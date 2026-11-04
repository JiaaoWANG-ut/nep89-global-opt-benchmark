"""Benchmark problem definition.

A benchmark lives in ``benchmarks/<name>/`` and consists of ``problem.json`` and,
for supported systems, a substrate file. The full system is
``substrate atoms (indices 0..n_sub-1) + search atoms (n_sub..n-1)``.

problem.json keys
-----------------
name, title, category ("cluster" | "surface" | "zeolite" | "bulk"), description
search_symbols      list of element symbols placed/moved by the global search
substrate_file      extxyz with the (pre-relaxed) host; omit for clusters/bulk
cell                3x3 cell when there is no substrate (cluster box / bulk guess)
pbc                 periodicity (NEP CPU always treats the cell as periodic;
                    clusters therefore sit in a box with > 2*cutoff of vacuum)
fixed_substrate     "all" (default) or list of substrate indices kept frozen;
relax_substrate     list of substrate indices free during local relaxation
                    (never moved by global moves); default none
region              where search atoms are generated:
                    {"type": "sphere", "center": [x,y,z], "radius": r}
                    {"type": "box", "lo": [x,y,z], "hi": [x,y,z]}
                    {"type": "cell"}  (whole cell, fractional sampling; bulk)
connected           require generated search atoms to form one connected
                    cluster (default true unless bulk)
variable_cell       relax the cell too (bulk crystal-structure prediction)
volume_per_atom     bulk only: volume guess (A^3/atom) for random cells
budget              ionic-step budget of one run
fmax                local-relaxation threshold (eV/A), default 0.05
e_tol               success tolerance above the putative global minimum (eV)
max_relax_steps     cap on steps of one local relaxation (default 500)
dmin_factor         minimum distance = factor * (r_cov_i + r_cov_j), default 0.7
"""
import hashlib
import json
import os

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from ase.geometry import cellpar_to_cell
from ase.io import read

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = os.environ.get("GOPT_BENCH_DIR", os.path.join(ROOT, "benchmarks"))


def list_benchmarks(root=BENCH_DIR):
    return sorted(d for d in os.listdir(root)
                  if os.path.isfile(os.path.join(root, d, "problem.json")))


def _images(pbc):
    rng = [(-1, 0, 1) if p else (0,) for p in pbc]
    return np.array([(i, j, k) for i in rng[0] for j in rng[1] for k in rng[2]], float)


def min_image_distances(a, b, cell, pbc):
    """Minimum-image distance matrix between point sets a (n,3) and b (m,3)."""
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    d = a[:, None, :] - b[None, :, :]
    if not np.any(pbc):
        return np.linalg.norm(d, axis=-1)
    inv = np.linalg.inv(cell)
    f = d @ inv
    for k in range(3):
        if pbc[k]:
            f[..., k] -= np.round(f[..., k])
    best = None
    for img in _images(pbc):
        dd = np.linalg.norm((f + img) @ cell, axis=-1)
        best = dd if best is None else np.minimum(best, dd)
    return best


class Problem:
    def __init__(self, name, root=BENCH_DIR):
        self.dir = os.path.join(root, name)
        with open(os.path.join(self.dir, "problem.json")) as f:
            spec = json.load(f)
        self.spec = spec
        # The hash identifies the energy landscape only (atoms, host, cell, frozen
        # atoms); budget, region, tolerances and texts may change without
        # invalidating earlier runs.
        land = {k: spec.get(k) for k in ("search_symbols", "pbc", "fixed_substrate",
                                         "relax_substrate", "variable_cell")}
        if not spec.get("substrate_file") and not spec.get("variable_cell"):
            land["cell"] = spec.get("cell")
        self.guard_factor = float(spec.get("guard_factor", 0.75))
        self.guard_k = float(spec.get("guard_k", 20.0))
        land["pes"] = f"nep89+guard({self.guard_factor:g},{self.guard_k:g})"
        # optional per-element-pair guard floors {"O-O": 1.9}: r0 = max(factor * rcov sum, floor)
        self.guard_pairs = {}
        for key, val in (spec.get("guard_pairs") or {}).items():
            a, b = key.split("-")
            self.guard_pairs[tuple(sorted((a, b)))] = float(val)
        if self.guard_pairs:
            land["pes"] += "+pairs(" + ",".join(
                f"{a}-{b}:{v:g}" for (a, b), v in sorted(self.guard_pairs.items())) + ")"
        self.name = spec.get("name", name)
        self.title = spec.get("title", self.name)
        self.category = spec["category"]
        self.description = spec.get("description", "")
        sub_file = spec.get("substrate_file")
        if sub_file:
            with open(os.path.join(self.dir, sub_file), "rb") as f:
                sub_bytes = f.read()
            sub = read(os.path.join(self.dir, sub_file))
            cell = np.array(sub.cell.array, float)
            pbc = spec.get("pbc", [bool(x) for x in sub.pbc])
        else:
            sub_bytes = None
            sub = Atoms()
            cell = np.array(spec["cell"], float).reshape(3, 3)
            pbc = spec.get("pbc", [True, True, True])
        # Slabs are periodic along the normal too (NEP_CPU is fully periodic), so search
        # atoms could cross the vacuum and bind to the frozen, bulk-terminated bottom face
        # of the periodic image. A flat-bottom wall half-way across the vacuum keeps them
        # on the top face; it is exactly zero for every structure that stays closer to it.
        # It was added without changing the hash: it is zero for every stored slab result
        # (the few runs that had ended on the far side were removed from results/).
        self.z_wall = None
        if self.category == "surface" and len(sub) and pbc[2] and not spec.get("variable_cell"):
            nrm = np.cross(cell[0], cell[1])
            nrm /= np.linalg.norm(nrm)
            period = float(cell[2] @ nrm)
            hgt = sub.positions @ nrm
            if period < 0:
                nrm, period, hgt = -nrm, -period, -hgt
            top, bot = float(hgt.max()), float(hgt.min())
            half = 0.5 * (period - (top - bot))
            self.z_wall = dict(normal=nrm, period=period, mid=0.5 * (top + bot), top=top, height=half)
        h = hashlib.sha1(json.dumps(land, sort_keys=True).encode())
        if sub_bytes is not None:
            h.update(sub_bytes)
        self.hash = h.hexdigest()[:12]
        self.cell0 = cell
        self.pbc = np.array(pbc, bool)
        self.n_sub = len(sub)
        self.sub_pos = sub.positions.copy()
        self.sub_symbols = sub.get_chemical_symbols()
        self.search_symbols = list(spec["search_symbols"])
        self.symbols = self.sub_symbols + self.search_symbols
        self.n = len(self.symbols)
        self.n_search = len(self.search_symbols)
        self.numbers = np.array([atomic_numbers[s] for s in self.symbols])
        self.search_idx = np.arange(self.n_sub, self.n)
        free = np.zeros(self.n, bool)
        free[self.search_idx] = True
        for i in spec.get("relax_substrate", []) or []:
            free[int(i)] = True
        fixed = spec.get("fixed_substrate", "all")
        if fixed != "all" and fixed is not None:
            # everything in the substrate not listed as fixed is relaxable
            fixed = set(int(i) for i in fixed)
            for i in range(self.n_sub):
                if i not in fixed:
                    free[i] = True
        self.free_mask = free
        self.free_idx = np.where(free)[0]
        self.variable_cell = bool(spec.get("variable_cell", False))
        self.region = spec.get("region", {"type": "cell"})
        self.connected = bool(spec.get("connected", not self.variable_cell))
        self.budget = int(spec.get("budget", 30000))
        self.fmax = float(spec.get("fmax", 0.05))
        self.e_tol = float(spec.get("e_tol", 0.02))
        self.max_relax_steps = int(spec.get("max_relax_steps", 500))
        self.vpa = spec.get("volume_per_atom")
        self.dmin_factor = float(spec.get("dmin_factor", 0.7))
        self.rcov = covalent_radii[self.numbers]
        self.species = sorted(set(self.search_symbols))

    # ------------------------------------------------------------------ helpers
    def __repr__(self):
        return (f"Problem({self.name}: {self.category}, n={self.n}, search={self.n_search}, "
                f"free={self.free_mask.sum()}, vc={self.variable_cell})")

    def to_atoms(self, pos, cell=None):
        return Atoms(self.symbols, positions=pos, cell=self.cell0 if cell is None else cell,
                     pbc=self.pbc)

    def dmin(self, i, j_idx):
        return self.dmin_factor * (self.rcov[i] + self.rcov[j_idx])

    def bond_cut(self, i, j_idx, factor=1.35):
        return factor * (self.rcov[i] + self.rcov[j_idx])

    def in_region(self, x, cell=None):
        r = self.region
        t = r["type"]
        if t == "sphere":
            return np.linalg.norm(np.asarray(x) - np.asarray(r["center"])) <= r["radius"]
        if t == "box":
            return bool(np.all(x >= np.asarray(r["lo"])) and np.all(x <= np.asarray(r["hi"])))
        return True

    def sample_point(self, rng, cell):
        r = self.region
        t = r["type"]
        if t == "sphere":
            v = rng.normal(size=3)
            v /= np.linalg.norm(v)
            return np.asarray(r["center"], float) + v * r["radius"] * rng.random() ** (1 / 3)
        if t == "box":
            lo, hi = np.asarray(r["lo"], float), np.asarray(r["hi"], float)
            return lo + rng.random(3) * (hi - lo)
        return rng.random(3) @ cell

    def random_cell(self, rng):
        """Random cell for bulk problems with volume ~ volume_per_atom * n."""
        vol = float(self.vpa) * self.n_search * rng.uniform(0.9, 1.3)
        for _ in range(1000):
            lengths = rng.uniform(0.6, 1.6, 3)
            angles = rng.uniform(60, 120, 3)
            try:
                cell = cellpar_to_cell(list(lengths) + list(angles))
            except Exception:
                continue
            v = abs(np.linalg.det(cell))
            if not np.isfinite(v) or v < 1e-3:
                continue
            cell *= (vol / v) ** (1 / 3)
            # reject flat cells: every interplanar spacing >= 0.45 * vol^(1/3)
            rec = np.linalg.inv(cell).T
            heights = 1.0 / np.linalg.norm(rec, axis=1)
            if heights.min() >= 0.45 * vol ** (1 / 3):
                return cell
        return np.eye(3) * vol ** (1 / 3)

    def place(self, pos, idx, rng, cell=None, connected=None, max_tries=3000):
        """(Re)place search atoms ``idx`` at random positions in the region.

        Atoms not in ``idx`` keep their positions and act as obstacles. Returns a
        new position array, or None if placement failed.
        """
        cell = self.cell0 if cell is None else cell
        connected = self.connected if connected is None else connected
        pos = np.array(pos, float)
        idx = list(idx)
        keep = np.ones(self.n, bool)
        keep[idx] = False
        # substrate atoms are always obstacles; search atoms being placed are not yet
        others = list(np.where(keep)[0])
        search_set = set(self.search_idx.tolist())
        for i in idx:
            ok = False
            for _ in range(max_tries):
                x = self.sample_point(rng, cell)
                if others:
                    oth = np.array(others)
                    d = min_image_distances(x, pos[oth], cell, self.pbc)[0]
                    if np.any(d < self.dmin(i, oth)):
                        continue
                    if connected:
                        cl = [j for j in others if j in search_set]
                        if cl:
                            cl = np.array(cl)
                            dc = min_image_distances(x, pos[cl], cell, self.pbc)[0]
                            if not np.any(dc < self.bond_cut(i, cl)):
                                continue
                ok = True
                break
            if not ok:
                return None
            pos[i] = x
            others.append(i)
        return pos

    def random_structure(self, rng):
        """Random initial structure: (positions[n,3], cell[3,3])."""
        for attempt in range(50):
            cell = self.random_cell(rng) if self.variable_cell else self.cell0.copy()
            pos = np.zeros((self.n, 3))
            pos[: self.n_sub] = self.sub_pos
            order = rng.permutation(self.search_idx)
            new = self.place(pos, order, rng, cell)
            if new is not None:
                return new, cell
        raise RuntimeError(f"{self.name}: could not generate a random structure")

    def min_dist_ok(self, pos, cell=None, factor=1.0):
        """True if no search atom is closer than factor*dmin to any other atom."""
        cell = self.cell0 if cell is None else cell
        s = self.search_idx
        d = min_image_distances(pos[s], pos, cell, self.pbc)
        lim = factor * self.dmin_factor * (self.rcov[s][:, None] + self.rcov[None, :])
        d[np.arange(len(s)), s] = np.inf
        return bool(np.all(d >= lim))

    def cluster_components(self, pos, cell=None, factor=1.35):
        """Number of connected components among the search atoms."""
        cell = self.cell0 if cell is None else cell
        s = self.search_idx
        d = min_image_distances(pos[s], pos[s], cell, self.pbc)
        adj = d < factor * (self.rcov[s][:, None] + self.rcov[s][None, :])
        seen = np.zeros(len(s), bool)
        ncomp = 0
        for k in range(len(s)):
            if seen[k]:
                continue
            ncomp += 1
            stack = [k]
            seen[k] = True
            while stack:
                u = stack.pop()
                for v in np.where(adj[u] & ~seen)[0]:
                    seen[v] = True
                    stack.append(v)
        return ncomp
