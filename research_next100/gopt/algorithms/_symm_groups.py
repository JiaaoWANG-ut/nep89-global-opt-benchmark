"""Symmetry machinery for the ``symm`` algorithm (no energy calls, no randomness
except through the ``rng`` arguments).

* point groups (Cn, Cnv, Cnh, Dn, Dnh, Dnd, S2n, T, Td, Th, O, Oh, I, Ih) as
  closed sets of 3x3 orthogonal matrices, with their orbit ("Wyckoff") classes:
  centre, rotation axes, mirror planes, general position;
* a few dozen common space groups (generators -> closure modulo lattice
  translations) with their Wyckoff positions enumerated numerically on a
  deterministic 24^3 fractional grid (site-symmetry group -> fixed affine subspace);
* composition partitioning into orbit multiplicities (bounded-knapsack bitsets);
* approximate point-group detection and Reynolds symmetrisation of clusters.
"""
from functools import lru_cache
from math import gcd

import numpy as np
from scipy.optimize import linear_sum_assignment

PHI = (1.0 + 5.0 ** 0.5) / 2.0
I3 = np.eye(3)


# ===================================================================== utils
def rot(axis, ang):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return I3 + np.sin(ang) * k + (1 - np.cos(ang)) * k @ k


def mirror(normal):
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    return I3 - 2.0 * np.outer(n, n)


def closure(gens, maxn=240, tol=1e-6):
    els = [I3.copy()]
    arr = np.array(els)
    queue = [I3.copy()]
    while queue:
        a = queue.pop()
        for g in gens:
            b = a @ g
            if np.abs(arr - b).reshape(len(arr), -1).max(axis=1).min() > tol:
                els.append(b)
                arr = np.array(els)
                queue.append(b)
                if len(els) > maxn:
                    raise ValueError("group closure too large")
    return arr


def nullspace(a, tol=1e-6):
    a = np.atleast_2d(a)
    u, s, vt = np.linalg.svd(a)
    rank = int((s > tol).sum())
    return vt[rank:]


def random_unit(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([
        [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
        [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
        [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d]])


# ============================================================== point groups
def _rz(n):
    return rot([0, 0, 1], 2 * np.pi / n)


def pg_generators(name):
    """Generators with the principal axis along z (cubic/icosahedral: C2 along x,y,z)."""
    c3 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)  # C3 along [111]
    c2x = rot([1, 0, 0], np.pi)
    sh = mirror([0, 0, 1])
    sv = mirror([0, 1, 0])
    if name == "C1":
        return []
    if name == "Cs":
        return [sv]
    if name == "Ci":
        return [-I3]
    if name in ("T", "Td", "Th"):
        g = [rot([0, 0, 1], np.pi), c3]
        if name == "Td":
            g.append(mirror([1, -1, 0]))
        if name == "Th":
            g.append(-I3)
        return g
    if name in ("O", "Oh"):
        g = [_rz(4), c3]
        if name == "Oh":
            g.append(-I3)
        return g
    if name in ("I", "Ih"):
        g = [rot([0, 0, 1], np.pi), c3, rot([0, 1, PHI], 2 * np.pi / 5)]
        if name == "Ih":
            g.append(-I3)
        return g
    if name[0] == "S":
        n2 = int(name[1:])
        return [sh @ _rz(n2)]
    kind = name[0]
    rest = name[1:]
    suffix = rest.lstrip("0123456789")
    n = int(rest[: len(rest) - len(suffix)])
    g = [_rz(n)]
    if kind == "C":
        if suffix == "v":
            g.append(sv)
        elif suffix == "h":
            g.append(sh)
        return g
    # D groups
    g.append(c2x)
    if suffix == "h":
        g.append(sh)
    elif suffix == "d":
        a = np.pi / (2 * n) + np.pi / 2
        g.append(mirror([np.cos(a), np.sin(a), 0]))
    return g


CLUSTER_GROUPS = ["Cs", "C2", "Ci", "C2v", "C2h", "D2", "S4", "C3", "C3v", "C3h", "D3", "S6",
                  "D2h", "D2d", "C4v", "C5v", "C6v", "D3h", "D3d", "D4", "C4h", "D4h", "D4d",
                  "D5h", "D5d", "D6h", "D6d", "T", "Td", "Th", "O", "Oh", "I", "Ih"]
# groups with a vertical (z) axis and no operation flipping z (supported clusters)
SURFACE_GROUPS = ["Cs", "C2", "C2v", "C3", "C3v", "C4", "C4v", "C5v", "C6v"]


class PointGroup:
    def __init__(self, name):
        self.name = name
        gens = pg_generators(name)
        self.ops = closure(gens) if gens else I3[None].copy()
        self.order = len(self.ops)
        self.sites = self._sites()
        self.level = 0 if self.order <= 4 else (1 if self.order <= 12 else 2)

    def _sites(self):
        """Orbit classes: dict(dim, basis (dim,3), mult, cap, sym)."""
        G = self.ops
        lines, planes = [], []
        for g in G[1:]:
            ns = nullspace(g - I3)
            if len(ns) == 1:
                u = ns[0]
                if not any(abs(abs(u @ v) - 1) < 1e-6 for v in lines):
                    lines.append(u)
            elif len(ns) == 2:
                nv = np.cross(ns[0], ns[1])
                nv /= np.linalg.norm(nv)
                if not any(abs(abs(nv @ v) - 1) < 1e-6 for v in planes):
                    planes.append(nv)
        out = [dict(dim=0, basis=np.zeros((0, 3)), mult=1, cap=1, sym=True)]
        for kind, items in (("line", lines), ("plane", planes)):
            classes = []
            for u in items:
                if any(any(abs(abs((g @ u) @ c) - 1) < 1e-6 for g in G) for c in classes):
                    continue
                classes.append(u)
            for u in classes:
                if kind == "line":
                    basis = u[None]
                    x = 0.7 * u
                else:
                    a = np.cross(u, [1.0, 0.3, 0.1])
                    if np.linalg.norm(a) < 1e-3:
                        a = np.cross(u, [0.2, 1.0, 0.0])
                    a /= np.linalg.norm(a)
                    b = np.cross(u, a)
                    basis = np.array([a, b])
                    x = 0.7 * a + 0.37 * b
                stab = int((np.abs(G @ x - x).max(axis=1) < 1e-6).sum())
                mult = self.order // stab
                # are +x and -x in the same orbit (line sampling on one side suffices)?
                sym = bool(np.any(np.abs(G @ x + x).max(axis=1) < 1e-6))
                out.append(dict(dim=len(basis), basis=basis, mult=mult,
                                cap=(3 if kind == "line" else 99), sym=sym))
        out.append(dict(dim=3, basis=I3.copy(), mult=self.order, cap=99, sym=True))
        return out

    def orbit(self, x, tol=1e-3):
        pts = self.ops @ x
        keep = [0]
        for i in range(1, len(pts)):
            if np.min(np.linalg.norm(pts[keep] - pts[i], axis=1)) > tol:
                keep.append(i)
        return pts[keep]


@lru_cache(maxsize=None)
def point_group(name):
    return PointGroup(name)


# ============================================================== space groups
_C = [(0.5, 0.5, 0.0)]
_I = [(0.5, 0.5, 0.5)]
_F = [(0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)]
_R = [(2 / 3, 1 / 3, 1 / 3), (1 / 3, 2 / 3, 2 / 3)]
# primitive-cell transformations (rows: primitive vectors in conventional fractions)
_PRIM = {
    "C": np.array([[0.5, -0.5, 0], [0.5, 0.5, 0], [0, 0, 1]]),
    "I": np.array([[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]]),
    "F": np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]),
    "R": np.array([[2 / 3, 1 / 3, 1 / 3], [-1 / 3, 1 / 3, 1 / 3], [-1 / 3, -2 / 3, 1 / 3]]),
}
# (symbol, lattice system, centring, generators); ITA standard settings
SPACE_GROUPS = [
    ("P-1", "tric", "P", ["-x,-y,-z"]),
    ("P2_1", "mono", "P", ["-x,y+1/2,-z"]),
    ("Cm", "mono", "C", ["x,-y,z"]),
    ("P2_1/m", "mono", "P", ["-x,y+1/2,-z", "-x,-y,-z"]),
    ("C2/m", "mono", "C", ["-x,y,-z", "-x,-y,-z"]),
    ("P2_1/c", "mono", "P", ["-x,y+1/2,-z+1/2", "-x,-y,-z"]),
    ("C2/c", "mono", "C", ["-x,y,-z+1/2", "-x,-y,-z"]),
    ("Pna2_1", "orth", "P", ["-x,-y,z+1/2", "x+1/2,-y+1/2,z"]),
    ("Pmmm", "orth", "P", ["-x,-y,z", "-x,y,-z", "-x,-y,-z"]),
    ("Cmmm", "orth", "C", ["-x,-y,z", "-x,y,-z", "-x,-y,-z"]),
    ("Immm", "orth", "I", ["-x,-y,z", "-x,y,-z", "-x,-y,-z"]),
    ("Fmmm", "orth", "F", ["-x,-y,z", "-x,y,-z", "-x,-y,-z"]),
    ("Pbcn", "orth", "P", ["-x+1/2,-y+1/2,z+1/2", "-x,y,-z+1/2", "-x,-y,-z"]),
    ("Pbca", "orth", "P", ["-x+1/2,-y,z+1/2", "-x,y+1/2,-z+1/2", "-x,-y,-z"]),
    ("Pnma", "orth", "P", ["-x+1/2,-y,z+1/2", "-x,y+1/2,-z", "-x,-y,-z"]),
    ("Cmcm", "orth", "C", ["-x,-y,z+1/2", "-x,y,-z+1/2", "-x,-y,-z"]),
    ("P4/mmm", "tetr", "P", ["-y,x,z", "x,-y,-z", "-x,-y,-z"]),
    ("P4/nmm", "tetr", "P", ["-x+1/2,-y+1/2,z", "-y+1/2,x,z", "-x,y+1/2,-z", "-x,-y,-z"]),
    ("P4_2/mnm", "tetr", "P", ["-y+1/2,x+1/2,z+1/2", "x+1/2,-y+1/2,-z+1/2", "-x,-y,-z"]),
    ("I4/mmm", "tetr", "I", ["-y,x,z", "x,-y,-z", "-x,-y,-z"]),
    ("I4_1/amd", "tetr", "I", ["-x+1/2,-y,z+1/2", "-y+1/4,x+3/4,z+1/4", "-x+1/2,y,-z+1/2",
                               "-x,-y,-z"]),
    ("I-42d", "tetr", "I", ["-x,-y,z", "y,-x,-z", "-x+1/2,y,-z+3/4"]),
    ("P3_121", "hexa", "P", ["-y,x-y,z+1/3", "y,x,-z"]),
    ("R-3", "hexa", "R", ["-y,x-y,z", "-x,-y,-z"]),
    ("P-3m1", "hexa", "P", ["-y,x-y,z", "y,x,-z", "-x,-y,-z"]),
    ("R-3m", "hexa", "R", ["-y,x-y,z", "y,x,-z", "-x,-y,-z"]),
    ("R-3c", "hexa", "R", ["-y,x-y,z", "y,x,-z+1/2", "-x,-y,-z"]),
    ("P6_3mc", "hexa", "P", ["-y,x-y,z", "-x,-y,z+1/2", "-y,-x,z"]),
    ("P-6m2", "hexa", "P", ["-y,x-y,z", "x,y,-z", "-y,-x,z"]),
    ("P6/mmm", "hexa", "P", ["-y,x-y,z", "-x,-y,z", "y,x,-z", "-x,-y,-z"]),
    ("P6_3/mmc", "hexa", "P", ["-y,x-y,z", "-x,-y,z+1/2", "y,x,-z", "-x,-y,-z"]),
    ("Pa-3", "cubi", "P", ["-x+1/2,-y,z+1/2", "z,x,y", "-x,-y,-z"]),
    ("Pm-3m", "cubi", "P", ["-y,x,z", "z,x,y", "-x,-y,-z"]),
    ("F-43m", "cubi", "F", ["y,-x,-z", "z,x,y"]),
    ("Fm-3m", "cubi", "F", ["-y,x,z", "z,x,y", "-x,-y,-z"]),
    ("Fd-3m", "cubi", "F", ["y,-x,-z", "z,x,y", "-x+1/4,-y+1/4,-z+1/4"]),
    ("Im-3m", "cubi", "I", ["-y,x,z", "z,x,y", "-x,-y,-z"]),
]


def _parse_op(s):
    W = np.zeros((3, 3))
    w = np.zeros(3)
    for r, comp in enumerate(s.split(",")):
        comp = comp.replace("-", "+-")
        for term in comp.split("+"):
            term = term.strip()
            if not term:
                continue
            sign = -1.0 if term.startswith("-") else 1.0
            t = term.lstrip("-")
            if t in ("x", "y", "z"):
                W[r, "xyz".index(t)] += sign
            else:
                a, b = t.split("/") if "/" in t else (t, "1")
                w[r] += sign * float(a) / float(b)
    return W, w


def _sg_closure(gens, cent, maxn=400):
    """All operations (W, w mod 1) of the conventional cell, centring included."""
    ops = [(I3.copy(), np.zeros(3))] + [(I3.copy(), np.array(c, float)) for c in cent]
    gens = list(gens) + [(I3.copy(), np.array(c, float)) for c in cent]

    def key(W, w):
        ww = np.round(w % 1.0, 5) % 1.0
        return tuple(np.round(W).astype(int).ravel()) + tuple(ww)

    seen = {key(W, w) for W, w in ops}
    queue = list(ops)
    while queue:
        W1, w1 = queue.pop()
        for W2, w2 in gens:
            W = W1 @ W2
            w = (W1 @ w2 + w1) % 1.0
            k = key(W, w)
            if k not in seen:
                seen.add(k)
                ops.append((W, w))
                queue.append((W, w))
                if len(ops) > maxn:
                    raise ValueError("space group closure too large")
    return ops


class SpaceGroup:
    def __init__(self, symbol, system, centring, gens):
        self.symbol, self.system, self.centring = symbol, system, centring
        cent = {"P": [], "C": _C, "I": _I, "F": _F, "R": _R}[centring]
        self.ncent = 1 + len(cent)
        ops = _sg_closure([_parse_op(g) for g in gens], cent)
        self.W = np.array([o[0] for o in ops])
        self.w = np.array([o[1] for o in ops])
        self.order = len(ops)
        self.prim = _PRIM.get(centring)
        self._wyckoff()

    def orbit(self, x, tol=1e-4):
        pts = (np.einsum("kij,j->ki", self.W, x) + self.w) % 1.0
        keep = []
        for i in range(len(pts)):
            if keep:
                d = pts[keep] - pts[i]
                d -= np.round(d)
                if np.min(np.abs(d).max(axis=1)) < tol:
                    continue
            keep.append(i)
        return pts[keep]

    def _wyckoff(self):
        g = np.arange(24) / 24.0
        X = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
        # images[k, p] = W_k x_p + w_k ; fixed if equal modulo lattice
        stab = np.zeros((len(X), self.order), bool)
        trans = np.zeros((len(X), self.order, 3), np.int8)
        for k in range(self.order):
            d = X @ self.W[k].T + self.w[k] - X
            t = np.round(d)
            stab[:, k] = np.abs(d - t).max(axis=1) < 1e-6
            trans[:, k] = np.where(stab[:, k][:, None], t, 0).astype(np.int8)
        sel = np.where(stab.sum(axis=1) > 1)[0]
        keyarr = np.ascontiguousarray(np.hstack([np.packbits(stab[sel], axis=1).astype(np.int8),
                                                 trans[sel].reshape(len(sel), 3 * self.order)]))
        firstd = {}
        for i, row in enumerate(keyarr):
            firstd.setdefault(row.tobytes(), sel[i])
        first = np.array(sorted(firstd.values()), int)
        points0, lines = {}, {}
        for p in first:
            S = np.where(stab[p])[0]
            if len(S) <= 1:
                continue
            x = X[p]
            A = np.vstack([self.W[k] - I3 for k in S])
            N = nullspace(A)
            mult = self.order // len(S)
            if len(N) == 0:
                orb = self.orbit(x)
                key = tuple(sorted(tuple(np.round(o, 4) % 1.0) for o in orb))
                if key not in points0:
                    points0[key] = dict(dim=0, x0=x.copy(), N=np.zeros((0, 3)), mult=mult, cap=1)
            else:
                k2 = (mult, len(N))
                lst = lines.setdefault(k2, [])
                if len(lst) < 12:
                    lst.append((x.copy(), N))
        sites = list(points0.values())
        for (mult, dim), reps in lines.items():
            sites.append(dict(dim=dim, reps=reps, mult=mult, cap=99))
        sites.append(dict(dim=3, reps=[(np.zeros(3), I3.copy())], mult=self.order, cap=99))
        self.sites = sites

    def sample_site(self, site, rng):
        if site["dim"] == 0:
            return site["x0"].copy()
        x0, N = site["reps"][rng.integers(len(site["reps"]))]
        r = rng.uniform(-1.5, 1.5, len(N))
        return (x0 + r @ N) % 1.0


@lru_cache(maxsize=None)
def space_group(symbol):
    for s, sysm, cen, gens in SPACE_GROUPS:
        if s == symbol:
            return SpaceGroup(s, sysm, cen, gens)
    raise KeyError(symbol)


def lattice(system, rng, vol):
    """Random conventional cell (rows) of the lattice system with volume vol."""
    lu = lambda lo, hi: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))  # noqa: E731
    if system == "cubi":
        a = b = c = 1.0
        al = be = ga = 90.0
    elif system == "tetr":
        a = b = 1.0
        c = lu(0.45, 2.8)
        al = be = ga = 90.0
    elif system == "hexa":
        a = b = 1.0
        c = lu(0.45, 3.5)
        al = be = 90.0
        ga = 120.0
    elif system == "orth":
        a, b, c = 1.0, lu(0.45, 2.2), lu(0.45, 2.2)
        al = be = ga = 90.0
    elif system == "mono":
        a, b, c = 1.0, lu(0.45, 2.2), lu(0.45, 2.2)
        al = ga = 90.0
        be = rng.uniform(90.0, 125.0)
    else:
        a, b, c = 1.0, lu(0.6, 1.6), lu(0.6, 1.6)
        al, be, ga = rng.uniform(65, 115, 3)
    from ase.geometry import cellpar_to_cell
    cell = cellpar_to_cell([a, b, c, al, be, ga])
    v = abs(np.linalg.det(cell))
    return cell * (vol / v) ** (1.0 / 3.0)


# ============================================================ partitioning
def _reach_mask(r, sizes_caps):
    """Bitmask of totals <= r reachable with bounded multiplicities."""
    full = (1 << (r + 1)) - 1
    m = 1
    for size, cap in sizes_caps:
        if size > r or cap <= 0:
            continue
        for _ in range(min(cap, r // size)):
            new = (m | (m << size)) & full
            if new == m:
                break
            m = new
    return m


def feasible(counts, sites):
    """Quick necessary check: every species count is reachable (capacities per species)."""
    sc = [(s["mult"], s["cap"]) for s in sites]
    return all((_reach_mask(c, sc) >> c) & 1 for c in counts)


def partition(counts, sites, rng, alpha=0.5, tries=20):
    """Random assignment of orbits to species: list of (species index, site index)."""
    for _ in range(tries):
        cap = [s["cap"] for s in sites]
        plan = []
        ok = True
        for si in rng.permutation(len(counts)):
            r = int(counts[si])
            while r > 0:
                opts, w = [], []
                for k, s in enumerate(sites):
                    m = s["mult"]
                    if m > r or cap[k] <= 0:
                        continue
                    cap[k] -= 1
                    rest = r - m
                    good = rest == 0 or (_reach_mask(rest, [(sites[j]["mult"], cap[j])
                                                            for j in range(len(sites))]) >> rest) & 1
                    cap[k] += 1
                    if good:
                        opts.append(k)
                        w.append(float(m) ** alpha)
                if not opts:
                    ok = False
                    break
                w = np.array(w)
                k = opts[rng.choice(len(opts), p=w / w.sum())]
                cap[k] -= 1
                plan.append((int(si), k))
                r -= sites[k]["mult"]
            if not ok:
                break
        if ok:
            return plan
    return None


def divisors(n):
    return [k for k in range(1, n + 1) if n % k == 0]


def gcd_list(xs):
    g = 0
    for x in xs:
        g = gcd(g, int(x))
    return g


# ======================================================= detection (clusters)
def _match(Y, X, labels, tol):
    """Permutation p with Y[i] ~ X[p[i]] (same labels), or None if worse than tol."""
    n = len(X)
    p = np.empty(n, int)
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        d = np.linalg.norm(Y[idx][:, None, :] - X[idx][None, :, :], axis=-1)
        j = d.argmin(axis=1)
        if len(np.unique(j)) == len(idx) and d[np.arange(len(idx)), j].max() < tol:
            p[idx] = idx[j]
            continue
        if d.min(axis=1).max() > tol:
            return None
        r, c = linear_sum_assignment(d)
        if d[r, c].max() > tol:
            return None
        p[idx[r]] = idx[c]
    return p


def detect_ops(X, labels, tol=0.15, max_axes=150):
    """Point-group operations (about the centroid) mapping X onto itself within tol."""
    X = np.asarray(X, float)
    X = X - X.mean(axis=0)
    n = len(X)
    labels = np.asarray(labels)
    found = []
    if n < 2:
        return [I3.copy()]
    cand = []
    w, v = np.linalg.eigh(np.einsum("ij,ik->jk", X, X))
    cand.extend(v.T)
    r = np.linalg.norm(X, axis=1)
    for i in range(n):
        if r[i] > 0.3:
            cand.append(X[i] / r[i])
    d = np.linalg.norm(X[:, None] - X[None], axis=-1)
    for i in range(n):
        same = np.where((labels == labels[i]) & (np.abs(r - r[i]) < tol))[0]
        same = same[same > i]
        if len(same) > 4:
            same = same[np.argsort(d[i, same])[:4]]
        for j in same:
            m = 0.5 * (X[i] + X[j])
            if np.linalg.norm(m) > 0.3:
                cand.append(m / np.linalg.norm(m))
            dv = X[i] - X[j]
            cand.append(dv / np.linalg.norm(dv))
    axes = []
    for u in cand:
        if all(abs(u @ a) < 0.9995 for a in axes):
            axes.append(u)
        if len(axes) >= max_axes:
            break
    ops = [-I3]
    for u in axes:
        for k in (2, 3, 4, 5, 6):
            ops.append(rot(u, 2 * np.pi / k))
        ops.append(mirror(u))
        sh = mirror(u)
        ops.append(sh @ rot(u, np.pi / 2))
        ops.append(sh @ rot(u, np.pi / 3))
    for g in ops:
        if _match(X @ g.T, X, labels, tol) is not None:
            found.append(g)
    if not found:
        return [I3.copy()]
    # largest consistent group generated by the passing operations
    found.sort(key=lambda g: -abs(np.trace(g)))
    group = [I3.copy()]
    for g in found:
        if any(np.abs(g - h).max() < 1e-4 for h in group):
            continue
        try:
            cl = closure(group[1:] + [g], maxn=120)
        except ValueError:
            continue
        if all(_match(X @ h.T, X, labels, 2 * tol) is not None for h in cl):
            group = list(cl)
    return group


def symmetrize(X, labels, group, tol=0.6, iters=3):
    """Reynolds projection of X (about its centroid) onto the group; None if no match."""
    X = np.asarray(X, float)
    c = X.mean(axis=0)
    Y = X - c
    labels = np.asarray(labels)
    for _ in range(iters):
        acc = np.zeros_like(Y)
        for g in group:
            p = _match(Y @ g.T, Y, labels, tol)
            if p is None:
                return None
            # g y_i ~ y_p(i)  =>  y_i ~ g^T y_p(i)
            acc += Y[p] @ g
        Y = acc / len(group)
    return Y + c


def orbits_of(X, labels, group, tol=0.3):
    """Atom orbits (lists of indices) of an (almost) symmetric structure."""
    X = np.asarray(X, float)
    Y = X - X.mean(axis=0)
    n = len(Y)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for g in group:
        p = _match(Y @ g.T, Y, np.asarray(labels), tol)
        if p is None:
            continue
        for i in range(n):
            a, b = find(i), find(int(p[i]))
            if a != b:
                parent[a] = b
    out = {}
    for i in range(n):
        out.setdefault(find(i), []).append(i)
    return list(out.values())


def group_name(group):
    """Rough Schoenflies label from the element structure (for diagnostics)."""
    n = len(group)
    if n == 1:
        return "C1"
    tr = np.array([np.trace(g) for g in group])
    det = np.array([np.linalg.det(g) for g in group])
    inv = bool(np.any(np.abs(np.array(group) + I3).reshape(n, -1).max(axis=1) < 1e-4))
    nmir = int(np.sum((det < 0) & (np.abs(tr - 1) < 1e-4)))
    proper = tr[det > 0]
    ang = np.arccos(np.clip((proper - 1) / 2, -1, 1))
    orders = [int(round(2 * np.pi / a)) for a in ang if a > 1e-3]
    nmax = max(orders) if orders else 1
    return f"G{n}(n={nmax},m={nmir},i={int(inv)})"
