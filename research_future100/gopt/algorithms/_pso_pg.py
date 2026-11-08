"""Point-group-constrained random clusters for the CALYPSO-style PSO (pso.py).

CALYPSO's cluster mode (Lv, Wang, Zhu & Ma, JCP 137, 084104 (2012); Wang et al.,
JPCM 27, 203203 (2015)) draws its random clusters with point-group symmetry.
``PGGen.structure`` picks a random point group (uniformly among the groups that
can reproduce the composition) from the 32 crystallographic point groups (C1
included) plus the five-fold groups C5, C5v, C5h, D5, D5h, D5d, I and Ih, then
adds one orbit at a time per species (largest count first):

* a site class is drawn uniformly among the classes whose orbit size fits the
  remaining count (and leaves a decomposable remainder): general position,
  mirror plane, rotation axis or the centre (used at most once);
* the site is a uniform random point of a ball of radius R projected onto the
  fixed set of an operation of that class; R is the radius of a compact cluster
  of the n_search atoms (sum of covalent-radius spheres at 0.74 packing),
  capped by the search region;
* the orbit is the set of distinct images under all operations.

Orbits must keep every distance >= dmin (as Problem.random_structure), the
cluster must be connected (Problem.bond_cut) if the problem requires it and
inside the search region. Returns (pos, cell) or None; never calls ctx.
"""
import numpy as np

_PHI = (1.0 + 5.0 ** 0.5) / 2.0


def _rot(axis, ang):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(ang) * K + (1.0 - np.cos(ang)) * (K @ K)


def _rz(n):
    return _rot((0, 0, 1), 2.0 * np.pi / n)


_MZ = np.diag([1.0, 1.0, -1.0])          # sigma_h
_MX = np.diag([-1.0, 1.0, 1.0])          # sigma_v (yz plane)
_C2X = np.diag([1.0, -1.0, -1.0])        # C2' about x
_INV = -np.eye(3)
_C3P = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])   # C3 about (1,1,1)
_MXY = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # sigma_d (x = y)
_C5I = _rot((0.0, 1.0, _PHI), 2.0 * np.pi / 5.0)                        # C5 of the icosahedron


def _gens():
    S4, S6, S10 = _rz(4) @ _MZ, _rz(6) @ _MZ, _rz(10) @ _MZ
    return {
        "C1": [], "Ci": [_INV], "C2": [_rz(2)], "Cs": [_MZ],
        "C2h": [_rz(2), _MZ], "D2": [_rz(2), _C2X], "C2v": [_rz(2), _MX],
        "D2h": [_rz(2), _C2X, _MZ],
        "C4": [_rz(4)], "S4": [S4], "C4h": [_rz(4), _MZ], "D4": [_rz(4), _C2X],
        "C4v": [_rz(4), _MX], "D2d": [S4, _C2X], "D4h": [_rz(4), _C2X, _MZ],
        "C3": [_rz(3)], "S6": [S6], "D3": [_rz(3), _C2X], "C3v": [_rz(3), _MX],
        "D3d": [S6, _C2X],
        "C6": [_rz(6)], "C3h": [_rz(3), _MZ], "C6h": [_rz(6), _MZ],
        "D6": [_rz(6), _C2X], "C6v": [_rz(6), _MX], "D3h": [_rz(3), _C2X, _MZ],
        "D6h": [_rz(6), _C2X, _MZ],
        "T": [_rz(2), _C3P], "Th": [_rz(2), _C3P, _INV], "O": [_rz(4), _C3P],
        "Td": [_rz(2), _C3P, _MXY], "Oh": [_rz(4), _C3P, _INV],
        "C5": [_rz(5)], "C5v": [_rz(5), _MX], "C5h": [_rz(5), _MZ],
        "D5": [_rz(5), _C2X], "D5h": [_rz(5), _C2X, _MZ], "D5d": [S10, _C2X],
        "I": [_rz(2), _C3P, _C5I], "Ih": [_rz(2), _C3P, _C5I, _INV],
    }


def _closure(gens, max_order=120):
    key = lambda m: tuple(np.round(m, 6).ravel() + 0.0)
    ops = {key(np.eye(3)): np.eye(3)}
    frontier = [np.eye(3)]
    while frontier:
        new = []
        for g in frontier:
            for h in gens:
                m = h @ g
                k = key(m)
                if k not in ops:
                    ops[k] = m
                    new.append(m)
        frontier = new
        if len(ops) > max_order:
            raise RuntimeError("point group closure did not converge")
    return np.array(list(ops.values()))


def _unique_points(X, tol=1e-3):
    d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    dup = np.tril(d < tol, -1).any(axis=1)
    return X[~dup]


def _group_data(ops):
    """Site classes of a point group: orbit size m, fixed-set dimension d, projectors."""
    eye = np.eye(3)
    v = np.array([0.2718281828, 0.5772156649, 0.8414709848])
    classes = {}
    seen = set()
    for g in ops:
        k, m = 1, g.copy()
        while not np.allclose(m, eye, atol=1e-8) and k < 13:
            m = m @ g
            k += 1
        if k >= 13:
            continue
        P = np.zeros((3, 3))
        m = eye.copy()
        for _ in range(k):
            P += m
            m = m @ g
        P /= k
        key = tuple(np.round(P, 6).ravel() + 0.0)
        if key in seen:
            continue
        seen.add(key)
        d = int(round(np.trace(P)))
        size = 1 if d == 0 else len(_unique_points(ops @ (P @ v)))
        classes.setdefault((size, d), []).append(P)
    if not any(size == 1 for size, _ in classes):
        classes[(1, 0)] = [np.zeros((3, 3))]
    return [dict(m=size, d=d, P=np.array(P)) for (size, d), P in sorted(classes.items())]


class PGGen:
    def __init__(self, p, max_groups=8, tries_per_atom=10, pack=0.74):
        self.p = p
        s = p.search_idx
        sym = np.array(p.symbols)[s]
        species = sorted(set(sym), key=lambda x: (-(sym == x).sum(), x))
        self.groups = [(x, s[sym == x]) for x in species]
        self.max_groups = int(max_groups)
        self.tries_per_atom = int(tries_per_atom)
        r = p.region
        if r.get("type") == "sphere":
            self.center = np.asarray(r["center"], float)
            rmax = float(r["radius"])
        elif r.get("type") == "box":
            lo, hi = np.asarray(r["lo"], float), np.asarray(r["hi"], float)
            self.center = 0.5 * (lo + hi)
            rmax = 0.5 * float(np.min(hi - lo))
        else:
            self.center = np.full(3, 0.5) @ p.cell0
            rmax = 0.5 / float(np.max(np.linalg.norm(np.linalg.inv(p.cell0), axis=0)))
        rc = p.rcov[s]
        self.radius = float(min(rmax, (np.sum(rc ** 3) / pack) ** (1.0 / 3.0)))
        counts = [len(i) for _, i in self.groups]
        nmax = max(counts)
        self.pg = []
        for name, gens in _gens().items():
            ops = _closure(gens)
            classes = _group_data(ops)
            sizes = sorted(set(c["m"] for c in classes if c["d"] > 0))
            R = np.zeros(nmax + 1, bool)
            R[0] = True
            for k in range(1, nmax + 1):
                R[k] = any(k >= m and R[k - m] for m in sizes)
            has_origin = any(c["d"] == 0 for c in classes)
            need = [i for i, n in enumerate(counts) if not R[n]]
            if len(need) > 1 or (need and not (has_origin and R[counts[need[0]] - 1])):
                continue
            self.pg.append(dict(name=name, ops=ops, classes=classes, R=R,
                                origin=has_origin, owner=need[0] if need else None))
        self.n_ok = 0
        self.n_fail = 0

    @staticmethod
    def _fits(g, c, need, use_origin):
        rest = need - c["m"]
        if rest < 0:
            return False
        if c["d"] == 0:
            return use_origin and bool(g["R"][rest])
        return bool(g["R"][rest]) or (use_origin and rest >= 1 and bool(g["R"][rest - 1]))

    def _connected(self, pos):
        from scipy.sparse.csgraph import connected_components
        p = self.p
        s = p.search_idx
        x = pos[s]
        rc = p.rcov[s]
        d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
        adj = d < 1.35 * (rc[:, None] + rc[None, :])
        return connected_components(adj, directed=False)[0] == 1

    def structure(self, rng):
        """Random cluster with point-group symmetry (pos, cell), or None."""
        p = self.p
        if not self.pg:
            self.n_fail += 1
            return None
        for _ in range(self.max_groups):
            g = self.pg[int(rng.integers(len(self.pg)))]
            owner = g["owner"]
            if owner is None and g["origin"]:
                owner = int(rng.integers(len(self.groups)))
            placed = np.zeros((0, 3))
            prad = np.zeros(0)
            out = {}
            ok = True
            for si, (x, idx) in enumerate(self.groups):
                rc = float(p.rcov[idx[0]])
                need = len(idx)
                use_origin = si == owner
                pts = []
                tries = 0
                while need > 0 and tries < self.tries_per_atom * len(idx) + 20:
                    tries += 1
                    cand = [c for c in g["classes"] if self._fits(g, c, need, use_origin)]
                    if not cand:
                        break
                    c = cand[int(rng.integers(len(cand)))]
                    P = c["P"][int(rng.integers(len(c["P"])))]
                    v = rng.normal(size=3)
                    v *= self.radius * rng.random() ** (1.0 / 3.0) / max(np.linalg.norm(v), 1e-12)
                    orb = _unique_points(g["ops"] @ (P @ v))
                    m = len(orb)
                    if m != c["m"]:
                        continue
                    if m > 1:
                        d = np.linalg.norm(orb[:, None, :] - orb[None, :, :], axis=-1)
                        d[np.arange(m), np.arange(m)] = np.inf
                        if d.min() < p.dmin_factor * 2.0 * rc:
                            continue
                    if len(placed):
                        d = np.linalg.norm(orb[:, None, :] - placed[None, :, :], axis=-1)
                        if np.any(d < p.dmin_factor * (rc + prad[None, :])):
                            continue
                    pts.append(orb)
                    placed = np.vstack([placed, orb])
                    prad = np.concatenate([prad, np.full(m, rc)])
                    need -= m
                    if c["d"] == 0:
                        use_origin = False
                if need > 0:
                    ok = False
                    break
                out[x] = np.vstack(pts)
            if not ok:
                continue
            pos = np.zeros((p.n, 3))
            pos[: p.n_sub] = p.sub_pos
            for x, idx in self.groups:
                pos[idx] = self.center + out[x][rng.permutation(len(idx))]
            if not all(p.in_region(pos[i]) for i in p.search_idx):
                continue
            if p.connected and not self._connected(pos):
                continue
            self.n_ok += 1
            return pos, p.cell0.copy()
        self.n_fail += 1
        return None
