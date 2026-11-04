"""GOFEE helper: vectorised candidate-generation primitives (no energy calls).

Same distributions as the per-trial loops in gopt.problem / the GOFEE operators, but all
trials of one atom are tested in a single numpy call.
"""
import numpy as np

from ..problem import min_image_distances


def _heights(cell):
    return 1.0 / np.linalg.norm(np.linalg.inv(cell).T, axis=1)


def mic_dist(p, a, b, cell, rmax=6.0):
    """Minimum-image distances (len(a), len(b)); exact for distances < rmax.

    Uses one rounding step when every periodic cell height exceeds 2*rmax (clusters,
    slabs, zeolites), otherwise falls back to the full image search.
    """
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    if not np.any(p.pbc):
        return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    h = _heights(cell)
    if np.any(h[p.pbc] < 2.0 * rmax):
        return min_image_distances(a, b, cell, p.pbc)
    inv = np.linalg.inv(cell)
    f = (a[:, None, :] - b[None, :, :]) @ inv
    f[..., p.pbc] -= np.round(f[..., p.pbc])
    v = f @ cell
    return np.sqrt((v * v).sum(-1))


def _sample(p, rng, cell, k):
    """k points distributed like p.sample_point."""
    r = p.region
    t = r.get("type", "cell")
    if t == "sphere":
        v = rng.normal(size=(k, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return np.asarray(r["center"], float) + v * (r["radius"] * rng.random(k) ** (1 / 3))[:, None]
    if t == "box":
        lo, hi = np.asarray(r["lo"], float), np.asarray(r["hi"], float)
        return lo + rng.random((k, 3)) * (hi - lo)
    return rng.random((k, 3)) @ cell


def place(p, pos, idx, rng, cell=None, max_tries=3000, chunk=64):
    """Vectorised equivalent of Problem.place (sequential rejection sampling)."""
    cell = p.cell0 if cell is None else cell
    pos = np.array(pos, float)
    keep = np.ones(p.n, bool)
    keep[list(idx)] = False
    others = list(np.where(keep)[0])
    search = np.zeros(p.n, bool)
    search[p.search_idx] = True
    rmax = 1.35 * 2.0 * float(np.max(p.rcov))
    for i in idx:
        oth = np.array(others, int)
        lim = p.dmin(i, oth) if len(oth) else None
        cl = oth[search[oth]] if (p.connected and len(oth)) else np.zeros(0, int)
        bc = p.bond_cut(i, cl) if len(cl) else None
        ok_x = None
        tries = 0
        while tries < max_tries and ok_x is None:
            k = min(chunk, max_tries - tries)
            tries += k
            X = _sample(p, rng, cell, k)
            good = np.ones(k, bool)
            if len(oth):
                d = mic_dist(p, X, pos[oth], cell, rmax)
                good &= np.all(d >= lim[None, :], axis=1)
                if len(cl):
                    good &= np.any(d[:, search[oth]] < bc[None, :], axis=1)
            hit = np.nonzero(good)[0]
            if len(hit):
                ok_x = X[hit[0]]
        if ok_x is None:
            return None
        pos[i] = ok_x
        others.append(i)
    return pos


def random_structure(p, rng):
    """Vectorised equivalent of Problem.random_structure."""
    for _ in range(50):
        cell = p.random_cell(rng) if p.variable_cell else p.cell0.copy()
        pos = np.zeros((p.n, 3))
        pos[: p.n_sub] = p.sub_pos
        new = place(p, pos, rng.permutation(p.search_idx), rng, cell)
        if new is not None:
            return new, cell
    return p.random_structure(rng)


def rattle(p, pos, cell, rng, n_rattle, rattle_range, blmin, blmax, inside=None, ntrial=200):
    """GOFEE RattleMutation: each search atom with prob. n_rattle/n is moved uniformly within a
    sphere of radius rattle_range; a trial is kept if it respects blmin (and, for non-bulk, has a
    neighbour within blmax) w.r.t. all other atoms; the first valid of ntrial trials is used."""
    pos = np.array(pos, float)
    s = p.search_idx
    prob = min(1.0, n_rattle / len(s))
    idx = s[rng.random(len(s)) < prob]
    if len(idx) == 0:
        idx = np.array([rng.choice(s)])
    need = not p.variable_cell
    rmax = blmax * 2.0 * float(np.max(p.rcov))
    allidx = np.arange(p.n)
    for i in rng.permutation(idx):
        v = rng.normal(size=(ntrial, 3))
        v *= (rattle_range * rng.random(ntrial) ** (1 / 3) / np.linalg.norm(v, axis=1))[:, None]
        X = pos[i] + v
        good = np.ones(ntrial, bool) if inside is None else inside(X)
        oth = allidx[allidx != i]
        d = mic_dist(p, X, pos[oth], cell, rmax)
        rc = p.rcov[i] + p.rcov[oth]
        good &= np.all(d >= blmin * rc[None, :], axis=1)
        if need:
            good &= np.any(d < blmax * rc[None, :], axis=1)
        hit = np.nonzero(good)[0]
        if len(hit):
            pos[i] = X[hit[0]]
    return pos


def valid(p, pos, cell, blmin, blmax):
    """No search atom closer than blmin*(rc_i+rc_j) to any atom; non-bulk: every search atom
    has a neighbour within blmax*(rc_i+rc_j) (no detached atoms)."""
    s = p.search_idx
    rmax = blmax * 2.0 * float(np.max(p.rcov))
    d = mic_dist(p, pos[s], pos, cell, rmax)
    d[np.arange(len(s)), s] = np.inf
    rc = p.rcov[s][:, None] + p.rcov[None, :]
    if np.any(d < blmin * rc):
        return False
    if not p.variable_cell and not np.all(np.any(d < blmax * rc, axis=1)):
        return False
    return True
