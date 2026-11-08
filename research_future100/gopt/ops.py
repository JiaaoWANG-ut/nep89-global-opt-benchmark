"""Generic structure operators shared by the algorithms (no energy calls here)."""
import numpy as np
from ase.data import atomic_masses

from .problem import min_image_distances

KB = 8.617333262e-5  # eV/K


# ---------------------------------------------------------------- geometry
def search_pos(p, pos):
    return np.asarray(pos)[p.search_idx]


def center_of_search(p, pos, cell=None):
    """Center of the search atoms (unwrapped around the first search atom)."""
    cell = p.cell0 if cell is None else cell
    s = np.asarray(pos)[p.search_idx]
    if not np.any(p.pbc):
        return s.mean(axis=0)
    ref = s[0]
    d = s - ref
    inv = np.linalg.inv(cell)
    f = d @ inv
    for k in range(3):
        if p.pbc[k]:
            f[:, k] -= np.round(f[:, k])
    return ref + (f @ cell).mean(axis=0)


def unwrap_search(p, pos, cell=None):
    """Search-atom coordinates made contiguous with minimum-image vectors (copy)."""
    cell = p.cell0 if cell is None else cell
    pos = np.array(pos, float)
    s = p.search_idx
    if not np.any(p.pbc) or len(s) == 0:
        return pos
    c = center_of_search(p, pos, cell)
    d = pos[s] - c
    inv = np.linalg.inv(cell)
    f = d @ inv
    for k in range(3):
        if p.pbc[k]:
            f[:, k] -= np.round(f[:, k])
    pos[s] = c + f @ cell
    return pos


def random_unit(rng, n=None):
    v = rng.normal(size=(3,) if n is None else (n, 3))
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([
        [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
        [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
        [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d]])


def rotation_about(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


# ---------------------------------------------------------------- repair
def repair_overlaps(p, pos, cell=None, factor=1.0, iters=50):
    """Push apart search atoms that sit closer than factor*dmin (no energy calls)."""
    cell = p.cell0 if cell is None else cell
    pos = np.array(pos, float)
    s = p.search_idx
    is_search = np.zeros(p.n, bool)
    is_search[s] = True
    lim_all = factor * p.dmin_factor * (p.rcov[s][:, None] + p.rcov[None, :])
    inv = np.linalg.inv(cell)
    for _ in range(iters):
        d = min_image_distances(pos[s], pos, cell, p.pbc)
        d[np.arange(len(s)), s] = np.inf
        bad = d < lim_all
        if not bad.any():
            return pos
        for a, j in zip(*np.where(bad)):
            i = s[a]
            v = pos[i] - pos[j]
            if np.any(p.pbc):
                f = v @ inv
                for k in range(3):
                    if p.pbc[k]:
                        f[k] -= np.round(f[k])
                v = f @ cell
            nv = np.linalg.norm(v)
            v = v / nv if nv > 1e-8 else np.array([np.cos(i), np.sin(i), 0.3]) / 1.044
            push = 0.5 * (lim_all[a, j] - min(nv, lim_all[a, j])) + 0.02
            pos[i] += v * push
            if is_search[j]:
                pos[j] -= v * push
    return pos


def detached_atoms(p, pos, cell=None, factor=1.6):
    """Indices of search atoms not bonded (within factor*rcov sum) to any other atom."""
    cell = p.cell0 if cell is None else cell
    s = p.search_idx
    d = min_image_distances(pos[s], pos, cell, p.pbc)
    d[np.arange(len(s)), s] = np.inf
    lim = factor * (p.rcov[s][:, None] + p.rcov[None, :])
    return s[~np.any(d < lim, axis=1)]


# ---------------------------------------------------------------- moves
def displace(p, pos, rng, step, idx=None, cell=None):
    """Uniform random displacement inside a sphere of radius ``step`` for atoms idx."""
    pos = np.array(pos, float)
    idx = p.search_idx if idx is None else np.asarray(idx)
    r = step * rng.random(len(idx)) ** (1 / 3)
    pos[idx] += random_unit(rng, len(idx)) * r[:, None]
    return repair_overlaps(p, pos, cell)


def rotate_subset(p, pos, rng, idx, angle=None, cell=None):
    """Rotate atoms idx rigidly about the search-atom centre."""
    cell = p.cell0 if cell is None else cell
    pos = unwrap_search(p, pos, cell)
    c = center_of_search(p, pos, cell)
    rot = random_rotation(rng) if angle is None else rotation_about(random_unit(rng), angle)
    if p.category in ("surface",):
        rot = rotation_about([0, 0, 1], rng.uniform(0, 2 * np.pi) if angle is None else angle)
    pos[idx] = (pos[idx] - c) @ rot.T + c
    return repair_overlaps(p, pos, cell)


def swap_unlike(p, pos, rng, nswap=1):
    """Exchange positions of unlike search atoms (no-op for one species)."""
    pos = np.array(pos, float)
    sym = np.array(p.symbols)
    s = p.search_idx
    if len(set(sym[s])) < 2:
        return pos
    for _ in range(nswap):
        i = rng.choice(s)
        cand = s[sym[s] != sym[i]]
        j = rng.choice(cand)
        pos[[i, j]] = pos[[j, i]]
    return pos


def relocate_atoms(p, pos, rng, idx, cell=None, shell=True):
    """Move atoms idx to random positions on the surface of the rest of the search set.

    The new position is placed at a bonding distance from a randomly chosen
    other search atom, away from the centre, avoiding overlaps.
    """
    cell = p.cell0 if cell is None else cell
    pos = unwrap_search(p, pos, cell)
    s = p.search_idx
    idx = list(np.atleast_1d(idx))
    rest = [i for i in s if i not in idx]
    if not rest:
        res = p.place(pos, idx, rng, cell)
        return pos if res is None else res
    c = pos[rest].mean(axis=0)
    out = np.array(pos)
    for i in idx:
        placed = False
        others = np.delete(np.arange(p.n), i)
        dm = p.dmin(i, others)
        xo = out[others]
        for _ in range(200):
            j = rng.choice(rest)
            u = pos[j] - c
            n = np.linalg.norm(u)
            u = u / n if n > 1e-6 and shell else random_unit(rng)
            u = u + 0.6 * random_unit(rng)
            if p.category == "surface" and u[2] < 0:
                u[2] = -u[2]
            u /= np.linalg.norm(u)
            x = out[j] + u * (p.rcov[i] + p.rcov[j]) * rng.uniform(0.95, 1.1)
            d = min_image_distances(x, xo, cell, p.pbc)[0]
            if np.all(d >= dm):
                out[i] = x
                placed = True
                break
        if not placed:
            res = p.place(out, [i], rng, cell)
            if res is not None:
                out = res
        rest.append(i)
    return out


def cut_and_splice(p, pos_a, pos_b, rng, cell=None):
    """Deaven-Ho crossover with composition repair.

    A random plane through the centre of parent A (vertical plane for supported
    clusters) keeps A's atoms on the positive side; the rest is filled with B's
    atoms from the negative side of a parallel plane through B's centre (B is
    translated onto A's centre). Species counts are always preserved.
    """
    cell = p.cell0 if cell is None else cell
    a = unwrap_search(p, pos_a, cell)
    b = unwrap_search(p, pos_b, cell)
    s = p.search_idx
    sym = np.array(p.symbols)
    n = random_unit(rng)
    if p.category == "surface":
        n[2] = 0.0
        n /= np.linalg.norm(n)
    ca, cb = a[s].mean(axis=0), b[s].mean(axis=0)
    pa = (a[s] - ca) @ n
    pb = (b[s] - cb) @ n
    need = {}
    for x in p.search_symbols:
        need[x] = need.get(x, 0) + 1
    chosen = []
    n_from_a = int(np.clip((pa >= 0).sum(), 1, len(s) - 1))
    for k in np.argsort(-pa)[:n_from_a]:
        i = s[k]
        if need[sym[i]] > 0:
            chosen.append((sym[i], a[i]))
            need[sym[i]] -= 1
    for k in np.argsort(pb):
        i = s[k]
        if need[sym[i]] > 0:
            chosen.append((sym[i], b[i] - cb + ca))
            need[sym[i]] -= 1
    child = np.array(a)
    slots = {x: [i for i in s if sym[i] == x] for x in set(sym[s])}
    for x, xyz in chosen:
        child[slots[x].pop(0)] = xyz
    return repair_overlaps(p, child, cell)


def strain_cell(pos, cell, rng, sigma=0.1):
    """Random symmetric strain of the cell, atoms follow (fractional coordinates kept)."""
    eps = rng.normal(0, sigma, (3, 3))
    eps = 0.5 * (eps + eps.T)
    new = cell @ (np.eye(3) + eps)
    frac = np.asarray(pos) @ np.linalg.inv(cell)
    return frac @ new, new


# ---------------------------------------------------------------- dynamics
def mb_velocities(p, temperature, rng, idx=None):
    """Maxwell-Boltzmann velocities (A/fs) at temperature (K) for atoms idx (default: free)."""
    m = atomic_masses[p.numbers]
    v = np.zeros((p.n, 3))
    idx = np.where(p.free_mask)[0] if idx is None else np.asarray(idx)
    # sigma = sqrt(kT/m) in A/fs with kT in eV, m in amu
    sig = np.sqrt(KB * temperature / m[idx] / 103.642697)
    v[idx] = rng.normal(size=(len(idx), 3)) * sig[:, None]
    v[idx] -= v[idx].mean(axis=0)
    return v


# ---------------------------------------------------------------- similarity
def fingerprint(p, pos, cell=None, rmax=6.0, nbins=60):
    """Structure fingerprint.

    Non-bulk: sorted minimum-image distances among search atoms plus sorted
    distances from each search atom to its nearest substrate atom.
    Bulk: per-species-pair Gaussian-smeared RDF (cell invariant).
    """
    cell = p.cell0 if cell is None else cell
    s = p.search_idx
    if not p.variable_cell:
        d = min_image_distances(pos[s], pos[s], cell, p.pbc)
        iu = np.triu_indices(len(s), 1)
        fp = [np.sort(d[iu])]
        if p.n_sub:
            dsub = min_image_distances(pos[s], pos[: p.n_sub], cell, p.pbc).min(axis=1)
            fp.append(np.sort(dsub))
        return np.concatenate(fp)
    # bulk RDF fingerprint
    frac = np.asarray(pos) @ np.linalg.inv(cell)
    if np.ptp(frac, axis=0).max() >= 1.0:
        # the image shifts below only cover fractional separations < 1
        frac = frac - np.floor(frac)
    rec = np.linalg.inv(cell).T
    heights = 1.0 / np.linalg.norm(rec, axis=1)
    nrep = np.ceil(rmax / heights).astype(int)
    shifts = np.array([(i, j, k) for i in range(-nrep[0], nrep[0] + 1)
                       for j in range(-nrep[1], nrep[1] + 1)
                       for k in range(-nrep[2], nrep[2] + 1)], float)
    sym = np.array(p.symbols)
    species = sorted(set(sym))
    edges = np.linspace(0.5, rmax, nbins + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    out = []
    vol = abs(np.linalg.det(cell))
    for ai, x in enumerate(species):
        for y in species[ai:]:
            ia = np.where(sym == x)[0]
            ib = np.where(sym == y)[0]
            dd = []
            for i in ia:
                dv = (frac[ib][None, :, :] + shifts[:, None, :] - frac[i]) @ cell
                r = np.linalg.norm(dv, axis=-1).ravel()
                dd.append(r[(r > 1e-6) & (r < rmax)])
            dd = np.concatenate(dd) if dd else np.zeros(0)
            h = np.exp(-((centers[:, None] - dd[None, :]) ** 2) / (2 * 0.08 ** 2)).sum(axis=1)
            h = h / (centers ** 2) / max(len(ia) * len(ib), 1) * vol
            out.append(h)
    v = np.concatenate(out)
    return v / (np.linalg.norm(v) + 1e-12)


def fp_distance(p, a, b):
    if p.variable_cell:
        return float(1.0 - np.dot(a, b))
    return float(np.mean(np.abs(a - b)))


def same_structure(p, e1, fp1, e2, fp2, de=0.01, dfp=None):
    if abs(e1 - e2) > de:
        return False
    dfp = (0.01 if p.variable_cell else 0.03) if dfp is None else dfp
    return fp_distance(p, fp1, fp2) < dfp
