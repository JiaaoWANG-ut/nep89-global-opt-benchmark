"""Diversity-archived, atom-matched path relinking for structure search."""
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from ase.geometry import find_mic
from scipy.optimize import linear_sum_assignment

from gopt import ops

LABEL = "Archive path relinking"
FAMILY = "innovative"
REFERENCE = "Path relinking is an established optimization concept; no novelty claim."
DESCRIPTION = (
    "A bounded quality-diversity archive of relaxed structures supplies pairs of "
    "parents. Same-species atoms are matched, and several intermediate geometries "
    "along their aligned displacement path are independently relaxed. A periodic "
    "immigrant maintains exploration; a small shake handles a singleton archive. "
    "Archive pruning protects the energy elite and balances energy rank with "
    "nearest-neighbour fingerprint distance. All relaxations use the shared budget."
)
DEFAULTS = dict(archive_size=8, initial=3, path_points=3, immigrant_every=6,
                diversity_weight=0.5, shake=0.35, rotations=3)

_BAD = 1e5


@dataclass
class _Member:
    pos: np.ndarray
    cell: np.ndarray
    e: float
    fp: np.ndarray


def _distance(p, a, b):
    if not len(a) and not len(b):
        return 0.0
    if len(a) != len(b):
        raise ValueError("fingerprints must have the same length")
    if p.variable_cell:
        azero = np.linalg.norm(a) < 1e-12
        bzero = np.linalg.norm(b) < 1e-12
        if azero or bzero:
            return 0.0 if azero and bzero else 1.0
    return max(0.0, ops.fp_distance(p, a, b))


def _insert(p, archive, member, capacity, diversity_weight):
    """Keep the best duplicate, then prune by rank and nearest-neighbour novelty."""
    for i, old in enumerate(archive):
        duplicate = abs(old.e - member.e) <= 0.01 and (
            _distance(p, old.fp, member.fp) < (0.01 if p.variable_cell else 0.03))
        if duplicate:
            if member.e < old.e:
                archive[i] = member
                archive.sort(key=lambda x: x.e)
                return True
            return False
    archive.append(member)
    archive.sort(key=lambda x: x.e)
    if len(archive) > capacity:
        n = len(archive)
        nearest = np.array([min(_distance(p, x.fp, y.fp)
                                for j, y in enumerate(archive) if i != j)
                            for i, x in enumerate(archive)])
        scale = max(float(nearest.max()), 1e-12)
        scores = (1.0 - diversity_weight) * (1.0 - np.arange(n) / (n - 1))
        scores += diversity_weight * nearest / scale
        scores[0] = np.inf
        removed = archive.pop(int(np.argmin(scores)))
        return removed is not member
    return True


def _match(ref, mov, groups, cell=None, pbc=None):
    """Assign donor atoms only within their chemical species."""
    perm = np.arange(len(ref))
    for group in groups:
        delta = mov[group][None, :, :] - ref[group][:, None, :]
        if cell is not None and np.any(pbc):
            delta = find_mic(delta.reshape(-1, 3), cell, pbc=pbc)[0].reshape(delta.shape)
        cost = np.sum(delta * delta, axis=2)
        rows, cols = linear_sum_assignment(cost)
        perm[group[rows]] = group[cols]
    return perm


def _aligned(p, a, b, rng, rotations):
    """Return matched target coordinates and displacement in A's Cartesian frame."""
    s = np.asarray(p.search_idx)[np.asarray(p.free_mask)[p.search_idx]]
    sym = np.asarray(p.symbols)[s]
    groups = [np.flatnonzero(sym == x) for x in sorted(set(sym))]
    if not len(s):
        return s, np.empty((0, 3)), np.empty((0, 3))

    free_cluster = p.category == "cluster" and p.n_sub == 0 and not p.variable_cell
    if free_cluster:
        ref = ops.unwrap_search(p, a.pos, a.cell)[s]
        mov = ops.unwrap_search(p, b.pos, b.cell)[s]
        center = ref.mean(axis=0)
        ref0, mov0 = ref - center, mov - mov.mean(axis=0)
        starts = [np.eye(3)] + [ops.random_rotation(rng) for _ in range(rotations)]
        best, best_cost = None, np.inf
        for rot in starts:
            x = mov0 @ rot
            for _ in range(5):
                perm = _match(ref0, x, groups)
                u, _, vt = np.linalg.svd(x[perm].T @ ref0)
                d = np.linalg.det(u @ vt)
                x = x @ (u @ np.diag([1., 1., d]) @ vt)
            perm = _match(ref0, x, groups)
            aligned = x[perm] + center
            cost = np.sum((aligned - ref) ** 2)
            if cost < best_cost:
                best, best_cost = aligned, cost
        return s, ref, best - ref

    ref = np.asarray(a.pos)[s]
    if p.variable_cell:
        mov = (np.asarray(b.pos)[s] @ np.linalg.inv(b.cell)) @ a.cell
    else:
        mov = np.asarray(b.pos)[s].copy()
    # Only a host-free fully periodic system has arbitrary origin symmetry.
    shifts = [np.zeros(3)]
    if p.variable_cell and p.n_sub == 0 and np.all(p.pbc) and np.all(p.free_mask):
        group = min(groups, key=len)
        shifts += [ref[group[0]] - mov[k] for k in group[:6]]
    best, best_cost = None, np.inf
    for shift in shifts:
        x = mov + shift
        perm = _match(ref, x, groups, a.cell, p.pbc)
        delta = x[perm] - ref
        if np.any(p.pbc):
            delta = find_mic(delta, a.cell, pbc=p.pbc)[0]
        cost = np.sum(delta * delta)
        if cost < best_cost:
            best, best_cost = delta, cost
    return s, ref, best


def _valid_cell(cell, reference):
    det, refdet = np.linalg.det(cell), np.linalg.det(reference)
    vol, vref = abs(det), abs(refdet)
    if (not np.isfinite(det) or not np.isfinite(refdet) or det * refdet <= 0
            or not 0.5 * vref < vol < 2 * vref):
        return False
    height = 1.0 / np.linalg.norm(np.linalg.inv(cell), axis=0).max()
    return height > 0.3 * vol ** (1 / 3)


def _path_geometry(p, a, b, aligned, fraction):
    """Build one image; only mobile search atoms and optionally the cell move."""
    s, ref, delta = aligned
    cell = np.array(a.cell, copy=True)
    pos = np.array(a.pos, copy=True)
    xyz = ref + fraction * delta
    if p.variable_cell:
        proposed = (1.0 - fraction) * a.cell + fraction * b.cell
        if _valid_cell(proposed, a.cell):
            cell = proposed
            xyz = (xyz @ np.linalg.inv(a.cell)) @ cell
    pos[s] = xyz
    pos = ops.repair_overlaps(p, pos, cell)
    # repair_overlaps moves search atoms; never allow it to move frozen atoms.
    pos[~p.free_mask] = a.pos[~p.free_mask]
    return pos, cell


def _member(p, res):
    if res.aborted or not np.isfinite(res.e) or res.e >= _BAD:
        return None
    pos, cell = np.array(res.pos, float), np.array(res.cell, float)
    if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(cell)):
        return None
    if np.any(~p.free_mask):
        frozen = np.flatnonzero(~p.free_mask)
        host = frozen[frozen < len(p.sub_pos)]
        pos[host] = p.sub_pos[host]
    if not p.variable_cell and abs(np.linalg.det(cell)) < 1e-8:
        return None
    # Relaxed bulk cells need not retain the initial volume, but must be usable.
    if p.variable_cell and not _valid_cell(cell, cell):
        return None
    fp = ops.fingerprint(p, pos, cell)
    if not np.all(np.isfinite(fp)):
        return None
    return _Member(pos, cell, float(res.e), fp)


def run(ctx, rng, archive_size=8, initial=3, path_points=3, immigrant_every=6,
        diversity_weight=0.5, shake=0.35, rotations=3):
    for name, value, minimum in (
            ("archive_size", archive_size, 2), ("initial", initial, 1),
            ("path_points", path_points, 1), ("immigrant_every", immigrant_every, 1),
            ("rotations", rotations, 0)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for name, value in (("diversity_weight", diversity_weight), ("shake", shake)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite real number")
        try:
            finite = np.isfinite(float(value))
        except (OverflowError, TypeError, ValueError):
            finite = False
        if not finite:
            raise ValueError(f"{name} must be a finite real number")
    if not 0 <= diversity_weight <= 1:
        raise ValueError("diversity_weight must be between 0 and 1")
    if shake <= 0 or float(shake) == 0:
        raise ValueError("shake must be positive")
    diversity_weight, shake = float(diversity_weight), float(shake)

    p = ctx.p
    capacity = archive_size
    archive = []
    stats = ctx.stats.setdefault("path_relink", {})
    for name in ("random", "paths", "images", "shake", "admitted", "invalid"):
        stats.setdefault(name, 0)

    def relax(pos, cell):
        res = ctx.relax(pos, cell)
        member = _member(p, res)
        if member is None:
            stats["invalid"] += 1
        elif _insert(p, archive, member, capacity, diversity_weight):
            stats["admitted"] += 1

    def immigrant():
        pos, cell = p.random_structure(rng)
        stats["random"] += 1
        relax(pos, cell)

    for _ in range(initial):
        immigrant()
    cycle = 0
    while True:
        cycle += 1
        if not archive or cycle % immigrant_every == 0:
            immigrant()
        elif len(archive) == 1:
            parent = archive[0]
            s = np.asarray(p.search_idx)[np.asarray(p.free_mask)[p.search_idx]]
            pos = ops.displace(p, parent.pos, rng, shake, idx=s, cell=parent.cell)
            pos[~p.free_mask] = parent.pos[~p.free_mask]
            stats["shake"] += 1
            relax(pos, parent.cell)
        else:
            ranks = 1.0 / (1.0 + np.arange(len(archive)))
            i = int(rng.choice(len(archive), p=ranks / ranks.sum()))
            a = archive[i]
            candidates = [j for j in range(len(archive)) if j != i]
            distances = np.array([_distance(p, a.fp, archive[j].fp) for j in candidates])
            weights = ranks[candidates] * (0.1 + distances / max(distances.max(), 1e-12))
            j = int(rng.choice(candidates, p=weights / weights.sum()))
            b = archive[j]
            aligned = _aligned(p, a, b, rng, rotations)
            stats["paths"] += 1
            for k in range(1, path_points + 1):
                pos, cell = _path_geometry(p, a, b, aligned, k / (path_points + 1))
                stats["images"] += 1
                relax(pos, cell)
