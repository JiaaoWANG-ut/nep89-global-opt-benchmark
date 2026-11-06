"""Free (no energy calls) structure moves for the `race` algorithm."""
import numpy as np

from .. import ops
from ..problem import min_image_distances


def _local_set(p, pos, cell, center, radius):
    """Indices of substrate atoms within `radius` of `center` (minimum image)."""
    if not p.n_sub:
        return np.zeros(0, int)
    d = min_image_distances(center, pos[:p.n_sub], cell, p.pbc)[0]
    return np.flatnonzero(d < radius)


def site_relocate(p, pos, rng, idx, cell=None, n_cand=40, sub_anchor=0.3):
    """Cut atoms `idx` and regrow them one by one at the best-coordinated free site.

    Candidate sites are put at bonding distance from random anchor atoms (search
    atoms, and for supported/confined clusters also nearby host atoms); sites closer
    than 0.85 x the covalent-radius sum to any atom are rejected, the rest is scored
    by the number of neighbours within 1.3 x the radius sum (search and host atoms)
    plus a small random tie-breaker. Connected problems keep the atom bonded to the
    other search atoms. Returns new positions (or None if no site was found).
    """
    cell = p.cell0 if cell is None else cell
    pos = ops.unwrap_search(p, pos, cell)
    s = p.search_idx
    idx = [int(i) for i in np.atleast_1d(idx)]
    moving = set(idx)
    placed = [i for i in s if i not in moving]
    if not placed:
        return None
    c = pos[placed].mean(axis=0)
    rad = float(np.sqrt(((pos[placed] - c) ** 2).sum(axis=1).max())) if len(placed) > 1 else 0.0
    host = _local_set(p, pos, cell, c, rad + 7.0)
    host_anchor = host[min_image_distances(c, pos[host], cell, p.pbc)[0] < rad + 3.5] \
        if len(host) else host
    out = np.array(pos)
    for i in idx:
        nb = np.concatenate([np.array(placed, int), host])
        nb_search = len(placed)
        ri = p.rcov[i]
        n_h = int(round(n_cand * sub_anchor)) if len(host_anchor) else 0
        a_s = rng.choice(np.array(placed), size=n_cand - n_h)
        anchors = a_s if not n_h else np.concatenate([a_s, rng.choice(host_anchor, size=n_h)])
        u = ops.random_unit(rng, len(anchors))
        if p.category == "surface":
            u[:, 2] = np.abs(u[:, 2])
        x = out[anchors] + u * ((ri + p.rcov[anchors]) * rng.uniform(0.95, 1.05, len(anchors)))[:, None]
        d = min_image_distances(x, out[nb], cell, p.pbc)
        rs = ri + p.rcov[nb]
        ok = np.all(d >= 0.85 * rs, axis=1)
        bonded = d < 1.3 * rs
        if p.connected:
            ok &= bonded[:, :nb_search].any(axis=1)
        if not ok.any():
            res = ops.relocate_atoms(p, out, rng, [i], cell)
            out = res
            placed.append(i)
            continue
        score = bonded.sum(axis=1) + 0.5 * rng.random(len(x))
        score[~ok] = -1.0
        out[i] = x[int(np.argmax(score))]
        placed.append(i)
    return out


def rigid_move(p, pos, rng, cell=None, shift=2.0):
    """Rigid rotation + translation of the search cluster (supported: about z, in-plane)."""
    cell = p.cell0 if cell is None else cell
    pos = ops.unwrap_search(p, pos, cell)
    s = p.search_idx
    c = pos[s].mean(axis=0)
    if p.category == "surface":
        rot = ops.rotation_about([0, 0, 1], rng.uniform(0, 2 * np.pi))
        t = rng.normal(size=3)
        t[2] = 0.0
    else:
        rot = ops.rotation_about(ops.random_unit(rng), rng.uniform(0, np.pi))
        t = rng.normal(size=3)
    t *= shift * rng.random() / max(np.linalg.norm(t), 1e-9)
    pos[s] = (pos[s] - c) @ rot.T + c + t
    return ops.repair_overlaps(p, pos, cell)
