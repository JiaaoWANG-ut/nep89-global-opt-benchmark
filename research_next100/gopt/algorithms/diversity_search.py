"""Adaptive quality-diversity search over relaxed structures.

The archive is a bounded set of geometrically distinct, locally relaxed
structures, not a census of energy-defined basins.  Parent selection combines
energy rank, descriptor isolation and how often the entry has been tried.
"""
import numpy as np

from gopt import ops
from gopt.problem import min_image_distances

LABEL = "Adaptive diversity archive search"
FAMILY = "innovative"
REFERENCE = ("Quality-diversity inspiration: Mouret & Clune, arXiv:1504.04909 "
             "(2015), MAP-Elites; Lehman & Stanley, Evolutionary Computation "
             "19, 189-223 (2011), novelty search. This implementation is neither "
             "a reproduction nor a claim of novelty.")
DESCRIPTION = (
    "A bounded archive of relaxed structures uses species-resolved geometric "
    "descriptors and adaptive nearest-neighbour spacing. Similar candidates "
    "compete on energy; novel candidates replace crowded archive members, "
    "with the lowest-energy member protected. Parent choice balances energy "
    "rank, isolation and use count; perturbation size adapts to archive "
    "admission, with occasional independent random starts.")
DEFAULTS = dict(capacity=16, step=0.6, strain=0.05, p_random=0.12,
                p_swap=0.15, p_relocate=0.15, adapt_every=8)


class Descriptor:
    """Block-balanced, species-resolved distances and local host environments."""

    def __init__(self, p):
        self.p = p
        self.idx = np.asarray(p.search_idx, int)
        sym = np.asarray(p.symbols)
        self.species = sorted(set(sym[self.idx]))
        self.search_groups = [np.flatnonzero(sym[self.idx] == s) for s in self.species]
        self.host_groups = [np.flatnonzero(sym[:p.n_sub] == s)
                            for s in sorted(set(sym[:p.n_sub]))]
        self.pair_groups = []
        i, j = np.triu_indices(len(self.idx), 1)
        for a, ga in enumerate(self.search_groups):
            for gb in self.search_groups[a:]:
                m = (np.isin(i, ga) & np.isin(j, gb)) | (np.isin(i, gb) & np.isin(j, ga))
                if m.any():
                    self.pair_groups.append((i[m], j[m]))

    def __call__(self, pos, cell):
        p = self.p
        x = np.asarray(pos)[self.idx]
        blocks = []
        d = min_image_distances(x, x, cell, p.pbc)
        for i, j in self.pair_groups:
            blocks.append(np.sort(d[i, j]))
        if self.host_groups:
            host_d = min_image_distances(x, np.asarray(pos)[:p.n_sub], cell, p.pbc)
            for a in self.search_groups:
                for b in self.host_groups:
                    blocks.append(np.sort(host_d[np.ix_(a, b)].min(axis=1)))
        if p.variable_cell:
            # Volume and reduced lattice lengths distinguish, imperfectly,
            # periodic structures with the same in-cell pair distances.
            from ase.geometry import minkowski_reduce
            reduced = np.asarray(minkowski_reduce(cell, p.pbc)[0])
            blocks.append(np.r_[abs(np.linalg.det(cell)) ** (1 / 3),
                                np.sort(np.linalg.norm(reduced, axis=1))])
        if not blocks:
            blocks.append(np.zeros(1))
        # Euclidean distance of the concatenation is the RMS block distance.
        n = len(blocks)
        return np.concatenate([np.asarray(b) / np.sqrt(n * len(b)) for b in blocks])


class Archive:
    """Energy competition within a niche and crowding replacement between niches."""

    def __init__(self, capacity, min_spacing=0.03):
        self.capacity = capacity
        self.min_spacing = min_spacing
        self.entries = []

    def distances(self):
        f = np.array([a["fp"] for a in self.entries])
        return np.linalg.norm(f[:, None, :] - f[None, :, :], axis=2)

    def add(self, result, fp):
        entry = dict(pos=np.array(result.pos, copy=True), cell=np.array(result.cell, copy=True),
                     e=float(result.e), fp=np.array(fp, copy=True), uses=0)
        if not self.entries:
            self.entries.append(entry)
            return "novel"
        dist = np.array([np.linalg.norm(fp - a["fp"]) for a in self.entries])
        nearest = int(np.argmin(dist))
        if len(self.entries) > 1:
            dm = self.distances()
            np.fill_diagonal(dm, np.inf)
            spacing = max(self.min_spacing, 0.5 * float(np.median(dm.min(axis=1))))
        else:
            spacing = self.min_spacing
        if dist[nearest] < spacing:
            if entry["e"] < self.entries[nearest]["e"]:
                self.entries[nearest] = entry
                return "improved"
            return "rejected"
        if len(self.entries) < self.capacity:
            self.entries.append(entry)
            return "novel"
        # Only add a new niche if it is more isolated than the most crowded
        # existing pair. Protect the current energy elite in all cases.
        i, j = np.unravel_index(np.argmin(dm), dm.shape)
        elite = int(np.argmin([a["e"] for a in self.entries]))
        if dist[nearest] <= dm[i, j] and entry["e"] >= self.entries[elite]["e"]:
            return "rejected"
        victim = j if i == elite else i if j == elite else (
            i if self.entries[i]["e"] >= self.entries[j]["e"] else j)
        self.entries[victim] = entry
        return "novel"

    def choose(self, rng):
        n = len(self.entries)
        energies = np.array([a["e"] for a in self.entries])
        order = np.argsort(energies, kind="stable")
        rank = np.empty(n, int)
        rank[order] = np.arange(n)
        if n == 1:
            isolation = np.ones(1)
        else:
            dm = self.distances()
            np.fill_diagonal(dm, np.inf)
            isolation = dm.min(axis=1)
            isolation = np.clip(isolation / max(float(np.median(isolation)), 1e-12), 0.25, 4)
        uses = np.array([a["uses"] for a in self.entries])
        weights = np.sqrt(isolation) / np.sqrt((rank + 1) * (uses + 1))
        i = int(rng.choice(n, p=weights / weights.sum()))
        self.entries[i]["uses"] += 1
        return self.entries[i]


def _fixed_positions(p, pos):
    pos = np.array(pos, copy=True)
    frozen = np.flatnonzero(~np.asarray(p.free_mask, bool))
    if len(frozen):
        pos[frozen] = np.asarray(p.sub_pos)[frozen]
    return pos


def _move(p, parent, rng, step, strain, p_swap, p_relocate):
    pos, cell = parent["pos"].copy(), parent["cell"].copy()
    if p.variable_cell:
        pos, cell = ops.strain_cell(pos, cell, rng, strain)
        pos = _fixed_positions(p, pos)
    u = rng.random()
    if u < p_swap:
        pos = ops.swap_unlike(p, pos, rng)
        pos = ops.displace(p, pos, rng, 0.25 * step, cell=cell)
    elif u < p_swap + p_relocate and not p.variable_cell and len(p.search_idx) > 1:
        idx = rng.choice(p.search_idx, size=1, replace=False)
        pos = ops.relocate_atoms(p, pos, rng, idx, cell)
    else:
        pos = ops.displace(p, pos, rng, step, cell=cell)
    if not p.variable_cell and p.category != "bulk":
        lost = ops.detached_atoms(p, pos, cell)
        if len(lost):
            pos = ops.relocate_atoms(p, pos, rng, lost, cell)
    return _fixed_positions(p, pos), cell


def run(ctx, rng, capacity=16, step=0.6, strain=0.05, p_random=0.12,
        p_swap=0.15, p_relocate=0.15, adapt_every=8):
    if (not np.isfinite([capacity, step, strain, p_random, p_swap, p_relocate,
                         adapt_every]).all()
            or capacity < 2 or int(capacity) != capacity or adapt_every < 1
            or int(adapt_every) != adapt_every or step <= 0 or strain < 0
            or not 0 <= p_random <= 1 or min(p_swap, p_relocate) < 0
            or p_swap + p_relocate > 1):
        raise ValueError("invalid diversity search parameters")
    p = ctx.p
    descriptor, archive = Descriptor(p), Archive(int(capacity))
    ntrial = admissions = window_admissions = 0
    while True:
        if not archive.entries or rng.random() < p_random:
            pos, cell = p.random_structure(rng)
            pos = _fixed_positions(p, pos)
        else:
            pos, cell = _move(p, archive.choose(rng), rng, step, strain,
                              p_swap, p_relocate)
        result = ctx.relax(pos, cell)
        outcome = "rejected"
        if (not result.aborted and np.isfinite(result.e) and result.e < 1e5
                and np.isfinite(result.pos).all() and np.isfinite(result.cell).all()
                and abs(np.linalg.det(result.cell)) > 1e-12):
            result.pos = _fixed_positions(p, result.pos)
            fp = descriptor(result.pos, result.cell)
            if np.isfinite(fp).all():
                outcome = archive.add(result, fp)
        ntrial += 1
        if outcome != "rejected":
            admissions += 1
            window_admissions += 1
        if ntrial % adapt_every == 0:
            # Low admission indicates repeatedly rediscovering occupied niches.
            step *= 1.2 if window_admissions / adapt_every < 0.3 else 0.9
            step = float(np.clip(step, 0.1, 2.0))
            window_admissions = 0
        ctx.stats.update(archive_size=len(archive.entries), trials=ntrial,
                         admissions=admissions, step=step)
