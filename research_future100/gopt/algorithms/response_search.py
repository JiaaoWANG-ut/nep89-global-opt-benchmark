"""Relaxation-response guided population search (experimental)."""
import numpy as np

from gopt import ops
from gopt.algorithms.diversity_search import Descriptor

LABEL = "Relaxation-response population search"
FAMILY = "innovative"
REFERENCE = "Experimental proposal; relationship to prior literature has not been established."
DESCRIPTION = (
    "A population of relaxed structures learns per-parent displacement directions "
    "from observed escapes. It reuses the relaxed structural response, rather than "
    "the initial perturbation, and increases amplitude after repeated basin returns."
)
DEFAULTS = dict(capacity=10, step=0.55, spacing=0.06, restart=0.15,
                direction_weight=0.75, strain=0.04)


def _project(p, pos, cell):
    """Constrain global proposals, not the subsequent local minimisation."""
    x = np.array(pos, dtype=float, copy=True)
    idx = np.asarray(p.search_idx, dtype=int)
    region = p.region
    if region["type"] == "sphere":
        center = np.asarray(region["center"], float)
        delta = x[idx] - center
        norm = np.linalg.norm(delta, axis=1)
        radius = float(region["radius"]) * (1 - 1e-12)
        x[idx] = center + delta * np.minimum(1, radius / np.maximum(norm, 1e-12))[:, None]
    elif region["type"] == "box":
        x[idx] = np.clip(x[idx], region["lo"], region["hi"])
    else:
        frac = x[idx] @ np.linalg.inv(cell)
        frac[:, p.pbc] %= 1.0
        frac[:, ~p.pbc] = np.clip(frac[:, ~p.pbc], 0.0, 1.0)
        x[idx] = frac @ cell
    frozen = np.flatnonzero(~p.free_mask)
    if len(frozen):
        x[frozen] = p.sub_pos[frozen]
    return x


def _rms(v):
    return float(np.sqrt(np.mean(np.sum(v * v, axis=1)))) if len(v) else 0.0


def _response(p, before, after, cell):
    """Index-preserving minimum-image response, with cluster translation removed."""
    idx = p.search_idx
    delta = after[idx] - before[idx]
    frac = delta @ np.linalg.inv(cell)
    frac[:, p.pbc] -= np.round(frac[:, p.pbc])
    delta = frac @ cell
    if p.category == "cluster" and len(idx) > 1:
        delta -= delta.mean(axis=0)
    return delta


def _learn(parent, response, escaped, base_step):
    """Return-to-basin increases step; an escape stores its relaxed direction."""
    if escaped:
        size = _rms(response)
        if size > 1e-8:
            direction = response / size
            old = parent["direction"]
            if old is not None:
                direction = 0.5 * old + 0.5 * direction
                size = _rms(direction)
                direction = direction / size if size > 1e-8 else None
            parent["direction"] = direction
        parent["step"] = max(base_step * 0.35, parent["step"] * 0.85)
        parent["escapes"] += 1
    else:
        parent["step"] = min(base_step * 4.0, parent["step"] * 1.25)
        parent["returns"] += 1


def _proposal(p, parent, rng, direction_weight, strain):
    pos, cell = parent["pos"].copy(), parent["cell"].copy()
    if p.variable_cell and strain:
        strained_pos, strained_cell = ops.strain_cell(pos, cell, rng, strain)
        old_volume = abs(np.linalg.det(cell))
        volume = abs(np.linalg.det(strained_cell))
        if np.isfinite(volume) and 0.5 * old_volume < volume < 2 * old_volume:
            heights = 1.0 / np.linalg.norm(np.linalg.inv(strained_cell).T, axis=1)
            if heights.min() >= 0.3 * volume ** (1 / 3):
                pos, cell = strained_pos, strained_cell
    idx = p.search_idx
    noise = rng.normal(size=(len(idx), 3))
    if p.category == "cluster" and len(idx) > 1:
        noise -= noise.mean(axis=0)
    size = _rms(noise)
    if size > 1e-12:
        noise /= size
    direction = parent["direction"]
    if direction is not None and direction_weight > 0:
        noise = (1 - direction_weight) * noise + direction_weight * direction
        size = _rms(noise)
        if size > 1e-12:
            noise /= size
    pos[idx] += parent["step"] * noise
    pos = _project(p, pos, cell)
    pos = ops.repair_overlaps(p, pos, cell=cell, iters=20)
    return _project(p, pos, cell), cell


def _entry(result, fp, step):
    return dict(pos=result.pos.copy(), cell=result.cell.copy(), e=float(result.e),
                fp=fp.copy(), step=step, direction=None, uses=0, escapes=0, returns=0)


def _insert(pop, result, fp, capacity, spacing, step):
    candidate = _entry(result, fp, step)
    if not pop:
        pop.append(candidate)
        return "new"
    distances = np.array([np.linalg.norm(fp - a["fp"]) for a in pop])
    nearest = int(np.argmin(distances))
    if distances[nearest] < spacing:
        if result.e < pop[nearest]["e"]:
            pop[nearest] = candidate
            return "improved"
        return "duplicate"
    if len(pop) < capacity:
        pop.append(candidate)
    else:
        # Protect the energy elite; replace the worst-energy nonelite.
        elite = int(np.argmin([a["e"] for a in pop]))
        others = [i for i in range(len(pop)) if i != elite]
        victim = max(others, key=lambda i: pop[i]["e"])
        pop[victim] = candidate
    return "new"


def run(ctx, rng, capacity=10, step=0.55, spacing=0.06, restart=0.15,
        direction_weight=0.75, strain=0.04):
    vals = [capacity, step, spacing, restart, direction_weight, strain]
    if (not np.isfinite(vals).all() or int(capacity) != capacity or capacity < 2
            or step <= 0 or spacing <= 0 or not 0 <= restart <= 1
            or not 0 <= direction_weight <= 1 or not 0 <= strain <= 0.2):
        raise ValueError("invalid response search parameters")
    p = ctx.p
    descriptor, pop = Descriptor(p), []
    trials = escapes = returns = directed = started = 0
    ctx.stats.update(trials=0, population=0, escapes=0, returns=0,
                     proposals_started=0, directed_proposals=0)
    while ctx.remaining() > 0:
        parent = None
        if not pop or rng.random() < restart:
            pos, cell = p.random_structure(rng)
            pos = _project(p, pos, cell)
        else:
            energies = np.array([a["e"] for a in pop])
            order = np.argsort(energies, kind="stable")
            ranks = np.empty(len(pop), int)
            ranks[order] = np.arange(len(pop))
            weights = 1.0 / np.sqrt((1 + ranks) * (1 + np.array([a["uses"] for a in pop])))
            parent = pop[int(rng.choice(len(pop), p=weights / weights.sum()))]
            parent["uses"] += 1
            if parent["direction"] is not None and direction_weight > 0:
                directed += 1
            pos, cell = _proposal(p, parent, rng, direction_weight, strain)
        started += 1
        # A local relaxation can exhaust the budget before returning a result.
        ctx.stats.update(proposals_started=started, directed_proposals=directed)
        result = ctx.relax(pos, cell)
        trials += 1
        if (not result.aborted and np.isfinite(result.e) and result.e < 1e5
                and np.isfinite(result.pos).all() and np.isfinite(result.cell).all()
                and abs(np.linalg.det(result.cell)) > 1e-12):
            fp = descriptor(result.pos, result.cell)
            if np.isfinite(fp).all():
                if parent is not None:
                    escaped = np.linalg.norm(fp - parent["fp"]) >= spacing
                    response = _response(p, parent["pos"], result.pos, result.cell)
                    _learn(parent, response, escaped, step)
                    escapes += int(escaped)
                    returns += int(not escaped)
                _insert(pop, result, fp, int(capacity), spacing, step)
        ctx.stats.update(trials=trials, population=len(pop), escapes=escapes,
                         returns=returns, directed_proposals=directed)
