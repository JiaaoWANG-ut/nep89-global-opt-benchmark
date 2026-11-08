"""Contextual, transition-novelty adaptive basin hopping (experimental)."""
import math
import numbers

import numpy as np

from .. import ops
from ..context import BAD_ENERGY
from .diversity_v2 import Descriptor
from ._race_moves import rigid_move

LABEL = "Transition-context adaptive basin hopping"
FAMILY = "experimental"
REFERENCE = ("Wales & Doye, J. Phys. Chem. A 101, 5111 (1997) [basin hopping]; "
             "Fialho et al., Ann. Math. Artif. Intell. 60, 25 (2010) "
             "[adaptive operator selection]; Mouret & Clune, arXiv:1504.04909 "
             "[quality-diversity inspiration]")
DESCRIPTION = (
    "A Metropolis basin walk learns discounted operator utility per ionic step, "
    "conditioned on whether the parent is near the best energy and repeatedly "
    "visited. Utility combines best-energy improvement and first observation "
    "of a directed transition between approximate geometric niches. A bounded "
    "transition memory, forced exploration and random starts limit lock-in.")
DEFAULTS = dict(kT=0.15, step=0.6, strain=0.05, spacing=0.08,
                energy_bin=0.5, capacity=32, decay=0.97, explore=0.15,
                contextual=True, transition_reward=True)


class TransitionMemory:
    """Bounded approximate niche graph; these are not proven attraction basins."""

    def __init__(self, capacity, spacing):
        self.capacity = capacity
        self.spacing = spacing
        self.nodes = {}
        self.edges = {}
        self.next_id = 0
        self.clock = 0

    def observe(self, fp, energy, protect=None):
        self.clock += 1
        nearest = None
        if self.nodes:
            nearest = min(self.nodes, key=lambda i: np.linalg.norm(fp - self.nodes[i]["fp"]))
        if (nearest is not None
                and np.linalg.norm(fp - self.nodes[nearest]["fp"]) < self.spacing
                and abs(energy - self.nodes[nearest]["energy"]) < 0.1):
            node = self.nodes[nearest]
            node["visits"] += 1
            node["time"] = self.clock
            return nearest
        if len(self.nodes) >= self.capacity:
            candidates = [i for i in self.nodes if i != protect]
            victim = min(candidates, key=lambda i: self.nodes[i]["time"])
            del self.nodes[victim]
            self.edges = {edge: n for edge, n in self.edges.items() if victim not in edge}
        i = self.next_id
        self.next_id += 1
        self.nodes[i] = dict(fp=np.array(fp, copy=True), energy=float(energy),
                             visits=1, time=self.clock)
        return i

    def transition(self, source, target):
        edge = (source, target)
        count = self.edges.get(edge, 0)
        self.edges[edge] = count + 1
        return source != target and count == 0


class ContextualSelector:
    """Discounted utility/cost estimates plus an exploration bonus and floor."""

    def __init__(self, arms, decay, explore):
        self.arms = list(arms)
        self.decay, self.explore = decay, explore
        shape = (4, len(arms))
        self.count = np.zeros(shape)
        self.reward = np.zeros(shape)
        self.cost = np.zeros(shape)

    def choose(self, state, rng):
        if rng.random() < self.explore:
            return int(rng.integers(len(self.arms)))
        n = self.count[state]
        # A prior of one observation costing 20 steps prevents tiny samples
        # from looking infinitely efficient.
        rate = (self.reward[state] + 0.25) / (self.cost[state] + 20.0) * 20.0
        bonus = 0.4 * np.sqrt(np.log(2.0 + n.sum()) / (1.0 + n))
        score = rate + bonus
        best = np.flatnonzero(np.isclose(score, score.max(), rtol=0, atol=1e-12))
        return int(rng.choice(best))

    def update(self, state, arm, reward, cost):
        self.count *= self.decay
        self.reward *= self.decay
        self.cost *= self.decay
        self.count[state, arm] += 1
        self.reward[state, arm] += reward
        self.cost[state, arm] += max(1, cost)


def _valid(ctx, result):
    if result.aborted or not np.isfinite(result.e) or result.e >= BAD_ENERGY:
        return False
    pos, cell = np.asarray(result.pos), np.asarray(result.cell)
    if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(cell)):
        return False
    if abs(np.linalg.det(cell)) < 1e-6:
        return False
    if ctx.guard is not None:
        # Exclude minima still inside the framework's short-range guard or slab wall.
        energy, _, _, _ = ctx.guard(pos, cell)
        if not np.isfinite(energy) or energy > 1e-8:
            return False
    return True


def _arms(p):
    arms = ["small", "large", "random"]
    if p.n_search > 1 and not p.variable_cell:
        arms.append("relocate")
    if p.category in ("surface", "zeolite") and not p.variable_cell:
        arms.append("rigid")
    if len(set(p.search_symbols)) > 1:
        arms.append("swap")
    if p.variable_cell:
        arms.append("strain")
    return arms


def _context(energy, best, visits, energy_bin):
    return 2 * int(energy > best + energy_bin) + int(visits >= 3)


def _validate(kT, step, strain, spacing, energy_bin, capacity, decay, explore,
              contextual, transition_reward):
    values = dict(kT=kT, step=step, strain=strain, spacing=spacing,
                  energy_bin=energy_bin, decay=decay, explore=explore)
    for name, value in values.items():
        try:
            finite = (isinstance(value, numbers.Real) and not isinstance(value, bool)
                      and math.isfinite(value))
        except (OverflowError, TypeError, ValueError):
            finite = False
        if not finite:
            raise ValueError(f"{name} must be a finite real number")
    if (kT <= 0 or step <= 0 or strain < 0 or spacing <= 0 or energy_bin <= 0
            or not 0 < decay <= 1 or not 0 <= explore <= 1):
        raise ValueError("invalid transition_aos parameter range")
    if isinstance(capacity, bool) or not isinstance(capacity, numbers.Integral) or capacity < 2:
        raise ValueError("capacity must be an integer >= 2")
    if type(contextual) is not bool or type(transition_reward) is not bool:
        raise ValueError("contextual and transition_reward must be booleans")


def _move(p, parent, arm, rng, step, strain):
    if arm == "random":
        return p.random_structure(rng)
    pos, cell = np.array(parent.pos, copy=True), np.array(parent.cell, copy=True)
    if arm in ("small", "large"):
        pos = ops.displace(p, pos, rng, step * (0.5 if arm == "small" else 1.8), cell=cell)
    elif arm == "relocate":
        idx = rng.choice(p.search_idx, size=1, replace=False)
        pos = ops.relocate_atoms(p, pos, rng, idx, cell)
    elif arm == "rigid":
        pos = rigid_move(p, pos, rng, cell, shift=2 * step)
    elif arm == "swap":
        pos = ops.swap_unlike(p, pos, rng)
    elif arm == "strain":
        pos, cell = ops.strain_cell(pos, cell, rng, strain)
        pos = ops.displace(p, pos, rng, step * 0.5, cell=cell)
    return pos, cell


def run(ctx, rng, kT=0.15, step=0.6, strain=0.05, spacing=0.08,
        energy_bin=0.5, capacity=32, decay=0.97, explore=0.15,
        contextual=True, transition_reward=True):
    _validate(kT, step, strain, spacing, energy_bin, capacity, decay, explore,
              contextual, transition_reward)
    p = ctx.p
    descriptor = Descriptor(p)
    memory = TransitionMemory(capacity, spacing)
    selector = ContextualSelector(_arms(p), decay, explore)
    stats = ctx.stats
    stats.update(trials=0, accepted=0, invalid=0, novel_transitions=0,
                 arm_pulls={name: 0 for name in selector.arms},
                 arm_costs={name: 0 for name in selector.arms},
                 arm_rewards={name: 0.0 for name in selector.arms},
                 context_pulls=[0, 0, 0, 0])
    while True:
        pos, cell = p.random_structure(rng)
        cur = ctx.relax(pos, cell)
        if _valid(ctx, cur):
            break
        stats["invalid"] += 1
    current = memory.observe(descriptor(cur.pos, cur.cell), cur.e)
    best = float(cur.e)
    while True:
        # Four contexts: energy quality x repeat visits to this niche.
        state = (_context(cur.e, best, memory.nodes[current]["visits"], energy_bin)
                 if contextual else 0)
        a = selector.choose(state, rng)
        arm = selector.arms[a]
        pos, cell = _move(p, cur, arm, rng, step, strain)
        before = ctx.nsteps
        trial = ctx.relax(pos, cell)
        cost = ctx.nsteps - before
        stats["trials"] += 1
        stats["arm_pulls"][arm] += 1
        stats["arm_costs"][arm] += cost
        stats["context_pulls"][state] += 1
        reward = 0.0
        if _valid(ctx, trial):
            target = memory.observe(descriptor(trial.pos, trial.cell), trial.e, protect=current)
            novel = memory.transition(current, target)
            improvement = max(0.0, best - trial.e)
            reward = (0.5 * float(novel and transition_reward)
                      + 0.5 * min(improvement / energy_bin, 1.0))
            stats["novel_transitions"] += int(novel)
            best = min(best, float(trial.e))
            de = trial.e - cur.e
            if de <= 0 or rng.random() < np.exp(-de / kT):
                cur, current = trial, target
                stats["accepted"] += 1
        else:
            stats["invalid"] += 1
        selector.update(state, a, reward, cost)
        stats["arm_rewards"][arm] += reward
