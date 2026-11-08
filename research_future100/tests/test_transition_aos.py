import os
import unittest
from types import SimpleNamespace

import numpy as np

from gopt.algorithms import transition_aos as alg
from gopt.context import BAD_ENERGY, BudgetExhausted
from gopt.problem import Problem
from gopt.run import run_one


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, "benches-dev")


class MemoryTests(unittest.TestCase):
    def test_directed_edges_and_revisit(self):
        memory = alg.TransitionMemory(3, 0.1)
        a = memory.observe(np.array([0.0]), -1)
        self.assertEqual(a, memory.observe(np.array([0.01]), -1.02))
        self.assertEqual(memory.nodes[a]["visits"], 2)
        b = memory.observe(np.array([1.0]), -1, protect=a)
        self.assertTrue(memory.transition(a, b))
        self.assertFalse(memory.transition(a, b))
        self.assertTrue(memory.transition(b, a))
        self.assertFalse(memory.transition(a, a))

    def test_capacity_evicts_edges_not_current(self):
        memory = alg.TransitionMemory(2, 0.1)
        a = memory.observe(np.array([0.0]), 0)
        b = memory.observe(np.array([1.0]), 0, protect=a)
        memory.transition(a, b)
        c = memory.observe(np.array([2.0]), 0, protect=a)
        self.assertEqual(set(memory.nodes), {a, c})
        self.assertEqual(memory.edges, {})

    def test_energy_prevents_descriptor_collision(self):
        m = alg.TransitionMemory(3, 0.1)
        self.assertNotEqual(m.observe(np.array([0.0]), 0),
                            m.observe(np.array([0.0]), 1))


class SelectorTests(unittest.TestCase):
    def test_four_contexts(self):
        self.assertEqual(alg._context(0.0, 0.0, 1, 0.5), 0)
        self.assertEqual(alg._context(0.5, 0.0, 3, 0.5), 1)
        self.assertEqual(alg._context(0.6, 0.0, 1, 0.5), 2)
        self.assertEqual(alg._context(0.6, 0.0, 3, 0.5), 3)

    def test_context_and_cost_credit(self):
        selector = alg.ContextualSelector(["cheap", "expensive"], 1.0, 0)
        for _ in range(10):
            selector.update(0, 0, 1, 5)
            selector.update(0, 1, 1, 100)
        self.assertEqual(selector.choose(0, np.random.default_rng(1)), 0)
        self.assertTrue(np.all(selector.count[1] == 0))

    def test_discount_and_seeded_exploration(self):
        sel = alg.ContextualSelector(["a", "b"], 0.5, 1)
        sel.update(0, 0, 1, 10)
        sel.update(1, 1, 0, 20)
        self.assertEqual(sel.count[0, 0], 0.5)
        r1, r2 = np.random.default_rng(12), np.random.default_rng(12)
        self.assertEqual([sel.choose(0, r1) for _ in range(30)],
                         [sel.choose(0, r2) for _ in range(30)])


class ValidityTests(unittest.TestCase):
    def test_invalid_results_never_accepted(self):
        ctx = SimpleNamespace(guard=None)
        r = SimpleNamespace(aborted=False, e=-1.0, pos=np.zeros((1, 3)), cell=np.eye(3))
        self.assertTrue(alg._valid(ctx, r))
        for field, value in [("aborted", True), ("e", BAD_ENERGY),
                             ("e", np.nan), ("cell", np.zeros((3, 3))),
                             ("pos", np.full((1, 3), np.nan))]:
            other = SimpleNamespace(**vars(r))
            setattr(other, field, value)
            self.assertFalse(alg._valid(ctx, other))
        ctx.guard = lambda pos, cell: (0.01, None, None, None)
        self.assertFalse(alg._valid(ctx, r))

    def test_invalid_first_relaxation_retries_and_charges(self):
        p = Problem("cl_Pt13", root=BENCH)

        class FakeContext:
            guard = None
            nsteps = 0
            stats = {}

            def __init__(self):
                self.p = p
                self.stats = {}

            def relax(self, pos, cell):
                self.nsteps += 1
                if self.nsteps == 4:
                    raise BudgetExhausted
                return SimpleNamespace(pos=pos, cell=cell, e=BAD_ENERGY,
                                       aborted=False)

        ctx = FakeContext()
        with self.assertRaises(BudgetExhausted):
            alg.run(ctx, np.random.default_rng(0))
        self.assertEqual(ctx.stats["invalid"], 3)

    def test_invalid_trial_is_charged_but_not_credited(self):
        p = Problem("cl_Pt13", root=BENCH)

        class FakeContext:
            guard = None

            def __init__(self):
                self.p, self.nsteps, self.stats = p, 0, {}

            def relax(self, pos, cell):
                if self.nsteps >= 3:
                    raise BudgetExhausted
                self.nsteps += 1
                e = BAD_ENERGY if self.nsteps == 2 else -1.0
                return SimpleNamespace(pos=pos, cell=cell, e=e, aborted=False)

        ctx = FakeContext()
        with self.assertRaises(BudgetExhausted):
            alg.run(ctx, np.random.default_rng(9))
        self.assertEqual(ctx.stats["invalid"], 1)
        self.assertEqual(ctx.stats["trials"], 2)
        self.assertEqual(sum(ctx.stats["arm_costs"].values()), 2)
        self.assertLessEqual(sum(ctx.stats["arm_rewards"].values()), 0.5)

    def test_rejects_bad_parameters(self):
        with self.assertRaises(ValueError):
            alg.run(None, np.random.default_rng(0), capacity=1)


class IntegrationTests(unittest.TestCase):
    def test_moves_and_runner_all_categories_reproducible(self):
        for name in ("cl_Pt13", "bu_Si8", "su_Pt8_MgO100", "ze_Pt4_CHA"):
            with self.subTest(name=name):
                p = Problem(name, root=BENCH)
                pos, cell = p.random_structure(np.random.default_rng(3))
                parent = SimpleNamespace(pos=pos, cell=cell)
                for arm in alg._arms(p):
                    x, c = alg._move(p, parent, arm, np.random.default_rng(4), .6, .05)
                    self.assertTrue(np.isfinite(x).all())
                    self.assertTrue(np.isfinite(c).all())
                    if not p.variable_cell:
                        np.testing.assert_array_equal(x[:p.n_sub], pos[:p.n_sub])
                p.guard_factor = 0
                p.max_relax_steps = 3

                class HarmonicNEP:
                    def compute(self, x, cell_arg):
                        e = 0.01 * float((x * x).sum())
                        return e, -0.02 * x, np.zeros((3, 3)), np.zeros(len(x))

                out1 = run_one(p, "transition_aos", 7, budget=30, nep=HarmonicNEP())
                out2 = run_one(p, "transition_aos", 7, budget=30, nep=HarmonicNEP())
                for out in (out1, out2):
                    self.assertEqual(out["status"], "ok", out["error"])
                    self.assertEqual(out["nsteps"], 30)
                for key in ("best_e", "best_step", "best_free_pos", "best_cell",
                            "trace_steps", "trace_e", "stats"):
                    self.assertEqual(out1[key], out2[key])

    def test_swap_only_exchanges_search_species(self):
        p = Problem("cl_Pt7Cu6", root=BENCH)
        pos, cell = p.random_structure(np.random.default_rng(2))
        parent = SimpleNamespace(pos=pos, cell=cell)
        self.assertIn("swap", alg._arms(p))
        moved, _ = alg._move(p, parent, "swap", np.random.default_rng(2), .6, .05)
        self.assertEqual(sorted(map(tuple, pos)), sorted(map(tuple, moved)))
        self.assertEqual(np.count_nonzero(np.any(pos != moved, axis=1)), 2)


if __name__ == "__main__":
    unittest.main()
