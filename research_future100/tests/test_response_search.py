import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from gopt.algorithms import response_search as alg
from gopt.context import BudgetExhausted, Context
from gopt.problem import Problem


ROOT = Path(__file__).resolve().parents[1]
BENCHES = ("cl_Pt13", "su_Pt8_MgO100", "ze_Pt4_CHA", "bu_Si8")


class CheapPotential:
    def __init__(self):
        self.cell = None

    def compute(self, pos, cell=None):
        # A smooth deterministic landscape; zero force gives one-call relaxations.
        return float(np.sum(np.sin(pos) ** 2)), np.zeros_like(pos), np.zeros((3, 3)), np.zeros(len(pos))


class ResponseSearchTests(unittest.TestCase):
    def test_learning_and_directional_proposal(self):
        p = Problem("cl_Pt13", root=ROOT / "benches-dev")
        pos, cell = p.random_structure(np.random.default_rng(1))
        parent = dict(pos=pos, cell=cell, step=0.5, direction=None, escapes=0, returns=0)
        alg._learn(parent, np.ones((p.n_search, 3)), False, 0.5)
        self.assertAlmostEqual(parent["step"], 0.625)
        self.assertIsNone(parent["direction"])
        direction = np.arange(p.n_search * 3, dtype=float).reshape(-1, 3)
        direction -= direction.mean(axis=0)
        alg._learn(parent, direction, True, 0.5)
        self.assertAlmostEqual(alg._rms(parent["direction"]), 1.0)
        self.assertEqual((parent["returns"], parent["escapes"]), (1, 1))
        a, _ = alg._proposal(p, parent, np.random.default_rng(19), 1.0, 0)
        parent["direction"] = -parent["direction"]
        b, _ = alg._proposal(p, parent, np.random.default_rng(19), 1.0, 0)
        self.assertGreater(np.linalg.norm(a - b), 0.1)

    def test_projection_and_frozen_across_categories(self):
        for name in BENCHES:
            with self.subTest(name=name):
                p = Problem(name, root=ROOT / "benches-dev")
                pos, cell = p.random_structure(np.random.default_rng(8))
                pos[p.search_idx] += 100
                x = alg._project(p, pos, cell)
                self.assertTrue(all(p.in_region(v, cell) for v in x[p.search_idx]))
                if p.region["type"] == "cell":
                    f = x[p.search_idx] @ np.linalg.inv(cell)
                    self.assertTrue(np.all(f >= -1e-10) and np.all(f <= 1 + 1e-10))
                fixed = np.flatnonzero(~p.free_mask)
                np.testing.assert_allclose(x[fixed], p.sub_pos[fixed])

    def test_budget_and_seed_determinism_all_categories(self):
        for name in BENCHES:
            with self.subTest(name=name):
                p = Problem(name, root=ROOT / "benches-dev")
                def experiment():
                    ctx = Context(p, budget=8, nep=CheapPotential())
                    ctx.guard = None
                    try:
                        alg.run(ctx, np.random.default_rng(42), restart=0, spacing=0.005)
                    except BudgetExhausted:
                        pass
                    return ctx
                a, b = experiment(), experiment()
                self.assertEqual(a.nsteps, 8)
                self.assertEqual(a.n_frozen_fix, 0)
                self.assertEqual(a.stats, b.stats)
                self.assertEqual(a.best_e, b.best_e)
                np.testing.assert_array_equal(a.best_pos, b.best_pos)
                self.assertGreater(a.stats["trials"], 0)

    def test_parameter_validation(self):
        p = Problem("cl_Pt13", root=ROOT / "benches-dev")
        ctx = Context(p, budget=1, nep=CheapPotential())
        for kwargs in ({"capacity": 1}, {"step": 0}, {"restart": 1.1},
                       {"direction_weight": -1}, {"strain": 0.3}, {"spacing": np.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                alg.run(ctx, np.random.default_rng(0), **kwargs)
        self.assertEqual(ctx.nsteps, 0)

    def test_zero_direction_ablation_is_bitwise_independent_of_memory(self):
        p = Problem("cl_Pt13", root=ROOT / "benches-dev")
        pos, cell = p.random_structure(np.random.default_rng(1))
        parent = dict(pos=pos, cell=cell, step=0.5,
                      direction=np.ones((p.n_search, 3)))
        a, ca = alg._proposal(p, parent, np.random.default_rng(23), 0.0, 0.0)
        parent["direction"] = None
        b, cb = alg._proposal(p, parent, np.random.default_rng(23), 0.0, 0.0)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(ca, cb)

    def test_collapsed_strain_is_rejected(self):
        p = Problem("bu_Si8", root=ROOT / "benches-dev")
        pos, cell = p.random_structure(np.random.default_rng(1))
        parent = dict(pos=pos, cell=cell, step=0.5, direction=None)
        with patch.object(alg.ops, "strain_cell", return_value=(pos, np.zeros((3, 3)))):
            _, new_cell = alg._proposal(p, parent, np.random.default_rng(2), 0.5, 0.1)
        np.testing.assert_array_equal(new_cell, cell)

    def test_interrupted_guided_proposal_is_counted(self):
        p = Problem("cl_Pt13", root=ROOT / "benches-dev")

        class FakeContext:
            def __init__(self):
                self.p, self.nsteps, self.stats = p, 0, {}

            def remaining(self):
                return 3 - self.nsteps

            def relax(self, pos, cell):
                self.nsteps += 1
                if self.nsteps == 3:
                    raise BudgetExhausted()
                class Result:
                    aborted = False
                    e = 0.
                r = Result()
                r.pos, r.cell = pos.copy(), cell.copy()
                return r

        class ChangingDescriptor:
            def __init__(self, _p):
                self.count = 0

            def __call__(self, pos, cell):
                self.count += 1
                return np.array([float(self.count)])

        original_insert = alg._insert
        def first_only(pop, *args):
            return original_insert(pop, *args) if not pop else "new"

        ctx = FakeContext()
        with patch.object(alg, "Descriptor", ChangingDescriptor), patch.object(alg, "_insert", first_only):
            with self.assertRaises(BudgetExhausted):
                alg.run(ctx, np.random.default_rng(42), restart=0)
        self.assertEqual(ctx.stats["trials"], 2)
        self.assertEqual(ctx.stats["proposals_started"], 3)
        self.assertEqual(ctx.stats["directed_proposals"], 1)

    def test_archive_duplicate_and_elite_protection(self):
        class Result:
            pos = np.zeros((2, 3))
            cell = np.eye(3)
            e = 0.0
        pop = []
        r = Result()
        self.assertEqual(alg._insert(pop, r, np.array([0.]), 2, 0.1, 0.5), "new")
        r.e = 2.
        self.assertEqual(alg._insert(pop, r, np.array([1.]), 2, 0.1, 0.5), "new")
        r.e = 3.
        self.assertEqual(alg._insert(pop, r, np.array([2.]), 2, 0.1, 0.5), "new")
        self.assertEqual(sorted(x["e"] for x in pop), [0., 3.])
        r.e = -1.
        self.assertEqual(alg._insert(pop, r, np.array([0.01]), 2, 0.1, 0.5), "improved")


if __name__ == "__main__":
    unittest.main()
