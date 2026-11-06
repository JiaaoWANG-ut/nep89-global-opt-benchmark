import unittest
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from gopt.context import BudgetExhausted
from gopt.problem import min_image_distances
from gopt.algorithms import relink_v2 as alg


class Problem:
    def __init__(self, category):
        self.category = category
        self.variable_cell = category == "bulk"
        self.n_sub = 1 if category in ("surface", "zeolite") else 0
        self.n = 4 + self.n_sub
        self.search_idx = np.arange(self.n_sub, self.n)
        self.n_search = len(self.search_idx)
        self.symbols = ["O"] * self.n_sub + ["Pt", "Pt", "Cu", "Cu"]
        self.free_mask = np.ones(self.n, bool)
        self.free_mask[:self.n_sub] = False
        self.sub_pos = np.array([[1., 1., 1.]]) if self.n_sub else np.empty((0, 3))
        self.cell0 = np.eye(3) * 10
        self.pbc = np.ones(3, bool)
        self.rcov = np.full(self.n, 0.6)
        self.dmin_factor = 0.7

    def random_structure(self, rng):
        cell = self.cell0.copy()
        if self.variable_cell:
            cell *= rng.uniform(0.95, 1.05)
        pos = np.zeros((self.n, 3))
        pos[:self.n_sub] = self.sub_pos
        pos[self.search_idx] = np.array([[3, 3, 3], [5, 3, 3],
                                         [3, 5, 3], [3, 3, 5]]) + rng.normal(0, .3, (4, 3))
        return pos, cell


class MockContext:
    def __init__(self, p, budget=20):
        self.p, self.budget = p, budget
        self.nsteps = 0
        self.stats = {}
        self.inputs = []

    def relax(self, pos, cell):
        if self.nsteps == self.budget:
            raise BudgetExhausted
        self.nsteps += 1
        self.inputs.append((np.array(pos), np.array(cell)))
        e = float(np.sum(np.asarray(pos)[self.p.search_idx] ** 2))
        return SimpleNamespace(pos=np.array(pos), cell=np.array(cell), e=e, aborted=False)


class RelinkingTests(unittest.TestCase):
    def test_invalid_parameters_fail_before_work(self):
        cases = {
            "archive_size": [0, 1, -2, 2.5, 2.0, True, np.nan, np.inf, "3", None],
            "initial": [0, -1, 1.5, False, np.nan, np.inf],
            "path_points": [0, -1, 1.5, False, np.nan, np.inf],
            "immigrant_every": [0, -1, 1.5, False, np.nan, np.inf],
            "rotations": [-1, 1.5, False, np.nan, np.inf],
            "host_every": [0, -1, 1.5, False, np.nan, np.inf],
            "diversity_weight": [-.01, 1.01, np.nan, np.inf, -np.inf, True, "0.5"],
            "shake": [0, -1, np.nan, np.inf, -np.inf, True, "1", 10 ** 400,
                      Fraction(1, 10 ** 400)],
            "host_step": [0, -1, np.nan, np.inf, -np.inf, True, "1", 10 ** 400,
                          Fraction(1, 10 ** 400)],
        }
        for name, values in cases.items():
            for value in values:
                with self.subTest(name=name, value=value):
                    ctx = MockContext(Problem("cluster"), budget=1)
                    with self.assertRaisesRegex(ValueError, name):
                        alg.run(ctx, np.random.default_rng(1), **{name: value})
                    self.assertEqual(ctx.nsteps, 0)
                    self.assertEqual(ctx.stats, {})

    def test_boundary_parameters_are_valid(self):
        for weight in (0., 1., Fraction(1, 2)):
            ctx = MockContext(Problem("cluster"), budget=1)
            with self.assertRaises(BudgetExhausted):
                alg.run(ctx, np.random.default_rng(1), archive_size=2, initial=1,
                        path_points=1, immigrant_every=1, diversity_weight=weight,
                        shake=0.01, rotations=0, host_every=1, host_step=0.01)
            self.assertEqual(ctx.nsteps, 1)

    def test_assignment_respects_species_and_periodic_images(self):
        ref = np.array([[.1, 0, 0], [4, 0, 0], [8, 0, 0]])
        mov = np.array([[4, 0, 0], [10.1, 0, 0], [8, 0, 0]])
        perm = alg._match(ref, mov, [np.array([0, 1]), np.array([2])],
                          np.eye(3) * 10, np.ones(3, bool))
        np.testing.assert_array_equal(perm, [1, 0, 2])

    def test_archive_duplicate_elite_and_capacity(self):
        p = Problem("cluster")
        archive = []
        def member(e, x):
            return alg._Member(np.zeros((p.n, 3)), p.cell0, e, np.array([x]))
        for e, x in [(1., 0.), (2., .1), (3., 1.), (4., 1.1)]:
            alg._insert(p, archive, member(e, x), 3, .5)
        self.assertEqual(len(archive), 3)
        self.assertEqual(archive[0].e, 1.)
        self.assertFalse(alg._insert(p, archive, member(1.005, 0.), 3, .5))
        self.assertTrue(alg._insert(p, archive, member(.995, 0.), 3, .5))
        self.assertEqual(archive[0].e, .995)

    def test_recombination_frozen_and_cell_fallback(self):
        p = Problem("surface")
        rng = np.random.default_rng(42)
        pa, ca = p.random_structure(rng)
        pb, cb = p.random_structure(rng)
        a = alg._Member(pa, ca, 0., np.array([0.]))
        b = alg._Member(pb, cb, 1., np.array([1.]))
        aligned = alg._aligned(p, a, b, rng, 0)
        child, cell = alg._path_geometry(p, a, b, aligned, .5)
        np.testing.assert_array_equal(child[:p.n_sub], p.sub_pos)
        np.testing.assert_array_equal(cell, ca)
        p.variable_cell = True
        b.cell = -ca
        child, cell = alg._path_geometry(p, a, b, aligned, .5)
        np.testing.assert_array_equal(cell, ca)
        np.testing.assert_array_equal(child[:p.n_sub], p.sub_pos)

    def test_bulk_origin_shift_and_valid_cell_interpolation(self):
        p = Problem("bulk")
        rng = np.random.default_rng(42)
        pa, ca = p.random_structure(rng)
        cb = ca * 1.1
        pb = (pa @ np.linalg.inv(ca) + [.3, .2, .1]) @ cb
        a = alg._Member(pa, ca, 0., np.array([0.]))
        b = alg._Member(pb, cb, 1., np.array([1.]))
        aligned = alg._aligned(p, a, b, rng, 0)
        np.testing.assert_allclose(aligned[2], 0., atol=1e-12)
        child, cell = alg._path_geometry(p, a, b, aligned, .5)
        np.testing.assert_allclose(cell, (ca + cb) / 2)
        np.testing.assert_allclose(child @ np.linalg.inv(cell), pa @ np.linalg.inv(ca))

    def test_deterministic_search_budget_and_all_categories(self):
        for category in ("cluster", "surface", "zeolite", "bulk"):
            def execute():
                p = Problem(category)
                ctx = MockContext(p)
                with self.assertRaises(BudgetExhausted):
                    alg.run(ctx, np.random.default_rng(345), rotations=1)
                return p, ctx
            p, a = execute()
            _, b = execute()
            self.assertEqual(a.nsteps, a.budget)
            self.assertEqual(a.stats, b.stats)
            self.assertGreater(a.stats["path_relink"]["images"], 0)
            self.assertGreater(a.stats["path_relink"]["random"], 3)
            if p.n_sub:
                self.assertGreater(a.stats["path_relink"]["host"], 0)
            else:
                self.assertEqual(a.stats["path_relink"]["host"], 0)
            for (pa, ca), (pb, cb) in zip(a.inputs, b.inputs):
                np.testing.assert_allclose(pa, pb)
                np.testing.assert_allclose(ca, cb)
                np.testing.assert_array_equal(pa[:p.n_sub], p.sub_pos)
                self.assertGreater(np.linalg.det(ca), 0)

    def test_singleton_archive_uses_shake_and_propagates_budget(self):
        p = Problem("cluster")
        ctx = MockContext(p, budget=4)
        with self.assertRaises(BudgetExhausted):
            alg.run(ctx, np.random.default_rng(1), initial=1)
        self.assertGreater(ctx.stats["path_relink"]["shake"], 0)

    def test_empty_and_zero_fingerprints_and_partial_stats(self):
        p = Problem("cluster")
        p.n = p.n_search = 1
        p.search_idx = np.array([0])
        p.symbols = ["Pt"]
        p.free_mask = np.array([True])
        p.rcov = np.array([.6])
        p.random_structure = lambda rng: (rng.uniform(2, 4, (1, 3)), p.cell0.copy())
        res = SimpleNamespace(pos=np.array([[3., 3., 3.]]), cell=p.cell0,
                              e=1., aborted=False)
        member = alg._member(p, res)
        self.assertEqual(len(member.fp), 0)
        archive = []
        self.assertTrue(alg._insert(p, archive, member, 2, .5))
        self.assertFalse(alg._insert(p, archive, member, 2, .5))
        self.assertEqual(alg._distance(p, member.fp, member.fp), 0.)
        with self.assertRaises(ValueError):
            alg._distance(p, member.fp, np.array([0.]))

        p.variable_cell = True
        self.assertEqual(alg._distance(p, np.zeros(3), np.zeros(3)), 0.)
        self.assertEqual(alg._distance(p, np.zeros(3), np.array([1., 0., 0.])), 1.)
        p.variable_cell = False
        ctx = MockContext(p, budget=1)
        ctx.stats["path_relink"] = {"random": 7}
        with self.assertRaises(BudgetExhausted):
            alg.run(ctx, np.random.default_rng(1), initial=1)
        self.assertEqual(ctx.stats["path_relink"], {
            "random": 8, "paths": 0, "images": 0, "shake": 1, "host": 0,
            "host_fallback": 0, "admitted": 1, "invalid": 0})

    def test_host_translation_preserves_group_and_screened_host(self):
        for category in ("surface", "zeolite"):
            p = Problem(category)
            # Tilt the slab plane to ensure the move does not assume Cartesian z.
            if category == "surface":
                p.cell0 = np.array([[10., 0., 4.], [0., 10., 2.], [0., 0., 10.]])
            rng = np.random.default_rng(42)
            pos, cell = p.random_structure(rng)
            a = alg._Member(pos, cell, 0., np.array([0.]))
            child = alg._host_translation(p, a, rng, 1.)
            self.assertIsNotNone(child)
            delta = child[p.search_idx] - pos[p.search_idx]
            np.testing.assert_allclose(delta, np.broadcast_to(delta[0], delta.shape))
            self.assertGreater(np.linalg.norm(delta[0]), 0.)
            np.testing.assert_array_equal(child[:p.n_sub], pos[:p.n_sub])
            d = min_image_distances(child[p.search_idx], pos[:p.n_sub], cell, p.pbc)
            limits = p.dmin_factor * (p.rcov[p.search_idx, None] + p.rcov[None, :p.n_sub])
            self.assertTrue(np.all(d >= limits))
            if category == "surface":
                normal = np.cross(cell[0], cell[1])
                self.assertAlmostEqual(float(delta[0] @ normal), 0., places=10)

    def test_translation_keeps_frozen_search_atoms_fixed(self):
        p = Problem("surface")
        p.free_mask[p.search_idx[0]] = False
        rng = np.random.default_rng(42)
        pos, cell = p.random_structure(rng)
        parent = alg._Member(pos, cell, 0., np.array([0.]))
        child = alg._host_translation(p, parent, rng, 1.)
        self.assertIsNotNone(child)
        np.testing.assert_array_equal(child[~p.free_mask], pos[~p.free_mask])
        self.assertTrue(np.any(child[p.free_mask] != pos[p.free_mask]))

    def test_infeasible_host_move_uses_shake_fallback(self):
        p = Problem("zeolite")
        ctx = MockContext(p, budget=4)
        with patch.object(alg, "_host_translation", return_value=None):
            with self.assertRaises(BudgetExhausted):
                alg.run(ctx, np.random.default_rng(1), initial=1, host_every=1)
        self.assertGreater(ctx.stats["path_relink"]["host_fallback"], 0)
        for pos, _ in ctx.inputs:
            np.testing.assert_array_equal(pos[:p.n_sub], p.sub_pos)

    def test_coincident_forced_moves_do_not_starve(self):
        ctx = MockContext(Problem("surface"), budget=8)
        with self.assertRaises(BudgetExhausted):
            alg.run(ctx, np.random.default_rng(42), initial=1,
                    immigrant_every=1, host_every=1)
        stats = ctx.stats["path_relink"]
        self.assertGreater(stats["random"], 1)
        self.assertGreater(stats["host"], 0)


if __name__ == "__main__":
    unittest.main()
