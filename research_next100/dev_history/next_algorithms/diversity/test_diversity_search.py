import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from gopt.algorithms import diversity_v2 as alg
from gopt.algorithms import diversity_search as original
from gopt.context import BudgetExhausted, Context
from gopt.problem import Problem


def result(e, fp_value):
    return SimpleNamespace(e=e, pos=np.array([[fp_value, 0., 0.]]),
                           cell=np.eye(3), aborted=False)


class ArchiveTests(unittest.TestCase):
    def test_quality_novelty_crowding_and_elite(self):
        a = alg.Archive(3, min_spacing=0.1)
        add = lambda e, x: a.add(result(e, x), np.array([x]))
        self.assertEqual(add(0, 0), "novel")
        self.assertEqual(add(5, 1), "novel")
        self.assertEqual(add(4, 3), "novel")
        self.assertEqual(add(6, 1.01), "rejected")
        self.assertEqual(add(3, 1.01), "improved")
        self.assertEqual(add(9, 8), "novel")
        self.assertEqual(sorted(x["e"] for x in a.entries), [0, 4, 9])
        self.assertEqual(add(-1, 7.5), "improved")
        self.assertIn(-1, [x["e"] for x in a.entries])
        self.assertEqual(len(a.entries), 3)

    def test_full_archive_accepts_new_energy_elite(self):
        a = alg.Archive(2, min_spacing=0.01)
        for energy, x in [(1, 0), (2, 10)]:
            a.add(result(energy, x), np.array([x]))
        self.assertEqual(a.add(result(-1, 5), np.array([5.])), "novel")
        self.assertIn(-1, [e["e"] for e in a.entries])

    def test_rank_isolation_use_count_selection_deterministic(self):
        def choices():
            a = alg.Archive(3)
            for e, x in [(0, 0), (1, 1), (2, 4)]:
                a.add(result(e, x), np.array([x]))
            rng = np.random.default_rng(42)
            return [a.choose(rng)["e"] for _ in range(20)]
        self.assertEqual(choices(), choices())
        self.assertGreater(len(set(choices())), 1)


class DescriptorTests(unittest.TestCase):
    def test_periodic_translation_and_same_species_permutation(self):
        p = Problem("cl_Pt7Cu6")
        x, cell = p.random_structure(np.random.default_rng(3))
        desc = alg.Descriptor(p)
        base = desc(x, cell)
        changed = x.copy()
        same = [i for i in p.search_idx if p.symbols[i] == p.symbols[p.search_idx[0]]]
        changed[same[:2]] = changed[same[:2]][::-1]
        changed[p.search_idx[0]] += cell[0]
        np.testing.assert_allclose(desc(changed, cell), base, atol=1e-10)

    def test_host_environments_and_volume_are_included(self):
        p = Problem("su_Pt8_MgO100")
        x, cell = p.random_structure(np.random.default_rng(7))
        d = alg.Descriptor(p)
        moved = x.copy()
        moved[p.search_idx] += [0, 0, 1]
        self.assertGreater(np.linalg.norm(d(x, cell) - d(moved, cell)), 1e-3)
        p = Problem("bu_Si8")
        x, cell = p.random_structure(np.random.default_rng(8))
        d = alg.Descriptor(p)
        self.assertGreater(np.linalg.norm(d(x, cell) - d(x * 1.1, cell * 1.1)), 1e-3)

    def test_nearest_host_descriptor_can_alias_distinct_sites(self):
        p = SimpleNamespace(search_idx=np.array([3]), symbols=["O", "O", "O", "Pt"],
                            n_sub=3, pbc=np.array([False] * 3), variable_cell=False)
        x = np.array([[0., 0, 0], [3, 0, 0], [7, 0, 0], [.5, 0, 0]])
        y = x.copy()
        y[3, 0] = 3.5
        np.testing.assert_array_equal(alg.Descriptor(p)(x, np.eye(3)),
                                      alg.Descriptor(p)(y, np.eye(3)))


class HostedShiftTests(unittest.TestCase):
    def test_rigid_moves_keep_host_and_stay_surface_parallel(self):
        for name in ("su_Pt8_MgO100", "ze_Pt4_CHA"):
            with self.subTest(name=name):
                p = Problem(name)
                x, cell = p.random_structure(np.random.default_rng(16))
                if p.category == "surface":
                    # Exercise a slab whose normal is not the Cartesian z axis.
                    rotation = np.array([[1., 0, 0], [0, 0, -1.], [0, 1., 0]])
                    x, cell = x @ rotation, cell @ rotation
                y = alg._host_shift(p, x, cell, np.random.default_rng(9))
                self.assertIsNotNone(y)
                np.testing.assert_array_equal(y[:p.n_sub], x[:p.n_sub])
                displacements = y[p.search_idx] - x[p.search_idx]
                np.testing.assert_allclose(displacements, np.tile(displacements[0], (len(displacements), 1)),
                                           atol=1e-12)
                self.assertGreater(np.linalg.norm(displacements[0]), 0)
                if p.category == "surface":
                    normal = np.cross(cell[0], cell[1])
                    self.assertAlmostEqual(float(displacements[0] @ normal), 0, places=10)
                d = alg.min_image_distances(y[p.search_idx], y[:p.n_sub], cell, p.pbc)
                limits = p.dmin_factor * (p.rcov[p.search_idx, None] + p.rcov[None, :p.n_sub])
                self.assertTrue(np.all(d >= limits))

    def test_exhausted_trials_fall_back_without_changing_parent(self):
        p = Problem("su_Pt8_MgO100")
        x, cell = p.random_structure(np.random.default_rng(3))
        initial = x.copy()
        with patch.object(alg, "min_image_distances",
                          return_value=np.zeros((p.n_search, p.n_sub))) as distances:
            self.assertIsNone(alg._host_shift(p, x, cell, np.random.default_rng(4), tries=3))
        self.assertEqual(distances.call_count, 4)
        np.testing.assert_array_equal(x, initial)

    def test_attached_cluster_is_not_translated_away_from_host(self):
        p = Problem("su_Pt8_MgO100")
        x, cell = p.random_structure(np.random.default_rng(3))
        shape = (p.n_search, p.n_sub)
        with patch.object(alg, "min_image_distances",
                          side_effect=[np.zeros(shape)] + [np.full(shape, 100.)] * 3):
            self.assertIsNone(alg._host_shift(p, x, cell, np.random.default_rng(4), tries=3))

    def test_zeolite_region_rejects_outside_translations(self):
        p = Problem("ze_Pt4_CHA")
        x, cell = p.random_structure(np.random.default_rng(3))
        with patch.object(p, "in_region", return_value=False) as region:
            self.assertIsNone(alg._host_shift(p, x, cell, np.random.default_rng(4), tries=3))
        self.assertEqual(region.call_count, 3)

    def test_host_shift_disabled_reproduces_original_trajectory(self):
        for name in ("cl_Pt13", "su_Pt8_MgO100", "ze_Pt4_CHA", "bu_Si8"):
            with self.subTest(name=name):
                p = Problem(name)
                a, b = FakeContext(p, 12), FakeContext(p, 12)
                with self.assertRaises(BudgetExhausted):
                    original.run(a, np.random.default_rng(25), capacity=4, adapt_every=3)
                with self.assertRaises(BudgetExhausted):
                    alg.run(b, np.random.default_rng(25), capacity=4, adapt_every=3, p_host_shift=0)
                self.assertEqual(a.stats, {k: b.stats[k] for k in a.stats})
                self.assertEqual(b.stats["host_shift_attempts"], 0)
                for (x, c), (y, d) in zip(a.inputs, b.inputs):
                    np.testing.assert_array_equal(x, y)
                    np.testing.assert_array_equal(c, d)

    def test_run_uses_host_shift_and_fallback(self):
        p = Problem("su_Pt8_MgO100")
        for succeeds in (True, False):
            c = FakeContext(p, 8)
            fn = ((lambda p, pos, cell, rng: np.array(pos, copy=True))
                  if succeeds else (lambda *args: None))
            with patch.object(alg, "_host_shift", side_effect=fn) as move:
                with self.assertRaises(BudgetExhausted):
                    alg.run(c, np.random.default_rng(5), capacity=4, p_random=0,
                            p_host_shift=1)
            self.assertGreater(move.call_count, 0)
            self.assertGreater(c.stats["host_shift_attempts"], 0)
            self.assertEqual(c.stats["host_shifts"],
                             c.stats["host_shift_attempts"] if succeeds else 0)


class FakeContext:
    def __init__(self, p, budget):
        self.p, self.budget, self.nsteps, self.stats = p, budget, 0, {}
        self.inputs = []

    def relax(self, pos, cell):
        if self.nsteps == self.budget:
            raise BudgetExhausted()
        self.nsteps += 1
        frozen = np.flatnonzero(~self.p.free_mask)
        np.testing.assert_allclose(pos[frozen], self.p.sub_pos[frozen], atol=1e-12)
        self.inputs.append((np.array(pos), np.array(cell)))
        # Exercise both failed and non-finite relaxations.
        bad = self.nsteps in (2, 3)
        return SimpleNamespace(pos=np.array(pos), cell=np.array(cell),
                               e=np.nan if self.nsteps == 3 else float(np.sum(pos ** 2)),
                               aborted=bad and self.nsteps == 2)


class SearchTests(unittest.TestCase):
    def test_four_categories_budget_fixed_atoms_and_seed(self):
        for name in ("cl_Pt13", "su_Pt8_MgO100", "ze_Pt4_CHA", "bu_Si8"):
            with self.subTest(name=name):
                p = Problem(name)
                histories = []
                for _ in range(2):
                    c = FakeContext(p, 12)
                    with self.assertRaises(BudgetExhausted):
                        alg.run(c, np.random.default_rng(11), capacity=4, adapt_every=3)
                    self.assertEqual(c.nsteps, 12)
                    self.assertEqual(c.stats["trials"], 12)
                    self.assertLessEqual(c.stats["archive_size"], 4)
                    self.assertGreater(c.stats["archive_size"], 0)
                    self.assertLessEqual(c.stats["admissions"], 10)
                    histories.append(c)
                self.assertEqual(histories[0].stats, histories[1].stats)
                for (x, cell), (y, cell2) in zip(histories[0].inputs, histories[1].inputs):
                    np.testing.assert_array_equal(x, y)
                    np.testing.assert_array_equal(cell, cell2)

    def test_variable_cell_move_restores_fixed_host(self):
        p = SimpleNamespace(variable_cell=True, search_idx=np.array([1, 2]),
                            free_mask=np.array([False, True, True]),
                            sub_pos=np.array([[1., 2., 3.]]), category="bulk")
        parent = dict(pos=np.array([[1., 2., 3.], [2., 2., 2.], [3., 3., 3.]]),
                      cell=np.eye(3) * 5)
        with patch.object(alg.ops, "strain_cell",
                          side_effect=lambda pos, cell, rng, sigma: (pos + 2, cell * 1.1)), \
             patch.object(alg.ops, "displace",
                          side_effect=lambda p, pos, rng, step, cell: pos):
            pos, cell = alg._move(p, parent, np.random.default_rng(1), .5, .05, 0, 0)
        np.testing.assert_array_equal(pos[0], p.sub_pos[0])
        np.testing.assert_array_equal(cell, np.eye(3) * 5.5)

    def test_invalid_parameters(self):
        for kwargs in ({"capacity": 1}, {"step": 0}, {"p_swap": .9, "p_relocate": .2},
                       {"p_host_shift": -0.1}, {"p_host_shift": float("nan")},
                       {"adapt_every": 0}):
            with self.assertRaises(ValueError):
                alg.run(None, np.random.default_rng(0), **kwargs)

    def test_real_context_budget_accounting(self):
        class FakeNEP:
            def compute(self, x, cell):
                return float(np.sum(x * x)), -2 * x, np.zeros((3, 3)), np.sum(x * x, axis=1)

        p = Problem("cl_Pt13")
        p.max_relax_steps = 1
        ctx = Context(p, budget=10, nep=FakeNEP())
        with self.assertRaises(BudgetExhausted):
            alg.run(ctx, np.random.default_rng(12), capacity=4)
        self.assertEqual(ctx.nsteps, 10)
        self.assertEqual(ctx.remaining(), 0)
        self.assertTrue(np.isfinite(ctx.best_e))


if __name__ == "__main__":
    unittest.main()
