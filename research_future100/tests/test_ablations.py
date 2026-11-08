import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from gopt.algorithms import transition_aos as alg
from gopt.context import BudgetExhausted
from gopt.problem import Problem
from gopt.run import run_one


class AblationTests(unittest.TestCase):
    @staticmethod
    def run_scripted(**params):
        class FakeProblem:
            category = "cluster"
            n_search = 2
            variable_cell = False
            search_symbols = ["Pt", "Pt"]

            def random_structure(self, rng):
                return None, None

        class FakeContext:
            guard = None

            def __init__(self):
                self.p, self.nsteps, self.stats = FakeProblem(), 0, {}
                self.history = []

            def relax(self, pos, cell):
                sequence = [(0.0, -1.0), (1.0, -0.4), (1.0, -0.4),
                            (1.0, -0.4), (2.0, -1.2)]
                if self.nsteps == len(sequence):
                    raise BudgetExhausted
                x, energy = sequence[self.nsteps]
                self.nsteps += 1
                self.history.append((x, energy))
                return SimpleNamespace(pos=np.array([[x, 0, 0], [0, 0, 0]]),
                                       cell=np.eye(3), e=energy, aborted=False)

        instances = []

        class SpySelector(alg.ContextualSelector):
            def __init__(self, *args):
                super().__init__(*args)
                self.choices, self.updates = [], []
                instances.append(self)

            def choose(self, state, rng):
                self.choices.append(state)
                return 0

            def update(self, state, arm, reward, cost):
                self.updates.append((state, arm, reward, cost))
                super().update(state, arm, reward, cost)

        ctx = FakeContext()
        rng = SimpleNamespace(random=lambda: 0.0)
        with patch.object(alg, "Descriptor", return_value=lambda pos, cell: pos[0, :1]), \
                patch.object(alg, "ContextualSelector", SpySelector), \
                patch.object(alg, "_move", return_value=(None, None)):
            with unittest.TestCase().assertRaises(BudgetExhausted):
                alg.run(ctx, rng, **params)
        return ctx, instances[0]

    def test_defaults_match_signature_and_explicit_enabled(self):
        signature = inspect.signature(alg.run)
        self.assertEqual(alg.DEFAULTS,
                         {name: p.default for name, p in signature.parameters.items()
                          if name not in ("ctx", "rng")})
        default, ds = self.run_scripted()
        explicit, es = self.run_scripted(contextual=True, transition_reward=True)
        self.assertEqual(default.history, explicit.history)
        self.assertEqual(default.stats, explicit.stats)
        self.assertEqual(ds.choices, es.choices)
        self.assertEqual(ds.updates, es.updates)

    def test_context_ablation_pools_only_selector_state(self):
        for transition_reward in (True, False):
            with self.subTest(transition_reward=transition_reward):
                full, fs = self.run_scripted(contextual=True, transition_reward=transition_reward)
                pooled, ps = self.run_scripted(contextual=False, transition_reward=transition_reward)
                self.assertEqual(fs.choices[:4], [0, 2, 2, 3])
                self.assertEqual(ps.choices[:4], [0, 0, 0, 0])
                self.assertEqual([u[1:] for u in fs.updates], [u[1:] for u in ps.updates])
                self.assertEqual(full.history, pooled.history)
                self.assertGreater(fs.count[2].sum(), 0)
                self.assertEqual(ps.count[2].sum(), 0)
                self.assertAlmostEqual(fs.count.sum(), ps.count.sum())
                self.assertAlmostEqual(fs.reward.sum(), ps.reward.sum())
                a, b = dict(full.stats), dict(pooled.stats)
                self.assertEqual(a.pop("context_pulls"), [1, 0, 2, 1])
                self.assertEqual(b.pop("context_pulls"), [4, 0, 0, 0])
                self.assertEqual(a, b)

    def test_transition_ablation_removes_only_novelty_credit(self):
        for contextual in (True, False):
            with self.subTest(contextual=contextual):
                full, fs = self.run_scripted(contextual=contextual, transition_reward=True)
                ablated, ns = self.run_scripted(contextual=contextual, transition_reward=False)
                self.assertEqual(fs.choices, ns.choices)
                self.assertEqual(full.history, ablated.history)
                self.assertEqual([u[:2] + u[3:] for u in fs.updates],
                                 [u[:2] + u[3:] for u in ns.updates])
                self.assertEqual(full.stats["novel_transitions"], 2)
                self.assertEqual(ablated.stats["novel_transitions"], 2)
                np.testing.assert_array_equal(fs.count, ns.count)
                np.testing.assert_array_equal(fs.cost, ns.cost)
                for actual, expected in zip([u[2] for u in fs.updates],
                                            [0.5, 0.0, 0.0, 0.7]):
                    self.assertAlmostEqual(actual, expected)
                for actual, expected in zip([u[2] for u in ns.updates],
                                            [0.0, 0.0, 0.0, 0.2]):
                    self.assertAlmostEqual(actual, expected)
                a, b = dict(full.stats), dict(ablated.stats)
                a.pop("arm_rewards")
                b.pop("arm_rewards")
                self.assertEqual(a, b)

    def test_runner_four_modes_are_seed_reproducible_and_budgeted(self):
        import os
        bench = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "benches-dev")
        p = Problem("cl_Pt13", root=bench)
        p.guard_factor = 0
        p.max_relax_steps = 3

        class HarmonicNEP:
            def compute(self, x, cell_arg):
                return (0.01 * float((x * x).sum()), -0.02 * x,
                        np.zeros((3, 3)), np.zeros(len(x)))

        for contextual in (False, True):
            for transition_reward in (False, True):
                params = dict(contextual=contextual, transition_reward=transition_reward)
                with self.subTest(**params):
                    first = run_one(p, "transition_aos", 19, budget=40,
                                    params=params, nep=HarmonicNEP())
                    replay = run_one(p, "transition_aos", 19, budget=40,
                                     params=params, nep=HarmonicNEP())
                    self.assertEqual(first["status"], "ok", first["error"])
                    self.assertEqual(first["nsteps"], 40)
                    for key in ("params", "best_e", "best_step", "best_free_pos",
                                "best_cell", "trace_steps", "trace_e", "stats"):
                        self.assertEqual(first[key], replay[key])


class ParameterTests(unittest.TestCase):
    def test_reject_nonfinite_and_nonreal_numeric_parameters(self):
        for name in ("kT", "step", "strain", "spacing", "energy_bin", "decay", "explore"):
            for value in (float("nan"), float("inf"), float("-inf"), True, "1", None, 1j):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        alg.run(None, None, **{name: value})

    def test_reject_invalid_ranges_capacity_and_toggles(self):
        cases = dict(kT=[0, -1], step=[0, -1], strain=[-1], spacing=[0, -1],
                     energy_bin=[0, -1], decay=[0, -1, 1.01], explore=[-0.01, 1.01],
                     capacity=[1, 2.0, 2.5, True, float("nan"), float("inf"), "32", None],
                     contextual=[0, 1, None, "false", float("nan")],
                     transition_reward=[0, 1, None, "false", float("nan")])
        for name, values in cases.items():
            for value in values:
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        alg.run(None, None, **{name: value})

    def test_valid_boundaries(self):
        for kwargs in (dict(strain=0), dict(decay=1), dict(explore=0),
                       dict(explore=1), dict(capacity=2),
                       dict(contextual=False, transition_reward=False)):
            params = {**alg.DEFAULTS, **kwargs}
            alg._validate(**params)


if __name__ == "__main__":
    unittest.main()
