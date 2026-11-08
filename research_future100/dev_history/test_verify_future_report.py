import math
import unittest

from verify_future_report import compute_cell


class RecomputeTests(unittest.TestCase):
    def test_timeout_hits_and_unsuccessful_timeouts(self):
        runs = [
            {"status": "ok", "trace_steps": [2], "trace_e": [-1], "nsteps": 4},
            {"status": "timeout", "trace_steps": [3], "trace_e": [-1], "nsteps": 4},
            {"status": "timeout", "trace_steps": [4], "trace_e": [0], "nsteps": 4},
        ]
        self.assertEqual(compute_cell(runs, -0.5, 10), (7.5, 2, 1))

    def test_no_hits(self):
        runs = [{"status": "ok", "trace_steps": [1], "trace_e": [0], "nsteps": 8}]
        ert, hits, timed_hits = compute_cell(runs, -1, 10)
        self.assertTrue(math.isinf(ert))
        self.assertEqual((hits, timed_hits), (0, 0))

    def test_errors_are_not_silently_counted(self):
        with self.assertRaises(ValueError):
            compute_cell([{"status": "error"}], 0, 10)


if __name__ == "__main__":
    unittest.main()
