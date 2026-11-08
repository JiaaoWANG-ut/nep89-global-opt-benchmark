import tempfile
import unittest
from pathlib import Path

from scripts.extended100_report import reference_paths, seed_range


class ScopeTests(unittest.TestCase):
    def test_seed_ranges_are_consecutive_and_disjoint_from_pilot(self):
        self.assertEqual(list(seed_range(0)), list(range(100)))
        self.assertEqual(list(seed_range(100)), list(range(100, 200)))
        self.assertFalse(set(seed_range(100)) & set(range(30, 35)))
        with self.assertRaises(ValueError):
            seed_range(-1)

    def test_reference_campaign_restriction(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("formal", "pilot", "unrelated"):
                path = root / name / "bench" / "bh"
                path.mkdir(parents=True)
                (path / "s000.json").touch()
            paths = reference_paths(str(root), "bench", "formal", ["pilot", "formal"])
            self.assertEqual(len(paths), 2)
            self.assertTrue(any("/formal/" in path for path in paths))
            self.assertTrue(any("/pilot/" in path for path in paths))
            self.assertEqual(len(reference_paths(str(root), "bench", "formal", None)), 3)
            for invalid in ("missing", "../outside", "*", ""):
                with self.assertRaises(ValueError):
                    reference_paths(str(root), "bench", "formal", [invalid])


if __name__ == "__main__":
    unittest.main()
