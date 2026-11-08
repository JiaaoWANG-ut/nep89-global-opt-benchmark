import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from verify_queue import ROOT, verify


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.queue = Path(self.tmp.name)
        for name in ("pending", "done", "failed", "running", "control"):
            (self.queue / name).mkdir()
        self.config = {
            "prefix": "test", "campaign": "test-campaign",
            "benchmarks": ["x"], "algorithms": ["a", "b"],
            "seed_range_inclusive": [0, 2], "capped_algorithms": ["b"],
            "time_limit_seconds": 20, "capped_task_timeout_seconds": 60,
            "other_task_timeout_seconds": 100, "capped_seeds_per_task": 1,
            "other_seeds_per_task": 2,
        }
        self.add_task("a", "0,1", 100)
        self.add_task("a", "2", 100)
        for seed in range(3):
            self.add_task("b", str(seed), 60)

    def add_task(self, alg, seeds, timeout, suffix=""):
        command = (f"python -m gopt.run --bench x --alg {alg} --seeds {seeds} "
                   "--campaign test-campaign --print")
        if alg == "b":
            command += " --time-limit 20"
        task = {"name": f"test-x-{alg}-{seeds}{suffix}", "cwd": str(ROOT),
                "cmd": command, "timeout": timeout}
        (self.queue / "pending" / f"{alg}-{seeds}{suffix}.json").write_text(json.dumps(task))

    def check(self):
        return verify(self.queue, self.config, ["x"], ["a", "b"])

    def test_complete_cartesian_product(self):
        self.assertEqual(self.check(), (5, 6))

    def test_duplicate_run_is_rejected(self):
        self.add_task("a", "0,1", 100, "-duplicate")
        with self.assertRaisesRegex(ValueError, "Duplicate run"):
            self.check()

    def test_missing_run_is_rejected(self):
        (self.queue / "pending" / "b-2.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing=1"):
            self.check()

    def test_wrong_timeout_is_rejected(self):
        self.add_task("b", "0", 61)
        with self.assertRaisesRegex(ValueError, "Wrong task timeout"):
            self.check()

    def test_control_flag_is_rejected(self):
        (self.queue / "control" / "DRAIN").touch()
        with self.assertRaisesRegex(ValueError, "control"):
            self.check()

    def test_duplicate_option_is_rejected(self):
        path = self.queue / "pending" / "a-2.json"
        task = json.loads(path.read_text())
        task["cmd"] += " --budget 1 --budget 2"
        path.write_text(json.dumps(task))
        with self.assertRaisesRegex(ValueError, "Duplicate option"):
            self.check()

    def test_environment_override_is_rejected(self):
        path = self.queue / "pending" / "a-2.json"
        task = json.loads(path.read_text())
        task["env"] = {"GOPT_BENCH_DIR": "/other"}
        path.write_text(json.dumps(task))
        with self.assertRaisesRegex(ValueError, "execution settings"):
            self.check()

    def test_real_task_generator_output(self):
        for path in (self.queue / "pending").iterdir():
            path.unlink()
        env = dict(os.environ, FARM_QUEUE=str(self.queue))
        for alg, chunk, timeout, capped in (("a", "2", "100", False),
                                             ("b", "1", "60", True)):
            args = [sys.executable, str(ROOT / "scripts" / "make_tasks.py"),
                    "--prefix", "test", "--campaign", "test-campaign",
                    "--algs", alg, "--benches", "x", "--seeds", "0-2",
                    "--per-task", chunk, "--timeout", timeout, "--submit"]
            if capped:
                args.extend(["--time-limit", "20"])
            subprocess.run(args, cwd=ROOT, env=env, check=True, capture_output=True)
        self.assertEqual(self.check(), (5, 6))


if __name__ == "__main__":
    unittest.main()
