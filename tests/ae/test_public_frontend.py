import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class PublicFrontendTest(unittest.TestCase):
    def test_failed_party_stops_its_waiting_peer(self):
        from frontend import run_experiment
        failed = subprocess.Popen([sys.executable, "-c", "raise SystemExit(1)"])
        waiting = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            with self.assertRaisesRegex(run_experiment.ExperimentFailure, "exit"):
                run_experiment.wait_for_parties(failed, waiting, timeout_seconds=2)
            self.assertIsNotNone(waiting.poll())
        finally:
            for process in (failed, waiting):
                if process.poll() is None:
                    process.kill()
                    process.wait()

    def test_party_timeout_stops_both_processes(self):
        from frontend import run_experiment
        processes = [subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]) for _ in range(2)]
        try:
            with self.assertRaisesRegex(run_experiment.ExperimentFailure, "timeout"):
                run_experiment.wait_for_parties(*processes, timeout_seconds=0.1)
            self.assertTrue(all(process.poll() is not None for process in processes))
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.wait()

    def test_pipeline_help_from_a_different_working_directory(self):
        with tempfile.TemporaryDirectory() as working:
            result = subprocess.run(
                [sys.executable, str(ROOT / "frontend/run_frontend_pipeline.py"), "--help"],
                cwd=working, capture_output=True, text=True,
            )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("--threads", result.stdout)

    def test_temporary_config_restores_optimized_default_after_failure(self):
        from frontend.run_experiment import temporary_config
        with tempfile.TemporaryDirectory() as working:
            config = Path(working) / "config.h"
            original = (
                "#define MVZK_CONFIG_SCALE 10\n"
                "#define MVZK_CONFIG_MULT_PRODUCT_THRESHOLD 256\n"
                "#define MVZK_CONFIG_DEFAULT_BITLEN 32\n"
                "#define MVZK_CONFIG_NON_LINEAR_RELU_DIGDEC_K 3\n"
                "#define MVZK_CONFIG_RANGE_CHECK_REQUEST_BUFFER_THRESHOLD 260000\n"
            ).encode()
            config.write_bytes(original)
            with self.assertRaises(RuntimeError):
                with temporary_config(config, 8, 128, 16, 2):
                    self.assertIn("THRESHOLD 260000", config.read_text())
                    raise RuntimeError("failed experiment")
            self.assertEqual(original, config.read_bytes())


if __name__ == "__main__":
    unittest.main()
