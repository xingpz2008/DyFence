import importlib.util
import subprocess
import sys
import time
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = REPO_ROOT / "build-ae"


def load_run_experiment_module():
    module_path = REPO_ROOT / "frontend" / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("dyfence_run_experiment", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_consistency_runner_module():
    module_path = REPO_ROOT / "test" / "functional_test" / "run_benchmark_consistency.py"
    spec = importlib.util.spec_from_file_location("dyfence_consistency_runner", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FailurePropagationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run(
            [
                "cmake", "-S", str(REPO_ROOT), "-B", str(BUILD_DIR),
                "-DDYFENCE_USE_BUNDLED_EMP=ON", "-DTEST_SELECT=FUNC",
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["cmake", "--build", str(BUILD_DIR), "--target", "benchmark_consistency", "-j", "2"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

    def test_consistency_rejects_unknown_mode_with_nonzero_exit(self):
        result = subprocess.run(
            [str(BUILD_DIR / "benchmark_consistency"), "1", "24561", "2", "8", "99"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )

        self.assertNotEqual(0, result.returncode)
        self.assertIn("mode", (result.stdout + result.stderr).lower())

    def test_consistency_runner_rejects_any_party_failure(self):
        module = load_consistency_runner_module()

        class Process:
            def __init__(self, returncode):
                self.returncode = returncode

        module.require_successful_processes("phase", [Process(0), Process(0)])
        with self.assertRaises(RuntimeError):
            module.require_successful_processes("phase", [Process(0), Process(3)])

    def test_consistency_runner_times_out_and_terminates_all_parties(self):
        module = load_consistency_runner_module()
        processes = [
            subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]),
            subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]),
        ]
        try:
            with self.assertRaisesRegex(TimeoutError, "phase-test"):
                module.wait_for_processes(
                    "phase-test",
                    processes,
                    deadline=time.monotonic() + 0.1,
                )
            self.assertTrue(all(process.poll() is not None for process in processes))
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.wait()

    def test_frontend_rejects_either_party_failure(self):
        module = load_run_experiment_module()

        module.require_successful_parties(prover_returncode=0, verifier_returncode=0)
        with self.assertRaises(module.ExperimentFailure):
            module.require_successful_parties(prover_returncode=1, verifier_returncode=0)
        with self.assertRaises(module.ExperimentFailure):
            module.require_successful_parties(prover_returncode=0, verifier_returncode=1)

    def test_frontend_run_status_is_nonzero_when_any_experiment_failed(self):
        module = load_run_experiment_module()

        self.assertEqual(0, module.experiment_exit_code([]))
        self.assertEqual(1, module.experiment_exit_code(["resnet18-32x32"]))


if __name__ == "__main__":
    unittest.main()
