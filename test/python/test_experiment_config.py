import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "frontend" / "run_experiment.py"


def load_module():
    spec = importlib.util.spec_from_file_location("dyfence_experiment_config", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ExperimentConfigurationTest(unittest.TestCase):
    def test_all_table_viii_configurations_are_paper_values(self):
        ae_path = REPO_ROOT / "ae.py"
        spec = importlib.util.spec_from_file_location("dyfence_table_viii", ae_path)
        ae_module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = ae_module
        spec.loader.exec_module(ae_module)
        cases = ae_module.load_paper_cases()["tables"]["VIII"]

        configured = {
            (case.get("bit_length"), case.get("scale"), case.get("decomposition_k"))
            for case in cases
            if "scale" in case
        }
        self.assertEqual({(16, 8, 2), (32, 10, 3), (40, 12, 3)}, configured)

    def test_table_viii_runner_applies_case_decomposition(self):
        ae_path = REPO_ROOT / "ae.py"
        spec = importlib.util.spec_from_file_location("dyfence_table_viii_runner", ae_path)
        ae_module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = ae_module
        spec.loader.exec_module(ae_module)
        case = ae_module.select_paper_cases(
            ae_module.load_paper_cases(), table="VIII", case_id="relu-16-8"
        )[0]
        observed = {}

        with tempfile.TemporaryDirectory() as tmp:
            temp_root = Path(tmp)
            config_path = temp_root / "config.h"
            original = (
                "#define MVZK_CONFIG_SCALE 10\n"
                "#define MVZK_CONFIG_MULT_PRODUCT_THRESHOLD 256\n"
                "#define MVZK_CONFIG_DEFAULT_BITLEN 32\n"
                "#define MVZK_CONFIG_NON_LINEAR_RELU_DIGDEC_K 3\n"
            )
            config_path.write_text(original, encoding="utf-8")

            def capture_run(specification, binary_path, output_dir, threads):
                observed["arguments"] = specification.arguments
                observed["config"] = config_path.read_text(encoding="utf-8")
                return {"status": "passed"}

            with (
                patch.object(ae_module, "REPO_ROOT", temp_root),
                patch.object(ae_module, "run_command"),
                patch.object(ae_module, "run_two_party", side_effect=capture_run),
            ):
                ae_module.run_manifest_case(
                    case,
                    build_dir=temp_root / "build",
                    output_dir=temp_root / "results",
                    threads=4,
                    jobs=1,
                    network="none",
                )

            self.assertEqual(("800000", "16", "2", "1"), observed["arguments"])
            self.assertIn("MVZK_CONFIG_NON_LINEAR_RELU_DIGDEC_K 2", observed["config"])
            self.assertEqual(original, config_path.read_text(encoding="utf-8"))

    def test_consistency_runner_receives_threads_timeout_and_live_output(self):
        ae_path = REPO_ROOT / "ae.py"
        spec = importlib.util.spec_from_file_location("dyfence_consistency_config", ae_path)
        ae_module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = ae_module
        spec.loader.exec_module(ae_module)
        case = ae_module.select_paper_cases(
            ae_module.load_paper_cases(), table="V", case_id="resnet18-n2"
        )[0]
        observed = {}

        def capture_run(command, cwd, environment, timeout_seconds):
            observed["command"] = [str(part) for part in command]
            observed["environment"] = environment
            observed["timeout_seconds"] = timeout_seconds
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory() as tmp:
            temp_root = Path(tmp)
            with (
                patch.object(ae_module, "REPO_ROOT", REPO_ROOT),
                patch.object(ae_module, "find_safe_consistency_port", return_value=12000),
                patch.object(ae_module, "run_process_group", side_effect=capture_run),
                patch.object(ae_module, "load_consistency_party_metrics", return_value={
                    "prover_metrics": {"runtime_seconds": 1.0},
                    "verifier_metrics": {"runtime_seconds": 1.0},
                }),
            ):
                result = ae_module.run_manifest_case(
                    case,
                    build_dir=temp_root / "build",
                    output_dir=temp_root / "results",
                    threads=4,
                    jobs=8,
                    network="none",
                )

        self.assertEqual("passed", result["status"])
        self.assertEqual("4", observed["environment"]["OMP_NUM_THREADS"])
        self.assertIn("--timeout_seconds", observed["command"])
        self.assertEqual(7500, observed["timeout_seconds"])

    def test_consistency_outer_timeout_is_reported_without_loading_metrics(self):
        ae_path = REPO_ROOT / "ae.py"
        spec = importlib.util.spec_from_file_location("dyfence_consistency_timeout", ae_path)
        ae_module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = ae_module
        spec.loader.exec_module(ae_module)
        case = ae_module.select_paper_cases(
            ae_module.load_paper_cases(), table="V", case_id="resnet18-n2"
        )[0]

        with tempfile.TemporaryDirectory() as tmp:
            temp_root = Path(tmp)
            with (
                patch.object(ae_module, "find_safe_consistency_port", return_value=12000),
                patch.object(ae_module, "run_process_group", side_effect=subprocess.TimeoutExpired(["runner"], 1)),
                patch.object(
                    ae_module,
                    "load_consistency_party_metrics",
                    side_effect=AssertionError("metrics must not be loaded after a timeout"),
                ),
            ):
                result = ae_module.run_manifest_case(
                    case,
                    build_dir=temp_root / "build",
                    output_dir=temp_root / "results",
                    threads=4,
                    jobs=8,
                    network="none",
                )

        self.assertEqual("failed", result["status"])
        self.assertIn("timed out", result["error"])

    def test_frontend_pipeline_exposes_thread_limit(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "frontend" / "run_frontend_pipeline.py"), "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )

        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertIn("--threads", result.stdout)

    def test_eight_core_machine_defaults_to_four_threads_per_party(self):
        module = load_module()
        self.assertEqual(4, module.default_party_threads(cpu_count=8))
        self.assertEqual(1, module.default_party_threads(cpu_count=1))

    def test_explicit_build_jobs_override_party_thread_default(self):
        module = load_module()

        self.assertEqual(8, module.resolve_build_jobs(jobs=None, threads=4))
        self.assertEqual(1, module.resolve_build_jobs(jobs=1, threads=4))
        with self.assertRaises(ValueError):
            module.resolve_build_jobs(jobs=0, threads=4)

    def test_macos_party_commands_do_not_use_linux_affinity_or_preload(self):
        module = load_module()
        commands = module.build_party_commands(
            binary_path=Path("/artifact/model_test"),
            model_dir=Path("/artifact/model"),
            port=23111,
            threads=2,
            cpu_affinity=True,
            platform_name="Darwin",
            cpu_count=8,
            jemalloc_path=None,
        )

        self.assertEqual("/artifact/model_test", commands.prover[0])
        self.assertEqual("/artifact/model_test", commands.verifier[0])
        self.assertNotIn("taskset", commands.prover)
        self.assertNotIn("LD_PRELOAD", commands.environment)
        self.assertEqual("2", commands.environment["OMP_NUM_THREADS"])

    def test_linux_affinity_splits_available_cores_between_parties(self):
        module = load_module()
        commands = module.build_party_commands(
            binary_path=Path("/artifact/model_test"),
            model_dir=Path("/artifact/model"),
            port=23112,
            threads=4,
            cpu_affinity=True,
            platform_name="Linux",
            cpu_count=8,
            jemalloc_path=Path("/usr/lib/libjemalloc.so"),
            taskset_path="/usr/bin/taskset",
        )

        self.assertEqual(["/usr/bin/taskset", "-c", "0-3"], commands.prover[:3])
        self.assertEqual(["/usr/bin/taskset", "-c", "4-7"], commands.verifier[:3])
        self.assertEqual("/usr/lib/libjemalloc.so", commands.environment["LD_PRELOAD"])

    def test_temporary_config_restores_original_bytes_after_failure(self):
        module = load_module()
        original = (
            "#define MVZK_CONFIG_SCALE 7\n"
            "#define MVZK_CONFIG_MULT_PRODUCT_THRESHOLD 128\n"
            "#define MVZK_CONFIG_DEFAULT_BITLEN 24\n"
            "#define MVZK_CONFIG_NON_LINEAR_RELU_DIGDEC_K 2\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.h"
            config_path.write_text(original, encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "injected"):
                with module.temporary_config(
                    config_path,
                    scale=10,
                    product_limit=256,
                    bit_length=32,
                    decomposition_k=3,
                ):
                    changed = config_path.read_text(encoding="utf-8")
                    self.assertIn("MVZK_CONFIG_SCALE 10", changed)
                    self.assertIn("MVZK_CONFIG_DEFAULT_BITLEN 32", changed)
                    self.assertIn("MVZK_CONFIG_NON_LINEAR_RELU_DIGDEC_K 3", changed)
                    raise RuntimeError("injected")

            self.assertEqual(original, config_path.read_text(encoding="utf-8"))

    def test_dry_run_uses_paper_configuration_without_mutating_config(self):
        config_path = REPO_ROOT / "config.h"
        before = config_path.read_bytes()
        env = os.environ.copy()
        result = subprocess.run(
            [
                sys.executable,
                str(MODULE_PATH),
                "--profile",
                "quick",
                "--models",
                "resnet18",
                "--resolutions",
                "32",
                "--threads",
                "2",
                "--dry-run",
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )

        output = result.stdout + result.stderr
        self.assertEqual(0, result.returncode, output)
        self.assertIn("Bit=32", output)
        self.assertIn("K=3", output)
        self.assertNotIn("0-15", output)
        self.assertNotIn("16-31", output)
        self.assertNotIn("/usr/lib/x86_64-linux-gnu/libjemalloc.so", output)
        self.assertEqual(before, config_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
