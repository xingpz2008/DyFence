import importlib.util
import contextlib
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from contextlib import redirect_stdout
from unittest import mock
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
AE_PATH = REPO_ROOT / "ae.py"


def load_module():
    spec = importlib.util.spec_from_file_location("dyfence_ae", AE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_fake_party_binary(directory, body):
    path = Path(directory) / "fake_party.py"
    path.write_text("#!/usr/bin/env python3\n" + textwrap.dedent(body), encoding="utf-8")
    path.chmod(0o755)
    return path


class AeCliTest(unittest.TestCase):
    def test_reproduction_reports_case_progress_and_output_directories(self):
        module = load_module()
        cases = [
            {"id": "first-case", "table": "I", "kind": "benchmark"},
            {"id": "second-case", "table": "I", "kind": "benchmark"},
        ]

        def completed_case(case, *_args, **_kwargs):
            return {"name": case["id"], "status": "passed"}

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "results"
            report = io.StringIO()
            with (
                mock.patch.object(module, "select_paper_cases", return_value=cases),
                mock.patch.object(module, "active_network_profile", return_value=contextlib.nullcontext()),
                mock.patch.object(module, "run_manifest_case", side_effect=completed_case),
                mock.patch.object(module, "write_summary"),
                mock.patch.object(module, "write_results_csv"),
                redirect_stdout(report),
            ):
                succeeded = module.run_reproduction(
                    table="I",
                    case_id="all",
                    network="lan",
                    build_dir=Path(tmp) / "build",
                    output_dir=output_dir,
                    threads=4,
                    jobs=8,
                )

        output = report.getvalue()
        self.assertTrue(succeeded)
        self.assertIn("[PLAN] Table I: 2 cases, network=lan", output)
        self.assertIn("[RUN 1/2] first-case", output)
        self.assertIn("[PASSED 1/2] first-case", output)
        self.assertIn(str(output_dir / "table-i" / "first-case"), output)
        self.assertIn("[RUN 2/2] second-case", output)
        self.assertIn("[PASSED 2/2] second-case", output)

    def test_consistency_port_leaves_room_for_protocol_offsets(self):
        module = load_module()
        port = module.find_safe_consistency_port(num_verifiers=32)

        self.assertGreaterEqual(port, 1024)
        self.assertLess(max(module.consistency_protocol_ports(port, 32)), 32768)

    def test_consistency_port_rejects_a_busy_derived_protocol_port(self):
        module = load_module()
        base_port = 12000
        busy_port = base_port + 200

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
            occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            occupied.bind(("127.0.0.1", busy_port))
            occupied.listen(1)
            with self.assertRaisesRegex(RuntimeError, "no safe consistency"):
                module.find_safe_consistency_port(
                    num_verifiers=2,
                    candidate_bases=(base_port,),
                )

    def test_network_profile_commands_reuse_throttle_owner(self):
        module = load_module()
        self.assertEqual([], module.network_profile_commands("none"))
        self.assertEqual(
            [["bash", str(REPO_ROOT / "test" / "throttle.sh"), "lan"],
             ["bash", str(REPO_ROOT / "test" / "throttle.sh"), "del"]],
            module.network_profile_commands("lan"),
        )
        self.assertEqual("wan", module.network_profile_commands("wan")[0][-1])

    def test_network_profile_cleanup_runs_after_experiment_failure(self):
        module = load_module()
        calls = []

        def record(command, **_kwargs):
            calls.append(command)
            return mock.Mock(returncode=0)

        with mock.patch.object(module.subprocess, "run", side_effect=record):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                with module.active_network_profile("wan"):
                    raise RuntimeError("injected")

        self.assertEqual("wan", calls[0][-1])
        self.assertEqual("del", calls[-1][-1])
    def test_case_selector_returns_only_requested_manifest_case(self):
        module = load_module()
        selected = module.select_paper_cases(module.load_paper_cases(), "VIII", "relu-16-8")

        self.assertEqual(1, len(selected))
        self.assertEqual("VIII", selected[0]["table"])
        self.assertEqual("relu-16-8", selected[0]["id"])
        self.assertEqual(16, selected[0]["bit_length"])
        self.assertEqual(8, selected[0]["scale"])

    def test_unknown_case_is_rejected_instead_of_falling_back(self):
        module = load_module()
        with self.assertRaisesRegex(ValueError, "not found"):
            module.select_paper_cases(module.load_paper_cases(), "I", "invented")

    def test_table_iii_selection_expands_to_shared_table_ii_case(self):
        module = load_module()
        selected = module.select_paper_cases(module.load_paper_cases(), "III", "all")

        self.assertEqual(["resnet18-cifar10", "vgg16-cifar10"], [case["id"] for case in selected])
        self.assertTrue(all(case["kind"] == "model" for case in selected))
        self.assertTrue(all(case["reported_in"] == ["II", "III"] for case in selected))

    def test_metric_parser_reads_party_runtime_and_communication(self):
        module = load_module()
        report = "Total Runtime: 6.32 s\nTotal Communication: 50.13 MB\n"
        self.assertEqual(
            {"runtime_seconds": 6.32, "communication": 50.13, "communication_unit": "MB"},
            module.parse_benchmark_metrics(report),
        )

    def test_metric_parser_reads_consistency_report_totals(self):
        module = load_module()
        report = "Total Time (P1+P2) : 922 ms\nTotal Comm (P1+P2) : 179.97 MB\n"

        self.assertEqual(
            {
                "runtime_seconds": 0.922,
                "communication": 179.97,
                "communication_unit": "MB",
            },
            module.parse_benchmark_metrics(report),
        )

    def test_metric_parser_prefers_exact_parenthesized_milliseconds(self):
        module = load_module()
        report = "Total Time  : 0s 911ms  (911 ms)\nTotal Comm  : 89.77 MB\n"

        self.assertEqual(
            {
                "runtime_seconds": 0.911,
                "communication": 89.77,
                "communication_unit": "MB",
            },
            module.parse_benchmark_metrics(report),
        )

    def test_frontend_metrics_are_loaded_from_reported_party_logs(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp) / "log_Scale10_Thresh256_Bit32_K3"
            log_dir.mkdir()
            (log_dir / "prover.log").write_text(
                "Total Runtime: 3.23 s\nTotal Communication: 0.09 GB\n",
                encoding="utf-8",
            )
            (log_dir / "verifier.log").write_text(
                "Total Runtime: 3.39 s\nTotal Communication: 1.70 MB\n",
                encoding="utf-8",
            )
            report = f"[SUCCESS] ResNet18_CIFAR10: logs saved to {log_dir}\n"

            self.assertEqual(
                {
                    "prover_log": str(log_dir / "prover.log"),
                    "verifier_log": str(log_dir / "verifier.log"),
                    "prover_metrics": {
                        "runtime_seconds": 3.23,
                        "communication": 0.09,
                        "communication_unit": "GB",
                    },
                    "verifier_metrics": {
                        "runtime_seconds": 3.39,
                        "communication": 1.70,
                        "communication_unit": "MB",
                    },
                },
                module.load_frontend_party_metrics(report),
            )

    def test_consistency_metrics_are_loaded_from_party_logs(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp)
            log_dir = case_dir / "logs_N100"
            log_dir.mkdir()
            (log_dir / "phase2_prover.log").write_text(
                "Total Time (P1+P2) : 922 ms\nTotal Comm (P1+P2) : 179.97 MB\n",
                encoding="utf-8",
            )
            (log_dir / "phase2_verifier_1.log").write_text(
                "Total Time (P1+P2) : 310 ms\nTotal Comm (P1+P2) : 1.70 MB\n",
                encoding="utf-8",
            )

            loaded = module.load_consistency_party_metrics(case_dir, 100)

            self.assertEqual(0.922, loaded["prover_metrics"]["runtime_seconds"])
            self.assertEqual(179.97, loaded["prover_metrics"]["communication"])
            self.assertEqual(0.31, loaded["verifier_metrics"]["runtime_seconds"])
            self.assertEqual(1.70, loaded["verifier_metrics"]["communication"])

    def test_results_csv_contains_one_row_per_party(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            result = {
                "name": "relu-resnet18", "table": "I", "network": "lan", "status": "passed",
                "prover_metrics": {"runtime_seconds": 6.32, "communication": 50.13, "communication_unit": "MB"},
                "verifier_metrics": {"runtime_seconds": 6.29, "communication": 1.31, "communication_unit": "KB"},
            }
            path = Path(tmp) / "results.csv"
            module.write_results_csv(path, [result])
            rows = path.read_text(encoding="utf-8").splitlines()

            self.assertEqual(3, len(rows))
            self.assertIn("relu-resnet18,I,lan,prover,passed,6.32,50.13,MB", rows[1])
            self.assertIn("relu-resnet18,I,lan,verifier,passed,6.29,1.31,KB", rows[2])
    def test_git_commit_is_unavailable_when_git_is_not_installed(self):
        module = load_module()
        with mock.patch.object(module.subprocess, "run", side_effect=FileNotFoundError):
            self.assertEqual("unavailable", module.git_commit())

    def test_help_lists_stable_evaluator_commands(self):
        result = subprocess.run(
            [sys.executable, str(AE_PATH), "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        for command in ("build", "quick", "reproduce", "accuracy", "validate"):
            self.assertIn(command, result.stdout)

    def test_reproduce_accepts_generic_table_case_and_network_selectors(self):
        result = subprocess.run(
            [sys.executable, str(AE_PATH), "reproduce", "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertIn("--table", result.stdout)
        self.assertIn("--case", result.stdout)
        self.assertIn("--network", result.stdout)
        self.assertIn("{all,I,II,III,V,VI,VIII,IX}", result.stdout)
        self.assertNotIn("VII,", result.stdout)

    def test_reproduce_rejects_a_table_outside_the_appendix_scope(self):
        result = subprocess.run(
            [sys.executable, str(AE_PATH), "reproduce", "--table", "VII"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertNotEqual(0, result.returncode)
        self.assertIn("invalid choice: 'VII'", result.stderr)

    def test_accuracy_accepts_all_paper_case_dimensions(self):
        result = subprocess.run(
            [sys.executable, str(AE_PATH), "accuracy", "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        for option in ("--model", "--dataset", "--samples", "--data-root"):
            self.assertIn(option, result.stdout)

    def test_two_party_runner_requires_both_exit_codes_and_success_marker(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            binary = make_fake_party_binary(
                tmp,
                """
                import sys
                print(f"party={sys.argv[1]} BENCHMARK RESULT")
                raise SystemExit(0)
                """,
            )
            spec = module.BenchmarkSpec(
                name="fake",
                binary="fake_party.py",
                arguments=("8",),
                paper_result="test-only",
                success_marker="BENCHMARK RESULT",
                timeout_seconds=5,
            )
            result = module.run_two_party(
                spec=spec,
                binary_path=binary,
                output_dir=Path(tmp) / "logs",
                threads=1,
                port=23001,
            )

            self.assertEqual("passed", result["status"])
            self.assertEqual(0, result["prover_exit_code"])
            self.assertEqual(0, result["verifier_exit_code"])
            self.assertTrue(Path(result["prover_log"]).is_file())
            self.assertTrue(Path(result["verifier_log"]).is_file())

    def test_two_party_runner_fails_when_marker_is_missing(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            binary = make_fake_party_binary(tmp, "print('finished without marker')\n")
            spec = module.BenchmarkSpec("missing", "fake_party.py", (), "test-only", "REQUIRED", 5)
            result = module.run_two_party(spec, binary, Path(tmp) / "logs", threads=1, port=23002)
            self.assertEqual("failed", result["status"])
            self.assertIn("success marker", result["error"])

    def test_two_party_runner_times_out_and_reports_failure(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            binary = make_fake_party_binary(tmp, "import time\ntime.sleep(2)\n")
            spec = module.BenchmarkSpec("slow", "fake_party.py", (), "test-only", "RESULT", 0.1)
            result = module.run_two_party(spec, binary, Path(tmp) / "logs", threads=1, port=23003)
            self.assertEqual("failed", result["status"])
            self.assertIn("timeout", result["error"])

    def test_process_group_runner_enforces_outer_timeout(self):
        module = load_module()
        started = time.monotonic()
        with self.assertRaises(subprocess.TimeoutExpired):
            module.run_process_group(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                cwd=REPO_ROOT,
                environment=os.environ.copy(),
                timeout_seconds=0.1,
            )
        self.assertLess(time.monotonic() - started, 3)

    def test_validate_returns_nonzero_for_failed_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / "summary.json"
            summary.write_text(
                json.dumps({"schema_version": 1, "experiments": [{"name": "relu", "status": "failed"}]}),
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, str(AE_PATH), "validate", "--result", str(summary)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertNotEqual(0, result.returncode)
            self.assertIn("relu", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
