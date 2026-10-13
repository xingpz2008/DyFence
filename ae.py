#!/usr/bin/env python3
import argparse
import csv
import copy
import hashlib
import json
import os
import platform
import re
import signal
import shlex
import shutil
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

from frontend.run_experiment import default_party_threads, find_free_port, temporary_config


REPO_ROOT = Path(__file__).resolve().parent
PAPER_CASES_PATH = REPO_ROOT / "ae_paper_cases.json"
FRONTEND_DIR = REPO_ROOT / "frontend"
MODEL_WEIGHTS_DIR = REPO_ROOT / "model_weights" / "cifar10"

ACCURACY_MODELS = {
    "resnet18": {"cifar_script": "resnet18_cifar10.py", "cifar_class": "ResNet18_CIFAR10", "imagenet_model": "resnet18"},
    "resnet50": {"cifar_script": "resnet50_cifar10.py", "cifar_class": "ResNet50_CIFAR10", "imagenet_model": "resnet50"},
    "resnet101": {"cifar_script": "resnet101_cifar10.py", "cifar_class": "ResNet101_CIFAR10", "imagenet_model": "resnet101"},
    "vgg11": {"cifar_script": "vgg11bn_cifar10.py", "cifar_class": "VGG11_CIFAR10", "imagenet_model": "vgg11_bn"},
    "vgg16": {"cifar_script": "vgg16bn_cifar10.py", "cifar_class": "VGG16_CIFAR10", "imagenet_model": "vgg16_bn"},
}


class BenchmarkSpec(NamedTuple):
    name: str
    binary: str
    arguments: tuple[str, ...]
    paper_result: str
    success_marker: str
    timeout_seconds: float


SMOKE_SPECS = (
    BenchmarkSpec("conv", "benchmark_conv", ("1", "1", "4", "4", "1", "3", "1", "1"), "E1/Table I functional smoke", "CONV2D MICRO-BENCHMARK RESULT", 120),
    BenchmarkSpec("relu", "benchmark_relu", ("64", "32", "3", "1"), "E1/Table I functional smoke", "RELU MICRO-BENCHMARK RESULT", 120),
    BenchmarkSpec("maxpool", "benchmark_maxpool", ("1", "1", "4", "4", "2", "2", "0", "32", "3"), "E1/Table I functional smoke", "MAXPOOL MICRO-BENCHMARK RESULT", 120),
    BenchmarkSpec("integrated_nl", "benchmark_integrated_nl", ("1", "1", "4", "4", "1", "1", "32", "3"), "E1/Table I functional smoke", "INTEGRATED NON-LINEAR", 120),
    BenchmarkSpec("polynomial", "benchmark_polynomial", ("16", "3"), "E1/Table IX functional smoke", "POLY EVAL (x^n) MICRO-BENCHMARK RESULT", 120),
)

def load_paper_cases(path=PAPER_CASES_PATH):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("tables"), dict):
        raise ValueError("unsupported paper-case manifest")
    return payload


def resolve_sample_count(value, dataset):
    if value in (50, 100, "50", "100"):
        return int(value)
    if value == "full":
        try:
            return int(load_paper_cases()["datasets"][dataset]["full_samples"])
        except KeyError as error:
            raise ValueError(f"unknown dataset: {dataset}") from error
    raise ValueError("--samples must be 50, 100, or full")


def select_accuracy_cases(model="all", dataset="all"):
    return [
        entry for entry in load_paper_cases()["tables"]["IV"]
        if model in ("all", entry["model"]) and dataset in ("all", entry["dataset"])
    ]


def select_paper_cases(manifest, table="all", case_id="all"):
    tables = manifest["tables"]
    selected_tables = [key for key in tables if key != "IV"] if table == "all" else [table]
    selected = []
    for table_id in selected_tables:
        for entry in tables[table_id]:
            if case_id != "all" and entry["id"] != case_id:
                continue
            resolved = copy.deepcopy(entry)
            if resolved.get("kind") == "reuse":
                reuse_reference = copy.deepcopy(resolved.get("reference", {}))
                source_table = resolved["source_table"]
                source = next(
                    source for source in tables[source_table]
                    if source["id"] == resolved["source_case"]
                )
                resolved = copy.deepcopy(source)
                resolved["reported_in"] = [source_table, table_id]
                resolved["references"] = {
                    source_table: copy.deepcopy(source.get("reference", {})),
                    table_id: reuse_reference,
                }
            else:
                resolved["reported_in"] = [table_id]
                if "reference" in resolved:
                    resolved["references"] = {table_id: copy.deepcopy(resolved["reference"])}
            resolved["table"] = table_id
            selected.append(resolved)
    if case_id != "all" and not selected:
        raise ValueError(f"case {case_id!r} not found in Table {table}")
    if table == "all":
        unique = []
        by_key = {}
        for entry in selected:
            key = (entry.get("kind"), entry["id"])
            if key in by_key:
                for reported_table in entry["reported_in"]:
                    if reported_table not in by_key[key]["reported_in"]:
                        by_key[key]["reported_in"].append(reported_table)
                by_key[key].setdefault("references", {}).update(entry.get("references", {}))
                continue
            by_key[key] = entry
            unique.append(entry)
        selected = unique
    return selected


def parse_benchmark_metrics(report):
    plain = re.sub(r"\x1b\[[0-9;]*m", "", report)
    runtime_exact_ms = re.search(
        r"(?:Total Runtime|Execution Time|Total Time(?: \(P1\+P2\))?)[^\n]*\(([0-9.]+)\s*ms\)",
        plain,
    )
    runtime = re.search(
        r"(?:Total Runtime|Execution Time|Total Time(?: \(P1\+P2\))?)\s*:\s*([0-9.]+)\s*(ms|s)",
        plain,
    )
    communication = re.search(
        r"(?:Total Communication|Communication|Total Comm(?: \(P1\+P2\))?)\s*:\s*([0-9.]+)\s*(KB|MB|GB)",
        plain,
    )
    metrics = {}
    if runtime_exact_ms:
        metrics["runtime_seconds"] = float(runtime_exact_ms.group(1)) / 1000.0
    elif runtime:
        value = float(runtime.group(1))
        metrics["runtime_seconds"] = value / 1000.0 if runtime.group(2) == "ms" else value
    if communication:
        metrics["communication"] = float(communication.group(1))
        metrics["communication_unit"] = communication.group(2)
    return metrics


def load_frontend_party_metrics(report):
    plain = re.sub(r"\x1b\[[0-9;]*m", "", report)
    matches = re.findall(r"\[SUCCESS\].*?: logs saved to (.+)", plain)
    if not matches:
        raise ValueError("frontend success output does not report its log directory")
    log_dir = Path(matches[-1].strip())
    prover_log = log_dir / "prover.log"
    verifier_log = log_dir / "verifier.log"
    if not prover_log.is_file() or not verifier_log.is_file():
        raise ValueError(f"frontend party logs are missing from {log_dir}")
    prover_metrics = parse_benchmark_metrics(prover_log.read_text(encoding="utf-8", errors="replace"))
    verifier_metrics = parse_benchmark_metrics(verifier_log.read_text(encoding="utf-8", errors="replace"))
    if not prover_metrics or not verifier_metrics:
        raise ValueError(f"frontend party metrics are missing from {log_dir}")
    return {
        "prover_log": str(prover_log),
        "verifier_log": str(verifier_log),
        "prover_metrics": prover_metrics,
        "verifier_metrics": verifier_metrics,
    }


def load_consistency_party_metrics(case_output, num_params):
    log_dir = Path(case_output) / f"logs_N{num_params}"
    prover_log = log_dir / "phase2_prover.log"
    verifier_log = log_dir / "phase2_verifier_1.log"
    if not prover_log.is_file() or not verifier_log.is_file():
        raise ValueError(f"consistency party logs are missing from {log_dir}")
    prover_metrics = parse_benchmark_metrics(prover_log.read_text(encoding="utf-8", errors="replace"))
    verifier_metrics = parse_benchmark_metrics(verifier_log.read_text(encoding="utf-8", errors="replace"))
    if not prover_metrics or not verifier_metrics:
        raise ValueError(f"consistency party metrics are missing from {log_dir}")
    return {
        "prover_log": str(prover_log),
        "verifier_log": str(verifier_log),
        "prover_metrics": prover_metrics,
        "verifier_metrics": verifier_metrics,
    }


def write_results_csv(path, experiments):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case", "table", "network", "party", "status", "runtime_seconds",
        "communication", "communication_unit", "plaintext_accuracy",
        "dyfence_accuracy", "difference_percentage_points", "output_size",
        "parameter_size", "ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for experiment in experiments:
            party_rows = ["prover", "verifier"] if "prover_metrics" in experiment else [""]
            for party in party_rows:
                metrics = experiment.get(f"{party}_metrics", {})
                writer.writerow({
                    "case": experiment["name"],
                    "table": experiment.get("table", ""),
                    "network": experiment.get("network", ""),
                    "party": party,
                    "status": experiment["status"],
                    "runtime_seconds": metrics.get("runtime_seconds", ""),
                    "communication": metrics.get("communication", ""),
                    "communication_unit": metrics.get("communication_unit", ""),
                    "plaintext_accuracy": experiment.get("plaintext", {}).get("percentage", ""),
                    "dyfence_accuracy": experiment.get("dyfence", {}).get("percentage", ""),
                    "difference_percentage_points": experiment.get("difference_percentage_points", ""),
                    "output_size": experiment.get("output", ""),
                    "parameter_size": experiment.get("parameters", ""),
                    "ratio": experiment.get("ratio", ""),
                })


def manifest_case_to_spec(case):
    return BenchmarkSpec(
        case["id"], case["binary"], tuple(case["arguments"]),
        f"Table {case['table']}", case["marker"], float(case.get("timeout_seconds", 7200)),
    )


def network_profile_commands(profile):
    if profile == "none":
        return []
    if profile not in ("lan", "wan"):
        raise ValueError(f"unknown network profile: {profile}")
    script = str(REPO_ROOT / "test" / "throttle.sh")
    return [["bash", script, profile], ["bash", script, "del"]]


def consistency_protocol_ports(base_port, num_verifiers):
    total_parties = num_verifiers + 1
    ports = {base_port + party * 100 for party in range(2, total_parties + 1)}
    if total_parties > 2:
        broadcast_base = base_port + 10000
        ports.update(
            broadcast_base + left * total_parties + right
            for left in range(1, total_parties + 1)
            for right in range(left + 1, total_parties + 1)
        )
    return ports


def find_safe_consistency_port(num_verifiers, candidate_bases=None):
    candidates = candidate_bases if candidate_bases is not None else range(12000, 20001)
    for candidate in candidates:
        probes = []
        try:
            for port in consistency_protocol_ports(candidate, num_verifiers):
                probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                probes.append(probe)
                probe.bind(("0.0.0.0", port))
            return candidate
        except OSError:
            continue
        finally:
            for probe in probes:
                probe.close()
    raise RuntimeError("no safe consistency base port is available")


@contextmanager
def active_network_profile(profile):
    commands = network_profile_commands(profile)
    if not commands:
        yield
        return
    setup, cleanup = commands
    subprocess.run(setup, cwd=REPO_ROOT, check=True)
    try:
        yield
    finally:
        subprocess.run(cleanup, cwd=REPO_ROOT, check=True)


def terminate_process(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


def run_process_group(command, cwd, environment, timeout_seconds):
    process = subprocess.Popen(
        [str(part) for part in command],
        cwd=cwd,
        env=environment,
        start_new_session=True,
    )
    try:
        return subprocess.CompletedProcess(command, process.wait(timeout=timeout_seconds))
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2)
        raise


def run_two_party(spec, binary_path, output_dir, threads, cwd=None, port=None):
    binary_path = Path(binary_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_cwd = Path(cwd).resolve() if cwd is not None else output_dir
    run_cwd.mkdir(parents=True, exist_ok=True)

    selected_port = port if port is not None else find_free_port()
    prover_command = [str(binary_path), "1", str(selected_port), *spec.arguments]
    verifier_command = [str(binary_path), "2", str(selected_port), *spec.arguments]
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(threads)

    prover_log = output_dir / "prover.log"
    verifier_log = output_dir / "verifier.log"
    prover = None
    verifier = None
    timed_out = False
    started_at = time.monotonic()

    try:
        with prover_log.open("w", encoding="utf-8") as prover_out, verifier_log.open("w", encoding="utf-8") as verifier_out:
            verifier = subprocess.Popen(
                verifier_command,
                cwd=run_cwd,
                env=environment,
                stdout=verifier_out,
                stderr=subprocess.STDOUT,
            )
            time.sleep(0.2)
            prover = subprocess.Popen(
                prover_command,
                cwd=run_cwd,
                env=environment,
                stdout=prover_out,
                stderr=subprocess.STDOUT,
            )
            deadline = time.monotonic() + spec.timeout_seconds
            prover.wait(timeout=max(0.01, deadline - time.monotonic()))
            verifier.wait(timeout=max(0.01, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        terminate_process(prover)
        terminate_process(verifier)

    prover_code = None if prover is None else prover.returncode
    verifier_code = None if verifier is None else verifier.returncode
    prover_text = prover_log.read_text(encoding="utf-8", errors="replace") if prover_log.exists() else ""
    verifier_text = verifier_log.read_text(encoding="utf-8", errors="replace") if verifier_log.exists() else ""
    marker_present = spec.success_marker in prover_text and spec.success_marker in verifier_text

    error = None
    if timed_out:
        error = f"timeout after {spec.timeout_seconds} seconds"
    elif prover_code != 0 or verifier_code != 0:
        error = f"party exit codes: prover={prover_code}, verifier={verifier_code}"
    elif not marker_present:
        error = f"required success marker missing: {spec.success_marker}"

    return {
        "name": spec.name,
        "paper_result": spec.paper_result,
        "status": "failed" if error else "passed",
        "error": error,
        "command_arguments": list(spec.arguments),
        "port": selected_port,
        "threads_per_party": threads,
        "elapsed_seconds": round(time.monotonic() - started_at, 3),
        "prover_exit_code": prover_code,
        "verifier_exit_code": verifier_code,
        "prover_log": str(prover_log),
        "verifier_log": str(verifier_log),
        "prover_metrics": parse_benchmark_metrics(prover_text),
        "verifier_metrics": parse_benchmark_metrics(verifier_text),
    }


def git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unavailable"
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def write_summary(path, profile, threads, experiments):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "profile": profile,
        "git_commit": git_commit(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "threads_per_party": threads,
        "experiments": experiments,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return payload


def build_targets(build_dir, jobs):
    build_dir = Path(build_dir).resolve()
    subprocess.run(
        [
            "cmake",
            "-S",
            str(REPO_ROOT),
            "-B",
            str(build_dir),
            "-DDYFENCE_USE_BUNDLED_EMP=ON",
            "-DTEST_SELECT=FUNC",
        ],
        check=True,
    )
    subprocess.run(["cmake", "--build", str(build_dir), "--parallel", str(jobs)], check=True)


def run_specs(specs, build_dir, output_dir, threads):
    results = []
    for spec in specs:
        print(f"[RUN] {spec.name}: {spec.paper_result}")
        result = run_two_party(
            spec,
            Path(build_dir) / spec.binary,
            Path(output_dir) / spec.name,
            threads,
        )
        results.append(result)
        print(f"[{result['status'].upper()}] {spec.name}")
        if result["status"] != "passed":
            break
    return results


def run_manifest_case(case, build_dir, output_dir, threads, jobs, network):
    case_output = Path(output_dir) / f"table-{case['table'].lower()}" / case["id"]
    case_output.mkdir(parents=True, exist_ok=True)
    kind = case["kind"]
    if kind == "benchmark":
        spec = manifest_case_to_spec(case)
        if "scale" in case:
            with temporary_config(
                REPO_ROOT / "config.h", scale=case["scale"], product_limit=256,
                bit_length=case["bit_length"], decomposition_k=case["decomposition_k"],
            ):
                run_command(["cmake", "--build", build_dir, "--target", spec.binary, "--parallel", jobs])
                result = run_two_party(spec, Path(build_dir) / spec.binary, case_output, threads)
        else:
            result = run_two_party(spec, Path(build_dir) / spec.binary, case_output, threads)
        result.update({"table": case["table"], "reported_in": case["reported_in"], "network": network,
                       "reference": case.get("reference"), "references": case.get("references")})
        return result

    if kind == "model":
        model_alias = {"vgg11": "vgg11_bn", "vgg16": "vgg16_bn"}.get(case["model"], case["model"])
        command = [sys.executable, FRONTEND_DIR / "run_experiment.py", "--profile", "quick",
                   "--models", model_alias, "--resolutions", case["resolution"],
                   "--threads", threads, "--jobs", jobs]
        log_path = case_output / "experiment.log"
        completed = subprocess.run([str(part) for part in command], cwd=REPO_ROOT, capture_output=True, text=True)
        report = completed.stdout + completed.stderr
        log_path.write_text(report, encoding="utf-8")
        error = None
        party_metrics = {}
        if completed.returncode != 0:
            error = f"frontend experiment exited with {completed.returncode}"
        else:
            try:
                party_metrics = load_frontend_party_metrics(report)
            except ValueError as metrics_error:
                error = str(metrics_error)
        return {"name": case["id"], "table": case["table"], "reported_in": case["reported_in"],
                "network": network, "status": "failed" if error else "passed",
                "error": error,
                "command": [str(part) for part in command], "log": str(log_path),
                "reference": case.get("reference"), "references": case.get("references"),
                **party_metrics}

    if kind == "consistency":
        port = find_safe_consistency_port(case["verifiers"])
        timeout_seconds = float(case.get("timeout_seconds", 7200))
        command = [sys.executable, REPO_ROOT / "test" / "functional_test" / "run_benchmark_consistency.py",
                   "--num_params", case["parameters"], "--num_verifiers", case["verifiers"],
                   "--port", port, "--build_dir", Path(build_dir).resolve(),
                   "--binary_path", Path(build_dir).resolve() / "benchmark_consistency",
                   "--timeout_seconds", timeout_seconds]
        log_path = case_output / "experiment.log"
        environment = os.environ.copy()
        environment["OMP_NUM_THREADS"] = str(threads)
        rendered_command = [str(part) for part in command]
        print(f"[CONSISTENCY] {case['id']} using base port {port}; detailed party logs: {case_output}", flush=True)
        try:
            completed = run_process_group(
                rendered_command,
                cwd=case_output,
                environment=environment,
                timeout_seconds=timeout_seconds + 300,
            )
            timeout_error = None
        except subprocess.TimeoutExpired:
            completed = subprocess.CompletedProcess(rendered_command, 124)
            timeout_error = f"consistency experiment timed out after {timeout_seconds} seconds"
        log_path.write_text(
            f"Command: {shlex.join(rendered_command)}\nExit code: {completed.returncode}\n",
            encoding="utf-8",
        )
        error = timeout_error
        party_metrics = {}
        if error is None:
            if completed.returncode != 0:
                error = f"consistency experiment exited with {completed.returncode}"
            else:
                try:
                    party_metrics = load_consistency_party_metrics(case_output, case["parameters"])
                except ValueError as metrics_error:
                    error = str(metrics_error)
        return {"name": case["id"], "table": case["table"], "reported_in": case["reported_in"],
                "network": network, "status": "failed" if error else "passed",
                "error": error,
                "command": [str(part) for part in command], "log": str(log_path),
                "reference": case.get("reference"), **party_metrics}

    raise ValueError(f"unsupported paper case kind: {kind}")


def run_reproduction(table, case_id, network, build_dir, output_dir, threads, jobs):
    cases = select_paper_cases(load_paper_cases(), table, case_id)
    results = []
    total_cases = len(cases)
    print(
        f"[PLAN] Table {table}: {total_cases} cases, network={network}, output={Path(output_dir)}",
        flush=True,
    )
    with active_network_profile(network):
        for index, case in enumerate(cases, start=1):
            case_output = Path(output_dir) / f"table-{case['table'].lower()}" / case["id"]
            print(f"[RUN {index}/{total_cases}] {case['id']}", flush=True)
            started_at = time.monotonic()
            result = run_manifest_case(case, build_dir, output_dir, threads, jobs, network)
            results.append(result)
            elapsed = time.monotonic() - started_at
            print(
                f"[{result['status'].upper()} {index}/{total_cases}] {case['id']} "
                f"in {elapsed:.1f}s; outputs: {case_output}",
                flush=True,
            )
            if result["status"] != "passed":
                break
    summary_path = Path(output_dir) / "summary.json"
    write_summary(summary_path, f"table-{table}", threads, results)
    write_results_csv(Path(output_dir) / "results.csv", results)
    print(f"[RESULT] {summary_path}")
    return bool(results) and all(entry["status"] == "passed" for entry in results)


def run_consistency_smoke(build_dir, output_dir, threads):
    work_dir = Path(output_dir) / "consistency-work"
    binary = Path(build_dir) / "benchmark_consistency"
    mode1 = BenchmarkSpec("consistency-mode1", "benchmark_consistency", ("2", "8", "1"), "E1/Tables V-VI functional smoke", "dumped successfully", 120)
    mode2 = BenchmarkSpec("consistency-mode2", "benchmark_consistency", ("2", "8", "2"), "E1/Tables V-VI functional smoke", "CONSISTENCY CHECK REPORT", 120)
    first = run_two_party(mode1, binary, Path(output_dir) / mode1.name, threads, cwd=work_dir)
    if first["status"] != "passed":
        return [first]
    second = run_two_party(mode2, binary, Path(output_dir) / mode2.name, threads, cwd=work_dir)
    return [first, second]


def validate_summary(result_path):
    payload = json.loads(Path(result_path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported or missing summary schema_version")
    experiments = payload.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("summary contains no experiments")
    failed = [entry.get("name", "unnamed") for entry in experiments if entry.get("status") != "passed"]
    if failed:
        raise ValueError("failed experiments: " + ", ".join(failed))
    return len(experiments)


def parse_accuracy_report(report, expected_total):
    plain_report = re.sub(r"\x1b\[[0-9;]*m", "", report)
    matches = re.findall(
        r"ZK Model Accuracy\s*:\s*([0-9]+(?:\.[0-9]+)?)%\s*\((\d+)/(\d+)\)",
        plain_report,
    )
    if not matches:
        raise ValueError("ZK Model Accuracy marker is missing")
    percentage, matched, total = matches[-1]
    matched = int(matched)
    total = int(total)
    if total != expected_total:
        raise ValueError(f"accuracy report contains {total} cases; expected {expected_total}")
    return {"percentage": float(percentage), "correct": matched, "total": total}


def parse_plaintext_accuracy(report, expected_total):
    match = re.search(
        r"Plaintext Top-1 Accuracy\s*:\s*([0-9]+(?:\.[0-9]+)?)%\s*\((\d+)/(\d+)\)",
        report,
    )
    if not match:
        raise ValueError("Plaintext Top-1 Accuracy marker is missing")
    percentage, correct, total = match.groups()
    if int(total) != expected_total:
        raise ValueError(f"plaintext report contains {total} cases; expected {expected_total}")
    return {"percentage": float(percentage), "correct": int(correct), "total": int(total)}


def run_command(command, cwd=REPO_ROOT):
    print(f"[COMMAND] {shlex.join(str(part) for part in command)}")
    subprocess.run([str(part) for part in command], cwd=cwd, check=True)


def run_accuracy_case(model, dataset, samples, data_root, output_dir, build_dir, jobs, threads):
    output_dir = Path(output_dir).resolve()
    build_dir = Path(build_dir).resolve()
    data_root = Path(data_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_count = resolve_sample_count(samples, dataset)
    case_id = f"{model}-{dataset}"
    generated_dir = REPO_ROOT / "generated_model" / f"ae_accuracy_{case_id}_{sample_count}"
    case_dir = generated_dir / "test_cases" / "case_1"
    oracle_log = output_dir / "plaintext.log"
    backend_log = output_dir / "dyfence.log"
    model_config = ACCURACY_MODELS[model]
    resolution = 32 if dataset == "cifar10" else 224
    experiment = {
        "name": case_id,
        "table": "IV",
        "dataset": dataset,
        "model": model,
        "sample_count": sample_count,
        "paper_result": "Table IV",
        "status": "failed",
        "error": None,
        "plaintext_log": str(oracle_log),
        "dyfence_log": str(backend_log),
    }

    try:
        if generated_dir.exists():
            shutil.rmtree(generated_dir)
        extractor = [sys.executable, FRONTEND_DIR / "extractor.py", "--out_dir", generated_dir,
                     "--batch", sample_count, "--channels", 3, "--height", resolution, "--width", resolution]
        architecture = []
        if dataset == "cifar10":
            record = load_paper_cases()["cifar10_checkpoints"][model]
            checkpoint = MODEL_WEIGHTS_DIR / record["file"]
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if digest != record["sha256"]:
                raise ValueError(f"checkpoint hash mismatch: {checkpoint}")
            architecture = ["--custom_script", REPO_ROOT / "local_models" / model_config["cifar_script"],
                            "--model_class", model_config["cifar_class"]]
            extractor += architecture + ["--weights", checkpoint]
            generated_model_name = model_config["cifar_class"]
        else:
            architecture = ["--vision_model", model_config["imagenet_model"]]
            extractor += architecture
            generated_model_name = model_config["imagenet_model"]
        run_command(extractor)
        run_command([sys.executable, FRONTEND_DIR / "input_processor.py", "--target_dir", case_dir,
                     "--dataset", dataset, "--batch", sample_count, "--data_root", data_root])

        oracle_command = [sys.executable, FRONTEND_DIR / "ground_truth.py", "--weights_pth",
                          generated_dir / "pytorch_weights.pth", "--input_npy", case_dir / "input_0.npy",
                          "--labels_npy", case_dir / "labels_0.npy", "--output_dir", case_dir, *architecture]
        oracle = subprocess.run([str(part) for part in oracle_command], cwd=REPO_ROOT, capture_output=True, text=True)
        oracle_text = oracle.stdout + oracle.stderr
        oracle_log.write_text(oracle_text, encoding="utf-8")
        if oracle.returncode != 0:
            raise RuntimeError(f"plaintext oracle exited with code {oracle.returncode}")
        plaintext = parse_plaintext_accuracy(oracle_text, sample_count)

        run_command(
            [
                sys.executable,
                FRONTEND_DIR / "compiler.py",
                "--json",
                generated_dir / "graph.json",
                "--model_name",
                generated_model_name,
            ]
        )
        run_command(
            [
                "cmake",
                "-S",
                REPO_ROOT,
                "-B",
                build_dir,
                "-DDYFENCE_USE_BUNDLED_EMP=ON",
                "-DTEST_SELECT=GEN",
            ]
        )
        run_command(
            [
                "cmake",
                "--build",
                build_dir,
                "--target",
                f"{generated_dir.name}_{generated_model_name}_test",
                "--parallel",
                jobs,
            ]
        )

        binary = generated_dir / f"{generated_model_name}_test"
        backend_command = [binary, "1", "-1", generated_dir, "4"]
        print(f"[COMMAND] OMP_NUM_THREADS={threads} {shlex.join(str(part) for part in backend_command)}")
        environment = os.environ.copy()
        environment["OMP_NUM_THREADS"] = str(threads)
        backend = subprocess.run(
            [str(part) for part in backend_command],
            cwd=REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
        report = backend.stdout + backend.stderr
        backend_log.write_text(report, encoding="utf-8")
        print(report, end="")
        if backend.returncode != 0:
            raise RuntimeError(f"accuracy backend exited with code {backend.returncode}")
        dyfence = parse_accuracy_report(report, sample_count)
        experiment.update(
            {
                "status": "passed",
                "plaintext": plaintext,
                "dyfence": dyfence,
                "difference_percentage_points": round(dyfence["percentage"] - plaintext["percentage"], 2),
            }
        )
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        experiment["error"] = str(error)
        print(f"[FAILED] {case_id}: {error}", file=sys.stderr)

    summary_path = output_dir / "summary.json"
    write_summary(summary_path, f"accuracy-{case_id}-{samples}", threads, [experiment])
    write_results_csv(output_dir / "results.csv", [experiment])
    print(f"[RESULT] {summary_path}")
    return experiment["status"] == "passed"


def run_accuracy_matrix(model, dataset, samples, data_root, output_dir, build_dir, jobs, threads):
    selected = select_accuracy_cases(model, dataset)
    if len(selected) == 1:
        case = selected[0]
        return run_accuracy_case(
            case["model"], case["dataset"], samples, data_root,
            output_dir, build_dir, jobs, threads,
        )
    experiments = []
    for case in selected:
        case_output = Path(output_dir) / case["id"]
        passed = run_accuracy_case(
            case["model"], case["dataset"], samples, data_root,
            case_output, build_dir, jobs, threads,
        )
        experiments.extend(json.loads((case_output / "summary.json").read_text(encoding="utf-8"))["experiments"])
        if not passed:
            break
    write_summary(Path(output_dir) / "summary.json", f"accuracy-{model}-{dataset}-{samples}", threads, experiments)
    write_results_csv(Path(output_dir) / "results.csv", experiments)
    return len(experiments) == len(selected) and all(entry["status"] == "passed" for entry in experiments)


def add_common_run_arguments(parser):
    parser.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build-ae")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--output-dir", type=Path)


def build_parser():
    parser = argparse.ArgumentParser(description="DyFence benchmarks and paper workloads")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build functional benchmark targets")
    build.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build-ae")
    build.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))

    quick = subparsers.add_parser("quick", help="run a bounded functional check")
    add_common_run_arguments(quick)

    reproduce = subparsers.add_parser("reproduce", help="run DyFence claim workloads")
    reproduce.add_argument("--table", choices=("all", "I", "II", "III", "V", "VI", "VIII", "IX"), default="all")
    reproduce.add_argument("--case", default="all", help="manifest case id, or all")
    reproduce.add_argument("--network", choices=("lan", "wan", "none"), default="none")
    reproduce.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    add_common_run_arguments(reproduce)

    accuracy = subparsers.add_parser("accuracy", help="run a Table IV model/dataset case")
    accuracy.add_argument("--model", choices=("all", "resnet18", "resnet50", "resnet101", "vgg11", "vgg16"), required=True)
    accuracy.add_argument("--dataset", choices=("all", "cifar10", "imagenet"), required=True)
    accuracy.add_argument("--samples", choices=("50", "100", "full"), default="50")
    accuracy.add_argument("--data-root", type=Path, required=True)
    accuracy.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build-accuracy")
    accuracy.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    accuracy.add_argument("--threads", type=int, default=default_party_threads())
    accuracy.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "accuracy")

    validate = subparsers.add_parser("validate", help="validate a generated summary")
    validate.add_argument("--result", type=Path, required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "build":
        build_targets(args.build_dir, args.jobs)
        print(f"[SUCCESS] Built DyFence targets in {args.build_dir}")
        return 0

    if args.command == "validate":
        try:
            count = validate_summary(args.result)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(f"[FAILED] {error}", file=sys.stderr)
            return 1
        print(f"[SUCCESS] Validated {count} experiments in {args.result}")
        return 0

    if args.command == "accuracy":
        if args.jobs < 1 or args.threads < 1:
            print("[FAILED] --jobs and --threads must be at least 1", file=sys.stderr)
            return 2
        return 0 if run_accuracy_matrix(
            args.model, args.dataset, args.samples, args.data_root,
            args.output_dir, args.build_dir, args.jobs, args.threads,
        ) else 1

    threads = args.threads if args.threads is not None else default_party_threads()
    if threads < 1:
        print("[FAILED] --threads must be at least 1", file=sys.stderr)
        return 2
    output_dir = args.output_dir or REPO_ROOT / "results" / (
        "quick" if args.command == "quick" else f"table-{args.table}"
    )

    if args.command == "reproduce":
        try:
            success = run_reproduction(
                args.table, args.case, args.network, args.build_dir, output_dir,
                threads, args.jobs,
            )
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            print(f"[FAILED] {error}", file=sys.stderr)
            return 1
        return 0 if success else 1

    specs = SMOKE_SPECS
    results = run_specs(specs, args.build_dir, output_dir, threads)
    if args.command == "quick" and all(entry["status"] == "passed" for entry in results):
        results.extend(run_consistency_smoke(args.build_dir, output_dir, threads))

    summary_path = Path(output_dir) / "summary.json"
    write_summary(summary_path, args.command if args.command == "quick" else f"table-{args.table}", threads, results)
    print(f"[RESULT] {summary_path}")
    return 0 if results and all(entry["status"] == "passed" for entry in results) else 1


if __name__ == "__main__":
    sys.exit(main())
