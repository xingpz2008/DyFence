import argparse
import itertools
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.h"
BUILD_DIR = REPO_ROOT / "build"
FRONTEND_DIR = Path(__file__).resolve().parent
GENERATED_DIR = REPO_ROOT / "generated_model"

PAPER_SCALE = 10
PAPER_PRODUCT_LIMIT = 256
PAPER_BIT_LENGTH = 32
PAPER_DECOMPOSITION_K = 3

FULL_MODELS = ["vgg11_bn", "vgg16_bn", "resnet18", "resnet50", "resnet101"]
FULL_RESOLUTIONS = [(32, 32), (224, 224)]

CIFAR10_CUSTOM_MODELS = {
    "vgg11_bn": {"script": "../local_models/vgg11bn_cifar10.py", "class": "VGG11_CIFAR10"},
    "vgg16_bn": {"script": "../local_models/vgg16bn_cifar10.py", "class": "VGG16_CIFAR10"},
    "resnet18": {"script": "../local_models/resnet18_cifar10.py", "class": "ResNet18_CIFAR10"},
    "resnet50": {"script": "../local_models/resnet50_cifar10.py", "class": "ResNet50_CIFAR10"},
    "resnet101": {"script": "../local_models/resnet101_cifar10.py", "class": "ResNet101_CIFAR10"},
}


class ExperimentFailure(RuntimeError):
    """Raised when an experiment cannot produce a valid two-party result."""


class PartyCommands(NamedTuple):
    prover: list[str]
    verifier: list[str]
    environment: dict[str, str]


def default_party_threads(cpu_count=None):
    available = cpu_count if cpu_count is not None else (os.cpu_count() or 1)
    return max(1, available // 2)


def resolve_build_jobs(jobs, threads):
    selected = threads * 2 if jobs is None else jobs
    if selected < 1:
        raise ValueError("build jobs must be at least 1")
    return selected


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def find_jemalloc():
    candidates = (
        Path("/usr/lib/x86_64-linux-gnu/libjemalloc.so.2"),
        Path("/usr/lib/x86_64-linux-gnu/libjemalloc.so"),
        Path("/usr/local/lib/libjemalloc.so"),
        Path("/opt/homebrew/lib/libjemalloc.dylib"),
    )
    return next((path for path in candidates if path.is_file()), None)


def build_party_commands(
    binary_path,
    model_dir,
    port,
    threads,
    cpu_affinity=False,
    platform_name=None,
    cpu_count=None,
    jemalloc_path=None,
    taskset_path=None,
):
    if threads < 1:
        raise ValueError("threads must be at least 1")

    binary_path = str(Path(binary_path))
    model_dir = str(Path(model_dir))
    port = str(port)
    environment = {
        "OMP_NUM_THREADS": str(threads),
        "OMP_PROC_BIND": "TRUE",
        "OMP_PLACES": "CORES",
    }
    if jemalloc_path is not None:
        environment["LD_PRELOAD"] = str(jemalloc_path)

    prover = [binary_path, "1", port, model_dir]
    verifier = [binary_path, "2", port, model_dir]

    current_platform = platform_name or platform.system()
    available_cpus = cpu_count if cpu_count is not None else (os.cpu_count() or 1)
    resolved_taskset = taskset_path or shutil.which("taskset")
    if cpu_affinity and current_platform == "Linux" and resolved_taskset and available_cpus >= threads * 2:
        prover = [resolved_taskset, "-c", f"0-{threads - 1}"] + prover
        verifier = [resolved_taskset, "-c", f"{threads}-{threads * 2 - 1}"] + verifier

    return PartyCommands(prover=prover, verifier=verifier, environment=environment)


def require_successful_parties(prover_returncode, verifier_returncode):
    if prover_returncode != 0 or verifier_returncode != 0:
        raise ExperimentFailure(
            "ZK backend failed: "
            f"prover exit code={prover_returncode}, "
            f"verifier exit code={verifier_returncode}"
        )


def stop_party(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


def wait_for_parties(prover, verifier, timeout_seconds=7200):
    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            codes = (prover.poll(), verifier.poll())
            if any(code is not None and code != 0 for code in codes):
                require_successful_parties(*codes)
            if all(code is not None for code in codes):
                return
            if time.monotonic() >= deadline:
                raise ExperimentFailure(f"party timeout after {timeout_seconds} seconds")
            time.sleep(0.02)
    finally:
        stop_party(prover)
        stop_party(verifier)


def experiment_exit_code(failed_experiments):
    return 1 if failed_experiments else 0


def update_config_h(config_path, scale, product_limit, bit_length, decomposition_k):
    content = Path(config_path).read_text(encoding="utf-8")
    replacements = (
        (r"(#define\s+MVZK_CONFIG_SCALE\s+)\d+", rf"\g<1>{scale}"),
        (r"(#define\s+MVZK_CONFIG_MULT_PRODUCT_THRESHOLD\s+)\d+", rf"\g<1>{product_limit}"),
        (r"(#define\s+MVZK_CONFIG_DEFAULT_BITLEN\s+)\d+", rf"\g<1>{bit_length}"),
        (r"(#define\s+MVZK_CONFIG_NON_LINEAR_RELU_DIGDEC_K\s+)\d+", rf"\g<1>{decomposition_k}"),
    )
    for pattern, replacement in replacements:
        content, count = re.subn(pattern, replacement, content)
        if count != 1:
            raise ExperimentFailure(f"Expected exactly one config macro matching {pattern!r}; found {count}")
    Path(config_path).write_text(content, encoding="utf-8")


@contextmanager
def temporary_config(config_path, scale, product_limit, bit_length, decomposition_k):
    config_path = Path(config_path)
    original = config_path.read_bytes()
    try:
        update_config_h(config_path, scale, product_limit, bit_length, decomposition_k)
        yield
    finally:
        config_path.write_bytes(original)


def global_disk_cleanup():
    print("\n[Cleanup] Removing generated weights, test cases, and model binaries...")
    if not GENERATED_DIR.exists():
        return

    for directory in GENERATED_DIR.iterdir():
        if not directory.is_dir():
            continue
        for item in ("weights", "test_cases", "pytorch_weights.pth"):
            path = directory / item
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        for path in directory.glob("*_test"):
            if path.is_file() and path.suffix == "":
                path.unlink()


def resolve_jemalloc(value):
    if value == "off":
        return None
    if value == "auto":
        return find_jemalloc()
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ExperimentFailure(f"jemalloc library does not exist: {path}")
    return path


def execute_zk_backend(
    model_name,
    latest_model_dir,
    scale,
    product_limit,
    bit_length,
    decomposition_k,
    threads,
    build_jobs=None,
    port=None,
    cpu_affinity=False,
    jemalloc="auto",
    timeout_seconds=7200,
):
    latest_model_dir = Path(latest_model_dir)
    binary_path = latest_model_dir / f"{model_name}_test"
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    with temporary_config(
        CONFIG_PATH,
        scale=scale,
        product_limit=product_limit,
        bit_length=bit_length,
        decomposition_k=decomposition_k,
    ):
        subprocess.run(
            ["cmake", "-S", str(REPO_ROOT), "-B", str(BUILD_DIR), "-DCMAKE_BUILD_TYPE=Release", "-DTEST_SELECT=GEN"],
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", str(BUILD_DIR), "--parallel", str(resolve_build_jobs(build_jobs, threads))],
            check=True,
        )

        if not binary_path.is_file():
            raise ExperimentFailure(f"compiled executable not found: {binary_path}")

        selected_port = port if port is not None else find_free_port()
        commands = build_party_commands(
            binary_path=binary_path,
            model_dir=latest_model_dir,
            port=selected_port,
            threads=threads,
            cpu_affinity=cpu_affinity,
            jemalloc_path=resolve_jemalloc(jemalloc),
        )
        run_env = os.environ.copy()
        run_env.update(commands.environment)

        log_dir = latest_model_dir / f"log_Scale{scale}_Thresh{product_limit}_Bit{bit_length}_K{decomposition_k}"
        if log_dir.exists():
            shutil.rmtree(log_dir)
        log_dir.mkdir(parents=True)

        prover_log = log_dir / "prover.log"
        verifier_log = log_dir / "verifier.log"
        with prover_log.open("w", encoding="utf-8") as prover_out, verifier_log.open("w", encoding="utf-8") as verifier_out:
            prover = verifier = None
            try:
                prover = subprocess.Popen(
                    commands.prover,
                    env=run_env,
                    cwd=latest_model_dir,
                    stdout=prover_out,
                    stderr=subprocess.STDOUT,
                )
                time.sleep(0.5)
                verifier = subprocess.Popen(
                    commands.verifier,
                    env=run_env,
                    cwd=latest_model_dir,
                    stdout=verifier_out,
                    stderr=subprocess.STDOUT,
                )
                wait_for_parties(prover, verifier, timeout_seconds)
            finally:
                stop_party(prover)
                stop_party(verifier)

        require_successful_parties(prover.returncode, verifier.returncode)
        print(f"[SUCCESS] {model_name}: logs saved to {log_dir}")
        return log_dir


def parse_resolution(value):
    normalized = value.lower().replace("x", " ").split()
    if len(normalized) == 1:
        size = int(normalized[0])
        return size, size
    if len(normalized) == 2:
        return int(normalized[0]), int(normalized[1])
    raise argparse.ArgumentTypeError(f"invalid resolution: {value}; use 32 or 224x224")


def build_pipeline_command(model, height, width):
    common = [
        sys.executable,
        str(FRONTEND_DIR / "run_frontend_pipeline.py"),
        "--batch",
        "1",
        "--height",
        str(height),
        "--width",
        str(width),
    ]
    if (height, width) == (32, 32):
        custom = CIFAR10_CUSTOM_MODELS[model]
        return (
            common
            + ["--custom_script", custom["script"], "--model_class", custom["class"]],
            custom["class"],
        )
    return common + ["--vision_model", model], model


def build_parser():
    parser = argparse.ArgumentParser(description="Run DyFence end-to-end model experiments")
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--models", nargs="+", choices=FULL_MODELS)
    parser.add_argument("--resolutions", nargs="+", type=parse_resolution)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--jobs", type=int, help="parallel C++ build jobs (defaults to twice --threads)")
    parser.add_argument("--port", type=int)
    parser.add_argument("--timeout-seconds", type=float, default=7200, help="maximum two-party execution time per model")
    parser.add_argument("--cpu-affinity", action="store_true")
    parser.add_argument("--jemalloc", default="auto", help="auto, off, or an explicit library path")
    parser.add_argument("--cleanup", action="store_true", help="remove generated model inputs and binaries")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    threads = args.threads if args.threads is not None else default_party_threads()
    if threads < 1:
        raise ExperimentFailure("--threads must be at least 1")
    if args.timeout_seconds <= 0:
        raise ExperimentFailure("--timeout-seconds must be positive")
    try:
        build_jobs = resolve_build_jobs(args.jobs, threads)
    except ValueError as error:
        raise ExperimentFailure(f"--jobs must be at least 1: {error}") from error

    models = args.models or (["resnet18"] if args.profile == "quick" else FULL_MODELS)
    resolutions = args.resolutions or ([(32, 32)] if args.profile == "quick" else FULL_RESOLUTIONS)
    macro_combinations = [(PAPER_SCALE, PAPER_PRODUCT_LIMIT, PAPER_BIT_LENGTH, PAPER_DECOMPOSITION_K)]

    print(
        f"[Experiment Config] Profile={args.profile} Threads={threads} Jobs={build_jobs} "
        f"Scale={PAPER_SCALE} Bit={PAPER_BIT_LENGTH} K={PAPER_DECOMPOSITION_K}"
    )

    if args.dry_run:
        for model, (height, width) in itertools.product(models, resolutions):
            command, actual_model = build_pipeline_command(model, height, width)
            print(f"[DRY RUN] {actual_model} ({height}x{width}): {' '.join(command)}")
        return 0

    if args.cleanup:
        global_disk_cleanup()

    failed_experiments = []
    for model, (height, width) in itertools.product(models, resolutions):
        command, actual_model = build_pipeline_command(model, height, width)
        print(f"\n[Pipeline] Generating {actual_model} ({height}x{width})")
        subprocess.run(command, cwd=FRONTEND_DIR, check=True)

        matching_dirs = [
            path for path in GENERATED_DIR.iterdir() if path.is_dir() and actual_model in path.name
        ]
        if not matching_dirs:
            raise ExperimentFailure(f"generated model directory not found for {actual_model}")
        latest_dir = max(matching_dirs, key=lambda path: path.stat().st_mtime)

        for scale, product_limit, bit_length, decomposition_k in macro_combinations:
            identity = (
                f"{actual_model}-{height}x{width}-Scale{scale}-"
                f"Thresh{product_limit}-Bit{bit_length}-K{decomposition_k}"
            )
            try:
                execute_zk_backend(
                    actual_model,
                    latest_dir,
                    scale,
                    product_limit,
                    bit_length,
                    decomposition_k,
                    threads=threads,
                    build_jobs=build_jobs,
                    port=args.port,
                    cpu_affinity=args.cpu_affinity,
                    jemalloc=args.jemalloc,
                    timeout_seconds=args.timeout_seconds,
                )
            except Exception as error:
                print(f"[FAILED] {identity}: {error}", file=sys.stderr)
                failed_experiments.append(identity)

    if args.cleanup:
        global_disk_cleanup()

    if failed_experiments:
        print("[FAILED] Experiment run completed with failures:", file=sys.stderr)
        for identity in failed_experiments:
            print(f"  - {identity}", file=sys.stderr)
    else:
        print("[SUCCESS] All requested experiments completed")
    return experiment_exit_code(failed_experiments)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ExperimentFailure, subprocess.CalledProcessError) as error:
        print(f"[FATAL] {error}", file=sys.stderr)
        sys.exit(1)
