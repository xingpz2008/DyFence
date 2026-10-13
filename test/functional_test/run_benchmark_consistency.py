"""
================================================================================
Parametric ZK-ML Multi-Party Consistency Experiment Automation
================================================================================
Usage:
  python run_param_benchmark.py --num_params 10000000 --num_verifiers 3
================================================================================
"""
import os
import sys
import time
import shutil
import argparse
import subprocess
from pathlib import Path

# --- Terminal Color Configuration ---
C_CYAN = "\033[1;36m"
C_GREEN = "\033[1;32m"
C_YELLOW = "\033[1;33m"
C_RED = "\033[1;31m"
C_MAGENTA = "\033[1;35m"
C_RESET = "\033[0m"

def run_cmd(cmd_list, step_name, cwd=None):
    print(f"\n{C_CYAN}================================================={C_RESET}")
    print(f"{C_CYAN}[STEP] {step_name}{C_RESET}")
    print(f"{C_CYAN}================================================={C_RESET}")
    cmd_str = " ".join([str(x) for x in cmd_list])
    print(f"{C_YELLOW}> {cmd_str}{C_RESET}\n")

    result = subprocess.run(cmd_list, cwd=cwd)
    if result.returncode != 0:
        print(f"\n{C_RED}[FATAL] Pipeline aborted at {step_name}!{C_RESET}")
        sys.exit(result.returncode)


def require_successful_processes(phase_name, processes):
    """Raise when any process participating in a protocol phase failed."""
    failures = [
        f"party {index} exited with status {process.returncode}"
        for index, process in enumerate(processes, start=1)
        if process.returncode != 0
    ]
    if failures:
        raise RuntimeError(f"{phase_name} failed: " + "; ".join(failures))


def terminate_processes(processes):
    active = [process for process in processes if process.poll() is None]
    for process in active:
        process.terminate()
    for process in active:
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


def wait_for_processes(phase_name, processes, deadline):
    try:
        while True:
            finished = [process for process in processes if process.poll() is not None]
            failed = [process for process in finished if process.returncode != 0]
            if failed:
                require_successful_processes(phase_name, finished)
            if len(finished) == len(processes):
                require_successful_processes(phase_name, processes)
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{phase_name} timed out")
            time.sleep(0.05)
    except (RuntimeError, TimeoutError):
        terminate_processes(processes)
        raise

def create_symlinks_for_experiment(ckpt_dir, num_verifiers):
    print(f"\n{C_MAGENTA}[MAGIC] Creating zero-storage Symlinks for {num_verifiers} Verifiers...{C_RESET}")

    src_prover_dir = ckpt_dir / "Prover_with_Verifier_1"
    src_verifier_dir = ckpt_dir / "Verifier_1_with_Prover"

    if not src_prover_dir.exists() or not src_verifier_dir.exists():
        print(f"{C_RED}[FATAL] Base checkpoints not found at {ckpt_dir}. Did Phase 1 fail?{C_RESET}")
        sys.exit(1)

    for i in range(2, num_verifiers + 1):
        dst_prover_dir = ckpt_dir / f"Prover_with_Verifier_{i}"
        dst_verifier_dir = ckpt_dir / f"Verifier_{i}_with_Prover"

        # Clean up old symlinks if they exist from previous runs
        if dst_prover_dir.exists() or dst_prover_dir.is_symlink():
            os.unlink(dst_prover_dir)
        if dst_verifier_dir.exists() or dst_verifier_dir.is_symlink():
            os.unlink(dst_verifier_dir)

        # Create symlinks (using absolute paths to point to the source directories)
        os.symlink(src_prover_dir.resolve(), dst_prover_dir)
        os.symlink(src_verifier_dir.resolve(), dst_verifier_dir)

        print(f"  -> Linked Verifier {i} data to Verifier 1 (Saved Storage!)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parametric Multi-Party ZK Benchmark")

    # Experiment-specific parameters
    parser.add_argument('--num_params', type=str, default="1000000", help="Number of parameters (e.g., 10000000)")
    parser.add_argument('--num_verifiers', type=int, default=3, help="Total number of verifiers to simulate")
    parser.add_argument('--port', type=str, default="12345", help="Base port for networking")

    # Build and Binary paths
    parser.add_argument('--build_dir', type=str, default="../../build", help="Path to cmake build directory")
    parser.add_argument('--binary_path', type=str, default="../../build/benchmark_consistency", help="Path to the target C++ binary")
    parser.add_argument('--timeout_seconds', type=float, default=7200, help="Maximum time for both protocol phases")

    args = parser.parse_args()

    num_params = args.num_params
    total_parties = args.num_verifiers + 1
    deadline = time.monotonic() + args.timeout_seconds

    # Step 1: Automated C++ Compilation (Restored from your original script)

    build_path = Path(args.build_dir).resolve()
    if not build_path.exists():
        build_path.mkdir(parents=True, exist_ok=True)

    run_cmd(["cmake", "..", "-DTEST_SELECT=FUNC"], "CMake Configuration", cwd=build_path)
    run_cmd(["make", "-j"], "Make C++ Binary", cwd=build_path)

    # Verify if the binary was successfully compiled
    binary_exe = Path(args.binary_path).resolve()
    if not binary_exe.exists():
        print(f"{C_RED}[FATAL] C++ binary not found at {binary_exe} after compilation!{C_RESET}")
        sys.exit(1)

    # Log and file path configuration
    work_dir = Path.cwd()
    log_dir = work_dir / f"logs_N{num_params}"
    ckpt_dir = work_dir / f"checkpoints_N{num_params}"

    # Clean up old data
    if log_dir.exists(): shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    if ckpt_dir.exists(): shutil.rmtree(ckpt_dir)

    # Step 2: Run Phase 1 (Mode 1: Generate Base Checkpoints)

    print(f"\n{C_CYAN}================================================={C_RESET}")
    print(f"{C_CYAN}[STEP] Phase 1: Generating Base Checkpoints (2 Parties){C_RESET}")
    print(f"{C_CYAN}================================================={C_RESET}")

    # C++ Args: <party> <port> <total_parties> <num_params> <mode>
    cmd_p1 = [str(binary_exe), "1", args.port, "2", num_params, "1"]
    cmd_v1 = [str(binary_exe), "2", args.port, "2", num_params, "1"]

    print(f"{C_YELLOW}> Launching Prover and Verifier 1 concurrently...{C_RESET}", flush=True)

    with open(log_dir / "phase1_prover.log", "w") as fp, open(log_dir / "phase1_verifier.log", "w") as fv:
        proc_v1 = subprocess.Popen(cmd_v1, stdout=fv, stderr=subprocess.STDOUT, cwd=work_dir)
        time.sleep(0.5) # Give the Verifier time to start listening
        proc_p1 = subprocess.Popen(cmd_p1, stdout=fp, stderr=subprocess.STDOUT, cwd=work_dir)
        wait_for_processes("Phase 1", [proc_p1, proc_v1], deadline)

    try:
        require_successful_processes("Phase 1", [proc_p1, proc_v1])
    except RuntimeError as error:
        print(f"{C_RED}[FATAL] {error}. Check logs in {log_dir}{C_RESET}")
        sys.exit(1)

    # Step 3: Trigger Symlink Magic

    create_symlinks_for_experiment(ckpt_dir, args.num_verifiers)

    # Step 4: Run Phase 2 (Mode 2: Multi-Party Consistency Check)

    print(f"\n{C_CYAN}================================================={C_RESET}")
    print(f"{C_CYAN}[STEP] Phase 2: Multi-Party Consistency Check ({total_parties} Parties){C_RESET}")
    print(f"{C_CYAN}================================================={C_RESET}")

    # C++ Args: <party> <port> <total_parties> <num_params> <mode>
    cmd_p2 = [str(binary_exe), "1", args.port, str(total_parties), num_params, "2"]

    v_procs = []
    v_files = []

    print(f"{C_YELLOW}> Launching {args.num_verifiers} Verifiers in background...{C_RESET}", flush=True)
    # Launch all Verifiers (Party IDs from 2 to total_parties)
    for p in range(2, total_parties + 1):
        fv = open(log_dir / f"phase2_verifier_{p-1}.log", "w")
        v_files.append(fv)
        cmd_v = [str(binary_exe), str(p), args.port, str(total_parties), num_params, "2"]
        proc_v = subprocess.Popen(cmd_v, stdout=fv, stderr=subprocess.STDOUT, cwd=work_dir)
        v_procs.append(proc_v)

    time.sleep(1) # Wait for Verifiers to initialize network listeners

    print(f"{C_YELLOW}> Launching Prover (Broadcaster)...{C_RESET}", flush=True)
    with open(log_dir / "phase2_prover.log", "w") as fp:
        proc_p2 = subprocess.Popen(cmd_p2, stdout=fp, stderr=subprocess.STDOUT, cwd=work_dir)
        try:
            wait_for_processes("Phase 2", [proc_p2, *v_procs], deadline)
        finally:
            terminate_processes([proc_p2, *v_procs])
            for fv in v_files:
                fv.close()

    try:
        require_successful_processes("Phase 2", [proc_p2, *v_procs])
    except RuntimeError as error:
        print(f"{C_RED}[FATAL] {error}. Check logs in {log_dir}{C_RESET}")
        sys.exit(1)

    print(f"\n{C_GREEN}================================================={C_RESET}")
    print(f"{C_GREEN} EXPERIMENT FINISHED SUCCESSFULLY {C_RESET}")
    print(f"{C_GREEN}================================================={C_RESET}")
    print(f"Results and detailed logs are saved in: {C_YELLOW}{log_dir}{C_RESET}")
