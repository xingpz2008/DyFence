"""
Usage:
  python benchmark_consistency_with_torch_model.py --vision_model resnet18 --num_verifiers 5
"""
import os
import sys
import time
import shutil
import argparse
import subprocess
from pathlib import Path

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

def create_symlinks_for_experiment(ckpt_dir, num_verifiers):
    print(f"\n{C_MAGENTA}[MAGIC] Creating zero-storage Symlinks for {num_verifiers} Verifiers...{C_RESET}")

    src_prover_dir = ckpt_dir / "Prover_with_Verifier_1"
    src_verifier_dir = ckpt_dir / "Verifier_1_with_Prover"

    if not src_prover_dir.exists() or not src_verifier_dir.exists():
        print(f"{C_RED}[FATAL] Base checkpoints not found. Did Phase 1 fail?{C_RESET}")
        sys.exit(1)

    for i in range(2, num_verifiers + 1):
        dst_prover_dir = ckpt_dir / f"Prover_with_Verifier_{i}"
        dst_verifier_dir = ckpt_dir / f"Verifier_{i}_with_Prover"

        if dst_prover_dir.exists() or dst_prover_dir.is_symlink():
            os.unlink(dst_prover_dir)
        if dst_verifier_dir.exists() or dst_verifier_dir.is_symlink():
            os.unlink(dst_verifier_dir)

        os.symlink(src_prover_dir.resolve(), dst_prover_dir)
        os.symlink(src_verifier_dir.resolve(), dst_verifier_dir)

        print(f"  -> Linked Verifier {i} data to Verifier 1 (Saved Storage!)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZK-ML Automated Multi-Party Experiment")

    parser.add_argument('--vision_model', type=str, default="resnet18")
    parser.add_argument('--custom_script', type=str, default=None)
    parser.add_argument('--model_class', type=str, default=None)
    parser.add_argument('--batch', type=str, default="1")
    parser.add_argument('--channels', type=str, default="3")
    parser.add_argument('--height', type=str, default="32")
    parser.add_argument('--width', type=str, default="32")

    parser.add_argument('--num_verifiers', type=int, default=3, help="Total number of verifiers to simulate")
    parser.add_argument('--port', type=str, default="12345", help="Base port for networking")
    parser.add_argument('--build_dir', type=str, default="../build", help="Path to cmake build directory")
    parser.add_argument('--skip_inference', action='store_true', help="Skip Phase 1 inference, only generate ZK shares for Phase 2")

    args = parser.parse_args()

    model_name = args.vision_model if args.vision_model else args.model_class
    total_parties = args.num_verifiers + 1

    extractor_cmd = [sys.executable, "extractor.py", "--batch", args.batch, "--channels", args.channels, "--height", args.height, "--width", args.width]
    if args.vision_model:
        extractor_cmd.extend(["--vision_model", args.vision_model])
    else:
        extractor_cmd.extend(["--custom_script", args.custom_script, "--model_class", args.model_class])
    run_cmd(extractor_cmd, "Model Extraction & Graph Generation")

    base_dir = Path(__file__).resolve().parent.parent / "generated_model"
    out_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and model_name in d.name], key=os.path.getmtime)
    target_dir = str(out_dirs[-1])
    test_case_dir = str(out_dirs[-1] / "test_cases" / "case_1")

    run_cmd([sys.executable, "input_processor.py", "--target_dir", test_case_dir], "Dummy Data Generation")

    gt_cmd = [sys.executable, "ground_truth.py", "--weights_pth", str(Path(target_dir) / "pytorch_weights.pth"), "--input_npy", str(Path(test_case_dir) / "input_0.npy"), "--output_dir", test_case_dir]
    if args.vision_model: gt_cmd.extend(["--vision_model", args.vision_model])
    else: gt_cmd.extend(["--custom_script", args.custom_script, "--model_class", args.model_class])
    run_cmd(gt_cmd, "Oracle Ground Truth Calculation")

    run_cmd([sys.executable, "compiler.py", "--json", str(Path(target_dir) / "graph.json"), "--model_name", model_name], "AOT ZK-Circuit Compilation")

    build_path = Path(args.build_dir).resolve()
    run_cmd(["cmake", "..", "-DTEST_SELECT=GEN"], "CMake Configuration", cwd=build_path)
    run_cmd(["make", "-j"], "Make C++ Binary", cwd=build_path)

    binary_exe = str(Path(target_dir) / f"{model_name}_test")
    log_dir = Path(target_dir) / "log"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{C_CYAN}================================================={C_RESET}")
    print(f"{C_CYAN}[STEP] Phase 1: Real 1-to-1 Execution & Dump{C_RESET}")
    print(f"{C_CYAN}================================================={C_RESET}")

    ckpt_dir = Path(target_dir) / "checkpoints"
    if ckpt_dir.exists(): shutil.rmtree(ckpt_dir)

    phase1_mode = "3" if args.skip_inference else "1"

    if args.skip_inference:
        print(f"{C_MAGENTA}[FAST TRACK] Skipping Forward Pass! Only injecting weights and dumping checkpoints...{C_RESET}")

    cmd_p1 = [binary_exe, "1", args.port, target_dir, phase1_mode, "2", "1"]
    cmd_v1 = [binary_exe, "2", args.port, target_dir, phase1_mode, "-1", "1"]

    print(f"{C_YELLOW}> Launching Prover and Verifier 1 concurrently...{C_RESET}")

    with open(log_dir / "phase1_prover.log", "w") as fp, open(log_dir / "phase1_verifier.log", "w") as fv:
        proc_v1 = subprocess.Popen(cmd_v1, stdout=fv, stderr=subprocess.STDOUT)
        time.sleep(0.5)
        proc_p1 = subprocess.Popen(cmd_p1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        for line in proc_p1.stdout:
            print(line, end="")
            fp.write(line)

        proc_p1.wait()
        proc_v1.wait()

    if proc_p1.returncode != 0 or proc_v1.returncode != 0:
        print(f"{C_RED}[FATAL] Phase 1 Execution Failed! Check logs in {log_dir}{C_RESET}")
        sys.exit(1)

    create_symlinks_for_experiment(ckpt_dir, args.num_verifiers)

    print(f"\n{C_CYAN}================================================={C_RESET}")
    print(f"{C_CYAN}[STEP] Phase 2: Multi-Party Consistency Check ({total_parties} Parties){C_RESET}")
    print(f"{C_CYAN}================================================={C_RESET}")

    cmd_p2 = [binary_exe, "1", args.port, target_dir, "2", str(total_parties)]
    v_procs = []
    v_files = []

    print(f"{C_YELLOW}> Launching {args.num_verifiers} Verifiers in background...{C_RESET}")

    for p in range(2, total_parties + 1):
        fv = open(log_dir / f"phase2_verifier_{p-1}.log", "w")
        v_files.append(fv)
        cmd_v = [binary_exe, str(p), args.port, target_dir, "2", str(total_parties)]
        proc_v = subprocess.Popen(cmd_v, stdout=fv, stderr=subprocess.STDOUT)
        v_procs.append(proc_v)

    time.sleep(1)

    print(f"{C_YELLOW}> Launching Prover (Broadcaster)...{C_RESET}")

    with open(log_dir / "phase2_prover.log", "w") as fp:
        proc_p2 = subprocess.Popen(cmd_p2, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc_p2.stdout:
            print(line, end="")
            fp.write(line)
        proc_p2.wait()

    for proc in v_procs:
        proc.wait()
    for fv in v_files:
        fv.close()

    print(f"\n{C_GREEN}================================================={C_RESET}")
    print(f"{C_GREEN} EXPERIMENT FINISHED SUCCESSFULLY {C_RESET}")
    print(f"{C_GREEN}================================================={C_RESET}")
    print(f"Results and detailed logs are saved in: {C_YELLOW}{log_dir}{C_RESET}")