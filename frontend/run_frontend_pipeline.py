"""
Usage:
  python run_frontend_pipeline.py --vision_model resnet18 --height 32 --width 32
  python run_frontend_pipeline.py --custom_script my_net.py --model_class SuperNet
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parent

C_CYAN = "\033[1;36m"
C_GREEN = "\033[1;32m"
C_YELLOW = "\033[1;33m"
C_RED = "\033[1;31m"
C_RESET = "\033[0m"

def run_cmd(cmd_list, step_name):
    print(f"\n{C_CYAN}================================================={C_RESET}")
    print(f"{C_CYAN}[STEP] {step_name}{C_RESET}")
    print(f"{C_CYAN}================================================={C_RESET}")
    cmd_str = " ".join(cmd_list)
    print(f"{C_YELLOW}> {cmd_str}{C_RESET}\n")

    result = subprocess.run(cmd_list)
    if result.returncode != 0:
        print(f"\n{C_RED}[FATAL] Pipeline aborted at {step_name}!{C_RESET}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZK-ML Automated Pipeline")
    parser.add_argument('--vision_model', type=str, default=None)
    parser.add_argument('--custom_script', type=str, default=None)
    parser.add_argument('--model_class', type=str, default=None)
    parser.add_argument('--weights', type=str, default=None, help="Path to custom .pth weights")

    parser.add_argument('--batch', type=str, default="1")
    parser.add_argument('--channels', type=str, default="3")
    parser.add_argument('--height', type=str, default="32")
    parser.add_argument('--width', type=str, default="32")

    parser.add_argument('--dataset', type=str, default=None, help="Name of standard dataset (e.g., cifar10, mnist)")
    parser.add_argument('--image_path', type=str, default=None, help="Path to a single real image file")
    parser.add_argument('--data_root', type=str, default='./frontend_dataset', help="Root directory for datasets")

    parser.add_argument('--eval_acc', action='store_true', help="Automatically run fast plaintext accuracy evaluation after compilation")
    parser.add_argument('--threads', type=int, default=None, help="OpenMP/build threads (default: half of available CPUs)")

    args = parser.parse_args()
    thread_limit = args.threads if args.threads is not None else max(1, (os.cpu_count() or 1) // 2)
    if thread_limit < 1:
        parser.error("--threads must be at least 1")

    extractor_cmd = [sys.executable, str(FRONTEND_DIR / "extractor.py"),
                     "--batch", args.batch, "--channels", args.channels,
                     "--height", args.height, "--width", args.width]
    if args.weights:
        extractor_cmd.extend(["--weights", args.weights])

    if args.vision_model:
        extractor_cmd.extend(["--vision_model", args.vision_model])
        model_name = args.vision_model
    elif args.custom_script and args.model_class:
        extractor_cmd.extend(["--custom_script", args.custom_script, "--model_class", args.model_class])
        model_name = args.model_class
    else:
        print(f"{C_RED}Please specify --vision_model OR (--custom_script and --model_class){C_RESET}")
        sys.exit(1)

    run_cmd(extractor_cmd, "Model Extraction & Graph Generation")

    base_dir = Path(__file__).resolve().parent.parent / "generated_model"

    out_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and model_name in d.name], key=os.path.getmtime)
    if not out_dirs:
        print(f"{C_RED}[FATAL] Cannot find generated model directory!{C_RESET}")
        sys.exit(1)

    target_dir = str(out_dirs[-1])
    test_case_dir = str(out_dirs[-1] / "test_cases" / "case_1")

    input_cmd = [sys.executable, str(FRONTEND_DIR / "input_processor.py"), "--target_dir", test_case_dir]

    input_cmd.extend(["--data_root", args.data_root])

    if args.dataset:
        input_cmd.extend(["--dataset", args.dataset])
    elif args.image_path:
        input_cmd.extend(["--image_path", args.image_path])

    run_cmd(input_cmd, "Input Data Generation")

    gt_cmd = [sys.executable, str(FRONTEND_DIR / "ground_truth.py"),
              "--weights_pth", str(Path(target_dir) / "pytorch_weights.pth"),
              "--input_npy", str(Path(test_case_dir) / "input_0.npy"),
              "--output_dir", test_case_dir]

    if args.vision_model:
        gt_cmd.extend(["--vision_model", args.vision_model])
    else:
        gt_cmd.extend(["--custom_script", args.custom_script, "--model_class", args.model_class])

    run_cmd(gt_cmd, "Oracle Ground Truth Calculation")

    compiler_cmd = [sys.executable, str(FRONTEND_DIR / "compiler.py"),
                    "--json", str(Path(target_dir) / "graph.json"),
                    "--model_name", model_name]
    run_cmd(compiler_cmd, "AOT ZK-Circuit Compilation")

    print(f"\n{C_GREEN}================================================={C_RESET}")
    print(f"{C_GREEN} PIPELINE FINISHED SUCCESSFULLY {C_RESET}")
    print(f"{C_GREEN}================================================={C_RESET}")

    repo_dir = FRONTEND_DIR.parent

    if args.eval_acc:
        print(f"\n{C_CYAN}[STEP] Auto-evaluating Plaintext Accuracy (F_p Simulation){C_RESET}")

        build_dir = Path(__file__).resolve().parent.parent / "build"
        build_dir.mkdir(parents=True, exist_ok=True)

        # Verify CMake execution explicitly
        cmake_result = subprocess.run(["cmake", "..", "-DTEST_SELECT=GEN"], cwd=str(build_dir))
        if cmake_result.returncode != 0:
            print(f"\n{C_RED}[FATAL] CMake configuration process failed. (Return Code: {cmake_result.returncode}){C_RESET}")
            sys.exit(cmake_result.returncode)

        # Verify Make compilation explicitly
        make_result = subprocess.run(
            ["cmake", "--build", ".", "--parallel", str(thread_limit)],
            cwd=str(build_dir),
        )
        if make_result.returncode != 0:
            print(f"\n{C_RED}[FATAL] Compilation process failed. (Return Code: {make_result.returncode}){C_RESET}")
            sys.exit(make_result.returncode)

        bin_path = Path(target_dir) / f"{model_name}_test"

        if not bin_path.exists():
            print(f"\n{C_RED}[FATAL] Cannot find compiled binary at {bin_path}! Compilation might have failed.{C_RESET}")
            sys.exit(1)

        eval_cmd = [str(bin_path), "1", "-1", target_dir, "4"]

        run_env = os.environ.copy()

        run_env["OMP_NUM_THREADS"] = str(thread_limit)

        print(f"{C_YELLOW}> env OMP_NUM_THREADS={thread_limit} {' '.join(eval_cmd)}{C_RESET}\n")

        # Capture and verify the evaluation execution result
        eval_result = subprocess.run(eval_cmd, env=run_env)
        if eval_result.returncode != 0:
            print(f"\n{C_RED}[FATAL] Accuracy evaluation execution aborted. (Return Code: {eval_result.returncode}){C_RESET}")
            sys.exit(eval_result.returncode)

        print(f"\n{C_GREEN}==> Accuracy Evaluation Finished!{C_RESET}")

    print(f"Generated C++ files are located at: {C_YELLOW}{target_dir}{C_RESET}")
    print(f"\nNext Steps to run the model with ZK Backend:")
    print(f"1. cmake -S {repo_dir} -B {repo_dir / 'build-model'} -DTEST_SELECT=GEN")
    print(f"2. cmake --build {repo_dir / 'build-model'} --parallel {thread_limit}")
    print(f"3. cd {target_dir}\n")

    print(f"{C_CYAN}[Phase 1: Inference & Checkpoint Dump]{C_RESET}")
    print(f"To run 1-on-1 inference between Prover and Verifier 1 (Party 2):")
    print(f"   [Terminal 1]: ./{model_name}_test 1 12345 . 1 2 1")
    print(f"   [Terminal 2]: ./{model_name}_test 2 12345 . 1 2 1\n")
    print("Repeat Phase 1 for each additional verifier, replacing target party 2")
    print("with its party number in both commands and the verifier's first argument.")
    print("Keep the final save flag 1 to prepare fresh consistency state.\n")

    print(f"{C_CYAN}[Phase 2: Multi-Party Consistency Check]{C_RESET}")
    print("After preparing state for parties 2 and 3, launch all three commands:")
    print(f"   [Terminal 1]: ./{model_name}_test 1 12345 . 2 3")
    print(f"   [Terminal 2]: ./{model_name}_test 2 12345 . 2 3")
    print(f"   [Terminal 3]: ./{model_name}_test 3 12345 . 2 3\n")
