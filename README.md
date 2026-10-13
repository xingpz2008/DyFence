# DyFence

DyFence is a VOLE-based zero-knowledge framework for auditing
Machine-Learning-as-a-Service. It provides a C++ backend for authenticated tensor
computation and a Python frontend that converts supported PyTorch model graphs
into executable C++ protocols.

This public release includes the optimized tensor storage, linear kernels,
range-product computation, and batched degree refresh implementation. The
framework supports CNN inference verification and cross-client model-consistency
checks, with reusable operator and model benchmarks.

The associated paper is **DyFence: Affordable Zero-Knowledge Auditing for
Machine-Learning-as-a-Service**, NDSS 2027. The archived paper artifact is
available through the [Zenodo concept DOI](https://doi.org/10.5281/zenodo.22108428).

## Quick start

Obtain this repository and enter its root directory. Git clones use Git LFS for
the five CIFAR-10 model checkpoints.

```bash
git lfs install
git lfs pull
```

Build the image, prepare persistent output directories, and enter the environment.

```bash
docker build --platform linux/amd64 -t dyfence .
```

```bash
mkdir -p results datasets generated_model
```

```bash
docker run --rm -it --platform linux/amd64 \
  --cap-add=NET_ADMIN \
  -v "$PWD/results:/artifact/results" \
  -v "$PWD/datasets:/artifact/datasets" \
  -v "$PWD/generated_model:/artifact/generated_model" \
  --entrypoint bash dyfence
```

All remaining Python and CMake commands run in this shell. The image includes
Ubuntu 22.04, Python 3.10, CMake 3.22, the compiler, pinned PyTorch/TorchVision packages, and
the bundled EMP source. `NET_ADMIN` allows the benchmark runner to apply LAN/WAN
profiles using Linux traffic control.

Run the seven small operator and consistency checks.

```bash
python3 ae.py quick --build-dir build-ae --threads 4 --output-dir results/quick
```

```bash
python3 ae.py validate --result results/quick/summary.json
```

Expected final output is `[SUCCESS] Validated 7 experiments`. Results include
`summary.json`, `results.csv`, and per-party logs. `ae.py` retains its historical
filename and provides the public benchmark, accuracy, and result-checking commands.

## Convert and run a PyTorch model

This example uses the packaged ResNet-18 CIFAR-10 checkpoint and one generated
32×32 input. It exercises graph extraction, C++ generation, compilation, and
two-party inference. Dataset accuracy is evaluated separately below.

```bash
python3 frontend/run_frontend_pipeline.py \
  --custom_script local_models/resnet18_cifar10.py \
  --model_class ResNet18_CIFAR10 \
  --weights model_weights/cifar10/resnet18_cifar10.pth \
  --batch 1 --channels 3 --height 32 --width 32 --threads 4
```

```bash
cmake -S . -B build-model -DTEST_SELECT=GEN
cmake --build build-model --parallel 4
```

Select the generated model directory and launch both parties.

```bash
MODEL_DIR=$(ls -td generated_model/ResNet18_CIFAR10_* | head -n 1)
```

```bash
OMP_NUM_THREADS=4 "$MODEL_DIR/ResNet18_CIFAR10_test" 2 12345 "$MODEL_DIR" 1 \
  > results/resnet18-verifier.log 2>&1 &
VERIFIER_PID=$!
OMP_NUM_THREADS=4 "$MODEL_DIR/ResNet18_CIFAR10_test" 1 12345 "$MODEL_DIR" 1 \
  > results/resnet18-prover.log 2>&1
wait "$VERIFIER_PID"
```

Both logs should end with `[SUCCESS] ResNet18_CIFAR10 Test Execution Complete!`;
the verifier also reports its PyTorch/DyFence output alignment. Generated C++,
input tensors, and binaries are under `generated_model/`.

Supported graph operations include convolution, linear layers, ReLU, pooling,
residual arithmetic, and flattening. See [docs/frontend.md](docs/frontend.md)
for custom models, fixed-point parameters, and two-host inference.

## Benchmarks and accuracy

The manifest preserves the paper's workload cases for Tables I–VI, VIII, and IX.
Select a table and network profile with the existing runner, for example:

```bash
python3 ae.py reproduce --table II --case all --network lan \
  --build-dir build-ae --jobs 8 --threads 4 --output-dir results/table-II-lan
```

```bash
python3 ae.py validate --result results/table-II-lan/summary.json
```

The optimized release produces its own measurements for these workloads. The
manifest's reference values are the published paper measurements.

Test all five CIFAR-10 checkpoints on 50 real images.

```bash
python3 ae.py accuracy --model all --dataset cifar10 --samples 50 \
  --data-root datasets --build-dir build-accuracy --jobs 1 --threads 4 \
  --output-dir results/accuracy-cifar10-50
```

For the complete accuracy matrix, use `--dataset all --samples full`. Full means
10,000 CIFAR-10 test images or 50,000 ImageNet validation images per model.
[docs/reproduce.md](docs/reproduce.md) provides all cases, full commands, and
paper reference tables.

`validate` checks summary format, nonempty experiments, and their recorded
successful statuses. The runners detect failed party processes, missing required
outputs, wrong accuracy sample counts, and incorrect CIFAR-10 checkpoint hashes.
Runtime and communication references are available for comparison in the result
metadata; `validate` does not enforce a performance tolerance.

## Requirements and assets

- Docker 24 or newer and an x86-64 execution environment. Apple Silicon uses
  the documented `linux/amd64` image through Docker emulation.
- An 8-core CPU and 16 GB RAM for the small checks and most paper workloads.
  More than eight cores are recommended for the Table V 32-client case.
- Allocate 10–12 GB to the container for the Table II VGG ImageNet cases.
- Budget about 50 GB RAM and several days for full accuracy; the largest
  Table VI workloads use a 72 GB memory budget.

The resource budgets above come from the archived artifact. Actual runtime and
memory usage depend on the workload, thread count, and hardware.

The five CIFAR-10 checkpoints are under `model_weights/cifar10/`; model wrappers
are under `local_models/`. Hashes are in [model_weights/VERSIONS.md](model_weights/VERSIONS.md).
ImageNet weights and CIFAR-10 data are obtained through TorchVision. Obtain the
ILSVRC2012 validation set from [ImageNet](https://www.image-net.org/) and arrange
it as `datasets/imagenet/val/<class>/<image>`.

## Repository layout

| Path | Contents |
|---|---|
| `backend/`, `exec/` | Prover/verifier implementation and execution interfaces |
| `data_type/`, `operations/`, `utils/` | Authenticated tensors, operators, and utilities |
| `frontend/` | PyTorch extraction, generation, templates, and automation |
| `local_models/` | CIFAR-10 model wrappers and handwritten C++ models |
| `third_party/` | Fixed EMP source and original licenses |
| `model_weights/` | Model checkpoints and provenance |
| `test/`, `tests/` | C++ checks and Python regression tests |
| `ae.py`, `ae_paper_cases.json` | Benchmark/accuracy runner and paper workloads |
| `docs/` | Model usage and experiment instructions |

## Citation and license

Please cite *DyFence: Affordable Zero-Knowledge Auditing for
Machine-Learning-as-a-Service*, NDSS 2027, when using this implementation.
Software citation metadata is in [CITATION.cff](CITATION.cff).

DyFence is distributed under the [MIT License](LICENSE). Bundled EMP components
retain their individual licenses and provenance in
[third_party/VERSIONS.md](third_party/VERSIONS.md).
