# DyFence Paper-Case Reproduction Guide

The commands run the optimized public implementation on the paper's workload
cases. Numbers in the tables are the original published measurements. The
runner records new measurements in JSON/CSV without changing these references.
The resource/time budgets below describe the archived artifact.

Run commands from the repository root inside the Docker environment entered in
the README. The single source of truth is `ae_paper_cases.json`. Quick checks
use small tensors to establish functionality. Reproduction commands use every
reported case at the exact paper size. Accuracy supports 50-image and
100-image bounded samples and the complete dataset.

## Generic Commands

Replace `TABLE` according to the claim mapping below and run each table with
`NETWORK` set to `lan` and then `wan`.

```bash
python3 ae.py reproduce --table TABLE --case all --network NETWORK \
  --build-dir build-ae --jobs 8 --threads 4 \
  --output-dir results/table-TABLE-NETWORK
```

```bash
python3 ae.py validate --result results/table-TABLE-NETWORK/summary.json
```

The claim mapping and expected paper-host budgets are:

- C1 uses Tables I and VIII. The four LAN/WAN runs take 5 human-minutes and
  10--20 compute-minutes.
- C2 uses Table II and its two shared Table III rows. The two network runs take
  10 human-minutes and 30--60 compute-minutes.
- C3 uses Tables V and VI. The four LAN/WAN runs take 5 human-minutes and
  60--120 compute-minutes. Table VI uses 72 GB RAM.
- C4 uses Table IX. The two LAN/WAN runs take 5 human-minutes and under 10
  compute-minutes.

The runs execute 6 cases for Table I, 10 cases for Table II, 3 cases for Table
V, 4 cases for Table VI, 10 cases for Table VIII, and 6 cases for Table IX.
The validators report the same experiment counts.

The five CIFAR-10 model-definition wrappers are under `local_models/` as the
`*_cifar10.py` files, and their checkpoints are under
`model_weights/cifar10/`. The wrappers instantiate the official TorchVision
architectures and adapt their output layers to ten classes. ImageNet model
definitions and pretrained weights, together with the CIFAR-10 dataset, come
from the official TorchVision package. The ImageNet dataset is not
redistributed due to its licensing restrictions. Users can obtain the
ILSVRC2012 validation set from <https://www.image-net.org/>.

Accuracy accepts 50, 100, or full. Full is 10,000 CIFAR-10 images and 50,000
ImageNet images. Dataset preparation takes about 5 human-minutes.

```bash
python3 ae.py accuracy --model all --dataset all --samples full \
  --data-root datasets --build-dir build-accuracy --jobs 8 --threads 4 \
  --output-dir results/accuracy-full
```

The full-set run may take several days and require about 50 GB of RAM.
All paper accuracy values were measured on the full datasets. The commands below are
quick samples of the same path and take about 5--15 compute-minutes.

```bash
python3 ae.py accuracy --model all --dataset cifar10 --samples 50 \
  --data-root datasets --build-dir build-accuracy --jobs 1 --threads 4 \
  --output-dir results/accuracy-cifar10-50
```

```bash
python3 ae.py accuracy --model all --dataset cifar10 --samples 100 \
  --data-root datasets --build-dir build-accuracy --jobs 1 --threads 4 \
  --output-dir results/accuracy-cifar10-100
```

The five CIFAR checkpoints are in `model_weights/cifar10`. ImageNet validation
images must be in `datasets/imagenet/val`. The torchvision ImageNet weights
are pinned by `frontend/requirements.txt` and `model_weights/VERSIONS.md`.

## Table I

Six cases reproduce the largest ResNet-18 and VGG-16 operator layers.

| Case | Exact workload | Prover LAN/WAN s | Prover comm | Verifier LAN/WAN s | Verifier comm |
|---|---|---:|---:|---:|---:|
| `conv-resnet18` | 1x3x224x224, 64 filters, 7x7, stride 2 | 0.32/0.33 | 1.22 MB | 0.59/0.61 | 0.02 KB |
| `conv-vgg16` | 1x64x224x224, 64 filters, 3x3 | 4.03/4.83 | 24.78 MB | 10.24/11.08 | 0.02 KB |
| `relu-resnet18` | N=802,816, bit 32, K=3 | 6.32/10.02 | 50.13 MB | 6.29/10.01 | 1.31 KB |
| `relu-vgg16` | N=3,211,264, bit 32, K=3 | 27.37/41.81 | 202.17 MB | 27.44/41.91 | 3.41 MB |
| `maxpool-resnet18` | 1x64x112x112, 3x3, stride 2 | 13.47/21.11 | 93.73 MB | 13.47/21.13 | 1.71 MB |
| `maxpool-vgg16` | 1x64x224x224, 2x2, stride 2 | 24.90/39.16 | 183.80 MB | 24.97/39.25 | 3.41 MB |

## Table II

Ten cases use one input and the production frontend/backend. Times below are
prover LAN/WAN, verifier LAN/WAN, prover communication, verifier
communication.

| Case | Prover s | Verifier s | Prover comm GB | Verifier comm MB |
|---|---:|---:|---:|---:|
| `resnet18-cifar10` | 3.23/6.73 | 3.39/6.96 | 0.09 | 1.70 |
| `resnet18-imagenet` | 30.68/48.16 | 30.81/48.32 | 0.27 | 5.12 |
| `resnet50-cifar10` | 7.55/14.87 | 7.83/15.19 | 0.19 | 3.41 |
| `resnet50-imagenet` | 95.12/147.07 | 95.44/147.43 | 0.78 | 17.06 |
| `resnet101-cifar10` | 13.10/26.18 | 13.67/26.78 | 0.34 | 6.82 |
| `resnet101-imagenet` | 144.39/221.38 | 145.01/222.02 | 1.21 | 25.59 |
| `vgg11-cifar10` | 3.90/7.17 | 3.98/7.28 | 0.08 | 1.70 |
| `vgg11-imagenet` | 117.86/190.99 | 119.89/193.03 | 1.47 | 32.40 |
| `vgg16-cifar10` | 6.03/11.20 | 6.20/11.40 | 0.13 | 1.71 |
| `vgg16-imagenet` | 178.67/277.92 | 180.76/280.05 | 1.86 | 40.93 |

## Table III

The two DyFence rows are the same measurements as Table II. The driver records
their shared provenance. `resnet18-cifar10` reports
3.23 s prover, 3.39 s verifier, and 89.77 MB communication. `vgg16-cifar10`
reports 6.03 s prover, 6.20 s verifier, and 129.06 MB communication.

## Table IV

Each of the ten cases uses the real model, checkpoint, dataset preprocessing,
PyTorch oracle, generated C++, and fixed-point backend.

| Model | CIFAR plaintext/DyFence | ImageNet plaintext/DyFence |
|---|---:|---:|
| ResNet-18 | 83.97/83.93 | 69.76/69.66 |
| ResNet-50 | 91.67/91.65 | 80.84/80.36 |
| ResNet-101 | 90.86/90.82 | 81.87/81.65 |
| VGG-11 BN | 91.43/91.45 | 70.37/70.39 |
| VGG-16 BN | 93.24/93.22 | 73.36/73.38 |

The 50- and 100-image results are bounded samples of the complete accuracy
path, and their percentages describe the selected deterministic prefixes. The
full command reproduces the Table IV accuracy claim.

## Table V

All cases use 11,684,712 ResNet-18 parameters.

| Case | Verifiers | Prover LAN/WAN s | Prover comm GB | Verifier LAN/WAN s | Verifier comm MB |
|---|---:|---:|---:|---:|---:|
| `resnet18-n2` | 2 | 5.14/11.48 | 0.18 | 3.09/6.31 | 1.70 |
| `resnet18-n16` | 16 | 40.50/92.86 | 1.41 | 5.64/9.90 | 1.71 |
| `resnet18-n32` | 32 | 79.52/181.36 | 2.81 | 7.93/11.87 | 1.71 |

## Table VI

All four cases use two verifiers.

| Case | Parameters | Prover LAN/WAN s | Prover comm GB | Verifier LAN/WAN s | Verifier comm MB |
|---|---:|---:|---:|---:|---:|
| `resnet18` | 11,684,712 | 5.14/11.48 | 0.18 | 3.09/6.31 | 1.70 |
| `vgg16` | 138,357,544 | 70.01/146.80 | 2.08 | 39.09/77.58 | 22.16 |
| `bert-large` | 340,000,000 | 153.32/339.54 | 5.12 | 79.71/172.74 | 56.25 |
| `gpt2-large` | 774,000,000 | 354.22/779.17 | 11.66 | 184.11/396.21 | 131.26 |

## Table VIII

Ten cases cover convolution and all three `(bit length, scale)` settings for
ReLU, MaxPool, and integrated non-linear computation.
The decomposition parameter is `K=2` for `(16,8)` and `K=3` for `(32,10)`
and `(40,12)`.

| Case | N | Configuration `(bit,scale,K)` | Prover LAN/WAN s | Prover comm MB | Verifier LAN/WAN s | Verifier comm |
|---|---:|---:|---:|---:|---:|---:|
| `conv-resnet18` | 1.18e8 | paper default | 0.32/0.33 | 1.22 | 0.59/0.61 | 0.02 KB |
| `relu-16-8` | 8.00e5 | (16,8,2) | 4.25/6.81 | 36.96 | 4.23/6.80 | 0.88 KB |
| `relu-32-10` | 8.00e5 | (32,10,3) | 6.32/10.02 | 50.13 | 6.29/10.01 | 1.31 KB |
| `relu-40-12` | 8.00e5 | (40,12,3) | 7.15/10.96 | 56.10 | 7.12/10.97 | 1.34 KB |
| `maxpool-16-8` | 1.58e6 | (16,8,2) | 8.77/13.60 | 63.24 | 8.76/13.62 | 1.97 KB |
| `maxpool-32-10` | 1.58e6 | (32,10,3) | 13.47/21.11 | 93.73 | 13.47/21.13 | 1.71 MB |
| `maxpool-40-12` | 1.58e6 | (40,12,3) | 15.85/23.45 | 107.24 | 15.86/23.47 | 1.71 MB |
| `integrated-16-8` | 2.40e6 | (16,8,2) | 9.62/14.99 | 69.42 | 9.60/15.00 | 2.17 KB |
| `integrated-32-10` | 2.40e6 | (32,10,3) | 14.96/23.34 | 103.19 | 14.95/23.35 | 1.71 MB |
| `integrated-40-12` | 2.40e6 | (40,12,3) | 17.41/25.89 | 118.17 | 17.40/25.91 | 1.71 MB |

## Table IX

| Case | N | Degree | Prover LAN/WAN s | Prover comm GB | Verifier LAN/WAN s | Verifier comm KB |
|---|---:|---:|---:|---:|---:|---:|
| `bert-p3` | 4,200,000 | 3 | 1.83/3.05 | 0.03 | 1.93/3.19 | 0.02 |
| `bert-p5` | 4,200,000 | 5 | 2.62/3.87 | 0.03 | 2.73/4.01 | 0.02 |
| `bert-p7` | 4,200,000 | 7 | 3.23/4.53 | 0.03 | 3.34/4.66 | 0.02 |
| `gpt2-p3` | 5,240,000 | 3 | 2.39/3.70 | 0.04 | 2.54/3.86 | 0.02 |
| `gpt2-p5` | 5,240,000 | 5 | 3.42/4.78 | 0.04 | 3.58/4.97 | 0.02 |
| `gpt2-p7` | 5,240,000 | 7 | 4.29/5.59 | 0.04 | 4.46/5.78 | 0.02 |

## Validation and Interpretation

```bash
python3 ae.py validate --result results/table-TABLE-lan/summary.json
```

```bash
python3 ae.py validate --result results/table-TABLE-wan/summary.json
```

The archived artifact used 20 percent runtime and 1 percent communication
comparison tolerances. The public release records measurements for its optimized
implementation alongside the published references. `validate` checks summary
schema, nonempty experiments and their successful statuses; it does not enforce
those comparison tolerances. `results.csv` contains one row per measured party. `summary.json`
records the exact case, arguments, network profile, platform, thread count,
commit, status, and log paths.
