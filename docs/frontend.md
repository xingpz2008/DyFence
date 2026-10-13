# PyTorch frontend

The frontend extracts supported PyTorch graphs, prepares inputs and oracle
outputs, and generates C++ model protocols. Its entry point is
`frontend/run_frontend_pipeline.py`, which can be invoked from the repository root.

## CIFAR-10 model example

Run in the Docker shell described in the README:

```bash
python3 frontend/run_frontend_pipeline.py \
  --custom_script local_models/resnet18_cifar10.py \
  --model_class ResNet18_CIFAR10 \
  --weights model_weights/cifar10/resnet18_cifar10.pth \
  --batch 1 --channels 3 --height 32 --width 32 --threads 4
```

The pipeline performs graph extraction, input generation, PyTorch oracle
evaluation and C++ generation. It reports its output directory under
`generated_model/ResNet18_CIFAR10_<timestamp>/`. The default input is generated
data; `--dataset cifar10 --data_root datasets` uses dataset images.

Build the generated targets:

```bash
cmake -S . -B build-model -DTEST_SELECT=GEN
cmake --build build-model --parallel 4
```

For inference, launch the generated executable with arguments
`<party> <base-port> <model-directory> 1`. The README provides a complete
two-party invocation and saves both logs.

The wrapper files map as follows:

| Model | Script under `local_models/` | Python class |
|---|---|---|
| ResNet-18 | `resnet18_cifar10.py` | `ResNet18_CIFAR10` |
| ResNet-50 | `resnet50_cifar10.py` | `ResNet50_CIFAR10` |
| ResNet-101 | `resnet101_cifar10.py` | `ResNet101_CIFAR10` |
| VGG-11 BN | `vgg11bn_cifar10.py` | `VGG11_CIFAR10` |
| VGG-16 BN | `vgg16bn_cifar10.py` | `VGG16_CIFAR10` |

Pass the matching `.pth` file from `model_weights/cifar10/` through `--weights`.
The C++ files in `local_models/` are handwritten backend examples; the Python
wrappers are used by the frontend.

## TorchVision and custom models

For a supported ImageNet architecture, use:

```bash
python3 frontend/run_frontend_pipeline.py --vision_model resnet18 \
  --batch 1 --height 224 --width 224 --threads 4
```

For a custom model, select its script and class with `--custom_script` and
`--model_class`. Give the expected input shape through `--batch`, `--channels`,
`--height` and `--width`, and supply trained weights through `--weights` when
needed. Use a static graph composed of supported operators. The operator mapping
is in `frontend/backend_cpp_templates.py`; graph extraction is in
`frontend/extractor.py`.

Generated files include the model header, implementation, test driver, weight
tensors, graph JSON, and input/oracle data. C++ template files are maintained under
`frontend/templates/`.

## Two-host inference

Copy the generated model data to both hosts and use the same binary/configuration.
On the prover, run:

```bash
"$MODEL_DIR/ResNet18_CIFAR10_test" 1 12345 "$MODEL_DIR" 1
```

On the verifier, set the prover's reachable address:

```bash
DYFENCE_PROVER_IP=<prover-ip> \
  "$MODEL_DIR/ResNet18_CIFAR10_test" 2 12345 "$MODEL_DIR" 1
```

An unset or empty variable selects localhost. Allow inbound VOLE ports
`base_port + 200` through `base_port + 200 + MVZK_CONFIG_THREADS_NUM - 1` on
the prover. This setting configures two-party inference; the multi-party
broadcast mesh uses a separate peer-address vector in the backend.

## Fixed-point and accuracy settings

The release defaults are bit length 32, scale 10 and digit-decomposition `K=3`.
`config.h` controls those settings, authenticated I/O threads and range buffers.
`frontend/run_experiment.py` provides configurable model sweeps and restores
the original configuration after each run. Its `--timeout-seconds` option bounds
each two-party protocol run (default 7200 seconds); failed parties stop their peers.

`ae.py accuracy` compares PyTorch and DyFence fixed-point top-1 accuracy on
real data. It supports each model/dataset case, 50/100-image samples, and full
datasets. Use the complete commands in [reproduce.md](reproduce.md).

Model executables support mode `1` for inference, `2` for consistency, `3` for
fast state dumping and `4` for plaintext fixed-point evaluation. Generate protocol
state with this release before running a consistency check.

Consistency mode uses at least two verifiers (three total parties). For each
verifier party `J`, prepare its authenticated state with mode `1` and arguments
`<party> <base-port> <model-directory> 1 J 1`, running parties `1` and `J`
together. The last `1` enables state saving. After preparing every verifier,
launch all parties with `<party> <base-port> <model-directory> 2 <total-parties>`.
The pipeline prints the commands for the three-party case.
