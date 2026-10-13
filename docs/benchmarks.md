# Benchmarks

This document summarizes the common benchmark targets and command patterns.
The exact runtime depends on hardware, network shaping, OpenMP settings, and the
values in `config.h`.

For commands that correspond directly to paper tables, see
[reproduce.md](reproduce.md).

## Network Throttling

`test/throttle.sh` provides the network settings used by the evaluation:

```bash
# 400 Mbps LAN setting
bash test/throttle.sh lan

# 150 Mbps WAN setting with approximately 30 ms RTT on loopback
bash test/throttle.sh wan

# remove any throttling rule installed by the script
bash test/throttle.sh del
```

The script defaults to the loopback interface `lo`. If running across machines
or over a different interface, edit the `DEV` variable in `test/throttle.sh`
before applying the rule. It runs `tc` as root or through `sudo`; the README's
Docker command supplies the required `NET_ADMIN` capability.

## Build Benchmark Targets

```bash
cmake -S . -B build -DTEST_SELECT=FUNC
cmake --build build --parallel 4
```

For legacy local model benchmarks:

```bash
cmake -S . -B build -DTEST_SELECT=LOCALMODEL
cmake --build build --parallel 4
```

This mode uses the C++ and header files in `local_models/` together with
`test/local_model_test/`. The Python files in `local_models/` are CIFAR-10
custom model definitions for frontend `--custom_script` commands and are not
part of `TEST_SELECT=LOCALMODEL`.

For generated frontend models:

```bash
cmake -S . -B build -DTEST_SELECT=GEN
cmake --build build --parallel 4
```

## Functional Benchmarks

Functional benchmarks follow the EMP party convention:

```bash
<binary> <party> <port> <workload arguments...>
```

Use party `1` for the prover and party `2` for the verifier.

### ReLU

```bash
# Terminal 1
./build/benchmark_relu 1 12345 100000

# Terminal 2
./build/benchmark_relu 2 12345 100000
```

Optional arguments control bit length, digit-decomposition parameter, and
truncation mode:

```bash
./build/benchmark_relu <party> <port> <N> [BitLen] [K] [Truncate]
```

### Convolution

```bash
./build/benchmark_conv <party> <port> <N> <C_in> <H> <W> <C_out> <K> <Stride> <Pad>
```

Example ResNet-style first convolution:

```bash
./build/benchmark_conv <party> 12345 1 3 224 224 64 7 2 3
```

### MaxPool

```bash
./build/benchmark_maxpool <party> <port> <N> <C> <H> <W> [Kernel] [Stride] [Pad] [BitLen] [K]
```

### Polynomial Evaluation

```bash
./build/benchmark_polynomial <party> <port> <N_elements> <Power_N>
```

### Integrated Non-linear Block

```bash
./build/benchmark_integrated_nl <party> <port> <N> <C> <H> <W> [UseFusion] [Trunc] [BitLen] [K]
```

## Consistency Check Benchmark

The consistency benchmark uses one prover and multiple verifier parties:

```bash
./build/benchmark_consistency <party> <port> <total_parties> <num_params> <mode>
```

The executable selects the number of parties from `total_parties` at runtime.
Consistency mode uses at least two verifiers. The helper prepares state and
launches all parties; the paper-case runner also handles result collection and
network profiles.

Modes:

- `1`: dump or prepare mock committed state
- `2`: run consistency check

There is also a helper script:

```bash
python3 test/functional_test/run_benchmark_consistency.py \
  --build_dir build \
  --binary_path build/benchmark_consistency
```

## Recording Results

When reporting results, record:

- commit hash
- compiler and CMake versions
- CPU model and memory
- OpenMP thread count
- network bandwidth and latency settings
- `config.h` changes
- command line for each prover and verifier

This makes performance comparisons reproducible and easier to interpret.
