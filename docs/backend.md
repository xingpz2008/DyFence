# Backend API Overview

The C++ backend provides authenticated tensors, neural-network operators and
prover/verifier execution interfaces. This guide covers their usage, tensor
ownership, protocol state and CMake integration.

## Execution Model

The backend follows the EMP two-party convention:

- party `1`: prover
- party `2`: verifier

For multi-verifier consistency checks, party `1` is still the prover and parties
`2..N` are verifiers.

The main execution classes are:

- `MVZKExec`: abstract execution interface shared by prover and verifier.
- `MVZKExecProver<IO>`: prover-side implementation.
- `MVZKExecVerifier<IO>`: verifier-side implementation.
- `MVZKNetworkManager<IO>`: network setup for VOLE channels and optional
  broadcast channels.
- `MVZKNetBase<IO>`: common network-backed execution base.

Most high-level code does not call prover/verifier methods directly. Instead,
it constructs the correct executor for the party and then uses tensor
operations. `PolyTensor` and `PolyDelta` route arithmetic through the global
`MVZKExec::mvzk_exec` pointer set during executor construction.

`exec/ExecProver.h` and `exec/ExecVerifier.h` contain the public class
declarations, constructors, and template include points. The method bodies are
split by responsibility under `backend/ProverImpl/` and
`backend/VerifierImpl/`.

## Data Structures

### `PolyTensor`

`PolyTensor` is the primary tensor abstraction for authenticated values. It
stores:

- `shape`: tensor dimensions.
- `total_elements`: flattened element count.
- `degree`: polynomial/authentication degree.
- `flat_coeffs`: prover-side polynomial coefficients.
- `flat_keys`: verifier-side MAC keys.

Authenticated tensors allocate coefficients on the prover and keys on the
verifier. `PolyTensor::Store` also supports explicit `Both`, `Coeffs`, and `Keys`
storage; plaintext execution uses coefficient storage. Executor construction sets
the default store for its party.

The class uses a structure-of-arrays layout: all degree-0 coefficients are
stored contiguously, followed by degree-1 coefficients, and so on. The highest
degree coefficient is treated as the plaintext value on the prover side.

Common methods:

- `PolyTensor(shape, degree)`: allocate a tensor of the given shape and degree.
- `PolyTensor::from_public(shape, data)`: create a public degree-0 tensor.
- `clone()`: deep-copy a tensor.
- `flatten()` and `reshape(new_shape)`: shape-only transformations.
- `refresh_degree(check_name)`: check and re-authenticate a high-degree tensor.
- `MatMul(rhs)`: matrix multiplication.
- arithmetic operators: `+`, `-`, `*`, `+=`, `-=`, `*=`.
- relation helpers: `store_relation`, `store_zero_relation`,
  `store_self_relation`.
- `save_to_file` / `load_from_file`: checkpoint tensor state.

The mutable `flatten()` and `reshape()` overloads move the buffers into their
result, leaving the input empty. The const overloads deep-copy. Use `clone()` or
a const reference when the original tensor must remain available. Generated
models transfer ownership at the graph's final shape transformations.

Saved `.bin` tensor state uses this release's internal layout. Generate fresh
state with this release before consistency checks. PyTorch `.pth` checkpoint
files are independent of that layout.

### `PolyDelta`

`PolyDelta` is the scalar counterpart of `PolyTensor`. It represents one
authenticated field element with a polynomial degree, prover coefficients, and a
verifier key. It is useful for scalar constraints and lower-level protocol
building blocks.

Common methods:

- `clone()`
- arithmetic operators: `+`, `-`, `*`, `+=`, `-=`, scalar multiplication
- relation helpers: `store_relation(lhs, rhs)` and `store_relation(lhs, value)`

## Input and Output

The core input API is exposed through `MVZKExec`:

```cpp
PolyTensor input(const std::vector<int>& shape,
                 const std::vector<uint64_t>& raw_data);
```

On the prover side, `raw_data` contains the committed field elements. On the
verifier side, the same call is made with shape-compatible placeholder data so
that both parties execute the same protocol schedule.

To reveal an output tensor to the verifier:

```cpp
std::vector<uint64_t> y = exec->reveal(output_tensor);
```

Generated model tests use this pattern internally. Functional benchmarks also
call relation helpers and finalization routines so buffered checks are executed
before reporting success.

## Model Operations

The public operation wrappers live in `operations/` and dispatch to the current
executor.

### Linear Operations

Header: `operations/linear.h`

- `Conv2D(input, weight, bias, stride, padding, dilation)`
- `Conv1D(input, weight, bias, stride, padding, dilation)`
- `Linear(input, weight, bias)`
- `AvgPool2D(x, kernel_size, stride, padding, back_to_sum_pool)`
- `BatchNorm2D(...)` is deprecated for online verification; use offline
  Conv-BN folding in the frontend instead.

Expected tensor layouts:

- Conv2D input: `(N, C_in, H_in, W_in)`
- Conv2D weight: `(C_out, C_in, kH, kW)`
- Conv2D output: `(N, C_out, H_out, W_out)`
- Conv1D input: `(N, C_in, L_in)`
- Linear weight: `(Out, In)`

### Non-linear Operations

Header: `operations/nonlinear.h`

- `ReLU(x, bitlen, digdec_k, do_truncation, scale)`
- `MaxPool2D(x, kernel_size, stride, padding, bitlen, digdec_k, scale)`
- `IntegratedNL(x, kernel_size, stride, padding, bitlen, digdec_k,
  do_truncation, scale)`

Defaults are taken from `config.h`, including:

- `MVZK_CONFIG_DEFAULT_BITLEN`
- `MVZK_CONFIG_NON_LINEAR_RELU_DIGDEC_K`
- `MVZK_CONFIG_NON_LINEAR_RELU_DO_TRUNCATION`
- `MVZK_CONFIG_SCALE`

`IntegratedNL` is intended for fused non-linear blocks, such as ReLU with
MaxPool, when the model structure allows it.

## Lookup and Range Checks

Lookup and range-check logic is exposed through `MVZKExec` and wrapped by
helper classes in `operations/lut.h`.

The core flow is:

1. register a public table;
2. submit lookup or range-check requests;
3. flush the buffered requests;
4. run the accumulated polynomial checks.

Relevant methods:

- `register_lut_table(data)`
- `submit_lut_check(keys, vals, table_id)`
- `flush_all_luts(require_immediate_check)`
- `register_range_check_table(data)`
- `submit_range_check(x, table_id)`
- `flush_all_range_checks(require_immediate_check)`

In most model code, users call higher-level operators such as `ReLU`,
`MaxPool2D`, or `IntegratedNL`; these operators register and flush the necessary
range checks internally.

## Consistency Checks

Consistency checks compare committed tensors across verifier sessions. The
backend API is:

```cpp
void consistency_check(const std::vector<std::vector<PolyTensor*>>& data);
```

The outer vector groups verifier sessions or checkpoint chunks, and the inner
vector contains tensor pointers to be checked. In practice, generated model
executables and benchmark scripts handle checkpoint loading and call this API
for model-weight consistency experiments.

## Typical Backend Usage

For hand-written tests, the flow is:

```cpp
#include "exec/ExecProver.h"
#include "exec/ExecVerifier.h"
#include "operations/linear.h"
#include "operations/nonlinear.h"

using namespace emp;

int party = std::atoi(argv[1]);
int port = std::atoi(argv[2]);

MVZKExec* exec = nullptr;
if (party == ALICE) {
    exec = new MVZKExecProver<NetIO>(party, port);
} else {
    exec = new MVZKExecVerifier<NetIO>(party, port);
}

PolyTensor x = exec->input({1, 3, 224, 224}, input_data);
PolyTensor w = exec->input({64, 3, 7, 7}, weight_data);
PolyTensor b = exec->input({64}, bias_data);

PolyTensor y = Conv2D(x, w, b, 2, 3);
y = ReLU(y);

std::vector<uint64_t> opened = exec->reveal(y);
exec->finalize_protocol();

delete exec;
```

Verifier-side callers still pass vectors with the correct shape. The verifier
does not learn prover-private inputs from those vectors; they are placeholders
used to keep the local execution schedule aligned.

Most users should prefer the frontend-generated code path described in
[frontend.md](frontend.md). The backend API is mainly useful for adding new
operators, writing microbenchmarks, and debugging generated models.

## Install Surface

For CMake integration, use the source tree so dependency linkage is propagated:

```cmake
add_subdirectory(path/to/DyFence dyfence)
target_link_libraries(your_target PRIVATE DyFence::dyfence_core)
```

The bundled EMP source is selected by default. Set
`-DDYFENCE_USE_BUNDLED_EMP=OFF` to use an existing EMP installation.

The C++ install target exposes the backend as `lib/libdyfence_core.a` plus
headers under `include/dyfence/`. Installed headers preserve the repository
layout below that prefix, for example:

```cpp
#include "exec/ExecProver.h"
#include "exec/ExecVerifier.h"
#include "operations/linear.h"
```

For manual installed-library integration, put `<prefix>/include/dyfence` on the
include path and supply EMP headers/libraries, OpenMP, OpenSSL and threading
dependencies. The install target packages the DyFence archive and headers; it
does not export a standalone `find_package(DyFence)` configuration.

The install surface intentionally excludes frontend scripts, tests, generated
models, datasets, virtual environments, logs, and temporary development files.

## Where to Look First

- `data_type/PolyTensor.h`: tensor representation and tensor arithmetic surface.
- `exec/MVZKExec.h`: abstract execution interface.
- `exec/ExecProver.h` and `exec/ExecVerifier.h`: party-specific declarations
  and constructors.
- `backend/ProverImpl/` and `backend/VerifierImpl/`: party-specific protocol
  method bodies split by responsibility.
- `exec/MVZKNetworkManager.h`: party, port, and multi-verifier network setup.
- `operations/linear.h`: model linear operators.
- `operations/nonlinear.h`: ReLU, MaxPool, and fused non-linear operators.
- `frontend/templates/`: generated model headers, implementations and test drivers.
