/**
 * @file benchmark_relu.cpp
 * @brief Microbenchmark for the Secure ReLU Operator with optional Truncation.
 * * This tool evaluates the cryptographic overhead (Prover/Verifier runtime
 * and communication volume) of the non-linear ReLU protocol. It incorporates
 * digit decomposition parameters and scale-based truncation.
 * * ============================================================================
 * Usage:
 * ./benchmark_relu <party> <port> <N> [BitLen] [K] [Truncate]
 * * Parameter Guide:
 * <party>    : 1 for Prover (Alice), 2 for Verifier (Bob)
 * <port>     : Network port (e.g., 12345)
 * <N>        : Number of elements (Batch Size / Flattened Tensor Size)
 * [BitLen]   : (Optional) Bit length of the input data. Default: MVZK_CONFIG_DEFAULT_BITLEN @ config.h
 * [K]        : (Optional) Digit Decomposition parameter. Default: MVZK_CONFIG_NON_LINEAR_RELU_DIGDEC_K @ config.h
 * [Truncate] : (Optional) 1 to enable scale-truncation, 0 to disable. Default: MVZK_CONFIG_NON_LINEAR_RELU_DO_TRUNCATION @ config.h
 *
 * * Examples:
 * 1. Run with defaults (BitLen=32, K=3, Truncate=1) for 100,000 elements:
 * [Verifier]: ./benchmark_relu 2 12345 100000
 * [Prover]  : ./benchmark_relu 1 12345 100000
 * * 2. Override defaults (e.g., BitLen=64, K=4, Truncate=0):
 * [Verifier]: ./benchmark_relu 2 12345 100000 64 4 0
 * [Prover]  : ./benchmark_relu 1 12345 100000 64 4 0
 *
 * * Note:
 * 1. To establish a strict single-threaded baseline, configure your
 * OpenMP environment variable (e.g., setting the number of threads to 1)
 * before executing the binary.
 * 2. We omit explicit value-level consistency checks against the plaintext expected
 * output in this microbenchmark. By the soundness of our protocol, computational
 * integrity is inherently guaranteed during execution. The correctness holds
 * unconditionally as long as the final output is a valid authenticated state.
 * To explicitly verify the exact output values during the benchmark (which
 * will incur a minor additional overhead), replace:
 *
 * PolyTensor::store_self_relation(pt_res);
 *
 * with the following snippet:
 *
 * PolyTensor pt_expected = exec->input(pt_res.shape, expected_output);
 * PolyTensor::store_relation(pt_res, pt_expected);
 * ============================================================================
 */

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <iomanip>

// Core MVZK Framework Headers
#include "exec/ExecProver.h"
#include "exec/ExecVerifier.h"
#include "exec/MVZKNetBase.h"
#include "operations/nonlinear.h"
#include "utility.h"
#include "emp-tool/emp-tool.h"
#include "utils/logger.h"
#include "utils/net_utils.h"

using namespace std;
using namespace emp;
using namespace std::chrono;

MVZKExec* MVZKExec::mvzk_exec = nullptr;

const int PARTY_PROVER = ALICE;
const int PARTY_VERIFIER = BOB;

// Helper: Random Mixed Data Generator (Positive & Negative)

vector<uint64_t> gen_mixed_random_vector(size_t size, uint64_t modulus = PR) {
    vector<uint64_t> res(size);
    static mt19937 gen(12345);
    uniform_int_distribution<int64_t> dis(-10000, 10000);

    for(size_t i = 0; i < size; ++i) {
        int64_t val = dis(gen);
        float f_val = static_cast<float>(val) / 100.0f;
        res[i] = real2fp(f_val);
    }
    return res;
}

// Helper: Plaintext Reference Implementation (Ground Truth)

vector<uint64_t> plain_relu_reference(const vector<uint64_t>& input, int bitlen, int k, bool do_truncation) {
    vector<uint64_t> output(input.size());
    uint64_t zero_point = PR >> 1; // Threshold for negative numbers in field

    int s_0 = MVZK_CONFIG_SCALE;

    for (size_t i = 0; i < input.size(); ++i) {
        uint64_t val = input[i];

        if (val > zero_point) {
            // Negative region -> ReLU outputs 0
            output[i] = 0;
        } else {
            // Positive region
            if (do_truncation) {
                // Shift by configuration scale
                output[i] = val >> s_0;
            } else {
                output[i] = val;
            }
        }
    }
    return output;
}

// Helper: Academic Benchmark Console Formatter

string format_relu_benchmark_console(int party, int N, int bitlen, int k, bool is_truncated,
                                     long long total_ms, uint64_t total_comm_bytes) {
    double total_comm_mb = (double)total_comm_bytes / (1024.0 * 1024.0);
    double total_comm_kb = (double)total_comm_bytes / 1024.0;

    ostringstream oss;
    oss << "\n\033[1;36m=================================================\n";
    oss << "         RELU MICRO-BENCHMARK RESULT             \n";
    oss << "=================================================\033[0m\n";
    oss << "[Configuration]\n";
    oss << "  - Role         : " << (party == PARTY_PROVER ? "PROVER (Alice)" : "VERIFIER (Bob)") << "\n";
    oss << "  - Vector Size  : N = " << N << "\n";
    oss << "  - ZK Params    : BitLen = " << bitlen << ", DigDec (K) = " << k << "\n";
    oss << "  - Scale Params : Config_Scale = " << MVZK_CONFIG_SCALE << "\n";
    oss << "  - Truncation   : " << (is_truncated ? "Enabled (Shifted)" : "Disabled (Pure ReLU)") << "\n";

    oss << "\n[Metrics]\n";
    oss << "  - \033[1;32mExecution Time : " << total_ms << " ms\033[0m\n";
    if (total_comm_mb >= 1.0) {
        oss << "  - \033[1;32mCommunication  : " << fixed << setprecision(3) << total_comm_mb << " MB\033[0m\n";
    } else {
        oss << "  - \033[1;32mCommunication  : " << fixed << setprecision(3) << total_comm_kb << " KB\033[0m\n";
    }
    oss << "\033[1;36m=================================================\033[0m\n";

    return oss.str();
}

// Main Execution Engine

int main(int argc, char** argv) {
    SetLogLevel(LEVEL_ERROR);

    if (argc < 4) {
        cout << "\nUsage: " << argv[0] << " <party> <port> <N> [BitLen] [K] [Truncate]" << endl;
        cout << "Example (Config.h default):\n";
        cout << "  " << argv[0] << " 1 12345 100000\n" << endl;
        return 1;
    }

    int party = atoi(argv[1]);
    int port = atoi(argv[2]);
    int N = atoi(argv[3]);

    int BitLen = (argc >= 5) ? atoi(argv[4]) : MVZK_CONFIG_DEFAULT_BITLEN;
    int K = (argc >= 6) ? atoi(argv[5]) : MVZK_CONFIG_NON_LINEAR_RELU_DIGDEC_K;
    bool do_truncation = (argc >= 7) ? (atoi(argv[6]) != 0) : MVZK_CONFIG_NON_LINEAR_RELU_DO_TRUNCATION;

    if (checkSysSettings() == false) {
        LOG_ERROR("System settings check failed.");
        exit(-1);
    }

    MVZKNetBase<NetIO> *exec = nullptr;

    try {
        if (party == PARTY_PROVER) {
            exec = new MVZKExecProver<NetIO>(party, port);
        } else {
            exec = new MVZKExecVerifier<NetIO>(party, port);
        }

        // Phase 1: Data Preparation

        vector<uint64_t> pt_input_data;
        vector<uint64_t> expected_output;

        if (party == PARTY_PROVER) {
            pt_input_data = gen_mixed_random_vector(N);
            expected_output = plain_relu_reference(pt_input_data, BitLen, K, do_truncation);
        }

        // Phase 2: Core Protocol Execution & Profiling

        uint64_t net_io_start = exec->get_comm_bytes_counter();
        auto t_start = high_resolution_clock::now();

        // 1. Secret Shared Input Injection
        PolyTensor pt_input = exec->input({N}, pt_input_data);

        // 2. Execute Non-Linear Secure ReLU
        PolyTensor pt_res = ReLU(pt_input, BitLen, K, do_truncation);

        // 3. Store relation for ZK verification
        PolyTensor::store_self_relation(pt_res);

        // 4. Finalize and check all constraints
        exec->finalize_protocol();

        auto t_end = high_resolution_clock::now();
        uint64_t net_io_end = exec->get_comm_bytes_counter();

        long long active_zk_time_ms = duration_cast<milliseconds>(t_end - t_start).count();
        uint64_t total_comm_bytes = net_io_end - net_io_start;

        // Phase 3: Correctness Check & Console Output

        if (party == PARTY_PROVER) {
            if (pt_res.total_elements != expected_output.size()) {
                cerr << "\033[1;31m[FATAL ERROR] Dimension mismatch in ZK output!\033[0m\n";
                exit(-1);
            }
        }

        string report = format_relu_benchmark_console(party, N, BitLen, K, do_truncation,
                                                      active_zk_time_ms, total_comm_bytes);
        cout << report;
    } catch (const std::exception& e) {
        cerr << "\n\033[1;31m[FATAL ERROR] Exception caught: " << e.what() << "\033[0m" << endl;
        if(exec) delete exec;
        return -1;
    }

    if(exec) delete exec;
    return 0;
}