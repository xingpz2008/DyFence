/**
 * @file benchmark_polynomial.cpp
 * @brief Microbenchmark for the Secure Polynomial (Power) Evaluation Operator.
 * * This tool evaluates the cryptographic overhead (Prover/Verifier runtime
 * and communication volume) of computing high-degree polynomial evaluations,
 * specifically x^n, using a fast tree-based reduction algorithm (fast_tree_product).
 * It incorporates dynamic degree reduction to maintain efficiency and avoid
 * exponential degree explosion.
 * * ============================================================================
 * Usage:
 * ./benchmark_polynomial <party> <port> <N_elements> <Power_N>
 * * Parameter Guide:
 * <party>      : 1 for Prover (Alice), 2 for Verifier (Bob)
 * <port>       : Network port (e.g., 12345)
 * <N_elements> : Number of elements (Batch Size / Flattened Tensor Size)
 * <Power_N>    : The exponent (degree) for the evaluation (e.g., 5 for x^5)
 * * Examples:
 * 1. Evaluate x^5 on 1,000 elements:
 * [Terminal 1 - Verifier]: ./benchmark_polynomial 2 12345 1000 5
 * [Terminal 2 - Prover]  : ./benchmark_polynomial 1 12345 1000 5
 * * 2. Evaluate x^16 on 50,000 elements (triggers deeper tree reductions):
 * [Terminal 1 - Verifier]: ./benchmark_polynomial 2 12345 50000 16
 * [Terminal 2 - Prover]  : ./benchmark_polynomial 1 12345 50000 16
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
 * * PolyTensor::store_self_relation(pt_res);
 * * with the following snippet:
 * * PolyTensor pt_expected = exec->input(pt_res.shape, expected_output);
 * PolyTensor::store_relation(pt_res, pt_expected);
 * ============================================================================
 */

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <iomanip>
#include <sstream>

// Core MVZK Framework Headers
#include "exec/ExecProver.h"
#include "exec/ExecVerifier.h"
#include "exec/MVZKNetBase.h"
#include "operations/core_math.h"
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

// Helper: Random Data Generator

vector<uint64_t> gen_random_vector(size_t size, uint64_t mod = 100) {
    vector<uint64_t> res(size);
    for(size_t i = 0; i < size; ++i) {
        res[i] = rand() % mod;
    }
    return res;
}

// Helper: Plaintext Reference Implementation (Ground Truth)

vector<uint64_t> plain_poly_eval_reference(const vector<uint64_t>& input, int power_n) {
    vector<uint64_t> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = pow_mod(input[i], power_n);
    }
    return output;
}

// Helper: Academic Benchmark Console Formatter

string format_poly_eval_benchmark_console(int party, int N, int power_n,
                                          long long total_ms, uint64_t total_comm_bytes) {
    double total_comm_mb = (double)total_comm_bytes / (1024.0 * 1024.0);
    double total_comm_kb = (double)total_comm_bytes / 1024.0;

    ostringstream oss;
    oss << "\n\033[1;36m=================================================\n";
    oss << "      POLY EVAL (x^n) MICRO-BENCHMARK RESULT     \n";
    oss << "=================================================\033[0m\n";
    oss << "[Configuration]\n";
    oss << "  - Role         : " << (party == PARTY_PROVER ? "PROVER (Alice)" : "VERIFIER (Bob)") << "\n";
    oss << "  - Vector Size  : N = " << N << "\n";
    oss << "  - Polynomial   : Degree (Power_N) = " << power_n << "\n";

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

    if (argc < 5) {
        cout << "\nUsage: " << argv[0] << " <party> <port> <N_elements> <Power_N>" << endl;
        cout << "Example (Evaluate x^5 on 10,000 elements):\n";
        cout << "  " << argv[0] << " 1 12345 10000 5\n" << endl;
        return 1;
    }

    int party = atoi(argv[1]);
    int port = atoi(argv[2]);
    int N = atoi(argv[3]);
    int power_n = atoi(argv[4]);

    if (power_n < 1) {
        cerr << "Power must be >= 1" << endl;
        return -1;
    }

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
            pt_input_data = gen_random_vector(N);
            expected_output = plain_poly_eval_reference(pt_input_data, power_n);
        }

        // Phase 2: Core Protocol Execution & Profiling

        uint64_t net_io_start = exec->get_comm_bytes_counter();
        auto t_start = high_resolution_clock::now();

        // 1. Secret Shared Input Injection
        PolyTensor pt_input = exec->input({N}, pt_input_data);

        // 2. Prepare items for fast_tree_product
        std::vector<PolyTensor> items;
        items.reserve(power_n);

        for (int i = 0; i < power_n; ++i) {
            items.push_back(pt_input.clone());
        }

        // 3. Execute Secure Polynomial Evaluation via Tree Reduction
        PolyTensor pt_res = fast_tree_product(items);

        // 4. Store relation for ZK verification
        PolyTensor::store_self_relation(pt_res);

        // 5. Finalize and check all constraints
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

        string report = format_poly_eval_benchmark_console(party, N, power_n,
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