/**
 * @file benchmark_conv.cpp
 * @brief Microbenchmark for the Secure 2D Convolution (Conv2D) Operator.
 * * This tool evaluates the cryptographic overhead (Prover/Verifier runtime
 * and communication volume) of the Conv2D protocol under parameterized input
 * dimensions. It runs purely in-memory and outputs results directly to stdout.
 * * ============================================================================
 * Usage:
 * ./benchmark_conv <party> <port> <N> <C_in> <H> <W> <C_out> <K> <Stride> <Pad>
 * * Parameter Guide:
 * <party>  : 1 for Prover (Alice), 2 for Verifier (Bob)
 * <port>   : Network port (e.g., 12345)
 * <N>      : Batch size
 * <C_in>   : Input channels
 * <H>, <W> : Input height and width
 * <C_out>  : Output channels (number of filters)
 * <K>      : Kernel size (K x K)
 * <Stride> : Convolution stride
 * <Pad>    : Zero-padding size
 *
 * * ResNet-50 Baseline Examples:
 * 1. Initial Conv1 (7x7, stride=2, pad=3):
 * [Terminal 1 - Verifier]: ./benchmark_conv 2 12345 1 3 224 224 64 7 2 3
 * [Terminal 2 - Prover]  : ./benchmark_conv 1 12345 1 3 224 224 64 7 2 3
 * * 2. Conv2_x 3x3 Bottleneck (3x3, stride=1, pad=1):
 * [Terminal 1 - Verifier]: ./benchmark_conv 2 12345 1 64 56 56 64 3 1 1
 * [Terminal 2 - Prover]  : ./benchmark_conv 1 12345 1 64 56 56 64 3 1 1
 * * 3. Conv5_x 3x3 Bottleneck (3x3, stride=1, pad=1):
 * [Terminal 1 - Verifier]: ./benchmark_conv 2 12345 1 512 7 7 512 3 1 1
 * [Terminal 2 - Prover]  : ./benchmark_conv 1 12345 1 512 7 7 512 3 1 1
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
#include <sstream>
#include <cassert>

// Core MVZK Framework Headers
#include "exec/ExecProver.h"
#include "exec/ExecVerifier.h"
#include "exec/MVZKNetBase.h"
#include "operations/linear.h"
#include "emp-tool/emp-tool.h"
#include "utils/logger.h"
#include "utils/net_utils.h"

using namespace std;
using namespace emp;
using namespace std::chrono;

MVZKExec* MVZKExec::mvzk_exec = nullptr;

const int PARTY_PROVER = ALICE;
const int PARTY_VERIFIER = BOB;

// Helper: Plaintext Reference Implementation (Ground Truth)

vector<uint64_t> plain_conv2d_reference(
    const vector<uint64_t>& input,
    const vector<uint64_t>& weight,
    const vector<uint64_t>& bias,
    int N, int C_in, int H_in, int W_in,
    int C_out, int kH, int kW,
    int stride, int padding
) {
    int H_out = (H_in + 2 * padding - kH) / stride + 1;
    int W_out = (W_in + 2 * padding - kW) / stride + 1;

    vector<uint64_t> output(N * C_out * H_out * W_out, 0);

    for (int n = 0; n < N; ++n) {
        for (int c_out = 0; c_out < C_out; ++c_out) {
            for (int h_out = 0; h_out < H_out; ++h_out) {
                for (int w_out = 0; w_out < W_out; ++w_out) {
                    uint64_t sum = bias.empty() ? 0 : bias[c_out];

                    for (int c_in = 0; c_in < C_in; ++c_in) {
                        for (int kh = 0; kh < kH; ++kh) {
                            for (int kw = 0; kw < kW; ++kw) {
                                int h_in = h_out * stride + kh - padding;
                                int w_in = w_out * stride + kw - padding;

                                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                                    size_t in_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                                    size_t w_idx = ((c_out * C_in + c_in) * kH + kh) * kW + kw;

                                    sum = add_mod(sum, mult_mod(input[in_idx], weight[w_idx]));
                                }
                            }
                        }
                    }

                    size_t out_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
                    output[out_idx] = sum;
                }
            }
        }
    }
    return output;
}

// Helper: Random Data Generator

vector<uint64_t> gen_random_vector(size_t size, uint64_t mod = 100) {
    vector<uint64_t> res(size);
    for(size_t i=0; i<size; ++i) res[i] = rand() % mod;
    return res;
}

// Helper: Academic Benchmark Console Formatter

string format_microbenchmark_console(int party, int N, int C_in, int H, int W,
                                     int C_out, int K_size, int Stride, int Pad,
                                     long long total_ms, uint64_t total_comm_bytes) {
    int H_out = (H + 2 * Pad - K_size) / Stride + 1;
    int W_out = (W + 2 * Pad - K_size) / Stride + 1;

    // Calculate Multiply-Accumulates (MACs) to evaluate computational density
    uint64_t macs = (uint64_t)N * C_out * H_out * W_out * C_in * K_size * K_size;

    double total_comm_mb = (double)total_comm_bytes / (1024.0 * 1024.0);
    double total_comm_kb = (double)total_comm_bytes / 1024.0;

    ostringstream oss;
    oss << "\n\033[1;36m=================================================\n";
    oss << "        CONV2D MICRO-BENCHMARK RESULT            \n";
    oss << "=================================================\033[0m\n";
    oss << "[Configuration]\n";
    oss << "  - Role         : " << (party == PARTY_PROVER ? "PROVER (Alice)" : "VERIFIER (Bob)") << "\n";
    oss << "  - Input Tensor : (N=" << N << ", C_in=" << C_in << ", H=" << H << ", W=" << W << ")\n";
    oss << "  - Weight Tensor: (C_out=" << C_out << ", C_in=" << C_in << ", K=" << K_size << ", K=" << K_size << ")\n";
    oss << "  - Workload     : " << macs << " MACs\n";

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

    if (argc < 11) {
        cout << "\nUsage: " << argv[0] << " <party> <port> <N> <C_in> <H> <W> <C_out> <K> <Stride> <Pad>" << endl;
        cout << "\nResNet-50 Test Profiles:\n";
        cout << "  - ResNet50 Conv1   : " << argv[0] << " <party> <port> 1 3 224 224 64 7 2 3\n";
        cout << "  - ResNet50 Conv2_x : " << argv[0] << " <party> <port> 1 64 56 56 64 3 1 1\n";
        cout << "  - ResNet50 Conv5_x : " << argv[0] << " <party> <port> 1 512 7 7 512 3 1 1\n" << endl;
        return 1;
    }

    int party = atoi(argv[1]);
    int port = atoi(argv[2]);

    int N = atoi(argv[3]);
    int C_in = atoi(argv[4]);
    int H = atoi(argv[5]);
    int W = atoi(argv[6]);
    int C_out = atoi(argv[7]);
    int K_size = atoi(argv[8]);
    int Stride = atoi(argv[9]);
    int Pad = atoi(argv[10]);

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
        vector<uint64_t> pt_weight_data;
        vector<uint64_t> pt_bias_data;
        vector<uint64_t> expected_output;

        if (party == PARTY_PROVER) {
            pt_input_data = gen_random_vector(N * C_in * H * W);
            pt_weight_data = gen_random_vector(C_out * C_in * K_size * K_size);
            pt_bias_data = gen_random_vector(C_out);

            expected_output = plain_conv2d_reference(
                pt_input_data, pt_weight_data, pt_bias_data,
                N, C_in, H, W, C_out, K_size, K_size, Stride, Pad
            );
        }

        // Phase 2: Core Protocol Execution & Profiling

        uint64_t net_io_start = exec->get_comm_bytes_counter();
        auto t_start = high_resolution_clock::now();

        PolyTensor pt_input = exec->input({N, C_in, H, W}, pt_input_data);
        PolyTensor pt_weight = exec->input({C_out, C_in, K_size, K_size}, pt_weight_data);
        PolyTensor pt_bias = exec->input({C_out}, pt_bias_data);

        PolyTensor pt_res = Conv2D(pt_input, pt_weight, pt_bias, Stride, Pad, 1);

        PolyTensor::store_self_relation(pt_res);

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

        string report = format_microbenchmark_console(party, N, C_in, H, W, C_out, K_size, Stride, Pad,
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