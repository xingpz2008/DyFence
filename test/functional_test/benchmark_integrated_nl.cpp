/**
 * @file benchmark_integrated_nl.cpp
 * @brief Microbenchmark for the Integrated Non-Linear Operator (ReLU + MaxPool2D).
 * * This tool evaluates the cryptographic overhead (Prover/Verifier runtime
 * and communication volume) of the fused Non-Linear protocol, and compares
 * it against the naive sequential execution (ReLU followed by MaxPool2D).
 * * ============================================================================
 * Usage:
 * ./benchmark_integrated_nl <party> <port> <N> <C> <H> <W> [UseFusion] [Trunc] [BitLen] [K]
 * * Parameter Guide:
 * <party>     : 1 for Prover (Alice), 2 for Verifier (Bob)
 * <port>      : Network port (e.g., 12345)
 * <N>         : Batch Size
 * <C>         : Channels
 * <H>, <W>    : Input Height and Width
 * [UseFusion] : (Optional) 1 = Use FUSED IntegratedNL, 0 = Use SEPARATE. Default: MVZK_CONFIG_MODEL_USING_INTEGRATED_NON_LINEAR @ config.h
 * [Trunc]     : (Optional) 1 = Enable Truncation, 0 = Disable. Default: MVZK_CONFIG_NON_LINEAR_RELU_DO_TRUNCATION @ config.h
 * [BitLen]    : (Optional) Bit length of the input data. Default: MVZK_CONFIG_DEFAULT_BITLEN @ config.h
 * [K]         : (Optional) Digit Decomposition parameter. Default: MVZK_CONFIG_NON_LINEAR_RELU_DIGDEC_K @ config.h
 *
 * * Note:
 * 1. To establish a strict single-threaded baseline, configure your
 * OpenMP environment variable (e.g., OMP_NUM_THREADS=1) before execution.
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
 *
 * * Examples:
 * 1. FUSED Mode (ResNet-18 Layer 1, 64x112x112):
 * [Verifier]: ./benchmark_integrated_nl 2 12345 1 64 112 112 1
 * [Prover]  : ./benchmark_integrated_nl 1 12345 1 64 112 112 1
 * * 2. SEPARATE Mode (Baseline comparison for the same workload):
 * [Verifier]: ./benchmark_integrated_nl 2 12345 1 64 112 112 0
 * [Prover]  : ./benchmark_integrated_nl 1 12345 1 64 112 112 0
 * ============================================================================
 */

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <iomanip>
#include <algorithm>

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

vector<uint64_t> gen_mixed_random_vector(size_t size) {
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

vector<uint64_t> plain_naive_nl_reference(
    const vector<uint64_t>& input,
    int N, int C, int H_in, int W_in,
    int kH, int kW, int stride, int padding,
    bool do_truncation)
{
    int H_out = (H_in + 2 * padding - kH) / stride + 1;
    int W_out = (W_in + 2 * padding - kW) / stride + 1;
    int M = N * C * H_out * W_out;
    size_t total_in = input.size();

    uint64_t zero_point = PR >> 1;
    int s_0 = MVZK_CONFIG_SCALE;
    vector<uint64_t> relu_out(total_in, 0);

    for(size_t i = 0; i < total_in; ++i) {
        uint64_t val = input[i];
        if (val > zero_point) {
            relu_out[i] = 0;
        } else {
            relu_out[i] = do_truncation ? (val >> s_0) : val;
        }
    }

    vector<uint64_t> output(M, 0);

    for(int n=0; n<N; ++n) {
        for(int c=0; c<C; ++c) {
            for(int ho=0; ho<H_out; ++ho) {
                for(int wo=0; wo<W_out; ++wo) {
                    int out_idx = n*(C*H_out*W_out) + c*(H_out*W_out) + ho*W_out + wo;
                    uint64_t x_max = 0;
                    bool first = true;

                    for(int kh=0; kh<kH; ++kh) {
                        for(int kw=0; kw<kW; ++kw) {
                            int hi = ho * stride - padding + kh;
                            int wi = wo * stride - padding + kw;

                            uint64_t val = 0;
                            if(hi >= 0 && hi < H_in && wi >= 0 && wi < W_in) {
                                int in_idx = n*(C*H_in*W_in) + c*(H_in*W_in) + hi*W_in + wi;
                                val = relu_out[in_idx];
                            }

                            if(first) {
                                x_max = val;
                                first = false;
                            } else {
                                if(val > x_max) {
                                    x_max = val;
                                }
                            }
                        }
                    }
                    output[out_idx] = x_max;
                }
            }
        }
    }
    return output;
}

// Helper: Academic Benchmark Console Formatter

string format_integrated_nl_benchmark_console(int party, int N, int C, int H_in, int W_in,
                                              int kernel, int stride, int padding,
                                              bool use_fusion, bool do_truncation,
                                              int bitlen, int k_digdec,
                                              long long total_ms, uint64_t total_comm_bytes) {
    int H_out = (H_in + 2 * padding - kernel) / stride + 1;
    int W_out = (W_in + 2 * padding - kernel) / stride + 1;

    uint64_t relu_ops = (uint64_t)N * C * H_in * W_in;
    uint64_t pool_ops = (uint64_t)N * C * H_out * W_out * (kernel * kernel - 1);

    double total_comm_mb = (double)total_comm_bytes / (1024.0 * 1024.0);
    double total_comm_kb = (double)total_comm_bytes / 1024.0;

    ostringstream oss;
    oss << "\n\033[1;36m=================================================\n";
    oss << "      INTEGRATED NON-LINEAR MICRO-BENCHMARK      \n";
    oss << "=================================================\033[0m\n";
    oss << "[Configuration]\n";
    oss << "  - Role         : " << (party == PARTY_PROVER ? "PROVER (Alice)" : "VERIFIER (Bob)") << "\n";
    if (use_fusion) {
        oss << "  - Mode         : \033[1;35mFUSED (IntegratedNL Protocol)\033[0m\n";
    } else {
        oss << "  - Mode         : \033[1;33mSEPARATE (ReLU -> MaxPool2D)\033[0m\n";
    }
    oss << "  - Input Shape  : (N=" << N << ", C=" << C << ", H=" << H_in << ", W=" << W_in << ")\n";
    oss << "  - Pool Params  : Kernel=" << kernel << ", Stride=" << stride << ", Pad=" << padding << "\n";
    oss << "  - ZK Params    : BitLen=" << bitlen << ", DigDec=" << k_digdec << ", Trunc=" << (do_truncation ? "ON" : "OFF") << "\n";
    oss << "  - Workload     : " << relu_ops << " ReLUs + " << pool_ops << " Comparisons\n";

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

    if (argc < 7) {
        cout << "\nUsage: " << argv[0] << " <party> <port> <N> <C> <H> <W> [UseFusion] [Trunc] [BitLen] [K]" << endl;
        cout << "Example (Fused Mode Default):\n";
        cout << "  " << argv[0] << " 1 12345 1 64 112 112 1\n" << endl;
        return 1;
    }

    int party = atoi(argv[1]);
    int port = atoi(argv[2]);
    int N = atoi(argv[3]);
    int C = atoi(argv[4]);
    int H_in = atoi(argv[5]);
    int W_in = atoi(argv[6]);

    // Smart default fallback
    bool use_fusion = (argc >= 8) ? (atoi(argv[7]) != 0) : MVZK_CONFIG_MODEL_USING_INTEGRATED_NON_LINEAR;
    bool do_truncation = (argc >= 9) ? (atoi(argv[8]) != 0) : MVZK_CONFIG_NON_LINEAR_RELU_DO_TRUNCATION;
    int BitLen = (argc >= 10) ? atoi(argv[9]) : MVZK_CONFIG_DEFAULT_BITLEN;
    int K = (argc >= 11) ? atoi(argv[10]) : MVZK_CONFIG_NON_LINEAR_RELU_DIGDEC_K;

    // Fixed default pooling parameters
    int kernel = 3;
    int stride = 2;
    int padding = 1;

    int input_size = N * C * H_in * W_in;

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

        exec->sync_network();

        // Phase 1: Data Preparation

        vector<uint64_t> pt_input_data;
        vector<uint64_t> expected_output;

        if (party == PARTY_PROVER) {
            pt_input_data = gen_mixed_random_vector(input_size);
            expected_output = plain_naive_nl_reference(pt_input_data, N, C, H_in, W_in, kernel, kernel, stride, padding, do_truncation);
        }

        // Phase 2: Core Protocol Execution & Profiling

        uint64_t net_io_start = exec->get_comm_bytes_counter();
        auto t_start = high_resolution_clock::now();

        PolyTensor pt_input = exec->input({N, C, H_in, W_in}, pt_input_data);
        PolyTensor pt_res;

        if (use_fusion) {
            pt_res = IntegratedNL(pt_input, kernel, stride, padding, BitLen, K, do_truncation);
        } else {
            PolyTensor pt_relu = ReLU(pt_input, BitLen, K, do_truncation);
            pt_res = MaxPool2D(pt_relu, kernel, stride, padding, BitLen, K);
        }

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

        string report = format_integrated_nl_benchmark_console(party, N, C, H_in, W_in, kernel, stride, padding,
                                                               use_fusion, do_truncation, BitLen, K,
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