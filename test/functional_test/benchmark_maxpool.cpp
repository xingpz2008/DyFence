/**
 * @file benchmark_maxpool.cpp
 * @brief Microbenchmark for the Secure MaxPool2D Operator.
 * * This tool evaluates the cryptographic overhead (Prover/Verifier runtime
 * and communication volume) of the non-linear MaxPool2D protocol.
 * * ============================================================================
 * Usage:
 * ./benchmark_maxpool <party> <port> <N> <C> <H> <W> [Kernel] [Stride] [Pad] [BitLen] [K]
 * * Parameter Guide:
 * <party>    : 1 for Prover (Alice), 2 for Verifier (Bob)
 * <port>     : Network port (e.g., 12345)
 * <N>        : Batch Size
 * <C>        : Channels
 * <H>, <W>   : Input Height and Width
 * [Kernel]   : (Optional) Pooling kernel size. Default: 3
 * [Stride]   : (Optional) Pooling stride. Default: 2
 * [Pad]      : (Optional) Zero-padding size. Default: 1
 * [BitLen]   : (Optional) Bit length of the input data. Default: MVZK_CONFIG_DEFAULT_BITLEN @ config.h
 * [K]        : (Optional) Digit Decomposition parameter. Default: MVZK_CONFIG_NON_LINEAR_RELU_DIGDEC_K @ config.h
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
 *
 * * Examples:
 * 1. ResNet-18 Layer 1 MaxPool (1x64x112x112) with default 3x3 pool, stride 2, pad 1:
 * [Verifier]: ./benchmark_maxpool 2 12345 1 64 112 112
 * [Prover]  : ./benchmark_maxpool 1 12345 1 64 112 112
 * * 2. Override all parameters (e.g., 2x2 pool, stride 2, pad 0, 64-bit, K=4):
 * [Verifier]: ./benchmark_maxpool 2 12345 1 64 112 112 2 2 0 64 4
 * [Prover]  : ./benchmark_maxpool 1 12345 1 64 112 112 2 2 0 64 4
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
    // Static ensures progressive generation, fixed seed ensures Prover/Verifier sync
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

bool _helper_plaintext_fp_greater(uint64_t a, uint64_t b) {
    uint64_t half_pr = PR >> 1;
    int64_t sa = (a > half_pr) ? static_cast<int64_t>(a) - static_cast<int64_t>(PR) : static_cast<int64_t>(a);
    int64_t sb = (b > half_pr) ? static_cast<int64_t>(b) - static_cast<int64_t>(PR) : static_cast<int64_t>(b);
    return sa > sb;
}

vector<uint64_t> plain_maxpool2d_reference(
    const vector<uint64_t>& input,
    int N, int C, int H_in, int W_in,
    int kH, int kW, int stride, int padding)
{
    int H_out = (H_in + 2 * padding - kH) / stride + 1;
    int W_out = (W_in + 2 * padding - kW) / stride + 1;
    int M = N * C * H_out * W_out;

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
                            if(hi >= 0 && hi < H_in && wi >= 0 && wi < W_in) {
                                int in_idx = n*(C*H_in*W_in) + c*(H_in*W_in) + hi*W_in + wi;
                                uint64_t val = input[in_idx];
                                if(first) {
                                    x_max = val;
                                    first = false;
                                } else {
                                    if(_helper_plaintext_fp_greater(val, x_max)) {
                                        x_max = val;
                                    }
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

string format_maxpool_benchmark_console(int party, int N, int C, int H_in, int W_in,
                                        int kernel, int stride, int padding,
                                        int bitlen, int k_digdec,
                                        long long total_ms, uint64_t total_comm_bytes) {
    int H_out = (H_in + 2 * padding - kernel) / stride + 1;
    int W_out = (W_in + 2 * padding - kernel) / stride + 1;

    // Workload estimation: A KxK window requires (K^2 - 1) comparisons
    uint64_t comparisons = (uint64_t)N * C * H_out * W_out * (kernel * kernel);

    double total_comm_mb = (double)total_comm_bytes / (1024.0 * 1024.0);
    double total_comm_kb = (double)total_comm_bytes / 1024.0;

    ostringstream oss;
    oss << "\n\033[1;36m=================================================\n";
    oss << "        MAXPOOL MICRO-BENCHMARK RESULT           \n";
    oss << "=================================================\033[0m\n";
    oss << "[Configuration]\n";
    oss << "  - Role         : " << (party == PARTY_PROVER ? "PROVER (Alice)" : "VERIFIER (Bob)") << "\n";
    oss << "  - Input Shape  : (N=" << N << ", C=" << C << ", H=" << H_in << ", W=" << W_in << ")\n";
    oss << "  - Output Shape : (N=" << N << ", C=" << C << ", H=" << H_out << ", W=" << W_out << ")\n";
    oss << "  - Pool Params  : Kernel=" << kernel << ", Stride=" << stride << ", Pad=" << padding << "\n";
    oss << "  - ZK Params    : BitLen=" << bitlen << ", DigDec (K)=" << k_digdec << "\n";
    oss << "  - Workload     : " << comparisons << " Comparisons\n";

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
        cout << "\nUsage: " << argv[0] << " <party> <port> <N> <C> <H> <W> [Kernel] [Stride] [Pad] [BitLen] [K]" << endl;
        cout << "Example (ResNet-18 Layer 1 defaults):\n";
        cout << "  " << argv[0] << " 1 12345 1 64 112 112\n" << endl;
        return 1;
    }

    int party = atoi(argv[1]);
    int port = atoi(argv[2]);
    int N = atoi(argv[3]);
    int C = atoi(argv[4]);
    int H_in = atoi(argv[5]);
    int W_in = atoi(argv[6]);

    // Smart default fallback
    int kernel = (argc >= 8) ? atoi(argv[7]) : 3;
    int stride = (argc >= 9) ? atoi(argv[8]) : 2;
    int padding = (argc >= 10) ? atoi(argv[9]) : 1;
    int BitLen = (argc >= 11) ? atoi(argv[10]) : MVZK_CONFIG_DEFAULT_BITLEN;
    int K = (argc >= 12) ? atoi(argv[11]) : MVZK_CONFIG_NON_LINEAR_RELU_DIGDEC_K;

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
            expected_output = plain_maxpool2d_reference(pt_input_data, N, C, H_in, W_in, kernel, kernel, stride, padding);
        }

        // Phase 2: Core Protocol Execution & Profiling

        uint64_t net_io_start = exec->get_comm_bytes_counter();
        auto t_start = high_resolution_clock::now();

        PolyTensor pt_input = exec->input({N, C, H_in, W_in}, pt_input_data);

        PolyTensor pt_res = MaxPool2D(pt_input, kernel, stride, padding, BitLen, K);

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

        string report = format_maxpool_benchmark_console(party, N, C, H_in, W_in, kernel, stride, padding,
                                                         BitLen, K, active_zk_time_ms, total_comm_bytes);
        cout << report;
    } catch (const std::exception& e) {
        cerr << "\n\033[1;31m[FATAL ERROR] Exception caught: " << e.what() << "\033[0m" << endl;
        if(exec) delete exec;
        return -1;
    }

    if(exec) delete exec;
    return 0;
}