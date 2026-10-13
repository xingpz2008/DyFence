/*
 * Usage:
 * * Mode 1: ./<binary_name> <party> <port> <model_dir> 1 [target_party] [save_state]
 * * Mode 2: ./<binary_name> <party> <port> <model_dir> 2 [total_parties]
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <ctime>
#include <omp.h>
#include <cstdlib>
#include <limits>
#include <filesystem>

#include "exec/ExecProver.h"
#include "exec/ExecVerifier.h"
#include "exec/MVZKNetBase.h"
#include "exec/ExecPlaintext.h"
#include "utils/logger.h"
#include "utils/net_utils.h"
#include "emp-tool/emp-tool.h"
#include "data_type/tensor_loader.h"

// Consistency-check helper utilities.
#include "utils/consistency_utils.h"

// Generated model header.
#include "{model_name}.h"

using namespace std;
using namespace emp;
using namespace std::chrono;

MVZKExec* MVZKExec::mvzk_exec = nullptr;
const int PARTY_PROVER = ALICE;
const int PARTY_VERIFIER = BOB;

enum TestMode {{
    MODE_INFERENCE = 1,
    MODE_CONSISTENCY = 2,
    MODE_FAST_DUMP = 3,
    MODE_PLAINTEXT = 4
}};

// ==========================================
// Timing helper.
// ==========================================
template <typename Func>
long long measure_time(Func&& func, string step_name) {{
    auto start = high_resolution_clock::now();
    func();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "[TIME] " << left << setw(45) << step_name << ": " << duration.count() << " ms" << endl;
    return duration.count();
}}

// ==========================================
// Remove ANSI color escape sequences before writing reports.
// ==========================================
string strip_ansi(const string& input) {{
    string output;
    bool in_escape = false;
    for (char c : input) {{
        if (c == '\033') {{
            in_escape = true;
        }} else if (in_escape) {{
            if (c == 'm') in_escape = false;
        }} else {{
            output += c;
        }}
    }}
    return output;
}}

// ==========================================
// Format the in-memory report for inference mode.
// ==========================================
string format_base_report(const string& model_name, int party, int N, int C, int H, int W,
                       uint64_t bitlen, uint64_t digdec_k, bool do_truncation,
                       int num_threads, int omp_threads, uint64_t net_io_counter_start, uint64_t net_io_counter_end,
                       const string& start_time_str, const string& stop_time_str, long long total_ms, long long prep_time) {{

    long long h  = total_ms / 3600000;
    long long m  = (total_ms % 3600000) / 60000;
    long long s  = (total_ms % 60000) / 1000;
    long long ms = total_ms % 1000;

    uint64_t total_comm_bytes = net_io_counter_end - net_io_counter_start;
    double total_comm_mb = (double)total_comm_bytes / (1024.0 * 1024.0);
    double total_comm_gb = total_comm_mb / 1024.0;

    ostringstream oss;
    oss << "\n=================================================\n";
    oss << "                 FINAL TEST REPORT               \n";
    oss << "=================================================\n";
    oss << "[Model & Task Info]\n";
    oss << "  - Model Name   : " << model_name << "\n";
    oss << "  - Input Shape  : (N=" << N << ", C=" << C << ", H=" << H << ", W=" << W << ")\n";
    oss << "  - Role         : " << (party == PARTY_PROVER ? "PROVER (Alice)" : "VERIFIER (Bob)") << "\n";
    oss << "  - Source       : Frontend Generated \n";

    oss << "\n[Test Runtime Parameters]\n";
    oss << "  - Run: Bit Length   : " << bitlen << "\n";
    oss << "  - Run: DigDec_k     : " << digdec_k << "\n";
    oss << "  - Run: Truncation   : " << (do_truncation ? "Enabled" : "Disabled") << "\n";
    oss << "  - Run: Threads (IO) : " << num_threads << "\n";
    oss << "  - Run: Threads (OMP): " << omp_threads << "\n";

    oss << "\n[Compiled System Configs (config.h)]\n";
    oss << "  - MVZK_CONFIG_THREADS_NUM      : " << MVZK_CONFIG_THREADS_NUM << "\n";
    oss << "  - MVZK_CONFIG_SCALE            : " << MVZK_CONFIG_SCALE << "\n";
    oss << "  - MVZK_CONFIG_DEFAULT_BITLEN   : " << MVZK_CONFIG_DEFAULT_BITLEN << "\n";
    oss << "  - MVZK_CONFIG_RELU_DIGDEC_K    : " << MVZK_CONFIG_NON_LINEAR_RELU_DIGDEC_K << "\n";
    oss << "  - MVZK_OMP_SIZE_THRESHOLD      : " << MVZK_OMP_SIZE_THRESHOLD << "\n";
    oss << "  - MVZK_MULT_CHECK_CNT          : " << MVZK_MULT_CHECK_CNT << "\n";
    oss << "  - MVZK_CONFIG_TENSOR_CHK_CNT   : " << MVZK_CONFIG_TENSOR_CHECK_CNT << "\n";
    oss << "  - MVZK_CONFIG_RANGE_CHK_BUF    : " << MVZK_CONFIG_RANGE_CHECK_REQUEST_BUFFER_THRESHOLD << "\n";
    oss << "  - MVZK_CONFIG_MULT_PRODUCT_THR : " << MVZK_CONFIG_MULT_PRODUCT_THRESHOLD << "\n";
    oss << "  - MVZK_CONFIG_NN_PUBLIC_INPUT  : " << (MVZK_CONFIG_NN_PUBLIC_INPUT ? "Yes" : "No") << "\n";
    oss << "  - MVZK_CONFIG_NN_PUBLIC_WEIGHT : " << (MVZK_CONFIG_NN_PUBLIC_WEIGHT ? "Yes" : "No") << "\n";

    oss << "\n[Execution Time]\n";
    oss << "  - Start Time   : " << start_time_str << "\n";
    oss << "  - End Time     : " << stop_time_str << "\n";

    long long p_h  = prep_time / 3600000;
    long long p_m  = (prep_time % 3600000) / 60000;
    long long p_s  = (prep_time % 60000) / 1000;
    long long p_ms = prep_time % 1000;
    oss << "  - Prep Time    : ";
    if (p_h > 0) oss << p_h << "h ";
    if (p_m > 0 || p_h > 0) oss << p_m << "m ";
    oss << p_s << "s " << p_ms << "ms  (" << prep_time << " ms)\n";

    oss << "  - ZK Execute Time  : ";
    if (h > 0) oss << h << "h ";
    if (m > 0 || h > 0) oss << m << "m ";
    oss << s << "s " << ms << "ms  (" << total_ms << " ms)\n";

    oss << "\n[Communication]\n";
    oss << "  - Data Sent    : " << fixed << setprecision(2) << total_comm_mb << " MB";
    if (total_comm_gb > 1.0) oss << " (" << total_comm_gb << " GB)";
    oss << "\n=================================================\n";

    return oss.str();
}}

string format_consistency_report(const string& model_name, int party, int total_parties, size_t total_params,
                                 const string& start_time_str, const string& stop_time_str,
                                 long long commit_ms, long long check_ms,
                                 uint64_t commit_comm, uint64_t check_comm) {{

    long long total_ms = commit_ms + check_ms;
    uint64_t total_comm = commit_comm + check_comm;

    double commit_mb = (double)commit_comm / (1024.0 * 1024.0);
    double check_mb = (double)check_comm / (1024.0 * 1024.0);
    double total_mb = (double)total_comm / (1024.0 * 1024.0);

    string role_str = (party == PARTY_PROVER) ? "PROVER (Alice)" : ("VERIFIER " + to_string(party - 1) + " (Bob)");

    ostringstream oss;
    oss << "\n\033[1;36m=================================================\n";
    oss << "          CONSISTENCY CHECK REPORT               \n";
    oss << "=================================================\033[0m\n";
    oss << "[Task Info]\n";
    oss << "  - Model Name   : " << model_name << "\n";
    oss << "  - Role         : " << role_str << "\n";
    oss << "  - Total Parties: " << total_parties << " (1 Prover, " << (total_parties - 1) << " Verifiers)\n";
    oss << "  - Params Chk'd : " << total_params << " weights/biases\n";
    oss << "  - Ideal Chnl   : " << (MVZK_CONFIG_BROADCAST_ASSUME_IDEAL ? "Yes" : "No") << "\n";

    oss << "\n[Execution Time]\n";
    oss << "  - Start Time   : " << start_time_str << "\n";
    oss << "  - End Time     : " << stop_time_str << "\n";
    oss << "  - Phase 1 (Commit Time) : " << commit_ms << " ms\n";
    oss << "  - Phase 2 (Check Time)  : " << check_ms << " ms\n";
    oss << "  - [1;32mTotal Time (P1+P2)      : " << total_ms << " ms[0m\n";

    oss << "\n[Communication Overhead]\n";
    oss << "  - Phase 1 (Commit Comm) : " << fixed << setprecision(2) << commit_mb << " MB\n";
    oss << "  - Phase 2 (Check Comm)  : " << fixed << setprecision(2) << check_mb << " MB\n";
    oss << "  - [1;32mTotal Comm (P1+P2)      : " << fixed << setprecision(2) << total_mb << " MB[0m\n";
    oss << "\033[1;36m=================================================\033[0m\n";

    return oss.str();
}}

string format_plaintext_report(const string& model_name, int N, int C, int H, int W,
                               uint64_t bitlen, uint64_t digdec_k, bool do_truncation,
                               const string& start_time_str, const string& stop_time_str,
                               long long total_ms, int accuracy_count, bool has_labels) {{

    long long m  = (total_ms % 3600000) / 60000;
    long long s  = (total_ms % 60000) / 1000;
    long long ms = total_ms % 1000;

    ostringstream oss;
    oss << "\n\033[1;36m=================================================\n";
    oss << "      PLAINTEXT SIMULATION FINAL REPORT          \n";
    oss << "=================================================\033[0m\n";
    oss << "[Task Info]\n";
    oss << "  - Model Name   : " << model_name << "\n";
    oss << "  - Total Images : " << N << " (Shape: " << C << "x" << H << "x" << W << ")\n";

    oss << "\n[Math Parameters]\n";
    oss << "  - Bit Length   : " << bitlen << " bits\n";
    oss << "  - Scale        : " << MVZK_CONFIG_SCALE << "\n";
    oss << "  - Truncation   : " << (do_truncation ? "Enabled" : "Disabled") << "\n";

    oss << "\n[Execution Time]\n";
    oss << "  - Start Time   : " << start_time_str << "\n";
    oss << "  - End Time     : " << stop_time_str << "\n";
    oss << "  - Pure F_p Math: ";
    if (m > 0) oss << m << "m ";
    oss << s << "s " << ms << "ms  (" << total_ms << " ms)\n";

    if (has_labels) {{
        double accuracy = (double)accuracy_count / N * 100.0;
        oss << "\n\033[1;35m[====> ACCURACY RESULTS <====]\033[0m\n";
        oss << "  - \033[1;32mModel Accuracy : " << fixed << setprecision(2) << accuracy << "% (" << accuracy_count << "/" << N << ")\033[0m\n";
    }} else {{
        oss << "\n\033[1;33m[====> ACCURACY RESULTS <====]\033[0m\n";
        oss << "  - Ground truth labels not found. Accuracy not computed.\n";
    }}
    oss << "\033[1;36m=================================================\033[0m\n\n";

    return oss.str();
}}

// ==========================================
// Entry point.
// ==========================================
int main(int argc, char** argv) {{
    SetLogLevel(LEVEL_ERROR);

    // Command-line usage.
    if (argc < 4) {{
        cout << "\nUsage: " << endl;
        cout << "  Mode 1 (Inference): " << argv[0] << " <party> <port> <model_dir> 1 [target_party_id] [save_state]" << endl;
        cout << "  Mode 2 (Check)    : " << argv[0] << " <party> <port> <model_dir> 2 [total_parties]" << endl;
        return 1;
    }}
    int party = atoi(argv[1]);
    int port = atoi(argv[2]);
    std::string model_dir = argv[3];

    // Parse execution mode and optional context parameters.
    TestMode mode = (argc >= 5) ? static_cast<TestMode>(atoi(argv[4])) : MODE_INFERENCE;
    bool save_for_audit = false;

    int target_party = -1;
    int cli_total_parties = MVZK_CONFIG_NUM_PARTIES;

    if (mode == MODE_INFERENCE || mode == MODE_FAST_DUMP || mode == MODE_PLAINTEXT) {{
        target_party = (argc >= 6) ? atoi(argv[5]) : -1;
        save_for_audit = (argc >= 7) ? (atoi(argv[6]) == 1) : false;
    }} else if (mode == MODE_CONSISTENCY) {{
        cli_total_parties = (argc >= 6) ? atoi(argv[5]) : MVZK_CONFIG_NUM_PARTIES;
        if (cli_total_parties < 3) {{
            cerr << "Consistency mode requires at least 3 total parties (1 prover and 2 verifiers)." << endl;
            return 1;
        }}
    }}

    std::string bin_dir = model_dir + "/weights";
    std::string input_bin_path = model_dir + "/test_cases/case_1/input_0.bin";
    std::string log_dir = model_dir + "/log/";

    int actual_target_party = (target_party != -1) ? target_party : 2;
    int verifier_idx = (party == PARTY_PROVER) ? (actual_target_party - 1) : (party - 1);

    std::vector<std::string> net_ips;
    const char* prover_ip_env = std::getenv("DYFENCE_PROVER_IP");
    if (prover_ip_env != nullptr && prover_ip_env[0] != '\0') {{
        net_ips.emplace_back(prover_ip_env);
    }}

    std::string checkpoint_dir = model_dir + "/checkpoints";
    std::string sub_dir = checkpoint_dir + (party == PARTY_PROVER ?
                          "/Prover_with_Verifier_" + to_string(verifier_idx) :
                          "/Verifier_" + to_string(verifier_idx) + "_with_Prover");

    std::filesystem::create_directories(sub_dir);

    std::string delta_path = sub_dir + "/verifier_delta.bin";
    std::string snapshot_path = sub_dir + "/commit_snapshot.bin";

    const int N = {N};
    const int C = {C};
    const int H = {H};
    const int W = {W};

    uint64_t bitlen = MVZK_CONFIG_DEFAULT_BITLEN;
    uint64_t digdec_k = MVZK_CONFIG_NON_LINEAR_RELU_DIGDEC_K;
    bool do_truncation = MVZK_CONFIG_NON_LINEAR_RELU_DO_TRUNCATION;

    cout << "=================================================" << endl;
    cout << "{model_name} Full ZK Graph Test" << endl;
    cout << "Role: " << (party == PARTY_PROVER ? "PROVER" : "VERIFIER") << endl;
    if (mode == MODE_CONSISTENCY) cout << "Mode: Phase 2 (Consistency Check Only | Total Parties: " << cli_total_parties << ")" << endl;
    else if (mode == MODE_PLAINTEXT) cout << "Mode: Plaintext Accuracy Simulation (No ZK Overhead)" << endl;
    else cout << "Mode: Phase 1 (Target Verifier Party ID: " << ((target_party == -1) ? 2 : target_party) << " | Verifier " << verifier_idx << ")" << endl;
    cout << "=================================================" << endl;

    if (checkSysSettings(mode) == false) {{
        LOG_ERROR("System settings check failed.");
        exit(-1);
    }}

    int num_threads = MVZK_CONFIG_THREADS_NUM;
    int omp_threads = omp_get_max_threads();

    auto prep_start = high_resolution_clock::now();

    MVZKNetBase<NetIO> *exec = nullptr;
    if (mode == MODE_PLAINTEXT) {{
        exec = new MVZKExecPlaintext<NetIO>(party);
    }} else if (party == PARTY_PROVER) {{
        exec = new MVZKExecProver<NetIO>(party, port, net_ips, target_party, cli_total_parties);
    }} else {{
        if (mode == MODE_CONSISTENCY) {{
            exec = new MVZKExecVerifier<NetIO>(party, port, net_ips, delta_path, cli_total_parties);
        }} else {{
            exec = new MVZKExecVerifier<NetIO>(party, port, net_ips, "", cli_total_parties);
        }}
    }}

    auto prep_end = high_resolution_clock::now();
    long long prep_time = duration_cast<milliseconds>(prep_end - prep_start).count();

    try {{
        time_t start_time_t = system_clock::to_time_t(system_clock::now());
        string start_time_str = ctime(&start_time_t);
        start_time_str.pop_back();

        uint64_t net_io_start = exec->get_comm_bytes_counter();

        if (mode == MODE_PLAINTEXT) {{
            cout << "\n[TEST]  Phase 1: Loading Weights and Full Input Batch..." << endl;
            {model_name}HostWeights host_weights = load_{model_name_lower}_host_weights(party, bin_dir);
            std::vector<uint64_t> host_input_data = load_raw_data_from_bin(party, {{N, C, H, W}}, input_bin_path, TensorDataType::FP32, false, MVZK_CONFIG_NN_PUBLIC_INPUT);

            cout << "[TEST]  Phase 2: Mock Commit weights." << endl;
            {model_name}PolyWeights poly_weights = inject_{model_name_lower}_weights_to_zk(exec, host_weights);

            int CHUNK_SIZE = MVZK_CONFIG_NN_MAX_BATCHSIZE;
            int num_chunks = (N + CHUNK_SIZE - 1) / CHUNK_SIZE;

            std::vector<float> expected_logits;
            std::vector<int32_t> true_labels;
            size_t last_slash = input_bin_path.find_last_of('/');
            std::string expected_bin_path = (last_slash == std::string::npos) ? "expected_logits.bin" : input_bin_path.substr(0, last_slash) + "/expected_logits.bin";
            std::string labels_bin_path = (last_slash == std::string::npos) ? "labels_0.bin" : input_bin_path.substr(0, last_slash) + "/labels_0.bin";

            std::ifstream ifs(expected_bin_path, std::ios::binary);
            if(ifs.is_open()) {{
                ifs.seekg(0, std::ios::end); size_t size = ifs.tellg(); ifs.seekg(0, std::ios::beg);
                expected_logits.resize(size / sizeof(float)); ifs.read(reinterpret_cast<char*>(expected_logits.data()), size); ifs.close();
            }}

            std::ifstream ifs_labels(labels_bin_path, std::ios::binary);
            if(ifs_labels.is_open()) {{
                ifs_labels.seekg(0, std::ios::end); size_t size = ifs_labels.tellg(); ifs_labels.seekg(0, std::ios::beg);
                true_labels.resize(size / sizeof(int32_t)); ifs_labels.read(reinterpret_cast<char*>(true_labels.data()), size); ifs_labels.close();
            }}

            int accuracy_count = 0;
            size_t num_classes = 0;

            cout << "[TEST]  Phase 3: Executing Plaintext F_p Math in " << num_chunks << " Chunks..." << endl;
            long long active_zk_time_ms = 0;

            for (int chunk = 0; chunk < num_chunks; ++chunk) {{
                int current_batch = (chunk == num_chunks - 1 && N % CHUNK_SIZE != 0) ? (N % CHUNK_SIZE) : CHUNK_SIZE;
                cout << "\033[1;36m  -> Processing Chunk " << chunk + 1 << "/" << num_chunks << " (Batch Size: " << current_batch << ")...[0m " << flush;

                size_t offset = (size_t)chunk * (size_t)CHUNK_SIZE * (size_t)C * (size_t)H * (size_t)W;
                size_t length = (size_t)current_batch * (size_t)C * (size_t)H * (size_t)W;
                std::vector<uint64_t> chunk_data(host_input_data.begin() + offset, host_input_data.begin() + offset + length);

                auto t3 = high_resolution_clock::now();
                PolyTensor image = MVZK_CONFIG_NN_PUBLIC_INPUT
                                   ? PolyTensor::from_public({{current_batch, C, H, W}}, chunk_data)
                                   : exec->input({{current_batch, C, H, W}}, chunk_data);

                PolyTensor output = {model_name}_Forward(image, poly_weights, bitlen, digdec_k, do_truncation);
                auto t6 = high_resolution_clock::now();
                active_zk_time_ms += duration_cast<milliseconds>(t6 - t3).count();

                std::vector<uint64_t> raw_logits = exec->reveal(output);
                output.mark_consumed();

                if (!expected_logits.empty() && !raw_logits.empty()) {{
                    num_classes = raw_logits.size() / current_batch;
                    for (int b = 0; b < current_batch; ++b) {{
                        size_t global_b = (size_t)chunk * (size_t)CHUNK_SIZE + (size_t)b;
                        int zk_argmax = 0; int pt_argmax = 0;
                        double zk_max = std::numeric_limits<double>::lowest();
                        float pt_max = std::numeric_limits<float>::lowest();
                        int current_true_label = (!true_labels.empty() && global_b < true_labels.size()) ? true_labels[global_b] : -1;

                        for(size_t c = 0; c < num_classes; ++c) {{
                            size_t i_zk = (size_t)b * (size_t)num_classes + (size_t)c;
                            size_t i_pt = (size_t)global_b * (size_t)num_classes + (size_t)c;
                            uint64_t field_val = raw_logits[i_zk];
                            int64_t signed_val = (field_val > PR / 2) ? static_cast<int64_t>(field_val) - PR : field_val;
                            double pt_val = expected_logits[i_pt];

                            double total_scale = 1.0;
                            for(int s = 0; s < {final_scale_power}; ++s) total_scale *= (1ULL << MVZK_CONFIG_SCALE);
                            double zk_val_scaled = (double)signed_val / total_scale / (double)({pool_area});

                            if (zk_val_scaled > zk_max) {{ zk_max = zk_val_scaled; zk_argmax = c; }}
                            if (pt_val > pt_max) {{ pt_max = pt_val; pt_argmax = c; }}
                        }}
                        if (current_true_label != -1 && zk_argmax == current_true_label) accuracy_count++;
                    }}
                }}
                cout << "\033[1;32m (Running Acc: " << accuracy_count << "/" << (chunk * CHUNK_SIZE + current_batch) << ")[0m" << endl;
            }}
            exec->finalize_protocol();

            time_t stop_time_t = system_clock::to_time_t(system_clock::now());
            string stop_time_str = ctime(&stop_time_t); stop_time_str.pop_back();
            uint64_t net_io_end = exec->get_comm_bytes_counter();

            std::vector<PolyTensor*> current_tensors = get_all_weight_tensors(poly_weights);
            for (auto pt : current_tensors) pt->mark_consumed();

            string pt_report_str = format_plaintext_report("{model_name}", N, C, H, W, bitlen, digdec_k, do_truncation,
                                                           start_time_str, stop_time_str, active_zk_time_ms,
                                                           accuracy_count, !true_labels.empty());
            cout << pt_report_str;

            if (!true_labels.empty()) {{
                double accuracy = (double)accuracy_count / N * 100.0;
                cout << "\n\033[1;35m[====> GLOBAL ALIGNMENT & ACCURACY <====]\033[0m\n";
                cout << "  - \033[1;32mZK Model Accuracy    : " << fixed << setprecision(2) << accuracy << "% (" << accuracy_count << "/" << N << ")\033[0m\n";
                cout << "=================================================\n\n";
            }}
        }}

        // =========================================================================
        // MODE 1 & 4 (Inference / Plaintext / Fast Dump)
        // =========================================================================
        else if (mode == MODE_INFERENCE || mode == MODE_FAST_DUMP) {{
            bool skip_inference = (mode == MODE_FAST_DUMP);

            cout << "\n[TEST]  Phase 1: Loading Weights and Inputs from Disk ..." << endl;
            {model_name}HostWeights host_weights = load_{model_name_lower}_host_weights(party, bin_dir);
            std::vector<uint64_t> host_input_data;
            if (!skip_inference) {{
                host_input_data = load_raw_data_from_bin(party, {{{{N, C, H, W}}}}, input_bin_path, TensorDataType::FP32, false, MVZK_CONFIG_NN_PUBLIC_INPUT);
            }}

            long long active_zk_time_ms = 0;

            cout << "[TEST]  Phase 2: Commit weights and inputs with ZK backend." << endl;

            auto t1 = high_resolution_clock::now();
            {model_name}PolyWeights poly_weights = inject_{model_name_lower}_weights_to_zk(exec, host_weights);
            auto t2 = high_resolution_clock::now();
            if (!MVZK_CONFIG_NN_PUBLIC_WEIGHT) {{
                active_zk_time_ms += duration_cast<milliseconds>(t2 - t1).count();
            }}

            uint64_t net_io_commit_end = exec->get_comm_bytes_counter();
            long long commit_time_ms = active_zk_time_ms;
            uint64_t commit_comm_bytes = net_io_commit_end - net_io_start;

            PolyTensor output;
            if (!skip_inference) {{
                auto t3 = high_resolution_clock::now();
                PolyTensor image = MVZK_CONFIG_NN_PUBLIC_INPUT
                                   ? PolyTensor::from_public({{{{N, C, H, W}}}}, host_input_data)
                                   : exec->input({{{{N, C, H, W}}}}, host_input_data);
                auto t4 = high_resolution_clock::now();
                if (!MVZK_CONFIG_NN_PUBLIC_INPUT) {{
                    active_zk_time_ms += duration_cast<milliseconds>(t4 - t3).count();
                }}

                cout << "[TEST]  Phase 3: Executing {model_name} ZK Forward Pass..." << endl;

                auto t5 = high_resolution_clock::now();
                output = {model_name}_Forward(image, poly_weights, bitlen, digdec_k, do_truncation);
                PolyTensor::store_self_relation(output, "{model_name}_Final_Logits_Check");
                exec->finalize_protocol();

                auto t6 = high_resolution_clock::now();
                active_zk_time_ms += duration_cast<milliseconds>(t6 - t5).count();
            }}

            time_t stop_time_t = system_clock::to_time_t(system_clock::now());
            string stop_time_str = ctime(&stop_time_t);
            stop_time_str.pop_back();

            uint64_t net_io_end = exec->get_comm_bytes_counter();
            auto total_ms = active_zk_time_ms;

            std::vector<PolyTensor*> current_tensors;

            if (save_for_audit && mode != MODE_PLAINTEXT){{
                cout << "\n[TEST]  Dumping internal ZK states to isolated directories..." << endl;
                current_tensors = get_all_weight_tensors(poly_weights);

                save_tensors_1to1(current_tensors, party, verifier_idx, checkpoint_dir);

                if (party != PARTY_PROVER) {{
                    auto verifier = static_cast<MVZKExecVerifier<NetIO>*>(exec);
                    verifier->save_delta_to_file(delta_path);
                }}

                ofstream snapshot_out(snapshot_path, ios::binary);
                if(snapshot_out.is_open()) {{
                    snapshot_out.write(reinterpret_cast<const char*>(&commit_time_ms), sizeof(long long));
                    snapshot_out.write(reinterpret_cast<const char*>(&commit_comm_bytes), sizeof(uint64_t));
                    snapshot_out.close();
                }} else {{
                    LOG_WARN("Failed to save binary metrics snapshot to: " << snapshot_path);
                }}
            }}

            // Clean unconsumed variables
            output.mark_consumed();
            for (auto pt : current_tensors) {{
                pt->mark_consumed();
            }}

            if (!skip_inference) {{
                stringstream profiler_buffer;
                streambuf* old_cout = cout.rdbuf(profiler_buffer.rdbuf());
                exec->print_profiler_report();
                cout.rdbuf(old_cout);
                string captured_profiler_str = profiler_buffer.str();

                string base_report_str = format_base_report("{model_name}", party, N, C, H, W, bitlen, digdec_k, do_truncation,
                                                        num_threads, omp_threads, net_io_start, net_io_end, start_time_str, stop_time_str, total_ms, prep_time);

                cout << base_report_str << captured_profiler_str;

                long long h  = total_ms / 3600000;
                long long m  = (total_ms % 3600000) / 60000;
                long long s  = (total_ms % 60000) / 1000;
                long long ms = total_ms % 1000;

                double total_comm_mb = (double)(net_io_end - net_io_start) / (1024.0 * 1024.0);

                cout << "\n\033[1;36m[====> ZK PERFORMANCE METRICS <====]\033[0m" << endl;
                cout << "  - Total Time  : ";
                if (h > 0) cout << h << "h ";
                if (m > 0 || h > 0) cout << m << "m ";
                cout << s << "s " << ms << "ms  (" << total_ms << " ms)\n";
                cout << "  - Total Comm  : " << fixed << setprecision(2) << total_comm_mb << " MB\n";
                cout << "=================================================\n" << endl;

                string align_report_str = "";

                cout << "[TEST]  Phase 4: Revealing and Verifying Output against Oracle..." << endl;
                std::vector<uint64_t> raw_logits = exec->reveal(output);

                if ((party == PARTY_VERIFIER || mode == MODE_PLAINTEXT) && !raw_logits.empty()) {{
                    size_t last_slash = input_bin_path.find_last_of('/');
                    std::string expected_bin_path = (last_slash == std::string::npos) ? "expected_logits.bin" : input_bin_path.substr(0, last_slash) + "/expected_logits.bin";
                    std::string labels_bin_path = (last_slash == std::string::npos) ? "labels_0.bin" : input_bin_path.substr(0, last_slash) + "/labels_0.bin";

                    std::ifstream ifs(expected_bin_path, std::ios::binary);
                    std::vector<float> expected_logits;
                    if(ifs.is_open()) {{
                        ifs.seekg(0, std::ios::end);
                        size_t size = ifs.tellg();
                        ifs.seekg(0, std::ios::beg);
                        expected_logits.resize(size / sizeof(float));
                        ifs.read(reinterpret_cast<char*>(expected_logits.data()), size);
                        ifs.close();
                    }}

                    std::ifstream ifs_labels(labels_bin_path, std::ios::binary);
                    std::vector<int32_t> true_labels;
                    if(ifs_labels.is_open()) {{
                        ifs_labels.seekg(0, std::ios::end);
                        size_t size = ifs_labels.tellg();
                        ifs_labels.seekg(0, std::ios::beg);
                        true_labels.resize(size / sizeof(int32_t));
                        ifs_labels.read(reinterpret_cast<char*>(true_labels.data()), size);
                        ifs_labels.close();
                    }}

                    if (!expected_logits.empty()) {{
                        if (raw_logits.size() != expected_logits.size()) {{
                            cout << "\n\033[1;31m[FATAL ERROR] Dimension mismatch! ZK output size ("
                                << raw_logits.size() << ") != Oracle expected size ("
                                << expected_logits.size() << ").\033[0m" << endl;
                            exit(-1);
                        }}

                        double mse_unscaled = 0.0;
                        double mse_scaled = 0.0;
                        int match_count = 0;
                        int accuracy_count = 0;
                        size_t num_classes = raw_logits.size() / N;

                        ostringstream align_stream;

                        for (int b = 0; b < N; ++b) {{
                            int zk_argmax = 0;
                            int pt_argmax = 0;
                            double zk_max = std::numeric_limits<double>::lowest();
                            float pt_max = std::numeric_limits<float>::lowest();

                            int current_true_label = (!true_labels.empty() && b < true_labels.size()) ? true_labels[b] : -1;

                            const uint64_t PRIME = PR;
                            const uint64_t HALF_PRIME = PRIME / 2;

                            ostringstream img_debug;
                            img_debug << "\n\033[1;33m[DEBUG] --- Logits Comparison (Image " << b << ", First 15 Classes) ---\033[0m\n";
                            img_debug << left << setw(10) << "Class"
                                    << setw(22) << "Raw Int"
                                    << setw(15) << "ZK Output"
                                    << setw(15) << "Oracle"
                                    << "Abs Diff\n";

                            for(size_t c = 0; c < num_classes; ++c) {{
                                size_t i = b * num_classes + c;
                                uint64_t field_val = raw_logits[i];
                                int64_t signed_val;

                                if (field_val > HALF_PRIME) {{
                                    signed_val = static_cast<int64_t>(field_val) - static_cast<int64_t>(PRIME);
                                }} else {{
                                    signed_val = static_cast<int64_t>(field_val);
                                }}

                                double pt_val = expected_logits[i];

                                double scale_factor = static_cast<double>(1ULL << MVZK_CONFIG_SCALE);
                                double total_scale = 1.0;
                                for(int s = 0; s < {final_scale_power}; ++s) {{
                                    total_scale *= scale_factor;
                                }}

                                double zk_val_unscaled = (double)signed_val / total_scale;
                                double zk_val_scaled = zk_val_unscaled / (double)({pool_area});

                                double err_unscaled = zk_val_unscaled - pt_val;
                                mse_unscaled += err_unscaled * err_unscaled;

                                double err_scaled = zk_val_scaled - pt_val;
                                mse_scaled += err_scaled * err_scaled;

                                if (zk_val_scaled > zk_max) {{
                                    zk_max = zk_val_scaled;
                                    zk_argmax = c;
                                }}
                                if (pt_val > pt_max) {{
                                    pt_max = pt_val;
                                    pt_argmax = c;
                                }}

                                if (c < 15) {{
                                    img_debug << left << setw(10) << c
                                            << setw(22) << signed_val
                                            << setw(15) << fixed << setprecision(6) << zk_val_scaled
                                            << setw(15) << pt_val
                                            << abs(zk_val_scaled - pt_val) << "\n";
                                }}
                            }}

                            bool is_match = (zk_argmax == pt_argmax);
                            if (is_match) match_count++;

                            if (current_true_label != -1 && zk_argmax == current_true_label) accuracy_count++;

                            align_stream << "\n\033[1;36m[PREDICT] Image " << b
                                        << " | True Label = " << (current_true_label != -1 ? to_string(current_true_label) : "N/A")
                                        << " || ZK Argmax = " << zk_argmax << " (Val: " << zk_max
                                        << ") || Oracle Argmax = " << pt_argmax << " (Val: " << pt_max << ")\033[0m\n";

                            if (!is_match) {{
                                align_stream << img_debug.str();
                            }}
                        }}

                        mse_unscaled /= raw_logits.size();
                        mse_scaled /= raw_logits.size();

                        align_stream << "\n\033[1;35m[====> GROUND TRUTH ALIGNMENT & ACCURACY <====]\033[0m\n";
                        align_stream << "  - Oracle Answer Loaded : " << expected_bin_path << "\n";
                        align_stream << "  - MSE (Unscaled)       : " << scientific << mse_unscaled << fixed << "\n";
                        align_stream << "  - MSE (Scaled)         : " << scientific << mse_scaled << fixed << "\n";

                        if (!true_labels.empty()) {{
                            double accuracy = (double)accuracy_count / N * 100.0;
                            align_stream << "  - \033[1;32mZK Model Accuracy    : " << fixed << setprecision(2) << accuracy << "% (" << accuracy_count << "/" << N << ")\033[0m\n";
                        }}

                        if (match_count == N) {{
                            align_stream << "  - ZK/Oracle Consistency: \033[1;32m[ MATCH! (" << match_count << "/" << N << ") ]\033[0m\n";
                        }} else {{
                            align_stream << "  - ZK/Oracle Consistency: \033[1;31m[ MISMATCH! (" << match_count << "/" << N << ") ]\033[0m\n";
                        }}
                        align_stream << "=================================================\n\n";

                        align_report_str = align_stream.str();
                        cout << align_report_str;

                    }} else {{
                        cout << "\033[33m[WARNING] expected_logits.bin not found at " << expected_bin_path << ". Skipping MSE check.\033[0m\n" << endl;
                    }}
                }}

                string role_str = (party == PARTY_PROVER) ? "PROVER" : "VERIFIER";
                char time_buf[64];
                strftime(time_buf, sizeof(time_buf), "%Y%m%d_%H%M%S", localtime(&stop_time_t));
                string safe_time_str(time_buf);

                std::filesystem::create_directories(log_dir);

                string filename = log_dir + "report_{model_name}_" + role_str + "_"
                                + to_string(N) + "x" + to_string(H) + "x" + to_string(W)
                                + "_" + to_string(bitlen) + "bit_"
                                + safe_time_str + ".log";

                string full_report = base_report_str + captured_profiler_str + align_report_str;

                ofstream outfile(filename);
                if (outfile.is_open()) {{
                    outfile << strip_ansi(full_report);
                    outfile.close();
                    cout << "[INFO] Final complete report successfully saved to: \033[36m" << filename << "\033[0m\n" << endl;
                }} else {{
                    cerr << "\033[31m[ERROR] Failed to save report to " << filename << "\033[0m\n" << endl;
                }}
                cout << "\033[32m[SUCCESS] {model_name} Test Execution Complete!\033[0m" << endl;
            }} else {{
                cout << "\n\033[1;36m[====> FAST DUMP COMPLETE <====]\033[0m" << endl;
                cout << "  - Weights Injected : Yes" << endl;
                cout << "  - Checkpoint Saved : " << checkpoint_dir << endl;
                cout << "  - Commit Time      : " << commit_time_ms << " ms" << endl;
                cout << "\033[32m[SUCCESS] {model_name} Fast Dump Complete! Ready for Phase 2.\033[0m" << endl;
            }}
        }}

        else if (mode == MODE_CONSISTENCY) {{
            cout << "\n[TEST]  Phase 2: Waking up ZK states from isolated directories..." << endl;

            int total_parties = cli_total_parties;
            int num_verifiers = total_parties > 1 ? total_parties - 1 : 1;

            std::vector<{model_name}PolyWeights> p_weights_list;
            {model_name}PolyWeights v_weights;
            std::vector<std::vector<PolyTensor*>> check_data;

            if (party == PARTY_PROVER) {{
                p_weights_list.resize(num_verifiers);
                check_data.resize(num_verifiers);
                for (int j = 0; j < num_verifiers; ++j) {{
                    check_data[j] = get_all_weight_tensors(p_weights_list[j]);
                }}
            }} else {{
                check_data.push_back(get_all_weight_tensors(v_weights));
            }}

            load_tensors_1toN(check_data, party, num_verifiers, checkpoint_dir);

            size_t total_params_checked = 0;
            if (!check_data.empty()) {{
                for (auto* pt : check_data[0]) {{
                    total_params_checked += pt->total_elements;
                }}
            }}

            cout << "[TEST]  Executing Multi-Verifier Consistency Check (" << total_params_checked << " Params)..." << endl;
            exec->sync_network();
            uint64_t check_io_start = exec->get_comm_bytes_counter();
            auto t_chk_start = high_resolution_clock::now();

            exec->consistency_check(check_data);
            if (!MVZK_CONFIG_BROADCAST_ASSUME_IDEAL) exec->sync_network();

            auto t_chk_end = high_resolution_clock::now();
            uint64_t check_io_end = exec->get_comm_bytes_counter();

            long long consistency_check_time_ms = duration_cast<milliseconds>(t_chk_end - t_chk_start).count();
            uint64_t consistency_check_comm = check_io_end - check_io_start;

            long long total_commit_time_ms = 0;
            uint64_t total_commit_comm_bytes = 0;

            int loop_start = (party == PARTY_PROVER) ? 1 : (party - 1);
            int loop_end = (party == PARTY_PROVER) ? (num_verifiers + 1) : (party - 1 + 1);

            for (int i = loop_start; i < loop_end; ++i) {{
                string cur_sub_dir = checkpoint_dir + (party == PARTY_PROVER ?
                                     "/Prover_with_Verifier_" + to_string(i) :
                                     "/Verifier_" + to_string(i) + "_with_Prover");

                string cur_snap_path = cur_sub_dir + "/commit_snapshot.bin";
                ifstream snap_in(cur_snap_path, ios::binary);
                if (snap_in.is_open()) {{
                    long long t = 0;
                    uint64_t c = 0;
                    snap_in.read(reinterpret_cast<char*>(&t), sizeof(long long));
                    snap_in.read(reinterpret_cast<char*>(&c), sizeof(uint64_t));
                    total_commit_time_ms += t;
                    total_commit_comm_bytes += c;
                    snap_in.close();
                }} else {{
                    LOG_WARN("Phase 1 snapshot not found for Verifier " << i << " at " << cur_snap_path << "! Using 0 for this instance.");
                }}
            }}

            time_t stop_time_t = system_clock::to_time_t(system_clock::now());
            string stop_time_str = ctime(&stop_time_t); stop_time_str.pop_back();

            string consistency_report_str = format_consistency_report(
                "{model_name}", party, total_parties, total_params_checked,
                start_time_str, stop_time_str,
                total_commit_time_ms, consistency_check_time_ms,
                total_commit_comm_bytes, consistency_check_comm
            );

            cout << consistency_report_str;
            cout << "\033[32m[SUCCESS] Consistency Check Pipeline Complete!\033[0m" << endl;
        }}

    }} catch (const std::exception& e) {{
        cerr << "\n\033[31m[FATAL ERROR] " << e.what() << "\033[0m" << endl;
        if(exec) delete exec;
        return 1;
    }}

    if(exec) delete exec;

    return 0;
}}
