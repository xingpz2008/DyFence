/*
 * Usage:
 * ./vgg16_test <party> <port> <N> <H> <W>
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <ctime>
#include <filesystem>
#include <omp.h>
#include <cstdlib>

#include "exec/ExecProver.h"
#include "exec/ExecVerifier.h"
#include "utility.h"
#include "emp-tool/emp-tool.h"

#include "local_models/vgg16.h"

using namespace std;
using namespace emp;
using namespace std::chrono;

MVZKExec* MVZKExec::mvzk_exec = nullptr;
const int PARTY_PROVER = ALICE;
const int PARTY_VERIFIER = BOB;

template <typename Func>
long long measure_time(Func&& func, string step_name) {
    auto start = high_resolution_clock::now();
    func();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "[TIME] " << left << setw(45) << step_name << ": " << duration.count() << " ms" << endl;
    return duration.count();
}

PolyTensor create_dummy_tensor(MVZKExec* exec, int party, const vector<int>& shape) {
    size_t total_size = 1;
    for (int s : shape) total_size *= s;

    vector<uint64_t> data(total_size, 0);
    if (party == PARTY_PROVER) {
        static mt19937 gen(12345);
        uniform_int_distribution<int64_t> dis(-5, 5);
        for (size_t i = 0; i < total_size; ++i) {
            float f_val = static_cast<float>(dis(gen)) / 100.0f;
            data[i] = real2fp(f_val);
        }
    }
    return exec->input(shape, data);
}

string strip_ansi(const string& input) {
    string output;
    bool in_escape = false;
    for (char c : input) {
        if (c == '\033') {
            in_escape = true;
        } else if (in_escape) {
            if (c == 'm') in_escape = false;
        } else {
            output += c;
        }
    }
    return output;
}

void print_test_report(const string& model_name, int party, int N, int H, int W,
                       uint64_t bitlen, uint64_t digdec_k, bool do_truncation,
                       int num_threads, int omp_threads, uint64_t net_io_counter_start, uint64_t net_io_counter_end,
                       const string& start_time_str, const string& stop_time_str, time_t stop_time_t,
                       long long total_ms, const string& profiler_str) {
    long long h  = total_ms / 3600000;
    long long m  = (total_ms % 3600000) / 60000;
    long long s  = (total_ms % 60000) / 1000;
    long long ms = total_ms % 1000;

    uint64_t total_comm_bytes = net_io_counter_end - net_io_counter_start;
    double total_comm_kb = (double)total_comm_bytes / 1024.0;
    double total_comm_mb = (double)total_comm_bytes / (1024.0 * 1024.0);
    double total_comm_gb = total_comm_mb / 1024.0;

    ostringstream oss;
    oss << "\n=================================================\n";
    oss << "                 FINAL TEST REPORT               \n";
    oss << "=================================================\n";
    oss << "[Model & Task Info]\n";
    oss << "  - Model Name   : " << model_name << "\n";
    oss << "  - Input Shape  : (N=" << N << ", C=3, H=" << H << ", W=" << W << ")\n";
    oss << "  - Role         : " << (party == PARTY_PROVER ? "PROVER (Alice)" : "VERIFIER (Bob)") << "\n";

    oss << "\n[Test Runtime Parameters]\n";
    oss << "  - Run: Bit Length   : " << bitlen << "\n";
    oss << "  - Run: DigDec_k     : " << digdec_k << "\n";
    oss << "  - Run: Truncation   : " << (do_truncation ? "Enabled" : "Disabled") << "\n";
    oss << "  - Run: Threads (IO) : " << num_threads << "\n";
    oss << "  - Run: Threads (OMP): " << omp_threads << "\n";

    oss << "\n[Compiled System Configs (config.h)]\n";
    oss << "  - MVZK_CONFIG_THREADS_NUM     : " << MVZK_CONFIG_THREADS_NUM << "\n";
    oss << "  - MVZK_CONFIG_SCALE           : " << MVZK_CONFIG_SCALE << "\n";
    oss << "  - MVZK_CONFIG_DEFAULT_BITLEN  : " << MVZK_CONFIG_DEFAULT_BITLEN << "\n";
    oss << "  - MVZK_CONFIG_RELU_DIGDEC_K   : " << MVZK_CONFIG_NON_LINEAR_RELU_DIGDEC_K << "\n";
    oss << "  - MVZK_OMP_SIZE_THRESHOLD     : " << MVZK_OMP_SIZE_THRESHOLD << "\n";
    oss << "  - MVZK_MULT_CHECK_CNT         : " << MVZK_MULT_CHECK_CNT << "\n";
    oss << "  - MVZK_CONFIG_TENSOR_CHK_CNT  : " << MVZK_CONFIG_TENSOR_CHECK_CNT << "\n";
    oss << "  - MVZK_CONFIG_RANGE_CHK_BUF   : " << MVZK_CONFIG_RANGE_CHECK_REQUEST_BUFFER_THRESHOLD << "\n";

    oss << "\n[Execution Time]\n";
    oss << "  - Start Time   : " << start_time_str << "\n";
    oss << "  - End Time     : " << stop_time_str << "\n";
    oss << "  - Total Time   : ";
    if (h > 0) oss << h << "h ";
    if (m > 0 || h > 0) oss << m << "m ";
    oss << s << "s " << ms << "ms\n";

    oss << "\n[Communication]\n";
    oss << "  - Data Sent    : " << fixed << setprecision(2) << total_comm_mb << " MB";
    if (total_comm_gb > 1.0) oss << " (" << total_comm_gb << " GB)";
    oss << "\n=================================================\n";

    string report_str = oss.str();
    cout << report_str << endl;
    cout << profiler_str;

    string role_str = (party == PARTY_PROVER) ? "PROVER" : "VERIFIER";

    char time_buf[64];
    strftime(time_buf, sizeof(time_buf), "%Y%m%d_%H%M%S", localtime(&stop_time_t));
    string safe_time_str(time_buf);

    string log_dir = "log/";
    std::filesystem::create_directories(log_dir);

    string filename = log_dir + "report_" + model_name + "_" + role_str + "_"
                      + to_string(N) + "x" + to_string(H) + "x" + to_string(W)
                      + "_" + to_string(bitlen) + "bit_"
                      + safe_time_str + ".log";

    ofstream outfile(filename);
    if (outfile.is_open()) {
        outfile << report_str;
        outfile << strip_ansi(profiler_str);
        outfile.close();
        cout << "[INFO] Report successfully saved to: \033[36m" << filename << "\033[0m" << endl;
    } else {
        cerr << "\033[31m[ERROR] Failed to save report to " << filename << "\033[0m" << endl;
    }
}

VGG16Weights vgg16(MVZKExec* exec, int party, int input_H, int input_W) {
    VGG16Weights model;

    model.conv1_1_w = create_dummy_tensor(exec, party, {64, 3, 3, 3});
    model.conv1_1_b = create_dummy_tensor(exec, party, {64});
    model.conv1_2_w = create_dummy_tensor(exec, party, {64, 64, 3, 3});
    model.conv1_2_b = create_dummy_tensor(exec, party, {64});

    model.conv2_1_w = create_dummy_tensor(exec, party, {128, 64, 3, 3});
    model.conv2_1_b = create_dummy_tensor(exec, party, {128});
    model.conv2_2_w = create_dummy_tensor(exec, party, {128, 128, 3, 3});
    model.conv2_2_b = create_dummy_tensor(exec, party, {128});

    model.conv3_1_w = create_dummy_tensor(exec, party, {256, 128, 3, 3});
    model.conv3_1_b = create_dummy_tensor(exec, party, {256});
    model.conv3_2_w = create_dummy_tensor(exec, party, {256, 256, 3, 3});
    model.conv3_2_b = create_dummy_tensor(exec, party, {256});
    model.conv3_3_w = create_dummy_tensor(exec, party, {256, 256, 3, 3});
    model.conv3_3_b = create_dummy_tensor(exec, party, {256});

    model.conv4_1_w = create_dummy_tensor(exec, party, {512, 256, 3, 3});
    model.conv4_1_b = create_dummy_tensor(exec, party, {512});
    model.conv4_2_w = create_dummy_tensor(exec, party, {512, 512, 3, 3});
    model.conv4_2_b = create_dummy_tensor(exec, party, {512});
    model.conv4_3_w = create_dummy_tensor(exec, party, {512, 512, 3, 3});
    model.conv4_3_b = create_dummy_tensor(exec, party, {512});

    model.conv5_1_w = create_dummy_tensor(exec, party, {512, 512, 3, 3});
    model.conv5_1_b = create_dummy_tensor(exec, party, {512});
    model.conv5_2_w = create_dummy_tensor(exec, party, {512, 512, 3, 3});
    model.conv5_2_b = create_dummy_tensor(exec, party, {512});
    model.conv5_3_w = create_dummy_tensor(exec, party, {512, 512, 3, 3});
    model.conv5_3_b = create_dummy_tensor(exec, party, {512});

    int pooled_H = std::max(1, input_H / 32);
    int pooled_W = std::max(1, input_W / 32);
    int flattened_size = 512 * pooled_H * pooled_W;

    model.fc1_w = create_dummy_tensor(exec, party, {4096, flattened_size});
    model.fc1_b = create_dummy_tensor(exec, party, {4096});

    model.fc2_w = create_dummy_tensor(exec, party, {4096, 4096});
    model.fc2_b = create_dummy_tensor(exec, party, {4096});

    model.fc3_w = create_dummy_tensor(exec, party, {1000, 4096});
    model.fc3_b = create_dummy_tensor(exec, party, {1000});

    return model;
}

int main(int argc, char** argv) {
    SetLogLevel(LEVEL_ERROR);
    if (argc < 6) {
        cout << "Usage: " << argv[0] << " <party> <port> <N> <H> <W>" << endl;
        return 1;
    }

    int party = atoi(argv[1]);
    int port = atoi(argv[2]);
    int N = atoi(argv[3]), H = atoi(argv[4]), W = atoi(argv[5]);

    uint64_t bitlen = MVZK_CONFIG_DEFAULT_BITLEN;
    uint64_t digdec_k = MVZK_CONFIG_NON_LINEAR_RELU_DIGDEC_K;
    bool do_truncation = MVZK_CONFIG_NON_LINEAR_RELU_DO_TRUNCATION;

    cout << "=================================================" << endl;
    cout << "VGG-16 Full ZK Graph Test" << endl;
    cout << "Role: " << (party == PARTY_PROVER ? "PROVER" : "VERIFIER") << endl;
    cout << "=================================================" << endl;

    int num_threads = MVZK_CONFIG_THREADS_NUM;

    int omp_threads = omp_get_max_threads();

    NetIO **io_arr = new NetIO*[num_threads];
    for (int i = 0; i < num_threads; ++i) {
        io_arr[i] = new NetIO(party == PARTY_PROVER ? nullptr : "127.0.0.1", port + i);
    }

    MVZKExec *exec = nullptr;
    if (party == PARTY_PROVER) exec = new MVZKExecProver<NetIO>(io_arr);
    else exec = new MVZKExecVerifier<NetIO>(io_arr);

    try {
        uint64_t net_io_start = comm(io_arr, num_threads);

        auto start_zk = high_resolution_clock::now();

        time_t start_time_t = system_clock::to_time_t(system_clock::now());
        string start_time_str = ctime(&start_time_t);
        start_time_str.pop_back();

        cout << "[TEST] Loading Dummy Image..." << endl;
        PolyTensor image = create_dummy_tensor(exec, party, {N, 3, H, W});

        cout << "[TEST] Instantiating VGG-16 Model Weights..." << endl;
        VGG16Weights model = vgg16(exec, party, H, W);

        PolyTensor output;
        output = VGG16_Forward(image, model, bitlen, digdec_k, do_truncation);

        cout << "[TEST] Output Shape: ("
             << output.shape[0] << ", " << output.shape[1] << ")" << endl;

        PolyTensor::store_self_relation(output, "VGG16_Final_Logits_Check");

        exec->finalize_protocol();

        auto stop_zk = high_resolution_clock::now();

        time_t stop_time_t = system_clock::to_time_t(system_clock::now());
        string stop_time_str = ctime(&stop_time_t);
        stop_time_str.pop_back();

        uint64_t net_io_end = comm(io_arr, num_threads);

        auto total_ms = duration_cast<milliseconds>(stop_zk - start_zk).count();

        stringstream profiler_buffer;
        streambuf* old_cout = cout.rdbuf(profiler_buffer.rdbuf());
        exec->print_profiler_report();
        cout.rdbuf(old_cout);
        string captured_profiler_str = profiler_buffer.str();

        print_test_report("VGG16", party, N, H, W, bitlen, digdec_k, do_truncation,
                          num_threads, omp_threads, net_io_start, net_io_end, start_time_str, stop_time_str,
                          stop_time_t, total_ms, captured_profiler_str);

        long long h  = total_ms / 3600000;
        long long m  = (total_ms % 3600000) / 60000;
        long long s  = (total_ms % 60000) / 1000;
        long long ms = total_ms % 1000;

        cout << "\n\033[1;36m[====> TOTAL EXECUTION TIME <====]\033[0m" << endl;
        cout << "  ";
        if (h > 0) cout << h << "h ";
        if (m > 0 || h > 0) cout << m << "m ";
        cout << s << "s " << ms << "ms  (" << total_ms << " ms)\n" << endl;
        cout << "=================================================\n" << endl;

        cout << "\033[32m[SUCCESS] VGG-16 Test Passed Beautifully!\033[0m" << endl;
    } catch (const std::exception& e) {
        cerr << "\n\033[31m[FATAL ERROR] " << e.what() << "\033[0m" << endl;
    }

    if(exec) delete exec;

    for (int i = 0; i < num_threads; ++i) {
        delete io_arr[i];
    }
    delete[] io_arr;

    return 0;
}
