
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <string>
#include <random>
#include <algorithm> // For std::min
#include <filesystem>
#include <omp.h>

#include "exec/ExecProver.h"
#include "exec/ExecVerifier.h"
#include "exec/MVZKNetBase.h"
#include "utils/consistency_utils.h"
#include "emp-tool/emp-tool.h"

using namespace std;
using namespace emp;
using namespace std::chrono;

MVZKExec* MVZKExec::mvzk_exec = nullptr;
const int PARTY_PROVER = ALICE;

// Core: Chunked Mock Commit generation to bypass OOM and Int32 limits

template<typename IO>
long long generate_base_mock_tensor(MVZKExec* exec, int party, size_t num_params, std::vector<PolyTensor*>& out_data) {
    long long total_elapsed_ms = 0;

    // THE SILVER BULLET: Limit maximum contiguous allocation to 10 Million
    const size_t CHUNK_LIMIT = MVZK_CONFIG_BENCHMARK_CONSISTENCY_CHUNK_LIMIT;

    if (party == PARTY_PROVER) {
        auto prover = static_cast<MVZKExecProver<IO>*>(exec);

        // Fixed random seed to ensure that when symlinks are expanded,
        // all Verifiers validate the same plaintext X
        std::mt19937 gen(12345);
        IO* current_io = prover->net_manager->vole_ios[0];

        // Streaming loop: Bite off CHUNK_LIMIT pieces at a time
        for (size_t offset = 0; offset < num_params; offset += CHUNK_LIMIT) {
            size_t current_batch = std::min(CHUNK_LIMIT, num_params - offset);

            std::vector<uint64_t> X(current_batch);
            for(size_t i = 0; i < current_batch; ++i) X[i] = gen() % PR;

            auto t_start = high_resolution_clock::now();

            // SUPER SAFE: current_batch will NEVER exceed 10,000,000.
            // (int)cast is mathematically guaranteed to NOT overflow here!
            PolyTensor* pt = new PolyTensor({(int)current_batch}, 1);
            uint64_t* m_ptr = pt->get_coeffs_ptr(0);
            uint64_t* x_ptr = pt->get_coeffs_ptr(1);

            std::vector<__uint128_t> vole_res(current_batch);
            prover->voles[0]->extend(vole_res.data(), current_batch);

            std::vector<uint64_t> lam_buf(current_batch);

            #pragma omp parallel for if (current_batch >= MVZK_OMP_SIZE_THRESHOLD)
            for (size_t i = 0; i < current_batch; ++i) {
                uint64_t M_u = (uint64_t)LOW64(vole_res[i]);
                uint64_t u   = (uint64_t)HIGH64(vole_res[i]);
                m_ptr[i] = M_u;
                x_ptr[i] = X[i];
                lam_buf[i] = add_mod(u, X[i]);
            }

            current_io->send_data(lam_buf.data(), current_batch * sizeof(uint64_t));
            current_io->flush();
            out_data.push_back(pt);

            auto t_end = high_resolution_clock::now();
            total_elapsed_ms += duration_cast<milliseconds>(t_end - t_start).count();
        }
    } else {
        auto verifier = static_cast<MVZKExecVerifier<IO>*>(exec);
        uint64_t delta = verifier->delta;

        // Streaming loop for verifier
        for (size_t offset = 0; offset < num_params; offset += CHUNK_LIMIT) {
            size_t current_batch = std::min(CHUNK_LIMIT, num_params - offset);
            auto t_start = high_resolution_clock::now();

            PolyTensor* pt = new PolyTensor({(int)current_batch}, 1);
            uint64_t* k_ptr = pt->get_keys_ptr();

            std::vector<__uint128_t> vole_res(current_batch);
            verifier->vole->extend(vole_res.data(), current_batch);

            std::vector<uint64_t> lam_buf(current_batch);
            verifier->io->recv_data(lam_buf.data(), current_batch * sizeof(uint64_t));

            #pragma omp parallel for if (current_batch >= MVZK_OMP_SIZE_THRESHOLD)
            for (size_t i = 0; i < current_batch; ++i) {
                uint64_t delta_mult_lam = mult_mod(delta, lam_buf[i]);
                k_ptr[i] = add_mod(LOW64(vole_res[i]), delta_mult_lam);
            }
            out_data.push_back(pt);

            auto t_end = high_resolution_clock::now();
            total_elapsed_ms += duration_cast<milliseconds>(t_end - t_start).count();
        }
    }
    return total_elapsed_ms;
}

// Minimalist Main Control Flow

int main(int argc, char** argv) {
    if (argc < 6) {
        cerr << "Usage: " << argv[0] << " <party> <port> <total_parties> <num_params> <mode (1=Dump, 2=Check)>" << endl;
        return -1;
    }

    int party = atoi(argv[1]);
    int port = atoi(argv[2]);
    int total_parties = atoi(argv[3]);
    size_t num_params = stoull(argv[4]);
    int mode = atoi(argv[5]);

    if (mode != 1 && mode != 2) {
        cerr << "Invalid mode " << mode << "; expected 1 (Dump) or 2 (Check)." << endl;
        return 1;
    }

    // Directory planning
    string ckpt_dir = "./checkpoints_N" + to_string(num_params);
    std::filesystem::create_directories(ckpt_dir);

    MVZKNetBase<NetIO> *exec = nullptr;

    try {
        // MODE 1: Generate base files (Forced 2-party network execution)

        if (mode == 1) {
            if (party > 2) return 0; // Mode 1 only requires P1 and P2 to participate

            if (party == PARTY_PROVER) exec = new MVZKExecProver<NetIO>(party, port, {}, 2, 2);
            else exec = new MVZKExecVerifier<NetIO>(party, port, {}, "", 2);

            exec->sync_network();
            cout << "[MODE 1] Generating chunked mock tensor for " << num_params << " parameters..." << endl;

            std::vector<PolyTensor*> mock_tensors;
            uint64_t io_start = exec->get_comm_bytes_counter();
            long long time_ms = generate_base_mock_tensor<NetIO>(exec, party, num_params, mock_tensors);
            uint64_t comm_bytes = exec->get_comm_bytes_counter() - io_start;

            int verifier_idx = 1; // The base template is always bound to ID 1
            string sub_dir = ckpt_dir + (party == PARTY_PROVER ? "/Prover_with_Verifier_1" : "/Verifier_1_with_Prover");
            std::filesystem::create_directories(sub_dir);

            save_tensors_1to1(mock_tensors, party, verifier_idx, ckpt_dir);
            if (party != PARTY_PROVER) {
                static_cast<MVZKExecVerifier<NetIO>*>(exec)->save_delta_to_file(sub_dir + "/verifier_delta.bin");
            }

            ofstream snap_out(sub_dir + "/commit_snapshot.bin", ios::binary);
            if(snap_out.is_open()) {
                snap_out.write(reinterpret_cast<const char*>(&time_ms), sizeof(long long));
                snap_out.write(reinterpret_cast<const char*>(&comm_bytes), sizeof(uint64_t));
                snap_out.close();
            }

            for(auto* pt : mock_tensors) delete pt;
            cout << "[MODE 1] Base snapshots dumped successfully to: " << ckpt_dir << endl;
        }

        // MODE 2: Consistency Check

        else if (mode == 2) {
            int num_verifiers = total_parties - 1;
            int verifier_idx = (party == PARTY_PROVER) ? 1 : (party - 1);

            string sub_dir = ckpt_dir + (party == PARTY_PROVER ?
                             "/Prover_with_Verifier_" + to_string(verifier_idx) :
                             "/Verifier_" + to_string(verifier_idx) + "_with_Prover");
            string delta_path = sub_dir + "/verifier_delta.bin";

            if (party == PARTY_PROVER) exec = new MVZKExecProver<NetIO>(party, port, {}, -1, total_parties);
            else exec = new MVZKExecVerifier<NetIO>(party, port, {}, delta_path, total_parties);

            cout << "[MODE 2] Loading checkpoints/symlinks from disk..." << endl;
            std::vector<std::vector<PolyTensor*>> check_data;
            if (party == PARTY_PROVER) check_data.resize(num_verifiers);
            else check_data.resize(1);

            // C++ layer resolves symlinks natively
            load_tensors_1toN(check_data, party, num_verifiers, ckpt_dir);

            cout << "[MODE 2] Executing Multi-Verifier Consistency Check..." << endl;
            exec->sync_network();

            uint64_t chk_io_start = exec->get_comm_bytes_counter();
            auto t_chk_start = high_resolution_clock::now();

            exec->consistency_check(check_data);
            if (!MVZK_CONFIG_BROADCAST_ASSUME_IDEAL) exec->sync_network();

            auto t_chk_end = high_resolution_clock::now();
            long long consistency_check_time_ms = duration_cast<milliseconds>(t_chk_end - t_chk_start).count();
            uint64_t consistency_check_comm = exec->get_comm_bytes_counter() - chk_io_start;

            // Replicate Phase 1 Snapshot Loading Logic

            long long total_commit_time_ms = 0;
            uint64_t total_commit_comm_bytes = 0;

            int loop_start = (party == PARTY_PROVER) ? 1 : (party - 1);
            int loop_end = (party == PARTY_PROVER) ? (num_verifiers + 1) : (party - 1 + 1);

            for (int i = loop_start; i < loop_end; ++i) {
                string cur_sub_dir = ckpt_dir + (party == PARTY_PROVER ?
                                     "/Prover_with_Verifier_" + to_string(i) :
                                     "/Verifier_" + to_string(i) + "_with_Prover");

                string cur_snap_path = cur_sub_dir + "/commit_snapshot.bin";
                ifstream snap_in(cur_snap_path, ios::binary);
                if (snap_in.is_open()) {
                    long long t = 0;
                    uint64_t c = 0;
                    snap_in.read(reinterpret_cast<char*>(&t), sizeof(long long));
                    snap_in.read(reinterpret_cast<char*>(&c), sizeof(uint64_t));
                    total_commit_time_ms += t;
                    total_commit_comm_bytes += c;
                    snap_in.close();
                } else {
                    LOG_WARN("Phase 1 snapshot not found for Verifier " << i << " at " << cur_snap_path << "! Using 0 for this instance.");
                }
            }

            // Output Complete Two-Phase Report

            double commit_mb = (double)total_commit_comm_bytes / (1024.0 * 1024.0);
            double check_mb = (double)consistency_check_comm / (1024.0 * 1024.0);
            double total_mb = commit_mb + check_mb;
            long long total_ms = total_commit_time_ms + consistency_check_time_ms;
            string role_str = (party == PARTY_PROVER) ? "PROVER (Alice)" : ("VERIFIER " + to_string(party - 1) + " (Bob)");

            cout << "\n\033[1;36m=================================================\n";
            cout << "          CONSISTENCY CHECK REPORT               \n";
            cout << "=================================================\033[0m\n";
            cout << "[Task Info]\n";
            cout << "  - Role         : " << role_str << "\n";
            cout << "  - Total Parties: " << total_parties << " (1 Prover, " << num_verifiers << " Verifiers)\n";
            cout << "  - Params Chk'd : " << num_params << " elements\n";
            cout << "  - Ideal Chnl   : " << (MVZK_CONFIG_BROADCAST_ASSUME_IDEAL ? "Yes" : "No") << "\n";

            cout << "\n[Execution Time]\n";
            cout << "  - Phase 1 (Commit Time) : " << total_commit_time_ms << " ms\n";
            cout << "  - Phase 2 (Check Time)  : " << consistency_check_time_ms << " ms\n";
            cout << "  - \033[1;32mTotal Time (P1+P2)      : " << total_ms << " ms\033[0m\n";

            cout << "\n[Communication Overhead]\n";
            cout << "  - Phase 1 (Commit Comm) : " << fixed << setprecision(2) << commit_mb << " MB\n";
            cout << "  - Phase 2 (Check Comm)  : " << fixed << setprecision(2) << check_mb << " MB\n";
            cout << "  - \033[1;32mTotal Comm (P1+P2)      : " << fixed << setprecision(2) << total_mb << " MB\033[0m\n";
            cout << "\033[1;36m=================================================\033[0m\n";

            for(auto& vec : check_data) { for(auto* pt : vec) delete pt; }
        }
    } catch (const exception& e) {
        cerr << "[FATAL ERROR] " << e.what() << endl;
        if(exec) delete exec;
        return 1;
    }

    if(exec) delete exec;
    return 0;
}
