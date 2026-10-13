#ifndef MVZK_BACKEND_VERIFIER_IMPL_CONSISTENCY_HPP__
#define MVZK_BACKEND_VERIFIER_IMPL_CONSISTENCY_HPP__

#include "../../exec/ExecVerifier.h"

template <typename IO>
uint64_t MVZKExecVerifier<IO>::broadcast(uint64_t dummy_val, bool assume_ideal_broadcast) {
    uint64_t received_val = 0;
    auto b_ios = this->net_manager->broadcast_ios;
    int num_parties = this->net_manager->num_parties;
    int my_id = this->net_manager->my_party_id;

    if (b_ios == nullptr || num_parties <= 2) {
        if (b_ios && b_ios[ALICE]) {
            b_ios[ALICE]->recv_data(&received_val, sizeof(uint64_t));
        } else {
            this->io->recv_data(&received_val, sizeof(uint64_t));
        }
        return received_val;
    }

    b_ios[ALICE]->recv_data(&received_val, sizeof(uint64_t));

    if (!assume_ideal_broadcast){
        emp::Hash hash_instance;
        char local_dig[emp::Hash::DIGEST_SIZE];
        hash_instance.put(&received_val, sizeof(uint64_t));
        hash_instance.digest(local_dig);

        for (int i = 2; i <= num_parties; ++i) {
            if (i == my_id || b_ios[i] == nullptr) continue;
            b_ios[i]->send_data(local_dig, emp::Hash::DIGEST_SIZE);
            b_ios[i]->flush();
            LOG_DEBUG("[Party " << my_id << "] Sending Hash to Verifier " << i);
        }

        bool equivocation_detected = false;
        for (int i = 2; i <= num_parties; ++i) {
            if (i == my_id || b_ios[i] == nullptr) continue;

            char peer_dig[emp::Hash::DIGEST_SIZE];
            b_ios[i]->recv_data(peer_dig, emp::Hash::DIGEST_SIZE);

            if (std::memcmp(local_dig, peer_dig, emp::Hash::DIGEST_SIZE) != 0) {
                equivocation_detected = true;
            }
        }

        if (equivocation_detected) {
            LOG_ERROR("Broadcast cross-check failed! Prover is equivocating.");
            if (MVZK_CONFIG_INSTANT_ABORT) {
                exit(-1);
            }
        }
    }
    return received_val;
}


template <typename IO>
void MVZKExecVerifier<IO>::consistency_check(const std::vector<std::vector<PolyTensor*>>& data) {
    if (data.empty() || data[0].empty()) return;

    size_t num_weights = 0;
    for (const auto* pt : data[0]) {
        num_weights += pt->total_elements;
    }
    if (num_weights == 0) return;

    int d_j = data[0][0]->degree;

    __uint128_t vole_returned;
    this->vole->extend(&vole_returned, 1);
    LOG_DEBUG("END EXTEND");

    uint64_t lam;
    this->io->recv_data(&lam, sizeof(uint64_t));

    uint64_t K_r = add_mod(LOW64(vole_returned), mult_mod(this->delta, lam));
    LOG_DEBUG("END R COMMIT");

    block seed = this->coin_tossing();
    emp::PRG shared_prg;
    shared_prg.reseed(&seed);

    std::vector<uint64_t> chi(num_weights + 1);
    shared_prg.random_data(chi.data(), (num_weights + 1) * sizeof(uint64_t));
    for(size_t i = 0; i < num_weights + 1; ++i) chi[i] %= PR;
    LOG_DEBUG("END COIN");

    uint64_t Z_val = this->broadcast();
    LOG_DEBUG("END BD Z");

    std::vector<uint64_t> delta_pow(d_j + 1);
    delta_pow[0] = 1;
    for (int k = 1; k <= d_j; ++k) delta_pow[k] = mult_mod(delta_pow[k-1], this->delta);

    uint64_t K_r_shifted = mult_mod(K_r, delta_pow[d_j - 1]);
    uint64_t K_Z = mult_mod(K_r_shifted, chi[num_weights]);

    size_t c_idx = 0;
    for (const auto* pt : data[0]) {
        const uint64_t* k_ptr = pt->get_keys_ptr();
        size_t size = pt->total_elements;

        if (size >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel()) {
            int num_threads = omp_get_max_threads();
            std::vector<uint64_t> local_sums(num_threads, 0);

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                uint64_t l_sum = 0;

                #pragma omp for nowait
                for (size_t i = 0; i < size; ++i) {
                    l_sum = add_mod(l_sum, mult_mod(k_ptr[i], chi[c_idx + i]));
                }
                local_sums[tid] = l_sum;
            }

            for (int t = 0; t < num_threads; ++t) {
                K_Z = add_mod(K_Z, local_sums[t]);
            }
        } else {
            for (size_t i = 0; i < size; ++i) {
                K_Z = add_mod(K_Z, mult_mod(k_ptr[i], chi[c_idx + i]));
            }
        }
        c_idx += size;
    }

    std::vector<uint64_t> M_Z(d_j);
    this->io->recv_data(M_Z.data(), d_j * sizeof(uint64_t));

    uint64_t expected_K = 0;
    for (int k = 0; k < d_j; ++k) {
        expected_K = add_mod(expected_K, mult_mod(M_Z[k], delta_pow[k]));
    }
    expected_K = add_mod(expected_K, mult_mod(Z_val, delta_pow[d_j]));

    if (expected_K != K_Z) {
        LOG_ERROR("Weight Consistency Check Failed! Prover equivocation detected.");
        if (MVZK_CONFIG_INSTANT_ABORT) exit(-1);
    } else {
        LOG_INFO("Weight Consistency Check Passed.");
    }
}

#endif
