#ifndef MVZK_BACKEND_PROVER_IMPL_CONSISTENCY_HPP__
#define MVZK_BACKEND_PROVER_IMPL_CONSISTENCY_HPP__

#include "../../exec/ExecProver.h"

template <typename IO>
void MVZKExecProver<IO>::consistency_check(const std::vector<std::vector<PolyTensor*>>& data) {
    if (data.empty() || data[0].empty()) return;

    size_t num_weights = 0;
    for (const auto* pt : data[0]) {
        num_weights += pt->total_elements;
    }
    if (num_weights == 0) return;

    uint64_t r;
    this->prg.random_data(&r, sizeof(uint64_t));
    r %= PR;

    int num_verifiers = this->voles.size();
    std::vector<PolyDelta> r_pds(num_verifiers);
    for (int i = 0; i < num_verifiers; ++i) {
        __uint128_t vole_returned;
        this->voles[i]->extend(&vole_returned, 1);

        uint64_t M_u = (uint64_t)LOW64(vole_returned);
        uint64_t u   = (uint64_t)HIGH64(vole_returned);

        r_pds[i].degree = 1;
        r_pds[i].coeffs.resize(2);
        r_pds[i].coeffs[0] = M_u;
        r_pds[i].coeffs[1] = r;
        r_pds[i].is_consumed = false;

        uint64_t lam = add_mod(u, r);
        IO* current_io = this->net_manager->vole_ios[i * this->threads];
        current_io->send_data(&lam, sizeof(uint64_t));
        current_io->flush();
    }
    LOG_DEBUG("END R COMMIT");

    // Step 2: Jointly sample a seed (Challenge)

    block seed = this->coin_tossing();
    emp::PRG shared_prg;
    shared_prg.reseed(&seed);

    std::vector<uint64_t> chi(num_weights + 1);
    shared_prg.random_data(chi.data(), (num_weights + 1) * sizeof(uint64_t));
    for(size_t i = 0; i < num_weights + 1; ++i) chi[i] %= PR;
    LOG_DEBUG("END COIN");

    uint64_t Z_val = 0;
    size_t idx = 0;

    for (const auto* pt : data[0]) {
        const uint64_t* x_ptr = pt->get_coeffs_ptr(pt->degree);
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
                    l_sum = add_mod(l_sum, mult_mod(x_ptr[i], chi[idx + i]));
                }
                local_sums[tid] = l_sum;
            }

            for (int t = 0; t < num_threads; ++t) {
                Z_val = add_mod(Z_val, local_sums[t]);
            }
        } else {
            for (size_t i = 0; i < size; ++i) {
                Z_val = add_mod(Z_val, mult_mod(x_ptr[i], chi[idx + i]));
            }
        }
        idx += size;
    }

    Z_val = add_mod(Z_val, mult_mod(r, chi[num_weights]));
    this->broadcast(Z_val);
    LOG_DEBUG("END BROADCAST");

    #pragma omp parallel for if(num_verifiers >= 2 && !omp_in_parallel())
    for (int j = 0; j < num_verifiers; ++j) {
        int d_j = data[j][0]->degree;
        std::vector<uint64_t> M_Z_j(d_j, 0);

        size_t c_idx = 0;
        for (const auto* pt : data[j]) {
            for (int k = 0; k < d_j; ++k) {
                const uint64_t* m_ptr = pt->get_coeffs_ptr(k);
                for (size_t i = 0; i < pt->total_elements; ++i) {
                    M_Z_j[k] = add_mod(M_Z_j[k], mult_mod(m_ptr[i], chi[c_idx + i]));
                }
            }
            c_idx += pt->total_elements;
            pt->mark_consumed();
        }

        M_Z_j[d_j - 1] = add_mod(M_Z_j[d_j - 1], mult_mod(r_pds[j].coeffs[0], chi[num_weights]));
        r_pds[j].is_consumed = true;

        IO* current_io = this->net_manager->vole_ios[j * this->threads];
        current_io->send_data(M_Z_j.data(), d_j * sizeof(uint64_t));
        current_io->flush();
    }
    LOG_DEBUG("END SEND");
}


template <typename IO>
uint64_t MVZKExecProver<IO>::broadcast(uint64_t val, bool assume_ideal_broadcast) {
    auto b_ios = this->net_manager->broadcast_ios;
    int num_parties = this->net_manager->num_parties;

    if (b_ios == nullptr || num_parties <= 2) {
        return val;
    }

    for (int i = 2; i <= num_parties; ++i) {
        if (b_ios[i] != nullptr) {
            b_ios[i]->send_data(&val, sizeof(uint64_t));
            b_ios[i]->flush();
            LOG_DEBUG("[Prover] Sending broadcast data to Verifier " << i);
        }
    }
    return val;
}

#endif
