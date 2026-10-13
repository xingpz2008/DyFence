#ifndef MVZK_NETWORK_MANAGER_H__
#define MVZK_NETWORK_MANAGER_H__

#include <cstdint>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <thread>
#include "emp-tool/emp-tool.h"
#include "emp-tool/utils/ThreadPool.h"
#include "../config.h"
#include "emp-tool/utils/constants.h"
#include "utils/logger.h"

template <typename IO>
class MVZKNetworkManager {
public:
    int my_party_id;
    int num_parties;
    int num_vole_ios;

    IO** vole_ios = nullptr;
    IO** broadcast_ios = nullptr;
    ThreadPool* net_pool = nullptr;
    uint64_t sync_round = 0;

    MVZKNetworkManager() = default;

    MVZKNetworkManager(int party_id, int base_port, const std::vector<std::string>& ips = {}, int target_party_id = -1, int override_num_parties = 0)
        : my_party_id(party_id) {
        // Directly return for plaintext acc
        if (base_port == -1) {
            LOG_INFO("Skip all netwotk construction for plaintext acc mode.");
            this->num_parties = 1;
            this->num_vole_ios = 0;
            return;
        }

        this->num_parties = (override_num_parties > 0) ? override_num_parties : MVZK_CONFIG_NUM_PARTIES;
        int threads = MVZK_CONFIG_THREADS_NUM;
        int vole_base_port = base_port;

        if (is_prover()) {
            if (target_party_id != -1) {
                this->num_vole_ios = threads;
                this->vole_ios = new IO*[this->num_vole_ios];
                for (int t = 0; t < threads; ++t) {
                    this->vole_ios[t] = new IO(nullptr, vole_base_port + target_party_id * 100 + t);
                }
            }

            else {
                int num_verifiers = (this->num_parties > 1) ? (this->num_parties - 1) : 1;
                this->num_vole_ios = num_verifiers * threads;
                this->vole_ios = new IO*[this->num_vole_ios];

                for (int p = 2; p <= this->num_parties; ++p) {
                    for (int t = 0; t < threads; ++t) {
                        int flat_index = (p - 2) * threads + t;
                        this->vole_ios[flat_index] = new IO(nullptr, vole_base_port + p * 100 + t);
                    }
                }
            }
        } else {
            this->num_vole_ios = threads;
            this->vole_ios = new IO*[threads];
            std::string prover_ip = ips.empty() ? "127.0.0.1" : ips[0];

            for (int t = 0; t < threads; ++t) {
                this->vole_ios[t] = new IO(prover_ip.c_str(), vole_base_port + my_party_id * 100 + t);
            }
        }

        if (this->num_parties > 2 && target_party_id == -1) {
            this->broadcast_ios = new IO*[this->num_parties + 1];
            for (int i = 0; i <= this->num_parties; ++i) this->broadcast_ios[i] = nullptr;

            int broadcast_base_port = base_port + 10000;
            for (int i = 1; i <= this->num_parties; ++i) {
                for (int j = i + 1; j <= this->num_parties; ++j) {
                    int pair_port = broadcast_base_port + i * this->num_parties + j;
                    if (my_party_id == i) {
                        this->broadcast_ios[j] = new IO(nullptr, pair_port);
                    } else if (my_party_id == j) {
                        std::string target_ip = (ips.size() > i) ? ips[i - 1] : "127.0.0.1";
                        this->broadcast_ios[i] = new IO(target_ip.c_str(), pair_port);
                    }
                }
            }

            if (this->num_parties > MVZK_CONFIG_BROADCAST_USING_THREAD_POOL_THRE) {
                this->net_pool = new ThreadPool(this->num_parties + 2);
            }
        }
    }

    ~MVZKNetworkManager() {
        if (num_vole_ios == -1) {
            return;
        }

        if (vole_ios) {
            for (int i = 0; i < num_vole_ios; ++i) {
                if (vole_ios[i]) {
                    vole_ios[i]->flush();
                    delete vole_ios[i];
                }
            }
            delete[] vole_ios;
        }

        if (broadcast_ios) {
            for (int i = 0; i <= num_parties; ++i) {
                if (broadcast_ios[i]) {
                    broadcast_ios[i]->flush();
                    delete broadcast_ios[i];
                }
            }
            delete[] broadcast_ios;
        }

        if (net_pool) {
            delete net_pool;
        }
    }

    bool is_prover() const { return my_party_id == emp::ALICE; }
    IO** get_primary_ios() const { return vole_ios; }

    void sync_network() {
        if (broadcast_ios == nullptr || num_parties < 2) return;

        const uint64_t round = ++sync_round;
        auto require_round = [round](uint64_t received) {
            if (received != round) {
                LOG_ERROR("Network synchronization round mismatch: expected " << round << ", received " << received);
                std::exit(1);
            }
        };

        if (is_prover()) {
            for (int party = 2; party <= num_parties; ++party) {
                uint64_t ready = 0;
                broadcast_ios[party]->recv_data(&ready, sizeof(ready));
                require_round(ready);
            }
            for (int party = 2; party <= num_parties; ++party) {
                broadcast_ios[party]->send_data(&round, sizeof(round));
                broadcast_ios[party]->flush();
            }
            for (int party = 2; party <= num_parties; ++party) {
                uint64_t acknowledgement = 0;
                broadcast_ios[party]->recv_data(&acknowledgement, sizeof(acknowledgement));
                require_round(acknowledgement);
            }
            return;
        }

        IO* coordinator = broadcast_ios[emp::ALICE];
        if (coordinator == nullptr) {
            LOG_ERROR("Network synchronization coordinator channel is missing for party " << my_party_id);
            std::exit(1);
        }
        coordinator->send_data(&round, sizeof(round));
        coordinator->flush();

        uint64_t release = 0;
        coordinator->recv_data(&release, sizeof(release));
        require_round(release);

        coordinator->send_data(&round, sizeof(round));
        coordinator->flush();
    }

    uint64_t get_comm_bytes_counter() const {
        uint64_t total = 0;
        if (vole_ios) {
            for (int i = 0; i < num_vole_ios; ++i) {
                if (vole_ios[i]) total += vole_ios[i]->counter;
            }
        }
        if (broadcast_ios) {
            for (int i = 1; i <= num_parties; ++i) {
                if (broadcast_ios[i]) total += broadcast_ios[i]->counter;
            }
        }
        return total;
    }
};

#endif // MVZK_NETWORK_MANAGER_H__
