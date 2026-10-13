#ifndef MVZK_NET_BASE_H__
#define MVZK_NET_BASE_H__

#include "MVZKExec.h"
#include "MVZKNetworkManager.h"
#include "../data_type/PolyTensor.h"
#include "emp-tool/emp-tool.h"
#include <vector>
#include <cstring>

template <typename IO>
class MVZKNetBase : public MVZKExec {
public:
    MVZKNetworkManager<IO>* net_manager;
    bool owns_net_manager = false;

    MVZKNetBase(MVZKNetworkManager<IO>* net) : MVZKExec(), net_manager(net) {
        if (!net) {
            LOG_ERROR("Network Manager cannot be null!");
            exit(-1);
        }

        this->party = net->my_party_id;
        PolyTensor::set_default_store(net->my_party_id);
    }

    virtual ~MVZKNetBase() {
        if (owns_net_manager && net_manager) {
            delete net_manager;
            net_manager = nullptr;
        }
    }

    void sync_network() override {
        if (net_manager) {
            net_manager->sync_network();
        } else {
            LOG_ERROR("Cannot sync network: net_manager is null!");
        }
    }

    emp::block coin_tossing() override {
        if (net_manager == nullptr || net_manager->broadcast_ios == nullptr || net_manager->num_parties < 2) {
            return emp::zero_block;
        }

        int my_id = net_manager->my_party_id;
        int num_parties = net_manager->num_parties;
        auto b_ios = net_manager->broadcast_ios;

        emp::block my_rand;
        this->prg.random_block(&my_rand, 1);

        std::vector<char> my_commit(emp::Hash::DIGEST_SIZE);
        emp::Hash hash;
        hash.put(&my_rand, sizeof(emp::block));
        hash.digest(my_commit.data());

        std::vector<std::vector<char>> all_commits(num_parties + 1, std::vector<char>(emp::Hash::DIGEST_SIZE));
        std::vector<emp::block> all_reveals(num_parties + 1);

        memcpy(all_commits[my_id].data(), my_commit.data(), emp::Hash::DIGEST_SIZE);
        all_reveals[my_id] = my_rand;

        for (int turn = 1; turn <= num_parties; ++turn) {
            if (turn == my_id) {
                for (int i = 1; i <= num_parties; ++i) {
                    if (i == my_id || b_ios[i] == nullptr) continue;
                    b_ios[i]->send_data(my_commit.data(), emp::Hash::DIGEST_SIZE);
                    b_ios[i]->flush();
                }
            } else {
                if (b_ios[turn] != nullptr) {
                    b_ios[turn]->recv_data(all_commits[turn].data(), emp::Hash::DIGEST_SIZE);
                }
            }
        }

        for (int turn = 1; turn <= num_parties; ++turn) {
            if (turn == my_id) {
                for (int i = 1; i <= num_parties; ++i) {
                    if (i == my_id || b_ios[i] == nullptr) continue;
                    b_ios[i]->send_data(&my_rand, sizeof(emp::block));
                    b_ios[i]->flush();
                }
            } else {
                if (b_ios[turn] != nullptr) {
                    b_ios[turn]->recv_data(&all_reveals[turn], sizeof(emp::block));
                }
            }
        }

        emp::block final_coin = my_rand;
        for (int i = 1; i <= num_parties; ++i) {
            if (i == my_id || b_ios[i] == nullptr) continue;

            std::vector<char> check_commit(emp::Hash::DIGEST_SIZE);
            emp::Hash check_hash;
            check_hash.put(&all_reveals[i], sizeof(emp::block));
            check_hash.digest(check_commit.data());

            if (memcmp(check_commit.data(), all_commits[i].data(), emp::Hash::DIGEST_SIZE) != 0) {
                LOG_ERROR("Inconsistency detected in coin tossing from Party " << i << "!");
                if (MVZK_CONFIG_INSTANT_ABORT) exit(-1);
            }

            final_coin = final_coin ^ all_reveals[i];
        }

        return final_coin;
    }

    uint64_t get_comm_bytes_counter() const {
        if (net_manager) {
            return net_manager->get_comm_bytes_counter();
        }
        return 0;
    }
};

#endif // MVZK_NET_BASE_H__
