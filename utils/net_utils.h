#pragma once
#include <vector>
#include <string>
#include "emp-tool/emp-tool.h"
#include "../config.h"
#include "logger.h"

// Comm Data
// NOTE: Legacy func!
template <typename IO>
inline uint64_t comm(IO** io_arr, int num){
    uint64_t c = 0;
    for (int i = 0; i < num; i++){
        c += io_arr[i]->counter;
    }
    return c;
}

// NOTE: Legacy func!
inline uint64_t comm(emp::NetIO** io_arr, int num){
    uint64_t c = 0;
    for (int i = 0; i < num; i++){
        c += io_arr[i]->counter;
    }
    return c;
}

inline bool checkSysSettings(int test_mode = 1){
    // Test mode 1 -> 1-on-1 ZK Proof
    // Test mode 2 -> 1-on-N MP Proof

    if (test_mode == 1){
        if (MVZK_CONFIG_SCALE >= MVZK_CONFIG_DEFAULT_BITLEN){
            LOG_ERROR("Bitlen must greater than scale. Now :(MVZK_CONFIG_SCALE >= MVZK_CONFIG_DEFAULT_BITLEN)");
            return false;
        }
        if (MVZK_CONFIG_NN_PUBLIC_INPUT && MVZK_CONFIG_NN_PUBLIC_WEIGHT){
            LOG_ERROR("Input and weight CANNOT be public simutanouely. Now: (MVZK_CONFIG_NN_PUBLIC_INPUT && MVZK_CONFIG_NN_PUBLIC_WEIGHT)");
            LOG_ERROR("Hint: For plaintext accuracy check, run the binary with mode 4.");
            return false;
        }
        if (MVZK_CONFIG_NUM_PARTIES != 2){
            LOG_ERROR("ZK Framework only supports 2 parties. This is NOT a consistency check code that supports multiple parties! Now: (MVZK_CONFIG_NUM_PARTIES != 2)");
            return false;
        }
    } else if (test_mode == 2) {
        if (MVZK_CONFIG_THREADS_NUM != 1){
            LOG_ERROR("This is a reserved variable. Currently only recommend 1. Now: (MVZK_CONFIG_THREADS_NUM != 1)");
            return false;
        }
        if (MVZK_CONFIG_NUM_PARTIES <= 2){
            LOG_WARN("Consistency check must be conducted under multiple (>2) parties, please ENSURE cli_total_parties > 2! Now: (MVZK_CONFIG_NUM_PARTIES <= 2)");
        }
        if (MVZK_CONFIG_BROADCAST_ASSUME_IDEAL){
            LOG_WARN("An ideal broadcast channel is assumed. Boradcast consistency is naturally established and NO broadcast verification will be performed! Now: (MVZK_CONFIG_BROADCAST_ASSUME_IDEAL = true)");
        }
    }

    return true;
}