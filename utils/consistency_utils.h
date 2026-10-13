#pragma once

#include <vector>
#include <string>
#include <filesystem>
#include <iostream>
#include "../data_type/PolyTensor.h"
#include "emp-tool/emp-tool.h"
#include "logger.h"

inline void save_tensors_1to1(const std::vector<PolyTensor*>& data, int party, int verifier_idx, const std::string& checkpoints_dir) {
    std::string sub_dir = checkpoints_dir + (party == emp::ALICE ?
                          "/Prover_with_Verifier_" + std::to_string(verifier_idx) :
                          "/Verifier_" + std::to_string(verifier_idx) + "_with_Prover");

    std::filesystem::create_directories(sub_dir);

    for (size_t i = 0; i < data.size(); ++i) {
        std::string filepath = sub_dir + "/tensor_" + std::to_string(i) + ".bin";
        data[i]->save_to_file(filepath);
    }
    LOG_INFO("Successfully dumped " << data.size() << " tensors to " << sub_dir);
}

inline void load_tensors_1toN(std::vector<std::vector<PolyTensor*>>& data, int party, int num_verifiers, const std::string& checkpoints_dir) {
    int outer_size = (party == emp::ALICE) ? num_verifiers : 1;

    for (int j = 0; j < outer_size; ++j) {
        int verifier_idx = (party == emp::ALICE) ? (j + 1) : (party - 1);

        std::string sub_dir = checkpoints_dir + (party == emp::ALICE ?
                              "/Prover_with_Verifier_" + std::to_string(verifier_idx) :
                              "/Verifier_" + std::to_string(verifier_idx) + "_with_Prover");

        for (size_t i = 0; i < data[j].size(); ++i) {
            std::string filepath = sub_dir + "/tensor_" + std::to_string(i) + ".bin";
            data[j][i]->load_from_file(filepath);
        }
    }
    LOG_INFO("Successfully loaded tensor matrix from sub-directories.");
}
