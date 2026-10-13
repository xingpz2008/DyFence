#pragma once

#include "exec/ExecProver.h"
#include "data_type/PolyTensor.h"

template <typename IO>
class MVZKExecPlaintext : public MVZKExecProver<IO> {
public:
    // Initialize as Prover (ALICE) with dummy network parameters.
    // No actual network connection is required.
    MVZKExecPlaintext(int party_id)
        : MVZKExecProver<IO>(ALICE, -1, {}, -1, 1)
    {
        // Force identity to ALICE to trigger real plaintext math operations
        this->party = ALICE;
        MVZKExec::mvzk_exec = this;
    }

    ~MVZKExecPlaintext() = default;

    // 1. I/O Interception: Force degree-0 plaintext

    PolyTensor input(const std::vector<int>& shape, const std::vector<uint64_t>& raw_data) override {
        // Convert all secret inputs to degree-0 public tensors
        PolyTensor res = PolyTensor::from_public(shape, raw_data);
        res.mark_consumed();
        return res;
    }

    std::vector<uint64_t> reveal(const PolyTensor& pt) override {
        // Safely extract the real values from the 0-degree tensor
        return pt.get_real_vals_vector();
    }

    // 2. Bypass Constraint Generation

    // Intercept LUT and range checks to prevent buffering and memory overhead
    void submit_lut_check(const PolyTensor& keys, const PolyTensor& vals, int table_id) override {}
    void submit_range_check(const PolyTensor& x, int table_id) override {}

    void submit_to_buffer(PolyDelta&& pd) override {}
    void submit_tensor_to_buffer(PolyTensor&& pt) override {}

    // 3. Bypass Network & Verification

    // Disable all cryptographic checks and network synchronizations
    void check_all() override {}
    void consistency_check(const std::vector<std::vector<PolyTensor*>>& data) override {}
    void sync_network() override {}
    void finalize_protocol() override {}
    uint64_t broadcast(uint64_t val = 0, bool assume_ideal_broadcast = true) override { return val; }
};