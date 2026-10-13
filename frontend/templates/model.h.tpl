#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include "data_type/PolyTensor.h"

// ==========================================
// Phase 1: Offline Host Weights (RAM Storage)
// ==========================================
struct {model_name}HostWeights {{
{host_fields}
}};

// ==========================================
// Phase 2: Online PolyTensor Weights (ZK Engine)
// ==========================================
struct {model_name}PolyWeights {{
{poly_fields}
}};

// ==========================================
// Phase 3: ZK Consistency Check Helper
// ==========================================
inline std::vector<PolyTensor*> get_all_weight_tensors({model_name}PolyWeights& weights) {{
    std::vector<PolyTensor*> vec;
{get_weights_cpp}
    return vec;
}}

// ==========================================
// Function Declarations
// ==========================================
{model_name}HostWeights load_{model_name_lower}_host_weights(int party, const std::string& bin_dir);
{model_name}PolyWeights inject_{model_name_lower}_weights_to_zk(MVZKExec* exec, const {model_name}HostWeights& hw);
PolyTensor {model_name}_Forward(
    PolyTensor& {input_name},
    {model_name}PolyWeights& poly_weights,
    uint64_t bitlen,
    uint64_t digdec_k,
    bool do_truncation
);
