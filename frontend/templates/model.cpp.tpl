#include "{model_name}.h"
#include "data_type/tensor_loader.h"
#include "operations/linear.h"
#include "operations/nonlinear.h"
#include "exec/MVZKExec.h"
#include <iostream>
#include <utility>

{model_name}HostWeights load_{model_name_lower}_host_weights(int party, const std::string& bin_dir) {{
    {model_name}HostWeights hw;
{load_statements}
    return hw;
}}

{model_name}PolyWeights inject_{model_name_lower}_weights_to_zk(MVZKExec* exec, const {model_name}HostWeights& hw) {{
    {model_name}PolyWeights pw;
{inject_statements}
    return pw;
}}

PolyTensor {model_name}_Forward(PolyTensor& {input_name}, {model_name}PolyWeights& poly_weights, uint64_t bitlen, uint64_t digdec_k, bool do_truncation) {{
{forward_statements}
    return std::move({output_name});
}}
