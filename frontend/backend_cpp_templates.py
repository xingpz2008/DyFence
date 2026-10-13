"""
================================================================================
ZK-ML Compiler Configuration & Templates
================================================================================
This file contains the Operator Registry mapping and the static C++ templates.
Modify this file if the underlying C++ MPC framework's API changes.
================================================================================
"""

# MODULE 1: OPERATOR REGISTRY (CPP API Mapping)

CPP_API_MAP = {
    "conv2d": "PolyTensor {out_var} = Conv2D({in_var}, poly_weights.{node_name}_w, poly_weights.{node_name}_b, {stride}, {padding});",
    "linear": "PolyTensor {out_var} = Linear({in_var}, poly_weights.{node_name}_w, poly_weights.{node_name}_b);",
    "relu": "PolyTensor {out_var} = ReLU({in_var}, bitlen, digdec_k, do_truncation);",
    "integrated_nl": "PolyTensor {out_var} = IntegratedNL({in_var}, {kernel_size}, {stride}, {padding}, bitlen, digdec_k, do_truncation);",
    "max_pool2d": "PolyTensor {out_var} = MaxPool2D({in_var}, {kernel_size}, {stride}, {padding}, bitlen, digdec_k);",
    "avgpool2d": "PolyTensor {out_var} = AvgPool2D({in_var}, {kernel_size}, {stride}, {padding}, false);",
    "global_avg_pool2d": "PolyTensor {out_var} = AvgPool2D({in_var}, {in_var}.shape[2], 1, 0, false);",
    "global_sum_pool2d": "PolyTensor {out_var} = AvgPool2D({in_var}, {in_var}.shape[2], 1, 0, true);",
    "func_add": "PolyTensor {out_var} = {in_var_0} + {in_var_1};",
    "func_sub": "PolyTensor {out_var} = {in_var_0} - {in_var_1};",
    "func_mul": "PolyTensor {out_var} = {in_var_0} * {in_var_1};",
    "flatten": "PolyTensor {out_var} = {in_var}.flatten();",
    "module_identity": "PolyTensor {out_var} = {in_var}.clone();\n    {in_var}.mark_consumed();",
    "module_dropout": "PolyTensor {out_var} = {in_var}.clone();\n    {in_var}.mark_consumed();",
    "unknown": "// [WARNING] Unmapped operation: {node_type} for {out_var}"
}

# MODULE 2: STATIC TEMPLATES

from pathlib import Path

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

def _load_template(filename):
    return (_TEMPLATE_DIR / filename).read_text(encoding="utf-8")

TEMPLATE_HEADER = _load_template("model.h.tpl")
TEMPLATE_CPP = _load_template("model.cpp.tpl")
TEMPLATE_TEST = _load_template("model_test.cpp.tpl")
