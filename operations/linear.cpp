#include "linear.h"
#include <cmath>
#include "../exec/MVZKExec.h"

PolyTensor Conv2D(
    const PolyTensor& input,
    const PolyTensor& weight,
    const PolyTensor& bias,
    int stride,
    int padding,
    int dilation
) {
    if (!MVZKExec::mvzk_exec) {
        LOG_ERROR(" MVZKExec not initialized!");
        exit(-1);
    }

    if (input.shape.size() != 4 || weight.shape.size() != 4) {
        LOG_ERROR("Conv2D expects 4D input and weight!");
        exit(-1);
    }

    return MVZKExec::mvzk_exec->conv2d(input, weight, bias, stride, padding, dilation);
}

PolyTensor Linear(
    const PolyTensor& input,
    const PolyTensor& weight,
    const PolyTensor& bias
) {
    if (!MVZKExec::mvzk_exec) {
        LOG_ERROR(" MVZKExec not initialized!");
        exit(-1);
    }
    return MVZKExec::mvzk_exec->linear(input, weight, bias);
}

PolyTensor Conv1D(
    const PolyTensor& input,
    const PolyTensor& weight,
    const PolyTensor& bias,
    int stride, int padding, int dilation
) {
    if (!MVZKExec::mvzk_exec) {
        LOG_ERROR(" MVZKExec not initialized!");
        exit(-1);
    }
    if (input.shape.size() != 3 || weight.shape.size() != 3) {
        LOG_ERROR("Conv1D expects 3D input (N, C, L) and weight (Co, Ci, K)!");
        exit(-1);
    }
    return MVZKExec::mvzk_exec->conv1d(input, weight, bias, stride, padding, dilation);
}

PolyTensor AvgPool2D(PolyTensor& x, int kernel_size, int stride, int padding, bool back_to_sum_pool) {
    if (!MVZKExec::mvzk_exec) {
        LOG_ERROR(" MVZKExec not initialized!");
        exit(-1);
    }

    if (stride == -1) {
        stride = kernel_size;
    }
    return MVZKExec::mvzk_exec->avgpool2d(x, kernel_size, stride, padding, back_to_sum_pool);
}

[[deprecated("WARNING: DO NOT USE Standalone BatchNorm2D. Please use offline Conv-BN folding instead.")]]
PolyTensor BatchNorm2D(
    PolyTensor& x,
    const std::vector<double>& weight,
    const std::vector<double>& bias,
    const std::vector<double>& running_mean,
    const std::vector<double>& running_var,
    double eps)
{
    RED("[CRITICAL WARNING] DO NOT USE Standalone BatchNorm2D. Please use offline Conv-BN folding instead.");
    if (!MVZKExec::mvzk_exec) {
        LOG_ERROR(" MVZKExec not initialized!");
        exit(-1);
    }

    int C = weight.size();
    if (C != x.shape[1]) {
        LOG_ERROR("BatchNorm2D channel mismatch!");
        exit(-1);
    }

    std::vector<double> A_float(C);
    std::vector<double> B_float(C);
    for (int c = 0; c < C; ++c) {
        A_float[c] = weight[c] / std::sqrt(running_var[c] + eps);
        B_float[c] = bias[c] - running_mean[c] * A_float[c];
    }

    std::vector<uint64_t> A_quantized(C);
    std::vector<uint64_t> B_quantized(C);
    for (int c = 0; c < C; ++c) {
        A_quantized[c] = real2fp(A_float[c]);
        B_quantized[c] = real2fp(B_float[c]);
    }

    return MVZKExec::mvzk_exec->batchnorm2d(x, A_quantized, B_quantized);
}