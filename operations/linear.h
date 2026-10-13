#ifndef MVZK_LINEAR_H__
#define MVZK_LINEAR_H__

#include "../data_type/PolyTensor.h"

/**
 * @brief 2D convolution interface similar to PyTorch Conv2d.
 *
 * @param input Input tensor with shape (N, C_in, H_in, W_in).
 * @param weight Weight tensor with shape (C_out, C_in, kH, kW).
 * @param bias Bias tensor with shape (C_out), or an empty tensor.
 * @param stride Stride, default 1.
 * @param padding Padding, default 0.
 * @param dilation Dilation parameter, currently reserved.
 * @return Output tensor with shape (N, C_out, H_out, W_out).
 */
PolyTensor Conv2D(
    const PolyTensor& input,
    const PolyTensor& weight,
    const PolyTensor& bias,
    int stride = 1,
    int padding = 0,
    int dilation = 1
);

// Fully connected layer.
PolyTensor Linear(
    const PolyTensor& input,
    const PolyTensor& weight,
    const PolyTensor& bias
);

/**
 * @brief 1D convolution interface similar to PyTorch Conv1d.
 * @param input   Shape: (N, C_in, L_in)
 * @param weight  Shape: (C_out, C_in, K)
 * @param bias    Shape: (C_out)
 */
PolyTensor Conv1D(
    const PolyTensor& input,
    const PolyTensor& weight,
    const PolyTensor& bias,
    int stride = 1,
    int padding = 0,
    int dilation = 1
);

PolyTensor AvgPool2D(PolyTensor& x,
    int kernel_size,
    int stride = -1,
    int padding = 0,
    bool back_to_sum_pool = false);

// last 4 parameters are public
[[deprecated("WARNING: DO NOT USE Standalone BatchNorm2D. Please use offline Conv-BN folding instead.")]]
PolyTensor BatchNorm2D(
    PolyTensor& x,
    const std::vector<double>& weight,
    const std::vector<double>& bias,
    const std::vector<double>& running_mean,
    const std::vector<double>& running_var,
    double eps = 1e-5
);

#endif
