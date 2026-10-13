#pragma once

#include <vector>

#include "../data_type/PolyTensor.h"
#include "../config.h"

struct BottleneckWeights {
    static const int expansion = 4;

    PolyTensor conv1_w, conv1_b;
    PolyTensor conv2_w, conv2_b;
    PolyTensor conv3_w, conv3_b;

    bool has_downsample;
    PolyTensor downsample_w, downsample_b;
};

struct ResNet50Weights {
    PolyTensor conv1_w, conv1_b;

    std::vector<BottleneckWeights> layer1;
    std::vector<BottleneckWeights> layer2;
    std::vector<BottleneckWeights> layer3;
    std::vector<BottleneckWeights> layer4;

    PolyTensor fc_w, fc_b;
};

PolyTensor Bottleneck_Forward(
    PolyTensor& x,
    BottleneckWeights& block,
    int stride,
    uint64_t bitlen,
    uint64_t digdec_k,
    bool do_truncation
);

PolyTensor ResNet50_Forward(
    PolyTensor& image,
    ResNet50Weights& net,
    uint64_t bitlen,
    uint64_t digdec_k,
    bool do_truncation,
    bool using_integrated_nl = MVZK_CONFIG_MODEL_USING_INTEGRATED_NON_LINEAR
);