#pragma once

#include <vector>
#include "../data_type/PolyTensor.h"
#include "../config.h"

struct VGG16Weights {
    // Block 1 (2x Conv)
    PolyTensor conv1_1_w, conv1_1_b;
    PolyTensor conv1_2_w, conv1_2_b;

    // Block 2 (2x Conv)
    PolyTensor conv2_1_w, conv2_1_b;
    PolyTensor conv2_2_w, conv2_2_b;

    // Block 3 (3x Conv)
    PolyTensor conv3_1_w, conv3_1_b;
    PolyTensor conv3_2_w, conv3_2_b;
    PolyTensor conv3_3_w, conv3_3_b;

    // Block 4 (3x Conv)
    PolyTensor conv4_1_w, conv4_1_b;
    PolyTensor conv4_2_w, conv4_2_b;
    PolyTensor conv4_3_w, conv4_3_b;

    // Block 5 (3x Conv)
    PolyTensor conv5_1_w, conv5_1_b;
    PolyTensor conv5_2_w, conv5_2_b;
    PolyTensor conv5_3_w, conv5_3_b;

    // Classifier (3x FC)
    PolyTensor fc1_w, fc1_b;
    PolyTensor fc2_w, fc2_b;
    PolyTensor fc3_w, fc3_b;
};

PolyTensor VGG16_Forward(
    PolyTensor& image,
    VGG16Weights& net,
    uint64_t bitlen,
    uint64_t digdec_k,
    bool do_truncation,
    bool using_integrated_nl = MVZK_CONFIG_MODEL_USING_INTEGRATED_NON_LINEAR
);