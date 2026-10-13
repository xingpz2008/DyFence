#include "vgg11.h"
#include <iostream>

#include "../operations/linear.h"
#include "../operations/nonlinear.h"

PolyTensor VGG11_Forward(
    PolyTensor& image,
    VGG11Weights& net,
    uint64_t bitlen, uint64_t digdec_k, bool do_truncation, bool using_integrated_nl)
{
    std::cout << "\n=================================================" << std::endl;
    std::cout << "[Model Inference] Starting VGG-11 ZK Inference" << std::endl;
    std::cout << "=================================================" << std::endl;

    PolyTensor x = image.clone();
    image.mark_consumed();

    // Block 1 (1x Conv)

    std::cout << "[Model Inference] Executing Block 1..." << std::endl;
    x = Conv2D(x, net.conv1_1_w, net.conv1_1_b, 1, 1);
    if (using_integrated_nl) {
        x = IntegratedNL(x, 2, 2, 0, bitlen, digdec_k, do_truncation);
    } else {
        x = ReLU(x, bitlen, digdec_k, do_truncation);
        x = MaxPool2D(x, 2, 2, 0, bitlen, digdec_k);
    }

    // Block 2 (1x Conv)

    std::cout << "[Model Inference] Executing Block 2..." << std::endl;
    x = Conv2D(x, net.conv2_1_w, net.conv2_1_b, 1, 1);
    if (using_integrated_nl) {
        x = IntegratedNL(x, 2, 2, 0, bitlen, digdec_k, do_truncation);
    } else {
        x = ReLU(x, bitlen, digdec_k, do_truncation);
        x = MaxPool2D(x, 2, 2, 0, bitlen, digdec_k);
    }

    // Block 3 (2x Conv)

    std::cout << "[Model Inference] Executing Block 3..." << std::endl;
    x = Conv2D(x, net.conv3_1_w, net.conv3_1_b, 1, 1);
    x = ReLU(x, bitlen, digdec_k, do_truncation);

    x = Conv2D(x, net.conv3_2_w, net.conv3_2_b, 1, 1);
    if (using_integrated_nl) {
        x = IntegratedNL(x, 2, 2, 0, bitlen, digdec_k, do_truncation);
    } else {
        x = ReLU(x, bitlen, digdec_k, do_truncation);
        x = MaxPool2D(x, 2, 2, 0, bitlen, digdec_k);
    }

    // Block 4 (2x Conv)

    std::cout << "[Model Inference] Executing Block 4..." << std::endl;
    x = Conv2D(x, net.conv4_1_w, net.conv4_1_b, 1, 1);
    x = ReLU(x, bitlen, digdec_k, do_truncation);

    x = Conv2D(x, net.conv4_2_w, net.conv4_2_b, 1, 1);
    if (using_integrated_nl) {
        x = IntegratedNL(x, 2, 2, 0, bitlen, digdec_k, do_truncation);
    } else {
        x = ReLU(x, bitlen, digdec_k, do_truncation);
        x = MaxPool2D(x, 2, 2, 0, bitlen, digdec_k);
    }

    // Block 5 (2x Conv)

    std::cout << "[Model Inference] Executing Block 5..." << std::endl;
    x = Conv2D(x, net.conv5_1_w, net.conv5_1_b, 1, 1);
    x = ReLU(x, bitlen, digdec_k, do_truncation);

    x = Conv2D(x, net.conv5_2_w, net.conv5_2_b, 1, 1);
    if (using_integrated_nl) {
        x = IntegratedNL(x, 2, 2, 0, bitlen, digdec_k, do_truncation);
    } else {
        x = ReLU(x, bitlen, digdec_k, do_truncation);
        x = MaxPool2D(x, 2, 2, 0, bitlen, digdec_k);
    }

    // Classifier (FC Layers)

    std::cout << "[Model Inference] Executing Classifier (FC Layers)..." << std::endl;

    x = x.flatten();

    x = Linear(x, net.fc1_w, net.fc1_b);
    x = ReLU(x, bitlen, digdec_k, do_truncation);

    x = Linear(x, net.fc2_w, net.fc2_b);
    x = ReLU(x, bitlen, digdec_k, do_truncation);

    x = Linear(x, net.fc3_w, net.fc3_b);

    std::cout << "=================================================" << std::endl;
    std::cout << "\033[32m[SUCCESS] VGG-11 ZK Inference Complete!\033[0m" << std::endl;
    std::cout << "=================================================\n" << std::endl;

    return x;
}