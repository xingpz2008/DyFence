#include "resnet50.h"
#include <iostream>

#include "../operations/linear.h"
#include "../operations/nonlinear.h"

PolyTensor Bottleneck_Forward(
    PolyTensor& x,
    BottleneckWeights& block,
    int stride,
    uint64_t bitlen, uint64_t digdec_k, bool do_truncation)
{
    PolyTensor identity = x.clone();

    PolyTensor out = Conv2D(x, block.conv1_w, block.conv1_b, 1, 0);
    out = ReLU(out, bitlen, digdec_k, do_truncation);

    out = Conv2D(out, block.conv2_w, block.conv2_b, stride, 1);
    out = ReLU(out, bitlen, digdec_k, do_truncation);

    out = Conv2D(out, block.conv3_w, block.conv3_b, 1, 0);

    if (block.has_downsample) {
        identity = Conv2D(x, block.downsample_w, block.downsample_b, stride, 0);
    }else{
        identity = identity * real2fp(1.0);
    }

    out = out + identity;
    out = ReLU(out, bitlen, digdec_k, do_truncation);

    return out;
}

PolyTensor ResNet50_Forward(
    PolyTensor& image,
    ResNet50Weights& net,
    uint64_t bitlen, uint64_t digdec_k, bool do_truncation, bool using_integrated_nl)
{
    std::cout << "\n=================================================" << std::endl;
    std::cout << "[Model Inference] Starting ResNet-50 ZK Inference" << std::endl;
    std::cout << "=================================================" << std::endl;

    std::cout << "[Model Inference] Executing Stem Layer..." << std::endl;
    PolyTensor x = Conv2D(image, net.conv1_w, net.conv1_b, 2, 3);

    if (using_integrated_nl){
        x = IntegratedNL(x, 3, 2, 1, bitlen, digdec_k, do_truncation);
    } else {
        x = ReLU(x, bitlen, digdec_k, do_truncation);
        x = MaxPool2D(x, 3, 2, 1, bitlen, digdec_k);
    }

    std::cout << "[Model Inference] Executing Layer 1 (3 Blocks)..." << std::endl;
    for (int i = 0; i < 3; ++i) {
        x = Bottleneck_Forward(x, net.layer1[i], 1, bitlen, digdec_k, do_truncation);
    }

    std::cout << "[Model Inference] Executing Layer 2 (4 Blocks)..." << std::endl;
    for (int i = 0; i < 4; ++i) {
        int stride = (i == 0) ? 2 : 1;
        x = Bottleneck_Forward(x, net.layer2[i], stride, bitlen, digdec_k, do_truncation);
    }

    std::cout << "[Model Inference] Executing Layer 3 (6 Blocks)..." << std::endl;
    for (int i = 0; i < 6; ++i) {
        int stride = (i == 0) ? 2 : 1;
        x = Bottleneck_Forward(x, net.layer3[i], stride, bitlen, digdec_k, do_truncation);
    }

    std::cout << "[Model Inference] Executing Layer 4 (3 Blocks)..." << std::endl;
    for (int i = 0; i < 3; ++i) {
        int stride = (i == 0) ? 2 : 1;
        x = Bottleneck_Forward(x, net.layer4[i], stride, bitlen, digdec_k, do_truncation);
    }

    std::cout << "[Model Inference] Executing Global Avg Pooling & FC Layer..." << std::endl;
    x = AvgPool2D(x, 7, 1, 0);
    //x = Conv2D(x, net.fc_w, net.fc_b, 1, 0);
    x = Linear(x, net.fc_w, net.fc_b);

    std::cout << "=================================================" << std::endl;
    std::cout << "\033[32m[SUCCESS] ResNet-50 ZK Inference Complete!\033[0m" << std::endl;
    std::cout << "=================================================\n" << std::endl;

    return x;
}