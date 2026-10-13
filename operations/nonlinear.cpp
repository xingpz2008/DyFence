#include "nonlinear.h"
#include "../exec/MVZKExec.h"

PolyTensor ReLU(PolyTensor& x, uint64_t bitlen, uint64_t digdec_k, bool do_truncation, uint64_t scale){
    if (!MVZKExec::mvzk_exec) {
        LOG_ERROR("MVZKExec not initialized!");
        exit(-1);
    }
    return MVZKExec::mvzk_exec->relu(x, bitlen, digdec_k, do_truncation, scale);
}

PolyTensor MaxPool2D(PolyTensor& x, int kernel_size, int stride, int padding, uint64_t bitlen, uint64_t digdec_k, uint64_t scale){
    if (!MVZKExec::mvzk_exec) {
        LOG_ERROR("MVZKExec not initialized!");
        exit(-1);
    }
    return MVZKExec::mvzk_exec->maxpool2d(x, kernel_size, stride, padding, bitlen, digdec_k, scale);
}

PolyTensor IntegratedNL(PolyTensor& x, int kernel_size, int stride, int padding, uint64_t bitlen, uint64_t digdec_k,bool do_truncation, uint64_t scale){
    if (!MVZKExec::mvzk_exec) {
        RED("[ERROR] MVZKExec not initialized!");
        exit(-1);
    }
    
    return MVZKExec::mvzk_exec->integrated_nl(x, kernel_size, stride, padding, bitlen, digdec_k, do_truncation, scale);
}