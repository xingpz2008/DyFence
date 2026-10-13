#ifndef MVZK_BACKEND_VERIFIER_IMPL_ARITH_HPP__
#define MVZK_BACKEND_VERIFIER_IMPL_ARITH_HPP__

#include "../../exec/ExecVerifier.h"

template <typename IO>
PolyDelta MVZKExecVerifier<IO>::add(const PolyDelta& lhs, const PolyDelta& rhs) {
    PolyDelta res;
    res.degree = std::max(lhs.degree, rhs.degree);

    size_t offset_lhs = res.degree - lhs.degree;
    size_t offset_rhs = res.degree - rhs.degree;

    uint64_t k1 = lhs.key;
    if (offset_lhs > 0) {
        k1 = mult_mod(k1, delta_pow(this->delta, offset_lhs));
    }

    uint64_t k2 = rhs.key;
    if (offset_rhs > 0) {
        k2 = mult_mod(k2, delta_pow(this->delta, offset_rhs));
    }

    res.key = add_mod(k1, k2);

    res.is_consumed = false;
    lhs.is_consumed = true;
    rhs.is_consumed = true;
    return res;
}


template <typename IO>
PolyTensor MVZKExecVerifier<IO>::add(const PolyTensor& lhs, const PolyTensor& rhs) {
    if (lhs.total_elements != rhs.total_elements) {
        LOG_ERROR("Tensor shape mismatch in verifier add!");
        exit(-1);
    }

    int res_degree = std::max(lhs.degree, rhs.degree);
    PolyTensor res(lhs.shape, res_degree, PolyTensor::Keys);

    size_t size = lhs.total_elements;

    const uint64_t* lhs_keys = lhs.get_keys_ptr();
    const uint64_t* rhs_keys = rhs.get_keys_ptr();
    uint64_t* res_keys = res.get_keys_ptr();

    uint64_t local_delta = (uint64_t)this->delta;

    int offset_lhs = res_degree - lhs.degree;
    int offset_rhs = res_degree - rhs.degree;

    if (offset_lhs == 0 && offset_rhs == 0) {
        // K_res = K_lhs + K_rhs

        #pragma omp parallel for if(size>= MVZK_OMP_SIZE_THRESHOLD  && !omp_in_parallel())
        for (size_t i = 0; i < size; ++i) {
            res_keys[i] = add_mod(lhs_keys[i], rhs_keys[i]);
        }
    }
    else if (offset_lhs > 0 && offset_rhs == 0) {
        // K_res = K_lhs * delta^diff + K_rhs
        uint64_t scale = delta_pow(local_delta, offset_lhs);

        #pragma omp parallel for if(size>= MVZK_OMP_SIZE_THRESHOLD  && !omp_in_parallel())
        for (size_t i = 0; i < size; ++i) {
            uint64_t term_lhs = mult_mod(lhs_keys[i], scale);
            res_keys[i] = add_mod(term_lhs, rhs_keys[i]);
        }
    }
    else if (offset_lhs == 0 && offset_rhs > 0) {
        // K_res = K_lhs + K_rhs * delta^diff
        uint64_t scale = delta_pow(local_delta, offset_rhs);

        #pragma omp parallel for if(size>= MVZK_OMP_SIZE_THRESHOLD  && !omp_in_parallel())
        for (size_t i = 0; i < size; ++i) {
            uint64_t term_rhs = mult_mod(rhs_keys[i], scale);
            res_keys[i] = add_mod(lhs_keys[i], term_rhs);
        }
    }
    else {
        uint64_t scale_lhs = delta_pow(local_delta, offset_lhs);
        uint64_t scale_rhs = delta_pow(local_delta, offset_rhs);

        #pragma omp parallel for if(size>= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
        for (size_t i = 0; i < size; ++i) {
            uint64_t term_lhs = mult_mod(lhs_keys[i], scale_lhs);
            uint64_t term_rhs = mult_mod(rhs_keys[i], scale_rhs);
            res_keys[i] = add_mod(term_lhs, term_rhs);
        }
    }

    res.is_consumed = false;
    lhs.is_consumed = true;
    rhs.is_consumed = true;
    return res; // RVO
}


template <typename IO>
PolyDelta MVZKExecVerifier<IO>::sub(const PolyDelta& lhs, const PolyDelta& rhs) {
    PolyDelta res;
    res.degree = std::max(lhs.degree, rhs.degree);

    size_t offset_lhs = res.degree - lhs.degree;
    size_t offset_rhs = res.degree - rhs.degree;

    uint64_t k1 = lhs.key;
    if (offset_lhs > 0) {
        k1 = mult_mod(k1, delta_pow(this->delta, offset_lhs));
    }

    uint64_t k2 = rhs.key;
    if (offset_rhs > 0) {
        k2 = mult_mod(k2, delta_pow(this->delta, offset_rhs));
    }

    res.key = add_mod(k1, (PR - k2));

    res.is_consumed = false;
    lhs.is_consumed = true;
    rhs.is_consumed = true;
    return res;
}


template <typename IO>
PolyTensor MVZKExecVerifier<IO>::sub(const PolyTensor& lhs, const PolyTensor& rhs) {
    if (lhs.total_elements != rhs.total_elements) {
        LOG_ERROR("Tensor shape mismatch in verifier sub!");
        exit(-1);
    }

    int res_degree = std::max(lhs.degree, rhs.degree);
    PolyTensor res(lhs.shape, res_degree, PolyTensor::Keys);

    size_t size = lhs.total_elements;

    const uint64_t* lhs_keys = lhs.get_keys_ptr();
    const uint64_t* rhs_keys = rhs.get_keys_ptr();
    uint64_t* res_keys = res.get_keys_ptr();

    uint64_t local_delta = (uint64_t)this->delta;
    int offset_lhs = res_degree - lhs.degree;
    int offset_rhs = res_degree - rhs.degree;

    if (offset_lhs == 0 && offset_rhs == 0) {
        #pragma omp parallel for if(size>= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
        for (size_t i = 0; i < size; ++i) {
            res_keys[i] = add_mod(lhs_keys[i], PR - rhs_keys[i]);
        }
    }
    else if (offset_lhs > 0 && offset_rhs == 0) {
        uint64_t scale = delta_pow(local_delta, offset_lhs);

        #pragma omp parallel for if(size>= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
        for (size_t i = 0; i < size; ++i) {
            uint64_t term_lhs = mult_mod(lhs_keys[i], scale);
            res_keys[i] = add_mod(term_lhs, PR - rhs_keys[i]);
        }
    }
    else if (offset_lhs == 0 && offset_rhs > 0) {
        uint64_t scale = delta_pow(local_delta, offset_rhs);

        #pragma omp parallel for if(size>= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
        for (size_t i = 0; i < size; ++i) {
            uint64_t term_rhs = mult_mod(rhs_keys[i], scale);
            res_keys[i] = add_mod(lhs_keys[i], PR - term_rhs);
        }
    }
    else {
        uint64_t scale_lhs = delta_pow(local_delta, offset_lhs);
        uint64_t scale_rhs = delta_pow(local_delta, offset_rhs);

        #pragma omp parallel for if(size>= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
        for (size_t i = 0; i < size; ++i) {
            uint64_t term_lhs = mult_mod(lhs_keys[i], scale_lhs);
            uint64_t term_rhs = mult_mod(rhs_keys[i], scale_rhs);
            res_keys[i] = add_mod(term_lhs, PR - term_rhs);
        }
    }

    res.is_consumed = false;
    lhs.is_consumed = true;
    rhs.is_consumed = true;
    return res;
}


template <typename IO>
PolyDelta MVZKExecVerifier<IO>::mul(const PolyDelta& lhs, const PolyDelta& rhs) {
    // Verifer side multiplciation
    PolyDelta res;

    res.degree = lhs.degree + rhs.degree;

    res.key = mult_mod(lhs.key, rhs.key);

    if (res.key > PR) LOG_ERROR("keys! > PR (mul)");

    res.is_consumed = false;
    lhs.is_consumed = true;
    rhs.is_consumed = true;

    return res;
}


template <typename IO>
PolyTensor MVZKExecVerifier<IO>::mul(const PolyTensor& lhs, const PolyTensor& rhs) {
    if (lhs.total_elements != rhs.total_elements) {
        LOG_ERROR("Tensor shape mismatch in verifier mul!");
        exit(-1);
    }

    int res_degree = lhs.degree + rhs.degree;
    PolyTensor res(lhs.shape, res_degree, PolyTensor::Keys);

    size_t size = lhs.total_elements;

    const uint64_t* lhs_keys = lhs.get_keys_ptr();
    const uint64_t* rhs_keys = rhs.get_keys_ptr();
    uint64_t* res_keys = res.get_keys_ptr();

    #pragma omp parallel for if(size>= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
    for (size_t i = 0; i < size; ++i) {
        res_keys[i] = mult_mod(lhs_keys[i], rhs_keys[i]);
    }

    res.is_consumed = false;
    lhs.is_consumed = true;
    rhs.is_consumed = true;
    return res;
}


template <typename IO>
void MVZKExecVerifier<IO>::add_assign(PolyDelta& lhs, const PolyDelta& rhs) {
    if (rhs.degree > lhs.degree){
        size_t offset_diff = rhs.degree - lhs.degree;
        lhs.key = add_mod(rhs.key, mult_mod(lhs.key, delta_pow(this->delta, offset_diff)));
        lhs.degree = rhs.degree;
    }else{
        size_t offset_diff = lhs.degree - rhs.degree;
        lhs.key = add_mod(lhs.key, mult_mod(rhs.key, delta_pow(this->delta, offset_diff)));
    }

    lhs.is_consumed = false;
    rhs.is_consumed = true;
}


template <typename IO>
void MVZKExecVerifier<IO>::add_assign(PolyTensor& lhs, const PolyTensor& rhs) {
    if (lhs.total_elements != rhs.total_elements) {
        LOG_ERROR("Tensor shape mismatch in verifier add_assign!");
        exit(-1);
    }

    size_t size = lhs.total_elements;
    uint64_t* lhs_keys = lhs.get_keys_ptr();
    const uint64_t* rhs_keys = rhs.get_keys_ptr();

    uint64_t local_delta = (uint64_t)this->delta;

    if (rhs.degree > lhs.degree) {
        int offset = rhs.degree - lhs.degree;
        uint64_t scale = delta_pow(local_delta, offset);

        #pragma omp parallel for if(size>= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
        for (size_t i = 0; i < size; ++i) {
            uint64_t shifted_lhs = mult_mod(lhs_keys[i], scale);
            lhs_keys[i] = add_mod(shifted_lhs, rhs_keys[i]);
        }

        lhs.degree = rhs.degree;
    } else {
        int offset = lhs.degree - rhs.degree;

        if (offset == 0) {
            #pragma omp parallel for if(size>= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
            for (size_t i = 0; i < size; ++i) {
                lhs_keys[i] = add_mod(lhs_keys[i], rhs_keys[i]);
            }
        } else {
            uint64_t scale = delta_pow(local_delta, offset);
            #pragma omp parallel for if(size>= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
            for (size_t i = 0; i < size; ++i) {
                uint64_t shifted_rhs = mult_mod(rhs_keys[i], scale);
                lhs_keys[i] = add_mod(lhs_keys[i], shifted_rhs);
            }
        }
    }

    lhs.is_consumed = false;
    rhs.is_consumed = true;
}


template <typename IO>
void MVZKExecVerifier<IO>::sub_assign(PolyDelta& lhs, const PolyDelta& rhs) {
    if (rhs.degree > lhs.degree){
        size_t offset_diff = rhs.degree - lhs.degree;
        lhs.key = add_mod((PR - rhs.key), mult_mod(lhs.key, delta_pow(this->delta, offset_diff)));
        lhs.degree = rhs.degree;
    }else{
        size_t offset_diff = lhs.degree - rhs.degree;
        lhs.key = add_mod(lhs.key, (PR - mult_mod(rhs.key, delta_pow(this->delta, offset_diff))));
    }

    lhs.is_consumed = false;
    rhs.is_consumed = true;
}


template <typename IO>
void MVZKExecVerifier<IO>::sub_assign(PolyTensor& lhs, const PolyTensor& rhs) {
    if (lhs.total_elements != rhs.total_elements) {
        LOG_ERROR("Tensor shape mismatch in verifier add_assign!");
        exit(-1);
    }

    size_t size = lhs.total_elements;
    uint64_t* lhs_keys = lhs.get_keys_ptr();
    const uint64_t* rhs_keys = rhs.get_keys_ptr();

    uint64_t local_delta = (uint64_t)this->delta;

    if (rhs.degree > lhs.degree) {
        int offset = rhs.degree - lhs.degree;
        uint64_t scale = delta_pow(local_delta, offset);

        #pragma omp parallel for if(size>= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
        for (size_t i = 0; i < size; ++i) {
            uint64_t shifted_lhs = mult_mod(lhs_keys[i], scale);
            lhs_keys[i] = add_mod(shifted_lhs, PR - rhs_keys[i]);
        }

        lhs.degree = rhs.degree;
    } else {
        int offset = lhs.degree - rhs.degree;

        if (offset == 0) {
            #pragma omp parallel for if(size>= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
            for (size_t i = 0; i < size; ++i) {
                lhs_keys[i] = add_mod(lhs_keys[i], PR - rhs_keys[i]);
            }
        } else {
            uint64_t scale = delta_pow(local_delta, offset);
            #pragma omp parallel for if(size>= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
            for (size_t i = 0; i < size; ++i) {
                uint64_t shifted_rhs = mult_mod(rhs_keys[i], scale);
                lhs_keys[i] = add_mod(lhs_keys[i], PR - shifted_rhs);
            }
        }
    }

    lhs.is_consumed = false;
    rhs.is_consumed = true;
}


template <typename IO>
void MVZKExecVerifier<IO>::mul_assign(PolyDelta& lhs, uint64_t val) {
    lhs.key = mult_mod(lhs.key, val);

    if (lhs.key > PR) LOG_WARN("keys! > PR (mul assign)");

    lhs.is_consumed = false;
}


template <typename IO>
void MVZKExecVerifier<IO>::mul_assign(PolyTensor& lhs, uint64_t val) {
    size_t size = lhs.total_elements;
    uint64_t* keys_ptr = lhs.get_keys_ptr();

    #pragma omp parallel for if(size>= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
    for (size_t i = 0; i < size; ++i) {
        keys_ptr[i] = mult_mod(keys_ptr[i], val);
    }

    lhs.is_consumed = false;
}


template <typename IO>
void MVZKExecVerifier<IO>::mul_assign(PolyTensor& lhs, const PolyTensor& rhs) {
    if (lhs.total_elements != rhs.total_elements) {
        LOG_ERROR("Tensor shape mismatch in verifier mul_assign!");
        exit(-1);
    }

    size_t size = lhs.total_elements;
    uint64_t* lhs_keys = lhs.get_keys_ptr();
    const uint64_t* rhs_keys = rhs.get_keys_ptr();

    #pragma omp parallel for if(size>= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
    for (size_t i = 0; i < size; ++i) {
        lhs_keys[i] = mult_mod(lhs_keys[i], rhs_keys[i]);
    }

    lhs.degree += rhs.degree;

    lhs.is_consumed = false;
    rhs.is_consumed = true;
}


template <typename IO>
void MVZKExecVerifier<IO>::add_assign_const(PolyDelta& lhs, uint64_t val) {
    // K' = K + val * delta^deg
    uint64_t shift_val = mult_mod(val, delta_pow(this->delta, lhs.degree));
    lhs.key = add_mod(lhs.key, shift_val);

    lhs.is_consumed = false;
}


template <typename IO>
void MVZKExecVerifier<IO>::add_assign_const(PolyTensor& lhs, uint64_t val) {
    size_t size = lhs.total_elements;
    uint64_t* keys_ptr = lhs.get_keys_ptr();

    // Shift = val * Delta^degree

    uint64_t shift = mult_mod(val, delta_pow(this->delta, lhs.degree));

    // K_new = K_old + Shift
    #pragma omp parallel for if(size>= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
    for (size_t i = 0; i < size; ++i) {
        keys_ptr[i] = add_mod(keys_ptr[i], shift);
    }

    lhs.is_consumed = false;
}


template <typename IO>
void MVZKExecVerifier<IO>::sub_assign_const(PolyDelta& lhs, uint64_t val) {
    // K' = K - val * delta^deg
    uint64_t shift_val = mult_mod(val, delta_pow(this->delta, lhs.degree));

    uint64_t neg_shift_val = (shift_val == 0) ? 0 : (PR - shift_val);
    lhs.key = add_mod(lhs.key, neg_shift_val);

    lhs.is_consumed = false;
}


template <typename IO>
PolyDelta MVZKExecVerifier<IO>::add_const(const PolyDelta& lhs, uint64_t val) {
    // Verifier side: K' = K + c * delta ^n
    if (lhs.degree == 0){
        LOG_WARN("add_const called on degree 0 PolyDelta! Potential circuit design issue.");
    }

    PolyDelta res;

    res = lhs.clone();
    auto delta_powers_n = delta_pow(this->delta, lhs.degree);
    auto val_mult_delta_powers_n = mult_mod(val, delta_powers_n);
    res.key = add_mod(res.key, val_mult_delta_powers_n);

    res.is_consumed = false;
    lhs.is_consumed = true;

    return res;
}


template <typename IO>
PolyDelta MVZKExecVerifier<IO>::sub_const(const PolyDelta& lhs, uint64_t val) {
    // Verifier site: nearly identical to add const
    if (lhs.degree == 0){
        LOG_WARN("sub_const called on degree 0 PolyDelta! Potential circuit design issue.");
    }

    PolyDelta res;

    res = lhs.clone();
    auto delta_powers_n = delta_pow(this->delta, lhs.degree);
    auto val_mult_delta_powers_n = mult_mod(val, delta_powers_n);
    res.key = add_mod(res.key, (PR - val_mult_delta_powers_n));

    res.is_consumed = false;
    lhs.is_consumed = true;

    return res;
}


template <typename IO>
PolyDelta MVZKExecVerifier<IO>::sub_const_rev(uint64_t val, const PolyDelta& rhs) {
    // Verifier site, K'=-K+c*delta^n
    if (rhs.degree == 0){
        LOG_WARN("sub_const_rev called on degree 0 PolyDelta! Potential circuit design issue.");
    }

    PolyDelta res;

    res = rhs.clone();
    auto delta_powers_n = delta_pow(this->delta, rhs.degree);
    auto val_mult_delta_powers_n = mult_mod(val, delta_powers_n);
    res.key = add_mod((PR - res.key), val_mult_delta_powers_n);

    res.is_consumed = false;
    rhs.is_consumed = true;

    return res;
}


template <typename IO>
PolyDelta MVZKExecVerifier<IO>::mul_const(const PolyDelta& lhs, uint64_t val) {
    PolyDelta res;

    res = lhs.clone();
    res.key = mult_mod(res.key, val);

    if (res.key > PR) {
        LOG_WARN("key > PR! (mul_const)");
        exit(-1);
    }

    res.is_consumed = false;
    lhs.is_consumed = true;

    return res;
}


template <typename IO>
PolyTensor MVZKExecVerifier<IO>::MatMul(const PolyTensor& lhs, const PolyTensor& rhs) {
    // Matrix multiplication verifier 2d

    if (lhs.shape.size() != 2 || rhs.shape.size() != 2 || lhs.shape[1] != rhs.shape[0]) {
        LOG_ERROR("Invalid shape for matmul!");
        exit(-1);
    }
    int M = lhs.shape[0];
    int K = lhs.shape[1];
    int N = rhs.shape[1];

    return MatMul(lhs, rhs, M, K, N);
}

template <typename IO>
PolyTensor MVZKExecVerifier<IO>::MatMul(const PolyTensor& lhs, const PolyTensor& rhs, int M, int K, int N) {
    if ((size_t)M * (size_t)K != lhs.total_elements || rhs.shape.size() != 2 ||
        rhs.shape[0] != K || rhs.shape[1] != N) {
        LOG_ERROR("Invalid shape for matmul!");
        exit(-1);
    }

    int res_degree = lhs.degree + rhs.degree;
    PolyTensor res({M, N}, res_degree, PolyTensor::Keys);

    matrix_mul_acc_kernel(
        res.get_keys_ptr(),  // Dst Key Matrix
        lhs.get_keys_ptr(),  // LHS Key Matrix
        rhs.get_keys_ptr(),  // RHS Key Matrix
        M, K, N
    );

    res.is_consumed = false;
    lhs.is_consumed = true;
    rhs.is_consumed = true;
    return res;
}

#endif
