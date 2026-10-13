#ifndef MVZK_BACKEND_PROVER_IMPL_ARITH_HPP__
#define MVZK_BACKEND_PROVER_IMPL_ARITH_HPP__

#include "../../exec/ExecProver.h"

template <typename IO>
PolyDelta MVZKExecProver<IO>::add(const PolyDelta& lhs, const PolyDelta& rhs) {
    // Prover side, need align the polydelta

    PolyDelta res;

    res.degree = std::max(lhs.degree, rhs.degree);

    size_t size = res.degree + 1;
    res.coeffs.resize(size);

    size_t offset_lhs = res.degree - lhs.degree;
    size_t offset_rhs = res.degree - rhs.degree;

    for (size_t i = 0; i < size; ++i) {
        uint64_t u = (i < offset_lhs) ? 0 : lhs.coeffs[i - offset_lhs];
        uint64_t v = (i < offset_rhs) ? 0 : rhs.coeffs[i - offset_rhs];

        res.coeffs[i] = add_mod(u, v);
    }

    res.is_consumed = false;
    lhs.is_consumed = true;
    rhs.is_consumed = true;

    return res;
}


template <typename IO>
PolyTensor MVZKExecProver<IO>::add(const PolyTensor& lhs, const PolyTensor& rhs) {
    if (lhs.total_elements != rhs.total_elements) {
        LOG_ERROR("shape mismatch in add!");
        exit(-1);
    }

    int res_degree = std::max(lhs.degree, rhs.degree);
    PolyTensor res(lhs.shape, res_degree, PolyTensor::Coeffs);

    size_t size = lhs.total_elements;

    const std::vector<uint64_t>& lhs_coeffs = lhs.flat_coeffs;
    const std::vector<uint64_t>& rhs_coeffs = rhs.flat_coeffs;
    std::vector<uint64_t>& res_coeffs = res.flat_coeffs;

    int offset_lhs = res_degree - lhs.degree;
    int offset_rhs = res_degree - rhs.degree;

    size_t total_coeffs_len = size * (res_degree + 1);

    for (int d = 0; d <= res_degree; ++d) {
        int d_lhs = d - offset_lhs;
        int d_rhs = d - offset_rhs;

        uint64_t* res_ptr = res.get_coeffs_ptr(d);

        const uint64_t* lhs_ptr = (d_lhs >= 0) ? lhs.get_coeffs_ptr(d_lhs) : nullptr;
        const uint64_t* rhs_ptr = (d_rhs >= 0) ? rhs.get_coeffs_ptr(d_rhs) : nullptr;

        #pragma omp parallel for if(size >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
        for (size_t i = 0; i < size; ++i) {
            uint64_t u = (lhs_ptr) ? lhs_ptr[i] : 0;
            uint64_t v = (rhs_ptr) ? rhs_ptr[i] : 0;
            res_ptr[i] = add_mod(u, v);
        }
    }

    res.is_consumed = false;
    lhs.is_consumed = true;
    rhs.is_consumed = true;
    return res;
}


template <typename IO>
PolyDelta MVZKExecProver<IO>::sub(const PolyDelta& lhs, const PolyDelta& rhs) {
    // Prover side, need align the polydelta

    PolyDelta res;

    res.degree = std::max(lhs.degree, rhs.degree);

    size_t size = res.degree + 1;
    res.coeffs.resize(size);

    size_t offset_lhs = res.degree - lhs.degree;
    size_t offset_rhs = res.degree - rhs.degree;

    for (size_t i = 0; i < size; ++i) {
        uint64_t u = (i < offset_lhs) ? 0 : lhs.coeffs[i - offset_lhs];
        uint64_t v = (i < offset_rhs) ? 0 : rhs.coeffs[i - offset_rhs];

        res.coeffs[i] = add_mod(u, (PR - v));
    }

    res.is_consumed = false;
    lhs.is_consumed = true;
    rhs.is_consumed = true;

    return res;
}


template <typename IO>
PolyTensor MVZKExecProver<IO>::sub(const PolyTensor& lhs, const PolyTensor& rhs) {
    if (lhs.total_elements != rhs.total_elements) {
        LOG_ERROR("shape mismatch in sub!");
        exit(-1);
    }

    int res_degree = std::max(lhs.degree, rhs.degree);
    PolyTensor res(lhs.shape, res_degree, PolyTensor::Coeffs);

    size_t size = lhs.total_elements;

    int offset_lhs = res_degree - lhs.degree;
    int offset_rhs = res_degree - rhs.degree;

    for (int d = 0; d <= res_degree; ++d) {
        int d_lhs = d - offset_lhs;
        int d_rhs = d - offset_rhs;

        uint64_t* res_ptr = res.get_coeffs_ptr(d);

        const uint64_t* lhs_ptr = (d_lhs >= 0) ? lhs.get_coeffs_ptr(d_lhs) : nullptr;
        const uint64_t* rhs_ptr = (d_rhs >= 0) ? rhs.get_coeffs_ptr(d_rhs) : nullptr;

        #pragma omp parallel for if(size >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
        for (size_t i = 0; i < size; ++i) {
            uint64_t u = (lhs_ptr) ? lhs_ptr[i] : 0;
            uint64_t v = (rhs_ptr) ? rhs_ptr[i] : 0;

            res_ptr[i] = add_mod(u, PR - v);
        }
    }

    res.is_consumed = false;
    lhs.is_consumed = true;
    rhs.is_consumed = true;
    return res;
}


template <typename IO>
PolyDelta MVZKExecProver<IO>::mul(const PolyDelta& lhs, const PolyDelta& rhs) {
    // Prover side multiplication
    PolyDelta res;

    res.degree = lhs.degree + rhs.degree;

    size_t size = res.degree + 1;
    res.coeffs.assign(size, 0);

    for (size_t i = 0; i < lhs.coeffs.size(); ++i) {
        if (lhs.coeffs[i] == 0) continue;

        for (size_t j = 0; j < rhs.coeffs.size(); ++j) {
            uint64_t term = mult_mod(lhs.coeffs[i], rhs.coeffs[j]);

            // res[i+j] += term
            res.coeffs[i + j] = add_mod(res.coeffs[i + j], term);
        }
    }

    res.is_consumed = false;

    lhs.is_consumed = true;
    rhs.is_consumed = true;

    return res;
}


template <typename IO>
PolyTensor MVZKExecProver<IO>::mul(const PolyTensor& lhs, const PolyTensor& rhs) {
    if (lhs.total_elements != rhs.total_elements) {
        LOG_ERROR("Tensor shape mismatch in mul!");
        exit(-1);
    }

    int res_degree = lhs.degree + rhs.degree;
    PolyTensor res(lhs.shape, res_degree, PolyTensor::Coeffs);
    size_t size = lhs.total_elements;

    karatsuba_core(
        lhs.get_coeffs_ptr(0), lhs.degree,
        rhs.get_coeffs_ptr(0), rhs.degree,
        res.get_coeffs_ptr(0), size
    );

    res.is_consumed = false;
    lhs.is_consumed = true;
    rhs.is_consumed = true;
    return res;
}


template <typename IO>
void MVZKExecProver<IO>::add_assign(PolyDelta& lhs, const PolyDelta& rhs) {
    if (rhs.degree > lhs.degree) {
        size_t old_size = lhs.coeffs.size();
        size_t new_size = rhs.coeffs.size();
        size_t offset_diff = rhs.degree - lhs.degree;

        lhs.coeffs.resize(new_size);

        for (int i = (int)old_size - 1; i >= 0; --i) {
            lhs.coeffs[i + offset_diff] = lhs.coeffs[i];
        }

        for (size_t i = 0; i < offset_diff; ++i) {
            lhs.coeffs[i] = 0;
        }

        lhs.degree = rhs.degree;
    }

    size_t offset_rhs = lhs.degree - rhs.degree;
    size_t rhs_len = rhs.coeffs.size();

    for (size_t i = 0; i < rhs_len; ++i) {
        size_t target_idx = i + offset_rhs;
        lhs.coeffs[target_idx] = add_mod(lhs.coeffs[target_idx], rhs.coeffs[i]);
    }

    // Update consumption flag
    rhs.is_consumed = true;
    lhs.is_consumed = false;
}


template <typename IO>
void MVZKExecProver<IO>::add_assign(PolyTensor& lhs, const PolyTensor& rhs) {
    if (lhs.total_elements != rhs.total_elements) {
        LOG_ERROR("Tensor shape mismatch in prover add_assign!");
        exit(-1);
    }

    size_t size = lhs.total_elements;

    if (rhs.degree > lhs.degree) {
        int old_deg = lhs.degree;
        int new_deg = rhs.degree;
        int offset = new_deg - old_deg;

        lhs.flat_coeffs.resize(size * (new_deg + 1));

        uint64_t* lhs_base = lhs.flat_coeffs.data();
        const uint64_t* rhs_base = rhs.flat_coeffs.data();

        size_t bytes_to_move = size * (old_deg + 1) * sizeof(uint64_t);

        // Dest: lhs_base + offset * size
        // Src:  lhs_base
        std::memmove(lhs_base + (offset * size), lhs_base, bytes_to_move);

        std::memset(lhs_base, 0, offset * size * sizeof(uint64_t));

        lhs.degree = new_deg;

        size_t total_len = size * (new_deg + 1);
        #pragma omp parallel for if(size >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
        for (size_t i = 0; i < total_len; ++i) {
            lhs_base[i] = add_mod(lhs_base[i], rhs_base[i]);
        }
    } else {
        // LHS: [Deg0] [Deg1] ...
        // RHS: [Deg0] ...

        // P_new = P_lhs + P_rhs * X^diff

        int offset = lhs.degree - rhs.degree;

        size_t start_idx = offset * size;
        size_t rhs_len = size * (rhs.degree + 1);

        uint64_t* lhs_base = lhs.flat_coeffs.data();
        const uint64_t* rhs_base = rhs.flat_coeffs.data();

        #pragma omp parallel for if(size >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
        for (size_t i = 0; i < rhs_len; ++i) {
            lhs_base[start_idx + i] = add_mod(lhs_base[start_idx + i], rhs_base[i]);
        }
    }

    lhs.is_consumed = false;
    rhs.is_consumed = true;
}


template <typename IO>
void MVZKExecProver<IO>::sub_assign(PolyDelta& lhs, const PolyDelta& rhs) {
    if (rhs.degree > lhs.degree) {
        size_t old_size = lhs.coeffs.size();
        size_t new_size = rhs.coeffs.size();
        size_t offset_diff = rhs.degree - lhs.degree;

        lhs.coeffs.resize(new_size);

        for (int i = (int)old_size - 1; i >= 0; --i) {
            lhs.coeffs[i + offset_diff] = lhs.coeffs[i];
        }

        for (size_t i = 0; i < offset_diff; ++i) {
            lhs.coeffs[i] = 0;
        }

        lhs.degree = rhs.degree;
    }

    size_t offset_rhs = lhs.degree - rhs.degree;
    size_t rhs_len = rhs.coeffs.size();

    for (size_t i = 0; i < rhs_len; ++i) {
        size_t target_idx = i + offset_rhs;

        uint64_t subtrahend = (rhs.coeffs[i] == 0) ? 0 : (PR - rhs.coeffs[i]);

        lhs.coeffs[target_idx] = add_mod(lhs.coeffs[target_idx], subtrahend);
    }
    rhs.is_consumed = true;
    lhs.is_consumed = false;
}


template <typename IO>
void MVZKExecProver<IO>::sub_assign(PolyTensor& lhs, const PolyTensor& rhs) {
    if (lhs.total_elements != rhs.total_elements) {
        LOG_ERROR("Tensor shape mismatch in prover add_assign!");
        exit(-1);
    }

    size_t size = lhs.total_elements;

    if (rhs.degree > lhs.degree) {
        int old_deg = lhs.degree;
        int new_deg = rhs.degree;
        int offset = new_deg - old_deg;

        lhs.flat_coeffs.resize(size * (new_deg + 1));

        uint64_t* lhs_base = lhs.flat_coeffs.data();
        const uint64_t* rhs_base = rhs.flat_coeffs.data();

        size_t bytes_to_move = size * (old_deg + 1) * sizeof(uint64_t);

        // Dest: lhs_base + offset * size
        // Src:  lhs_base
        std::memmove(lhs_base + (offset * size), lhs_base, bytes_to_move);

        std::memset(lhs_base, 0, offset * size * sizeof(uint64_t));

        lhs.degree = new_deg;

        size_t total_len = size * (new_deg + 1);
        #pragma omp parallel for if(size >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
        for (size_t i = 0; i < total_len; ++i) {
            lhs_base[i] = add_mod(lhs_base[i], PR - rhs_base[i]);
        }
    } else {
        // LHS: [Deg0] [Deg1] ...
        // RHS: [Deg0] ...

        // P_new = P_lhs + P_rhs * X^diff

        int offset = lhs.degree - rhs.degree;

        size_t start_idx = offset * size;
        size_t rhs_len = size * (rhs.degree + 1);

        uint64_t* lhs_base = lhs.flat_coeffs.data();
        const uint64_t* rhs_base = rhs.flat_coeffs.data();

        #pragma omp parallel for if(size >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
        for (size_t i = 0; i < rhs_len; ++i) {
            lhs_base[start_idx + i] = add_mod(lhs_base[start_idx + i], PR - rhs_base[i]);
        }
    }

    lhs.is_consumed = false;
    rhs.is_consumed = true;
}


template <typename IO>
void MVZKExecProver<IO>::mul_assign(PolyDelta& lhs, uint64_t val) {
    for (auto& c : lhs.coeffs) {
        c = mult_mod(c, val);
    }
    lhs.is_consumed = false;
}


template <typename IO>
void MVZKExecProver<IO>::mul_assign(PolyTensor& lhs, uint64_t val) {
    size_t size = lhs.total_elements;

    size_t total_coeffs_len = lhs.flat_coeffs.size();
    uint64_t* coeffs_ptr = lhs.flat_coeffs.data();

    #pragma omp parallel for if(size >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
    for (size_t i = 0; i < total_coeffs_len; ++i) {
        coeffs_ptr[i] = mult_mod(coeffs_ptr[i], val);
    }

    lhs.is_consumed = false;
}


template <typename IO>
void MVZKExecProver<IO>::mul_assign(PolyTensor& lhs, const PolyTensor& rhs) {
    LOG_WARN("Invocation of in-place multiplication for PolyTensor may cause performance bottleneck!");
    if (lhs.total_elements != rhs.total_elements) {
        LOG_ERROR("Tensor shape mismatch in mul_assign!");
        exit(-1);
    }

    PolyTensor old_lhs = lhs.clone();
    int new_degree = old_lhs.degree + rhs.degree;
    size_t size = lhs.total_elements;

    lhs.flat_coeffs.assign(size * (new_degree + 1), 0);
    lhs.degree = new_degree;

    if (size >= MVZK_OMP_SIZE_THRESHOLD) {
        for (int k = 0; k <= new_degree; ++k) {
            uint64_t* res_ptr = lhs.get_coeffs_ptr(k);
            int start_i = std::max(0, k - rhs.degree);
            int end_i = std::min(old_lhs.degree, k);

            #pragma omp parallel for if(size >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
            for (size_t idx = 0; idx < size; ++idx) {
                uint64_t sum = 0;
                for (int i = start_i; i <= end_i; ++i) {
                    int j = k - i;
                    uint64_t term = mult_mod(old_lhs.get_coeffs_ptr(i)[idx], rhs.get_coeffs_ptr(j)[idx]);
                    sum = add_mod(sum, term);
                }
                res_ptr[idx] = sum;
            }
        }
    } else {
        #pragma omp parallel for if(new_degree >= MVZK_OMP_DEGREE_THRESHOLD && !omp_in_parallel())
        for (int k = 0; k <= new_degree; ++k) {
            uint64_t* res_ptr = lhs.get_coeffs_ptr(k);
            int start_i = std::max(0, k - rhs.degree);
            int end_i = std::min(old_lhs.degree, k);

            for (size_t idx = 0; idx < size; ++idx) {
                uint64_t sum = 0;
                for (int i = start_i; i <= end_i; ++i) {
                    int j = k - i;
                    uint64_t term = mult_mod(old_lhs.get_coeffs_ptr(i)[idx], rhs.get_coeffs_ptr(j)[idx]);
                    sum = add_mod(sum, term);
                }
                res_ptr[idx] = sum;
            }
        }
    }

    lhs.is_consumed = false;
    rhs.is_consumed = true;
}


template <typename IO>
void MVZKExecProver<IO>::add_assign_const(PolyDelta& lhs, uint64_t val) {
    lhs.coeffs[lhs.degree] = add_mod(lhs.coeffs[lhs.degree], val);
    lhs.is_consumed = false;
}


template <typename IO>
void MVZKExecProver<IO>::sub_assign_const(PolyDelta& lhs, uint64_t val) {
    lhs.coeffs[lhs.degree] = add_mod(lhs.coeffs[lhs.degree], (PR - val));
    lhs.is_consumed = false;
}


template <typename IO>
PolyDelta MVZKExecProver<IO>::add_const(const PolyDelta& lhs, uint64_t val) {
    // At the prover site, nothing but the highest coef (same as real_val) has to be changed.
    PolyDelta res;

    res = lhs.clone();
    res.coeffs[lhs.degree] = add_mod(res.coeffs[lhs.degree], val);

    res.is_consumed = false;
    lhs.is_consumed = true;

    return res;
}


template <typename IO>
PolyDelta MVZKExecProver<IO>::sub_const(const PolyDelta& lhs, uint64_t val) {
    // Prover site, nearly identical to add const

    PolyDelta res;

    res = lhs.clone();
    res.coeffs[lhs.degree] = add_mod(res.coeffs[lhs.degree], (PR - val));

    res.is_consumed = false;
    lhs.is_consumed = true;

    return res;
}


template <typename IO>
PolyDelta MVZKExecProver<IO>::sub_const_rev(uint64_t val, const PolyDelta& rhs) {
    // Prover side: set all coefs to -coefs

    PolyDelta res;

    res = rhs.clone();
    for (size_t i = 0; i < res.coeffs.size(); i++){
        res.coeffs[i] = add_mod(0, (PR - res.coeffs[i]));
    }
    res.coeffs[rhs.degree] = add_mod(res.coeffs[rhs.degree], val);

    res.is_consumed = false;
    rhs.is_consumed = true;

    return res;
}


template <typename IO>
PolyDelta MVZKExecProver<IO>::mul_const(const PolyDelta& lhs, uint64_t val) {
    // Prover site, all times c
    PolyDelta res;

    res = lhs.clone();
    for (size_t i = 0; i < res.coeffs.size(); i++){
        res.coeffs[i] = mult_mod(res.coeffs[i], val);
    }

    res.is_consumed = false;
    lhs.is_consumed = true;

    return res;
}


template <typename IO>
void MVZKExecProver<IO>::add_assign_const(PolyTensor& lhs, uint64_t val) {
    size_t size = lhs.total_elements;

    // P_new(X) = P_old(X) + val * X^degree

    uint64_t* highest_coeff_ptr = lhs.get_coeffs_ptr(lhs.degree);

    #pragma omp parallel for if(size >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
    for (size_t i = 0; i < size; ++i) {
        highest_coeff_ptr[i] = add_mod(highest_coeff_ptr[i], val);
    }

    lhs.is_consumed = false;
}


template <typename IO>
PolyTensor MVZKExecProver<IO>::MatMul(const PolyTensor& lhs, const PolyTensor& rhs) {
    // Verifier Site MatMul2D

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
PolyTensor MVZKExecProver<IO>::MatMul(const PolyTensor& lhs, const PolyTensor& rhs, int M, int K, int N) {
    if ((size_t)M * (size_t)K != lhs.total_elements || rhs.shape.size() != 2 ||
        rhs.shape[0] != K || rhs.shape[1] != N) {
        LOG_ERROR("Invalid shape for matmul!");
        exit(-1);
    }

    int res_degree = lhs.degree + rhs.degree;

    PolyTensor res({M, N}, res_degree, PolyTensor::Coeffs);

    for (int k = 0; k <= res_degree; ++k) {
        uint64_t* dst_ptr = res.get_coeffs_ptr(k);

        int start_i = std::max(0, k - rhs.degree);
        int end_i   = std::min(lhs.degree, k);

        for (int i = start_i; i <= end_i; ++i) {
            int j = k - i;

            matrix_mul_acc_kernel(
                dst_ptr,                // Dst (+=)
                lhs.get_coeffs_ptr(i),  // LHS Matrix A_i
                rhs.get_coeffs_ptr(j),  // RHS Matrix B_j
                M, K, N
            );
        }
    }

    res.is_consumed = false;
    lhs.is_consumed = true;
    rhs.is_consumed = true;
    return res;
}

#endif
