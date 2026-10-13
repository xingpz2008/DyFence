#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <algorithm>
#include <cstring>
#include <map>
#include <omp.h>
#include <cassert>
#include "../config.h"
#include "../utils/logger.h"
#include "emp-zk/emp-vole/utility.h"
#include "../data_type/PolyTensor.h"

inline uint64_t pow_mod(uint64_t a, uint64_t deg) {
    assert(deg >= 0);
    uint64_t res = 1;
    uint64_t base = a;

    if (deg == 1) return a;

    while (deg > 0) {
        if (deg & 1) {
            res = mult_mod(res, base);
        }
        base = mult_mod(base, base);
        deg >>= 1;
    }
    return res;
}

static void matrix_mul_acc_kernel(
    uint64_t* __restrict__ Dst,
    const uint64_t* __restrict__ LHS,
    const uint64_t* __restrict__ RHS,
    int M, int K, int N
) {
    size_t total_work = (size_t)M * N * K;

    if (total_work < MVZK_OMP_SIZE_THRESHOLD) {
        for (int i = 0; i < M; ++i) {
            for (int k = 0; k < K; ++k) {
                uint64_t a_val = LHS[i * K + k];
                if (a_val == 0) continue;
                for (int j = 0; j < N; ++j) {
                    uint64_t prod = mult_mod(a_val, RHS[k * N + j]);
                    Dst[i * N + j] = add_mod(Dst[i * N + j], prod);
                }
            }
        }
        return;
    }

    if (M >= 4 && !omp_in_parallel()) {
        #pragma omp parallel for
        for (int i = 0; i < M; ++i) {
            for (int k = 0; k < K; ++k) {
                uint64_t a_val = LHS[i * K + k];
                if (a_val == 0) continue;
                for (int j = 0; j < N; ++j) {
                    uint64_t prod = mult_mod(a_val, RHS[k * N + j]);
                    Dst[i * N + j] = add_mod(Dst[i * N + j], prod);
                }
            }
        }
    }

    else {
        for (int i = 0; i < M; ++i) {
            #pragma omp parallel for if (!omp_in_parallel())
            for (int j = 0; j < N; ++j) {
                uint64_t sum = 0;
                for (int k = 0; k < K; ++k) {
                    // Dst[i, j] += LHS[i, k] * RHS[k, j]
                    uint64_t term = mult_mod(LHS[i * K + k], RHS[k * N + j]);
                    sum = add_mod(sum, term);
                }

                Dst[i * N + j] = add_mod(Dst[i * N + j], sum);
            }
        }
    }
}

inline void refresh_batch(std::vector<PolyTensor>& items,
                          const std::vector<size_t>& ids,
                          const char* refresh_name) {
    if (ids.empty()) return;

    if (ids.size() == 1) {
        size_t id = ids[0];
        items[id] = items[id].refresh_degree(refresh_name);
        return;
    }

    const PolyTensor& first = items[ids[0]];
    PolyTensor batch({(int)ids.size()}, first.degree, first.store);

    for (size_t i = 0; i < ids.size(); ++i) {
        batch.copy_from(items[ids[i]], i);
    }

    PolyTensor fresh = batch.refresh_degree(refresh_name);

    for (size_t i = 0; i < ids.size(); ++i) {
        PolyTensor item({1}, fresh.degree, fresh.store);

        if (!fresh.flat_coeffs.empty()) {
            for (int d = 0; d <= fresh.degree; ++d) {
                item.get_coeffs_ptr(d)[0] = fresh.get_coeffs_ptr(d)[i];
            }
        }

        if (!fresh.flat_keys.empty()) {
            item.get_keys_ptr()[0] = fresh.get_keys_ptr()[i];
        }

        items[ids[i]] = std::move(item);
    }

    fresh.mark_consumed();
}

inline PolyTensor fast_tree_product(std::vector<PolyTensor>& items,
                                    const char* refresh_name = "Refreshed Tensor") {
    if (items.empty()) {
        LOG_ERROR("Empty PolyTensor received at fast_tree_product!");
        exit(-1);
    }

    if (items.size() == 1) {
        return std::move(items[0]);
    }

    std::vector<PolyTensor> current_layer = std::move(items);

    while (current_layer.size() > 1) {
        size_t current_size = current_layer.size();
        size_t next_size = (current_size + 1) / 2;

        std::vector<PolyTensor> next_layer(next_size);

        #pragma omp parallel for schedule(guided) if(current_size >= MVZK_CONFIG_OMP_FAST_TREE_PRODUCT_SIZE_THRESHOLD && !omp_in_parallel())
        for (size_t i = 0; i < current_size / 2; ++i) {
            next_layer[i] = current_layer[2 * i] * current_layer[2 * i + 1];
        }

        if (current_size % 2 != 0) {
            next_layer.back() = std::move(current_layer.back());
        }

        std::map<std::pair<int, PolyTensor::Store>, std::vector<size_t>> refresh_groups;
        for (size_t i = 0; i < next_size; ++i) {
            if (next_layer[i].degree >= MVZK_CONFIG_MULT_PRODUCT_THRESHOLD) {
                refresh_groups[{next_layer[i].degree, next_layer[i].store}].push_back(i);
            }
        }

        for (auto& group : refresh_groups) {
            refresh_batch(next_layer, group.second, refresh_name);
        }

        current_layer = std::move(next_layer);
    }

    return std::move(current_layer[0]);
}

inline PolyTensor product_blocks(const PolyTensor& src, const char* refresh_name) {
    size_t len = src.total_elements;
    if (len == 0) {
        LOG_ERROR("Empty PolyTensor received at product_blocks!");
        exit(-1);
    }

    if (src.flat_coeffs.empty() && src.flat_keys.empty()) {
        LOG_ERROR("PolyTensor without storage received at product_blocks!");
        exit(-1);
    }

    size_t block_size = len;
    if (src.degree > 0) {
        size_t target_degree = std::max<size_t>(1, MVZK_CONFIG_MULT_PRODUCT_THRESHOLD / 2);
        block_size = std::max<size_t>(
            1,
            target_degree / (size_t)src.degree
        );
    }

    size_t block_count = (len + block_size - 1) / block_size;
    std::vector<PolyTensor> blocks(block_count);

    #pragma omp parallel for schedule(static) if(block_count >= MVZK_CONFIG_OMP_FAST_TREE_PRODUCT_SIZE_THRESHOLD && !omp_in_parallel())
    for (size_t b = 0; b < block_count; ++b) {
        size_t begin = b * block_size;
        size_t end = std::min(begin + block_size, len);
        size_t count = end - begin;
        int block_degree = src.degree * (int)count;

        PolyTensor acc({1}, block_degree, src.store);

        if (!src.flat_coeffs.empty() && !acc.flat_coeffs.empty()) {
            if (src.degree == 1) {
                uint64_t* coeffs = acc.flat_coeffs.data();
                coeffs[0] = src.flat_coeffs[begin];
                coeffs[1] = src.flat_coeffs[len + begin];

                int acc_degree = 1;
                for (size_t i = begin + 1; i < end; ++i) {
                    uint64_t c0 = src.flat_coeffs[i];
                    uint64_t c1 = src.flat_coeffs[len + i];

                    coeffs[acc_degree + 1] = mult_mod(coeffs[acc_degree], c1);
                    for (int d = acc_degree; d >= 1; --d) {
                        uint64_t keep = mult_mod(coeffs[d], c0);
                        uint64_t grow = mult_mod(coeffs[d - 1], c1);
                        coeffs[d] = add_mod(keep, grow);
                    }
                    coeffs[0] = mult_mod(coeffs[0], c0);
                    ++acc_degree;
                }
            } else {
                std::vector<uint64_t> coeffs(block_degree + 1, 0);
                std::vector<uint64_t> next(block_degree + 1, 0);
                for (int d = 0; d <= src.degree; ++d) {
                    coeffs[d] = src.flat_coeffs[(size_t)d * len + begin];
                }

                int acc_degree = src.degree;
                for (size_t i = begin + 1; i < end; ++i) {
                    int next_degree = acc_degree + src.degree;
                    std::fill(next.begin(), next.begin() + next_degree + 1, 0);

                    for (int a = 0; a <= acc_degree; ++a) {
                        uint64_t lhs = coeffs[a];
                        if (lhs == 0) continue;

                        for (int d = 0; d <= src.degree; ++d) {
                            uint64_t rhs = src.flat_coeffs[(size_t)d * len + i];
                            if (rhs == 0) continue;

                            next[a + d] = add_mod(next[a + d], mult_mod(lhs, rhs));
                        }
                    }

                    coeffs.swap(next);
                    acc_degree = next_degree;
                }

                for (int d = 0; d <= acc_degree; ++d) {
                    acc.flat_coeffs[d] = coeffs[d];
                }
            }
        }

        if (!src.flat_keys.empty() && !acc.flat_keys.empty()) {
            uint64_t key = src.flat_keys[begin];
            for (size_t i = begin + 1; i < end; ++i) {
                key = mult_mod(key, src.flat_keys[i]);
            }
            acc.flat_keys[0] = key;
        }

        blocks[b] = std::move(acc);
    }

    src.mark_consumed();
    return fast_tree_product(blocks, refresh_name);
}

inline uint64_t sub_mod(const uint64_t& a, const uint64_t& b) {
    return add_mod(a, PR-b);
}

inline void im2col_kernel(
    const uint64_t* __restrict__ src,
    uint64_t* __restrict__ dst,
    int N, int C, int H, int W,
    int kH, int kW,
    int padding, int stride, int dilation,
    int H_out, int W_out
) {
    int K_dim = C * kH * kW;

    size_t total_parallel_tasks = (size_t)N * H_out * W_out;
    #pragma omp parallel for collapse(2) if(total_parallel_tasks >= (MVZK_OMP_SIZE_THRESHOLD / 2) && !omp_in_parallel())
    for (int n = 0; n < N; ++n) {
        for (int h_out = 0; h_out < H_out; ++h_out) {
            for (int w_out = 0; w_out < W_out; ++w_out) {
                // Row layout: [Batch 0 (Pixels...)] [Batch 1 (Pixels...)] ...
                size_t dst_row = ((size_t)n * H_out + h_out) * W_out + w_out;

                for (int c = 0; c < C; ++c) {
                    for (int kh = 0; kh < kH; ++kh) {
                        for (int kw = 0; kw < kW; ++kw) {
                            int h_in = h_out * stride + kh * dilation - padding;
                            int w_in = w_out * stride + kw * dilation - padding;

                            int dst_col = (c * kH + kh) * kW + kw;

                            size_t dst_idx = dst_row * K_dim + dst_col;

                            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                size_t src_idx = ((size_t)n * C + c) * H * W + h_in * W + w_in;
                                dst[dst_idx] = src[src_idx];
                            } else {
                                dst[dst_idx] = 0;
                            }
                        }
                    }
                }
            }
        }
    }
}

// 2. Weight Transpose Kernel: (C_out, C_in, kH, kW) -> (C_in * kH * kW, C_out)

inline void transpose_weight_kernel(
    const uint64_t* __restrict__ src,
    uint64_t* __restrict__ dst,
    int C_out, int C_in, int kH, int kW
) {
    int K = C_in * kH * kW;

    size_t total_elements = (size_t)C_out * K;

    #pragma omp parallel for collapse(2) if(total_elements >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
    for (int co = 0; co < C_out; ++co) {
        for (int k = 0; k < K; ++k) {
            dst[k * C_out + co] = src[co * K + k];
        }
    }
}

// 3. Permute & Add Bias Kernel: (N, H, W, C) -> (N, C, H, W) + Bias

inline void permute_and_add_bias_kernel(
    const uint64_t* __restrict__ src,
    uint64_t* __restrict__ dst,
    const uint64_t* __restrict__ bias,
    uint64_t bias_scale,
    int N, int H_out, int W_out, int C_out
) {
    bool use_scale = (bias_scale != 1);

    size_t total_elements = (size_t)N * H_out * W_out * C_out;

    #pragma omp parallel for collapse(3) if(total_elements >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_out; ++c) {
            for (int h = 0; h < H_out; ++h) {
                uint64_t final_bias_val = 0;
                if (bias) {
                    final_bias_val = bias[c];
                    if (use_scale) {
                        final_bias_val = mult_mod(final_bias_val, bias_scale);
                    }
                }

                for (int w = 0; w < W_out; ++w) {
                    size_t src_idx = (((size_t)n * H_out + h) * W_out + w) * C_out + c;
                    size_t dst_idx = (((size_t)n * C_out + c) * H_out + h) * W_out + w;

                    uint64_t val = src[src_idx];

                    if (bias) {
                        val = add_mod(val, final_bias_val);
                    }

                    dst[dst_idx] = val;
                }
            }
        }
    }
}

// 4. Matrix Transpose Kernel: (Rows, Cols) -> (Cols, Rows)

inline void transpose_matrix_kernel(
    const uint64_t* __restrict__ src,
    uint64_t* __restrict__ dst,
    int Rows, int Cols
) {
    size_t total_elements = (size_t)Rows * Cols;

    #pragma omp parallel for collapse(2) if(total_elements >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
    for (int r = 0; r < Rows; ++r) {
        for (int c = 0; c < Cols; ++c) {
            // Src: [r, c]
            // Dst: [c, r]
            dst[c * Rows + r] = src[r * Cols + c];
        }
    }
}

// 5. Add Bias Row Broadcast Kernel: Dst[row][col] += Bias[col]

inline void add_bias_row_broadcast_kernel(
    uint64_t* __restrict__ data,
    const uint64_t* __restrict__ bias,
    uint64_t bias_scale,
    int Rows, int Cols
) {
    bool use_scale = (bias_scale != 1);

    size_t total_elements = (size_t)Rows * Cols;

    #pragma omp parallel for if(total_elements >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
    for (int r = 0; r < Rows; ++r) {
        for (int c = 0; c < Cols; ++c) {
            uint64_t b_val = bias[c];

            if (use_scale) {
                b_val = mult_mod(b_val, bias_scale);
            }

            // data[r, c] += bias[c]
            int idx = r * Cols + c;
            data[idx] = add_mod(data[idx], b_val);
        }
    }
}

// 6. Im2Col 1D Kernel
// Input: (N, C, L_in) -> Output Matrix: (N * L_out, C * K)
inline void im2col_1d_kernel(
    const uint64_t* __restrict__ src,
    uint64_t* __restrict__ dst,
    int N, int C, int L_in,
    int K_size,
    int padding, int stride, int dilation,
    int L_out
) {
    int K_dim = C * K_size;

    #pragma omp parallel for collapse(2) if (!omp_in_parallel())
    for (int n = 0; n < N; ++n) {
        for (int l_out = 0; l_out < L_out; ++l_out) {
            // Dst Row Index: [Batch, L_out]
            size_t dst_row = (size_t)n * L_out + l_out;

            for (int c = 0; c < C; ++c) {
                for (int k = 0; k < K_size; ++k) {
                    int l_in = l_out * stride + k * dilation - padding;

                    // Dst Col Index
                    int dst_col = c * K_size + k;
                    size_t dst_idx = dst_row * K_dim + dst_col;

                    if (l_in >= 0 && l_in < L_in) {
                        size_t src_idx = ((size_t)n * C + c) * L_in + l_in;
                        dst[dst_idx] = src[src_idx];
                    } else {
                        dst[dst_idx] = 0; // Padding
                    }
                }
            }
        }
    }
}

// 7. Permute & Add Bias 1D Kernel
// Src: (N * L_out, C_out) -> logical (N, L_out, C_out)
// Dst: (N, C_out, L_out)
// Bias: (C_out)
inline void permute_and_add_bias_1d_kernel(
    const uint64_t* __restrict__ src,
    uint64_t* __restrict__ dst,
    const uint64_t* __restrict__ bias,
    uint64_t bias_scale,
    int N, int L_out, int C_out
) {
    bool use_scale = (bias_scale != 1);

    #pragma omp parallel for collapse(2) if (!omp_in_parallel())
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C_out; ++c) {
            // Hoisting Bias Calculation
            uint64_t final_bias = 0;
            if (bias) {
                final_bias = bias[c];
                if (use_scale) final_bias = mult_mod(final_bias, bias_scale);
            }

            for (int l = 0; l < L_out; ++l) {
                // Src Index (Row-Major from MatMul): (n, l, c)
                size_t src_idx = ((size_t)n * L_out + l) * C_out + c;

                // Dst Index (Target NCL): (n, c, l)
                size_t dst_idx = ((size_t)n * C_out + c) * L_out + l;

                uint64_t val = src[src_idx];
                if (bias) {
                    val = add_mod(val, final_bias);
                }
                dst[dst_idx] = val;
            }
        }
    }
}

inline void base_mul_core(
    const uint64_t* __restrict__ A, int dA,
    const uint64_t* __restrict__ B, int dB,
    uint64_t* __restrict__ Res, size_t size)
{
    int dRes = dA + dB;
    size_t total_ops = (size_t)(dRes + 1) * size;

    if (total_ops >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel()) {
        #pragma omp parallel for
        for (size_t i = 0; i < total_ops; ++i) Res[i] = 0;
    } else {
        std::memset(Res, 0, total_ops * sizeof(uint64_t));
    }

    if (size >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel()) {
        #pragma omp parallel
        {
            for (int k = 0; k <= dRes; ++k) {
                uint64_t* res_ptr = Res + k * size;
                int start_i = std::max(0, k - dB);
                int end_i = std::min(dA, k);

                #pragma omp for
                for (size_t idx = 0; idx < size; ++idx) {
                    uint64_t sum = 0;
                    for (int i = start_i; i <= end_i; ++i) {
                        int j = k - i;

                        uint64_t term = mult_mod(A[i * size + idx], B[j * size + idx]);
                        sum = add_mod(sum, term);
                    }
                    res_ptr[idx] = sum;
                }
            }
        }
    } else {
        for (int k = 0; k <= dRes; ++k) {
            uint64_t* res_ptr = Res + k * size;
            int start_i = std::max(0, k - dB);
            int end_i = std::min(dA, k);

            for (size_t idx = 0; idx < size; ++idx) {
                uint64_t sum = 0;
                for (int i = start_i; i <= end_i; ++i) {
                    int j = k - i;
                    uint64_t term = mult_mod(A[i * size + idx], B[j * size + idx]);
                    sum = add_mod(sum, term);
                }
                res_ptr[idx] = sum;
            }
        }
    }
}

inline void karatsuba_core(
    const uint64_t* __restrict__ A, int dA,
    const uint64_t* __restrict__ B, int dB,
    uint64_t* __restrict__ Res, size_t size)
{
#if MVZK_CONFIG_ENABLE_KARATSUBA_OPT
    if (dA < MVZK_CONFIG_KARATSUBA_THRESHOLD || dB < MVZK_CONFIG_KARATSUBA_THRESHOLD ||
        std::max(dA, dB) > 2 * std::min(dA, dB)) {
        base_mul_core(A, dA, B, dB, Res, size);
        return;
    }

    int m = (std::max(dA, dB) + 1) / 2;

    int dA_lo = std::min(dA, m - 1);
    int dA_hi = (dA >= m) ? (dA - m) : -1;

    int dB_lo = std::min(dB, m - 1);
    int dB_hi = (dB >= m) ? (dB - m) : -1;

    int dAsum = std::max(dA_lo, dA_hi);
    int dBsum = std::max(dB_lo, dB_hi);

    std::vector<uint64_t> Asum((dAsum + 1) * size, 0);
    std::vector<uint64_t> Bsum((dBsum + 1) * size, 0);

    for (int i = 0; i <= dAsum; ++i) {
        const uint64_t* p_lo = (i <= dA_lo) ? (A + i * size) : nullptr;
        const uint64_t* p_hi = (i <= dA_hi) ? (A + (m + i) * size) : nullptr;
        uint64_t* p_sum = Asum.data() + i * size;

        for (size_t idx = 0; idx < size; ++idx) {
            uint64_t val_lo = p_lo ? p_lo[idx] : 0;
            uint64_t val_hi = p_hi ? p_hi[idx] : 0;
            p_sum[idx] = add_mod(val_lo, val_hi);
        }
    }
    for (int i = 0; i <= dBsum; ++i) {
        const uint64_t* p_lo = (i <= dB_lo) ? (B + i * size) : nullptr;
        const uint64_t* p_hi = (i <= dB_hi) ? (B + (m + i) * size) : nullptr;
        uint64_t* p_sum = Bsum.data() + i * size;

        for (size_t idx = 0; idx < size; ++idx) {
            uint64_t val_lo = p_lo ? p_lo[idx] : 0;
            uint64_t val_hi = p_hi ? p_hi[idx] : 0;
            p_sum[idx] = add_mod(val_lo, val_hi);
        }
    }

    int dP1 = dA_lo + dB_lo;
    int dP2 = (dA_hi >= 0 && dB_hi >= 0) ? (dA_hi + dB_hi) : -1;
    int dP3 = dAsum + dBsum;

    std::vector<uint64_t> P1((dP1 + 1) * size, 0);
    std::vector<uint64_t> P3((dP3 + 1) * size, 0);

    karatsuba_core(A, dA_lo, B, dB_lo, P1.data(), size);
    karatsuba_core(Asum.data(), dAsum, Bsum.data(), dBsum, P3.data(), size);

    std::vector<uint64_t> P2;
    if (dP2 >= 0) {
        P2.resize((dP2 + 1) * size, 0);
        karatsuba_core(A + m * size, dA_hi, B + m * size, dB_hi, P2.data(), size);
    }

    int dRes = dA + dB;
    for (size_t i = 0; i < (size_t)(dRes + 1) * size; ++i) Res[i] = 0;

    for (int i = 0; i <= dP1; ++i) {
        uint64_t* dst = Res + i * size;
        uint64_t* src = P1.data() + i * size;
        for (size_t idx = 0; idx < size; ++idx) dst[idx] = add_mod(dst[idx], src[idx]);
    }
    if (dP2 >= 0) {
        for (int i = 0; i <= dP2; ++i) {
            uint64_t* dst = Res + (2 * m + i) * size;
            uint64_t* src = P2.data() + i * size;
            for (size_t idx = 0; idx < size; ++idx) dst[idx] = add_mod(dst[idx], src[idx]);
        }
    }
    for (int i = 0; i <= dP3; ++i) {
        uint64_t* dst = Res + (m + i) * size;
        uint64_t* src3 = P3.data() + i * size;
        const uint64_t* src1 = (i <= dP1) ? (P1.data() + i * size) : nullptr;
        const uint64_t* src2 = (dP2 >= 0 && i <= dP2) ? (P2.data() + i * size) : nullptr;

        for (size_t idx = 0; idx < size; ++idx) {
            uint64_t val = src3[idx];
            if (src1) val = sub_mod(val, src1[idx]);
            if (src2) val = sub_mod(val, src2[idx]);
            dst[idx] = add_mod(dst[idx], val);
        }
    }
#else
    base_mul_core(A, dA, B, dB, Res, size);
#endif
}
