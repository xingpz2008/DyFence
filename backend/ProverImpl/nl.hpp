#ifndef MVZK_BACKEND_PROVER_IMPL_NL_HPP__
#define MVZK_BACKEND_PROVER_IMPL_NL_HPP__

#include "../../exec/ExecProver.h"

template <typename IO>
PolyTensor MVZKExecProver<IO>::relu(PolyTensor& x, uint64_t bitlen, uint64_t digdec_k, bool do_truncation, uint64_t scale) {
    size_t n = x.total_elements;

    int s_0 = scale;
    int s_rest = (bitlen > (uint64_t)s_0 && digdec_k > 1) ?
                 (bitlen - s_0 + digdec_k - 2) / (digdec_k - 1) : 0;

    std::vector<uint64_t> plain_abs_x(n);
    std::vector<uint64_t> plain_sign_bq(n);
    const uint64_t* x_ptr = x.get_real_vals_ptr();
    uint64_t zero_point = PR >> 1;

    std::vector<uint64_t> flat_dig_dec(n * digdec_k, 0);
    uint64_t mask_0 = (1ULL << s_0) - 1;
    uint64_t mask_rest = (1ULL << s_rest) - 1;

    #pragma omp parallel for schedule(static) if(n >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
    for(size_t i = 0; i < n; ++i) {
        uint64_t val = x_ptr[i];
        if (val > zero_point) {
            plain_abs_x[i] = PR - val; // |x|
            plain_sign_bq[i] = 0;      // b_Q = 0 (Negative)
        } else {
            plain_abs_x[i] = val;      // |x|
            plain_sign_bq[i] = 1;      // b_Q = 1 (Positive)
        }

        uint64_t abs_val = plain_abs_x[i];
        flat_dig_dec[i * digdec_k + 0] = abs_val & mask_0;
        abs_val >>= s_0;
        for(int j = 1; j < digdec_k; ++j) {
            flat_dig_dec[i * digdec_k + j] = abs_val & mask_rest;
            abs_val >>= s_rest;
        }
    }

    // Step 2 & 3: Authenticate b_Q

    PolyTensor authenticated_bq = input(x.shape, plain_sign_bq);

    PolyTensor b_Q_check = authenticated_bq * (authenticated_bq - 1);
    PolyTensor::store_zero_relation(b_Q_check, "[ReLU] b_Q");

    uint64_t table_size_0 = 1ULL << s_0;
    std::vector<uint64_t> rangeTableData_0(table_size_0);
    for (size_t i = 0; i < table_size_0; i++) rangeTableData_0[i] = i;
    RangeCheckTable rangeTable_0(rangeTableData_0);

    uint64_t table_size_rest = 1ULL << s_rest;
    std::vector<uint64_t> rangeTableData_rest(table_size_rest);
    for (size_t i = 0; i < table_size_rest; i++) rangeTableData_rest[i] = i;
    RangeCheckTable rangeTable_rest(rangeTableData_rest);

    PolyTensor X_recon;
    PolyTensor X_out;

    for (int j = 0; j < digdec_k; ++j) {
        std::vector<uint64_t> seg_j_data(n);
        #pragma omp parallel for schedule(static) if(n >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
        for(size_t i = 0; i < n; ++i) seg_j_data[i] = flat_dig_dec[i * digdec_k + j];

        PolyTensor authenticated_seg_j = input(x.shape, seg_j_data);

        if (j == 0) {
            rangeTable_0.range_check(authenticated_seg_j, "[ReLU] x_i range (low)");
            X_recon = authenticated_seg_j.clone();
        } else {
            rangeTable_rest.range_check(authenticated_seg_j, "[ReLU] x_i range (high)");

            uint64_t shift_recon = 1ULL << (s_0 + (j - 1) * s_rest);
            X_recon = X_recon + authenticated_seg_j * shift_recon;

            if (do_truncation) {
                uint64_t trunc_shift = 1ULL << ((j - 1) * s_rest);
                if (j == 1) X_out = authenticated_seg_j * trunc_shift;
                else        X_out = X_out + authenticated_seg_j * trunc_shift;
            }
        }
    }

    if (!do_truncation) {
        X_out = X_recon.clone();
    }

    // Step 6, 7, 8: Consistency Check

    PolyTensor term1 = authenticated_bq * (X_recon - x);
    PolyTensor term2 = (1 - authenticated_bq) * (X_recon + x);

    PolyTensor consistency = term1 + term2;
    PolyTensor::store_zero_relation(consistency, "[ReLU (Conv)] from prev");

    // Step 9: Compute Output

    PolyTensor result = authenticated_bq * X_out;

    return result;
}


template <typename IO>
PolyTensor MVZKExecProver<IO>::maxpool2d(PolyTensor& pt_in, int kernel_size, int stride, int padding, uint64_t bitlen, uint64_t digdec_k, uint64_t scale) {
    int N = pt_in.shape[0], C = pt_in.shape[1], H_in = pt_in.shape[2], W_in = pt_in.shape[3];
    int kH = kernel_size, kW = kernel_size;
    int H_out = (H_in + 2 * padding - kH) / stride + 1;
    int W_out = (W_in + 2 * padding - kW) / stride + 1;

    int M = N * C * H_out * W_out;
    int h = kH * kW;

    // Step 1: Plaintext Pre-computation (Prover finds Max)

    std::vector<uint64_t> plain_max(M, 0);
    const uint64_t* in_ptr = pt_in.get_real_vals_ptr();

    #pragma omp parallel for collapse(2) if(M >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
    for(int n=0; n<N; ++n) {
        for(int c=0; c<C; ++c) {
            for(int ho=0; ho<H_out; ++ho) {
                for(int wo=0; wo<W_out; ++wo) {
                    int out_idx = n*(C*H_out*W_out) + c*(H_out*W_out) + ho*W_out + wo;
                    uint64_t x_max = 0;
                    bool first = true;

                    for(int kh=0; kh<kH; ++kh) {
                        for(int kw=0; kw<kW; ++kw) {
                            int hi = ho * stride - padding + kh;
                            int wi = wo * stride - padding + kw;
                            if(hi >= 0 && hi < H_in && wi >= 0 && wi < W_in) {
                                int in_idx = n*(C*H_in*W_in) + c*(H_in*W_in) + hi*W_in + wi;
                                uint64_t val = in_ptr[in_idx];
                                if(first) {
                                    x_max = val;
                                    first = false;
                                } else {
                                    if(helper_plaintext_fp_greater(val, x_max)) x_max = val;
                                }
                            }
                        }
                    }
                    plain_max[out_idx] = x_max;
                }
            }
        }
    }

    PolyTensor pt_max = this->input({N, C, H_out, W_out}, plain_max);

    int s_0 = scale;
    int s_rest = (bitlen > (uint64_t)s_0 && digdec_k > 1) ?
                 (bitlen - s_0 + digdec_k - 2) / (digdec_k - 1) : 0;

    uint64_t table_size_0 = 1ULL << s_0;
    std::vector<uint64_t> rangeTableData_0(table_size_0);
    for (size_t i = 0; i < table_size_0; i++) rangeTableData_0[i] = i;
    RangeCheckTable rangeTable_0(rangeTableData_0);

    uint64_t table_size_rest = 1ULL << s_rest;
    std::vector<uint64_t> rangeTableData_rest(table_size_rest);
    for (size_t i = 0; i < table_size_rest; i++) rangeTableData_rest[i] = i;
    RangeCheckTable rangeTable_rest(rangeTableData_rest);

    PolyTensor pt_prod;

    for (int j = 0; j < h; ++j) {
        int kh = j / kW;
        int kw = j % kW;

        PolyTensor pt_in_aligned({N, C, H_out, W_out}, pt_in.degree, PolyTensor::Coeffs);

        #pragma omp parallel for collapse(4) schedule(static) if(M >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
        for(int n=0; n<N; ++n) {
            for(int c=0; c<C; ++c) {
                for(int ho=0; ho<H_out; ++ho) {
                    for(int wo=0; wo<W_out; ++wo) {
                        int m = n*(C*H_out*W_out) + c*(H_out*W_out) + ho*W_out + wo;
                        int hi = ho * stride - padding + kh;
                        int wi = wo * stride - padding + kw;

                        if(hi >= 0 && hi < H_in && wi >= 0 && wi < W_in) {
                            int in_idx = n*(C*H_in*W_in) + c*(H_in*W_in) + hi*W_in + wi;
                            // pt_in_aligned.flat_real_vals[m] = pt_in.flat_real_vals[in_idx];
                            for(int d=0; d<=pt_in.degree; ++d) {
                                pt_in_aligned.get_coeffs_ptr(d)[m] = pt_in.get_coeffs_ptr(d)[in_idx];
                            }
                        } else {
                            // pt_in_aligned.flat_real_vals[m] = add_mod(pt_max.flat_real_vals[m], PR - 1);
                            for(int d=0; d<pt_in.degree; ++d) {
                                pt_in_aligned.get_coeffs_ptr(d)[m] = (d <= pt_max.degree) ? pt_max.get_coeffs_ptr(d)[m] : 0;
                            }
                            uint64_t max_coeff_high = (pt_in.degree <= pt_max.degree) ? pt_max.get_coeffs_ptr(pt_in.degree)[m] : 0;
                            pt_in_aligned.get_coeffs_ptr(pt_in.degree)[m] = add_mod(max_coeff_high, PR - 1);
                        }
                    }
                }
            }
        }

        PolyTensor pt_y_hat_j = pt_max - pt_in_aligned;

        // --- Upper Bound Check ---

        std::vector<uint64_t> flat_dig_dec(M * digdec_k, 0);
        uint64_t mask_0 = (1ULL << s_0) - 1;
        uint64_t mask_rest = (1ULL << s_rest) - 1;

        const uint64_t* y_hat_ptr = pt_y_hat_j.get_real_vals_ptr();

        #pragma omp parallel for schedule(static) if(M >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
        for(size_t i = 0; i < M; ++i) {
            uint64_t abs_val = y_hat_ptr[i];
            flat_dig_dec[i * digdec_k + 0] = abs_val & mask_0;
            abs_val >>= s_0;
            for(int d = 1; d < digdec_k; ++d) {
                flat_dig_dec[i * digdec_k + d] = abs_val & mask_rest;
                abs_val >>= s_rest;
            }
        }

        PolyTensor X_recon;

        for (int d = 0; d < digdec_k; ++d) {
            std::vector<uint64_t> seg_data(M);
            #pragma omp parallel for schedule(static) if(M >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
            for(size_t i=0; i<M; ++i) seg_data[i] = flat_dig_dec[i * digdec_k + d];

            PolyTensor auth_seg = this->input({M}, seg_data);

            if (d == 0) {
                rangeTable_0.range_check(auth_seg, "[MaxPool] x_i range (low)");
                X_recon = auth_seg.clone();
            } else {
                rangeTable_rest.range_check(auth_seg, "[MaxPool] x_i range (high)");
                uint64_t shift_recon = 1ULL << (s_0 + (d - 1) * s_rest);
                X_recon = X_recon + auth_seg * shift_recon;
            }
        }

        PolyTensor::store_relation(X_recon, pt_y_hat_j, "[MaxPool] greater than");

        // --- Existence Check ---
        if (j == 0) {
            pt_prod = pt_y_hat_j.clone();
            pt_y_hat_j.is_consumed = true;
        } else {
            pt_prod = pt_prod * pt_y_hat_j;
        }
    }

    PolyTensor::store_zero_relation(pt_prod, "[MaxPool] Existence check");

    pt_in.mark_consumed();

    return pt_max;
}


template <typename IO>
PolyTensor MVZKExecProver<IO>::integrated_nl(PolyTensor& pt_in, int kernel_size, int stride, int padding, uint64_t bitlen, uint64_t digdec_k, bool do_truncation, uint64_t scale) {
    int N = pt_in.shape[0], C = pt_in.shape[1], H_in = pt_in.shape[2], W_in = pt_in.shape[3];
    int kH = kernel_size, kW = kernel_size;
    int H_out = (H_in + 2 * padding - kH) / stride + 1;
    int W_out = (W_in + 2 * padding - kW) / stride + 1;

    int M = N * C * H_out * W_out;
    int h = kH * kW;

    // Step 1: Plaintext Pre-computation

    std::vector<uint64_t> plain_max(M, 0);
    const uint64_t* in_ptr = pt_in.get_real_vals_ptr();

    #pragma omp parallel for collapse(2) if(M >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
    for(int n=0; n<N; ++n) {
        for(int c=0; c<C; ++c) {
            for(int ho=0; ho<H_out; ++ho) {
                for(int wo=0; wo<W_out; ++wo) {
                    int out_idx = n*(C*H_out*W_out) + c*(H_out*W_out) + ho*W_out + wo;
                    uint64_t x_max = 0;

                    for(int kh=0; kh<kH; ++kh) {
                        for(int kw=0; kw<kW; ++kw) {
                            int hi = ho * stride - padding + kh;
                            int wi = wo * stride - padding + kw;
                            if(hi >= 0 && hi < H_in && wi >= 0 && wi < W_in) {
                                int in_idx = n*(C*H_in*W_in) + c*(H_in*W_in) + hi*W_in + wi;
                                uint64_t val = in_ptr[in_idx];
                                if(helper_plaintext_fp_greater(val, x_max)) x_max = val;
                            }
                        }
                    }
                    plain_max[out_idx] = x_max;
                }
            }
        }
    }

    // Step 2: Authenticate x_max
    PolyTensor pt_max = this->input({N, C, H_out, W_out}, plain_max);

    int s_0 = scale;
    int s_rest = (bitlen > (uint64_t)s_0 && digdec_k > 1) ?
                 (bitlen - s_0 + digdec_k - 2) / (digdec_k - 1) : 0;

    uint64_t table_size_0 = 1ULL << s_0;
    std::vector<uint64_t> rangeTableData_0(table_size_0);
    for (size_t i = 0; i < table_size_0; i++) rangeTableData_0[i] = i;
    RangeCheckTable rangeTable_0(rangeTableData_0);

    uint64_t table_size_rest = 1ULL << s_rest;
    std::vector<uint64_t> rangeTableData_rest(table_size_rest);
    for (size_t i = 0; i < table_size_rest; i++) rangeTableData_rest[i] = i;
    RangeCheckTable rangeTable_rest(rangeTableData_rest);

    uint64_t mask_0 = (1ULL << s_0) - 1;
    uint64_t mask_rest = (1ULL << s_rest) - 1;

    // Step 3 & 9: Range Check x_max & Reconstruct Output (Truncated)

    std::vector<uint64_t> flat_max_dec(M * digdec_k, 0);
    #pragma omp parallel for schedule(static) if(M >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
    for(size_t i = 0; i < M; ++i) {
        uint64_t abs_val = plain_max[i];
        flat_max_dec[i * digdec_k + 0] = abs_val & mask_0;
        abs_val >>= s_0;
        for(int d = 1; d < digdec_k; ++d) {
            flat_max_dec[i * digdec_k + d] = abs_val & mask_rest;
            abs_val >>= s_rest;
        }
    }

    PolyTensor X_max_recon;
    PolyTensor X_max_out;

    for (int d = 0; d < digdec_k; ++d) {
        std::vector<uint64_t> seg_data(M);
        #pragma omp parallel for schedule(static) if(M >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
        for(size_t i=0; i<M; ++i) seg_data[i] = flat_max_dec[i * digdec_k + d];

        PolyTensor auth_seg = this->input({M}, seg_data);

        if (d == 0) {
            rangeTable_0.range_check(auth_seg, "[Integrated NL] x_max range (low)");
            X_max_recon = auth_seg.clone();
        } else {
            rangeTable_rest.range_check(auth_seg, "[Integrated NL] x_max range (high)");
            uint64_t shift_recon = 1ULL << (s_0 + (d - 1) * s_rest);
            X_max_recon = X_max_recon + auth_seg * shift_recon;

            if (do_truncation) {
                uint64_t trunc_shift = 1ULL << ((d - 1) * s_rest);
                if (d == 1) X_max_out = auth_seg * trunc_shift;
                else        X_max_out = X_max_out + auth_seg * trunc_shift;
            }
        }
    }
    if (!do_truncation) X_max_out = X_max_recon.clone();

    PolyTensor::store_relation(X_max_recon, pt_max, "[Integrated NL] x_max digdec check");

    // Step 4 - 8: Loop over sub-matrix elements

    PolyTensor pt_prod = pt_max.clone();

    for (int j = 0; j < h; ++j) {
        int kh = j / kW;
        int kw = j % kW;

        PolyTensor pt_in_aligned({N, C, H_out, W_out}, pt_in.degree, PolyTensor::Coeffs);

        #pragma omp parallel for collapse(4) schedule(static) if(M >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
        for(int n=0; n<N; ++n) {
            for(int c=0; c<C; ++c) {
                for(int ho=0; ho<H_out; ++ho) {
                    for(int wo=0; wo<W_out; ++wo) {
                        int m = n*(C*H_out*W_out) + c*(H_out*W_out) + ho*W_out + wo;
                        int hi = ho * stride - padding + kh;
                        int wi = wo * stride - padding + kw;

                        if(hi >= 0 && hi < H_in && wi >= 0 && wi < W_in) {
                            int in_idx = n*(C*H_in*W_in) + c*(H_in*W_in) + hi*W_in + wi;
                            //pt_in_aligned.flat_real_vals[m] = pt_in.flat_real_vals[in_idx];
                            for(int d=0; d<=pt_in.degree; ++d) {
                                pt_in_aligned.get_coeffs_ptr(d)[m] = pt_in.get_coeffs_ptr(d)[in_idx];
                            }
                        } else {
                            //pt_in_aligned.flat_real_vals[m] = add_mod(pt_max.flat_real_vals[m], PR - 1);
                            for(int d=0; d<pt_in.degree; ++d) {
                                pt_in_aligned.get_coeffs_ptr(d)[m] = (d <= pt_max.degree) ? pt_max.get_coeffs_ptr(d)[m] : 0;
                            }
                            uint64_t max_coeff_high = (pt_in.degree <= pt_max.degree) ? pt_max.get_coeffs_ptr(pt_in.degree)[m] : 0;
                            pt_in_aligned.get_coeffs_ptr(pt_in.degree)[m] = add_mod(max_coeff_high, PR - 1);
                        }
                    }
                }
            }
        }

        PolyTensor pt_y_hat_j = pt_max - pt_in_aligned;

        std::vector<uint64_t> flat_y_hat_dec(M * digdec_k, 0);
        const uint64_t* y_hat_ptr = pt_y_hat_j.get_real_vals_ptr();
        #pragma omp parallel for schedule(static) if(M >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
        for(size_t i = 0; i < M; ++i) {
            uint64_t abs_val = y_hat_ptr[i];
            flat_y_hat_dec[i * digdec_k + 0] = abs_val & mask_0;
            abs_val >>= s_0;
            for(int d = 1; d < digdec_k; ++d) {
                flat_y_hat_dec[i * digdec_k + d] = abs_val & mask_rest;
                abs_val >>= s_rest;
            }
        }

        PolyTensor X_recon;
        for (int d = 0; d < digdec_k; ++d) {
            std::vector<uint64_t> seg_data(M);
            #pragma omp parallel for schedule(static) if(M >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
            for(size_t i=0; i<M; ++i) seg_data[i] = flat_y_hat_dec[i * digdec_k + d];

            PolyTensor auth_seg = this->input({M}, seg_data);

            if (d == 0) {
                rangeTable_0.range_check(auth_seg, "[Integrated NL] x_max - x (low)");
                X_recon = auth_seg.clone();
            } else {
                rangeTable_rest.range_check(auth_seg, "[Integrated NL] x_max - x (high)");
                uint64_t shift_recon = 1ULL << (s_0 + (d - 1) * s_rest);
                X_recon = X_recon + auth_seg * shift_recon;
            }
        }

        PolyTensor::store_relation(X_recon, pt_y_hat_j, "[Integrated NL] x_max greater than (size = from Conv)");
        pt_prod = pt_prod * pt_y_hat_j;
    }

    PolyTensor::store_zero_relation(pt_prod, "[Integrated NL] existence check");
    pt_in.mark_consumed();

    X_max_out.shape = {N, C, H_out, W_out};
    return X_max_out;
}

#endif
