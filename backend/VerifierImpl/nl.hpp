#ifndef MVZK_BACKEND_VERIFIER_IMPL_NL_HPP__
#define MVZK_BACKEND_VERIFIER_IMPL_NL_HPP__

#include "../../exec/ExecVerifier.h"

template <typename IO>
PolyTensor MVZKExecVerifier<IO>::relu(PolyTensor& x, uint64_t bitlen, uint64_t digdec_k, bool do_truncation, uint64_t scale) {
    size_t n = x.total_elements;

    int s_0 = scale;
    int s_rest = (bitlen > (uint64_t)s_0 && digdec_k > 1) ?
                 (bitlen - s_0 + digdec_k - 2) / (digdec_k - 1) : 0;

    std::vector<uint64_t> dummy_zeros(n, 0);

    PolyTensor authenticated_bq = input(x.shape, dummy_zeros);
    PolyTensor b_Q_check = authenticated_bq * (authenticated_bq - 1);
    PolyTensor::store_zero_relation(b_Q_check);

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
        PolyTensor authenticated_seg_j = input(x.shape, dummy_zeros);

        if (j == 0) {
            rangeTable_0.range_check(authenticated_seg_j);
            X_recon = authenticated_seg_j.clone();
        } else {
            rangeTable_rest.range_check(authenticated_seg_j);

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

    PolyTensor term1 = authenticated_bq * (X_recon - x);
    PolyTensor term2 = (1 - authenticated_bq) * (X_recon + x);

    PolyTensor consistency = term1 + term2;
    PolyTensor::store_zero_relation(consistency);

    PolyTensor result = authenticated_bq * X_out;

    return result;
}


template <typename IO>
PolyTensor MVZKExecVerifier<IO>::relu_legacy(PolyTensor& x, uint64_t bitlen, uint64_t digdec_k, bool do_truncation) {
    size_t n = x.total_elements;
    int s = (bitlen + digdec_k - 1) / digdec_k;
    std::vector<uint64_t> dummy_zeros(n, 0);

    PolyTensor authenticated_bq = input(x.shape, dummy_zeros);
    PolyTensor b_Q_check = authenticated_bq * (authenticated_bq - 1);
    PolyTensor::store_zero_relation(b_Q_check);

    uint64_t table_size = 1ULL << s;
    std::vector<uint64_t> rangeTableData(table_size);
    for (size_t i = 0; i < table_size; i++) rangeTableData[i] = i;
    RangeCheckTable rangeTable(rangeTableData);

    PolyTensor X_recon;
    PolyTensor X_out;

    for (int j = 0; j < digdec_k; ++j) {
        PolyTensor authenticated_seg_j = input(x.shape, dummy_zeros);
        rangeTable.range_check(authenticated_seg_j);

        uint64_t shift_scale = 1ULL << (j * s);
        if (j == 0) X_recon = authenticated_seg_j * shift_scale;
        else        X_recon = X_recon + authenticated_seg_j * shift_scale;

        if (do_truncation && j > 0) {
            uint64_t trunc_shift = 1ULL << ((j - 1) * s);
            if (j == 1) X_out = authenticated_seg_j * trunc_shift;
            else        X_out = X_out + authenticated_seg_j * trunc_shift;
        }
    }

    if (!do_truncation) {
        X_out = X_recon.clone();
    }

    PolyTensor term1 = authenticated_bq * (X_recon - x);
    PolyTensor term2 = (1 - authenticated_bq) * (X_recon + x);

    PolyTensor consistency = term1 + term2;
    PolyTensor::store_zero_relation(consistency);

    PolyTensor result = authenticated_bq * X_out;

    return result;
}


template <typename IO>
PolyTensor MVZKExecVerifier<IO>::maxpool2d(PolyTensor& pt_in, int kernel_size, int stride, int padding, uint64_t bitlen, uint64_t digdec_k, uint64_t scale) {
    int N = pt_in.shape[0], C = pt_in.shape[1], H_in = pt_in.shape[2], W_in = pt_in.shape[3];
    int kH = kernel_size, kW = kernel_size;
    int H_out = (H_in + 2 * padding - kH) / stride + 1;
    int W_out = (W_in + 2 * padding - kW) / stride + 1;

    int M = N * C * H_out * W_out;
    int h = kH * kW;

    std::vector<uint64_t> dummy_max(M, 0);
    PolyTensor pt_max = this->input({N, C, H_out, W_out}, dummy_max);

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

        PolyTensor pt_in_aligned({N, C, H_out, W_out}, pt_in.degree, PolyTensor::Keys);

        #pragma omp parallel for collapse(4) schedule(static) if(M>= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
        for(int n=0; n<N; ++n) {
            for(int c=0; c<C; ++c) {
                for(int ho=0; ho<H_out; ++ho) {
                    for(int wo=0; wo<W_out; ++wo) {
                        int m = n*(C*H_out*W_out) + c*(H_out*W_out) + ho*W_out + wo;
                        int hi = ho * stride - padding + kh;
                        int wi = wo * stride - padding + kw;

                        if(hi >= 0 && hi < H_in && wi >= 0 && wi < W_in) {
                            int in_idx = n*(C*H_in*W_in) + c*(H_in*W_in) + hi*W_in + wi;
                            pt_in_aligned.flat_keys[m] = pt_in.flat_keys[in_idx];
                        } else {
                            pt_in_aligned.flat_keys[m] = add_mod(pt_max.flat_keys[m], PR - delta_pow(this->delta, pt_in.degree));
                        }
                    }
                }
            }
        }

        PolyTensor pt_y_hat_j = pt_max - pt_in_aligned;

        // --- Upper Bound Check ---
        PolyTensor X_recon;

        for (int d = 0; d < digdec_k; ++d) {
            PolyTensor auth_seg = this->input({M}, dummy_max);

            if (d == 0) {
                rangeTable_0.range_check(auth_seg);
                X_recon = auth_seg.clone();
            } else {
                rangeTable_rest.range_check(auth_seg);
                uint64_t shift_recon = 1ULL << (s_0 + (d - 1) * s_rest);
                X_recon = X_recon + auth_seg * shift_recon;
            }
        }

        PolyTensor::store_relation(X_recon, pt_y_hat_j);

        // --- Existence Check ---
        if (j == 0) {
            pt_prod = pt_y_hat_j.clone();
            pt_y_hat_j.is_consumed = true;
        } else {
            pt_prod = pt_prod * pt_y_hat_j;
        }
    }

    PolyTensor::store_zero_relation(pt_prod);

    pt_in.mark_consumed();

    return pt_max;
}


template <typename IO>
PolyTensor MVZKExecVerifier<IO>::maxpool2d_legacy(PolyTensor& pt_in, int kernel_size, int stride, int padding, uint64_t bitlen, uint64_t digdec_k) {
    int N = pt_in.shape[0], C = pt_in.shape[1], H_in = pt_in.shape[2], W_in = pt_in.shape[3];
    int kH = kernel_size, kW = kernel_size;
    int H_out = (H_in + 2 * padding - kH) / stride + 1;
    int W_out = (W_in + 2 * padding - kW) / stride + 1;

    int M = N * C * H_out * W_out;
    int h = kH * kW;

    std::vector<uint64_t> dummy_max(M, 0);
    PolyTensor pt_max = this->input({N, C, H_out, W_out}, dummy_max);

    int s = (bitlen + digdec_k - 1) / digdec_k;
    uint64_t table_size = 1ULL << s;
    std::vector<uint64_t> rangeTableData(table_size);
    for (size_t i = 0; i < table_size; i++) rangeTableData[i] = i;
    RangeCheckTable rangeTable(rangeTableData);

    PolyTensor pt_prod;

    for (int j = 0; j < h; ++j) {
        int kh = j / kW;
        int kw = j % kW;

        PolyTensor pt_in_aligned({N, C, H_out, W_out}, pt_in.degree, PolyTensor::Keys);

        #pragma omp parallel for collapse(4) schedule(static) if(M>= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
        for(int n=0; n<N; ++n) {
            for(int c=0; c<C; ++c) {
                for(int ho=0; ho<H_out; ++ho) {
                    for(int wo=0; wo<W_out; ++wo) {
                        int m = n*(C*H_out*W_out) + c*(H_out*W_out) + ho*W_out + wo;
                        int hi = ho * stride - padding + kh;
                        int wi = wo * stride - padding + kw;

                        if(hi >= 0 && hi < H_in && wi >= 0 && wi < W_in) {
                            int in_idx = n*(C*H_in*W_in) + c*(H_in*W_in) + hi*W_in + wi;
                            pt_in_aligned.flat_keys[m] = pt_in.flat_keys[in_idx];
                        } else {
                            // Padding: K_in = K_max - Delta^(pt_in.degree)

                            pt_in_aligned.flat_keys[m] = add_mod(pt_max.flat_keys[m], PR - delta_pow(this->delta, pt_in.degree));
                        }
                    }
                }
            }
        }

        PolyTensor pt_y_hat_j = pt_max - pt_in_aligned;

        // --- Upper Bound Check ---
        PolyTensor X_recon;

        for (int d = 0; d < digdec_k; ++d) {
            PolyTensor auth_seg = this->input({M}, dummy_max);
            rangeTable.range_check(auth_seg);

            uint64_t shift_scale = 1ULL << (d * s);
            if (d == 0) X_recon = auth_seg * shift_scale;
            else        X_recon = X_recon + auth_seg * shift_scale;
        }

        //PolyTensor pt_y_for_bound = pt_y_hat_j.clone();
        PolyTensor::store_relation(X_recon, pt_y_hat_j);

        // --- Existence Check ---
        if (j == 0) {
            pt_prod = pt_y_hat_j.clone();
            pt_y_hat_j.is_consumed = true;
        } else {
            pt_prod = pt_prod * pt_y_hat_j;
        }
    }

    PolyTensor::store_zero_relation(pt_prod);

    pt_in.mark_consumed();

    return pt_max;
}


template <typename IO>
PolyTensor MVZKExecVerifier<IO>::integrated_nl(PolyTensor& pt_in, int kernel_size, int stride, int padding, uint64_t bitlen, uint64_t digdec_k, bool do_truncation, uint64_t scale) {
    int N = pt_in.shape[0], C = pt_in.shape[1], H_in = pt_in.shape[2], W_in = pt_in.shape[3];
    int kH = kernel_size, kW = kernel_size;
    int H_out = (H_in + 2 * padding - kH) / stride + 1;
    int W_out = (W_in + 2 * padding - kW) / stride + 1;

    int M = N * C * H_out * W_out;
    int h = kH * kW;

    std::vector<uint64_t> dummy_max(M, 0);
    PolyTensor pt_max = this->input({N, C, H_out, W_out}, dummy_max);

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

    // Step 3 & 9: Range Check x_max & Reconstruct Output

    PolyTensor X_max_recon;
    PolyTensor X_max_out;

    for (int d = 0; d < digdec_k; ++d) {
        PolyTensor auth_seg = this->input({M}, dummy_max);

        if (d == 0) {
            rangeTable_0.range_check(auth_seg);
            X_max_recon = auth_seg.clone();
        } else {
            rangeTable_rest.range_check(auth_seg);
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

    PolyTensor::store_relation(X_max_recon, pt_max);

    // Step 4 - 8: Loop over sub-matrix elements

    PolyTensor pt_prod = pt_max.clone();

    for (int j = 0; j < h; ++j) {
        int kh = j / kW;
        int kw = j % kW;

        PolyTensor pt_in_aligned({N, C, H_out, W_out}, pt_in.degree, PolyTensor::Keys);

        #pragma omp parallel for collapse(4) schedule(static) if(M>= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
        for(int n=0; n<N; ++n) {
            for(int c=0; c<C; ++c) {
                for(int ho=0; ho<H_out; ++ho) {
                    for(int wo=0; wo<W_out; ++wo) {
                        int m = n*(C*H_out*W_out) + c*(H_out*W_out) + ho*W_out + wo;
                        int hi = ho * stride - padding + kh;
                        int wi = wo * stride - padding + kw;

                        if(hi >= 0 && hi < H_in && wi >= 0 && wi < W_in) {
                            int in_idx = n*(C*H_in*W_in) + c*(H_in*W_in) + hi*W_in + wi;
                            pt_in_aligned.flat_keys[m] = pt_in.flat_keys[in_idx];
                        } else {
                            pt_in_aligned.flat_keys[m] = add_mod(pt_max.flat_keys[m], PR - delta_pow(this->delta, pt_in.degree));
                        }
                    }
                }
            }
        }

        PolyTensor pt_y_hat_j = pt_max - pt_in_aligned;
        PolyTensor X_recon;

        for (int d = 0; d < digdec_k; ++d) {
            PolyTensor auth_seg = this->input({M}, dummy_max);

            if (d == 0) {
                rangeTable_0.range_check(auth_seg);
                X_recon = auth_seg.clone();
            } else {
                rangeTable_rest.range_check(auth_seg);
                uint64_t shift_recon = 1ULL << (s_0 + (d - 1) * s_rest);
                X_recon = X_recon + auth_seg * shift_recon;
            }
        }

        PolyTensor::store_relation(X_recon, pt_y_hat_j);
        pt_prod = pt_prod * pt_y_hat_j;
    }

    PolyTensor::store_zero_relation(pt_prod);
    pt_in.mark_consumed();

    X_max_out.shape = {N, C, H_out, W_out};
    return X_max_out;
}


template <typename IO>
PolyTensor MVZKExecVerifier<IO>::integrated_nl_legacy(PolyTensor& pt_in, int kernel_size, int stride, int padding, uint64_t bitlen, uint64_t digdec_k, bool do_truncation) {
    int N = pt_in.shape[0], C = pt_in.shape[1], H_in = pt_in.shape[2], W_in = pt_in.shape[3];
    int kH = kernel_size, kW = kernel_size;
    int H_out = (H_in + 2 * padding - kH) / stride + 1;
    int W_out = (W_in + 2 * padding - kW) / stride + 1;

    int M = N * C * H_out * W_out;
    int h = kH * kW;

    std::vector<uint64_t> dummy_max(M, 0);
    PolyTensor pt_max = this->input({N, C, H_out, W_out}, dummy_max);

    int s = (bitlen + digdec_k - 1) / digdec_k;
    uint64_t table_size = 1ULL << s;
    std::vector<uint64_t> rangeTableData(table_size);
    for (size_t i = 0; i < table_size; i++) rangeTableData[i] = i;
    RangeCheckTable rangeTable(rangeTableData);

    // Step 3 & 9: Range Check x_max & Reconstruct Output

    PolyTensor X_max_recon;
    PolyTensor X_max_out;

    for (int d = 0; d < digdec_k; ++d) {
        PolyTensor auth_seg = this->input({M}, dummy_max);
        rangeTable.range_check(auth_seg);

        uint64_t shift_scale = 1ULL << (d * s);
        if (d == 0) X_max_recon = auth_seg * shift_scale;
        else        X_max_recon = X_max_recon + auth_seg * shift_scale;

        if (do_truncation && d > 0) {
            uint64_t trunc_shift = 1ULL << ((d - 1) * s);
            if (d == 1) X_max_out = auth_seg * trunc_shift;
            else        X_max_out = X_max_out + auth_seg * trunc_shift;
        }
    }
    if (!do_truncation) X_max_out = X_max_recon.clone();

    PolyTensor::store_relation(X_max_recon, pt_max);

    // Step 4 - 8: Loop over sub-matrix elements

    PolyTensor pt_prod = pt_max.clone(); // d_0 = x_max

    for (int j = 0; j < h; ++j) {
        int kh = j / kW;
        int kw = j % kW;

        PolyTensor pt_in_aligned({N, C, H_out, W_out}, pt_in.degree, PolyTensor::Keys);

        #pragma omp parallel for collapse(4) schedule(static)
        for(int n=0; n<N; ++n) {
            for(int c=0; c<C; ++c) {
                for(int ho=0; ho<H_out; ++ho) {
                    for(int wo=0; wo<W_out; ++wo) {
                        int m = n*(C*H_out*W_out) + c*(H_out*W_out) + ho*W_out + wo;
                        int hi = ho * stride - padding + kh;
                        int wi = wo * stride - padding + kw;

                        if(hi >= 0 && hi < H_in && wi >= 0 && wi < W_in) {
                            int in_idx = n*(C*H_in*W_in) + c*(H_in*W_in) + hi*W_in + wi;
                            pt_in_aligned.flat_keys[m] = pt_in.flat_keys[in_idx];
                        } else {
                            pt_in_aligned.flat_keys[m] = add_mod(pt_max.flat_keys[m], PR - delta_pow(this->delta, pt_in.degree));
                        }
                    }
                }
            }
        }

        PolyTensor pt_y_hat_j = pt_max - pt_in_aligned;
        PolyTensor X_recon;

        for (int d = 0; d < digdec_k; ++d) {
            PolyTensor auth_seg = this->input({M}, dummy_max);
            rangeTable.range_check(auth_seg);

            uint64_t shift_scale = 1ULL << (d * s);
            if (d == 0) X_recon = auth_seg * shift_scale;
            else        X_recon = X_recon + auth_seg * shift_scale;
        }

        PolyTensor::store_relation(X_recon, pt_y_hat_j);
        pt_prod = pt_prod * pt_y_hat_j;
    }

    PolyTensor::store_zero_relation(pt_prod);
    pt_in.mark_consumed();

    X_max_out.shape = {N, C, H_out, W_out};
    return X_max_out;
}

#endif
