#ifndef MVZK_BACKEND_PROVER_IMPL_LINEAR_HPP__
#define MVZK_BACKEND_PROVER_IMPL_LINEAR_HPP__

#include "../../exec/ExecProver.h"

template <typename IO>
void MVZKExecProver<IO>::permute_and_add_bias(
    const PolyTensor& src, PolyTensor& dst, const PolyTensor& bias,
    int N, int H_out, int W_out, int C_out
) {
    bool has_bias = (bias.total_elements > 0);

    if (!src.flat_coeffs.empty()) {
        int offset = 0;

        if (src.degree >= bias.degree) {
            offset = src.degree - bias.degree;
        } else{
            LOG_WARN("Non-implemented with bias.deg > WX.deg at Conv2D!");
        }

        for (int d = 0; d <= src.degree; ++d) {
            const uint64_t* b_ptr = nullptr;

            int d_bias = d - offset;

            if (has_bias && d_bias >= 0 && d_bias <= bias.degree) {
                b_ptr = bias.get_coeffs_ptr(d_bias);
            }

            permute_and_add_bias_kernel(
                src.get_coeffs_ptr(d), dst.get_coeffs_ptr(d), b_ptr,
                1, // scale = 1
                N, H_out, W_out, C_out
            );
        }
    }

    src.is_consumed = true;
    bias.is_consumed = true;
    dst.is_consumed = false;
}


template <typename IO>
void MVZKExecProver<IO>::helper_linear_add_bias(
    PolyTensor& data, const PolyTensor& bias, int Rows, int Cols
) {
    bool has_bias = (bias.total_elements > 0);

    if (!data.flat_coeffs.empty()) {
        int offset = 0;
        if (data.degree >= bias.degree) {
            offset = data.degree - bias.degree;
        } else {
            LOG_WARN("Linear: bias.deg > data.deg");
        }

        for (int d = 0; d <= data.degree; ++d) {
            const uint64_t* b_ptr = nullptr;

            // d_data = d_bias + offset
            int d_bias = d - offset;

            if (has_bias && d_bias >= 0 && d_bias <= bias.degree) {
                b_ptr = bias.get_coeffs_ptr(d_bias);
            }

            if (b_ptr){
                // If b_ptr is null, it means that this deg has been shifted.
                add_bias_row_broadcast_kernel(data.get_coeffs_ptr(d), b_ptr, 1, Rows, Cols);
            }
        }
    }
}


template <typename IO>
void MVZKExecProver<IO>::helper_permute_and_add_bias_1d(
    const PolyTensor& src, PolyTensor& dst, const PolyTensor& bias,
    int N, int L_out, int C_out
) {
    bool has_bias = (bias.total_elements > 0);

    if (!src.flat_coeffs.empty()) {
        int offset = 0;
        if (src.degree >= bias.degree) {
            offset = src.degree - bias.degree;
        } else {
        }

        for (int d = 0; d <= src.degree; ++d) {
            const uint64_t* b_ptr = nullptr;

            // d_src = d_bias + offset  =>  d_bias = d - offset
            int d_bias = d - offset;

            if (has_bias && d_bias >= 0 && d_bias <= bias.degree) {
                b_ptr = bias.get_coeffs_ptr(d_bias);
            }

            permute_and_add_bias_1d_kernel(
                src.get_coeffs_ptr(d), dst.get_coeffs_ptr(d), b_ptr,
                1,
                N, L_out, C_out);
        }
    }
    src.is_consumed = true;
}


template <typename IO>
PolyTensor MVZKExecProver<IO>::avgpool2d(PolyTensor& pt_in, int kernel_size, int stride, int padding, bool back_to_sum_pool) {
    int N = pt_in.shape[0], C = pt_in.shape[1], H_in = pt_in.shape[2], W_in = pt_in.shape[3];
    int H_out = (H_in + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W_in + 2 * padding - kernel_size) / stride + 1;

    PolyTensor pt_out({N, C, H_out, W_out}, pt_in.degree, PolyTensor::Coeffs);

    uint64_t pool_area = (uint64_t)(kernel_size * kernel_size);
    uint64_t inv_area = pow_mod(pool_area, PR - 2);

    size_t total_elements = (size_t)N * C * H_out * W_out;

    #pragma omp parallel for collapse(4) if(total_elements >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
    for(int n=0; n<N; ++n) {
        for(int c=0; c<C; ++c) {
            for(int ho=0; ho<H_out; ++ho) {
                for(int wo=0; wo<W_out; ++wo) {
                    int out_idx = n*(C*H_out*W_out) + c*(H_out*W_out) + ho*W_out + wo;

                    std::vector<uint64_t> sum_coeffs(pt_in.degree + 1, 0);

                    for(int kh=0; kh<kernel_size; ++kh) {
                        for(int kw=0; kw<kernel_size; ++kw) {
                            int hi = ho * stride - padding + kh;
                            int wi = wo * stride - padding + kw;

                            if(hi >= 0 && hi < H_in && wi >= 0 && wi < W_in) {
                                int in_idx = n*(C*H_in*W_in) + c*(H_in*W_in) + hi*W_in + wi;

                                for(int d=0; d<=pt_in.degree; ++d) {
                                    sum_coeffs[d] = add_mod(sum_coeffs[d], pt_in.get_coeffs_ptr(d)[in_idx]);
                                }
                            }
                        }
                    }

                    if (back_to_sum_pool == false){
                        for(int d=0; d<=pt_in.degree; ++d) {
                            pt_out.get_coeffs_ptr(d)[out_idx] = mult_mod(sum_coeffs[d], inv_area);
                        }
                    } else {
                        for(int d=0; d<=pt_in.degree; ++d) {
                            pt_out.get_coeffs_ptr(d)[out_idx] = sum_coeffs[d];
                        }
                    }
                }
            }
        }
    }

    pt_in.mark_consumed();
    return pt_out;
}


template <typename IO>
PolyTensor MVZKExecProver<IO>::batchnorm2d(PolyTensor& pt_in, const std::vector<uint64_t>& A, const std::vector<uint64_t>& B) {
    int N = pt_in.shape[0], C = pt_in.shape[1], H = pt_in.shape[2], W = pt_in.shape[3];

    if (A.size() != (size_t)C || B.size() != (size_t)C) {
        LOG_ERROR("BatchNorm2D Scale/Shift vector size mismatch!");
        exit(-1);
    }

    PolyTensor pt_out({N, C, H, W}, pt_in.degree, PolyTensor::Coeffs);

    const uint64_t* in_real = pt_in.get_real_vals_ptr();
    uint64_t* out_real = pt_out.get_real_vals_ptr();

    int max_deg = pt_in.degree;

    std::vector<const uint64_t*> in_coeffs(max_deg + 1);
    std::vector<uint64_t*> out_coeffs(max_deg + 1);
    for(int d = 0; d <= max_deg; ++d) {
        in_coeffs[d] = pt_in.get_coeffs_ptr(d);
        out_coeffs[d] = pt_out.get_coeffs_ptr(d);
    }

    #pragma omp parallel for collapse(2)
    for(int n=0; n<N; ++n) {
        for(int c=0; c<C; ++c) {
            uint64_t a_val = A[c];
            uint64_t b_val = B[c];

            for(int h=0; h<H; ++h) {
                for(int w=0; w<W; ++w) {
                    size_t idx = ((size_t)n * C + c) * H * W + h * W + w;

                    out_real[idx] = add_mod(mult_mod(in_real[idx], a_val), b_val);

                    for(int d=0; d<=max_deg; ++d) {
                        uint64_t coeff = mult_mod(in_coeffs[d][idx], a_val);

                        if (d == max_deg) {
                            coeff = add_mod(coeff, b_val);
                        }
                        out_coeffs[d][idx] = coeff;
                    }
                }
            }
        }
    }

    pt_in.mark_consumed();
    return pt_out;
}

#endif
