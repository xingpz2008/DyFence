#ifndef MVZK_BACKEND_VERIFIER_IMPL_LINEAR_HPP__
#define MVZK_BACKEND_VERIFIER_IMPL_LINEAR_HPP__

#include "../../exec/ExecVerifier.h"

template <typename IO>
void MVZKExecVerifier<IO>::permute_and_add_bias(
    const PolyTensor& src, PolyTensor& dst, const PolyTensor& bias,
    int N, int H_out, int W_out, int C_out
) {
    bool has_bias = (bias.total_elements > 0);

    if(src.degree < bias.degree){
        LOG_WARN("Non-implemented with bias.deg > WX.deg at Conv2D!");
    }

    if (!src.flat_keys.empty()) {
        const uint64_t* b_ptr = has_bias ? bias.get_keys_ptr() : nullptr;

        uint64_t scale = 1;

        if (has_bias && src.degree > bias.degree) {
            int offset = src.degree - bias.degree;
            scale = delta_pow(this->delta, offset);
        }

        permute_and_add_bias_kernel(
            src.get_keys_ptr(), dst.get_keys_ptr(), b_ptr,
            scale,
            N, H_out, W_out, C_out
        );
    }

    // Update bias flag
    src.is_consumed = true;
    bias.is_consumed = true;
    dst.is_consumed = false;
}


template <typename IO>
void MVZKExecVerifier<IO>::helper_linear_add_bias(
    PolyTensor& data, const PolyTensor& bias, int Rows, int Cols
) {
    bool has_bias = (bias.total_elements > 0);

    if (!data.flat_keys.empty()) {
        const uint64_t* b_ptr = has_bias ? bias.get_keys_ptr() : nullptr;

        uint64_t scale = 1;
        if (has_bias && data.degree > bias.degree) {
            int offset = data.degree - bias.degree;
            scale = delta_pow(this->delta, offset);
        }

        if (b_ptr){
            add_bias_row_broadcast_kernel(data.get_keys_ptr(), b_ptr, scale, Rows, Cols);
        }
    }
}


template <typename IO>
void MVZKExecVerifier<IO>::helper_permute_and_add_bias_1d(
    const PolyTensor& src, PolyTensor& dst, const PolyTensor& bias,
    int N, int L_out, int C_out
) {
    bool has_bias = (bias.total_elements > 0);

    if (!src.flat_keys.empty()) {
        const uint64_t* b_ptr = has_bias ? bias.get_keys_ptr() : nullptr;

        uint64_t scale = 1;
        if (has_bias && src.degree > bias.degree) {
            int offset = src.degree - bias.degree;
            scale = delta_pow(this->delta, offset);
        }

        permute_and_add_bias_1d_kernel(
            src.get_keys_ptr(), dst.get_keys_ptr(), b_ptr,
            scale,
            N, L_out, C_out
        );
    }
    src.is_consumed = true;
}


template <typename IO>
PolyTensor MVZKExecVerifier<IO>::avgpool2d(PolyTensor& pt_in, int kernel_size, int stride, int padding, bool back_to_sum_pool) {
    int N = pt_in.shape[0], C = pt_in.shape[1], H_in = pt_in.shape[2], W_in = pt_in.shape[3];
    int H_out = (H_in + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W_in + 2 * padding - kernel_size) / stride + 1;

    PolyTensor pt_out({N, C, H_out, W_out}, pt_in.degree, PolyTensor::Keys);

    uint64_t pool_area = (uint64_t)(kernel_size * kernel_size);
    uint64_t inv_area = pow_mod(pool_area, PR - 2);

    const uint64_t* in_keys = pt_in.get_keys_ptr();
    uint64_t* out_keys = pt_out.get_keys_ptr();

    #pragma omp parallel for collapse(4)
    for(int n=0; n<N; ++n) {
        for(int c=0; c<C; ++c) {
            for(int ho=0; ho<H_out; ++ho) {
                for(int wo=0; wo<W_out; ++wo) {
                    int out_idx = n*(C*H_out*W_out) + c*(H_out*W_out) + ho*W_out + wo;

                    uint64_t sum_keys = 0;

                    for(int kh=0; kh<kernel_size; ++kh) {
                        for(int kw=0; kw<kernel_size; ++kw) {
                            int hi = ho * stride - padding + kh;
                            int wi = wo * stride - padding + kw;

                            if(hi >= 0 && hi < H_in && wi >= 0 && wi < W_in) {
                                int in_idx = n*(C*H_in*W_in) + c*(H_in*W_in) + hi*W_in + wi;
                                sum_keys = add_mod(sum_keys, in_keys[in_idx]);
                            }
                        }
                    }
                    if (back_to_sum_pool == false){
                        out_keys[out_idx] = mult_mod(sum_keys, inv_area);
                    } else {
                        out_keys[out_idx] = sum_keys;
                    }
                }
            }
        }
    }

    pt_in.mark_consumed();
    return pt_out;
}


template <typename IO>
PolyTensor MVZKExecVerifier<IO>::batchnorm2d(PolyTensor& pt_in, const std::vector<uint64_t>& A, const std::vector<uint64_t>& B) {
    int N = pt_in.shape[0], C = pt_in.shape[1], H = pt_in.shape[2], W = pt_in.shape[3];
    PolyTensor pt_out({N, C, H, W}, pt_in.degree, PolyTensor::Keys);

    const uint64_t* in_keys = pt_in.get_keys_ptr();
    uint64_t* out_keys = pt_out.get_keys_ptr();

    #pragma omp parallel for collapse(2)
    for(int n=0; n<N; ++n) {
        for(int c=0; c<C; ++c) {
            uint64_t a_val = A[c];
            uint64_t b_val = B[c];

            uint64_t b_scaled = mult_mod(b_val, delta_pow(this->delta, pt_in.degree));

            for(int h=0; h<H; ++h) {
                for(int w=0; w<W; ++w) {
                    size_t idx = ((size_t)n * C + c) * H * W + h * W + w;
                    out_keys[idx] = add_mod(mult_mod(in_keys[idx], a_val), b_scaled);
                }
            }
        }
    }

    pt_in.mark_consumed();
    return pt_out;
}

#endif
