#pragma once

#include "../data_type/PolyDelta.h"
#include "../data_type/PolyTensor.h"
#include "../utility.h"
#include "../config.h"

PolyTensor ReLU(PolyTensor& x, uint64_t bitlen = MVZK_CONFIG_DEFAULT_BITLEN,
    uint64_t digdec_k = MVZK_CONFIG_NON_LINEAR_RELU_DIGDEC_K,
    bool do_truncation = MVZK_CONFIG_NON_LINEAR_RELU_DO_TRUNCATION,
    uint64_t scale = MVZK_CONFIG_SCALE);

PolyTensor MaxPool2D(PolyTensor& x,
    int kernel_size,
    int stride,
    int padding,
    uint64_t bitlen = MVZK_CONFIG_DEFAULT_BITLEN,
    uint64_t digdec_k = MVZK_CONFIG_NON_LINEAR_RELU_DIGDEC_K,
    uint64_t scale = MVZK_CONFIG_SCALE);

PolyTensor IntegratedNL(PolyTensor& x,
    int kernel_size,
    int stride,
    int padding,
    uint64_t bitlen = MVZK_CONFIG_DEFAULT_BITLEN,
    uint64_t digdec_k = MVZK_CONFIG_NON_LINEAR_RELU_DIGDEC_K,
    bool do_truncation = MVZK_CONFIG_NON_LINEAR_RELU_DO_TRUNCATION,
    uint64_t scale = MVZK_CONFIG_SCALE);

// Plaintext helper functions for reference computations.

/**
 * @brief Plaintext digit decomposition.
 * Splits each l-bit input into k segments. If l is not divisible by k, the
 * representation is treated as zero-padded at the high bits.
 *
 * @param src Input vector of length N.
 * @param k Number of decomposition segments.
 * @param l Effective input bit length.
 * @return Flattened output of length N * k.
 * Layout: [x0_seg0, ..., x0_seg{k-1}, x1_seg0, ...]
 */
inline std::vector<uint64_t> helper_plaintext_digit_decomposition(
    const std::vector<uint64_t>& src,
    int k,
    int l = MVZK_CONFIG_DEFAULT_BITLEN
) {
    if (k <= 0) return {};
    size_t n = src.size();

    // Number of bits in each segment, rounded up.
    int bits_per_seg = (l + k - 1) / k;

    // Mask for one segment.
    uint64_t mask = (bits_per_seg >= 64) ? 0xFFFFFFFFFFFFFFFFULL : ((1ULL << bits_per_seg) - 1);

    std::vector<uint64_t> output(n * k);
    const uint64_t* __restrict__ in_ptr = src.data();
    uint64_t* __restrict__ out_ptr = output.data();

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        uint64_t val = in_ptr[i];

        for (int j = 0; j < k; ++j) {
            int shift = j * bits_per_seg;
            uint64_t segment = 0;

            // Shifts beyond the uint64_t width contribute zero.
            if (shift < 64) {
                segment = (val >> shift) & mask;
            }

            out_ptr[i * k + j] = segment;
        }
    }

    return output;
}

/**
 * @brief Plaintext sign indicator for field-encoded fixed-point values.
 *
 * Negative values map to 0 and non-negative values map to 1.
 */
inline std::vector<uint64_t> helper_plaintext_sign(
    const std::vector<uint64_t>& src,
    uint64_t zero_point = 0
) {
    size_t n = src.size();
    std::vector<uint64_t> output(n);

    // Default signed-field zero point.
    if (zero_point == 0) zero_point = PR >> 1;

    const uint64_t* __restrict__ in_ptr = src.data();
    uint64_t* __restrict__ out_ptr = output.data();

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        if (in_ptr[i] > zero_point) {
            out_ptr[i] = 0; // Negative
        } else {
            out_ptr[i] = 1; // Positive
        }
    }
    return output;
}

inline bool helper_plaintext_fp_greater(uint64_t a, uint64_t b, uint64_t zero_point = 0) {
    if (zero_point == 0) zero_point = PR >> 1;

    bool a_is_pos = (a <= zero_point);
    bool b_is_pos = (b <= zero_point);

    // Positive values are greater than negative values.
    if (a_is_pos && !b_is_pos) return true;
    if (!a_is_pos && b_is_pos) return false;

    // Values with the same sign can be compared directly in this encoding.
    return a > b;
}
