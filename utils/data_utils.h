#pragma once
#include <vector>
#include <cmath>
#include <utility>
#include <functional>
#include <cstdint>
#include <cstddef>
#include <omp.h>
#include "../config.h"
#include "logger.h"
#include "emp-zk/emp-vole/utility.h"

inline void hash_combine(size_t& seed, size_t value) {
    seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <typename T>
inline size_t compute_vector_hash(const std::vector<T>& vec) {
    size_t seed = 0;
    std::hash<T> hasher;
    for (const auto& item : vec) {
        hash_combine(seed, hasher(item));
    }
    return seed;
}

inline size_t compute_vector_hash(const std::vector<std::pair<uint64_t, uint64_t>>& vec) {
    size_t seed = 0;
    std::hash<uint64_t> hasher;
    for (const auto& item : vec) {
        hash_combine(seed, hasher(item.first));
        hash_combine(seed, hasher(item.second));
    }
    return seed;
}

inline void print_u128(__uint128_t n) {
    if (n == 0) {
        std::cout << "0";
        return;
    }

    std::string s;
    while (n > 0) {
        s += (char)('0' + (n % 10));
        n /= 10;
    }
    std::reverse(s.begin(), s.end());
    std::cout << s;
}

inline uint64_t real2fp(double x) {
    const double scale_factor = static_cast<double>(1ULL << MVZK_CONFIG_SCALE);

    int64_t val = std::round(x * scale_factor);

    int64_t rem = val % (int64_t)PR;
    if (rem < 0) {
        rem += (int64_t)PR;
    }
    return (uint64_t)rem;
}

inline std::vector<uint64_t> real2fp(const std::vector<float>& input) {
    size_t n = input.size();

    std::vector<uint64_t> output(n);

    const double scale_factor = static_cast<double>(1ULL << MVZK_CONFIG_SCALE);

    const int64_t signed_PR = (int64_t)PR;

    const float* __restrict__ in_ptr = input.data();
    uint64_t* __restrict__ out_ptr = output.data();

    #pragma omp parallel for schedule(static) if(n >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
    for (size_t i = 0; i < n; ++i) {
        double raw_val = (double)in_ptr[i] * scale_factor;
        int64_t val = std::round(raw_val);

        int64_t rem = val % signed_PR;

        if (rem < 0) {
            rem += signed_PR;
        }

        out_ptr[i] = (uint64_t)rem;
    }

    return output;
}

template <typename T> size_t get_total_elements(const T& val) { return 1; }
template <typename T> size_t get_total_elements(const std::vector<T>& vec) {
    if (vec.empty()) return 0;
    return vec.size() * get_total_elements(vec[0]);
}

template <typename T> void flatten_recursive(const T& val, std::vector<float>& buffer) {
    buffer.push_back(static_cast<float>(val));
}
template <typename T> void flatten_recursive(const std::vector<T>& vec, std::vector<float>& buffer) {
    for (const auto& item : vec) flatten_recursive(item, buffer);
}

template <typename T>
inline std::vector<uint64_t> real2fp_ndim(const std::vector<T>& input) {
    size_t total = get_total_elements(input);
    std::vector<float> flat_floats;
    flat_floats.reserve(total);
    flatten_recursive(input, flat_floats);

    return real2fp(flat_floats);
}