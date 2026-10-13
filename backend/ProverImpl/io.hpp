#ifndef MVZK_BACKEND_PROVER_IMPL_IO_HPP__
#define MVZK_BACKEND_PROVER_IMPL_IO_HPP__

#include "../../exec/ExecProver.h"

template <typename IO>
void MVZKExecProver<IO>::input(std::vector<PolyDelta>& pdList, const std::vector<uint64_t>& raw_data) {
    size_t len = raw_data.size();

    pdList.resize(len);

    std::vector<__uint128_t> vole_returned(len);
    std::vector<uint64_t> lam(len);

    vole->extend(vole_returned.data(), len);

    for (size_t i = 0; i < len; i++) {
        uint64_t M_u = (uint64_t)LOW64(vole_returned[i]);
        uint64_t u   = (uint64_t)HIGH64(vole_returned[i]);
        uint64_t x   = raw_data[i];

        pdList[i].degree = 1;
        pdList[i].coeffs.resize(2);
        pdList[i].coeffs[0] = M_u;
        pdList[i].coeffs[1] = x;
        pdList[i].is_consumed = false;

        lam[i] = add_mod(u, x);
    }

    io->send_data(lam.data(), len * sizeof(uint64_t));
}


template <typename IO>
PolyDelta MVZKExecProver<IO>::input(uint64_t raw_data) {
    __uint128_t vole_returned;
    uint64_t lam;
    PolyDelta res;

    vole->extend(&vole_returned, 1);
    uint64_t M_u = (uint64_t)LOW64(vole_returned);
    uint64_t u   = (uint64_t)HIGH64(vole_returned);
    uint64_t x   = raw_data;

    res.degree = 1;
    res.coeffs.resize(2);
    res.coeffs[0] = M_u;
    res.coeffs[1] = x;
    res.is_consumed = false;

    lam = add_mod(u, x);
    io->send_data(&lam, sizeof(uint64_t));
    return res;
}


template <typename IO>
PolyTensor MVZKExecProver<IO>::input(const std::vector<int>& shape, const std::vector<uint64_t>& raw_data) {
    PolyTensor res(shape, 1, PolyTensor::Coeffs);
    size_t size = res.total_elements;

    if (raw_data.size() != size) {
        LOG_ERROR("Input data size mismatch! Expected " << size << ", got " << raw_data.size());
        exit(-1);
    }

    std::vector<__uint128_t> vole_returned(size);

    std::vector<uint64_t> lams(size);

    vole->extend(vole_returned.data(), size);

    uint64_t* coeffs_0 = res.get_coeffs_ptr(0);

    uint64_t* coeffs_1 = res.get_coeffs_ptr(1);

    #pragma omp parallel for if(size >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
    for (size_t i = 0; i < size; ++i) {
        uint64_t M_u = (uint64_t)LOW64(vole_returned[i]);
        uint64_t u   = (uint64_t)HIGH64(vole_returned[i]);
        uint64_t x   = raw_data[i];

        coeffs_0[i] = M_u;
        coeffs_1[i] = x;

        lams[i] = add_mod(u, x);
    }

    io->send_data(lams.data(), size * sizeof(uint64_t));

    res.is_consumed = false;

    return res;
}


template <typename IO>
std::vector<PolyDelta> MVZKExecProver<IO>::extend_vope_from_vole(uint64_t deg, int size) {
    // If we are asked to pre-generate a deg-n VOPE, where deg=1 -> standard VOLE,
    // then we will need the VOLE triple size to be 2 * (n-1) + 1 = 2 * n - 1
    assert(size > 0 && deg > 0);

    int each_vole_size = (2 * deg - 1);
    int vole_size = size * each_vole_size;

    std::vector<PolyDelta> res;
    res.resize(size);
    std::vector<__uint128_t> vole_returned;
    vole_returned.resize(vole_size);

    this->vole->extend(vole_returned.data(), vole_size);

    std::vector<PolyDelta> from_vole_returned = vole2pd(vole_returned, vole_size);

    for (int i = 0; i < size; i++) {
        // Base offset for this batch
        int base_offset = i * each_vole_size;

        // 2. Base term (Index 0)
        res[i] = std::move(from_vole_returned[base_offset]);

        for (int j = 0; j < deg - 1; j++) {
            // Mul Index: base + 1 + j

            res[i] = res[i] * from_vole_returned[base_offset + 1 + j]
                     + from_vole_returned[base_offset + deg + j];
        }
        res[i].is_consumed = false;
        res[i].is_pre_generated = true;
    }
    return res;
}


template <typename IO>
std::vector<PolyDelta> MVZKExecProver<IO>::vole2pd(std::vector<__uint128_t>& vole_data, int size) {
    std::vector<PolyDelta> res;
    res.resize(size);
    for (int i = 0; i < size; i++){
        res[i].is_consumed = false;
        res[i].degree = 1;
        res[i].coeffs.resize(2);
        res[i].coeffs[1] = HIGH64(vole_data[i]);
        // Note: this step is to make K = M + delta * x, instead of M = K + delta * x
        res[i].coeffs[0] = add_mod((PR - LOW64(vole_data[i])), 0);
    }
    return res;
}


template <typename IO>
void MVZKExecProver<IO>::send_poly(const PolyDelta& pd) {
    size_t num_coeffs = pd.coeffs.size();
    size_t total_size = sizeof(int) + num_coeffs * sizeof(uint64_t);

    std::vector<uint8_t> buffer(total_size);
    uint8_t* ptr = buffer.data();

    memcpy(ptr, &pd.degree, sizeof(int));
    ptr += sizeof(int);

    if (num_coeffs > 0) {
        memcpy(ptr, pd.coeffs.data(), num_coeffs * sizeof(uint64_t));
    }

    io->send_data(buffer.data(), total_size);
}


template <typename IO>
void MVZKExecProver<IO>::send_poly(const std::vector<PolyDelta>& pd_list) {
    size_t count = pd_list.size();

    size_t total_size = sizeof(size_t);

    for (const auto& pd : pd_list) {
        // [Degree] + [RealVal] + [Coeffs]
        total_size += sizeof(int);
        total_size += pd.coeffs.size() * sizeof(uint64_t);
    }

    std::vector<uint8_t> buffer(total_size);
    uint8_t* ptr = buffer.data();

    memcpy(ptr, &count, sizeof(size_t));
    ptr += sizeof(size_t);

    for (const auto& pd : pd_list) {
        // 4.1 Degree
        memcpy(ptr, &pd.degree, sizeof(int));
        ptr += sizeof(int);

        // 4.3 Coefficients
        size_t num_coeffs = pd.coeffs.size();
        if (num_coeffs > 0) {
            memcpy(ptr, pd.coeffs.data(), num_coeffs * sizeof(uint64_t));
            ptr += num_coeffs * sizeof(uint64_t);
        }
    }

    io->send_data(buffer.data(), total_size);
}


template <typename IO>
std::vector<uint64_t> MVZKExecProver<IO>::reveal(const PolyTensor& pt) {
    io->send_data(&pt.total_elements, sizeof(size_t));
    io->send_data(&pt.degree, sizeof(int));

    size_t total_coeffs_size = pt.flat_coeffs.size();
    io->send_data(pt.flat_coeffs.data(), total_coeffs_size * sizeof(uint64_t));
    std::vector<uint64_t> placeholder;
    return placeholder;
}

#endif
