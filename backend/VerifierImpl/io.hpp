#ifndef MVZK_BACKEND_VERIFIER_IMPL_IO_HPP__
#define MVZK_BACKEND_VERIFIER_IMPL_IO_HPP__

#include "../../exec/ExecVerifier.h"

template <typename IO>
void MVZKExecVerifier<IO>::input(std::vector<PolyDelta>& pdList, const std::vector<uint64_t>& raw_data) {
    // Here raw_data will not be used.

    size_t len = pdList.size();
    if (len == 0) {
        LOG_ERROR("Verifier input list must be pre-sized!");
        exit(-1);
    }

    std::vector<__uint128_t> vole_returned(len);
    std::vector<uint64_t> lam(len);

    vole->extend(vole_returned.data(), len);

    io->recv_data(lam.data(), len * sizeof(uint64_t));

    for (size_t i = 0; i < len; i++){
        uint64_t delta_mult_u_add_x = mult_mod(this->delta, lam[i]);
        pdList[i].key = add_mod(vole_returned[i], delta_mult_u_add_x);
        pdList[i].degree = 1;
        pdList[i].is_consumed = false;
    }
}


template <typename IO>
PolyDelta MVZKExecVerifier<IO>::input(uint64_t raw_data) {
    PolyDelta res;
    __uint128_t vole_returned;
    uint64_t lam;

    vole->extend(&vole_returned, 1);

    io->recv_data(&lam, sizeof(uint64_t));
    uint64_t delta_mult_u_add_x = mult_mod(this->delta, lam);
    res.key = add_mod(vole_returned, delta_mult_u_add_x);
    res.degree = 1;
    res.is_consumed = false;

    return res;
}


template <typename IO>
std::vector<PolyDelta> MVZKExecVerifier<IO>::extend_vope_from_vole(uint64_t deg, int size) {
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
PolyTensor MVZKExecVerifier<IO>::input(const std::vector<int>& shape, const std::vector<uint64_t>& raw_data) {
    // The second varibale raw data will not be used in the verifier side, just set it to zero.

    PolyTensor res(shape, 1, PolyTensor::Keys);
    size_t size = res.total_elements;

    std::vector<__uint128_t> vole_returned(size);
    std::vector<uint64_t> lams(size);

    vole->extend(vole_returned.data(), size);

    io->recv_data(lams.data(), size * sizeof(uint64_t));

    uint64_t* keys = res.get_keys_ptr();

    uint64_t local_delta = (uint64_t)this->delta;

    #pragma omp parallel for if (size>=MVZK_OMP_SIZE_THRESHOLD  && !omp_in_parallel())
    for (size_t i = 0; i < size; ++i) {
        uint64_t K = (uint64_t)LOW64(vole_returned[i]);
        uint64_t lam = lams[i];

        uint64_t delta_term = mult_mod(local_delta, lam);

        keys[i] = add_mod(K, delta_term);

        if (keys[i] > PR) LOG_WARN("keys! > PR (input)");
    }

    res.is_consumed = false;
    return res;
}


template <typename IO>
uint64_t MVZKExecVerifier<IO>::delta_pow(uint64_t a, uint64_t deg) {
    assert(a == (uint64_t)this->delta);
    if (deg < delta_powers_cache.size()) {
        return delta_powers_cache[deg];
    }

    size_t old_size = delta_powers_cache.size();

    delta_powers_cache.resize(deg + 1);

    // cache[i] = cache[i-1] * delta
    for (size_t i = old_size; i <= deg; ++i) {
        delta_powers_cache[i] = mult_mod(delta_powers_cache[i-1], this->delta);
    }

    return delta_powers_cache[deg];
}


template <typename IO>
std::vector<PolyDelta> MVZKExecVerifier<IO>::vole2pd(std::vector<__uint128_t>& vole_data, int size) {
    std::vector<PolyDelta> res;
    res.resize(size);
    for (int i = 0; i < size; i++){
        res[i].is_consumed = false;
        res[i].degree = 1;
        // Note: this step is to make K = M + delta * x, instead of M = K + delta * x
        res[i].key = add_mod((PR - LOW64(vole_data[i])), 0);
    }
    return res;
}


template <typename IO>
PolyDelta MVZKExecVerifier<IO>::recv_poly() {
    PolyDelta res;

    io->recv_data(&res.degree, sizeof(int));

    size_t num_coeffs = res.degree + 1;
    res.coeffs.resize(num_coeffs);

    io->recv_data(res.coeffs.data(), num_coeffs * sizeof(uint64_t));

    res.key = 0;

    res.is_consumed = true;

    return res;
}


template <typename IO>
std::vector<PolyDelta> MVZKExecVerifier<IO>::recv_poly_vector() {
    size_t count;
    io->recv_data(&count, sizeof(size_t));

    std::vector<PolyDelta> res;
    res.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        res.push_back(recv_poly());
    }

    return res;
}


template <typename IO>
bool MVZKExecVerifier<IO>::verify_poly(PolyDelta& mine, PolyDelta& hers) {
    // This function check the validity of a received PolyDelta.
    // Hers from prover, have coefs and real value

    if (mine.degree != hers.degree){
        LOG_ERROR("Inconsistent degree for verification! Mine = " << mine.degree << ", hers = " << hers.degree);
        return false;
    }

    if (hers.coeffs.back() != 0) {
        LOG_ERROR("ZK Zero-Check Failed! The accumulated constraint evaluates to " << hers.coeffs.back() << " instead of 0!");
        return false;
    }

    // If the two condition passed, we calculate.
    uint64_t res = 0;
    for (int i = 0; i < hers.coeffs.size() - 1; i++){
        res = add_mod(res, mult_mod(hers.coeffs[i], delta_pow(this->delta, i)));
    }
    if (res != mine.key){
        LOG_ERROR("Commitment check failed! RES = " << res << ", Key = " << mine.key);
        if ((res % PR) == (mine.key % PR)){
            LOG_ERROR("MULT MOD ERROR....");
        }else{
            LOG_ERROR("REAL ERROR!!!!!");
        }
        return false;
    }

    return true;
}


template <typename IO>
std::vector<uint64_t> MVZKExecVerifier<IO>::reveal(const PolyTensor& pt) {
    size_t recv_len;
    int recv_deg;
    io->recv_data(&recv_len, sizeof(size_t));
    io->recv_data(&recv_deg, sizeof(int));

    if (recv_len != pt.total_elements) {
        LOG_ERROR(" Reveal Size Mismatch!");
        std::cout << "Recv Len: " << recv_len << ", Local Len: " << pt.total_elements << std::endl;

        std::vector<uint64_t> trash(recv_len * (recv_deg + 1));
        io->recv_data(trash.data(), trash.size() * sizeof(uint64_t));
        exit(-1);
    }

    std::vector<uint64_t> recv_coeffs(recv_len * (recv_deg + 1));
    io->recv_data(recv_coeffs.data(), recv_coeffs.size() * sizeof(uint64_t));

    LOG_INFO("Received Tensor Data, Verifying...");

    std::vector<uint64_t> d_pows(recv_deg + 1);
    for(int d = 0; d <= recv_deg; ++d) {
        d_pows[d] = delta_pow(this->delta, d);
    }

    const uint64_t* local_keys = pt.get_keys_ptr();
    bool all_pass = true;
    int fail_count = 0;

    for (size_t i = 0; i < recv_len; ++i) {
        uint64_t rhs = 0;

        for (int d = 0; d <= recv_deg; ++d) {
            uint64_t coeff = recv_coeffs[d * recv_len + i];
            uint64_t term = mult_mod(coeff, d_pows[d]);
            rhs = add_mod(rhs, term);
        }

        if (rhs != local_keys[i]) {
            if (fail_count < 10) {
                std::cout << "\033[31m[FAIL] Index " << i
                          << " | Calc RHS=" << rhs
                          << " != Local Key=" << local_keys[i] << "\033[0m" << std::endl;
            }
            all_pass = false;
            fail_count++;
        }
    }

    if (all_pass) {
        size_t offset = recv_deg * recv_len;
        std::vector<uint64_t> reveal_output(
            recv_coeffs.begin() + offset,
            recv_coeffs.end()
        );

        return reveal_output;
    } else {
        LOG_ERROR("INSTANT TENSOR CHECK FAILED");
        std::cout << "Total Failures: " << fail_count << " / " << recv_len << std::endl;
        exit(-1);
    }
}


template <typename IO>
void MVZKExecVerifier<IO>::delta_gen() {
    PRG prg;
    prg.random_data(&delta, sizeof(__uint128_t));
    extract_fp(delta);
}


template <typename IO>
void MVZKExecVerifier<IO>::save_delta_to_file(const std::string& filepath) const {
    std::ofstream out(filepath, std::ios::binary);
    if (!out) throw std::runtime_error("[ERROR] Failed to save Delta: " + filepath);
    out.write(reinterpret_cast<const char*>(&delta), sizeof(__uint128_t));
    out.close();
}


template <typename IO>
void MVZKExecVerifier<IO>::load_delta_from_file(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in) throw std::runtime_error("[ERROR] Failed to load Delta: " + filepath);
    in.read(reinterpret_cast<char*>(&delta), sizeof(__uint128_t));
    in.close();
}

#endif
