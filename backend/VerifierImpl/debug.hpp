#ifndef MVZK_BACKEND_VERIFIER_IMPL_DEBUG_HPP__
#define MVZK_BACKEND_VERIFIER_IMPL_DEBUG_HPP__

#include "../../exec/ExecVerifier.h"

template <typename IO>
void MVZKExecVerifier<IO>::debug_print(const PolyDelta& pd, std::string name) {
    std::cout << std::left;
    std::cout << "\033[36m[Verif ] " << std::setw(15) << name << "\033[0m";
    std::cout << " (Deg=" << pd.degree << ") ";
    std::cout << "Delta=" << (uint64_t)this->delta << ", ";
    std::cout << "Key=" << pd.key << std::endl;
}


template <typename IO>
void MVZKExecVerifier<IO>::debug_print(const PolyTensor& pt, std::string name, int limit) {
    std::cout << "--------------------------------------------------------" << std::endl;
    size_t count = (size_t)limit < pt.total_elements ? (size_t)limit : pt.total_elements;

    for(size_t i=0; i<count; ++i) {
        std::string item_name = name + "[" + std::to_string(i) + "]";
        std::cout << "\033[36m[Verif ] " << std::setw(15) << item_name << "\033[0m";
        std::cout << " (Deg=" << pt.degree << ") ";

        if (i < pt.flat_keys.size())
            std::cout << "Key=" << pt.flat_keys[i];
        else
            std::cout << "Key=ERR";
        std::cout << std::endl;
    }
    if (pt.total_elements > count) std::cout << " ... (" << (pt.total_elements - count) << " omitted)" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
}


template <typename IO>
void MVZKExecVerifier<IO>::debug_print_recvd(const PolyDelta& pd, std::string name) {
    std::cout << std::left;
    std::cout << "\033[32m[Received] " << std::setw(15) << name << "\033[0m";
    std::cout << " (Deg=" << pd.degree << ") ";
    std::cout << "Val=" << std::setw(6) << pd.get_real_val() << " | Poly = ";

    if (pd.coeffs.empty()) {
        std::cout << "(empty)";
    } else {
        for (size_t i = 0; i < pd.coeffs.size(); ++i) {
            if (i > 0) std::cout << " + ";
            std::cout << pd.coeffs[i];
            if (i == 1) std::cout << "x";
            else if (i > 1) std::cout << "x^" << i;
        }
    }
    std::cout << std::endl;
}


template <typename IO>
bool MVZKExecVerifier<IO>::debug_instant_check(const PolyDelta& pd) {
    YELLOW("[Verifier] Instant Check");
    debug_print(pd, "");
    PolyDelta recvd = recv_poly();
    WHITE("Received:");
    debug_print_recvd(recvd, "");
    // Compute
    uint64_t rhs = 0;
    for (size_t i = 0; i < recvd.coeffs.size(); i++){
        uint64_t this_term = mult_mod(recvd.coeffs[i], delta_pow(this->delta, i));
        rhs = add_mod(rhs, this_term);
    }
    std::cout << "RHS = " << rhs << ", LHS (Key) = " << pd.key << std::endl;
    bool res = (rhs == pd.key);
    if (res){
        GREEN("[Verifer] INSTANT CHECK PASS");
        return res;
    }
    else{
        RED("[VERIFIER] INSTANT CHECK FAILED");
        return res;
    }
}


template <typename IO>
bool MVZKExecVerifier<IO>::debug_instant_check(const PolyTensor& pt) {
    YELLOW("[Verifier] Instant Tensor Check");

    size_t recv_len;
    int recv_deg;
    io->recv_data(&recv_len, sizeof(size_t));
    io->recv_data(&recv_deg, sizeof(int));

    if (recv_len != pt.total_elements) {
        RED("[Verifier] Size Mismatch!");
        std::cout << "Recv Len: " << recv_len << ", Local Len: " << pt.total_elements << std::endl;

        std::vector<uint64_t> trash(recv_len * (recv_deg + 1));
        io->recv_data(trash.data(), trash.size() * sizeof(uint64_t));
        return false;
    }

    std::vector<uint64_t> recv_coeffs(recv_len * (recv_deg + 1));
    io->recv_data(recv_coeffs.data(), recv_coeffs.size() * sizeof(uint64_t));

    WHITE("Received Tensor Data, Verifying...");

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
        GREEN("[Verifier] INSTANT TENSOR CHECK PASS");
        return true;
    } else {
        RED("[Verifier] INSTANT TENSOR CHECK FAILED");
        std::cout << "Total Failures: " << fail_count << " / " << recv_len << std::endl;
        return false;
    }
}

#endif
