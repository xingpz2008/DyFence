#ifndef MVZK_BACKEND_PROVER_IMPL_DEBUG_HPP__
#define MVZK_BACKEND_PROVER_IMPL_DEBUG_HPP__

#include "../../exec/ExecProver.h"

template <typename IO>
void MVZKExecProver<IO>::debug_print(const PolyDelta& pd, std::string name) {
    std::cout << std::left;
    std::cout << "\033[32m[Prover] " << std::setw(15) << name << "\033[0m";
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
void MVZKExecProver<IO>::debug_print(const PolyTensor& pt, std::string name, int limit) {
    std::cout << "--------------------------------------------------------" << std::endl;
    size_t count = (size_t)limit < pt.total_elements ? (size_t)limit : pt.total_elements;

    for(size_t i=0; i<count; ++i) {
        std::string item_name = name + "[" + std::to_string(i) + "]";
        std::cout << "\033[32m[Prover] " << std::setw(15) << item_name << "\033[0m";
        std::cout << " (Deg=" << pt.degree << ") ";

        if (i < pt.total_elements)
            std::cout << "Val=" << std::setw(6) << pt.get_real_vals_ptr()[i] << " | Poly = ";
        else
            std::cout << "Val=ERR    | Poly = ";

        for (int d = 0; d <= pt.degree; ++d) {
            if (d > 0) std::cout << " + ";

            const uint64_t* ptr = pt.get_coeffs_ptr(d);
            if (ptr) std::cout << ptr[i]; else std::cout << "?";

            if (d == 1) std::cout << "x";
            else if (d > 1) std::cout << "x^" << d;
        }
        std::cout << std::endl;
    }
    if (pt.total_elements > count) std::cout << " ... (" << (pt.total_elements - count) << " omitted)" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
}


template <typename IO>
bool MVZKExecProver<IO>::debug_instant_check(const PolyDelta& pd) {
    YELLOW("[Prover] Instant PD Check");
    debug_print(pd, "");
    send_poly(pd);
    return true;
}


template <typename IO>
bool MVZKExecProver<IO>::debug_instant_check(const PolyTensor& pt) {
    YELLOW("[Prover] Instant Tensor Check");
    debug_print(pt, "Sent Tensor");

    io->send_data(&pt.total_elements, sizeof(size_t));
    io->send_data(&pt.degree, sizeof(int));

    size_t total_coeffs_size = pt.flat_coeffs.size();
    io->send_data(pt.flat_coeffs.data(), total_coeffs_size * sizeof(uint64_t));

    return true;
}

#endif
