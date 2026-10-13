#ifndef MVZK_BACKEND_PROVER_IMPL_CHECK_HPP__
#define MVZK_BACKEND_PROVER_IMPL_CHECK_HPP__

#include "../../exec/ExecProver.h"

template <typename IO>
void MVZKExecProver<IO>::submit_to_buffer(PolyDelta&& pd) {
    if (!pd.is_constraint){
        LOG_WARN("A non-constraint PolyDelta received by buffer!");
    }
    pd.is_consumed = true;
    check_buffer.emplace_back(std::move(pd));

    if (check_buffer.size() >= MVZK_MULT_CHECK_CNT) {
        check_all();
    }
}


template <typename IO>
void MVZKExecProver<IO>::submit_tensor_to_buffer(PolyTensor&& pt) {
    if (!pt.is_constraint) {
         LOG_WARN("A non-constraint PolyTensor received by buffer!");
    }
    pt.is_consumed = true;
    check_tensor_buffer.emplace_back(std::move(pt));

    if (check_tensor_buffer.size() >= MVZK_CONFIG_TENSOR_CHECK_CNT) {
        check_all();
    }
}


template <typename IO>
void MVZKExecProver<IO>::submit_non_zero_tensor_to_buffer(const PolyTensor& target) {
    target.is_consumed = true;

    int new_degree = target.degree + 1;
    PolyTensor padded_tensor(target.shape, new_degree, PolyTensor::Coeffs);
    padded_tensor.is_constraint = true;

    for (int d = 0; d <= target.degree; ++d) {
        const uint64_t* in_coeffs = target.get_coeffs_ptr(d);
        uint64_t* out_coeffs = padded_tensor.get_coeffs_ptr(d);

        std::copy(in_coeffs, in_coeffs + target.total_elements, out_coeffs);
    }

    uint64_t* highest_coeffs = padded_tensor.get_coeffs_ptr(new_degree);
    std::fill(highest_coeffs, highest_coeffs + target.total_elements, 0);

    this->check_tensor_buffer.push_back(std::move(padded_tensor));
    if (check_tensor_buffer.size() >= MVZK_CONFIG_TENSOR_CHECK_CNT) {
        check_all();
    }
}


template <typename IO>
PolyTensor MVZKExecProver<IO>::refresh_tensor_degree(const PolyTensor& high_degree_tensor, const std::string& check_name) {
    std::vector<uint64_t> plaintext_val = high_degree_tensor.get_real_vals_vector();
    std::vector<int> shape_vec = high_degree_tensor.shape;

    PolyTensor res = input(shape_vec, plaintext_val);

    PolyTensor::store_relation(high_degree_tensor, res, check_name);

    return res;
}


template <typename IO>
void MVZKExecProver<IO>::check_all() {
    if (check_buffer.empty() && check_tensor_buffer.empty()){
        LOG_WARN("check_all() called with no items in all buffers.");
        return;
    }

    block seed;
    io->recv_data(&seed, sizeof(block));
    this->prg.reseed(&seed);

    PolyDelta final_checked_item;

    accumulate_delta_buffer(final_checked_item);
    accumulate_tensor_buffer(final_checked_item);

    if (final_checked_item.degree > MVZK_FFT_WARNING_LIMIT){
        LOG_WARN("FFT Optimization is recommended, deg = " << final_checked_item.degree << ", alert limit = " << MVZK_FFT_WARNING_LIMIT << ".");
    }

    if (final_checked_item.degree > 1){
        PolyDelta vope = std::move(extend_vope_from_vole(final_checked_item.degree - 1, 1)[0]);

        // Here we have to mask it, instead of directly add it.
        for (int i = 0; i < final_checked_item.coeffs.size() - 1; i++){
            final_checked_item.coeffs[i] = add_mod(final_checked_item.coeffs[i], vope.coeffs[i]);
        }
        vope.is_consumed = true;
        //final_checked_item = final_checked_item + vope;

        send_poly(final_checked_item);
    }else{
        // We consider another different case. If final checked item degree = 0, then it is check zero by a hash oracle.
        emp::Hash hash_instance;
        hash_instance.put(&final_checked_item.coeffs[0], sizeof(uint64_t));
        char dig[emp::Hash::DIGEST_SIZE];
        hash_instance.digest(dig);
        io->send_data(dig, emp::Hash::DIGEST_SIZE);
    }
    final_checked_item.is_consumed = true;
}


template <typename IO>
void MVZKExecProver<IO>::finalize_protocol() {
    flush_all_luts();
    flush_all_range_checks();
    check_all();
}


template <typename IO>
void MVZKExecProver<IO>::accumulate_delta_buffer(PolyDelta& final_item) {
    if (check_buffer.empty()) return;

    size_t size = check_buffer.size();
    std::vector<uint64_t> chi(size);
    this->prg.random_data(chi.data(), size * sizeof(uint64_t));

    for (size_t i = 0; i < size; i++){
        // final += chi[i] * item[i]

        chi[i] = chi[i] % PR;
        final_item += chi[i] * check_buffer[i];
        check_buffer[i].is_consumed = true;
    }
    check_buffer.clear();
}


template <typename IO>
void MVZKExecProver<IO>::accumulate_tensor_buffer(PolyDelta& final_item) {
    if (check_tensor_buffer.empty()) return;

    if (final_item.coeffs.empty()) {
        final_item.coeffs.resize(final_item.degree + 1, 0);
    }

    int max_deg = final_item.degree;
    for (auto& tensor : check_tensor_buffer) {
        max_deg = std::max(max_deg, tensor.degree);
    }

    if (final_item.degree < max_deg) {
        int offset_diff = max_deg - final_item.degree;
        std::vector<uint64_t> new_coeffs(max_deg + 1, 0);
        for (size_t i = 0; i <= final_item.degree; ++i) {
            new_coeffs[i + offset_diff] = final_item.coeffs[i];
        }
        final_item.coeffs = std::move(new_coeffs);
        final_item.degree = max_deg;
    }

    for (auto& tensor : check_tensor_buffer) {
        size_t len = tensor.total_elements;

        std::vector<uint64_t> chi_vec(len);
        this->prg.random_data(chi_vec.data(), len * sizeof(uint64_t));

        #pragma omp parallel for if(len >= MVZK_OMP_SIZE_THRESHOLD && !omp_in_parallel())
        for(size_t i=0; i<len; ++i) {
            chi_vec[i] = chi_vec[i] % PR;
        }

        int offset = max_deg - tensor.degree;

        for (int d = 0; d <= tensor.degree; ++d) {
            const uint64_t* coeff_ptr = tensor.get_coeffs_ptr(d);
            uint64_t sum_coeff = 0;

            if (len >= MVZK_OMP_SIZE_THRESHOLD  && !omp_in_parallel()) {
                #pragma omp parallel
                {
                    uint64_t local_sum = 0;
                    #pragma omp for nowait
                    for (size_t i = 0; i < len; ++i) {
                        uint64_t term = mult_mod(coeff_ptr[i], chi_vec[i]);
                        local_sum = add_mod(local_sum, term);
                    }
                    #pragma omp critical
                    {
                        sum_coeff = add_mod(sum_coeff, local_sum);
                    }
                }
            } else {
                for (size_t i = 0; i < len; ++i) {
                    uint64_t term = mult_mod(coeff_ptr[i], chi_vec[i]);
                    sum_coeff = add_mod(sum_coeff, term);
                }
            }

            final_item.coeffs[d + offset] = add_mod(final_item.coeffs[d + offset], sum_coeff);
        }

        tensor.is_consumed = true;
    }
    check_tensor_buffer.clear();
}

#endif
