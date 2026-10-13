#ifndef MVZK_BACKEND_VERIFIER_IMPL_CHECK_HPP__
#define MVZK_BACKEND_VERIFIER_IMPL_CHECK_HPP__

#include "../../exec/ExecVerifier.h"

template <typename IO>
void MVZKExecVerifier<IO>::submit_to_buffer(PolyDelta&& pd) {
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
void MVZKExecVerifier<IO>::submit_tensor_to_buffer(PolyTensor&& pt) {
    if (!pt.is_constraint){
        LOG_WARN("A non-constraint PolyTensor received by buffer!");
    }
    pt.is_consumed = true;
    check_tensor_buffer.emplace_back(std::move(pt));
    if (check_tensor_buffer.size() >= MVZK_CONFIG_TENSOR_CHECK_CNT) {
        check_all();
    }
}


template <typename IO>
void MVZKExecVerifier<IO>::submit_non_zero_tensor_to_buffer(const PolyTensor& target) {
    target.is_consumed = true;

    int new_degree = target.degree + 1;
    PolyTensor padded_tensor(target.shape, new_degree, PolyTensor::Keys);
    padded_tensor.is_constraint = true;

    const uint64_t* in_keys = target.get_keys_ptr();
    uint64_t* out_keys = padded_tensor.get_keys_ptr();

    std::copy(in_keys, in_keys + target.total_elements, out_keys);

    this->check_tensor_buffer.push_back(std::move(padded_tensor));
    if (check_tensor_buffer.size() >= MVZK_CONFIG_TENSOR_CHECK_CNT) {
        check_all();
    }
}


template <typename IO>
PolyTensor MVZKExecVerifier<IO>::refresh_tensor_degree(const PolyTensor& high_degree_tensor, const std::string& check_name) {
    // There is no plaintext for verifier
    std::vector<uint64_t> plaintext_data_dummy;
    std::vector<int> shape = high_degree_tensor.shape;

    PolyTensor res = input(shape, plaintext_data_dummy);
    PolyTensor::store_relation(high_degree_tensor, res, check_name);

    return res;
}


template <typename IO>
void MVZKExecVerifier<IO>::check_all() {
    if (check_buffer.empty() && check_tensor_buffer.empty()) return;

    block seed;
    prg.random_block(&seed, 1);
    io->send_data(&seed, sizeof(block));
    this->prg.reseed(&seed);

    PolyDelta final_checked_item;

    accumulate_delta_buffer(final_checked_item);
    accumulate_tensor_buffer(final_checked_item);

    if (final_checked_item.degree > MVZK_FFT_WARNING_LIMIT){
        LOG_WARN("FFT Optimization is recommended, deg = " << final_checked_item.degree << ", alert limit = " << MVZK_FFT_WARNING_LIMIT << ".");
    }

    if (final_checked_item.degree > 1){
        PolyDelta vope = std::move(extend_vope_from_vole(final_checked_item.degree - 1, 1)[0]);

        final_checked_item.key = add_mod(final_checked_item.key, vope.key);
        vope.is_consumed = true;
        //final_checked_item = final_checked_item + vope;

        PolyDelta recvd_poly = std::move(recv_poly());

        if (verify_poly(final_checked_item, recvd_poly) == false){
            LOG_ERROR("Check failed. (Continue to other test case if in dev_test mode.)");
            if (MVZK_CONFIG_INSTANT_ABORT){
                exit(-1);
            }
        }else{
            LOG_INFO("Check success.");
        }
    }else{
        emp::Hash hash_instance;
        char dig[emp::Hash::DIGEST_SIZE];
        hash_instance.put(&final_checked_item.key, sizeof(uint64_t));
        hash_instance.digest(dig);
        char dig_recv[emp::Hash::DIGEST_SIZE];
        io->recv_data(dig_recv, emp::Hash::DIGEST_SIZE);
        if (!cmpBlock((block *)dig, (block *)dig_recv,
                emp::Hash::DIGEST_SIZE / 16)) {
            LOG_ERROR("Check failed. (Hash failure)");
            if (MVZK_CONFIG_INSTANT_ABORT){ exit(-1); }
        }else{
            LOG_INFO("Check success.");
        }
    }
    final_checked_item.is_consumed = true;
}


template <typename IO>
void MVZKExecVerifier<IO>::finalize_protocol() {
    flush_all_luts();
    flush_all_range_checks();
    check_all();
}


template <typename IO>
void MVZKExecVerifier<IO>::accumulate_delta_buffer(PolyDelta& final_item) {
    if (check_buffer.empty()) return;

    size_t size = check_buffer.size();
    std::vector<uint64_t> chi(size);
    this->prg.random_data(chi.data(), size * sizeof(uint64_t));

    for (size_t i = 0; i < size; i++) {
        chi[i] = chi[i] % PR;
        final_item += chi[i] * check_buffer[i];
        check_buffer[i].is_consumed = true;
    }
    check_buffer.clear();
}


template <typename IO>
void MVZKExecVerifier<IO>::accumulate_tensor_buffer(PolyDelta& final_item) {
    if (check_tensor_buffer.empty()) return;

    int max_deg = final_item.degree;
    for (auto& tensor : check_tensor_buffer) {
        max_deg = std::max(max_deg, tensor.degree);
    }

    if (final_item.degree < max_deg) {
        int offset_diff = max_deg - final_item.degree;
        final_item.key = mult_mod(final_item.key, delta_pow(this->delta, offset_diff));
        final_item.degree = max_deg;
    }

    for (auto& tensor : check_tensor_buffer) {
        size_t len = tensor.total_elements;

        std::vector<uint64_t> chi_vec(len);
        this->prg.random_data(chi_vec.data(), len * sizeof(uint64_t));

        #pragma omp parallel for if (len >= MVZK_OMP_SIZE_THRESHOLD  && !omp_in_parallel())
        for(size_t i=0; i<len; ++i) {
            chi_vec[i] = chi_vec[i] % PR;
        }

        const uint64_t* key_ptr = tensor.get_keys_ptr();
        uint64_t sum_key = 0;

        if (len >= MVZK_OMP_SIZE_THRESHOLD  && !omp_in_parallel()) {
            #pragma omp parallel
            {
                uint64_t local_sum = 0;

                #pragma omp for nowait
                for (size_t i = 0; i < len; ++i) {
                    uint64_t term = mult_mod(key_ptr[i], chi_vec[i]);
                    local_sum = add_mod(local_sum, term);
                }

                #pragma omp critical
                {
                    sum_key = add_mod(sum_key, local_sum);
                }
            }
        } else {
            for (size_t i = 0; i < len; ++i) {
                uint64_t term = mult_mod(key_ptr[i], chi_vec[i]);
                sum_key = add_mod(sum_key, term);
            }
        }

        int offset = max_deg - tensor.degree;

        uint64_t shifted_key = mult_mod(sum_key, delta_pow(this->delta, offset));
        final_item.key = add_mod(final_item.key, shifted_key);

        tensor.is_consumed = true;
    }
    check_tensor_buffer.clear();
}

#endif
