#ifndef MVZK_BACKEND_PROVER_IMPL_LUT_HPP__
#define MVZK_BACKEND_PROVER_IMPL_LUT_HPP__

#include "../../exec/ExecProver.h"

template <typename IO>
int MVZKExecProver<IO>::register_lut_table(const std::vector<std::pair<uint64_t, uint64_t>>& data) {
    size_t new_hash = compute_vector_hash(data);

    for (size_t i = 0; i < lut_tables.size(); ++i) {
        LUTData* existing_table = lut_tables[i].get();

        if (existing_table->data_hash != new_hash) {
            continue;
        }

        if (existing_table->data.size() != data.size()) {
            continue;
        }

        if (existing_table->data == data) {
            LOG_INFO("LUT Table Deduplication: Reusing ID " << i);
            return (int)i;
        }
    }
    auto entry = std::make_unique<LUTData>(data);
    for (const auto& kv : data) {
        entry->version_tracker[kv] = 0;
    }
    lut_tables.push_back(std::move(entry));
    return (int)lut_tables.size() - 1;
}


template <typename IO>
void MVZKExecProver<IO>::submit_lut_check(const PolyTensor& keys, const PolyTensor& vals, int table_id) {
    // When submit to the lut, there is only the public table, read and a version tracker. Storing Wirte is not necessary.

    if (table_id < 0 || table_id >= lut_tables.size()) {
        LOG_ERROR("[ERROR] Invalid Table ID: " << table_id);
        return;
    }

    // Find the right LUT table instance.
    LUTData* table = lut_tables[table_id].get();

    LUTData::Request req;
    req.keys = keys.clone();
    req.vals = vals.clone();

    size_t len = keys.total_elements;
    req.v_j_hint.reserve(len);

    const uint64_t* k_ptr = keys.get_real_vals_ptr();
    const uint64_t* v_ptr = vals.get_real_vals_ptr();

    for(size_t i = 0; i < len; ++i) {
        std::pair<uint64_t, uint64_t> item = {k_ptr[i], v_ptr[i]};

        req.v_j_hint.push_back(table->version_tracker[item]++);
    }

    table->buffer.push_back(std::move(req));
    table->current_buffer_size += len;

    keys.is_consumed = true;
    vals.is_consumed = true;

    if (table->current_buffer_size >= MVZK_CONFIG_LUT_REQUEST_BUFFER_THRESHOLD) {
        process_lut_batch(table);
    }
}


template <typename IO>
void MVZKExecProver<IO>::process_lut_batch(LUTData* table) {
    if (table->buffer.empty()) return;

    size_t N_Query = table->current_buffer_size;
    size_t N_Table = table->data.size();

    std::vector<uint64_t> hints_flat;
    hints_flat.reserve(N_Query);
    for (const auto& req : table->buffer) {
        hints_flat.insert(hints_flat.end(), req.v_j_hint.begin(), req.v_j_hint.end());
    }

    std::vector<uint64_t> finals_flat;
    finals_flat.reserve(N_Table);
    for (const auto& kv : table->data) {
        finals_flat.push_back(table->version_tracker[kv]);
    }

    PolyTensor V_J = this->input({(int)N_Query, 1}, hints_flat);
    PolyTensor V_FINAL = this->input({(int)N_Table, 1}, finals_flat);

    block seed;
    io->recv_data(&seed, sizeof(block));
    this->prg.reseed(&seed);

    uint64_t sx, sy, sv, r;
    this->prg.random_data(&sx, sizeof(uint64_t));
    this->prg.random_data(&sy, sizeof(uint64_t));
    this->prg.random_data(&sv, sizeof(uint64_t));
    this->prg.random_data(&r, sizeof(uint64_t));
    sx = sx % PR;
    sy = sy % PR;
    sv = sv % PR;
    r = r % PR;

    // Phase 3: RLC & Term Construction

    auto [Big_Keys, Big_Vals] = concat_buffer_tensors(table);

    // Term_Read = K*sx + V*sy + v*sv - r
    PolyTensor Term_Q_Read = (Big_Keys * sx) + (Big_Vals * sy) + (V_J * sv) - r;

    PolyTensor Term_Q_Write = Term_Q_Read + sv;

    std::vector<uint64_t> table_consts;
    table_consts.reserve(N_Table);
    uint64_t P_Init_Scalar = 1;

    for (const auto& kv : table->data) {
        // Process write const
        uint64_t key = kv.first;
        uint64_t val = kv.second;

        uint64_t term_val = add_mod(mult_mod(key, sx), mult_mod(val, sy));
        table_consts.push_back(term_val);

        P_Init_Scalar = mult_mod(P_Init_Scalar, add_mod(term_val, PR - r));
        //P_Init_Scalar *= (term_val - r);
    }

    // Term_Final = (K*sx + V*sy) + v_final*sv - r
    PolyTensor P_Table_Const = PolyTensor::from_public({(int)N_Table, 1}, table_consts);

    PolyTensor Term_Final = P_Table_Const + (V_FINAL * sv) - r;

    std::vector<PolyTensor> lhs_list = split_to_scalars(Term_Q_Read);
    std::vector<PolyTensor> final_list = split_to_scalars(Term_Final);

    lhs_list.reserve(lhs_list.size() + final_list.size());
    lhs_list.insert(lhs_list.end(),
                    std::make_move_iterator(final_list.begin()),
                    std::make_move_iterator(final_list.end()));

    PolyTensor P_LHS = fast_tree_product(lhs_list);

    std::vector<PolyTensor> rhs_list = split_to_scalars(Term_Q_Write);
    PolyTensor P_RHS_Write = fast_tree_product(rhs_list);

    // Phase 5: Check Zero

    // Z = LHS - (Write_Product * Init_Scalar)
    PolyTensor Z = P_LHS - (P_RHS_Write * P_Init_Scalar);
    Z.is_constraint = true;

    // Phase 6: Cleanup

    table->buffer.clear();
    table->current_buffer_size = 0;

    for (auto& pair : table->version_tracker) {
        pair.second = 0;
    }

    this->submit_tensor_to_buffer(std::move(Z));
}


template <typename IO>
int MVZKExecProver<IO>::register_range_check_table(const std::vector<uint64_t>& data) {
    if (data[0] != real2fp(0)){
        LOG_ERROR("Not implemented: Range Table must start from 0 in F_p, range check with offset is not implemented currently.");
        exit(-1);
    }
    size_t new_hash = compute_vector_hash(data);

    for (size_t i = 0; i < range_check_tables.size(); ++i) {
        RangeCheckData* existing_table = range_check_tables[i].get();

        if (existing_table->data_hash != new_hash) {
            continue;
        }

        if (existing_table->data.size() != data.size()) {
            continue;
        }

        if (existing_table->data == data) {
            LOG_INFO("Range Check Table Deduplication: Reusing ID " << i);
            return (int)i;
        }
    }
    auto entry = std::make_unique<RangeCheckData>(data);

    range_check_tables.push_back(std::move(entry));
    return (int)range_check_tables.size() - 1;
}


template <typename IO>
void MVZKExecProver<IO>::submit_range_check(const PolyTensor& x, int table_id) {
    if (table_id < 0 || table_id >= range_check_tables.size()) {
        LOG_ERROR("Invalid Table ID: " << table_id);
        return;
    }

    RangeCheckData* table = range_check_tables[table_id].get();
    size_t threshold = MVZK_CONFIG_RANGE_CHECK_REQUEST_BUFFER_THRESHOLD;

    uint64_t* tracker_ptr = table->version_tracker.data();
    size_t tracker_size = table->version_tracker.size();

    const uint64_t* x_ptr = x.get_real_vals_ptr();
    size_t len = x.total_elements;

    size_t offset = 0;

    while (offset < len) {
        size_t capacity = threshold - table->current_buffer_size;
        size_t current_chunk = std::min(capacity, len - offset);

        PolyTensor sub_x({(int)current_chunk}, x.degree, PolyTensor::Coeffs);
        if (!x.flat_coeffs.empty()) {
            for (int d = 0; d <= x.degree; ++d) {
                std::memcpy(sub_x.get_coeffs_ptr(d), x.get_coeffs_ptr(d) + offset, current_chunk * sizeof(uint64_t));
            }
        }

        RangeCheckData::Request req;
        req.v_j_hint.reserve(current_chunk);

        const uint64_t* chunk_ptr = sub_x.get_real_vals_ptr();

        for(size_t i = 0; i < current_chunk; ++i) {
            uint64_t item = chunk_ptr[i];

            #ifndef NDEBUG
            if (item >= tracker_size) {
                LOG_ERROR("Range Check Value OOB! Val: " << item << " Size: " << tracker_size);
                exit(-1);
            }
            #endif

            req.v_j_hint.push_back(tracker_ptr[item]++);
        }

        req.vals = std::move(sub_x);
        table->buffer.push_back(std::move(req));
        table->current_buffer_size += current_chunk;

        if (table->current_buffer_size >= threshold) {
            process_range_check_batch(table);
        }
        offset += current_chunk;
    }
    x.is_consumed = true;
}


template <typename IO>
void MVZKExecProver<IO>::process_range_check_batch(RangeCheckData* table) {
    if (table->buffer.empty()) return;

    size_t N_Query = table->current_buffer_size;
    size_t N_Table = table->data.size();

    std::vector<uint64_t> hints_flat;
    hints_flat.reserve(N_Query);
    for (const auto& req : table->buffer) {
        hints_flat.insert(hints_flat.end(), req.v_j_hint.begin(), req.v_j_hint.end());
    }

    std::vector<uint64_t> finals_flat;
    finals_flat.reserve(N_Table);

    const uint64_t* tracker_ptr = table->version_tracker.data();
    for (const auto& kv : table->data) {
        finals_flat.push_back(tracker_ptr[kv]);
    }

    PolyTensor V_J = this->input({(int)N_Query, 1}, hints_flat);
    PolyTensor V_FINAL = this->input({(int)N_Table, 1}, finals_flat);

    block seed;
    io->recv_data(&seed, sizeof(block));
    this->prg.reseed(&seed);
    uint64_t sy, sv, r;
    this->prg.random_data(&sy, sizeof(uint64_t));
    this->prg.random_data(&sv, sizeof(uint64_t));
    this->prg.random_data(&r, sizeof(uint64_t));
    sy = sy % PR; sv = sv % PR; r = r % PR;

    auto Big_Vals = concat_buffer_tensors(table);
    PolyTensor Term_Q_Read = (Big_Vals * sy) + (V_J * sv) - r;
    PolyTensor Term_Q_Write = Term_Q_Read + sv;

    std::vector<uint64_t> table_consts;
    table_consts.reserve(N_Table);
    uint64_t P_Init_Scalar = 1;

    for (const auto& kv : table->data) {
        uint64_t term_val = mult_mod(kv, sy);
        table_consts.push_back(term_val);
        P_Init_Scalar = mult_mod(P_Init_Scalar, add_mod(term_val, PR - r));
    }

    PolyTensor P_Table_Const = PolyTensor::from_public({(int)N_Table, 1}, table_consts);
    PolyTensor Term_Final = P_Table_Const + (V_FINAL * sv) - r;

    // Phase 4: Grand Product
    std::vector<PolyTensor> lhs_parts;
    lhs_parts.reserve(2);
    lhs_parts.emplace_back(product_blocks(Term_Q_Read, "RangeCheck Read Product"));
    lhs_parts.emplace_back(product_blocks(Term_Final, "RangeCheck Final Product"));
    PolyTensor P_LHS = fast_tree_product(lhs_parts);

    PolyTensor P_RHS_Write = product_blocks(Term_Q_Write, "RangeCheck Write Product");

    // Phase 5: Check Zero
    PolyTensor Z = P_LHS - (P_RHS_Write * P_Init_Scalar);
    Z.is_constraint = true;

    table->buffer.clear();
    table->current_buffer_size = 0;

    std::fill(table->version_tracker.begin(), table->version_tracker.end(), 0);
    this->submit_tensor_to_buffer(std::move(Z));
}

#endif
