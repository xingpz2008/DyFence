#ifndef MVZK_BACKEND_VERIFIER_IMPL_LUT_HPP__
#define MVZK_BACKEND_VERIFIER_IMPL_LUT_HPP__

#include "../../exec/ExecVerifier.h"

template <typename IO>
int MVZKExecVerifier<IO>::register_lut_table(const std::vector<std::pair<uint64_t, uint64_t>>& data) {
    auto entry = std::make_unique<LUTData>(data);
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
            return (int)i;
        }
    }
    lut_tables.push_back(std::move(entry));
    return (int)lut_tables.size() - 1;
}


template <typename IO>
void MVZKExecVerifier<IO>::submit_lut_check(const PolyTensor& keys, const PolyTensor& vals, int table_id) {
    if (table_id < 0 || table_id >= lut_tables.size()) {
        LOG_ERROR("Invalid Table ID: " << table_id);
        return;
    }

    LUTData* table = lut_tables[table_id].get();

    LUTData::Request req;
    req.keys = keys.clone();
    req.vals = vals.clone();

    table->buffer.push_back(std::move(req));
    table->current_buffer_size += keys.total_elements;
    keys.is_consumed = true;
    vals.is_consumed = true;

    if (table->current_buffer_size >= MVZK_CONFIG_LUT_REQUEST_BUFFER_THRESHOLD) {
        process_lut_batch(table);
    }
}


template <typename IO>
void MVZKExecVerifier<IO>::process_lut_batch(LUTData* table) {
    if (table->buffer.empty()) return;

    size_t N_Query = table->current_buffer_size;
    size_t N_Table = table->data.size();

    std::vector<uint64_t> null_raw_data;
    PolyTensor V_J = this->input({(int)N_Query, 1}, null_raw_data);
    PolyTensor V_FINAL = this->input({(int)N_Table, 1}, null_raw_data);

    block seed;
    this->prg.random_block(&seed, 1);
    io->send_data(&seed, sizeof(block));
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

    auto [Big_Keys, Big_Vals] = concat_buffer_tensors(table);

    PolyTensor Term_Q_Read = (Big_Keys * sx) + (Big_Vals * sy) + (V_J * sv) - r;

    PolyTensor Term_Q_Write = Term_Q_Read + sv;

    std::vector<uint64_t> table_consts;
    table_consts.reserve(N_Table);
    uint64_t P_Init_Scalar = 1;

    for (const auto& kv : table->data) {
        uint64_t key = kv.first;
        uint64_t val = kv.second;

        uint64_t term_val = add_mod(mult_mod(key, sx), mult_mod(val, sy));
        table_consts.push_back(term_val);

        P_Init_Scalar = mult_mod(P_Init_Scalar, add_mod(term_val, PR - r));
    }

    PolyTensor P_Table_Const = PolyTensor::from_public({(int)N_Table, 1}, table_consts);

    PolyTensor Term_Final = P_Table_Const + (V_FINAL * sv) - r;

    // 4.1 LHS
    std::vector<PolyTensor> lhs_list = split_to_scalars(Term_Q_Read);
    std::vector<PolyTensor> final_list = split_to_scalars(Term_Final);

    lhs_list.reserve(lhs_list.size() + final_list.size());
    lhs_list.insert(lhs_list.end(),
                    std::make_move_iterator(final_list.begin()),
                    std::make_move_iterator(final_list.end()));

    PolyTensor P_LHS = fast_tree_product(lhs_list);

    // 4.2 RHS
    std::vector<PolyTensor> rhs_list = split_to_scalars(Term_Q_Write);
    PolyTensor P_RHS_Write = fast_tree_product(rhs_list);

    // Phase 5: Check Zero

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
int MVZKExecVerifier<IO>::register_range_check_table(const std::vector<uint64_t>& data) {
    if (data[0] != real2fp(0)){
        LOG_ERROR("Not implemented: Range Table must start from 0 in F_p, range check with offset is not implemented currently.");
        exit(-1);
    }
    auto entry = std::make_unique<RangeCheckData>(data);
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
            return (int)i;
        }
    }
    range_check_tables.push_back(std::move(entry));
    return (int)range_check_tables.size() - 1;
}


template <typename IO>
void MVZKExecVerifier<IO>::submit_range_check(const PolyTensor& x, int table_id) {
    if (table_id < 0 || table_id >= range_check_tables.size()) {
        LOG_ERROR("Invalid Table ID: " << table_id);
        return;
    }

    RangeCheckData* table = range_check_tables[table_id].get();

    size_t len = x.total_elements;
    size_t offset = 0;
    size_t threshold = MVZK_CONFIG_RANGE_CHECK_REQUEST_BUFFER_THRESHOLD;

    const uint64_t* x_keys_ptr = x.get_keys_ptr();

    while (offset < len) {
        size_t capacity = threshold - table->current_buffer_size;
        size_t current_chunk = std::min(capacity, len - offset);

        PolyTensor sub_x({(int)current_chunk}, x.degree, PolyTensor::Keys);

        if (!x.flat_keys.empty()) {
            std::memcpy(sub_x.get_keys_ptr(), x_keys_ptr + offset, current_chunk * sizeof(uint64_t));
        }

        RangeCheckData::Request req;
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
void MVZKExecVerifier<IO>::process_range_check_batch(RangeCheckData* table) {
    if (table->buffer.empty()) return;

    size_t N_Query = table->current_buffer_size;
    size_t N_Table = table->data.size();

    std::vector<uint64_t> null_raw_data;

    PolyTensor V_J = this->input({(int)N_Query, 1}, null_raw_data);
    PolyTensor V_FINAL = this->input({(int)N_Table, 1}, null_raw_data);

    block seed;
    this->prg.random_block(&seed, 1);
    io->send_data(&seed, sizeof(block));
    this->prg.reseed(&seed);

    uint64_t sy, sv, r;
    //this->prg.random_data(&sx, sizeof(uint64_t));
    this->prg.random_data(&sy, sizeof(uint64_t));
    this->prg.random_data(&sv, sizeof(uint64_t));
    this->prg.random_data(&r, sizeof(uint64_t));
    //sx = sx % PR;
    sy = sy % PR;
    sv = sv % PR;
    r = r % PR;

    PolyTensor Big_Vals = concat_buffer_tensors(table);

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

    // 4.1 LHS
    std::vector<PolyTensor> lhs_parts;
    lhs_parts.reserve(2);
    lhs_parts.emplace_back(product_blocks(Term_Q_Read, "RangeCheck Read Product"));
    lhs_parts.emplace_back(product_blocks(Term_Final, "RangeCheck Final Product"));
    PolyTensor P_LHS = fast_tree_product(lhs_parts);

    // 4.2 RHS
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
