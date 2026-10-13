#ifndef MVZK_EXECUTION_H__
#define MVZK_EXECUTION_H__

#include <vector>
#include <cstdint>
#include <map>
#include <memory>
#include <sstream>
#include "emp-zk/emp-vole/utility.h"
#include "../utility.h"

class PolyDelta;
class PolyTensor;

class MVZKExec {
public:
    static MVZKExec *mvzk_exec;

    int party; // ALICE(Prover) or BOB(Verifier)
    PRG prg;

    // IO is a template name, we do not define it here.
    // IO **broadcast_ios = nullptr;

    MVZKExec() { mvzk_exec = this; }
    virtual ~MVZKExec() {}

protected:
    virtual void accumulate_delta_buffer(PolyDelta& final_item) = 0;
    virtual void accumulate_tensor_buffer(PolyDelta& final_item) = 0;

    static PolyTensor tree_product(std::vector<PolyTensor>& items) {
        if (items.empty()) return PolyTensor::from_public({1}, {1});
        if (items.size() == 1) return std::move(items[0]);

        size_t mid = items.size() / 2;
        std::vector<PolyTensor> left(std::make_move_iterator(items.begin()),
                                     std::make_move_iterator(items.begin() + mid));
        std::vector<PolyTensor> right(std::make_move_iterator(items.begin() + mid),
                                      std::make_move_iterator(items.end()));

        return tree_product(left) * tree_product(right);
    }

    std::vector<PolyTensor> split_to_scalars(const PolyTensor& big_tensor) {
        size_t len = big_tensor.total_elements;

        if (len < MVZK_OMP_SIZE_THRESHOLD * 2 || omp_in_parallel()) {
            std::vector<PolyTensor> res;
            res.reserve(len);

            for(size_t i = 0; i < len; ++i) {
                res.emplace_back(std::vector<int>{1}, big_tensor.degree);
                PolyTensor& t = res.back();

                // Copy Coeffs
                if (!big_tensor.flat_coeffs.empty()) {
                    for(int d = 0; d <= big_tensor.degree; ++d) {
                        t.flat_coeffs[d] = big_tensor.flat_coeffs[d * len + i];
                    }
                }

                // Copy Keys
                if(!big_tensor.flat_keys.empty()) {
                    t.flat_keys[0] = big_tensor.flat_keys[i];
                }

                t.is_consumed = false;
            }

            big_tensor.is_consumed = true;
            return res;
        }

        int max_threads = omp_get_max_threads();

        std::vector<std::vector<PolyTensor>> thread_buffers(max_threads);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto& local_res = thread_buffers[tid];

            local_res.reserve(len / max_threads + 16);

            #pragma omp for schedule(static) nowait
            for(size_t i = 0; i < len; ++i) {
                local_res.emplace_back(std::vector<int>{1}, big_tensor.degree);
                PolyTensor& t = local_res.back();

                if (!big_tensor.flat_coeffs.empty()) {
                    for(int d = 0; d <= big_tensor.degree; ++d) {
                        t.flat_coeffs[d] = big_tensor.flat_coeffs[d * len + i];
                    }
                }
                if(!big_tensor.flat_keys.empty()) {
                    t.flat_keys[0] = big_tensor.flat_keys[i];
                }
                t.is_consumed = false;
            }
        }

        std::vector<PolyTensor> res;
        res.reserve(len);

        for(auto& buf : thread_buffers) {
            res.insert(res.end(),
                       std::make_move_iterator(buf.begin()),
                       std::make_move_iterator(buf.end()));
        }

        big_tensor.is_consumed = true;
        return res;
    }

    struct LUTData {
        // LUTData is the meta LUT instance, containing public table data and all its related lookup requests.

        std::vector<std::pair<uint64_t, uint64_t>> data;

        std::map<std::pair<uint64_t, uint64_t>, uint64_t> version_tracker;

        struct Request {
            // Request struct may involve a bunch of x-y-v pairs.
            PolyTensor keys;
            PolyTensor vals;
            std::vector<uint64_t> v_j_hint;
            // Version number: We use this plaintext format when submiting lookup reqests. At verification, before obtaining challenge,
            // we will convert it to commited value.
        };
        // The buffer contains a lot of requests.
        std::vector<Request> buffer;

        size_t current_buffer_size = 0;

        size_t data_hash;

        LUTData(const std::vector<std::pair<uint64_t, uint64_t>>& _data)
            : data(_data) {
            data_hash = compute_vector_hash(_data);
        }
    };

    struct RangeCheckData {
        // LUTData is the meta LUT instance, containing public table data and all its related lookup requests.

        std::vector<uint64_t> data;

        // Version counters indexed by table value.
        std::vector<uint64_t> version_tracker;

        struct Request {
            // Request struct may involve a bunch of x-y-v pairs.
            PolyTensor vals;
            std::vector<uint64_t> v_j_hint;
            // Version number: We use this plaintext format when submiting lookup reqests. At verification, before obtaining challenge,
            // we will convert it to commited value.
        };
        // The buffer contains a lot of requests.
        std::vector<Request> buffer;

        size_t current_buffer_size = 0;

        size_t data_hash;

        RangeCheckData(const std::vector<uint64_t>& _data)
            : data(_data) {
            data_hash = compute_vector_hash(_data);
            version_tracker.resize(_data.size(), 0);
        }
    };

    std::pair<PolyTensor, PolyTensor> concat_buffer_tensors(const LUTData* table) {
        if (table->buffer.empty()) {
            return {PolyTensor(), PolyTensor()};
        }

        size_t N_Query = table->current_buffer_size;

        int max_deg_key = 0;
        int max_deg_val = 0;

        int first_deg_key = table->buffer[0].keys.degree;
        int first_deg_val = table->buffer[0].vals.degree;

        bool inconsistent_key = false;
        bool inconsistent_val = false;

        for (const auto& req : table->buffer) {
            max_deg_key = std::max(max_deg_key, req.keys.degree);
            max_deg_val = std::max(max_deg_val, req.vals.degree);

            // Updating flag
            req.keys.is_consumed = true;
            req.vals.is_consumed = true;

            if (req.keys.degree != first_deg_key) inconsistent_key = true;
            if (req.vals.degree != first_deg_val) inconsistent_val = true;
        }

        if (inconsistent_key) {
            LOG_WARN("LUT Keys have mixed degrees! Promoting to max degree: " << max_deg_key);
        }
        if (inconsistent_val) {
            LOG_WARN("Values have mixed degrees! Promoting to max degree: " << max_deg_val);
        }

        PolyTensor Big_Keys({(int)N_Query, 1}, max_deg_key);
        PolyTensor Big_Vals({(int)N_Query, 1}, max_deg_val);

        size_t offset = 0;

        for (const auto& req : table->buffer) {
            Big_Keys.copy_from(req.keys, offset);
            Big_Vals.copy_from(req.vals, offset);

            offset += req.keys.total_elements;
        }

        if (offset != N_Query) {
            LOG_ERROR("Buffer size mismatch in concat! Expected " << N_Query << ", got " << offset);
            exit(-1);
        }

        return {std::move(Big_Keys), std::move(Big_Vals)};
    }

    PolyTensor concat_buffer_tensors(const RangeCheckData* table) {
        if (table->buffer.empty()) {
            return PolyTensor();
        }

        size_t N_Query = table->current_buffer_size;

        int max_deg_key = 0;
        int max_deg_val = 0;

        int first_deg_val = table->buffer[0].vals.degree;

        bool inconsistent_val = false;

        for (const auto& req : table->buffer) {
            max_deg_val = std::max(max_deg_val, req.vals.degree);

            // Updating flag
            req.vals.is_consumed = true;

            if (req.vals.degree != first_deg_val) inconsistent_val = true;
        }

        if (inconsistent_val) {
            LOG_WARN("LUT Values have mixed degrees! Promoting to max degree: " << max_deg_val);
        }

        PolyTensor Big_Vals({(int)N_Query, 1}, max_deg_val);

        size_t offset = 0;

        for (const auto& req : table->buffer) {
            Big_Vals.copy_from(req.vals, offset);

            offset += req.vals.total_elements;
        }

        if (offset != N_Query) {
            LOG_ERROR("Buffer size mismatch in concat! Expected " << N_Query << ", got " << offset);
            exit(-1);
        }

        return std::move(Big_Vals);
    }

    void helper_im2col_polytensor(
        const PolyTensor& src, PolyTensor& dst,
        int N, int C, int H, int W,
        int kH, int kW,
        int padding, int stride, int dilation,
        int H_out, int W_out
    ) {
        // 1. Verifier Keys
        if (!src.flat_keys.empty()) {
            im2col_kernel(src.get_keys_ptr(), dst.get_keys_ptr(),
                          N, C, H, W, kH, kW, padding, stride, dilation, H_out, W_out);
        }
        // 2. Prover Real Values
        // Removed

        if (!src.flat_coeffs.empty()) {
            for (int d = 0; d <= src.degree; ++d) {
                im2col_kernel(src.get_coeffs_ptr(d), dst.get_coeffs_ptr(d),
                              N, C, H, W, kH, kW, padding, stride, dilation, H_out, W_out);
            }
        }
    }

    void helper_transpose_weight_polytensor(
        const PolyTensor& src, PolyTensor& dst,
        int C_out, int C_in, int kH, int kW
    ) {
        if (!src.flat_keys.empty()) {
            transpose_weight_kernel(src.get_keys_ptr(), dst.get_keys_ptr(), C_out, C_in, kH, kW);
        }
        if (!src.flat_coeffs.empty()) {
            for (int d = 0; d <= src.degree; ++d) {
                transpose_weight_kernel(src.get_coeffs_ptr(d), dst.get_coeffs_ptr(d), C_out, C_in, kH, kW);
            }
        }
    }

    virtual void permute_and_add_bias(
        const PolyTensor& src, PolyTensor& dst, const PolyTensor& bias,
        int N, int H_out, int W_out, int C_out
    ) = 0;

    void helper_transpose_matrix_polytensor(
        const PolyTensor& src, PolyTensor& dst, int Rows, int Cols
    ) {
        if (!src.flat_keys.empty()) {
            transpose_matrix_kernel(src.get_keys_ptr(), dst.get_keys_ptr(), Rows, Cols);
        }
        if (!src.flat_coeffs.empty()) {
            for (int d = 0; d <= src.degree; ++d) {
                transpose_matrix_kernel(src.get_coeffs_ptr(d), dst.get_coeffs_ptr(d), Rows, Cols);
            }
        }
    }

    virtual void helper_linear_add_bias(
        PolyTensor& data, const PolyTensor& bias, int Rows, int Cols
    ) = 0;

    virtual void helper_permute_and_add_bias_1d(
        const PolyTensor& src, PolyTensor& dst, const PolyTensor& bias,
        int N, int L_out, int C_out
    ) = 0;

public:
    virtual void submit_to_buffer(PolyDelta&& pd) = 0;
    virtual void check_all() = 0;
    virtual void submit_tensor_to_buffer(PolyTensor&& pt) = 0;
    virtual void submit_non_zero_tensor_to_buffer(const PolyTensor& target) = 0;
    virtual void consistency_check(const std::vector<std::vector<PolyTensor*>>& data) = 0;
    virtual void finalize_protocol() = 0;
    //virtual void submit_matmul_tensor_to_buffer(PolyTensor&& pt) = 0;

    virtual void input(std::vector<PolyDelta>& pdList, const std::vector<uint64_t>& raw_data) = 0;
    virtual PolyDelta input(uint64_t raw_data) = 0;
    virtual std::vector<PolyDelta> vole2pd(std::vector<__uint128_t>& vole_data, int size) = 0;
    virtual std::vector<PolyDelta> extend_vope_from_vole(uint64_t deg, int size) = 0;
    virtual PolyTensor refresh_tensor_degree(const PolyTensor& high_degree_tensor, const std::string& check_name) = 0;

    // Poly vs Poly
    virtual PolyDelta add(const PolyDelta& lhs, const PolyDelta& rhs) = 0;
    virtual PolyDelta sub(const PolyDelta& lhs, const PolyDelta& rhs) = 0;
    virtual PolyDelta mul(const PolyDelta& lhs, const PolyDelta& rhs) = 0;

    // Poly vs Const
    virtual PolyDelta add_const(const PolyDelta& lhs, uint64_t val) = 0;
    virtual PolyDelta sub_const(const PolyDelta& lhs, uint64_t val) = 0;
    virtual PolyDelta sub_const_rev(uint64_t val, const PolyDelta& rhs) = 0; // val - poly
    virtual PolyDelta mul_const(const PolyDelta& lhs, uint64_t val) = 0;

    // Poly vs Poly
    virtual void add_assign(PolyDelta& lhs, const PolyDelta& rhs) = 0;
    virtual void sub_assign(PolyDelta& lhs, const PolyDelta& rhs) = 0;

    // Poly vs Const
    virtual void add_assign_const(PolyDelta& lhs, uint64_t val) = 0;
    virtual void sub_assign_const(PolyDelta& lhs, uint64_t val) = 0;
    virtual void mul_assign(PolyDelta& lhs, uint64_t val) = 0;

    // virtual bool verify_poly(PolyDelta& mine, PolyDelta& hers) = 0;

    virtual PolyTensor input(const std::vector<int>& shape, const std::vector<uint64_t>& raw_data) = 0;

    // Tensor vs Tensor
    virtual PolyTensor add(const PolyTensor& lhs, const PolyTensor& rhs) = 0;
    virtual PolyTensor sub(const PolyTensor& lhs, const PolyTensor& rhs) = 0;
    virtual PolyTensor mul(const PolyTensor& lhs, const PolyTensor& rhs) = 0;

    // Tensor vs Tensor
    virtual void add_assign(PolyTensor& lhs, const PolyTensor& rhs) = 0;
    virtual void sub_assign(PolyTensor& lhs, const PolyTensor& rhs) = 0;
    virtual void mul_assign(PolyTensor& lhs, const PolyTensor& rhs) = 0;

    // Tensor vs Scalar (Const)
    virtual void add_assign_const(PolyTensor& lhs, uint64_t val) = 0;

    virtual void mul_assign(PolyTensor& lhs, uint64_t val) = 0;

    // Matrix Multiplication
    virtual PolyTensor MatMul(const PolyTensor& lhs, const PolyTensor& rhs) = 0;
    virtual PolyTensor MatMul(const PolyTensor& lhs, const PolyTensor& rhs, int M, int K, int N) = 0;

    virtual std::vector<uint64_t> reveal(const PolyTensor& pt) = 0;

    // Part 3: LUT Section

    // Stored place for LUT table instance. This instance contains the LUT public table itself, and the related requests.
    std::vector<std::unique_ptr<LUTData>> lut_tables;
    // LUT registration, return the table ID
    virtual int register_lut_table(const std::vector<std::pair<uint64_t, uint64_t>>& data) = 0;
    virtual void submit_lut_check(const PolyTensor& keys, const PolyTensor& vals, int table_id) = 0;
    // This function process one public table with all its lookup requests.
    // For security and performance consideration, we only choose to check one public table each time.
    virtual void process_lut_batch(LUTData* table) = 0;

    void flush_all_luts(bool require_immediate_check = false) {
        for (auto& table_ptr : lut_tables) {
            if (table_ptr->current_buffer_size > 0) {
                process_lut_batch(table_ptr.get());
            }
        }
        for (auto& range_check_ptr : range_check_tables) {
            if (range_check_ptr->current_buffer_size > 0) {
                process_range_check_batch(range_check_ptr.get());
            }
        }

        if (require_immediate_check){
            check_all();
        }
    }

    // Range Check Section
    std::vector<std::unique_ptr<RangeCheckData>> range_check_tables;
    virtual int register_range_check_table(const std::vector<uint64_t>& data) = 0;
    virtual void submit_range_check(const PolyTensor& x, int table_id) = 0;
    virtual void process_range_check_batch(RangeCheckData* table) = 0;

    void flush_all_range_checks(bool require_immediate_check = false) {
        for (auto& table_ptr : range_check_tables) {
            if (table_ptr->current_buffer_size > 0) {
                process_range_check_batch(table_ptr.get());
            }
        }

        if (require_immediate_check){
            check_all();
        }
    }

    // Part 4: Linear Operation Section

    virtual PolyTensor conv2d(
        const PolyTensor& input,
        const PolyTensor& weight,
        const PolyTensor& bias,
        int stride, int padding, int dilation
    ) {
        int N = input.shape[0];
        int C_in = input.shape[1];
        int H_in = input.shape[2];
        int W_in = input.shape[3];

        int C_out = weight.shape[0];

        int kH = weight.shape[2];
        int kW = weight.shape[3];

        if (input.shape[1] != weight.shape[1]) {
            LOG_ERROR("Conv2D Channel Mismatch! Input: " << C_in << ", Weight: " << weight.shape[1]);
            exit(-1);
        }

        // H_out = floor((H_in + 2*padding - dilation*(kH-1) - 1) / stride + 1)
        int H_out = (H_in + 2 * padding - dilation * (kH - 1) - 1) / stride + 1;
        int W_out = (W_in + 2 * padding - dilation * (kW - 1) - 1) / stride + 1;

        if (H_out <= 0 || W_out <= 0) {
            LOG_ERROR("Conv2D output dimension <= 0. Check your padding/stride settings!");
            exit(-1);
        }

        // Step 1: Im2Col (Input -> Matrix)

        int im2col_M = N * H_out * W_out;

        int im2col_K = C_in * kH * kW;

        PolyTensor col_matrix({im2col_M, im2col_K}, input.degree);

        helper_im2col_polytensor(input, col_matrix,
                                 N, C_in, H_in, W_in, kH, kW,
                                 padding, stride, dilation,
                                 H_out, W_out);

        // Weight: (C_out, K) -> Weight_T: (K, C_out)
        PolyTensor weight_matrix({im2col_K, C_out}, weight.degree);
        helper_transpose_weight_polytensor(weight, weight_matrix, C_out, C_in, kH, kW);

        // Step 3: GEMM (MatMul)

        // (M x K) * (K x N) -> (M x N)

        PolyTensor output_matrix = col_matrix.MatMul(weight_matrix);

        // Step 4: Reshape & Bias Add

        PolyTensor final_res({N, C_out, H_out, W_out}, output_matrix.degree);

        this->permute_and_add_bias(output_matrix, final_res, bias, N, H_out, W_out, C_out);

        input.is_consumed = true;
        weight.is_consumed = true;
        if (bias.total_elements > 0) bias.is_consumed = true;

        return final_res;
    }

    // --- Linear Layer ---

    // Weight: (Out_Features, In_Features)
    // Bias: (Out_Features)
    virtual PolyTensor linear(
        const PolyTensor& input,
        const PolyTensor& weight,
        const PolyTensor& bias
    ) {
        if (weight.shape.size() != 2) {
            LOG_ERROR("Linear weight must be 2D (Out, In)");
            exit(-1);
        }
        int Out_Features = weight.shape[0];
        int In_Features = weight.shape[1];

        bool is_valid_shape = false;
        if (input.shape.back() == In_Features) {
            is_valid_shape = true;
        } else if (input.shape.size() >= 2 && input.shape[1] == In_Features && input.total_elements % In_Features == 0) {
            is_valid_shape = true;
        }

        if (!is_valid_shape && (GlobalLogLevel() <= LEVEL_ERROR)) {
            std::ostringstream shape_stream;
            for(int s : input.shape) shape_stream << s << ",";
            LOG_ERROR("Linear input/weight mismatch! Input shape: ("
                      << shape_stream.str()
                      << ") Weight in_features: " << In_Features);
            exit(-1);
        }

        size_t K = In_Features;
        size_t M = input.total_elements / K;

        // 2. Weight Transpose: (Out, In) -> (In, Out)

        PolyTensor weight_T({(int)K, Out_Features}, weight.degree);
        helper_transpose_matrix_polytensor(weight, weight_T, Out_Features, In_Features);

        // 3. MatMul
        // (M, K) * (K, Out) -> (M, Out)
        PolyTensor output_mat = MatMul(input, weight_T, (int)M, (int)K, Out_Features);

        // 4. Bias Add (Row Broadcast)
        if (bias.total_elements > 0) {
            this->helper_linear_add_bias(output_mat, bias, M, Out_Features);
        }

        // 5. Restore Shape
        if (input.shape.back() == In_Features) {
            std::vector<int> out_shape = input.shape;
            out_shape.back() = Out_Features;
            output_mat.shape = out_shape;
        } else {
            output_mat.shape = {(int)M, Out_Features};
        }

        // Cleanup
        input.is_consumed = true;
        weight.is_consumed = true;
        if (bias.total_elements > 0) bias.is_consumed = true;

        return output_mat;
    }

    // --- Conv1D Layer ---
    virtual PolyTensor conv1d(
        const PolyTensor& input,
        const PolyTensor& weight,
        const PolyTensor& bias,
        int stride, int padding, int dilation
    ) {
        // 1. Shapes
        int N = input.shape[0];
        int C_in = input.shape[1];
        int L_in = input.shape[2];

        int C_out = weight.shape[0];
        // weight.shape[1] == C_in
        int K_size = weight.shape[2];

        // 2. Calculate Output Length
        int L_out = (L_in + 2 * padding - dilation * (K_size - 1) - 1) / stride + 1;
        if (L_out <= 0) {
            LOG_ERROR("[ERROR] Conv1D output length <= 0!");
            exit(-1);
        }

        // Step 1: Im2Col 1D (Input -> Matrix)

        // M = N * L_out
        // K = C_in * K_size
        int im2col_M = N * L_out;
        int im2col_K = C_in * K_size;

        PolyTensor col_matrix({im2col_M, im2col_K}, input.degree);

        // Helper 1D Im2Col (Need to define this specific helper below or use generic one)

        if (!input.flat_keys.empty()) {
            im2col_1d_kernel(input.get_keys_ptr(), col_matrix.get_keys_ptr(),
                             N, C_in, L_in, K_size, padding, stride, dilation, L_out);
        }
        if (!input.flat_coeffs.empty()) {
            for (int d = 0; d <= input.degree; ++d) {
                im2col_1d_kernel(input.get_coeffs_ptr(d), col_matrix.get_coeffs_ptr(d),
                                 N, C_in, L_in, K_size, padding, stride, dilation, L_out);
            }
        }

        // Step 2: Weight Transpose

        // Weight (C_out, K_dim) -> (K_dim, C_out)

        PolyTensor weight_matrix({im2col_K, C_out}, weight.degree);

        helper_transpose_matrix_polytensor(weight, weight_matrix, C_out, im2col_K);

        // Step 3: GEMM

        PolyTensor output_matrix = col_matrix.MatMul(weight_matrix);

        // Step 4: Reshape & Bias Add (1D Specific)

        PolyTensor final_res({N, C_out, L_out}, output_matrix.degree);

        this->helper_permute_and_add_bias_1d(output_matrix, final_res, bias, N, L_out, C_out);

        // Cleanup
        input.is_consumed = true;
        weight.is_consumed = true;
        if (bias.total_elements > 0) bias.is_consumed = true;

        return final_res;
    }

    virtual PolyTensor batchnorm2d(PolyTensor& pt_in, const std::vector<uint64_t>& A, const std::vector<uint64_t>& B) = 0;

    virtual PolyTensor avgpool2d(PolyTensor& pt_in, int kernel_size, int stride, int padding, bool back_to_sum_pool) = 0;

    // Part 5: Non-linear Operation Section

    virtual PolyTensor relu(PolyTensor& x, uint64_t bitlen, uint64_t digdec_k, bool do_truncation, uint64_t scale) = 0;

    virtual PolyTensor maxpool2d(PolyTensor& pt_in, int kernel_size, int stride, int padding, uint64_t bitlen, uint64_t digdec_k, uint64_t scale) = 0;

    virtual PolyTensor integrated_nl(PolyTensor& pt_in, int kernel_size, int stride, int padding, uint64_t bitlen, uint64_t digdec_k, bool do_truncation, uint64_t scale) = 0;

    // Part 6: Consistency Check Section

    virtual void sync_network() = 0;
    virtual emp::block coin_tossing() = 0;
    virtual uint64_t broadcast(uint64_t val = 0, bool assume_ideal_broadcast = MVZK_CONFIG_BROADCAST_ASSUME_IDEAL) = 0;

    static uint64_t generate_uniform_fp(emp::PRG& shared_prg) {
        uint64_t val;
        do {
            shared_prg.random_data(&val, sizeof(uint64_t));
        } while (val >= PR);
        return val;
    }

    // Part -1: Debug Helper Section

    virtual void debug_print(const PolyDelta& pd, std::string name = "") {
        std::cout << "[Base] " << name << " (Not implemented)" << std::endl;
    }

    virtual void debug_print(const PolyTensor& pt, std::string name, int limit = 4) {
        std::cout << "[Base] " << name << " (Not implemented)" << std::endl;
    }

    virtual bool debug_instant_check(const PolyDelta& pd) = 0;
    virtual bool debug_instant_check(const PolyTensor& pt) = 0;

    std::map<std::string, std::map<int, size_t>> stat_constraints;
    std::map<std::string, size_t> stat_range_checks;

    void print_profiler_report() {
        std::cout << "\n===========================================================" << std::endl;
        std::cout << "          ZK Constraint Profiler Report (Degree-Aware)     " << std::endl;
        std::cout << "===========================================================" << std::endl;

        std::cout << "[Polynomial Constraints]" << std::endl;
        size_t total_constraints = 0;

        for (const auto& tag_pair : stat_constraints) {
            std::cout << "  \033[36m[" << tag_pair.first << "]\033[0m" << std::endl;

            size_t sub_total = 0;

            for (const auto& deg_pair : tag_pair.second) {
                std::cout << "    |-- Degree " << std::left << std::setw(5) << deg_pair.first
                          << ": " << std::setw(10) << deg_pair.second << " constraints" << std::endl;
                sub_total += deg_pair.second;
                total_constraints += deg_pair.second;
            }
            std::cout << "    `-- Sub-Total : " << sub_total << "\n" << std::endl;
        }
        std::cout << "  > TOTAL Constraints Generated : " << total_constraints << "\n\n";

        std::cout << "[Range Checks (LUT Lookups)]" << std::endl;
        size_t total_rc = 0;
        for (const auto& pair : stat_range_checks) {
            std::cout << "  - " << std::left << std::setw(25) << pair.first
                      << ": " << pair.second << " elements" << std::endl;
            total_rc += pair.second;
        }
        std::cout << "  > TOTAL Range Checks          : " << total_rc << std::endl;
        std::cout << "===========================================================\n" << std::endl;
    }
};

#endif
