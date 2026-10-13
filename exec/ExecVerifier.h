#ifndef MVZK_EXECUTION_VERIFIER_H__
#define MVZK_EXECUTION_VERIFIER_H__

#include "MVZKNetBase.h"
#include "emp-zk/emp-zk-arith/ostriple.h"
#include "emp-zk/emp-vole/emp-vole.h"
#include "emp-zk/emp-vole/vole_triple.h"
#include "../config.h"
#include "../data_type/PolyDelta.h"
#include "../data_type/PolyTensor.h"
#include "../utility.h"
#include "emp-tool/utils/hash.h"
#include "../operations/nonlinear.h"
#include "../operations/lut.h"
#include <omp.h>
#include <vector>

#define LOW64(x) _mm_extract_epi64((block)x, 0)
#define HIGH64(x) _mm_extract_epi64((block)x, 1)

template <typename IO>
class MVZKExecVerifier : public MVZKNetBase<IO> {
public:
    IO *io = nullptr;
    VoleTriple<IO> *vole = nullptr;
    int party;
    int threads;
    __uint128_t delta;
    std::vector<uint64_t> delta_powers_cache;

    std::vector<PolyDelta> check_buffer;
    std::vector<PolyTensor> check_tensor_buffer;

    using typename MVZKExec::LUTData;
    using typename MVZKExec::RangeCheckData;
    using MVZKExec::lut_tables;
    using MVZKExec::range_check_tables;
    using MVZKExec::flush_all_luts;
    using MVZKExec::flush_all_range_checks;
    using MVZKExec::concat_buffer_tensors;
    using MVZKExec::split_to_scalars;
    using MVZKExec::prg;

    MVZKExecVerifier(MVZKNetworkManager<IO>* net, const std::string& delta_filepath = "") : MVZKNetBase<IO>(net) {
        this->threads = MVZK_CONFIG_THREADS_NUM;

        MVZKExec::mvzk_exec = this;

        this->io = net->vole_ios[0];

        this->vole = new VoleTriple<IO>(ALICE, this->threads, net->vole_ios);

        if (delta_filepath.empty()) {
            delta_gen();
            LOG_INFO("Generated fresh Delta for new session.");
        } else {
            load_delta_from_file(delta_filepath);
            LOG_INFO("Loaded existing Delta from Checkpoint.");
        }
        vole->setup(delta);

        __uint128_t tmp;
        vole->extend(&tmp, 1);

        delta_powers_cache.resize(2);
        delta_powers_cache[0] = 1;
        delta_powers_cache[1] = (uint64_t)delta;

        if (net->broadcast_ios != nullptr) {
            LOG_INFO("Multi Broadcast Net Enabled. Sending Ready Signal to Prover.");
            this->sync_network();
            LOG_INFO("All READY. Proceed to next step.");
        }
    }

    MVZKExecVerifier(IO **ios, IO **init_broadcast_ios = nullptr, int broadcast_party_id = -1, const std::string& delta_filepath = "")
        : MVZKExecVerifier(create_legacy_net(ios, init_broadcast_ios, broadcast_party_id), delta_filepath) {
    }

    MVZKExecVerifier(int party_id, int base_port, const std::vector<std::string>& ips = {}, const std::string& delta_path = "", int override_num_parties = 0)
        : MVZKExecVerifier(new MVZKNetworkManager<IO>(party_id, base_port, ips, -1, override_num_parties), delta_path) {
        this->owns_net_manager = true;
    }

    virtual ~MVZKExecVerifier() {
        if(vole) delete vole;

        if (this->net_manager && this->net_manager->num_vole_ios == -1) {
            delete this->net_manager;
        }
    }

private:
    static MVZKNetworkManager<IO>* create_legacy_net(IO** ios, IO** b_ios, int party_id) {
        auto net = new MVZKNetworkManager<IO>();

        net->my_party_id = (party_id == -1) ? 2 : party_id;
        net->vole_ios = ios;
        net->broadcast_ios = b_ios;
        net->num_parties = b_ios ? MVZK_CONFIG_NUM_PARTIES : 2;

        net->num_vole_ios = -1;
        return net;
    }

public:
    void submit_to_buffer(PolyDelta&& pd) override;
    void submit_tensor_to_buffer(PolyTensor&& pt) override;
    void submit_non_zero_tensor_to_buffer(const PolyTensor& target) override;
    PolyTensor refresh_tensor_degree(const PolyTensor& high_degree_tensor, const std::string& check_name) override;
    void check_all() override;
    void finalize_protocol() override;
    PolyDelta add(const PolyDelta& lhs, const PolyDelta& rhs) override;
    PolyTensor add(const PolyTensor& lhs, const PolyTensor& rhs) override;
    PolyDelta sub(const PolyDelta& lhs, const PolyDelta& rhs) override;
    PolyTensor sub(const PolyTensor& lhs, const PolyTensor& rhs) override;
    PolyDelta mul(const PolyDelta& lhs, const PolyDelta& rhs) override;
    PolyTensor mul(const PolyTensor& lhs, const PolyTensor& rhs) override;
    void add_assign(PolyDelta& lhs, const PolyDelta& rhs) override;
    void add_assign(PolyTensor& lhs, const PolyTensor& rhs) override;
    void sub_assign(PolyDelta& lhs, const PolyDelta& rhs) override;
    void sub_assign(PolyTensor& lhs, const PolyTensor& rhs) override;
    void mul_assign(PolyDelta& lhs, uint64_t val) override;
    void mul_assign(PolyTensor& lhs, uint64_t val) override;
    void mul_assign(PolyTensor& lhs, const PolyTensor& rhs) override;
    void add_assign_const(PolyDelta& lhs, uint64_t val) override;
    void add_assign_const(PolyTensor& lhs, uint64_t val) override;
    void sub_assign_const(PolyDelta& lhs, uint64_t val) override;
    PolyDelta add_const(const PolyDelta& lhs, uint64_t val) override;
    PolyDelta sub_const(const PolyDelta& lhs, uint64_t val) override;
    PolyDelta sub_const_rev(uint64_t val, const PolyDelta& rhs) override;
    PolyDelta mul_const(const PolyDelta& lhs, uint64_t val) override;
    PolyTensor MatMul(const PolyTensor& lhs, const PolyTensor& rhs) override;
    PolyTensor MatMul(const PolyTensor& lhs, const PolyTensor& rhs, int M, int K, int N) override;
    void permute_and_add_bias(
        const PolyTensor& src, PolyTensor& dst, const PolyTensor& bias,
        int N, int H_out, int W_out, int C_out
    ) override;
    void helper_linear_add_bias(
        PolyTensor& data, const PolyTensor& bias, int Rows, int Cols
    ) override;
    void helper_permute_and_add_bias_1d(
        const PolyTensor& src, PolyTensor& dst, const PolyTensor& bias,
        int N, int L_out, int C_out
    ) override;
    PolyTensor avgpool2d(PolyTensor& pt_in, int kernel_size, int stride, int padding, bool back_to_sum_pool) override;
    [[deprecated("WARNING: DO NOT USE Standalone BatchNorm2D. Please use offline Conv-BN folding instead.")]]
    PolyTensor batchnorm2d(PolyTensor& pt_in, const std::vector<uint64_t>& A, const std::vector<uint64_t>& B) override;
    PolyTensor relu(PolyTensor& x, uint64_t bitlen, uint64_t digdec_k, bool do_truncation, uint64_t scale) override;
    [[deprecated("This relu should not be used as it fails to consider the truncation bit and the scale.")]]
    PolyTensor relu_legacy(PolyTensor& x, uint64_t bitlen, uint64_t digdec_k, bool do_truncation);
    PolyTensor maxpool2d(PolyTensor& pt_in, int kernel_size, int stride, int padding, uint64_t bitlen, uint64_t digdec_k, uint64_t scale) override;
    [[deprecated("This maxpool should not be used as it fails to consider the truncation bit and the scale.")]]
    PolyTensor maxpool2d_legacy(PolyTensor& pt_in, int kernel_size, int stride, int padding, uint64_t bitlen, uint64_t digdec_k);
    PolyTensor integrated_nl(PolyTensor& pt_in, int kernel_size, int stride, int padding, uint64_t bitlen, uint64_t digdec_k, bool do_truncation, uint64_t scale) override;
    [[deprecated("This integrated_nl should not be used as it fails to consider the truncation bit and the scale.")]]
    PolyTensor integrated_nl_legacy(PolyTensor& pt_in, int kernel_size, int stride, int padding, uint64_t bitlen, uint64_t digdec_k, bool do_truncation);
    void input(std::vector<PolyDelta>& pdList, const std::vector<uint64_t>& raw_data) override;
    PolyDelta input(uint64_t raw_data) override;
    std::vector<PolyDelta> extend_vope_from_vole(uint64_t deg, int size = 1) override;
    PolyTensor input(const std::vector<int>& shape, const std::vector<uint64_t>& raw_data) override;
    uint64_t delta_pow(uint64_t a, uint64_t deg);
    std::vector<PolyDelta> vole2pd(std::vector<__uint128_t>& vole_data, int size) override;
    PolyDelta recv_poly();
    std::vector<PolyDelta> recv_poly_vector();
    bool verify_poly(PolyDelta& mine, PolyDelta& hers);
    std::vector<uint64_t> reveal(const PolyTensor& pt) override;
    void save_delta_to_file(const std::string& filepath) const;
    void load_delta_from_file(const std::string& filepath);
    uint64_t broadcast(uint64_t dummy_val = 0, bool assume_ideal_broadcast = MVZK_CONFIG_BROADCAST_ASSUME_IDEAL) override;
    void consistency_check(const std::vector<std::vector<PolyTensor*>>& data) override;
    void debug_print(const PolyDelta& pd, std::string name = "") override;
    void debug_print(const PolyTensor& pt, std::string name, int limit = 4) override;
    void debug_print_recvd(const PolyDelta& pd, std::string name);
    bool debug_instant_check(const PolyDelta& pd) override;
    bool debug_instant_check(const PolyTensor& pt) override;

protected:
    void accumulate_delta_buffer(PolyDelta& final_item) override;
    void accumulate_tensor_buffer(PolyDelta& final_item) override;
    int register_lut_table(const std::vector<std::pair<uint64_t, uint64_t>>& data) override;
    void submit_lut_check(const PolyTensor& keys, const PolyTensor& vals, int table_id) override;
    void process_lut_batch(LUTData* table) override;
    int register_range_check_table(const std::vector<uint64_t>& data) override;
    void submit_range_check(const PolyTensor& x, int table_id) override;
    void process_range_check_batch(RangeCheckData* table) override;

private:
    void delta_gen();
};

#include "../backend/VerifierImpl/check.hpp"
#include "../backend/VerifierImpl/arith.hpp"
#include "../backend/VerifierImpl/linear.hpp"
#include "../backend/VerifierImpl/nl.hpp"
#include "../backend/VerifierImpl/io.hpp"
#include "../backend/VerifierImpl/consistency.hpp"
#include "../backend/VerifierImpl/debug.hpp"
#include "../backend/VerifierImpl/lut.hpp"

#endif
