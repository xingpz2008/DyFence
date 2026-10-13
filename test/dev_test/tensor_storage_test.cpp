#include <filesystem>
#include <iostream>
#include <stdexcept>
#include "data_type/PolyTensor.h"
#include "exec/MVZKExec.h"
#include "exec/ExecPlaintext.h"
#include "operations/core_math.h"

MVZKExec* MVZKExec::mvzk_exec = nullptr;

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main(int argc, char** argv) {
    try {
        require(argc == 2, "usage: tensor_storage_test <scratch-directory>");
        for (int party : {emp::ALICE, emp::BOB, 3, 33}) {
            PolyTensor::set_default_store(party);
            PolyTensor original({2, 2}, 1);
            require(original.flat_coeffs.empty() == (party != emp::ALICE), "coefficient storage");
            require(original.flat_keys.empty() == (party == emp::ALICE), "key storage");
            const PolyTensor& constant = original;
            auto copied = constant.flatten();
            require(original.total_elements == 4, "const flatten preserves input");
            auto moved = original.reshape({4});
            require(original.total_elements == 0 && moved.total_elements == 4, "mutable reshape moves input");
            const auto file = std::filesystem::path(argv[1]) / (std::to_string(party) + " tensor.bin");
            std::filesystem::create_directories(file.parent_path());
            moved.save_to_file(file.string());
            PolyTensor restored;
            restored.load_from_file(file.string());
            require(restored.store == moved.store && restored.shape == moved.shape, "checkpoint metadata");
            require(restored.flat_coeffs == moved.flat_coeffs && restored.flat_keys == moved.flat_keys, "checkpoint storage");
            copied.mark_consumed(); moved.mark_consumed(); restored.mark_consumed();
        }
        PolyTensor::set_default_store(0);
        PolyTensor factors({3}, 1);
        factors.flat_coeffs = {2, 5, 11, 3, 7, 13};
        factors.flat_keys = {5, 12, 24};
        auto product = product_blocks(factors, "test product");
        require(product.flat_coeffs == std::vector<uint64_t>({110, 449, 608, 273}), "linear polynomial product");
        require(product.flat_keys == std::vector<uint64_t>({1440}), "key product");
        product.mark_consumed();
        PolyTensor quadratic({2}, 2);
        quadratic.flat_coeffs = {1, 4, 2, 5, 3, 6};
        quadratic.flat_keys = {6, 15};
        auto higher = product_blocks(quadratic, "test quadratic");
        require(higher.flat_coeffs == std::vector<uint64_t>({4, 13, 28, 27, 18}), "generic polynomial fallback");
        higher.mark_consumed();
        PolyTensor scalar({1}, 1);
        scalar.flat_coeffs = {17, 19}; scalar.flat_keys = {36};
        auto single = product_blocks(scalar, "test single");
        require(single.flat_coeffs == std::vector<uint64_t>({17, 19}), "single factor");
        single.mark_consumed();
        PolyTensor constants({3}, 0);
        constants.flat_coeffs = {2, 3, 5}; constants.flat_keys = {2, 3, 5};
        auto zero_degree = product_blocks(constants, "test constants");
        require(zero_degree.flat_coeffs == std::vector<uint64_t>({30}), "degree-zero product");
        zero_degree.mark_consumed();
        {
            MVZKExecPlaintext<emp::NetIO> plaintext(emp::ALICE);
            const int block_size = MVZK_CONFIG_MULT_PRODUCT_THRESHOLD / 2;
            for (int count : {block_size - 1, block_size, block_size + 1,
                              2 * block_size, 2 * block_size + 1, 4 * block_size + 1}) {
                PolyTensor boundary({count}, 1);
                std::fill(boundary.get_coeffs_ptr(1), boundary.get_coeffs_ptr(1) + count, 1);
                boundary.get_coeffs_ptr(1)[count - 1] = 7;
                auto reduced = product_blocks(boundary, "block and refresh boundary");
                require(reduced.get_real_vals_vector() == std::vector<uint64_t>({7}),
                        "odd blocks and refresh boundary preserve product");
                reduced.mark_consumed();
            }
        }
        MVZKExec::mvzk_exec = nullptr;
        std::cout << "[SUCCESS] Tensor storage, ownership, serialization and product checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
