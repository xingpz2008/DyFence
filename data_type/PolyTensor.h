#ifndef POLY_TENSOR_H__
#define POLY_TENSOR_H__

#include <vector>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <utility>
#include <cassert>
#include <cstring>
#include <iomanip>

// Forward declaration to avoid including the execution backend here.
class MVZKExec;

class PolyTensor {
public:
    enum Store { Both, Coeffs, Keys };

    static Store default_store;
    static void set_default_store(int party);

    // Metadata.

    std::vector<int> shape;
    size_t total_elements;
    int degree;
    Store store;

    // Flattened storage in structure-of-arrays layout.

    std::vector<uint64_t> flat_coeffs; // Prover-side coefficients.
    std::vector<uint64_t> flat_keys;   // Verifier-side keys.

    // Lifecycle and constraint flags.

    mutable bool is_consumed;
    bool is_constraint;

    // Constructors and ownership.

    PolyTensor()
        : total_elements(0), degree(0), store(Both),
          is_consumed(false), is_constraint(false) {}

    PolyTensor(const std::vector<int>& s, int d)
        : PolyTensor(s, d, default_store) {}

    PolyTensor(const std::vector<int>& s, int d, Store storage)
        : shape(s), total_elements(0), degree(d), store(storage),
          is_consumed(false), is_constraint(false) {
        if (!shape.empty()) {
            total_elements = 1;
            for (int dim : shape) total_elements *= (size_t)dim;
        }
        if (store == Both || store == Coeffs) {
            flat_coeffs.resize(total_elements * (d + 1), 0);
        }
        if (store == Both || store == Keys) {
            flat_keys.resize(total_elements, 0);
        }
    }

    // Move construction.
    PolyTensor(PolyTensor&& other) noexcept
        : shape(std::move(other.shape)),
          total_elements(other.total_elements),
          degree(other.degree),
          store(other.store),
          flat_coeffs(std::move(other.flat_coeffs)),
          flat_keys(std::move(other.flat_keys)),
          is_consumed(other.is_consumed),
          is_constraint(other.is_constraint) {
        other.is_consumed = true;
        other.total_elements = 0;
        other.degree = 0;
        other.store = Both;
    }

    // Move assignment.
    PolyTensor& operator=(PolyTensor&& other) noexcept {
        if (this != &other) {
            shape = std::move(other.shape);
            total_elements = other.total_elements;
            degree = other.degree;
            store = other.store;
            flat_coeffs = std::move(other.flat_coeffs);
            flat_keys = std::move(other.flat_keys);
            is_consumed = other.is_consumed;
            is_constraint = other.is_constraint;
            other.is_consumed = true;
            other.total_elements = 0;
            other.store = Both;
        }
        return *this;
    }

    // Copying is disabled because tensors are move-owned authenticated values.
    PolyTensor(const PolyTensor&) = delete;
    PolyTensor& operator=(const PolyTensor&) = delete;

    ~PolyTensor();

    // Storage accessors and utilities.

    // Pointer to the first coefficient at the requested degree.
    uint64_t* get_coeffs_ptr(int deg) {
        assert(deg <= degree);
        return &flat_coeffs[deg * total_elements];
    }
    const uint64_t* get_coeffs_ptr(int deg) const {
        assert(deg <= degree);
        return &flat_coeffs[deg * total_elements];
    }

    uint64_t* get_real_vals_ptr() {
        return get_coeffs_ptr(degree);
    }
    const uint64_t* get_real_vals_ptr() const {
        return get_coeffs_ptr(degree);
    }

    std::vector<uint64_t> get_real_vals_vector() const {
        const uint64_t* start_ptr = get_coeffs_ptr(degree);

        return std::vector<uint64_t>(start_ptr, start_ptr + total_elements);
    }

    uint64_t* get_keys_ptr() { return flat_keys.data(); }
    const uint64_t* get_keys_ptr() const { return flat_keys.data(); }

    void mark_consumed() const { is_consumed = true; }

    // Deep copy.
    PolyTensor clone() const;

    void copy_from(const PolyTensor& src, size_t dest_offset);

    // Public tensor factory.
    static PolyTensor from_public(const std::vector<int>& shape, const std::vector<uint64_t>& data);

    // Relation helpers.
    static void store_relation(const PolyTensor& lhs, const PolyTensor& rhs, const std::string& tag = "Unknown");
    static void store_zero_relation(PolyTensor& lhs, const std::string& tag = "Unknown");
    // Stores a non-zero tensor for validity checking.
    static void store_self_relation(const PolyTensor& lhs, const std::string& tag = "Unknown");
    //static void store_relation(const PolyTensor& lhs, const std::vector<uint64_t> rhs);

    // Shape manipulation.
    PolyTensor flatten();
    PolyTensor flatten() const;
    PolyTensor reshape(const std::vector<int>& new_shape);
    PolyTensor reshape(const std::vector<int>& new_shape) const;

    // Degree manipulation.
    PolyTensor refresh_degree(const std::string& check_name = "Refreshed Tensor");

    // In-place operators.

    // Tensor vs Tensor
    PolyTensor& operator+=(const PolyTensor& rhs);
    PolyTensor& operator-=(const PolyTensor& rhs);
    PolyTensor& operator*=(const PolyTensor& rhs);

    // Tensor vs Scalar
    PolyTensor& operator+=(uint64_t scalar);
    PolyTensor& operator-=(uint64_t scalar);
    PolyTensor& operator*=(uint64_t scalar);

    // Operators returning new tensors.

    // Tensor op Tensor
    PolyTensor operator+(const PolyTensor& rhs) const;
    PolyTensor operator-(const PolyTensor& rhs) const;
    PolyTensor operator*(const PolyTensor& rhs) const;

    // Tensor op Scalar
    PolyTensor operator+(uint64_t scalar) const;
    PolyTensor operator-(uint64_t scalar) const;
    PolyTensor operator*(uint64_t scalar) const;

    // Scalar on the left-hand side.

    friend PolyTensor operator+(uint64_t scalar, const PolyTensor& rhs);
    friend PolyTensor operator*(uint64_t scalar, const PolyTensor& rhs);
    friend PolyTensor operator-(uint64_t scalar, const PolyTensor& rhs);

    // Serialization helpers.

    void save_to_file(const std::string& filepath) const;

    void load_from_file(const std::string& filepath);

    // 10. Matrix Multiplication API

    PolyTensor MatMul(const PolyTensor& rhs) const;
    friend PolyTensor MatMul(const PolyTensor& lhs, const PolyTensor& rhs);

    void print_poly_tensor(size_t limit = 25, const std::string& tag = "Debug") {
        std::vector<uint64_t> vals = get_real_vals_vector();

        if (vals.empty()) {
            return;
        }

        std::cout << "\033[32m[" << tag << "] Role: PROVER (Alice) | Shape: [";
        for (size_t i = 0; i < this->shape.size(); ++i) {
            std::cout << shape[i] << (i == shape.size() - 1 ? "" : ", ");
        }
        std::cout << "] | Total: " << this->total_elements << " | Degree: " << degree << "\033[0m" << std::endl;

        size_t print_count = std::min((size_t)total_elements, limit);
        std::cout << "  Values (first " << print_count << "): ";

        for (size_t i = 0; i < print_count; ++i) {
            std::cout << vals[i] << (i == print_count - 1 ? "" : ", ");
        }

        if (total_elements > limit) {
            std::cout << " ... (omitted)";
        }
        std::cout << std::endl << std::endl;
    }
};

#endif
