// PolyTensor.cpp
#include "PolyTensor.h"
#include "emp-tool/emp-tool.h"
#include "../exec/MVZKExec.h"
#include "../utility.h"
#include <cstring>
#include <fstream>
#include <stdexcept>

PolyTensor::Store PolyTensor::default_store = PolyTensor::Both;

void PolyTensor::set_default_store(int party) {
    if (party == ALICE) {
        default_store = Coeffs;
    } else if (party >= BOB) {
        default_store = Keys;
    } else {
        default_store = Both;
    }
}

// Destructor and basic helpers.
PolyTensor::~PolyTensor() {
    // Moved-from and empty tensors do not need lifecycle checks.
    if (total_elements == 0) {
        return;
    }

    // Warn when an authenticated tensor is destroyed without being consumed or
    // explicitly submitted as a constraint.
    if (!is_consumed && !is_constraint && (GlobalLogLevel() <= LEVEL_WARN)) {
        std::cerr << "[WARNING] PolyTensor destroyed but NOT consumed! "
                  << "Potential circuit logic error (dangling wire)."
                  << " Shape: [";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cerr << shape[i] << (i == shape.size() - 1 ? "" : ", ");
        }
        std::cerr << "], Degree: " << degree << std::endl;
    }
}

PolyTensor PolyTensor::clone() const {
    PolyTensor res;
    res.shape = this->shape;
    res.total_elements = this->total_elements;
    res.degree = this->degree;
    res.store = this->store;

    res.flat_coeffs = this->flat_coeffs;
    res.flat_keys = this->flat_keys;

    res.is_consumed = false;
    res.is_constraint = false;

    return res;
}

PolyTensor PolyTensor::from_public(const std::vector<int>& shape, const std::vector<uint64_t>& data) {
    PolyTensor res(shape, 0);

    if (data.size() != res.total_elements) {
        LOG_ERROR("Size mismatch in PolyTensor::from_public");
        exit(-1);
    }

    // Public constants are represented as degree-0 values.
    if (res.store == Both || res.store == Coeffs) {
        std::memcpy(res.get_coeffs_ptr(0), data.data(), data.size() * sizeof(uint64_t));
    }
    if (res.store == Both || res.store == Keys) {
        std::memcpy(res.get_keys_ptr(), data.data(), data.size() * sizeof(uint64_t));
    }

    return res;
}

void PolyTensor::copy_from(const PolyTensor& src, size_t dest_offset) {
    if (dest_offset + src.total_elements > this->total_elements) {
        LOG_ERROR("Buffer overflow in copy_from!");
        exit(-1);
    }

    // The destination must have enough degree slots for the source tensor.
    if (this->degree < src.degree) {
        LOG_ERROR("Destination tensor degree too small in copy_from!");
        exit(-1);
    }

    if (src.total_elements == 0) {
        src.mark_consumed();
        return;
    }

    size_t copy_bytes = src.total_elements * sizeof(uint64_t);

    if (this->store == Both || this->store == Coeffs) {
        if (src.store != Both && src.store != Coeffs) {
            LOG_ERROR("Missing source coeffs in copy_from!");
            exit(-1);
        }

        // Copy available source degrees; higher destination degrees remain zero.
        for (int d = 0; d <= src.degree; ++d) {
            std::memcpy(this->get_coeffs_ptr(d) + dest_offset,
                        src.get_coeffs_ptr(d),
                        copy_bytes);
        }

        std::memcpy(this->get_real_vals_ptr() + dest_offset,
                    src.get_real_vals_ptr(),
                    copy_bytes);
    }

    // Verifier keys are stored as one block independent of degree.
    if (this->store == Both || this->store == Keys) {
        if (src.store != Both && src.store != Keys) {
            LOG_ERROR("Missing source keys in copy_from!");
            exit(-1);
        }

        std::memcpy(this->get_keys_ptr() + dest_offset,
                    src.get_keys_ptr(),
                    copy_bytes);
    }

    src.mark_consumed();
}

// 2. Tensor vs Tensor (In-Place)

PolyTensor& PolyTensor::operator+=(const PolyTensor& rhs) {
    if (MVZKExec::mvzk_exec) {
        MVZKExec::mvzk_exec->add_assign(*this, rhs);
    }
    return *this;
}

PolyTensor& PolyTensor::operator-=(const PolyTensor& rhs) {
    if (MVZKExec::mvzk_exec) {
        MVZKExec::mvzk_exec->sub_assign(*this, rhs);
    }
    return *this;
}

PolyTensor& PolyTensor::operator*=(const PolyTensor& rhs) {
    if (MVZKExec::mvzk_exec) {
        MVZKExec::mvzk_exec->mul_assign(*this, rhs);
    }
    return *this;
}

// 3. Tensor vs Scalar (In-Place)

PolyTensor& PolyTensor::operator+=(uint64_t scalar) {
    if (MVZKExec::mvzk_exec) {
        MVZKExec::mvzk_exec->add_assign_const(*this, scalar);
    }
    return *this;
}

PolyTensor& PolyTensor::operator-=(uint64_t scalar) {
    uint64_t neg_scalar = (scalar == 0) ? 0 : (PR - scalar);
    return this->operator+=(neg_scalar);
}

PolyTensor& PolyTensor::operator*=(uint64_t scalar) {
    if (MVZKExec::mvzk_exec) {
        MVZKExec::mvzk_exec->mul_assign(*this, scalar);
    }
    return *this;
}

// 4. Tensor vs Tensor operators returning new tensors.

PolyTensor PolyTensor::operator+(const PolyTensor& rhs) const {
    if (MVZKExec::mvzk_exec) {
        return MVZKExec::mvzk_exec->add(*this, rhs);
    }
    return PolyTensor();
}

PolyTensor PolyTensor::operator-(const PolyTensor& rhs) const {
    if (MVZKExec::mvzk_exec) {
        return MVZKExec::mvzk_exec->sub(*this, rhs);
    }
    return PolyTensor();
}

PolyTensor PolyTensor::operator*(const PolyTensor& rhs) const {
    if (MVZKExec::mvzk_exec) {
        return MVZKExec::mvzk_exec->mul(*this, rhs);
    }
    return PolyTensor();
}

// 5. Tensor vs Scalar operators returning new tensors.

PolyTensor PolyTensor::operator+(uint64_t scalar) const {
    PolyTensor res = this->clone();
    res += scalar;
    this->is_consumed = true;
    return res;
}

PolyTensor PolyTensor::operator-(uint64_t scalar) const {
    PolyTensor res = this->clone();
    res -= scalar;
    this->is_consumed = true;
    return res;
}

PolyTensor PolyTensor::operator*(uint64_t scalar) const {
    PolyTensor res = this->clone();
    res *= scalar;
    this->is_consumed = true;
    return res;
}

// 6. Friend operators for scalar left-hand sides.

// c + T -> T + c
PolyTensor operator+(uint64_t scalar, const PolyTensor& rhs) {
    return rhs + scalar;
}

// c * T -> T * c
PolyTensor operator*(uint64_t scalar, const PolyTensor& rhs) {
    return rhs * scalar;
}

// c - T -> -T + c -> T*(-1) + c
PolyTensor operator-(uint64_t scalar, const PolyTensor& rhs) {
    PolyTensor res = rhs.clone();

    if (MVZKExec::mvzk_exec) {
        MVZKExec::mvzk_exec->mul_assign(res, PR - 1);
    }

    if (MVZKExec::mvzk_exec) {
        MVZKExec::mvzk_exec->add_assign_const(res, scalar);
    }

    return res;
}

PolyTensor PolyTensor::MatMul(const PolyTensor& rhs) const {
    if (MVZKExec::mvzk_exec) {
        return MVZKExec::mvzk_exec->MatMul(*this, rhs);
    }
    return PolyTensor();
}

PolyTensor MatMul(const PolyTensor& lhs, const PolyTensor& rhs) {
    if (MVZKExec::mvzk_exec) {
        return MVZKExec::mvzk_exec->MatMul(lhs, rhs);
    }
    return PolyTensor();
}

// 7. Check Helper

void PolyTensor::store_relation(const PolyTensor& lhs, const PolyTensor& rhs, const std::string& tag) {
    int max_degree = std::max(lhs.degree, rhs.degree);
    MVZKExec::mvzk_exec->stat_constraints[tag][max_degree] += lhs.total_elements;

    PolyTensor diff = lhs - rhs;

    // A stored relation is represented as a zero-valued constraint tensor.
    diff.is_constraint = true;

    MVZKExec::mvzk_exec->submit_tensor_to_buffer(std::move(diff));

    lhs.mark_consumed();
    rhs.mark_consumed();
}

void PolyTensor::store_zero_relation(PolyTensor& lhs, const std::string& tag){
    MVZKExec::mvzk_exec->stat_constraints[tag][lhs.degree] += lhs.total_elements;
    lhs.is_constraint = true;
    MVZKExec::mvzk_exec->submit_tensor_to_buffer(std::move(lhs));
}

void PolyTensor::store_self_relation(const PolyTensor& lhs, const std::string& tag){
    // Check validity for a non-zero tensor through the backend buffer.
    MVZKExec::mvzk_exec->stat_constraints[tag][lhs.degree] += lhs.total_elements;
    lhs.mark_consumed();
    MVZKExec::mvzk_exec->submit_non_zero_tensor_to_buffer(lhs);
}

// Shape Manipulation

PolyTensor PolyTensor::flatten() {
    if (this->shape.size() < 2) {
        PolyTensor res = std::move(*this);
        res.is_consumed = false;
        res.is_constraint = false;
        return res;
    }

    int batch_size = this->shape[0];

    size_t flattened_dim = 1;
    for (size_t i = 1; i < this->shape.size(); ++i) {
        flattened_dim *= (size_t)this->shape[i];
    }

    PolyTensor res = std::move(*this);

    res.shape.clear();
    res.shape.push_back(batch_size);
    res.shape.push_back((int)flattened_dim);
    res.is_consumed = false;
    res.is_constraint = false;

    return res;
}

PolyTensor PolyTensor::flatten() const {
    if (this->shape.size() < 2) return this->clone();

    int batch_size = this->shape[0];

    size_t flattened_dim = 1;
    for (size_t i = 1; i < this->shape.size(); ++i) {
        flattened_dim *= (size_t)this->shape[i];
    }

    PolyTensor res = this->clone();

    res.shape.clear();
    res.shape.push_back(batch_size);
    res.shape.push_back((int)flattened_dim);

    this->is_consumed = true;

    return res;
}

PolyTensor PolyTensor::reshape(const std::vector<int>& new_shape) {
    size_t old_count = this->total_elements;
    size_t new_count = 1;
    for(int s : new_shape) new_count *= (size_t)s;

    if (old_count != new_count) {
        LOG_ERROR("Reshape size mismatch! Old: " << old_count << ", New: " << new_count);
        exit(-1);
    }

    PolyTensor res = std::move(*this);
    res.shape = new_shape;
    res.is_consumed = false;
    res.is_constraint = false;
    return res;
}

PolyTensor PolyTensor::reshape(const std::vector<int>& new_shape) const {
    size_t old_count = this->total_elements;
    size_t new_count = 1;
    for(int s : new_shape) new_count *= (size_t)s;

    if (old_count != new_count) {
        LOG_ERROR("Reshape size mismatch! Old: " << old_count << ", New: " << new_count);
        exit(-1);
    }

    PolyTensor res = this->clone();
    res.shape = new_shape;
    this->is_consumed = true;
    return res;
}

// Degree Manipulation

PolyTensor PolyTensor::refresh_degree(const std::string& check_name) {
    return MVZKExec::mvzk_exec->refresh_tensor_degree(*this, check_name);
}

// Binary IO

void PolyTensor::save_to_file(const std::string& filepath) const {
    std::ofstream out(filepath, std::ios::binary);
    if (!out) {
        LOG_ERROR("[ERROR] Failed to open file for writing: " + filepath);
        exit(-1);
    }

    // Shape metadata.
    size_t shape_size = shape.size();
    out.write(reinterpret_cast<const char*>(&shape_size), sizeof(size_t));
    if (shape_size > 0) {
        out.write(reinterpret_cast<const char*>(shape.data()), shape_size * sizeof(int));
    }

    // Core metadata.
    out.write(reinterpret_cast<const char*>(&total_elements), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&degree), sizeof(int));

    // Lifecycle flags.
    out.write(reinterpret_cast<const char*>(&is_consumed), sizeof(bool));
    out.write(reinterpret_cast<const char*>(&is_constraint), sizeof(bool));

    // Optional coefficient block.
    bool has_coeffs = !flat_coeffs.empty();
    out.write(reinterpret_cast<const char*>(&has_coeffs), sizeof(bool));
    if (has_coeffs) {
        size_t coeffs_size = flat_coeffs.size();
        out.write(reinterpret_cast<const char*>(&coeffs_size), sizeof(size_t));
        out.write(reinterpret_cast<const char*>(flat_coeffs.data()), coeffs_size * sizeof(uint64_t));
    }

    // Optional key block.
    bool has_keys = !flat_keys.empty();
    out.write(reinterpret_cast<const char*>(&has_keys), sizeof(bool));
    if (has_keys) {
        size_t keys_size = flat_keys.size();
        out.write(reinterpret_cast<const char*>(&keys_size), sizeof(size_t));
        out.write(reinterpret_cast<const char*>(flat_keys.data()), keys_size * sizeof(uint64_t));
    }

    out.close();
}

void PolyTensor::load_from_file(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in) {
        LOG_ERROR("[ERROR] Failed to open file for reading: " + filepath);
        exit(-1);
    }

    // Shape metadata.
    size_t shape_size;
    in.read(reinterpret_cast<char*>(&shape_size), sizeof(size_t));
    shape.resize(shape_size);
    if (shape_size > 0) {
        in.read(reinterpret_cast<char*>(shape.data()), shape_size * sizeof(int));
    }

    // Core metadata.
    in.read(reinterpret_cast<char*>(&total_elements), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&degree), sizeof(int));

    // Lifecycle flags.
    in.read(reinterpret_cast<char*>(&is_consumed), sizeof(bool));
    in.read(reinterpret_cast<char*>(&is_constraint), sizeof(bool));

    // Optional coefficient block.
    bool has_coeffs;
    in.read(reinterpret_cast<char*>(&has_coeffs), sizeof(bool));
    if (has_coeffs) {
        size_t coeffs_size;
        in.read(reinterpret_cast<char*>(&coeffs_size), sizeof(size_t));
        flat_coeffs.resize(coeffs_size);
        in.read(reinterpret_cast<char*>(flat_coeffs.data()), coeffs_size * sizeof(uint64_t));
    } else {
        flat_coeffs.clear();
    }

    // Optional key block.
    bool has_keys;
    in.read(reinterpret_cast<char*>(&has_keys), sizeof(bool));
    if (has_keys) {
        size_t keys_size;
        in.read(reinterpret_cast<char*>(&keys_size), sizeof(size_t));
        flat_keys.resize(keys_size);
        in.read(reinterpret_cast<char*>(flat_keys.data()), keys_size * sizeof(uint64_t));
    } else {
        flat_keys.clear();
    }

    if (has_coeffs && has_keys) {
        store = Both;
    } else if (has_coeffs) {
        store = Coeffs;
    } else if (has_keys) {
        store = Keys;
    } else {
        store = Both;
    }

    in.close();
}
