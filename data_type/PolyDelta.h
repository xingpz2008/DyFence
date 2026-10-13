#ifndef POLY_DELTA_H__
#define POLY_DELTA_H__

#include <vector>
#include <cstdint>
#include <utility>

// Forward declaration to avoid including the execution interface here.
class MVZKExec;

class PolyDelta {
public:
    // Public state used by the execution backend.

    int degree;
    std::vector<uint64_t> coeffs; // Prover
    uint64_t key;                 // Verifier
    mutable bool is_consumed;
    bool is_pre_generated;
    bool is_constraint;

    // Constructors and ownership.

    PolyDelta() : degree(0), key(0), is_consumed(false), is_pre_generated(false), is_constraint(false) {}

    // Move construction.
    PolyDelta(PolyDelta&& other) noexcept
        : degree(other.degree), coeffs(std::move(other.coeffs)),
          key(other.key), is_consumed(other.is_consumed), is_pre_generated(other.is_pre_generated),
          is_constraint(other.is_constraint) {
        other.is_consumed = true;
        other.degree = 0;
    }

    // Move assignment.
    PolyDelta& operator=(PolyDelta&& other) noexcept {
        if (this != &other) {
            degree = other.degree;
            coeffs = std::move(other.coeffs);
            key = other.key;
            is_consumed = other.is_consumed;
            other.is_consumed = true;
            other.degree = 0;
            is_pre_generated = other.is_pre_generated;
            is_constraint = other.is_constraint;
        }
        return *this;
    }

    // Copying is disabled because authenticated values are move-owned.
    PolyDelta(const PolyDelta&) = delete;
    PolyDelta& operator=(const PolyDelta&) = delete;
    PolyDelta clone() const;

    // The destructor is implemented in the .cpp file where MVZKExec is complete.
    ~PolyDelta();

    uint64_t get_real_val() const { return coeffs[degree]; }

    // Arithmetic operators.

    // PolyDelta vs PolyDelta.
    PolyDelta operator+(const PolyDelta& rhs) const;
    PolyDelta operator-(const PolyDelta& rhs) const;
    PolyDelta operator*(const PolyDelta& rhs) const;

    // PolyDelta vs scalar.
    PolyDelta operator+(uint64_t scalar) const;
    PolyDelta operator-(uint64_t scalar) const;
    PolyDelta operator*(uint64_t scalar) const;

    // Scalar on the left-hand side.
    friend PolyDelta operator+(uint64_t scalar, const PolyDelta& rhs);
    friend PolyDelta operator*(uint64_t scalar, const PolyDelta& rhs);
    friend PolyDelta operator-(uint64_t scalar, const PolyDelta& rhs);

    // Relation helpers.
    static void store_relation(const PolyDelta& lhs, const PolyDelta& rhs);
    static void store_relation(const PolyDelta& lhs, const uint64_t rhs);

    // In-place operators.
    PolyDelta& operator+=(const PolyDelta& rhs);
    PolyDelta& operator-=(const PolyDelta& rhs);
    PolyDelta& operator*=(uint64_t scalar);
    PolyDelta& operator+=(uint64_t scalar);
    PolyDelta& operator-=(uint64_t scalar);
};

#endif
