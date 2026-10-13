#include "PolyDelta.h"
#include "../exec/MVZKExec.h"

PolyDelta::~PolyDelta() {
    if (!is_consumed && degree > 0 && MVZKExec::mvzk_exec && !is_pre_generated) {
        MVZKExec::mvzk_exec->submit_to_buffer(std::move(*this));
    }
}

PolyDelta PolyDelta::clone() const {
    PolyDelta res;

    res.degree = this->degree;
    res.coeffs = this->coeffs;
    res.key = this->key;
    res.is_pre_generated = this->is_pre_generated;

    res.is_consumed = false;
    res.is_constraint = false;

    return res;
}

// 1. PolyDelta vs PolyDelta.

PolyDelta PolyDelta::operator+(const PolyDelta& rhs) const {
    return MVZKExec::mvzk_exec->add(*this, rhs);
}

PolyDelta PolyDelta::operator-(const PolyDelta& rhs) const {
    return MVZKExec::mvzk_exec->sub(*this, rhs);
}

PolyDelta PolyDelta::operator*(const PolyDelta& rhs) const {
    return MVZKExec::mvzk_exec->mul(*this, rhs);
}

// 2. PolyDelta vs scalar.

PolyDelta PolyDelta::operator+(uint64_t scalar) const {
    return MVZKExec::mvzk_exec->add_const(*this, scalar);
}

PolyDelta PolyDelta::operator-(uint64_t scalar) const {
    return MVZKExec::mvzk_exec->sub_const(*this, scalar);
}

PolyDelta PolyDelta::operator*(uint64_t scalar) const {
    return MVZKExec::mvzk_exec->mul_const(*this, scalar);
}

// 3. Friend operators.

// c + poly
PolyDelta operator+(uint64_t scalar, const PolyDelta& rhs) {
    return MVZKExec::mvzk_exec->add_const(rhs, scalar);
}

// c * poly
PolyDelta operator*(uint64_t scalar, const PolyDelta& rhs) {
    return MVZKExec::mvzk_exec->mul_const(rhs, scalar);
}

// c - poly
PolyDelta operator-(uint64_t scalar, const PolyDelta& rhs) {
    return MVZKExec::mvzk_exec->sub_const_rev(scalar, rhs);
}

void PolyDelta::store_relation(const PolyDelta& lhs, const PolyDelta& rhs){
    PolyDelta constraint = lhs - rhs;
    constraint.is_constraint = true;
    if (MVZKExec::mvzk_exec) {
        MVZKExec::mvzk_exec->submit_to_buffer(std::move(constraint));
    }
    return;
}

void PolyDelta::store_relation(const PolyDelta& lhs, const uint64_t rhs){
    PolyDelta constraint = lhs - rhs;
    constraint.is_constraint = true;
    if (MVZKExec::mvzk_exec) {
        MVZKExec::mvzk_exec->submit_to_buffer(std::move(constraint));
    }
    return;
}

PolyDelta& PolyDelta::operator+=(const PolyDelta& rhs) {
    if (MVZKExec::mvzk_exec) {
        MVZKExec::mvzk_exec->add_assign(*this, rhs);
    }
    return *this;
}

PolyDelta& PolyDelta::operator*=(uint64_t scalar) {
    if (MVZKExec::mvzk_exec) {
        MVZKExec::mvzk_exec->mul_assign(*this, scalar);
    }
    return *this;
}

PolyDelta& PolyDelta::operator-=(const PolyDelta& rhs) {
    if (MVZKExec::mvzk_exec) {
        MVZKExec::mvzk_exec->sub_assign(*this, rhs);
    }
    return *this;
}

PolyDelta& PolyDelta::operator+=(uint64_t scalar) {
    if (MVZKExec::mvzk_exec) {
        MVZKExec::mvzk_exec->add_assign_const(*this, scalar);
    }
    return *this;
}

PolyDelta& PolyDelta::operator-=(uint64_t scalar) {
    if (MVZKExec::mvzk_exec) {
        MVZKExec::mvzk_exec->sub_assign_const(*this, scalar);
    }
    return *this;
}
