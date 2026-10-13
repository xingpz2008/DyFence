#ifndef MVZK_LUT_H__
#define MVZK_LUT_H__

#include <vector>
#include <utility>
#include <cstdint>
#include "../data_type/PolyTensor.h"

class MVZKExec;

// Lightweight handle for a public lookup table managed by the active MVZKExec.
class PublicTable {
private:
    int table_id = -1;

public:
    PublicTable() = default;

    // Register a public table and keep its executor-side table id.
    PublicTable(const std::vector<std::pair<uint64_t, uint64_t>>& table_data);

    // Queue a lookup request for later batched verification.
    void lookup(const PolyTensor& keys, const PolyTensor& vals);

    int get_id() const { return table_id; }
};

// Lightweight handle for a range-check table managed by the active MVZKExec.
class RangeCheckTable{
private:
    int table_id = -1;

public:
    RangeCheckTable() = default;
    RangeCheckTable(const std::vector<uint64_t>& table_data);
    void range_check(const PolyTensor& x, const std::string& tag = "Unknown");
    int get_id() const {return table_id;}
};

void LUT_FlushAll();
void Range_Check_FlushAll();

#endif
