/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_PREPARED_ARRAY_H
#define OVSPATIALQUERY_PREPARED_ARRAY_H

#include <ovspatialquery_internal.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace ovspatialquery
{
namespace detail
{

class PreparedInputArray
{
  public:
    PreparedInputArray();
    ~PreparedInputArray();
    PreparedInputArray(PreparedInputArray&& other) noexcept;
    PreparedInputArray& operator=(PreparedInputArray&& other) noexcept;
    PreparedInputArray(const PreparedInputArray&)            = delete;
    PreparedInputArray& operator=(const PreparedInputArray&) = delete;

    static ovspatialquery_api_status_t prepare(const ovspatialquery_input_array_t* source, uint8_t expectedCode,
                                               uint8_t expectedBits, int32_t expectedRank,
                                               int64_t expectedTrailingDimension, const char* column,
                                               PreparedInputArray& outArray);

    const uint8_t* data() const;
    int64_t        shape(uint32_t dimension) const;
    uint64_t       byteCount() const;
    bool           initialized() const;
    void           reset();

  private:
    void swap(PreparedInputArray& other) noexcept;

    std::vector<int64_t>             mShape;
    std::vector<uint8_t>             mCopiedData;
    const uint8_t*                   mBorrowedData = nullptr;
    ovspatialquery_input_ownership_t mOwnership    = OVSPATIALQUERY_INPUT_BORROWED_UNTIL_TERMINAL;
    bool                             mInitialized  = false;
    uint64_t                         mByteCount    = 0u;
};

// Reads a fixed-width scalar of type T out of a prepared array's byte buffer
// at the given row index.
template <typename T>
T readPrimitive(const PreparedInputArray& array, size_t index) noexcept
{
    T value{};
    std::memcpy(&value, array.data() + index * sizeof(value), sizeof(value));
    return value;
}

} // namespace detail
} // namespace ovspatialquery

#endif
