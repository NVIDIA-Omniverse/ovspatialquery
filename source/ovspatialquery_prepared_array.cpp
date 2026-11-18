/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <ovspatialquery_prepared_array.h>

#include "ovspatialquery_input_layout.h"

#include <cstring>
#include <new>
#include <stdexcept>
#include <utility>

namespace ovspatialquery
{
namespace detail
{
// MSVC Debug vectors allocate iterator proxies even when empty. The count
// constructor permits those allocations to throw into prepare's
// OUT_OF_MEMORY handler; the noexcept default vector constructor terminates.
PreparedInputArray::PreparedInputArray() : mShape(0u), mCopiedData(0u) {}

PreparedInputArray::~PreparedInputArray() { reset(); }

PreparedInputArray::PreparedInputArray(PreparedInputArray&& other) noexcept { swap(other); }

PreparedInputArray& PreparedInputArray::operator=(PreparedInputArray&& other) noexcept
{
    if (this != &other)
    {
        reset();
        swap(other);
    }
    return *this;
}

void PreparedInputArray::swap(PreparedInputArray& other) noexcept
{
    mShape.swap(other.mShape);
    mCopiedData.swap(other.mCopiedData);
    std::swap(mBorrowedData, other.mBorrowedData);
    std::swap(mOwnership, other.mOwnership);
    std::swap(mInitialized, other.mInitialized);
    std::swap(mByteCount, other.mByteCount);
}

ovspatialquery_api_status_t PreparedInputArray::prepare(const ovspatialquery_input_array_t* source,
                                                        uint8_t expectedCode, uint8_t expectedBits,
                                                        int32_t expectedRank, int64_t expectedTrailingDimension,
                                                        const char* column, PreparedInputArray& outArray)
try
{
    if (source == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    const DLTensor& tensor = source->tensor;
    if (tensor.device.device_type != kDLCPU || tensor.device.device_id != 0 || tensor.dtype.code != expectedCode ||
        tensor.dtype.bits != expectedBits || tensor.dtype.lanes != 1u || tensor.ndim != expectedRank)
    {
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    if (tensor.shape == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    if (source->ownership != OVSPATIALQUERY_INPUT_BORROWED_UNTIL_TERMINAL &&
        source->ownership != OVSPATIALQUERY_INPUT_COPIED_ON_SUBMISSION)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }

    PreparedInputArray candidate;
    candidate.mShape.assign(tensor.shape, tensor.shape + expectedRank);
    if (candidate.mShape[0] < 0 || (expectedRank == 2 && candidate.mShape[1] != expectedTrailingDimension))
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    InputColumnLayout                 layout;
    const ovspatialquery_api_status_t layoutStatus = readInputColumnLayout(tensor, column, layout);
    if (layoutStatus != OVSPATIALQUERY_API_SUCCESS)
    {
        return layoutStatus;
    }
    const uint64_t byteCount = layout.compactBytes();
    if ((byteCount == 0u) != (tensor.data == nullptr) || (tensor.data == nullptr && tensor.byte_offset != 0u))
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    const uint8_t* logicalData =
        reinterpret_cast<const uint8_t*>(reinterpret_cast<uintptr_t>(tensor.data) + tensor.byte_offset);
    candidate.mInitialized = true;
    candidate.mByteCount   = byteCount;
    // The library reads compact rows, so it copies a column with any other
    // layout whatever its ownership (ADR-0053).
    candidate.mOwnership = layout.compact ? source->ownership : OVSPATIALQUERY_INPUT_COPIED_ON_SUBMISSION;
    if (candidate.mOwnership == OVSPATIALQUERY_INPUT_COPIED_ON_SUBMISSION)
    {
        candidate.mCopiedData.resize(static_cast<size_t>(byteCount));
        if (byteCount != 0u && layout.compact)
        {
            std::memcpy(candidate.mCopiedData.data(), logicalData, static_cast<size_t>(byteCount));
        }
        else if (byteCount != 0u)
        {
            gatherInputColumn(logicalData, layout, candidate.mCopiedData.data());
        }
    }
    else
    {
        candidate.mBorrowedData = logicalData;
    }
    outArray = std::move(candidate);
    return OVSPATIALQUERY_API_SUCCESS;
}
catch (const std::bad_alloc&)
{
    return OVSPATIALQUERY_API_OUT_OF_MEMORY;
}
catch (const std::length_error&)
{
    return OVSPATIALQUERY_API_OUT_OF_MEMORY;
}

const uint8_t* PreparedInputArray::data() const
{
    if (mByteCount == 0u)
    {
        return nullptr;
    }
    return mOwnership == OVSPATIALQUERY_INPUT_COPIED_ON_SUBMISSION ? mCopiedData.data() : mBorrowedData;
}

int64_t PreparedInputArray::shape(uint32_t dimension) const
{
    return dimension < mShape.size() ? mShape[dimension] : 0;
}

uint64_t PreparedInputArray::byteCount() const { return mByteCount; }

bool PreparedInputArray::initialized() const { return mInitialized; }

void PreparedInputArray::reset()
{
    mShape.clear();
    mCopiedData.clear();
    mBorrowedData = nullptr;
    mOwnership    = OVSPATIALQUERY_INPUT_BORROWED_UNTIL_TERMINAL;
    mInitialized  = false;
    mByteCount    = 0u;
}

} // namespace detail
} // namespace ovspatialquery
