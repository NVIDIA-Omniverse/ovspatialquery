/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_input_layout.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace ovspatialquery
{
namespace detail
{
namespace
{

thread_local std::string gInputLayoutDiagnostic;

ovspatialquery_api_status_t refuse(const char* column, const char* rule, long long stride = 0,
                                   int dimension = -1) noexcept
{
    // The first refused column keeps the diagnostic until it is cleared.
    if (!gInputLayoutDiagnostic.empty())
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    try
    {
        char text[160] = {};
        if (dimension >= 0)
        {
            std::snprintf(text, sizeof(text), "%s: stride %lld in dimension %d; %s", column, stride, dimension, rule);
        }
        else
        {
            std::snprintf(text, sizeof(text), "%s: %s", column, rule);
        }
        gInputLayoutDiagnostic = text;
    }
    catch (...)
    {
        gInputLayoutDiagnostic.clear();
    }
    return OVSPATIALQUERY_API_INVALID_ARGUMENT;
}

bool multiply(uint64_t left, uint64_t right, uint64_t& out) noexcept
{
    if (left != 0u && right > UINT64_MAX / left)
    {
        return false;
    }
    out = left * right;
    return true;
}

} // namespace

ovspatialquery_api_status_t readInputColumnLayout(const DLTensor& tensor, const char* column,
                                                  InputColumnLayout& out) noexcept
{
    out              = InputColumnLayout{};
    out.rows         = static_cast<uint64_t>(tensor.shape[0]);
    out.width        = tensor.ndim == 2 ? static_cast<uint64_t>(tensor.shape[1]) : 1u;
    out.elementBytes = tensor.dtype.bits / 8u;
    out.rowStride    = out.width;
    // An empty column's strides address nothing, and producers report any (ADR-0042).
    if (out.rows == 0u || out.width == 0u)
    {
        return OVSPATIALQUERY_API_SUCCESS;
    }
    if (tensor.strides != nullptr)
    {
        for (int dimension = 0; dimension < tensor.ndim; ++dimension)
        {
            if (tensor.strides[dimension] < 0)
            {
                return refuse(column, "strides must be nonnegative", static_cast<long long>(tensor.strides[dimension]),
                              dimension);
            }
        }
        // A dimension of extent one never steps, so its stride is not read.
        if (out.rows > 1u)
        {
            out.rowStride = static_cast<uint64_t>(tensor.strides[0]);
        }
        if (tensor.ndim == 2 && out.width > 1u)
        {
            out.columnStride = static_cast<uint64_t>(tensor.strides[1]);
        }
    }
    out.compact           = out.rowStride == out.width && out.columnStride == 1u;
    uint64_t compactBytes = 0u;
    uint64_t lastRow      = 0u;
    uint64_t lastColumn   = 0u;
    uint64_t spanBytes    = 0u;
    if (!multiply(out.rows, out.width, compactBytes) || !multiply(compactBytes, out.elementBytes, compactBytes) ||
        !multiply(out.rows - 1u, out.rowStride, lastRow) || !multiply(out.width - 1u, out.columnStride, lastColumn) ||
        lastColumn >= UINT64_MAX || lastRow > UINT64_MAX - lastColumn - 1u ||
        !multiply(lastRow + lastColumn + 1u, out.elementBytes, spanBytes))
    {
        return refuse(column, "the element or byte count overflows");
    }
    const uint64_t base = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tensor.data));
    if (tensor.byte_offset > UINTPTR_MAX || base > UINTPTR_MAX - tensor.byte_offset ||
        spanBytes > UINTPTR_MAX - base - tensor.byte_offset || compactBytes > SIZE_MAX)
    {
        return refuse(column, "the last element's address overflows");
    }
    return OVSPATIALQUERY_API_SUCCESS;
}

void gatherInputColumn(const uint8_t* first, const InputColumnLayout& layout, uint8_t* destination) noexcept
{
    const size_t elementBytes = static_cast<size_t>(layout.elementBytes);
    if (layout.width == 1u || layout.columnStride == 1u)
    {
        // A row's elements are adjacent: copy each row whole.
        const size_t rowBytes = static_cast<size_t>(layout.width) * elementBytes;
        for (uint64_t row = 0u; row < layout.rows; ++row)
        {
            std::memcpy(destination + row * rowBytes, first + row * layout.rowStride * elementBytes, rowBytes);
        }
        return;
    }
    for (uint64_t row = 0u; row < layout.rows; ++row)
    {
        const uint8_t* source = first + row * layout.rowStride * elementBytes;
        for (uint64_t element = 0u; element < layout.width; ++element)
        {
            std::memcpy(destination, source + element * layout.columnStride * elementBytes, elementBytes);
            destination += elementBytes;
        }
    }
}

void clearInputLayoutDiagnostic() noexcept { gInputLayoutDiagnostic.clear(); }

const char* inputLayoutDiagnostic(ovspatialquery_api_status_t status) noexcept
{
    return status == OVSPATIALQUERY_API_INVALID_ARGUMENT && !gInputLayoutDiagnostic.empty()
               ? gInputLayoutDiagnostic.c_str()
               : nullptr;
}

} // namespace detail
} // namespace ovspatialquery
