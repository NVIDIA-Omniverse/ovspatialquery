/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_INPUT_LAYOUT_H
#define OVSPATIALQUERY_INPUT_LAYOUT_H

#include <ovspatialquery/ovspatialquery.h>

#include <cstdint>

namespace ovspatialquery
{
namespace detail
{

// Where the elements of a rank-1 or rank-2 input column lie (ADR-0053). Strides
// are element counts; a rank-1 column has one element per row.
struct InputColumnLayout
{
    uint64_t rows         = 0u;
    uint64_t width        = 1u;
    uint64_t rowStride    = 1u;
    uint64_t columnStride = 1u;
    uint64_t elementBytes = 0u;
    bool     compact      = true;

    uint64_t compactBytes() const noexcept { return rows * width * elementBytes; }
};

// Reads a column's layout once its rank (1 or 2) and nonnegative shape are known.
// NULL strides mean compact row-major, and a dimension of extent one or less
// ignores its stride (ADR-0042). A negative stride, an element count or byte
// count that overflows, or a last element whose address overflows returns
// INVALID_ARGUMENT and names `column` in the calling thread's layout diagnostic,
// unless an earlier refusal already holds it.
// The tensor must be one whose shape and dtype the caller has already checked.
ovspatialquery_api_status_t readInputColumnLayout(const DLTensor& tensor, const char* column,
                                                  InputColumnLayout& out) noexcept;

// Copies a column's elements in row-major order into compact storage of
// `layout.compactBytes()` bytes.
void gatherInputColumn(const uint8_t* first, const InputColumnLayout& layout, uint8_t* destination) noexcept;

// Forgets the calling thread's layout diagnostic.
void clearInputLayoutDiagnostic() noexcept;

// The calling thread's layout diagnostic for an INVALID_ARGUMENT refusal, or
// NULL when the refusal had another cause. Valid until the thread's next
// layout read.
const char* inputLayoutDiagnostic(ovspatialquery_api_status_t status) noexcept;

} // namespace detail
} // namespace ovspatialquery

#endif
