/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_EXPLICIT_CUDA_CAPTURE_H
#define OVSPATIALQUERY_EXPLICIT_CUDA_CAPTURE_H

#include "ovspatialquery_cuda_backend.h"
#include "ovspatialquery_device_box_population.h"
#include "ovspatialquery_explicit_preparation.h"
#include <ovspatialquery/ovspatialquery.h>
#include <ovspatialquery/ovspatialquery_explicit.h>

#include <memory>

namespace ovspatialquery
{
namespace detail
{

// What a CUDA-row publication needs beyond its rows: the instance that charges
// its budgets and the population's device-preparation state (ADR-0024).
struct ExplicitCudaCaptureContext
{
    ovspatialquery_instance_handle_t     instance{};
    std::shared_ptr<DevicePreparedState> deviceState;
};

// Captures CUDA instance rows for a publication: a complete set when
// `removed` is null, otherwise an update that also removes those IDs.
// `slotCapacity` is a set's declared capacity for device preparation; zero
// means the set's rows. NOT_SUPPORTED in a build without the CUDA backend.
ovspatialquery_api_status_t prepareExplicitCudaCapture(
    const ovspatialquery_explicit_instance_table_t& instances, const ovspatialquery_input_array_t* removed,
    const ovspatialquery_dynamic_execution_t& execution, ovspatialquery_ordinal_t baseOrdinal,
    ovspatialquery_ordinal_t targetOrdinal, bool posesOnly, bool rowBits, uint64_t slotCapacity,
    std::shared_ptr<ExplicitPreparationRegistry> registry, const ExplicitCudaCaptureContext& captureContext,
    std::shared_ptr<CudaPopulationCapture>& outCapture) noexcept;

} // namespace detail
} // namespace ovspatialquery

#endif
