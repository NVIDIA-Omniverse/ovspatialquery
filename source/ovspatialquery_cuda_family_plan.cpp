/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_cuda_family_plan.h"

#include <cmath>
#include <limits>

namespace ovspatialquery
{
namespace detail
{
namespace
{

bool positiveFiniteExtents(const double values[3]) noexcept
{
    return std::isfinite(values[0]) && values[0] > 0.0 && std::isfinite(values[1]) && values[1] > 0.0 &&
           std::isfinite(values[2]) && values[2] > 0.0;
}

} // namespace

void readCudaFamilyInputLayout(CudaFamilyInput& input, const char* column) noexcept
{
    input.layoutValid      = false;
    const DLTensor& tensor = input.tensor;
    if (!input.present || tensor.shape == nullptr || (tensor.ndim != 1 && tensor.ndim != 2) || tensor.shape[0] < 0 ||
        (tensor.ndim == 2 && tensor.shape[1] < 0) || tensor.dtype.bits == 0u || (tensor.dtype.bits % 8u) != 0u)
    {
        return;
    }
    input.layoutValid = readInputColumnLayout(tensor, column, input.layout) == OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t validateCudaFamilyPlanConfig(const CudaFamilyPlanConfig& config,
                                                         uint64_t&                   outDeviceByteCharge) noexcept
{
    outDeviceByteCharge            = 0u;
    constexpr uint32_t knownFields = kCudaFamilyResultPosition | kCudaFamilyResultNormal | kCudaFamilyResultDistance |
                                     kCudaFamilyResultSourceFeature | cuda_family::kResultGeometryIdentity |
                                     cuda_family::kResultQueryPosition | cuda_family::kResultMaterial;
    if (config.device.device_type != kDLCUDA || config.device.device_id < 0 || config.maximumQueryCount == 0u ||
        config.maximumQueryCount > static_cast<uint64_t>(INT64_MAX) || config.hitCapacityPerSlot == 0u ||
        config.hitCapacityPerSlot > static_cast<uint64_t>(INT64_MAX) || config.resultSlotCount == 0u ||
        !std::isfinite(config.contactTolerance) || config.contactTolerance < 0.0 ||
        (config.resultFields & ~knownFields) != 0u ||
        (config.precision != CudaFamilyPrecision::boundsCandidate && config.precision != CudaFamilyPrecision::exact &&
         config.precision != CudaFamilyPrecision::exactSingle) ||
        (config.precision == CudaFamilyPrecision::exactSingle && config.family != CudaFamilyKind::raycast) ||
        (config.precision == CudaFamilyPrecision::boundsCandidate &&
         (config.resultFields & kCudaFamilyResultSourceFeature) != 0u) ||
        (config.family != CudaFamilyKind::raycast && config.family != CudaFamilyKind::sphereSweep &&
         config.family != CudaFamilyKind::boxSweep && config.family != CudaFamilyKind::overlap) ||
        (config.hitMode != CudaFamilyHitMode::closest && config.hitMode != CudaFamilyHitMode::any &&
         config.hitMode != CudaFamilyHitMode::multiple) ||
        (config.family == CudaFamilyKind::overlap &&
         (config.hitMode == CudaFamilyHitMode::closest ||
          (config.resultFields & ~(kCudaFamilyResultGeometryIdentity | kCudaFamilyResultMaterial)) != 0u ||
          (config.overlapKind != CudaOverlapKind::sphere && config.overlapKind != CudaOverlapKind::box &&
           config.overlapKind != CudaOverlapKind::aabb))) ||
        ((config.resultFields & cuda_family::kResultQueryPosition) != 0u &&
         config.family != CudaFamilyKind::sphereSweep && config.family != CudaFamilyKind::boxSweep) ||
        (config.hitMode == CudaFamilyHitMode::multiple && config.maximumHitsPerQuery == 0u) ||
        (config.family == CudaFamilyKind::sphereSweep &&
         (!std::isfinite(config.planRadius) || config.planRadius <= 0.0)) ||
        (config.family == CudaFamilyKind::boxSweep && !positiveFiniteExtents(config.planHalfExtents)) ||
        (config.family == CudaFamilyKind::overlap &&
         ((config.overlapKind == CudaOverlapKind::sphere &&
           (!std::isfinite(config.planRadius) || config.planRadius <= 0.0)) ||
          (config.overlapKind != CudaOverlapKind::sphere && !positiveFiniteExtents(config.planHalfExtents)))))
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    uint64_t       perSlot  = sizeof(uint64_t);
    const uint64_t capacity = config.hitCapacityPerSlot;
    const uint64_t perQueryScratch =
        sizeof(cuda_family::RowPlan) +
        (config.hitMode != CudaFamilyHitMode::multiple ? uint64_t(cuda_family::kStashRowBytes) : 0u);
    if (!addHostCharge(config.maximumQueryCount, sizeof(uint32_t), perSlot) ||
        !addHostCharge(config.maximumQueryCount, 2u * sizeof(uint64_t) + sizeof(uint8_t), perSlot) ||
        !addHostCharge(config.maximumQueryCount, perQueryScratch, perSlot) ||
        !addHostCharge(capacity, sizeof(uint64_t), perSlot) ||
        (config.family != CudaFamilyKind::overlap && !addHostCharge(capacity, sizeof(uint32_t), perSlot)) ||
        ((config.resultFields & kCudaFamilyResultPosition) != 0u &&
         !addHostCharge(capacity, 3u * sizeof(double), perSlot)) ||
        ((config.resultFields & kCudaFamilyResultNormal) != 0u &&
         !addHostCharge(capacity, 3u * sizeof(float), perSlot)) ||
        (config.family != CudaFamilyKind::overlap && !addHostCharge(capacity, sizeof(double), perSlot)) ||
        ((config.resultFields & kCudaFamilyResultGeometryIdentity) != 0u &&
         (!addHostCharge(capacity, sizeof(int32_t), perSlot) || !addHostCharge(capacity, sizeof(uint64_t), perSlot))) ||
        ((config.resultFields & kCudaFamilyResultQueryPosition) != 0u &&
         !addHostCharge(capacity, 3u * sizeof(double), perSlot)) ||
        ((config.resultFields & kCudaFamilyResultMaterial) != 0u &&
         !addHostCharge(capacity, sizeof(uint64_t), perSlot)) ||
        ((config.resultFields & kCudaFamilyResultSourceFeature) != 0u &&
         (!addHostCharge(capacity, sizeof(uint32_t) + sizeof(uint64_t), perSlot) ||
          !addHostCharge(capacity, 6u * sizeof(uint32_t), perSlot))) ||
        !addHostCharge(config.resultSlotCount, perSlot, outDeviceByteCharge))
    {
        outDeviceByteCharge = 0u;
        return OVSPATIALQUERY_API_DEVICE_BUDGET;
    }
    return OVSPATIALQUERY_API_SUCCESS;
}

bool layoutCudaFamilyMappedColumns(uint64_t maximumQueryCount, uint64_t hitCapacityPerSlot, uint32_t resultFields,
                                   CudaFamilyMappedLayout& outLayout) noexcept
{
    using Column         = CudaFamilyMappedColumn;
    outLayout            = {};
    const auto requested = [resultFields](uint32_t field) { return (resultFields & field) != 0u; };
    const struct
    {
        Column   column;
        bool     present;
        uint32_t rowBytes;
        uint32_t rows;
    } columns[] = {
        {Column::queryStatus, true, sizeof(uint32_t), cuda_family::kMappedRowsQuery},
        {Column::storedHitCount, true, sizeof(uint64_t), cuda_family::kMappedRowsOne},
        {Column::hitOffsets, true, sizeof(uint64_t), cuda_family::kMappedRowsQuery},
        {Column::hitCounts, true, sizeof(uint64_t), cuda_family::kMappedRowsQuery},
        {Column::truncated, true, sizeof(uint8_t), cuda_family::kMappedRowsQuery},
        {Column::objectIds, true, sizeof(uint64_t), cuda_family::kMappedRowsHit},
        {Column::geometryTypes, requested(kCudaFamilyResultGeometryIdentity), sizeof(int32_t),
         cuda_family::kMappedRowsHit},
        {Column::geometryValues, requested(kCudaFamilyResultGeometryIdentity), sizeof(uint64_t),
         cuda_family::kMappedRowsHit},
        {Column::materialIds, requested(kCudaFamilyResultMaterial), sizeof(uint64_t), cuda_family::kMappedRowsHit},
        {Column::positions, requested(kCudaFamilyResultPosition), 3u * sizeof(double), cuda_family::kMappedRowsHit},
        {Column::queryPositions, requested(kCudaFamilyResultQueryPosition), 3u * sizeof(double),
         cuda_family::kMappedRowsHit},
        {Column::normals, requested(kCudaFamilyResultNormal), 3u * sizeof(float), cuda_family::kMappedRowsHit},
        {Column::distances, requested(kCudaFamilyResultDistance), sizeof(double), cuda_family::kMappedRowsHit},
        {Column::featureKinds, requested(kCudaFamilyResultSourceFeature), sizeof(uint32_t),
         cuda_family::kMappedRowsHit},
        {Column::faceIndices, requested(kCudaFamilyResultSourceFeature), sizeof(uint64_t), cuda_family::kMappedRowsHit},
        {Column::corners, requested(kCudaFamilyResultSourceFeature), 3u * sizeof(uint32_t),
         cuda_family::kMappedRowsHit},
        {Column::barycentrics, requested(kCudaFamilyResultSourceFeature), 3u * sizeof(float),
         cuda_family::kMappedRowsHit},
    };
    constexpr uint64_t alignment = cuda_family::kMappedColumnAlignment;
    uint64_t           offset    = 0u;
    for (const auto& entry : columns)
    {
        if (!entry.present)
        {
            continue;
        }
        const uint64_t rows  = entry.rows == cuda_family::kMappedRowsQuery ? maximumQueryCount
                               : entry.rows == cuda_family::kMappedRowsHit ? hitCapacityPerSlot
                                                                           : 1u;
        uint64_t       bytes = 0u;
        if (!addHostCharge(rows, entry.rowBytes, bytes) || bytes > UINT64_MAX - offset ||
            offset + bytes > UINT64_MAX - (alignment - 1u))
        {
            outLayout = {};
            return false;
        }
        const size_t index        = static_cast<size_t>(entry.column);
        outLayout.offsets[index]  = offset;
        outLayout.rowBytes[index] = entry.rowBytes;
        outLayout.rows[index]     = entry.rows;
        offset                    = (offset + bytes + alignment - 1u) / alignment * alignment;
    }
    outLayout.bytes = offset;
    return true;
}

bool validateCudaFamilyLaunch(const CudaFamilyPlanConfig& config, const CudaFamilyLaunch& launch) noexcept
{
    const auto valid = [&config, queryCount = launch.queryCount](const CudaFamilyInput& input, uint8_t code,
                                                                 uint8_t bits, int32_t rank, int64_t trailing,
                                                                 bool optional)
    {
        if (!input.present)
        {
            return optional;
        }
        const DLTensor& tensor = input.tensor;
        // The kernels load each element at its natural width, so the first element
        // must sit on a multiple of the element size.
        const uint64_t firstElement =
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tensor.data)) + static_cast<uint64_t>(tensor.byte_offset);
        return (input.ownership == 0u || input.ownership == 1u) && tensor.device.device_type == kDLCUDA &&
               tensor.device.device_id == config.device.device_id && tensor.dtype.code == code &&
               tensor.dtype.bits == bits && tensor.dtype.lanes == 1u && tensor.ndim == rank &&
               tensor.shape != nullptr && input.layoutValid && tensor.shape[0] == static_cast<int64_t>(queryCount) &&
               (rank != 2 || tensor.shape[1] == trailing) && ((queryCount == 0u) == (tensor.data == nullptr)) &&
               (queryCount != 0u || tensor.byte_offset == 0u) && firstElement % (bits / 8u) == 0u;
    };
    const auto             absent    = [](const CudaFamilyInput& input) { return !input.present; };
    const CudaFamilyInput& position  = launch.positions;
    const CudaFamilyInput& direction = launch.directionsOrRotations;
    if (launch.queryCount > config.maximumQueryCount || !valid(launch.queryBits, kDLUInt, 64u, 2, 2, true) ||
        (config.hitMode == CudaFamilyHitMode::multiple ? !valid(launch.rowMaximumHits, kDLUInt, 64u, 1, 1, true)
                                                       : !absent(launch.rowMaximumHits)))
    {
        return false;
    }
    if (config.family == CudaFamilyKind::overlap)
    {
        if (config.overlapKind == CudaOverlapKind::sphere)
        {
            // SPHERE rows may carry their own radii.
            return valid(position, kDLFloat, 64u, 2, 3, false) && absent(direction) && absent(launch.halfExtents) &&
                   valid(launch.radii, kDLFloat, 64u, 1, 1, true);
        }
        if (!absent(launch.radii))
        {
            return false;
        }
        if (config.overlapKind == CudaOverlapKind::box)
        {
            // Rotations in float32 or float64 (ADR-0026); rows may carry their own half extents.
            return valid(position, kDLFloat, 64u, 2, 3, false) &&
                   (valid(direction, kDLFloat, 32u, 2, 4, false) || valid(direction, kDLFloat, 64u, 2, 4, false)) &&
                   valid(launch.halfExtents, kDLFloat, 64u, 2, 3, true);
        }
        // AABB rows may carry their own half extents.
        return valid(position, kDLFloat, 64u, 2, 3, false) && absent(direction) &&
               valid(launch.halfExtents, kDLFloat, 64u, 2, 3, true);
    }
    // Box sweep rows carry rotations and may carry their own half extents;
    // sphere sweep rows may carry their own radii.
    return (config.family != CudaFamilyKind::boxSweep || (valid(launch.halfExtents, kDLFloat, 64u, 2, 3, true) &&
                                                          valid(launch.rotations, kDLFloat, 32u, 2, 4, false))) &&
           (config.family != CudaFamilyKind::sphereSweep || valid(launch.radii, kDLFloat, 64u, 1, 1, true)) &&
           valid(position, kDLFloat, 64u, 2, 3, false) && valid(direction, kDLFloat, 32u, 2, 3, false) &&
           valid(launch.minimumDistances, kDLFloat, 64u, 1, 1, true) &&
           valid(launch.maximumDistances, kDLFloat, 64u, 1, 1, false);
}

} // namespace detail
} // namespace ovspatialquery
