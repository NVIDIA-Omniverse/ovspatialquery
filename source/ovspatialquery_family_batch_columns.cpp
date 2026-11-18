/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_family_batch_columns.h"

#include "ovspatialquery_instance_host_ledger.h"

#include <algorithm>
#include <array>
#include <type_traits>
#include <utility>
#include <vector>

namespace ovspatialquery
{
namespace detail
{
namespace
{

uint32_t rowStatus(RowOutcome value) noexcept
{
    switch (value)
    {
    case RowOutcome::success: return static_cast<uint32_t>(OVSPATIALQUERY_BATCH_ROW_SUCCESS);
    case RowOutcome::invalidArgument: return static_cast<uint32_t>(OVSPATIALQUERY_BATCH_ROW_INVALID_ARGUMENT);
    case RowOutcome::notSupported: return static_cast<uint32_t>(OVSPATIALQUERY_BATCH_ROW_NOT_SUPPORTED);
    case RowOutcome::failed: return static_cast<uint32_t>(OVSPATIALQUERY_BATCH_ROW_FAILED);
    }
    return static_cast<uint32_t>(OVSPATIALQUERY_BATCH_ROW_FAILED);
}

} // namespace

template <typename Policy>
struct FamilyBatchColumnStorage<Policy>::Impl
{
    using Config      = FamilyBatchColumnConfig;
    using Result      = typename Policy::Result;
    using Hit         = typename Policy::Hit;
    using Reservation = typename Policy::Reservation;
    using Fetched     = typename Policy::Fetched;

    struct TensorMetadata
    {
        std::array<int64_t, 2u> shape{};
        std::array<int64_t, 2u> strides{};

        DLTensor oneDimensional(void* data, uint8_t code, uint8_t bits, int64_t count) noexcept
        {
            shape   = {count, 0};
            strides = {1, 0};
            return {data, {kDLCPU, 0}, 1, {code, bits, 1u}, shape.data(), strides.data(), 0u};
        }

        DLTensor triples(void* data, uint8_t code, uint8_t bits, int64_t count) noexcept
        {
            shape   = {count, 3};
            strides = {3, 1};
            return {data, {kDLCPU, 0}, 2, {code, bits, 1u}, shape.data(), strides.data(), 0u};
        }
    };

    struct SnapshotMetadata
    {
        TensorMetadata queryStatus;
        TensorMetadata storedHitCount;
        TensorMetadata hitOffsets;
        TensorMetadata hitCounts;
        TensorMetadata truncated;
        TensorMetadata objectIds;
        TensorMetadata geometryTypes;
        TensorMetadata geometryValues;
        TensorMetadata materialIds;
        TensorMetadata positions;
        TensorMetadata normals;
        TensorMetadata queryPositions;
        TensorMetadata distances;
        TensorMetadata featureKinds;
        TensorMetadata sourceFaces;
        TensorMetadata sourceCorners;
        TensorMetadata barycentrics;
    };

    explicit Impl(Config value) : config(value) {}

    bool requested(uint32_t field) const noexcept { return (config.resultFields & field) != 0u; }

    void allocate()
    {
        const size_t rows = static_cast<size_t>(config.maximumQueryCount);
        const size_t hits = static_cast<size_t>(config.hitCapacity);
        queryStatus.resize(rows);
        hitOffsets.resize(rows);
        hitCounts.resize(rows);
        truncated.resize(rows);
        objectIds.resize(hits);
        if (requested(kBatchResultGeometryIdentity))
        {
            geometryTypes.resize(hits);
            geometryValues.resize(hits);
        }
        if (requested(kBatchResultMaterial))
        {
            materialIds.resize(hits);
        }
        if (requested(kBatchResultPosition))
        {
            positions.resize(3u * hits);
        }
        if (requested(kBatchResultNormal))
        {
            normals.resize(3u * hits);
        }
        if (requested(kBatchResultQueryPosition))
        {
            queryPositions.resize(3u * hits);
        }
        if (requested(kBatchResultDistance))
        {
            distances.resize(hits);
        }
        if (requested(kBatchResultSourceFeature))
        {
            featureKinds.resize(hits);
            sourceFaces.resize(hits);
            sourceCorners.resize(3u * hits);
            barycentrics.resize(3u * hits);
        }
    }

    ovspatialquery_batch_result_common_t common(uint32_t slotIndex, uint64_t rowCount,
                                                ovspatialquery_ordinal_t ordinal) const noexcept
    {
        ovspatialquery_batch_result_common_t value{};
        value.result_slot      = slotIndex;
        value.query_count      = rowCount;
        value.hit_capacity     = config.hitCapacity;
        value.truth            = config.truth == HitTruth::exactSurface ? OVSPATIALQUERY_HIT_EXACT_SURFACE
                                                                        : OVSPATIALQUERY_HIT_BOUNDS_CANDIDATE;
        value.source_ordinal   = ordinal;
        value.execution_device = {kDLCPU, 0};
        return value;
    }

    void commonColumns(typename Policy::Columns& value, SnapshotMetadata& metadata, int64_t rows, int64_t hits) noexcept
    {
        value.hit_ranges.query_status = metadata.queryStatus.oneDimensional(queryStatus.data(), kDLUInt, 32u, rows);
        value.hit_ranges.stored_hit_count =
            metadata.storedHitCount.oneDimensional(storedHitCount.data(), kDLUInt, 64u, 1);
        value.hit_ranges.hit_offsets = metadata.hitOffsets.oneDimensional(hitOffsets.data(), kDLUInt, 64u, rows);
        value.hit_ranges.hit_counts  = metadata.hitCounts.oneDimensional(hitCounts.data(), kDLUInt, 64u, rows);
        value.hit_ranges.truncated   = metadata.truncated.oneDimensional(truncated.data(), kDLUInt, 8u, rows);
        value.object_ids             = metadata.objectIds.oneDimensional(objectIds.data(), kDLUInt, 64u, hits);
        if (requested(kBatchResultGeometryIdentity))
        {
            value.geometry.types  = metadata.geometryTypes.oneDimensional(geometryTypes.data(), kDLInt, 32u, hits);
            value.geometry.values = metadata.geometryValues.oneDimensional(geometryValues.data(), kDLUInt, 64u, hits);
        }
        if (requested(kBatchResultMaterial))
        {
            value.material_ids = metadata.materialIds.oneDimensional(materialIds.data(), kDLUInt, 64u, hits);
        }
    }

    void featureColumns(ovspatialquery_exact_feature_columns_t& value, SnapshotMetadata& metadata,
                        int64_t hits) noexcept
    {
        if (!requested(kBatchResultSourceFeature))
        {
            return;
        }
        value.kinds                 = metadata.featureKinds.oneDimensional(featureKinds.data(), kDLUInt, 32u, hits);
        value.source_face_indices   = metadata.sourceFaces.oneDimensional(sourceFaces.data(), kDLUInt, 64u, hits);
        value.source_corner_indices = metadata.sourceCorners.triples(sourceCorners.data(), kDLUInt, 32u, hits);
        value.barycentrics          = metadata.barycentrics.triples(barycentrics.data(), kDLFloat, 32u, hits);
    }

    typename Policy::Columns columns(SnapshotMetadata& metadata, uint64_t rowCount, uint64_t hitCount) noexcept
    {
        const int64_t            rows = static_cast<int64_t>(rowCount);
        const int64_t            hits = static_cast<int64_t>(hitCount);
        typename Policy::Columns value{};
        commonColumns(value, metadata, rows, hits);
        if constexpr (!std::is_same_v<Policy, OverlapBatchColumnPolicy>)
        {
            const bool exact           = config.truth == HitTruth::exactSurface;
            auto&      contact         = value.contact;
            DLTensor*  positionsTensor = exact ? &contact.exact.positions : &contact.candidate.bounds_positions;
            DLTensor*  normalsTensor   = exact ? &contact.exact.normals : &contact.candidate.bounds_normals;
            DLTensor*  distancesTensor = exact ? &contact.exact.distances : &contact.candidate.distances;
            if (requested(kBatchResultPosition))
            {
                *positionsTensor = metadata.positions.triples(positions.data(), kDLFloat, 64u, hits);
            }
            if (requested(kBatchResultNormal))
            {
                *normalsTensor = metadata.normals.triples(normals.data(), kDLFloat, 32u, hits);
            }
            if constexpr (std::is_same_v<Policy, SphereSweepBatchColumnPolicy>)
            {
                if (requested(kBatchResultQueryPosition))
                {
                    (exact ? contact.exact.query_positions : contact.candidate.query_positions) =
                        metadata.queryPositions.triples(queryPositions.data(), kDLFloat, 64u, hits);
                }
            }
            if (requested(kBatchResultDistance))
            {
                *distancesTensor = metadata.distances.oneDimensional(distances.data(), kDLFloat, 64u, hits);
            }
            if (exact)
            {
                featureColumns(contact.exact.features, metadata, hits);
            }
        }
        return value;
    }

    bool validSemantic(const Result& semantic) const noexcept
    {
        const size_t rows = semantic.common.rowOutcomes.size();
        if (semantic.common.storedHitCount != semantic.hits.size() ||
            semantic.common.storedHitCount > config.hitCapacity || rows != queryCount ||
            semantic.common.hitOffsets.size() != rows || semantic.common.hitCounts.size() != rows ||
            semantic.common.truncated.size() != rows)
        {
            return false;
        }
        for (size_t row = 0u; row < rows; ++row)
        {
            const uint64_t offset = semantic.common.hitOffsets[row];
            const uint64_t count  = semantic.common.hitCounts[row];
            if (offset > semantic.common.storedHitCount || count > semantic.common.storedHitCount - offset ||
                (semantic.common.rowOutcomes[row] != RowOutcome::success &&
                 (count != 0u || semantic.common.truncated[row] != 0u)))
            {
                return false;
            }
        }
        return true;
    }

    const SemanticHitIdentity& identity(const Hit& hit) const noexcept
    {
        if constexpr (std::is_same_v<Policy, OverlapBatchColumnPolicy>)
        {
            return hit.identity;
        }
        else
        {
            return config.truth == HitTruth::exactSurface ? hit.exact.identity : hit.boundsCandidate.identity;
        }
    }

    void copyFeature(size_t index, const GeometryFeature& feature) noexcept
    {
        if (!requested(kBatchResultSourceFeature))
        {
            return;
        }
        featureKinds[index] = feature.kind == GeometryFeatureKind::triangle ? OVSPATIALQUERY_GEOMETRY_FEATURE_TRIANGLE
                                                                            : OVSPATIALQUERY_GEOMETRY_FEATURE_NONE;
        if (feature.kind == GeometryFeatureKind::triangle)
        {
            sourceFaces[index] = feature.triangle.sourceFace;
            for (size_t corner = 0u; corner < 3u; ++corner)
            {
                sourceCorners[index * 3u + corner] = feature.triangle.sourceCorners[corner];
                barycentrics[index * 3u + corner]  = feature.triangle.barycentrics[corner];
            }
        }
    }

    void copyHit(size_t index, const Hit& hit) noexcept
    {
        const SemanticHitIdentity& hitIdentity = identity(hit);
        objectIds[index]                       = hitIdentity.objectId.value;
        if (requested(kBatchResultGeometryIdentity))
        {
            geometryTypes[index]  = static_cast<int32_t>(hitIdentity.geometry.family) + kGeometryTypeFirstBuiltin;
            geometryValues[index] = hitIdentity.geometry.value;
        }
        if (requested(kBatchResultMaterial))
        {
            materialIds[index] = hitIdentity.materialId;
        }
        if constexpr (!std::is_same_v<Policy, OverlapBatchColumnPolicy>)
        {
            const bool exact = config.truth == HitTruth::exactSurface;
            if constexpr (std::is_same_v<Policy, RaycastBatchColumnPolicy>)
            {
                copyContact(index, exact ? hit.exact.position : hit.boundsCandidate.boundsPosition,
                            exact ? hit.exact.normal : hit.boundsCandidate.boundsNormal,
                            exact ? hit.exact.distance : hit.boundsCandidate.distance);
            }
            else
            {
                copyContact(index, exact ? hit.exact.objectPosition : hit.boundsCandidate.boundsPosition,
                            exact ? hit.exact.normal : hit.boundsCandidate.boundsNormal,
                            exact ? hit.exact.distance : hit.boundsCandidate.distance);
                if (requested(kBatchResultQueryPosition))
                {
                    const std::array<double, 3u>& queryPosition =
                        exact ? hit.exact.queryPosition : hit.boundsCandidate.queryPosition;
                    std::copy(queryPosition.begin(), queryPosition.end(), queryPositions.begin() + index * 3u);
                }
            }
            if (exact)
            {
                copyFeature(index, hit.exact.feature);
            }
        }
    }

    void copyContact(size_t index, const std::array<double, 3u>& position, const std::array<float, 3u>& normal,
                     double distance) noexcept
    {
        for (size_t axis = 0u; axis < 3u; ++axis)
        {
            if (requested(kBatchResultPosition))
            {
                positions[index * 3u + axis] = position[axis];
            }
            if (requested(kBatchResultNormal))
            {
                normals[index * 3u + axis] = normal[axis];
            }
        }
        if (requested(kBatchResultDistance))
        {
            distances[index] = distance;
        }
    }

    Config                   config;
    uint32_t                 resultSlot    = UINT32_MAX;
    uint64_t                 queryCount    = 0u;
    ovspatialquery_ordinal_t sourceOrdinal = 0;
    std::vector<uint32_t>    queryStatus;
    std::array<uint64_t, 1u> storedHitCount{};
    std::vector<uint64_t>    hitOffsets;
    std::vector<uint64_t>    hitCounts;
    std::vector<uint8_t>     truncated;
    std::vector<uint64_t>    objectIds;
    std::vector<int32_t>     geometryTypes;
    std::vector<uint64_t>    geometryValues;
    std::vector<uint64_t>    materialIds;
    std::vector<double>      positions;
    std::vector<float>       normals;
    std::vector<double>      queryPositions;
    std::vector<double>      distances;
    std::vector<uint32_t>    featureKinds;
    std::vector<uint64_t>    sourceFaces;
    std::vector<uint32_t>    sourceCorners;
    std::vector<float>       barycentrics;
    SnapshotMetadata         reservationMetadata;
    SnapshotMetadata         resultMetadata;
    Fetched                  resultSnapshot{};
};

template <typename Policy>
FamilyBatchColumnStorage<Policy>::FamilyBatchColumnStorage(std::unique_ptr<Impl> impl) noexcept : mImpl(std::move(impl))
{}

template <typename Policy>
FamilyBatchColumnStorage<Policy>::~FamilyBatchColumnStorage() = default;

template <typename Policy>
bool FamilyBatchColumnStorage<Policy>::estimateResidentBytes(Config config, uint64_t& outBytes) noexcept
{
    uint64_t       total   = sizeof(FamilyBatchColumnStorage) + sizeof(Impl);
    const uint64_t rowSize = sizeof(uint32_t) + 2u * sizeof(uint64_t) + sizeof(uint8_t);
    uint64_t       hitSize = sizeof(uint64_t);
    if ((config.resultFields & kBatchResultGeometryIdentity) != 0u)
    {
        hitSize += sizeof(int32_t) + sizeof(uint64_t);
    }
    if ((config.resultFields & kBatchResultMaterial) != 0u)
    {
        hitSize += sizeof(uint64_t);
    }
    if ((config.resultFields & kBatchResultPosition) != 0u)
    {
        hitSize += 3u * sizeof(double);
    }
    if ((config.resultFields & kBatchResultNormal) != 0u)
    {
        hitSize += 3u * sizeof(float);
    }
    if ((config.resultFields & kBatchResultQueryPosition) != 0u)
    {
        hitSize += 3u * sizeof(double);
    }
    if ((config.resultFields & kBatchResultDistance) != 0u)
    {
        hitSize += sizeof(double);
    }
    if ((config.resultFields & kBatchResultSourceFeature) != 0u)
    {
        hitSize += sizeof(uint32_t) + sizeof(uint64_t) + 3u * sizeof(uint32_t) + 3u * sizeof(float);
    }
    return addHostCharge(config.maximumQueryCount, rowSize, total) &&
           addHostCharge(config.hitCapacity, hitSize, total) && (outBytes = total, true);
}

template <typename Policy>
std::unique_ptr<FamilyBatchColumnStorage<Policy>> FamilyBatchColumnStorage<Policy>::create(Config config)
{
    std::unique_ptr<Impl> impl(new Impl(config));
    impl->allocate();
    return std::unique_ptr<FamilyBatchColumnStorage>(new FamilyBatchColumnStorage(std::move(impl)));
}

template <typename Policy>
void FamilyBatchColumnStorage<Policy>::begin(uint32_t resultSlot, uint64_t queryCount,
                                             ovspatialquery_ordinal_t sourceOrdinal,
                                             Reservation&             outReservation) noexcept
{
    mImpl->resultSlot        = resultSlot;
    mImpl->queryCount        = queryCount;
    mImpl->sourceOrdinal     = sourceOrdinal;
    mImpl->storedHitCount[0] = 0u;
    outReservation           = {mImpl->common(resultSlot, queryCount, sourceOrdinal),
                                mImpl->columns(mImpl->reservationMetadata, queryCount, mImpl->config.hitCapacity)};
}

template <typename Policy>
FamilyBatchColumnsOutcome FamilyBatchColumnStorage<Policy>::publish(const Result& semantic) noexcept
{
    if (!mImpl->validSemantic(semantic))
    {
        return FamilyBatchColumnsOutcome::invalidArgument;
    }
    for (size_t row = 0u; row < mImpl->queryCount; ++row)
    {
        mImpl->queryStatus[row] = rowStatus(semantic.common.rowOutcomes[row]);
        mImpl->hitOffsets[row] =
            semantic.common.rowOutcomes[row] == RowOutcome::success ? semantic.common.hitOffsets[row] : 0u;
        mImpl->hitCounts[row] = semantic.common.hitCounts[row];
        mImpl->truncated[row] = semantic.common.truncated[row];
    }
    for (size_t index = 0u; index < semantic.hits.size(); ++index)
    {
        mImpl->copyHit(index, semantic.hits[index]);
    }
    mImpl->storedHitCount[0] = semantic.common.storedHitCount;
    mImpl->resultSnapshot    = {mImpl->common(mImpl->resultSlot, mImpl->queryCount, mImpl->sourceOrdinal),
                                mImpl->columns(mImpl->resultMetadata, mImpl->queryCount, semantic.common.storedHitCount)};
    return FamilyBatchColumnsOutcome::success;
}

template <typename Policy>
const typename FamilyBatchColumnStorage<Policy>::Fetched& FamilyBatchColumnStorage<Policy>::result() const noexcept
{
    return mImpl->resultSnapshot;
}

template class FamilyBatchColumnStorage<RaycastBatchColumnPolicy>;
template class FamilyBatchColumnStorage<SphereSweepBatchColumnPolicy>;
template class FamilyBatchColumnStorage<OverlapBatchColumnPolicy>;

} // namespace detail
} // namespace ovspatialquery
