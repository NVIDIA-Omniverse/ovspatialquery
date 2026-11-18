/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_input.h"
#include "ovspatialquery_typed_geometry.h"

#include "ovspatialquery_backend.h"
#include "ovspatialquery_cuda_backend.h"
#include <ovspatialquery_prepared_array.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace ovspatialquery
{
namespace detail
{
namespace
{

using PreparedArray = PreparedInputArray;

void addRetainedBytes(uint64_t bytes, uint64_t& total)
{
    total = bytes > std::numeric_limits<uint64_t>::max() - total ? std::numeric_limits<uint64_t>::max() : total + bytes;
}

template <typename T>
void addVectorBytes(const std::vector<T>& values, uint64_t& total)
{
    if (values.capacity() > std::numeric_limits<uint64_t>::max() / sizeof(T))
    {
        total = std::numeric_limits<uint64_t>::max();
        return;
    }
    addRetainedBytes(static_cast<uint64_t>(values.capacity()) * sizeof(T), total);
}

} // namespace

struct PreparedPopulationInput::Impl
{
    ovspatialquery_ordinal_t baseOrdinal                = 0u;
    ovspatialquery_ordinal_t targetOrdinal              = 0u;
    bool                     completeSet                = false;
    bool                     requireUnreadyPopulation   = false;
    bool                     chainPendingPublications   = false;
    bool                     enforceObjectValueLifetime = false;
    bool                     preMaterialized            = false;
    ObjectTable              copiedObjects;
    // Carry the admission measurement with the unchanged CPU object vector; a
    // CUDA capture reports its own, if any, when it materializes the rows.
    std::optional<uint64_t>                copiedGeometryCharge;
    std::optional<VersionChangeRecord>     changes;
    std::vector<PreparedInputArray>        inputLeases;
    std::shared_ptr<CudaPopulationCapture> cudaCapture;
};

PreparedPopulationInput::PreparedPopulationInput()                                                    = default;
PreparedPopulationInput::~PreparedPopulationInput()                                                   = default;
PreparedPopulationInput::PreparedPopulationInput(PreparedPopulationInput&& other) noexcept            = default;
PreparedPopulationInput& PreparedPopulationInput::operator=(PreparedPopulationInput&& other) noexcept = default;

PreparedPopulationInput::PreparedPopulationInput(std::unique_ptr<Impl> impl) : mImpl(std::move(impl)) {}

ovspatialquery_api_status_t PreparedPopulationInput::prepareCopiedSet(ovspatialquery_ordinal_t targetOrdinal,
                                                                      ObjectTable              objects,
                                                                      PreparedPopulationInput& outInput)
{
    if (targetOrdinal < 0)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    try
    {
        std::unique_ptr<Impl> candidate(new Impl());
        candidate->completeSet     = true;
        candidate->preMaterialized = true;
        candidate->targetOrdinal   = targetOrdinal;
        candidate->copiedObjects   = std::move(objects);
        outInput                   = PreparedPopulationInput(std::move(candidate));
        return OVSPATIALQUERY_API_SUCCESS;
    }
    catch (const std::bad_alloc&)
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    catch (...)
    {
        return OVSPATIALQUERY_API_INTERNAL;
    }
}

ovspatialquery_api_status_t PreparedPopulationInput::prepareCopiedSet(ovspatialquery_ordinal_t      targetOrdinal,
                                                                      std::vector<PopulationObject> objects,
                                                                      PreparedPopulationInput&      outInput)
{
    try
    {
        return prepareCopiedSet(targetOrdinal, ObjectTable(std::move(objects)), outInput);
    }
    catch (const std::bad_alloc&)
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    catch (...)
    {
        return OVSPATIALQUERY_API_INTERNAL;
    }
}

ovspatialquery_api_status_t PreparedPopulationInput::prepareCopiedCompleteUpdate(
    ovspatialquery_ordinal_t baseOrdinal, ovspatialquery_ordinal_t targetOrdinal, ObjectTable objects,
    PreparedPopulationInput& outInput, std::optional<uint64_t> exactGeometryCharge,
    std::optional<VersionChangeRecord> changes)
{
    if (baseOrdinal < 0 || targetOrdinal <= baseOrdinal)
    {
        return targetOrdinal <= baseOrdinal ? OVSPATIALQUERY_API_STALE_ORDINAL : OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    try
    {
        std::unique_ptr<Impl> candidate(new Impl());
        candidate->baseOrdinal          = baseOrdinal;
        candidate->targetOrdinal        = targetOrdinal;
        candidate->preMaterialized      = true;
        candidate->copiedObjects        = std::move(objects);
        candidate->copiedGeometryCharge = exactGeometryCharge;
        candidate->changes              = std::move(changes);
        outInput                        = PreparedPopulationInput(std::move(candidate));
        return OVSPATIALQUERY_API_SUCCESS;
    }
    catch (const std::bad_alloc&)
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    catch (...)
    {
        return OVSPATIALQUERY_API_INTERNAL;
    }
}

bool PreparedPopulationInput::isSet() const { return mImpl != nullptr && mImpl->completeSet; }

void PreparedPopulationInput::restrictToUnreadyPopulation()
{
    if (mImpl != nullptr)
    {
        mImpl->requireUnreadyPopulation = true;
    }
}

bool PreparedPopulationInput::requiresUnreadyPopulation() const
{
    return mImpl != nullptr && mImpl->requireUnreadyPopulation;
}

void PreparedPopulationInput::chainPendingPublications()
{
    if (mImpl != nullptr)
    {
        mImpl->chainPendingPublications = true;
    }
}

bool PreparedPopulationInput::chainsPendingPublications() const
{
    return mImpl != nullptr && mImpl->chainPendingPublications;
}

void PreparedPopulationInput::enforceObjectValueNoReuse()
{
    if (mImpl != nullptr)
    {
        mImpl->enforceObjectValueLifetime = true;
    }
}

bool PreparedPopulationInput::enforcesObjectValueNoReuse() const
{
    return mImpl != nullptr && mImpl->enforceObjectValueLifetime;
}

ovspatialquery_ordinal_t PreparedPopulationInput::baseOrdinal() const
{
    return mImpl == nullptr ? 0u : mImpl->baseOrdinal;
}

ovspatialquery_ordinal_t PreparedPopulationInput::targetOrdinal() const
{
    return mImpl == nullptr ? 0u : mImpl->targetOrdinal;
}

void PreparedPopulationInput::setTargetOrdinal(ovspatialquery_ordinal_t ordinal)
{
    if (mImpl != nullptr)
    {
        mImpl->targetOrdinal = ordinal;
    }
}

void PreparedPopulationInput::adoptInputLease(PreparedInputArray lease)
{
    if (mImpl != nullptr)
    {
        mImpl->inputLeases.push_back(std::move(lease));
    }
}

void PreparedPopulationInput::adoptCudaCapture(std::shared_ptr<CudaPopulationCapture> capture)
{
    if (mImpl != nullptr)
    {
        mImpl->cudaCapture = std::move(capture);
    }
}

std::shared_ptr<CudaPopulationCapture> PreparedPopulationInput::cudaCapture() const
{
    return mImpl == nullptr ? std::shared_ptr<CudaPopulationCapture>() : mImpl->cudaCapture;
}

uint64_t PreparedPopulationInput::retainedByteCount()
{
    if (mImpl == nullptr)
    {
        return 0u;
    }
    uint64_t total = static_cast<uint64_t>(sizeof(Impl));
    for (const PreparedInputArray& lease : mImpl->inputLeases)
    {
        addRetainedBytes(lease.byteCount(), total);
    }
    // A derived table owns only its new chunks; the base's stay charged to it.
    addRetainedBytes(mImpl->copiedObjects.ownBatchBytes(), total);
    addRetainedBytes(mImpl->copiedObjects.chunkEntryBytes(), total);
    const uint64_t exactGeometryBytes = mImpl->copiedGeometryCharge
                                            ? *mImpl->copiedGeometryCharge
                                            : exactGeometryResidentByteCount(mImpl->copiedObjects);
    mImpl->copiedGeometryCharge       = exactGeometryBytes;
    if (exactGeometryBytes == std::numeric_limits<uint64_t>::max())
    {
        return exactGeometryBytes;
    }
    addRetainedBytes(exactGeometryBytes, total);
    addVectorBytes(mImpl->inputLeases, total);
    if (mImpl->changes)
    {
        addRetainedBytes(mImpl->changes->residentByteCount(), total);
    }
    if (mImpl->cudaCapture != nullptr)
    {
        addRetainedBytes(mImpl->cudaCapture->retainedOperationByteCharge(), total);
    }
    return total;
}

uint64_t PreparedPopulationInput::materializationPeakByteCount(uint64_t baseObjectCount,
                                                               uint64_t retainedByteCount) const
{
    if (mImpl == nullptr)
    {
        return 0u;
    }
    // A host input is prepared at submission, and materialize moves its
    // objects into the version, as it does a prepared derived table: neither
    // needs anything beyond what it retains.
    const bool derived =
        mImpl->preMaterialized && mImpl->copiedObjects.contiguousData() == nullptr && !mImpl->copiedObjects.empty();
    if (mImpl->cudaCapture == nullptr || derived)
    {
        return retainedByteCount;
    }
    const uint64_t changedCount   = mImpl->preMaterialized ? static_cast<uint64_t>(mImpl->copiedObjects.size()) : 0u;
    const uint64_t candidateCount = mImpl->completeSet
                                        ? changedCount
                                        : (changedCount > std::numeric_limits<uint64_t>::max() - baseObjectCount
                                               ? std::numeric_limits<uint64_t>::max()
                                               : baseObjectCount + changedCount);
    if (candidateCount == std::numeric_limits<uint64_t>::max())
    {
        return candidateCount;
    }

    // Materialization needs per-object geometry binding/attachment scratch
    // alongside the final contiguous vector; the node allowance keeps the
    // conservative envelope that also covers triangle/source normalization
    // vectors coexisting with the retained arrays.
    constexpr uint64_t mapNodeAllowance = 4u * sizeof(void*) + 32u;
    constexpr uint64_t perCandidate     = 2u * sizeof(PopulationObject) + mapNodeAllowance;
    uint64_t           total            = retainedByteCount;
    if (perCandidate != 0u && candidateCount > (std::numeric_limits<uint64_t>::max() - total) / perCandidate)
    {
        return std::numeric_limits<uint64_t>::max();
    }
    total += candidateCount * perCandidate;
    return total;
}

uint64_t PreparedPopulationInput::objectCount() const
{
    return mImpl == nullptr ? 0u : static_cast<uint64_t>(mImpl->copiedObjects.size());
}

ovspatialquery_api_status_t PreparedPopulationInput::materialize(const GenerationLease& base, ObjectTable& outObjects,
                                                                 std::optional<uint64_t>& outGeometryCharge)
{
    outGeometryCharge.reset();
    if (mImpl == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    try
    {
        if (mImpl->cudaCapture != nullptr && mImpl->cudaCapture->materializesHostObjects())
        {
            return mImpl->cudaCapture->materializeHostObjects(base, outObjects, mImpl->changes, outGeometryCharge);
        }
        if (mImpl->preMaterialized)
        {
            outObjects = std::move(mImpl->copiedObjects);
            if (mImpl->cudaCapture == nullptr)
            {
                outGeometryCharge = mImpl->copiedGeometryCharge;
            }
            mImpl->copiedGeometryCharge.reset();
            return OVSPATIALQUERY_API_SUCCESS;
        }
        return OVSPATIALQUERY_API_INTERNAL;
    }
    catch (const std::bad_alloc&)
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    catch (...)
    {
        return OVSPATIALQUERY_API_INTERNAL;
    }
}

std::optional<VersionChangeRecord> PreparedPopulationInput::releaseChanges()
{
    return mImpl == nullptr ? std::nullopt : std::move(mImpl->changes);
}

void PreparedPopulationInput::reset() { mImpl.reset(); }

} // namespace detail
} // namespace ovspatialquery
