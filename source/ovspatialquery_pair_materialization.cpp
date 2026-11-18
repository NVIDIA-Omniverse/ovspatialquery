/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_pair_materialization.h"

#include "ovspatialquery_backend.h"
#include "ovspatialquery_cuda_pair_types.h"
#include "ovspatialquery_float_bounds.h"
#include "ovspatialquery_internal.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <new>
#include <stdexcept>

namespace ovspatialquery
{
namespace detail
{
namespace
{

#if defined(OVSPATIALQUERY_ENABLE_PAIR_MATERIALIZATION_TEST_HOOK)
std::atomic<bool>     gFailNextPairMaterializationAllocation{false};
std::atomic<uint64_t> gPairSnapshotObservedMatchCount{0u};
#endif

PairMaterializationOutcome appendPairBounded(const ovspatialquery_pair_t& pair, uint64_t activePairCapacity,
                                             std::vector<ovspatialquery_pair_t>& pairs)
{
    if (pairs.size() >= activePairCapacity)
    {
        return PairMaterializationOutcome::activeCapacity;
    }
    if (pairs.size() == pairs.capacity())
    {
#if defined(OVSPATIALQUERY_ENABLE_PAIR_MATERIALIZATION_TEST_HOOK)
        if (gFailNextPairMaterializationAllocation.exchange(false, std::memory_order_acq_rel))
        {
            return PairMaterializationOutcome::resourceExhausted;
        }
#endif
        // Only a bounded snapshot appends to an unreserved vector: its first
        // match reserves the whole bound, so no uncharged old-plus-new growth
        // peak follows, and a bound past max_size() fails as exhaustion.
        pairs.reserve(static_cast<size_t>(std::min<uint64_t>(activePairCapacity, pairs.max_size())));
    }
    pairs.push_back(pair);
    return PairMaterializationOutcome::success;
}

// Each endpoint's selection membership and participation bits, resolved once
// per evaluation: every object passes an endpoint with neither, otherwise one
// flag per object replaces a membership search and a participation read per
// candidate. An identical selection shares the first endpoint's flags.
class PairEvaluation
{
  public:
    PairEvaluation(const PairMaterializationEndpoint& firstEndpoint, const PairMaterializationEndpoint& secondEndpoint,
                   PairEndpointRelation relationValue, double tolerance)
        : first(firstEndpoint), second(secondEndpoint), relation(relationValue), contactTolerance(tolerance)
    {
        resolve(first, mFirstAll, mFirstFlags);
        if (relation != PairEndpointRelation::identicalSelection)
        {
            resolve(second, mSecondAll, mSecondFlags);
        }
    }

    bool admitsFirst(size_t objectIndex) const { return mFirstAll || mFirstFlags[objectIndex] != 0u; }
    bool admitsSecond(size_t objectIndex) const
    {
        return relation == PairEndpointRelation::identicalSelection ? admitsFirst(objectIndex)
                                                                    : mSecondAll || mSecondFlags[objectIndex] != 0u;
    }

    const PairMaterializationEndpoint& first;
    const PairMaterializationEndpoint& second;
    const PairEndpointRelation         relation;
    const double                       contactTolerance;

  private:
    static void resolve(const PairMaterializationEndpoint& endpoint, bool& outAll, std::vector<uint8_t>& outFlags)
    {
        outAll = !endpoint.target.generation.hasSelection() && participationIsZero(endpoint.queryBits);
        if (!outAll)
        {
            endpoint.target.generation.admittedObjects(endpoint.queryBits, outFlags);
        }
    }

    bool                 mFirstAll  = true;
    bool                 mSecondAll = true;
    std::vector<uint8_t> mFirstFlags;
    std::vector<uint8_t> mSecondFlags;
};

// Exact rules applied to one candidate index pair: selection membership,
// participation bits, the self-pair and unordered rules, then the double
// predicate with the caller tolerance. Returns the visitor's continue flag;
// a non-match continues.
template <typename Visitor>
bool visitCandidate(const PairEvaluation& evaluation, size_t firstIndex, size_t secondIndex, Visitor& visitor)
{
    const ObjectTable& firstObjects  = evaluation.first.target.generation.objects();
    const ObjectTable& secondObjects = evaluation.second.target.generation.objects();
    if (firstIndex >= firstObjects.size() || secondIndex >= secondObjects.size())
    {
        return true;
    }
    const PopulationObject& firstObject  = firstObjects[firstIndex];
    const PopulationObject& secondObject = secondObjects[secondIndex];
    // An identical selection visits each unordered index pair once.
    if ((evaluation.relation == PairEndpointRelation::identicalSelection && secondIndex <= firstIndex) ||
        !evaluation.admitsFirst(firstIndex) || !evaluation.admitsSecond(secondIndex) ||
        (evaluation.relation != PairEndpointRelation::distinctPopulations &&
         objectIdEqual(firstObject.id, secondObject.id)) ||
        !pairBoundsOverlap(firstObject.bounds.minimum, firstObject.bounds.maximum, secondObject.bounds.minimum,
                           secondObject.bounds.maximum, evaluation.contactTolerance))
    {
        return true;
    }
    ovspatialquery_object_id_t firstId  = firstObject.id;
    ovspatialquery_object_id_t secondId = secondObject.id;
    if (evaluation.relation == PairEndpointRelation::identicalSelection && objectIdLess(secondId, firstId))
    {
        std::swap(firstId, secondId);
    }
    return visitor(ovspatialquery_pair_t{firstId, secondId});
}

// Streams tree-pair candidates from findPairs into the exact rules.
template <typename Visitor>
class TreePairSink final : public BackendPairSink
{
  public:
    TreePairSink(const PairEvaluation& evaluation, Visitor& visitor) : mEvaluation(evaluation), mVisitor(visitor) {}

    bool reportPair(uint32_t objectIndex, uint32_t otherObjectIndex) override
    {
        return visitCandidate(mEvaluation, objectIndex, otherObjectIndex, mVisitor);
    }

    ovspatialquery_api_status_t status() const override { return OVSPATIALQUERY_API_SUCCESS; }

  private:
    const PairEvaluation& mEvaluation;
    Visitor&              mVisitor;
};

// Streams each candidate of one query once using fixed-size, caller-charged
// visit stamps: each query passes a token no earlier query of the same stamps
// used, so repeated backend reports neither grow scratch storage nor count
// twice, and `onCandidate` applies the pair rules.
template <typename OnCandidate>
class StampedCandidateSink final : public BackendCandidateSink
{
  public:
    StampedCandidateSink(std::vector<size_t>& stamps, size_t objectCount, size_t token, OnCandidate& onCandidate)
        : mStamps(stamps), mObjectCount(objectCount), mToken(token), mOnCandidate(onCandidate)
    {}

    bool stopped = false;

    bool reportCandidate(uint32_t objectIndex) override
    {
        if (objectIndex >= mObjectCount)
        {
            mStatus = OVSPATIALQUERY_API_INTERNAL;
            return false;
        }
        if (mStamps[objectIndex] == mToken)
        {
            return true;
        }
        mStamps[objectIndex] = mToken;
        stopped              = !mOnCandidate(static_cast<size_t>(objectIndex));
        return !stopped;
    }

    ovspatialquery_api_status_t status() const override { return mStatus; }

  private:
    std::vector<size_t>&        mStamps;
    const size_t                mObjectCount;
    const size_t                mToken;
    OnCandidate&                mOnCandidate;
    ovspatialquery_api_status_t mStatus = OVSPATIALQUERY_API_SUCCESS;
};

// Reports once each object of `generation` that may lie within `tolerance` of
// `bounds`: the generation's tree holds conservative float proxies, so its
// candidates are a superset of the exact matches, and the fallback objects the
// backend keeps outside its tree arrive through the same sink. A generation
// without a CPU backend is an internal failure, as in `visitMatchingPairs`.
// `onCandidate` returns false to stop, which sets `outStopped`.
template <typename OnCandidate>
ovspatialquery_api_status_t visitCandidates(const GenerationLease& generation, const ovspatialquery_aabb_t& bounds,
                                            double tolerance, std::vector<size_t>& stamps, size_t token,
                                            bool& outStopped, OnCandidate&& onCandidate)
{
    const size_t objectCount = generation.objects().size();
    outStopped               = false;
    if (!generation.hasBackend())
    {
        return OVSPATIALQUERY_API_INTERNAL;
    }
    // Clamping to the float range loses nothing: every tree object lies
    // inside it, as every published bound lies inside the input domain.
    FloatAabb query{};
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        query.minimum[axis] = conservativeFloatBound(bounds.minimum[axis] - tolerance, true);
        query.maximum[axis] = conservativeFloatBound(bounds.maximum[axis] + tolerance, false);
    }
    StampedCandidateSink<OnCandidate> sink(stamps, objectCount, token, onCandidate);
    const ovspatialquery_api_status_t status = generation.backend().aabbOverlapCandidates(query, sink);
    outStopped                               = sink.stopped;
    return status;
}

// Visits every matching pair through the backend trees of both generations: a
// tree-pair traversal for distinct populations at zero tolerance, otherwise
// one tolerance-inflated AABB traversal of the second tree per admitted first
// object. Both trees hold conservative float proxies, so the candidates are a
// superset of the exact pairs; the fallback objects the backend keeps outside
// its tree arrive through the same sinks. Every caller reaches here with CPU
// backends (a device-prepared version through its host copy), so a generation
// without one is an internal failure. Stopping early is a success. Every path
// visits a matching pair once, which the bounded capacity accounting of both
// callers relies on.
template <typename Visitor>
ovspatialquery_api_status_t visitMatchingPairs(const PairMaterializationEndpoint& first,
                                               const PairMaterializationEndpoint& second, PairEndpointRelation relation,
                                               double contactTolerance, Visitor&& visitor)
{
    const GenerationLease& firstGeneration  = first.target.generation;
    const GenerationLease& secondGeneration = second.target.generation;
    if (!firstGeneration.hasBackend() || !secondGeneration.hasBackend())
    {
        return OVSPATIALQUERY_API_INTERNAL;
    }
    const ObjectTable&   firstObjects  = firstGeneration.objects();
    const ObjectTable&   secondObjects = secondGeneration.objects();
    const PairEvaluation evaluation(first, second, relation, contactTolerance);
    if (relation == PairEndpointRelation::distinctPopulations && contactTolerance == 0.0)
    {
        TreePairSink<Visitor> sink(evaluation, visitor);
        return firstGeneration.backend().findPairs(secondGeneration.backend(), sink);
    }
    std::vector<size_t> stamps(firstObjects.empty() ? 0u : secondObjects.size(), std::numeric_limits<size_t>::max());
    for (size_t firstIndex = 0u; firstIndex < firstObjects.size(); ++firstIndex)
    {
        if (!evaluation.admitsFirst(firstIndex))
        {
            continue;
        }
        bool                              stopped = false;
        const ovspatialquery_api_status_t status  = visitCandidates(
            secondGeneration, firstObjects[firstIndex].bounds, contactTolerance, stamps, firstIndex, stopped,
            [&](size_t other) { return visitCandidate(evaluation, firstIndex, other, visitor); });
        if (status != OVSPATIALQUERY_API_SUCCESS || stopped)
        {
            return status;
        }
    }
    return OVSPATIALQUERY_API_SUCCESS;
}

// Visits every matching pair with a changed endpoint once. A changed first
// object queries the second endpoint's candidates; a changed second object
// queries the first endpoint's and skips changed first objects, whose pairs
// the first pass visited. In an identical selection a pair of two changed
// objects is visited from the lower position. Stopping early is a success.
template <typename Visitor>
ovspatialquery_api_status_t visitChangedPairs(const PairMaterializationRequest& request,
                                              const PairChangedObjects&         firstChanged,
                                              const PairChangedObjects& secondChanged, Visitor&& visitor)
{
    const GenerationLease& firstGeneration  = request.first.target.generation;
    const GenerationLease& secondGeneration = request.second.target.generation;
    const size_t           firstCount       = firstGeneration.objects().size();
    const size_t           secondCount      = secondGeneration.objects().size();
    const PairEvaluation   evaluation(request.first, request.second, request.relation, request.contactTolerance);
    std::vector<size_t>    stamps(std::max(firstCount, secondCount), std::numeric_limits<size_t>::max());
    size_t                 token     = 0u;
    bool                   stopped   = false;
    const bool             identical = request.relation == PairEndpointRelation::identicalSelection;
    for (const uint32_t position : firstChanged.positions)
    {
        if (!evaluation.admitsFirst(position))
        {
            continue;
        }
        const ovspatialquery_api_status_t status =
            visitCandidates(secondGeneration, firstGeneration.objects()[position].bounds, request.contactTolerance,
                            stamps, token++, stopped,
                            [&](size_t other)
                            {
                                if (!identical)
                                {
                                    return visitCandidate(evaluation, position, other, visitor);
                                }
                                if (other == position || (firstChanged.flags[other] != 0u && other < position))
                                {
                                    return true;
                                }
                                return visitCandidate(evaluation, std::min<size_t>(position, other),
                                                      std::max<size_t>(position, other), visitor);
                            });
        if (status != OVSPATIALQUERY_API_SUCCESS || stopped)
        {
            return status;
        }
    }
    if (identical)
    {
        return OVSPATIALQUERY_API_SUCCESS;
    }
    for (const uint32_t position : secondChanged.positions)
    {
        if (!evaluation.admitsSecond(position))
        {
            continue;
        }
        const ovspatialquery_api_status_t status = visitCandidates(
            firstGeneration, secondGeneration.objects()[position].bounds, request.contactTolerance, stamps, token++,
            stopped,
            [&](size_t other)
            { return firstChanged.flags[other] != 0u || visitCandidate(evaluation, other, position, visitor); });
        if (status != OVSPATIALQUERY_API_SUCCESS || stopped)
        {
            return status;
        }
    }
    return OVSPATIALQUERY_API_SUCCESS;
}

// Orders pairs by first, then second identity. A per-object traversal visits
// first objects in index order and reports each one's pairs together, so for
// generations in ascending ID order only each group's second identities need
// sorting; any other order takes the complete sort.
void sortPairs(std::vector<ovspatialquery_pair_t>& pairs)
{
    const auto firstLess = [](const ovspatialquery_pair_t& left, const ovspatialquery_pair_t& right)
    { return left.first.value < right.first.value; };
    if (!std::is_sorted(pairs.begin(), pairs.end(), firstLess))
    {
        std::sort(pairs.begin(), pairs.end(), PairLess{});
        return;
    }
    for (auto group = pairs.begin(); group != pairs.end();)
    {
        const auto end = std::find_if(group, pairs.end(),
                                      [&group](const ovspatialquery_pair_t& pair)
                                      { return pair.first.value != group->first.value; });
        std::sort(group, end, PairLess{});
        group = end;
    }
}

PairMaterializationOutcome traversalOutcome(ovspatialquery_api_status_t status)
{
    if (status == OVSPATIALQUERY_API_SUCCESS)
    {
        return PairMaterializationOutcome::success;
    }
    return status == OVSPATIALQUERY_API_OUT_OF_MEMORY ? PairMaterializationOutcome::resourceExhausted
                                                      : PairMaterializationOutcome::internalFailure;
}

// Collects the pairs one traversal visits into a sorted unique set whose
// storage never exceeds the request's active capacity. `traverse` runs the
// traversal with a visitor that returns false to stop. With `outTruncated`,
// a pair past the capacity stops the traversal and sets it instead of
// failing, and the storage is reserved at the first match.
template <typename Traverse>
PairMaterializationOutcome collectPairs(const PairMaterializationRequest& request, Traverse&& traverse,
                                        std::vector<ovspatialquery_pair_t>& outPairs,
                                        bool*                               outTruncated = nullptr) noexcept
{
    try
    {
        std::vector<ovspatialquery_pair_t> pairs;
        bool                               truncated = false;
        if (outTruncated == nullptr)
        {
#if defined(OVSPATIALQUERY_ENABLE_PAIR_MATERIALIZATION_TEST_HOOK)
            if (request.activePairCapacity != 0u &&
                gFailNextPairMaterializationAllocation.exchange(false, std::memory_order_acq_rel))
            {
                return PairMaterializationOutcome::resourceExhausted;
            }
#endif
            pairs.reserve(static_cast<size_t>(request.activePairCapacity));
        }
        PairMaterializationOutcome        appendOutcome = PairMaterializationOutcome::success;
        const ovspatialquery_api_status_t status        = traverse(
            [&](const ovspatialquery_pair_t& pair)
            {
                if (outTruncated != nullptr)
                {
#if defined(OVSPATIALQUERY_ENABLE_PAIR_MATERIALIZATION_TEST_HOOK)
                    gPairSnapshotObservedMatchCount.fetch_add(1u, std::memory_order_relaxed);
#endif
                    if (pairs.size() >= request.activePairCapacity)
                    {
                        truncated = true;
                        return false;
                    }
                }
                appendOutcome = appendPairBounded(pair, request.activePairCapacity, pairs);
                return appendOutcome == PairMaterializationOutcome::success;
            });
        if (appendOutcome != PairMaterializationOutcome::success)
        {
            return appendOutcome;
        }
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return traversalOutcome(status);
        }
        sortPairs(pairs);
        // Every traversal visits a matching pair once, so a snapshot needs no duplicate scan.
        if (outTruncated == nullptr)
        {
            pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
        }
        else
        {
            *outTruncated = truncated;
        }
        outPairs = std::move(pairs);
        return PairMaterializationOutcome::success;
    }
    catch (const std::bad_alloc&)
    {
        return PairMaterializationOutcome::resourceExhausted;
    }
    catch (const std::length_error&)
    {
        return PairMaterializationOutcome::resourceExhausted;
    }
    catch (...)
    {
        return PairMaterializationOutcome::internalFailure;
    }
}

} // namespace

PairMaterializationOutcome materializePairSet(const PairMaterializationRequest&   request,
                                              std::vector<ovspatialquery_pair_t>& outPairs) noexcept
{
    return collectPairs(
        request,
        [&request](auto&& visitor) {
            return visitMatchingPairs(request.first, request.second, request.relation, request.contactTolerance,
                                      visitor);
        },
        outPairs);
}

PairMaterializationOutcome materializeChangedPairs(const PairMaterializationRequest&   request,
                                                   const PairChangedObjects&           firstChanged,
                                                   const PairChangedObjects&           secondChanged,
                                                   std::vector<ovspatialquery_pair_t>& outPairs) noexcept
{
    return collectPairs(
        request, [&](auto&& visitor) { return visitChangedPairs(request, firstChanged, secondChanged, visitor); },
        outPairs);
}

PairMaterializationOutcome materializeBoundedPairSnapshot(const PairMaterializationRequest&   request,
                                                          std::vector<ovspatialquery_pair_t>& outPairs,
                                                          bool&                               outTruncated) noexcept
{
    return collectPairs(
        request,
        [&request](auto&& visitor) {
            return visitMatchingPairs(request.first, request.second, request.relation, request.contactTolerance,
                                      visitor);
        },
        outPairs, &outTruncated);
}

#if defined(OVSPATIALQUERY_ENABLE_PAIR_MATERIALIZATION_TEST_HOOK)
void failNextPairMaterializationAllocationForTest() noexcept
{
    gFailNextPairMaterializationAllocation.store(true, std::memory_order_release);
}

void resetPairSnapshotObservedMatchCountForTest() noexcept
{
    gPairSnapshotObservedMatchCount.store(0u, std::memory_order_release);
}

uint64_t pairSnapshotObservedMatchCountForTest() noexcept
{
    return gPairSnapshotObservedMatchCount.load(std::memory_order_acquire);
}
#endif

} // namespace detail
} // namespace ovspatialquery
