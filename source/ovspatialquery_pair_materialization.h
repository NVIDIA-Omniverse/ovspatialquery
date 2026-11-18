/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_PAIR_MATERIALIZATION_H
#define OVSPATIALQUERY_PAIR_MATERIALIZATION_H

#include "ovspatialquery_target_resolver.h"

#include <ovspatialquery/ovspatialquery_pairs.h>

#include <cstdint>
#include <vector>

inline bool operator==(const ovspatialquery_pair_t& left, const ovspatialquery_pair_t& right) noexcept
{
    return left.first.value == right.first.value && left.second.value == right.second.value;
}

namespace ovspatialquery
{
namespace detail
{

// Orders pair rows by first, then second identity.
struct PairLess
{
    bool operator()(const ovspatialquery_pair_t& left, const ovspatialquery_pair_t& right) const noexcept
    {
        return left.first.value < right.first.value ||
               (left.first.value == right.first.value && left.second.value < right.second.value);
    }
};

enum class PairEndpointRelation : uint8_t
{
    /** Different population identity; equal numeric object IDs remain valid. */
    distinctPopulations,
    /** One population/ordinal, but distinct directed target selections. */
    samePopulationDistinctSelections,
    /** Identical target, resolved ordinal, and query-bit selection. */
    identicalSelection,
};

// The relation of two endpoints whose resolved targets do or do not share a population.
inline PairEndpointRelation pairEndpointRelation(bool samePopulation, const ovspatialquery_target_ref_t& first,
                                                 const ParticipationBits&           firstQueryBits,
                                                 const ovspatialquery_target_ref_t& second,
                                                 const ParticipationBits&           secondQueryBits) noexcept
{
    const bool sameTarget =
        first.kind == second.kind && ((first.kind == 0u && first.handle.population == second.handle.population) ||
                                      (first.kind == 1u && first.handle.view == second.handle.view));
    return !samePopulation ? PairEndpointRelation::distinctPopulations
                           : (sameTarget && participationEqual(firstQueryBits, secondQueryBits)
                                  ? PairEndpointRelation::identicalSelection
                                  : PairEndpointRelation::samePopulationDistinctSelections);
}

struct PairMaterializationEndpoint
{
    /** Atomically resolved generation plus its generation-owned membership. */
    ResolvedTarget target;
    /** Zero accepts all members; otherwise participation bits must intersect. */
    ParticipationBits queryBits{};
};

// One admission flag per object of an endpoint with a selection or query bits;
// an identical selection's second endpoint shares the first's.
inline uint64_t pairAdmissionFlagBytes(const ResolvedTarget& first, const ParticipationBits& firstQueryBits,
                                       const ResolvedTarget& second, const ParticipationBits& secondQueryBits,
                                       PairEndpointRelation relation)
{
    uint64_t bytes = 0u;
    if (first.generation.hasSelection() || !participationIsZero(firstQueryBits))
    {
        bytes += first.generation.objects().size();
    }
    if (relation != PairEndpointRelation::identicalSelection &&
        (second.generation.hasSelection() || !participationIsZero(secondQueryBits)))
    {
        bytes += second.generation.objects().size();
    }
    return bytes;
}

// Scratch bytes of one evaluation, which callers include in their execution
// staging lease before evaluating: the admission flags and one visit stamp per
// second-side object for duplicate-safe per-object traversal.
inline uint64_t pairTraversalScratchBytes(const ResolvedTarget& first, const ParticipationBits& firstQueryBits,
                                          const ResolvedTarget& second, const ParticipationBits& secondQueryBits,
                                          PairEndpointRelation relation, double contactTolerance)
{
    uint64_t bytes = pairAdmissionFlagBytes(first, firstQueryBits, second, secondQueryBits, relation);
    if (first.generation.hasBackend() && second.generation.hasBackend() && !first.generation.objects().empty() &&
        (first.population == second.population || contactTolerance != 0.0))
    {
        bytes += uint64_t(second.generation.objects().size()) * sizeof(size_t);
    }
    return bytes;
}

// Scratch bytes of an incremental evaluation (ADR-0019): the admission flags,
// one visit stamp per object of the larger endpoint, and per object of each
// endpoint population a changed flag and, for at most a quarter of them, a
// position, an identity and an identity-filter word.
inline uint64_t pairIncrementalScratchBytes(const ResolvedTarget& first, const ParticipationBits& firstQueryBits,
                                            const ResolvedTarget& second, const ParticipationBits& secondQueryBits,
                                            PairEndpointRelation relation)
{
    constexpr uint64_t perObject   = 1u + (sizeof(uint32_t) + 2u * sizeof(uint64_t)) / 4u;
    const uint64_t     firstCount  = first.generation.objects().size();
    const uint64_t     secondCount = second.generation.objects().size();
    return pairAdmissionFlagBytes(first, firstQueryBits, second, secondQueryBits, relation) +
           (first.population == second.population ? firstCount : firstCount + secondCount) * perObject +
           (firstCount > secondCount ? firstCount : secondCount) * sizeof(size_t);
}

struct PairMaterializationRequest
{
    PairMaterializationEndpoint first;
    PairMaterializationEndpoint second;
    /** Caller-resolved relationship between the two target selections. */
    PairEndpointRelation relation = PairEndpointRelation::distinctPopulations;
    /** One finite nonnegative margin for the inclusive world-AABB predicate. */
    double contactTolerance = 0.0;
    /** Hard upper bound for the complete candidate pair set; a bounded
        snapshot retains at most this many rows and reports truncation. */
    uint64_t activePairCapacity = 0u;
};

enum class PairMaterializationOutcome : uint8_t
{
    success,
    activeCapacity,
    resourceExhausted,
    internalFailure,
};

/**
 * Builds one complete sorted unique pair set from already-resolved immutable
 * generation leases. Failure preserves `outPairs`. Candidates come from the
 * backend trees of both generations (a generation without a CPU backend is an
 * internal failure) and pass the exact predicate; O(P) pair storage whose
 * vector capacity never exceeds `activePairCapacity`, plus the caller-charged
 * traversal scratch above.
 */
PairMaterializationOutcome materializePairSet(const PairMaterializationRequest&   request,
                                              std::vector<ovspatialquery_pair_t>& outPairs) noexcept;

// The objects of one endpoint that changed since a reporter's accepted
// version, by position in the endpoint's resolved generation.
struct PairChangedObjects
{
    /** Sorted unique positions. */
    std::vector<uint32_t> positions;
    /** One flag per object of the generation, set at `positions`. */
    std::vector<uint8_t> flags;
};

/**
 * Builds the sorted unique pairs of the request with at least one changed
 * endpoint: a first object in `firstChanged` or a second object in
 * `secondChanged` (the same set for one population). Each changed object
 * queries the other endpoint's tree once; a pair of two changed objects is
 * visited once. Bounds, outcomes and failure behaviour are those of
 * `materializePairSet`, whose complete set is these pairs plus the pairs of
 * unchanged objects.
 */
PairMaterializationOutcome materializeChangedPairs(const PairMaterializationRequest&   request,
                                                   const PairChangedObjects&           firstChanged,
                                                   const PairChangedObjects&           secondChanged,
                                                   std::vector<ovspatialquery_pair_t>& outPairs) noexcept;

/**
 * Evaluates one bounded stateless snapshot over already-resolved immutable
 * generations. The function retains at most `activePairCapacity` rows, in
 * deterministic order, and stops after observing the first additional unique
 * match needed to prove truncation. Pair storage is O(K), where K is
 * `activePairCapacity`; every traversal visits a matching pair once using the
 * caller-charged traversal scratch. Failure preserves `outPairs` and
 * `outTruncated`; no reporter baseline or other persistent state participates
 * in this seam.
 */
PairMaterializationOutcome materializeBoundedPairSnapshot(const PairMaterializationRequest&   request,
                                                          std::vector<ovspatialquery_pair_t>& outPairs,
                                                          bool&                               outTruncated) noexcept;

#if defined(OVSPATIALQUERY_ENABLE_PAIR_MATERIALIZATION_TEST_HOOK)
/** Makes the next candidate allocation report resource exhaustion. */
void failNextPairMaterializationAllocationForTest() noexcept;
/** Resets the bounded snapshot's count of matching pairs observed. */
void resetPairSnapshotObservedMatchCountForTest() noexcept;
/** Returns matching pairs observed since the last reset. */
uint64_t pairSnapshotObservedMatchCountForTest() noexcept;
#endif

} // namespace detail
} // namespace ovspatialquery

#endif
