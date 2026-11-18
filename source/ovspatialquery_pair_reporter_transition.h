/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_PAIR_REPORTER_TRANSITION_H
#define OVSPATIALQUERY_PAIR_REPORTER_TRANSITION_H

#include "ovspatialquery_pair_materialization.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace ovspatialquery
{
namespace detail
{

struct PairReporterAcceptedState
{
    ovspatialquery_ordinal_t firstOrdinal  = 0;
    ovspatialquery_ordinal_t secondOrdinal = 0;
    /** Sorted unique pairs from the last accepted advance. */
    std::vector<ovspatialquery_pair_t> activePairs;
};

enum class PairReporterResultOutcome : uint8_t
{
    advanced,
    activeCapacity,
    deltaCapacity,
};

struct PairReporterTransition
{
    PairReporterResultOutcome                outcome = PairReporterResultOutcome::advanced;
    std::vector<ovspatialquery_pair_t>       entered;
    std::vector<ovspatialquery_pair_t>       exited;
    std::optional<PairReporterAcceptedState> replacement;
};

/**
 * Purely evaluates one reporter transition over already-resolved ordinals.
 *
 * `activePairs` and an accepted state's pairs use PairLess ordering and
 * contain no duplicates. `activePairs` is a complete set that fit the active
 * capacity (materialization reports an overflow before evaluation), and the
 * ordinals advance past an accepted state's (admission rejects backward and
 * stale advances). Only `advanced` returns replacement accepted state, which
 * takes `activePairs`; every other outcome preserves the caller-owned baseline.
 */
PairReporterTransition evaluatePairReporterTransition(const std::optional<PairReporterAcceptedState>& accepted,
                                                      ovspatialquery_ordinal_t                        firstOrdinal,
                                                      ovspatialquery_ordinal_t                        secondOrdinal,
                                                      std::vector<ovspatialquery_pair_t>              activePairs,
                                                      uint64_t deltaPairCapacity);

/**
 * The transition a recomputation of the complete pair set would return,
 * derived from an accepted state instead (ADR-0019); the ordinals advance past
 * the accepted state's, as for `evaluatePairReporterTransition`. A pair
 * has a changed endpoint when its first identity is in `firstChangedIds` or its
 * second in `secondChangedIds` (both sorted). `changedPairs` holds every
 * current pair with a changed endpoint, sorted and unique; an accepted pair
 * without one is still current. One merge pass builds the replacement and the
 * deltas. The replacement's pairs are written into `activeStorage`, whose
 * allocation the replacement takes on `advanced`; any other outcome leaves the
 * allocation in `activeStorage`, so a caller can reuse one buffer across
 * advances.
 */
PairReporterTransition evaluatePatchedPairReporterTransition(
    const PairReporterAcceptedState& accepted, ovspatialquery_ordinal_t firstOrdinal,
    ovspatialquery_ordinal_t secondOrdinal, const std::vector<ovspatialquery_pair_t>& changedPairs,
    const std::vector<uint64_t>& firstChangedIds, const std::vector<uint64_t>& secondChangedIds,
    uint64_t activePairCapacity, uint64_t deltaPairCapacity, std::vector<ovspatialquery_pair_t>& activeStorage);

} // namespace detail
} // namespace ovspatialquery

#endif
