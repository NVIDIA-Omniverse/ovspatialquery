/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_pair_reporter_transition.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <utility>

namespace
{

// Membership in a sorted identity list, answered for most absent identities
// by one bit of a hashed filter with 32 to 64 bits per identity.
class IdentityFilter
{
  public:
    explicit IdentityFilter(const std::vector<uint64_t>& ids) : mIds(ids)
    {
        if (ids.empty())
        {
            return;
        }
        size_t bitCount = 64u;
        mShift          = 58u;
        while (bitCount * 2u <= ids.size() * 64u)
        {
            bitCount *= 2u;
            --mShift;
        }
        mBits.assign(bitCount / 64u, 0u);
        for (const uint64_t id : ids)
        {
            const uint64_t bit = hash(id);
            mBits[bit / 64u] |= uint64_t(1u) << (bit % 64u);
        }
    }

    bool contains(uint64_t id) const
    {
        if (mBits.empty())
        {
            return false;
        }
        const uint64_t bit = hash(id);
        return (mBits[bit / 64u] >> (bit % 64u) & 1u) != 0u && std::binary_search(mIds.begin(), mIds.end(), id);
    }

  private:
    uint64_t hash(uint64_t id) const { return (id * UINT64_C(0x9E3779B97F4A7C15)) >> mShift; }

    const std::vector<uint64_t>& mIds;
    std::vector<uint64_t>        mBits;
    unsigned                     mShift = 58u;
};

// Finds accepted pairs with a changed endpoint. Accepted pairs are ordered by
// first identity, so one cursor answers the first endpoint; the second needs
// the filter.
class ChangedEndpoints
{
  public:
    ChangedEndpoints(const std::vector<uint64_t>& firstIds, const std::vector<uint64_t>& secondIds)
        : mFirst(firstIds.data()), mFirstEnd(firstIds.data() + firstIds.size()), mSecond(secondIds)
    {}

    // The first pair in [begin, end) with a changed endpoint, or end.
    const ovspatialquery_pair_t* find(const ovspatialquery_pair_t* begin, const ovspatialquery_pair_t* end)
    {
        const uint64_t* first = mFirst;
        for (; begin != end; ++begin)
        {
            while (first != mFirstEnd && *first < begin->first.value)
            {
                ++first;
            }
            if ((first != mFirstEnd && *first == begin->first.value) || mSecond.contains(begin->second.value))
            {
                break;
            }
        }
        mFirst = first;
        return begin;
    }

  private:
    const uint64_t*      mFirst;
    const uint64_t*      mFirstEnd;
    const IdentityFilter mSecond;
};

// The first pair in [begin, end) not ordered before `pair`, searched in
// doubling steps from `begin`: the next changed pair is usually far ahead at
// low motion and close at high motion.
const ovspatialquery_pair_t* lowerBoundFrom(const ovspatialquery_pair_t* begin, const ovspatialquery_pair_t* end,
                                            const ovspatialquery_pair_t& pair)
{
    const ovspatialquery::detail::PairLess less;
    size_t                                 step = 1u;
    while (static_cast<size_t>(end - begin) > step && less(begin[step], pair))
    {
        begin += step;
        step *= 2u;
    }
    return std::lower_bound(begin, begin + std::min(step, static_cast<size_t>(end - begin)), pair, less);
}

} // namespace

namespace ovspatialquery
{
namespace detail
{

PairReporterTransition evaluatePairReporterTransition(const std::optional<PairReporterAcceptedState>& accepted,
                                                      ovspatialquery_ordinal_t                        firstOrdinal,
                                                      ovspatialquery_ordinal_t                        secondOrdinal,
                                                      std::vector<ovspatialquery_pair_t>              activePairs,
                                                      uint64_t                                        deltaPairCapacity)
{
    // A first advance compares with an empty baseline: everything enters.
    const std::vector<ovspatialquery_pair_t>  none;
    const std::vector<ovspatialquery_pair_t>& baseline = accepted.has_value() ? accepted->activePairs : none;
    PairReporterTransition                    transition;
    if (static_cast<uint64_t>(activePairs.size()) > deltaPairCapacity && baseline.empty())
    {
        // Everything would enter: stop before copying past the delta capacity.
        transition.outcome = PairReporterResultOutcome::deltaCapacity;
        return transition;
    }
    transition.entered.reserve(static_cast<size_t>(deltaPairCapacity));
    transition.exited.reserve(static_cast<size_t>(deltaPairCapacity));
    std::set_difference(activePairs.begin(), activePairs.end(), baseline.begin(), baseline.end(),
                        std::back_inserter(transition.entered), PairLess{});
    std::set_difference(baseline.begin(), baseline.end(), activePairs.begin(), activePairs.end(),
                        std::back_inserter(transition.exited), PairLess{});
    if (static_cast<uint64_t>(transition.entered.size() + transition.exited.size()) > deltaPairCapacity)
    {
        transition.outcome = PairReporterResultOutcome::deltaCapacity;
        return transition;
    }
    transition.replacement = PairReporterAcceptedState{firstOrdinal, secondOrdinal, std::move(activePairs)};
    return transition;
}

PairReporterTransition evaluatePatchedPairReporterTransition(
    const PairReporterAcceptedState& accepted, ovspatialquery_ordinal_t firstOrdinal,
    ovspatialquery_ordinal_t secondOrdinal, const std::vector<ovspatialquery_pair_t>& changedPairs,
    const std::vector<uint64_t>& firstChangedIds, const std::vector<uint64_t>& secondChangedIds,
    uint64_t activePairCapacity, uint64_t deltaPairCapacity, std::vector<ovspatialquery_pair_t>& activeStorage)
{
    PairReporterTransition              transition;
    ChangedEndpoints                    changedEndpoints(firstChangedIds, secondChangedIds);
    std::vector<ovspatialquery_pair_t>& active = activeStorage;
    std::vector<ovspatialquery_pair_t>  entered;
    std::vector<ovspatialquery_pair_t>  exited;
    active.clear();
    active.reserve(static_cast<size_t>(activePairCapacity));
    entered.reserve(static_cast<size_t>(deltaPairCapacity));
    exited.reserve(static_cast<size_t>(deltaPairCapacity));
    // An active overflow decides the outcome whatever the deltas, so the pass
    // stops there; a delta overflow only stops recording deltas.
    bool       deltaOverflow = false;
    const auto keep          = [&](const ovspatialquery_pair_t* begin, const ovspatialquery_pair_t* end)
    {
        if (static_cast<uint64_t>(end - begin) > activePairCapacity - static_cast<uint64_t>(active.size()))
        {
            return false;
        }
        active.insert(active.end(), begin, end);
        return true;
    };
    const auto record = [&](std::vector<ovspatialquery_pair_t>& delta, const ovspatialquery_pair_t& pair)
    {
        deltaOverflow = deltaOverflow || static_cast<uint64_t>(entered.size() + exited.size()) >= deltaPairCapacity;
        if (!deltaOverflow)
        {
            delta.push_back(pair);
        }
    };
    // Most accepted pairs are still current: keep them in runs, which end at
    // an accepted pair with a changed endpoint (it exited).
    const ovspatialquery_pair_t*       next               = accepted.activePairs.data();
    const ovspatialquery_pair_t* const acceptedEnd        = next + accepted.activePairs.size();
    const auto                         keepAcceptedBefore = [&](const ovspatialquery_pair_t* end)
    {
        while (next != end)
        {
            const ovspatialquery_pair_t* changed = changedEndpoints.find(next, end);
            if (!keep(next, changed))
            {
                return false;
            }
            next = changed;
            if (changed != end)
            {
                record(exited, *changed);
                ++next;
            }
        }
        return true;
    };
    const PairLess less;
    for (const ovspatialquery_pair_t& pair : changedPairs)
    {
        if (!keepAcceptedBefore(lowerBoundFrom(next, acceptedEnd, pair)))
        {
            transition.outcome = PairReporterResultOutcome::activeCapacity;
            return transition;
        }
        // An accepted pair equal to this one is still current.
        if (next != acceptedEnd && !less(pair, *next))
        {
            ++next;
        }
        else
        {
            record(entered, pair);
        }
        if (!keep(&pair, &pair + 1))
        {
            transition.outcome = PairReporterResultOutcome::activeCapacity;
            return transition;
        }
    }
    if (!keepAcceptedBefore(acceptedEnd))
    {
        transition.outcome = PairReporterResultOutcome::activeCapacity;
        return transition;
    }
    if (deltaOverflow)
    {
        transition.outcome = PairReporterResultOutcome::deltaCapacity;
        return transition;
    }
    transition.entered     = std::move(entered);
    transition.exited      = std::move(exited);
    transition.replacement = PairReporterAcceptedState{firstOrdinal, secondOrdinal, std::move(active)};
    return transition;
}

} // namespace detail
} // namespace ovspatialquery
