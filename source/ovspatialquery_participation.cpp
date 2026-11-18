/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_participation.h"

#include "ovspatialquery_internal.h"

#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>

namespace ovspatialquery
{
namespace detail
{
namespace
{

constexpr uint64_t kAllocationAllowance = 128u;

using Entry = std::pair<uint64_t, ParticipationBits>;

uint64_t residentByteCountFor(size_t entryCount) noexcept
{
    const uint64_t fixed = static_cast<uint64_t>(sizeof(ParticipationTable)) + kAllocationAllowance;
    if (entryCount > static_cast<size_t>((std::numeric_limits<uint64_t>::max() - fixed) / sizeof(Entry)))
    {
        return std::numeric_limits<uint64_t>::max();
    }
    return fixed + static_cast<uint64_t>(entryCount) * static_cast<uint64_t>(sizeof(Entry));
}

} // namespace

ParticipationBits ParticipationTable::bitsFor(uint64_t objectId) const noexcept
{
    const std::vector<Entry>::const_iterator found =
        std::lower_bound(mEntries.begin(), mEntries.end(), objectId,
                         [](const Entry& entry, uint64_t value) { return entry.first < value; });
    return found != mEntries.end() && found->first == objectId ? found->second : mDefaultBits;
}

uint64_t ParticipationTable::residentByteCount() const noexcept { return mResidentByteCount; }

uint64_t participationTableReservationByteCount(size_t count) noexcept { return residentByteCountFor(count); }

ParticipationOutcome copyParticipationTable(const ParticipationTableDesc&              desc,
                                            std::shared_ptr<const ParticipationTable>& outTable) noexcept
{
    outTable.reset();
    if (residentByteCountFor(desc.count) == std::numeric_limits<uint64_t>::max())
    {
        return ParticipationOutcome::outOfMemory;
    }
    try
    {
        std::shared_ptr<ParticipationTable> candidate = std::make_shared<ParticipationTable>();
        candidate->mDefaultBits                       = desc.defaultBits;
        candidate->mEntries.reserve(desc.count);
        for (size_t index = 0u; index < desc.count; ++index)
        {
            if (desc.paths[index] == 0u)
            {
                return ParticipationOutcome::invalidArgument;
            }
            candidate->mEntries.emplace_back(desc.paths[index], desc.bits[index]);
        }
        std::sort(candidate->mEntries.begin(), candidate->mEntries.end(),
                  [](const Entry& first, const Entry& second) { return first.first < second.first; });
        if (std::adjacent_find(candidate->mEntries.begin(), candidate->mEntries.end(),
                               [](const Entry& first, const Entry& second)
                               { return first.first == second.first; }) != candidate->mEntries.end())
        {
            return ParticipationOutcome::invalidArgument;
        }
        candidate->mResidentByteCount = residentByteCountFor(candidate->mEntries.capacity());
        if (candidate->mResidentByteCount == std::numeric_limits<uint64_t>::max())
        {
            return ParticipationOutcome::outOfMemory;
        }
        outTable = std::move(candidate);
        return ParticipationOutcome::success;
    }
    catch (const std::bad_alloc&)
    {
        return ParticipationOutcome::outOfMemory;
    }
    catch (const std::length_error&)
    {
        return ParticipationOutcome::outOfMemory;
    }
    catch (...)
    {
        return ParticipationOutcome::internalFailure;
    }
}

void applyParticipationTable(const ParticipationTable* table, std::vector<PopulationObject>& objects) noexcept
{
    for (PopulationObject& object : objects)
    {
        object.participationBits = table == nullptr ? kAllParticipationBits : table->bitsFor(object.id.value);
    }
}

} // namespace detail
} // namespace ovspatialquery
