/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_PARTICIPATION_H
#define OVSPATIALQUERY_PARTICIPATION_H

#include "ovspatialquery_private_api.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace ovspatialquery
{
namespace detail
{

struct PopulationObject;

enum class ParticipationOutcome : uint8_t
{
    success,
    invalidArgument,
    invalidHandle,
    invalidTargetSource,
    busy,
    hostBudget,
    outOfMemory,
    internalFailure,
};

/* The host's table as passed: `count` object IDs (Ovstage prim paths) and their bits; the
 * arrays are null exactly when `count` is zero, which the public setter checks. */
struct ParticipationTableDesc
{
    ParticipationBits        defaultBits = kAllParticipationBits;
    const uint64_t*          paths       = nullptr;
    const ParticipationBits* bits        = nullptr;
    size_t                   count       = 0u;
};

/* A deep copy of one table, its entries sorted by object ID. */
class ParticipationTable
{
  public:
    ParticipationTable() = default;

    /* The listed entry's bits, or the default bits for an unlisted object. */
    ParticipationBits bitsFor(uint64_t objectId) const noexcept;
    uint64_t          residentByteCount() const noexcept;

  private:
    friend ParticipationOutcome copyParticipationTable(const ParticipationTableDesc&,
                                                       std::shared_ptr<const ParticipationTable>&) noexcept;

    ParticipationBits                                   mDefaultBits = kAllParticipationBits;
    std::vector<std::pair<uint64_t, ParticipationBits>> mEntries;
    uint64_t                                            mResidentByteCount = 0u;
};

uint64_t participationTableReservationByteCount(size_t count) noexcept;

// Private first-slice admission reserves the complete candidate while the
// installed table remains charged. A net-fitting replacement may therefore fail
// its host budget at the old-plus-new transient peak.

ParticipationOutcome copyParticipationTable(const ParticipationTableDesc&              desc,
                                            std::shared_ptr<const ParticipationTable>& outTable) noexcept;

/* Gives each object its table bits, or all bits for every object without a table. */
void applyParticipationTable(const ParticipationTable* table, std::vector<PopulationObject>& objects) noexcept;

} // namespace detail
} // namespace ovspatialquery

#endif
