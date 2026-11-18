/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_POPULATION_SOURCE_H
#define OVSPATIALQUERY_POPULATION_SOURCE_H

#include "ovspatialquery_private_api.h"

#include <memory>
#include <vector>

namespace ovspatialquery
{
namespace detail
{

class DevicePreparedState;
struct CopiedSourceSnapshot;
class GeometryIdentityRegistry;
struct PopulationObject;
class ObjectTable;
struct SourceViewRequest;

/* The generation an update replaces, offered to the acquirer so that it can
 * reuse what the source reports unchanged since that ordinal. Absent for the
 * first publication of a population. */
struct SourceAcquisitionBase
{
    ovspatialquery_ordinal_t ordinal = 0;
    const ObjectTable*       objects = nullptr;
    /* The instance's identity registry, which retains the definition of every
     * prepared mesh the base generation holds. */
    const GeometryIdentityRegistry* geometryIdentities = nullptr;
};

using SourceSnapshotAcquirer = ovspatialquery_api_status_t (*)(
    const void* sourceInstance, const void* sourcePathDictionary, const std::shared_ptr<void>& populationSourceState,
    const std::vector<SourceViewRequest>& viewRequests, ovspatialquery_ordinal_t targetOrdinal,
    const SourceAcquisitionBase* base, CopiedSourceSnapshot& outSnapshot);

enum class PopulationPreparationStrategy : uint8_t
{
    untypedInternal,
    typedFiveFamily,
};

// A population owns this binding. Copies are short-lived leases that keep its
// source-specific state alive while acquisition resolves the raw source APIs.
struct PopulationSourceBinding
{
    ovspatialquery_source_kind_t  kind           = OVSPATIALQUERY_SOURCE_EXPLICIT;
    const void*                   instance       = nullptr;
    const void*                   pathDictionary = nullptr;
    std::shared_ptr<void>         sourceState;
    SourceSnapshotAcquirer        acquireSnapshot = nullptr;
    PopulationPreparationStrategy preparation     = PopulationPreparationStrategy::untypedInternal;
    // Explicit populations only: what device preparation knows (ADR-0024).
    std::shared_ptr<DevicePreparedState> deviceState{};
};

} // namespace detail
} // namespace ovspatialquery

#endif
