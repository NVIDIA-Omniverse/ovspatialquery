/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_target_resolver.h"

#include <utility>

namespace ovspatialquery
{
namespace detail
{
namespace
{
TargetResolveOutcome unavailableViewOutcome(const TargetSelection& selection) noexcept
{
    return selection.kind == TargetSelectionKind::retainedExact || selection.kind == TargetSelectionKind::admittedExact
               ? TargetResolveOutcome::ordinalNotRetained
               : TargetResolveOutcome::notReady;
}
} // namespace

TargetResolver::TargetResolver(const PopulationTable& populations) : mPopulations(populations) {}

ovspatialquery_api_status_t TargetResolver::materializeHostCopy(const GenerationLease& generation,
                                                                InstanceHostLedger*    hostLedger) const
{
    return mPopulations.materializeHostCopy(generation, hostLedger);
}

TargetResolveOutcome TargetResolver::resolve(const TargetRequest& request, ResolvedTarget& outTarget) const
{
    outTarget = ResolvedTarget{};
    GenerationLease            generation;
    const TargetResolveOutcome outcome =
        mPopulations.pinResolvedTarget({request.target.population, request.selection}, generation);
    if (outcome != TargetResolveOutcome::success)
    {
        return outcome;
    }
    const std::shared_ptr<ViewState>& view = request.target.view;
    if (view != nullptr)
    {
        if (view->automatic)
        {
            if (!generation.selectViewMembership(view->handle))
            {
                return unavailableViewOutcome(request.selection);
            }
        }
        else if (view->selectedIds == nullptr)
        {
            return unavailableViewOutcome(request.selection);
        }
        else
        {
            generation.selectIds(view->selectedIds);
        }
    }
    ResolvedTarget resolved;
    resolved.population = request.target.population;
    resolved.ordinal    = generation.ordinal();
    resolved.generation = std::move(generation);
    outTarget           = std::move(resolved);
    return TargetResolveOutcome::success;
}

TargetResolveOutcome TargetResolver::resolvePair(const std::array<TargetRequest, 2u>& requests,
                                                 std::array<ResolvedTarget, 2u>&      outTargets) const
{
    outTargets = {};
    std::array<GenerationSelectionRequest, 2u> generationRequests{};
    for (size_t index = 0u; index < requests.size(); ++index)
    {
        generationRequests[index] = {requests[index].target.population, requests[index].selection};
    }
    std::array<GenerationLease, 2u> generations{};
    const TargetResolveOutcome      outcome = mPopulations.pinResolvedPair(generationRequests, generations);
    if (outcome != TargetResolveOutcome::success)
    {
        return outcome;
    }
    std::array<ResolvedTarget, 2u> resolved{};
    for (size_t index = 0u; index < requests.size(); ++index)
    {
        GenerationLease&                  generation = generations[index];
        const std::shared_ptr<ViewState>& view       = requests[index].target.view;
        if (view != nullptr)
        {
            if (view->automatic)
            {
                if (!generation.selectViewMembership(view->handle))
                {
                    return unavailableViewOutcome(requests[index].selection);
                }
            }
            else if (view->selectedIds == nullptr)
            {
                return unavailableViewOutcome(requests[index].selection);
            }
            else
            {
                generation.selectIds(view->selectedIds);
            }
        }
        resolved[index].population = requests[index].target.population;
        resolved[index].ordinal    = generation.ordinal();
        resolved[index].generation = std::move(generation);
    }
    outTargets = std::move(resolved);
    return TargetResolveOutcome::success;
}

} // namespace detail
} // namespace ovspatialquery
