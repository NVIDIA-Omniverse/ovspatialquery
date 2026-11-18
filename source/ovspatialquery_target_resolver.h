/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_TARGET_RESOLVER_H
#define OVSPATIALQUERY_TARGET_RESOLVER_H

#include "ovspatialquery_handle_directory.h"

#include <array>
#include <cstdint>
#include <memory>

namespace ovspatialquery
{
namespace detail
{

enum class TargetSelectionKind : uint8_t
{
    head,
    retainedExact,
    // A completed retained ordinal or a device-prepared publication that is
    // admitted and not yet complete (ADR-0021); CUDA plan launches only.
    admittedExact,
};

struct TargetSelection
{
    TargetSelectionKind      kind    = TargetSelectionKind::head;
    ovspatialquery_ordinal_t ordinal = 0;
};

// A view's target population is its parent: the API producers derive
// `population` from `view->parent`, so the resolver does not check it again.
struct PopulationTarget
{
    PopulationKey              population;
    std::shared_ptr<ViewState> view;
};

struct TargetRequest
{
    PopulationTarget target;
    TargetSelection  selection;
};

struct GenerationSelectionRequest
{
    PopulationKey   population;
    TargetSelection selection;
};

enum class TargetResolveOutcome : uint8_t
{
    success,
    invalid,
    notReady,
    ordinalNotRetained,
    ordinalMismatch,
    resourceExhausted,
};

// The admission status of a launch whose target did not resolve; the
// synchronous queries map `invalid` to INVALID_ARGUMENT instead.
inline ovspatialquery_api_status_t admissionStatus(TargetResolveOutcome outcome) noexcept
{
    switch (outcome)
    {
    case TargetResolveOutcome::success: return OVSPATIALQUERY_API_SUCCESS;
    case TargetResolveOutcome::invalid: return OVSPATIALQUERY_API_INVALID_HANDLE;
    case TargetResolveOutcome::notReady: return OVSPATIALQUERY_API_NOT_READY;
    case TargetResolveOutcome::ordinalNotRetained: return OVSPATIALQUERY_API_ORDINAL_NOT_RETAINED;
    case TargetResolveOutcome::ordinalMismatch: return OVSPATIALQUERY_API_ORDINAL_MISMATCH;
    case TargetResolveOutcome::resourceExhausted: return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    return OVSPATIALQUERY_API_INTERNAL;
}

struct ResolvedTarget
{
    PopulationKey            population;
    ovspatialquery_ordinal_t ordinal = 0;
    // The generation lease also owns the membership selected for a view.
    GenerationLease generation;
};

#if defined(OVSPATIALQUERY_ENABLE_TARGET_RESOLVER_TEST_HOOK)
using TargetResolverAfterFirstPinHook = void (*)();
void setTargetResolverAfterFirstPinHook(TargetResolverAfterFirstPinHook hook);
#endif

class TargetResolver
{
  public:
    explicit TargetResolver(const PopulationTable& populations);

    TargetResolveOutcome resolve(const TargetRequest& request, ResolvedTarget& outTarget) const;
    TargetResolveOutcome resolvePair(const std::array<TargetRequest, 2u>& requests,
                                     std::array<ResolvedTarget, 2u>&      outTargets) const;
    // A CPU reader's host copy of a readable device-prepared version; see
    // PopulationTable::materializeHostCopy.
    ovspatialquery_api_status_t materializeHostCopy(const GenerationLease& generation,
                                                    InstanceHostLedger*    hostLedger) const;

  private:
    const PopulationTable& mPopulations;
};

} // namespace detail
} // namespace ovspatialquery

#endif
