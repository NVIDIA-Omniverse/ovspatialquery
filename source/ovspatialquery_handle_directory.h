/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_HANDLE_DIRECTORY_H
#define OVSPATIALQUERY_HANDLE_DIRECTORY_H

#include "ovspatialquery_participation.h"
#include "ovspatialquery_population_source.h"
#include "ovspatialquery_population_table.h"

#include "ovspatialquery_private_api.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace ovspatialquery
{
namespace detail
{

struct SourceViewRequest;
struct PreparedSourceViewMembership;

enum class HandleKind : uint32_t
{
    population = 1u,
    view       = 2u,
};

struct ViewState
{
    ovspatialquery_view_handle_t                 handle = OVSPATIALQUERY_INVALID_VIEW;
    PopulationKey                                parent;
    std::shared_ptr<const std::vector<uint64_t>> selectedIds;
    bool                                         ready              = true;
    bool                                         automatic          = false;
    size_t                                       pendingUpdateCount = 0u;
    // Batch plans and pair reporters that target the view.
    size_t                planReferenceCount = 0u;
    std::shared_ptr<void> sourceState;
};

struct PopulationState
{
    PopulationKey                             key;
    PopulationSourceBinding                   source;
    std::shared_ptr<const ParticipationTable> participation;
};

struct HandleCell
{
    HandleKind                 kind = HandleKind::population;
    PopulationState            population;
    std::shared_ptr<ViewState> view;
};

class HandleDirectory
{
  public:
    HandleDirectory()  = default;
    ~HandleDirectory() = default;

    HandleDirectory(const HandleDirectory&)            = delete;
    HandleDirectory& operator=(const HandleDirectory&) = delete;

    ovspatialquery_api_status_t createPopulation(PopulationKey                       population,
                                                 ovspatialquery_population_handle_t& outHandle,
                                                 PopulationSourceBinding             source);
    ovspatialquery_api_status_t resolvePopulationSource(ovspatialquery_population_handle_t handle,
                                                        PopulationKey&                     outPopulation,
                                                        PopulationSourceBinding&           outSource) const;
    // The population key, and its source binding only when asked: a query that does not
    // read the source skips copying it.
    ovspatialquery_api_status_t resolvePopulation(ovspatialquery_population_handle_t handle,
                                                  PopulationKey&                     outPopulation,
                                                  PopulationSourceBinding*           outSource = nullptr) const;
    ovspatialquery_api_status_t invalidatePopulation(
        ovspatialquery_population_handle_t handle, PopulationSourceBinding& outRetiredSource,
        std::shared_ptr<const ParticipationTable>* outRetiredParticipation = nullptr);
    ovspatialquery_api_status_t createView(PopulationKey parent, std::vector<uint64_t> explicitIds,
                                           ovspatialquery_view_handle_t& outHandle);
    ovspatialquery_api_status_t createSourceView(PopulationKey parent, std::shared_ptr<void> sourceState,
                                                 ovspatialquery_view_handle_t& outHandle);
    ovspatialquery_api_status_t resolveView(ovspatialquery_view_handle_t handle,
                                            std::shared_ptr<ViewState>&  outView) const;
    // A batch plan requires a ready view (NOT_READY otherwise); a pair reporter does not.
    ovspatialquery_api_status_t retainView(ovspatialquery_view_handle_t handle, bool requireReady,
                                           std::shared_ptr<ViewState>& outView);
    void                        finishViewReference(const std::shared_ptr<ViewState>& view);
    ovspatialquery_api_status_t invalidateView(ovspatialquery_view_handle_t handle);
    bool                        hasLiveView(PopulationKey parent) const;
    ovspatialquery_api_status_t beginSourcePublication(PopulationKey                              parent,
                                                       std::vector<SourceViewRequest>&            outRequests,
                                                       std::shared_ptr<const ParticipationTable>& outParticipation);
    ovspatialquery_api_status_t validateSourceViewUpdate(
        const std::vector<SourceViewRequest>&            requests,
        const std::vector<PreparedSourceViewMembership>& memberships) const;
    void                        publishSourceViewUpdate(const std::vector<SourceViewRequest>&            requests,
                                                        const std::vector<PreparedSourceViewMembership>& memberships);
    ovspatialquery_api_status_t finishSourceViewUpdate(const std::vector<SourceViewRequest>& requests);
    ParticipationOutcome        replaceParticipationTable(ovspatialquery_population_handle_t         population,
                                                          std::shared_ptr<const ParticipationTable>  table,
                                                          uint64_t                                   maximumResidentByteCount,
                                                          std::shared_ptr<const ParticipationTable>& outRetired);

    // Batch plans and pair reporters, by a handle whose top byte names the
    // kind; the caller casts the state back to the kind it checked.
    void                  insertAdapter(uint64_t handle, std::shared_ptr<void> adapter);
    std::shared_ptr<void> findAdapter(uint64_t handle) const;
    void                  eraseAdapter(uint64_t handle);
    size_t                adapterCount() const;

  private:
    ovspatialquery_api_status_t insertCell(std::unique_ptr<HandleCell> cell, uint64_t& outHandle);

    mutable std::mutex                                        mMutex;
    std::unordered_map<uint64_t, std::unique_ptr<HandleCell>> mCells;
    std::unordered_map<uint64_t, std::shared_ptr<void>>       mAdapters;
};

} // namespace detail
} // namespace ovspatialquery

#endif
