/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_handle_directory.h"
#include "ovspatialquery_instance_source.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <new>
#include <utility>

namespace
{

std::atomic<uint64_t> gNextHandle{1u};

// Population and view handles share one sequence; their kind bits keep them apart.
ovspatialquery_api_status_t allocateHandle(uint64_t kind, uint64_t& outHandle)
{
    const uint64_t value = gNextHandle.fetch_add(1u, std::memory_order_relaxed);
    if (value >= ovspatialquery::detail::kHandlePayloadLimit)
    {
        return OVSPATIALQUERY_API_INTERNAL;
    }
    outHandle = kind | value;
    return OVSPATIALQUERY_API_SUCCESS;
}

} // namespace

namespace ovspatialquery
{
namespace detail
{

ovspatialquery_api_status_t HandleDirectory::createPopulation(PopulationKey                       population,
                                                              ovspatialquery_population_handle_t& outHandle,
                                                              PopulationSourceBinding             source)
{
    outHandle = OVSPATIALQUERY_INVALID_POPULATION;
    try
    {
        std::unique_ptr<HandleCell> cell(new HandleCell());
        cell->population.key    = population;
        cell->population.source = std::move(source);
        return insertCell(std::move(cell), outHandle.raw_id);
    }
    catch (const std::bad_alloc&)
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    catch (...)
    {
        return OVSPATIALQUERY_API_INTERNAL;
    }
}

ovspatialquery_api_status_t HandleDirectory::createView(PopulationKey parent, std::vector<uint64_t> explicitIds,
                                                        ovspatialquery_view_handle_t& outHandle)
{
    outHandle = OVSPATIALQUERY_INVALID_VIEW;
    try
    {
        std::shared_ptr<ViewState> view = std::make_shared<ViewState>();
        view->parent                    = parent;
        std::sort(explicitIds.begin(), explicitIds.end());
        view->selectedIds = std::make_shared<const std::vector<uint64_t>>(std::move(explicitIds));
        std::unique_ptr<HandleCell> cell(new HandleCell());
        cell->kind = HandleKind::view;
        cell->view = std::move(view);
        return insertCell(std::move(cell), outHandle.raw_id);
    }
    catch (const std::bad_alloc&)
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    catch (...)
    {
        return OVSPATIALQUERY_API_INTERNAL;
    }
}

// Its callers catch what the cell's allocations throw. Handles never repeat, so
// the insertion cannot find the handle already present.
ovspatialquery_api_status_t HandleDirectory::insertCell(std::unique_ptr<HandleCell> cell, uint64_t& outHandle)
{
    uint64_t                          handle = 0u;
    const ovspatialquery_api_status_t status =
        allocateHandle(cell->kind == HandleKind::view ? kViewHandleKind : kPopulationHandleKind, handle);
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        return status;
    }
    if (cell->view != nullptr)
    {
        cell->view->handle = ovspatialquery_view_handle_t{handle};
    }
    std::lock_guard<std::mutex> lock(mMutex);
    mCells.emplace(handle, std::move(cell));
    outHandle = handle;
    return OVSPATIALQUERY_API_SUCCESS;
}

// Its callers catch what the map node's allocation throws.
void HandleDirectory::insertAdapter(uint64_t handle, std::shared_ptr<void> adapter)
{
    std::lock_guard<std::mutex> lock(mMutex);
    mAdapters.emplace(handle, std::move(adapter));
}

std::shared_ptr<void> HandleDirectory::findAdapter(uint64_t handle) const
{
    std::lock_guard<std::mutex> lock(mMutex);
    const auto                  found = mAdapters.find(handle);
    return found != mAdapters.end() ? found->second : nullptr;
}

// The adapter is destroyed after the lock is released: its view retention
// finishes the view reference through this directory.
void HandleDirectory::eraseAdapter(uint64_t handle)
{
    std::shared_ptr<void>       erased;
    std::lock_guard<std::mutex> lock(mMutex);
    const auto                  found = mAdapters.find(handle);
    if (found != mAdapters.end())
    {
        erased = std::move(found->second);
        mAdapters.erase(found);
    }
}

size_t HandleDirectory::adapterCount() const
{
    std::lock_guard<std::mutex> lock(mMutex);
    return mAdapters.size();
}

ovspatialquery_api_status_t HandleDirectory::resolveView(ovspatialquery_view_handle_t handle,
                                                         std::shared_ptr<ViewState>&  outView) const
{
    if (handle == OVSPATIALQUERY_INVALID_VIEW)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    std::lock_guard<std::mutex>                                                     lock(mMutex);
    const std::unordered_map<uint64_t, std::unique_ptr<HandleCell>>::const_iterator found = mCells.find(handle.raw_id);
    if (found == mCells.end() || found->second->kind != HandleKind::view)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    outView = found->second->view;
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t HandleDirectory::retainView(ovspatialquery_view_handle_t handle, bool requireReady,
                                                        std::shared_ptr<ViewState>& outView)
{
    std::lock_guard<std::mutex> lock(mMutex);
    const auto                  found = mCells.find(handle.raw_id);
    if (found == mCells.end() || found->second->kind != HandleKind::view)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    if (requireReady && !found->second->view->ready)
    {
        return OVSPATIALQUERY_API_NOT_READY;
    }
    ++found->second->view->planReferenceCount;
    outView = found->second->view;
    return OVSPATIALQUERY_API_SUCCESS;
}

void HandleDirectory::finishViewReference(const std::shared_ptr<ViewState>& view)
{
    std::lock_guard<std::mutex> lock(mMutex);
    if (view != nullptr && view->planReferenceCount != 0u)
    {
        --view->planReferenceCount;
    }
}

ovspatialquery_api_status_t HandleDirectory::invalidateView(ovspatialquery_view_handle_t handle)
{
    if (handle == OVSPATIALQUERY_INVALID_VIEW)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    std::lock_guard<std::mutex>                                               lock(mMutex);
    const std::unordered_map<uint64_t, std::unique_ptr<HandleCell>>::iterator found = mCells.find(handle.raw_id);
    if (found == mCells.end() || found->second->kind != HandleKind::view)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    if (found->second->view->pendingUpdateCount != 0u || found->second->view->planReferenceCount != 0u)
    {
        return OVSPATIALQUERY_API_BUSY;
    }
    mCells.erase(found);
    return OVSPATIALQUERY_API_SUCCESS;
}

bool HandleDirectory::hasLiveView(PopulationKey parent) const
{
    std::lock_guard<std::mutex> lock(mMutex);
    for (const std::pair<const uint64_t, std::unique_ptr<HandleCell>>& entry : mCells)
    {
        const HandleCell& cell = *entry.second;
        if (cell.kind == HandleKind::view && cell.view->parent == parent)
        {
            return true;
        }
    }
    return false;
}

ovspatialquery_api_status_t HandleDirectory::createSourceView(PopulationKey parent, std::shared_ptr<void> sourceState,
                                                              ovspatialquery_view_handle_t& outHandle)
{
    outHandle = OVSPATIALQUERY_INVALID_VIEW;
    if (sourceState == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    try
    {
        std::shared_ptr<ViewState> view = std::make_shared<ViewState>();
        view->parent                    = parent;
        view->selectedIds               = std::make_shared<const std::vector<uint64_t>>();
        view->ready                     = false;
        view->automatic                 = true;
        view->sourceState               = std::move(sourceState);
        std::unique_ptr<HandleCell> cell(new HandleCell());
        cell->kind = HandleKind::view;
        cell->view = std::move(view);
        return insertCell(std::move(cell), outHandle.raw_id);
    }
    catch (const std::bad_alloc&)
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    catch (...)
    {
        return OVSPATIALQUERY_API_INTERNAL;
    }
}

ovspatialquery_api_status_t HandleDirectory::beginSourcePublication(
    PopulationKey parent, std::vector<SourceViewRequest>& outRequests,
    std::shared_ptr<const ParticipationTable>& outParticipation)
{
    try
    {
        std::lock_guard<std::mutex>               lock(mMutex);
        std::vector<SourceViewRequest>            requests;
        std::shared_ptr<const ParticipationTable> participation;
        for (const std::pair<const uint64_t, std::unique_ptr<HandleCell>>& entry : mCells)
        {
            const HandleCell& cell = *entry.second;
            if (cell.kind == HandleKind::view && cell.view->automatic && cell.view->parent == parent)
            {
                requests.push_back({ovspatialquery_view_handle_t{entry.first}, cell.view->sourceState});
            }
            if (cell.kind == HandleKind::population && cell.population.key == parent)
            {
                participation = cell.population.participation;
            }
        }
        for (const SourceViewRequest& request : requests)
        {
            HandleCell& cell = *mCells.find(request.view.raw_id)->second;
            ++cell.view->pendingUpdateCount;
        }
        outRequests      = std::move(requests);
        outParticipation = std::move(participation);
        return OVSPATIALQUERY_API_SUCCESS;
    }
    catch (const std::bad_alloc&)
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    catch (...)
    {
        return OVSPATIALQUERY_API_INTERNAL;
    }
}

ovspatialquery_api_status_t HandleDirectory::validateSourceViewUpdate(
    const std::vector<SourceViewRequest>& requests, const std::vector<PreparedSourceViewMembership>& memberships) const
{
    std::lock_guard<std::mutex> lock(mMutex);
    if (requests.size() != memberships.size())
    {
        return OVSPATIALQUERY_API_INTERNAL;
    }
    for (size_t index = 0u; index < requests.size(); ++index)
    {
        const SourceViewRequest&                                                        request    = requests[index];
        const PreparedSourceViewMembership&                                             membership = memberships[index];
        const std::unordered_map<uint64_t, std::unique_ptr<HandleCell>>::const_iterator found =
            mCells.find(request.view.raw_id);
        if (found == mCells.end() || found->second->kind != HandleKind::view || !found->second->view->automatic ||
            found->second->view->pendingUpdateCount == 0u || found->second->view->sourceState != request.sourceState ||
            membership.view != request.view || membership.selectedIds == nullptr)
        {
            return OVSPATIALQUERY_API_INTERNAL;
        }
    }
    return OVSPATIALQUERY_API_SUCCESS;
}

void HandleDirectory::publishSourceViewUpdate(const std::vector<SourceViewRequest>&            requests,
                                              const std::vector<PreparedSourceViewMembership>& memberships)
{
    std::lock_guard<std::mutex> lock(mMutex);
    for (size_t index = 0u; index < requests.size(); ++index)
    {
        HandleCell& cell       = *mCells.find(requests[index].view.raw_id)->second;
        cell.view->selectedIds = memberships[index].selectedIds;
        cell.view->ready       = true;
    }
}

ovspatialquery_api_status_t HandleDirectory::finishSourceViewUpdate(const std::vector<SourceViewRequest>& requests)
{
    std::lock_guard<std::mutex> lock(mMutex);
    for (const SourceViewRequest& request : requests)
    {
        const std::unordered_map<uint64_t, std::unique_ptr<HandleCell>>::iterator found =
            mCells.find(request.view.raw_id);
        if (found == mCells.end() || found->second->kind != HandleKind::view || !found->second->view->automatic ||
            found->second->view->pendingUpdateCount == 0u || found->second->view->sourceState != request.sourceState)
        {
            return OVSPATIALQUERY_API_INTERNAL;
        }
    }
    for (const SourceViewRequest& request : requests)
    {
        --mCells.find(request.view.raw_id)->second->view->pendingUpdateCount;
    }
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t HandleDirectory::resolvePopulationSource(ovspatialquery_population_handle_t handle,
                                                                     PopulationKey&                     outPopulation,
                                                                     PopulationSourceBinding&           outSource) const
{
    return resolvePopulation(handle, outPopulation, &outSource);
}

ovspatialquery_api_status_t HandleDirectory::resolvePopulation(ovspatialquery_population_handle_t handle,
                                                               PopulationKey&                     outPopulation,
                                                               PopulationSourceBinding*           outSource) const
{
    if (handle == OVSPATIALQUERY_INVALID_POPULATION)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    std::lock_guard<std::mutex> lock(mMutex);
    const auto                  found = mCells.find(handle.raw_id);
    if (found == mCells.end() || found->second->kind != HandleKind::population)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    outPopulation = found->second->population.key;
    if (outSource != nullptr)
    {
        *outSource = found->second->population.source;
    }
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t HandleDirectory::invalidatePopulation(
    ovspatialquery_population_handle_t handle, PopulationSourceBinding& outRetiredSource,
    std::shared_ptr<const ParticipationTable>* outRetiredParticipation)
{
    outRetiredSource = {};
    if (handle == OVSPATIALQUERY_INVALID_POPULATION)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    std::lock_guard<std::mutex>                                               lock(mMutex);
    const std::unordered_map<uint64_t, std::unique_ptr<HandleCell>>::iterator found = mCells.find(handle.raw_id);
    if (found == mCells.end() || found->second->kind != HandleKind::population)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    if (outRetiredParticipation != nullptr)
    {
        *outRetiredParticipation = std::move(found->second->population.participation);
    }
    outRetiredSource = std::move(found->second->population.source);
    mCells.erase(found);
    return OVSPATIALQUERY_API_SUCCESS;
}

ParticipationOutcome HandleDirectory::replaceParticipationTable(ovspatialquery_population_handle_t        population,
                                                                std::shared_ptr<const ParticipationTable> table,
                                                                uint64_t maximumResidentByteCount,
                                                                std::shared_ptr<const ParticipationTable>& outRetired)
{
    outRetired.reset();
    try
    {
        std::lock_guard<std::mutex>                                               lock(mMutex);
        const std::unordered_map<uint64_t, std::unique_ptr<HandleCell>>::iterator populationCell =
            mCells.find(population.raw_id);
        if (populationCell == mCells.end() || populationCell->second->kind != HandleKind::population)
        {
            return ParticipationOutcome::invalidHandle;
        }
        if (populationCell->second->population.source.kind != OVSPATIALQUERY_SOURCE_OVSTAGE)
        {
            return ParticipationOutcome::invalidTargetSource;
        }
        if (table != nullptr && table->residentByteCount() > maximumResidentByteCount)
        {
            return ParticipationOutcome::hostBudget;
        }
        outRetired                                       = std::move(populationCell->second->population.participation);
        populationCell->second->population.participation = std::move(table);
        return ParticipationOutcome::success;
    }
    catch (...)
    {
        return ParticipationOutcome::internalFailure;
    }
}

} // namespace detail
} // namespace ovspatialquery
