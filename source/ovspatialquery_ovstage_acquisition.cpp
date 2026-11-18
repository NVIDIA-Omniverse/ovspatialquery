/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_backend.h"
#include "ovspatialquery_device_box_population.h"
#include "ovspatialquery_ovstage_device_publication.h"
#include "ovspatialquery_ovstage_filter.h"
#include <ovspatialquery_instance_source.h>

#include <ovstage/ovstage_api/ovstage_api.h>
#include <ovx/path_dictionary/path_dictionary.h>

#include <ovspatialquery/ovspatialquery.h>
#include <ovspatialquery/ovspatialquery_ovstage.h>

#include <algorithm>
#include <iterator>
#include <memory>
#include <new>
#include <string>
#include <utility>

/* current API installed Ovstage adapter. */
namespace
{

using ovspatialquery::detail::OvstageLayout;
using ovspatialquery::detail::stageContext;
using ovspatialquery::detail::stageVtable;

constexpr const char* kMalformedFilter =
    "a filter predicate names no attribute, or has a null value array or a null value with a nonzero length";
constexpr const char* kMalformedScope =
    "a scope list's pointer and count disagree, or an entry is zero or not a path of the stage's path dictionary";

ovspatialquery_api_status_t apiValidateStage(ovstage_instance_t* stage, path_dictionary_instance_t*& outDictionary,
                                             std::string& outDiagnostic)
{
    outDictionary = nullptr;
    if (stage == nullptr || stageVtable(stage) == nullptr || stageContext(stage) == nullptr ||
        stageVtable(stage)->get_version == nullptr || stageVtable(stage)->get_path_dictionary == nullptr ||
        stageVtable(stage)->wait_op == nullptr || stageVtable(stage)->release_op == nullptr ||
        stageVtable(stage)->get_attribute_write_floor == nullptr || stageVtable(stage)->fetch_ordinal == nullptr ||
        stageVtable(stage)->release_ordinal_query == nullptr || stageVtable(stage)->query == nullptr ||
        stageVtable(stage)->fetch_query_result == nullptr || stageVtable(stage)->release_query_result == nullptr ||
        stageVtable(stage)->release_query == nullptr || stageVtable(stage)->read_attributes == nullptr ||
        stageVtable(stage)->fetch_read_next == nullptr || stageVtable(stage)->release_groups == nullptr ||
        stageVtable(stage)->release_read == nullptr)
    {
        outDiagnostic = stage == nullptr ? "the population's stage is null"
                                         : "the stage's function table lacks a slot ovstage populations call";
        return stage == nullptr ? OVSPATIALQUERY_API_INVALID_ARGUMENT : OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    if (ovspatialquery::detail::ovstageLayout(stage) == OvstageLayout::unsupported)
    {
        uint32_t major = 0u, minor = 0u, patch = 0u;
        stageVtable(stage)->get_version(stageContext(stage), &major, &minor, &patch);
        outDiagnostic = "the stage is ovstage " + std::to_string(major) + "." + std::to_string(minor) + "." +
                        std::to_string(patch) + "; ovstage populations accept ovstage 0.3 and 0.4";
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    path_dictionary_instance_t* dictionary = stageVtable(stage)->get_path_dictionary(stageContext(stage));
    if (dictionary == nullptr || dictionary->vtable == nullptr || dictionary->context == nullptr ||
        dictionary->vtable->release_error == nullptr || dictionary->vtable->create_tokens_from_strings == nullptr ||
        dictionary->vtable->get_tokens_from_paths == nullptr ||
        dictionary->vtable->get_num_paths_from_path_list == nullptr ||
        dictionary->vtable->get_paths_from_path_list == nullptr)
    {
        outDiagnostic = "the stage's path dictionary lacks a slot ovstage populations call";
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    outDictionary = dictionary;
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t apiParticipationStatus(ovspatialquery::detail::ParticipationOutcome outcome)
{
    using Outcome = ovspatialquery::detail::ParticipationOutcome;
    switch (outcome)
    {
    case Outcome::success: return OVSPATIALQUERY_API_SUCCESS;
    case Outcome::invalidArgument:
    case Outcome::invalidTargetSource: return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    case Outcome::invalidHandle: return OVSPATIALQUERY_API_INVALID_HANDLE;
    case Outcome::busy: return OVSPATIALQUERY_API_BUSY;
    case Outcome::hostBudget: return OVSPATIALQUERY_API_HOST_BUDGET;
    case Outcome::outOfMemory: return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    case Outcome::internalFailure: return OVSPATIALQUERY_API_INTERNAL;
    }
    return OVSPATIALQUERY_API_INTERNAL;
}

} // namespace

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_create_ovstage_population(
    ovspatialquery_instance_handle_t instance, const ovspatialquery_ovstage_population_desc_t* desc,
    ovspatialquery_population_handle_t* outPopulation)
{
    if (outPopulation == nullptr)
    {
        return ovspatialquery::detail::apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT, nullptr);
    }
    *outPopulation = OVSPATIALQUERY_INVALID_POPULATION;
    if (desc == nullptr)
    {
        return ovspatialquery::detail::apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT,
                                                 "the population descriptor is null");
    }
    std::string                                          diagnostic;
    path_dictionary_instance_t*                          dictionary = nullptr;
    ovspatialquery_api_status_t                          status = apiValidateStage(desc->stage, dictionary, diagnostic);
    std::shared_ptr<ovspatialquery::detail::OwnedFilter> filter;
    ovspatialquery::detail::SourceGeometryPreparationContext preparation{};
    if (status == OVSPATIALQUERY_API_SUCCESS)
    {
        status = ovspatialquery::detail::getSourceGeometryPreparationContext(instance, preparation);
    }
    if (status == OVSPATIALQUERY_API_SUCCESS)
    {
        status = ovspatialquery::detail::copyFilter(desc->stage, desc->filter, dictionary, filter, preparation);
        if (status == OVSPATIALQUERY_API_INVALID_ARGUMENT)
        {
            diagnostic = kMalformedFilter;
        }
    }
    if (status == OVSPATIALQUERY_API_SUCCESS)
    {
        status = ovspatialquery::detail::PathScope::create(dictionary, desc->scope, preparation.convexHullBudget,
                                                           filter->scope);
        if (status == OVSPATIALQUERY_API_INVALID_ARGUMENT)
        {
            diagnostic = kMalformedScope;
        }
    }
    uint64_t filterCharge = 0u;
    if (status == OVSPATIALQUERY_API_SUCCESS && !ovspatialquery::detail::estimateFilterCharge(*filter, filterCharge))
    {
        status = OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    std::shared_ptr<ovspatialquery::detail::DevicePreparedState> deviceState;
    if (status == OVSPATIALQUERY_API_SUCCESS)
    {
        try
        {
            deviceState = std::make_shared<ovspatialquery::detail::DevicePreparedState>();
        }
        catch (const std::bad_alloc&)
        {
            status = OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
    }
    if (status == OVSPATIALQUERY_API_SUCCESS)
    {
        // Host publications stay one at a time; device ones may chain (ADR-0047).
        status = ovspatialquery::detail::createEmptyPopulationForSource(
            instance,
            ovspatialquery::detail::PopulationSourceBinding{
                OVSPATIALQUERY_SOURCE_OVSTAGE,
                desc->stage,
                dictionary,
                std::move(filter),
                ovspatialquery::detail::acquireTypedSnapshot,
                ovspatialquery::detail::PopulationPreparationStrategy::typedFiveFamily,
                std::move(deviceState),
            },
            desc->common.retained_ordinal_capacity, filterCharge, *outPopulation,
            desc->common.pending_publication_capacity);
    }
    return ovspatialquery::detail::apiResult(
        status, status == OVSPATIALQUERY_API_SUCCESS || diagnostic.empty() ? nullptr : diagnostic.c_str());
}

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_create_ovstage_view(
    ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population,
    const ovspatialquery_ovstage_view_desc_t* desc, ovspatialquery_view_handle_t* outView)
{
    if (outView == nullptr)
    {
        return ovspatialquery::detail::apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT, nullptr);
    }
    *outView = OVSPATIALQUERY_INVALID_VIEW;
    if (desc == nullptr)
    {
        return ovspatialquery::detail::apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT, nullptr);
    }
    ovspatialquery::detail::PopulationSourceBinding source;
    ovspatialquery_api_status_t                     status =
        ovspatialquery::detail::getPopulationSource(instance, population, OVSPATIALQUERY_SOURCE_OVSTAGE, source);
    std::shared_ptr<ovspatialquery::detail::OwnedFilter> filter;
    const char*                                          diagnostic = nullptr;
    // No view selects a device-prepared version (ADR-0047).
    if (status == OVSPATIALQUERY_API_SUCCESS && source.deviceState != nullptr &&
        source.deviceState->current() != nullptr)
    {
        status     = OVSPATIALQUERY_API_NOT_SUPPORTED;
        diagnostic = "a population prepared on the device has no view";
    }
    if (status == OVSPATIALQUERY_API_SUCCESS)
    {
        status = ovspatialquery::detail::copyFilter(
            const_cast<ovstage_instance_t*>(static_cast<const ovstage_instance_t*>(source.instance)), desc->filter,
            const_cast<path_dictionary_instance_t*>(
                static_cast<const path_dictionary_instance_t*>(source.pathDictionary)),
            filter);
        diagnostic = status == OVSPATIALQUERY_API_INVALID_ARGUMENT ? kMalformedFilter : nullptr;
    }
    // The view's scope charges its cache to the instance like the population's does.
    ovspatialquery::detail::SourceGeometryPreparationContext preparation{};
    if (status == OVSPATIALQUERY_API_SUCCESS && (desc->scope.include_count != 0u || desc->scope.exclude_count != 0u))
    {
        status = ovspatialquery::detail::getSourceGeometryPreparationContext(instance, preparation);
    }
    if (status == OVSPATIALQUERY_API_SUCCESS)
    {
        status = ovspatialquery::detail::PathScope::create(
            const_cast<path_dictionary_instance_t*>(
                static_cast<const path_dictionary_instance_t*>(source.pathDictionary)),
            desc->scope, preparation.convexHullBudget, filter->scope);
        diagnostic = status == OVSPATIALQUERY_API_INVALID_ARGUMENT ? kMalformedScope : nullptr;
    }
    uint64_t filterCharge = 0u;
    if (status == OVSPATIALQUERY_API_SUCCESS && !ovspatialquery::detail::estimateFilterCharge(*filter, filterCharge))
    {
        status = OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    if (status == OVSPATIALQUERY_API_SUCCESS)
    {
        status = ovspatialquery::detail::createViewForSource(instance, OVSPATIALQUERY_SOURCE_OVSTAGE, population,
                                                             std::move(filter), filterCharge, *outView);
    }
    return ovspatialquery::detail::apiResult(status, diagnostic);
}

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_set_ovstage_participation_map(
    ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population,
    const ovspatialquery_ovstage_participation_table_t* table)
{
    if (table != nullptr &&
        ((table->count == 0u) != (table->paths == nullptr) || (table->count == 0u) != (table->bits == nullptr)))
    {
        return ovspatialquery::detail::apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT, nullptr);
    }
    try
    {
        ovspatialquery::detail::ParticipationTableDesc descriptor{};
        if (table != nullptr)
        {
            static_assert(sizeof(ovx_primpath_t) == sizeof(uint64_t));
            descriptor = {table->default_bits, reinterpret_cast<const uint64_t*>(table->paths), table->bits,
                          table->count};
        }
        return ovspatialquery::detail::apiResult(
            apiParticipationStatus(ovspatialquery::detail::installParticipationTable(
                instance, population, table == nullptr ? nullptr : &descriptor)),
            nullptr);
    }
    catch (const std::bad_alloc&)
    {
        return ovspatialquery::detail::apiResult(OVSPATIALQUERY_API_OUT_OF_MEMORY, nullptr);
    }
    catch (...)
    {
        return ovspatialquery::detail::apiResult(OVSPATIALQUERY_API_INTERNAL, nullptr);
    }
}

namespace
{

// ON_HOST with a CPU device and no stream is the host call; anything else on the host
// path is malformed, and the device choices go to the device path (ADR-0047).
ovspatialquery_enqueue_result_t enqueueOvstagePublication(ovspatialquery_instance_handle_t          instance,
                                                          ovspatialquery_population_handle_t        population,
                                                          const ovspatialquery_ordinal_t*           baseOrdinal,
                                                          ovspatialquery_ordinal_t                  targetOrdinal,
                                                          const ovspatialquery_dynamic_execution_t* execution)
{
    ovspatialquery_op_id_t      operation = OVSPATIALQUERY_INVALID_OP;
    std::string                 diagnostic;
    ovspatialquery_api_status_t status = OVSPATIALQUERY_API_SUCCESS;
    try
    {
        if (execution == nullptr || execution->preparation == OVSPATIALQUERY_PREPARE_ON_HOST)
        {
            if (execution != nullptr &&
                (execution->device.device_type != kDLCPU || execution->device.device_id != 0 ||
                 execution->cuda_sync.has_stream != 0u || execution->cuda_sync.wait_event != 0u))
            {
                status = OVSPATIALQUERY_API_INVALID_ARGUMENT;
            }
            else
            {
                status = ovspatialquery::detail::submitPopulationSnapshotForSource(
                    instance, OVSPATIALQUERY_SOURCE_OVSTAGE, population, baseOrdinal, targetOrdinal, operation,
                    &diagnostic);
            }
        }
        else if (execution->preparation == OVSPATIALQUERY_PREPARE_ON_DEVICE ||
                 execution->preparation == OVSPATIALQUERY_PREPARE_ON_DEVICE_HOST_READABLE)
        {
            status = ovspatialquery::detail::submitOvstageDevicePublication(
                instance, population, baseOrdinal, targetOrdinal, *execution, operation, diagnostic);
        }
        else
        {
            status = OVSPATIALQUERY_API_INVALID_ARGUMENT;
        }
    }
    catch (const std::bad_alloc&)
    {
        status = OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    catch (...)
    {
        status = OVSPATIALQUERY_API_INTERNAL;
    }
    return ovspatialquery::detail::sourcePublicEnqueueResult(status, operation,
                                                             diagnostic.empty() ? nullptr : diagnostic.c_str());
}

} // namespace

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t
ovspatialquery_fetch_ovstage_dropped_triangles(ovspatialquery_instance_handle_t instance, ovspatialquery_op_id_t op_id,
                                               ovspatialquery_ovstage_dropped_triangles_t* out_result)
{
    if (out_result == nullptr)
    {
        return ovspatialquery::detail::apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    DLTensor                          objects{};
    DLTensor                          counts{};
    const ovspatialquery_api_status_t status =
        ovspatialquery::detail::fetchDroppedTriangles(instance, op_id, objects, counts);
    if (status == OVSPATIALQUERY_API_SUCCESS)
    {
        *out_result = {objects, counts};
    }
    return ovspatialquery::detail::apiResult(status);
}

extern "C" OVSPATIALQUERY_API ovspatialquery_enqueue_result_t ovspatialquery_enqueue_ovstage_population_set(
    ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population,
    ovspatialquery_ordinal_t targetOrdinal)
{
    return enqueueOvstagePublication(instance, population, nullptr, targetOrdinal, nullptr);
}

extern "C" OVSPATIALQUERY_API ovspatialquery_enqueue_result_t ovspatialquery_enqueue_ovstage_population_update(
    ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population,
    ovspatialquery_ordinal_t baseOrdinal, ovspatialquery_ordinal_t targetOrdinal)
{
    return enqueueOvstagePublication(instance, population, &baseOrdinal, targetOrdinal, nullptr);
}

extern "C" OVSPATIALQUERY_API ovspatialquery_enqueue_result_t
ovspatialquery_enqueue_ovstage_population_set_with_execution(ovspatialquery_instance_handle_t          instance,
                                                             ovspatialquery_population_handle_t        population,
                                                             ovspatialquery_ordinal_t                  targetOrdinal,
                                                             const ovspatialquery_dynamic_execution_t* execution)
{
    return enqueueOvstagePublication(instance, population, nullptr, targetOrdinal, execution);
}

extern "C" OVSPATIALQUERY_API ovspatialquery_enqueue_result_t
ovspatialquery_enqueue_ovstage_population_update_with_execution(ovspatialquery_instance_handle_t          instance,
                                                                ovspatialquery_population_handle_t        population,
                                                                ovspatialquery_ordinal_t                  baseOrdinal,
                                                                ovspatialquery_ordinal_t                  targetOrdinal,
                                                                const ovspatialquery_dynamic_execution_t* execution)
{
    return enqueueOvstagePublication(instance, population, &baseOrdinal, targetOrdinal, execution);
}

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t
ovspatialquery_query_ovstage_capability(ovspatialquery_instance_handle_t instance, ovstage_instance_t* stage,
                                        DLDevice device, ovspatialquery_capability_t capability)
{
    path_dictionary_instance_t*       dictionary = nullptr;
    std::string                       diagnostic;
    const ovspatialquery_api_status_t stageStatus = apiValidateStage(stage, dictionary, diagnostic);
    if (stageStatus != OVSPATIALQUERY_API_SUCCESS)
    {
        return ovspatialquery::detail::apiResult(stageStatus, diagnostic.empty() ? nullptr : diagnostic.c_str());
    }
    const ovspatialquery_result_t answer = ovspatialquery_query_capability(instance, device, capability);
    if (answer.status != OVSPATIALQUERY_API_SUCCESS || capability != OVSPATIALQUERY_CAPABILITY_PREPARE_ON_DEVICE ||
        ovspatialquery::detail::ovstagePlacementExtension(stage, diagnostic) != nullptr)
    {
        return answer;
    }
    return ovspatialquery::detail::apiResult(OVSPATIALQUERY_API_NOT_SUPPORTED, diagnostic.c_str());
}
