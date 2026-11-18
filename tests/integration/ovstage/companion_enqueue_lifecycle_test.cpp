/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <ovspatialquery/ovspatialquery_ovstage.h>

#include <ovstage/ovstage.h>
#include <ovx/path_dictionary/path_dictionary.h>

#include "../../support/native/ovstage_companion_fixtures.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace
{

ovstage_instance_t* gRealStage                = nullptr;
bool                gInjectWaitTimeout        = false;
bool                gInjectWaitInternal       = false;
bool                gInjectReadReleaseFull    = false;
bool                gInjectQueryReleaseFull   = false;
bool                gInjectOrdinalReleaseFull = false;
uint32_t            gWaitTimeoutCount         = 0u;
uint32_t            gWaitInternalCount        = 0u;
uint32_t            gReadReleaseCount         = 0u;
uint32_t            gQueryReleaseCount        = 0u;
uint32_t            gOrdinalReleaseCount      = 0u;

enum class GroupMutation
{
    none,
    wrongAttribute,
    wrongDevice,
    wrongDtype,
    wrongLanes,
    wrongRank,
    wrongSemantic,
    wrongForm,
    zeroFixedCount,
    brokenCorrelation,
    duplicateToken,
    nonfiniteBounds,
    invertedBounds
};

GroupMutation         gGroupMutation = GroupMutation::none;
ovstage_read_group_t  gOriginalGroup{};
bool                  gHasOriginalGroup = false;
ovstage_read_column_t gMutatedColumn{};
DLTensor              gMutatedTensor{};
uint32_t              gDuplicatePrimRows[2] = {0u, 0u};
std::vector<double>   gMutatedBounds;
bool                  gObservedCopiedNullAndEmpty = false;
ovx_token_t           gBoundsAttribute            = OVX_INVALID_TOKEN;
ovx_primpath_t        gMalformedPath              = OVX_INVALID_PRIMPATH;
uint32_t              gMalformedLogicalRow        = 0u;

ovstage_api_status_t injectedWait(ovstage_data_plane_context_t* context, ovstage_op_id_t operation,
                                  ovstage_timeout_ns_t timeout, ovstage_op_wait_result_t* result)
{
    if (gInjectWaitInternal)
    {
        gInjectWaitInternal = false;
        ++gWaitInternalCount;
        *result = {};
        return OVSTAGE_ERROR_INTERNAL;
    }
    if (gInjectWaitTimeout)
    {
        gInjectWaitTimeout = false;
        ++gWaitTimeoutCount;
        *result = {};
        return OVSTAGE_ERROR_TIMEOUT;
    }
    return stageVtable(gRealStage)->wait_op(context, operation, timeout, result);
}

ovstage_enqueue_result_t injectedReleaseRead(ovstage_data_plane_context_t* context, ovstage_read_handle_t handle)
{
    ++gReadReleaseCount;
    if (gInjectReadReleaseFull)
    {
        gInjectReadReleaseFull = false;
        return {OVSTAGE_ERROR_QUEUE_FULL, OVSTAGE_INVALID_OP_ID};
    }
    return stageVtable(gRealStage)->release_read(context, handle);
}

ovstage_enqueue_result_t injectedReleaseQuery(ovstage_data_plane_context_t* context, ovstage_query_handle_t handle)
{
    ++gQueryReleaseCount;
    if (gInjectQueryReleaseFull)
    {
        gInjectQueryReleaseFull = false;
        return {OVSTAGE_ERROR_QUEUE_FULL, OVSTAGE_INVALID_OP_ID};
    }
    return stageVtable(gRealStage)->release_query(context, handle);
}

ovstage_enqueue_result_t injectedReleaseOrdinalQuery(ovstage_data_plane_context_t*  context,
                                                     ovstage_ordinal_query_handle_t handle)
{
    ++gOrdinalReleaseCount;
    if (gInjectOrdinalReleaseFull)
    {
        gInjectOrdinalReleaseFull = false;
        return {OVSTAGE_ERROR_QUEUE_FULL, OVSTAGE_INVALID_OP_ID};
    }
    return stageVtable(gRealStage)->release_ordinal_query(context, handle);
}

ovstage_api_status_t injectedFetchReadNext(ovstage_data_plane_context_t* context, ovstage_read_handle_t handle,
                                           ovstage_timeout_ns_t timeout, ovstage_read_group_t* groups, size_t capacity,
                                           size_t* outCount)
{
    const ovstage_api_status_t status =
        stageVtable(gRealStage)->fetch_read_next(context, handle, timeout, groups, capacity, outCount);
    if (status != OVSTAGE_OK || gGroupMutation == GroupMutation::none || *outCount != 1u ||
        groups[0].column_count != 1u)
    {
        return status;
    }
    ovstage_read_group_t* group = &groups[0];
    if (gGroupMutation == GroupMutation::nonfiniteBounds || gGroupMutation == GroupMutation::invertedBounds)
    {
        if (group->columns[0].attribute != gBoundsAttribute)
            return status;
        auto* dictionary = ovstage_get_path_dictionary(gRealStage);
        bool  selected   = false;
        for (uint32_t row = 0u; row < group->prims.count; ++row)
        {
            const uint32_t index =
                group->prims.index_map == nullptr ? group->prims.offset + row : group->prims.index_map[row];
            ovx_primpath_t path          = OVX_INVALID_PRIMPATH;
            size_t         resolvedCount = 0u;
            if (dictionaryCallSucceeded(
                    dictionary, dictionary->vtable->get_paths_from_path_list(dictionary->context, group->prims.list,
                                                                             index, 1u, &path, &resolvedCount)) &&
                resolvedCount == 1u && path == gMalformedPath)
            {
                gMalformedLogicalRow = row;
                selected             = true;
                break;
            }
        }
        if (!selected)
            return status;
    }
    gOriginalGroup                = *group;
    gHasOriginalGroup             = true;
    gMutatedColumn                = group->columns[0];
    group->columns                = &gMutatedColumn;
    ovstage_read_column_t* column = &gMutatedColumn;
    gMutatedTensor                = column->data.tensors[0];
    switch (gGroupMutation)
    {
    case GroupMutation::wrongAttribute:
        column->attribute = makeHandle<ovx_token_t>(handleValue(column->attribute) + 1u);
        break;
    case GroupMutation::wrongDevice:
        gMutatedTensor.device.device_type = kDLCUDA;
        column->data.tensors              = &gMutatedTensor;
        break;
    case GroupMutation::wrongDtype:
        gMutatedTensor.dtype.code = kDLInt;
        column->data.tensors      = &gMutatedTensor;
        break;
    case GroupMutation::wrongLanes:
        gMutatedTensor.dtype.lanes = 3u;
        column->data.tensors       = &gMutatedTensor;
        break;
    case GroupMutation::wrongRank:
        gMutatedTensor.ndim  = 2;
        column->data.tensors = &gMutatedTensor;
        break;
    case GroupMutation::wrongSemantic: column->semantic = OVSTAGE_SEMANTIC_POINT; break;
    case GroupMutation::wrongForm: column->is_array = true; break;
    case GroupMutation::zeroFixedCount: column->data.count = 0u; break;
    case GroupMutation::brokenCorrelation: column->data.count = group->prims.count + 1u; break;
    case GroupMutation::duplicateToken:
        group->prims.offset    = 0u;
        group->prims.index_map = gDuplicatePrimRows;
        break;
    case GroupMutation::nonfiniteBounds:
    case GroupMutation::invertedBounds: {
        // The filtered group can select rows from a larger physical tensor.
        gMutatedBounds.resize(static_cast<size_t>(gMutatedTensor.shape[0]) * 6u);
        std::memcpy(gMutatedBounds.data(), gMutatedTensor.data, gMutatedBounds.size() * sizeof(double));
        const size_t firstBounds =
            (column->data.index_map == nullptr ? static_cast<size_t>(gMalformedLogicalRow)
                                               : static_cast<size_t>(column->data.index_map[gMalformedLogicalRow])) *
            6u;
        if (gGroupMutation == GroupMutation::nonfiniteBounds)
        {
            gMutatedBounds[firstBounds] = std::numeric_limits<double>::infinity();
        }
        else
        {
            gMutatedBounds[firstBounds] = gMutatedBounds[firstBounds + 3u] + 1.0;
        }
        gMutatedTensor.data  = gMutatedBounds.data();
        column->data.tensors = &gMutatedTensor;
        break;
    }
    case GroupMutation::none: break;
    }
    return status;
}

ovstage_api_status_t injectedReleaseGroups(ovstage_data_plane_context_t* context, const ovstage_read_group_t* groups,
                                           size_t count)
{
    const ovstage_read_group_t* released = gHasOriginalGroup && count == 1u ? &gOriginalGroup : groups;
    const ovstage_api_status_t  status   = stageVtable(gRealStage)->release_groups(context, released, count);
    gHasOriginalGroup                    = false;
    return status;
}

ovstage_enqueue_result_t inspectCopiedFilterThenReject(ovstage_data_plane_context_t*, const ovstage_filter_t* filter,
                                                       const ovx_token_t*, size_t, ovstage_query_handle_t* outHandle)
{
    *outHandle = OVSTAGE_INVALID_QUERY_HANDLE;
    gObservedCopiedNullAndEmpty =
        filter != nullptr && filter->count == 1u && filter->predicates != nullptr &&
        filter->predicates[0].value_count == 2u && filter->predicates[0].values != nullptr &&
        filter->predicates[0].values[0].ptr == nullptr && filter->predicates[0].values[0].length == 0u &&
        filter->predicates[0].values[1].ptr != nullptr && filter->predicates[0].values[1].length == 0u;
    return {OVSTAGE_ERROR_NOT_SUPPORTED, OVSTAGE_INVALID_OP_ID};
}

int testFilterLogicalCopy(ovstage_instance_t* stage)
{
    gRealStage                         = stage;
    gObservedCopiedNullAndEmpty        = false;
    ovstage_data_plane_vtable_t vtable = *stageVtable(stage);
    vtable.query                       = inspectCopiedFilterThenReject;
    ovstage_instance_t wrappedStage{};
    wrappedStage.data_plane                           = {&vtable, stageContext(stage)};
    const ovspatialquery_instance_desc_t instanceDesc = commonInstanceDesc();
    ovspatialquery_instance_handle_t     instance     = OVSPATIALQUERY_INVALID_INSTANCE;
    CHECK(succeeded(ovspatialquery_create_instance(&instanceDesc, &instance)));

    char                attribute[]    = "usd-path";
    char                emptyStorage[] = "";
    ovx_string_t        values[]       = {{nullptr, 0u}, {emptyStorage, 0u}};
    ovstage_predicate_t predicate{};
    predicate.attribute.string = ovx_string_t{attribute, sizeof(attribute) - 1u};
    predicate.op               = OVSTAGE_FILTER_OP_IN;
    predicate.values           = values;
    predicate.value_count      = 2u;
    ovspatialquery_ovstage_population_desc_t populationDesc{};
    populationDesc.stage                          = &wrappedStage;
    populationDesc.filter                         = ovstage_filter_t{&predicate, 1u};
    populationDesc.common                         = aabbPopulationDesc();
    ovspatialquery_population_handle_t population = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &populationDesc, &population)));
    std::memset(attribute, 'x', sizeof(attribute));
    values[0] = ovx_string_t{"changed", sizeof("changed") - 1u};
    values[1] = ovx_string_t{nullptr, 0u};

    const ovspatialquery_enqueue_result_t operation =
        ovspatialquery_enqueue_ovstage_population_set(instance, population, 2);
    CHECK(operation.status == OVSPATIALQUERY_API_NOT_SUPPORTED);
    CHECK(operation.op_id == OVSPATIALQUERY_INVALID_OP);
    CHECK(gObservedCopiedNullAndEmpty);
    ovspatialquery_population_info_t info{};
    CHECK(succeeded(ovspatialquery_get_population_info(instance, population, &info)));
    CHECK(info.ready == 0u);
    CHECK(info.object_count == 0u && info.update_pending == 0u);
    CHECK(succeeded(ovspatialquery_destroy_population(instance, population)));
    CHECK(succeeded(ovspatialquery_destroy_instance(instance)));
    gRealStage = nullptr;
    return 0;
}

int testTransientCleanup(ovstage_instance_t* stage)
{
    gRealStage                         = stage;
    gWaitTimeoutCount                  = 0u;
    gWaitInternalCount                 = 0u;
    gReadReleaseCount                  = 0u;
    gQueryReleaseCount                 = 0u;
    gOrdinalReleaseCount               = 0u;
    ovstage_data_plane_vtable_t vtable = *stageVtable(stage);
    vtable.wait_op                     = injectedWait;
    vtable.release_read                = injectedReleaseRead;
    vtable.release_query               = injectedReleaseQuery;
    vtable.release_ordinal_query       = injectedReleaseOrdinalQuery;
    ovstage_instance_t wrappedStage{};
    wrappedStage.data_plane = {&vtable, stageContext(stage)};

    const ovspatialquery_instance_desc_t instanceDesc = commonInstanceDesc(1u);
    ovspatialquery_instance_handle_t     instance     = OVSPATIALQUERY_INVALID_INSTANCE;
    CHECK(succeeded(ovspatialquery_create_instance(&instanceDesc, &instance)));

    const ovx_string_t  missingPath{"/World/MissingForCleanup", 24u};
    ovstage_predicate_t predicates[2]{};
    predicates[0].attribute.string = ovx_string_t{"_worldExtent", sizeof("_worldExtent") - 1u};
    predicates[0].op               = OVSTAGE_FILTER_OP_HAS;
    predicates[1].attribute.string = ovx_string_t{"usd-path", sizeof("usd-path") - 1u};
    predicates[1].op               = OVSTAGE_FILTER_OP_IN;
    predicates[1].values           = &missingPath;
    predicates[1].value_count      = 1u;
    ovspatialquery_ovstage_population_desc_t populationDesc{};
    populationDesc.stage                          = &wrappedStage;
    populationDesc.filter                         = ovstage_filter_t{predicates, 2u};
    populationDesc.common                         = aabbPopulationDesc();
    ovspatialquery_population_handle_t population = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &populationDesc, &population)));

    gInjectWaitTimeout                        = true;
    gInjectWaitInternal                       = true;
    gInjectReadReleaseFull                    = true;
    gInjectQueryReleaseFull                   = true;
    gInjectOrdinalReleaseFull                 = true;
    ovspatialquery_enqueue_result_t operation = ovspatialquery_enqueue_ovstage_population_set(instance, population, 2);
    CHECK(operation.status == OVSPATIALQUERY_API_SUCCESS);
    CHECK(gWaitTimeoutCount == 1u);
    CHECK(gWaitInternalCount == 1u);
    CHECK(gReadReleaseCount != 0u);
    CHECK(gQueryReleaseCount != 0u);
    CHECK(gOrdinalReleaseCount != 0u);
    ovspatialquery_op_wait_result_t waited{};
    CHECK(succeeded(ovspatialquery_wait_op(instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited)));
    const uint32_t                        readReleases    = gReadReleaseCount;
    const uint32_t                        queryReleases   = gQueryReleaseCount;
    const uint32_t                        ordinalReleases = gOrdinalReleaseCount;
    const ovspatialquery_enqueue_result_t rejected =
        ovspatialquery_enqueue_ovstage_population_update(instance, population, 2, 3);
    CHECK(rejected.status == OVSPATIALQUERY_API_OPERATION_CAPACITY);
    CHECK(rejected.op_id == OVSPATIALQUERY_INVALID_OP);
    CHECK(gReadReleaseCount == readReleases);
    CHECK(gQueryReleaseCount == queryReleases);
    CHECK(gOrdinalReleaseCount == ordinalReleases);
    CHECK(succeeded(ovspatialquery_destroy_op(instance, &operation.op_id)));
    CHECK(succeeded(ovspatialquery_destroy_population(instance, population)));
    CHECK(succeeded(ovspatialquery_destroy_instance(instance)));
    gRealStage = nullptr;
    return 0;
}

int testQualifiedGroupRejections(ovstage_instance_t* stage)
{
    struct TestCase
    {
        GroupMutation               mutation;
        ovspatialquery_api_status_t expected;
    };
    const TestCase cases[] = {{GroupMutation::zeroFixedCount, OVSPATIALQUERY_API_NOT_SUPPORTED},
                              {GroupMutation::wrongAttribute, OVSPATIALQUERY_API_NOT_SUPPORTED},
                              {GroupMutation::wrongDevice, OVSPATIALQUERY_API_NOT_SUPPORTED},
                              {GroupMutation::wrongDtype, OVSPATIALQUERY_API_NOT_SUPPORTED},
                              {GroupMutation::wrongLanes, OVSPATIALQUERY_API_NOT_SUPPORTED},
                              {GroupMutation::wrongRank, OVSPATIALQUERY_API_NOT_SUPPORTED},
                              {GroupMutation::wrongSemantic, OVSPATIALQUERY_API_NOT_SUPPORTED},
                              {GroupMutation::wrongForm, OVSPATIALQUERY_API_NOT_SUPPORTED},
                              {GroupMutation::brokenCorrelation, OVSPATIALQUERY_API_NOT_SUPPORTED},
                              {GroupMutation::duplicateToken, OVSPATIALQUERY_API_INVALID_STATE},
                              {GroupMutation::nonfiniteBounds, OVSPATIALQUERY_API_SUCCESS},
                              {GroupMutation::invertedBounds, OVSPATIALQUERY_API_SUCCESS}};
    for (const TestCase& test : cases)
    {
        gRealStage                         = stage;
        gGroupMutation                     = test.mutation;
        gHasOriginalGroup                  = false;
        ovstage_data_plane_vtable_t vtable = *stageVtable(stage);
        vtable.fetch_read_next             = injectedFetchReadNext;
        vtable.release_groups              = injectedReleaseGroups;
        ovstage_instance_t wrappedStage{};
        wrappedStage.data_plane                           = {&vtable, stageContext(stage)};
        const ovspatialquery_instance_desc_t instanceDesc = commonInstanceDesc();
        ovspatialquery_instance_handle_t     instance     = OVSPATIALQUERY_INVALID_INSTANCE;
        CHECK(succeeded(ovspatialquery_create_instance(&instanceDesc, &instance)));

        const ovx_string_t paths[] = {{"/World/AuthoredParent/Authored", sizeof("/World/AuthoredParent/Authored") - 1u},
                                      {"/World/Synthesized", sizeof("/World/Synthesized") - 1u}};
        ovstage_predicate_t predicates[2]{};
        predicates[0].attribute.string = ovx_string_t{"_worldExtent", sizeof("_worldExtent") - 1u};
        predicates[0].op               = OVSTAGE_FILTER_OP_HAS;
        predicates[1].attribute.string = ovx_string_t{"usd-path", sizeof("usd-path") - 1u};
        predicates[1].op               = OVSTAGE_FILTER_OP_IN;
        predicates[1].values           = paths;
        predicates[1].value_count      = 2u;
        auto*          dictionary      = ovstage_get_path_dictionary(stage);
        ovx_primpath_t expectedPaths[2]{};
        CHECK(dictionaryCallSucceeded(
            dictionary, dictionary->vtable->create_paths_from_strings(dictionary->context, paths, 2u, expectedPaths)));
        gMalformedPath = expectedPaths[0];
        const ovx_string_t boundsAttribute{"_worldExtent", sizeof("_worldExtent") - 1u};
        CHECK(dictionaryCallSucceeded(dictionary, dictionary->vtable->create_tokens_from_strings(
                                                      dictionary->context, &boundsAttribute, 1u, &gBoundsAttribute)));
        ovspatialquery_ovstage_population_desc_t desc{};
        desc.stage                                    = &wrappedStage;
        desc.filter                                   = ovstage_filter_t{predicates, 2u};
        desc.common                                   = aabbPopulationDesc();
        ovspatialquery_population_handle_t population = OVSPATIALQUERY_INVALID_POPULATION;
        CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &desc, &population)));
        ovspatialquery_enqueue_result_t operation =
            ovspatialquery_enqueue_ovstage_population_set(instance, population, 2);
        CHECK(operation.status == test.expected);
        ovspatialquery_population_info_t info{};
        if (test.expected == OVSPATIALQUERY_API_SUCCESS)
        {
            ovspatialquery_op_wait_result_t waited{};
            CHECK(
                succeeded(ovspatialquery_wait_op(instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited)));
            ovspatialquery_population_update_result_t publication{};
            CHECK(succeeded(ovspatialquery_fetch_population_update(instance, operation.op_id, &publication)));
            CHECK(publication.excluded_object_count == 0u);
            CHECK(finishPopulationUpdate(instance, operation, 2, 2u, 0u));
            ovspatialquery_overlap_desc_t overlap{};
            overlap.common.target    = populationTarget(population);
            overlap.common.precision = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
            overlap.shape.kind       = OVSPATIALQUERY_OVERLAP_AABB;
            for (uint32_t axis = 0u; axis < 3u; ++axis)
            {
                overlap.shape.data.aabb.min_corner[axis] = -100.0;
                overlap.shape.data.aabb.max_corner[axis] = 100.0;
            }
            ovspatialquery_overlap_any_result_t result{};
            CHECK(succeeded(ovspatialquery_overlap_any_sync(instance, &overlap, &result)));
            CHECK(result.source_ordinal == 2 && result.has_hit == 1u);
            CHECK(succeeded(ovspatialquery_get_population_info(instance, population, &info)));
            CHECK(info.ready == 1u && info.object_count == 2u);
        }
        else
        {
            CHECK(operation.op_id == OVSPATIALQUERY_INVALID_OP);
            CHECK(succeeded(ovspatialquery_get_population_info(instance, population, &info)));
            CHECK(info.ready == 0u);
            CHECK(info.object_count == 0u);
        }
        CHECK(!gHasOriginalGroup);
        CHECK(succeeded(ovspatialquery_destroy_population(instance, population)));
        CHECK(succeeded(ovspatialquery_destroy_instance(instance)));
    }
    gGroupMutation = GroupMutation::none;
    gRealStage     = nullptr;
    return 0;
}

} // namespace

int main()
{
    ovstage_instance_t*           stage = nullptr;
    const ovstage_instance_desc_t stageDesc{"ovspatialquery-companion-test"};
    CHECK(ovstage_create_instance(&stageDesc, &stage) == OVSTAGE_OK);
    CHECK(stage != nullptr);
    path_dictionary_instance_t* dictionary = ovstage_get_path_dictionary(stage);
    CHECK(dictionary != nullptr);
    CHECK(populateBoundsFixture(stage, dictionary));
    CHECK(testFilterLogicalCopy(stage) == 0);
    CHECK(testTransientCleanup(stage) == 0);
    CHECK(testQualifiedGroupRejections(stage) == 0);
    CHECK(ovstage_destroy_instance(stage) == OVSTAGE_OK);
    return 0;
}
