/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_OVSTAGE_COMPANION_FIXTURES_H
#define OVSPATIALQUERY_OVSTAGE_COMPANION_FIXTURES_H

#include "../../../source/ovspatialquery_ovstage_abi.h"
#include "public_api_inputs.h"

#include <ovspatialquery/ovspatialquery_ovstage.h>

#include <ovstage/ovstage.h>
#include <ovx/path_dictionary/path_dictionary.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

using ovspatialquery::detail::handleValue;
using ovspatialquery::detail::makeHandle;
using ovspatialquery::detail::stageContext;
using ovspatialquery::detail::stageVtable;

/* Ovstage 0.4 populates physics data only through a description's selectors. These tests
 * populate RENDERING plus every prim's `physics` and `material:binding` properties, which the
 * companion reads for convex hulls and physics materials and which 0.3's ALL domain held, and
 * its `physxConvexHullCollision` properties, which carry a convex hull's vertex limit. */
inline ovstage_population_enqueue_result_t openUsdWithPhysics(ovstage_instance_t* stage, ovx_string_t usda,
                                                              ovstage_ordinal_t ordinal, double time)
{
    static const ovx_string_t kNamespaces[] = {
        {"physics", 7u}, {"physxConvexHullCollision", 24u}, {"material:binding", 16u}};
    ovstage_population_selector_t selector{};
    selector.prim_predicate.kind               = OVSTAGE_POPULATION_PRIM_PREDICATE_ALL;
    selector.property_predicate.kind           = OVSTAGE_POPULATION_PROPERTY_PREDICATE_IN_NAMESPACE;
    selector.property_predicate.values.strings = kNamespaces;
    selector.property_predicate.value_count    = 3u;
    ovstage_population_desc_t desc{};
    desc.domains        = OVSTAGE_POPULATION_DOMAIN_RENDERING;
    desc.selectors      = &selector;
    desc.selector_count = 1u;
    return ovstage_population_open_usd_from_string_with_desc(stage, usda, ordinal, time, &desc, 1u);
}

inline int fail(const char* message, int line)
{
    std::fprintf(stderr, "ovspatialquery ovstage API failure at line %d: %s\n", line, message);
    return 1;
}

#define CHECK(condition)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
        {                                                                                                              \
            return fail(#condition, __LINE__);                                                                         \
        }                                                                                                              \
    } while (false)

inline bool succeeded(ovspatialquery_result_t result) { return result.status == OVSPATIALQUERY_API_SUCCESS; }

inline ovspatialquery_instance_desc_t commonInstanceDesc(uint32_t operationCapacity = 4u)
{
    ovspatialquery_instance_desc_t desc{};
    desc.operation_capacity  = operationCapacity;
    desc.max_cpu_concurrency = 2u;
    desc.host_byte_budget    = UINT64_C(64) << 20u;
    return desc;
}

inline ovspatialquery_population_desc_t aabbPopulationDesc()
{
    ovspatialquery_population_desc_t desc{};
    desc.retained_ordinal_capacity = 3u;
    return desc;
}

inline bool waitStageOperation(ovstage_instance_t* stage, ovstage_enqueue_result_t operation)
{
    if (operation.status != OVSTAGE_OK || operation.op_index == OVSTAGE_INVALID_OP_ID)
    {
        return false;
    }
    ovstage_op_wait_result_t   result{};
    const ovstage_api_status_t waitStatus =
        stageVtable(stage)->wait_op(stageContext(stage), operation.op_index, OVSTAGE_TIMEOUT_INFINITE, &result);
    const ovstage_api_status_t releaseStatus = stageVtable(stage)->release_op(stageContext(stage), operation.op_index);
    return waitStatus == OVSTAGE_OK && result.error_op_id_count == 0u && releaseStatus == OVSTAGE_OK;
}

inline double absentBoundsValue(bool minimum)
{
    const double extreme = static_cast<double>(std::numeric_limits<float>::max());
    return minimum ? extreme : -extreme;
}

inline bool dictionaryCallSucceeded(path_dictionary_instance_t* dictionary, const ovx_api_result_t& result)
{
    const bool succeededResult = result.status == OVX_API_SUCCESS;
    if (result.error.ptr != nullptr)
    {
        dictionary->vtable->release_error(dictionary->context, result.error);
    }
    return succeededResult;
}

inline bool withPathQuery(ovstage_instance_t* stage, path_dictionary_instance_t* dictionary, const ovx_string_t& path,
                          ovstage_query_handle_t& query, ovx_primpath_list_t& pathList)
{
    pathList = OVX_INVALID_PRIMPATH_LIST;
    query    = OVSTAGE_INVALID_QUERY_HANDLE;
    const ovx_api_result_t listResult =
        dictionary->vtable->create_path_list_from_strings(dictionary->context, &path, 1u, &pathList);
    if (!dictionaryCallSucceeded(dictionary, listResult) || pathList == OVX_INVALID_PRIMPATH_LIST)
    {
        return false;
    }
    if (ovstage_query_from_path_list(stage, pathList, &query) != OVSTAGE_OK || query == OVSTAGE_INVALID_QUERY_HANDLE)
    {
        const ovx_api_result_t releaseResult =
            dictionary->vtable->release_path_list_reference(dictionary->context, pathList);
        (void)dictionaryCallSucceeded(dictionary, releaseResult);
        pathList = OVX_INVALID_PRIMPATH_LIST;
        return false;
    }
    return true;
}

inline bool releasePathQuery(ovstage_instance_t* stage, path_dictionary_instance_t* dictionary,
                             ovstage_query_handle_t query, ovx_primpath_list_t pathList)
{
    const bool             queryReleased = waitStageOperation(stage, ovstage_release_query(stage, query));
    const ovx_api_result_t listReleased =
        dictionary->vtable->release_path_list_reference(dictionary->context, pathList);
    return queryReleased && dictionaryCallSucceeded(dictionary, listReleased);
}

// Writes one attribute of one prim at an ordinal through the write API: count elements of
// dtype, as an array value when isArray. The caller seals the floor.
inline bool writeAttribute(ovstage_instance_t* stage, path_dictionary_instance_t* dictionary, const char* pathText,
                           size_t pathLength, const char* attributeName, ovstage_ordinal_t ordinal, void* values,
                           DLDataType dtype, size_t count = 1u,
                           ovstage_attribute_semantic_t semantic = OVSTAGE_SEMANTIC_NONE, bool isArray = false)
{
    const ovx_string_t     path{pathText, pathLength};
    ovstage_query_handle_t query    = OVSTAGE_INVALID_QUERY_HANDLE;
    ovx_primpath_list_t    pathList = OVX_INVALID_PRIMPATH_LIST;
    if (!withPathQuery(stage, dictionary, path, query, pathList))
    {
        return false;
    }
    int64_t  shape = static_cast<int64_t>(count);
    DLTensor tensor{};
    tensor.data               = values;
    tensor.device.device_type = kDLCPU;
    tensor.ndim               = 1;
    tensor.dtype              = dtype;
    tensor.shape              = &shape;
    ovstage_write_data_t data{};
    data.tensors      = &tensor;
    data.tensor_count = 1u;
    data.semantic     = semantic;
    data.is_array     = isArray;
    decltype(ovstage_predicate_t::attribute) attribute{};
    attribute.string = ovx_string_t{attributeName, std::strlen(attributeName)};
    const bool wrote = waitStageOperation(
        stage, ovstage_write_attribute(stage, query, attribute, ordinal, data, OVSTAGE_PRIM_MODE_UPSERT));
    return releasePathQuery(stage, dictionary, query, pathList) && wrote;
}

inline bool writeFloatPointArray(ovstage_instance_t* stage, path_dictionary_instance_t* dictionary,
                                 const char* pathText, size_t pathLength, ovstage_ordinal_t ordinal, float* values,
                                 size_t pointCount)
{
    return writeAttribute(stage, dictionary, pathText, pathLength, "points", ordinal, values,
                          DLDataType{kDLFloat, 32u, 3u}, pointCount, OVSTAGE_SEMANTIC_POINT, true);
}

inline bool writeBounds(ovstage_instance_t* stage, path_dictionary_instance_t* dictionary, const char* pathText,
                        size_t pathLength, ovstage_ordinal_t ordinal, double (&bounds)[6])
{
    return writeAttribute(stage, dictionary, pathText, pathLength, "_worldExtent", ordinal, bounds,
                          DLDataType{kDLFloat, 64u, 6u});
}

inline bool populateBoundsFixture(ovstage_instance_t* stage, path_dictionary_instance_t* dictionary)
{
    static const char                         usda[] = R"usda(#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
    metersPerUnit = 1
)
def Xform "World"
{
    def Xform "AuthoredParent"
    {
        matrix4d xformOp:transform = ((2, 0, 0, 0), (0, 3, 0, 0), (0, 0, 4, 0), (10, 20, 30, 1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
        def Cube "Authored"
        {
            double size = 2
            float3[] extent = [(-1, -2, -3), (1, 2, 3)]
        }
    }
    def Cube "Synthesized"
    {
        double size = 1.4
        double3 xformOp:translate = (-3, 0, 0.35)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def Cube "Added"
    {
        double size = 2
        double3 xformOp:translate = (5, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def Cube "MissingBounds"
    {
        double size = 1
        double3 xformOp:translate = (0, 0, 10)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
}
)usda";
    const ovstage_population_enqueue_result_t population =
        openUsdWithPhysics(stage, ovx_string_t{usda, sizeof(usda) - 1u}, 1u, 0.0);
    ovstage_population_op_wait_result_t populationWait{};
    if (population.status != OVSTAGE_OK ||
        ovstage_population_wait_op(stage, population.op_index, OVSTAGE_TIMEOUT_INFINITE, &populationWait) != OVSTAGE_OK)
    {
        return false;
    }
    const ovstage_write_floor_desc_t floorOne{1u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
    if (!waitStageOperation(stage, ovstage_advance_write_floor(stage, &floorOne)) ||
        !waitStageOperation(stage,
                            ovstage_compute_hierarchy(stage, OVSTAGE_HIERARCHY_COMPUTATION_MODEL_DEFAULT_CPU, 1u, 2u)))
    {
        return false;
    }
    double absentBounds[6]{
        absentBoundsValue(true),  absentBoundsValue(true),  absentBoundsValue(true),
        absentBoundsValue(false), absentBoundsValue(false), absentBoundsValue(false),
    };
    if (!writeBounds(stage, dictionary, "/World/Added", sizeof("/World/Added") - 1u, 2u, absentBounds))
    {
        return false;
    }
    const ovstage_write_floor_desc_t floorTwo{2u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
    return waitStageOperation(stage, ovstage_advance_write_floor(stage, &floorTwo));
}

inline bool finishPopulationUpdate(ovspatialquery_instance_handle_t instance, ovspatialquery_enqueue_result_t operation,
                                   ovspatialquery_ordinal_t expectedOrdinal, uint64_t objects, uint64_t excluded)
{
    if (operation.status != OVSPATIALQUERY_API_SUCCESS || operation.op_id == OVSPATIALQUERY_INVALID_OP)
    {
        return false;
    }
    ovspatialquery_op_wait_result_t waitResult{};
    if (!succeeded(ovspatialquery_wait_op(instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waitResult)) ||
        waitResult.state != OVSPATIALQUERY_OP_COMPLETED)
    {
        return false;
    }
    ovspatialquery_population_update_result_t first{};
    ovspatialquery_population_update_result_t second{};
    if (!succeeded(ovspatialquery_fetch_population_update(instance, operation.op_id, &first)) ||
        !succeeded(ovspatialquery_fetch_population_update(instance, operation.op_id, &second)) ||
        first.published_ordinal != expectedOrdinal || second.published_ordinal != expectedOrdinal ||
        first.object_count != objects || first.excluded_object_count != excluded || second.object_count != objects ||
        second.excluded_object_count != excluded)
    {
        return false;
    }
    ovspatialquery_op_status_t status{};
    if (!succeeded(ovspatialquery_query_op_status(instance, operation.op_id, &status)) ||
        status.state != OVSPATIALQUERY_OP_COMPLETED || status.terminal_status != OVSPATIALQUERY_API_SUCCESS)
    {
        return false;
    }
    if (!succeeded(ovspatialquery_destroy_op(instance, &operation.op_id)) ||
        operation.op_id != OVSPATIALQUERY_INVALID_OP)
    {
        return false;
    }
    return true;
}

inline bool readHostBytes(ovspatialquery_instance_handle_t instance, uint64_t& outBytes)
{
    outBytes = 0u;
    return succeeded(ovspatialquery_get_instance_memory_usage(instance, DLDevice{kDLCPU, 0}, &outBytes));
}

#endif
