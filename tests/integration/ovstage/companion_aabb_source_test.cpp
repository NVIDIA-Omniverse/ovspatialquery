/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <ovspatialquery/ovspatialquery_ovstage.h>
#include <ovspatialquery/ovspatialquery_pairs.h>

#include <ovstage/ovstage.h>
#include <ovx/path_dictionary/path_dictionary.h>

#include "../../support/native/ovstage_companion_fixtures.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

extern "C" void                                 ovspatialquery_test_ovstage_fail_next_group_materialization();
typedef struct ovspatialquery_backend_runtime_t ovspatialquery_backend_runtime_t;
extern "C" ovspatialquery_backend_runtime_t*    ovspatialquery_test_oracle_get_default_runtime();
extern "C" void     ovspatialquery_test_oracle_block_before_publish(ovspatialquery_backend_runtime_t* runtime,
                                                                    ovspatialquery_ordinal_t          ordinal);
extern "C" uint32_t ovspatialquery_test_oracle_wait_before_publish(ovspatialquery_backend_runtime_t* runtime);
extern "C" void     ovspatialquery_test_oracle_release_before_publish(ovspatialquery_backend_runtime_t* runtime);
extern "C" void     ovspatialquery_test_oracle_fail_next_build(ovspatialquery_backend_runtime_t* runtime,
                                                               ovspatialquery_api_status_t       failure);

namespace
{

bool fetchPopulationUpdateAndDestroy(ovspatialquery_instance_handle_t instance,
                                     ovspatialquery_enqueue_result_t  operation,
                                     ovspatialquery_ordinal_t expectedOrdinal, uint64_t objects, uint64_t excluded)
{
    ovspatialquery_population_update_result_t   result{};
    ovspatialquery_result_t                     fetch{OVSPATIALQUERY_API_NOT_READY};
    const std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (fetch.status == OVSPATIALQUERY_API_NOT_READY && std::chrono::steady_clock::now() < deadline)
    {
        fetch = ovspatialquery_fetch_population_update(instance, operation.op_id, &result);
        std::this_thread::yield();
    }
    return succeeded(fetch) && result.published_ordinal == expectedOrdinal && result.object_count == objects &&
           result.excluded_object_count == excluded &&
           succeeded(ovspatialquery_destroy_op(instance, &operation.op_id)) &&
           operation.op_id == OVSPATIALQUERY_INVALID_OP;
}

int testRealAabbSource(ovstage_instance_t* stage, path_dictionary_instance_t* dictionary)
{
    const ovspatialquery_instance_desc_t instanceDesc = commonInstanceDesc();
    ovspatialquery_instance_handle_t     instance     = OVSPATIALQUERY_INVALID_INSTANCE;
    CHECK(succeeded(ovspatialquery_create_instance(&instanceDesc, &instance)));

    char               extentAttribute[] = "_worldExtent";
    char               pathAttribute[]   = "usd-path";
    char               authoredPath[]    = "/World/AuthoredParent/Authored";
    char               synthesizedPath[] = "/World/Synthesized";
    char               addedPath[]       = "/World/Added";
    ovx_string_t       pathValues[]      = {{authoredPath, sizeof(authoredPath) - 1u},
                                            {synthesizedPath, sizeof(synthesizedPath) - 1u},
                                            {addedPath, sizeof(addedPath) - 1u}};
    const ovx_string_t expectedStrings[] = {
        {"/World/AuthoredParent/Authored", sizeof("/World/AuthoredParent/Authored") - 1u},
        {"/World/Synthesized", sizeof("/World/Synthesized") - 1u},
        {"/World/Added", sizeof("/World/Added") - 1u}};
    ovx_primpath_t expectedPaths[3]{};
    CHECK(dictionaryCallSucceeded(dictionary, dictionary->vtable->create_paths_from_strings(
                                                  dictionary->context, expectedStrings, 3u, expectedPaths)));

    ovstage_predicate_t predicates[2]{};
    predicates[0].attribute.string = ovx_string_t{extentAttribute, sizeof(extentAttribute) - 1u};
    predicates[0].op               = OVSTAGE_FILTER_OP_HAS;
    predicates[1].attribute.string = ovx_string_t{pathAttribute, sizeof(pathAttribute) - 1u};
    predicates[1].op               = OVSTAGE_FILTER_OP_IN;
    predicates[1].values           = pathValues;
    predicates[1].value_count      = 3u;
    ovspatialquery_ovstage_population_desc_t sourceDesc{};
    sourceDesc.stage                              = stage;
    sourceDesc.filter                             = ovstage_filter_t{predicates, 2u};
    sourceDesc.common                             = aabbPopulationDesc();
    ovspatialquery_population_handle_t population = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &sourceDesc, &population)));

    ovstage_predicate_t viewPredicate = predicates[1];
    viewPredicate.values              = &pathValues[1];
    viewPredicate.value_count         = 1u;
    ovspatialquery_ovstage_view_desc_t viewDesc{};
    viewDesc.filter                   = ovstage_filter_t{&viewPredicate, 1u};
    ovspatialquery_view_handle_t view = OVSPATIALQUERY_INVALID_VIEW;
    CHECK(succeeded(ovspatialquery_create_ovstage_view(instance, population, &viewDesc, &view)));

    ovspatialquery_overlap_desc_t viewOverlap{};
    viewOverlap.common.target    = viewTarget(view);
    viewOverlap.common.precision = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    viewOverlap.shape.kind       = OVSPATIALQUERY_OVERLAP_AABB;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        viewOverlap.shape.data.aabb.min_corner[axis] = -100.0;
        viewOverlap.shape.data.aabb.max_corner[axis] = 100.0;
    }
    ovspatialquery_overlap_any_result_t preserved{};
    preserved.source_ordinal               = 77;
    preserved.has_hit                      = 1u;
    preserved.hit.identity.object_id.value = 91u;
    CHECK(ovspatialquery_overlap_any_sync(instance, &viewOverlap, &preserved).status == OVSPATIALQUERY_API_NOT_READY);
    CHECK(preserved.source_ordinal == 77);
    CHECK(preserved.has_hit == 1u);
    CHECK(preserved.hit.identity.object_id.value == 91u);

    std::memset(extentAttribute, 'x', sizeof(extentAttribute));
    std::memset(pathAttribute, 'x', sizeof(pathAttribute));
    std::memset(authoredPath, 'x', sizeof(authoredPath));
    std::memset(synthesizedPath, 'x', sizeof(synthesizedPath));
    std::memset(addedPath, 'x', sizeof(addedPath));

    ovspatialquery_test_ovstage_fail_next_group_materialization();
    ovspatialquery_enqueue_result_t operation = ovspatialquery_enqueue_ovstage_population_set(instance, population, 2);
    CHECK(operation.status == OVSPATIALQUERY_API_OUT_OF_MEMORY);
    CHECK(operation.op_id == OVSPATIALQUERY_INVALID_OP);
    ovspatialquery_population_info_t info{};
    CHECK(succeeded(ovspatialquery_get_population_info(instance, population, &info)));
    CHECK(info.ready == 0u);
    CHECK(info.object_count == 0u);

    operation = ovspatialquery_enqueue_ovstage_population_set(instance, population, 2);
    CHECK(operation.status == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_op_wait_result_t firstPublicationWait{};
    CHECK(succeeded(
        ovspatialquery_wait_op(instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &firstPublicationWait)));
    ovspatialquery_population_update_result_t firstPublication{};
    CHECK(succeeded(ovspatialquery_fetch_population_update(instance, operation.op_id, &firstPublication)));
    CHECK(firstPublication.excluded_object_count == 1u);
    CHECK(firstPublication.excluded_objects.data != nullptr);
    CHECK(firstPublication.exclusion_reasons.data != nullptr);
    CHECK(*static_cast<const uint64_t*>(firstPublication.excluded_objects.data) == handleValue(expectedPaths[2]));
    CHECK(*static_cast<const int32_t*>(firstPublication.exclusion_reasons.data) ==
          OVSPATIALQUERY_OVSTAGE_EXCLUSION_NO_SPATIAL_SOURCE);
    CHECK(finishPopulationUpdate(instance, operation, 2, 2u, 1u));
    CHECK(succeeded(ovspatialquery_get_population_info(instance, population, &info)));
    CHECK(info.ready == 1u);
    CHECK(info.head_ordinal == 2);
    CHECK(info.object_count == 2u);

    ovspatialquery_overlap_desc_t overlap = viewOverlap;
    overlap.common.target                 = populationTarget(population);
    std::array<ovspatialquery_overlap_hit_t, 4> hits{};
    ovspatialquery_overlap_hit_buffer_t         hitBuffer{hits.data(), hits.size()};
    ovspatialquery_multiple_result_t            hitResult{};
    CHECK(succeeded(ovspatialquery_overlap_multiple_sync(instance, &overlap, &hitBuffer, &hitResult)));
    CHECK(hitResult.source_ordinal == 2);
    CHECK(hitResult.stored_hit_count == 2u);
    CHECK(hitResult.truncated == 0u);
    // An Ovstage object ID's value is its prim path.
    std::array<ovx_primpath_t, 2> converted{makeHandle<ovx_primpath_t>(hits[0].identity.object_id.value),
                                            makeHandle<ovx_primpath_t>(hits[1].identity.object_id.value)};
    std::sort(converted.begin(), converted.end());
    std::array<ovx_primpath_t, 2> expectedInitial{expectedPaths[0], expectedPaths[1]};
    std::sort(expectedInitial.begin(), expectedInitial.end());
    CHECK(converted == expectedInitial);

    ovspatialquery_multiple_result_t            viewResult{};
    std::array<ovspatialquery_overlap_hit_t, 2> viewHits{};
    ovspatialquery_overlap_hit_buffer_t         viewBuffer{viewHits.data(), viewHits.size()};
    CHECK(succeeded(ovspatialquery_overlap_multiple_sync(instance, &viewOverlap, &viewBuffer, &viewResult)));
    CHECK(viewResult.source_ordinal == 2);
    CHECK(viewResult.stored_hit_count == 1u);
    CHECK(viewHits[0].identity.object_id.value == handleValue(expectedPaths[1]));

    ovspatialquery_raycast_desc_t ray{};
    ray.common.target         = populationTarget(population);
    ray.common.precision      = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    ray.ray.origin[0]         = -10.0;
    ray.ray.unit_direction[0] = 1.0f;
    ray.ray.max_distance      = 100.0;
    ovspatialquery_raycast_single_result_t rayResult{};
    CHECK(succeeded(ovspatialquery_raycast_closest_sync(instance, &ray, &rayResult)));
    CHECK(rayResult.source_ordinal == 2);
    CHECK(rayResult.has_hit == 1u);
    CHECK(rayResult.hit.truth == OVSPATIALQUERY_HIT_BOUNDS_CANDIDATE);
    CHECK(rayResult.hit.data.bounds_candidate.identity.object_id.value == handleValue(expectedPaths[1]));

    ovspatialquery_sphere_sweep_desc_t sweep{};
    sweep.common                  = ray.common;
    sweep.sweep.center[0]         = -10.0;
    sweep.sweep.radius            = 0.5;
    sweep.sweep.unit_direction[0] = 1.0f;
    sweep.sweep.max_distance      = 100.0;
    ovspatialquery_sphere_sweep_closest_result_t sweepResult{};
    CHECK(succeeded(ovspatialquery_sweep_sphere_closest_sync(instance, &sweep, &sweepResult)));
    CHECK(sweepResult.source_ordinal == 2);
    CHECK(sweepResult.has_hit == 1u);
    CHECK(sweepResult.hit.data.bounds_candidate.identity.object_id.value == handleValue(expectedPaths[1]));

    ovspatialquery_pair_snapshot_desc_t pairDesc{};
    pairDesc.first  = populationTarget(population);
    pairDesc.second = viewTarget(view);
    std::array<ovspatialquery_pair_t, 4>  pairs{};
    ovspatialquery_pair_buffer_t          pairBuffer{pairs.data(), pairs.size()};
    ovspatialquery_pair_snapshot_result_t pairResult{};
    CHECK(succeeded(ovspatialquery_query_pairs_sync(instance, &pairDesc, &pairBuffer, &pairResult)));
    CHECK(pairResult.first_source_ordinal == 2);
    CHECK(pairResult.second_source_ordinal == 2);

    // A table that names two paths: the listed bits replace default_bits, and /World/Added,
    // absent at ordinal 2, takes its entry once it appears; one entry also sets bit 104. Malformed
    // tables leave it installed.
    const ovx_primpath_t                               tablePaths[] = {expectedPaths[2], expectedPaths[1]};
    const ovspatialquery_participation_bits_t          tableBits[]  = {{{UINT64_C(0x4), 0u}},
                                                                       {{UINT64_C(0x2), UINT64_C(1) << 40u}}};
    const ovspatialquery_participation_bits_t          eight{{UINT64_C(0x8), 0u}};
    const ovx_primpath_t                               zeroPath[]   = {OVX_INVALID_PRIMPATH};
    const ovx_primpath_t                               twicePaths[] = {expectedPaths[1], expectedPaths[1]};
    const ovspatialquery_ovstage_participation_table_t categories{{{UINT64_C(0x1), 0u}}, tablePaths, tableBits, 2u};
    const ovspatialquery_ovstage_participation_table_t malformed[] = {
        {eight, nullptr, tableBits, 1u},
        {eight, tablePaths, nullptr, 1u},
        {eight, zeroPath, tableBits, 1u},
        {eight, twicePaths, tableBits, 2u},
    };
    CHECK(succeeded(ovspatialquery_set_ovstage_participation_map(instance, population, &categories)));
    for (const ovspatialquery_ovstage_participation_table_t& table : malformed)
    {
        CHECK(ovspatialquery_set_ovstage_participation_map(instance, population, &table).status ==
              OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }

    double addedBounds[6]{4.0, -1.0, -1.0, 6.0, 1.0, 1.0};
    CHECK(writeBounds(stage, dictionary, "/World/Added", sizeof("/World/Added") - 1u, 3u, addedBounds));
    const ovstage_write_floor_desc_t floorThree{3u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
    CHECK(waitStageOperation(stage, ovstage_advance_write_floor(stage, &floorThree)));
    ovspatialquery_backend_runtime_t* runtime = ovspatialquery_test_oracle_get_default_runtime();
    CHECK(runtime != nullptr);
    ovspatialquery_test_oracle_block_before_publish(runtime, 3);
    operation = ovspatialquery_enqueue_ovstage_population_update(instance, population, 2, 3);
    CHECK(operation.status == OVSPATIALQUERY_API_SUCCESS);
    CHECK(ovspatialquery_test_oracle_wait_before_publish(runtime) == 1u);
    ovspatialquery_population_update_result_t pendingResult{};
    std::memset(&pendingResult, 0xa5, sizeof(pendingResult));
    const ovspatialquery_population_update_result_t preservedPendingResult = pendingResult;
    const ovspatialquery_result_t                   pendingFetch =
        ovspatialquery_fetch_population_update(instance, operation.op_id, &pendingResult);
    const bool pendingOutputPreserved =
        std::memcmp(&pendingResult, &preservedPendingResult, sizeof(pendingResult)) == 0;
    hitResult = {};
    const bool pendingReadsPreviousOrdinal =
        succeeded(ovspatialquery_overlap_multiple_sync(instance, &overlap, &hitBuffer, &hitResult)) &&
        hitResult.source_ordinal == 2 && hitResult.stored_hit_count == 2u;
    const ovspatialquery_result_t pendingViewDestroy       = ovspatialquery_destroy_view(instance, view);
    const ovspatialquery_result_t pendingPopulationDestroy = ovspatialquery_destroy_population(instance, population);
    ovspatialquery_test_oracle_release_before_publish(runtime);
    CHECK(pendingFetch.status == OVSPATIALQUERY_API_NOT_READY);
    CHECK(pendingOutputPreserved);
    CHECK(pendingReadsPreviousOrdinal);
    CHECK(pendingViewDestroy.status == OVSPATIALQUERY_API_BUSY);
    CHECK(pendingPopulationDestroy.status == OVSPATIALQUERY_API_BUSY);
    CHECK(fetchPopulationUpdateAndDestroy(instance, operation, 3, 3u, 0u));
    hitResult = {};
    CHECK(succeeded(ovspatialquery_overlap_multiple_sync(instance, &overlap, &hitBuffer, &hitResult)));
    CHECK(hitResult.source_ordinal == 3);
    CHECK(hitResult.stored_hit_count == 3u);
    // Each query bit admits exactly the objects whose published bits hold it: ordinal 3 captured
    // the table, while the retained ordinal 2 keeps the all-bits value it published without one.
    const auto admitted = [&](uint64_t queryBits, ovspatialquery_ordinal_t ordinal,
                              std::vector<ovx_primpath_t> expected, uint64_t highQueryBits = 0u)
    {
        ovspatialquery_overlap_desc_t categoryOverlap = overlap;
        categoryOverlap.common.query_bits             = {{queryBits, highQueryBits}};
        categoryOverlap.common.target.ordinal         = {OVSPATIALQUERY_ORDINAL_RETAINED_EXACT, ordinal};
        std::array<ovspatialquery_overlap_hit_t, 4> categoryHits{};
        ovspatialquery_overlap_hit_buffer_t         categoryBuffer{categoryHits.data(), categoryHits.size()};
        ovspatialquery_multiple_result_t            categoryResult{};
        std::vector<ovx_primpath_t>                 paths;
        const bool                                  queried = succeeded(
            ovspatialquery_overlap_multiple_sync(instance, &categoryOverlap, &categoryBuffer, &categoryResult));
        for (uint64_t index = 0u; queried && index < categoryResult.stored_hit_count; ++index)
        {
            paths.push_back(makeHandle<ovx_primpath_t>(categoryHits[index].identity.object_id.value));
        }
        std::sort(paths.begin(), paths.end());
        std::sort(expected.begin(), expected.end());
        return queried && paths == expected;
    };
    CHECK(admitted(UINT64_C(0x1), 3, {expectedPaths[0]}));
    CHECK(admitted(UINT64_C(0x2), 3, {expectedPaths[1]}));
    CHECK(admitted(UINT64_C(0x4), 3, {expectedPaths[2]}));
    CHECK(admitted(UINT64_C(0x8), 3, {}));
    CHECK(admitted(UINT64_C(0x8), 2, {expectedPaths[0], expectedPaths[1]}));
    CHECK(admitted(0u, 3, {expectedPaths[1]}, UINT64_C(1) << 40u));
    CHECK(admitted(0u, 2, {expectedPaths[0], expectedPaths[1]}, UINT64_C(1) << 63u));
    CHECK(succeeded(ovspatialquery_set_ovstage_participation_map(instance, population, nullptr)));

    double absentBounds[6]{
        absentBoundsValue(true),  absentBoundsValue(true),  absentBoundsValue(true),
        absentBoundsValue(false), absentBoundsValue(false), absentBoundsValue(false),
    };
    CHECK(
        writeBounds(stage, dictionary, "/World/MissingBounds", sizeof("/World/MissingBounds") - 1u, 4u, absentBounds));
    const ovstage_write_floor_desc_t floorFour{4u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
    CHECK(waitStageOperation(stage, ovstage_advance_write_floor(stage, &floorFour)));
    ovspatialquery_test_oracle_fail_next_build(runtime, OVSPATIALQUERY_API_INTERNAL);
    operation = ovspatialquery_enqueue_ovstage_population_update(instance, population, 3, 4);
    CHECK(operation.status == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_op_wait_result_t failedWait{};
    CHECK(ovspatialquery_wait_op(instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &failedWait).status ==
          OVSPATIALQUERY_API_SUCCESS);
    const ovspatialquery_string_t error = ovspatialquery_get_last_op_error(instance, operation.op_id);
    CHECK(error.ptr != nullptr && error.length != 0u);
    ovspatialquery_population_update_result_t preservedUpdate{};
    preservedUpdate.published_ordinal = 91;
    preservedUpdate.object_count      = 92u;
    CHECK(ovspatialquery_fetch_population_update(instance, operation.op_id, &preservedUpdate).status ==
          OVSPATIALQUERY_API_OPERATION_FAILED);
    CHECK(preservedUpdate.published_ordinal == 91);
    CHECK(preservedUpdate.object_count == 92u);
    CHECK(succeeded(ovspatialquery_destroy_op(instance, &operation.op_id)));
    CHECK(succeeded(ovspatialquery_get_population_info(instance, population, &info)));
    CHECK(info.head_ordinal == 3);
    CHECK(info.object_count == 3u);

    const ovx_string_t  absentPath{"/World/DoesNotExist", sizeof("/World/DoesNotExist") - 1u};
    ovstage_predicate_t emptyPredicates[2]{};
    emptyPredicates[0].attribute.string = ovx_string_t{"_worldExtent", sizeof("_worldExtent") - 1u};
    emptyPredicates[0].op               = OVSTAGE_FILTER_OP_HAS;
    emptyPredicates[1].attribute.string = ovx_string_t{"usd-path", sizeof("usd-path") - 1u};
    emptyPredicates[1].op               = OVSTAGE_FILTER_OP_IN;
    emptyPredicates[1].values           = &absentPath;
    emptyPredicates[1].value_count      = 1u;
    ovspatialquery_ovstage_population_desc_t emptyDesc{};
    emptyDesc.stage                                    = stage;
    emptyDesc.filter                                   = ovstage_filter_t{emptyPredicates, 2u};
    emptyDesc.common                                   = aabbPopulationDesc();
    ovspatialquery_population_handle_t emptyPopulation = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &emptyDesc, &emptyPopulation)));
    operation = ovspatialquery_enqueue_ovstage_population_set(instance, emptyPopulation, 3);
    CHECK(operation.status == OVSPATIALQUERY_API_STALE_ORDINAL);
    CHECK(operation.op_id == OVSPATIALQUERY_INVALID_OP);
    operation = ovspatialquery_enqueue_ovstage_population_set(instance, emptyPopulation, 4);
    CHECK(finishPopulationUpdate(instance, operation, 4, 0u, 0u));
    CHECK(succeeded(ovspatialquery_destroy_population(instance, emptyPopulation)));

    CHECK(succeeded(ovspatialquery_destroy_view(instance, view)));
    CHECK(succeeded(ovspatialquery_destroy_population(instance, population)));
    CHECK(succeeded(ovspatialquery_destroy_instance(instance)));
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
    CHECK(testRealAabbSource(stage, dictionary) == 0);
    CHECK(ovstage_destroy_instance(stage) == OVSTAGE_OK);
    return 0;
}
