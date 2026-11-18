/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../support/native/public_api_inputs.h"
#include "../../support/native/test_failure.h"
#include <ovspatialquery/ovspatialquery_explicit.h>
#include <ovspatialquery/ovspatialquery_pairs.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>

// Explicit views: a copied-ID stable subset that shares its parent
// population's published versions. Queries against a view see only the
// selected IDs present in the pinned generation, and the parent population
// stays busy while any view over it is live.
//
// View readiness and query-target introspection have no public equivalent
// and are not exercised here. Membership resolution for head and
// retained-exact ordinals is exercised at the internal-detail level by
// family_query_sync_test.cpp and target_resolver_test.cpp.

namespace
{

constexpr uint64_t kUnitSphere = 1u;

ovspatialquery_instance_handle_t createInstance()
{
    const ovspatialquery_instance_desc_t desc{1u, 1u, UINT64_C(16) * 1024u * 1024u, nullptr, 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    return instance;
}

// Publishes ordinal 7 with two unit spheres: object 10 at x=0.5 and object
// 20 at x=4.5, each with a published world AABB one unit wide.
ovspatialquery_population_handle_t createTwoObjectPopulation(ovspatialquery_instance_handle_t instance)
{
    const ovspatialquery_population_handle_t population = createExplicitPopulation(instance, 0u);
    InstanceRows                             rows;
    rows.add(10u, OVSPATIALQUERY_GEOMETRY_SPHERE, kUnitSphere, 0.5, 0.5, 0.5);
    rows.add(20u, OVSPATIALQUERY_GEOMETRY_SPHERE, kUnitSphere, 4.5, 0.5, 0.5);
    const ovspatialquery_explicit_population_set_t set{7, rows.table(), nullptr, 0u};
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));

    ovspatialquery_population_info_t info{};
    REQUIRE(ovspatialquery_get_population_info(instance, population, &info).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(info.ready != 0u && info.head_ordinal == 7 && info.object_count == 2u);
    return population;
}

ovspatialquery_view_handle_t createView(ovspatialquery_instance_handle_t   instance,
                                        ovspatialquery_population_handle_t population, const InputArray& ids)
{
    const ovspatialquery_explicit_view_desc_t desc{&ids.value};
    ovspatialquery_view_handle_t              view = OVSPATIALQUERY_INVALID_VIEW;
    REQUIRE(ovspatialquery_create_explicit_view(instance, population, &desc, &view).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(view != OVSPATIALQUERY_INVALID_VIEW);
    return view;
}

// Rejected view creation always zeros out_view; the caller passes a sentinel
// to confirm that.
void requireViewRejected(ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population,
                         const InputArray& ids, ovspatialquery_api_status_t expected)
{
    const ovspatialquery_explicit_view_desc_t desc{&ids.value};
    ovspatialquery_view_handle_t              view = ovspatialquery_view_handle_t{UINT64_MAX};
    const ovspatialquery_result_t status = ovspatialquery_create_explicit_view(instance, population, &desc, &view);
    REQUIRE(status.status == expected);
    REQUIRE(view == OVSPATIALQUERY_INVALID_VIEW);
}

// A view selecting object 20 and one absent ID (99) filters every family of
// synchronous query to the one present member, at the pinned parent ordinal.
void testExplicitViewFiltersQueries()
{
    const ovspatialquery_instance_handle_t instance = createInstance();
    registerSphere(instance, kUnitSphere, 0.5);
    const ovspatialquery_population_handle_t population = createTwoObjectPopulation(instance);

    uint64_t                            selectedIds[2] = {20u, 99u};
    InputArray                          selector(selectedIds, 2, kDLUInt);
    const ovspatialquery_view_handle_t  view   = createView(instance, population, selector);
    const ovspatialquery_query_target_t target = viewTarget(view);

    ovspatialquery_raycast_desc_t ray{};
    ray.common.target         = target;
    ray.common.precision      = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    ray.ray.origin[0]         = -2.0;
    ray.ray.origin[1]         = 0.5;
    ray.ray.origin[2]         = 0.5;
    ray.ray.unit_direction[0] = 1.0f;
    ray.ray.max_distance      = 20.0;

    ovspatialquery_raycast_single_result_t closest{};
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &closest).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(closest.source_ordinal == 7 && closest.has_hit == 1u &&
            closest.hit.data.bounds_candidate.identity.object_id.value == 20u);

    ovspatialquery_raycast_single_result_t any{};
    REQUIRE(ovspatialquery_raycast_any_sync(instance, &ray, &any).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(any.has_hit == 1u && any.hit.data.bounds_candidate.identity.object_id.value == 20u);

    std::array<ovspatialquery_raycast_hit_t, 2u> rayRows{};
    ovspatialquery_raycast_hit_buffer_t          rayBuffer{rayRows.data(), rayRows.size()};
    ovspatialquery_multiple_result_t             rayMultiple{};
    REQUIRE(ovspatialquery_raycast_multiple_sync(instance, &ray, &rayBuffer, &rayMultiple).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(rayMultiple.source_ordinal == 7 && rayMultiple.stored_hit_count == 1u &&
            rayRows[0].data.bounds_candidate.identity.object_id.value == 20u);

    ovspatialquery_sphere_sweep_desc_t sweep{};
    sweep.common                  = ray.common;
    sweep.sweep.center[0]         = -2.0;
    sweep.sweep.center[1]         = 0.5;
    sweep.sweep.center[2]         = 0.5;
    sweep.sweep.radius            = 0.5;
    sweep.sweep.unit_direction[0] = 1.0f;
    sweep.sweep.max_distance      = 20.0;
    ovspatialquery_sphere_sweep_closest_result_t sweepResult{};
    REQUIRE(ovspatialquery_sweep_sphere_closest_sync(instance, &sweep, &sweepResult).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(sweepResult.source_ordinal == 7 && sweepResult.has_hit == 1u &&
            sweepResult.hit.data.bounds_candidate.identity.object_id.value == 20u);

    ovspatialquery_overlap_desc_t overlap{};
    overlap.common.target                 = target;
    overlap.common.precision              = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    overlap.shape.kind                    = OVSPATIALQUERY_OVERLAP_AABB;
    overlap.shape.data.aabb.min_corner[0] = -1.0;
    overlap.shape.data.aabb.min_corner[1] = -1.0;
    overlap.shape.data.aabb.min_corner[2] = -1.0;
    overlap.shape.data.aabb.max_corner[0] = 6.0;
    overlap.shape.data.aabb.max_corner[1] = 2.0;
    overlap.shape.data.aabb.max_corner[2] = 2.0;

    ovspatialquery_overlap_any_result_t overlapAny{};
    REQUIRE(ovspatialquery_overlap_any_sync(instance, &overlap, &overlapAny).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(overlapAny.source_ordinal == 7 && overlapAny.has_hit == 1u &&
            overlapAny.hit.identity.object_id.value == 20u);

    std::array<ovspatialquery_overlap_hit_t, 2u> overlapRows{};
    ovspatialquery_overlap_hit_buffer_t          overlapBuffer{overlapRows.data(), overlapRows.size()};
    ovspatialquery_multiple_result_t             overlapMultiple{};
    REQUIRE(ovspatialquery_overlap_multiple_sync(instance, &overlap, &overlapBuffer, &overlapMultiple).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(overlapMultiple.source_ordinal == 7 && overlapMultiple.stored_hit_count == 1u &&
            overlapRows[0].identity.object_id.value == 20u);

    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_BUSY);
    REQUIRE(ovspatialquery_destroy_view(instance, view).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(
        ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_SPHERE, kUnitSphere}).status ==
        OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

// Selector validation is atomic (a rejected call never creates a view), and a
// view is usable only from its own instance and only until destroyed.
void testExplicitViewValidationAndAtomicity()
{
    const ovspatialquery_instance_handle_t instance        = createInstance();
    const ovspatialquery_instance_handle_t foreignInstance = createInstance();
    registerSphere(instance, kUnitSphere, 0.5);
    const ovspatialquery_population_handle_t population = createTwoObjectPopulation(instance);

    uint64_t   selectedIds[2] = {10u, 10u};
    InputArray duplicate(selectedIds, 2, kDLUInt);
    requireViewRejected(instance, population, duplicate, OVSPATIALQUERY_API_INVALID_ARGUMENT);

    selectedIds[1] = 0u;
    InputArray zeroId(selectedIds, 2, kDLUInt);
    requireViewRejected(instance, population, zeroId, OVSPATIALQUERY_API_INVALID_ARGUMENT);

    selectedIds[1] = 99u;
    InputArray borrowed(selectedIds, 2, kDLUInt);
    borrowed.value.ownership = OVSPATIALQUERY_INPUT_BORROWED_UNTIL_TERMINAL;
    requireViewRejected(instance, population, borrowed, OVSPATIALQUERY_API_INVALID_ARGUMENT);

    InputArray oversized(selectedIds, 2, kDLUInt);
    // Byte count exceeds vector::max_size without overflowing size_t: reject
    // deterministically rather than asking the allocator for exabytes.
    oversized.shape[0] = std::numeric_limits<int64_t>::max() / 8 + 1;
    requireViewRejected(instance, population, oversized, OVSPATIALQUERY_API_OUT_OF_MEMORY);

    InputArray                         selector(selectedIds, 2, kDLUInt);
    const ovspatialquery_view_handle_t view = createView(instance, population, selector);

    ovspatialquery_raycast_desc_t ray{};
    ray.common.target         = viewTarget(view);
    ray.common.precision      = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    ray.ray.unit_direction[0] = 1.0f;
    ray.ray.max_distance      = 1.0;
    ovspatialquery_raycast_single_result_t foreignResult{};
    REQUIRE(ovspatialquery_raycast_closest_sync(foreignInstance, &ray, &foreignResult).status ==
            OVSPATIALQUERY_API_INVALID_HANDLE);

    REQUIRE(ovspatialquery_destroy_view(instance, view).status == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_raycast_single_result_t staleResult{};
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &staleResult).status ==
            OVSPATIALQUERY_API_INVALID_HANDLE);

    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(
        ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_SPHERE, kUnitSphere}).status ==
        OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(foreignInstance).status == OVSPATIALQUERY_API_SUCCESS);
}

// A plan or pair reporter that targets a view retains it: destroying the
// view is BUSY until the plan or reporter is destroyed.
void testPlanAndReporterRetainView()
{
    const ovspatialquery_instance_handle_t instance = createInstance();
    registerSphere(instance, kUnitSphere, 0.5);
    const ovspatialquery_population_handle_t population = createTwoObjectPopulation(instance);
    uint64_t                                 selectedId = 20u;
    InputArray                               selector(&selectedId, 1, kDLUInt);

    ovspatialquery_view_handle_t       view = createView(instance, population, selector);
    ovspatialquery_overlap_plan_desc_t planDesc{};
    planDesc.target                         = viewRef(view);
    planDesc.hit_mode                       = OVSPATIALQUERY_HIT_MODE_ANY;
    planDesc.precision                      = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    planDesc.shape.kind                     = OVSPATIALQUERY_OVERLAP_SPHERE;
    planDesc.shape.data.sphere.radius       = 1.0;
    planDesc.device                         = {kDLCPU, 0};
    planDesc.maximum_query_count            = 1u;
    planDesc.maximum_hits_per_query         = 1u;
    planDesc.hit_capacity_per_slot          = 1u;
    planDesc.result_slot_count              = 1u;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_overlap_plan(instance, &planDesc, &plan).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_view(instance, view).status == OVSPATIALQUERY_API_BUSY);
    REQUIRE(ovspatialquery_destroy_batch_plan(instance, plan).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_view(instance, view).status == OVSPATIALQUERY_API_SUCCESS);

    view = createView(instance, population, selector);
    ovspatialquery_pair_reporter_desc_t reporterDesc{};
    reporterDesc.first                = viewRef(view);
    reporterDesc.second               = populationRef(population);
    reporterDesc.active_pair_capacity = 1u;
    reporterDesc.delta_pair_capacity  = 1u;
    reporterDesc.result_slot_count    = 1u;
    reporterDesc.result_device        = {kDLCPU, 0};
    ovspatialquery_pair_reporter_handle_t reporter{};
    REQUIRE(ovspatialquery_create_pair_reporter(instance, &reporterDesc, &reporter).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_view(instance, view).status == OVSPATIALQUERY_API_BUSY);
    REQUIRE(ovspatialquery_destroy_pair_reporter(instance, reporter).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_view(instance, view).status == OVSPATIALQUERY_API_SUCCESS);

    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(
        ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_SPHERE, kUnitSphere}).status ==
        OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

} // namespace

int main()
{
    testExplicitViewFiltersQueries();
    testExplicitViewValidationAndAtomicity();
    testPlanAndReporterRetainView();
    std::puts("explicit view API test passed");
    return 0;
}
