/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../support/native/public_api_inputs.h"
#include "../../support/native/test_failure.h"
#include <ovspatialquery/ovspatialquery_explicit.h>

#include "ovspatialquery_backend.h"
#include "ovspatialquery_physx_backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <set>
#include <thread>
#include <vector>

// Real PhysX correctness for the public synchronous raycasts and sphere sweep:
// closest, any and multiple results, the tie, capacity and truncation contract of
// the *_multiple_sync functions, strided CPU plan query columns, and EXACT_SINGLE
// admission of CPU query plans.
// Overlap modes live in overlap_modes_test.cpp. One concurrency scenario uses the
// PhysX ray-blocking test hooks (ovspatialquery_physx_backend.h) and the
// process-wide default backend runtime every public instance binds to.

namespace
{

bool nearlyEqual(double left, double right, double epsilon = 1.0e-6) { return std::abs(left - right) < epsilon; }

ovspatialquery_instance_handle_t createInstance()
{
    const ovspatialquery_instance_desc_t instanceDesc{4u, 1u, UINT64_C(64) * 1024u * 1024u, nullptr, 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&instanceDesc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    return instance;
}

// Every instance in one buildBoxPopulation call shares the single box
// definition registered by registerBox, so only its id and translation vary.
struct BoxInstance
{
    uint64_t id;
    double   center[3];
};

constexpr uint64_t kBoxDefinition    = 1u;
constexpr uint64_t kSphereDefinition = 2u;

void registerBox(ovspatialquery_instance_handle_t instance, double halfExtent)
{
    uint64_t                                    value          = kBoxDefinition;
    double                                      halfExtents[3] = {halfExtent, halfExtent, halfExtent};
    InputArray                                  values(&value, 1, kDLUInt);
    InputArray                                  extents(halfExtents, 1, 3, kDLFloat);
    const ovspatialquery_box_definition_table_t boxes{&values.value, &extents.value};
    waitAndDestroy(instance, ovspatialquery_enqueue_box_registration(instance, &boxes));
}

// Every instance in one row set shares the box definition registered by
// registerBox; only id and translation vary, matching how the box was
// registered above. Shared by the first-set and republish paths below.
void addBoxes(InstanceRows& rows, const std::vector<BoxInstance>& boxes)
{
    for (const BoxInstance& box : boxes)
    {
        rows.add(box.id, OVSPATIALQUERY_GEOMETRY_BOX, kBoxDefinition, box.center[0], box.center[1], box.center[2]);
    }
}

ovspatialquery_population_handle_t buildBoxPopulation(ovspatialquery_instance_handle_t instance,
                                                      ovspatialquery_ordinal_t         ordinal,
                                                      const std::vector<BoxInstance>&  boxes)
{
    const ovspatialquery_population_handle_t population = createExplicitPopulation(instance, 0u);
    InstanceRows                             rows;
    addBoxes(rows, boxes);
    const ovspatialquery_explicit_population_set_t set{ordinal, rows.table(), nullptr, 0u};
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));
    return population;
}

// Republishes a complete box-instance replacement at `targetOrdinal`: every
// id in `removed` leaves the population and every row in `boxes` is the
// complete row for its id afterward.
void updateBoxPopulation(ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population,
                         ovspatialquery_ordinal_t baseOrdinal, ovspatialquery_ordinal_t targetOrdinal,
                         const std::vector<BoxInstance>& boxes, std::vector<uint64_t> removed)
{
    InstanceRows rows;
    addBoxes(rows, boxes);
    InputArray removedArray(removed.data(), static_cast<int64_t>(removed.size()), kDLUInt);
    const ovspatialquery_explicit_population_update_t update{
        baseOrdinal,  targetOrdinal,       OVSPATIALQUERY_EXPLICIT_UPDATE_COMPLETE_ROWS,
        rows.table(), &removedArray.value, nullptr};
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_update(instance, population, &update));
}

ovspatialquery_population_handle_t buildSpherePopulation(ovspatialquery_instance_handle_t instance,
                                                         ovspatialquery_ordinal_t ordinal, uint64_t id,
                                                         const double center[3])
{
    const ovspatialquery_population_handle_t population = createExplicitPopulation(instance, 0u);
    InstanceRows                             rows;
    rows.add(id, OVSPATIALQUERY_GEOMETRY_SPHERE, kSphereDefinition, center[0], center[1], center[2]);
    const ovspatialquery_explicit_population_set_t set{ordinal, rows.table(), nullptr, 0u};
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));
    return population;
}

ovspatialquery_query_common_t commonQuery(ovspatialquery_population_handle_t population,
                                          ovspatialquery_query_precision_t   precision)
{
    ovspatialquery_query_common_t common{};
    common.target    = populationTarget(population);
    common.precision = precision;
    return common;
}

// hasBox/hasSphere say which definitions registerBox/registerSphere placed in
// this instance, so only those are torn down.
void destroyPopulationAndInstance(ovspatialquery_instance_handle_t                instance,
                                  std::vector<ovspatialquery_population_handle_t> populations, bool hasBox,
                                  bool hasSphere)
{
    for (ovspatialquery_population_handle_t population : populations)
    {
        REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    }
    if (hasBox)
    {
        REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_BOX, kBoxDefinition})
                    .status == OVSPATIALQUERY_API_SUCCESS);
    }
    if (hasSphere)
    {
        REQUIRE(
            ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_SPHERE, kSphereDefinition})
                .status == OVSPATIALQUERY_API_SUCCESS);
    }
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

// Real PhysX raycast_closest/any/multiple against real box geometry: hit
// identity, distance, position, and outward face normal (family_query_sync_test
// proves the generic dispatch contract through the oracle backend, not real
// PhysX collision math, so the geometric values checked here are not
// duplicated there).
void testRaycastFamilyCorrectness()
{
    ovspatialquery_instance_handle_t instance = createInstance();
    registerBox(instance, 0.5);
    const std::vector<BoxInstance> boxes{
        {20u, {0.5, 0.5, 0.5}},
        {10u, {4.5, 0.5, 0.5}},
        {30u, {0.5, 3.5, 0.5}},
    };
    const ovspatialquery_population_handle_t population = buildBoxPopulation(instance, 7, boxes);

    ovspatialquery_raycast_desc_t ray{};
    ray.common                = commonQuery(population, OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE);
    ray.ray.origin[0]         = -2.0;
    ray.ray.origin[1]         = 0.5;
    ray.ray.origin[2]         = 0.5;
    ray.ray.unit_direction[0] = 1.0f;
    ray.ray.max_distance      = 10.0;

    ovspatialquery_raycast_single_result_t closest{};
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &closest).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(closest.source_ordinal == 7 && closest.has_hit == 1u);
    REQUIRE(closest.hit.truth == OVSPATIALQUERY_HIT_BOUNDS_CANDIDATE);
    const ovspatialquery_raycast_bounds_candidate_t& hit = closest.hit.data.bounds_candidate;
    REQUIRE(hit.identity.object_id.value == 20u);
    REQUIRE(nearlyEqual(hit.distance, 2.0));
    REQUIRE(nearlyEqual(hit.bounds_position[0], 0.0));
    REQUIRE(hit.bounds_normal[0] == -1.0f);

    ovspatialquery_raycast_single_result_t any{};
    REQUIRE(ovspatialquery_raycast_any_sync(instance, &ray, &any).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(any.source_ordinal == 7 && any.has_hit == 1u);
    REQUIRE(any.hit.data.bounds_candidate.identity.object_id.value == 10u ||
            any.hit.data.bounds_candidate.identity.object_id.value == 20u);

    ovspatialquery_raycast_desc_t missRay = ray;
    missRay.ray.origin[1]                 = 2.0;
    ovspatialquery_raycast_single_result_t miss{};
    REQUIRE(ovspatialquery_raycast_any_sync(instance, &missRay, &miss).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(miss.source_ordinal == 7 && miss.has_hit == 0u);

    ovspatialquery_raycast_hit_t        hits[2]{};
    ovspatialquery_raycast_hit_buffer_t buffer{hits, 2u};
    ovspatialquery_multiple_result_t    multiple{};
    REQUIRE(ovspatialquery_raycast_multiple_sync(instance, &ray, &buffer, &multiple).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(multiple.source_ordinal == 7 && multiple.stored_hit_count == 2u && multiple.truncated == 0u);
    std::set<uint64_t> multipleIds;
    for (const ovspatialquery_raycast_hit_t& row : hits)
    {
        multipleIds.insert(row.data.bounds_candidate.identity.object_id.value);
    }
    REQUIRE(multipleIds == (std::set<uint64_t>{10u, 20u}));

    destroyPopulationAndInstance(instance, {population}, true, false);
}

// A CPU raycast plan reads its query columns in place whatever their nonnegative
// strides (ADR-0053): origins sliced from one (Q, 6) ray array, column-major
// directions, one minimum repeated by a zero stride and every other entry of a
// maximum array return what compact columns return, row for row, so each row
// reads its own values. A column one row shorter than the launch is refused,
// however many of its rows are present.
void testStridedCpuRaycastColumns()
{
    ovspatialquery_instance_handle_t instance = createInstance();
    registerBox(instance, 0.5);
    const ovspatialquery_population_handle_t population =
        buildBoxPopulation(instance, 3, {{10u, {0.5, 0.5, 0.5}}, {20u, {0.5, 3.5, 0.5}}, {30u, {0.5, 6.5, 0.5}}});
    constexpr uint64_t kRows = 4u;
    // Rows 0 and 1 hit the faces of boxes 10 and 20 1.5 m away; row 2's maximum stops
    // before box 30 and row 3 passes between the boxes.
    const double kY[kRows]       = {0.5, 3.5, 6.5, 2.0};
    const double kMaximum[kRows] = {10.0, 9.0, 1.25, 10.0};

    ovspatialquery_raycast_plan_desc_t desc{};
    desc.target                             = populationRef(population);
    desc.hit_mode                           = OVSPATIALQUERY_HIT_MODE_CLOSEST;
    desc.precision                          = OVSPATIALQUERY_QUERY_EXACT;
    desc.device                             = {kDLCPU, 0};
    desc.maximum_query_count                = kRows;
    desc.maximum_hits_per_query             = 1u;
    desc.hit_capacity_per_slot              = kRows;
    desc.result_slot_count                  = 1u;
    desc.result_fields                      = OVSPATIALQUERY_BATCH_RESULT_DISTANCE;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_raycast_plan(instance, &desc, &plan).status == OVSPATIALQUERY_API_SUCCESS);

    struct Row
    {
        uint64_t count;
        uint64_t id;
        double   distance;
    };
    const auto launch = [&](const ovspatialquery_raycast_batch_t& batch)
    {
        ovspatialquery_raycast_batch_reservation_t reservation{};
        const ovspatialquery_enqueue_result_t      admission =
            ovspatialquery_enqueue_raycast_plan(instance, plan, &batch, &reservation);
        REQUIRE(admission.status == OVSPATIALQUERY_API_SUCCESS);
        ovspatialquery_op_wait_result_t waited{};
        REQUIRE(ovspatialquery_wait_op(instance, admission.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
                    OVSPATIALQUERY_API_SUCCESS &&
                waited.state == OVSPATIALQUERY_OP_COMPLETED);
        ovspatialquery_raycast_batch_result_t result{};
        REQUIRE(ovspatialquery_fetch_raycast_batch(instance, admission.op_id, &result).status ==
                OVSPATIALQUERY_API_SUCCESS);
        const auto*      offsets   = static_cast<const uint64_t*>(result.columns.hit_ranges.hit_offsets.data);
        const auto*      counts    = static_cast<const uint64_t*>(result.columns.hit_ranges.hit_counts.data);
        const auto*      ids       = static_cast<const uint64_t*>(result.columns.object_ids.data);
        const auto*      distances = static_cast<const double*>(result.columns.contact.exact.distances.data);
        std::vector<Row> rows;
        for (uint64_t row = 0u; row < kRows; ++row)
        {
            rows.push_back(counts[row] == 0u ? Row{0u, 0u, 0.0}
                                             : Row{counts[row], ids[offsets[row]], distances[offsets[row]]});
        }
        ovspatialquery_op_id_t operation = admission.op_id;
        REQUIRE(ovspatialquery_destroy_op(instance, &operation).status == OVSPATIALQUERY_API_SUCCESS);
        return rows;
    };

    double origins[kRows * 3u]{};
    float  directions[kRows * 3u]{};
    double rays[kRows * 6u]{};
    float  columnMajor[3u * kRows]{};
    double everyOtherMaximum[kRows * 2u]{};
    for (uint64_t row = 0u; row < kRows; ++row)
    {
        const double origin[3]    = {-1.5, kY[row], 0.5};
        const float  direction[3] = {1.0f, 0.0f, 0.0f};
        for (uint64_t axis = 0u; axis < 3u; ++axis)
        {
            origins[row * 3u + axis]        = origin[axis];
            directions[row * 3u + axis]     = direction[axis];
            rays[row * 6u + axis]           = origin[axis];
            rays[row * 6u + 3u + axis]      = double(direction[axis]);
            columnMajor[axis * kRows + row] = direction[axis];
        }
        everyOtherMaximum[row * 2u]      = kMaximum[row];
        everyOtherMaximum[row * 2u + 1u] = -1.0; // never read
    }
    const double                   minimum = 0.0;
    double                         compactMinimums[kRows]{};
    InputArray                     compactOrigins(origins, kRows, 3, kDLFloat);
    InputArray                     compactDirections(directions, kRows, 3, kDLFloat);
    InputArray                     compactMinimumColumn(compactMinimums, kRows, kDLFloat);
    InputArray                     compactMaximums(kMaximum, kRows, kDLFloat);
    ovspatialquery_raycast_batch_t batch{};
    batch.ordinal.kind             = OVSPATIALQUERY_ORDINAL_HEAD;
    batch.query_count              = kRows;
    batch.origins                  = &compactOrigins.value;
    batch.unit_directions          = &compactDirections.value;
    batch.min_distances            = &compactMinimumColumn.value;
    batch.max_distances            = &compactMaximums.value;
    const std::vector<Row> compact = launch(batch);
    REQUIRE(compact[0].count == 1u && compact[0].id == 10u && nearlyEqual(compact[0].distance, 1.5));
    REQUIRE(compact[1].count == 1u && compact[1].id == 20u && nearlyEqual(compact[1].distance, 1.5));
    REQUIRE(compact[2].count == 0u && compact[3].count == 0u);

    InputArray stridedOrigins(rays, kRows, 3, kDLFloat);
    InputArray stridedDirections(columnMajor, kRows, 3, kDLFloat);
    InputArray repeatedMinimum(&minimum, kRows, kDLFloat);
    InputArray stridedMaximums(everyOtherMaximum, kRows, kDLFloat);
    stridedOrigins.strides         = {6, 1};
    stridedDirections.strides      = {1, int64_t(kRows)};
    repeatedMinimum.strides        = {0, 0};
    stridedMaximums.strides        = {2, 0};
    batch.origins                  = &stridedOrigins.value;
    batch.unit_directions          = &stridedDirections.value;
    batch.min_distances            = &repeatedMinimum.value;
    batch.max_distances            = &stridedMaximums.value;
    const std::vector<Row> strided = launch(batch);
    for (uint64_t row = 0u; row < kRows; ++row)
    {
        REQUIRE(strided[row].count == compact[row].count && strided[row].id == compact[row].id &&
                strided[row].distance == compact[row].distance);
    }

    // Only the last row is missing: a check of the first rows alone would read past the column.
    InputArray shortMaximums(kMaximum, kRows - 1u, kDLFloat);
    batch.max_distances = &shortMaximums.value;
    ovspatialquery_raycast_batch_reservation_t reservation{};
    REQUIRE(ovspatialquery_enqueue_raycast_plan(instance, plan, &batch, &reservation).status ==
            OVSPATIALQUERY_API_INVALID_ARGUMENT);

    REQUIRE(ovspatialquery_destroy_batch_plan(instance, plan).status == OVSPATIALQUERY_API_SUCCESS);
    destroyPopulationAndInstance(instance, {population}, true, false);
}

// The same EXACT_SINGLE contract for reusable CPU plans: every ray hit mode
// creates a plan whose exact columns carry PhysX hits with feature kinds,
// sweep and overlap plans are NOT_SUPPORTED, and an unknown precision is
// INVALID_ARGUMENT for every family. `population` holds boxes 20 at [0,1] and
// 10 at [4,5] on x, published at ordinal 12.
void testExactSingleRayPlans(ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population)
{
    const ovspatialquery_target_ref_t target = populationRef(population);

    double     origin[3]    = {-2.0, 0.5, 0.5};
    float      direction[3] = {1.0f, 0.0f, 0.0f};
    double     minDistance  = 0.0;
    double     maxDistance  = 10.0;
    InputArray origins(origin, 1, 3, kDLFloat);
    InputArray directions(direction, 1, 3, kDLFloat);
    InputArray minimums(&minDistance, 1, kDLFloat);
    InputArray maximums(&maxDistance, 1, kDLFloat);

    for (ovspatialquery_hit_mode_t mode :
         {OVSPATIALQUERY_HIT_MODE_CLOSEST, OVSPATIALQUERY_HIT_MODE_ANY, OVSPATIALQUERY_HIT_MODE_MULTIPLE})
    {
        const uint64_t                     hitCapacity = mode == OVSPATIALQUERY_HIT_MODE_MULTIPLE ? 2u : 1u;
        ovspatialquery_raycast_plan_desc_t desc{};
        desc.target                 = target;
        desc.hit_mode               = mode;
        desc.precision              = OVSPATIALQUERY_QUERY_EXACT_SINGLE;
        desc.device                 = {kDLCPU, 0};
        desc.maximum_query_count    = 1u;
        desc.maximum_hits_per_query = hitCapacity;
        desc.hit_capacity_per_slot  = hitCapacity;
        desc.result_slot_count      = 1u;
        desc.result_fields          = OVSPATIALQUERY_BATCH_RESULT_POSITION | OVSPATIALQUERY_BATCH_RESULT_DISTANCE |
                             OVSPATIALQUERY_BATCH_RESULT_SOURCE_FEATURE;
        ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
        REQUIRE(ovspatialquery_create_raycast_plan(instance, &desc, &plan).status == OVSPATIALQUERY_API_SUCCESS);

        uint64_t                       maximumHits = hitCapacity;
        InputArray                     maximumHitsArray(&maximumHits, 1, kDLUInt);
        ovspatialquery_raycast_batch_t batch{};
        batch.ordinal.kind    = OVSPATIALQUERY_ORDINAL_HEAD;
        batch.query_count     = 1u;
        batch.origins         = &origins.value;
        batch.unit_directions = &directions.value;
        batch.min_distances   = &minimums.value;
        batch.max_distances   = &maximums.value;
        batch.maximum_hits    = mode == OVSPATIALQUERY_HIT_MODE_MULTIPLE ? &maximumHitsArray.value : nullptr;
        ovspatialquery_raycast_batch_reservation_t reservation{};
        const ovspatialquery_enqueue_result_t      admission =
            ovspatialquery_enqueue_raycast_plan(instance, plan, &batch, &reservation);
        REQUIRE(admission.status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(reservation.common.truth == OVSPATIALQUERY_HIT_EXACT_SURFACE);
        REQUIRE(reservation.columns.contact.exact.features.kinds.data != nullptr);

        ovspatialquery_op_wait_result_t waited{};
        REQUIRE(ovspatialquery_wait_op(instance, admission.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
                OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(waited.state == OVSPATIALQUERY_OP_COMPLETED);
        ovspatialquery_raycast_batch_result_t result{};
        REQUIRE(ovspatialquery_fetch_raycast_batch(instance, admission.op_id, &result).status ==
                OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(result.common.source_ordinal == 12 && result.common.truth == OVSPATIALQUERY_HIT_EXACT_SURFACE);
        const uint64_t stored = *static_cast<const uint64_t*>(result.columns.hit_ranges.stored_hit_count.data);
        REQUIRE(stored == hitCapacity);
        const uint64_t* ids = static_cast<const uint64_t*>(result.columns.object_ids.data);
        if (mode == OVSPATIALQUERY_HIT_MODE_CLOSEST)
        {
            REQUIRE(ids[0] == 20u);
            REQUIRE(nearlyEqual(static_cast<const double*>(result.columns.contact.exact.distances.data)[0], 2.0));
            REQUIRE(nearlyEqual(static_cast<const double*>(result.columns.contact.exact.positions.data)[0], 0.0));
        }
        else if (mode == OVSPATIALQUERY_HIT_MODE_ANY)
        {
            REQUIRE(ids[0] == 20u || ids[0] == 10u);
        }
        else
        {
            REQUIRE((ids[0] == 20u && ids[1] == 10u) || (ids[0] == 10u && ids[1] == 20u));
        }
        const uint32_t* featureKinds = static_cast<const uint32_t*>(result.columns.contact.exact.features.kinds.data);
        for (uint64_t index = 0u; index < stored; ++index)
        {
            REQUIRE(featureKinds[index] == OVSPATIALQUERY_GEOMETRY_FEATURE_NONE);
        }
        ovspatialquery_op_id_t operation = admission.op_id;
        REQUIRE(ovspatialquery_destroy_op(instance, &operation).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(ovspatialquery_destroy_batch_plan(instance, plan).status == OVSPATIALQUERY_API_SUCCESS);
    }

    ovspatialquery_sphere_sweep_plan_desc_t sweep{};
    sweep.target                                = target;
    sweep.precision                             = OVSPATIALQUERY_QUERY_EXACT_SINGLE;
    sweep.radius                                = 0.25;
    sweep.device                                = {kDLCPU, 0};
    sweep.maximum_query_count                   = 1u;
    sweep.result_slot_count                     = 1u;
    sweep.hit_capacity_per_slot                 = 1u;
    ovspatialquery_batch_plan_handle_t rejected = ovspatialquery_batch_plan_handle_t{UINT64_C(0xa5a5a5a5)};
    REQUIRE(ovspatialquery_create_sphere_sweep_plan(instance, &sweep, &rejected).status ==
            OVSPATIALQUERY_API_NOT_SUPPORTED);
    REQUIRE(rejected == OVSPATIALQUERY_INVALID_BATCH_PLAN);

    ovspatialquery_overlap_plan_desc_t overlap{};
    overlap.target                   = target;
    overlap.hit_mode                 = OVSPATIALQUERY_HIT_MODE_ANY;
    overlap.precision                = OVSPATIALQUERY_QUERY_EXACT_SINGLE;
    overlap.shape.kind               = OVSPATIALQUERY_OVERLAP_SPHERE;
    overlap.shape.data.sphere.radius = 0.25;
    overlap.device                   = {kDLCPU, 0};
    overlap.maximum_query_count      = 1u;
    overlap.maximum_hits_per_query   = 1u;
    overlap.hit_capacity_per_slot    = 1u;
    overlap.result_slot_count        = 1u;
    rejected                         = ovspatialquery_batch_plan_handle_t{UINT64_C(0xa5a5a5a5)};
    REQUIRE(ovspatialquery_create_overlap_plan(instance, &overlap, &rejected).status ==
            OVSPATIALQUERY_API_NOT_SUPPORTED);
    REQUIRE(rejected == OVSPATIALQUERY_INVALID_BATCH_PLAN);

    ovspatialquery_raycast_plan_desc_t ray{};
    ray.target                 = target;
    ray.hit_mode               = OVSPATIALQUERY_HIT_MODE_CLOSEST;
    ray.device                 = {kDLCPU, 0};
    ray.maximum_query_count    = 1u;
    ray.maximum_hits_per_query = 1u;
    ray.hit_capacity_per_slot  = 1u;
    ray.result_slot_count      = 1u;
    for (int precision : {-1, 99})
    {
        const ovspatialquery_query_precision_t unknown = static_cast<ovspatialquery_query_precision_t>(precision);
        ray.precision                                  = unknown;
        rejected                                       = ovspatialquery_batch_plan_handle_t{UINT64_C(0xa5a5a5a5)};
        REQUIRE(ovspatialquery_create_raycast_plan(instance, &ray, &rejected).status ==
                OVSPATIALQUERY_API_INVALID_ARGUMENT);
        REQUIRE(rejected == OVSPATIALQUERY_INVALID_BATCH_PLAN);

        sweep.precision = unknown;
        rejected        = ovspatialquery_batch_plan_handle_t{UINT64_C(0xa5a5a5a5)};
        REQUIRE(ovspatialquery_create_sphere_sweep_plan(instance, &sweep, &rejected).status ==
                OVSPATIALQUERY_API_INVALID_ARGUMENT);
        REQUIRE(rejected == OVSPATIALQUERY_INVALID_BATCH_PLAN);

        overlap.precision = unknown;
        rejected          = ovspatialquery_batch_plan_handle_t{UINT64_C(0xa5a5a5a5)};
        REQUIRE(ovspatialquery_create_overlap_plan(instance, &overlap, &rejected).status ==
                OVSPATIALQUERY_API_INVALID_ARGUMENT);
        REQUIRE(rejected == OVSPATIALQUERY_INVALID_BATCH_PLAN);
    }
}

// EXACT_SINGLE is an exact-result request even when the CPU backend reuses its
// existing evaluator. All ray hit modes admit it; the two other query families
// recognize the value but reject it before dispatch.
void testExactSingleRayModesToleranceAndAdmission()
{
    ovspatialquery_instance_handle_t instance = createInstance();
    registerBox(instance, 0.5);
    const std::vector<BoxInstance> boxes{
        {20u, {0.5, 0.5, 0.5}},
        {10u, {4.5, 0.5, 0.5}},
    };
    const ovspatialquery_population_handle_t population = buildBoxPopulation(instance, 12, boxes);

    ovspatialquery_raycast_desc_t ray{};
    ray.common                = commonQuery(population, OVSPATIALQUERY_QUERY_EXACT_SINGLE);
    ray.ray.origin[0]         = -2.0;
    ray.ray.origin[1]         = 0.5;
    ray.ray.origin[2]         = 0.5;
    ray.ray.unit_direction[0] = 1.0f;
    ray.ray.max_distance      = 10.0;

    ovspatialquery_raycast_single_result_t closest{};
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &closest).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(closest.source_ordinal == 12 && closest.has_hit == 1u);
    REQUIRE(closest.hit.truth == OVSPATIALQUERY_HIT_EXACT_SURFACE);
    REQUIRE(closest.hit.data.exact.identity.object_id.value == 20u);
    REQUIRE(nearlyEqual(closest.hit.data.exact.distance, 2.0));
    REQUIRE(nearlyEqual(closest.hit.data.exact.position[0], 0.0));

    ovspatialquery_raycast_single_result_t any{};
    REQUIRE(ovspatialquery_raycast_any_sync(instance, &ray, &any).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(any.source_ordinal == 12 && any.has_hit == 1u);
    REQUIRE(any.hit.truth == OVSPATIALQUERY_HIT_EXACT_SURFACE);
    REQUIRE(any.hit.data.exact.identity.object_id.value == 10u || any.hit.data.exact.identity.object_id.value == 20u);

    ovspatialquery_raycast_hit_t        hits[2]{};
    ovspatialquery_raycast_hit_buffer_t buffer{hits, 2u};
    ovspatialquery_multiple_result_t    multiple{};
    REQUIRE(ovspatialquery_raycast_multiple_sync(instance, &ray, &buffer, &multiple).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(multiple.source_ordinal == 12 && multiple.stored_hit_count == 2u && multiple.truncated == 0u);
    std::set<uint64_t> ids;
    for (const ovspatialquery_raycast_hit_t& hit : hits)
    {
        REQUIRE(hit.truth == OVSPATIALQUERY_HIT_EXACT_SURFACE);
        ids.insert(hit.data.exact.identity.object_id.value);
    }
    REQUIRE(ids == (std::set<uint64_t>{10u, 20u}));

    ovspatialquery_raycast_desc_t grazing = ray;
    grazing.ray.origin[1]                 = 1.05;
    ovspatialquery_raycast_single_result_t miss{};
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &grazing, &miss).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(miss.source_ordinal == 12 && miss.has_hit == 0u);
    grazing.common.contact_tolerance = 0.1;
    ovspatialquery_raycast_single_result_t tolerated{};
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &grazing, &tolerated).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(tolerated.source_ordinal == 12 && tolerated.has_hit == 1u);
    REQUIRE(tolerated.hit.truth == OVSPATIALQUERY_HIT_EXACT_SURFACE);

    ovspatialquery_sphere_sweep_desc_t sweep{};
    sweep.common                  = ray.common;
    sweep.sweep.center[0]         = -2.0;
    sweep.sweep.center[1]         = 0.5;
    sweep.sweep.center[2]         = 0.5;
    sweep.sweep.radius            = 0.25;
    sweep.sweep.unit_direction[0] = 1.0f;
    sweep.sweep.max_distance      = 10.0;
    ovspatialquery_sphere_sweep_closest_result_t sweepResult{};
    REQUIRE(ovspatialquery_sweep_sphere_closest_sync(instance, &sweep, &sweepResult).status ==
            OVSPATIALQUERY_API_NOT_SUPPORTED);

    ovspatialquery_overlap_desc_t overlap{};
    overlap.common                      = ray.common;
    overlap.shape.kind                  = OVSPATIALQUERY_OVERLAP_SPHERE;
    overlap.shape.data.sphere.center[0] = 0.5;
    overlap.shape.data.sphere.center[1] = 0.5;
    overlap.shape.data.sphere.center[2] = 0.5;
    overlap.shape.data.sphere.radius    = 0.25;
    ovspatialquery_overlap_any_result_t overlapResult{};
    REQUIRE(ovspatialquery_overlap_any_sync(instance, &overlap, &overlapResult).status ==
            OVSPATIALQUERY_API_NOT_SUPPORTED);

    for (int precision : {-1, 99})
    {
        ovspatialquery_raycast_desc_t unknownRay = ray;
        unknownRay.common.precision              = static_cast<ovspatialquery_query_precision_t>(precision);
        ovspatialquery_raycast_single_result_t unknownRayResult{};
        REQUIRE(ovspatialquery_raycast_closest_sync(instance, &unknownRay, &unknownRayResult).status ==
                OVSPATIALQUERY_API_INVALID_ARGUMENT);

        ovspatialquery_sphere_sweep_desc_t unknownSweep = sweep;
        unknownSweep.common.precision                   = static_cast<ovspatialquery_query_precision_t>(precision);
        ovspatialquery_sphere_sweep_closest_result_t unknownSweepResult{};
        REQUIRE(ovspatialquery_sweep_sphere_closest_sync(instance, &unknownSweep, &unknownSweepResult).status ==
                OVSPATIALQUERY_API_INVALID_ARGUMENT);

        ovspatialquery_overlap_desc_t unknownOverlap = overlap;
        unknownOverlap.common.precision              = static_cast<ovspatialquery_query_precision_t>(precision);
        ovspatialquery_overlap_any_result_t unknownOverlapResult{};
        REQUIRE(ovspatialquery_overlap_any_sync(instance, &unknownOverlap, &unknownOverlapResult).status ==
                OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }

    testExactSingleRayPlans(instance, population);
    destroyPopulationAndInstance(instance, {population}, true, false);
}

// Real PhysX sweep_sphere_closest against a real stored sphere, EXACT
// precision: contact position, target-surface position, and normal (the
// oracle differential in closest_query_test checks distance only).
void testSweepFamilyCorrectness()
{
    ovspatialquery_instance_handle_t instance = createInstance();
    registerSphere(instance, kSphereDefinition, 1.0);
    const double                             center[3]  = {5.0, 0.5, 0.5};
    const ovspatialquery_population_handle_t population = buildSpherePopulation(instance, 11, 61u, center);

    ovspatialquery_sphere_sweep_desc_t sweep{};
    sweep.common                  = commonQuery(population, OVSPATIALQUERY_QUERY_EXACT);
    sweep.sweep.center[0]         = 0.0;
    sweep.sweep.center[1]         = 0.5;
    sweep.sweep.center[2]         = 0.5;
    sweep.sweep.radius            = 0.5;
    sweep.sweep.unit_direction[0] = 1.0f;
    sweep.sweep.max_distance      = 20.0;

    ovspatialquery_sphere_sweep_closest_result_t result{};
    REQUIRE(ovspatialquery_sweep_sphere_closest_sync(instance, &sweep, &result).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.source_ordinal == 11 && result.has_hit == 1u);
    REQUIRE(result.hit.truth == OVSPATIALQUERY_HIT_EXACT_SURFACE);
    const ovspatialquery_sphere_sweep_exact_hit_t& hit = result.hit.data.exact;
    REQUIRE(hit.identity.object_id.value == 61u);
    // Combined radius 1.5 against a 5-unit separation: contact at distance 3.5.
    REQUIRE(nearlyEqual(hit.distance, 3.5));
    REQUIRE(nearlyEqual(hit.query_position[0], 3.5));
    REQUIRE(nearlyEqual(hit.object_position[0], 4.0));
    REQUIRE(hit.normal[0] == -1.0f);

    destroyPopulationAndInstance(instance, {population}, false, true);
}

// MULTIPLE rows of a tied pair and the buffer-capacity stored/truncated
// contract. Row order is unspecified, but a truncated raycast or sweep keeps
// its nearest rows by distance, then object ID, so a buffer of one keeps the
// lower ID of two objects at the same distance, inserted here after the higher.
template <typename Hit, typename Buffer, typename Desc, typename Multiple, typename Read>
void expectTruncatedTie(ovspatialquery_instance_handle_t instance, const Desc& desc, Multiple multiple, Read read,
                        double tiedDistance)
{
    for (const uint64_t capacity : {UINT64_C(1), UINT64_C(2)})
    {
        Hit                              rows[2]{};
        Buffer                           buffer{rows, capacity};
        ovspatialquery_multiple_result_t result{};
        REQUIRE(multiple(instance, &desc, &buffer, &result).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(result.source_ordinal == 1 && result.stored_hit_count == capacity && result.truncated == 1u);
        std::set<uint64_t> ids;
        for (uint64_t index = 0u; index < capacity; ++index)
        {
            uint64_t id       = 0u;
            double   distance = 0.0;
            read(rows[index], id, distance);
            REQUIRE(nearlyEqual(distance, tiedDistance));
            ids.insert(id);
        }
        REQUIRE(ids == (capacity == 1u ? std::set<uint64_t>{10u} : std::set<uint64_t>{10u, 20u}));
    }
}

void testMultipleTieAndCapacity()
{
    ovspatialquery_instance_handle_t instance = createInstance();
    registerBox(instance, 0.5);
    const std::vector<BoxInstance> boxes{
        {20u, {5.5, 0.5, 0.5}},
        {10u, {5.5, 0.5, 0.5}},
        {5u, {8.5, 0.5, 0.5}},
    };
    const ovspatialquery_population_handle_t population = buildBoxPopulation(instance, 1, boxes);

    ovspatialquery_raycast_desc_t ray{};
    ray.common                = commonQuery(population, OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE);
    ray.ray.unit_direction[0] = 1.0f;
    ray.ray.max_distance      = 20.0;

    // At full capacity every candidate is stored: the tied pair at distance
    // 5.0 (either row order) and the farther object at distance 8.0.
    ovspatialquery_raycast_hit_t        hits[3]{};
    ovspatialquery_raycast_hit_buffer_t full{hits, 3u};
    ovspatialquery_multiple_result_t    fullResult{};
    REQUIRE(ovspatialquery_raycast_multiple_sync(instance, &ray, &full, &fullResult).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(fullResult.source_ordinal == 1 && fullResult.stored_hit_count == 3u && fullResult.truncated == 0u);
    std::set<uint64_t> tiedIds;
    std::set<uint64_t> farIds;
    for (const ovspatialquery_raycast_hit_t& row : hits)
    {
        const uint64_t id       = row.data.bounds_candidate.identity.object_id.value;
        const double   distance = row.data.bounds_candidate.distance;
        if (nearlyEqual(distance, 5.0))
        {
            tiedIds.insert(id);
        }
        else if (nearlyEqual(distance, 8.0))
        {
            farIds.insert(id);
        }
    }
    REQUIRE(tiedIds == (std::set<uint64_t>{10u, 20u}));
    REQUIRE(farIds == (std::set<uint64_t>{5u}));

    // A buffer smaller than the complete match count reports the stored count,
    // sets truncated and keeps the nearest rows, of a tie the lower ID.
    expectTruncatedTie<ovspatialquery_raycast_hit_t, ovspatialquery_raycast_hit_buffer_t>(
        instance, ray, ovspatialquery_raycast_multiple_sync,
        [](const ovspatialquery_raycast_hit_t& row, uint64_t& id, double& distance)
        {
            REQUIRE(row.truth == OVSPATIALQUERY_HIT_BOUNDS_CANDIDATE);
            id       = row.data.bounds_candidate.identity.object_id.value;
            distance = row.data.bounds_candidate.distance;
        },
        5.0);

    // Sphere and box sweeps of a quarter unit along the boxes' centre line
    // meet the tied pair at 4.75 and the farther box at 7.75.
    const auto readSweep = [](const ovspatialquery_sphere_sweep_hit_t& row, uint64_t& id, double& distance)
    {
        REQUIRE(row.truth == OVSPATIALQUERY_HIT_EXACT_SURFACE);
        id       = row.data.exact.identity.object_id.value;
        distance = row.data.exact.distance;
    };
    ovspatialquery_sphere_sweep_desc_t sphere{};
    sphere.common                  = commonQuery(population, OVSPATIALQUERY_QUERY_EXACT);
    sphere.sweep.center[1]         = 0.5;
    sphere.sweep.center[2]         = 0.5;
    sphere.sweep.radius            = 0.25;
    sphere.sweep.unit_direction[0] = 1.0f;
    sphere.sweep.max_distance      = 20.0;
    expectTruncatedTie<ovspatialquery_sphere_sweep_hit_t, ovspatialquery_sphere_sweep_hit_buffer_t>(
        instance, sphere, ovspatialquery_sweep_sphere_multiple_sync, readSweep, 4.75);
    ovspatialquery_box_sweep_desc_t box{};
    box.common = sphere.common;
    for (int axis = 0; axis < 3; ++axis)
    {
        box.sweep.center[axis]       = sphere.sweep.center[axis];
        box.sweep.half_extents[axis] = 0.25;
    }
    box.sweep.rotation_xyzw[3]  = 1.0f;
    box.sweep.unit_direction[0] = 1.0f;
    box.sweep.max_distance      = 20.0;
    expectTruncatedTie<ovspatialquery_box_sweep_hit_t, ovspatialquery_box_sweep_hit_buffer_t>(
        instance, box, ovspatialquery_sweep_box_multiple_sync, readSweep, 4.75);

    destroyPopulationAndInstance(instance, {population}, true, false);
}

// A real miss reports zero stored hits without truncation; a rejected call
// leaves the caller's buffer and result struct untouched.
void testRaycastMultipleMissAndInvalidArgumentPreservesOutputs()
{
    ovspatialquery_instance_handle_t instance = createInstance();
    registerBox(instance, 0.5);
    const std::vector<BoxInstance>           boxes{{20u, {0.5, 0.5, 0.5}}};
    const ovspatialquery_population_handle_t population = buildBoxPopulation(instance, 7, boxes);

    ovspatialquery_raycast_desc_t ray{};
    ray.common                = commonQuery(population, OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE);
    ray.ray.origin[0]         = -2.0;
    ray.ray.origin[1]         = 2.0;
    ray.ray.unit_direction[0] = 1.0f;
    ray.ray.max_distance      = 10.0;

    ovspatialquery_raycast_hit_buffer_t emptyBuffer{nullptr, 0u};
    ovspatialquery_multiple_result_t    missResult{};
    REQUIRE(ovspatialquery_raycast_multiple_sync(instance, &ray, &emptyBuffer, &missResult).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(missResult.source_ordinal == 7 && missResult.stored_hit_count == 0u && missResult.truncated == 0u);

    ovspatialquery_raycast_desc_t invalid = ray;
    invalid.ray.unit_direction[0]         = 2.0f;
    ovspatialquery_raycast_hit_t hits[2];
    std::memset(hits, 0xa5, sizeof(hits));
    ovspatialquery_multiple_result_t before;
    std::memset(&before, 0xa5, sizeof(before));
    ovspatialquery_multiple_result_t    after = before;
    ovspatialquery_raycast_hit_buffer_t buffer{hits, 2u};
    const ovspatialquery_result_t status = ovspatialquery_raycast_multiple_sync(instance, &invalid, &buffer, &after);
    REQUIRE(status.status == OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(std::memcmp(&after, &before, sizeof(after)) == 0);
    REQUIRE(reinterpret_cast<const unsigned char*>(hits)[0] == 0xa5u);

    destroyPopulationAndInstance(instance, {population}, true, false);
}

// A minimal representative sample of dispatch-boundary validation: a handle
// from a foreign instance, a malformed target kind, a negative or NaN contact
// tolerance or one beyond the input domain, ray and sphere values beyond it,
// and a negative retained ordinal, each rejected before any
// generation is pinned. The tolerance is checked before the target handle is
// resolved; while the target request is built, the target kind and handle are
// checked first and the ordinal after them. The last step, an unretained
// ordinal, fails after resolution starts and must leave the output untouched
// as well. Not-ready targets are proven backend-neutrally by
// family_query_sync_test.cpp.
void testValidationRepresentative()
{
    ovspatialquery_instance_handle_t instance = createInstance();
    registerBox(instance, 0.5);
    const std::vector<BoxInstance>           boxes{{20u, {0.5, 0.5, 0.5}}};
    const ovspatialquery_population_handle_t population = buildBoxPopulation(instance, 7, boxes);

    ovspatialquery_instance_handle_t foreignInstance = createInstance();
    registerBox(foreignInstance, 0.5);
    const ovspatialquery_population_handle_t foreignPopulation = buildBoxPopulation(foreignInstance, 7, boxes);

    ovspatialquery_raycast_desc_t ray{};
    ray.common                = commonQuery(population, OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE);
    ray.ray.origin[0]         = -2.0;
    ray.ray.unit_direction[0] = 1.0f;
    ray.ray.max_distance      = 10.0;

    ovspatialquery_raycast_single_result_t sentinel;
    std::memset(&sentinel, 0xa5, sizeof(sentinel));

    ovspatialquery_raycast_desc_t foreignHandle          = ray;
    foreignHandle.common.target.target.handle.population = foreignPopulation;
    ovspatialquery_raycast_single_result_t foreignOut    = sentinel;
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &foreignHandle, &foreignOut).status ==
            OVSPATIALQUERY_API_INVALID_HANDLE);
    REQUIRE(std::memcmp(&foreignOut, &sentinel, sizeof(foreignOut)) == 0);

    ovspatialquery_raycast_desc_t malformed             = ray;
    malformed.common.target.target.kind                 = static_cast<ovspatialquery_target_kind_t>(99);
    ovspatialquery_raycast_single_result_t malformedOut = sentinel;
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &malformed, &malformedOut).status ==
            OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(std::memcmp(&malformedOut, &sentinel, sizeof(malformedOut)) == 0);

    // An overlap sphere that contains the box still overlaps it after a
    // negative tolerance shrinks the radius, so only the facade rejects it.
    ovspatialquery_overlap_desc_t overlap{};
    overlap.common                      = commonQuery(population, OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE);
    overlap.shape.kind                  = OVSPATIALQUERY_OVERLAP_SPHERE;
    overlap.shape.data.sphere.center[0] = 0.5;
    overlap.shape.data.sphere.center[1] = 0.5;
    overlap.shape.data.sphere.center[2] = 0.5;
    overlap.shape.data.sphere.radius    = 2.0;
    ovspatialquery_overlap_any_result_t contained{};
    REQUIRE(ovspatialquery_overlap_any_sync(instance, &overlap, &contained).status == OVSPATIALQUERY_API_SUCCESS &&
            contained.has_hit == 1u);
    ovspatialquery_overlap_any_result_t overlapSentinel;
    std::memset(&overlapSentinel, 0xa5, sizeof(overlapSentinel));
    for (const double tolerance : {-1.0, std::numeric_limits<double>::quiet_NaN(), 0x1p61})
    {
        overlap.common.contact_tolerance                 = tolerance;
        ovspatialquery_overlap_any_result_t toleranceOut = overlapSentinel;
        REQUIRE(ovspatialquery_overlap_any_sync(instance, &overlap, &toleranceOut).status ==
                OVSPATIALQUERY_API_INVALID_ARGUMENT);
        REQUIRE(std::memcmp(&toleranceOut, &overlapSentinel, sizeof(toleranceOut)) == 0);
    }
    // The input domain (ADR-0032): a finite value beyond 2^60 caller units is
    // malformed like a nonfinite one, while values at the bound are answered.
    overlap.common.contact_tolerance                 = 0.0;
    overlap.shape.data.sphere.radius                 = 0x1p61;
    ovspatialquery_overlap_any_result_t farSphereOut = overlapSentinel;
    REQUIRE(ovspatialquery_overlap_any_sync(instance, &overlap, &farSphereOut).status ==
            OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(std::memcmp(&farSphereOut, &overlapSentinel, sizeof(farSphereOut)) == 0);
    ovspatialquery_raycast_desc_t far = ray;
    far.ray.origin[0]                 = -0x1p60;
    far.ray.max_distance              = 0x1p60;
    ovspatialquery_raycast_single_result_t farOut{};
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &far, &farOut).status == OVSPATIALQUERY_API_SUCCESS &&
            farOut.has_hit == 1u);
    for (const double origin : {-0x1p61, -2.0})
    {
        far.ray.origin[0]    = origin;
        far.ray.max_distance = origin == -2.0 ? 0x1p61 : 0x1p60;
        farOut               = sentinel;
        REQUIRE(ovspatialquery_raycast_closest_sync(instance, &far, &farOut).status ==
                OVSPATIALQUERY_API_INVALID_ARGUMENT);
        REQUIRE(std::memcmp(&farOut, &sentinel, sizeof(farOut)) == 0);
    }
    ovspatialquery_raycast_desc_t negativeOrdinal             = ray;
    negativeOrdinal.common.target.ordinal.kind                = OVSPATIALQUERY_ORDINAL_RETAINED_EXACT;
    negativeOrdinal.common.target.ordinal.source_ordinal      = -1;
    ovspatialquery_raycast_single_result_t negativeOrdinalOut = sentinel;
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &negativeOrdinal, &negativeOrdinalOut).status ==
            OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(std::memcmp(&negativeOrdinalOut, &sentinel, sizeof(negativeOrdinalOut)) == 0);
    // A failure found while running the query, after the target is pinned,
    // leaves the output untouched too.
    ovspatialquery_raycast_desc_t unretained             = ray;
    unretained.common.target.ordinal.kind                = OVSPATIALQUERY_ORDINAL_RETAINED_EXACT;
    unretained.common.target.ordinal.source_ordinal      = 999;
    ovspatialquery_raycast_single_result_t unretainedOut = sentinel;
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &unretained, &unretainedOut).status ==
            OVSPATIALQUERY_API_ORDINAL_NOT_RETAINED);
    REQUIRE(std::memcmp(&unretainedOut, &sentinel, sizeof(unretainedOut)) == 0);

    destroyPopulationAndInstance(foreignInstance, {foreignPopulation}, true, false);
    destroyPopulationAndInstance(instance, {population}, true, false);
}

// Double-precision correctness through real PhysX at scales an oracle-driven
// unit test would not exercise: a sub-unit exact hit, a grazing miss one ULP
// past the object's edge, and a hit found across a ray origin/distance offset
// of roughly 1e12 without cancellation error.
void testDoublePrecisionBoundary()
{
    ovspatialquery_instance_handle_t instance = createInstance();
    registerBox(instance, 0.05);
    const std::vector<BoxInstance>           boxes{{101u, {0.15, 0.15, 0.15}}};
    const ovspatialquery_population_handle_t population = buildBoxPopulation(instance, 11, boxes);

    ovspatialquery_raycast_desc_t ray{};
    ray.common                = commonQuery(population, OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE);
    ray.ray.origin[0]         = -0.1;
    ray.ray.origin[1]         = 0.15;
    ray.ray.origin[2]         = 0.15;
    ray.ray.unit_direction[0] = 1.0f;
    ray.ray.max_distance      = 1.0;

    ovspatialquery_raycast_single_result_t exactHit{};
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &exactHit).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(exactHit.source_ordinal == 11 && exactHit.has_hit == 1u);
    REQUIRE(exactHit.hit.data.bounds_candidate.identity.object_id.value == 101u);
    REQUIRE(std::abs(exactHit.hit.data.bounds_candidate.distance - 0.2) < 1.0e-12);
    // The box's world AABB min is computed as center - half_extent
    // (0.15 - 0.05), which does not round to the literal 0.1 bit-for-bit, so
    // this compares within a tight epsilon rather than by exact equality.
    REQUIRE(nearlyEqual(exactHit.hit.data.bounds_candidate.bounds_position[0], 0.1, 1.0e-15));

    ovspatialquery_raycast_desc_t grazingMiss = ray;
    grazingMiss.ray.origin[0]                 = -1.0;
    grazingMiss.ray.origin[1]                 = std::nextafter(0.2, std::numeric_limits<double>::infinity());
    ovspatialquery_raycast_single_result_t grazeResult{};
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &grazingMiss, &grazeResult).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(grazeResult.source_ordinal == 11 && grazeResult.has_hit == 0u);

    ovspatialquery_raycast_desc_t offsetRay = ray;
    offsetRay.ray.origin[0]                 = -1.0e12;
    offsetRay.ray.min_distance              = 1.0e12;
    offsetRay.ray.max_distance              = 1.0e12 + 1.0;
    ovspatialquery_raycast_single_result_t offsetResult{};
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &offsetRay, &offsetResult).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(offsetResult.has_hit == 1u);
    REQUIRE(offsetResult.hit.data.bounds_candidate.identity.object_id.value == 101u);
    // A tight-but-not-bit-exact tolerance: this proves the distance is found
    // without catastrophic cancellation at this magnitude, without depending
    // on an exact internal rounding order this test cannot observe.
    REQUIRE(nearlyEqual(offsetResult.hit.data.bounds_candidate.distance, 1000000000000.1, 1.0e-3));
    REQUIRE(nearlyEqual(offsetResult.hit.data.bounds_candidate.bounds_position[0], 0.1, 1.0e-15));

    destroyPopulationAndInstance(instance, {population}, true, false);
}

// A population published with zero objects builds a valid, queryable
// (always-miss) real PhysX acceleration structure.
void testEmptyPopulationMiss()
{
    ovspatialquery_instance_handle_t instance = createInstance();
    registerBox(instance, 0.5);
    const ovspatialquery_population_handle_t population = buildBoxPopulation(instance, 3, {});

    ovspatialquery_raycast_desc_t ray{};
    ray.common                = commonQuery(population, OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE);
    ray.ray.unit_direction[0] = 1.0f;
    ray.ray.max_distance      = 10.0;
    ovspatialquery_raycast_single_result_t result{};
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &result).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.source_ordinal == 3 && result.has_hit == 0u);

    destroyPopulationAndInstance(instance, {population}, true, false);
}

// Blocks PhysX rays so a public raycast_closest_sync is still in flight when
// a concurrent population update publishes a new head, then confirms the
// in-flight query stayed pinned to the version it read at admission while a
// fresh query afterward observes the update: reusable acceleration data does
// not shift out from under a query already running against it.
//
// ovspatialquery_create_instance always binds the process-wide default PhysX
// runtime (acquireDefaultBackendRuntime in ovspatialquery_backend.h, called
// from source/ovspatialquery_instance.cpp), so this test reaches that same
// runtime instance through its own extra reference to arm and disarm the ray
// gate; releasing that reference afterward only drops this test's own count,
// not the instance's.
void testInFlightQuerySeesPreUpdateVersion()
{
    ovspatialquery_instance_handle_t instance = createInstance();
    registerBox(instance, 0.5);
    const ovspatialquery_population_handle_t population = buildBoxPopulation(instance, 7, {{10u, {0.5, 0.5, 0.5}}});

    ovspatialquery_raycast_desc_t ray{};
    ray.common                = commonQuery(population, OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE);
    ray.ray.origin[0]         = -2.0;
    ray.ray.origin[1]         = 0.5;
    ray.ray.origin[2]         = 0.5;
    ray.ray.unit_direction[0] = 1.0f;
    ray.ray.max_distance      = 10.0;

    ovspatialquery_backend_runtime_t* runtime = nullptr;
    REQUIRE(ovspatialquery::detail::acquireDefaultBackendRuntime(&runtime) == OVSPATIALQUERY_API_SUCCESS);

    ovspatialquery_physx_test_block_rays(runtime, 1u);
    ovspatialquery_result_t                pinnedStatus{};
    ovspatialquery_raycast_single_result_t pinned{};
    std::thread queryThread([&]() { pinnedStatus = ovspatialquery_raycast_closest_sync(instance, &ray, &pinned); });
    REQUIRE(ovspatialquery_physx_test_wait_for_blocked_rays(runtime) == 1u);

    // Complete AABB replacement while the in-flight query still holds the
    // ordinal-7 version: object 10 leaves, object 40 takes its place.
    updateBoxPopulation(instance, population, 7, 8, {{40u, {2.5, 0.5, 0.5}}}, {10u});

    ovspatialquery_physx_test_release_rays(runtime);
    queryThread.join();
    ovspatialquery::detail::releaseDefaultBackendRuntime(runtime);

    REQUIRE(pinnedStatus.status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(pinned.source_ordinal == 7 && pinned.has_hit == 1u);
    REQUIRE(pinned.hit.data.bounds_candidate.identity.object_id.value == 10u);

    ovspatialquery_raycast_single_result_t after{};
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &after).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(after.source_ordinal == 8 && after.has_hit == 1u);
    REQUIRE(after.hit.data.bounds_candidate.identity.object_id.value == 40u);

    destroyPopulationAndInstance(instance, {population}, true, false);
}

} // namespace

int main()
{
    testRaycastFamilyCorrectness();
    testStridedCpuRaycastColumns();
    testExactSingleRayModesToleranceAndAdmission();
    testSweepFamilyCorrectness();
    testMultipleTieAndCapacity();
    testRaycastMultipleMissAndInvalidArgumentPreservesOutputs();
    testValidationRepresentative();
    testDoublePrecisionBoundary();
    testEmptyPopulationMiss();
    testInFlightQuerySeesPreUpdateVersion();
    std::puts("query api contract test passed");
    return 0;
}
