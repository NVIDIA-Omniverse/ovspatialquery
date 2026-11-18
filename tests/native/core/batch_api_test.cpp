/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../support/native/float64_box_row_scene.h"
#include "../../support/native/public_api_inputs.h"
#include "../../support/native/test_failure.h"
#include "oracle_backend.h"
#include <ovspatialquery/ovspatialquery_explicit.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

// Arms the next operation a CPU plan admits to fail with exhausted resources
// (built only into test runtimes): no real input produces that failure on demand.
extern "C" ovspatialquery_api_status_t ovspatialquery_test_fail_next_family_plan_resource(
    ovspatialquery_instance_handle_t instance, ovspatialquery_batch_plan_handle_t plan);

// Reusable raycast, sphere-sweep, and overlap batch plans through the public
// API only: closest/any/multiple hit modes, hit-range offsets/counts/
// truncation, candidate and exact contact columns, absent unrequested columns,
// reservation-to-fetch backing, copied-vs-borrowed input ownership,
// fetch repeatability and output preservation, and admission/plan-creation
// validation. Populations are built from registered box definitions (a box
// with identity rotation reproduces any AABB) published through the explicit
// source adapter.

namespace
{

template <typename T>
bool isBytePattern(const T& value, unsigned char byte)
{
    std::vector<unsigned char> expected(sizeof(T), byte);
    return std::memcmp(&value, expected.data(), sizeof(T)) == 0;
}

template <typename T>
void fillBytePattern(T& value, unsigned char byte)
{
    std::memset(&value, byte, sizeof(T));
}

// A present CPU column with its documented dtype, rank and row count; a
// two-dimensional column is a row of three. An unrequested optional column is
// instead the all-zero absent descriptor, isBytePattern(tensor, 0u).
bool hasLayout(const DLTensor& tensor, uint8_t code, uint8_t bits, int32_t ndim, int64_t rows)
{
    return tensor.shape != nullptr && tensor.device.device_type == kDLCPU && tensor.dtype.code == code &&
           tensor.dtype.bits == bits && tensor.dtype.lanes == 1u && tensor.ndim == ndim && tensor.shape[0] == rows &&
           (ndim == 1 || tensor.shape[1] == 3);
}

// A reservation and its fetched result describe one backing store: equal data,
// dtype, device, strides and byte_offset, separate shape and strides arrays,
// and shape[0] moving from the capacity C to the stored count H.
bool sharesBacking(const DLTensor& reserved, const DLTensor& fetched, int64_t capacity, int64_t stored)
{
    return reserved.data == fetched.data && reserved.dtype.code == fetched.dtype.code &&
           reserved.dtype.bits == fetched.dtype.bits && reserved.device.device_type == fetched.device.device_type &&
           reserved.ndim == fetched.ndim && reserved.byte_offset == fetched.byte_offset &&
           reserved.strides != fetched.strides && reserved.strides[0] == fetched.strides[0] &&
           reserved.shape != fetched.shape && reserved.shape[0] == capacity && fetched.shape[0] == stored;
}

ovspatialquery_instance_handle_t createInstance(uint32_t operationCapacity, uint32_t maxCpuConcurrency,
                                                uint64_t hostBudget)
{
    const ovspatialquery_instance_desc_t desc{operationCapacity, maxCpuConcurrency, hostBudget, nullptr, 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(instance != OVSPATIALQUERY_INVALID_INSTANCE);
    return instance;
}

// One box-shaped object, described by its world AABB. A box definition with
// identity rotation reproduces the AABB exactly: half_extents is half the
// AABB size and the instance translation is its center.
struct BoxObject
{
    uint64_t               id;
    std::array<double, 3u> minCorner;
    std::array<double, 3u> maxCorner;
};

// Definition values registered for the current test instance; destroyed with it.
std::vector<uint64_t> gRegisteredBoxDefinitions;

void registerBoxes(ovspatialquery_instance_handle_t instance, const std::vector<BoxObject>& boxes)
{
    for (const BoxObject& box : boxes)
    {
        gRegisteredBoxDefinitions.push_back(box.id);
    }
    std::vector<uint64_t> values;
    std::vector<double>   halfExtents;
    for (const BoxObject& box : boxes)
    {
        values.push_back(box.id);
        for (int axis = 0; axis < 3; ++axis)
        {
            halfExtents.push_back(0.5 * (box.maxCorner[axis] - box.minCorner[axis]));
        }
    }
    const int64_t                               count = static_cast<int64_t>(boxes.size());
    InputArray                                  valueArray(values.data(), count, kDLUInt);
    InputArray                                  extentArray(halfExtents.data(), count, 3, kDLFloat);
    const ovspatialquery_box_definition_table_t table{&valueArray.value, &extentArray.value};
    waitAndDestroy(instance, ovspatialquery_enqueue_box_registration(instance, &table));
}

// Destroys every box definition the test registered, then the instance; the
// instance reports BUSY while any definition is still live.
void destroyInstanceAndDefinitions(ovspatialquery_instance_handle_t instance)
{
    for (uint64_t value : gRegisteredBoxDefinitions)
    {
        REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_BOX, value}).status ==
                OVSPATIALQUERY_API_SUCCESS);
    }
    gRegisteredBoxDefinitions.clear();
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

// Registers each box's geometry and publishes one instance per object, whose
// geometry definition value equals the object id (unique per test population).
ovspatialquery_population_handle_t publishBoxPopulation(ovspatialquery_instance_handle_t instance,
                                                        ovspatialquery_ordinal_t         ordinal,
                                                        const std::vector<BoxObject>&    boxes)
{
    registerBoxes(instance, boxes);
    const ovspatialquery_population_handle_t population = createExplicitPopulation(instance, 1u);
    InstanceRows                             rows;
    for (const BoxObject& box : boxes)
    {
        rows.add(box.id, OVSPATIALQUERY_GEOMETRY_BOX, box.id, 0.5 * (box.minCorner[0] + box.maxCorner[0]),
                 0.5 * (box.minCorner[1] + box.maxCorner[1]), 0.5 * (box.minCorner[2] + box.maxCorner[2]));
    }
    const ovspatialquery_explicit_population_set_t set{ordinal, rows.table(), nullptr, 0u};
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));
    return population;
}

ovspatialquery_ordinal_selection_t headOrdinal()
{
    ovspatialquery_ordinal_selection_t ordinal{};
    ordinal.kind = OVSPATIALQUERY_ORDINAL_HEAD;
    return ordinal;
}

ovspatialquery_batch_plan_handle_t createRaycastPlan(ovspatialquery_instance_handle_t instance,
                                                     ovspatialquery_target_ref_t      target,
                                                     ovspatialquery_hit_mode_t hitMode, uint64_t maximumQueryCount,
                                                     uint64_t maximumHitsPerQuery, uint64_t hitCapacityPerSlot,
                                                     uint32_t                    resultFields,
                                                     ovspatialquery_api_status_t expected = OVSPATIALQUERY_API_SUCCESS,
                                                     ovspatialquery_mesh_hits_t meshHits = OVSPATIALQUERY_MESH_HITS_ALL)
{
    ovspatialquery_raycast_plan_desc_t desc{};
    desc.target                               = target;
    desc.hit_mode                             = hitMode;
    desc.mesh_hits                            = meshHits;
    desc.precision                            = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    desc.device                               = DLDevice{kDLCPU, 0};
    desc.maximum_query_count                  = maximumQueryCount;
    desc.maximum_hits_per_query               = maximumHitsPerQuery;
    desc.hit_capacity_per_slot                = hitCapacityPerSlot;
    desc.result_slot_count                    = 2u;
    desc.result_fields                        = resultFields;
    ovspatialquery_batch_plan_handle_t plan   = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    const ovspatialquery_api_status_t  status = ovspatialquery_create_raycast_plan(instance, &desc, &plan).status;
    REQUIRE(status == expected);
    REQUIRE((plan != OVSPATIALQUERY_INVALID_BATCH_PLAN) == (expected == OVSPATIALQUERY_API_SUCCESS));
    return plan;
}

ovspatialquery_batch_plan_handle_t createSphereSweepPlan(
    ovspatialquery_instance_handle_t instance, ovspatialquery_target_ref_t target, double radius,
    uint64_t maximumQueryCount, uint32_t resultFields,
    ovspatialquery_api_status_t expected = OVSPATIALQUERY_API_SUCCESS)
{
    ovspatialquery_sphere_sweep_plan_desc_t desc{};
    desc.target                             = target;
    desc.precision                          = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    desc.radius                             = radius;
    desc.device                             = DLDevice{kDLCPU, 0};
    desc.maximum_query_count                = maximumQueryCount;
    desc.result_slot_count                  = 2u;
    desc.result_fields                      = resultFields;
    desc.hit_capacity_per_slot              = maximumQueryCount;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_sphere_sweep_plan(instance, &desc, &plan).status == expected);
    REQUIRE((plan != OVSPATIALQUERY_INVALID_BATCH_PLAN) == (expected == OVSPATIALQUERY_API_SUCCESS));
    return plan;
}

ovspatialquery_batch_plan_handle_t createOverlapPlan(
    ovspatialquery_instance_handle_t instance, ovspatialquery_target_ref_t target, ovspatialquery_hit_mode_t hitMode,
    ovspatialquery_overlap_batch_shape_t shape, uint64_t maximumQueryCount, uint64_t maximumHitsPerQuery,
    uint64_t hitCapacityPerSlot, uint32_t resultFields,
    ovspatialquery_query_precision_t precision = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE,
    ovspatialquery_api_status_t      expected  = OVSPATIALQUERY_API_SUCCESS)
{
    ovspatialquery_overlap_plan_desc_t desc{};
    desc.target                             = target;
    desc.hit_mode                           = hitMode;
    desc.precision                          = precision;
    desc.shape                              = shape;
    desc.device                             = DLDevice{kDLCPU, 0};
    desc.maximum_query_count                = maximumQueryCount;
    desc.maximum_hits_per_query             = maximumHitsPerQuery;
    desc.hit_capacity_per_slot              = hitCapacityPerSlot;
    desc.result_slot_count                  = 2u;
    desc.result_fields                      = resultFields;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_overlap_plan(instance, &desc, &plan).status == expected);
    REQUIRE((plan != OVSPATIALQUERY_INVALID_BATCH_PLAN) == (expected == OVSPATIALQUERY_API_SUCCESS));
    return plan;
}

ovspatialquery_op_id_t enqueueRaycast(ovspatialquery_instance_handle_t            instance,
                                      ovspatialquery_batch_plan_handle_t          plan,
                                      const ovspatialquery_raycast_batch_t&       batch,
                                      ovspatialquery_raycast_batch_reservation_t& reservation)
{
    const ovspatialquery_enqueue_result_t admission =
        ovspatialquery_enqueue_raycast_plan(instance, plan, &batch, &reservation);
    REQUIRE(admission.status == OVSPATIALQUERY_API_SUCCESS && admission.op_id != OVSPATIALQUERY_INVALID_OP);
    return admission.op_id;
}

// Waits for an operation that must complete and fetches its result.
template <typename Result>
Result waitAndFetch(ovspatialquery_instance_handle_t instance, ovspatialquery_op_id_t operation,
                    ovspatialquery_result_t (*fetch)(ovspatialquery_instance_handle_t, ovspatialquery_op_id_t, Result*))
{
    ovspatialquery_op_wait_result_t wait{};
    REQUIRE(ovspatialquery_wait_op(instance, operation, OVSPATIALQUERY_TIMEOUT_INFINITE, &wait).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(wait.state == OVSPATIALQUERY_OP_COMPLETED);
    // Prefilled so an absent descriptor reads as zero only if fetch wrote it.
    Result result{};
    fillBytePattern(result, 0xA5u);
    REQUIRE(fetch(instance, operation, &result).status == OVSPATIALQUERY_API_SUCCESS);
    return result;
}

void destroyOperation(ovspatialquery_instance_handle_t instance, ovspatialquery_op_id_t operation)
{
    REQUIRE(ovspatialquery_destroy_op(instance, &operation).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(operation == OVSPATIALQUERY_INVALID_OP);
}

// Zero-row and single-row closest raycast batches: hit-range shapes, the
// admitted ordinal reported even for a zero-row launch, and the candidate
// contact columns for a hit.
//
// Already covered elsewhere: ANY-mode single-row admission/reservation/fetch
// shape basics (publication_status_test's requireRaycastAndSweepReservationRuntime).
void testZeroAndClosestRayBatch()
{
    const ovspatialquery_instance_handle_t   instance = createInstance(4u, 2u, UINT64_C(64) << 20);
    const ovspatialquery_population_handle_t population =
        publishBoxPopulation(instance, 7, {{77u, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}});
    const ovspatialquery_target_ref_t        target = populationRef(population);
    const ovspatialquery_batch_plan_handle_t plan =
        createRaycastPlan(instance, target, OVSPATIALQUERY_HIT_MODE_CLOSEST, 1u, 1u, 1u,
                          OVSPATIALQUERY_BATCH_RESULT_POSITION | OVSPATIALQUERY_BATCH_RESULT_NORMAL |
                              OVSPATIALQUERY_BATCH_RESULT_DISTANCE);

    // Zero submitted rows still pins and reports the target ordinal.
    ovspatialquery_raycast_batch_t zeroBatch{};
    zeroBatch.ordinal     = headOrdinal();
    zeroBatch.query_count = 0u;
    ovspatialquery_raycast_batch_reservation_t zeroReservation{};
    const ovspatialquery_op_id_t zeroOperation = enqueueRaycast(instance, plan, zeroBatch, zeroReservation);
    const ovspatialquery_raycast_batch_result_t zeroResult =
        waitAndFetch(instance, zeroOperation, ovspatialquery_fetch_raycast_batch);
    REQUIRE(zeroResult.common.query_count == 0u);
    REQUIRE(zeroResult.common.source_ordinal == 7);
    REQUIRE(zeroResult.common.truth == OVSPATIALQUERY_HIT_BOUNDS_CANDIDATE);
    REQUIRE(zeroResult.columns.hit_ranges.query_status.shape[0] == 0);
    REQUIRE(*static_cast<const uint64_t*>(zeroResult.columns.hit_ranges.stored_hit_count.data) == 0u);
    REQUIRE(zeroResult.columns.object_ids.shape[0] == 0);
    REQUIRE(zeroResult.columns.contact.candidate.bounds_positions.shape[0] == 0);
    destroyOperation(instance, zeroOperation);

    // A single row hits the box's world AABB face.
    double                         origin[3]    = {-1.0, 0.5, 0.5};
    float                          direction[3] = {1.0f, 0.0f, 0.0f};
    double                         minDistance  = 0.0;
    double                         maxDistance  = 10.0;
    InputArray                     originArray(origin, 1, 3, kDLFloat);
    InputArray                     directionArray(direction, 1, 3, kDLFloat);
    InputArray                     minArray(&minDistance, 1, kDLFloat);
    InputArray                     maxArray(&maxDistance, 1, kDLFloat);
    ovspatialquery_raycast_batch_t closestBatch{};
    closestBatch.ordinal         = headOrdinal();
    closestBatch.query_count     = 1u;
    closestBatch.origins         = &originArray.value;
    closestBatch.unit_directions = &directionArray.value;
    closestBatch.min_distances   = &minArray.value;
    closestBatch.max_distances   = &maxArray.value;
    ovspatialquery_raycast_batch_reservation_t closestReservation{};
    const ovspatialquery_op_id_t closestOperation = enqueueRaycast(instance, plan, closestBatch, closestReservation);
    const ovspatialquery_raycast_batch_result_t closestResult =
        waitAndFetch(instance, closestOperation, ovspatialquery_fetch_raycast_batch);
    REQUIRE(closestResult.common.source_ordinal == 7);
    REQUIRE(*static_cast<const uint64_t*>(closestResult.columns.hit_ranges.stored_hit_count.data) == 1u);
    REQUIRE(static_cast<const uint32_t*>(closestResult.columns.hit_ranges.query_status.data)[0] ==
            OVSPATIALQUERY_BATCH_ROW_SUCCESS);
    REQUIRE(static_cast<const uint64_t*>(closestResult.columns.hit_ranges.hit_offsets.data)[0] == 0u);
    REQUIRE(static_cast<const uint64_t*>(closestResult.columns.hit_ranges.hit_counts.data)[0] == 1u);
    REQUIRE(static_cast<const uint8_t*>(closestResult.columns.hit_ranges.truncated.data)[0] == 0u);
    REQUIRE(static_cast<const uint64_t*>(closestResult.columns.object_ids.data)[0] == 77u);
    REQUIRE(hasLayout(closestResult.columns.contact.candidate.bounds_positions, kDLFloat, 64u, 2, 1));
    const double* position = static_cast<const double*>(closestResult.columns.contact.candidate.bounds_positions.data);
    const float*  normal   = static_cast<const float*>(closestResult.columns.contact.candidate.bounds_normals.data);
    const double* distance = static_cast<const double*>(closestResult.columns.contact.candidate.distances.data);
    if (!(position[0] == 0.0 && position[1] == 0.5 && position[2] == 0.5 && normal[0] == -1.0f && normal[1] == 0.0f &&
          normal[2] == 0.0f && distance[0] == 1.0))
    {
        std::fprintf(stderr,
                     "closest candidate hit: position (%g, %g, %g) normal (%g, %g, %g) "
                     "distance %g\n",
                     position[0], position[1], position[2], static_cast<double>(normal[0]),
                     static_cast<double>(normal[1]), static_cast<double>(normal[2]), distance[0]);
    }
    REQUIRE(position[0] == 0.0 && position[1] == 0.5 && position[2] == 0.5 && normal[0] == -1.0f && normal[1] == 0.0f &&
            normal[2] == 0.0f && distance[0] == 1.0);
    destroyOperation(instance, closestOperation);

    // A plan-fixed contact tolerance widens every row's hit test: the box face
    // 2.0 along this ray lies beyond max_distance 1.75 and is reached only
    // through the plan's 0.25 tolerance.
    ovspatialquery_raycast_plan_desc_t tolerantDesc{};
    tolerantDesc.target                             = target;
    tolerantDesc.hit_mode                           = OVSPATIALQUERY_HIT_MODE_CLOSEST;
    tolerantDesc.precision                          = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    tolerantDesc.device                             = DLDevice{kDLCPU, 0};
    tolerantDesc.maximum_query_count                = 1u;
    tolerantDesc.maximum_hits_per_query             = 1u;
    tolerantDesc.hit_capacity_per_slot              = 1u;
    tolerantDesc.result_slot_count                  = 1u;
    tolerantDesc.contact_tolerance                  = 0.25;
    ovspatialquery_batch_plan_handle_t tolerantPlan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_raycast_plan(instance, &tolerantDesc, &tolerantPlan).status ==
            OVSPATIALQUERY_API_SUCCESS);
    origin[0]   = -2.0;
    maxDistance = 1.75;
    ovspatialquery_raycast_batch_reservation_t tolerantReservation{};
    const ovspatialquery_op_id_t               tolerantOperation =
        enqueueRaycast(instance, tolerantPlan, closestBatch, tolerantReservation);
    const ovspatialquery_raycast_batch_result_t tolerantResult =
        waitAndFetch(instance, tolerantOperation, ovspatialquery_fetch_raycast_batch);
    REQUIRE(*static_cast<const uint64_t*>(tolerantResult.columns.hit_ranges.stored_hit_count.data) == 1u);
    REQUIRE(static_cast<const uint64_t*>(tolerantResult.columns.object_ids.data)[0] == 77u);
    destroyOperation(instance, tolerantOperation);
    REQUIRE(ovspatialquery_destroy_batch_plan(instance, tolerantPlan).status == OVSPATIALQUERY_API_SUCCESS);

    REQUIRE(ovspatialquery_destroy_batch_plan(instance, plan).status == OVSPATIALQUERY_API_SUCCESS);
    const ovspatialquery_result_t populationDestroy = ovspatialquery_destroy_population(instance, population);
    if (populationDestroy.status != OVSPATIALQUERY_API_SUCCESS)
    {
        std::fprintf(stderr, "destroy_population after the plan returned %u\n",
                     static_cast<unsigned>(populationDestroy.status));
    }
    REQUIRE(populationDestroy.status == OVSPATIALQUERY_API_SUCCESS);
    destroyInstanceAndDefinitions(instance);
}

// A raycast launch long enough for an instance's workers to share. Row i crosses box 20
// and box 10 at y = 0.5, only box 10 (starting between them) or nothing, by i % 3. In
// MULTIPLE mode the slot capacity runs out part way, and rows are granted it in query order,
// as in a launch of a few rows. Rows keep one hit per object (NEAREST), whose execution fits
// the peak admission charges; every-crossing rows charge their buffers as they grow
// (ADR-0049). A CLOSEST plan holds a hit for every row: box 20 or box 10 by i % 3.
constexpr uint64_t kLongRows = 512u, kLongCapacity = 400u;

ovspatialquery_batch_plan_handle_t createLongRayPlan(
    ovspatialquery_instance_handle_t instance, ovspatialquery_target_ref_t target,
    ovspatialquery_hit_mode_t hitMode = OVSPATIALQUERY_HIT_MODE_MULTIPLE)
{
    const bool closest = hitMode == OVSPATIALQUERY_HIT_MODE_CLOSEST;
    return createRaycastPlan(instance, target, hitMode, kLongRows, closest ? 1u : 2u,
                             closest ? kLongRows : kLongCapacity, 0u, OVSPATIALQUERY_API_SUCCESS,
                             OVSPATIALQUERY_MESH_HITS_NEAREST);
}

// Launches the long rays and checks every row; false when admission refuses the launch
// for the host budget.
bool launchLongRays(ovspatialquery_instance_handle_t instance, ovspatialquery_batch_plan_handle_t plan,
                    bool closest = false)
{
    std::vector<double> origins, minimums(kLongRows, 0.0), maximums(kLongRows, 10.0);
    std::vector<float>  directions;
    for (uint64_t row = 0u; row < kLongRows; ++row)
    {
        const double start[3][3] = {{-2.0, 0.5, 0.5}, {2.0, 0.5, 0.5}, {-2.0, 10.0, 0.5}};
        origins.insert(origins.end(), start[row % 3u], start[row % 3u] + 3);
        directions.insert(directions.end(), {1.0f, 0.0f, 0.0f});
    }
    InputArray                     originArray(origins.data(), static_cast<int64_t>(kLongRows), 3, kDLFloat);
    InputArray                     directionArray(directions.data(), static_cast<int64_t>(kLongRows), 3, kDLFloat);
    InputArray                     minArray(minimums.data(), static_cast<int64_t>(kLongRows), kDLFloat);
    InputArray                     maxArray(maximums.data(), static_cast<int64_t>(kLongRows), kDLFloat);
    ovspatialquery_raycast_batch_t batch{};
    batch.ordinal         = headOrdinal();
    batch.query_count     = kLongRows;
    batch.origins         = &originArray.value;
    batch.unit_directions = &directionArray.value;
    batch.min_distances   = &minArray.value;
    batch.max_distances   = &maxArray.value;
    ovspatialquery_raycast_batch_reservation_t reservation{};
    const ovspatialquery_enqueue_result_t      admission =
        ovspatialquery_enqueue_raycast_plan(instance, plan, &batch, &reservation);
    if (admission.status == OVSPATIALQUERY_API_HOST_BUDGET)
    {
        return false;
    }
    REQUIRE(admission.status == OVSPATIALQUERY_API_SUCCESS);
    const ovspatialquery_raycast_batch_result_t result =
        waitAndFetch(instance, admission.op_id, ovspatialquery_fetch_raycast_batch);
    const uint64_t* offsets   = static_cast<const uint64_t*>(result.columns.hit_ranges.hit_offsets.data);
    const uint64_t* counts    = static_cast<const uint64_t*>(result.columns.hit_ranges.hit_counts.data);
    const uint8_t*  truncated = static_cast<const uint8_t*>(result.columns.hit_ranges.truncated.data);
    const uint64_t* ids       = static_cast<const uint64_t*>(result.columns.object_ids.data);
    uint64_t        granted   = 0u;
    for (uint64_t row = 0u; row < kLongRows; ++row)
    {
        const uint64_t crossed = row % 3u == 0u ? 2u : (row % 3u == 1u ? 1u : 0u);
        if (closest)
        {
            REQUIRE(counts[row] == (crossed != 0u ? 1u : 0u) && truncated[row] == 0u);
            REQUIRE(crossed == 0u || (offsets[row] == granted && ids[granted] == (crossed == 2u ? 20u : 10u)));
            granted += counts[row];
            continue;
        }
        const uint64_t kept = std::min(crossed, kLongCapacity - granted);
        REQUIRE(counts[row] == kept && truncated[row] == (kept < crossed ? 1u : 0u));
        REQUIRE(kept == 0u || offsets[row] == granted);
        REQUIRE(kept != 1u || crossed != 1u || ids[offsets[row]] == 10u);
        granted += kept;
    }
    const uint64_t expected = closest ? (kLongRows * 2u + 2u) / 3u : kLongCapacity;
    REQUIRE(granted == expected &&
            *static_cast<const uint64_t*>(result.columns.hit_ranges.stored_hit_count.data) == expected);
    destroyOperation(instance, admission.op_id);
    return true;
}

// MULTIPLE mode partitions two rows through disjoint boxes with a per-query
// hit cap that truncates the nearer row, and ANY mode returns one
// unspecified admitted hit per row.
void testRayMultipleAndAnyPartitions()
{
    const ovspatialquery_instance_handle_t   instance   = createInstance(4u, 2u, UINT64_C(64) << 20);
    const ovspatialquery_population_handle_t population = publishBoxPopulation(
        instance, 7, {{20u, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}, {10u, {3.0, 0.0, 0.0}, {4.0, 1.0, 1.0}}});
    const ovspatialquery_target_ref_t target = populationRef(population);

    double     origins[6]      = {-2.0, 0.5, 0.5, -2.0, 10.0, 0.5};
    float      directions[6]   = {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    double     minDistances[2] = {0.0, 0.0};
    double     maxDistances[2] = {10.0, 10.0};
    InputArray originArray(origins, 2, 3, kDLFloat);
    InputArray directionArray(directions, 2, 3, kDLFloat);
    InputArray minArray(minDistances, 2, kDLFloat);
    InputArray maxArray(maxDistances, 2, kDLFloat);

    // Row 0's ray enters both box AABBs; a per-query cap of one hit forces
    // truncation there while row 1 (offset in Y) misses everything.
    const ovspatialquery_batch_plan_handle_t multiplePlan = createRaycastPlan(
        instance, target, OVSPATIALQUERY_HIT_MODE_MULTIPLE, 2u, 1u, 2u, OVSPATIALQUERY_BATCH_RESULT_DISTANCE);
    ovspatialquery_raycast_batch_t multipleBatch{};
    multipleBatch.ordinal         = headOrdinal();
    multipleBatch.query_count     = 2u;
    multipleBatch.origins         = &originArray.value;
    multipleBatch.unit_directions = &directionArray.value;
    multipleBatch.min_distances   = &minArray.value;
    multipleBatch.max_distances   = &maxArray.value;
    ovspatialquery_raycast_batch_reservation_t multipleReservation{};
    fillBytePattern(multipleReservation, 0xA5u);
    const ovspatialquery_op_id_t multipleOperation =
        enqueueRaycast(instance, multiplePlan, multipleBatch, multipleReservation);
    const ovspatialquery_raycast_batch_result_t multipleResult =
        waitAndFetch(instance, multipleOperation, ovspatialquery_fetch_raycast_batch);
    const uint64_t* offsets   = static_cast<const uint64_t*>(multipleResult.columns.hit_ranges.hit_offsets.data);
    const uint64_t* counts    = static_cast<const uint64_t*>(multipleResult.columns.hit_ranges.hit_counts.data);
    const uint8_t*  truncated = static_cast<const uint8_t*>(multipleResult.columns.hit_ranges.truncated.data);
    REQUIRE(*static_cast<const uint64_t*>(multipleResult.columns.hit_ranges.stored_hit_count.data) == 1u);
    REQUIRE(offsets[0] == 0u && counts[0] == 1u && truncated[0] == 1u);
    REQUIRE(counts[1] == 0u && truncated[1] == 0u);
    REQUIRE(static_cast<const uint64_t*>(multipleResult.columns.object_ids.data)[0] == 20u);
    REQUIRE(static_cast<const double*>(multipleResult.columns.contact.candidate.distances.data)[0] == 2.0);

    // The reservation is Q- and C-shaped; the fetch keeps its backing and moves
    // only shape[0] from C to H. DISTANCE alone leaves the other contact columns,
    // the geometry identity group and the reserved descriptors absent in both.
    REQUIRE(multipleReservation.common.hit_capacity == 2u &&
            hasLayout(multipleReservation.columns.hit_ranges.query_status, kDLUInt, 32u, 1, 2) &&
            hasLayout(multipleReservation.columns.hit_ranges.stored_hit_count, kDLUInt, 64u, 1, 1) &&
            hasLayout(multipleReservation.columns.object_ids, kDLUInt, 64u, 1, 2));
    REQUIRE(sharesBacking(multipleReservation.columns.object_ids, multipleResult.columns.object_ids, 2, 1));
    REQUIRE(sharesBacking(multipleReservation.columns.contact.candidate.distances,
                          multipleResult.columns.contact.candidate.distances, 2, 1));
    REQUIRE(multipleReservation.columns.hit_ranges.query_status.data ==
            multipleResult.columns.hit_ranges.query_status.data);
    for (const ovspatialquery_raycast_batch_columns_t* columns :
         std::array<const ovspatialquery_raycast_batch_columns_t*, 2u>{&multipleReservation.columns,
                                                                       &multipleResult.columns})
    {
        REQUIRE(isBytePattern(columns->contact.candidate.bounds_positions, 0u) &&
                isBytePattern(columns->contact.candidate.bounds_normals, 0u) &&
                isBytePattern(columns->geometry.types, 0u) && isBytePattern(columns->geometry.values, 0u) &&
                isBytePattern(columns->reserved, 0u));
    }
    destroyOperation(instance, multipleOperation);
    REQUIRE(ovspatialquery_destroy_batch_plan(instance, multiplePlan).status == OVSPATIALQUERY_API_SUCCESS);

    const ovspatialquery_batch_plan_handle_t anyPlan =
        createRaycastPlan(instance, target, OVSPATIALQUERY_HIT_MODE_ANY, 2u, 1u, 2u, 0u);
    ovspatialquery_raycast_batch_t             anyBatch = multipleBatch;
    ovspatialquery_raycast_batch_reservation_t anyReservation{};
    const ovspatialquery_op_id_t anyOperation = enqueueRaycast(instance, anyPlan, anyBatch, anyReservation);
    const ovspatialquery_raycast_batch_result_t anyResult =
        waitAndFetch(instance, anyOperation, ovspatialquery_fetch_raycast_batch);
    const uint64_t* anyCounts    = static_cast<const uint64_t*>(anyResult.columns.hit_ranges.hit_counts.data);
    const uint8_t*  anyTruncated = static_cast<const uint8_t*>(anyResult.columns.hit_ranges.truncated.data);
    REQUIRE(anyCounts[0] == 1u && anyTruncated[0] == 0u);
    REQUIRE(anyCounts[1] == 0u);
    const uint64_t anyId = static_cast<const uint64_t*>(anyResult.columns.object_ids.data)[0];
    REQUIRE(anyId == 20u || anyId == 10u);
    // An identity-only plan publishes no contact column at all.
    REQUIRE(isBytePattern(anyResult.columns.contact.candidate.distances, 0u) &&
            isBytePattern(anyResult.columns.contact.candidate.bounds_positions, 0u));
    destroyOperation(instance, anyOperation);
    REQUIRE(ovspatialquery_destroy_batch_plan(instance, anyPlan).status == OVSPATIALQUERY_API_SUCCESS);

    // A launch long enough for the instance's workers to share.
    const ovspatialquery_batch_plan_handle_t longPlan = createLongRayPlan(instance, target);
    REQUIRE(launchLongRays(instance, longPlan));
    REQUIRE(ovspatialquery_destroy_batch_plan(instance, longPlan).status == OVSPATIALQUERY_API_SUCCESS);
    const ovspatialquery_batch_plan_handle_t closestPlan =
        createLongRayPlan(instance, target, OVSPATIALQUERY_HIT_MODE_CLOSEST);
    REQUIRE(launchLongRays(instance, closestPlan, true));
    REQUIRE(ovspatialquery_destroy_batch_plan(instance, closestPlan).status == OVSPATIALQUERY_API_SUCCESS);

    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    destroyInstanceAndDefinitions(instance);
}

// A long launch whose workers cannot all be charged their row execution peak still
// completes, row by row, with the same columns. The free budget is swept from below the
// launch's own admission charge upward in steps much smaller than one row peak over a
// thousand objects, by a ballast plan sized through the instance's measured usage; every
// admitted launch must complete with the in-order columns.
void testLongLaunchUnderTightBudgets()
{
    constexpr uint64_t                     kBudget  = UINT64_C(64) << 20;
    const ovspatialquery_instance_handle_t instance = createInstance(4u, 2u, kBudget);
    std::vector<BoxObject> boxes{{20u, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}, {10u, {3.0, 0.0, 0.0}, {4.0, 1.0, 1.0}}};
    for (uint64_t index = 0u; index < 1000u; ++index)
    {
        const double x = static_cast<double>(index % 40u) * 2.0, y = static_cast<double>(index / 40u) * 2.0;
        boxes.push_back({1000u + index, {x, y, 100.0}, {x + 1.0, y + 1.0, 101.0}});
    }
    const ovspatialquery_population_handle_t population = publishBoxPopulation(instance, 1, boxes);
    const ovspatialquery_target_ref_t        target     = populationRef(population);
    const ovspatialquery_batch_plan_handle_t plan       = createLongRayPlan(instance, target);
    const auto                               usage      = [instance]()
    {
        uint64_t bytes = 0u;
        REQUIRE(ovspatialquery_get_instance_memory_usage(instance, DLDevice{kDLCPU, 0}, &bytes).status ==
                OVSPATIALQUERY_API_SUCCESS);
        return bytes;
    };
    // A ballast raycast plan's charge, measured at two hit capacities.
    const uint64_t base   = usage();
    const auto     create = [instance, target](uint64_t capacity)
    { return createRaycastPlan(instance, target, OVSPATIALQUERY_HIT_MODE_MULTIPLE, 1u, capacity, capacity, 0u); };
    ovspatialquery_batch_plan_handle_t ballast = create(1u);
    const uint64_t                     fixed   = usage() - base;
    REQUIRE(ovspatialquery_destroy_batch_plan(instance, ballast).status == OVSPATIALQUERY_API_SUCCESS);
    ballast                  = create(UINT64_C(1) << 16);
    const uint64_t perHitX64 = ((usage() - base - fixed) << 6) / ((UINT64_C(1) << 16) - 1u);
    REQUIRE(ovspatialquery_destroy_batch_plan(instance, ballast).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(perHitX64 != 0u && base + fixed < kBudget);

    constexpr uint64_t kStep    = UINT64_C(8) << 10;
    uint64_t           admitted = 0u;
    for (uint64_t free = 0u; free <= (UINT64_C(2) << 20); free += kStep)
    {
        const uint64_t ballastBytes = kBudget - base - fixed - free;
        ballast                     = create(1u + (ballastBytes << 6) / perHitX64);
        admitted += launchLongRays(instance, plan) ? 1u : 0u;
        REQUIRE(ovspatialquery_destroy_batch_plan(instance, ballast).status == OVSPATIALQUERY_API_SUCCESS);
    }
    REQUIRE(admitted != 0u);
    REQUIRE(ovspatialquery_destroy_batch_plan(instance, plan).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    destroyInstanceAndDefinitions(instance);
}

// Sphere-sweep candidate contact columns (query position and object bounds
// position are distinct), and overlap sphere/box/AABB shapes including a
// multi-row overlap that exercises geometry-identity columns.
void testSweepAndOverlapFamilies()
{
    const ovspatialquery_instance_handle_t   instance   = createInstance(4u, 2u, UINT64_C(64) << 20);
    const ovspatialquery_population_handle_t population = publishBoxPopulation(
        instance, 7, {{20u, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}, {10u, {3.0, 0.0, 0.0}, {4.0, 1.0, 1.0}}});
    const ovspatialquery_target_ref_t target = populationRef(population);

    // A sphere sweep of radius 0.5 moving toward object 20's near face
    // touches it 1.5 units later, at query center x=-0.5 and object x=0.
    double                                   center[3]    = {-2.0, 0.5, 0.5};
    float                                    direction[3] = {1.0f, 0.0f, 0.0f};
    double                                   minDistance  = 0.0;
    double                                   maxDistance  = 10.0;
    InputArray                               centerArray(center, 1, 3, kDLFloat);
    InputArray                               directionArray(direction, 1, 3, kDLFloat);
    InputArray                               minArray(&minDistance, 1, kDLFloat);
    InputArray                               maxArray(&maxDistance, 1, kDLFloat);
    const ovspatialquery_batch_plan_handle_t sweepPlan =
        createSphereSweepPlan(instance, target, 0.5, 1u,
                              OVSPATIALQUERY_BATCH_RESULT_POSITION | OVSPATIALQUERY_BATCH_RESULT_NORMAL |
                                  OVSPATIALQUERY_BATCH_RESULT_DISTANCE | OVSPATIALQUERY_BATCH_RESULT_QUERY_POSITION);
    ovspatialquery_sphere_sweep_batch_t sweepBatch{};
    sweepBatch.ordinal         = headOrdinal();
    sweepBatch.query_count     = 1u;
    sweepBatch.centers         = &centerArray.value;
    sweepBatch.unit_directions = &directionArray.value;
    sweepBatch.min_distances   = &minArray.value;
    sweepBatch.max_distances   = &maxArray.value;
    ovspatialquery_sphere_sweep_batch_reservation_t sweepReservation{};
    const ovspatialquery_enqueue_result_t           sweepAdmission =
        ovspatialquery_enqueue_sphere_sweep_plan(instance, sweepPlan, &sweepBatch, &sweepReservation);
    REQUIRE(sweepAdmission.status == OVSPATIALQUERY_API_SUCCESS && sweepAdmission.op_id != OVSPATIALQUERY_INVALID_OP);
    const ovspatialquery_sphere_sweep_batch_result_t sweepResult =
        waitAndFetch(instance, sweepAdmission.op_id, ovspatialquery_fetch_sphere_sweep_batch);
    REQUIRE(sweepResult.common.source_ordinal == 7);
    REQUIRE(static_cast<const uint64_t*>(sweepResult.columns.object_ids.data)[0] == 20u);
    const double* boundsPosition =
        static_cast<const double*>(sweepResult.columns.contact.candidate.bounds_positions.data);
    const double* queryPosition =
        static_cast<const double*>(sweepResult.columns.contact.candidate.query_positions.data);
    const float*  boundsNormal  = static_cast<const float*>(sweepResult.columns.contact.candidate.bounds_normals.data);
    const double* sweepDistance = static_cast<const double*>(sweepResult.columns.contact.candidate.distances.data);
    REQUIRE(sweepDistance[0] == 1.5);
    REQUIRE(std::abs(boundsPosition[0] - 0.0) < 1.0e-12);
    REQUIRE(std::abs(queryPosition[0] - (-0.5)) < 1.0e-12);
    REQUIRE(boundsNormal[0] == -1.0f);
    destroyOperation(instance, sweepAdmission.op_id);
    REQUIRE(ovspatialquery_destroy_batch_plan(instance, sweepPlan).status == OVSPATIALQUERY_API_SUCCESS);

    // One plan accepts different radii and leaves a malformed value local to its row.
    // A radius beyond the input domain's 2^60 is malformed for a plan and for a row (the last one below).
    (void)createSphereSweepPlan(instance, target, 0x1p61, 4u, OVSPATIALQUERY_BATCH_RESULT_DISTANCE,
                                OVSPATIALQUERY_API_INVALID_ARGUMENT);
    double     variedCenters[12]    = {-2.0, 0.5, 0.5, -2.0, 0.5, 0.5, -2.0, 0.5, 0.5, -2.0, 0.5, 0.5};
    float      variedDirections[12] = {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    double     variedMaximums[4]    = {10.0, 10.0, 10.0, 10.0};
    double     variedRadii[4]       = {0.25, 0.75, 0.0, 0x1p61};
    InputArray variedCenterArray(variedCenters, 4, 3, kDLFloat);
    InputArray variedDirectionArray(variedDirections, 4, 3, kDLFloat);
    InputArray variedMaximumArray(variedMaximums, 4, kDLFloat);
    InputArray variedRadiusArray(variedRadii, 4, kDLFloat);
    const ovspatialquery_batch_plan_handle_t variedSweepPlan =
        createSphereSweepPlan(instance, target, 0.5, 4u, OVSPATIALQUERY_BATCH_RESULT_DISTANCE);
    ovspatialquery_sphere_sweep_batch_t variedBatch{};
    variedBatch.ordinal         = headOrdinal();
    variedBatch.query_count     = 4u;
    variedBatch.centers         = &variedCenterArray.value;
    variedBatch.unit_directions = &variedDirectionArray.value;
    variedBatch.max_distances   = &variedMaximumArray.value;
    variedBatch.radii           = &variedRadiusArray.value;
    ovspatialquery_sphere_sweep_batch_reservation_t variedReservation{};
    float                                           wrongWidthRadii[4] = {0.25f, 0.75f, 1.0f, 1.0f};
    InputArray                                      wrongWidthRadiusArray(wrongWidthRadii, 4, kDLFloat);
    variedBatch.radii = &wrongWidthRadiusArray.value;
    REQUIRE(
        ovspatialquery_enqueue_sphere_sweep_plan(instance, variedSweepPlan, &variedBatch, &variedReservation).status ==
        OVSPATIALQUERY_API_INVALID_ARGUMENT);
    variedBatch.radii = &variedRadiusArray.value;
    const ovspatialquery_enqueue_result_t variedAdmission =
        ovspatialquery_enqueue_sphere_sweep_plan(instance, variedSweepPlan, &variedBatch, &variedReservation);
    REQUIRE(variedAdmission.status == OVSPATIALQUERY_API_SUCCESS);
    const ovspatialquery_sphere_sweep_batch_result_t variedResult =
        waitAndFetch(instance, variedAdmission.op_id, ovspatialquery_fetch_sphere_sweep_batch);
    const auto* variedStatuses  = static_cast<const uint32_t*>(variedResult.columns.hit_ranges.query_status.data);
    const auto* variedCounts    = static_cast<const uint64_t*>(variedResult.columns.hit_ranges.hit_counts.data);
    const auto* variedDistances = static_cast<const double*>(variedResult.columns.contact.candidate.distances.data);
    REQUIRE(variedStatuses[0] == OVSPATIALQUERY_BATCH_ROW_SUCCESS);
    REQUIRE(variedStatuses[1] == OVSPATIALQUERY_BATCH_ROW_SUCCESS);
    REQUIRE(variedStatuses[2] == OVSPATIALQUERY_BATCH_ROW_INVALID_ARGUMENT);
    REQUIRE(variedStatuses[3] == OVSPATIALQUERY_BATCH_ROW_INVALID_ARGUMENT);
    REQUIRE(variedCounts[0] == 1u && variedCounts[1] == 1u && variedCounts[2] == 0u && variedCounts[3] == 0u);
    // A non-success row has a zero offset, not the running hit count.
    REQUIRE(static_cast<const uint64_t*>(variedResult.columns.hit_ranges.hit_offsets.data)[2] == 0u);
    REQUIRE(std::abs(variedDistances[0] - 1.75) < 1.0e-12);
    REQUIRE(std::abs(variedDistances[1] - 1.25) < 1.0e-12);
    destroyOperation(instance, variedAdmission.op_id);
    REQUIRE(ovspatialquery_destroy_batch_plan(instance, variedSweepPlan).status == OVSPATIALQUERY_API_SUCCESS);

    // The box keeps its orientation while moving, and a box plan has its own handle family.
    float                            boxRotation[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    InputArray                       sweepBoxRotationArray(boxRotation, 1, 4, kDLFloat);
    ovspatialquery_box_sweep_batch_t boxBatch{};
    boxBatch.ordinal         = headOrdinal();
    boxBatch.query_count     = 1u;
    boxBatch.centers         = &centerArray.value;
    boxBatch.rotations_xyzw  = &sweepBoxRotationArray.value;
    boxBatch.unit_directions = &directionArray.value;
    boxBatch.max_distances   = &maxArray.value;
    for (const ovspatialquery_query_precision_t precision :
         {OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE, OVSPATIALQUERY_QUERY_EXACT})
    {
        ovspatialquery_box_sweep_plan_desc_t boxPlanDesc{};
        boxPlanDesc.target              = target;
        boxPlanDesc.precision           = precision;
        boxPlanDesc.half_extents[0]     = 0.25;
        boxPlanDesc.half_extents[1]     = 0.25;
        boxPlanDesc.half_extents[2]     = 0.25;
        boxPlanDesc.device              = {kDLCPU, 0};
        boxPlanDesc.maximum_query_count = 1u;
        boxPlanDesc.result_slot_count   = 1u;
        boxPlanDesc.result_fields = OVSPATIALQUERY_BATCH_RESULT_DISTANCE | OVSPATIALQUERY_BATCH_RESULT_QUERY_POSITION |
                                    OVSPATIALQUERY_BATCH_RESULT_POSITION;
        boxPlanDesc.hit_capacity_per_slot          = 1u;
        ovspatialquery_batch_plan_handle_t boxPlan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
        REQUIRE(ovspatialquery_create_box_sweep_plan(instance, &boxPlanDesc, &boxPlan).status ==
                OVSPATIALQUERY_API_SUCCESS);
        ovspatialquery_box_sweep_batch_reservation_t boxReservation{};
        REQUIRE(ovspatialquery_enqueue_sphere_sweep_plan(instance, boxPlan, &sweepBatch, &boxReservation).status ==
                OVSPATIALQUERY_API_INVALID_HANDLE);
        const ovspatialquery_enqueue_result_t boxAdmission =
            ovspatialquery_enqueue_box_sweep_plan(instance, boxPlan, &boxBatch, &boxReservation);
        REQUIRE(boxAdmission.status == OVSPATIALQUERY_API_SUCCESS);
        const ovspatialquery_box_sweep_batch_result_t boxResult =
            waitAndFetch(instance, boxAdmission.op_id, ovspatialquery_fetch_box_sweep_batch);
        REQUIRE(boxResult.common.truth == (precision == OVSPATIALQUERY_QUERY_EXACT
                                               ? OVSPATIALQUERY_HIT_EXACT_SURFACE
                                               : OVSPATIALQUERY_HIT_BOUNDS_CANDIDATE));
        REQUIRE(static_cast<const uint64_t*>(boxResult.columns.hit_ranges.hit_counts.data)[0] == 1u);
        const double* boxDistance =
            precision == OVSPATIALQUERY_QUERY_EXACT
                ? static_cast<const double*>(boxResult.columns.contact.exact.distances.data)
                : static_cast<const double*>(boxResult.columns.contact.candidate.distances.data);
        REQUIRE(std::abs(boxDistance[0] - 1.75) < 1.0e-6);
        // Each arm carries the box center at contact, 1.75 along +X from -2.
        const double* boxQueryPosition =
            precision == OVSPATIALQUERY_QUERY_EXACT
                ? static_cast<const double*>(boxResult.columns.contact.exact.query_positions.data)
                : static_cast<const double*>(boxResult.columns.contact.candidate.query_positions.data);
        REQUIRE(std::abs(boxQueryPosition[0] - (-0.25)) < 1.0e-6);
        destroyOperation(instance, boxAdmission.op_id);
        REQUIRE(ovspatialquery_destroy_batch_plan(instance, boxPlan).status == OVSPATIALQUERY_API_SUCCESS);
    }

    // Box sweep rows with their own half extents, on the three varied-radius
    // rows: the leading face sits 0.25 and 0.75 ahead of the center, so the
    // rows reach object 20 at 1.75 and 1.25; a zero extent is that row's
    // INVALID_ARGUMENT. A malformed column (float32) is the enqueue's.
    {
        float      variedRotations[12] = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
        double     rowHalfExtents[9]   = {0.25, 0.25, 0.25, 0.75, 0.1, 0.1, 0.0, 0.1, 0.1};
        float      rowHalfExtents32[9] = {0.25f, 0.25f, 0.25f, 0.75f, 0.1f, 0.1f, 0.0f, 0.1f, 0.1f};
        InputArray variedRotationArray, rowHalfExtentArray, rowHalfExtent32Array;
        variedRotationArray.set(variedRotations, 3, 4, kDLFloat);
        rowHalfExtentArray.set(rowHalfExtents, 3, 3, kDLFloat);
        rowHalfExtent32Array.set(rowHalfExtents32, 3, 3, kDLFloat);
        constexpr uint32_t sizedBoxFields = OVSPATIALQUERY_BATCH_RESULT_DISTANCE |
                                            OVSPATIALQUERY_BATCH_RESULT_POSITION |
                                            OVSPATIALQUERY_BATCH_RESULT_SOURCE_FEATURE;
        ovspatialquery_box_sweep_plan_desc_t sizedBoxPlanDesc{};
        sizedBoxPlanDesc.target                         = target;
        sizedBoxPlanDesc.precision                      = OVSPATIALQUERY_QUERY_EXACT;
        sizedBoxPlanDesc.half_extents[0]                = 0.5;
        sizedBoxPlanDesc.half_extents[1]                = 0.5;
        sizedBoxPlanDesc.half_extents[2]                = 0.5;
        sizedBoxPlanDesc.device                         = {kDLCPU, 0};
        sizedBoxPlanDesc.maximum_query_count            = 3u;
        sizedBoxPlanDesc.result_slot_count              = 1u;
        sizedBoxPlanDesc.result_fields                  = sizedBoxFields;
        sizedBoxPlanDesc.hit_capacity_per_slot          = 3u;
        ovspatialquery_batch_plan_handle_t sizedBoxPlan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
        REQUIRE(ovspatialquery_create_box_sweep_plan(instance, &sizedBoxPlanDesc, &sizedBoxPlan).status ==
                OVSPATIALQUERY_API_SUCCESS);
        ovspatialquery_box_sweep_batch_t sizedBoxBatch{};
        sizedBoxBatch.ordinal         = headOrdinal();
        sizedBoxBatch.query_count     = 3u;
        sizedBoxBatch.centers         = &variedCenterArray.value;
        sizedBoxBatch.rotations_xyzw  = &variedRotationArray.value;
        sizedBoxBatch.unit_directions = &variedDirectionArray.value;
        sizedBoxBatch.max_distances   = &variedMaximumArray.value;
        sizedBoxBatch.half_extents    = &rowHalfExtent32Array.value;
        ovspatialquery_box_sweep_batch_reservation_t sizedBoxReservation{};
        REQUIRE(ovspatialquery_enqueue_box_sweep_plan(instance, sizedBoxPlan, &sizedBoxBatch, &sizedBoxReservation)
                    .status == OVSPATIALQUERY_API_INVALID_ARGUMENT);
        sizedBoxBatch.half_extents = &rowHalfExtentArray.value;
        const ovspatialquery_enqueue_result_t sizedBoxAdmission =
            ovspatialquery_enqueue_box_sweep_plan(instance, sizedBoxPlan, &sizedBoxBatch, &sizedBoxReservation);
        REQUIRE(sizedBoxAdmission.status == OVSPATIALQUERY_API_SUCCESS);
        const ovspatialquery_box_sweep_batch_result_t sizedBoxResult =
            waitAndFetch(instance, sizedBoxAdmission.op_id, ovspatialquery_fetch_box_sweep_batch);
        const auto* sizedStatuses  = static_cast<const uint32_t*>(sizedBoxResult.columns.hit_ranges.query_status.data);
        const auto* sizedCounts    = static_cast<const uint64_t*>(sizedBoxResult.columns.hit_ranges.hit_counts.data);
        const auto* sizedDistances = static_cast<const double*>(sizedBoxResult.columns.contact.exact.distances.data);
        REQUIRE(sizedStatuses[0] == OVSPATIALQUERY_BATCH_ROW_SUCCESS && sizedCounts[0] == 1u);
        REQUIRE(sizedStatuses[1] == OVSPATIALQUERY_BATCH_ROW_SUCCESS && sizedCounts[1] == 1u);
        REQUIRE(sizedStatuses[2] == OVSPATIALQUERY_BATCH_ROW_INVALID_ARGUMENT && sizedCounts[2] == 0u);
        REQUIRE(std::abs(sizedDistances[0] - 1.75) < 1.0e-6);
        REQUIRE(std::abs(sizedDistances[1] - 1.25) < 1.0e-6);
        // Exact [H,3] columns and the feature group: a box surface is feature NONE.
        const ovspatialquery_exact_feature_columns_t& sizedFeatures = sizedBoxResult.columns.contact.exact.features;
        REQUIRE(hasLayout(sizedBoxResult.columns.contact.exact.positions, kDLFloat, 64u, 2, 2) &&
                hasLayout(sizedFeatures.kinds, kDLUInt, 32u, 1, 2) &&
                hasLayout(sizedFeatures.source_face_indices, kDLUInt, 64u, 1, 2) &&
                hasLayout(sizedFeatures.source_corner_indices, kDLUInt, 32u, 2, 2) &&
                hasLayout(sizedFeatures.barycentrics, kDLFloat, 32u, 2, 2));
        const auto* sizedKinds = static_cast<const uint32_t*>(sizedFeatures.kinds.data);
        REQUIRE(sizedKinds[0] == OVSPATIALQUERY_GEOMETRY_FEATURE_NONE &&
                sizedKinds[1] == OVSPATIALQUERY_GEOMETRY_FEATURE_NONE);
        destroyOperation(instance, sizedBoxAdmission.op_id);
        REQUIRE(ovspatialquery_destroy_batch_plan(instance, sizedBoxPlan).status == OVSPATIALQUERY_API_SUCCESS);
    }

    // Sphere overlap ANY: a small sphere at object 20's center overlaps it.
    // A sphere has no orientation, so the batch leaves rotations_xyzw null
    // as the header requires; supplying a rotation column is rejected.
    double                               sphereCenter[3] = {0.5, 0.5, 0.5};
    InputArray                           sphereCenterArray(sphereCenter, 1, 3, kDLFloat);
    ovspatialquery_overlap_batch_shape_t sphereShape{};
    sphereShape.kind               = OVSPATIALQUERY_OVERLAP_SPHERE;
    sphereShape.data.sphere.radius = 0.1;
    const ovspatialquery_batch_plan_handle_t sphereOverlapPlan =
        createOverlapPlan(instance, target, OVSPATIALQUERY_HIT_MODE_ANY, sphereShape, 1u, 1u, 1u, 0u);
    ovspatialquery_overlap_batch_t sphereOverlapBatch{};
    sphereOverlapBatch.ordinal      = headOrdinal();
    sphereOverlapBatch.query_count  = 1u;
    sphereOverlapBatch.translations = &sphereCenterArray.value;
    {
        float                          sphereRotation[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        InputArray                     sphereRotationArray(sphereRotation, 1, 4, kDLFloat);
        ovspatialquery_overlap_batch_t rotated = sphereOverlapBatch;
        rotated.rotations_xyzw                 = &sphereRotationArray.value;
        ovspatialquery_overlap_batch_reservation_t rotatedReservation{};
        const ovspatialquery_enqueue_result_t      rotatedAdmission =
            ovspatialquery_enqueue_overlap_plan(instance, sphereOverlapPlan, &rotated, &rotatedReservation);
        REQUIRE(rotatedAdmission.status == OVSPATIALQUERY_API_INVALID_ARGUMENT);
        REQUIRE(rotatedAdmission.op_id == OVSPATIALQUERY_INVALID_OP);
        // Per-row half extents belong to AABB and BOX plans, not spheres (ADR-0025).
        ovspatialquery_overlap_batch_t sized = sphereOverlapBatch;
        sized.half_extents                   = &sphereCenterArray.value;
        REQUIRE(ovspatialquery_enqueue_overlap_plan(instance, sphereOverlapPlan, &sized, &rotatedReservation).status ==
                OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    ovspatialquery_overlap_batch_reservation_t sphereOverlapReservation{};
    const ovspatialquery_enqueue_result_t      sphereOverlapAdmission = ovspatialquery_enqueue_overlap_plan(
        instance, sphereOverlapPlan, &sphereOverlapBatch, &sphereOverlapReservation);
    if (sphereOverlapAdmission.status != OVSPATIALQUERY_API_SUCCESS)
    {
        std::fprintf(stderr, "sphere overlap plan enqueue returned %u\n",
                     static_cast<unsigned>(sphereOverlapAdmission.status));
    }
    REQUIRE(sphereOverlapAdmission.status == OVSPATIALQUERY_API_SUCCESS);
    const ovspatialquery_overlap_batch_result_t sphereOverlapResult =
        waitAndFetch(instance, sphereOverlapAdmission.op_id, ovspatialquery_fetch_overlap_batch);
    REQUIRE(static_cast<const uint64_t*>(sphereOverlapResult.columns.object_ids.data)[0] == 20u);
    destroyOperation(instance, sphereOverlapAdmission.op_id);
    REQUIRE(ovspatialquery_destroy_batch_plan(instance, sphereOverlapPlan).status == OVSPATIALQUERY_API_SUCCESS);

    // Box overlap ANY: a small box at object 10's center overlaps it.
    double                               boxCenter[3] = {3.5, 0.5, 0.5};
    InputArray                           boxCenterArray(boxCenter, 1, 3, kDLFloat);
    const float                          identityRotation[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    InputArray                           boxRotationArray(identityRotation, 1, 4, kDLFloat);
    ovspatialquery_overlap_batch_shape_t boxShape{};
    boxShape.kind                     = OVSPATIALQUERY_OVERLAP_BOX;
    boxShape.data.box.half_extents[0] = 0.1;
    boxShape.data.box.half_extents[1] = 0.1;
    boxShape.data.box.half_extents[2] = 0.1;
    const ovspatialquery_batch_plan_handle_t boxOverlapPlan =
        createOverlapPlan(instance, target, OVSPATIALQUERY_HIT_MODE_ANY, boxShape, 1u, 1u, 1u, 0u);
    ovspatialquery_overlap_batch_t boxOverlapBatch{};
    boxOverlapBatch.ordinal        = headOrdinal();
    boxOverlapBatch.query_count    = 1u;
    boxOverlapBatch.translations   = &boxCenterArray.value;
    boxOverlapBatch.rotations_xyzw = &boxRotationArray.value;
    ovspatialquery_overlap_batch_reservation_t boxOverlapReservation{};
    const ovspatialquery_enqueue_result_t      boxOverlapAdmission =
        ovspatialquery_enqueue_overlap_plan(instance, boxOverlapPlan, &boxOverlapBatch, &boxOverlapReservation);
    REQUIRE(boxOverlapAdmission.status == OVSPATIALQUERY_API_SUCCESS);
    const ovspatialquery_overlap_batch_result_t boxOverlapResult =
        waitAndFetch(instance, boxOverlapAdmission.op_id, ovspatialquery_fetch_overlap_batch);
    REQUIRE(static_cast<const uint64_t*>(boxOverlapResult.columns.object_ids.data)[0] == 10u);
    destroyOperation(instance, boxOverlapAdmission.op_id);
    REQUIRE(ovspatialquery_destroy_batch_plan(instance, boxOverlapPlan).status == OVSPATIALQUERY_API_SUCCESS);

    // BOX rows may carry float64 rotations and their own half extents (ADR-0026,
    // ADR-0025): under a 0.1 plan shape, row 0 grown to 2.5 spans both objects
    // and row 1 only object 10, as the AABB rows below do.
    {
        double     sizedBoxTranslations[6] = {2.0, 0.5, 0.5, 6.0, 0.5, 0.5};
        double     sizedBoxRotations[8]    = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        double     sizedBoxHalfExtents[6]  = {2.5, 2.5, 2.5, 2.5, 2.5, 2.5};
        InputArray sizedBoxTranslationArray, sizedBoxRotationArray, sizedBoxHalfExtentArray;
        sizedBoxTranslationArray.set(sizedBoxTranslations, 2, 3, kDLFloat);
        sizedBoxRotationArray.set(sizedBoxRotations, 2, 4, kDLFloat);
        sizedBoxHalfExtentArray.set(sizedBoxHalfExtents, 2, 3, kDLFloat);
        const ovspatialquery_batch_plan_handle_t sizedBoxPlan =
            createOverlapPlan(instance, target, OVSPATIALQUERY_HIT_MODE_MULTIPLE, boxShape, 2u, 2u, 4u, 0u);
        ovspatialquery_overlap_batch_t sizedBox{};
        sizedBox.ordinal        = headOrdinal();
        sizedBox.query_count    = 2u;
        sizedBox.translations   = &sizedBoxTranslationArray.value;
        sizedBox.rotations_xyzw = &sizedBoxRotationArray.value;
        sizedBox.half_extents   = &sizedBoxHalfExtentArray.value;
        ovspatialquery_overlap_batch_reservation_t sizedBoxReservation{};
        const ovspatialquery_enqueue_result_t      sizedBoxAdmission =
            ovspatialquery_enqueue_overlap_plan(instance, sizedBoxPlan, &sizedBox, &sizedBoxReservation);
        REQUIRE(sizedBoxAdmission.status == OVSPATIALQUERY_API_SUCCESS);
        const ovspatialquery_overlap_batch_result_t sizedBoxResult =
            waitAndFetch(instance, sizedBoxAdmission.op_id, ovspatialquery_fetch_overlap_batch);
        const uint64_t* sizedBoxCounts =
            static_cast<const uint64_t*>(sizedBoxResult.columns.hit_ranges.hit_counts.data);
        REQUIRE(sizedBoxCounts[0] == 2u && sizedBoxCounts[1] == 1u);
        destroyOperation(instance, sizedBoxAdmission.op_id);
        REQUIRE(ovspatialquery_destroy_batch_plan(instance, sizedBoxPlan).status == OVSPATIALQUERY_API_SUCCESS);
    }

    // AABB overlap MULTIPLE across two rows, with geometry identity columns:
    // row 0 spans both objects; row 1 covers only object 10.
    double translations[6] = {
        2.0, 0.5, 0.5, 6.0, 0.5, 0.5,
    };
    InputArray                           translationArray(translations, 2, 3, kDLFloat);
    ovspatialquery_overlap_batch_shape_t aabbShapeWide{};
    aabbShapeWide.kind                      = OVSPATIALQUERY_OVERLAP_AABB;
    aabbShapeWide.data.aabb.half_extents[0] = 2.5;
    aabbShapeWide.data.aabb.half_extents[1] = 2.5;
    aabbShapeWide.data.aabb.half_extents[2] = 2.5;
    const ovspatialquery_batch_plan_handle_t aabbPlan =
        createOverlapPlan(instance, target, OVSPATIALQUERY_HIT_MODE_MULTIPLE, aabbShapeWide, 2u, 2u, 3u,
                          OVSPATIALQUERY_BATCH_RESULT_GEOMETRY_IDENTITY);
    ovspatialquery_overlap_batch_t aabbBatch{};
    aabbBatch.ordinal      = headOrdinal();
    aabbBatch.query_count  = 2u;
    aabbBatch.translations = &translationArray.value;
    ovspatialquery_overlap_batch_reservation_t aabbReservation{};
    const ovspatialquery_enqueue_result_t      aabbAdmission =
        ovspatialquery_enqueue_overlap_plan(instance, aabbPlan, &aabbBatch, &aabbReservation);
    REQUIRE(aabbAdmission.status == OVSPATIALQUERY_API_SUCCESS);
    const ovspatialquery_overlap_batch_result_t aabbResult =
        waitAndFetch(instance, aabbAdmission.op_id, ovspatialquery_fetch_overlap_batch);
    const uint64_t* aabbOffsets   = static_cast<const uint64_t*>(aabbResult.columns.hit_ranges.hit_offsets.data);
    const uint64_t* aabbCounts    = static_cast<const uint64_t*>(aabbResult.columns.hit_ranges.hit_counts.data);
    const uint8_t*  aabbTruncated = static_cast<const uint8_t*>(aabbResult.columns.hit_ranges.truncated.data);
    REQUIRE(aabbCounts[0] == 2u && aabbTruncated[0] == 0u);
    REQUIRE(aabbCounts[1] == 1u && aabbTruncated[1] == 0u);
    REQUIRE(*static_cast<const uint64_t*>(aabbResult.columns.hit_ranges.stored_hit_count.data) == 3u);
    const uint64_t* aabbIds = static_cast<const uint64_t*>(aabbResult.columns.object_ids.data);
    REQUIRE((aabbIds[aabbOffsets[0]] == 10u || aabbIds[aabbOffsets[0]] == 20u) &&
            (aabbIds[aabbOffsets[0] + 1u] == 10u || aabbIds[aabbOffsets[0] + 1u] == 20u) &&
            aabbIds[aabbOffsets[0]] != aabbIds[aabbOffsets[0] + 1u]);
    REQUIRE(aabbIds[aabbOffsets[1]] == 10u);
    const int32_t* geometryTypes = static_cast<const int32_t*>(aabbResult.columns.geometry.types.data);
    REQUIRE(geometryTypes[aabbOffsets[1]] == static_cast<int32_t>(OVSPATIALQUERY_GEOMETRY_BOX));
    REQUIRE(static_cast<const uint64_t*>(aabbResult.columns.geometry.values.data)[aabbOffsets[1]] == 10u);
    destroyOperation(instance, aabbAdmission.op_id);
    REQUIRE(ovspatialquery_destroy_batch_plan(instance, aabbPlan).status == OVSPATIALQUERY_API_SUCCESS);

    // AABB rows with their own half extents (ADR-0025): three rows at one
    // centre, a thin box that reaches no object, a wide one that reaches both,
    // and a zero one, which is that row's INVALID_ARGUMENT; a malformed column
    // (rank 1) is the enqueue's.
    double                                   sizedTranslations[9] = {2.0, 0.5, 0.5, 2.0, 0.5, 0.5, 2.0, 0.5, 0.5};
    double                                   rowHalfExtents[9]    = {0.2, 0.2, 0.2, 2.5, 2.5, 2.5, 0.0, 0.0, 0.0};
    InputArray                               sizedTranslationArray(sizedTranslations, 3, 3, kDLFloat);
    InputArray                               rowHalfExtentArray(rowHalfExtents, 3, 3, kDLFloat);
    const ovspatialquery_batch_plan_handle_t sizedPlan =
        createOverlapPlan(instance, target, OVSPATIALQUERY_HIT_MODE_MULTIPLE, aabbShapeWide, 3u, 2u, 4u, 0u);
    ovspatialquery_overlap_batch_t sizedBatch{};
    sizedBatch.ordinal      = headOrdinal();
    sizedBatch.query_count  = 3u;
    sizedBatch.translations = &sizedTranslationArray.value;
    sizedBatch.half_extents = &rowHalfExtentArray.value;
    ovspatialquery_overlap_batch_reservation_t sizedReservation{};
    {
        InputArray                     malformed(rowHalfExtents, 9, kDLFloat);
        ovspatialquery_overlap_batch_t flat = sizedBatch;
        flat.half_extents                   = &malformed.value;
        REQUIRE(ovspatialquery_enqueue_overlap_plan(instance, sizedPlan, &flat, &sizedReservation).status ==
                OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    const ovspatialquery_enqueue_result_t sizedAdmission =
        ovspatialquery_enqueue_overlap_plan(instance, sizedPlan, &sizedBatch, &sizedReservation);
    REQUIRE(sizedAdmission.status == OVSPATIALQUERY_API_SUCCESS);
    const ovspatialquery_overlap_batch_result_t sizedResult =
        waitAndFetch(instance, sizedAdmission.op_id, ovspatialquery_fetch_overlap_batch);
    const uint64_t* sizedCounts = static_cast<const uint64_t*>(sizedResult.columns.hit_ranges.hit_counts.data);
    const uint32_t* sizedStatus = static_cast<const uint32_t*>(sizedResult.columns.hit_ranges.query_status.data);
    REQUIRE(sizedStatus[0] == OVSPATIALQUERY_BATCH_ROW_SUCCESS && sizedCounts[0] == 0u);
    REQUIRE(sizedStatus[1] == OVSPATIALQUERY_BATCH_ROW_SUCCESS && sizedCounts[1] == 2u);
    REQUIRE(sizedStatus[2] == OVSPATIALQUERY_BATCH_ROW_INVALID_ARGUMENT && sizedCounts[2] == 0u);
    destroyOperation(instance, sizedAdmission.op_id);
    REQUIRE(ovspatialquery_destroy_batch_plan(instance, sizedPlan).status == OVSPATIALQUERY_API_SUCCESS);

    // SPHERE rows with their own radii, the same three rows as above: a thin
    // sphere that reaches no object, a wide one that reaches both, and a zero
    // one, which is that row's INVALID_ARGUMENT. A malformed column (float32)
    // and a radii column on an AABB plan are the enqueue's.
    {
        double     rowRadii[3]   = {0.2, 2.5, 0.0};
        float      rowRadii32[3] = {0.2f, 2.5f, 0.0f};
        InputArray rowRadiusArray, rowRadius32Array;
        rowRadiusArray.set(rowRadii, 3, kDLFloat);
        rowRadius32Array.set(rowRadii32, 3, kDLFloat);
        const ovspatialquery_batch_plan_handle_t radiusPlan =
            createOverlapPlan(instance, target, OVSPATIALQUERY_HIT_MODE_MULTIPLE, sphereShape, 3u, 2u, 4u, 0u);
        ovspatialquery_overlap_batch_t radiusBatch{};
        radiusBatch.ordinal      = headOrdinal();
        radiusBatch.query_count  = 3u;
        radiusBatch.translations = &sizedTranslationArray.value;
        radiusBatch.radii        = &rowRadiusArray.value;
        ovspatialquery_overlap_batch_reservation_t radiusReservation{};
        ovspatialquery_overlap_batch_t             narrow = radiusBatch;
        narrow.radii                                      = &rowRadius32Array.value;
        REQUIRE(ovspatialquery_enqueue_overlap_plan(instance, radiusPlan, &narrow, &radiusReservation).status ==
                OVSPATIALQUERY_API_INVALID_ARGUMENT);
        const ovspatialquery_batch_plan_handle_t aabbRadiusPlan =
            createOverlapPlan(instance, target, OVSPATIALQUERY_HIT_MODE_MULTIPLE, aabbShapeWide, 3u, 2u, 4u, 0u);
        REQUIRE(
            ovspatialquery_enqueue_overlap_plan(instance, aabbRadiusPlan, &radiusBatch, &radiusReservation).status ==
            OVSPATIALQUERY_API_INVALID_ARGUMENT);
        REQUIRE(ovspatialquery_destroy_batch_plan(instance, aabbRadiusPlan).status == OVSPATIALQUERY_API_SUCCESS);
        const ovspatialquery_enqueue_result_t radiusAdmission =
            ovspatialquery_enqueue_overlap_plan(instance, radiusPlan, &radiusBatch, &radiusReservation);
        REQUIRE(radiusAdmission.status == OVSPATIALQUERY_API_SUCCESS);
        const ovspatialquery_overlap_batch_result_t radiusResult =
            waitAndFetch(instance, radiusAdmission.op_id, ovspatialquery_fetch_overlap_batch);
        const uint64_t* radiusCounts = static_cast<const uint64_t*>(radiusResult.columns.hit_ranges.hit_counts.data);
        const uint32_t* radiusStatus = static_cast<const uint32_t*>(radiusResult.columns.hit_ranges.query_status.data);
        REQUIRE(radiusStatus[0] == OVSPATIALQUERY_BATCH_ROW_SUCCESS && radiusCounts[0] == 0u);
        REQUIRE(radiusStatus[1] == OVSPATIALQUERY_BATCH_ROW_SUCCESS && radiusCounts[1] == 2u);
        REQUIRE(radiusStatus[2] == OVSPATIALQUERY_BATCH_ROW_INVALID_ARGUMENT && radiusCounts[2] == 0u);
        destroyOperation(instance, radiusAdmission.op_id);
        REQUIRE(ovspatialquery_destroy_batch_plan(instance, radiusPlan).status == OVSPATIALQUERY_API_SUCCESS);
    }

    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    destroyInstanceAndDefinitions(instance);
}

// A float64 BOX row orients the exact test in double precision (ADR-0026), on
// the scene float64_box_row_scene.h describes. Rows 2e-9 beyond each object's
// face miss with a float64 rotation column and rows 2e-9 inside hit. With a
// float32 column the box and capsule rows hit on both sides; the hull and mesh
// take PhysX's float32 test then, which does not resolve 2e-9.
void testFloat64BoxRowsOrientTheExactTest()
{
    const ovspatialquery_instance_handle_t instance = createInstance(4u, 1u, UINT64_C(64) << 20);
    Float64BoxRowScene                     scene(instance);
    constexpr uint32_t                     kRows = Float64BoxRowScene::kRows;
    ovspatialquery_overlap_batch_shape_t   shape{};
    shape.kind = OVSPATIALQUERY_OVERLAP_BOX;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        shape.data.box.half_extents[axis] = Float64BoxRowScene::kHalfExtents[axis];
    }
    const ovspatialquery_batch_plan_handle_t plan =
        createOverlapPlan(instance, populationRef(scene.population), OVSPATIALQUERY_HIT_MODE_MULTIPLE, shape, kRows, 2u,
                          kRows * 2u, 0u, OVSPATIALQUERY_QUERY_EXACT);

    InputArray translationArray(scene.translations, kRows, 3, kDLFloat);
    InputArray rotation64Array(scene.rotations64, kRows, 4, kDLFloat);
    InputArray rotation32Array(scene.rotations32, kRows, 4, kDLFloat);
    const auto hitCounts = [&](const InputArray& rotationArray)
    {
        ovspatialquery_overlap_batch_t batch{};
        batch.ordinal        = headOrdinal();
        batch.query_count    = kRows;
        batch.translations   = &translationArray.value;
        batch.rotations_xyzw = &rotationArray.value;
        ovspatialquery_overlap_batch_reservation_t reservation{};
        const ovspatialquery_enqueue_result_t      admission =
            ovspatialquery_enqueue_overlap_plan(instance, plan, &batch, &reservation);
        REQUIRE(admission.status == OVSPATIALQUERY_API_SUCCESS);
        const ovspatialquery_overlap_batch_result_t result =
            waitAndFetch(instance, admission.op_id, ovspatialquery_fetch_overlap_batch);
        std::array<uint64_t, kRows> counts{};
        std::memcpy(counts.data(), result.columns.hit_ranges.hit_counts.data, sizeof(counts));
        destroyOperation(instance, admission.op_id);
        return counts;
    };
    REQUIRE(hitCounts(rotation64Array) == std::array<uint64_t, kRows>{0u, 1u, 0u, 1u, 0u, 1u, 0u, 1u});
    const std::array<uint64_t, kRows> float32Counts = hitCounts(rotation32Array);
    REQUIRE(float32Counts[0] == 1u && float32Counts[1] == 1u && float32Counts[2] == 1u && float32Counts[3] == 1u);
    // A float32 row evaluates as the synchronous float32 shape does, on every
    // family: only a float64 column carries its rotation in double.
    for (uint32_t row = 0u; row < kRows; ++row)
    {
        ovspatialquery_overlap_desc_t query{};
        query.common.target.target  = populationRef(scene.population);
        query.common.target.ordinal = headOrdinal();
        query.common.precision      = OVSPATIALQUERY_QUERY_EXACT;
        query.shape.kind            = OVSPATIALQUERY_OVERLAP_BOX;
        for (uint32_t axis = 0u; axis < 3u; ++axis)
        {
            query.shape.data.box.center[axis]       = scene.translations[row * 3u + axis];
            query.shape.data.box.half_extents[axis] = shape.data.box.half_extents[axis];
        }
        std::memcpy(query.shape.data.box.rotation_xyzw, scene.rotations32 + row * 4u, sizeof(float) * 4u);
        ovspatialquery_overlap_any_result_t any{};
        REQUIRE(ovspatialquery_overlap_any_sync(instance, &query, &any).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(any.has_hit == float32Counts[row]);
    }

    REQUIRE(ovspatialquery_destroy_batch_plan(instance, plan).status == OVSPATIALQUERY_API_SUCCESS);
    scene.destroy();
    destroyInstanceAndDefinitions(instance);
}

// Copied-on-submission inputs may be overwritten as soon as admission
// succeeds; borrowed inputs must stay alive until terminal, exercised by
// blocking ray execution with the oracle backend gate; fetch is repeatable
// and frozen under concurrent readers; a recoverable operation failure
// preserves the caller's fetch output and reports its terminal cause.
void testOwnershipCopiedTargetsAndFrozenFetch()
{
    const ovspatialquery_instance_handle_t instance = createInstance(4u, 2u, UINT64_C(64) << 20);
    ovspatialquery_backend_runtime_t*      runtime  = ovspatialquery_test_oracle_get_default_runtime();
    REQUIRE(runtime != nullptr);
    const ovspatialquery_population_handle_t population =
        publishBoxPopulation(instance, 13, {{41u, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}});
    const ovspatialquery_target_ref_t        target = populationRef(population);
    const ovspatialquery_batch_plan_handle_t plan = createRaycastPlan(instance, target, OVSPATIALQUERY_HIT_MODE_CLOSEST,
                                                                      1u, 1u, 1u, OVSPATIALQUERY_BATCH_RESULT_DISTANCE);

    double     origin[3]    = {-1.0, 0.5, 0.5};
    float      direction[3] = {1.0f, 0.0f, 0.0f};
    double     minDistance  = 0.0;
    double     maxDistance  = 10.0;
    InputArray originArray(origin, 1, 3, kDLFloat);
    InputArray directionArray(direction, 1, 3, kDLFloat);
    directionArray.value.ownership = OVSPATIALQUERY_INPUT_BORROWED_UNTIL_TERMINAL;
    InputArray                     minArray(&minDistance, 1, kDLFloat);
    InputArray                     maxArray(&maxDistance, 1, kDLFloat);
    ovspatialquery_raycast_batch_t batch{};
    batch.ordinal         = headOrdinal();
    batch.query_count     = 1u;
    batch.origins         = &originArray.value;
    batch.unit_directions = &directionArray.value;
    batch.min_distances   = &minArray.value;
    batch.max_distances   = &maxArray.value;

    ovspatialquery_test_oracle_block_rays(runtime, 1u);
    ovspatialquery_raycast_batch_reservation_t reservation{};
    const ovspatialquery_op_id_t               operation = enqueueRaycast(instance, plan, batch, reservation);

    // Admission has already copied the origin; mutating it now must not
    // change the result. The borrowed direction stays untouched throughout,
    // honoring its ownership rule while ray execution is still blocked.
    origin[0] = 1000.0;
    REQUIRE(ovspatialquery_test_oracle_wait_for_blocked_rays(runtime) == 1u);

    ovspatialquery_raycast_batch_result_t pending{};
    fillBytePattern(pending, 0x5Bu);
    REQUIRE(ovspatialquery_fetch_raycast_batch(instance, operation, &pending).status == OVSPATIALQUERY_API_NOT_READY);
    REQUIRE(isBytePattern(pending, 0x5Bu));

    ovspatialquery_test_oracle_release_rays(runtime);
    const ovspatialquery_raycast_batch_result_t first =
        waitAndFetch(instance, operation, ovspatialquery_fetch_raycast_batch);
    REQUIRE(static_cast<const uint64_t*>(first.columns.object_ids.data)[0] == 41u);
    REQUIRE(static_cast<const double*>(first.columns.contact.candidate.distances.data)[0] == 1.0);

    // Frozen concurrent repeat fetch: every reader observes the same column
    // storage while the operation stays alive.
    constexpr uint32_t       fetcherCount = 4u;
    std::atomic<uint32_t>    fetchFailures{0u};
    std::vector<std::thread> fetchers;
    for (uint32_t index = 0u; index < fetcherCount; ++index)
    {
        fetchers.emplace_back(
            [&]()
            {
                for (uint32_t iteration = 0u; iteration < 50u; ++iteration)
                {
                    ovspatialquery_raycast_batch_result_t repeated{};
                    if (ovspatialquery_fetch_raycast_batch(instance, operation, &repeated).status !=
                            OVSPATIALQUERY_API_SUCCESS ||
                        repeated.common.query_count != first.common.query_count ||
                        repeated.columns.object_ids.data != first.columns.object_ids.data ||
                        repeated.columns.contact.candidate.distances.data !=
                            first.columns.contact.candidate.distances.data)
                    {
                        fetchFailures.fetch_add(1u);
                    }
                }
            });
    }
    for (std::thread& fetcher : fetchers)
    {
        fetcher.join();
    }
    REQUIRE(fetchFailures.load() == 0u);

    // An op_id is instance-scoped: another instance's fetch reports
    // INVALID_HANDLE and leaves the caller's output untouched.
    const ovspatialquery_instance_handle_t foreignInstance = createInstance(4u, 1u, UINT64_C(64) << 20);
    ovspatialquery_raycast_batch_result_t  foreignResult{};
    fillBytePattern(foreignResult, 0x4Du);
    REQUIRE(ovspatialquery_fetch_raycast_batch(foreignInstance, operation, &foreignResult).status ==
            OVSPATIALQUERY_API_INVALID_HANDLE);
    REQUIRE(isBytePattern(foreignResult, 0x4Du));
    REQUIRE(ovspatialquery_destroy_instance(foreignInstance).status == OVSPATIALQUERY_API_SUCCESS);
    destroyOperation(instance, operation);

    ovspatialquery_test_oracle_fail_next_ray(runtime, OVSPATIALQUERY_API_INTERNAL);
    ovspatialquery_raycast_batch_reservation_t failedReservation{};
    const ovspatialquery_op_id_t    failedOperation = enqueueRaycast(instance, plan, batch, failedReservation);
    ovspatialquery_op_wait_result_t failedWait{};
    REQUIRE(ovspatialquery_wait_op(instance, failedOperation, OVSPATIALQUERY_TIMEOUT_INFINITE, &failedWait).status ==
            OVSPATIALQUERY_API_SUCCESS);
    // A backend failure on one ray is a per-row outcome, not an operation
    // failure: the operation completes, the row reports BATCH_ROW_FAILED with
    // zero offset, count, and truncation, and no payload row is readable.
    REQUIRE(failedWait.state == OVSPATIALQUERY_OP_COMPLETED);
    ovspatialquery_op_status_t failedStatus{};
    REQUIRE(ovspatialquery_query_op_status(instance, failedOperation, &failedStatus).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(failedStatus.terminal_status == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_raycast_batch_result_t failedResult{};
    REQUIRE(ovspatialquery_fetch_raycast_batch(instance, failedOperation, &failedResult).status ==
            OVSPATIALQUERY_API_SUCCESS);
    const uint32_t failedRowStatus = static_cast<const uint32_t*>(failedResult.columns.hit_ranges.query_status.data)[0];
    if (failedRowStatus != OVSPATIALQUERY_BATCH_ROW_FAILED)
    {
        std::fprintf(stderr, "failed ray row reported status %u\n", static_cast<unsigned>(failedRowStatus));
    }
    REQUIRE(failedRowStatus == OVSPATIALQUERY_BATCH_ROW_FAILED);
    REQUIRE(*static_cast<const uint64_t*>(failedResult.columns.hit_ranges.stored_hit_count.data) == 0u);
    REQUIRE(static_cast<const uint64_t*>(failedResult.columns.hit_ranges.hit_offsets.data)[0] == 0u &&
            static_cast<const uint64_t*>(failedResult.columns.hit_ranges.hit_counts.data)[0] == 0u &&
            static_cast<const uint8_t*>(failedResult.columns.hit_ranges.truncated.data)[0] == 0u);
    destroyOperation(instance, failedOperation);

    // An operation failure, here exhausted resources, reaches every entry
    // point as the header promises: wait_op succeeds with the FAILURE state,
    // query_op_status carries the terminal cause, and fetch returns
    // OPERATION_FAILED and preserves its output. After destroy_op the stale id
    // is INVALID_HANDLE and leaves output untouched, and the released result
    // slot serves the next operations.
    origin[0] = -1.0;
    REQUIRE(ovspatialquery_test_fail_next_family_plan_resource(instance, plan) == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_raycast_batch_reservation_t exhaustedReservation{};
    const ovspatialquery_op_id_t    exhaustedOperation = enqueueRaycast(instance, plan, batch, exhaustedReservation);
    ovspatialquery_op_wait_result_t exhaustedWait{};
    REQUIRE(
        ovspatialquery_wait_op(instance, exhaustedOperation, OVSPATIALQUERY_TIMEOUT_INFINITE, &exhaustedWait).status ==
        OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(exhaustedWait.state == OVSPATIALQUERY_OP_FAILURE);
    ovspatialquery_op_status_t exhaustedStatus{};
    REQUIRE(ovspatialquery_query_op_status(instance, exhaustedOperation, &exhaustedStatus).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(exhaustedStatus.state == OVSPATIALQUERY_OP_FAILURE &&
            exhaustedStatus.terminal_status == OVSPATIALQUERY_API_OUT_OF_MEMORY);
    ovspatialquery_raycast_batch_result_t exhaustedResult{};
    fillBytePattern(exhaustedResult, 0x6Du);
    REQUIRE(ovspatialquery_fetch_raycast_batch(instance, exhaustedOperation, &exhaustedResult).status ==
            OVSPATIALQUERY_API_OPERATION_FAILED);
    REQUIRE(isBytePattern(exhaustedResult, 0x6Du));
    destroyOperation(instance, exhaustedOperation);
    ovspatialquery_op_wait_result_t staleWait{};
    REQUIRE(ovspatialquery_wait_op(instance, exhaustedOperation, 0u, &staleWait).status ==
            OVSPATIALQUERY_API_INVALID_HANDLE);
    ovspatialquery_raycast_batch_result_t staleResult{};
    fillBytePattern(staleResult, 0x6Eu);
    REQUIRE(ovspatialquery_fetch_raycast_batch(instance, exhaustedOperation, &staleResult).status ==
            OVSPATIALQUERY_API_INVALID_HANDLE);
    REQUIRE(isBytePattern(staleResult, 0x6Eu));
    // Both of the plan's result slots admit again, so the failed one was released.
    ovspatialquery_raycast_batch_reservation_t firstReservation{};
    ovspatialquery_raycast_batch_reservation_t secondReservation{};
    const ovspatialquery_op_id_t               firstOperation = enqueueRaycast(instance, plan, batch, firstReservation);
    const ovspatialquery_op_id_t secondOperation = enqueueRaycast(instance, plan, batch, secondReservation);
    REQUIRE(firstReservation.common.result_slot != secondReservation.common.result_slot);
    REQUIRE(static_cast<const uint64_t*>(waitAndFetch(instance, firstOperation, ovspatialquery_fetch_raycast_batch)
                                             .columns.object_ids.data)[0] == 41u);
    REQUIRE(static_cast<const uint64_t*>(waitAndFetch(instance, secondOperation, ovspatialquery_fetch_raycast_batch)
                                             .columns.object_ids.data)[0] == 41u);
    destroyOperation(instance, firstOperation);
    destroyOperation(instance, secondOperation);

    REQUIRE(ovspatialquery_destroy_batch_plan(instance, plan).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    destroyInstanceAndDefinitions(instance);
}

// Plan creation rejects result fields a plan cannot serve, and admission
// rejects input columns that disagree on their row count, both atomically: no
// plan or operation is issued and caller outputs are untouched.
void testPlanCreationAndAdmissionValidation()
{
    const ovspatialquery_instance_handle_t   instance = createInstance(4u, 2u, UINT64_C(64) << 20);
    const ovspatialquery_population_handle_t population =
        publishBoxPopulation(instance, 17, {{900u, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}});
    const ovspatialquery_target_ref_t target = populationRef(population);

    // A plan may not request SOURCE_FEATURE or QUERY_POSITION on a candidate
    // (non-exact) raycast plan; both are rejected atomically at creation.
    const ovspatialquery_batch_plan_handle_t rejectedFeature =
        createRaycastPlan(instance, target, OVSPATIALQUERY_HIT_MODE_CLOSEST, 1u, 1u, 1u,
                          OVSPATIALQUERY_BATCH_RESULT_SOURCE_FEATURE, OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(rejectedFeature == OVSPATIALQUERY_INVALID_BATCH_PLAN);
    const ovspatialquery_batch_plan_handle_t rejectedQueryPosition =
        createRaycastPlan(instance, target, OVSPATIALQUERY_HIT_MODE_CLOSEST, 1u, 1u, 1u,
                          OVSPATIALQUERY_BATCH_RESULT_QUERY_POSITION, OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(rejectedQueryPosition == OVSPATIALQUERY_INVALID_BATCH_PLAN);
    // SOURCE_FEATURE is exact-only for sphere and box sweeps too, and
    // QUERY_POSITION belongs to sweeps alone, so an overlap plan rejects it at
    // every precision.
    createSphereSweepPlan(instance, target, 0.5, 1u, OVSPATIALQUERY_BATCH_RESULT_SOURCE_FEATURE,
                          OVSPATIALQUERY_API_INVALID_ARGUMENT);
    ovspatialquery_overlap_batch_shape_t sphereShape{};
    sphereShape.kind               = OVSPATIALQUERY_OVERLAP_SPHERE;
    sphereShape.data.sphere.radius = 0.5;
    createOverlapPlan(instance, target, OVSPATIALQUERY_HIT_MODE_ANY, sphereShape, 1u, 1u, 1u,
                      OVSPATIALQUERY_BATCH_RESULT_QUERY_POSITION, OVSPATIALQUERY_QUERY_EXACT,
                      OVSPATIALQUERY_API_INVALID_ARGUMENT);
    ovspatialquery_box_sweep_plan_desc_t boxDesc{};
    boxDesc.target                             = target;
    boxDesc.precision                          = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    boxDesc.half_extents[0]                    = 0.5;
    boxDesc.half_extents[1]                    = 0.5;
    boxDesc.half_extents[2]                    = 0.5;
    boxDesc.device                             = {kDLCPU, 0};
    boxDesc.maximum_query_count                = 1u;
    boxDesc.result_slot_count                  = 1u;
    boxDesc.result_fields                      = OVSPATIALQUERY_BATCH_RESULT_SOURCE_FEATURE;
    boxDesc.hit_capacity_per_slot              = 1u;
    ovspatialquery_batch_plan_handle_t boxPlan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_box_sweep_plan(instance, &boxDesc, &boxPlan).status ==
            OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(boxPlan == OVSPATIALQUERY_INVALID_BATCH_PLAN);

    // A plan whose byte size does not fit in 64 bits exceeds any host budget.
    createRaycastPlan(instance, target, OVSPATIALQUERY_HIT_MODE_CLOSEST, UINT64_C(1) << 62, 1u, 1u, 0u,
                      OVSPATIALQUERY_API_HOST_BUDGET);

    // Columns that disagree with each other (and with query_count) on their
    // row count are rejected atomically: no operation is issued and the
    // caller's reservation is left untouched.
    const ovspatialquery_batch_plan_handle_t mismatchPlan =
        createRaycastPlan(instance, target, OVSPATIALQUERY_HIT_MODE_CLOSEST, 2u, 1u, 2u, 0u);
    double mismatchedOrigins[6] = {
        -1.0, 0.5, 0.5, -1.0, 0.5, 0.5,
    };
    float                          mismatchedDirection[3] = {1.0f, 0.0f, 0.0f};
    double                         mismatchedMin[2]       = {0.0, 0.0};
    double                         mismatchedMax[2]       = {10.0, 10.0};
    InputArray                     mismatchedOriginArray(mismatchedOrigins, 2, 3, kDLFloat);
    InputArray                     mismatchedDirectionArray(mismatchedDirection, 1, 3, kDLFloat);
    InputArray                     mismatchedMinArray(mismatchedMin, 2, kDLFloat);
    InputArray                     mismatchedMaxArray(mismatchedMax, 2, kDLFloat);
    ovspatialquery_raycast_batch_t mismatchedBatch{};
    mismatchedBatch.ordinal         = headOrdinal();
    mismatchedBatch.query_count     = 2u;
    mismatchedBatch.origins         = &mismatchedOriginArray.value;
    mismatchedBatch.unit_directions = &mismatchedDirectionArray.value;
    mismatchedBatch.min_distances   = &mismatchedMinArray.value;
    mismatchedBatch.max_distances   = &mismatchedMaxArray.value;
    ovspatialquery_raycast_batch_reservation_t mismatchedReservation{};
    fillBytePattern(mismatchedReservation, 0x3Cu);
    const ovspatialquery_enqueue_result_t mismatchedAdmission =
        ovspatialquery_enqueue_raycast_plan(instance, mismatchPlan, &mismatchedBatch, &mismatchedReservation);
    REQUIRE(mismatchedAdmission.status == OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(mismatchedAdmission.op_id == OVSPATIALQUERY_INVALID_OP);
    REQUIRE(isBytePattern(mismatchedReservation, 0x3Cu));
    REQUIRE(ovspatialquery_destroy_batch_plan(instance, mismatchPlan).status == OVSPATIALQUERY_API_SUCCESS);

    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    destroyInstanceAndDefinitions(instance);
}

// Room reserved for later versions (ADR-0062) is refused with NOT_SUPPORTED
// and no effect: a non-null reserved batch column or synchronous input.
void testReservedRoomIsRefused()
{
    const ovspatialquery_instance_handle_t   instance = createInstance(4u, 2u, UINT64_C(64) << 20);
    const ovspatialquery_population_handle_t population =
        publishBoxPopulation(instance, 5, {{55u, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}});
    const ovspatialquery_target_ref_t target = populationRef(population);

    const ovspatialquery_batch_plan_handle_t plan =
        createRaycastPlan(instance, target, OVSPATIALQUERY_HIT_MODE_CLOSEST, 1u, 1u, 1u, 0u);
    double                         origin[3]    = {-1.0, 0.5, 0.5};
    float                          direction[3] = {1.0f, 0.0f, 0.0f};
    double                         maxDistance  = 10.0;
    InputArray                     originArray(origin, 1, 3, kDLFloat);
    InputArray                     directionArray(direction, 1, 3, kDLFloat);
    InputArray                     maxArray(&maxDistance, 1, kDLFloat);
    ovspatialquery_raycast_batch_t batch{};
    batch.ordinal         = headOrdinal();
    batch.query_count     = 1u;
    batch.origins         = &originArray.value;
    batch.unit_directions = &directionArray.value;
    batch.max_distances   = &maxArray.value;
    for (size_t slot = 0u; slot != 2u; ++slot)
    {
        batch.reserved[slot] = &maxArray.value;
        ovspatialquery_raycast_batch_reservation_t reservation{};
        fillBytePattern(reservation, 0x5Au);
        const ovspatialquery_enqueue_result_t refused =
            ovspatialquery_enqueue_raycast_plan(instance, plan, &batch, &reservation);
        REQUIRE(refused.status == OVSPATIALQUERY_API_NOT_SUPPORTED && refused.op_id == OVSPATIALQUERY_INVALID_OP);
        REQUIRE(isBytePattern(reservation, 0x5Au));
        batch.reserved[slot] = nullptr;
    }
    // Both slots null, the same launch is admitted: the refusal had no effect on the plan.
    ovspatialquery_raycast_batch_reservation_t  reservation{};
    const ovspatialquery_op_id_t                operation = enqueueRaycast(instance, plan, batch, reservation);
    const ovspatialquery_raycast_batch_result_t result =
        waitAndFetch(instance, operation, ovspatialquery_fetch_raycast_batch);
    REQUIRE(static_cast<const uint64_t*>(result.columns.object_ids.data)[0] == 55u);
    destroyOperation(instance, operation);
    REQUIRE(ovspatialquery_destroy_batch_plan(instance, plan).status == OVSPATIALQUERY_API_SUCCESS);

    ovspatialquery_raycast_desc_t ray{};
    ray.common.target.target  = target;
    ray.common.target.ordinal = headOrdinal();
    ray.ray.origin[0]         = -1.0;
    ray.ray.origin[1]         = 0.5;
    ray.ray.origin[2]         = 0.5;
    ray.ray.unit_direction[0] = 1.0f;
    ray.ray.max_distance      = 10.0;
    const int reservedInput   = 1;
    for (size_t slot = 0u; slot != 2u; ++slot)
    {
        ray.common.reserved[slot] = &reservedInput;
        ovspatialquery_raycast_single_result_t single{};
        fillBytePattern(single, 0x5Au);
        REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &single).status ==
                OVSPATIALQUERY_API_NOT_SUPPORTED);
        REQUIRE(isBytePattern(single, 0x5Au));
        ray.common.reserved[slot] = nullptr;
    }
    ovspatialquery_raycast_single_result_t single{};
    REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &single).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(single.has_hit == 1u && single.hit.data.bounds_candidate.identity.object_id.value == 55u);

    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    destroyInstanceAndDefinitions(instance);
}

} // namespace

int main()
{
    testZeroAndClosestRayBatch();
    testRayMultipleAndAnyPartitions();
    testLongLaunchUnderTightBudgets();
    testSweepAndOverlapFamilies();
    testFloat64BoxRowsOrientTheExactTest();
    testOwnershipCopiedTargetsAndFrozenFetch();
    testPlanCreationAndAdmissionValidation();
    testReservedRoomIsRefused();
    std::puts("batch API test passed");
    return 0;
}
