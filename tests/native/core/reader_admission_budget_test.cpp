/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Budgets refuse only at admission (ovspatialquery_op_status_t, ADR-0075): every reader kind,
// as the first exact reader of each geometry family, succeeds under the budget its scene was
// admitted under plus that reader's own charge, and returns the hit computed for the scene.
// A reader that builds a structure its scene's admissions did not reserve is refused or fails
// here, as the first exact query on a large triangle mesh once was.

#include "../../support/native/public_api_inputs.h"
#include "../../support/native/test_failure.h"
#include <ovspatialquery/ovspatialquery_explicit.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace
{

constexpr uint64_t kDefinition = 11u;
constexpr uint64_t kObject     = 12u;

// Every family presents its surface at x = -1 to the readers below, which start at x = -5 and
// move along +x through (y, z) = (kY, kZ), off every triangle edge of the mesh.
constexpr double kY = 0.01, kZ = 0.02, kStart = -5.0, kSweepRadius = 0.25, kBoxHalf = 0.25;

enum class Family
{
    sphere,
    box,
    capsule,
    convex,
    mesh,
};

enum class Reader
{
    raycast,
    sphereSweep,
    boxSweep,
    overlap,
    raycastBatch,
    sphereSweepBatch,
    boxSweepBatch,
    overlapBatch,
};

constexpr Family kFamilies[] = {Family::sphere, Family::box, Family::capsule, Family::convex, Family::mesh};
constexpr Reader kReaders[]  = {Reader::raycastBatch, Reader::sphereSweepBatch, Reader::boxSweepBatch,
                                Reader::overlapBatch, Reader::raycast,          Reader::sphereSweep,
                                Reader::boxSweep,     Reader::overlap};

// Waits for an admitted operation, which must complete, and destroys it; returns the
// admission's status.
ovspatialquery_api_status_t finish(ovspatialquery_instance_handle_t instance, ovspatialquery_enqueue_result_t admission)
{
    if (admission.status == OVSPATIALQUERY_API_SUCCESS)
        waitAndDestroy(instance, admission);
    return admission.status;
}

// Unit sphere, unit-half-extent box, a radius-one capsule along local y, the cube of
// corners +-1, and a double-sided square at x = -1 whose mesh also holds a 128 x 128
// grid far from every reader: 32770 triangles whose structure a reader must not pay for.
ovspatialquery_api_status_t registerFamily(ovspatialquery_instance_handle_t instance, Family family)
{
    uint64_t   value = kDefinition, zero = 0u;
    InputArray values(&value, 1, kDLUInt), offsets(&zero, 1, kDLUInt);
    switch (family)
    {
    case Family::sphere: {
        double                                         radius = 1.0;
        InputArray                                     radii(&radius, 1, kDLFloat);
        const ovspatialquery_sphere_definition_table_t spheres{&values.value, &radii.value};
        return finish(instance, ovspatialquery_enqueue_sphere_registration(instance, &spheres));
    }
    case Family::box: {
        double                                      half[3] = {1.0, 1.0, 1.0};
        InputArray                                  halves(half, 1, 3, kDLFloat);
        const ovspatialquery_box_definition_table_t boxes{&values.value, &halves.value};
        return finish(instance, ovspatialquery_enqueue_box_registration(instance, &boxes));
    }
    case Family::capsule: {
        double     radius = 1.0, halfSegment = 0.5;
        InputArray radii(&radius, 1, kDLFloat), segments(&halfSegment, 1, kDLFloat);
        const ovspatialquery_capsule_definition_table_t capsules{&values.value, &radii.value, &segments.value};
        return finish(instance, ovspatialquery_enqueue_capsule_registration(instance, &capsules));
    }
    case Family::convex: {
        float corners[24];
        for (int corner = 0; corner < 8; ++corner)
            for (int axis = 0; axis < 3; ++axis)
                corners[corner * 3 + axis] = (corner >> axis) & 1 ? 1.0f : -1.0f;
        uint64_t                                       count = 8u;
        InputArray                                     counts(&count, 1, kDLUInt), points(corners, 8, 3, kDLFloat);
        const ovspatialquery_convex_definition_table_t convexes{
            &values.value, &offsets.value, &counts.value, nullptr, nullptr, &points.value,
            nullptr,       nullptr,        nullptr,       0.0,     0u};
        return finish(instance, ovspatialquery_enqueue_convex_registration(instance, &convexes));
    }
    case Family::mesh: break;
    }
    constexpr uint32_t    kCells = 128u;
    std::vector<float>    vertices{-1, -2, -2, -1, 2, -2, -1, 2, 2, -1, -2, 2};
    std::vector<uint32_t> triangles{0, 1, 2, 0, 2, 3};
    for (uint32_t row = 0u; row <= kCells; ++row)
        for (uint32_t column = 0u; column <= kCells; ++column)
            vertices.insert(vertices.end(), {50.0f, float(column) / kCells, float(row) / kCells});
    for (uint32_t row = 0u; row < kCells; ++row)
        for (uint32_t column = 0u; column < kCells; ++column)
        {
            const uint32_t corner = 4u + row * (kCells + 1u) + column;
            triangles.insert(triangles.end(), {corner, corner + 1u, corner + kCells + 2u, corner, corner + kCells + 2u,
                                               corner + kCells + 1u});
        }
    uint64_t   vertexCount = vertices.size() / 3u, triangleCount = triangles.size() / 3u;
    uint8_t    doubleSided = 1u;
    InputArray vertexCounts(&vertexCount, 1, kDLUInt), triangleCounts(&triangleCount, 1, kDLUInt);
    InputArray points(vertices.data(), static_cast<int64_t>(vertexCount), 3, kDLFloat);
    InputArray indices(triangles.data(), static_cast<int64_t>(triangleCount), 3, kDLUInt);
    InputArray sidedness(&doubleSided, 1, kDLUInt);
    const ovspatialquery_triangle_mesh_definition_table_t meshes{
        &values.value, &offsets.value, &vertexCounts.value, &offsets.value, &triangleCounts.value,
        &points.value, &indices.value, &sidedness.value,    nullptr,        nullptr,
        nullptr};
    return finish(instance, ovspatialquery_enqueue_triangle_mesh_registration(instance, &meshes));
}

ovspatialquery_geometry_type_t geometryType(Family family)
{
    switch (family)
    {
    case Family::sphere: return OVSPATIALQUERY_GEOMETRY_SPHERE;
    case Family::box: return OVSPATIALQUERY_GEOMETRY_BOX;
    case Family::capsule: return OVSPATIALQUERY_GEOMETRY_CAPSULE;
    case Family::convex: return OVSPATIALQUERY_GEOMETRY_CONVEX;
    case Family::mesh: break;
    }
    return OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH;
}

// The distance each reader travels to the family's surface, computed for the scene. A box
// sweep's leading face covers (0, 0), so it meets every family at its leftmost point.
double expectedDistance(Family family, Reader reader)
{
    const bool   ray   = reader == Reader::raycast || reader == Reader::raycastBatch;
    const bool   box   = reader == Reader::boxSweep || reader == Reader::boxSweepBatch;
    const double reach = ray ? 1.0 : 1.0 + kSweepRadius;
    if (box)
        return -1.0 - kBoxHalf - kStart;
    if (family == Family::sphere)
        return -std::sqrt(reach * reach - kY * kY - kZ * kZ) - kStart;
    if (family == Family::capsule)
        return -std::sqrt(reach * reach - kZ * kZ) - kStart;
    return (ray ? -1.0 : -1.0 - kSweepRadius) - kStart;
}

struct Scene
{
    Family                             family     = Family::sphere;
    ovspatialquery_instance_handle_t   instance   = OVSPATIALQUERY_INVALID_INSTANCE;
    bool                               registered = false;
    ovspatialquery_population_handle_t population = OVSPATIALQUERY_INVALID_POPULATION;
    ovspatialquery_batch_plan_handle_t plans[4] = {OVSPATIALQUERY_INVALID_BATCH_PLAN, OVSPATIALQUERY_INVALID_BATCH_PLAN,
                                                   OVSPATIALQUERY_INVALID_BATCH_PLAN,
                                                   OVSPATIALQUERY_INVALID_BATCH_PLAN};
};

void releaseScene(Scene& scene)
{
    for (ovspatialquery_batch_plan_handle_t plan : scene.plans)
        if (plan != OVSPATIALQUERY_INVALID_BATCH_PLAN)
            REQUIRE(ovspatialquery_destroy_batch_plan(scene.instance, plan).status == OVSPATIALQUERY_API_SUCCESS);
    if (scene.population != OVSPATIALQUERY_INVALID_POPULATION)
        REQUIRE(ovspatialquery_destroy_population(scene.instance, scene.population).status ==
                OVSPATIALQUERY_API_SUCCESS);
    if (scene.registered)
        REQUIRE(ovspatialquery_destroy_geometry_definition(scene.instance, {geometryType(scene.family), kDefinition})
                    .status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(scene.instance).status == OVSPATIALQUERY_API_SUCCESS);
    scene = Scene{};
}

// Registers the family, publishes one object at the origin and creates one exact CPU plan
// per batch reader: the admissions a reader runs after. The first refusal releases the
// scene and is returned.
ovspatialquery_api_status_t admitScene(Family family, uint64_t budget, Scene& scene)
{
    scene.family = family;
    const ovspatialquery_instance_desc_t desc{8u, 1u, budget, nullptr, 0u};
    ovspatialquery_api_status_t          status = ovspatialquery_create_instance(&desc, &scene.instance).status;
    if (status != OVSPATIALQUERY_API_SUCCESS)
        return status;
    status           = registerFamily(scene.instance, family);
    scene.registered = status == OVSPATIALQUERY_API_SUCCESS;
    const ovspatialquery_population_desc_t populationDesc{1u, 1u};
    if (status == OVSPATIALQUERY_API_SUCCESS)
        status = ovspatialquery_create_explicit_population(scene.instance, &populationDesc, &scene.population).status;
    InstanceRows rows;
    rows.add(kObject, static_cast<int32_t>(geometryType(family)), kDefinition, 0.0, 0.0, 0.0);
    const ovspatialquery_explicit_population_set_t set{1, rows.table(), nullptr, 0u};
    if (status == OVSPATIALQUERY_API_SUCCESS)
        status = finish(scene.instance,
                        ovspatialquery_enqueue_explicit_population_set(scene.instance, scene.population, &set));

    const ovspatialquery_target_ref_t  target = populationRef(scene.population);
    ovspatialquery_raycast_plan_desc_t ray{};
    ray.target                 = target;
    ray.hit_mode               = OVSPATIALQUERY_HIT_MODE_CLOSEST;
    ray.precision              = OVSPATIALQUERY_QUERY_EXACT;
    ray.device                 = {kDLCPU, 0};
    ray.maximum_query_count    = 1u;
    ray.maximum_hits_per_query = 1u;
    ray.hit_capacity_per_slot  = 1u;
    ray.result_slot_count      = 1u;
    ray.result_fields          = OVSPATIALQUERY_BATCH_RESULT_DISTANCE;
    if (status == OVSPATIALQUERY_API_SUCCESS)
        status = ovspatialquery_create_raycast_plan(scene.instance, &ray, &scene.plans[0]).status;
    ovspatialquery_sphere_sweep_plan_desc_t sphere{};
    sphere.target                 = target;
    sphere.precision              = OVSPATIALQUERY_QUERY_EXACT;
    sphere.radius                 = kSweepRadius;
    sphere.device                 = {kDLCPU, 0};
    sphere.maximum_query_count    = 1u;
    sphere.result_slot_count      = 1u;
    sphere.result_fields          = OVSPATIALQUERY_BATCH_RESULT_DISTANCE;
    sphere.hit_capacity_per_slot  = 1u;
    sphere.hit_mode               = OVSPATIALQUERY_HIT_MODE_CLOSEST;
    sphere.maximum_hits_per_query = 1u;
    if (status == OVSPATIALQUERY_API_SUCCESS)
        status = ovspatialquery_create_sphere_sweep_plan(scene.instance, &sphere, &scene.plans[1]).status;
    ovspatialquery_box_sweep_plan_desc_t box{};
    box.target    = target;
    box.precision = OVSPATIALQUERY_QUERY_EXACT;
    std::fill(std::begin(box.half_extents), std::end(box.half_extents), kBoxHalf);
    box.device                 = {kDLCPU, 0};
    box.maximum_query_count    = 1u;
    box.result_slot_count      = 1u;
    box.result_fields          = OVSPATIALQUERY_BATCH_RESULT_DISTANCE;
    box.hit_capacity_per_slot  = 1u;
    box.hit_mode               = OVSPATIALQUERY_HIT_MODE_CLOSEST;
    box.maximum_hits_per_query = 1u;
    if (status == OVSPATIALQUERY_API_SUCCESS)
        status = ovspatialquery_create_box_sweep_plan(scene.instance, &box, &scene.plans[2]).status;
    ovspatialquery_overlap_plan_desc_t overlap{};
    overlap.target                   = target;
    overlap.hit_mode                 = OVSPATIALQUERY_HIT_MODE_ANY;
    overlap.precision                = OVSPATIALQUERY_QUERY_EXACT;
    overlap.shape.kind               = OVSPATIALQUERY_OVERLAP_SPHERE;
    overlap.shape.data.sphere.radius = 0.5;
    overlap.device                   = {kDLCPU, 0};
    overlap.maximum_query_count      = 1u;
    overlap.maximum_hits_per_query   = 1u;
    overlap.hit_capacity_per_slot    = 1u;
    overlap.result_slot_count        = 1u;
    if (status == OVSPATIALQUERY_API_SUCCESS)
        status = ovspatialquery_create_overlap_plan(scene.instance, &overlap, &scene.plans[3]).status;
    if (status != OVSPATIALQUERY_API_SUCCESS)
        releaseScene(scene);
    return status;
}

// The smallest host budget under which every admission of the family's scene succeeds,
// transient staging included: one byte less refuses one of them with HOST_BUDGET.
uint64_t admissionBudget(Family family)
{
    uint64_t refused = 0u, admitted = UINT64_C(1) << 30u;
    while (admitted - refused > 1u)
    {
        const uint64_t              budget = refused + (admitted - refused) / 2u;
        Scene                       scene;
        ovspatialquery_api_status_t status = admitScene(family, budget, scene);
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            admitted = budget;
            releaseScene(scene);
        }
        else
        {
            REQUIRE(status == OVSPATIALQUERY_API_HOST_BUDGET);
            refused = budget;
        }
    }
    return admitted;
}

// Runs one reader. A batch reader's hit is checked on its fetched columns; `outCharge`
// receives what the launch held while its operation was live.
void runReader(const Scene& scene, Family family, Reader reader, uint64_t* outCharge = nullptr)
{
    double                        origin[3] = {kStart, kY, kZ}, maximum = 10.0, overlapCenter[3] = {-1.25, kY, kZ};
    float                         direction[3] = {1.0f, 0.0f, 0.0f}, identity[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    double                        halves[3] = {kBoxHalf, kBoxHalf, kBoxHalf};
    ovspatialquery_query_common_t common{};
    common.target     = populationTarget(scene.population);
    common.precision  = OVSPATIALQUERY_QUERY_EXACT;
    common.query_bits = participationBits(UINT64_MAX, UINT64_MAX);
    double distance   = -1.0;
    switch (reader)
    {
    case Reader::raycast: {
        ovspatialquery_raycast_desc_t desc{};
        desc.common = common;
        std::copy_n(origin, 3, desc.ray.origin);
        std::copy_n(direction, 3, desc.ray.unit_direction);
        desc.ray.max_distance = maximum;
        ovspatialquery_raycast_single_result_t result{};
        REQUIRE(ovspatialquery_raycast_closest_sync(scene.instance, &desc, &result).status ==
                OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(result.has_hit && result.hit.data.exact.identity.object_id.value == kObject);
        distance = result.hit.data.exact.distance;
        break;
    }
    case Reader::sphereSweep:
    case Reader::boxSweep: {
        ovspatialquery_sphere_sweep_closest_result_t result{};
        if (reader == Reader::sphereSweep)
        {
            ovspatialquery_sphere_sweep_desc_t desc{};
            desc.common = common;
            std::copy_n(origin, 3, desc.sweep.center);
            desc.sweep.radius = kSweepRadius;
            std::copy_n(direction, 3, desc.sweep.unit_direction);
            desc.sweep.max_distance = maximum;
            REQUIRE(ovspatialquery_sweep_sphere_closest_sync(scene.instance, &desc, &result).status ==
                    OVSPATIALQUERY_API_SUCCESS);
        }
        else
        {
            ovspatialquery_box_sweep_desc_t desc{};
            desc.common = common;
            std::copy_n(origin, 3, desc.sweep.center);
            std::copy_n(halves, 3, desc.sweep.half_extents);
            std::copy_n(identity, 4, desc.sweep.rotation_xyzw);
            std::copy_n(direction, 3, desc.sweep.unit_direction);
            desc.sweep.max_distance = maximum;
            REQUIRE(ovspatialquery_sweep_box_closest_sync(scene.instance, &desc, &result).status ==
                    OVSPATIALQUERY_API_SUCCESS);
        }
        REQUIRE(result.has_hit && result.hit.data.exact.identity.object_id.value == kObject);
        distance = result.hit.data.exact.distance;
        break;
    }
    case Reader::overlap: {
        ovspatialquery_overlap_desc_t desc{};
        desc.common     = common;
        desc.shape.kind = OVSPATIALQUERY_OVERLAP_SPHERE;
        std::copy_n(overlapCenter, 3, desc.shape.data.sphere.center);
        desc.shape.data.sphere.radius = 0.5;
        ovspatialquery_overlap_any_result_t result{};
        REQUIRE(ovspatialquery_overlap_any_sync(scene.instance, &desc, &result).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(result.has_hit && result.hit.identity.object_id.value == kObject);
        return;
    }
    default: {
        const uint64_t                  before = hostUsage(scene.instance);
        InputArray                      origins(origin, 1, 3, kDLFloat), directions(direction, 1, 3, kDLFloat);
        InputArray                      maximums(&maximum, 1, kDLFloat), rotations(identity, 1, 4, kDLFloat);
        InputArray                      centers(overlapCenter, 1, 3, kDLFloat);
        ovspatialquery_enqueue_result_t launch{};
        if (reader == Reader::raycastBatch)
        {
            ovspatialquery_raycast_batch_t batch{};
            batch.ordinal.kind    = OVSPATIALQUERY_ORDINAL_HEAD;
            batch.query_count     = 1u;
            batch.origins         = &origins.value;
            batch.unit_directions = &directions.value;
            batch.max_distances   = &maximums.value;
            ovspatialquery_raycast_batch_reservation_t reservation{};
            launch = ovspatialquery_enqueue_raycast_plan(scene.instance, scene.plans[0], &batch, &reservation);
        }
        else if (reader == Reader::sphereSweepBatch)
        {
            ovspatialquery_sphere_sweep_batch_t batch{};
            batch.ordinal.kind    = OVSPATIALQUERY_ORDINAL_HEAD;
            batch.query_count     = 1u;
            batch.centers         = &origins.value;
            batch.unit_directions = &directions.value;
            batch.max_distances   = &maximums.value;
            ovspatialquery_sphere_sweep_batch_reservation_t reservation{};
            launch = ovspatialquery_enqueue_sphere_sweep_plan(scene.instance, scene.plans[1], &batch, &reservation);
        }
        else if (reader == Reader::boxSweepBatch)
        {
            ovspatialquery_box_sweep_batch_t batch{};
            batch.ordinal.kind    = OVSPATIALQUERY_ORDINAL_HEAD;
            batch.query_count     = 1u;
            batch.centers         = &origins.value;
            batch.rotations_xyzw  = &rotations.value;
            batch.unit_directions = &directions.value;
            batch.max_distances   = &maximums.value;
            ovspatialquery_box_sweep_batch_reservation_t reservation{};
            launch = ovspatialquery_enqueue_box_sweep_plan(scene.instance, scene.plans[2], &batch, &reservation);
        }
        else
        {
            ovspatialquery_overlap_batch_t batch{};
            batch.ordinal.kind = OVSPATIALQUERY_ORDINAL_HEAD;
            batch.query_count  = 1u;
            batch.translations = &centers.value;
            ovspatialquery_overlap_batch_reservation_t reservation{};
            launch = ovspatialquery_enqueue_overlap_plan(scene.instance, scene.plans[3], &batch, &reservation);
        }
        REQUIRE(launch.status == OVSPATIALQUERY_API_SUCCESS);
        ovspatialquery_op_wait_result_t waited{};
        REQUIRE(ovspatialquery_wait_op(scene.instance, launch.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
                OVSPATIALQUERY_API_SUCCESS);
        if (waited.state != OVSPATIALQUERY_OP_COMPLETED)
        {
            const ovspatialquery_string_t error = ovspatialquery_get_last_op_error(scene.instance, launch.op_id);
            std::fprintf(stderr, "family %d reader %d ended %d: %.*s\n", static_cast<int>(family),
                         static_cast<int>(reader), static_cast<int>(waited.state), static_cast<int>(error.length),
                         error.ptr);
        }
        REQUIRE(waited.state == OVSPATIALQUERY_OP_COMPLETED);
        if (outCharge != nullptr)
        {
            const uint64_t during = hostUsage(scene.instance);
            REQUIRE(during >= before);
            *outCharge = during - before;
        }
        const DLTensor*                            counts    = nullptr;
        const DLTensor*                            distances = nullptr;
        const DLTensor*                            objects   = nullptr;
        ovspatialquery_raycast_batch_result_t      rays{};
        ovspatialquery_sphere_sweep_batch_result_t sweeps{};
        ovspatialquery_overlap_batch_result_t      overlaps{};
        if (reader == Reader::raycastBatch)
        {
            REQUIRE(ovspatialquery_fetch_raycast_batch(scene.instance, launch.op_id, &rays).status ==
                    OVSPATIALQUERY_API_SUCCESS);
            counts    = &rays.columns.hit_ranges.hit_counts;
            distances = &rays.columns.contact.exact.distances;
            objects   = &rays.columns.object_ids;
        }
        else if (reader == Reader::overlapBatch)
        {
            REQUIRE(ovspatialquery_fetch_overlap_batch(scene.instance, launch.op_id, &overlaps).status ==
                    OVSPATIALQUERY_API_SUCCESS);
            counts  = &overlaps.columns.hit_ranges.hit_counts;
            objects = &overlaps.columns.object_ids;
        }
        else
        {
            REQUIRE((reader == Reader::sphereSweepBatch
                         ? ovspatialquery_fetch_sphere_sweep_batch(scene.instance, launch.op_id, &sweeps)
                         : ovspatialquery_fetch_box_sweep_batch(scene.instance, launch.op_id, &sweeps))
                        .status == OVSPATIALQUERY_API_SUCCESS);
            counts    = &sweeps.columns.hit_ranges.hit_counts;
            distances = &sweeps.columns.contact.exact.distances;
            objects   = &sweeps.columns.object_ids;
        }
        REQUIRE(static_cast<const uint64_t*>(counts->data)[0] == 1u);
        REQUIRE(static_cast<const uint64_t*>(objects->data)[0] == kObject);
        if (distances != nullptr)
            distance = static_cast<const double*>(distances->data)[0];
        REQUIRE(ovspatialquery_destroy_op(scene.instance, &launch.op_id).status == OVSPATIALQUERY_API_SUCCESS);
        if (distances == nullptr)
            return;
        break;
    }
    }
    if (std::abs(distance - expectedDistance(family, reader)) > 1.0e-6)
        std::fprintf(stderr, "family %d reader %d: distance %.9f, expected %.9f\n", static_cast<int>(family),
                     static_cast<int>(reader), distance, expectedDistance(family, reader));
    REQUIRE(std::abs(distance - expectedDistance(family, reader)) <= 1.0e-6);
}

// The synchronous readers' own scratch, which memory usage cannot observe after they
// return; far below what building the mesh family's 32770-triangle structure takes.
constexpr uint64_t kSynchronousReaderScratch = 256u * 1024u;

void testEveryFirstReaderSucceedsWithinItsAdmissions()
{
    for (Family family : kFamilies)
    {
        // Each batch reader's own charge, measured with room to spare on the scene the
        // synchronous readers have built.
        Scene measured;
        REQUIRE(admitScene(family, UINT64_C(1) << 30u, measured) == OVSPATIALQUERY_API_SUCCESS);
        for (Reader reader : {Reader::raycast, Reader::sphereSweep, Reader::boxSweep, Reader::overlap})
            runReader(measured, family, reader);
        uint64_t readerCharge = kSynchronousReaderScratch;
        for (Reader reader :
             {Reader::raycastBatch, Reader::sphereSweepBatch, Reader::boxSweepBatch, Reader::overlapBatch})
        {
            uint64_t charge = 0u;
            runReader(measured, family, reader, &charge);
            readerCharge = std::max(readerCharge, charge);
        }
        releaseScene(measured);

        const uint64_t budget = admissionBudget(family) + readerCharge;
        for (Reader reader : kReaders)
        {
            Scene scene;
            REQUIRE(admitScene(family, budget, scene) == OVSPATIALQUERY_API_SUCCESS);
            runReader(scene, family, reader);
            releaseScene(scene);
        }
    }
}

} // namespace

int main()
{
    testEveryFirstReaderSucceedsWithinItsAdmissions();
    return 0;
}
