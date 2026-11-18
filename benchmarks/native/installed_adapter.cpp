/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter comparison across the direct baseline and both installed paths.
#include "installed_adapter.h"
#include "../../support/native/cuda_driver.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <thread>

namespace bench
{
struct Spatial::BatchPlan::CudaState
{
    CudaDriver                cuda{CudaDriver::Ownership::tracked};
    CudaDriver::Stream        stream    = nullptr;
    CudaDriver::Event         ready     = nullptr;
    CudaDriver::DevicePointer positions = 0, rotations = 0, directions = 0, minimums = 0, maximums = 0, bits = 0;
    std::vector<std::vector<uint8_t>> columns; // Host mirrors of the fetched device columns.
};

InstanceRows::InstanceRows(const Fixture& f, size_t count)
{
    const auto& objects = f.objects;
    for (size_t i = 0; i < count; ++i)
    {
        const auto& ob = objects[i];
        ids.push_back(ob.id);
        definitions.push_back(ob.geometryValue());
        bits.insert(bits.end(), {ob.bits, 0u}); // the workloads' categories are bits 0-63
        types.push_back(ob.geometryType());
        positions.push_back(ob.p);
        rotations.push_back(ob.q);
    }
    idInput                  = std::make_unique<Input>(ids.data(), count, 0, kDLUInt, 64);
    defInput                 = std::make_unique<Input>(definitions.data(), count, 0, kDLUInt, 64);
    bitInput                 = std::make_unique<Input>(bits.data(), count, 2, kDLUInt, 64);
    typeInput                = std::make_unique<Input>(types.data(), count, 0, kDLInt, 32);
    posInput                 = std::make_unique<Input>(positions.data(), count, 3, kDLFloat, 64);
    rotInput                 = std::make_unique<Input>(rotations.data(), count, 4, kDLFloat, 32);
    table.object_values      = &idInput->value;
    table.geometry_values    = &defInput->value;
    table.participation_bits = &bitInput->value;
    table.geometry_types     = &typeInput->value;
    table.translations       = &posInput->value;
    table.rotations_xyzw     = &rotInput->value;
}
void Spatial::checkRelease(ovspatialquery_result_t result, const char* action) noexcept
{
    if (result.status != OVSPATIALQUERY_API_SUCCESS)
    {
        std::cerr << action << " failed during teardown: " << result.status << '\n';
        std::exit(EXIT_FAILURE);
    }
}
ovspatialquery_ordinal_t Spatial::finish(ovspatialquery_enqueue_result_t admitted, bool publication, const char* action)
{
    ovspatialquery_ordinal_t published = 0;
    check({admitted.status}, action);
    ovspatialquery_op_wait_result_t wait{};
    check(ovspatialquery_wait_op(instance, admitted.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &wait), "wait");
    if (wait.state != OVSPATIALQUERY_OP_COMPLETED)
    {
        const auto        error = ovspatialquery_get_last_op_error(instance, admitted.op_id);
        const std::string message(error.ptr ? error.ptr : "", size_t(error.length));
        check(ovspatialquery_destroy_op(instance, &admitted.op_id), "destroy failed op");
        throw std::runtime_error("operation did not complete: " + message);
    }
    if (publication)
    {
        ovspatialquery_population_update_result_t result{};
        check(ovspatialquery_fetch_population_update(instance, admitted.op_id, &result), "fetch publication");
        published = result.published_ordinal;
    }
    check(ovspatialquery_destroy_op(instance, &admitted.op_id), "destroy op");
    return published;
}
ovspatialquery_population_handle_t Spatial::publishPopulation(const Fixture& f, ovspatialquery_ordinal_t& outOrdinal)
{
    ovspatialquery_population_desc_t   pd{2, 1};
    ovspatialquery_population_handle_t handle = OVSPATIALQUERY_INVALID_POPULATION;
    check(ovspatialquery_create_explicit_population(instance, &pd, &handle), "create population");
    InstanceRows                             rows(f, f.objects.size());
    ovspatialquery_explicit_population_set_t set{};
    set.target_ordinal = 1;
    set.instances      = rows.table;
    outOrdinal = finish(ovspatialquery_enqueue_explicit_population_set(instance, handle, &set), true, "population set");
    return handle;
}
namespace
{
bool gMeshSourceFaces = false;

ovspatialquery_instance_handle_t createInstance()
{
    ovspatialquery_instance_desc_t desc{};
    desc.operation_capacity  = 4;
    desc.max_cpu_concurrency = 1;
    desc.host_byte_budget    = hostBudget;
    // The warehouse demo's instance budget for CUDA device 0; CPU-only hosts leave it unused.
    static const ovspatialquery_device_budget_t deviceBudget{{kDLCUDA, 0}, deviceByteBudget};
    desc.device_budgets                       = &deviceBudget;
    desc.device_budget_count                  = 1;
    ovspatialquery_instance_handle_t instance = OVSPATIALQUERY_INVALID_INSTANCE;
    check(ovspatialquery_create_instance(&desc, &instance), "create instance");
    return instance;
}
} // namespace
void registerMeshSourceFaces(bool enabled) { gMeshSourceFaces = enabled; }

Spatial::Spatial(InstanceOnly) : instance(createInstance()) {}
Spatial::Spatial(const Fixture& f) : instance(createInstance()), overlapHits(f.objects.size())
{
    const auto geometryStart = Clock::now();
    const bool needsSphere   = std::any_of(f.objects.begin(), f.objects.end(),
                                           [](const Object& o) { return !o.mesh && !o.convex && o.sphere; });
    const bool needsBox      = std::any_of(f.objects.begin(), f.objects.end(),
                                           [](const Object& o) { return !o.mesh && !o.convex && !o.sphere && !o.capsule; });
    if (needsSphere || needsBox)
    {
        uint64_t id = 1;
        Input    ids(&id, 1, 0, kDLUInt, 64);
        if (needsSphere)
        {
            double                                   radius = sphereRadius;
            Input                                    radii(&radius, 1, 0, kDLFloat, 64);
            ovspatialquery_sphere_definition_table_t spheres{&ids.value, &radii.value};
            finish(ovspatialquery_enqueue_sphere_registration(instance, &spheres), false, "sphere registration");
            sphereRegistered = true;
        }
        if (needsBox)
        {
            Vec                                   half = boxHalf;
            Input                                 halves(half.data(), 1, 3, kDLFloat, 64);
            ovspatialquery_box_definition_table_t boxes{&ids.value, &halves.value};
            finish(ovspatialquery_enqueue_box_registration(instance, &boxes), false, "box registration");
            boxRegistered = true;
        }
    }
    if (std::any_of(f.objects.begin(), f.objects.end(), [](const Object& object) { return object.capsule; }))
    {
        uint64_t id     = 1;
        double   radius = .5, halfSegment = .625;
        Input    ids(&id, 1, 0, kDLUInt, 64), radii(&radius, 1, 0, kDLFloat, 64),
            segments(&halfSegment, 1, 0, kDLFloat, 64);
        ovspatialquery_capsule_definition_table_t capsules{&ids.value, &radii.value, &segments.value};
        finish(ovspatialquery_enqueue_capsule_registration(instance, &capsules), false, "capsule registration");
        capsuleRegistered = true;
    }
    for (size_t i = 0; i < f.meshes.size(); ++i)
    {
        const auto& mesh  = f.meshes[i];
        uint64_t    value = i + 1, zero = 0, vertices = mesh.vertices.size(), triangles = mesh.triangles.size();
        uint8_t     sided = mesh.doubleSided;
        Input       valueIn(&value, 1, 0, kDLUInt, 64), zeroIn(&zero, 1, 0, kDLUInt, 64),
            verticesIn(&vertices, 1, 0, kDLUInt, 64), trianglesIn(&triangles, 1, 0, kDLUInt, 64),
            points(const_cast<std::array<float, 3>*>(mesh.vertices.data()), vertices, 3, kDLFloat, 32),
            indices(const_cast<std::array<uint32_t, 3>*>(mesh.triangles.data()), triangles, 3, kDLUInt, 32),
            sidedIn(&sided, 1, 0, kDLUInt, 8);
        ovspatialquery_triangle_mesh_definition_table_t table{
            &valueIn.value, &zeroIn.value,  &verticesIn.value, &zeroIn.value, &trianglesIn.value,
            &points.value,  &indices.value, &sidedIn.value,    nullptr,       nullptr,
            nullptr};
        // The submitted numbering as source faces, so what a hit reports stays the same.
        std::vector<uint64_t> faces;
        std::vector<uint32_t> corners;
        std::optional<Input>  facesIn, cornersIn;
        if (gMeshSourceFaces)
        {
            faces.resize(triangles);
            corners.resize(3u * triangles);
            for (size_t t = 0; t < triangles; ++t)
            {
                faces[t] = t;
                for (uint32_t c = 0; c < 3u; ++c)
                    corners[3u * t + c] = c;
            }
            facesIn.emplace(faces.data(), triangles, 0, kDLUInt, 64);
            cornersIn.emplace(corners.data(), triangles, 3, kDLUInt, 32);
            table.source_face_indices        = &facesIn->value;
            table.source_face_corner_indices = &cornersIn->value;
        }
        finish(ovspatialquery_enqueue_triangle_mesh_registration(instance, &table), false, "mesh registration");
        ++meshDefinitions;
    }
    for (size_t i = 0; i < f.convexes.size(); ++i)
    {
        const auto& vertices = f.convexes[i];
        uint64_t    value = i + 1, zero = 0, count = vertices.size();
        Input valueIn(&value, 1, 0, kDLUInt, 64), zeroIn(&zero, 1, 0, kDLUInt, 64), countIn(&count, 1, 0, kDLUInt, 64),
            points(const_cast<std::array<float, 3>*>(vertices.data()), count, 3, kDLFloat, 32);
        ovspatialquery_convex_definition_table_t table{};
        table.definition_values = &valueIn.value;
        table.vertex_offsets    = &zeroIn.value;
        table.vertex_counts     = &countIn.value;
        table.vertices          = &points.value;
        finish(ovspatialquery_enqueue_convex_registration(instance, &table), false, "convex registration");
        ++convexDefinitions;
    }
    geometrySetupMs             = elapsed(geometryStart);
    const auto publicationStart = Clock::now();
    population                  = publishPopulation(f, ordinal);
    std::vector<uint64_t> selected;
    for (const auto& ob : f.objects)
        if (ob.id % 2 == 0)
            selected.push_back(ob.id);
    Input                               subset(selected.data(), selected.size(), 0, kDLUInt, 64);
    ovspatialquery_explicit_view_desc_t vd{&subset.value};
    check(ovspatialquery_create_explicit_view(instance, population, &vd, &view), "create even view");
    publicationMs = elapsed(publicationStart);
}
Spatial::~Spatial()
{
    defaultPlan.reset();
    if (view != OVSPATIALQUERY_INVALID_VIEW)
        checkRelease(ovspatialquery_destroy_view(instance, view), "destroy view");
    if (population != OVSPATIALQUERY_INVALID_POPULATION)
        checkRelease(ovspatialquery_destroy_population(instance, population), "destroy population");
    if (instance != OVSPATIALQUERY_INVALID_INSTANCE)
    {
        for (size_t i = 0; i < convexDefinitions; ++i)
            checkRelease(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_CONVEX, i + 1}),
                         "destroy convex definition");
        for (size_t i = 0; i < meshDefinitions; ++i)
            checkRelease(
                ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, i + 1}),
                "destroy mesh definition");
        if (capsuleRegistered)
            checkRelease(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_CAPSULE, 1}),
                         "destroy capsule definition");
        if (boxRegistered)
            checkRelease(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_BOX, 1}),
                         "destroy box definition");
        if (sphereRegistered)
            checkRelease(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_SPHERE, 1}),
                         "destroy sphere definition");
        checkRelease(ovspatialquery_destroy_instance(instance), "destroy instance");
    }
}
ovspatialquery_target_ref_t Spatial::target(bool even) const
{
    ovspatialquery_target_ref_t result{};
    result.kind = even ? OVSPATIALQUERY_TARGET_VIEW : OVSPATIALQUERY_TARGET_POPULATION;
    if (even)
        result.handle.view = view;
    else
        result.handle.population = population;
    return result;
}
ovspatialquery_query_common_t Spatial::common(const Fixture& f, const Query& q, ovspatialquery_ordinal_t exact) const
{
    ovspatialquery_query_common_t c{};
    c.target.target = target(f.even);
    if (exact >= 0)
        c.target.ordinal = {OVSPATIALQUERY_ORDINAL_RETAINED_EXACT, exact};
    c.precision  = f.boundsCandidates ? OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE : OVSPATIALQUERY_QUERY_EXACT;
    c.query_bits = {{q.bits, 0u}};
    return c;
}
ovspatialquery_overlap_desc_t Spatial::overlap(const Fixture& f, const Query& q, ovspatialquery_ordinal_t exact) const
{
    ovspatialquery_overlap_desc_t d{};
    d.common = common(f, q, exact);
    if (sphereShape(f.kind))
    {
        d.shape.kind = OVSPATIALQUERY_OVERLAP_SPHERE;
        std::copy(q.p.begin(), q.p.end(), d.shape.data.sphere.center);
        d.shape.data.sphere.radius = f.rowRadii ? q.radius : f.radius;
        return d;
    }
    d.shape.kind = OVSPATIALQUERY_OVERLAP_BOX;
    std::copy(q.p.begin(), q.p.end(), d.shape.data.box.center);
    std::copy(q.q.begin(), q.q.end(), d.shape.data.box.rotation_xyzw);
    std::copy(f.half.begin(), f.half.end(), d.shape.data.box.half_extents);
    return d;
}
void Spatial::update(const Fixture& f, size_t count) { update(f, count, population, ordinal); }
void Spatial::update(const Fixture& f, size_t count, ovspatialquery_population_handle_t target,
                     ovspatialquery_ordinal_t& ioOrdinal)
{
    InstanceRows                                rows(f, count);
    Input                                       removed(nullptr, 0, 0, kDLUInt, 64);
    ovspatialquery_explicit_population_update_t change{};
    change.removed_object_values = &removed.value;
    change.base_ordinal          = ioOrdinal;
    change.target_ordinal        = ioOrdinal + 1;
    change.changed_instances     = rows.table;
    if (f.posesOnly)
    {
        change.kind                                 = OVSPATIALQUERY_EXPLICIT_UPDATE_POSES;
        change.changed_instances.geometry_types     = nullptr;
        change.changed_instances.geometry_values    = nullptr;
        change.changed_instances.participation_bits = nullptr;
    }
    ioOrdinal =
        finish(ovspatialquery_enqueue_explicit_population_update(instance, target, &change), true, "population update");
}
void Spatial::scalar(const Fixture& f, Results& result, ovspatialquery_ordinal_t exact)
{
    const auto wanted = exact >= 0 ? exact : ordinal;
    for (size_t i = 0; i < f.queries.size(); ++i)
    {
        const auto& q    = f.queries[i];
        auto&       hits = result[i].hits;
        hits.clear();
        if (f.kind == QueryKind::rays)
        {
            ovspatialquery_raycast_desc_t desc{};
            desc.common = common(f, q, exact);
            std::copy(q.p.begin(), q.p.end(), desc.ray.origin);
            std::copy(q.direction.begin(), q.direction.end(), desc.ray.unit_direction);
            desc.ray.min_distance = q.nearDistance;
            desc.ray.max_distance = q.farDistance;
            ovspatialquery_raycast_single_result_t r{};
            check(ovspatialquery_raycast_closest_sync(instance, &desc, &r), "raycast");
            require(r.source_ordinal == wanted, "ray ordinal mismatch");
            if (r.has_hit && f.boundsCandidates)
            {
                require(r.hit.truth == OVSPATIALQUERY_HIT_BOUNDS_CANDIDATE, "ray precision mismatch");
                const auto& h = r.hit.data.bounds_candidate;
                hits.push_back({h.identity.object_id.value,
                                h.distance,
                                {h.bounds_position[0], h.bounds_position[1], h.bounds_position[2]},
                                {h.bounds_normal[0], h.bounds_normal[1], h.bounds_normal[2]},
                                h.identity.geometry.type,
                                h.identity.geometry.value});
            }
            else if (r.has_hit)
            {
                require(r.hit.truth == OVSPATIALQUERY_HIT_EXACT_SURFACE, "ray precision mismatch");
                const auto& h = r.hit.data.exact;
                hits.push_back({h.identity.object_id.value,
                                h.distance,
                                {h.position[0], h.position[1], h.position[2]},
                                {h.normal[0], h.normal[1], h.normal[2]},
                                h.identity.geometry.type,
                                h.identity.geometry.value,
                                h.feature.kind == OVSPATIALQUERY_GEOMETRY_FEATURE_TRIANGLE
                                    ? h.feature.data.triangle.source_face_index
                                    : UINT64_MAX});
            }
        }
        else if (f.kind == QueryKind::sweep)
        {
            ovspatialquery_sphere_sweep_desc_t d{};
            d.common = common(f, q, exact);
            std::copy(q.p.begin(), q.p.end(), d.sweep.center);
            std::copy(q.direction.begin(), q.direction.end(), d.sweep.unit_direction);
            d.sweep.radius       = f.rowRadii ? q.radius : f.radius;
            d.sweep.min_distance = q.nearDistance;
            d.sweep.max_distance = q.farDistance;
            ovspatialquery_sphere_sweep_closest_result_t r{};
            check(ovspatialquery_sweep_sphere_closest_sync(instance, &d, &r), "sphere sweep");
            require(r.source_ordinal == wanted, "sweep ordinal mismatch");
            if (r.has_hit)
            {
                require(r.hit.truth == OVSPATIALQUERY_HIT_EXACT_SURFACE, "sweep precision mismatch");
                const auto& h = r.hit.data.exact;
                hits.push_back({h.identity.object_id.value,
                                h.distance,
                                {h.object_position[0], h.object_position[1], h.object_position[2]},
                                {h.normal[0], h.normal[1], h.normal[2]},
                                h.identity.geometry.type,
                                h.identity.geometry.value,
                                h.feature.kind == OVSPATIALQUERY_GEOMETRY_FEATURE_TRIANGLE
                                    ? h.feature.data.triangle.source_face_index
                                    : UINT64_MAX});
            }
        }
        else if (f.kind == QueryKind::boxSweep)
        {
            ovspatialquery_box_sweep_desc_t d{};
            d.common = common(f, q, exact);
            std::copy(q.p.begin(), q.p.end(), d.sweep.center);
            std::copy(f.half.begin(), f.half.end(), d.sweep.half_extents);
            std::copy(q.q.begin(), q.q.end(), d.sweep.rotation_xyzw);
            std::copy(q.direction.begin(), q.direction.end(), d.sweep.unit_direction);
            d.sweep.min_distance = q.nearDistance;
            d.sweep.max_distance = q.farDistance;
            ovspatialquery_box_sweep_closest_result_t r{};
            check(ovspatialquery_sweep_box_closest_sync(instance, &d, &r), "box sweep");
            require(r.source_ordinal == wanted, "box sweep ordinal mismatch");
            if (r.has_hit)
            {
                require(r.hit.truth == OVSPATIALQUERY_HIT_EXACT_SURFACE, "box sweep precision mismatch");
                const auto& h = r.hit.data.exact;
                hits.push_back({h.identity.object_id.value,
                                h.distance,
                                {h.object_position[0], h.object_position[1], h.object_position[2]},
                                {h.normal[0], h.normal[1], h.normal[2]},
                                h.identity.geometry.type,
                                h.identity.geometry.value,
                                h.feature.kind == OVSPATIALQUERY_GEOMETRY_FEATURE_TRIANGLE
                                    ? h.feature.data.triangle.source_face_index
                                    : UINT64_MAX});
            }
        }
        else
        {
            auto desc = overlap(f, q, exact);
            if (anyOverlap(f.kind))
            {
                ovspatialquery_overlap_any_result_t r{};
                check(ovspatialquery_overlap_any_sync(instance, &desc, &r), "overlap any");
                require(r.source_ordinal == wanted, "any ordinal mismatch");
                if (r.has_hit)
                {
                    require(r.hit.truth == OVSPATIALQUERY_HIT_EXACT_SURFACE, "any precision mismatch");
                    hits.push_back(Hit{1});
                }
            }
            else
            {
                ovspatialquery_overlap_hit_buffer_t buffer{overlapHits.data(), overlapHits.size()};
                ovspatialquery_multiple_result_t    r{};
                check(ovspatialquery_overlap_multiple_sync(instance, &desc, &buffer, &r), "overlap multiple");
                require(r.source_ordinal == wanted && !r.truncated, "multiple ordinal/truncation mismatch");
                for (size_t j = 0; j < r.stored_hit_count; ++j)
                {
                    require(overlapHits[j].truth == OVSPATIALQUERY_HIT_EXACT_SURFACE, "multiple precision mismatch");
                    hits.push_back(Hit{overlapHits[j].identity.object_id.value});
                }
            }
        }
    }
}
bool cudaPlansAvailable(std::string& message)
{
    try
    {
        CudaDriver probe(CudaDriver::Ownership::tracked);
    }
    catch (const std::exception& error)
    {
        message = error.what();
        return false;
    }
    Fixture f;
    f.objects.push_back({1, 1, true, {0, 0, 0}, identity});
    f.queries.resize(1);
    Spatial                            spatial(f);
    ovspatialquery_raycast_plan_desc_t d{};
    d.target                                  = spatial.target(false);
    d.precision                               = OVSPATIALQUERY_QUERY_EXACT;
    d.device                                  = {kDLCUDA, 0};
    d.maximum_query_count                     = 1;
    d.maximum_hits_per_query                  = 1;
    d.hit_capacity_per_slot                   = 1;
    d.result_slot_count                       = 1;
    ovspatialquery_batch_plan_handle_t plan   = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    const ovspatialquery_result_t      result = ovspatialquery_create_raycast_plan(spatial.instance, &d, &plan);
    if (result.status != OVSPATIALQUERY_API_SUCCESS)
    {
        message = "installed library refused a CUDA plan with status " + std::to_string(result.status);
        return false;
    }
    Spatial::checkRelease(ovspatialquery_destroy_batch_plan(spatial.instance, plan), "destroy probe plan");
    return true;
}
Spatial::BatchPlan::BatchPlan(Spatial& source, const Fixture& f, size_t maximumQueries, DLDevice planDevice,
                              ovspatialquery_query_precision_t planPrecision)
    : owner(source), instance(source.instance), device(planDevice), capacity(maximumQueries), precision(planPrecision)
{
    if (device.device_type == kDLCUDA)
    {
        cuda         = std::make_unique<CudaState>(); // Throws when the driver or device is absent.
        cuda->stream = cuda->cuda.createStream();
        cuda->ready  = cuda->cuda.createEvent();
        // Device columns sized once at plan capacity; pack() refreshes their contents.
        cuda->positions  = cuda->cuda.allocate(capacity * sizeof(Vec));
        cuda->rotations  = cuda->cuda.allocate(capacity * sizeof(std::array<float, 4>));
        cuda->directions = cuda->cuda.allocate(capacity * sizeof(std::array<float, 3>));
        cuda->minimums   = cuda->cuda.allocate(capacity * sizeof(double));
        cuda->maximums   = cuda->cuda.allocate(capacity * sizeof(double));
        cuda->bits       = cuda->cuda.allocate(capacity * 2u * sizeof(uint64_t));
        cuda->columns.resize(16);
    }
    pack(f);
    if (f.kind == QueryKind::rays)
    {
        ovspatialquery_raycast_plan_desc_t d{};
        d.target                 = owner.target(f.even);
        d.hit_mode               = OVSPATIALQUERY_HIT_MODE_CLOSEST;
        d.precision              = precision;
        d.device                 = device;
        d.maximum_query_count    = capacity;
        d.maximum_hits_per_query = 1;
        d.hit_capacity_per_slot  = capacity;
        d.result_slot_count      = 1;
        d.result_fields          = OVSPATIALQUERY_BATCH_RESULT_POSITION | OVSPATIALQUERY_BATCH_RESULT_NORMAL |
                          OVSPATIALQUERY_BATCH_RESULT_DISTANCE | OVSPATIALQUERY_BATCH_RESULT_GEOMETRY_IDENTITY;
        if (f.pointCloud)
            d.result_fields = OVSPATIALQUERY_BATCH_RESULT_POSITION | OVSPATIALQUERY_BATCH_RESULT_DISTANCE;
        else if (!f.meshes.empty())
            d.result_fields |= OVSPATIALQUERY_BATCH_RESULT_SOURCE_FEATURE;
        check(ovspatialquery_create_raycast_plan(instance, &d, &plan), "create ray plan");
    }
    else if (f.kind == QueryKind::sweep)
    {
        ovspatialquery_sphere_sweep_plan_desc_t d{};
        d.target                = owner.target(f.even);
        d.precision             = OVSPATIALQUERY_QUERY_EXACT;
        d.device                = device;
        d.radius                = f.radius;
        d.maximum_query_count   = capacity;
        d.hit_capacity_per_slot = capacity;
        d.result_slot_count     = 1;
        d.result_fields         = OVSPATIALQUERY_BATCH_RESULT_POSITION | OVSPATIALQUERY_BATCH_RESULT_NORMAL |
                          OVSPATIALQUERY_BATCH_RESULT_DISTANCE | OVSPATIALQUERY_BATCH_RESULT_GEOMETRY_IDENTITY;
        if (!f.meshes.empty())
            d.result_fields |= OVSPATIALQUERY_BATCH_RESULT_SOURCE_FEATURE;
        check(ovspatialquery_create_sphere_sweep_plan(instance, &d, &plan), "create sweep plan");
    }
    else if (f.kind == QueryKind::boxSweep)
    {
        ovspatialquery_box_sweep_plan_desc_t d{};
        d.target    = owner.target(f.even);
        d.precision = OVSPATIALQUERY_QUERY_EXACT;
        std::copy(f.half.begin(), f.half.end(), d.half_extents);
        d.device                = device;
        d.maximum_query_count   = capacity;
        d.hit_capacity_per_slot = capacity;
        d.result_slot_count     = 1;
        d.result_fields         = OVSPATIALQUERY_BATCH_RESULT_POSITION | OVSPATIALQUERY_BATCH_RESULT_NORMAL |
                          OVSPATIALQUERY_BATCH_RESULT_DISTANCE | OVSPATIALQUERY_BATCH_RESULT_GEOMETRY_IDENTITY;
        if (!f.meshes.empty())
            d.result_fields |= OVSPATIALQUERY_BATCH_RESULT_SOURCE_FEATURE;
        check(ovspatialquery_create_box_sweep_plan(instance, &d, &plan), "create box sweep plan");
    }
    else
    {
        ovspatialquery_overlap_plan_desc_t d{};
        d.target                 = owner.target(f.even);
        d.hit_mode               = (f.kind == QueryKind::zones || f.kind == QueryKind::sphereOverlap)
                                       ? OVSPATIALQUERY_HIT_MODE_MULTIPLE
                                       : OVSPATIALQUERY_HIT_MODE_ANY;
        d.precision              = OVSPATIALQUERY_QUERY_EXACT;
        d.device                 = device;
        d.maximum_query_count    = capacity;
        d.maximum_hits_per_query = (f.kind == QueryKind::zones || f.kind == QueryKind::sphereOverlap)
                                       ? (f.zoneHitCapacity ? f.zoneHitCapacity : owner.overlapHits.size())
                                       : 1;
        d.hit_capacity_per_slot  = capacity * d.maximum_hits_per_query;
        d.result_slot_count      = 1;
        if (sphereShape(f.kind))
        {
            d.shape.kind               = OVSPATIALQUERY_OVERLAP_SPHERE;
            d.shape.data.sphere.radius = f.radius;
        }
        else
        {
            d.shape.kind = OVSPATIALQUERY_OVERLAP_BOX;
            std::copy(f.half.begin(), f.half.end(), d.shape.data.box.half_extents);
        }
        check(ovspatialquery_create_overlap_plan(instance, &d, &plan), "create overlap plan");
    }
}
Spatial::BatchPlan::~BatchPlan()
{
    if (plan == OVSPATIALQUERY_INVALID_BATCH_PLAN)
        return;
    // A CUDA plan stays busy until the consumer event transferred by its last retirement completes.
    const auto deadline = Clock::now() + std::chrono::seconds(10);
    for (;;)
    {
        const auto result = ovspatialquery_destroy_batch_plan(instance, plan);
        if (result.status != OVSPATIALQUERY_API_BUSY || Clock::now() >= deadline)
        {
            checkRelease(result, "destroy batch plan");
            return;
        }
        std::this_thread::yield();
    }
}
void Spatial::BatchPlan::pack(const Fixture& f)
{
    require(f.queries.size() <= capacity, "plan query capacity exceeded");
    positions.clear();
    rotations.clear();
    directions.clear();
    minimums.clear();
    radii.clear();
    maximums.clear();
    queryBits.clear();
    const bool needsBits =
        !f.pointCloud || std::any_of(f.queries.begin(), f.queries.end(), [](const Query& q) { return q.bits != 0; });
    for (const auto& q : f.queries)
    {
        positions.push_back(q.p);
        if (!f.pointCloud)
            rotations.push_back({float(q.q[0]), float(q.q[1]), float(q.q[2]), float(q.q[3])});
        directions.push_back(q.direction);
        minimums.push_back(q.nearDistance);
        maximums.push_back(q.farDistance);
        if (f.rowRadii)
            radii.push_back(q.radius);
        if (needsBits)
            queryBits.insert(queryBits.end(), {q.bits, 0u});
    }
    const size_t n = f.queries.size();
    if (cuda)
    {
        // Refresh the device columns on the plan's stream and publish them through `ready`.
        const auto upload = [this](CudaDriver::DevicePointer destination, const void* source, size_t bytes)
        {
            if (bytes)
                cuda->cuda.uploadAsync(destination, source, bytes, cuda->stream);
        };
        upload(cuda->positions, positions.data(), n * sizeof(Vec));
        upload(cuda->directions, directions.data(), n * sizeof(std::array<float, 3>));
        upload(cuda->minimums, minimums.data(), n * sizeof(double));
        upload(cuda->maximums, maximums.data(), n * sizeof(double));
        if (!f.pointCloud)
            upload(cuda->rotations, rotations.data(), n * sizeof(std::array<float, 4>));
        if (needsBits)
            upload(cuda->bits, queryBits.data(), n * 2u * sizeof(uint64_t));
        cuda->cuda.record(cuda->ready, cuda->stream);
    }
    // A CPU column is the host vector, copied on submission; a CUDA column is its device copy.
    const auto column = [this](void* host, CudaDriver::DevicePointer CudaState::*member)
    { return cuda ? reinterpret_cast<void*>(static_cast<uintptr_t>((*cuda).*member)) : host; };
    posInput = std::make_unique<Input>(column(positions.data(), &CudaState::positions), n, 3, kDLFloat, 64, device);
    if (!f.pointCloud)
        rotInput = std::make_unique<Input>(column(rotations.data(), &CudaState::rotations), n, 4, kDLFloat, 32, device);
    dirInput = std::make_unique<Input>(column(directions.data(), &CudaState::directions), n, 3, kDLFloat, 32, device);
    minInput = std::make_unique<Input>(column(minimums.data(), &CudaState::minimums), n, 0, kDLFloat, 64, device);
    maxInput = std::make_unique<Input>(column(maximums.data(), &CudaState::maximums), n, 0, kDLFloat, 64, device);
    if (needsBits)
        bitsInput = std::make_unique<Input>(column(queryBits.data(), &CudaState::bits), n, 2, kDLUInt, 64, device);
    else
        bitsInput.reset();
    // Per-row radii are a CPU column here: the actor workloads do not run CUDA plans.
    require(!f.rowRadii || !cuda, "per-row radii are CPU-only in this adapter");
    if (f.rowRadii)
        radInput = std::make_unique<Input>(radii.data(), n, 0, kDLFloat, 64);
    else
        radInput.reset();
}
// Replaces a fetched device column with a host copy enqueued on the plan's stream, so the
// shared decoding below reads CPU and CUDA results through the same DLTensor contract.
void Spatial::BatchPlan::mirror(DLTensor& tensor, size_t index)
{
    if (tensor.data == nullptr)
        return;
    require(tensor.device.device_type == kDLCUDA, "CUDA result column is not on the device");
    const size_t rows  = size_t(tensor.shape[0]);
    const size_t inner = tensor.ndim == 2 ? size_t(tensor.shape[1]) : 1;
    require(tensor.strides == nullptr ||
                (size_t(tensor.strides[0]) == inner && (tensor.ndim < 2 || tensor.strides[1] == 1)),
            "CUDA result column is not contiguous");
    const size_t bytes   = rows * inner * (tensor.dtype.bits / 8);
    auto&        storage = cuda->columns.at(index);
    storage.resize(std::max<size_t>(bytes, 1));
    if (bytes)
        cuda->cuda.downloadAsync(storage.data(),
                                 static_cast<CudaDriver::DevicePointer>(reinterpret_cast<uintptr_t>(tensor.data)) +
                                     tensor.byte_offset,
                                 bytes, cuda->stream);
    tensor.data        = storage.data();
    tensor.byte_offset = 0;
    tensor.device      = {kDLCPU, 0};
}
void Spatial::BatchPlan::batch(const Fixture& f, Results& result, ovspatialquery_ordinal_t exact, bool* truncated)
{
    const bool sweepFamily = f.kind == QueryKind::sweep || f.kind == QueryKind::boxSweep;
    if (truncated)
        *truncated = false;
    ovspatialquery_enqueue_result_t            admitted{};
    ovspatialquery_raycast_batch_result_t      ray{};
    ovspatialquery_overlap_batch_result_t      ol{};
    ovspatialquery_sphere_sweep_batch_result_t sw{};
    ovspatialquery_ordinal_selection_t         selection{};
    if (exact >= 0)
        selection = {OVSPATIALQUERY_ORDINAL_RETAINED_EXACT, exact};
    ovspatialquery_cuda_sync_t sync{};
    if (cuda)
        sync = {1u, reinterpret_cast<uintptr_t>(cuda->stream), reinterpret_cast<uintptr_t>(cuda->ready)};
    last             = CudaTiming{};
    const auto start = Clock::now();
    if (f.kind == QueryKind::rays)
    {
        ovspatialquery_raycast_batch_t d{};
        d.ordinal         = selection;
        d.query_count     = f.queries.size();
        d.origins         = &posInput->value;
        d.unit_directions = &dirInput->value;
        d.min_distances   = &minInput->value;
        d.max_distances   = &maxInput->value;
        d.query_bits      = bitsInput ? &bitsInput->value : nullptr;
        d.cuda_sync       = sync;
        ovspatialquery_raycast_batch_reservation_t reservation{};
        admitted = ovspatialquery_enqueue_raycast_plan(instance, plan, &d, &reservation);
    }
    else if (f.kind == QueryKind::sweep)
    {
        ovspatialquery_sphere_sweep_batch_t d{};
        d.ordinal         = selection;
        d.query_count     = f.queries.size();
        d.query_bits      = bitsInput ? &bitsInput->value : nullptr;
        d.centers         = &posInput->value;
        d.unit_directions = &dirInput->value;
        d.min_distances   = &minInput->value;
        d.max_distances   = &maxInput->value;
        d.radii           = radInput ? &radInput->value : nullptr;
        d.cuda_sync       = sync;
        ovspatialquery_sphere_sweep_batch_reservation_t reservation{};
        admitted = ovspatialquery_enqueue_sphere_sweep_plan(instance, plan, &d, &reservation);
    }
    else if (f.kind == QueryKind::boxSweep)
    {
        ovspatialquery_box_sweep_batch_t d{};
        d.ordinal         = selection;
        d.query_count     = f.queries.size();
        d.query_bits      = bitsInput ? &bitsInput->value : nullptr;
        d.centers         = &posInput->value;
        d.rotations_xyzw  = &rotInput->value;
        d.unit_directions = &dirInput->value;
        d.min_distances   = &minInput->value;
        d.max_distances   = &maxInput->value;
        d.cuda_sync       = sync;
        ovspatialquery_box_sweep_batch_reservation_t reservation{};
        admitted = ovspatialquery_enqueue_box_sweep_plan(instance, plan, &d, &reservation);
    }
    else
    {
        ovspatialquery_overlap_batch_t d{};
        d.ordinal        = selection;
        d.query_count    = f.queries.size();
        d.translations   = &posInput->value;
        d.rotations_xyzw = sphereShape(f.kind) ? nullptr : &rotInput->value;
        d.query_bits     = &bitsInput->value;
        d.radii          = radInput ? &radInput->value : nullptr;
        d.cuda_sync      = sync;
        ovspatialquery_overlap_batch_reservation_t reservation{};
        admitted = ovspatialquery_enqueue_overlap_plan(instance, plan, &d, &reservation);
    }
    // The single result slot is free only once the previous CUDA retirement completed on the
    // device; a consumer waits for it like the demo does between scans. The wait counts as enqueue.
    const auto deadline = Clock::now() + std::chrono::seconds(10);
    while (cuda && admitted.status == OVSPATIALQUERY_API_RESULT_CAPACITY && Clock::now() < deadline)
    {
        std::this_thread::yield();
        if (f.kind == QueryKind::rays)
        {
            ovspatialquery_raycast_batch_t d{};
            d.ordinal         = selection;
            d.query_count     = f.queries.size();
            d.origins         = &posInput->value;
            d.unit_directions = &dirInput->value;
            d.min_distances   = &minInput->value;
            d.max_distances   = &maxInput->value;
            d.query_bits      = bitsInput ? &bitsInput->value : nullptr;
            d.cuda_sync       = sync;
            ovspatialquery_raycast_batch_reservation_t reservation{};
            admitted = ovspatialquery_enqueue_raycast_plan(instance, plan, &d, &reservation);
        }
        else if (f.kind == QueryKind::sweep)
        {
            ovspatialquery_sphere_sweep_batch_t d{};
            d.ordinal         = selection;
            d.query_count     = f.queries.size();
            d.query_bits      = bitsInput ? &bitsInput->value : nullptr;
            d.centers         = &posInput->value;
            d.unit_directions = &dirInput->value;
            d.min_distances   = &minInput->value;
            d.max_distances   = &maxInput->value;
            d.radii           = radInput ? &radInput->value : nullptr;
            d.cuda_sync       = sync;
            ovspatialquery_sphere_sweep_batch_reservation_t reservation{};
            admitted = ovspatialquery_enqueue_sphere_sweep_plan(instance, plan, &d, &reservation);
        }
        else if (f.kind == QueryKind::boxSweep)
        {
            ovspatialquery_box_sweep_batch_t d{};
            d.ordinal         = selection;
            d.query_count     = f.queries.size();
            d.query_bits      = bitsInput ? &bitsInput->value : nullptr;
            d.centers         = &posInput->value;
            d.rotations_xyzw  = &rotInput->value;
            d.unit_directions = &dirInput->value;
            d.min_distances   = &minInput->value;
            d.max_distances   = &maxInput->value;
            d.cuda_sync       = sync;
            ovspatialquery_box_sweep_batch_reservation_t reservation{};
            admitted = ovspatialquery_enqueue_box_sweep_plan(instance, plan, &d, &reservation);
        }
        else
        {
            ovspatialquery_overlap_batch_t d{};
            d.ordinal        = selection;
            d.query_count    = f.queries.size();
            d.translations   = &posInput->value;
            d.rotations_xyzw = sphereShape(f.kind) ? nullptr : &rotInput->value;
            d.query_bits     = &bitsInput->value;
            d.radii          = radInput ? &radInput->value : nullptr;
            d.cuda_sync      = sync;
            ovspatialquery_overlap_batch_reservation_t reservation{};
            admitted = ovspatialquery_enqueue_overlap_plan(instance, plan, &d, &reservation);
        }
    }
    check({admitted.status}, "enqueue batch");
    last.enqueue = elapsed(start);
    ovspatialquery_op_wait_result_t waited{};
    check(ovspatialquery_wait_op(instance, admitted.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited), "wait batch");
    last.wait = elapsed(start) - last.enqueue;
    if (waited.state != OVSPATIALQUERY_OP_COMPLETED)
    {
        const auto error = ovspatialquery_get_last_op_error(instance, admitted.op_id);
        throw std::runtime_error("batch operation failure: " +
                                 std::string(error.ptr ? error.ptr : "", size_t(error.length)));
    }
    if (f.kind == QueryKind::rays)
        check(ovspatialquery_fetch_raycast_batch(instance, admitted.op_id, &ray), "fetch ray batch");
    else if (f.kind == QueryKind::sweep)
        check(ovspatialquery_fetch_sphere_sweep_batch(instance, admitted.op_id, &sw), "fetch sweep batch");
    else if (f.kind == QueryKind::boxSweep)
        check(ovspatialquery_fetch_box_sweep_batch(instance, admitted.op_id, &sw), "fetch box sweep batch");
    else
        check(ovspatialquery_fetch_overlap_batch(instance, admitted.op_id, &ol), "fetch overlap batch");
    last.fetch               = elapsed(start) - last.enqueue - last.wait;
    const auto& commonResult = f.kind == QueryKind::rays ? ray.common : (sweepFamily ? sw.common : ol.common);
    const bool  bounds       = precision == OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    require(commonResult.source_ordinal == (exact >= 0 ? exact : owner.ordinal) &&
                commonResult.truth ==
                    (bounds ? OVSPATIALQUERY_HIT_BOUNDS_CANDIDATE : OVSPATIALQUERY_HIT_EXACT_SURFACE) &&
                commonResult.execution_device.device_type == device.device_type,
            "batch version/precision/device mismatch");
    CudaDriver::Event consumerDone = nullptr;
    if (cuda)
    {
        // Consume the library's completion event on the plan's stream, mirror every present
        // column to the host, then wait for the mirrors; the consumer event retires the operation.
        require(commonResult.result_ready.value != 0, "CUDA result has no completion event");
        cuda->cuda.wait(cuda->stream, reinterpret_cast<CudaDriver::Event>(commonResult.result_ready.value));
        auto*  r     = f.kind == QueryKind::rays ? &ray.columns.hit_ranges
                                                 : (sweepFamily ? &sw.columns.hit_ranges : &ol.columns.hit_ranges);
        size_t index = 0;
        for (DLTensor* t : {&r->query_status, &r->stored_hit_count, &r->hit_offsets, &r->hit_counts, &r->truncated})
            mirror(*t, index++);
        // A candidate's position, normal and distance lead its arm as an exact hit's lead the other.
        if (f.kind == QueryKind::rays && bounds)
            for (DLTensor* t :
                 {&ray.columns.object_ids, &ray.columns.geometry.types, &ray.columns.geometry.values,
                  &ray.columns.contact.candidate.bounds_positions, &ray.columns.contact.candidate.bounds_normals,
                  &ray.columns.contact.candidate.distances})
                mirror(*t, index++);
        else if (f.kind == QueryKind::rays)
            for (DLTensor* t : {&ray.columns.object_ids, &ray.columns.geometry.types, &ray.columns.geometry.values,
                                &ray.columns.contact.exact.positions, &ray.columns.contact.exact.normals,
                                &ray.columns.contact.exact.distances, &ray.columns.contact.exact.features.kinds,
                                &ray.columns.contact.exact.features.source_face_indices,
                                &ray.columns.contact.exact.features.source_corner_indices,
                                &ray.columns.contact.exact.features.barycentrics})
                mirror(*t, index++);
        else if (sweepFamily)
            for (DLTensor* t : {&sw.columns.object_ids, &sw.columns.geometry.types, &sw.columns.geometry.values,
                                &sw.columns.contact.exact.positions, &sw.columns.contact.exact.normals,
                                &sw.columns.contact.exact.distances, &sw.columns.contact.exact.features.kinds,
                                &sw.columns.contact.exact.features.source_face_indices,
                                &sw.columns.contact.exact.features.source_corner_indices,
                                &sw.columns.contact.exact.features.barycentrics})
                mirror(*t, index++);
        else
            for (DLTensor* t : {&ol.columns.object_ids, &ol.columns.geometry.types, &ol.columns.geometry.values})
                mirror(*t, index++);
        consumerDone = cuda->cuda.createEvent();
        cuda->cuda.record(consumerDone, cuda->stream);
        cuda->cuda.synchronize(consumerDone);
        last.readback = elapsed(start) - last.enqueue - last.wait - last.fetch;
    }
    const auto& ranges = f.kind == QueryKind::rays ? ray.columns.hit_ranges
                                                   : (sweepFamily ? sw.columns.hit_ranges : ol.columns.hit_ranges);
    const auto& ids    = f.kind == QueryKind::rays ? ray.columns.object_ids
                                                   : (sweepFamily ? sw.columns.object_ids : ol.columns.object_ids);
    for (size_t i = 0; i < f.queries.size(); ++i)
    {
        require(element<uint32_t>(ranges.query_status, i) == OVSPATIALQUERY_BATCH_ROW_SUCCESS, "batch row failed");
        const bool shortRow = element<uint8_t>(ranges.truncated, i) != 0;
        if (truncated)
            *truncated = *truncated || shortRow;
        else
            require(!shortRow, "batch row truncated");
        auto& hits = result[i].hits;
        hits.clear();
        const uint64_t count = element<uint64_t>(ranges.hit_counts, i);
        if (!count)
            continue;
        const uint64_t offset = element<uint64_t>(ranges.hit_offsets, i);
        for (size_t j = 0; j < count; ++j)
        {
            Hit h;
            h.id = anyOverlap(f.kind) ? 1 : element<uint64_t>(ids, offset + j);
            if (f.kind == QueryKind::rays || sweepFamily)
            {
                const auto& geometry         = f.kind == QueryKind::rays ? ray.columns.geometry : sw.columns.geometry;
                const bool  candidateRay     = f.kind == QueryKind::rays && bounds;
                const auto& distances        = candidateRay                ? ray.columns.contact.candidate.distances
                                               : f.kind == QueryKind::rays ? ray.columns.contact.exact.distances
                                                                           : sw.columns.contact.exact.distances;
                const auto& contactPositions = candidateRay ? ray.columns.contact.candidate.bounds_positions
                                               : f.kind == QueryKind::rays ? ray.columns.contact.exact.positions
                                                                           : sw.columns.contact.exact.positions;
                const auto& normals          = candidateRay ? ray.columns.contact.candidate.bounds_normals
                                               : f.kind == QueryKind::rays ? ray.columns.contact.exact.normals
                                                                           : sw.columns.contact.exact.normals;
                if (!f.pointCloud)
                {
                    h.geometryType  = element<int32_t>(geometry.types, offset + j);
                    h.geometryValue = element<uint64_t>(geometry.values, offset + j);
                }
                // A scene may mix meshes with other geometry; only triangle hits carry a face.
                if (!f.pointCloud && !f.meshes.empty() && h.geometryType == OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH)
                {
                    const auto& features = f.kind == QueryKind::rays ? ray.columns.contact.exact.features
                                                                     : sw.columns.contact.exact.features;
                    require(element<uint32_t>(features.kinds, offset + j) == OVSPATIALQUERY_GEOMETRY_FEATURE_TRIANGLE,
                            "mesh hit missing feature");
                    h.sourceFace = element<uint64_t>(features.source_face_indices, offset + j);
                    for (size_t k = 0; k < 3; ++k)
                        require(element<uint32_t>(features.source_corner_indices, offset + j, k) == k,
                                "explicit triangle corner mapping");
                }
                h.distance = element<double>(distances, offset + j);
                for (size_t k = 0; k < 3; ++k)
                {
                    h.position[k] = element<double>(contactPositions, offset + j, k);
                    if (!f.pointCloud)
                        h.normal[k] = element<float>(normals, offset + j, k);
                }
            }
            hits.push_back(h);
        }
    }
    if (cuda)
    {
        ovspatialquery_cuda_event_t done{reinterpret_cast<uintptr_t>(consumerDone)};
        check(ovspatialquery_destroy_op_after_cuda(instance, &admitted.op_id, &done), "retire CUDA batch");
        require(done.value == 0, "CUDA retirement did not take the consumer event");
        cuda->cuda.forgetTransferredEvent(consumerDone);
        return;
    }
    check(ovspatialquery_destroy_op(instance, &admitted.op_id), "release batch results");
}
void Spatial::makePlan(const Fixture& f, bool onCuda, ovspatialquery_query_precision_t precision)
{
    defaultPlan.reset();
    if (f.boundsCandidates)
        precision = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    defaultPlan =
        std::make_unique<BatchPlan>(*this, f, f.queries.size(), DLDevice{onCuda ? kDLCUDA : kDLCPU, 0}, precision);
}
void Spatial::batch(const Fixture& f, Results& result, ovspatialquery_ordinal_t exact)
{
    defaultPlan->batch(f, result, exact);
}
void compareAdapters(Fixture& f, Direct& direct, Spatial& spatial, bool ties)
{
    Results expected(f.queries.size()), scalar(f.queries.size()), batch(f.queries.size());
    direct.query(f, expected);
    spatial.scalar(f, scalar);
    spatial.makePlan(f);
    spatial.batch(f, batch);
    compare(f, expected, scalar, "PhysX/scalar " + std::string(kindName(f.kind)), ties, !f.approximateBaselineContacts,
            nullptr, true);
    compare(f, expected, batch, "PhysX/batch " + std::string(kindName(f.kind)), ties, !f.approximateBaselineContacts,
            nullptr, true);
    compare(f, scalar, batch, "scalar/batch " + std::string(kindName(f.kind)), ties);
}
} // namespace bench
