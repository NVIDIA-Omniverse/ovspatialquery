/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../support/native/public_api_inputs.h"
#include "../../support/native/test_failure.h"
#include <ovspatialquery/ovspatialquery_explicit.h>

#ifdef OVSPATIALQUERY_TEST_CUDA_PLANS
#include "../../support/native/cuda_public_test_helpers.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <type_traits>
#include <utility>
#include <vector>

// ANY and MULTIPLE sphere and box sweeps (ADR-0067), synchronous and through
// CPU plans, and through CUDA plans in the CUDA build of this file. One row
// of spheres and a two-wall triangle mesh lie along +x; the query starts
// inside the first sphere, so that sphere is an initial overlap reported at
// min_distance, and the mesh contributes one row at its first wall although
// the sweep crosses both. A second population checks that both modes honour
// the MESH_BOTH_SIDES query flag (ADR-0065) and report the population's materials
// (ADR-0066).

namespace
{

constexpr uint64_t kSphereDefinition = 1u;
constexpr uint64_t kMeshDefinition   = 2u;
constexpr uint64_t kMeshObject       = 80u;
constexpr uint64_t kOffPathObject    = 99u;
constexpr double   kTolerance        = 1.0e-5;

// The second population: a single-sided floor at z = 0, front face up, whose
// two triangles carry face material 701 over its object material 600, and a
// ball of object material 900 above its middle. A shape moving up from
// z = -2 reaches the floor's back face at 1.75 and, at x = 0, the ball at 4.25.
constexpr uint64_t kFloorDefinition   = 3u;
constexpr uint64_t kFloorObject       = 300u;
constexpr uint64_t kFloorFaceMaterial = 701u;
constexpr uint64_t kBallObject        = 310u;
constexpr uint64_t kBallMaterial      = 900u;
constexpr double   kFloorDistance     = 1.75;
constexpr double   kBallDistance      = 4.25;

// Object -> first contact distance of a 0.25-radius sphere or 0.25 half-extent
// box moving from the origin along +x.
const std::map<uint64_t, double> kExpected = {{5u, 0.0}, {20u, 1.25}, {40u, 3.25}, {60u, 5.25}, {kMeshObject, 7.75}};

struct Scene
{
    ovspatialquery_instance_handle_t   instance   = OVSPATIALQUERY_INVALID_INSTANCE;
    ovspatialquery_population_handle_t population = OVSPATIALQUERY_INVALID_POPULATION;
};

Scene buildScene()
{
    Scene scene;
#ifdef OVSPATIALQUERY_TEST_CUDA_PLANS
    const ovspatialquery_device_budget_t budget{{kDLCUDA, 0}, UINT64_C(64) << 20};
    const ovspatialquery_instance_desc_t desc{8u, 1u, UINT64_C(64) << 20, &budget, 1u};
#else
    const ovspatialquery_instance_desc_t desc{8u, 1u, UINT64_C(64) << 20, nullptr, 0u};
#endif
    REQUIRE(ovspatialquery_create_instance(&desc, &scene.instance).status == OVSPATIALQUERY_API_SUCCESS);
    registerSphere(scene.instance, kSphereDefinition, 0.5);

    // Two double-sided walls, x = 0 and x = 1 in the mesh frame, each two triangles.
    uint64_t   meshValue     = kMeshDefinition;
    uint64_t   zero          = 0u;
    uint64_t   vertexCount   = 8u;
    uint64_t   triangleCount = 4u;
    float      vertices[24]  = {0.0f, -1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f, 1.0f, 1.0f, 0.0f, -1.0f, 1.0f,
                                1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f};
    uint32_t   triangles[12] = {0u, 1u, 2u, 0u, 2u, 3u, 4u, 5u, 6u, 4u, 6u, 7u};
    uint8_t    doubleSided   = 1u;
    InputArray values(&meshValue, 1, kDLUInt);
    InputArray offsets(&zero, 1, kDLUInt);
    InputArray vertexCounts(&vertexCount, 1, kDLUInt);
    InputArray triangleCounts(&triangleCount, 1, kDLUInt);
    InputArray vertexData(vertices, 8, 3, kDLFloat);
    InputArray triangleData(triangles, 4, 3, kDLUInt);
    InputArray sidedness(&doubleSided, 1, kDLUInt);
    const ovspatialquery_triangle_mesh_definition_table_t meshes{
        &values.value,     &offsets.value,      &vertexCounts.value, &offsets.value, &triangleCounts.value,
        &vertexData.value, &triangleData.value, &sidedness.value,    nullptr,        nullptr,
        nullptr,
    };
    waitAndDestroy(scene.instance, ovspatialquery_enqueue_triangle_mesh_registration(scene.instance, &meshes));

    scene.population = createExplicitPopulation(scene.instance, 0u);
    InstanceRows rows;
    rows.add(60u, OVSPATIALQUERY_GEOMETRY_SPHERE, kSphereDefinition, 6.0, 0.0, 0.0);
    rows.add(5u, OVSPATIALQUERY_GEOMETRY_SPHERE, kSphereDefinition, 0.0, 0.0, 0.0);
    rows.add(kMeshObject, OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, kMeshDefinition, 8.0, 0.0, 0.0);
    rows.add(40u, OVSPATIALQUERY_GEOMETRY_SPHERE, kSphereDefinition, 4.0, 0.0, 0.0);
    rows.add(kOffPathObject, OVSPATIALQUERY_GEOMETRY_SPHERE, kSphereDefinition, 0.0, 5.0, 0.0);
    rows.add(20u, OVSPATIALQUERY_GEOMETRY_SPHERE, kSphereDefinition, 2.0, 0.0, 0.0);
    const ovspatialquery_explicit_population_set_t set{1, rows.table(), nullptr, 0u};
    waitAndDestroy(scene.instance,
                   ovspatialquery_enqueue_explicit_population_set(scene.instance, scene.population, &set));
    return scene;
}

// Adds the floor-and-ball population to the instance of `scene`.
Scene buildFloorScene(const Scene& scene)
{
    uint64_t   floorValue       = kFloorDefinition;
    uint64_t   zero             = 0u;
    uint64_t   vertexCount      = 4u;
    uint64_t   triangleCount    = 2u;
    float      vertices[12]     = {-4.0f, -4.0f, 0.0f, 4.0f, -4.0f, 0.0f, 4.0f, 4.0f, 0.0f, -4.0f, 4.0f, 0.0f};
    uint32_t   triangles[6]     = {0u, 1u, 2u, 0u, 2u, 3u};
    uint8_t    singleSided      = 0u;
    uint64_t   faceMaterials[2] = {kFloorFaceMaterial, kFloorFaceMaterial};
    InputArray values(&floorValue, 1, kDLUInt);
    InputArray offsets(&zero, 1, kDLUInt);
    InputArray vertexCounts(&vertexCount, 1, kDLUInt);
    InputArray triangleCounts(&triangleCount, 1, kDLUInt);
    InputArray vertexData(vertices, 4, 3, kDLFloat);
    InputArray triangleData(triangles, 2, 3, kDLUInt);
    InputArray sidedness(&singleSided, 1, kDLUInt);
    InputArray materialData(faceMaterials, 2, kDLUInt);
    const ovspatialquery_triangle_mesh_definition_table_t meshes{
        &values.value,     &offsets.value,      &vertexCounts.value, &offsets.value,      &triangleCounts.value,
        &vertexData.value, &triangleData.value, &sidedness.value,    &materialData.value, nullptr,
        nullptr,
    };
    waitAndDestroy(scene.instance, ovspatialquery_enqueue_triangle_mesh_registration(scene.instance, &meshes));

    Scene floor;
    floor.instance   = scene.instance;
    floor.population = createExplicitPopulation(scene.instance, 0u);
    InstanceRows rows;
    rows.add(kFloorObject, OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, kFloorDefinition, 0.0, 0.0, 0.0);
    rows.add(kBallObject, OVSPATIALQUERY_GEOMETRY_SPHERE, kSphereDefinition, 0.0, 0.0, 3.0);
    rows.materials = {600u, kBallMaterial};
    const ovspatialquery_explicit_population_set_t set{1, rows.table(), nullptr, 0u};
    waitAndDestroy(scene.instance,
                   ovspatialquery_enqueue_explicit_population_set(scene.instance, floor.population, &set));
    return floor;
}

// The floor reports its face material, the ball its object material, each at
// its first contact.
void requireFloorOrBall(uint64_t object, double distance, uint64_t material)
{
    REQUIRE(object == kFloorObject || object == kBallObject);
    REQUIRE(material == (object == kFloorObject ? kFloorFaceMaterial : kBallMaterial));
    REQUIRE(std::abs(distance - (object == kFloorObject ? kFloorDistance : kBallDistance)) <= kTolerance);
}

void requireExpectedHit(uint64_t object, double distance)
{
    const auto expected = kExpected.find(object);
    if (expected == kExpected.end() || std::abs(distance - expected->second) > kTolerance)
    {
        std::fprintf(stderr, "unexpected hit: object %llu at %.9f\n", static_cast<unsigned long long>(object),
                     distance);
    }
    REQUIRE(expected != kExpected.end());
    REQUIRE(std::abs(distance - expected->second) <= kTolerance);
}

// The rows must be exactly the `count` nearest expected objects, in any order.
void requireNearest(const std::vector<uint64_t>& objects, const std::vector<double>& distances, size_t count)
{
    REQUIRE(objects.size() == count && distances.size() == count);
    std::map<uint64_t, double> seen;
    for (size_t index = 0u; index < count; ++index)
    {
        requireExpectedHit(objects[index], distances[index]);
        REQUIRE(seen.emplace(objects[index], distances[index]).second);
    }
    auto                                     expected = kExpected.begin();
    std::vector<std::pair<double, uint64_t>> byDistance;
    for (; expected != kExpected.end(); ++expected)
    {
        byDistance.emplace_back(expected->second, expected->first);
    }
    std::sort(byDistance.begin(), byDistance.end());
    for (size_t index = 0u; index < count; ++index)
    {
        REQUIRE(seen.count(byDistance[index].second) == 1u);
    }
}

ovspatialquery_query_common_t exactCommon(const Scene& scene)
{
    ovspatialquery_query_common_t common{};
    common.target    = populationTarget(scene.population);
    common.precision = OVSPATIALQUERY_QUERY_EXACT;
    return common;
}

ovspatialquery_sphere_sweep_desc_t sphereQuery(const Scene& scene, bool hit)
{
    ovspatialquery_sphere_sweep_desc_t desc{};
    desc.common                  = exactCommon(scene);
    desc.sweep.center[1]         = hit ? 0.0 : -3.0;
    desc.sweep.radius            = 0.25;
    desc.sweep.unit_direction[0] = hit ? 1.0f : 0.0f;
    desc.sweep.unit_direction[1] = hit ? 0.0f : -1.0f;
    desc.sweep.max_distance      = 20.0;
    return desc;
}

ovspatialquery_box_sweep_desc_t boxQuery(const Scene& scene, bool hit)
{
    ovspatialquery_box_sweep_desc_t desc{};
    desc.common                  = exactCommon(scene);
    desc.sweep.center[1]         = hit ? 0.0 : -3.0;
    desc.sweep.half_extents[0]   = 0.25;
    desc.sweep.half_extents[1]   = 0.25;
    desc.sweep.half_extents[2]   = 0.25;
    desc.sweep.rotation_xyzw[3]  = 1.0f;
    desc.sweep.unit_direction[0] = hit ? 1.0f : 0.0f;
    desc.sweep.unit_direction[1] = hit ? 0.0f : -1.0f;
    desc.sweep.max_distance      = 20.0;
    return desc;
}

// The synchronous ANY and MULTIPLE entry points of one sweep family.
template <typename Desc, typename AnyFunction, typename MultipleFunction>
void checkSync(const Scene& scene, const Desc& hit, const Desc& miss, AnyFunction any, MultipleFunction multiple)
{
    ovspatialquery_sphere_sweep_closest_result_t single{};
    REQUIRE(any(scene.instance, &hit, &single).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(single.source_ordinal == 1 && single.has_hit == 1u);
    REQUIRE(single.hit.truth == OVSPATIALQUERY_HIT_EXACT_SURFACE);
    requireExpectedHit(single.hit.data.exact.identity.object_id.value, single.hit.data.exact.distance);
    REQUIRE(any(scene.instance, &miss, &single).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(single.has_hit == 0u);

    ovspatialquery_sphere_sweep_hit_t        rows[8]{};
    ovspatialquery_sphere_sweep_hit_buffer_t buffer{rows, 8u};
    ovspatialquery_multiple_result_t         result{};
    REQUIRE(multiple(scene.instance, &hit, &buffer, &result).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.source_ordinal == 1 && result.stored_hit_count == kExpected.size() && result.truncated == 0u);
    std::vector<uint64_t> objects;
    std::vector<double>   distances;
    for (uint64_t index = 0u; index < result.stored_hit_count; ++index)
    {
        const ovspatialquery_sphere_sweep_exact_hit_t& row = rows[index].data.exact;
        REQUIRE(rows[index].truth == OVSPATIALQUERY_HIT_EXACT_SURFACE);
        objects.push_back(row.identity.object_id.value);
        distances.push_back(row.distance);
        // The mesh's one row is its first wall, with that triangle's feature.
        REQUIRE((row.feature.kind == OVSPATIALQUERY_GEOMETRY_FEATURE_TRIANGLE) == (objects.back() == kMeshObject));
        if (objects.back() == kMeshObject)
        {
            REQUIRE(row.feature.data.triangle.source_face_index < 2u);
        }
    }
    requireNearest(objects, distances, kExpected.size());

    // A short buffer keeps the nearest rows and reports truncation.
    buffer.capacity = 2u;
    REQUIRE(multiple(scene.instance, &hit, &buffer, &result).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.stored_hit_count == 2u && result.truncated == 1u);
    requireNearest({rows[0].data.exact.identity.object_id.value, rows[1].data.exact.identity.object_id.value},
                   {rows[0].data.exact.distance, rows[1].data.exact.distance}, 2u);
    ovspatialquery_sphere_sweep_hit_buffer_t empty{nullptr, 0u};
    REQUIRE(multiple(scene.instance, &hit, &empty, &result).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.stored_hit_count == 0u && result.truncated == 1u);
    REQUIRE(multiple(scene.instance, &miss, &buffer, &result).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.stored_hit_count == 0u && result.truncated == 0u);
    ovspatialquery_sphere_sweep_hit_buffer_t missing{nullptr, 1u};
    REQUIRE(multiple(scene.instance, &hit, &missing, &result).status == OVSPATIALQUERY_API_INVALID_ARGUMENT);
}

// An upward sweep from (x, 0, -2) through the floor population.
template <typename Desc>
Desc floorQuery(const Scene& floor, double x, uint32_t queryFlags)
{
    Desc desc{};
    desc.common                  = exactCommon(floor);
    desc.common.query_flags      = queryFlags;
    desc.sweep.center[0]         = x;
    desc.sweep.center[2]         = -2.0;
    desc.sweep.unit_direction[2] = 1.0f;
    desc.sweep.max_distance      = 10.0;
    if constexpr (std::is_same_v<Desc, ovspatialquery_sphere_sweep_desc_t>)
    {
        desc.sweep.radius = 0.25;
    }
    else
    {
        desc.sweep.half_extents[0]  = 0.25;
        desc.sweep.half_extents[1]  = 0.25;
        desc.sweep.half_extents[2]  = 0.25;
        desc.sweep.rotation_xyzw[3] = 1.0f;
    }
    return desc;
}

// Synchronous ANY and MULTIPLE against the single-sided floor: the sweep moves
// along the floor's front normal, so ADR-0051 ignores it unless the query sets
// MESH_BOTH_SIDES; every row carries the population's material.
template <typename Desc, typename AnyFunction, typename MultipleFunction>
void checkFloorSync(const Scene& floor, AnyFunction any, MultipleFunction multiple)
{
    ovspatialquery_sphere_sweep_closest_result_t single{};
    ovspatialquery_sphere_sweep_hit_t            rows[4]{};
    ovspatialquery_sphere_sweep_hit_buffer_t     buffer{rows, 4u};
    ovspatialquery_multiple_result_t             result{};
    const auto exact = [&](uint64_t index) -> const ovspatialquery_sphere_sweep_exact_hit_t&
    {
        REQUIRE(rows[index].truth == OVSPATIALQUERY_HIT_EXACT_SURFACE);
        return rows[index].data.exact;
    };

    // Through the floor's middle: the ball alone without MESH_BOTH_SIDES.
    Desc desc = floorQuery<Desc>(floor, 0.0, 0u);
    REQUIRE(any(floor.instance, &desc, &single).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(single.has_hit == 1u && single.hit.truth == OVSPATIALQUERY_HIT_EXACT_SURFACE);
    REQUIRE(single.hit.data.exact.identity.object_id.value == kBallObject);
    requireFloorOrBall(kBallObject, single.hit.data.exact.distance, single.hit.data.exact.identity.material_id);
    REQUIRE(multiple(floor.instance, &desc, &buffer, &result).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.stored_hit_count == 1u && result.truncated == 0u);
    REQUIRE(exact(0u).identity.object_id.value == kBallObject);
    requireFloorOrBall(kBallObject, exact(0u).distance, exact(0u).identity.material_id);

    // With MESH_BOTH_SIDES the floor's back face is reached first.
    desc = floorQuery<Desc>(floor, 0.0, OVSPATIALQUERY_QUERY_MESH_BOTH_SIDES);
    REQUIRE(any(floor.instance, &desc, &single).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(single.has_hit == 1u && single.hit.truth == OVSPATIALQUERY_HIT_EXACT_SURFACE);
    requireFloorOrBall(single.hit.data.exact.identity.object_id.value, single.hit.data.exact.distance,
                       single.hit.data.exact.identity.material_id);
    REQUIRE(multiple(floor.instance, &desc, &buffer, &result).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.stored_hit_count == 2u && result.truncated == 0u);
    REQUIRE(exact(0u).identity.object_id.value != exact(1u).identity.object_id.value);
    for (uint64_t index = 0u; index < 2u; ++index)
    {
        requireFloorOrBall(exact(index).identity.object_id.value, exact(index).distance,
                           exact(index).identity.material_id);
        REQUIRE((exact(index).feature.kind == OVSPATIALQUERY_GEOMETRY_FEATURE_TRIANGLE) ==
                (exact(index).identity.object_id.value == kFloorObject));
    }

    // Beside the ball only the floor lies on the path.
    desc = floorQuery<Desc>(floor, 2.5, 0u);
    REQUIRE(any(floor.instance, &desc, &single).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(single.has_hit == 0u);
    REQUIRE(multiple(floor.instance, &desc, &buffer, &result).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.stored_hit_count == 0u && result.truncated == 0u);
    desc = floorQuery<Desc>(floor, 2.5, 1u);
    REQUIRE(any(floor.instance, &desc, &single).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(single.has_hit == 1u && single.hit.data.exact.identity.object_id.value == kFloorObject);
    requireFloorOrBall(kFloorObject, single.hit.data.exact.distance, single.hit.data.exact.identity.material_id);
    REQUIRE(multiple(floor.instance, &desc, &buffer, &result).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.stored_hit_count == 1u && exact(0u).identity.object_id.value == kFloorObject);
    requireFloorOrBall(kFloorObject, exact(0u).distance, exact(0u).identity.material_id);
}

// Reads result columns on the host, from the device in the CUDA build.
struct ColumnReader
{
#ifdef OVSPATIALQUERY_TEST_CUDA_PLANS
    CudaDriver*        cuda     = nullptr;
    CudaDriver::Stream consumer = nullptr;
#endif
    bool device = false;

    template <typename T>
    std::vector<T> read(const DLTensor& tensor, size_t count) const
    {
        std::vector<T> values(count);
        if (count == 0u)
        {
            return values;
        }
#ifdef OVSPATIALQUERY_TEST_CUDA_PLANS
        if (device)
        {
            cuda->downloadAsync(values.data(), deviceAddress(tensor), count * sizeof(T), consumer);
            const CudaDriver::Event done = cuda->createEvent();
            cuda->record(done, consumer);
            cuda->synchronize(done);
            cuda->destroyEvent(done);
            return values;
        }
#endif
        std::memcpy(values.data(), static_cast<const char*>(tensor.data) + tensor.byte_offset, count * sizeof(T));
        return values;
    }
};

struct Launch
{
    std::vector<uint32_t> status;
    std::vector<uint64_t> offsets;
    std::vector<uint64_t> counts;
    std::vector<uint8_t>  truncated;
    std::vector<uint64_t> objects;
    std::vector<double>   distances;
    std::vector<uint64_t> materials;
};

// The two rows of a launch; by default the hit query and a miss.
struct Rows
{
    double   centers[6]    = {0.0, 0.0, 0.0, 0.0, -3.0, 0.0};
    float    directions[6] = {1.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f};
    uint32_t queryFlags    = 0u;    // the launch's query_flags
    bool     materials     = false; // whether the plan returns materials
};

// The rows' inputs, on the plan's device.
struct Inputs
{
    explicit Inputs(const Rows& rows)
    {
        std::copy(std::begin(rows.centers), std::end(rows.centers), centers);
        std::copy(std::begin(rows.directions), std::end(rows.directions), directions);
        queryFlags = rows.queryFlags;
    }

    double     centers[6]    = {};
    float      directions[6] = {};
    float      rotations[8]  = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    double     maximums[2]   = {20.0, 20.0};
    uint32_t   queryFlags    = 0u;
    InputArray centerArray;
    InputArray directionArray;
    InputArray rotationArray;
    InputArray maximumArray;
};

template <typename Batch>
void bindInputs(Inputs& inputs, Batch& batch, const ColumnReader& reader)
{
    (void)reader;
    void*    centers    = inputs.centers;
    void*    directions = inputs.directions;
    void*    rotations  = inputs.rotations;
    void*    maximums   = inputs.maximums;
    DLDevice device{kDLCPU, 0};
#ifdef OVSPATIALQUERY_TEST_CUDA_PLANS
    if (reader.device)
    {
        device          = {kDLCUDA, 0};
        const auto copy = [&](const void* source, size_t bytes)
        {
            const CudaDriver::DevicePointer pointer = reader.cuda->allocate(bytes);
            reader.cuda->uploadAsync(pointer, source, bytes, reader.consumer);
            return reinterpret_cast<void*>(static_cast<uintptr_t>(pointer));
        };
        centers                    = copy(inputs.centers, sizeof(inputs.centers));
        directions                 = copy(inputs.directions, sizeof(inputs.directions));
        rotations                  = copy(inputs.rotations, sizeof(inputs.rotations));
        maximums                   = copy(inputs.maximums, sizeof(inputs.maximums));
        batch.cuda_sync.has_stream = 1u;
        batch.cuda_sync.stream     = reinterpret_cast<uintptr_t>(reader.consumer);
    }
#endif
    inputs.centerArray.set(static_cast<double*>(centers), 2, 3, kDLFloat, device);
    inputs.directionArray.set(static_cast<float*>(directions), 2, 3, kDLFloat, device);
    inputs.rotationArray.set(static_cast<float*>(rotations), 2, 4, kDLFloat, device);
    inputs.maximumArray.set(static_cast<double*>(maximums), 2, kDLFloat, device);
    batch.ordinal.kind    = OVSPATIALQUERY_ORDINAL_HEAD;
    batch.query_count     = 2u;
    batch.centers         = &inputs.centerArray.value;
    batch.unit_directions = &inputs.directionArray.value;
    batch.max_distances   = &inputs.maximumArray.value;
    batch.query_flags     = inputs.queryFlags;
}

// Runs one two-row launch of `plan` and reads its ranges, IDs, distances and,
// when requested, materials.
template <bool Box>
Launch runPlan(const Scene& scene, ovspatialquery_batch_plan_handle_t plan, const ColumnReader& reader,
               const Rows& rows = Rows{})
{
    Inputs                                     inputs(rows);
    ovspatialquery_sphere_sweep_batch_result_t result{};
    ovspatialquery_enqueue_result_t            admission{};
    if constexpr (Box)
    {
        ovspatialquery_box_sweep_batch_t batch{};
        bindInputs(inputs, batch, reader);
        batch.rotations_xyzw = &inputs.rotationArray.value;
        ovspatialquery_box_sweep_batch_reservation_t reservation{};
        admission = ovspatialquery_enqueue_box_sweep_plan(scene.instance, plan, &batch, &reservation);
    }
    else
    {
        ovspatialquery_sphere_sweep_batch_t batch{};
        bindInputs(inputs, batch, reader);
        ovspatialquery_sphere_sweep_batch_reservation_t reservation{};
        admission = ovspatialquery_enqueue_sphere_sweep_plan(scene.instance, plan, &batch, &reservation);
    }
    REQUIRE(admission.status == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_op_wait_result_t waited{};
    REQUIRE(ovspatialquery_wait_op(scene.instance, admission.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(waited.state == OVSPATIALQUERY_OP_COMPLETED);
    REQUIRE((Box ? ovspatialquery_fetch_box_sweep_batch(scene.instance, admission.op_id, &result)
                 : ovspatialquery_fetch_sphere_sweep_batch(scene.instance, admission.op_id, &result))
                .status == OVSPATIALQUERY_API_SUCCESS);
#ifdef OVSPATIALQUERY_TEST_CUDA_PLANS
    if (reader.device)
    {
        reader.cuda->wait(reader.consumer, reinterpret_cast<CudaDriver::Event>(result.common.result_ready.value));
    }
#endif
    const ovspatialquery_batch_hit_ranges_t& ranges = result.columns.hit_ranges;
    Launch                                   launch;
    launch.status         = reader.read<uint32_t>(ranges.query_status, 2u);
    launch.offsets        = reader.read<uint64_t>(ranges.hit_offsets, 2u);
    launch.counts         = reader.read<uint64_t>(ranges.hit_counts, 2u);
    launch.truncated      = reader.read<uint8_t>(ranges.truncated, 2u);
    const uint64_t stored = reader.read<uint64_t>(ranges.stored_hit_count, 1u)[0];
    REQUIRE(static_cast<int64_t>(stored) == result.columns.object_ids.shape[0]);
    launch.objects   = reader.read<uint64_t>(result.columns.object_ids, static_cast<size_t>(stored));
    launch.distances = reader.read<double>(result.columns.contact.exact.distances, static_cast<size_t>(stored));
    if (rows.materials)
    {
        REQUIRE(result.columns.material_ids.shape != nullptr &&
                static_cast<int64_t>(stored) == result.columns.material_ids.shape[0]);
        launch.materials = reader.read<uint64_t>(result.columns.material_ids, static_cast<size_t>(stored));
    }
#ifdef OVSPATIALQUERY_TEST_CUDA_PLANS
    if (reader.device)
    {
        const CudaDriver::Event consumerDone = reader.cuda->createEvent();
        reader.cuda->record(consumerDone, reader.consumer);
        reader.cuda->synchronize(consumerDone);
        ovspatialquery_cuda_event_t done{reinterpret_cast<uintptr_t>(consumerDone)};
        REQUIRE(ovspatialquery_destroy_op_after_cuda(scene.instance, &admission.op_id, &done).status ==
                OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(reader.cuda->forgetTransferredEvent(consumerDone));
        return launch;
    }
#endif
    REQUIRE(ovspatialquery_destroy_op(scene.instance, &admission.op_id).status == OVSPATIALQUERY_API_SUCCESS);
    return launch;
}

template <typename PlanDesc>
PlanDesc planDesc(const Scene& scene, DLDevice device, ovspatialquery_hit_mode_t mode, uint64_t perQuery,
                  uint64_t capacity)
{
    PlanDesc desc{};
    desc.target                 = populationRef(scene.population);
    desc.precision              = OVSPATIALQUERY_QUERY_EXACT;
    desc.device                 = device;
    desc.maximum_query_count    = 2u;
    desc.result_slot_count      = 1u;
    desc.result_fields          = OVSPATIALQUERY_BATCH_RESULT_DISTANCE;
    desc.hit_capacity_per_slot  = capacity;
    desc.hit_mode               = mode;
    desc.maximum_hits_per_query = perQuery;
    if constexpr (std::is_same_v<PlanDesc, ovspatialquery_sphere_sweep_plan_desc_t>)
    {
        desc.radius = 0.25;
    }
    else
    {
        desc.half_extents[0] = 0.25;
        desc.half_extents[1] = 0.25;
        desc.half_extents[2] = 0.25;
    }
    return desc;
}

template <typename PlanDesc, typename Create>
ovspatialquery_api_status_t createPlan(const Scene& scene, const PlanDesc& desc, Create create,
                                       ovspatialquery_batch_plan_handle_t& plan)
{
    plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    return create(scene.instance, &desc, &plan).status;
}

void destroyPlan(const Scene& scene, ovspatialquery_batch_plan_handle_t plan)
{
#ifdef OVSPATIALQUERY_TEST_CUDA_PLANS
    REQUIRE(destroyPlanWhenRetired(scene.instance, plan));
#else
    REQUIRE(ovspatialquery_destroy_batch_plan(scene.instance, plan).status == OVSPATIALQUERY_API_SUCCESS);
#endif
}

// The ANY and MULTIPLE plans of one sweep family on one device.
template <bool Box, typename PlanDesc, typename Create>
void checkPlans(const Scene& scene, const ColumnReader& reader, Create create)
{
    const DLDevice                     device = reader.device ? DLDevice{kDLCUDA, 0} : DLDevice{kDLCPU, 0};
    ovspatialquery_batch_plan_handle_t plan   = OVSPATIALQUERY_INVALID_BATCH_PLAN;

    // Admission: MULTIPLE needs a positive per-query limit; CLOSEST and ANY one
    // hit per row, a limit of zero or one and a capacity of one per row.
    REQUIRE(createPlan(scene, planDesc<PlanDesc>(scene, device, OVSPATIALQUERY_HIT_MODE_MULTIPLE, 0u, 8u), create,
                       plan) == OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(createPlan(scene, planDesc<PlanDesc>(scene, device, OVSPATIALQUERY_HIT_MODE_ANY, 2u, 2u), create, plan) ==
            OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(createPlan(scene, planDesc<PlanDesc>(scene, device, OVSPATIALQUERY_HIT_MODE_ANY, 1u, 3u), create, plan) ==
            OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(createPlan(scene, planDesc<PlanDesc>(scene, device, static_cast<ovspatialquery_hit_mode_t>(3), 1u, 2u),
                       create, plan) == OVSPATIALQUERY_API_INVALID_ARGUMENT);

    // ANY: one hit of an object the sweep reaches; the miss row stores none.
    REQUIRE(createPlan(scene, planDesc<PlanDesc>(scene, device, OVSPATIALQUERY_HIT_MODE_ANY, 0u, 2u), create, plan) ==
            OVSPATIALQUERY_API_SUCCESS);
    Launch any = runPlan<Box>(scene, plan, reader);
    REQUIRE(any.status[0] == OVSPATIALQUERY_BATCH_ROW_SUCCESS && any.status[1] == OVSPATIALQUERY_BATCH_ROW_SUCCESS);
    REQUIRE(any.counts[0] == 1u && any.counts[1] == 0u && any.truncated[0] == 0u && any.objects.size() == 1u);
    requireExpectedHit(any.objects[0], any.distances[0]);
    destroyPlan(scene, plan);

    // MULTIPLE: the per-query limit keeps the three nearest objects.
    REQUIRE(createPlan(scene, planDesc<PlanDesc>(scene, device, OVSPATIALQUERY_HIT_MODE_MULTIPLE, 3u, 8u), create,
                       plan) == OVSPATIALQUERY_API_SUCCESS);
    Launch limited = runPlan<Box>(scene, plan, reader);
    REQUIRE(limited.status[0] == OVSPATIALQUERY_BATCH_ROW_SUCCESS);
    REQUIRE(limited.counts[0] == 3u && limited.truncated[0] == 1u);
    REQUIRE(limited.counts[1] == 0u && limited.truncated[1] == 0u);
    const auto rowSlice = [](const Launch& launch, std::vector<uint64_t>& objects, std::vector<double>& distances)
    {
        const size_t begin = static_cast<size_t>(launch.offsets[0]);
        const size_t end   = begin + static_cast<size_t>(launch.counts[0]);
        objects.assign(launch.objects.begin() + begin, launch.objects.begin() + end);
        distances.assign(launch.distances.begin() + begin, launch.distances.begin() + end);
    };
    std::vector<uint64_t> objects;
    std::vector<double>   distances;
    rowSlice(limited, objects, distances);
    requireNearest(objects, distances, 3u);
    destroyPlan(scene, plan);

    // Every object reached, one row each, when the limit allows; the slot
    // capacity truncates the same way.
    REQUIRE(createPlan(scene, planDesc<PlanDesc>(scene, device, OVSPATIALQUERY_HIT_MODE_MULTIPLE, 8u, 8u), create,
                       plan) == OVSPATIALQUERY_API_SUCCESS);
    Launch all = runPlan<Box>(scene, plan, reader);
    REQUIRE(all.counts[0] == kExpected.size() && all.truncated[0] == 0u);
    rowSlice(all, objects, distances);
    requireNearest(objects, distances, kExpected.size());
    destroyPlan(scene, plan);

    REQUIRE(createPlan(scene, planDesc<PlanDesc>(scene, device, OVSPATIALQUERY_HIT_MODE_MULTIPLE, 8u, 2u), create,
                       plan) == OVSPATIALQUERY_API_SUCCESS);
    Launch slot = runPlan<Box>(scene, plan, reader);
    REQUIRE(slot.counts[0] == 2u && slot.truncated[0] == 1u);
    rowSlice(slot, objects, distances);
    requireNearest(objects, distances, 2u);
    destroyPlan(scene, plan);
}

// The ANY and MULTIPLE plans of one sweep family against the floor population:
// row 0 rises through the floor's middle and the ball, row 1 beside the ball,
// and MESH_BOTH_SIDES in the launch's query_flags turns the floor on for both.
template <bool Box, typename PlanDesc, typename Create>
void checkFloorPlans(const Scene& floor, const ColumnReader& reader, Create create)
{
    const DLDevice device = reader.device ? DLDevice{kDLCUDA, 0} : DLDevice{kDLCPU, 0};
    Rows           rows;
    const double   centers[6]    = {0.0, 0.0, -2.0, 2.5, 0.0, -2.0};
    const float    directions[6] = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};
    std::copy(std::begin(centers), std::end(centers), rows.centers);
    std::copy(std::begin(directions), std::end(directions), rows.directions);
    rows.materials = true;
    for (const ovspatialquery_hit_mode_t mode : {OVSPATIALQUERY_HIT_MODE_ANY, OVSPATIALQUERY_HIT_MODE_MULTIPLE})
    {
        const bool multiple = mode == OVSPATIALQUERY_HIT_MODE_MULTIPLE;
        PlanDesc   desc     = planDesc<PlanDesc>(floor, device, mode, multiple ? 4u : 0u, multiple ? 8u : 2u);
        desc.result_fields |= OVSPATIALQUERY_BATCH_RESULT_MATERIAL;
        // A CUDA slot is free again only once its result is retired, so each
        // of the two launches below gets its own.
        desc.result_slot_count                  = 2u;
        ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
        REQUIRE(createPlan(floor, desc, create, plan) == OVSPATIALQUERY_API_SUCCESS);
        for (const uint32_t flags : {0u, static_cast<uint32_t>(OVSPATIALQUERY_QUERY_MESH_BOTH_SIDES)})
        {
            rows.queryFlags          = flags;
            const bool   firstSides  = flags != 0u;
            const bool   secondSides = flags != 0u;
            const Launch launch      = runPlan<Box>(floor, plan, reader, rows);
            const size_t firstBegin  = static_cast<size_t>(launch.offsets[0]);
            const size_t secondBegin = static_cast<size_t>(launch.offsets[1]);
            REQUIRE(launch.status[0] == OVSPATIALQUERY_BATCH_ROW_SUCCESS);
            REQUIRE(launch.status[1] == OVSPATIALQUERY_BATCH_ROW_SUCCESS);
            REQUIRE(launch.truncated[0] == 0u && launch.truncated[1] == 0u);
            // Row 0: the ball alone, or with MESH_BOTH_SIDES the floor too (MULTIPLE)
            // or either of them (ANY).
            REQUIRE(launch.counts[0] == (multiple && firstSides ? 2u : 1u));
            for (size_t index = firstBegin; index < firstBegin + launch.counts[0]; ++index)
            {
                REQUIRE(firstSides || launch.objects[index] == kBallObject);
                requireFloorOrBall(launch.objects[index], launch.distances[index], launch.materials[index]);
            }
            if (launch.counts[0] == 2u)
            {
                REQUIRE(launch.objects[firstBegin] != launch.objects[firstBegin + 1u]);
            }
            // Row 1: the floor only with MESH_BOTH_SIDES.
            REQUIRE(launch.counts[1] == (secondSides ? 1u : 0u));
            if (secondSides)
            {
                REQUIRE(launch.objects[secondBegin] == kFloorObject);
                requireFloorOrBall(kFloorObject, launch.distances[secondBegin], launch.materials[secondBegin]);
            }
        }
        destroyPlan(floor, plan);
    }
}

} // namespace

int main()
{
    const Scene scene = buildScene();

    checkSync(scene, sphereQuery(scene, true), sphereQuery(scene, false), ovspatialquery_sweep_sphere_any_sync,
              ovspatialquery_sweep_sphere_multiple_sync);
    checkSync(scene, boxQuery(scene, true), boxQuery(scene, false), ovspatialquery_sweep_box_any_sync,
              ovspatialquery_sweep_box_multiple_sync);
    const Scene floor = buildFloorScene(scene);
    checkFloorSync<ovspatialquery_sphere_sweep_desc_t>(floor, ovspatialquery_sweep_sphere_any_sync,
                                                       ovspatialquery_sweep_sphere_multiple_sync);
    checkFloorSync<ovspatialquery_box_sweep_desc_t>(floor, ovspatialquery_sweep_box_any_sync,
                                                    ovspatialquery_sweep_box_multiple_sync);

    ColumnReader cpu;
    checkPlans<false, ovspatialquery_sphere_sweep_plan_desc_t>(scene, cpu, ovspatialquery_create_sphere_sweep_plan);
    checkPlans<true, ovspatialquery_box_sweep_plan_desc_t>(scene, cpu, ovspatialquery_create_box_sweep_plan);
    checkFloorPlans<false, ovspatialquery_sphere_sweep_plan_desc_t>(floor, cpu,
                                                                    ovspatialquery_create_sphere_sweep_plan);
    checkFloorPlans<true, ovspatialquery_box_sweep_plan_desc_t>(floor, cpu, ovspatialquery_create_box_sweep_plan);

#ifdef OVSPATIALQUERY_TEST_CUDA_PLANS
    {
        CudaDriver   cuda(CudaDriver::Ownership::tracked);
        ColumnReader device;
        device.cuda     = &cuda;
        device.consumer = cuda.createStream();
        device.device   = true;
        checkPlans<false, ovspatialquery_sphere_sweep_plan_desc_t>(scene, device,
                                                                   ovspatialquery_create_sphere_sweep_plan);
        checkPlans<true, ovspatialquery_box_sweep_plan_desc_t>(scene, device, ovspatialquery_create_box_sweep_plan);
        checkFloorPlans<false, ovspatialquery_sphere_sweep_plan_desc_t>(floor, device,
                                                                        ovspatialquery_create_sphere_sweep_plan);
        checkFloorPlans<true, ovspatialquery_box_sweep_plan_desc_t>(floor, device,
                                                                    ovspatialquery_create_box_sweep_plan);
    }
#endif

    REQUIRE(ovspatialquery_destroy_population(scene.instance, scene.population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_population(scene.instance, floor.population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(
        ovspatialquery_destroy_geometry_definition(scene.instance, {OVSPATIALQUERY_GEOMETRY_SPHERE, kSphereDefinition})
            .status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_geometry_definition(scene.instance,
                                                       {OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, kMeshDefinition})
                .status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_geometry_definition(scene.instance,
                                                       {OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, kFloorDefinition})
                .status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(scene.instance).status == OVSPATIALQUERY_API_SUCCESS);
    std::puts("sweep modes test passed");
    return 0;
}
