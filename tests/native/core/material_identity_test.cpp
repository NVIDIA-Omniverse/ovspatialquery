/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Material identity on hits (ADR-0066) through the installed explicit API: an
// object's material on every family and precision, a triangle's own material
// on exact mesh contacts, the CPU batch column, and how a material belongs to
// the published version, and the caller's source faces and corners on a
// polygon mesh it triangulated, with the material of each face.

#include "public_api_inputs.h"
#include "test_failure.h"

#include <ovspatialquery/ovspatialquery.h>
#include <ovspatialquery/ovspatialquery_explicit.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace
{

constexpr uint64_t kSphereObject      = 10u;
constexpr uint64_t kMeshObject        = 20u;
constexpr uint64_t kSphereMaterial    = 1000u;
constexpr uint64_t kTriangleMaterial  = 2000u;
constexpr uint64_t kMeshMaterial      = 3000u;
constexpr uint64_t kReplacedMaterial  = 1111u;
constexpr uint64_t kSphereDefinition  = 1u;
constexpr uint64_t kMeshDefinition    = 2u;
constexpr uint64_t kBadMeshDefinition = 3u;

// Two triangles in the plane x = 5, front faces towards +X: triangle 0 below
// y = 0 carries its own material, triangle 1 above it carries zero and so
// reports the instance's.
void registerMesh(ovspatialquery_instance_handle_t instance, uint64_t value, const std::vector<uint64_t>& faceMaterials,
                  ovspatialquery_api_status_t expected, uint8_t doubleSided = 1u)
{
    uint64_t   zero          = 0u;
    uint64_t   vertexCount   = 6u;
    uint64_t   triangleCount = 2u;
    float      vertices[18]  = {5.0f, -3.0f, -1.0f, 5.0f, -1.0f, -1.0f, 5.0f, -2.0f, 1.0f,
                                5.0f, 1.0f,  -1.0f, 5.0f, 3.0f,  -1.0f, 5.0f, 2.0f,  1.0f};
    uint32_t   triangles[6]  = {0u, 1u, 2u, 3u, 4u, 5u};
    InputArray values(&value, 1, kDLUInt);
    InputArray offsets(&zero, 1, kDLUInt);
    InputArray vertexCounts(&vertexCount, 1, kDLUInt);
    InputArray triangleCounts(&triangleCount, 1, kDLUInt);
    InputArray vertexData(vertices, 6, 3, kDLFloat);
    InputArray triangleData(triangles, 2, 3, kDLUInt);
    InputArray sidedness(&doubleSided, 1, kDLUInt);
    InputArray materials(faceMaterials.data(), static_cast<int64_t>(faceMaterials.size()), kDLUInt);
    const ovspatialquery_triangle_mesh_definition_table_t meshes{
        &values.value,     &offsets.value,      &vertexCounts.value, &offsets.value,   &triangleCounts.value,
        &vertexData.value, &triangleData.value, &sidedness.value,    &materials.value, nullptr,
        nullptr,
    };
    const ovspatialquery_enqueue_result_t admission =
        ovspatialquery_enqueue_triangle_mesh_registration(instance, &meshes);
    if (expected == OVSPATIALQUERY_API_SUCCESS)
    {
        waitAndDestroy(instance, admission);
    }
    else
    {
        requireRejected(admission, expected);
    }
}

struct Scene
{
    explicit Scene(uint8_t doubleSidedMesh = 1u)
    {
        const ovspatialquery_instance_desc_t desc{8u, 1u, UINT64_C(64) << 20, nullptr, 0u};
        REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
        registerSphere(instance, kSphereDefinition, 1.0);
        registerMesh(instance, kMeshDefinition, {kTriangleMaterial, 0u}, OVSPATIALQUERY_API_SUCCESS, doubleSidedMesh);
        population = createExplicitPopulation(instance, 2u);
        InstanceRows rows;
        rows.add(kSphereObject, OVSPATIALQUERY_GEOMETRY_SPHERE, kSphereDefinition, 0.0, 10.0, 0.0);
        rows.add(kMeshObject, OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, kMeshDefinition, 0.0, 0.0, 0.0);
        rows.materials = {kSphereMaterial, kMeshMaterial};
        ovspatialquery_explicit_population_set_t set{};
        set.target_ordinal = 1;
        set.instances      = rows.table();
        waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));
    }

    ~Scene()
    {
        REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(
            ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_SPHERE, kSphereDefinition})
                .status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(ovspatialquery_destroy_geometry_definition(instance,
                                                           {OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, kMeshDefinition})
                    .status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
    }

    ovspatialquery_query_common_t common(ovspatialquery_query_precision_t precision,
                                         ovspatialquery_ordinal_t         ordinal = -1) const
    {
        ovspatialquery_query_common_t value{};
        value.target    = populationTarget(population);
        value.precision = precision;
        if (ordinal >= 0)
        {
            value.target.ordinal.kind           = OVSPATIALQUERY_ORDINAL_RETAINED_EXACT;
            value.target.ordinal.source_ordinal = ordinal;
        }
        return value;
    }

    ovspatialquery_instance_handle_t   instance   = OVSPATIALQUERY_INVALID_INSTANCE;
    ovspatialquery_population_handle_t population = OVSPATIALQUERY_INVALID_POPULATION;
};

// The three rays: down onto the sphere, and along +X onto each triangle.
const double kOrigins[3][3]    = {{0.0, 10.0, -5.0}, {0.0, -2.0, 0.0}, {0.0, 2.0, 0.0}};
const float  kDirections[3][3] = {{0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}};

ovspatialquery_raycast_desc_t ray(const Scene& scene, uint32_t row, ovspatialquery_query_precision_t precision,
                                  ovspatialquery_ordinal_t ordinal = -1)
{
    ovspatialquery_raycast_desc_t desc{};
    desc.common = scene.common(precision, ordinal);
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        desc.ray.origin[axis]         = kOrigins[row][axis];
        desc.ray.unit_direction[axis] = kDirections[row][axis];
    }
    desc.ray.max_distance = 20.0;
    return desc;
}

uint64_t closestRayMaterial(const Scene& scene, uint32_t row, ovspatialquery_query_precision_t precision,
                            ovspatialquery_ordinal_t ordinal = -1)
{
    const ovspatialquery_raycast_desc_t    desc = ray(scene, row, precision, ordinal);
    ovspatialquery_raycast_single_result_t result{};
    REQUIRE(ovspatialquery_raycast_closest_sync(scene.instance, &desc, &result).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.has_hit == 1u);
    return precision == OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE ? result.hit.data.bounds_candidate.identity.material_id
                                                              : result.hit.data.exact.identity.material_id;
}

// Exact contacts on a mesh report the face's material when it has one; bounds
// candidates and every other family report the object's.
void testSynchronousHitsReportTheSurfaceMaterial()
{
    Scene scene;
    REQUIRE(closestRayMaterial(scene, 0u, OVSPATIALQUERY_QUERY_EXACT) == kSphereMaterial);
    REQUIRE(closestRayMaterial(scene, 1u, OVSPATIALQUERY_QUERY_EXACT) == kTriangleMaterial);
    REQUIRE(closestRayMaterial(scene, 2u, OVSPATIALQUERY_QUERY_EXACT) == kMeshMaterial);
    REQUIRE(closestRayMaterial(scene, 1u, OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE) == kMeshMaterial);
    REQUIRE(closestRayMaterial(scene, 0u, OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE) == kSphereMaterial);

    ovspatialquery_raycast_desc_t          anyDesc = ray(scene, 1u, OVSPATIALQUERY_QUERY_EXACT);
    ovspatialquery_raycast_single_result_t any{};
    REQUIRE(ovspatialquery_raycast_any_sync(scene.instance, &anyDesc, &any).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(any.has_hit == 1u && any.hit.data.exact.identity.material_id == kTriangleMaterial);

    ovspatialquery_raycast_desc_t       multipleDesc = ray(scene, 1u, OVSPATIALQUERY_QUERY_EXACT);
    ovspatialquery_raycast_hit_t        hits[4]{};
    ovspatialquery_raycast_hit_buffer_t buffer{hits, 4u};
    ovspatialquery_multiple_result_t    multiple{};
    REQUIRE(ovspatialquery_raycast_multiple_sync(scene.instance, &multipleDesc, &buffer, &multiple).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(multiple.stored_hit_count == 1u && hits[0].data.exact.identity.material_id == kTriangleMaterial);

    ovspatialquery_sphere_sweep_desc_t sphereSweep{};
    sphereSweep.common = scene.common(OVSPATIALQUERY_QUERY_EXACT);
    sphereSweep.sweep  = {{0.0, -2.0, 0.0}, 0.1, {1.0f, 0.0f, 0.0f}, 0.0, 20.0};
    ovspatialquery_sphere_sweep_closest_result_t sphereHit{};
    REQUIRE(ovspatialquery_sweep_sphere_closest_sync(scene.instance, &sphereSweep, &sphereHit).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(sphereHit.has_hit == 1u && sphereHit.hit.data.exact.identity.material_id == kTriangleMaterial);

    ovspatialquery_box_sweep_desc_t boxSweep{};
    boxSweep.common = scene.common(OVSPATIALQUERY_QUERY_EXACT);
    boxSweep.sweep  = {{0.0, 2.0, 0.0}, {0.1, 0.1, 0.1}, {0.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}, 0.0, 20.0};
    ovspatialquery_box_sweep_closest_result_t boxHit{};
    REQUIRE(ovspatialquery_sweep_box_closest_sync(scene.instance, &boxSweep, &boxHit).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(boxHit.has_hit == 1u && boxHit.hit.data.exact.identity.material_id == kMeshMaterial);

    ovspatialquery_overlap_desc_t overlap{};
    overlap.common          = scene.common(OVSPATIALQUERY_QUERY_EXACT);
    overlap.shape.kind      = OVSPATIALQUERY_OVERLAP_AABB;
    overlap.shape.data.aabb = {{-20.0, -20.0, -20.0}, {20.0, 20.0, 20.0}};
    ovspatialquery_overlap_hit_t        overlapHits[4]{};
    ovspatialquery_overlap_hit_buffer_t overlapBuffer{overlapHits, 4u};
    REQUIRE(ovspatialquery_overlap_multiple_sync(scene.instance, &overlap, &overlapBuffer, &multiple).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(multiple.stored_hit_count == 2u);
    for (uint64_t index = 0u; index < multiple.stored_hit_count; ++index)
    {
        const ovspatialquery_hit_identity_t& identity = overlapHits[index].identity;
        REQUIRE(identity.material_id == (identity.object_id.value == kSphereObject ? kSphereMaterial : kMeshMaterial));
    }
}

// The CPU batch column carries the same values, and it is absent unless asked for.
void testCpuBatchColumnsCarryMaterials()
{
    Scene                              scene;
    ovspatialquery_raycast_plan_desc_t desc{};
    desc.target                             = populationRef(scene.population);
    desc.hit_mode                           = OVSPATIALQUERY_HIT_MODE_CLOSEST;
    desc.precision                          = OVSPATIALQUERY_QUERY_EXACT;
    desc.device                             = {kDLCPU, 0};
    desc.maximum_query_count                = 3u;
    desc.maximum_hits_per_query             = 1u;
    desc.hit_capacity_per_slot              = 3u;
    desc.result_slot_count                  = 1u;
    desc.result_fields                      = OVSPATIALQUERY_BATCH_RESULT_MATERIAL;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_raycast_plan(scene.instance, &desc, &plan).status == OVSPATIALQUERY_API_SUCCESS);

    double                         maximumDistances[3] = {20.0, 20.0, 20.0};
    InputArray                     origins(&kOrigins[0][0], 3, 3, kDLFloat);
    InputArray                     directions(&kDirections[0][0], 3, 3, kDLFloat);
    InputArray                     maximums(maximumDistances, 3, kDLFloat);
    ovspatialquery_raycast_batch_t batch{};
    batch.ordinal.kind    = OVSPATIALQUERY_ORDINAL_HEAD;
    batch.query_count     = 3u;
    batch.origins         = &origins.value;
    batch.unit_directions = &directions.value;
    batch.max_distances   = &maximums.value;
    ovspatialquery_raycast_batch_reservation_t reservation{};
    ovspatialquery_enqueue_result_t            operation =
        ovspatialquery_enqueue_raycast_plan(scene.instance, plan, &batch, &reservation);
    REQUIRE(operation.status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(reservation.columns.material_ids.shape != nullptr);
    ovspatialquery_op_wait_result_t waited{};
    REQUIRE(ovspatialquery_wait_op(scene.instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(waited.state == OVSPATIALQUERY_OP_COMPLETED);
    ovspatialquery_raycast_batch_result_t result{};
    REQUIRE(ovspatialquery_fetch_raycast_batch(scene.instance, operation.op_id, &result).status ==
            OVSPATIALQUERY_API_SUCCESS);
    const DLTensor& materials = result.columns.material_ids;
    REQUIRE(materials.ndim == 1 && materials.shape[0] == 3 && materials.dtype.code == kDLUInt &&
            materials.dtype.bits == 64u);
    const uint64_t* offsets     = static_cast<const uint64_t*>(result.columns.hit_ranges.hit_offsets.data);
    const uint64_t* values      = static_cast<const uint64_t*>(materials.data);
    const uint64_t  expected[3] = {kSphereMaterial, kTriangleMaterial, kMeshMaterial};
    for (uint32_t row = 0u; row < 3u; ++row)
    {
        REQUIRE(values[offsets[row]] == expected[row]);
    }
    REQUIRE(result.columns.reserved[0].shape == nullptr);
    REQUIRE(ovspatialquery_destroy_op(scene.instance, &operation.op_id).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_batch_plan(scene.instance, plan).status == OVSPATIALQUERY_API_SUCCESS);

    ovspatialquery_overlap_plan_desc_t overlapDesc{};
    overlapDesc.target                   = populationRef(scene.population);
    overlapDesc.hit_mode                 = OVSPATIALQUERY_HIT_MODE_MULTIPLE;
    overlapDesc.precision                = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    overlapDesc.shape.kind               = OVSPATIALQUERY_OVERLAP_SPHERE;
    overlapDesc.shape.data.sphere.radius = 0.5;
    overlapDesc.device                   = {kDLCPU, 0};
    overlapDesc.maximum_query_count      = 1u;
    overlapDesc.maximum_hits_per_query   = 2u;
    overlapDesc.hit_capacity_per_slot    = 2u;
    overlapDesc.result_slot_count        = 1u;
    overlapDesc.result_fields            = OVSPATIALQUERY_BATCH_RESULT_MATERIAL;
    REQUIRE(ovspatialquery_create_overlap_plan(scene.instance, &overlapDesc, &plan).status ==
            OVSPATIALQUERY_API_SUCCESS);
    double                         center[3] = {0.0, 10.0, 0.0};
    InputArray                     translations(center, 1, 3, kDLFloat);
    ovspatialquery_overlap_batch_t overlapBatch{};
    overlapBatch.ordinal.kind = OVSPATIALQUERY_ORDINAL_HEAD;
    overlapBatch.query_count  = 1u;
    overlapBatch.translations = &translations.value;
    ovspatialquery_overlap_batch_reservation_t overlapReservation{};
    operation = ovspatialquery_enqueue_overlap_plan(scene.instance, plan, &overlapBatch, &overlapReservation);
    REQUIRE(operation.status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_wait_op(scene.instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
            OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_overlap_batch_result_t overlapResult{};
    REQUIRE(ovspatialquery_fetch_overlap_batch(scene.instance, operation.op_id, &overlapResult).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(overlapResult.columns.material_ids.shape[0] == 1);
    REQUIRE(static_cast<const uint64_t*>(overlapResult.columns.material_ids.data)[0] == kSphereMaterial);
    REQUIRE(ovspatialquery_destroy_op(scene.instance, &operation.op_id).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_batch_plan(scene.instance, plan).status == OVSPATIALQUERY_API_SUCCESS);
}

// A material belongs to the version: a COMPLETE_ROWS update publishes a new
// one, a retained ordinal keeps the old one, and a poses update keeps it.
void testMaterialBelongsToThePublishedVersion()
{
    Scene        scene;
    InstanceRows changed;
    changed.add(kSphereObject, OVSPATIALQUERY_GEOMETRY_SPHERE, kSphereDefinition, 0.0, 10.0, 0.0);
    changed.materials                                     = {kReplacedMaterial};
    uint64_t                                    noRemoval = 0u;
    InputArray                                  removed(&noRemoval, 0, kDLUInt);
    ovspatialquery_explicit_population_update_t update{};
    update.removed_object_values = &removed.value;
    update.base_ordinal          = 1;
    update.target_ordinal        = 2;
    update.kind                  = OVSPATIALQUERY_EXPLICIT_UPDATE_COMPLETE_ROWS;
    update.changed_instances     = changed.table();
    waitAndDestroy(scene.instance,
                   ovspatialquery_enqueue_explicit_population_update(scene.instance, scene.population, &update));
    REQUIRE(closestRayMaterial(scene, 0u, OVSPATIALQUERY_QUERY_EXACT) == kReplacedMaterial);
    REQUIRE(closestRayMaterial(scene, 0u, OVSPATIALQUERY_QUERY_EXACT, 1) == kSphereMaterial);
    REQUIRE(closestRayMaterial(scene, 1u, OVSPATIALQUERY_QUERY_EXACT) == kTriangleMaterial);

    // Poses rows take no material and keep each object's.
    ovspatialquery_explicit_population_update_t poses = update;
    poses.base_ordinal                                = 2;
    poses.target_ordinal                              = 3;
    poses.kind                                        = OVSPATIALQUERY_EXPLICIT_UPDATE_POSES;
    poses.changed_instances.geometry_types            = nullptr;
    poses.changed_instances.geometry_values           = nullptr;
    requireRejected(ovspatialquery_enqueue_explicit_population_update(scene.instance, scene.population, &poses),
                    OVSPATIALQUERY_API_INVALID_ARGUMENT);
    poses.changed_instances.material_ids = nullptr;
    waitAndDestroy(scene.instance,
                   ovspatialquery_enqueue_explicit_population_update(scene.instance, scene.population, &poses));
    REQUIRE(closestRayMaterial(scene, 0u, OVSPATIALQUERY_QUERY_EXACT) == kReplacedMaterial);

    // A complete row without the column publishes zero, "no material".
    InstanceRows plain;
    plain.add(kSphereObject, OVSPATIALQUERY_GEOMETRY_SPHERE, kSphereDefinition, 0.0, 10.0, 0.0);
    update.base_ordinal      = 3;
    update.target_ordinal    = 4;
    update.changed_instances = plain.table();
    waitAndDestroy(scene.instance,
                   ovspatialquery_enqueue_explicit_population_update(scene.instance, scene.population, &update));
    REQUIRE(closestRayMaterial(scene, 0u, OVSPATIALQUERY_QUERY_EXACT) == 0u);
}

// One face material per submitted triangle; another length is refused.
void testFaceMaterialTableMatchesTheTriangles()
{
    Scene scene;
    registerMesh(scene.instance, kBadMeshDefinition, {kTriangleMaterial}, OVSPATIALQUERY_API_INVALID_ARGUMENT);
}

// Sidedness and materials are independent: on a single-sided copy of the
// mesh the +X rays and sweep reach only back faces, ignored until the query
// sets MESH_BOTH_SIDES (ADR-0065); the contacts it then admits report the same face
// and object materials as on the double-sided mesh.
void testMeshBothSidesKeepsTheSurfaceMaterial()
{
    Scene scene(0u);
    for (const uint32_t row : {1u, 2u})
    {
        ovspatialquery_raycast_desc_t          desc = ray(scene, row, OVSPATIALQUERY_QUERY_EXACT);
        ovspatialquery_raycast_single_result_t result{};
        REQUIRE(ovspatialquery_raycast_closest_sync(scene.instance, &desc, &result).status ==
                OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(result.has_hit == 0u);
        desc.common.query_flags = OVSPATIALQUERY_QUERY_MESH_BOTH_SIDES;
        REQUIRE(ovspatialquery_raycast_closest_sync(scene.instance, &desc, &result).status ==
                OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(result.has_hit == 1u);
        REQUIRE(result.hit.data.exact.identity.material_id == (row == 1u ? kTriangleMaterial : kMeshMaterial));
    }

    ovspatialquery_raycast_desc_t multipleDesc = ray(scene, 1u, OVSPATIALQUERY_QUERY_EXACT);
    multipleDesc.common.query_flags            = OVSPATIALQUERY_QUERY_MESH_BOTH_SIDES;
    ovspatialquery_raycast_hit_t        hits[4]{};
    ovspatialquery_raycast_hit_buffer_t buffer{hits, 4u};
    ovspatialquery_multiple_result_t    multiple{};
    REQUIRE(ovspatialquery_raycast_multiple_sync(scene.instance, &multipleDesc, &buffer, &multiple).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(multiple.stored_hit_count == 1u && hits[0].data.exact.identity.material_id == kTriangleMaterial);

    ovspatialquery_sphere_sweep_desc_t sphereSweep{};
    sphereSweep.common = scene.common(OVSPATIALQUERY_QUERY_EXACT);
    sphereSweep.sweep  = {{0.0, -2.0, 0.0}, 0.1, {1.0f, 0.0f, 0.0f}, 0.0, 20.0};
    ovspatialquery_sphere_sweep_closest_result_t sphereHit{};
    REQUIRE(ovspatialquery_sweep_sphere_closest_sync(scene.instance, &sphereSweep, &sphereHit).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(sphereHit.has_hit == 0u);
    sphereSweep.common.query_flags = OVSPATIALQUERY_QUERY_MESH_BOTH_SIDES;
    REQUIRE(ovspatialquery_sweep_sphere_closest_sync(scene.instance, &sphereSweep, &sphereHit).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(sphereHit.has_hit == 1u && sphereHit.hit.data.exact.identity.material_id == kTriangleMaterial);
}

// The polygon mesh (public_api_inputs.h) as object 20 with material 3000.
struct PolygonScene
{
    explicit PolygonScene(PolygonMesh& mesh, bool withSourceFaces = true)
    {
        const ovspatialquery_instance_desc_t desc{8u, 1u, UINT64_C(64) << 20, nullptr, 0u};
        REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
        mesh.value                                                  = kMeshDefinition;
        const ovspatialquery_triangle_mesh_definition_table_t table = mesh.table(withSourceFaces);
        waitAndDestroy(instance, ovspatialquery_enqueue_triangle_mesh_registration(instance, &table));
        population = createExplicitPopulation(instance, 1u);
        InstanceRows rows;
        rows.add(kMeshObject, OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, kMeshDefinition, 0.0, 0.0, 0.0);
        rows.materials = {kMeshMaterial};
        ovspatialquery_explicit_population_set_t set{};
        set.target_ordinal = 1;
        set.instances      = rows.table();
        waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));
    }

    ~PolygonScene()
    {
        REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(ovspatialquery_destroy_geometry_definition(instance,
                                                           {OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, kMeshDefinition})
                    .status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
    }

    ovspatialquery_raycast_desc_t ray(uint32_t triangle) const
    {
        ovspatialquery_raycast_desc_t desc{};
        desc.common.target         = populationTarget(population);
        desc.common.precision      = OVSPATIALQUERY_QUERY_EXACT;
        desc.ray.origin[1]         = kPolygonRayYz[triangle][0];
        desc.ray.origin[2]         = kPolygonRayYz[triangle][1];
        desc.ray.unit_direction[0] = 1.0f;
        desc.ray.max_distance      = 20.0;
        return desc;
    }

    ovspatialquery_instance_handle_t   instance   = OVSPATIALQUERY_INVALID_INSTANCE;
    ovspatialquery_population_handle_t population = OVSPATIALQUERY_INVALID_POPULATION;
};

// The face, corners and material an exact hit on submitted triangle `triangle` reports.
template <typename ExactHit>
void requireSourceFace(const PolygonMesh& mesh, uint32_t triangle, const ExactHit& hit)
{
    REQUIRE(hit.feature.kind == OVSPATIALQUERY_GEOMETRY_FEATURE_TRIANGLE);
    const ovspatialquery_triangle_feature_t& feature = hit.feature.data.triangle;
    REQUIRE(feature.source_face_index == mesh.faces[triangle]);
    REQUIRE(std::memcmp(feature.source_face_corner_indices, mesh.corners[triangle], sizeof(mesh.corners[triangle])) ==
            0);
    REQUIRE(hit.identity.material_id == (mesh.faces[triangle] == 7u ? kTriangleMaterial : kMeshMaterial));
}

// Exact hits on a caller-triangulated quad and pentagon report the polygon and
// its corners, and the polygon's material, on every synchronous hit mode, a
// sphere sweep and the CPU batch column; without the columns the same mesh
// reports submitted ordinals and corners { 0, 1, 2 }.
void testHitsReportCallerSourceFaces()
{
    PolygonMesh  mesh;
    PolygonScene scene(mesh);
    for (uint32_t triangle = 0u; triangle < PolygonMesh::kTriangleCount; ++triangle)
    {
        const ovspatialquery_raycast_desc_t    desc = scene.ray(triangle);
        ovspatialquery_raycast_single_result_t single{};
        REQUIRE(ovspatialquery_raycast_closest_sync(scene.instance, &desc, &single).status ==
                OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(single.has_hit == 1u);
        requireSourceFace(mesh, triangle, single.hit.data.exact);
        REQUIRE(ovspatialquery_raycast_any_sync(scene.instance, &desc, &single).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(single.has_hit == 1u);
        requireSourceFace(mesh, triangle, single.hit.data.exact);
        ovspatialquery_raycast_hit_t        hits[2]{};
        ovspatialquery_raycast_hit_buffer_t buffer{hits, 2u};
        ovspatialquery_multiple_result_t    multiple{};
        REQUIRE(ovspatialquery_raycast_multiple_sync(scene.instance, &desc, &buffer, &multiple).status ==
                OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(multiple.stored_hit_count == 1u);
        requireSourceFace(mesh, triangle, hits[0].data.exact);

        ovspatialquery_sphere_sweep_desc_t sweep{};
        sweep.common = desc.common;
        sweep.sweep  = {
            {0.0, kPolygonRayYz[triangle][0], kPolygonRayYz[triangle][1]}, 0.05, {1.0f, 0.0f, 0.0f}, 0.0, 20.0};
        ovspatialquery_sphere_sweep_closest_result_t swept{};
        REQUIRE(ovspatialquery_sweep_sphere_closest_sync(scene.instance, &sweep, &swept).status ==
                OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(swept.has_hit == 1u);
        requireSourceFace(mesh, triangle, swept.hit.data.exact);
    }

    ovspatialquery_raycast_plan_desc_t planDesc{};
    planDesc.target                 = populationRef(scene.population);
    planDesc.hit_mode               = OVSPATIALQUERY_HIT_MODE_CLOSEST;
    planDesc.precision              = OVSPATIALQUERY_QUERY_EXACT;
    planDesc.device                 = {kDLCPU, 0};
    planDesc.maximum_query_count    = PolygonMesh::kTriangleCount;
    planDesc.maximum_hits_per_query = 1u;
    planDesc.hit_capacity_per_slot  = PolygonMesh::kTriangleCount;
    planDesc.result_slot_count      = 1u;
    planDesc.result_fields          = OVSPATIALQUERY_BATCH_RESULT_SOURCE_FEATURE | OVSPATIALQUERY_BATCH_RESULT_MATERIAL;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_raycast_plan(scene.instance, &planDesc, &plan).status == OVSPATIALQUERY_API_SUCCESS);
    double origins[PolygonMesh::kTriangleCount][3]{};
    float  directions[PolygonMesh::kTriangleCount][3]{};
    double maximums[PolygonMesh::kTriangleCount]{};
    for (uint32_t triangle = 0u; triangle < PolygonMesh::kTriangleCount; ++triangle)
    {
        origins[triangle][1]    = kPolygonRayYz[triangle][0];
        origins[triangle][2]    = kPolygonRayYz[triangle][1];
        directions[triangle][0] = 1.0f;
        maximums[triangle]      = 20.0;
    }
    InputArray                     originColumn(&origins[0][0], PolygonMesh::kTriangleCount, 3, kDLFloat);
    InputArray                     directionColumn(&directions[0][0], PolygonMesh::kTriangleCount, 3, kDLFloat);
    InputArray                     maximumColumn(maximums, PolygonMesh::kTriangleCount, kDLFloat);
    ovspatialquery_raycast_batch_t batch{};
    batch.ordinal.kind    = OVSPATIALQUERY_ORDINAL_HEAD;
    batch.query_count     = PolygonMesh::kTriangleCount;
    batch.origins         = &originColumn.value;
    batch.unit_directions = &directionColumn.value;
    batch.max_distances   = &maximumColumn.value;
    ovspatialquery_raycast_batch_reservation_t reservation{};
    ovspatialquery_enqueue_result_t            operation =
        ovspatialquery_enqueue_raycast_plan(scene.instance, plan, &batch, &reservation);
    REQUIRE(operation.status == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_op_wait_result_t waited{};
    REQUIRE(ovspatialquery_wait_op(scene.instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(waited.state == OVSPATIALQUERY_OP_COMPLETED);
    ovspatialquery_raycast_batch_result_t result{};
    REQUIRE(ovspatialquery_fetch_raycast_batch(scene.instance, operation.op_id, &result).status ==
            OVSPATIALQUERY_API_SUCCESS);
    const ovspatialquery_exact_feature_columns_t& features = result.columns.contact.exact.features;
    const uint64_t* offsets   = static_cast<const uint64_t*>(result.columns.hit_ranges.hit_offsets.data);
    const uint64_t* counts    = static_cast<const uint64_t*>(result.columns.hit_ranges.hit_counts.data);
    const uint64_t* faces     = static_cast<const uint64_t*>(features.source_face_indices.data);
    const uint32_t* corners   = static_cast<const uint32_t*>(features.source_corner_indices.data);
    const uint64_t* materials = static_cast<const uint64_t*>(result.columns.material_ids.data);
    for (uint32_t triangle = 0u; triangle < PolygonMesh::kTriangleCount; ++triangle)
    {
        REQUIRE(counts[triangle] == 1u);
        const uint64_t hit = offsets[triangle];
        REQUIRE(faces[hit] == mesh.faces[triangle]);
        REQUIRE(std::memcmp(corners + 3u * hit, mesh.corners[triangle], sizeof(mesh.corners[triangle])) == 0);
        REQUIRE(materials[hit] == (mesh.faces[triangle] == 7u ? kTriangleMaterial : kMeshMaterial));
    }
    REQUIRE(ovspatialquery_destroy_op(scene.instance, &operation.op_id).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_batch_plan(scene.instance, plan).status == OVSPATIALQUERY_API_SUCCESS);

    // Without the columns: the submitted ordinal, corners { 0, 1, 2 }, and each
    // triangle's own material entry.
    PolygonMesh  plainMesh;
    PolygonScene plain(plainMesh, false);
    for (uint32_t triangle = 0u; triangle < PolygonMesh::kTriangleCount; ++triangle)
    {
        const ovspatialquery_raycast_desc_t    desc = plain.ray(triangle);
        ovspatialquery_raycast_single_result_t single{};
        REQUIRE(ovspatialquery_raycast_closest_sync(plain.instance, &desc, &single).status ==
                OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(single.has_hit == 1u);
        const ovspatialquery_triangle_feature_t& feature = single.hit.data.exact.feature.data.triangle;
        REQUIRE(feature.source_face_index == triangle);
        REQUIRE(feature.source_face_corner_indices[0] == 0u && feature.source_face_corner_indices[1] == 1u &&
                feature.source_face_corner_indices[2] == 2u);
        REQUIRE(single.hit.data.exact.identity.material_id == (triangle < 2u ? kTriangleMaterial : kMeshMaterial));
    }
}

// Refusals before admission, each beside the registration that differs from it in
// one column and succeeds: one of the two columns alone, a triangle naming a
// corner twice, two materials in one face, and a face table beyond the budget.
void testSourceFaceRefusals()
{
    const ovspatialquery_instance_desc_t desc{8u, 1u, UINT64_C(64) << 20, nullptr, 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    uint64_t   nextValue = 100u;
    const auto attempt   = [&](PolygonMesh& mesh, bool withSourceFaces = true)
    {
        mesh.value                                                  = nextValue++;
        const ovspatialquery_triangle_mesh_definition_table_t table = mesh.table(withSourceFaces);
        return ovspatialquery_enqueue_triangle_mesh_registration(instance, &table);
    };
    std::vector<uint64_t> registered;
    const auto            accept = [&](PolygonMesh& mesh, bool withSourceFaces = true)
    {
        waitAndDestroy(instance, attempt(mesh, withSourceFaces));
        registered.push_back(mesh.value);
    };
    const auto lastError = []
    {
        const ovspatialquery_string_t error = ovspatialquery_get_last_error();
        return std::string(error.ptr != nullptr ? error.ptr : "", error.length);
    };

    PolygonMesh accepted;
    accept(accepted);

    PolygonMesh alone;
    alone.value                                           = nextValue++;
    ovspatialquery_triangle_mesh_definition_table_t table = alone.table();
    table.source_face_corner_indices                      = nullptr;
    requireRejected(ovspatialquery_enqueue_triangle_mesh_registration(instance, &table),
                    OVSPATIALQUERY_API_INVALID_ARGUMENT);
    table                     = alone.table();
    table.source_face_indices = nullptr;
    requireRejected(ovspatialquery_enqueue_triangle_mesh_registration(instance, &table),
                    OVSPATIALQUERY_API_INVALID_ARGUMENT);

    PolygonMesh repeated;
    repeated.corners[3][2] = 1u;
    requireRejected(attempt(repeated), OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(lastError().find("triangle 3 names one source face corner twice") != std::string::npos);

    // Triangle 1 of face 7 takes another material; zero, the object's, is another too.
    for (const uint64_t other : {kTriangleMaterial + 1u, uint64_t{0u}})
    {
        PolygonMesh mixed;
        mixed.materials[1] = other;
        requireRejected(attempt(mixed), OVSPATIALQUERY_API_INVALID_ARGUMENT);
        REQUIRE(lastError().find("triangles 0 and 1 belong to source face 7") != std::string::npos);
        // The same materials on submitted triangles are accepted without source faces.
        accept(mixed, false);
    }

    // With face materials, a face value whose table exceeds the 64 MiB budget, and
    // one at 2^32 - 1, which CUDA rows cannot key; without them, no table is kept and
    // any face value is accepted.
    PolygonMesh large;
    large.faces[0] = large.faces[1] = UINT64_C(1) << 24;
    requireRejected(attempt(large), OVSPATIALQUERY_API_HOST_BUDGET);
    large.faces[0] = large.faces[1] = UINT64_C(0xffffffff);
    requireRejected(attempt(large), OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(lastError().find("triangle 0 names source face 4294967295") != std::string::npos);
    large.faces[0] = large.faces[1] = UINT64_C(1) << 40;
    large.materials[0] = large.materials[1] = 0u;
    accept(large);
    for (const uint64_t value : registered)
    {
        REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, value})
                    .status == OVSPATIALQUERY_API_SUCCESS);
    }
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

} // namespace

int main()
{
    testSynchronousHitsReportTheSurfaceMaterial();
    testCpuBatchColumnsCarryMaterials();
    testMaterialBelongsToThePublishedVersion();
    testFaceMaterialTableMatchesTheTriangles();
    testMeshBothSidesKeepsTheSurfaceMaterial();
    testHitsReportCallerSourceFaces();
    testSourceFaceRefusals();
    std::puts("material identity tests passed");
    return 0;
}
