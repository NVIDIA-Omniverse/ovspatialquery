/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <ovspatialquery/ovspatialquery_explicit.h>
#include <ovspatialquery/ovspatialquery_ovstage.h>

#include <ovstage/ovstage.h>
#include <ovx/path_dictionary/path_dictionary.h>

#include "../../../source/ovspatialquery_instance_source.h"
#include "../../../source/ovspatialquery_ovstage_mesh_geometry.h"
#include "../../../source/ovspatialquery_typed_geometry.h"

#include "../../support/native/ovstage_companion_fixtures.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace
{

using ovspatialquery::detail::ConvexHullPreparationOutcome;
using ovspatialquery::detail::GeometryIdentityRegistryInspection;
using ovspatialquery::detail::inspectGeometryIdentityRegistry;

int failNextConvexHullPreparation(ovspatialquery_instance_handle_t instance, ConvexHullPreparationOutcome failure)
{
    ovspatialquery::detail::SourceGeometryPreparationContext preparation{};
    CHECK(ovspatialquery::detail::getSourceGeometryPreparationContext(instance, preparation) ==
          OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery::detail::failNextConvexHullPreparationForTest(preparation.backendRuntime, failure);
    return 0;
}

ovspatialquery_overlap_shape_t sphereShape(double centerX, double radius)
{
    ovspatialquery_overlap_shape_t shape{};
    shape.kind                  = OVSPATIALQUERY_OVERLAP_SPHERE;
    shape.data.sphere.center[0] = centerX;
    shape.data.sphere.radius    = radius;
    return shape;
}

ovspatialquery_overlap_shape_t aabbShape(double minimumX, double maximumX, double halfYz)
{
    ovspatialquery_overlap_shape_t shape{};
    shape.kind                    = OVSPATIALQUERY_OVERLAP_AABB;
    shape.data.aabb.min_corner[0] = minimumX;
    shape.data.aabb.min_corner[1] = -halfYz;
    shape.data.aabb.min_corner[2] = -halfYz;
    shape.data.aabb.max_corner[0] = maximumX;
    shape.data.aabb.max_corner[1] = halfYz;
    shape.data.aabb.max_corner[2] = halfYz;
    return shape;
}

ovspatialquery_overlap_shape_t boxShape(double centerX, double halfX, double halfYz)
{
    ovspatialquery_overlap_shape_t shape{};
    shape.kind                      = OVSPATIALQUERY_OVERLAP_BOX;
    shape.data.box.center[0]        = centerX;
    shape.data.box.rotation_xyzw[3] = 1.0f;
    shape.data.box.half_extents[0]  = halfX;
    shape.data.box.half_extents[1]  = halfYz;
    shape.data.box.half_extents[2]  = halfYz;
    return shape;
}

struct IdentityRow
{
    uint64_t                      objectId = 0u;
    ovspatialquery_geometry_ref_t geometry{};
};
using IdentityRows = std::vector<IdentityRow>;

// Reads every object published at `ordinal` with its geometry identity through
// a bounds-candidate overlap enclosing the whole stage, sorted by object id.
ovspatialquery_api_status_t publishedIdentities(ovspatialquery_instance_handle_t   instance,
                                                ovspatialquery_population_handle_t population,
                                                ovspatialquery_ordinal_t ordinal, IdentityRows& rows)
{
    ovspatialquery_overlap_desc_t desc{};
    desc.common.target.target                 = populationRef(population);
    desc.common.target.ordinal.kind           = OVSPATIALQUERY_ORDINAL_RETAINED_EXACT;
    desc.common.target.ordinal.source_ordinal = ordinal;
    desc.common.precision                     = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    desc.shape                                = aabbShape(-1000.0, 1000.0, 1000.0);
    std::array<ovspatialquery_overlap_hit_t, 32u> hits{};
    ovspatialquery_overlap_hit_buffer_t           buffer{hits.data(), hits.size()};
    ovspatialquery_multiple_result_t              result{};
    const ovspatialquery_api_status_t             status =
        ovspatialquery_overlap_multiple_sync(instance, &desc, &buffer, &result).status;
    if (status != OVSPATIALQUERY_API_SUCCESS || result.truncated != 0u || result.source_ordinal != ordinal)
    {
        return status != OVSPATIALQUERY_API_SUCCESS ? status : OVSPATIALQUERY_API_INTERNAL;
    }
    rows.clear();
    for (uint64_t index = 0u; index < result.stored_hit_count; ++index)
    {
        rows.push_back({hits[index].identity.object_id.value, hits[index].identity.geometry});
    }
    std::sort(rows.begin(), rows.end(),
              [](const IdentityRow& left, const IdentityRow& right) { return left.objectId < right.objectId; });
    return OVSPATIALQUERY_API_SUCCESS;
}

// True when an exact overlap of `shape` against `population` reports `objectId`.
bool overlapReports(ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population,
                    const ovspatialquery_overlap_shape_t& shape, uint64_t objectId)
{
    ovspatialquery_overlap_desc_t desc{};
    desc.common.target    = populationTarget(population);
    desc.common.precision = OVSPATIALQUERY_QUERY_EXACT;
    desc.shape            = shape;
    std::array<ovspatialquery_overlap_hit_t, 8u> hits{};
    ovspatialquery_overlap_hit_buffer_t          buffer{hits.data(), hits.size()};
    ovspatialquery_multiple_result_t             result{};
    return succeeded(ovspatialquery_overlap_multiple_sync(instance, &desc, &buffer, &result)) &&
           result.truncated == 0u &&
           std::any_of(hits.begin(), hits.begin() + static_cast<std::ptrdiff_t>(result.stored_hit_count),
                       [objectId](const ovspatialquery_overlap_hit_t& hit) {
                           return hit.truth == OVSPATIALQUERY_HIT_EXACT_SURFACE &&
                                  hit.identity.object_id.value == objectId;
                       });
}

bool populatePreciseMeshFixture(ovstage_instance_t* stage)
{
    static const char                         usda[] = R"usda(#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
    metersPerUnit = 1
)
def Xform "World"
{
    def Sphere "PreciseSphere"
    {
        double radius = 0.5
        float3 xformOp:rotateXYZ = (17, 31, 43)
        float3 xformOp:scale = (2, 2, 2)
        double3 xformOp:translate = (3, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    }
    def Sphere "NonUniformSphere"
    {
        double radius = 0.5
        float3 xformOp:scale = (2, 1, 2)
        double3 xformOp:translate = (8, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }
    def Cube "PreciseBox"
    {
        double size = 2
        float xformOp:rotateZ = 90
        float3 xformOp:scale = (1, 2, 0.5)
        double3 xformOp:translate = (8, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateZ", "xformOp:scale"]
    }
    def Cube "ShearedBox"
    {
        double size = 2
        matrix4d xformOp:transform = ((1, 0.25, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (16, 0, 0, 1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }
    def Cube "UnderEnclosedBox"
    {
        double size = 2
        float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
        double3 xformOp:translate = (22, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def Capsule "PreciseCapsule"
    {
        uniform token axis = "Y"
        double height = 1
        double radius = 0.25
        float3 xformOp:scale = (2, 2, 2)
        double3 xformOp:translate = (12, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }
    def Capsule "NonUniformCapsule"
    {
        uniform token axis = "Y"
        double height = 1
        double radius = 0.25
        float3 xformOp:scale = (2, 1, 2)
        double3 xformOp:translate = (18, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }
    def Capsule "UnsupportedAxisCapsule"
    {
        uniform token axis = "Q"
        double height = 1
        double radius = 0.25
        double3 xformOp:translate = (20, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def Sphere "SharedSphere"
    {
        double radius = 0.5
        float3 xformOp:scale = (2, 2, 2)
        double3 xformOp:translate = (60, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }
    def Mesh "PreciseConvex" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    )
    {
        uniform token physics:approximation = "convexHull"
        uniform bool doubleSided = 0
        uniform token orientation = "rightHanded"
        uniform token subdivisionScheme = "none"
        int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
        int[] faceVertexIndices = [0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 3, 7, 6, 2, 0, 4, 7, 3, 1, 2, 6, 5]
        point3f[] points = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
        float xformOp:rotateZ = 90
        float3 xformOp:scale = (2, 2, 2)
        double3 xformOp:translate = (16, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateZ", "xformOp:scale"]
    }
    def Mesh "ConcaveTaggedConvex" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    )
    {
        uniform token physics:approximation = "convexHull"
        uniform token orientation = "rightHanded"
        uniform token subdivisionScheme = "none"
        int[] faceVertexCounts = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        int[] faceVertexIndices = [0, 3, 2, 0, 2, 1, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5]
        point3f[] points = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (0.25, 0.25, 0.25), (-1, 1, 1)]
        double3 xformOp:translate = (24, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def Mesh "OpenTaggedConvex" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    )
    {
        uniform token physics:approximation = "convexHull"
        uniform token orientation = "rightHanded"
        uniform token subdivisionScheme = "none"
        int[] faceVertexCounts = [4, 4, 4, 4, 4]
        int[] faceVertexIndices = [0, 3, 2, 1, 0, 1, 5, 4, 3, 7, 6, 2, 0, 4, 7, 3, 1, 2, 6, 5]
        point3f[] points = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
        double3 xformOp:translate = (28, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def Mesh "ReversedTaggedConvex" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    )
    {
        uniform token physics:approximation = "convexHull"
        uniform token orientation = "rightHanded"
        uniform token subdivisionScheme = "none"
        int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
        int[] faceVertexIndices = [5, 6, 2, 1, 3, 7, 4, 0, 2, 6, 7, 3, 4, 5, 1, 0, 7, 6, 5, 4, 1, 2, 3, 0]
        point3f[] points = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
        double3 xformOp:translate = (32, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def Mesh "NonmanifoldTaggedConvex" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    )
    {
        uniform token physics:approximation = "convexHull"
        uniform token orientation = "rightHanded"
        uniform token subdivisionScheme = "none"
        int[] faceVertexCounts = [4, 4, 4, 4, 4, 4, 4]
        int[] faceVertexIndices = [0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 3, 7, 6, 2, 0, 4, 7, 3, 1, 2, 6, 5, 0, 3, 2, 1]
        point3f[] points = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
        double3 xformOp:translate = (36, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def Mesh "ZeroVolumeTaggedConvex" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    )
    {
        uniform token physics:approximation = "convexHull"
        uniform token orientation = "rightHanded"
        uniform token subdivisionScheme = "none"
        int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
        int[] faceVertexIndices = [0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 3, 7, 6, 2, 0, 4, 7, 3, 1, 2, 6, 5]
        point3f[] points = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0), (-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
        double3 xformOp:translate = (40, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def Mesh "DegenerateTransformTaggedConvex" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    )
    {
        uniform token physics:approximation = "convexHull"
        uniform token orientation = "rightHanded"
        uniform token subdivisionScheme = "none"
        int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
        int[] faceVertexIndices = [0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 3, 7, 6, 2, 0, 4, 7, 3, 1, 2, 6, 5]
        point3f[] points = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
        float3 xformOp:scale = (1, 1, 0)
        double3 xformOp:translate = (44, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }
    def Mesh "PointOnlyTaggedConvex" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    )
    {
        uniform token physics:approximation = "convexHull"
        point3f[] points = [(1, 1, 1), (-1, -1, -1), (1, -1, 1), (-1, 1, -1), (0, 0, 0), (1, -1, -1), (-1, 1, 1), (-1, -1, 1), (1, 1, -1)]
        float3[] extent = [(-1, -1, -1), (1, 1, 1)]
        double3 xformOp:translate = (58, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def Mesh "ApproximationOnlyMesh"
    {
        uniform token physics:approximation = "convexHull"
        uniform token orientation = "rightHanded"
        uniform token subdivisionScheme = "none"
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(48, -1, -1), (48, 0, 1), (48, 1, -1)]
    }
    def Mesh "SchemaOnlyMesh" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    )
    {
        uniform token orientation = "rightHanded"
        uniform token subdivisionScheme = "none"
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(50, -1, -1), (50, 0, 1), (50, 1, -1)]
    }
    def Xform "MeshParent"
    {
        matrix4d xformOp:transform = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (3, 0, 0, 1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
        def Mesh "PreciseMesh"
        {
            uniform bool doubleSided = 0
            uniform token orientation = "rightHanded"
            uniform token subdivisionScheme = "none"
            int[] faceVertexCounts = [3, 4]
            int[] faceVertexIndices = [4, 5, 6, 0, 1, 2, 3]
            point3f[] points = [(2, -1, -1), (2, -1, 1), (2, 1, 1), (2, 1, -1), (1, -1, -1), (1, 0, 1), (1, 1, -1)]
            float3[] extent = [(1, -1, -1), (2, 1, 1)]
        }
        def Mesh "MalformedTopologyMesh"
        {
            uniform bool doubleSided = 0
            uniform token orientation = "rightHanded"
            uniform token subdivisionScheme = "none"
            int[] faceVertexCounts = [2]
            int[] faceVertexIndices = [0, 1]
            point3f[] points = [(52, -1, -1), (52, 0, 1), (52, 1, -1)]
            float3[] extent = [(52, -1, -1), (52, 1, 1)]
        }
        def Mesh "EmptyTopologyMesh"
        {
            uniform bool doubleSided = 0
            uniform token orientation = "rightHanded"
            uniform token subdivisionScheme = "none"
            int[] faceVertexCounts = []
            int[] faceVertexIndices = []
            point3f[] points = [(54, -1, -1), (54, 0, 1), (54, 1, -1)]
            float3[] extent = [(54, -1, -1), (54, 1, 1)]
        }
        def Mesh "CatmullClarkMesh"
        {
            uniform bool doubleSided = 0
            uniform token orientation = "rightHanded"
            uniform token subdivisionScheme = "catmullClark"
            int[] faceVertexCounts = [3]
            int[] faceVertexIndices = [0, 1, 2]
            point3f[] points = [(1, -1, -1), (1, 0, 1), (1, 1, -1)]
            float3[] extent = [(1, -1, -1), (2, 1, 1)]
        }
        def Mesh "UnsupportedOrientationMesh"
        {
            uniform bool doubleSided = 0
            uniform token orientation = "leftHanded"
            uniform token subdivisionScheme = "none"
            int[] faceVertexCounts = [4]
            int[] faceVertexIndices = [0, 1, 2, 3]
            point3f[] points = [(1, -1, -1), (1, -1, 1), (1, 1, 1), (1, 1, -1)]
            float3[] extent = [(1, -1, -1), (1, 1, 1)]
        }
        def Mesh "MeshWithHole"
        {
            uniform bool doubleSided = 0
            uniform token orientation = "rightHanded"
            uniform token subdivisionScheme = "none"
            int[] holeIndices = [0]
            int[] faceVertexCounts = [3, 4]
            int[] faceVertexIndices = [4, 5, 6, 0, 1, 2, 3]
            point3f[] points = [(2, -1, -1), (2, -1, 1), (2, 1, 1), (2, 1, -1), (1, -1, -1), (1, 0, 1), (1, 1, -1)]
            float3[] extent = [(1, -1, -1), (2, 1, 1)]
        }
        def Mesh "UnsupportedHoleMesh"
        {
            uniform bool doubleSided = 0
            uniform token orientation = "rightHanded"
            uniform token subdivisionScheme = "none"
            int[] holeIndices = [0]
            int[] faceVertexCounts = [3]
            int[] faceVertexIndices = [0, 1, 2]
            point3f[] points = [(1, -1, -1), (1, 0, 1), (1, 1, -1)]
            float3[] extent = [(1, -1, -1), (2, 1, 1)]
        }
        def Mesh "UnsupportedTransformMesh"
        {
            uniform bool doubleSided = 0
            uniform token orientation = "rightHanded"
            uniform token subdivisionScheme = "none"
            int[] faceVertexCounts = [3]
            int[] faceVertexIndices = [0, 1, 2]
            point3f[] points = [(56, -1, -1), (56, 0, 1), (56, 1, -1)]
            float3[] extent = [(55.75, -1, -1), (56.25, 1, 1)]
            matrix4d xformOp:transform = ((1, 0, 0, 0), (0.25, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
            uniform token[] xformOpOrder = ["xformOp:transform"]
        }
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
    // An invalid axis has no computed extent; provide finite spatial data so the
    // typed acquisition, rather than the missing-spatial rule, diagnoses it.
    auto*      dictionary = ovstage_get_path_dictionary(stage);
    double     invalidAxisBounds[6]{19.75, -0.75, -0.25, 20.25, 0.75, 0.25};
    const char invalidAxisPath[] = "/World/UnsupportedAxisCapsule";
    if (!writeBounds(stage, dictionary, invalidAxisPath, sizeof(invalidAxisPath) - 1u, 2u, invalidAxisBounds))
        return false;
    const ovstage_write_floor_desc_t floorTwo{2u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
    return waitStageOperation(stage, ovstage_advance_write_floor(stage, &floorTwo));
}

bool finishWithExclusion(ovspatialquery_instance_handle_t instance, ovspatialquery_enqueue_result_t operation,
                         ovspatialquery_ordinal_t ordinal, uint64_t objects, uint64_t objectId,
                         ovspatialquery_ovstage_exclusion_reason_t reason)
{
    if (operation.status != OVSPATIALQUERY_API_SUCCESS)
    {
        return false;
    }
    ovspatialquery_op_wait_result_t           waited{};
    ovspatialquery_population_update_result_t result{};
    if (!succeeded(ovspatialquery_wait_op(instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited)) ||
        waited.state != OVSPATIALQUERY_OP_COMPLETED ||
        !succeeded(ovspatialquery_fetch_population_update(instance, operation.op_id, &result)) ||
        result.excluded_object_count != 1u || result.excluded_objects.data == nullptr ||
        result.exclusion_reasons.data == nullptr || result.excluded_objects.ndim != 1 ||
        result.exclusion_reasons.ndim != 1 || result.excluded_objects.shape == nullptr ||
        result.exclusion_reasons.shape == nullptr || result.excluded_objects.shape[0] != 1 ||
        result.exclusion_reasons.shape[0] != 1 ||
        *static_cast<const uint64_t*>(result.excluded_objects.data) != objectId ||
        *static_cast<const int32_t*>(result.exclusion_reasons.data) != static_cast<int32_t>(reason))
    {
        std::fprintf(
            stderr, "exclusion mismatch: E=%llu expectedID=%llu actualID=%llu expectedReason=%u actualReason=%d\n",
            static_cast<unsigned long long>(result.excluded_object_count), static_cast<unsigned long long>(objectId),
            static_cast<unsigned long long>(
                result.excluded_objects.data ? *static_cast<const uint64_t*>(result.excluded_objects.data) : 0),
            static_cast<unsigned>(reason),
            result.exclusion_reasons.data ? *static_cast<const int32_t*>(result.exclusion_reasons.data) : 0);
        return false;
    }
    return finishPopulationUpdate(instance, operation, ordinal, objects, 1u);
}

int32_t createOvstagePopulation(ovspatialquery_instance_handle_t instance, ovstage_instance_t* stage,
                                const ovstage_filter_t& filter, uint32_t retainedOrdinalCapacity,
                                ovspatialquery_population_handle_t& outPopulation)
{
    ovspatialquery_ovstage_population_desc_t desc{};
    desc.stage                            = stage;
    desc.filter                           = filter;
    desc.common.retained_ordinal_capacity = retainedOrdinalCapacity;
    return static_cast<int32_t>(ovspatialquery_create_ovstage_population(instance, &desc, &outPopulation).status);
}

int testRealPreciseSource()
{
    ovstage_instance_t*           stage = nullptr;
    const ovstage_instance_desc_t stageDesc{"ovspatialquery-precise-source-test"};
    CHECK(ovstage_create_instance(&stageDesc, &stage) == OVSTAGE_OK);
    CHECK(stage != nullptr);
    path_dictionary_instance_t* dictionary = ovstage_get_path_dictionary(stage);
    CHECK(dictionary != nullptr);
    CHECK(populatePreciseMeshFixture(stage));

    const ovx_string_t exactPathValues[] = {
        {"/World/PreciseSphere", sizeof("/World/PreciseSphere") - 1u},
        {"/World/MeshParent/PreciseMesh", sizeof("/World/MeshParent/PreciseMesh") - 1u},
        {"/World/NonUniformSphere", sizeof("/World/NonUniformSphere") - 1u},
        {"/World/MeshParent/CatmullClarkMesh", sizeof("/World/MeshParent/CatmullClarkMesh") - 1u},
        {"/World/MeshParent/UnsupportedOrientationMesh", sizeof("/World/MeshParent/UnsupportedOrientationMesh") - 1u},
        {"/World/MeshParent/UnsupportedHoleMesh", sizeof("/World/MeshParent/UnsupportedHoleMesh") - 1u},
        {"/World/PreciseBox", sizeof("/World/PreciseBox") - 1u},
        {"/World/ShearedBox", sizeof("/World/ShearedBox") - 1u},
        {"/World/PreciseCapsule", sizeof("/World/PreciseCapsule") - 1u},
        {"/World/NonUniformCapsule", sizeof("/World/NonUniformCapsule") - 1u},
        {"/World/UnsupportedAxisCapsule", sizeof("/World/UnsupportedAxisCapsule") - 1u},
        {"/World/UnderEnclosedBox", sizeof("/World/UnderEnclosedBox") - 1u},
        {"/World/PreciseConvex", sizeof("/World/PreciseConvex") - 1u},
        {"/World/ConcaveTaggedConvex", sizeof("/World/ConcaveTaggedConvex") - 1u},
        {"/World/OpenTaggedConvex", sizeof("/World/OpenTaggedConvex") - 1u},
        {"/World/ReversedTaggedConvex", sizeof("/World/ReversedTaggedConvex") - 1u},
        {"/World/NonmanifoldTaggedConvex", sizeof("/World/NonmanifoldTaggedConvex") - 1u},
        {"/World/ZeroVolumeTaggedConvex", sizeof("/World/ZeroVolumeTaggedConvex") - 1u},
        {"/World/ApproximationOnlyMesh", sizeof("/World/ApproximationOnlyMesh") - 1u},
        {"/World/SchemaOnlyMesh", sizeof("/World/SchemaOnlyMesh") - 1u},
        {"/World/DegenerateTransformTaggedConvex", sizeof("/World/DegenerateTransformTaggedConvex") - 1u},
        {"/World/SharedSphere", sizeof("/World/SharedSphere") - 1u},
        {"/World/MeshParent/MalformedTopologyMesh", sizeof("/World/MeshParent/MalformedTopologyMesh") - 1u},
        {"/World/MeshParent/EmptyTopologyMesh", sizeof("/World/MeshParent/EmptyTopologyMesh") - 1u},
        {"/World/MeshParent/UnsupportedTransformMesh", sizeof("/World/MeshParent/UnsupportedTransformMesh") - 1u},
        {"/World/PointOnlyTaggedConvex", sizeof("/World/PointOnlyTaggedConvex") - 1u},
        {"/World/MeshParent/MeshWithHole", sizeof("/World/MeshParent/MeshWithHole") - 1u},
    };
    std::array<ovx_primpath_t, 27u> exactPathHandles{};
    exactPathHandles.fill(OVX_INVALID_PRIMPATH);
    CHECK(dictionaryCallSucceeded(
        dictionary, dictionary->vtable->create_paths_from_strings(dictionary->context, exactPathValues,
                                                                  exactPathHandles.size(), exactPathHandles.data())));
    // Object IDs are the paths' values.
    std::array<uint64_t, 27u> exactPaths{};
    for (size_t index = 0u; index < exactPaths.size(); ++index)
        exactPaths[index] = handleValue(exactPathHandles[index]);

    const ovx_string_t geometryTypes[] = {
        {"Sphere", sizeof("Sphere") - 1u},
        {"Mesh", sizeof("Mesh") - 1u},
        {"Cube", sizeof("Cube") - 1u},
        {"Capsule", sizeof("Capsule") - 1u},
    };
    const ovx_string_t exactGeometryPaths[] = {
        exactPathValues[0], exactPathValues[1], exactPathValues[6], exactPathValues[8], exactPathValues[12],
    };
    ovstage_predicate_t geometryPredicates[2]{};
    geometryPredicates[0].attribute.string = ovx_string_t{"usd-prim-type", sizeof("usd-prim-type") - 1u};
    geometryPredicates[0].op               = OVSTAGE_FILTER_OP_IN;
    geometryPredicates[0].values           = geometryTypes;
    geometryPredicates[0].value_count      = 4u;
    geometryPredicates[1].attribute.string = ovx_string_t{"usd-path", sizeof("usd-path") - 1u};
    geometryPredicates[1].op               = OVSTAGE_FILTER_OP_IN;
    geometryPredicates[1].values           = exactGeometryPaths;
    geometryPredicates[1].value_count      = 5u;

    const ovspatialquery_instance_desc_t instanceDesc = commonInstanceDesc();
    ovspatialquery_instance_handle_t     instance     = OVSPATIALQUERY_INVALID_INSTANCE;
    CHECK(succeeded(ovspatialquery_create_instance(&instanceDesc, &instance)));

    // Content exclusions publish alongside a valid sphere in the same population.
    const auto setOnePrim = [&](const ovx_string_t&, size_t pathIndex,
                                ovspatialquery_ovstage_exclusion_reason_t reason) -> int
    {
        const ovx_string_t  paths[]             = {exactPathValues[pathIndex], exactPathValues[21]};
        ovstage_predicate_t predicate           = geometryPredicates[1];
        predicate.values                        = paths;
        predicate.value_count                   = 2u;
        ovspatialquery_population_handle_t prim = OVSPATIALQUERY_INVALID_POPULATION;
        CHECK(createOvstagePopulation(instance, stage, ovstage_filter_t{&predicate, 1u}, 2u, prim) == 0);
        const ovspatialquery_enqueue_result_t set = ovspatialquery_enqueue_ovstage_population_set(instance, prim, 2);
        if (static_cast<int32_t>(reason) == 0)
        {
            CHECK(finishPopulationUpdate(instance, set, 2, 2u, 0u));
        }
        else
        {
            CHECK(finishWithExclusion(instance, set, 2, 1u, exactPaths[pathIndex], reason));
        }
        IdentityRows rows;
        CHECK(publishedIdentities(instance, prim, 2, rows) == OVSPATIALQUERY_API_SUCCESS);
        CHECK(std::any_of(rows.begin(), rows.end(),
                          [&](const IdentityRow& row) { return row.objectId == exactPaths[21]; }));
        CHECK(succeeded(ovspatialquery_destroy_population(instance, prim)));
        return 0;
    };
    const auto acceptsPrim = [&](const ovx_string_t& type, size_t pathIndex)
    { return setOnePrim(type, pathIndex, static_cast<ovspatialquery_ovstage_exclusion_reason_t>(0)) == 0; };
    const auto excludesPrim =
        [&](const ovx_string_t& type, size_t pathIndex, ovspatialquery_ovstage_exclusion_reason_t reason)
    { return setOnePrim(type, pathIndex, reason) == 0; };
    CHECK(excludesPrim(geometryTypes[0], 2u, OVSPATIALQUERY_OVSTAGE_EXCLUSION_UNSUPPORTED_SHAPE));
    CHECK(acceptsPrim(geometryTypes[2], 7u));
    // A stale but present _worldExtent cannot exclude an otherwise valid Cube.
    CHECK(acceptsPrim(geometryTypes[2], 11u));
    CHECK(excludesPrim(geometryTypes[3], 9u, OVSPATIALQUERY_OVSTAGE_EXCLUSION_UNSUPPORTED_SHAPE));
    CHECK(excludesPrim(geometryTypes[3], 10u, OVSPATIALQUERY_OVSTAGE_EXCLUSION_UNSUPPORTED_SHAPE));
    CHECK(acceptsPrim(geometryTypes[1], 18u));
    // A prepared convex's definition leaves the registry, and its bytes the host
    // ledger, once no publication carries it; the next staging sweeps it. After the
    // concave and point-only convexes and a second mesh probe, the instance holds
    // what it held after the first mesh probe.
    uint64_t hostAfterMeshProbe = 0u;
    CHECK(readHostBytes(instance, hostAfterMeshProbe));
    CHECK(acceptsPrim(geometryTypes[1], 13u));
    CHECK(acceptsPrim(geometryTypes[1], 25u));
    CHECK(acceptsPrim(geometryTypes[1], 18u));
    uint64_t hostAfterConvexProbes = 0u;
    CHECK(readHostBytes(instance, hostAfterConvexProbes));
    CHECK(hostAfterConvexProbes == hostAfterMeshProbe);
    CHECK(acceptsPrim(geometryTypes[1], 19u));
    // Catmull-Clark control faces and left-handed winding are supported; an
    // all-holes mesh and a zero-determinant hull have no usable surface.
    CHECK(acceptsPrim(geometryTypes[1], 3u));
    CHECK(acceptsPrim(geometryTypes[1], 4u));
    CHECK(excludesPrim(geometryTypes[1], 5u, OVSPATIALQUERY_OVSTAGE_EXCLUSION_MALFORMED_GEOMETRY));
    CHECK(excludesPrim(geometryTypes[1], 20u, OVSPATIALQUERY_OVSTAGE_EXCLUSION_MALFORMED_GEOMETRY));
    CHECK(acceptsPrim(geometryTypes[1], 24u));

    // The hole removes authored face 0 at x=4. The remaining quad at x=5
    // still reports authored face 1. Left-handed winding makes +x its front normal.
    for (size_t pathIndex : {26u, 4u})
    {
        ovstage_predicate_t predicate                         = geometryPredicates[1];
        predicate.values                                      = &exactPathValues[pathIndex];
        predicate.value_count                                 = 1u;
        ovspatialquery_population_handle_t topologyPopulation = OVSPATIALQUERY_INVALID_POPULATION;
        CHECK(createOvstagePopulation(instance, stage, ovstage_filter_t{&predicate, 1u}, 2u, topologyPopulation) == 0);
        CHECK(finishPopulationUpdate(
            instance, ovspatialquery_enqueue_ovstage_population_set(instance, topologyPopulation, 2), 2, 1u, 0u));
        ovspatialquery_raycast_desc_t topologyRay{};
        topologyRay.common.target         = populationTarget(topologyPopulation);
        topologyRay.common.precision      = OVSPATIALQUERY_QUERY_EXACT;
        topologyRay.ray.max_distance      = 10.0;
        topologyRay.ray.origin[0]         = pathIndex == 4u ? 6.0 : 0.0;
        topologyRay.ray.unit_direction[0] = pathIndex == 4u ? -1.0f : 1.0f;
        ovspatialquery_raycast_single_result_t topologyHit{};
        CHECK(succeeded(ovspatialquery_raycast_closest_sync(instance, &topologyRay, &topologyHit)));
        CHECK(topologyHit.has_hit == 1u);
        CHECK(topologyHit.hit.data.exact.identity.object_id.value == exactPaths[pathIndex]);
        CHECK(std::abs(topologyHit.hit.data.exact.distance - (pathIndex == 4u ? 2.0 : 5.0)) < 1.0e-12);
        topologyRay.ray.origin[1] = 0.75;
        CHECK(succeeded(ovspatialquery_raycast_closest_sync(instance, &topologyRay, &topologyHit)));
        CHECK(topologyHit.has_hit == 1u);
        CHECK(topologyHit.hit.data.exact.feature.kind == OVSPATIALQUERY_GEOMETRY_FEATURE_TRIANGLE);
        const ovspatialquery_triangle_feature_t& feature = topologyHit.hit.data.exact.feature.data.triangle;
        CHECK(feature.source_face_index == (pathIndex == 4u ? 0u : 1u));
        CHECK(feature.source_face_corner_indices[0] == 0u);
        CHECK(feature.source_face_corner_indices[1] == (pathIndex == 4u ? 3u : 2u));
        CHECK(feature.source_face_corner_indices[2] == (pathIndex == 4u ? 2u : 3u));
        CHECK(topologyHit.hit.data.exact.normal[0] == (pathIndex == 4u ? 1.0f : -1.0f));
        topologyRay.ray.origin[0] = pathIndex == 4u ? 0.0 : 6.0;
        topologyRay.ray.unit_direction[0] *= -1.0f;
        CHECK(succeeded(ovspatialquery_raycast_closest_sync(instance, &topologyRay, &topologyHit)));
        CHECK(topologyHit.has_hit == 0u);
        CHECK(succeeded(ovspatialquery_destroy_population(instance, topologyPopulation)));
    }

    static const char   meshPath[] = "/World/MeshParent/PreciseMesh";
    const ovx_string_t  meshPathValue{meshPath, sizeof(meshPath) - 1u};
    const uint64_t      expectedPath = exactPaths[1];
    ovstage_predicate_t pathPredicate{};
    pathPredicate.attribute.string = ovx_string_t{"usd-path", sizeof("usd-path") - 1u};
    pathPredicate.op               = OVSTAGE_FILTER_OP_IN;
    pathPredicate.values           = &meshPathValue;
    pathPredicate.value_count      = 1u;

    ovspatialquery_ovstage_population_desc_t sourceDesc{};
    sourceDesc.stage                              = stage;
    sourceDesc.filter                             = ovstage_filter_t{&pathPredicate, 1u};
    sourceDesc.common                             = aabbPopulationDesc();
    ovspatialquery_population_handle_t population = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &sourceDesc, &population)));
    ovspatialquery_ovstage_view_desc_t viewDesc{};
    viewDesc.filter                   = ovstage_filter_t{&pathPredicate, 1u};
    ovspatialquery_view_handle_t view = OVSPATIALQUERY_INVALID_VIEW;
    CHECK(succeeded(ovspatialquery_create_ovstage_view(instance, population, &viewDesc, &view)));

    ovspatialquery_enqueue_result_t operation = ovspatialquery_enqueue_ovstage_population_set(instance, population, 2);
    CHECK(finishPopulationUpdate(instance, operation, 2, 1u, 0u));

    ovspatialquery_raycast_desc_t ray{};
    ray.common.target         = populationTarget(population);
    ray.common.precision      = OVSPATIALQUERY_QUERY_EXACT;
    ray.ray.unit_direction[0] = 1.0f;
    ray.ray.max_distance      = 20.0;
    ovspatialquery_raycast_single_result_t closest{};
    CHECK(succeeded(ovspatialquery_raycast_closest_sync(instance, &ray, &closest)));
    CHECK(closest.source_ordinal == 2);
    CHECK(closest.has_hit == 1u);
    CHECK(closest.hit.truth == OVSPATIALQUERY_HIT_EXACT_SURFACE);
    CHECK(closest.hit.data.exact.identity.object_id.value == expectedPath);
    CHECK(std::abs(closest.hit.data.exact.distance - 4.0) < 1.0e-12);
    CHECK(closest.hit.data.exact.feature.kind == OVSPATIALQUERY_GEOMETRY_FEATURE_TRIANGLE);
    CHECK(closest.hit.data.exact.feature.data.triangle.source_face_index == 0u);
    CHECK(std::abs(closest.hit.data.exact.feature.data.triangle.barycentrics[1] - 0.5f) < 1.0e-6f);
    CHECK(std::abs(closest.hit.data.exact.feature.data.triangle.barycentrics[2] - 0.25f) < 1.0e-6f);

    ovspatialquery_raycast_desc_t backRay = ray;
    backRay.ray.origin[0]                 = 6.0;
    backRay.ray.unit_direction[0]         = -1.0f;
    ovspatialquery_raycast_single_result_t backFace{};
    CHECK(succeeded(ovspatialquery_raycast_closest_sync(instance, &backRay, &backFace)));
    CHECK(backFace.has_hit == 0u);

    ovspatialquery_sphere_sweep_desc_t sweep{};
    sweep.common                  = ray.common;
    sweep.sweep.radius            = 0.25;
    sweep.sweep.unit_direction[0] = 1.0f;
    sweep.sweep.max_distance      = 20.0;
    ovspatialquery_sphere_sweep_closest_result_t sweepResult{};
    CHECK(succeeded(ovspatialquery_sweep_sphere_closest_sync(instance, &sweep, &sweepResult)));
    CHECK(sweepResult.source_ordinal == 2);
    CHECK(sweepResult.has_hit == 1u);
    CHECK(std::abs(sweepResult.hit.data.exact.distance - 3.75) < 1.0e-12);
    CHECK(sweepResult.hit.data.exact.feature.kind == OVSPATIALQUERY_GEOMETRY_FEATURE_TRIANGLE);
    CHECK(sweepResult.hit.data.exact.feature.data.triangle.source_face_index == 0u);

    ovspatialquery_overlap_desc_t overlap{};
    overlap.common                      = ray.common;
    overlap.shape.kind                  = OVSPATIALQUERY_OVERLAP_SPHERE;
    overlap.shape.data.sphere.center[0] = 4.0;
    overlap.shape.data.sphere.center[1] = 0.9;
    overlap.shape.data.sphere.center[2] = 0.9;
    overlap.shape.data.sphere.radius    = 0.05;
    ovspatialquery_overlap_any_result_t exactMiss{};
    CHECK(succeeded(ovspatialquery_overlap_any_sync(instance, &overlap, &exactMiss)));
    CHECK(exactMiss.has_hit == 0u);
    overlap.common.precision = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    ovspatialquery_overlap_any_result_t candidateHit{};
    CHECK(succeeded(ovspatialquery_overlap_any_sync(instance, &overlap, &candidateHit)));
    CHECK(candidateHit.has_hit == 1u);

    ray.common.target = viewTarget(view);
    closest           = {};
    CHECK(succeeded(ovspatialquery_raycast_closest_sync(instance, &ray, &closest)));
    CHECK(closest.has_hit == 1u);
    CHECK(closest.hit.data.exact.feature.data.triangle.source_face_index == 0u);

    ovspatialquery_raycast_plan_desc_t planDesc{};
    planDesc.target                         = viewRef(view);
    planDesc.hit_mode                       = OVSPATIALQUERY_HIT_MODE_CLOSEST;
    planDesc.precision                      = OVSPATIALQUERY_QUERY_EXACT;
    planDesc.device                         = DLDevice{kDLCPU, 0};
    planDesc.maximum_query_count            = 1u;
    planDesc.maximum_hits_per_query         = 1u;
    planDesc.hit_capacity_per_slot          = 1u;
    planDesc.result_slot_count              = 1u;
    planDesc.result_fields                  = OVSPATIALQUERY_BATCH_RESULT_SOURCE_FEATURE;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    CHECK(succeeded(ovspatialquery_create_raycast_plan(instance, &planDesc, &plan)));

    double                         origins[][3]       = {{0.0, 0.0, 0.0}};
    float                          directions[][3]    = {{1.0f, 0.0f, 0.0f}};
    double                         maximumDistances[] = {20.0};
    InputArray                     originColumn(origins[0], 1, 3, kDLFloat);
    InputArray                     directionColumn(directions[0], 1, 3, kDLFloat);
    InputArray                     maximumColumn(maximumDistances, 1, kDLFloat);
    ovspatialquery_raycast_batch_t launch{};
    launch.ordinal.kind    = OVSPATIALQUERY_ORDINAL_HEAD;
    launch.query_count     = 1u;
    launch.origins         = &originColumn.value;
    launch.unit_directions = &directionColumn.value;
    launch.max_distances   = &maximumColumn.value;
    ovspatialquery_raycast_batch_reservation_t acceptedViews{};
    operation = ovspatialquery_enqueue_raycast_plan(instance, plan, &launch, &acceptedViews);
    CHECK(operation.status == OVSPATIALQUERY_API_SUCCESS);
    CHECK(acceptedViews.common.source_ordinal == 2);
    CHECK(acceptedViews.common.truth == OVSPATIALQUERY_HIT_EXACT_SURFACE);
    ovspatialquery_op_wait_result_t waited{};
    CHECK(succeeded(ovspatialquery_wait_op(instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited)));
    ovspatialquery_raycast_batch_result_t fetched{};
    CHECK(succeeded(ovspatialquery_fetch_raycast_batch(instance, operation.op_id, &fetched)));
    CHECK(fetched.common.source_ordinal == 2);
    CHECK(fetched.columns.contact.exact.features.source_face_indices.data != nullptr);
    CHECK(*static_cast<const uint64_t*>(fetched.columns.contact.exact.features.source_face_indices.data) == 0u);
    CHECK(fetched.columns.object_ids.data != nullptr);
    CHECK(*static_cast<const uint64_t*>(fetched.columns.object_ids.data) == expectedPath);
    CHECK(succeeded(ovspatialquery_destroy_op(instance, &operation.op_id)));
    CHECK(succeeded(ovspatialquery_destroy_batch_plan(instance, plan)));

    uint8_t doubleSided = 1u;
    CHECK(writeAttribute(stage, dictionary, meshPath, sizeof(meshPath) - 1u, "doubleSided", 3u, &doubleSided,
                         DLDataType{kDLBool, 8u, 1u}));
    const ovstage_write_floor_desc_t floorThree{3u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
    CHECK(waitStageOperation(stage, ovstage_advance_write_floor(stage, &floorThree)));
    operation = ovspatialquery_enqueue_ovstage_population_update(instance, population, 2, 3);
    CHECK(finishPopulationUpdate(instance, operation, 3, 1u, 0u));
    backRay.common.target = populationTarget(population);
    backFace              = {};
    CHECK(succeeded(ovspatialquery_raycast_closest_sync(instance, &backRay, &backFace)));
    CHECK(backFace.source_ordinal == 3);
    CHECK(backFace.has_hit == 1u);
    CHECK(backFace.hit.data.exact.feature.data.triangle.source_face_index == 1u);

    ovspatialquery_population_handle_t typedPopulation = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(createOvstagePopulation(instance, stage, ovstage_filter_t{geometryPredicates, 2u}, 4u, typedPopulation) == 0);
    // One-object views of the box, capsule and convex let a sweep from the origin
    // reach each of them past the sphere in front.
    const size_t                                 familyViewPaths[] = {6u, 8u, 12u};
    std::array<ovspatialquery_view_handle_t, 3u> familyViews{};
    for (size_t index = 0u; index < familyViews.size(); ++index)
    {
        ovstage_predicate_t familyPredicate = geometryPredicates[1];
        familyPredicate.values              = &exactPathValues[familyViewPaths[index]];
        familyPredicate.value_count         = 1u;
        ovspatialquery_ovstage_view_desc_t familyViewDesc{};
        familyViewDesc.filter = ovstage_filter_t{&familyPredicate, 1u};
        CHECK(succeeded(
            ovspatialquery_create_ovstage_view(instance, typedPopulation, &familyViewDesc, &familyViews[index])));
    }
    operation = ovspatialquery_enqueue_ovstage_population_set(instance, typedPopulation, 3);
    CHECK(finishPopulationUpdate(instance, operation, 3, 5u, 0u));

    // A +x ray from the origin meets the sphere (radius 1 at x 3), the mesh triangle
    // at x 4, the box (half-extent 2 at x 8), the capsule (radius 0.5 at x 12) and
    // the convex (half-extent 2 at x 16); each object reports its own nearest hit.
    std::array<ovspatialquery_raycast_hit_t, 8u> typedRayHits{};
    ovspatialquery_raycast_hit_buffer_t          typedRayBuffer{typedRayHits.data(), typedRayHits.size()};
    ovspatialquery_multiple_result_t             typedRayResult{};
    const auto typedRayHit = [&](uint64_t objectId) -> const ovspatialquery_raycast_exact_hit_t*
    {
        for (uint64_t row = 0u; row < typedRayResult.stored_hit_count; ++row)
        {
            if (typedRayHits[row].truth == OVSPATIALQUERY_HIT_EXACT_SURFACE &&
                typedRayHits[row].data.exact.identity.object_id.value == objectId)
            {
                return &typedRayHits[row].data.exact;
            }
        }
        return nullptr;
    };
    ovspatialquery_raycast_desc_t typedRay{};
    typedRay.common.target         = populationTarget(typedPopulation);
    typedRay.common.precision      = OVSPATIALQUERY_QUERY_EXACT;
    typedRay.ray.unit_direction[0] = 1.0f;
    typedRay.ray.max_distance      = 20.0;
    typedRay.mesh_hits             = OVSPATIALQUERY_MESH_HITS_NEAREST;
    CHECK(succeeded(ovspatialquery_raycast_multiple_sync(instance, &typedRay, &typedRayBuffer, &typedRayResult)));
    CHECK(typedRayResult.source_ordinal == 3);
    CHECK(typedRayResult.stored_hit_count == 5u);
    CHECK(typedRayResult.truncated == 0u);
    const std::pair<size_t, double> typedRayDistances[] = {{0u, 2.0}, {1u, 4.0}, {6u, 6.0}, {8u, 11.5}, {12u, 14.0}};
    for (const auto& [pathIndex, distance] : typedRayDistances)
    {
        CHECK(typedRayHit(exactPaths[pathIndex]) != nullptr);
        CHECK(std::abs(typedRayHit(exactPaths[pathIndex])->distance - distance) < 1.0e-12);
    }
    CHECK(typedRayHit(exactPaths[12])->feature.kind == OVSPATIALQUERY_GEOMETRY_FEATURE_NONE);
    // At y 0.75 the ray passes beside the triangle and crosses the quad (source
    // face 1) inside its fan triangle over corners 0, 2 and 3.
    typedRay.ray.origin[1] = 0.75;
    CHECK(succeeded(ovspatialquery_raycast_multiple_sync(instance, &typedRay, &typedRayBuffer, &typedRayResult)));
    const ovspatialquery_raycast_exact_hit_t* quadHit = typedRayHit(exactPaths[1]);
    CHECK(quadHit != nullptr);
    CHECK(quadHit->feature.kind == OVSPATIALQUERY_GEOMETRY_FEATURE_TRIANGLE);
    CHECK(quadHit->feature.data.triangle.source_face_index == 1u);
    CHECK(quadHit->feature.data.triangle.source_face_corner_indices[0] == 0u);
    CHECK(quadHit->feature.data.triangle.source_face_corner_indices[1] == 2u);
    CHECK(quadHit->feature.data.triangle.source_face_corner_indices[2] == 3u);

    // A radius-0.25 sweep stops 0.25 before each surface the ray meets.
    ovspatialquery_sphere_sweep_desc_t typedSweep{};
    typedSweep.common                  = typedRay.common;
    typedSweep.sweep.radius            = 0.25;
    typedSweep.sweep.unit_direction[0] = 1.0f;
    typedSweep.sweep.max_distance      = 20.0;
    ovspatialquery_sphere_sweep_closest_result_t typedSweepHit{};
    CHECK(succeeded(ovspatialquery_sweep_sphere_closest_sync(instance, &typedSweep, &typedSweepHit)));
    CHECK(typedSweepHit.has_hit == 1u);
    CHECK(typedSweepHit.hit.data.exact.identity.object_id.value == exactPaths[0]);
    CHECK(std::abs(typedSweepHit.hit.data.exact.distance - 1.75) < 1.0e-12);
    const double familySweepDistances[] = {5.75, 11.25, 13.75};
    for (size_t index = 0u; index < familyViews.size(); ++index)
    {
        typedSweep.common.target = viewTarget(familyViews[index]);
        typedSweepHit            = {};
        CHECK(succeeded(ovspatialquery_sweep_sphere_closest_sync(instance, &typedSweep, &typedSweepHit)));
        CHECK(typedSweepHit.source_ordinal == 3);
        CHECK(typedSweepHit.has_hit == 1u);
        CHECK(typedSweepHit.hit.data.exact.identity.object_id.value == exactPaths[familyViewPaths[index]]);
        CHECK(std::abs(typedSweepHit.hit.data.exact.distance - familySweepDistances[index]) < 1.0e-12);
    }
    CHECK(typedSweepHit.hit.data.exact.feature.kind == OVSPATIALQUERY_GEOMETRY_FEATURE_NONE);

    // Sphere, AABB and box queries touching or entering each solid from +x
    // overlap it.
    const std::pair<size_t, std::array<ovspatialquery_overlap_shape_t, 3u>> familyOverlaps[] = {
        {0u, {sphereShape(4.25, 0.25), aabbShape(1.75, 2.0, 0.1), boxShape(1.75, 0.25, 0.1)}},
        {6u, {sphereShape(5.5, 0.5), aabbShape(5.5, 6.0, 0.25), boxShape(5.75, 0.25, 0.25)}},
        {8u, {sphereShape(11.0, 0.5), aabbShape(11.0, 11.5, 0.25), boxShape(11.25, 0.25, 0.25)}},
        {12u, {sphereShape(16.0, 0.1), aabbShape(15.9, 16.1, 0.1), boxShape(16.0, 0.1, 0.1)}},
    };
    for (const auto& [pathIndex, shapes] : familyOverlaps)
    {
        for (const ovspatialquery_overlap_shape_t& shape : shapes)
        {
            CHECK(overlapReports(instance, typedPopulation, shape, exactPaths[pathIndex]));
        }
    }
    for (ovspatialquery_view_handle_t familyView : familyViews)
    {
        CHECK(succeeded(ovspatialquery_destroy_view(instance, familyView)));
    }

    IdentityRows typedRows;
    CHECK(publishedIdentities(instance, typedPopulation, 3, typedRows) == OVSPATIALQUERY_API_SUCCESS);
    CHECK(typedRows.size() == 5u);
    const auto findIdentity = [&typedRows](uint64_t objectId)
    {
        return std::find_if(typedRows.begin(), typedRows.end(),
                            [objectId](const auto& row) { return row.objectId == objectId; });
    };
    const auto typedSphere  = findIdentity(exactPaths[0]);
    const auto typedBox     = findIdentity(exactPaths[6]);
    const auto typedCapsule = findIdentity(exactPaths[8]);
    const auto typedConvex  = findIdentity(exactPaths[12]);
    const auto typedMesh    = findIdentity(exactPaths[1]);
    CHECK(typedSphere != typedRows.end());
    CHECK(typedBox != typedRows.end());
    CHECK(typedCapsule != typedRows.end());
    CHECK(typedConvex != typedRows.end());
    CHECK(typedMesh != typedRows.end());
    CHECK(typedSphere->geometry.value != 0u);
    CHECK(typedSphere->geometry.type == OVSPATIALQUERY_GEOMETRY_SPHERE);
    CHECK(typedBox->geometry.type == OVSPATIALQUERY_GEOMETRY_BOX);
    CHECK(typedCapsule->geometry.type == OVSPATIALQUERY_GEOMETRY_CAPSULE);
    CHECK(typedConvex->geometry.type == OVSPATIALQUERY_GEOMETRY_CONVEX);
    CHECK(typedMesh->geometry.type == OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH);
    GeometryIdentityRegistryInspection registryBeforeCollision{};
    CHECK(inspectGeometryIdentityRegistry(instance, registryBeforeCollision) == OVSPATIALQUERY_API_SUCCESS);
    uint64_t                                       collidingValue  = typedSphere->geometry.value;
    double                                         collidingRadius = 3.0;
    InputArray                                     collidingValues(&collidingValue, 1, kDLUInt);
    InputArray                                     collidingRadii(&collidingRadius, 1, kDLFloat);
    const ovspatialquery_sphere_definition_table_t collidingSphere{&collidingValues.value, &collidingRadii.value};
    const ovspatialquery_enqueue_result_t          collision =
        ovspatialquery_enqueue_sphere_registration(instance, &collidingSphere);
    CHECK(collision.status == OVSPATIALQUERY_API_INVALID_ARGUMENT);
    CHECK(collision.op_id == OVSPATIALQUERY_INVALID_OP);
    GeometryIdentityRegistryInspection registryAfterCollision{};
    CHECK(inspectGeometryIdentityRegistry(instance, registryAfterCollision) == OVSPATIALQUERY_API_SUCCESS);
    CHECK(registryAfterCollision.definitionCount == registryBeforeCollision.definitionCount);
    CHECK(registryAfterCollision.nextValue == registryBeforeCollision.nextValue);
    CHECK(registryAfterCollision.residentByteCount == registryBeforeCollision.residentByteCount);

    const ovx_string_t  sharedSpherePaths[]             = {exactPathValues[0], exactPathValues[21]};
    ovstage_predicate_t sharedPredicates[2]             = {geometryPredicates[0], geometryPredicates[1]};
    sharedPredicates[0].values                          = &geometryTypes[0];
    sharedPredicates[0].value_count                     = 1u;
    sharedPredicates[1].values                          = sharedSpherePaths;
    sharedPredicates[1].value_count                     = 2u;
    ovspatialquery_population_handle_t sharedPopulation = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(createOvstagePopulation(instance, stage, ovstage_filter_t{sharedPredicates, 2u}, 4u, sharedPopulation) == 0);
    operation = ovspatialquery_enqueue_ovstage_population_set(instance, sharedPopulation, 3);
    CHECK(finishPopulationUpdate(instance, operation, 3, 2u, 0u));
    // Every population's IDs are checked against the stage dictionary's path values, which is
    // what makes populations over one stage agree on a prim's ID (ovspatialquery_ovstage.h).
    IdentityRows sharedAtThree;
    CHECK(publishedIdentities(instance, sharedPopulation, 3, sharedAtThree) == OVSPATIALQUERY_API_SUCCESS);
    CHECK(sharedAtThree.size() == 2u);
    CHECK(sharedAtThree[0].objectId != sharedAtThree[1].objectId);
    CHECK(sharedAtThree[0].geometry.type == sharedAtThree[1].geometry.type);
    CHECK(sharedAtThree[0].geometry.value == sharedAtThree[1].geometry.value);
    CHECK(sharedAtThree[0].geometry.type == typedSphere->geometry.type);
    CHECK(sharedAtThree[0].geometry.value == typedSphere->geometry.value);

    // Explicit spheres registered before an ovstage publication leave it its own references:
    // explicit values end below 2^63, where the values the adapter assigns begin.
    {
        ovspatialquery_instance_handle_t explicitFirst = OVSPATIALQUERY_INVALID_INSTANCE;
        CHECK(succeeded(ovspatialquery_create_instance(&instanceDesc, &explicitFirst)));
        uint64_t                                       reservedValue  = UINT64_C(1) << 63u;
        double                                         reservedRadius = 3.0;
        InputArray                                     reservedValues(&reservedValue, 1, kDLUInt);
        InputArray                                     reservedRadii(&reservedRadius, 1, kDLFloat);
        const ovspatialquery_sphere_definition_table_t reservedSphere{&reservedValues.value, &reservedRadii.value};
        const ovspatialquery_enqueue_result_t          reserved =
            ovspatialquery_enqueue_sphere_registration(explicitFirst, &reservedSphere);
        CHECK(reserved.status == OVSPATIALQUERY_API_INVALID_ARGUMENT);
        CHECK(reserved.op_id == OVSPATIALQUERY_INVALID_OP);

        uint64_t                                       explicitValues[2] = {1u, reservedValue - 1u};
        double                                         explicitRadii[2]  = {3.0, 4.0};
        InputArray                                     explicitValueColumn(explicitValues, 2, kDLUInt);
        InputArray                                     explicitRadiusColumn(explicitRadii, 2, kDLFloat);
        const ovspatialquery_sphere_definition_table_t explicitSpheres{&explicitValueColumn.value,
                                                                       &explicitRadiusColumn.value};
        ovspatialquery_enqueue_result_t                registration =
            ovspatialquery_enqueue_sphere_registration(explicitFirst, &explicitSpheres);
        CHECK(registration.status == OVSPATIALQUERY_API_SUCCESS);
        ovspatialquery_op_wait_result_t registered{};
        CHECK(succeeded(
            ovspatialquery_wait_op(explicitFirst, registration.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &registered)));
        CHECK(registered.state == OVSPATIALQUERY_OP_COMPLETED);
        CHECK(succeeded(ovspatialquery_destroy_op(explicitFirst, &registration.op_id)));

        ovspatialquery_population_handle_t afterExplicit = OVSPATIALQUERY_INVALID_POPULATION;
        CHECK(createOvstagePopulation(explicitFirst, stage, ovstage_filter_t{sharedPredicates, 2u}, 1u,
                                      afterExplicit) == 0);
        CHECK(finishPopulationUpdate(
            explicitFirst, ovspatialquery_enqueue_ovstage_population_set(explicitFirst, afterExplicit, 3), 3, 2u, 0u));
        IdentityRows afterExplicitRows;
        CHECK(publishedIdentities(explicitFirst, afterExplicit, 3, afterExplicitRows) == OVSPATIALQUERY_API_SUCCESS);
        CHECK(afterExplicitRows.size() == 2u);
        for (const IdentityRow& row : afterExplicitRows)
        {
            CHECK(row.geometry.type == OVSPATIALQUERY_GEOMETRY_SPHERE);
            CHECK(row.geometry.value >= reservedValue);
        }
        CHECK(succeeded(ovspatialquery_destroy_population(explicitFirst, afterExplicit)));
        for (uint64_t value : explicitValues)
        {
            CHECK(succeeded(ovspatialquery_destroy_geometry_definition(
                explicitFirst, ovspatialquery_geometry_ref_t{OVSPATIALQUERY_GEOMETRY_SPHERE, value})));
        }
        CHECK(succeeded(ovspatialquery_destroy_instance(explicitFirst)));
    }

    const char sharedSpherePath[] = "/World/SharedSphere";
    double     movedMatrix[16]    = {
        2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 62.0, 0.0, 0.0, 1.0,
    };
    double movedBounds[6] = {61.0, -1.0, -1.0, 63.0, 1.0, 1.0};
    CHECK(writeAttribute(stage, dictionary, sharedSpherePath, sizeof(sharedSpherePath) - 1u, "omni:fabric:worldMatrix",
                         4u, movedMatrix, DLDataType{kDLFloat, 64u, 16u}, 1u, OVSTAGE_SEMANTIC_MATRIX));
    CHECK(writeBounds(stage, dictionary, sharedSpherePath, sizeof(sharedSpherePath) - 1u, 4u, movedBounds));
    const ovstage_write_floor_desc_t floorFour{4u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
    CHECK(waitStageOperation(stage, ovstage_advance_write_floor(stage, &floorFour)));
    operation = ovspatialquery_enqueue_ovstage_population_update(instance, sharedPopulation, 3, 4);
    CHECK(finishPopulationUpdate(instance, operation, 4, 2u, 0u));
    IdentityRows sharedAtFour;
    CHECK(publishedIdentities(instance, sharedPopulation, 4, sharedAtFour) == OVSPATIALQUERY_API_SUCCESS);
    CHECK(sharedAtFour.size() == 2u);
    for (const auto& oldRow : sharedAtThree)
    {
        const auto current = std::find_if(sharedAtFour.begin(), sharedAtFour.end(),
                                          [&oldRow](const auto& row) { return row.objectId == oldRow.objectId; });
        CHECK(current != sharedAtFour.end());
        CHECK(current->geometry.type == oldRow.geometry.type);
        CHECK(current->geometry.value == oldRow.geometry.value);
    }
    IdentityRows retainedAtThree;
    CHECK(publishedIdentities(instance, sharedPopulation, 3, retainedAtThree) == OVSPATIALQUERY_API_SUCCESS);
    CHECK(retainedAtThree.size() == sharedAtThree.size());

    operation = ovspatialquery_enqueue_ovstage_population_update(instance, typedPopulation, 3, 4);
    CHECK(finishPopulationUpdate(instance, operation, 4, 5u, 0u));
    IdentityRows typedAtFour;
    CHECK(publishedIdentities(instance, typedPopulation, 4, typedAtFour) == OVSPATIALQUERY_API_SUCCESS);
    CHECK(typedAtFour.size() == typedRows.size());
    for (const auto& oldRow : typedRows)
    {
        const auto current = std::find_if(typedAtFour.begin(), typedAtFour.end(),
                                          [&oldRow](const auto& row) { return row.objectId == oldRow.objectId; });
        CHECK(current != typedAtFour.end());
        CHECK(current->geometry.type == oldRow.geometry.type);
        CHECK(current->geometry.value == oldRow.geometry.value);
    }

    const ovx_string_t topologyIndependentConvexPaths[] = {
        exactPathValues[12],
        exactPathValues[14],
        exactPathValues[15],
        exactPathValues[16],
    };
    ovstage_predicate_t topologyIndependentPredicates[2]             = {geometryPredicates[0], geometryPredicates[1]};
    topologyIndependentPredicates[0].values                          = &geometryTypes[1];
    topologyIndependentPredicates[0].value_count                     = 1u;
    topologyIndependentPredicates[1].values                          = topologyIndependentConvexPaths;
    topologyIndependentPredicates[1].value_count                     = 4u;
    ovspatialquery_population_handle_t topologyIndependentPopulation = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(createOvstagePopulation(instance, stage, ovstage_filter_t{topologyIndependentPredicates, 2u}, 2u,
                                  topologyIndependentPopulation) == 0);
    operation = ovspatialquery_enqueue_ovstage_population_set(instance, topologyIndependentPopulation, 4);
    CHECK(finishPopulationUpdate(instance, operation, 4, 4u, 0u));
    IdentityRows topologyIndependentRows;
    CHECK(publishedIdentities(instance, topologyIndependentPopulation, 4, topologyIndependentRows) ==
          OVSPATIALQUERY_API_SUCCESS);
    CHECK(topologyIndependentRows.size() == 4u);
    for (const auto& row : topologyIndependentRows)
    {
        CHECK(row.geometry.type == OVSPATIALQUERY_GEOMETRY_CONVEX);
        CHECK(row.geometry.value == typedConvex->geometry.value);
    }
    CHECK(succeeded(ovspatialquery_destroy_population(instance, topologyIndependentPopulation)));

    double editedRadius    = 0.75;
    double editedBounds[6] = {60.5, -1.5, -1.5, 63.5, 1.5, 1.5};
    CHECK(writeAttribute(stage, dictionary, sharedSpherePath, sizeof(sharedSpherePath) - 1u, "radius", 5u,
                         &editedRadius, DLDataType{kDLFloat, 64u, 1u}));
    CHECK(writeBounds(stage, dictionary, sharedSpherePath, sizeof(sharedSpherePath) - 1u, 5u, editedBounds));
    // A mesh whose authored content still matches the head keeps its preparation and
    // is not read again, and a mesh whose points alone moved keeps its topology and
    // takes only the new points, so the rejected source-face read below needs changed
    // topology at this ordinal: the precise mesh's quad is rewritten from another
    // corner (the same polygon, a different index array) and one point moves.
    float   precisePoints[]  = {2.f,  -1.f, -1.f, 2.f,  -1.f, 1.f, 2.f, 1.f, 1.f, 2.f, 1.f,
                                -1.f, 1.f,  -1.f, -1.f, 1.5f, 0.f, 1.f, 1.f, 1.f, -1.f};
    int32_t rotatedIndices[] = {4, 5, 6, 1, 2, 3, 0};
    CHECK(writeFloatPointArray(stage, dictionary, meshPath, sizeof(meshPath) - 1u, 5u, precisePoints, 7u));
    CHECK(writeAttribute(stage, dictionary, meshPath, sizeof(meshPath) - 1u, "faceVertexIndices", 5u, rotatedIndices,
                         DLDataType{kDLInt, 32u, 1u}, 7u, OVSTAGE_SEMANTIC_NONE, true));
    const ovstage_write_floor_desc_t floorFive{5u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
    CHECK(waitStageOperation(stage, ovstage_advance_write_floor(stage, &floorFive)));
    operation = ovspatialquery_enqueue_ovstage_population_update(instance, sharedPopulation, 4, 5);
    CHECK(finishPopulationUpdate(instance, operation, 5, 2u, 0u));
    IdentityRows sharedAtFive;
    CHECK(publishedIdentities(instance, sharedPopulation, 5, sharedAtFive) == OVSPATIALQUERY_API_SUCCESS);
    const auto edited = std::find_if(sharedAtFive.begin(), sharedAtFive.end(),
                                     [expected = exactPaths[21]](const auto& row) { return row.objectId == expected; });
    const auto prior  = std::find_if(sharedAtFour.begin(), sharedAtFour.end(),
                                     [expected = exactPaths[21]](const auto& row) { return row.objectId == expected; });
    CHECK(edited != sharedAtFive.end());
    CHECK(prior != sharedAtFour.end());
    CHECK(edited->geometry.type == prior->geometry.type);
    CHECK(edited->geometry.value != prior->geometry.value);
    const uint64_t editedSphereGeometryValue = edited->geometry.value;

    const auto sameRegistryState =
        [](const GeometryIdentityRegistryInspection& first, const GeometryIdentityRegistryInspection& second)
    {
        return first.definitionCount == second.definitionCount && first.nextValue == second.nextValue &&
               first.residentByteCount == second.residentByteCount;
    };

    CHECK(ovspatialquery::detail::ovstageSourceFaceFitsUint32(static_cast<uint64_t>(UINT32_MAX)));
    CHECK(!ovspatialquery::detail::ovstageSourceFaceFitsUint32(static_cast<uint64_t>(UINT32_MAX) + UINT64_C(1)));

    ovspatialquery_population_info_t typedInfoBeforeRejectedUpdate{};
    CHECK(succeeded(ovspatialquery_get_population_info(instance, typedPopulation, &typedInfoBeforeRejectedUpdate)));
    CHECK(typedInfoBeforeRejectedUpdate.head_ordinal == 4);
    CHECK(typedInfoBeforeRejectedUpdate.object_count == 5u);
    CHECK(typedInfoBeforeRejectedUpdate.update_pending == 0u);
    ovspatialquery_raycast_desc_t typedParticipationRay{};
    typedParticipationRay.common.target         = populationTarget(typedPopulation);
    typedParticipationRay.common.precision      = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    typedParticipationRay.common.query_bits     = {{UINT64_C(1) << 62u, 0u}};
    typedParticipationRay.ray.unit_direction[0] = 1.0f;
    typedParticipationRay.ray.max_distance      = 20.0;
    ovspatialquery_raycast_single_result_t typedHitBeforeRejection{};
    CHECK(succeeded(ovspatialquery_raycast_closest_sync(instance, &typedParticipationRay, &typedHitBeforeRejection)));
    CHECK(typedHitBeforeRejection.source_ordinal == 4);
    CHECK(typedHitBeforeRejection.has_hit == 1u);
    ovspatialquery::detail::rejectNextUnrepresentableOvstageSourceFaceFixture();
    const ovspatialquery_enqueue_result_t excludedSourceFaceUpdate =
        ovspatialquery_enqueue_ovstage_population_update(instance, typedPopulation, 4, 5);
    CHECK(finishWithExclusion(instance, excludedSourceFaceUpdate, 5, 4u, exactPaths[1],
                              OVSPATIALQUERY_OVSTAGE_EXCLUSION_MALFORMED_GEOMETRY));
    ovspatialquery_population_info_t typedInfoAfterExcludedUpdate{};
    CHECK(succeeded(ovspatialquery_get_population_info(instance, typedPopulation, &typedInfoAfterExcludedUpdate)));
    CHECK(typedInfoAfterExcludedUpdate.head_ordinal == 5);
    CHECK(typedInfoAfterExcludedUpdate.object_count == 4u);
    CHECK(typedInfoAfterExcludedUpdate.update_pending == 0u);
    IdentityRows typedAtThreeAfterExclusion;
    IdentityRows typedAtFourAfterExclusion;
    IdentityRows typedAtFiveAfterExclusion;
    CHECK(publishedIdentities(instance, typedPopulation, 3, typedAtThreeAfterExclusion) == OVSPATIALQUERY_API_SUCCESS);
    CHECK(publishedIdentities(instance, typedPopulation, 4, typedAtFourAfterExclusion) == OVSPATIALQUERY_API_SUCCESS);
    CHECK(publishedIdentities(instance, typedPopulation, 5, typedAtFiveAfterExclusion) == OVSPATIALQUERY_API_SUCCESS);
    CHECK(typedAtThreeAfterExclusion.size() == typedRows.size());
    CHECK(typedAtFourAfterExclusion.size() == typedAtFour.size());
    CHECK(std::none_of(typedAtFiveAfterExclusion.begin(), typedAtFiveAfterExclusion.end(),
                       [&](const IdentityRow& row) { return row.objectId == exactPaths[1]; }));
    ovspatialquery_raycast_single_result_t typedHitAfterExclusion{};
    CHECK(succeeded(ovspatialquery_raycast_closest_sync(instance, &typedParticipationRay, &typedHitAfterExclusion)));
    CHECK(typedHitAfterExclusion.source_ordinal == 5);
    CHECK(typedHitAfterExclusion.has_hit == 1u);
    CHECK(typedHitAfterExclusion.hit.data.bounds_candidate.identity.object_id.value ==
          typedHitBeforeRejection.hit.data.bounds_candidate.identity.object_id.value);

    const ovx_string_t  noExactGeometryPath{"/World/NoExactGeometryMatch", sizeof("/World/NoExactGeometryMatch") - 1u};
    ovstage_predicate_t zeroMatchPredicates[2]             = {geometryPredicates[0], geometryPredicates[1]};
    zeroMatchPredicates[0].values                          = &geometryTypes[0];
    zeroMatchPredicates[0].value_count                     = 1u;
    zeroMatchPredicates[1].values                          = &noExactGeometryPath;
    zeroMatchPredicates[1].value_count                     = 1u;
    ovspatialquery_population_handle_t zeroMatchPopulation = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(createOvstagePopulation(instance, stage, ovstage_filter_t{zeroMatchPredicates, 2u}, 3u,
                                  zeroMatchPopulation) == 0);
    GeometryIdentityRegistryInspection registryBeforeZeroSet{};
    CHECK(inspectGeometryIdentityRegistry(instance, registryBeforeZeroSet) == OVSPATIALQUERY_API_SUCCESS);
    operation = ovspatialquery_enqueue_ovstage_population_set(instance, zeroMatchPopulation, 5);
    CHECK(finishPopulationUpdate(instance, operation, 5, 0u, 0u));
    GeometryIdentityRegistryInspection registryAfterZeroSet{};
    CHECK(inspectGeometryIdentityRegistry(instance, registryAfterZeroSet) == OVSPATIALQUERY_API_SUCCESS);
    CHECK(sameRegistryState(registryBeforeZeroSet, registryAfterZeroSet));

    ovstage_predicate_t allExcludedPredicates[2]             = {geometryPredicates[0], geometryPredicates[1]};
    allExcludedPredicates[0].values                          = &geometryTypes[0];
    allExcludedPredicates[0].value_count                     = 1u;
    allExcludedPredicates[1].values                          = &exactPathValues[0];
    allExcludedPredicates[1].value_count                     = 1u;
    ovspatialquery_population_handle_t allExcludedPopulation = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(createOvstagePopulation(instance, stage, ovstage_filter_t{allExcludedPredicates, 2u}, 3u,
                                  allExcludedPopulation) == 0);
    operation = ovspatialquery_enqueue_ovstage_population_set(instance, allExcludedPopulation, 5);
    CHECK(finishPopulationUpdate(instance, operation, 5, 1u, 0u));
    IdentityRows allExcludedAtFive;
    CHECK(publishedIdentities(instance, allExcludedPopulation, 5, allExcludedAtFive) == OVSPATIALQUERY_API_SUCCESS);
    CHECK(allExcludedAtFive.size() == 1u);
    CHECK(allExcludedAtFive[0].objectId == exactPaths[0]);

    ovspatialquery_raycast_desc_t emptyPublicationRay{};
    emptyPublicationRay.common.target         = populationTarget(allExcludedPopulation);
    emptyPublicationRay.common.precision      = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    emptyPublicationRay.ray.unit_direction[0] = 1.0f;
    emptyPublicationRay.ray.max_distance      = 20.0;
    ovspatialquery_raycast_single_result_t beforeExclusionHit{};
    CHECK(succeeded(ovspatialquery_raycast_closest_sync(instance, &emptyPublicationRay, &beforeExclusionHit)));
    CHECK(beforeExclusionHit.source_ordinal == 5);
    CHECK(beforeExclusionHit.has_hit == 1u);
    CHECK(beforeExclusionHit.hit.data.bounds_candidate.identity.object_id.value == exactPaths[0]);

    const auto rejectedTypedPopulation = [&](size_t pathIndex, ovspatialquery_ordinal_t rejectedOrdinal,
                                             ovspatialquery_api_status_t expected) -> int
    {
        ovstage_predicate_t rejectedPredicates[2]   = {geometryPredicates[0], geometryPredicates[1]};
        rejectedPredicates[0].values                = &geometryTypes[1];
        rejectedPredicates[0].value_count           = 1u;
        rejectedPredicates[1].values                = &exactPathValues[pathIndex];
        rejectedPredicates[1].value_count           = 1u;
        ovspatialquery_population_handle_t rejected = OVSPATIALQUERY_INVALID_POPULATION;
        CHECK(createOvstagePopulation(instance, stage, ovstage_filter_t{rejectedPredicates, 2u}, 3u, rejected) == 0);
        GeometryIdentityRegistryInspection registryBefore{};
        CHECK(inspectGeometryIdentityRegistry(instance, registryBefore) == OVSPATIALQUERY_API_SUCCESS);
        ovspatialquery_population_info_t stableInfoBefore{};
        CHECK(succeeded(ovspatialquery_get_population_info(instance, typedPopulation, &stableInfoBefore)));
        ovspatialquery_raycast_single_result_t stableHitBefore{};
        CHECK(succeeded(ovspatialquery_raycast_closest_sync(instance, &typedParticipationRay, &stableHitBefore)));
        uint64_t hostBefore = 0u;
        CHECK(readHostBytes(instance, hostBefore));
        const ovspatialquery_enqueue_result_t rejectedSet =
            ovspatialquery_enqueue_ovstage_population_set(instance, rejected, rejectedOrdinal);
        CHECK(rejectedSet.status == expected);
        CHECK(rejectedSet.op_id == OVSPATIALQUERY_INVALID_OP);
        ovspatialquery_population_info_t rejectedInfo{};
        CHECK(succeeded(ovspatialquery_get_population_info(instance, rejected, &rejectedInfo)));
        CHECK(rejectedInfo.object_count == 0u);
        CHECK(rejectedInfo.ready == 0u);
        IdentityRows absentRejectedHistory;
        CHECK(publishedIdentities(instance, rejected, rejectedOrdinal, absentRejectedHistory) ==
              OVSPATIALQUERY_API_ORDINAL_NOT_RETAINED);
        GeometryIdentityRegistryInspection registryAfter{};
        CHECK(inspectGeometryIdentityRegistry(instance, registryAfter) == OVSPATIALQUERY_API_SUCCESS);
        CHECK(sameRegistryState(registryBefore, registryAfter));
        ovspatialquery_population_info_t stableInfoAfter{};
        CHECK(succeeded(ovspatialquery_get_population_info(instance, typedPopulation, &stableInfoAfter)));
        CHECK(stableInfoBefore.ready == stableInfoAfter.ready);
        CHECK(stableInfoBefore.head_ordinal == stableInfoAfter.head_ordinal);
        CHECK(stableInfoBefore.object_count == stableInfoAfter.object_count);
        CHECK(stableInfoBefore.update_pending == stableInfoAfter.update_pending);
        IdentityRows retainedStableAtThree;
        IdentityRows retainedStableAtFour;
        CHECK(publishedIdentities(instance, typedPopulation, 3, retainedStableAtThree) == OVSPATIALQUERY_API_SUCCESS);
        CHECK(publishedIdentities(instance, typedPopulation, 4, retainedStableAtFour) == OVSPATIALQUERY_API_SUCCESS);
        CHECK(retainedStableAtThree.size() == typedRows.size());
        CHECK(retainedStableAtFour.size() == typedAtFour.size());
        ovspatialquery_raycast_single_result_t stableHitAfter{};
        CHECK(succeeded(ovspatialquery_raycast_closest_sync(instance, &typedParticipationRay, &stableHitAfter)));
        CHECK(stableHitBefore.source_ordinal == stableHitAfter.source_ordinal);
        CHECK(stableHitBefore.has_hit == stableHitAfter.has_hit);
        CHECK(stableHitBefore.hit.data.bounds_candidate.identity.object_id.value ==
              stableHitAfter.hit.data.bounds_candidate.identity.object_id.value);
        uint64_t hostAfter = 0u;
        CHECK(readHostBytes(instance, hostAfter));
        CHECK(hostBefore == hostAfter);
        CHECK(succeeded(ovspatialquery_destroy_population(instance, rejected)));
        return 0;
    };
    const auto excludedTypedPopulation = [&](size_t pathIndex, ovspatialquery_ordinal_t ordinal,
                                             ovspatialquery_ovstage_exclusion_reason_t reason) -> int
    {
        const ovx_string_t  paths[]                   = {exactPathValues[pathIndex], exactPathValues[21]};
        ovstage_predicate_t predicate                 = geometryPredicates[1];
        predicate.values                              = paths;
        predicate.value_count                         = 2u;
        ovspatialquery_population_handle_t population = OVSPATIALQUERY_INVALID_POPULATION;
        CHECK(createOvstagePopulation(instance, stage, ovstage_filter_t{&predicate, 1u}, 3u, population) == 0);
        CHECK(finishWithExclusion(instance,
                                  ovspatialquery_enqueue_ovstage_population_set(instance, population, ordinal), ordinal,
                                  1u, exactPaths[pathIndex], reason));
        IdentityRows rows;
        CHECK(publishedIdentities(instance, population, ordinal, rows) == OVSPATIALQUERY_API_SUCCESS);
        CHECK(rows.size() == 1u);
        CHECK(rows[0].objectId == exactPaths[21]);
        CHECK(succeeded(ovspatialquery_destroy_population(instance, population)));
        return 0;
    };
    CHECK(excludedTypedPopulation(22u, 5, OVSPATIALQUERY_OVSTAGE_EXCLUSION_MALFORMED_GEOMETRY) == 0);

    ovstage_predicate_t concavePredicates[2]                     = {geometryPredicates[0], geometryPredicates[1]};
    concavePredicates[0].values                                  = &geometryTypes[1];
    concavePredicates[0].value_count                             = 1u;
    concavePredicates[1].values                                  = &exactPathValues[13u];
    concavePredicates[1].value_count                             = 1u;
    ovspatialquery_population_handle_t preparedConcavePopulation = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(createOvstagePopulation(instance, stage, ovstage_filter_t{concavePredicates, 2u}, 3u,
                                  preparedConcavePopulation) == 0);
    GeometryIdentityRegistryInspection registryBeforePreparedConvexes{};
    CHECK(inspectGeometryIdentityRegistry(instance, registryBeforePreparedConvexes) == OVSPATIALQUERY_API_SUCCESS);
    const ovspatialquery_enqueue_result_t preparedConcaveSet =
        ovspatialquery_enqueue_ovstage_population_set(instance, preparedConcavePopulation, 5);
    CHECK(finishPopulationUpdate(instance, preparedConcaveSet, 5, 1u, 0u));
    IdentityRows preparedConcaveRows;
    CHECK(publishedIdentities(instance, preparedConcavePopulation, 5, preparedConcaveRows) ==
          OVSPATIALQUERY_API_SUCCESS);
    CHECK(preparedConcaveRows.size() == 1u);
    CHECK(preparedConcaveRows[0].objectId == exactPaths[13u]);
    CHECK(preparedConcaveRows[0].geometry.type == OVSPATIALQUERY_GEOMETRY_CONVEX);
    CHECK(preparedConcaveRows[0].geometry.value != typedConvex->geometry.value);
    // The concave mesh is prepared as its hull, whose -x face at x 23 is intact.
    ovspatialquery_raycast_desc_t concaveRay{};
    concaveRay.common.target         = populationTarget(preparedConcavePopulation);
    concaveRay.common.precision      = OVSPATIALQUERY_QUERY_EXACT;
    concaveRay.ray.origin[0]         = 20.0;
    concaveRay.ray.unit_direction[0] = 1.0f;
    concaveRay.ray.max_distance      = 10.0;
    ovspatialquery_raycast_single_result_t concaveHit{};
    CHECK(succeeded(ovspatialquery_raycast_closest_sync(instance, &concaveRay, &concaveHit)));
    CHECK(concaveHit.source_ordinal == 5);
    CHECK(concaveHit.has_hit == 1u);
    CHECK(std::abs(concaveHit.hit.data.exact.distance - 3.0) < 1.0e-12);
    CHECK(concaveHit.hit.data.exact.feature.kind == OVSPATIALQUERY_GEOMETRY_FEATURE_NONE);
    ovspatialquery_sphere_sweep_desc_t concaveSweep{};
    concaveSweep.common                  = concaveRay.common;
    concaveSweep.sweep.center[0]         = 20.0;
    concaveSweep.sweep.radius            = 0.25;
    concaveSweep.sweep.unit_direction[0] = 1.0f;
    concaveSweep.sweep.max_distance      = 10.0;
    ovspatialquery_sphere_sweep_closest_result_t concaveSweepHit{};
    CHECK(succeeded(ovspatialquery_sweep_sphere_closest_sync(instance, &concaveSweep, &concaveSweepHit)));
    CHECK(concaveSweepHit.has_hit == 1u);
    CHECK(std::abs(concaveSweepHit.hit.data.exact.distance - 2.75) < 1.0e-12);
    CHECK(concaveSweepHit.hit.data.exact.feature.kind == OVSPATIALQUERY_GEOMETRY_FEATURE_NONE);
    for (const ovspatialquery_overlap_shape_t& shape :
         {sphereShape(24.0, 0.1), aabbShape(23.9, 24.1, 0.1), boxShape(24.0, 0.1, 0.1)})
    {
        CHECK(overlapReports(instance, preparedConcavePopulation, shape, exactPaths[13u]));
    }
    ovstage_predicate_t pointOnlyPredicates[2]             = {geometryPredicates[0], geometryPredicates[1]};
    pointOnlyPredicates[0].values                          = &geometryTypes[1];
    pointOnlyPredicates[0].value_count                     = 1u;
    pointOnlyPredicates[1].values                          = &exactPathValues[25u];
    pointOnlyPredicates[1].value_count                     = 1u;
    ovspatialquery_population_handle_t pointOnlyPopulation = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(createOvstagePopulation(instance, stage, ovstage_filter_t{pointOnlyPredicates, 2u}, 2u,
                                  pointOnlyPopulation) == 0);
    operation = ovspatialquery_enqueue_ovstage_population_set(instance, pointOnlyPopulation, 5);
    CHECK(finishPopulationUpdate(instance, operation, 5, 1u, 0u));
    IdentityRows pointOnlyRows;
    CHECK(publishedIdentities(instance, pointOnlyPopulation, 5, pointOnlyRows) == OVSPATIALQUERY_API_SUCCESS);
    CHECK(pointOnlyRows.size() == 1u);
    CHECK(pointOnlyRows[0].objectId == exactPaths[25u]);
    CHECK(pointOnlyRows[0].geometry.type == OVSPATIALQUERY_GEOMETRY_CONVEX);
    CHECK(succeeded(ovspatialquery_destroy_population(instance, pointOnlyPopulation)));
    GeometryIdentityRegistryInspection registryAfterPreparedConvexes{};
    CHECK(inspectGeometryIdentityRegistry(instance, registryAfterPreparedConvexes) == OVSPATIALQUERY_API_SUCCESS);
    CHECK(registryAfterPreparedConvexes.definitionCount == registryBeforePreparedConvexes.definitionCount + 2u);
    CHECK(failNextConvexHullPreparation(instance, ConvexHullPreparationOutcome::resourceExhausted) == 0);
    CHECK(rejectedTypedPopulation(13u, 5, OVSPATIALQUERY_API_HOST_BUDGET) == 0);
    CHECK(failNextConvexHullPreparation(instance, ConvexHullPreparationOutcome::outOfMemory) == 0);
    CHECK(rejectedTypedPopulation(13u, 5, OVSPATIALQUERY_API_OUT_OF_MEMORY) == 0);
    CHECK(failNextConvexHullPreparation(instance, ConvexHullPreparationOutcome::internalFailure) == 0);
    CHECK(rejectedTypedPopulation(13u, 5, OVSPATIALQUERY_API_INTERNAL) == 0);
    CHECK(ovspatialquery::detail::failNextTypedGeometryAggregateStagingForTest(instance) == OVSPATIALQUERY_API_SUCCESS);
    CHECK(rejectedTypedPopulation(13u, 5, OVSPATIALQUERY_API_HOST_BUDGET) == 0);
    // Points with no hull of their own, such as nearly coplanar ones under PhysX, exclude only
    // their prim (ADR-0076). This runtime's hull is exact, so the hook stands in for PhysX.
    CHECK(failNextConvexHullPreparation(instance, ConvexHullPreparationOutcome::noEnclosingHull) == 0);
    CHECK(excludedTypedPopulation(13u, 5, OVSPATIALQUERY_OVSTAGE_EXCLUSION_MALFORMED_GEOMETRY) == 0);
    CHECK(excludedTypedPopulation(23u, 5, OVSPATIALQUERY_OVSTAGE_EXCLUSION_MALFORMED_GEOMETRY) == 0);
    CHECK(excludedTypedPopulation(22u, 5, OVSPATIALQUERY_OVSTAGE_EXCLUSION_MALFORMED_GEOMETRY) == 0);

    double missingBounds[6] = {
        absentBoundsValue(true),  absentBoundsValue(true),  absentBoundsValue(true),
        absentBoundsValue(false), absentBoundsValue(false), absentBoundsValue(false),
    };
    for (size_t pathIndex : {0u, 1u, 13u, 14u, 17u, 22u, 23u})
    {
        CHECK(writeBounds(stage, dictionary, exactPathValues[pathIndex].ptr, exactPathValues[pathIndex].length, 6u,
                          missingBounds));
    }
    const ovstage_write_floor_desc_t floorSix{6u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
    CHECK(waitStageOperation(stage, ovstage_advance_write_floor(stage, &floorSix)));

    GeometryIdentityRegistryInspection registryBeforeEmptyUpdates{};
    CHECK(inspectGeometryIdentityRegistry(instance, registryBeforeEmptyUpdates) == OVSPATIALQUERY_API_SUCCESS);
    operation = ovspatialquery_enqueue_ovstage_population_update(instance, zeroMatchPopulation, 5, 6);
    CHECK(finishPopulationUpdate(instance, operation, 6, 0u, 0u));
    operation = ovspatialquery_enqueue_ovstage_population_update(instance, allExcludedPopulation, 5, 6);
    CHECK(finishWithExclusion(instance, operation, 6, 0u, exactPaths[0],
                              OVSPATIALQUERY_OVSTAGE_EXCLUSION_NO_SPATIAL_SOURCE));
    GeometryIdentityRegistryInspection registryAfterEmptyUpdates{};
    CHECK(inspectGeometryIdentityRegistry(instance, registryAfterEmptyUpdates) == OVSPATIALQUERY_API_SUCCESS);
    CHECK(sameRegistryState(registryBeforeEmptyUpdates, registryAfterEmptyUpdates));
    IdentityRows zeroAtFive;
    IdentityRows zeroAtSix;
    IdentityRows allExcludedAtSix;
    CHECK(publishedIdentities(instance, zeroMatchPopulation, 5, zeroAtFive) == OVSPATIALQUERY_API_SUCCESS);
    CHECK(publishedIdentities(instance, zeroMatchPopulation, 6, zeroAtSix) == OVSPATIALQUERY_API_SUCCESS);
    CHECK(publishedIdentities(instance, allExcludedPopulation, 5, allExcludedAtFive) == OVSPATIALQUERY_API_SUCCESS);
    CHECK(publishedIdentities(instance, allExcludedPopulation, 6, allExcludedAtSix) == OVSPATIALQUERY_API_SUCCESS);
    CHECK(zeroAtFive.empty());
    CHECK(zeroAtSix.empty());
    CHECK(allExcludedAtFive.size() == 1u);
    CHECK(allExcludedAtSix.empty());
    ovspatialquery_raycast_single_result_t afterExclusionHit{};
    CHECK(succeeded(ovspatialquery_raycast_closest_sync(instance, &emptyPublicationRay, &afterExclusionHit)));
    CHECK(afterExclusionHit.source_ordinal == 6);
    CHECK(afterExclusionHit.has_hit == 0u);

    for (size_t pathIndex : {22u, 17u, 23u, 1u})
    {
        CHECK(excludedTypedPopulation(pathIndex, 6, OVSPATIALQUERY_OVSTAGE_EXCLUSION_NO_SPATIAL_SOURCE) == 0);
    }

    ovspatialquery_population_handle_t excludingMissingConcavePopulation = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(createOvstagePopulation(instance, stage, ovstage_filter_t{concavePredicates, 2u}, 3u,
                                  excludingMissingConcavePopulation) == 0);
    GeometryIdentityRegistryInspection registryBeforeMissingConvex{};
    CHECK(inspectGeometryIdentityRegistry(instance, registryBeforeMissingConvex) == OVSPATIALQUERY_API_SUCCESS);
    operation = ovspatialquery_enqueue_ovstage_population_set(instance, excludingMissingConcavePopulation, 6);
    CHECK(finishWithExclusion(instance, operation, 6, 0u, exactPaths[13],
                              OVSPATIALQUERY_OVSTAGE_EXCLUSION_NO_SPATIAL_SOURCE));
    GeometryIdentityRegistryInspection registryAfterMissingConvex{};
    CHECK(inspectGeometryIdentityRegistry(instance, registryAfterMissingConvex) == OVSPATIALQUERY_API_SUCCESS);
    CHECK(sameRegistryState(registryBeforeMissingConvex, registryAfterMissingConvex));
    CHECK(succeeded(ovspatialquery_destroy_population(instance, excludingMissingConcavePopulation)));

    ovstage_predicate_t missingValidPredicates[2]             = {geometryPredicates[0], geometryPredicates[1]};
    const ovx_string_t  mixedMissingSpherePaths[]             = {exactPathValues[0], exactPathValues[21]};
    missingValidPredicates[0].values                          = &geometryTypes[0];
    missingValidPredicates[0].value_count                     = 1u;
    missingValidPredicates[1].values                          = mixedMissingSpherePaths;
    missingValidPredicates[1].value_count                     = 2u;
    ovspatialquery_population_handle_t missingValidPopulation = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(createOvstagePopulation(instance, stage, ovstage_filter_t{missingValidPredicates, 2u}, 3u,
                                  missingValidPopulation) == 0);
    operation = ovspatialquery_enqueue_ovstage_population_set(instance, missingValidPopulation, 6);
    CHECK(finishWithExclusion(instance, operation, 6, 1u, exactPaths[0],
                              OVSPATIALQUERY_OVSTAGE_EXCLUSION_NO_SPATIAL_SOURCE));
    CHECK(succeeded(ovspatialquery_destroy_population(instance, missingValidPopulation)));

    const char concavePath[]      = "/World/ConcaveTaggedConvex";
    float      farConcavePoints[] = {
        -1.0f,   -1.0f, -1.0f, 1.0f,  -1.0f, -1.0f, 1.0f, 1.0f,  -1.0f,
        -1.0f,   1.0f,  -1.0f, -1.0f, -1.0f, 1.0f,  1.0f, -1.0f, 1.0f,
        0x1p61f, // beyond the input domain, so the capture rejects it like a nonfinite point
        0.25f,   0.25f, -1.0f, 1.0f,  1.0f,
    };
    CHECK(writeFloatPointArray(stage, dictionary, concavePath, sizeof(concavePath) - 1u, 7u, farConcavePoints, 8u));
    double farConcaveBounds[6] = {23.0, -1.0, -1.0, 25.0, 1.0, 1.0};
    CHECK(writeBounds(stage, dictionary, concavePath, sizeof(concavePath) - 1u, 7u, farConcaveBounds));
    const ovstage_write_floor_desc_t floorSeven{7u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
    CHECK(waitStageOperation(stage, ovstage_advance_write_floor(stage, &floorSeven)));
    CHECK(excludedTypedPopulation(13u, 7, OVSPATIALQUERY_OVSTAGE_EXCLUSION_MALFORMED_GEOMETRY) == 0);

    float editedConcavePoints[] = {
        -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f,
        -1.0f, -1.0f, 1.0f,  1.0f, -1.0f, 1.0f,  0.5f, 0.5f, 0.5f,  -1.0f, 1.0f, 1.0f,
    };
    double restoredConcaveBounds[6] = {23.0, -1.0, -1.0, 25.0, 1.0, 1.0};
    CHECK(writeFloatPointArray(stage, dictionary, concavePath, sizeof(concavePath) - 1u, 8u, editedConcavePoints, 8u));
    CHECK(writeBounds(stage, dictionary, concavePath, sizeof(concavePath) - 1u, 8u, restoredConcaveBounds));
    const ovstage_write_floor_desc_t floorEight{8u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
    CHECK(waitStageOperation(stage, ovstage_advance_write_floor(stage, &floorEight)));
    operation = ovspatialquery_enqueue_ovstage_population_update(instance, preparedConcavePopulation, 5, 8);
    CHECK(finishPopulationUpdate(instance, operation, 8, 1u, 0u));
    IdentityRows retainedConcaveAtFive;
    IdentityRows editedConcaveAtEight;
    CHECK(publishedIdentities(instance, preparedConcavePopulation, 5, retainedConcaveAtFive) ==
          OVSPATIALQUERY_API_SUCCESS);
    CHECK(publishedIdentities(instance, preparedConcavePopulation, 8, editedConcaveAtEight) ==
          OVSPATIALQUERY_API_SUCCESS);
    CHECK(retainedConcaveAtFive.size() == 1u);
    CHECK(editedConcaveAtEight.size() == 1u);
    CHECK(retainedConcaveAtFive[0].geometry.value == preparedConcaveRows[0].geometry.value);
    CHECK(editedConcaveAtEight[0].geometry.value != retainedConcaveAtFive[0].geometry.value);
    concaveHit = {};
    CHECK(succeeded(ovspatialquery_raycast_closest_sync(instance, &concaveRay, &concaveHit)));
    CHECK(concaveHit.source_ordinal == 8);
    CHECK(concaveHit.has_hit == 1u);
    CHECK(std::abs(concaveHit.hit.data.exact.distance - 3.0) < 1.0e-12);
    CHECK(concaveHit.hit.data.exact.feature.kind == OVSPATIALQUERY_GEOMETRY_FEATURE_NONE);
    CHECK(succeeded(ovspatialquery_destroy_population(instance, preparedConcavePopulation)));
    CHECK(succeeded(ovspatialquery_destroy_population(instance, allExcludedPopulation)));
    CHECK(succeeded(ovspatialquery_destroy_population(instance, zeroMatchPopulation)));

    CHECK(succeeded(ovspatialquery_destroy_population(instance, sharedPopulation)));
    CHECK(succeeded(ovspatialquery_destroy_population(instance, typedPopulation)));

    ovstage_predicate_t recreatedPredicates[2]             = {geometryPredicates[0], geometryPredicates[1]};
    recreatedPredicates[0].values                          = &geometryTypes[0];
    recreatedPredicates[0].value_count                     = 1u;
    recreatedPredicates[1].values                          = &exactPathValues[21];
    recreatedPredicates[1].value_count                     = 1u;
    ovspatialquery_population_handle_t recreatedPopulation = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(createOvstagePopulation(instance, stage, ovstage_filter_t{recreatedPredicates, 2u}, 3u,
                                  recreatedPopulation) == 0);
    operation = ovspatialquery_enqueue_ovstage_population_set(instance, recreatedPopulation, 8);
    CHECK(finishPopulationUpdate(instance, operation, 8, 1u, 0u));
    IdentityRows recreatedRows;
    CHECK(publishedIdentities(instance, recreatedPopulation, 8, recreatedRows) == OVSPATIALQUERY_API_SUCCESS);
    CHECK(recreatedRows.size() == 1u);
    CHECK(recreatedRows[0].geometry.value == editedSphereGeometryValue);
    CHECK(succeeded(ovspatialquery_destroy_population(instance, recreatedPopulation)));

    CHECK(succeeded(ovspatialquery_destroy_view(instance, view)));
    CHECK(succeeded(ovspatialquery_destroy_population(instance, population)));
    CHECK(succeeded(ovspatialquery_destroy_instance(instance)));
    CHECK(ovstage_destroy_instance(stage) == OVSTAGE_OK);
    return 0;
}

} // namespace

int main()
{
    CHECK(testRealPreciseSource() == 0);
    return 0;
}
