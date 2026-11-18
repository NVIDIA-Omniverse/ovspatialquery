/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "example_runtime.h"

#include <ovspatialquery/ovspatialquery.h>
#include <ovspatialquery/ovspatialquery_ovstage.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

constexpr char kExample[]  = "06_precise_triangle";
constexpr char kMeshPath[] = "/World/MeshParent/PreciseMesh";

constexpr char kScene[] = R"usda(#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
    metersPerUnit = 1
)
def Material "Green"
{
    token outputs:surface.connect = </Green/P.outputs:surface>
    def Shader "P"
    {
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:diffuseColor = (0.04, 0.8, 0.23)
        float inputs:roughness = 0.28
        token outputs:surface
    }
}
def Material "Orange"
{
    token outputs:surface.connect = </Orange/P.outputs:surface>
    def Shader "P"
    {
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:diffuseColor = (1, 0.42, 0.01)
        color3f inputs:emissiveColor = (0.25, 0.05, 0)
        token outputs:surface
    }
}
# PRESENTATION:USDA:BEGIN
def Xform "World"
{
# PRESENTATION:USDA:END
    # PRESENTATION:USDA:BEGIN
    def Xform "MeshParent"
    {
        double3 xformOp:translate = (3, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Mesh "PreciseMesh" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        )
        {
            uniform bool doubleSided = 0
            uniform token subdivisionScheme = "none"
            int[] faceVertexCounts = [4, 3]
            int[] faceVertexIndices = [0, 1, 2, 3, 4, 5, 6]
            point3f[] points = [(2, -1, -1), (2, -1, 1), (2, 1, 1), (2, 1, -1), (1, -1, -1), (1, 0, 1), (1, 1, -1)]
            float3[] extent = [(1, -1, -1), (2, 1, 1)]
            rel material:binding = </Green>
        }
    }
    # PRESENTATION:USDA:END
    def Cube "Ray" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        double size = 2
        matrix4d xformOp:transform = ((2, 0, 0, 0), (0, 0.035, 0, 0), (0, 0, 0.035, 0), (2, 0, 0, 1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
        rel material:binding = </Orange>
    }
    def Sphere "Contact" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        double radius = 0.18
        double3 xformOp:translate = (4, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        rel material:binding = </Orange>
    }
# PRESENTATION:USDA:BEGIN
}
# PRESENTATION:USDA:END
def Xform "Render"
{
    def Camera "Camera"
    {
        float focalLength = 20
        double2 clippingRange = (0.01, 1000)
        matrix4d xformOp:transform = ((0.92, 0, -0.39, 0), (-0.1, 0.97, -0.23, 0), (0.38, 0.25, 0.89, 0), (10, 5, 15, 1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }
    def RenderProduct "CameraProduct"
    {
        int2 resolution = (1280, 720)
        rel camera = </Render/Camera>
        rel orderedVars = [<LdrColor>]
        def RenderVar "LdrColor"
        {
            string sourceName = "LdrColor"
        }
    }
    def SphereLight "Key"
    {
        float inputs:intensity = 28000
        float inputs:radius = 1.5
        double3 xformOp:translate = (-3, 7, 7)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def DomeLight "Environment"
    {
        color3f inputs:color = (0.12, 0.2, 0.36)
        float inputs:intensity = 260
    }
}
)usda";

bool succeeded(ovspatialquery_result_t result) { return result.status == OVSPATIALQUERY_API_SUCCESS; }

ovspatialquery_population_update_result_t finishPopulationUpdate(ovspatialquery_instance_handle_t instance,
                                                                 ovspatialquery_enqueue_result_t  admitted)
{
    example_runtime::require(admitted.status == OVSPATIALQUERY_API_SUCCESS &&
                                 admitted.op_id != OVSPATIALQUERY_INVALID_OP,
                             "population update admission failed");
    ovspatialquery_op_wait_result_t waited{};
    example_runtime::require(
        succeeded(ovspatialquery_wait_op(instance, admitted.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited)),
        "population update wait failed");
    ovspatialquery_population_update_result_t result{};
    example_runtime::require(succeeded(ovspatialquery_fetch_population_update(instance, admitted.op_id, &result)),
                             "population update fetch failed");
    example_runtime::require(succeeded(ovspatialquery_destroy_op(instance, &admitted.op_id)),
                             "population update destroy failed");
    return result;
}

struct SpatialResources
{
    ovspatialquery_instance_handle_t   instance   = OVSPATIALQUERY_INVALID_INSTANCE;
    ovspatialquery_population_handle_t population = OVSPATIALQUERY_INVALID_POPULATION;

    ~SpatialResources()
    {
        std::vector<std::string> ignored;
        close(ignored);
    }

    void close(std::vector<std::string>& errors)
    {
        if (population != OVSPATIALQUERY_INVALID_POPULATION)
        {
            if (!succeeded(ovspatialquery_destroy_population(instance, population)))
            {
                errors.emplace_back("destroy population failed");
            }
            population = OVSPATIALQUERY_INVALID_POPULATION;
        }
        if (instance != OVSPATIALQUERY_INVALID_INSTANCE)
        {
            if (!succeeded(ovspatialquery_destroy_instance(instance)))
            {
                errors.emplace_back("destroy instance failed");
            }
            instance = OVSPATIALQUERY_INVALID_INSTANCE;
        }
    }
};

int run(int argc, char** argv)
{
    const example_evidence::Options options = example_evidence::parseOptions(argc, argv);
    const std::filesystem::path     output  = std::filesystem::absolute(options.evidenceRoot) / kExample / "cpp";
    example_runtime::Runtime        runtime("ovspatialquery.06_precise_triangle.cpp", kScene);
    SpatialResources                spatial;
    std::vector<std::string>        cleanupErrors;
    runtime.openAndCompute();

    const ovx_string_t  pathValue{kMeshPath, sizeof(kMeshPath) - 1u};
    ovstage_predicate_t predicate{};
    predicate.attribute.string = ovx_string_t{"usd-path", sizeof("usd-path") - 1u};
    predicate.op               = OVSTAGE_FILTER_OP_IN;
    predicate.values           = &pathValue;
    predicate.value_count      = 1u;

    ovspatialquery_instance_desc_t instanceDesc{};
    instanceDesc.operation_capacity            = 8u;
    instanceDesc.max_cpu_concurrency           = 2u;
    instanceDesc.host_byte_budget              = UINT64_C(64) << 20u;
    ovspatialquery_instance_handle_t& instance = spatial.instance;
    example_runtime::require(succeeded(ovspatialquery_create_instance(&instanceDesc, &instance)),
                             "create core instance failed");

    // PRESENTATION:API:BEGIN
    ovspatialquery_population_handle_t&      population = spatial.population;
    ovspatialquery_ovstage_population_desc_t populationDesc{};
    populationDesc.stage                            = runtime.stage();
    populationDesc.filter                           = ovstage_filter_t{&predicate, 1u};
    populationDesc.common.retained_ordinal_capacity = 1u;
    example_runtime::require(
        succeeded(ovspatialquery_create_ovstage_population(instance, &populationDesc, &population)),
        "create triangle population failed");
    // PRESENTATION:API:END

    const ovspatialquery_population_update_result_t publication =
        finishPopulationUpdate(instance, ovspatialquery_enqueue_ovstage_population_set(instance, population, 2));

    // PRESENTATION:API:BEGIN
    ovspatialquery_raycast_desc_t ray{};
    ray.common.target.target.kind              = OVSPATIALQUERY_TARGET_POPULATION;
    ray.common.target.target.handle.population = population;
    ray.common.target.ordinal.kind             = OVSPATIALQUERY_ORDINAL_HEAD;
    ray.common.precision                       = OVSPATIALQUERY_QUERY_EXACT;
    ray.ray.unit_direction[0]                  = 1.0f;
    ray.ray.max_distance                       = 20.0;
    ovspatialquery_raycast_single_result_t hit{};
    example_runtime::require(succeeded(ovspatialquery_raycast_closest_sync(instance, &ray, &hit)) &&
                                 hit.has_hit == 1u && hit.source_ordinal == 2 &&
                                 hit.hit.truth == OVSPATIALQUERY_HIT_EXACT_SURFACE &&
                                 std::abs(hit.hit.data.exact.distance - 4.0) < 1.0e-12 &&
                                 hit.hit.data.exact.feature.kind == OVSPATIALQUERY_GEOMETRY_FEATURE_TRIANGLE &&
                                 hit.hit.data.exact.feature.data.triangle.source_face_index == 1u,
                             "precise triangle hit fields are unexpected");
    // PRESENTATION:API:END
    const ovx_primpath_t observed{hit.hit.data.exact.identity.object_id.value};
    example_runtime::require(observed == runtime.pathFromString(kMeshPath),
                             "precise object did not map to expected USD path");

    const std::vector<uint8_t> pixels         = runtime.render(options.frames, 2u);
    const std::string          nativeIdentity = example_runtime::nativeIdentityJson(runtime.stage());
    spatial.close(cleanupErrors);
    runtime.close(cleanupErrors);
    example_runtime::require(cleanupErrors.empty(), "presentation cleanup failed");

    const std::string  image = example_evidence::writePng(output / "frame.png", pixels);
    std::ostringstream json;
    json << std::setprecision(17) << "{"
         << "\"cleanup\":{\"clean\":true,\"errors\":[]},"
         << "\"errors\":[],"
         << "\"evidence\":" << image << ","
         << "\"example\":\"" << kExample << "\","
         << "\"execution\":{\"frames\":" << options.frames << ",\"headless\":" << (options.headless ? "true" : "false")
         << ",\"resolution\":[1280,720]},"
         << "\"inputs\":{\"ray\":{\"max_distance\":20.0,\"mode\":\"closest\","
         << "\"origin\":[0.0,0.0,0.0],\"precision\":\"exact\","
         << "\"unit_direction\":[1.0,0.0,0.0]},\"stage\":\"inline-usda\","
         << "\"target_paths\":[\"" << kMeshPath << "\"]},"
         << "\"language\":\"cpp\",\"native\":" << nativeIdentity << ","
         << "\"publication\":{\"excluded\":" << publication.excluded_object_count
         << ",\"object_count\":" << publication.object_count << ",\"source_ordinal\":" << publication.published_ordinal
         << "},"
         << "\"result\":{\"distance\":" << hit.hit.data.exact.distance << ",\"has_hit\":true,\"normal\":["
         << hit.hit.data.exact.normal[0] << "," << hit.hit.data.exact.normal[1] << "," << hit.hit.data.exact.normal[2]
         << "],\"object_id\":" << hit.hit.data.exact.identity.object_id.value << ",\"position\":["
         << hit.hit.data.exact.position[0] << "," << hit.hit.data.exact.position[1] << ","
         << hit.hit.data.exact.position[2] << "],\"prim_path\":\"" << kMeshPath
         << "\",\"source_ordinal\":" << hit.source_ordinal << ",\"truth\":\"exact_surface\",\"source_face_index\":"
         << hit.hit.data.exact.feature.data.triangle.source_face_index << ",\"source_face_corner_indices\":["
         << hit.hit.data.exact.feature.data.triangle.source_face_corner_indices[0] << ","
         << hit.hit.data.exact.feature.data.triangle.source_face_corner_indices[1] << ","
         << hit.hit.data.exact.feature.data.triangle.source_face_corner_indices[2] << "],\"barycentrics\":["
         << hit.hit.data.exact.feature.data.triangle.barycentrics[0] << ","
         << hit.hit.data.exact.feature.data.triangle.barycentrics[1] << ","
         << hit.hit.data.exact.feature.data.triangle.barycentrics[2] << "]},"
         << "\"schema_version\":1}";
    example_evidence::writeText(output / "semantic.json", json.str());
    if (!options.headless)
    {
        example_runtime::presentInteractive("ovspatialquery: precise triangle", pixels);
    }
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        return run(argc, argv);
    }
    catch (const std::exception& failure)
    {
        std::cerr << kExample << ": " << failure.what() << '\n';
        return 1;
    }
}
