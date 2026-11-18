/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "example_evidence.h"
#include "example_runtime.h"

#include <ovspatialquery/ovspatialquery_ovstage.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

constexpr char   kExample[]       = "02_sphere_sweep";
constexpr char   kHitPath[]       = "/World/Targets/Hit";
constexpr char   kOffPath[]       = "/World/Targets/OffPath";
constexpr double kCenter[3]       = {-4.0, 0.0, 0.0};
constexpr double kRadius          = 0.5;
constexpr float  kDirection[3]    = {1.0f, 0.0f, 0.0f};
constexpr double kMaximumDistance = 10.0;

constexpr char kScene[] = R"usda(#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
    metersPerUnit = 1
)
def Scope "Looks"
{
    def Material "Target"
    {
        token outputs:surface.connect = </Looks/Target/Preview.outputs:surface>
        def Shader "Preview"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (0.08, 0.32, 0.95)
            float inputs:roughness = 0.28
            token outputs:surface
        }
    }
    def Material "Hit"
    {
        token outputs:surface.connect = </Looks/Hit/Preview.outputs:surface>
        def Shader "Preview"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (0.05, 0.9, 0.28)
            float inputs:roughness = 0.24
            token outputs:surface
        }
    }
    def Material "Query"
    {
        token outputs:surface.connect = </Looks/Query/Preview.outputs:surface>
        def Shader "Preview"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (1, 0.58, 0.02)
            color3f inputs:emissiveColor = (0.35, 0.1, 0)
            float inputs:roughness = 0.35
            token outputs:surface
        }
    }
    def Material "Ground"
    {
        token outputs:surface.connect = </Looks/Ground/Preview.outputs:surface>
        def Shader "Preview"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (0.045, 0.06, 0.1)
            float inputs:roughness = 0.6
            token outputs:surface
        }
    }
}
def Xform "World"
{
    # PRESENTATION:USDA:BEGIN
    def Xform "Targets"
    {
        def Cube "Hit" (prepend apiSchemas = ["MaterialBindingAPI"])
        {
            double size = 2
            float3[] extent = [(-1, -1, -1), (1, 1, 1)]
            double3 xformOp:translate = (1, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
            rel material:binding = </Looks/Hit>
        }
        def Cube "OffPath" (prepend apiSchemas = ["MaterialBindingAPI"])
        {
            double size = 1.5
            float3[] extent = [(-0.75, -0.75, -0.75), (0.75, 0.75, 0.75)]
            double3 xformOp:translate = (3.2, 0, -3)
            uniform token[] xformOpOrder = ["xformOp:translate"]
            rel material:binding = </Looks/Target>
        }
    }
    # PRESENTATION:USDA:END
    def Xform "QueryVisual"
    {
        def Cube "Path" (prepend apiSchemas = ["MaterialBindingAPI"])
        {
            double size = 2
            float3[] extent = [(-1, -1, -1), (1, 1, 1)]
            matrix4d xformOp:transform = ((2, 0, 0, 0), (0, 0.035, 0, 0), (0, 0, 0.035, 0), (-2, 0, 0, 1))
            uniform token[] xformOpOrder = ["xformOp:transform"]
            rel material:binding = </Looks/Query>
        }
        def Sphere "Start" (prepend apiSchemas = ["MaterialBindingAPI"])
        {
            double radius = 0.5
            double3 xformOp:translate = (-4, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
            rel material:binding = </Looks/Query>
        }
        def Sphere "CandidateCenter" (prepend apiSchemas = ["MaterialBindingAPI"])
        {
            double radius = 0.18
            double3 xformOp:translate = (-0.5, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
            rel material:binding = </Looks/Hit>
        }
    }
    def Cube "Ground" (prepend apiSchemas = ["MaterialBindingAPI"])
    {
        double size = 2
        float3[] extent = [(-1, -1, -1), (1, 1, 1)]
        matrix4d xformOp:transform = ((6, 0, 0, 0), (0, 0.08, 0, 0), (0, 0, 4, 0), (0, -1.35, -0.8, 1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
        rel material:binding = </Looks/Ground>
    }
}
def Xform "Render"
{
    def Camera "Camera"
    {
        uniform token projection = "perspective"
        float focalLength = 18.45401
        float horizontalAperture = 27.183644
        float verticalAperture = 15.2908
        double2 clippingRange = (0.01, 1000)
        matrix4d xformOp:transform = ((0.88235294, 0, -0.47058824, 0), (-0.1448566, 0.95144452, -0.27160613, 0), (0.4477386, 0.30782029, 0.83950987, 0), (8, 5.5, 16, 1))
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
        color3f inputs:color = (1, 0.82, 0.66)
        float inputs:intensity = 32000
        float inputs:radius = 1.5
        double3 xformOp:translate = (-5, 7, 8)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def RectLight "Fill"
    {
        color3f inputs:color = (0.48, 0.68, 1)
        float inputs:intensity = 1100
        float inputs:width = 8
        float inputs:height = 8
        double3 xformOp:translate = (6, 4, 6)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def DomeLight "Environment"
    {
        color3f inputs:color = (0.16, 0.22, 0.38)
        float inputs:intensity = 220
    }
}
)usda";

bool succeeded(ovspatialquery_result_t result) { return result.status == OVSPATIALQUERY_API_SUCCESS; }

using example_runtime::require;

ovspatialquery_population_update_result_t finishPopulationUpdate(ovspatialquery_instance_handle_t instance,
                                                                 ovspatialquery_enqueue_result_t  operation)
{
    require(operation.status == OVSPATIALQUERY_API_SUCCESS && operation.op_id != OVSPATIALQUERY_INVALID_OP,
            "ovspatialquery rejected automatic population set");
    ovspatialquery_op_wait_result_t waited{};
    require(succeeded(ovspatialquery_wait_op(instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited)) &&
                waited.state == OVSPATIALQUERY_OP_COMPLETED,
            "automatic population set did not complete");
    ovspatialquery_population_update_result_t receipt{};
    require(succeeded(ovspatialquery_fetch_population_update(instance, operation.op_id, &receipt)),
            "automatic population receipt fetch failed");
    require(succeeded(ovspatialquery_destroy_op(instance, &operation.op_id)),
            "automatic population operation destroy failed");
    return receipt;
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
    example_runtime::Runtime        runtime(kExample, kScene);
    SpatialResources                spatial;
    std::vector<std::string>        cleanupErrors;
    runtime.openAndCompute();

    const ovx_string_t targetValues[] = {
        {kHitPath, sizeof(kHitPath) - 1u},
        {kOffPath, sizeof(kOffPath) - 1u},
    };
    ovstage_predicate_t predicate{};
    predicate.attribute.string = ovx_string_t{"usd-path", sizeof("usd-path") - 1u};
    predicate.op               = OVSTAGE_FILTER_OP_IN;
    predicate.values           = targetValues;
    predicate.value_count      = 2u;

    ovspatialquery_instance_desc_t instanceDesc{};
    instanceDesc.operation_capacity  = 8u;
    instanceDesc.max_cpu_concurrency = 2u;
    instanceDesc.host_byte_budget    = UINT64_C(64) << 20u;
    require(succeeded(ovspatialquery_create_instance(&instanceDesc, &spatial.instance)),
            "ovspatialquery_create_instance failed");

    ovspatialquery_ovstage_population_desc_t populationDesc{};
    populationDesc.stage                            = runtime.stage();
    populationDesc.filter                           = ovstage_filter_t{&predicate, 1u};
    populationDesc.common.retained_ordinal_capacity = 1u;
    require(succeeded(ovspatialquery_create_ovstage_population(spatial.instance, &populationDesc, &spatial.population)),
            "ovspatialquery_create_ovstage_population failed");

    const ovspatialquery_population_update_result_t publication = finishPopulationUpdate(
        spatial.instance, ovspatialquery_enqueue_ovstage_population_set(spatial.instance, spatial.population, 2));
    require(publication.object_count == 2u && publication.excluded_object_count == 0u,
            "automatic population receipt counts are unexpected");

    // PRESENTATION:API:BEGIN
    ovspatialquery_sphere_sweep_desc_t sweep{};
    sweep.common.target.target.kind              = OVSPATIALQUERY_TARGET_POPULATION;
    sweep.common.target.target.handle.population = spatial.population;
    sweep.common.target.ordinal.kind             = OVSPATIALQUERY_ORDINAL_HEAD;
    sweep.common.precision                       = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    std::copy(std::begin(kCenter), std::end(kCenter), sweep.sweep.center);
    sweep.sweep.radius = kRadius;
    std::copy(std::begin(kDirection), std::end(kDirection), sweep.sweep.unit_direction);
    sweep.sweep.max_distance = kMaximumDistance;
    ovspatialquery_sphere_sweep_closest_result_t hit{};
    require(succeeded(ovspatialquery_sweep_sphere_closest_sync(spatial.instance, &sweep, &hit)) && hit.has_hit == 1u &&
                hit.source_ordinal == 2 && hit.hit.truth == OVSPATIALQUERY_HIT_BOUNDS_CANDIDATE &&
                std::abs(hit.hit.data.bounds_candidate.distance - 3.5) < 1.0e-12 &&
                std::abs(hit.hit.data.bounds_candidate.query_position[0] + 0.5) < 1.0e-12,
            "candidate sphere sweep did not return the expected closest hit");
    // PRESENTATION:API:END
    const ovx_primpath_t observedPath{hit.hit.data.bounds_candidate.identity.object_id.value};
    require(observedPath == runtime.pathFromString(kHitPath),
            "candidate sweep hit did not map to the expected USD prim");

    const std::vector<uint8_t> pixels         = runtime.render(options.frames, 2u);
    const std::string          nativeIdentity = example_runtime::nativeIdentityJson(runtime.stage());

    spatial.close(cleanupErrors);
    runtime.close(cleanupErrors);
    require(cleanupErrors.empty(), "native example cleanup failed");

    const std::string image = example_evidence::writePng(output / "frame.png", pixels);

    std::ostringstream json;
    json << std::setprecision(17) << "{\n"
         << "  \"cleanup\":{\"clean\":true,\"errors\":[]},\n"
         << "  \"errors\":[],\n"
         << "  \"evidence\":" << image << ",\n"
         << "  \"example\":\"" << kExample << "\",\n"
         << "  \"execution\":{\"frames\":" << options.frames
         << ",\"headless\":" << (options.headless ? "true" : "false") << ",\"resolution\":[1280,720]},\n"
         << "  \"inputs\":{\"sphere_sweep\":{\"center\":[-4.0,0.0,0.0],"
         << "\"max_distance\":10.0,\"min_distance\":0.0,"
         << "\"mode\":\"closest\",\"precision\":\"aabb_candidate\","
         << "\"radius\":0.5,\"unit_direction\":[1.0,0.0,0.0]},"
         << "\"stage\":\"inline-usda\",\"target_paths\":[\"" << kHitPath << "\",\"" << kOffPath << "\"]},\n"
         << "  \"language\":\"cpp\",\n"
         << "  \"native\":" << nativeIdentity << ",\n"
         << "  \"publication\":{\"excluded\":" << publication.excluded_object_count
         << ",\"object_count\":" << publication.object_count << ",\"source_ordinal\":" << publication.published_ordinal
         << "},\n"
         << "  \"result\":{\"distance\":" << hit.hit.data.bounds_candidate.distance
         << ",\"geometry_type\":\"box\",\"has_hit\":true,\"truth\":\"bounds_candidate\""
         << ",\"normal\":[" << hit.hit.data.bounds_candidate.bounds_normal[0] << ","
         << hit.hit.data.bounds_candidate.bounds_normal[1] << "," << hit.hit.data.bounds_candidate.bounds_normal[2]
         << "],\"object_id\":" << hit.hit.data.bounds_candidate.identity.object_id.value << ",\"position\":["
         << hit.hit.data.bounds_candidate.query_position[0] << "," << hit.hit.data.bounds_candidate.query_position[1]
         << "," << hit.hit.data.bounds_candidate.query_position[2] << "],\"prim_path\":\"" << kHitPath
         << "\",\"source_ordinal\":" << hit.source_ordinal << "},\n"
         << "  \"schema_version\":1\n"
         << "}\n";
    example_evidence::writeText(output / "semantic.json", json.str());
    if (!options.headless)
    {
        example_runtime::presentInteractive("ovspatialquery: sphere sweep", pixels);
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
        std::cerr << kExample << ": " << failure.what() << "\n";
        return 1;
    }
}
