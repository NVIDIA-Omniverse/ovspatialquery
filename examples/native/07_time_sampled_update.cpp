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

constexpr char kExample[]    = "07_time_sampled_update";
constexpr char kTargetPath[] = "/World/Moving";

constexpr char kScene[] = R"usda(#usda 1.0
# PRESENTATION:USDA:BEGIN
(
    defaultPrim = "World"
    upAxis = "Y"
    metersPerUnit = 1
    startTimeCode = 0
    endTimeCode = 1
    timeCodesPerSecond = 1
)
# PRESENTATION:USDA:END
def Material "MovingLook"
{
    token outputs:surface.connect = </MovingLook/P.outputs:surface>
    def Shader "P"
    {
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:diffuseColor = (0.05, 0.72, 0.95)
        float inputs:metallic = 0.18
        float inputs:roughness = 0.24
        token outputs:surface
    }
}
def Material "PathLook"
{
    token outputs:surface.connect = </PathLook/P.outputs:surface>
    def Shader "P"
    {
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:diffuseColor = (1, 0.45, 0.01)
        color3f inputs:emissiveColor = (0.25, 0.04, 0)
        token outputs:surface
    }
}
def Xform "World"
{
    # PRESENTATION:USDA:BEGIN
    def Cube "Moving" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        double size = 2
        float3[] extent = [(-1, -1, -1), (1, 1, 1)]
        double3 xformOp:translate.timeSamples = {
            0: (1, 0, 0),
            1: (4, 0, 0),
        }
        uniform token[] xformOpOrder = ["xformOp:translate"]
        rel material:binding = </MovingLook>
    }
    # PRESENTATION:USDA:END
    def Cube "MotionPath" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        double size = 2
        matrix4d xformOp:transform = ((2, 0, 0, 0), (0, 0.028, 0, 0), (0, 0, 0.028, 0), (2.5, -1.25, 0, 1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
        rel material:binding = </PathLook>
    }
    def Sphere "BeforeContact" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        double radius = 0.15
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        rel material:binding = </PathLook>
    }
    def Sphere "AfterContact" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        double radius = 0.22
        double3 xformOp:translate = (3, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        rel material:binding = </PathLook>
    }
}
def Xform "Render"
{
    def Camera "Camera"
    {
        float focalLength = 18
        double2 clippingRange = (0.01, 1000)
        matrix4d xformOp:transform = ((0.91, 0, -0.41, 0), (-0.11, 0.96, -0.25, 0), (0.39, 0.27, 0.88, 0), (10, 5.8, 17, 1))
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
        double3 xformOp:translate = (-4, 7, 8)
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
    example_runtime::require(admitted.status == OVSPATIALQUERY_API_SUCCESS, "population update admission failed");
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

ovspatialquery_raycast_single_result_t query(ovspatialquery_instance_handle_t   instance,
                                             ovspatialquery_population_handle_t population)
{
    ovspatialquery_raycast_desc_t ray{};
    ray.common.target.target.kind              = OVSPATIALQUERY_TARGET_POPULATION;
    ray.common.target.target.handle.population = population;
    ray.common.target.ordinal.kind             = OVSPATIALQUERY_ORDINAL_HEAD;
    ray.common.precision                       = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    ray.ray.origin[0]                          = -4.0;
    ray.ray.unit_direction[0]                  = 1.0f;
    ray.ray.max_distance                       = 20.0;
    ovspatialquery_raycast_single_result_t result{};
    example_runtime::require(succeeded(ovspatialquery_raycast_closest_sync(instance, &ray, &result)) &&
                                 result.has_hit == 1u,
                             "time-sampled raycast failed");
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
    example_runtime::Runtime        runtime("ovspatialquery.07_time_sampled_update.cpp", kScene);
    SpatialResources                spatial;
    std::vector<std::string>        cleanupErrors;
    runtime.openAndCompute();

    const ovx_string_t  pathValue{kTargetPath, sizeof(kTargetPath) - 1u};
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
    ovspatialquery_ovstage_population_desc_t populationDesc{};
    populationDesc.stage                            = runtime.stage();
    populationDesc.filter                           = ovstage_filter_t{&predicate, 1u};
    populationDesc.common.retained_ordinal_capacity = 2u;
    ovspatialquery_population_handle_t& population  = spatial.population;
    example_runtime::require(
        succeeded(ovspatialquery_create_ovstage_population(instance, &populationDesc, &population)),
        "create population failed");

    const ovspatialquery_population_update_result_t publicationTwo =
        finishPopulationUpdate(instance, ovspatialquery_enqueue_ovstage_population_set(instance, population, 2));
    const ovspatialquery_raycast_single_result_t before = query(instance, population);

    // PRESENTATION:API:BEGIN
    example_runtime::waitPopulationOpen(runtime.stage(), ovstage_population_apply_usd_time(runtime.stage(), 3u, 1.0));
    const ovstage_write_floor_desc_t floorThree{3u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
    example_runtime::require(
        example_runtime::waitStageOperation(runtime.stage(), ovstage_advance_write_floor(runtime.stage(), &floorThree)),
        "time-sampled write floor failed");
    const ovspatialquery_population_update_result_t publicationThree =
        finishPopulationUpdate(instance, ovspatialquery_enqueue_ovstage_population_update(instance, population, 2, 3));
    const ovspatialquery_raycast_single_result_t after = query(instance, population);
    // PRESENTATION:API:END
    example_runtime::require(before.source_ordinal == 2 && after.source_ordinal == 3 &&
                                 std::abs(before.hit.data.bounds_candidate.distance - 4.0) < 1.0e-12 &&
                                 std::abs(after.hit.data.bounds_candidate.distance - 7.0) < 1.0e-12,
                             "time-sampled source ordinals or distances are unexpected");
    const ovx_primpath_t expected = runtime.pathFromString(kTargetPath);
    const ovx_primpath_t beforePath{before.hit.data.bounds_candidate.identity.object_id.value};
    const ovx_primpath_t afterPath{after.hit.data.bounds_candidate.identity.object_id.value};
    example_runtime::require(beforePath == expected && afterPath == expected,
                             "sampled hits did not map to the target path");

    const std::vector<uint8_t> pixels         = runtime.render(options.frames, 3u);
    const std::string          nativeIdentity = example_runtime::nativeIdentityJson(runtime.stage());
    spatial.close(cleanupErrors);
    runtime.close(cleanupErrors);
    example_runtime::require(cleanupErrors.empty(), "presentation cleanup failed");
    const std::string  image = example_evidence::writePng(output / "frame.png", pixels);
    std::ostringstream json;
    json << std::setprecision(17) << "{\"cleanup\":{\"clean\":true,\"errors\":[]},\"errors\":[],"
         << "\"evidence\":" << image << ",\"example\":\"" << kExample
         << "\",\"execution\":{\"frames\":" << options.frames
         << ",\"headless\":" << (options.headless ? "true" : "false") << ",\"resolution\":[1280,720]},"
         << "\"inputs\":{\"ray\":{\"max_distance\":20.0,\"mode\":\"closest\","
         << "\"origin\":[-4.0,0.0,0.0],\"precision\":\"aabb_candidate\","
         << "\"unit_direction\":[1.0,0.0,0.0]},\"samples\":["
         << "{\"sample_time\":0.0,\"source_ordinal\":2},"
         << "{\"sample_time\":1.0,\"source_ordinal\":3}],"
         << "\"stage\":\"inline-usda\",\"target_paths\":[\"" << kTargetPath
         << "\"]},\"language\":\"cpp\",\"native\":" << nativeIdentity
         << ",\"publication\":{\"after\":{\"object_count\":" << publicationThree.object_count
         << ",\"source_ordinal\":" << publicationThree.published_ordinal
         << "},\"before\":{\"object_count\":" << publicationTwo.object_count
         << ",\"source_ordinal\":" << publicationTwo.published_ordinal
         << "}},\"result\":{\"after\":{\"distance\":" << after.hit.data.bounds_candidate.distance
         << ",\"has_hit\":true,\"object_id\":" << after.hit.data.bounds_candidate.identity.object_id.value
         << ",\"normal\":[" << after.hit.data.bounds_candidate.bounds_normal[0] << ","
         << after.hit.data.bounds_candidate.bounds_normal[1] << "," << after.hit.data.bounds_candidate.bounds_normal[2]
         << "],\"position\":[" << after.hit.data.bounds_candidate.bounds_position[0] << ","
         << after.hit.data.bounds_candidate.bounds_position[1] << ","
         << after.hit.data.bounds_candidate.bounds_position[2] << "],\"prim_path\":\"" << kTargetPath
         << "\",\"source_ordinal\":" << after.source_ordinal << ",\"truth\":\"bounds_candidate\"},"
         << "\"before\":{\"distance\":" << before.hit.data.bounds_candidate.distance
         << ",\"has_hit\":true,\"object_id\":" << before.hit.data.bounds_candidate.identity.object_id.value
         << ",\"normal\":[" << before.hit.data.bounds_candidate.bounds_normal[0] << ","
         << before.hit.data.bounds_candidate.bounds_normal[1] << ","
         << before.hit.data.bounds_candidate.bounds_normal[2] << "],\"position\":["
         << before.hit.data.bounds_candidate.bounds_position[0] << ","
         << before.hit.data.bounds_candidate.bounds_position[1] << ","
         << before.hit.data.bounds_candidate.bounds_position[2] << "],\"prim_path\":\"" << kTargetPath
         << "\",\"source_ordinal\":" << before.source_ordinal << ",\"truth\":\"bounds_candidate\""
         << "}},\"schema_version\":1}";
    example_evidence::writeText(output / "semantic.json", json.str());
    if (!options.headless)
    {
        example_runtime::presentInteractive("ovspatialquery: time-sampled update", pixels);
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
