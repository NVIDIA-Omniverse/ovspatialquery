/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "example_evidence.h"
#include "example_runtime.h"

#include <ovspatialquery/ovspatialquery_ovstage.h>
#include <ovspatialquery/ovspatialquery_pairs.h>

#include <array>
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

constexpr char kExample[] = "05_two_set_pair_snapshot";
constexpr char kA0[]      = "/World/Sets/A/A0";
constexpr char kA1[]      = "/World/Sets/A/A1";
constexpr char kB0[]      = "/World/Sets/B/B0";
constexpr char kB1[]      = "/World/Sets/B/B1";

constexpr char kScene[] = R"usda(#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
    metersPerUnit = 1
)

def Scope "Looks"
{
    def Material "Pair"
    {
        token outputs:surface.connect = </Looks/Pair/Preview.outputs:surface>
        def Shader "Preview"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (0.08, 0.9, 0.28)
            color3f inputs:emissiveColor = (0.02, 0.18, 0.04)
            float inputs:roughness = 0.24
            token outputs:surface
        }
    }
    def Material "First"
    {
        token outputs:surface.connect = </Looks/First/Preview.outputs:surface>
        def Shader "Preview"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (1, 0.38, 0.04)
            float inputs:roughness = 0.3
            token outputs:surface
        }
    }
    def Material "Second"
    {
        token outputs:surface.connect = </Looks/Second/Preview.outputs:surface>
        def Shader "Preview"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (0.08, 0.3, 0.95)
            float inputs:roughness = 0.3
            token outputs:surface
        }
    }
    def Material "Ground"
    {
        token outputs:surface.connect = </Looks/Ground/Preview.outputs:surface>
        def Shader "Preview"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (0.04, 0.055, 0.09)
            float inputs:roughness = 0.65
            token outputs:surface
        }
    }
}

def Xform "World"
{
    # PRESENTATION:USDA:BEGIN
    def Xform "Sets"
    {
        def Xform "A"
        {
            def Cube "A0" (prepend apiSchemas = ["MaterialBindingAPI"])
            {
                double size = 2
                float3[] extent = [(-1, -1, -1), (1, 1, 1)]
                double3 xformOp:translate = (-1.2, 0, 0)
                uniform token[] xformOpOrder = ["xformOp:translate"]
                rel material:binding = </Looks/Pair>
            }
            def Cube "A1" (prepend apiSchemas = ["MaterialBindingAPI"])
            {
                double size = 1.6
                float3[] extent = [(-0.8, -0.8, -0.8), (0.8, 0.8, 0.8)]
                double3 xformOp:translate = (3.2, 0, 1.3)
                uniform token[] xformOpOrder = ["xformOp:translate"]
                rel material:binding = </Looks/First>
            }
        }
        def Xform "B"
        {
            def Cube "B0" (prepend apiSchemas = ["MaterialBindingAPI"])
            {
                double size = 2
                float3[] extent = [(-1, -1, -1), (1, 1, 1)]
                double3 xformOp:translate = (-0.35, 0, 0)
                uniform token[] xformOpOrder = ["xformOp:translate"]
                rel material:binding = </Looks/Pair>
            }
            def Cube "B1" (prepend apiSchemas = ["MaterialBindingAPI"])
            {
                double size = 1.5
                float3[] extent = [(-0.75, -0.75, -0.75), (0.75, 0.75, 0.75)]
                double3 xformOp:translate = (5.4, 0, -1.7)
                uniform token[] xformOpOrder = ["xformOp:translate"]
                rel material:binding = </Looks/Second>
            }
        }
    }
    # PRESENTATION:USDA:END
    def Cube "Ground" (prepend apiSchemas = ["MaterialBindingAPI"])
    {
        double size = 2
        float3[] extent = [(-1, -1, -1), (1, 1, 1)]
        matrix4d xformOp:transform = ((5.5, 0, 0, 0), (0, 0.08, 0, 0), (0, 0, 3.5, 0), (1.5, -1.25, 0, 1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
        rel material:binding = </Looks/Ground>
    }
}

def Xform "Render"
{
    def Camera "Camera"
    {
        uniform token projection = "perspective"
        float focalLength = 21
        float horizontalAperture = 27.183644
        float verticalAperture = 15.2908
        double2 clippingRange = (0.01, 1000)
        matrix4d xformOp:transform = ((0.94, 0, -0.342, 0), (-0.097, 0.959, -0.266, 0), (0.328, 0.284, 0.901, 0), (8.5, 5.2, 16.5, 1))
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
        float inputs:intensity = 30000
        float inputs:radius = 1.5
        double3 xformOp:translate = (-5, 7, 8)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def RectLight "Fill"
    {
        color3f inputs:color = (0.48, 0.68, 1)
        float inputs:intensity = 900
        float inputs:width = 8
        float inputs:height = 8
        double3 xformOp:translate = (6, 4, 6)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def DomeLight "Environment"
    {
        color3f inputs:color = (0.13, 0.19, 0.34)
        float inputs:intensity = 210
    }
}
)usda";

bool succeeded(ovspatialquery_result_t result) { return result.status == OVSPATIALQUERY_API_SUCCESS; }

using example_runtime::require;

ovspatialquery_population_update_result_t finishPopulationUpdate(ovspatialquery_instance_handle_t instance,
                                                                 ovspatialquery_enqueue_result_t  operation)
{
    require(operation.status == OVSPATIALQUERY_API_SUCCESS && operation.op_id != OVSPATIALQUERY_INVALID_OP,
            "Ovstage population set was rejected");
    ovspatialquery_op_wait_result_t waited{};
    require(succeeded(ovspatialquery_wait_op(instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited)) &&
                waited.state == OVSPATIALQUERY_OP_COMPLETED,
            "Ovstage population set did not complete");
    ovspatialquery_population_update_result_t receipt{};
    require(succeeded(ovspatialquery_fetch_population_update(instance, operation.op_id, &receipt)),
            "Ovstage population receipt fetch failed");
    require(succeeded(ovspatialquery_destroy_op(instance, &operation.op_id)),
            "Ovstage population operation destroy failed");
    return receipt;
}

ovspatialquery_query_target_t populationTarget(ovspatialquery_population_handle_t population)
{
    ovspatialquery_query_target_t target{};
    target.target.kind              = OVSPATIALQUERY_TARGET_POPULATION;
    target.target.handle.population = population;
    target.ordinal.kind             = OVSPATIALQUERY_ORDINAL_HEAD;
    return target;
}

struct Resources
{
    ovspatialquery_instance_handle_t                  instance = OVSPATIALQUERY_INVALID_INSTANCE;
    std::array<ovspatialquery_population_handle_t, 2> populations{OVSPATIALQUERY_INVALID_POPULATION,
                                                                  OVSPATIALQUERY_INVALID_POPULATION};
    std::vector<std::string>                          errors;

    ~Resources() noexcept { close(); }

    void close()
    {
        for (auto iterator = populations.rbegin(); iterator != populations.rend(); ++iterator)
        {
            if (*iterator != OVSPATIALQUERY_INVALID_POPULATION)
            {
                if (!succeeded(ovspatialquery_destroy_population(instance, *iterator)))
                {
                    errors.emplace_back("destroy population failed");
                }
                *iterator = OVSPATIALQUERY_INVALID_POPULATION;
            }
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
    const example_evidence::Options options =
        example_evidence::parseOptions(argc, argv); // --headless --frames --evidence
    const std::filesystem::path output = std::filesystem::absolute(options.evidenceRoot) / kExample / "cpp";
    example_runtime::Runtime    runtime("ovspatialquery.05_two_set_pair_snapshot.cpp", kScene);
    runtime.openAndCompute();
    Resources resources;

    const ovx_string_t setAValues[] = {
        {kA0, sizeof(kA0) - 1u},
        {kA1, sizeof(kA1) - 1u},
    };
    const ovx_string_t setBValues[] = {
        {kB0, sizeof(kB0) - 1u},
        {kB1, sizeof(kB1) - 1u},
    };
    ovstage_predicate_t predicates[2]{};
    for (ovstage_predicate_t& predicate : predicates)
    {
        predicate.attribute.string = ovx_string_t{"usd-path", sizeof("usd-path") - 1u};
        predicate.op               = OVSTAGE_FILTER_OP_IN;
        predicate.value_count      = 2u;
    }
    predicates[0].values = setAValues;
    predicates[1].values = setBValues;

    ovspatialquery_instance_desc_t instanceDesc{};
    instanceDesc.operation_capacity  = 8u;
    instanceDesc.max_cpu_concurrency = 2u;
    instanceDesc.host_byte_budget    = UINT64_C(64) << 20u;
    require(succeeded(ovspatialquery_create_instance(&instanceDesc, &resources.instance)),
            "ovspatialquery_create_instance failed");

    std::array<ovspatialquery_population_update_result_t, 2> publications{};
    for (size_t index = 0u; index < resources.populations.size(); ++index)
    {
        ovspatialquery_ovstage_population_desc_t populationDesc{};
        populationDesc.stage                            = runtime.stage();
        populationDesc.filter                           = ovstage_filter_t{&predicates[index], 1u};
        populationDesc.common.retained_ordinal_capacity = 1u;
        require(succeeded(ovspatialquery_create_ovstage_population(resources.instance, &populationDesc,
                                                                   &resources.populations[index])),
                "ovspatialquery_create_ovstage_population failed");
        publications[index] = finishPopulationUpdate(
            resources.instance,
            ovspatialquery_enqueue_ovstage_population_set(resources.instance, resources.populations[index], 2));
        require(publications[index].object_count == 2u && publications[index].excluded_object_count == 0u,
                "Ovstage pair endpoint receipt counts are unexpected");
    }

    // PRESENTATION:API:BEGIN
    ovspatialquery_pair_snapshot_desc_t pairDesc{};
    pairDesc.first             = populationTarget(resources.populations[0]);
    pairDesc.second            = populationTarget(resources.populations[1]);
    pairDesc.contact_tolerance = 0.0;
    std::array<ovspatialquery_pair_t, 8>  rows{};
    ovspatialquery_pair_buffer_t          buffer{rows.data(), rows.size()};
    ovspatialquery_pair_snapshot_result_t result{};
    require(succeeded(ovspatialquery_query_pairs_sync(resources.instance, &pairDesc, &buffer, &result)) &&
                result.stored_pair_count == 1u && result.truncated == 0u,
            "two-set pair snapshot did not return exactly one pair");
    // PRESENTATION:API:END

    const ovx_primpath_t firstPath{rows[0].first.value};
    const ovx_primpath_t secondPath{rows[0].second.value};
    require(firstPath == runtime.pathFromString(kA0) && secondPath == runtime.pathFromString(kB0),
            "pair endpoints did not map to the expected USD paths");

    std::vector<uint8_t> pixels         = runtime.render(options.frames, 2u);
    const std::string    nativeIdentity = example_runtime::nativeIdentityJson(runtime.stage());

    resources.close();
    runtime.close(resources.errors);
    require(resources.errors.empty(), "native example cleanup failed");
    const std::string image = example_evidence::writePng(output / "frame.png", pixels);

    std::ostringstream json;
    json << std::setprecision(17) << "{\n"
         << "  \"cleanup\":{\"clean\":true,\"errors\":[]},\n"
         << "  \"errors\":[],\n"
         << "  \"evidence\":" << image << ",\n"
         << "  \"example\":\"" << kExample << "\",\n"
         << "  \"execution\":{\"frames\":" << options.frames
         << ",\"headless\":" << (options.headless ? "true" : "false") << ",\"resolution\":[1280,720]},\n"
         << "  \"inputs\":{\"capacity\":8,\"contact_tolerance\":0.0,"
         << "\"first_paths\":[\"" << kA0 << "\",\"" << kA1 << "\"],"
         << "\"precision\":\"aabb_broadphase\","
         << "\"second_paths\":[\"" << kB0 << "\",\"" << kB1 << "\"],"
         << "\"stage\":\"inline-usda\"},\n"
         << "  \"language\":\"cpp\",\n"
         << "  \"native\":" << nativeIdentity << ",\n"
         << "  \"publication\":{\"sets\":["
         << "{\"excluded\":" << publications[0].excluded_object_count
         << ",\"object_count\":" << publications[0].object_count
         << ",\"name\":\"A\",\"source_ordinal\":" << publications[0].published_ordinal << "},"
         << "{\"excluded\":" << publications[1].excluded_object_count
         << ",\"object_count\":" << publications[1].object_count
         << ",\"name\":\"B\",\"source_ordinal\":" << publications[1].published_ordinal << "}]},\n"
         << "  \"result\":{\"first_source_ordinal\":" << result.first_source_ordinal
         << ",\"pair_count\":1,\"pairs\":[{\"first_object_id\":" << rows[0].first.value << ",\"first_path\":\"" << kA0
         << "\",\"second_object_id\":" << rows[0].second.value << ",\"second_path\":\"" << kB0
         << "\"}],\"second_source_ordinal\":" << result.second_source_ordinal << ",\"truncated\":false},\n"
         << "  \"schema_version\":1\n"
         << "}\n";
    example_evidence::writeText(output / "semantic.json", json.str());
    if (!options.headless)
    {
        example_runtime::presentInteractive("ovspatialquery: two-set pair snapshot", pixels);
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
    catch (const std::exception& error)
    {
        std::cerr << kExample << ": " << error.what() << "\n";
        return 1;
    }
}
