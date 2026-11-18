/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuda_driver.h"
#include "example_evidence.h"
#include "example_runtime.h"

#include <ovspatialquery/ovspatialquery_ovstage.h>
#include <ovspatialquery/ovspatialquery_pairs.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

constexpr char kExample[] = "10_pair_reporter_deltas";
constexpr char kA0[]      = "/World/Sets/A/A0";
constexpr char kB0[]      = "/World/Sets/B/B0";

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
def Scope "Looks"
{
    def Material "First"
    {
        token outputs:surface.connect = </Looks/First/Preview.outputs:surface>
        def Shader "Preview"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (0.08, 0.9, 0.28)
            float inputs:roughness = 0.24
            token outputs:surface
        }
    }
    def Material "Second"
    {
        token outputs:surface.connect = </Looks/Second/Preview.outputs:surface>
        def Shader "Preview"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (0.9, 0.15, 0.82)
            float inputs:roughness = 0.24
            token outputs:surface
        }
    }
    def Material "Baseline"
    {
        token outputs:surface.connect = </Looks/Baseline/Preview.outputs:surface>
        def Shader "Preview"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (0.25, 0.45, 1)
            float inputs:opacity = 0.24
            token outputs:surface
        }
    }
    def Material "Future"
    {
        token outputs:surface.connect = </Looks/Future/Preview.outputs:surface>
        def Shader "Preview"
        {
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (0.9, 0.15, 0.82)
            float inputs:opacity = 0.24
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
                rel material:binding = </Looks/First>
            }
        }
        def Xform "B"
        {
            def Cube "B0" (prepend apiSchemas = ["MaterialBindingAPI"])
            {
                double size = 2
                float3[] extent = [(-1, -1, -1), (1, 1, 1)]
                double3 xformOp:translate.timeSamples = {
                    0: (-0.35, 0, 0),
                    1: (4.1, 0, 0)
                }
                uniform token[] xformOpOrder = ["xformOp:translate"]
                rel material:binding = </Looks/Second>
            }
        }
    }
    # PRESENTATION:USDA:END
    def Cube "BaselineAtT0" (prepend apiSchemas = ["MaterialBindingAPI"])
    {
        double size = 2.08
        float3[] extent = [(-1.04, -1.04, -1.04), (1.04, 1.04, 1.04)]
        double3 xformOp:translate = (-0.35, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        rel material:binding = </Looks/Baseline>
    }
    def Cube "FutureAtT1" (prepend apiSchemas = ["MaterialBindingAPI"])
    {
        double size = 2.08
        float3[] extent = [(-1.04, -1.04, -1.04), (1.04, 1.04, 1.04)]
        double3 xformOp:translate = (4.1, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        rel material:binding = </Looks/Future>
    }
    def Cube "Ground" (prepend apiSchemas = ["MaterialBindingAPI"])
    {
        double size = 2
        float3[] extent = [(-1, -1, -1), (1, 1, 1)]
        matrix4d xformOp:transform = ((5.5, 0, 0, 0), (0, 0.08, 0, 0), (0, 0, 3.5, 0), (1.2, -1.25, 0, 1))
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
            "Ovstage population update was rejected");
    ovspatialquery_op_wait_result_t waited{};
    require(succeeded(ovspatialquery_wait_op(instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited)) &&
                waited.state == OVSPATIALQUERY_OP_COMPLETED,
            "Ovstage population update did not complete");
    ovspatialquery_population_update_result_t receipt{};
    require(succeeded(ovspatialquery_fetch_population_update(instance, operation.op_id, &receipt)),
            "Ovstage population receipt fetch failed");
    require(succeeded(ovspatialquery_destroy_op(instance, &operation.op_id)),
            "Ovstage population operation destroy failed");
    return receipt;
}

ovspatialquery_target_ref_t populationTarget(ovspatialquery_population_handle_t population)
{
    ovspatialquery_target_ref_t target{};
    target.kind              = OVSPATIALQUERY_TARGET_POPULATION;
    target.handle.population = population;
    return target;
}

// The reporter's result device: CUDA device 0 with the lesson's driver and stream, or the CPU.
struct Lane
{
    std::optional<CudaDriver> cuda;
    CudaDriver::Stream        stream = nullptr;

    DLDevice device() const { return cuda ? DLDevice{kDLCUDA, 0} : DLDevice{kDLCPU, 0}; }
};

uint64_t pairTensorRowCount(const DLTensor& tensor, DLDevice device)
{
    require(tensor.device.device_type == device.device_type && tensor.device.device_id == device.device_id &&
                tensor.ndim == 2 && tensor.dtype.code == kDLUInt && tensor.dtype.bits == 64u &&
                tensor.dtype.lanes == 1u && tensor.shape != nullptr && tensor.shape[0] >= 0 && tensor.shape[1] == 2 &&
                tensor.strides != nullptr && (tensor.shape[0] == 0 || tensor.data != nullptr),
            "pair-reporter delta tensor descriptor is unexpected");
    return static_cast<uint64_t>(tensor.shape[0]);
}

uint64_t firstPairTensorValue(const DLTensor& tensor, uint64_t column)
{
    require(pairTensorRowCount(tensor, DLDevice{kDLCPU, 0}) > 0u && column < 2u,
            "pair-reporter tensor index is out of range");
    constexpr int64_t row           = 0;
    const int64_t     elementOffset = row * tensor.strides[0] + static_cast<int64_t>(column) * tensor.strides[1];
    constexpr int64_t elementSize   = static_cast<int64_t>(sizeof(uint64_t));
    require(elementOffset >= (std::numeric_limits<std::ptrdiff_t>::min)() / elementSize &&
                elementOffset <= (std::numeric_limits<std::ptrdiff_t>::max)() / elementSize &&
                tensor.byte_offset <= static_cast<uint64_t>((std::numeric_limits<std::ptrdiff_t>::max)()),
            "pair-reporter tensor byte addressing is out of range");
    const std::ptrdiff_t byteDisplacement = static_cast<std::ptrdiff_t>(elementOffset * elementSize);
    const uint8_t*       logicalBase =
        static_cast<const uint8_t*>(tensor.data) + static_cast<std::ptrdiff_t>(tensor.byte_offset);
    uint64_t value = 0u;
    std::memcpy(&value, logicalBase + byteDisplacement, sizeof(value));
    return value;
}

// Copies the first pair of a delta column to the host: read in place on the CPU, or downloaded
// on CUDA on the lane's stream after `resultReady`.
std::array<uint64_t, 2> firstPair(Lane& lane, ovspatialquery_cuda_event_t resultReady, const DLTensor& tensor)
{
    if (!lane.cuda)
    {
        return {firstPairTensorValue(tensor, 0u), firstPairTensorValue(tensor, 1u)};
    }
    require(pairTensorRowCount(tensor, lane.device()) > 0u && tensor.strides[1] == 1,
            "CUDA pair-reporter delta column is unexpected");
    std::array<uint64_t, 2> pair{};
    const CudaDriver::Event copied = lane.cuda->createEvent();
    lane.cuda->wait(lane.stream, reinterpret_cast<CudaDriver::Event>(resultReady.value));
    lane.cuda->downloadAsync(
        pair.data(),
        reinterpret_cast<CudaDriver::DevicePointer>(static_cast<const uint8_t*>(tensor.data) + tensor.byte_offset),
        sizeof(pair), lane.stream);
    lane.cuda->record(copied, lane.stream);
    lane.cuda->synchronize(copied);
    return pair;
}

// Retires a CPU operation at once, and a CUDA one after the copies queued on the lane's stream.
void destroyAfterConsumption(ovspatialquery_instance_handle_t instance, ovspatialquery_op_id_t* operation, Lane& lane)
{
    if (lane.cuda)
    {
        const CudaDriver::Event consumerDone = lane.cuda->createEvent();
        lane.cuda->record(consumerDone, lane.stream);
        lane.cuda->synchronize(consumerDone);
        ovspatialquery_cuda_event_t value{reinterpret_cast<uintptr_t>(consumerDone)};
        require(succeeded(ovspatialquery_destroy_op_after_cuda(instance, operation, &value)) &&
                    lane.cuda->forgetTransferredEvent(consumerDone),
                "CUDA pair-reporter operation requires aggregate consumer_done ordering");
        return;
    }
    require(succeeded(ovspatialquery_destroy_op(instance, operation)), "pair-reporter operation destroy failed");
}

// The lane comes first so it outlives the cleanup below, which retires a CUDA operation on it.
struct Resources
{
    Lane                                              lane;
    ovspatialquery_instance_handle_t                  instance = OVSPATIALQUERY_INVALID_INSTANCE;
    std::array<ovspatialquery_population_handle_t, 2> populations{OVSPATIALQUERY_INVALID_POPULATION,
                                                                  OVSPATIALQUERY_INVALID_POPULATION};
    ovspatialquery_pair_reporter_handle_t             reporter  = OVSPATIALQUERY_INVALID_PAIR_REPORTER;
    ovspatialquery_op_id_t                            operation = OVSPATIALQUERY_INVALID_OP;
    std::vector<std::string>                          errors;

    ~Resources() noexcept { close(); }

    void close()
    {
        if (operation != OVSPATIALQUERY_INVALID_OP)
        {
            ovspatialquery_op_wait_result_t waited{};
            bool                            destroyed =
                succeeded(ovspatialquery_wait_op(instance, operation, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited));
            if (destroyed)
            {
                try
                {
                    destroyAfterConsumption(instance, &operation, lane);
                }
                catch (const std::exception&)
                {
                    destroyed = false;
                }
            }
            if (!destroyed)
            {
                errors.emplace_back("destroy pair-reporter operation failed");
            }
            operation = OVSPATIALQUERY_INVALID_OP;
        }
        if (reporter != OVSPATIALQUERY_INVALID_PAIR_REPORTER)
        {
            if (!example_runtime::destroyWhenRetired(
                    [&] { return ovspatialquery_destroy_pair_reporter(instance, reporter); }))
            {
                errors.emplace_back("destroy pair reporter failed");
            }
            reporter = OVSPATIALQUERY_INVALID_PAIR_REPORTER;
        }
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

struct Advance
{
    ovspatialquery_ordinal_t               firstSource     = 0;
    ovspatialquery_ordinal_t               secondSource    = 0;
    ovspatialquery_pair_reporter_outcome_t outcome         = OVSPATIALQUERY_PAIR_REPORTER_ADVANCED;
    uint64_t                               firstObjectId   = 0u;
    uint64_t                               secondObjectId  = 0u;
    uint64_t                               enteredCount    = 0u;
    uint64_t                               exitedCount     = 0u;
    uint64_t                               activePairCount = 0u;
};

// PRESENTATION:API:BEGIN
// Resources owns the admitted operation until it is retired, so a failed check still retires it.
Advance advancePairReporter(Resources& resources)
{
    const ovspatialquery_instance_handle_t instance = resources.instance;
    Lane&                                  lane     = resources.lane;
    ovspatialquery_pair_reporter_advance_t advance{};
    advance.first_ordinal.kind  = OVSPATIALQUERY_ORDINAL_HEAD;
    advance.second_ordinal.kind = OVSPATIALQUERY_ORDINAL_HEAD;
    ovspatialquery_pair_reporter_reservation_t accepted{};
    ovspatialquery_enqueue_result_t            operation =
        ovspatialquery_enqueue_pair_reporter_advance(instance, resources.reporter, &advance, &accepted);
    resources.operation = operation.op_id;
    require(operation.status == OVSPATIALQUERY_API_SUCCESS && operation.op_id != OVSPATIALQUERY_INVALID_OP,
            "pair-reporter advance was rejected with status " + std::to_string(operation.status));
    ovspatialquery_op_wait_result_t waited{};
    require(succeeded(ovspatialquery_wait_op(instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited)) &&
                waited.state == OVSPATIALQUERY_OP_COMPLETED,
            "pair-reporter advance did not complete");
    ovspatialquery_pair_reporter_result_t fetched{};
    require(succeeded(ovspatialquery_fetch_pair_reporter(instance, operation.op_id, &fetched)) &&
                (fetched.result_ready.value != 0u) == lane.cuda.has_value(),
            "pair-reporter result fetch failed");
    // PRESENTATION:API:END
    Advance result{};
    result.firstSource     = fetched.first_source_ordinal;
    result.secondSource    = fetched.second_source_ordinal;
    result.outcome         = fetched.outcome;
    result.enteredCount    = pairTensorRowCount(fetched.entered_pairs, lane.device());
    result.exitedCount     = pairTensorRowCount(fetched.exited_pairs, lane.device());
    result.activePairCount = fetched.accepted_active_pair_count;
    // Copy to the host only the pair this lesson records.
    if (result.enteredCount > 0u || result.exitedCount > 0u)
    {
        const std::array<uint64_t, 2> pair = firstPair(
            lane, fetched.result_ready, result.enteredCount > 0u ? fetched.entered_pairs : fetched.exited_pairs);
        result.firstObjectId  = pair[0];
        result.secondObjectId = pair[1];
    }
    destroyAfterConsumption(instance, &resources.operation, lane);
    return result;
}

int run(int argc, char** argv)
{
    const example_evidence::Options options =
        example_evidence::parseOptions(argc, argv); // --headless --frames --evidence
    const std::filesystem::path output = std::filesystem::absolute(options.evidenceRoot) / kExample / "cpp";
    example_runtime::Runtime    runtime("ovspatialquery.10_pair_reporter_deltas.cpp", kScene);
    runtime.openAndCompute();
    Resources resources;

    const ovx_string_t pathValues[2] = {
        {kA0, sizeof(kA0) - 1u},
        {kB0, sizeof(kB0) - 1u},
    };
    ovstage_predicate_t predicates[2]{};
    for (size_t index = 0u; index < 2u; ++index)
    {
        predicates[index].attribute.string = ovx_string_t{"usd-path", sizeof("usd-path") - 1u};
        predicates[index].op               = OVSTAGE_FILTER_OP_IN;
        predicates[index].values           = &pathValues[index];
        predicates[index].value_count      = 1u;
    }

    // CUDA work needs a byte budget for the device; the CPU needs none.
    const ovspatialquery_device_budget_t cudaBudget{DLDevice{kDLCUDA, 0}, UINT64_C(64) << 20u};
    ovspatialquery_instance_desc_t       instanceDesc{};
    instanceDesc.operation_capacity  = 8u;
    instanceDesc.max_cpu_concurrency = 2u;
    instanceDesc.host_byte_budget    = UINT64_C(64) << 20u;
    instanceDesc.device_budgets      = &cudaBudget;
    instanceDesc.device_budget_count = 1u;
    require(succeeded(ovspatialquery_create_instance(&instanceDesc, &resources.instance)),
            "ovspatialquery_create_instance failed");

    std::array<ovspatialquery_population_update_result_t, 2> initialPublications{};
    for (size_t index = 0u; index < resources.populations.size(); ++index)
    {
        ovspatialquery_ovstage_population_desc_t populationDesc{};
        populationDesc.stage                            = runtime.stage();
        populationDesc.filter                           = ovstage_filter_t{&predicates[index], 1u};
        populationDesc.common.retained_ordinal_capacity = 1u;
        require(succeeded(ovspatialquery_create_ovstage_population(resources.instance, &populationDesc,
                                                                   &resources.populations[index])),
                "ovspatialquery_create_ovstage_population failed");
        initialPublications[index] = finishPopulationUpdate(
            resources.instance,
            ovspatialquery_enqueue_ovstage_population_set(resources.instance, resources.populations[index], 2));
        require(initialPublications[index].object_count == 1u && initialPublications[index].excluded_object_count == 0u,
                "Ovstage pair-reporter endpoint receipt counts are unexpected");
    }

    // PRESENTATION:API:BEGIN
    // CUDA first: ask before creating the reporter. The CPU runs it only when this library has no
    // CUDA reporter on device 0; a CUDA reporter never falls back.
    Lane& lane = resources.lane;
    if (succeeded(ovspatialquery_query_capability(resources.instance, DLDevice{kDLCUDA, 0},
                                                  OVSPATIALQUERY_CAPABILITY_PAIR_REPORTER)))
    {
        lane.cuda.emplace(CudaDriver::Ownership::tracked);
        lane.stream = lane.cuda->createStream();
    }
    ovspatialquery_pair_reporter_desc_t reporterDesc{};
    reporterDesc.first                = populationTarget(resources.populations[0]);
    reporterDesc.second               = populationTarget(resources.populations[1]);
    reporterDesc.active_pair_capacity = 4u;
    reporterDesc.delta_pair_capacity  = 4u;
    // A CUDA slot returns to the reporter only after its consumers, so the second advance takes
    // the other slot instead of waiting for the first.
    reporterDesc.result_slot_count = 2u;
    reporterDesc.contact_tolerance = 0.0;
    reporterDesc.result_device     = lane.device();
    require(succeeded(ovspatialquery_create_pair_reporter(resources.instance, &reporterDesc, &resources.reporter)),
            "ovspatialquery_create_pair_reporter failed");

    const Advance entered = advancePairReporter(resources);
    // PRESENTATION:API:END
    require(entered.outcome == OVSPATIALQUERY_PAIR_REPORTER_ADVANCED && entered.enteredCount == 1u &&
                entered.exitedCount == 0u && entered.activePairCount == 1u && entered.firstSource == 2 &&
                entered.secondSource == 2,
            "initial pair-reporter advance did not emit one ENTERED delta");

    // PRESENTATION:API:BEGIN
    example_runtime::waitPopulationOpen(runtime.stage(), ovstage_population_apply_usd_time(runtime.stage(), 3u, 1.0));
    ovstage_write_floor_desc_t floorThree{};
    floorThree.ordinal = 3u;
    floorThree.scope   = OVSTAGE_SCOPE_ALL;
    require(
        example_runtime::waitStageOperation(runtime.stage(), ovstage_advance_write_floor(runtime.stage(), &floorThree)),
        "Ovstage time update write floor failed");
    const ovspatialquery_population_update_result_t updatedPublication = finishPopulationUpdate(
        resources.instance,
        ovspatialquery_enqueue_ovstage_population_update(resources.instance, resources.populations[1], 2, 3));
    // PRESENTATION:API:END
    require(updatedPublication.object_count == 1u && updatedPublication.excluded_object_count == 0u,
            "updated Ovstage endpoint receipt counts are unexpected");

    // PRESENTATION:API:BEGIN
    const Advance exited = advancePairReporter(resources);
    // PRESENTATION:API:END
    require(exited.outcome == OVSPATIALQUERY_PAIR_REPORTER_ADVANCED && exited.enteredCount == 0u &&
                exited.exitedCount == 1u && exited.activePairCount == 0u && exited.firstSource == 2 &&
                exited.secondSource == 3,
            "second pair-reporter advance did not emit one EXITED delta");

    // An Ovstage object ID's value is the prim's path in the stage's path dictionary.
    const ovx_primpath_t paths[4]{
        ovx_primpath_t{entered.firstObjectId},
        ovx_primpath_t{entered.secondObjectId},
        ovx_primpath_t{exited.firstObjectId},
        ovx_primpath_t{exited.secondObjectId},
    };
    require(paths[0] == runtime.pathFromString(kA0) && paths[1] == runtime.pathFromString(kB0) &&
                paths[2] == paths[0] && paths[3] == paths[1],
            "pair-reporter delta IDs did not map to the expected USD paths");

    // Presentation shows t0 and the authored t1 destination together; the
    // Reporter source ordinals and deltas above remain semantic authority.
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
         << "  \"inputs\":{\"active_pair_capacity\":4,"
         << "\"delta_pair_capacity\":4,\"device\":\"" << (lane.cuda ? "cuda:0" : "cpu:0") << "\",\"first_paths\":[\""
         << kA0 << "\"],\"precision\":\"aabb_broadphase\","
         << "\"result_slot_count\":2,\"second_paths\":[\"" << kB0
         << "\"],\"stage\":\"inline-usda\",\"times\":[0.0,1.0]},\n"
         << "  \"language\":\"cpp\",\n"
         << "  \"native\":" << nativeIdentity << ",\n"
         << "  \"publication\":{\"initial\":["
         << "{\"excluded\":0,\"object_count\":1,\"name\":\"A\","
         << "\"source_ordinal\":" << initialPublications[0].published_ordinal << "},"
         << "{\"excluded\":0,\"object_count\":1,\"name\":\"B\","
         << "\"source_ordinal\":" << initialPublications[1].published_ordinal << "}],"
         << "\"updated_second\":{\"excluded\":0,\"object_count\":1,"
         << "\"name\":\"B\",\"source_ordinal\":" << updatedPublication.published_ordinal << "}},\n"
         << "  \"result\":{\"advances\":["
         << "{\"active_pair_count\":1,\"entered_count\":1,"
         << "\"exited_count\":0,\"first_object_ids\":[" << entered.firstObjectId
         << "],\"first_source_ordinal\":" << entered.firstSource << ",\"pairs\":[{"
         << "\"first_path\":\"" << kA0 << "\",\"kind\":\"entered\","
         << "\"second_path\":\"" << kB0 << "\"}],"
         << "\"second_object_ids\":[" << entered.secondObjectId << "],"
         << "\"second_source_ordinal\":" << entered.secondSource << ",\"outcome\":\"advanced\"},"
         << "{\"active_pair_count\":0,\"entered_count\":0,"
         << "\"exited_count\":1,\"first_object_ids\":[" << exited.firstObjectId
         << "],\"first_source_ordinal\":" << exited.firstSource << ",\"pairs\":[{"
         << "\"first_path\":\"" << kA0 << "\",\"kind\":\"exited\","
         << "\"second_path\":\"" << kB0 << "\"}],"
         << "\"second_object_ids\":[" << exited.secondObjectId << "],"
         << "\"second_source_ordinal\":" << exited.secondSource << ",\"outcome\":\"advanced\"}]},\n"
         << "  \"schema_version\":1\n"
         << "}\n";
    example_evidence::writeText(output / "semantic.json", json.str());
    if (!options.headless)
    {
        example_runtime::presentInteractive("ovspatialquery: pair-reporter deltas", pixels);
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
