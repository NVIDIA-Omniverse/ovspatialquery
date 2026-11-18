/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuda_driver.h"
#include "example_runtime.h"

#include <ovspatialquery/ovspatialquery.h>
#include <ovspatialquery/ovspatialquery_ovstage.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

constexpr char kExample[]    = "08_reusable_cpu_batch";
constexpr char kHitPath[]    = "/World/Targets/Hit";
constexpr char kOffRayPath[] = "/World/Targets/OffRay";

constexpr char kScene[] = R"usda(#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
    metersPerUnit = 1
)
def Material "Target"
{
    token outputs:surface.connect = </Target/P.outputs:surface>
    def Shader "P"
    {
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:diffuseColor = (0.04, 0.7, 0.95)
        float inputs:roughness = 0.25
        token outputs:surface
    }
}
def Material "Ray"
{
    token outputs:surface.connect = </Ray/P.outputs:surface>
    def Shader "P"
    {
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:diffuseColor = (1, 0.42, 0.01)
        color3f inputs:emissiveColor = (0.3, 0.05, 0)
        token outputs:surface
    }
}
def Xform "World"
{
    # PRESENTATION:USDA:BEGIN
    def Xform "Targets"
    {
        def Cube "Hit" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        )
        {
            double size = 2
            float3[] extent = [(-1, -1, -1), (1, 1, 1)]
            double3 xformOp:translate = (1, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
            rel material:binding = </Target>
        }
        def Cube "OffRay" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        )
        {
            double size = 1.5
            float3[] extent = [(-0.75, -0.75, -0.75), (0.75, 0.75, 0.75)]
            double3 xformOp:translate = (3, 0, -3)
            uniform token[] xformOpOrder = ["xformOp:translate"]
            rel material:binding = </Target>
        }
    }
    # PRESENTATION:USDA:END
    def Cube "RayA" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        double size = 2
        matrix4d xformOp:transform = ((2, 0, 0, 0), (0, 0.025, 0, 0), (0, 0, 0.025, 0), (-2, 0, 0, 1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
        rel material:binding = </Ray>
    }
    def Cube "RayB" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        double size = 2
        matrix4d xformOp:transform = ((3, 0, 0, 0), (0, 0.025, 0, 0), (0, 0, 0.025, 0), (-1, 0, -3, 1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
        rel material:binding = </Ray>
    }
}
def Xform "Render"
{
    def Camera "Camera"
    {
        float focalLength = 18
        double2 clippingRange = (0.01, 1000)
        matrix4d xformOp:transform = ((0.88, 0, -0.47, 0), (-0.14, 0.95, -0.27, 0), (0.45, 0.31, 0.84, 0), (8, 5.5, 16, 1))
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

ovspatialquery_input_array_t copiedArray(void* data, DLDevice device, DLDataType dtype, int32_t rank, int64_t* shape,
                                         int64_t* strides)
{
    ovspatialquery_input_array_t result{};
    result.tensor.data    = data;
    result.tensor.device  = device;
    result.tensor.ndim    = rank;
    result.tensor.dtype   = dtype;
    result.tensor.shape   = shape;
    result.tensor.strides = strides;
    result.ownership      = OVSPATIALQUERY_INPUT_COPIED_ON_SUBMISSION;
    return result;
}

struct LaunchResult
{
    uint32_t                 index         = 0u;
    ovspatialquery_ordinal_t sourceOrdinal = 0;
    std::array<uint64_t, 2>  counts{};
    std::array<uint64_t, 2>  offsets{};
    std::vector<uint64_t>    ids;
    std::vector<std::string> paths;
};

bool tensorAbsent(const DLTensor& tensor) { return tensor.shape == nullptr; }

// The plan's device: CUDA device 0 with the lesson's driver and stream, or the CPU.
struct Lane
{
    std::optional<CudaDriver> cuda;
    CudaDriver::Stream        stream = nullptr;

    DLDevice device() const { return cuda ? DLDevice{kDLCUDA, 0} : DLDevice{kDLCPU, 0}; }
};

// Copies one result column to the host: read in place on the CPU, or downloaded on CUDA on the
// lane's stream after `resultReady`.
template <typename Element>
std::vector<Element> readTensor(Lane& lane, ovspatialquery_cuda_event_t resultReady, const DLTensor& tensor,
                                uint64_t count, uint8_t dataType, uint8_t bits)
{
    const DLDevice device = lane.device();
    example_runtime::require(tensor.device.device_type == device.device_type &&
                                 tensor.device.device_id == device.device_id && tensor.ndim == 1 &&
                                 tensor.shape != nullptr && tensor.shape[0] == static_cast<int64_t>(count) &&
                                 tensor.dtype.code == dataType && tensor.dtype.bits == bits && tensor.dtype.lanes == 1u,
                             "unexpected result tensor descriptor");
    if (count == 0u)
    {
        return {};
    }
    example_runtime::require(tensor.data != nullptr, "present nonempty result tensor has no data");
    const int64_t        stride = tensor.strides == nullptr ? 1 : tensor.strides[0];
    const uint8_t*       bytes  = static_cast<const uint8_t*>(tensor.data) + tensor.byte_offset;
    std::vector<Element> result(static_cast<size_t>(count));
    if (lane.cuda)
    {
        example_runtime::require(stride == 1, "CUDA result column is not compact");
        const CudaDriver::Event copied = lane.cuda->createEvent();
        lane.cuda->wait(lane.stream, reinterpret_cast<CudaDriver::Event>(resultReady.value));
        lane.cuda->downloadAsync(result.data(), reinterpret_cast<CudaDriver::DevicePointer>(bytes),
                                 result.size() * sizeof(Element), lane.stream);
        lane.cuda->record(copied, lane.stream);
        lane.cuda->synchronize(copied);
        return result;
    }
    const Element* values = reinterpret_cast<const Element*>(bytes);
    for (uint64_t index = 0u; index < count; ++index)
    {
        result[static_cast<size_t>(index)] = values[static_cast<int64_t>(index) * stride];
    }
    return result;
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
        example_runtime::require(succeeded(ovspatialquery_destroy_op_after_cuda(instance, operation, &value)) &&
                                     lane.cuda->forgetTransferredEvent(consumerDone),
                                 "CUDA batch operation requires aggregate consumer_done ordering");
        return;
    }
    example_runtime::require(succeeded(ovspatialquery_destroy_op(instance, operation)),
                             "CPU batch operation destroy failed");
}

// The lane comes first so it outlives the cleanup below, which retires a CUDA operation on it.
struct SpatialResources
{
    Lane                               lane;
    ovspatialquery_instance_handle_t   instance   = OVSPATIALQUERY_INVALID_INSTANCE;
    ovspatialquery_population_handle_t population = OVSPATIALQUERY_INVALID_POPULATION;
    ovspatialquery_batch_plan_handle_t plan       = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    ovspatialquery_op_id_t             operation  = OVSPATIALQUERY_INVALID_OP;

    ~SpatialResources()
    {
        std::vector<std::string> ignored;
        close(ignored);
    }

    void close(std::vector<std::string>& errors)
    {
        if (operation != OVSPATIALQUERY_INVALID_OP)
        {
            ovspatialquery_op_wait_result_t waited{};
            const bool                      waitedSuccessfully =
                succeeded(ovspatialquery_wait_op(instance, operation, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited));
            bool destroyedSuccessfully = false;
            if (waitedSuccessfully)
            {
                try
                {
                    destroyAfterConsumption(instance, &operation, lane);
                    destroyedSuccessfully = true;
                }
                catch (const std::exception&)
                {}
            }
            if (!destroyedSuccessfully)
            {
                errors.emplace_back("destroy batch operation failed");
            }
            operation = OVSPATIALQUERY_INVALID_OP;
        }
        if (plan != OVSPATIALQUERY_INVALID_BATCH_PLAN)
        {
            if (!example_runtime::destroyWhenRetired([&] { return ovspatialquery_destroy_batch_plan(instance, plan); }))
            {
                errors.emplace_back("destroy batch plan failed");
            }
            plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
        }
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
    example_runtime::Runtime        runtime("ovspatialquery.08_reusable_cpu_batch.cpp", kScene);
    SpatialResources                spatial;
    std::vector<std::string>        cleanupErrors;
    runtime.openAndCompute();

    const ovx_string_t  pathValues[] = {{kHitPath, sizeof(kHitPath) - 1u}, {kOffRayPath, sizeof(kOffRayPath) - 1u}};
    ovstage_predicate_t predicate{};
    predicate.attribute.string = ovx_string_t{"usd-path", sizeof("usd-path") - 1u};
    predicate.op               = OVSTAGE_FILTER_OP_IN;
    predicate.values           = pathValues;
    predicate.value_count      = 2u;

    // CUDA work needs a byte budget for the device; the CPU needs none.
    const ovspatialquery_device_budget_t cudaBudget{DLDevice{kDLCUDA, 0}, UINT64_C(64) << 20u};
    ovspatialquery_instance_desc_t       instanceDesc{};
    instanceDesc.operation_capacity            = 8u;
    instanceDesc.max_cpu_concurrency           = 2u;
    instanceDesc.host_byte_budget              = UINT64_C(64) << 20u;
    instanceDesc.device_budgets                = &cudaBudget;
    instanceDesc.device_budget_count           = 1u;
    ovspatialquery_instance_handle_t& instance = spatial.instance;
    example_runtime::require(succeeded(ovspatialquery_create_instance(&instanceDesc, &instance)),
                             "create core instance failed");
    ovspatialquery_ovstage_population_desc_t populationDesc{};
    populationDesc.stage                            = runtime.stage();
    populationDesc.filter                           = ovstage_filter_t{&predicate, 1u};
    populationDesc.common.retained_ordinal_capacity = 1u;
    ovspatialquery_population_handle_t& population  = spatial.population;
    example_runtime::require(
        succeeded(ovspatialquery_create_ovstage_population(instance, &populationDesc, &population)),
        "create population failed");
    const ovspatialquery_population_update_result_t publication =
        finishPopulationUpdate(instance, ovspatialquery_enqueue_ovstage_population_set(instance, population, 2));

    // PRESENTATION:API:BEGIN
    // CUDA first: ask before creating the plan. The CPU runs it only when this library cannot
    // plan raycasts on CUDA device 0; a CUDA plan never falls back.
    Lane& lane = spatial.lane;
    if (succeeded(
            ovspatialquery_query_capability(instance, DLDevice{kDLCUDA, 0}, OVSPATIALQUERY_CAPABILITY_RAYCAST_PLAN)))
    {
        lane.cuda.emplace(CudaDriver::Ownership::tracked);
        lane.stream = lane.cuda->createStream();
    }
    ovspatialquery_raycast_plan_desc_t planDesc{};
    planDesc.target.kind              = OVSPATIALQUERY_TARGET_POPULATION;
    planDesc.target.handle.population = population;
    planDesc.hit_mode                 = OVSPATIALQUERY_HIT_MODE_CLOSEST;
    planDesc.precision                = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    planDesc.device                   = lane.device();
    planDesc.maximum_query_count      = 2u;
    planDesc.maximum_hits_per_query   = 1u;
    planDesc.hit_capacity_per_slot    = 2u;
    // A CUDA slot returns to the plan only after its consumers, so the second launch takes
    // the other slot instead of waiting for the first.
    planDesc.result_slot_count               = 2u;
    planDesc.result_fields                   = 0u;
    ovspatialquery_batch_plan_handle_t& plan = spatial.plan;
    example_runtime::require(succeeded(ovspatialquery_create_raycast_plan(instance, &planDesc, &plan)),
                             "create batch plan failed");
    // PRESENTATION:API:END

    double originRows[2][2][3]   = {{{-4.0, 0.0, 0.0}, {-4.0, 0.0, 4.0}}, {{-4.0, 0.0, -3.0}, {-4.0, 0.0, 0.0}}};
    float directionRows[2][2][3] = {{{1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}}, {{1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}}};
    double                    minimumDistances[2] = {0.0, 0.0};
    double                    maximumDistances[2] = {20.0, 20.0};
    int64_t                   rows[]              = {2};
    int64_t                   rows3[]             = {2, 3};
    int64_t                   stride1[]           = {1};
    int64_t                   stride3[]           = {3, 1};
    const ovx_primpath_t      hitPath             = runtime.pathFromString(kHitPath);
    const ovx_primpath_t      offPath             = runtime.pathFromString(kOffRayPath);
    std::vector<LaunchResult> launches;
    for (uint32_t launchIndex = 0u; launchIndex < 2u; ++launchIndex)
    {
        // On CUDA the rows are uploaded on the lane's stream, which the launch then uses too.
        void* originData    = originRows[launchIndex];
        void* directionData = directionRows[launchIndex];
        void* minimumData   = minimumDistances;
        void* maximumData   = maximumDistances;
        if (lane.cuda)
        {
            auto upload = [&lane](const void* data, size_t bytes)
            {
                const CudaDriver::DevicePointer pointer = lane.cuda->allocate(bytes);
                lane.cuda->uploadAsync(pointer, data, bytes, lane.stream);
                return reinterpret_cast<void*>(pointer);
            };
            originData    = upload(originRows[launchIndex], sizeof(originRows[launchIndex]));
            directionData = upload(directionRows[launchIndex], sizeof(directionRows[launchIndex]));
            minimumData   = upload(minimumDistances, sizeof(minimumDistances));
            maximumData   = upload(maximumDistances, sizeof(maximumDistances));
        }
        ovspatialquery_input_array_t origins =
            copiedArray(originData, lane.device(), DLDataType{kDLFloat, 64u, 1u}, 2, rows3, stride3);
        ovspatialquery_input_array_t directions =
            copiedArray(directionData, lane.device(), DLDataType{kDLFloat, 32u, 1u}, 2, rows3, stride3);
        ovspatialquery_input_array_t minimum =
            copiedArray(minimumData, lane.device(), DLDataType{kDLFloat, 64u, 1u}, 1, rows, stride1);
        ovspatialquery_input_array_t maximum =
            copiedArray(maximumData, lane.device(), DLDataType{kDLFloat, 64u, 1u}, 1, rows, stride1);
        // PRESENTATION:API:BEGIN
        ovspatialquery_raycast_batch_t launch{};
        launch.ordinal.kind    = OVSPATIALQUERY_ORDINAL_HEAD;
        launch.query_count     = 2u;
        launch.origins         = &origins;
        launch.unit_directions = &directions;
        launch.min_distances   = &minimum;
        launch.max_distances   = &maximum;
        if (lane.cuda)
        {
            launch.cuda_sync.has_stream = 1u;
            launch.cuda_sync.stream     = reinterpret_cast<uintptr_t>(lane.stream);
        }
        ovspatialquery_raycast_batch_reservation_t accepted{};
        ovspatialquery_enqueue_result_t            operation =
            ovspatialquery_enqueue_raycast_plan(instance, plan, &launch, &accepted);
        spatial.operation = operation.op_id;
        example_runtime::require(operation.status == OVSPATIALQUERY_API_SUCCESS &&
                                     accepted.common.source_ordinal == 2 &&
                                     (accepted.common.result_ready.value != 0u) == lane.cuda.has_value(),
                                 "batch admission failed with status " + std::to_string(operation.status));
        ovspatialquery_op_wait_result_t waited{};
        example_runtime::require(
            succeeded(ovspatialquery_wait_op(instance, spatial.operation, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited)),
            "batch wait failed");
        ovspatialquery_raycast_batch_result_t fetched{};
        example_runtime::require(succeeded(ovspatialquery_fetch_raycast_batch(instance, operation.op_id, &fetched)),
                                 "batch fetch failed");
        // PRESENTATION:API:END
        const DLDevice device = lane.device();
        example_runtime::require(fetched.common.execution_device.device_type == device.device_type &&
                                     fetched.common.execution_device.device_id == device.device_id &&
                                     (fetched.common.result_ready.value != 0u) == lane.cuda.has_value() &&
                                     fetched.common.truth == OVSPATIALQUERY_HIT_BOUNDS_CANDIDATE &&
                                     tensorAbsent(fetched.columns.geometry.types) &&
                                     tensorAbsent(fetched.columns.geometry.values) &&
                                     tensorAbsent(fetched.columns.contact.candidate.bounds_positions) &&
                                     tensorAbsent(fetched.columns.contact.candidate.bounds_normals) &&
                                     tensorAbsent(fetched.columns.contact.candidate.distances),
                                 "identity-only result metadata is unexpected");
        // Copy to the host only the values this lesson records.
        const ovspatialquery_cuda_event_t ready = fetched.common.result_ready;
        const std::vector<uint32_t>       status =
            readTensor<uint32_t>(lane, ready, fetched.columns.hit_ranges.query_status, 2u, kDLUInt, 32u);
        const uint64_t stored =
            readTensor<uint64_t>(lane, ready, fetched.columns.hit_ranges.stored_hit_count, 1u, kDLUInt, 64u)[0];
        const std::vector<uint64_t> counts =
            readTensor<uint64_t>(lane, ready, fetched.columns.hit_ranges.hit_counts, 2u, kDLUInt, 64u);
        const std::vector<uint64_t> offsets =
            readTensor<uint64_t>(lane, ready, fetched.columns.hit_ranges.hit_offsets, 2u, kDLUInt, 64u);
        const std::vector<uint8_t> truncated =
            readTensor<uint8_t>(lane, ready, fetched.columns.hit_ranges.truncated, 2u, kDLUInt, 8u);
        example_runtime::require(
            status == std::vector<uint32_t>{OVSPATIALQUERY_BATCH_ROW_SUCCESS, OVSPATIALQUERY_BATCH_ROW_SUCCESS} &&
                truncated == std::vector<uint8_t>{0u, 0u},
            "batch row status or truncation is unexpected");
        std::vector<uint8_t> covered(static_cast<size_t>(stored), 0u);
        for (size_t query = 0u; query < counts.size(); ++query)
        {
            for (uint64_t hit = 0u; hit < counts[query]; ++hit)
            {
                const uint64_t flat = offsets[query] + hit;
                example_runtime::require(flat < stored && covered[static_cast<size_t>(flat)] == 0u,
                                         "Q-to-H ranges overlap or exceed stored hits");
                covered[static_cast<size_t>(flat)] = 1u;
            }
        }
        example_runtime::require(std::vector<uint8_t>(covered.size(), 1u) == covered,
                                 "Q-to-H ranges do not cover stored hits");
        const std::vector<uint64_t> objectIds =
            readTensor<uint64_t>(lane, ready, fetched.columns.object_ids, stored, kDLUInt, 64u);
        LaunchResult result{};
        result.index         = launchIndex;
        result.sourceOrdinal = fetched.common.source_ordinal;
        result.counts        = {counts[0], counts[1]};
        result.offsets       = {offsets[0], offsets[1]};
        for (uint64_t index = 0u; index < stored; ++index)
        {
            const uint64_t id = objectIds[static_cast<size_t>(index)];
            result.ids.push_back(id);
            const ovx_primpath_t path{id};
            result.paths.emplace_back(path == hitPath ? kHitPath : (path == offPath ? kOffRayPath : "unexpected"));
        }
        launches.push_back(result);
        destroyAfterConsumption(instance, &spatial.operation, lane);
    }
    example_runtime::require(launches[0].counts == std::array<uint64_t, 2>{1u, 0u} &&
                                 launches[0].paths == std::vector<std::string>{kHitPath} &&
                                 launches[1].counts == std::array<uint64_t, 2>{1u, 1u} &&
                                 launches[1].paths == std::vector<std::string>{kOffRayPath, kHitPath},
                             "reusable batch results are unexpected");

    const std::vector<uint8_t> pixels         = runtime.render(options.frames, 2u);
    const std::string          nativeIdentity = example_runtime::nativeIdentityJson(runtime.stage());
    spatial.close(cleanupErrors);
    runtime.close(cleanupErrors);
    example_runtime::require(cleanupErrors.empty(), "presentation cleanup failed");
    const std::string  image = example_evidence::writePng(output / "frame.png", pixels);
    std::ostringstream json;
    json << "{\"cleanup\":{\"clean\":true,\"errors\":[]},\"errors\":[],"
         << "\"evidence\":" << image << ",\"example\":\"" << kExample
         << "\",\"execution\":{\"frames\":" << options.frames
         << ",\"headless\":" << (options.headless ? "true" : "false")
         << ",\"resolution\":[1280,720]},\"inputs\":{\"batch_plan\":{"
         << "\"device\":\"" << (lane.cuda ? "cuda:0" : "cpu:0") << "\",\"family\":\"raycast\","
         << "\"hit_mode\":\"closest\",\"maximum_query_count\":2,"
         << "\"precision\":\"aabb_candidate\",\"result_slot_count\":2},"
         << "\"stage\":\"inline-usda\",\"target_paths\":[\"" << kHitPath << "\",\"" << kOffRayPath
         << "\"]},\"language\":\"cpp\",\"native\":" << nativeIdentity
         << ",\"publication\":{\"object_count\":" << publication.object_count
         << ",\"source_ordinal\":" << publication.published_ordinal << "},\"result\":{\"launches\":[";
    for (size_t launchIndex = 0u; launchIndex < launches.size(); ++launchIndex)
    {
        const LaunchResult& launch = launches[launchIndex];
        if (launchIndex != 0u)
        {
            json << ",";
        }
        json << "{\"hit_offsets\":[" << launch.offsets[0] << "," << launch.offsets[1]
             << "],\"launch_index\":" << launch.index << ",\"object_ids\":[";
        for (size_t index = 0u; index < launch.ids.size(); ++index)
        {
            json << (index == 0u ? "" : ",") << launch.ids[index];
        }
        json << "],\"prim_paths\":[";
        for (size_t index = 0u; index < launch.paths.size(); ++index)
        {
            json << (index == 0u ? "" : ",") << example_evidence::jsonString(launch.paths[index]);
        }
        json << "],\"query_hit_counts\":[" << launch.counts[0] << "," << launch.counts[1]
             << "],\"source_ordinal\":" << launch.sourceOrdinal << ",\"stored_hit_count\":" << launch.ids.size() << "}";
    }
    json << "],\"plan_create_count\":1},\"schema_version\":1}";
    example_evidence::writeText(output / "semantic.json", json.str());
    if (!options.headless)
    {
        example_runtime::presentInteractive("ovspatialquery: reusable batch plan", pixels);
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
