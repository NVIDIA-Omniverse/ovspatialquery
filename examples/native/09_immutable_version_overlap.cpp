/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuda_driver.h"
#include "example_runtime.h"

#include <ovspatialquery/ovspatialquery.h>
#include <ovspatialquery/ovspatialquery_ovstage.h>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

constexpr char kExample[]    = "09_immutable_version_overlap";
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
        color3f inputs:diffuseColor = (0.08, 0.4, 0.95)
        float inputs:metallic = 0.18
        float inputs:roughness = 0.24
        token outputs:surface
    }
}
def Material "PinnedLook"
{
    token outputs:surface.connect = </PinnedLook/P.outputs:surface>
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
    def Cube "Moving" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        double size = 2
        float3[] extent = [(-1, -1, -1), (1, 1, 1)]
        double3 xformOp:translate.timeSamples = {
            0: (1, 0, 0),
            1: (8, 0, 0),
        }
        uniform token[] xformOpOrder = ["xformOp:translate"]
        rel material:binding = </MovingLook>
    }
    # PRESENTATION:USDA:END
    def Cube "PinnedRay" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        double size = 2
        matrix4d xformOp:transform = ((2.5, 0, 0, 0), (0, 0.028, 0, 0), (0, 0, 0.028, 0), (-1.5, 0, 0, 1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
        rel material:binding = </PinnedLook>
    }
    def Sphere "PinnedContact" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        double radius = 0.2
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        rel material:binding = </PinnedLook>
    }
}
def Xform "Render"
{
    def Camera "Camera"
    {
        float focalLength = 18
        double2 clippingRange = (0.01, 1000)
        matrix4d xformOp:transform = ((0.92, 0, -0.39, 0), (-0.1, 0.97, -0.23, 0), (0.38, 0.25, 0.89, 0), (13, 6, 20, 1))
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
                errors.emplace_back("destroy pinned query operation failed");
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
    example_runtime::Runtime        runtime("ovspatialquery.09_immutable_version_overlap.cpp", kScene);
    SpatialResources                spatial;
    std::vector<std::string>        cleanupErrors;
    runtime.openAndCompute();

    const ovx_string_t  pathValue{kTargetPath, sizeof(kTargetPath) - 1u};
    ovstage_predicate_t predicate{};
    predicate.attribute.string = ovx_string_t{"usd-path", sizeof("usd-path") - 1u};
    predicate.op               = OVSTAGE_FILTER_OP_IN;
    predicate.values           = &pathValue;
    predicate.value_count      = 1u;

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
    populationDesc.common.retained_ordinal_capacity = 2u;
    ovspatialquery_population_handle_t& population  = spatial.population;
    example_runtime::require(
        succeeded(ovspatialquery_create_ovstage_population(instance, &populationDesc, &population)),
        "create population failed");
    (void)finishPopulationUpdate(instance, ovspatialquery_enqueue_ovstage_population_set(instance, population, 2));

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
    planDesc.target.kind                     = OVSPATIALQUERY_TARGET_POPULATION;
    planDesc.target.handle.population        = population;
    planDesc.hit_mode                        = OVSPATIALQUERY_HIT_MODE_CLOSEST;
    planDesc.precision                       = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    planDesc.device                          = lane.device();
    planDesc.maximum_query_count             = 1u;
    planDesc.maximum_hits_per_query          = 1u;
    planDesc.hit_capacity_per_slot           = 1u;
    planDesc.result_slot_count               = 1u;
    planDesc.result_fields                   = 0u;
    ovspatialquery_batch_plan_handle_t& plan = spatial.plan;
    example_runtime::require(succeeded(ovspatialquery_create_raycast_plan(instance, &planDesc, &plan)),
                             "create batch plan failed");

    double  origins[][3]       = {{-4.0, 0.0, 0.0}};
    float   directions[][3]    = {{1.0f, 0.0f, 0.0f}};
    double  minimumDistances[] = {0.0};
    double  maximumDistances[] = {6.0};
    int64_t rows[]             = {1};
    int64_t rows3[]            = {1, 3};
    int64_t stride1[]          = {1};
    int64_t stride3[]          = {3, 1};
    // On CUDA the rows are uploaded on the lane's stream, which the launch then uses too.
    void* originData    = origins;
    void* directionData = directions;
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
        originData    = upload(origins, sizeof(origins));
        directionData = upload(directions, sizeof(directions));
        minimumData   = upload(minimumDistances, sizeof(minimumDistances));
        maximumData   = upload(maximumDistances, sizeof(maximumDistances));
    }
    ovspatialquery_input_array_t originColumn =
        copiedArray(originData, lane.device(), DLDataType{kDLFloat, 64u, 1u}, 2, rows3, stride3);
    ovspatialquery_input_array_t directionColumn =
        copiedArray(directionData, lane.device(), DLDataType{kDLFloat, 32u, 1u}, 2, rows3, stride3);
    ovspatialquery_input_array_t minimumColumn =
        copiedArray(minimumData, lane.device(), DLDataType{kDLFloat, 64u, 1u}, 1, rows, stride1);
    ovspatialquery_input_array_t maximumColumn =
        copiedArray(maximumData, lane.device(), DLDataType{kDLFloat, 64u, 1u}, 1, rows, stride1);
    // PRESENTATION:API:BEGIN
    ovspatialquery_raycast_batch_t launch{};
    launch.ordinal.kind    = OVSPATIALQUERY_ORDINAL_HEAD;
    launch.query_count     = 1u;
    launch.origins         = &originColumn;
    launch.unit_directions = &directionColumn;
    launch.min_distances   = &minimumColumn;
    launch.max_distances   = &maximumColumn;
    if (lane.cuda)
    {
        launch.cuda_sync.has_stream = 1u;
        launch.cuda_sync.stream     = reinterpret_cast<uintptr_t>(lane.stream);
    }
    ovspatialquery_raycast_batch_reservation_t accepted{};
    ovspatialquery_enqueue_result_t            queryOperation =
        ovspatialquery_enqueue_raycast_plan(instance, plan, &launch, &accepted);
    spatial.operation = queryOperation.op_id;
    example_runtime::require(queryOperation.status == OVSPATIALQUERY_API_SUCCESS &&
                                 accepted.common.source_ordinal == 2 &&
                                 (accepted.common.result_ready.value != 0u) == lane.cuda.has_value(),
                             "async query did not pin ordinal 2");
    // PRESENTATION:API:END

    // PRESENTATION:API:BEGIN
    example_runtime::waitPopulationOpen(runtime.stage(), ovstage_population_apply_usd_time(runtime.stage(), 3u, 1.0));
    const ovstage_write_floor_desc_t floorThree{3u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
    example_runtime::require(
        example_runtime::waitStageOperation(runtime.stage(), ovstage_advance_write_floor(runtime.stage(), &floorThree)),
        "time-sampled write floor failed");
    const ovspatialquery_population_update_result_t nextPublication =
        finishPopulationUpdate(instance, ovspatialquery_enqueue_ovstage_population_update(instance, population, 2, 3));
    // PRESENTATION:API:END

    // PRESENTATION:API:BEGIN
    ovspatialquery_op_wait_result_t waited{};
    example_runtime::require(
        succeeded(ovspatialquery_wait_op(instance, spatial.operation, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited)),
        "pinned query wait failed");
    ovspatialquery_raycast_batch_result_t pinned{};
    example_runtime::require(succeeded(ovspatialquery_fetch_raycast_batch(instance, queryOperation.op_id, &pinned)) &&
                                 pinned.common.source_ordinal == 2 &&
                                 pinned.common.execution_device.device_type == lane.device().device_type &&
                                 pinned.common.execution_device.device_id == 0 &&
                                 (pinned.common.result_ready.value != 0u) == lane.cuda.has_value() &&
                                 pinned.common.truth == OVSPATIALQUERY_HIT_BOUNDS_CANDIDATE &&
                                 tensorAbsent(pinned.columns.geometry.types) &&
                                 tensorAbsent(pinned.columns.geometry.values) &&
                                 tensorAbsent(pinned.columns.contact.candidate.bounds_positions) &&
                                 tensorAbsent(pinned.columns.contact.candidate.bounds_normals) &&
                                 tensorAbsent(pinned.columns.contact.candidate.distances),
                             "pinned query lost the prior immutable version");
    // PRESENTATION:API:END
    // Copy to the host only the values this lesson records.
    const ovspatialquery_cuda_event_t ready = pinned.common.result_ready;
    const std::vector<uint32_t>       status =
        readTensor<uint32_t>(lane, ready, pinned.columns.hit_ranges.query_status, 1u, kDLUInt, 32u);
    const uint64_t pinnedCount =
        readTensor<uint64_t>(lane, ready, pinned.columns.hit_ranges.stored_hit_count, 1u, kDLUInt, 64u)[0];
    const std::vector<uint64_t> counts =
        readTensor<uint64_t>(lane, ready, pinned.columns.hit_ranges.hit_counts, 1u, kDLUInt, 64u);
    const std::vector<uint64_t> offsets =
        readTensor<uint64_t>(lane, ready, pinned.columns.hit_ranges.hit_offsets, 1u, kDLUInt, 64u);
    const std::vector<uint8_t> truncated =
        readTensor<uint8_t>(lane, ready, pinned.columns.hit_ranges.truncated, 1u, kDLUInt, 8u);
    example_runtime::require(status == std::vector<uint32_t>{OVSPATIALQUERY_BATCH_ROW_SUCCESS} && pinnedCount == 1u &&
                                 counts == std::vector<uint64_t>{1u} && offsets == std::vector<uint64_t>{0u} &&
                                 truncated == std::vector<uint8_t>{0u},
                             "pinned Q-to-H range is unexpected");
    const uint64_t pinnedId =
        readTensor<uint64_t>(lane, ready, pinned.columns.object_ids, pinnedCount, kDLUInt, 64u)[0];
    example_runtime::require(ovx_primpath_t{pinnedId} == runtime.pathFromString(kTargetPath),
                             "pinned result path is unexpected");
    destroyAfterConsumption(instance, &queryOperation.op_id, lane);
    spatial.operation = OVSPATIALQUERY_INVALID_OP;

    ovspatialquery_raycast_desc_t currentRay{};
    currentRay.common.target.target.kind              = OVSPATIALQUERY_TARGET_POPULATION;
    currentRay.common.target.target.handle.population = population;
    currentRay.common.target.ordinal.kind             = OVSPATIALQUERY_ORDINAL_HEAD;
    currentRay.common.precision                       = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    currentRay.ray.origin[0]                          = -4.0;
    currentRay.ray.unit_direction[0]                  = 1.0f;
    currentRay.ray.max_distance                       = 6.0;
    ovspatialquery_raycast_single_result_t current{};
    example_runtime::require(succeeded(ovspatialquery_raycast_closest_sync(instance, &currentRay, &current)) &&
                                 current.source_ordinal == 3 && current.has_hit == 0u,
                             "current query did not observe the next publication");

    const std::vector<uint8_t> pixels         = runtime.render(options.frames, 3u);
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
         << ",\"resolution\":[1280,720]},\"inputs\":{\"device\":\"" << (lane.cuda ? "cuda:0" : "cpu:0")
         << "\",\"ordering\":["
         << "\"admit asynchronous query\",\"submit and wait for next publication\","
         << "\"fetch pinned query\",\"query current publication\"],"
         << "\"stage\":\"inline-usda\",\"target_paths\":[\"" << kTargetPath
         << "\"]},\"language\":\"cpp\",\"native\":" << nativeIdentity
         << ",\"publication\":{\"next_publication\":{\"source_ordinal\":" << nextPublication.published_ordinal
         << "}},\"result\":{\"current_query\":{\"has_hit\":false,"
         << "\"source_ordinal\":" << current.source_ordinal
         << "},\"pinned_query\":{\"admission_source_ordinal\":" << accepted.common.source_ordinal << ",\"prim_path\":\""
         << kTargetPath << "\",\"result_source_ordinal\":" << pinned.common.source_ordinal
         << ",\"stored_hit_count\":1}},\"schema_version\":1}";
    example_evidence::writeText(output / "semantic.json", json.str());
    if (!options.headless)
    {
        example_runtime::presentInteractive("ovspatialquery: immutable version overlap", pixels);
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
