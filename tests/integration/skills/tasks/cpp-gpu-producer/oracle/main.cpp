// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// cpp-gpu-producer oracle: a GPU producer loop over ovspatialquery. Every step publishes
// device-resident POSES rows prepared on the device (PREPARE_ON_DEVICE) and launches a CUDA
// AABB overlap batch that selects the still-pending publication (ORDINAL_ADMITTED_EXACT) on
// the caller's stream; a device copy appends the step's hit counts to a history buffer. The
// host never waits for the step it just submitted: it retires finished operations by polling
// and blocks only when the pending window or every result slot is full.

#include <ovspatialquery/ovspatialquery.h>
#include <ovspatialquery/ovspatialquery_explicit.h>

#include <dlfcn.h>

#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace
{

[[noreturn]] void fail(const std::string& what)
{
    std::fprintf(stderr, "error: %s\n", what.c_str());
    std::exit(1);
}

// ---------------------------------------------------------------------------------------
// The CUDA driver API, loaded at run time; the few declarations it needs are written here.
// ---------------------------------------------------------------------------------------

using CUresult    = int;
using CUdeviceptr = unsigned long long;
using CUcontext   = void*;
using CUstream    = void*;
using CUevent     = void*;

struct Cuda
{
    CUresult (*cuInit)(unsigned);
    CUresult (*cuDeviceGet)(int*, int);
    CUresult (*cuDevicePrimaryCtxRetain)(CUcontext*, int);
    CUresult (*cuDevicePrimaryCtxRelease)(int);
    CUresult (*cuCtxSetCurrent)(CUcontext);
    CUresult (*cuCtxSynchronize)();
    CUresult (*cuStreamCreate)(CUstream*, unsigned);
    CUresult (*cuStreamDestroy)(CUstream);
    CUresult (*cuMemAlloc)(CUdeviceptr*, size_t);
    CUresult (*cuMemFree)(CUdeviceptr);
    CUresult (*cuMemcpyHtoD)(CUdeviceptr, const void*, size_t);
    CUresult (*cuMemcpyDtoH)(void*, CUdeviceptr, size_t);
    CUresult (*cuMemcpyDtoDAsync)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
    CUresult (*cuEventCreate)(CUevent*, unsigned);
    CUresult (*cuEventDestroy)(CUevent);
    CUresult (*cuEventRecord)(CUevent, CUstream);
    CUresult (*cuEventSynchronize)(CUevent);
    CUresult (*cuEventElapsedTime)(float*, CUevent, CUevent);

    void load()
    {
        void* driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        if (driver == nullptr)
        {
            fail("cannot load libcuda.so.1: no NVIDIA driver");
        }
        auto get = [driver](auto& slot, const char* name)
        {
            slot = reinterpret_cast<std::remove_reference_t<decltype(slot)>>(dlsym(driver, name));
            if (slot == nullptr)
            {
                fail(std::string("libcuda has no ") + name);
            }
        };
        get(cuInit, "cuInit");
        get(cuDeviceGet, "cuDeviceGet");
        get(cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain");
        get(cuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2");
        get(cuCtxSetCurrent, "cuCtxSetCurrent");
        get(cuCtxSynchronize, "cuCtxSynchronize");
        get(cuStreamCreate, "cuStreamCreate");
        get(cuStreamDestroy, "cuStreamDestroy_v2");
        get(cuMemAlloc, "cuMemAlloc_v2");
        get(cuMemFree, "cuMemFree_v2");
        get(cuMemcpyHtoD, "cuMemcpyHtoD_v2");
        get(cuMemcpyDtoH, "cuMemcpyDtoH_v2");
        get(cuMemcpyDtoDAsync, "cuMemcpyDtoDAsync_v2");
        get(cuEventCreate, "cuEventCreate");
        get(cuEventDestroy, "cuEventDestroy_v2");
        get(cuEventRecord, "cuEventRecord");
        get(cuEventSynchronize, "cuEventSynchronize");
        get(cuEventElapsedTime, "cuEventElapsedTime");
    }
};

Cuda cu;

void check(CUresult result, const char* what)
{
    if (result != 0)
    {
        fail(std::string(what) + " returned CUDA error " + std::to_string(result));
    }
}

template <typename T>
CUdeviceptr upload(const std::vector<T>& values)
{
    CUdeviceptr pointer = 0;
    check(cu.cuMemAlloc(&pointer, values.size() * sizeof(T)), "cuMemAlloc");
    check(cu.cuMemcpyHtoD(pointer, values.data(), values.size() * sizeof(T)), "cuMemcpyHtoD");
    return pointer;
}

// ---------------------------------------------------------------------------------------
// ovspatialquery helpers
// ---------------------------------------------------------------------------------------

ovspatialquery_instance_handle_t g_instance{};

std::string lastError()
{
    const ovspatialquery_string_t detail = ovspatialquery_get_last_error();
    return std::string(detail.ptr ? detail.ptr : "", detail.length);
}

void require(ovspatialquery_api_status_t status, const char* what)
{
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        fail(std::string(what) + " (status " + std::to_string(int(status)) + "): " + lastError());
    }
}

void waitAndDestroy(ovspatialquery_op_id_t op, const char* what)
{
    ovspatialquery_op_wait_result_t waited{};
    require(ovspatialquery_wait_op(g_instance, op, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status, what);
    if (waited.state != OVSPATIALQUERY_OP_COMPLETED)
    {
        const ovspatialquery_string_t detail = ovspatialquery_get_last_op_error(g_instance, op);
        fail(std::string(what) + ": " + std::string(detail.ptr ? detail.ptr : "", detail.length));
    }
    require(ovspatialquery_destroy_op(g_instance, &op).status, "destroy_op");
}

// One column descriptor. The shape array must outlive every operation that borrows it.
ovspatialquery_input_array_t column(const void* data, DLDevice device, DLDataTypeCode code, uint8_t bits, int32_t ndim,
                                    int64_t* shape)
{
    ovspatialquery_input_array_t array{};
    array.tensor.data    = const_cast<void*>(data);
    array.tensor.device  = device;
    array.tensor.ndim    = ndim;
    array.tensor.dtype   = DLDataType{uint8_t(code), bits, 1};
    array.tensor.shape   = shape;
    array.tensor.strides = nullptr; // compact row-major
    array.ownership      = OVSPATIALQUERY_INPUT_BORROWED_UNTIL_TERMINAL;
    return array;
}

// ---------------------------------------------------------------------------------------
// A minimal JSON reader for world.json: objects, arrays and numbers.
// ---------------------------------------------------------------------------------------

struct Json
{
    enum Kind
    {
        Number,
        Array,
        Object,
        Other
    } kind                             = Other;
    double                      number = 0.0;
    std::vector<Json>           items;
    std::map<std::string, Json> fields;

    const Json& operator[](const char* key) const
    {
        auto found = fields.find(key);
        if (found == fields.end())
        {
            fail(std::string("world.json has no ") + key);
        }
        return found->second;
    }
};

struct Parser
{
    const std::string& text;
    size_t             at = 0;

    void skip()
    {
        while (at < text.size() && std::isspace(static_cast<unsigned char>(text[at])))
        {
            ++at;
        }
    }
    std::string string()
    {
        std::string out;
        for (++at; at < text.size() && text[at] != '"'; ++at)
        {
            out += text[at];
        }
        ++at;
        return out;
    }
    Json value()
    {
        skip();
        Json out;
        if (text[at] == '{')
        {
            out.kind = Json::Object;
            for (++at, skip(); text[at] != '}'; skip())
            {
                const std::string key = string();
                skip();
                ++at; // ':'
                out.fields[key] = value();
                skip();
                if (text[at] == ',')
                {
                    ++at;
                }
            }
            ++at;
        }
        else if (text[at] == '[')
        {
            out.kind = Json::Array;
            for (++at, skip(); text[at] != ']'; skip())
            {
                out.items.push_back(value());
                skip();
                if (text[at] == ',')
                {
                    ++at;
                }
            }
            ++at;
        }
        else if (text[at] == '"')
        {
            string();
        }
        else
        {
            char* end  = nullptr;
            out.kind   = Json::Number;
            out.number = std::strtod(text.c_str() + at, &end);
            at         = size_t(end - text.c_str());
        }
        return out;
    }
};

// Each row of `rows` flattened, `width` values per row.
std::vector<double> flatten(const Json& rows, size_t width)
{
    std::vector<double> out;
    for (const Json& row : rows.items)
    {
        if (row.items.size() != width)
        {
            fail("unexpected row width in world.json");
        }
        for (const Json& value : row.items)
        {
            out.push_back(value.number);
        }
    }
    return out;
}

constexpr uint32_t kResultSlots     = 3; // launches in flight
constexpr uint32_t kPendingCapacity = 4; // publications in flight
constexpr uint64_t kMaxHitsPerQuery = 32;

} // namespace

int main()
{
    std::ifstream     file("world.json");
    std::stringstream buffer;
    buffer << file.rdbuf();
    const std::string text = buffer.str();
    if (text.empty())
    {
        fail("cannot read world.json");
    }
    Parser       parser{text};
    const Json   world   = parser.value();
    const int    steps   = int(world["steps"].number);
    const size_t objects = world["object_ids"].items.size();
    const size_t sensors = world["query_centers"].items.size();

    // ----- Host-side preparation, once: every step's poses (the stand-in producer). -----
    std::map<uint64_t, std::vector<double>> boxTypes;
    for (const Json& type : world["box_types"].items)
    {
        boxTypes[uint64_t(type["id"].number)] = flatten(Json{Json::Array, 0.0, {type["half_extents"]}, {}}, 3);
    }
    std::vector<uint64_t> ids(objects), boxes(objects);
    std::vector<int32_t>  types(objects, OVSPATIALQUERY_GEOMETRY_BOX);
    for (size_t i = 0; i < objects; ++i)
    {
        ids[i]   = uint64_t(world["object_ids"].items[i].number);
        boxes[i] = uint64_t(world["object_boxes"].items[i].number);
    }
    const std::vector<double> start    = flatten(world["object_positions"], 3);
    const std::vector<double> velocity = flatten(world["object_velocities"], 3);
    std::vector<double>       poses(size_t(steps + 1) * objects * 3);
    for (int s = 0; s <= steps; ++s)
    {
        for (size_t k = 0; k < objects * 3; ++k)
        {
            poses[size_t(s) * objects * 3 + k] = start[k] + double(s) * velocity[k];
        }
    }
    std::vector<double> rotations(objects * 4, 0.0);
    for (size_t i = 0; i < objects; ++i)
    {
        rotations[i * 4 + 3] = 1.0;
    }
    std::vector<uint64_t> definitionIds;
    std::vector<double>   definitionExtents;
    for (const auto& [id, extents] : boxTypes)
    {
        definitionIds.push_back(id);
        definitionExtents.insert(definitionExtents.end(), extents.begin(), extents.end());
    }

    // ----- CUDA: primary context (the one the library uses), a stream, device buffers. -----
    cu.load();
    check(cu.cuInit(0), "cuInit");
    int       device = 0;
    CUcontext context{};
    check(cu.cuDeviceGet(&device, 0), "cuDeviceGet");
    check(cu.cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
    check(cu.cuCtxSetCurrent(context), "cuCtxSetCurrent");
    CUstream stream{};
    check(cu.cuStreamCreate(&stream, 1 /* CU_STREAM_NON_BLOCKING */), "cuStreamCreate");

    const CUdeviceptr dIds       = upload(ids);
    const CUdeviceptr dTypes     = upload(types);
    const CUdeviceptr dBoxes     = upload(boxes);
    const CUdeviceptr dPoses     = upload(poses);
    const CUdeviceptr dRotations = upload(rotations);
    const CUdeviceptr dCenters   = upload(flatten(world["query_centers"], 3));
    const CUdeviceptr dExtents   = upload(flatten(world["query_half_extents"], 3));
    CUdeviceptr       dHistory   = 0; // steps x sensors hit counts, written on the device
    check(cu.cuMemAlloc(&dHistory, size_t(steps) * sensors * sizeof(uint64_t)), "cuMemAlloc");
    // Synchronous copies from pageable memory can return before the DMA lands, and the
    // non-blocking stream is not ordered after them: settle the uploads once, here.
    check(cu.cuCtxSynchronize(), "cuCtxSynchronize");

    // ----- Instance with a CUDA device budget; box definitions; a device-prepared population. -----
    const DLDevice                 gpu{kDLCUDA, device};
    const DLDevice                 cpu{kDLCPU, 0};
    ovspatialquery_device_budget_t budget{gpu, uint64_t(1) << 30};
    ovspatialquery_instance_desc_t instanceDesc{256, 2, uint64_t(256) << 20, &budget, 1};
    require(ovspatialquery_create_instance(&instanceDesc, &g_instance).status, "create_instance");

    int64_t                      definitionShape[] = {int64_t(definitionIds.size())};
    int64_t                      extentShape[]     = {int64_t(definitionIds.size()), 3};
    ovspatialquery_input_array_t definitionColumn  = column(definitionIds.data(), cpu, kDLUInt, 64, 1, definitionShape);
    ovspatialquery_input_array_t extentColumn = column(definitionExtents.data(), cpu, kDLFloat, 64, 2, extentShape);
    ovspatialquery_box_definition_table_t definitions{&definitionColumn, &extentColumn};
    const ovspatialquery_enqueue_result_t registered =
        ovspatialquery_enqueue_box_registration(g_instance, &definitions);
    require(registered.status, "enqueue_box_registration");
    waitAndDestroy(registered.op_id, "box registration");

    ovspatialquery_population_handle_t population{};
    ovspatialquery_population_desc_t   populationDesc{1, kPendingCapacity};
    require(ovspatialquery_create_explicit_population(g_instance, &populationDesc, &population).status,
            "create_explicit_population");

    int64_t rowShape[]  = {int64_t(objects)};
    int64_t poseShape[] = {int64_t(objects), 3};
    int64_t quatShape[] = {int64_t(objects), 4};
    auto    poseAt = [&](int s) { return reinterpret_cast<void*>(dPoses + size_t(s) * objects * 3 * sizeof(double)); };
    ovspatialquery_input_array_t idColumn   = column(reinterpret_cast<void*>(dIds), gpu, kDLUInt, 64, 1, rowShape);
    ovspatialquery_input_array_t typeColumn = column(reinterpret_cast<void*>(dTypes), gpu, kDLInt, 32, 1, rowShape);
    ovspatialquery_input_array_t boxColumn  = column(reinterpret_cast<void*>(dBoxes), gpu, kDLUInt, 64, 1, rowShape);
    ovspatialquery_input_array_t rotationColumn =
        column(reinterpret_cast<void*>(dRotations), gpu, kDLFloat, 64, 2, quatShape);

    // The library reads device rows on `stream` and prepares the version on the device.
    ovspatialquery_dynamic_execution_t execution{};
    execution.device      = gpu;
    execution.cuda_sync   = ovspatialquery_cuda_sync_t{1, reinterpret_cast<uintptr_t>(stream), 0};
    execution.preparation = OVSPATIALQUERY_PREPARE_ON_DEVICE;

    ovspatialquery_input_array_t             firstPoses = column(poseAt(0), gpu, kDLFloat, 64, 2, poseShape);
    ovspatialquery_explicit_population_set_t set{};
    set.target_ordinal            = 0;
    set.instances.object_values   = &idColumn;
    set.instances.geometry_types  = &typeColumn;
    set.instances.geometry_values = &boxColumn;
    set.instances.translations    = &firstPoses;
    set.instances.rotations_xyzw  = &rotationColumn;
    set.dynamic_execution         = &execution;
    const ovspatialquery_enqueue_result_t first =
        ovspatialquery_enqueue_explicit_population_set(g_instance, population, &set);
    require(first.status, "enqueue_explicit_population_set");
    waitAndDestroy(first.op_id, "first set"); // an update can only build on a completed set

    // ----- A CUDA AABB overlap plan. A device-prepared version serves BOUNDS precision only;
    // for axis-aligned boxes against axis-aligned queries the bounds are the boxes. -----
    ovspatialquery_overlap_plan_desc_t planDesc{};
    planDesc.target.kind                         = OVSPATIALQUERY_TARGET_POPULATION;
    planDesc.target.handle.population            = population;
    planDesc.hit_mode                            = OVSPATIALQUERY_HIT_MODE_MULTIPLE;
    planDesc.precision                           = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    planDesc.shape.kind                          = OVSPATIALQUERY_OVERLAP_AABB;
    planDesc.shape.data.aabb.half_extents[0]     = planDesc.shape.data.aabb.half_extents[1] =
        planDesc.shape.data.aabb.half_extents[2] = 1.0; // every launch supplies its own
    planDesc.device                              = gpu;
    planDesc.maximum_query_count                 = sensors;
    planDesc.maximum_hits_per_query              = kMaxHitsPerQuery;
    planDesc.hit_capacity_per_slot               = sensors * kMaxHitsPerQuery;
    planDesc.result_slot_count                   = kResultSlots;
    ovspatialquery_batch_plan_handle_t plan{};
    require(ovspatialquery_create_overlap_plan(g_instance, &planDesc, &plan).status, "create_overlap_plan");

    int64_t                      sensorShape[] = {int64_t(sensors), 3};
    ovspatialquery_input_array_t centerColumn =
        column(reinterpret_cast<void*>(dCenters), gpu, kDLFloat, 64, 2, sensorShape);
    ovspatialquery_input_array_t extentRows =
        column(reinterpret_cast<void*>(dExtents), gpu, kDLFloat, 64, 2, sensorShape);

    // ----- The loop. -----
    struct InFlight
    {
        ovspatialquery_op_id_t publication;
        ovspatialquery_op_id_t launch;
        CUevent                consumed;
    };
    std::deque<InFlight>                        inFlight;
    std::vector<ovspatialquery_input_array_t>   stepPoses(size_t(steps) + 1);
    ovspatialquery_explicit_population_update_t update{};
    ovspatialquery_input_array_t                noRemovals   = column(nullptr, gpu, kDLUInt, 64, 1, nullptr);
    int64_t                                     emptyShape[] = {0};
    noRemovals.tensor.shape                                  = emptyShape;

    auto terminal = [](ovspatialquery_op_id_t op, const char* what)
    {
        ovspatialquery_op_status_t status{};
        require(ovspatialquery_query_op_status(g_instance, op, &status).status, "query_op_status");
        if (status.state == OVSPATIALQUERY_OP_FAILURE)
        {
            fail(std::string(what) + " failed: " + std::to_string(int(status.terminal_status)));
        }
        return status.state == OVSPATIALQUERY_OP_COMPLETED;
    };
    // Retire every finished step, oldest first, without waiting for unfinished ones.
    auto retire = [&]()
    {
        while (!inFlight.empty())
        {
            InFlight& oldest = inFlight.front();
            if (oldest.publication.raw_id != 0 && terminal(oldest.publication, "publication"))
            {
                require(ovspatialquery_destroy_op(g_instance, &oldest.publication).status, "destroy publication");
            }
            if (oldest.publication.raw_id != 0 || !terminal(oldest.launch, "launch"))
            {
                return;
            }
            ovspatialquery_cuda_event_t done{reinterpret_cast<uintptr_t>(oldest.consumed)};
            require(ovspatialquery_destroy_op_after_cuda(g_instance, &oldest.launch, &done).status,
                    "destroy_op_after_cuda");
            if (done.value != 0)
            {
                cu.cuEventDestroy(oldest.consumed); // not transferred
            }
            inFlight.pop_front();
        }
    };

    CUevent loopStart{}, loopEnd{};
    check(cu.cuEventCreate(&loopStart, 0), "cuEventCreate");
    check(cu.cuEventCreate(&loopEnd, 0), "cuEventCreate");
    const auto wallStart = std::chrono::steady_clock::now();
    check(cu.cuEventRecord(loopStart, stream), "cuEventRecord");
    for (int s = 1; s <= steps; ++s)
    {
        // Publish step s from its slice of the device pose buffer.
        stepPoses[size_t(s)]                    = column(poseAt(s), gpu, kDLFloat, 64, 2, poseShape);
        update                                  = ovspatialquery_explicit_population_update_t{};
        update.base_ordinal                     = s - 1;
        update.target_ordinal                   = s;
        update.kind                             = OVSPATIALQUERY_EXPLICIT_UPDATE_POSES;
        update.changed_instances.object_values  = &idColumn;
        update.changed_instances.translations   = &stepPoses[size_t(s)];
        update.changed_instances.rotations_xyzw = &rotationColumn;
        update.removed_object_values            = &noRemovals;
        update.dynamic_execution                = &execution;
        ovspatialquery_enqueue_result_t published{};
        while (
            (published = ovspatialquery_enqueue_explicit_population_update(g_instance, population, &update)).status ==
            OVSPATIALQUERY_API_BUSY)
        {
            retire(); // the pending window is full: back-pressure
            std::this_thread::yield();
        }
        require(published.status, "enqueue_explicit_population_update");

        // Query step s while its publication may still be pending, on the same stream.
        ovspatialquery_overlap_batch_t batch{};
        batch.ordinal      = ovspatialquery_ordinal_selection_t{OVSPATIALQUERY_ORDINAL_ADMITTED_EXACT, s};
        batch.query_count  = sensors;
        batch.translations = &centerColumn;
        batch.half_extents = &extentRows;
        batch.cuda_sync    = ovspatialquery_cuda_sync_t{1, reinterpret_cast<uintptr_t>(stream), 0};
        ovspatialquery_overlap_batch_reservation_t reservation{};
        ovspatialquery_enqueue_result_t            launched{};
        while ((launched = ovspatialquery_enqueue_overlap_plan(g_instance, plan, &batch, &reservation)).status ==
               OVSPATIALQUERY_API_RESULT_CAPACITY)
        {
            retire(); // every result slot is busy: back-pressure
            std::this_thread::yield();
        }
        require(launched.status, "enqueue_overlap_plan");

        // The consumer: queued behind the launch on the same stream, so it needs no wait on
        // result_ready. It appends the step's hit counts to the history buffer.
        const DLTensor&   counts = reservation.columns.hit_ranges.hit_counts;
        const CUdeviceptr source = reinterpret_cast<CUdeviceptr>(counts.data) + counts.byte_offset;
        check(cu.cuMemcpyDtoDAsync(dHistory + size_t(s - 1) * sensors * sizeof(uint64_t), source,
                                   sensors * sizeof(uint64_t), stream),
              "cuMemcpyDtoDAsync");
        CUevent consumed{};
        check(cu.cuEventCreate(&consumed, 2 /* CU_EVENT_DISABLE_TIMING */), "cuEventCreate");
        check(cu.cuEventRecord(consumed, stream), "cuEventRecord");
        inFlight.push_back({published.op_id, launched.op_id, consumed});
        retire();
    }
    check(cu.cuEventRecord(loopEnd, stream), "cuEventRecord");
    while (!inFlight.empty()) // drain
    {
        retire();
        std::this_thread::yield();
    }
    check(cu.cuEventSynchronize(loopEnd), "cuEventSynchronize");
    const double wallSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - wallStart).count();
    float        gpuMs       = 0.0f;
    check(cu.cuEventElapsedTime(&gpuMs, loopStart, loopEnd), "cuEventElapsedTime");

    // One copy back, after the loop.
    std::vector<uint64_t> history(size_t(steps) * sensors);
    check(cu.cuMemcpyDtoH(history.data(), dHistory, history.size() * sizeof(uint64_t)), "cuMemcpyDtoH");

    std::printf("{\n  \"steps\": %d,\n  \"sensors\": %zu,\n  \"counts\": [", steps, sensors);
    for (int s = 0; s < steps; ++s)
    {
        std::printf("%s\n    [", s ? "," : "");
        for (size_t q = 0; q < sensors; ++q)
        {
            std::printf("%s%llu", q ? ", " : "", static_cast<unsigned long long>(history[size_t(s) * sensors + q]));
        }
        std::printf("]");
    }
    std::printf("\n  ],\n  \"gpu_ms_per_step\": %.6f,\n  \"steps_per_second\": %.3f\n}\n", gpuMs / steps,
                steps / wallSeconds);

    // ----- Teardown: the library releases retired slots after their events complete. -----
    auto destroyWhenIdle = [](auto destroy)
    {
        for (int attempt = 0; attempt < 10000; ++attempt)
        {
            const ovspatialquery_api_status_t status = destroy().status;
            if (status != OVSPATIALQUERY_API_BUSY)
            {
                require(status, "destroy");
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        fail("still busy at teardown");
    };
    destroyWhenIdle([&] { return ovspatialquery_destroy_batch_plan(g_instance, plan); });
    destroyWhenIdle([&] { return ovspatialquery_destroy_population(g_instance, population); });
    for (uint64_t id : definitionIds)
    {
        destroyWhenIdle(
            [&]
            {
                return ovspatialquery_destroy_geometry_definition(
                    g_instance, ovspatialquery_geometry_ref_t{OVSPATIALQUERY_GEOMETRY_BOX, id});
            });
    }
    require(ovspatialquery_destroy_instance(g_instance).status, "destroy_instance");
    cu.cuEventDestroy(loopStart);
    cu.cuEventDestroy(loopEnd);
    for (CUdeviceptr pointer : {dIds, dTypes, dBoxes, dPoses, dRotations, dCenters, dExtents, dHistory})
    {
        cu.cuMemFree(pointer);
    }
    cu.cuStreamDestroy(stream);
    cu.cuDevicePrimaryCtxRelease(device);
    return 0;
}
