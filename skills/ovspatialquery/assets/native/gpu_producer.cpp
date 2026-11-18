// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// C++17: a GPU producer loop. Boxes on a conveyor move every step; fixed sensor gates ask
// which boxes overlap them. Every step publishes the new poses from device memory, prepared
// on the device, and launches a CUDA overlap batch that reads that publication while it is
// still pending (ORDINAL_ADMITTED_EXACT), on the caller's stream. A device copy consumes the
// counts into a history buffer; the host never waits for the step it just submitted, retires
// finished steps by polling, and copies results back once. Times the loop with CUDA events.
//
// Exits 77 (skipped) when this build or host has no CUDA device; checks every count against
// a host computation otherwise. Needs only the CUDA driver (no toolkit): its API is loaded at
// run time. With the CUDA toolkit, use cuda.h or cudart instead of the declarations below.

#include <ovspatialquery/ovspatialquery.h>
#include <ovspatialquery/ovspatialquery_explicit.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace
{

constexpr int      kSteps        = 60;
constexpr size_t   kBoxes        = 512;  // 0.4 m cubes, 1 m apart along +x
constexpr size_t   kGates        = 16;   // fixed AABB sensors along the conveyor
constexpr double   kSpeed        = 0.05; // m per step
constexpr uint32_t kSlots        = 3;    // launches in flight (result slots)
constexpr uint32_t kPending      = 4;    // publications in flight (pending window)
constexpr uint64_t kHitsPerQuery = 8;

[[noreturn]] void fail(const std::string& what)
{
    std::fprintf(stderr, "error: %s\n", what.c_str());
    std::exit(1);
}

// ----- The CUDA driver API, loaded at run time. -----
using CUresult    = int;
using CUdeviceptr = unsigned long long;
using CUcontext   = void*;
using CUstream    = void*;
using CUevent     = void*;

struct Cuda
{
    CUresult (*cuInit)(unsigned);
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
#ifdef _WIN32
        HMODULE driver = LoadLibraryA("nvcuda.dll");
        auto    find   = [driver](const char* name) { return reinterpret_cast<void*>(GetProcAddress(driver, name)); };
#else
        void* driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        auto  find   = [driver](const char* name) { return dlsym(driver, name); };
#endif
        if (driver == nullptr)
        {
            fail("no CUDA driver");
        }
        auto get = [&](auto& slot, const char* name)
        {
            slot = reinterpret_cast<std::remove_reference_t<decltype(slot)>>(find(name));
            if (slot == nullptr)
            {
                fail(std::string("the CUDA driver has no ") + name);
            }
        };
        get(cuInit, "cuInit");
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
} cu;

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

// ----- ovspatialquery helpers. -----
ovspatialquery_instance_handle_t g_instance{};

void require(ovspatialquery_api_status_t status, const char* what)
{
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        const ovspatialquery_string_t detail = ovspatialquery_get_last_error();
        fail(std::string(what) + " (status " + std::to_string(int(status)) +
             "): " + std::string(detail.ptr ? detail.ptr : "", detail.length));
    }
}

void waitAndDestroy(ovspatialquery_op_id_t op, const char* what) // setup only, never in the loop
{
    ovspatialquery_op_wait_result_t waited{};
    require(ovspatialquery_wait_op(g_instance, op, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status, what);
    if (waited.state != OVSPATIALQUERY_OP_COMPLETED)
    {
        fail(std::string(what) + " failed");
    }
    require(ovspatialquery_destroy_op(g_instance, &op).status, "destroy_op");
}

// Borrowed, compact row-major column; `shape` must outlive every operation using it.
ovspatialquery_input_array_t column(void* data, DLDevice device, DLDataTypeCode code, uint8_t bits, int32_t ndim,
                                    int64_t* shape)
{
    ovspatialquery_input_array_t array{};
    array.tensor.data   = data;
    array.tensor.device = device;
    array.tensor.ndim   = ndim;
    array.tensor.dtype  = DLDataType{uint8_t(code), bits, 1};
    array.tensor.shape  = shape;
    array.ownership     = OVSPATIALQUERY_INPUT_BORROWED_UNTIL_TERMINAL;
    return array;
}

// The pending operation state; true once it completed, failure is fatal.
bool completed(ovspatialquery_op_id_t op)
{
    ovspatialquery_op_status_t status{};
    require(ovspatialquery_query_op_status(g_instance, op, &status).status, "query_op_status");
    if (status.state == OVSPATIALQUERY_OP_FAILURE)
    {
        fail("operation failed with status " + std::to_string(int(status.terminal_status)));
    }
    return status.state == OVSPATIALQUERY_OP_COMPLETED;
}

double boxX(size_t box, int step) { return double(box) + kSpeed * step; }
double gateX(size_t gate)
{
    return 3.013 + 7.0 * double(gate); // off the 0.05 m grid: no box face touches a gate face
}

} // namespace

int main()
{
    // ----- An instance with a CUDA device budget: without one nothing runs on a GPU. -----
    const DLDevice                 gpu{kDLCUDA, 0};
    const DLDevice                 cpu{kDLCPU, 0};
    ovspatialquery_device_budget_t budget{gpu, uint64_t(256) << 20};
    ovspatialquery_instance_desc_t desc{64, 2, uint64_t(64) << 20, &budget, 1};
    require(ovspatialquery_create_instance(&desc, &g_instance).status, "create_instance");
    for (ovspatialquery_capability_t capability :
         {OVSPATIALQUERY_CAPABILITY_PREPARE_ON_DEVICE, OVSPATIALQUERY_CAPABILITY_OVERLAP_PLAN})
    {
        if (ovspatialquery_query_capability(g_instance, gpu, capability).status != OVSPATIALQUERY_API_SUCCESS)
        {
            const ovspatialquery_string_t why = ovspatialquery_get_last_error();
            std::printf("skipped: %.*s\n", int(why.length), why.ptr ? why.ptr : "");
            ovspatialquery_destroy_instance(g_instance);
            return 77;
        }
    }

    // ----- The library works in the device's primary context; use it and a stream of ours. -----
    cu.load();
    check(cu.cuInit(0), "cuInit");
    CUcontext context{};
    check(cu.cuDevicePrimaryCtxRetain(&context, 0), "cuDevicePrimaryCtxRetain");
    check(cu.cuCtxSetCurrent(context), "cuCtxSetCurrent");
    CUstream stream{};
    check(cu.cuStreamCreate(&stream, 1 /* CU_STREAM_NON_BLOCKING */), "cuStreamCreate");

    // ----- Device-resident inputs, uploaded once. A real producer writes poses on the GPU;
    // here every step's poses are computed up front and step s reads its own slice. -----
    std::vector<uint64_t> ids(kBoxes), definitions(kBoxes, 1);
    std::vector<int32_t>  types(kBoxes, OVSPATIALQUERY_GEOMETRY_BOX);
    std::vector<double>   poses(size_t(kSteps + 1) * kBoxes * 3, 0.0), rotations(kBoxes * 4, 0.0);
    for (size_t i = 0; i < kBoxes; ++i)
    {
        ids[i]               = 1000 + i;
        rotations[4 * i + 3] = 1.0;
        for (int s = 0; s <= kSteps; ++s)
        {
            poses[(size_t(s) * kBoxes + i) * 3] = boxX(i, s);
        }
    }
    std::vector<double> centers(kGates * 3, 0.0), gateHalf(kGates * 3, 1.0);
    for (size_t g = 0; g < kGates; ++g)
    {
        centers[g * 3]  = gateX(g);
        gateHalf[g * 3] = 0.6;
    }
    const CUdeviceptr dIds = upload(ids), dTypes = upload(types), dDefinitions = upload(definitions);
    const CUdeviceptr dPoses = upload(poses), dRotations = upload(rotations);
    const CUdeviceptr dCenters = upload(centers), dGateHalf = upload(gateHalf);
    CUdeviceptr       dHistory = 0; // kSteps x kGates counts, written only on the device
    check(cu.cuMemAlloc(&dHistory, size_t(kSteps) * kGates * sizeof(uint64_t)), "cuMemAlloc");
    // Pageable copies may return before their DMA lands, and a non-blocking stream is not
    // ordered after them: settle the uploads once, before anything reads them.
    check(cu.cuCtxSynchronize(), "cuCtxSynchronize");

    // ----- One box definition (CPU columns are fine for setup). -----
    uint64_t                              definitionValue[] = {1};
    double                                halfExtents[]     = {0.2, 0.2, 0.2};
    int64_t                               oneShape[] = {1}, oneByThree[] = {1, 3};
    ovspatialquery_input_array_t          definitionColumn = column(definitionValue, cpu, kDLUInt, 64, 1, oneShape);
    ovspatialquery_input_array_t          extentColumn     = column(halfExtents, cpu, kDLFloat, 64, 2, oneByThree);
    ovspatialquery_box_definition_table_t table{&definitionColumn, &extentColumn};
    const ovspatialquery_enqueue_result_t registered = ovspatialquery_enqueue_box_registration(g_instance, &table);
    require(registered.status, "enqueue_box_registration");
    waitAndDestroy(registered.op_id, "box registration");

    // ----- A population whose versions are prepared on the device. pending_publication_capacity
    // lets several publications be in flight; each update builds on the newest admitted one. -----
    ovspatialquery_population_handle_t population{};
    ovspatialquery_population_desc_t   populationDesc{1, kPending};
    require(ovspatialquery_create_explicit_population(g_instance, &populationDesc, &population).status,
            "create_explicit_population");
    int64_t rows[] = {int64_t(kBoxes)}, poseShape[] = {int64_t(kBoxes), 3}, quatShape[] = {int64_t(kBoxes), 4};
    auto poseSlice = [&](int s) { return reinterpret_cast<void*>(dPoses + size_t(s) * kBoxes * 3 * sizeof(double)); };
    ovspatialquery_input_array_t idColumn    = column(reinterpret_cast<void*>(dIds), gpu, kDLUInt, 64, 1, rows);
    ovspatialquery_input_array_t typeColumn  = column(reinterpret_cast<void*>(dTypes), gpu, kDLInt, 32, 1, rows);
    ovspatialquery_input_array_t valueColumn = column(reinterpret_cast<void*>(dDefinitions), gpu, kDLUInt, 64, 1, rows);
    ovspatialquery_input_array_t rotationColumn =
        column(reinterpret_cast<void*>(dRotations), gpu, kDLFloat, 64, 2, quatShape);

    ovspatialquery_dynamic_execution_t execution{};
    execution.device      = gpu;
    execution.cuda_sync   = ovspatialquery_cuda_sync_t{1, reinterpret_cast<uintptr_t>(stream), 0};
    execution.preparation = OVSPATIALQUERY_PREPARE_ON_DEVICE; // BOUNDS CUDA plans and CUDA reporters only

    ovspatialquery_input_array_t             initialPoses = column(poseSlice(0), gpu, kDLFloat, 64, 2, poseShape);
    ovspatialquery_explicit_population_set_t set{};
    set.target_ordinal            = 0;
    set.instances.object_values   = &idColumn;
    set.instances.geometry_types  = &typeColumn;
    set.instances.geometry_values = &valueColumn;
    set.instances.translations    = &initialPoses;
    set.instances.rotations_xyzw  = &rotationColumn;
    set.dynamic_execution         = &execution;
    const ovspatialquery_enqueue_result_t first =
        ovspatialquery_enqueue_explicit_population_set(g_instance, population, &set);
    require(first.status, "enqueue_explicit_population_set");
    waitAndDestroy(first.op_id, "first set"); // updates build only on a completed set

    // ----- A CUDA overlap plan at BOUNDS precision: what a device-prepared version serves. -----
    ovspatialquery_overlap_plan_desc_t planDesc{};
    planDesc.target.kind                         = OVSPATIALQUERY_TARGET_POPULATION;
    planDesc.target.handle.population            = population;
    planDesc.hit_mode                            = OVSPATIALQUERY_HIT_MODE_MULTIPLE;
    planDesc.precision                           = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    planDesc.shape.kind                          = OVSPATIALQUERY_OVERLAP_AABB;
    planDesc.shape.data.aabb.half_extents[0]     = planDesc.shape.data.aabb.half_extents[1] =
        planDesc.shape.data.aabb.half_extents[2] = 1.0; // launches pass per-row half extents
    planDesc.device                              = gpu;
    planDesc.maximum_query_count                 = kGates;
    planDesc.maximum_hits_per_query              = kHitsPerQuery;
    planDesc.hit_capacity_per_slot               = kGates * kHitsPerQuery;
    planDesc.result_slot_count                   = kSlots; // > 1, or every launch waits for the previous one
    ovspatialquery_batch_plan_handle_t plan{};
    require(ovspatialquery_create_overlap_plan(g_instance, &planDesc, &plan).status, "create_overlap_plan");
    int64_t                      gateShape[] = {int64_t(kGates), 3};
    ovspatialquery_input_array_t centerColumn =
        column(reinterpret_cast<void*>(dCenters), gpu, kDLFloat, 64, 2, gateShape);
    ovspatialquery_input_array_t halfColumn =
        column(reinterpret_cast<void*>(dGateHalf), gpu, kDLFloat, 64, 2, gateShape);

    // ----- The loop. -----
    struct Step
    {
        ovspatialquery_op_id_t publication, launch;
        CUevent                consumed;
    };
    std::deque<Step>                          inFlight;
    std::vector<ovspatialquery_input_array_t> stepPoses(kSteps + 1); // borrowed until terminal
    int64_t                                   noRows[]   = {0};
    ovspatialquery_input_array_t              noRemovals = column(nullptr, gpu, kDLUInt, 64, 1, noRows);

    // Retires finished steps, oldest first, by polling: never waits.
    auto retire = [&]()
    {
        while (!inFlight.empty())
        {
            Step& oldest = inFlight.front();
            if (oldest.publication.raw_id != 0 && completed(oldest.publication))
            {
                require(ovspatialquery_destroy_op(g_instance, &oldest.publication).status, "destroy_op");
            }
            if (oldest.publication.raw_id != 0 || !completed(oldest.launch))
            {
                return;
            }
            // A CUDA launch is retired after its consumers: the library takes the event and
            // releases the result slot once it completes.
            ovspatialquery_cuda_event_t done{reinterpret_cast<uintptr_t>(oldest.consumed)};
            require(ovspatialquery_destroy_op_after_cuda(g_instance, &oldest.launch, &done).status,
                    "destroy_op_after_cuda");
            inFlight.pop_front();
        }
    };

    CUevent loopStart{}, loopEnd{};
    check(cu.cuEventCreate(&loopStart, 0), "cuEventCreate");
    check(cu.cuEventCreate(&loopEnd, 0), "cuEventCreate");
    const auto wallStart = std::chrono::steady_clock::now();
    check(cu.cuEventRecord(loopStart, stream), "cuEventRecord");
    for (int s = 1; s <= kSteps; ++s)
    {
        // Publish step s: POSES rows from device memory, prepared on the device, based on the
        // newest admitted ordinal. BUSY means the pending window is full.
        stepPoses[s] = column(poseSlice(s), gpu, kDLFloat, 64, 2, poseShape);
        ovspatialquery_explicit_population_update_t update{};
        update.base_ordinal                     = s - 1;
        update.target_ordinal                   = s;
        update.kind                             = OVSPATIALQUERY_EXPLICIT_UPDATE_POSES;
        update.changed_instances.object_values  = &idColumn;
        update.changed_instances.translations   = &stepPoses[s];
        update.changed_instances.rotations_xyzw = &rotationColumn;
        update.removed_object_values            = &noRemovals;
        update.dynamic_execution                = &execution;
        ovspatialquery_enqueue_result_t published{};
        while (
            (published = ovspatialquery_enqueue_explicit_population_update(g_instance, population, &update)).status ==
            OVSPATIALQUERY_API_BUSY)
        {
            retire(); // back-pressure
            std::this_thread::yield();
        }
        require(published.status, "enqueue_explicit_population_update");

        // Query step s now: ADMITTED_EXACT orders the launch after the pending publication on
        // the device. RESULT_CAPACITY means every result slot is in flight.
        ovspatialquery_overlap_batch_t batch{};
        batch.ordinal      = ovspatialquery_ordinal_selection_t{OVSPATIALQUERY_ORDINAL_ADMITTED_EXACT, s};
        batch.query_count  = kGates;
        batch.translations = &centerColumn;
        batch.half_extents = &halfColumn;
        batch.cuda_sync    = ovspatialquery_cuda_sync_t{1, reinterpret_cast<uintptr_t>(stream), 0};
        ovspatialquery_overlap_batch_reservation_t reservation{};
        ovspatialquery_enqueue_result_t            launched{};
        while ((launched = ovspatialquery_enqueue_overlap_plan(g_instance, plan, &batch, &reservation)).status ==
               OVSPATIALQUERY_API_RESULT_CAPACITY)
        {
            retire(); // back-pressure
            std::this_thread::yield();
        }
        require(launched.status, "enqueue_overlap_plan");

        // Consume on the device: the reservation's columns exist at enqueue, and work queued on
        // the same stream after it runs after the launch (no wait on result_ready needed).
        const DLTensor&   counts = reservation.columns.hit_ranges.hit_counts;
        const CUdeviceptr source = reinterpret_cast<CUdeviceptr>(counts.data) + counts.byte_offset;
        check(cu.cuMemcpyDtoDAsync(dHistory + size_t(s - 1) * kGates * sizeof(uint64_t), source,
                                   kGates * sizeof(uint64_t), stream),
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
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - wallStart).count();
    float        gpuMs   = 0.0f;
    check(cu.cuEventElapsedTime(&gpuMs, loopStart, loopEnd), "cuEventElapsedTime");

    // ----- One copy back, then check every count against the host. -----
    std::vector<uint64_t> history(size_t(kSteps) * kGates);
    check(cu.cuMemcpyDtoH(history.data(), dHistory, history.size() * sizeof(uint64_t)), "cuMemcpyDtoH");
    int failures = 0;
    for (int s = 1; s <= kSteps; ++s)
    {
        for (size_t g = 0; g < kGates; ++g)
        {
            uint64_t expected = 0;
            for (size_t i = 0; i < kBoxes; ++i)
            {
                expected += std::fabs(boxX(i, s) - gateX(g)) <= 0.6 + 0.2;
            }
            if (history[size_t(s - 1) * kGates + g] != expected && ++failures <= 5)
            {
                std::fprintf(stderr, "step %d gate %zu: %llu, expected %llu\n", s, g,
                             static_cast<unsigned long long>(history[size_t(s - 1) * kGates + g]),
                             static_cast<unsigned long long>(expected));
            }
        }
    }
    std::printf("%d steps x %zu gates: %s; %.1f steps/s, %.3f GPU ms/step\n", kSteps, kGates,
                failures ? "MISMATCH" : "all counts match", kSteps / seconds, gpuMs / kSteps);

    // ----- Teardown: slots retired after CUDA events are released once those events fire. -----
    auto whenIdle = [](auto destroy)
    {
        ovspatialquery_api_status_t status = destroy().status;
        for (int attempt = 0; status == OVSPATIALQUERY_API_BUSY && attempt < 10000; ++attempt)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            status = destroy().status;
        }
        require(status, "destroy");
    };
    whenIdle([&] { return ovspatialquery_destroy_batch_plan(g_instance, plan); });
    whenIdle([&] { return ovspatialquery_destroy_population(g_instance, population); });
    whenIdle(
        [&]
        {
            return ovspatialquery_destroy_geometry_definition(
                g_instance, ovspatialquery_geometry_ref_t{OVSPATIALQUERY_GEOMETRY_BOX, 1});
        });
    require(ovspatialquery_destroy_instance(g_instance).status, "destroy_instance");
    cu.cuEventDestroy(loopStart);
    cu.cuEventDestroy(loopEnd);
    for (CUdeviceptr pointer : {dIds, dTypes, dDefinitions, dPoses, dRotations, dCenters, dGateHalf, dHistory})
    {
        cu.cuMemFree(pointer);
    }
    cu.cuStreamDestroy(stream);
    cu.cuDevicePrimaryCtxRelease(0);
    return failures == 0 ? 0 : 1;
}
