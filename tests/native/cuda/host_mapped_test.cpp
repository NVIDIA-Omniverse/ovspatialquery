/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// A CUDA plan created with OVSPATIALQUERY_BATCH_RESULT_HOST_MAPPED (ADR-0077)
// keeps its result columns in mapped host memory: the host reads them in place
// once the operation is terminal, with no copy and no stream synchronization,
// and they hold exactly the rows a plain CUDA plan publishes on the device. The
// safe empty result of a failed launch reaches the host too, the columns are
// charged to the host budget, a host reader retires the launch with
// destroy_op, and a CPU plan ignores the bit. A launch whose ready poll fails
// waits for the device, mapped or not, and a mapped plan whose launch cannot
// be confirmed done keeps its columns allocated.

#include <ovspatialquery/ovspatialquery_explicit.h>

#include "../../support/native/cuda_public_test_helpers.h"
#include "../../support/native/public_api_inputs.h"
#include "cuda_driver.h"
#include "ovspatialquery_cuda_failure_points.h"
#include "ovspatialquery_instance_operation_runtime.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <vector>

extern "C" void ovspatialquery_test_set_cuda_failure_point(uint32_t point);

namespace
{

constexpr DLDevice kCuda{kDLCUDA, 0};
constexpr DLDevice kCpu{kDLCPU, 0};
constexpr DLDevice kMapped{kDLCUDAHost, 0};

const char* gPhase = "initialization";

void phase(const char* name)
{
    gPhase = name;
    std::fprintf(stderr, "phase: %s\n", name);
}

bool sameDevice(DLDevice a, DLDevice b) { return a.device_type == b.device_type && a.device_id == b.device_id; }

enum class Family
{
    raycast,
    sphereSweep,
    boxSweep,
    overlap,
};

constexpr uint64_t kRows        = 8u;
constexpr uint64_t kMaximumHits = 2u;
constexpr uint64_t kMeshValue   = 2u;
constexpr uint64_t kSphereValue = 1u;

// One result: its common block and every DLTensor of its column struct, in
// declaration order. Every family's column struct is a sequence of DLTensor
// members, so two results compare tensor by tensor.
struct Columns
{
    ovspatialquery_batch_result_common_t common{};
    std::vector<DLTensor>                tensors;
};

template <typename Value>
Columns flatten(const Value& value)
{
    static_assert(sizeof(value.columns) % sizeof(DLTensor) == 0u, "column structs hold DLTensors only");
    Columns out;
    out.common = value.common;
    out.tensors.resize(sizeof(value.columns) / sizeof(DLTensor));
    std::memcpy(out.tensors.data(), &value.columns, sizeof(value.columns));
    return out;
}

size_t tensorBytes(const DLTensor& tensor)
{
    size_t elements = 1u;
    for (int32_t axis = 0; axis < tensor.ndim; ++axis)
    {
        elements *= static_cast<size_t>(tensor.shape[axis]);
    }
    return elements * tensor.dtype.bits / 8u;
}

const uint8_t* hostData(const DLTensor& tensor)
{
    return static_cast<const uint8_t*>(tensor.data) + tensor.byte_offset;
}

template <typename T>
T hostValue(const DLTensor& tensor, size_t index)
{
    T value{};
    std::memcpy(&value, hostData(tensor) + index * sizeof(T), sizeof(T));
    return value;
}

// The hit ranges lead every column struct: status, H, offsets, counts, truncation.
const DLTensor& queryStatus(const Columns& columns) { return columns.tensors[0]; }
const DLTensor& storedHitCount(const Columns& columns) { return columns.tensors[1]; }

// Spheres in front of the even rows and a double-sided mesh wall behind every
// row, each with a material, so every family stores several hits with every
// optional column present.
struct Scene
{
    // Without the mesh, setup cooks nothing, so a budget a few bytes above its
    // usage admits it.
    explicit Scene(uint64_t hostBudget = UINT64_C(64) << 20, bool mesh = true) : withMesh(mesh)
    {
        const ovspatialquery_device_budget_t budget{kCuda, UINT64_C(64) << 20};
        const ovspatialquery_instance_desc_t desc{16u, 1u, hostBudget, &budget, 1u};
        REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
        registerSphere(instance, kSphereValue, 0.5);
        population = createExplicitPopulation(instance, 1u);
        InstanceRows rows;
        if (withMesh)
        {
            registerMesh();
            rows.add(100u, OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, kMeshValue, 0.0, 0.0, 0.0);
            rows.materials.push_back(800u);
        }
        for (uint64_t row = 0u; row < kRows; row += 2u)
        {
            rows.add(10u + row, OVSPATIALQUERY_GEOMETRY_SPHERE, kSphereValue, 2.5, rowY(row), 0.0);
            rows.materials.push_back(900u + row);
        }
        ovspatialquery_explicit_population_set_t set{};
        set.target_ordinal = 1;
        set.instances      = rows.table();
        waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));
    }

    // A double-sided wall in the plane x = 5 with one material per triangle.
    void registerMesh()
    {
        uint64_t   value          = kMeshValue;
        uint64_t   zero           = 0u;
        uint64_t   vertexCount    = 4u;
        uint64_t   triangleCount  = 2u;
        float      vertices[12]   = {5.0f, -6.0f, -2.0f, 5.0f, 6.0f, -2.0f, 5.0f, 6.0f, 2.0f, 5.0f, -6.0f, 2.0f};
        uint32_t   triangles[6]   = {0u, 1u, 2u, 0u, 2u, 3u};
        uint8_t    doubleSided    = 1u;
        uint64_t   faceMaterial[] = {700u, 701u};
        InputArray values(&value, 1, kDLUInt);
        InputArray offsets(&zero, 1, kDLUInt);
        InputArray vertexCounts(&vertexCount, 1, kDLUInt);
        InputArray triangleCounts(&triangleCount, 1, kDLUInt);
        InputArray vertexData(vertices, 4, 3, kDLFloat);
        InputArray triangleData(triangles, 2, 3, kDLUInt);
        InputArray sidedness(&doubleSided, 1, kDLUInt);
        InputArray materials(faceMaterial, 2, kDLUInt);
        const ovspatialquery_triangle_mesh_definition_table_t meshes{
            &values.value,     &offsets.value,      &vertexCounts.value, &offsets.value,   &triangleCounts.value,
            &vertexData.value, &triangleData.value, &sidedness.value,    &materials.value, nullptr,
            nullptr,
        };
        waitAndDestroy(instance, ovspatialquery_enqueue_triangle_mesh_registration(instance, &meshes));
    }

    static double rowY(uint64_t row) { return -3.5 + static_cast<double>(row); }

    void close()
    {
        REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_SPHERE, kSphereValue})
                    .status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(!withMesh || ovspatialquery_destroy_geometry_definition(
                                 instance, {OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, kMeshValue})
                                     .status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
    }

    ovspatialquery_instance_handle_t   instance   = OVSPATIALQUERY_INVALID_INSTANCE;
    ovspatialquery_population_handle_t population = OVSPATIALQUERY_INVALID_POPULATION;
    bool                               withMesh   = true;
};

// The rows of every launch: rays and sweeps start at x = 0 and go +x; an
// overlap sits between the spheres and the wall. CUDA lanes read device copies.
struct Rows
{
    Rows(CudaDriver* cuda, CudaDriver::Stream stream)
    {
        for (uint64_t row = 0u; row < kRows; ++row)
        {
            const double y = Scene::rowY(row);
            starts.insert(starts.end(), {0.0, y, 0.0});
            centers.insert(centers.end(), {3.5, y, 0.0});
            directions.insert(directions.end(), {1.0f, 0.0f, 0.0f});
            rotations.insert(rotations.end(), {0.0f, 0.0f, 0.0f, 1.0f});
            maximums.push_back(20.0);
        }
        const auto place = [&](auto& values)
        {
            if (cuda == nullptr)
            {
                return static_cast<void*>(values.data());
            }
            const size_t                    bytes   = values.size() * sizeof(values[0]);
            const CudaDriver::DevicePointer pointer = cuda->allocate(bytes);
            cuda->uploadAsync(pointer, values.data(), bytes, stream);
            return reinterpret_cast<void*>(static_cast<uintptr_t>(pointer));
        };
        const DLDevice device = cuda != nullptr ? kCuda : kCpu;
        const int64_t  n      = static_cast<int64_t>(kRows);
        startColumn.set(static_cast<double*>(place(starts)), n, 3, kDLFloat, device);
        centerColumn.set(static_cast<double*>(place(centers)), n, 3, kDLFloat, device);
        directionColumn.set(static_cast<float*>(place(directions)), n, 3, kDLFloat, device);
        rotationColumn.set(static_cast<float*>(place(rotations)), n, 4, kDLFloat, device);
        maximumColumn.set(static_cast<double*>(place(maximums)), n, kDLFloat, device);
    }

    std::vector<double> starts, centers, maximums;
    std::vector<float>  directions, rotations;
    InputArray          startColumn, centerColumn, directionColumn, rotationColumn, maximumColumn;
};

struct PlanSpec
{
    Family                           family    = Family::raycast;
    ovspatialquery_hit_mode_t        mode      = OVSPATIALQUERY_HIT_MODE_CLOSEST;
    ovspatialquery_query_precision_t precision = OVSPATIALQUERY_QUERY_EXACT;
    uint32_t                         fields    = 0u;
    DLDevice                         device    = kCuda;
    uint32_t                         slots     = 1u;
};

// Every optional column the family's spec allows.
uint32_t allFields(const PlanSpec& spec)
{
    if (spec.family == Family::overlap)
    {
        return OVSPATIALQUERY_BATCH_RESULT_GEOMETRY_IDENTITY | OVSPATIALQUERY_BATCH_RESULT_MATERIAL;
    }
    uint32_t fields = OVSPATIALQUERY_BATCH_RESULT_POSITION | OVSPATIALQUERY_BATCH_RESULT_NORMAL |
                      OVSPATIALQUERY_BATCH_RESULT_DISTANCE | OVSPATIALQUERY_BATCH_RESULT_GEOMETRY_IDENTITY |
                      OVSPATIALQUERY_BATCH_RESULT_MATERIAL;
    if (spec.precision != OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE)
    {
        fields |= OVSPATIALQUERY_BATCH_RESULT_SOURCE_FEATURE;
    }
    if (spec.family != Family::raycast)
    {
        fields |= OVSPATIALQUERY_BATCH_RESULT_QUERY_POSITION;
    }
    return fields;
}

ovspatialquery_api_status_t createPlan(const Scene& scene, const PlanSpec& spec,
                                       ovspatialquery_batch_plan_handle_t& plan)
{
    const bool     multiple = spec.mode == OVSPATIALQUERY_HIT_MODE_MULTIPLE;
    const uint64_t hits     = multiple ? kMaximumHits : 1u;
    plan                    = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    switch (spec.family)
    {
    case Family::raycast: {
        ovspatialquery_raycast_plan_desc_t d{};
        d.target                 = populationRef(scene.population);
        d.hit_mode               = spec.mode;
        d.precision              = spec.precision;
        d.device                 = spec.device;
        d.maximum_query_count    = kRows;
        d.maximum_hits_per_query = hits;
        d.hit_capacity_per_slot  = kRows * hits;
        d.result_slot_count      = spec.slots;
        d.result_fields          = spec.fields;
        return ovspatialquery_create_raycast_plan(scene.instance, &d, &plan).status;
    }
    case Family::sphereSweep: {
        ovspatialquery_sphere_sweep_plan_desc_t d{};
        d.target                 = populationRef(scene.population);
        d.precision              = spec.precision;
        d.radius                 = 0.25;
        d.device                 = spec.device;
        d.maximum_query_count    = kRows;
        d.result_slot_count      = spec.slots;
        d.result_fields          = spec.fields;
        d.hit_capacity_per_slot  = kRows * hits;
        d.hit_mode               = spec.mode;
        d.maximum_hits_per_query = multiple ? hits : 0u;
        return ovspatialquery_create_sphere_sweep_plan(scene.instance, &d, &plan).status;
    }
    case Family::boxSweep: {
        ovspatialquery_box_sweep_plan_desc_t d{};
        d.target                 = populationRef(scene.population);
        d.precision              = spec.precision;
        d.half_extents[0]        = 0.25;
        d.half_extents[1]        = 0.25;
        d.half_extents[2]        = 0.25;
        d.device                 = spec.device;
        d.maximum_query_count    = kRows;
        d.result_slot_count      = spec.slots;
        d.result_fields          = spec.fields;
        d.hit_capacity_per_slot  = kRows * hits;
        d.hit_mode               = spec.mode;
        d.maximum_hits_per_query = multiple ? hits : 0u;
        return ovspatialquery_create_box_sweep_plan(scene.instance, &d, &plan).status;
    }
    case Family::overlap: {
        ovspatialquery_overlap_plan_desc_t d{};
        d.target                   = populationRef(scene.population);
        d.hit_mode                 = spec.mode;
        d.precision                = spec.precision;
        d.shape.kind               = OVSPATIALQUERY_OVERLAP_SPHERE;
        d.shape.data.sphere.radius = 1.6;
        d.device                   = spec.device;
        d.maximum_query_count      = kRows;
        d.maximum_hits_per_query   = hits;
        d.hit_capacity_per_slot    = kRows * hits;
        d.result_slot_count        = spec.slots;
        d.result_fields            = spec.fields;
        return ovspatialquery_create_overlap_plan(scene.instance, &d, &plan).status;
    }
    }
    return OVSPATIALQUERY_API_INTERNAL;
}

ovspatialquery_batch_plan_handle_t requirePlan(const Scene& scene, const PlanSpec& spec)
{
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(createPlan(scene, spec, plan) == OVSPATIALQUERY_API_SUCCESS);
    return plan;
}

struct Launched
{
    ovspatialquery_enqueue_result_t operation{};
    Columns                         reservation;
};

// Enqueues `queryCount` of the rows on `stream` (a CPU plan takes none).
Launched launch(const Scene& scene, ovspatialquery_batch_plan_handle_t plan, Family family, Rows& rows,
                CudaDriver::Stream stream, uint64_t queryCount = kRows,
                ovspatialquery_ordinal_selection_t ordinal = {OVSPATIALQUERY_ORDINAL_HEAD, 0})
{
    const bool                       onCuda = stream != nullptr;
    const ovspatialquery_cuda_sync_t sync{onCuda ? 1u : 0u, reinterpret_cast<uintptr_t>(stream), 0u};
    InputArray* const columns[] = {&rows.startColumn, &rows.centerColumn, &rows.directionColumn, &rows.rotationColumn,
                                   &rows.maximumColumn};
    void*             data[std::size(columns)]{};
    for (size_t index = 0u; index < std::size(columns); ++index)
    {
        // A zero-row column carries null data.
        data[index]                       = columns[index]->value.tensor.data;
        columns[index]->shape[0]          = static_cast<int64_t>(queryCount);
        columns[index]->value.tensor.data = queryCount == 0u ? nullptr : data[index];
    }
    Launched out;
    switch (family)
    {
    case Family::raycast: {
        ovspatialquery_raycast_batch_t b{};
        b.ordinal         = ordinal;
        b.query_count     = queryCount;
        b.origins         = &rows.startColumn.value;
        b.unit_directions = &rows.directionColumn.value;
        b.max_distances   = &rows.maximumColumn.value;
        b.cuda_sync       = sync;
        ovspatialquery_raycast_batch_reservation_t r{};
        out.operation   = ovspatialquery_enqueue_raycast_plan(scene.instance, plan, &b, &r);
        out.reservation = flatten(r);
        break;
    }
    case Family::sphereSweep:
    case Family::boxSweep: {
        ovspatialquery_sphere_sweep_batch_reservation_t r{};
        if (family == Family::sphereSweep)
        {
            ovspatialquery_sphere_sweep_batch_t b{};
            b.ordinal         = ordinal;
            b.query_count     = queryCount;
            b.centers         = &rows.startColumn.value;
            b.unit_directions = &rows.directionColumn.value;
            b.max_distances   = &rows.maximumColumn.value;
            b.cuda_sync       = sync;
            out.operation     = ovspatialquery_enqueue_sphere_sweep_plan(scene.instance, plan, &b, &r);
        }
        else
        {
            ovspatialquery_box_sweep_batch_t b{};
            b.ordinal         = ordinal;
            b.query_count     = queryCount;
            b.centers         = &rows.startColumn.value;
            b.rotations_xyzw  = &rows.rotationColumn.value;
            b.unit_directions = &rows.directionColumn.value;
            b.max_distances   = &rows.maximumColumn.value;
            b.cuda_sync       = sync;
            out.operation     = ovspatialquery_enqueue_box_sweep_plan(scene.instance, plan, &b, &r);
        }
        out.reservation = flatten(r);
        break;
    }
    case Family::overlap: {
        ovspatialquery_overlap_batch_t b{};
        b.ordinal      = ordinal;
        b.query_count  = queryCount;
        b.translations = &rows.centerColumn.value;
        b.cuda_sync    = sync;
        ovspatialquery_overlap_batch_reservation_t r{};
        out.operation   = ovspatialquery_enqueue_overlap_plan(scene.instance, plan, &b, &r);
        out.reservation = flatten(r);
        break;
    }
    }
    for (size_t index = 0u; index < std::size(columns); ++index)
    {
        columns[index]->shape[0]          = static_cast<int64_t>(kRows);
        columns[index]->value.tensor.data = data[index];
    }
    return out;
}

ovspatialquery_op_state_t wait(const Scene& scene, ovspatialquery_op_id_t operation)
{
    ovspatialquery_op_wait_result_t waited{};
    REQUIRE(ovspatialquery_wait_op(scene.instance, operation, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
            OVSPATIALQUERY_API_SUCCESS);
    return waited.state;
}

ovspatialquery_api_status_t fetch(const Scene& scene, Family family, ovspatialquery_op_id_t operation, Columns& out)
{
    ovspatialquery_result_t status{};
    if (family == Family::raycast)
    {
        ovspatialquery_raycast_batch_result_t r{};
        status = ovspatialquery_fetch_raycast_batch(scene.instance, operation, &r);
        out    = flatten(r);
    }
    else if (family == Family::overlap)
    {
        ovspatialquery_overlap_batch_result_t r{};
        status = ovspatialquery_fetch_overlap_batch(scene.instance, operation, &r);
        out    = flatten(r);
    }
    else
    {
        ovspatialquery_sphere_sweep_batch_result_t r{};
        status = family == Family::sphereSweep ? ovspatialquery_fetch_sphere_sweep_batch(scene.instance, operation, &r)
                                               : ovspatialquery_fetch_box_sweep_batch(scene.instance, operation, &r);
        out    = flatten(r);
    }
    return status.status;
}

Columns completedColumns(const Scene& scene, Family family, ovspatialquery_op_id_t operation)
{
    REQUIRE(wait(scene, operation) == OVSPATIALQUERY_OP_COMPLETED);
    Columns result;
    REQUIRE(fetch(scene, family, operation, result) == OVSPATIALQUERY_API_SUCCESS);
    return result;
}

// Every present tensor's bytes: a device tensor through a download on
// `stream`, a host tensor read in place.
std::vector<std::vector<uint8_t>> columnBytes(CudaDriver& cuda, CudaDriver::Stream stream, const Columns& columns)
{
    std::vector<std::vector<uint8_t>> bytes(columns.tensors.size());
    const CudaDriver::Event           done = cuda.createEvent();
    for (size_t index = 0u; index < columns.tensors.size(); ++index)
    {
        const DLTensor& tensor = columns.tensors[index];
        if (tensor.shape == nullptr)
        {
            continue;
        }
        bytes[index].resize(tensorBytes(tensor));
        if (bytes[index].empty())
        {
            continue;
        }
        if (tensor.device.device_type == kDLCUDA)
        {
            cuda.downloadAsync(bytes[index].data(), deviceAddress(tensor), bytes[index].size(), stream);
        }
        else
        {
            std::memcpy(bytes[index].data(), hostData(tensor), bytes[index].size());
        }
    }
    cuda.record(done, stream);
    cuda.synchronize(done);
    cuda.destroyEvent(done);
    return bytes;
}

// An exact raycast's or sweep's source face, corners and barycentrics are
// unspecified on a row whose feature kind is NONE, so they are compared only on
// TRIANGLE rows. The feature group is the column struct's last four tensors
// before its four reserved ones.
void maskUnspecifiedFeatureRows(Family family, std::vector<std::vector<uint8_t>>& bytes)
{
    const size_t kinds = family == Family::raycast ? 12u : 13u;
    if (family == Family::overlap || bytes[kinds].empty())
    {
        return;
    }
    const size_t rows = bytes[kinds].size() / sizeof(uint32_t);
    for (size_t row = 0u; row < rows; ++row)
    {
        uint32_t kind = 0u;
        std::memcpy(&kind, bytes[kinds].data() + row * sizeof(uint32_t), sizeof(kind));
        if (kind == OVSPATIALQUERY_GEOMETRY_FEATURE_TRIANGLE)
        {
            continue;
        }
        for (size_t column = kinds + 1u; column <= kinds + 3u; ++column)
        {
            const size_t rowBytes = bytes[column].size() / rows;
            std::fill_n(bytes[column].begin() + static_cast<std::ptrdiff_t>(row * rowBytes), rowBytes, uint8_t{0u});
        }
    }
}

void retireAfter(CudaDriver& cuda, CudaDriver::Stream consumer, const Scene& scene, ovspatialquery_op_id_t& operation)
{
    const CudaDriver::Event consumerDone = cuda.createEvent();
    cuda.record(consumerDone, consumer);
    ovspatialquery_cuda_event_t done{reinterpret_cast<uintptr_t>(consumerDone)};
    REQUIRE(ovspatialquery_destroy_op_after_cuda(scene.instance, &operation, &done).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(cuda.forgetTransferredEvent(consumerDone));
}

// For one family, hit mode and precision: the mapped plan's descriptors name
// pinned host memory, its reservation and fetch share storage, and the rows
// the host reads in place equal the plain plan's device rows byte for byte.
void testMappedRowsEqualDeviceRows(const Scene& scene, CudaDriver& cuda, CudaDriver::Stream stream, Rows& rows,
                                   PlanSpec spec)
{
    spec.fields                                       = allFields(spec);
    const ovspatialquery_batch_plan_handle_t plain    = requirePlan(scene, spec);
    PlanSpec                                 hostSpec = spec;
    hostSpec.fields |= OVSPATIALQUERY_BATCH_RESULT_HOST_MAPPED;
    const ovspatialquery_batch_plan_handle_t mapped = requirePlan(scene, hostSpec);

    Launched device = launch(scene, plain, spec.family, rows, stream);
    Launched host   = launch(scene, mapped, spec.family, rows, stream);
    REQUIRE(device.operation.status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(host.operation.status == OVSPATIALQUERY_API_SUCCESS);
    const Columns deviceResult = completedColumns(scene, spec.family, device.operation.op_id);
    const Columns hostResult   = completedColumns(scene, spec.family, host.operation.op_id);

    REQUIRE(sameDevice(hostResult.common.execution_device, kCuda));
    REQUIRE(hostResult.common.result_ready.value != 0u);
    REQUIRE(hostResult.tensors.size() == deviceResult.tensors.size());
    for (size_t index = 0u; index < hostResult.tensors.size(); ++index)
    {
        const DLTensor& reserved = host.reservation.tensors[index];
        const DLTensor& fetched  = hostResult.tensors[index];
        const DLTensor& plainOne = deviceResult.tensors[index];
        REQUIRE((fetched.shape == nullptr) == (plainOne.shape == nullptr));
        REQUIRE((reserved.shape == nullptr) == (fetched.shape == nullptr));
        if (fetched.shape == nullptr)
        {
            continue;
        }
        REQUIRE(sameDevice(fetched.device, kMapped) && sameDevice(reserved.device, kMapped));
        REQUIRE(sameDevice(plainOne.device, kCuda));
        REQUIRE(fetched.data != nullptr && fetched.data == reserved.data &&
                fetched.byte_offset == reserved.byte_offset);
        REQUIRE(fetched.dtype.code == plainOne.dtype.code && fetched.dtype.bits == plainOne.dtype.bits &&
                fetched.ndim == plainOne.ndim);
        for (int32_t axis = 0; axis < fetched.ndim; ++axis)
        {
            REQUIRE(fetched.shape[axis] == plainOne.shape[axis]);
            REQUIRE(fetched.strides == nullptr || plainOne.strides == nullptr ||
                    fetched.strides[axis] == plainOne.strides[axis]);
        }
    }

    // The host reads the mapped rows with no stream wait: the wait above observed
    // the terminal state, which implies result_ready completed.
    std::vector<std::vector<uint8_t>> hostBytes = columnBytes(cuda, stream, hostResult);
    cuda.wait(stream, reinterpret_cast<CudaDriver::Event>(deviceResult.common.result_ready.value));
    std::vector<std::vector<uint8_t>> deviceBytes = columnBytes(cuda, stream, deviceResult);
    maskUnspecifiedFeatureRows(spec.family, hostBytes);
    maskUnspecifiedFeatureRows(spec.family, deviceBytes);
    for (size_t index = 0u; index < hostBytes.size(); ++index)
    {
        if (hostBytes[index] != deviceBytes[index])
        {
            const auto differs = std::mismatch(hostBytes[index].begin(), hostBytes[index].end(),
                                               deviceBytes[index].begin(), deviceBytes[index].end());
            std::fprintf(stderr, "%s: column %zu of %zu bytes differs at byte %td\n", gPhase, index,
                         hostBytes[index].size(), differs.first - hostBytes[index].begin());
        }
    }
    REQUIRE(hostBytes == deviceBytes);
    const uint64_t stored = hostValue<uint64_t>(storedHitCount(hostResult), 0u);
    REQUIRE(stored >= (spec.mode == OVSPATIALQUERY_HIT_MODE_MULTIPLE ? kRows + 1u : kRows / 2u));
    // An overlap between two spheres and the wall keeps two of its three hits.
    const DLTensor& truncated = hostResult.tensors[4];
    REQUIRE(spec.family != Family::overlap || spec.mode != OVSPATIALQUERY_HIT_MODE_MULTIPLE ||
            hostValue<uint8_t>(truncated, 1u) != 0u);

    // A host reader retires the mapped launch with destroy_op; the plain launch
    // still needs consumer_done.
    REQUIRE(ovspatialquery_destroy_op(scene.instance, &device.operation.op_id).status == OVSPATIALQUERY_API_BUSY);
    REQUIRE(ovspatialquery_destroy_op(scene.instance, &host.operation.op_id).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(host.operation.op_id == OVSPATIALQUERY_INVALID_OP);
    retireAfter(cuda, stream, scene, device.operation.op_id);
    REQUIRE(ovspatialquery_destroy_batch_plan(scene.instance, mapped).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(destroyPlanWhenRetired(scene.instance, plain));
}

// A launch that fails after its kernels publishes the safe empty result into
// the mapped columns, overwriting the rows its own publish pass wrote; a
// zero-row launch on the same slot then publishes H = 0; destroy_op frees the
// one slot each time, and destroy_op_after_cuda still retires a mapped launch.
void testMappedFailureAndSlotReuse(const Scene& scene, CudaDriver& cuda, CudaDriver::Stream stream, Rows& rows)
{
    PlanSpec spec;
    spec.family                                   = Family::overlap;
    spec.mode                                     = OVSPATIALQUERY_HIT_MODE_MULTIPLE;
    spec.fields                                   = OVSPATIALQUERY_BATCH_RESULT_HOST_MAPPED;
    const ovspatialquery_batch_plan_handle_t plan = requirePlan(scene, spec);

    phase("mapped launch failed after its kernels");
    ovspatialquery_test_set_cuda_failure_point(ovspatialquery::detail::kCudaFailureFamilyLaunchAfterKernels);
    Launched failed = launch(scene, plan, spec.family, rows, stream);
    REQUIRE(failed.operation.status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(wait(scene, failed.operation.op_id) == OVSPATIALQUERY_OP_FAILURE);
    Columns unavailable;
    REQUIRE(fetch(scene, spec.family, failed.operation.op_id, unavailable) == OVSPATIALQUERY_API_OPERATION_FAILED);
    const Columns& reserved = failed.reservation;
    REQUIRE(sameDevice(queryStatus(reserved).device, kMapped));
    REQUIRE(hostValue<uint64_t>(storedHitCount(reserved), 0u) == 0u);
    for (size_t row = 0u; row < kRows; ++row)
    {
        REQUIRE(hostValue<uint32_t>(queryStatus(reserved), row) == OVSPATIALQUERY_BATCH_ROW_FAILED);
        REQUIRE(hostValue<uint64_t>(reserved.tensors[2], row) == 0u);
        REQUIRE(hostValue<uint64_t>(reserved.tensors[3], row) == 0u);
        REQUIRE(hostValue<uint8_t>(reserved.tensors[4], row) == 0u);
    }
    REQUIRE(ovspatialquery_destroy_op(scene.instance, &failed.operation.op_id).status == OVSPATIALQUERY_API_SUCCESS);

    phase("mapped zero-row launch on a slot that stored hits");
    Launched full = launch(scene, plan, spec.family, rows, stream);
    REQUIRE(full.operation.status == OVSPATIALQUERY_API_SUCCESS);
    const Columns fullResult = completedColumns(scene, spec.family, full.operation.op_id);
    REQUIRE(hostValue<uint64_t>(storedHitCount(fullResult), 0u) > kRows);
    REQUIRE(ovspatialquery_destroy_op(scene.instance, &full.operation.op_id).status == OVSPATIALQUERY_API_SUCCESS);
    Launched empty = launch(scene, plan, spec.family, rows, stream, 0u);
    REQUIRE(empty.operation.status == OVSPATIALQUERY_API_SUCCESS);
    const Columns emptyResult = completedColumns(scene, spec.family, empty.operation.op_id);
    REQUIRE(storedHitCount(emptyResult).data == storedHitCount(fullResult).data);
    REQUIRE(hostValue<uint64_t>(storedHitCount(emptyResult), 0u) == 0u);

    phase("mapped launch retired after a device consumer");
    retireAfter(cuda, stream, scene, empty.operation.op_id);
    REQUIRE(destroyPlanWhenRetired(scene.instance, plan));
}

// H of a completed result, read in place from mapped columns or downloaded.
uint64_t storedHits(CudaDriver& cuda, CudaDriver::Stream stream, const Columns& columns)
{
    uint64_t value = 0u;
    std::memcpy(&value, columnBytes(cuda, stream, columns)[1].data(), sizeof(value));
    return value;
}

ovspatialquery_api_status_t failureCause(const Scene& scene, ovspatialquery_op_id_t operation)
{
    REQUIRE(wait(scene, operation) == OVSPATIALQUERY_OP_FAILURE);
    ovspatialquery_op_status_t status{};
    REQUIRE(ovspatialquery_query_op_status(scene.instance, operation, &status).status == OVSPATIALQUERY_API_SUCCESS);
    return status.terminal_status;
}

// A launch whose ready poll fails, as a failed switch to the plan's context does, cannot yet
// know that the device is done with its staging, its slot and a mapped plan's columns, so it
// waits for its ready event before it reports a terminal state: mapped or not, the launch
// completes with every row written and its slot serves the next launch. The workers stay
// paused until the failure is armed, so the first poll of whichever thread runs the launch
// takes it while the gate on the stream still holds the launch.
void testPollErrorWaitsForTheDevice(CudaDriver& cuda, bool mapped)
{
    phase(mapped ? "mapped launch whose ready poll fails" : "device launch whose ready poll fails");
    Scene                    scene(UINT64_C(64) << 20, false);
    const CudaDriver::Stream gated = cuda.createStream();
    Rows                     rows(&cuda, gated);
    PlanSpec                 spec;
    spec.fields = OVSPATIALQUERY_BATCH_RESULT_DISTANCE;
    if (mapped)
    {
        spec.fields |= OVSPATIALQUERY_BATCH_RESULT_HOST_MAPPED;
    }
    // A plain plan's slot returns only once its consumer event is reaped, so each of its
    // four launches has its own; a mapped plan's every launch reuses one.
    spec.slots                                      = mapped ? 1u : 4u;
    const ovspatialquery_batch_plan_handle_t plan   = requirePlan(scene, spec);
    const auto                               retire = [&](ovspatialquery_op_id_t& operation)
    {
        if (mapped)
        {
            REQUIRE(ovspatialquery_destroy_op(scene.instance, &operation).status == OVSPATIALQUERY_API_SUCCESS);
        }
        else
        {
            retireAfter(cuda, gated, scene, operation);
        }
    };
    // A first launch stages the catalogue and the copied inputs' buffers, whose allocations
    // synchronize with the device, so the held launches below allocate nothing. A zero-row
    // launch then leaves H at 0 in the slot, which a read before a held launch ends sees.
    for (const uint64_t queryCount : {kRows, uint64_t{0u}})
    {
        Launched warm = launch(scene, plan, spec.family, rows, gated, queryCount);
        REQUIRE(storedHits(cuda, gated, completedColumns(scene, spec.family, warm.operation.op_id)) ==
                (queryCount == 0u ? 0u : kRows / 2u));
        retire(warm.operation.op_id);
    }
    const auto launchHeld = [&](uint32_t failurePoint)
    {
        REQUIRE(ovspatialquery::detail::setInstanceOperationWorkersPausedForTest(scene.instance, true) ==
                OVSPATIALQUERY_API_SUCCESS);
        cuda.enqueueDelay(gated, UINT64_C(1000000000));
        Launched held = launch(scene, plan, spec.family, rows, gated);
        REQUIRE(held.operation.status == OVSPATIALQUERY_API_SUCCESS);
        ovspatialquery_test_set_cuda_failure_point(failurePoint);
        REQUIRE(ovspatialquery::detail::setInstanceOperationWorkersPausedForTest(scene.instance, false) ==
                OVSPATIALQUERY_API_SUCCESS);
        return held;
    };
    Launched held = launchHeld(ovspatialquery::detail::kCudaFailureResultPollContextSelection);
    // Four spheres sit in front of the even rows.
    REQUIRE(storedHits(cuda, gated, completedColumns(scene, spec.family, held.operation.op_id)) == kRows / 2u);
    retire(held.operation.op_id);
    Launched next = launch(scene, plan, spec.family, rows, gated, 0u);
    REQUIRE(storedHits(cuda, gated, completedColumns(scene, spec.family, next.operation.op_id)) == 0u);
    retire(next.operation.op_id);
    if (mapped)
    {
        // When the confirming wait fails too, the launch reports device loss, which promises
        // nothing about its columns, but the device may still write them: the plan ends a later
        // launch with device loss without touching the device, and keeps its page-locked memory
        // past its destruction, which starts while the gate still holds the launch (freeing the
        // plan's idle device buffers then waits for the device). The held rows land there.
        phase("mapped launch whose completion cannot be confirmed");
        Launched                unconfirmed = launchHeld(ovspatialquery::detail::kCudaFailureResultConfirmation);
        const CudaDriver::Event heldDone    = cuda.createEvent();
        cuda.record(heldDone, gated);
        REQUIRE(failureCause(scene, unconfirmed.operation.op_id) == OVSPATIALQUERY_API_DEVICE_LOST);
        const DLTensor heldCount = storedHitCount(unconfirmed.reservation);
        REQUIRE(ovspatialquery_destroy_op(scene.instance, &unconfirmed.operation.op_id).status ==
                OVSPATIALQUERY_API_SUCCESS);
        Launched later = launch(scene, plan, spec.family, rows, gated);
        REQUIRE(later.operation.status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(failureCause(scene, later.operation.op_id) == OVSPATIALQUERY_API_DEVICE_LOST);
        REQUIRE(ovspatialquery_destroy_op(scene.instance, &later.operation.op_id).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(!cuda.isComplete(heldDone));
        REQUIRE(ovspatialquery_destroy_batch_plan(scene.instance, plan).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(cuda.isPageLocked(heldCount.data));
        cuda.synchronize(heldDone);
        REQUIRE(hostValue<uint64_t>(heldCount, 0u) == kRows / 2u);
        cuda.destroyEvent(heldDone);
    }
    else
    {
        REQUIRE(destroyPlanWhenRetired(scene.instance, plan));
    }
    cuda.destroyStream(gated);
    scene.close();
}

// An ADMITTED_EXACT launch that selects a device-prepared publication which
// then fails reads FAILED rows and H = 0 in its mapped columns.
void testMappedLaunchBehindFailingPublication(CudaDriver& cuda)
{
    phase("mapped launch behind a failing device publication");
    Scene                    scene(UINT64_C(64) << 20, false);
    const CudaDriver::Stream producer = cuda.createStream();
    // Uploaded before the gate below holds the producer stream.
    Rows                    rows(&cuda, producer);
    const CudaDriver::Event ready  = cuda.createEvent();
    uint64_t                ids[2] = {42u, 99u};
    int32_t                 type   = OVSPATIALQUERY_GEOMETRY_SPHERE;
    uint64_t                value  = kSphereValue;
    double                  position[3]{2.5, 0.0, 0.0};
    float                   rotation[4]{0.0f, 0.0f, 0.0f, 1.0f};
    const auto              upload = [&](const void* data, size_t bytes)
    {
        const CudaDriver::DevicePointer pointer = cuda.allocate(bytes);
        cuda.uploadAsync(pointer, data, bytes, producer);
        return reinterpret_cast<void*>(static_cast<uintptr_t>(pointer));
    };
    void* const deviceIds = upload(ids, sizeof(ids));
    InputArray  known(static_cast<uint64_t*>(deviceIds), 1, kDLUInt, kCuda);
    InputArray  unknown(static_cast<uint64_t*>(deviceIds) + 1, 1, kDLUInt, kCuda);
    InputArray  types(static_cast<int32_t*>(upload(&type, sizeof(type))), 1, kDLInt, kCuda);
    InputArray  values(static_cast<uint64_t*>(upload(&value, sizeof(value))), 1, kDLUInt, kCuda);
    InputArray  positions(static_cast<double*>(upload(position, sizeof(position))), 1, 3, kDLFloat, kCuda);
    InputArray  rotations(static_cast<float*>(upload(rotation, sizeof(rotation))), 1, 4, kDLFloat, kCuda);
    InputArray  noRemovals(static_cast<uint64_t*>(nullptr), 0, kDLUInt, kCuda);
    cuda.record(ready, producer);
    const ovspatialquery_dynamic_execution_t onDevice{
        kCuda,
        {1u, reinterpret_cast<uintptr_t>(producer), reinterpret_cast<uintptr_t>(ready)},
        OVSPATIALQUERY_PREPARE_ON_DEVICE};
    const ovspatialquery_population_desc_t populationDesc{0u, 1u};
    ovspatialquery_population_handle_t     devicePopulation = OVSPATIALQUERY_INVALID_POPULATION;
    REQUIRE(ovspatialquery_create_explicit_population(scene.instance, &populationDesc, &devicePopulation).status ==
            OVSPATIALQUERY_API_SUCCESS);
    const ovspatialquery_explicit_population_set_t set{
        7,
        {&known.value, &types.value, &values.value, &positions.value, &rotations.value, nullptr, nullptr, nullptr},
        &onDevice,
        0u};
    waitAndDestroy(scene.instance,
                   ovspatialquery_enqueue_explicit_population_set(scene.instance, devicePopulation, &set));

    Scene    target = scene;
    PlanSpec spec;
    spec.precision    = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    spec.fields       = OVSPATIALQUERY_BATCH_RESULT_DISTANCE | OVSPATIALQUERY_BATCH_RESULT_HOST_MAPPED;
    target.population = devicePopulation;
    const ovspatialquery_batch_plan_handle_t plan = requirePlan(target, spec);
    // The gate keeps the failing update pending while the launch selects it; the library
    // refuses the selection once it has seen the publication fail.
    cuda.enqueueDelay(producer, UINT64_C(1000000000));
    const ovspatialquery_explicit_population_update_t poses{
        7,
        8,
        OVSPATIALQUERY_EXPLICIT_UPDATE_POSES,
        {&unknown.value, nullptr, nullptr, &positions.value, &rotations.value, nullptr, nullptr, nullptr},
        &noRemovals.value,
        &onDevice};
    ovspatialquery_enqueue_result_t publication =
        ovspatialquery_enqueue_explicit_population_update(scene.instance, devicePopulation, &poses);
    REQUIRE(publication.status == OVSPATIALQUERY_API_SUCCESS);

    Launched launched =
        launch(target, plan, spec.family, rows, producer, kRows, {OVSPATIALQUERY_ORDINAL_ADMITTED_EXACT, 8});
    REQUIRE(launched.operation.status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(wait(scene, launched.operation.op_id) == OVSPATIALQUERY_OP_FAILURE);
    REQUIRE(hostValue<uint64_t>(storedHitCount(launched.reservation), 0u) == 0u);
    for (size_t row = 0u; row < kRows; ++row)
    {
        REQUIRE(hostValue<uint32_t>(queryStatus(launched.reservation), row) == OVSPATIALQUERY_BATCH_ROW_FAILED);
        REQUIRE(hostValue<uint64_t>(launched.reservation.tensors[3], row) == 0u);
    }
    REQUIRE(ovspatialquery_destroy_op(scene.instance, &launched.operation.op_id).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(wait(scene, publication.op_id) == OVSPATIALQUERY_OP_FAILURE);
    REQUIRE(ovspatialquery_destroy_op(scene.instance, &publication.op_id).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_batch_plan(scene.instance, plan).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_population(scene.instance, devicePopulation).status == OVSPATIALQUERY_API_SUCCESS);
    scene.close();
}

// The mapped columns are host bytes: a mapped plan charges at least its result
// bytes more than the plain plan, creation one byte short of that charge
// returns HOST_BUDGET and charges nothing, and the exact charge fits.
void testMappedColumnsAreChargedToTheHostBudget()
{
    phase("calibrate the mapped plan's host charge");
    PlanSpec spec;
    spec.family = Family::raycast;
    spec.mode   = OVSPATIALQUERY_HIT_MODE_MULTIPLE;
    spec.fields = OVSPATIALQUERY_BATCH_RESULT_POSITION | OVSPATIALQUERY_BATCH_RESULT_NORMAL |
                  OVSPATIALQUERY_BATCH_RESULT_DISTANCE;
    spec.slots          = 4u;
    PlanSpec mappedSpec = spec;
    mappedSpec.fields |= OVSPATIALQUERY_BATCH_RESULT_HOST_MAPPED;
    Scene          calibration(UINT64_C(64) << 20, false);
    const uint64_t base   = hostUsage(calibration.instance);
    const auto     charge = [&](const PlanSpec& planSpec)
    {
        const ovspatialquery_batch_plan_handle_t plan  = requirePlan(calibration, planSpec);
        const uint64_t                           bytes = hostUsage(calibration.instance) - base;
        REQUIRE(ovspatialquery_destroy_batch_plan(calibration.instance, plan).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(hostUsage(calibration.instance) == base);
        return bytes;
    };
    const uint64_t plainCharge  = charge(spec);
    const uint64_t mappedCharge = charge(mappedSpec);
    calibration.close();
    // Per slot: status, offsets, counts and truncation per row, H, and per hit an
    // object ID, a position, a normal and a distance.
    const uint64_t hits        = kRows * kMaximumHits;
    const uint64_t resultBytes = spec.slots * (kRows * (4u + 8u + 8u + 1u) + 8u + hits * (8u + 24u + 12u + 8u));
    REQUIRE(mappedCharge >= plainCharge + resultBytes);

    phase("refuse a mapped plan one byte short of its charge");
    Scene short_(base + mappedCharge - 1u, false);
    REQUIRE(hostUsage(short_.instance) == base);
    ovspatialquery_batch_plan_handle_t refused = {UINT64_C(0xa5a5)};
    REQUIRE(createPlan(short_, mappedSpec, refused) == OVSPATIALQUERY_API_HOST_BUDGET);
    REQUIRE(refused == OVSPATIALQUERY_INVALID_BATCH_PLAN);
    REQUIRE(hostUsage(short_.instance) == base);
    // The plain plan fits in the same budget, so the refusal is the mapped columns'.
    const ovspatialquery_batch_plan_handle_t plain = requirePlan(short_, spec);
    REQUIRE(ovspatialquery_destroy_batch_plan(short_.instance, plain).status == OVSPATIALQUERY_API_SUCCESS);
    short_.close();

    phase("admit a mapped plan at exactly its charge");
    Scene                                    exact(base + mappedCharge, false);
    const ovspatialquery_batch_plan_handle_t admitted = requirePlan(exact, mappedSpec);
    REQUIRE(hostUsage(exact.instance) == base + mappedCharge);
    REQUIRE(ovspatialquery_destroy_batch_plan(exact.instance, admitted).status == OVSPATIALQUERY_API_SUCCESS);
    exact.close();
}

// A CPU plan accepts HOST_MAPPED and ignores it: the same rows on the CPU, the
// same host charge. Bit 256 stays unknown to every family on both devices.
void testCpuPlansIgnoreTheBitAndUnknownBitsStayRefused(const Scene& scene)
{
    phase("CPU plan ignores HOST_MAPPED");
    Rows     rows(nullptr, nullptr);
    PlanSpec spec;
    spec.family         = Family::raycast;
    spec.mode           = OVSPATIALQUERY_HIT_MODE_MULTIPLE;
    spec.device         = kCpu;
    spec.fields         = allFields(spec);
    PlanSpec mappedSpec = spec;
    mappedSpec.fields |= OVSPATIALQUERY_BATCH_RESULT_HOST_MAPPED;
    const uint64_t                           base        = hostUsage(scene.instance);
    const ovspatialquery_batch_plan_handle_t plain       = requirePlan(scene, spec);
    const uint64_t                           plainCharge = hostUsage(scene.instance) - base;
    const ovspatialquery_batch_plan_handle_t mapped      = requirePlan(scene, mappedSpec);
    REQUIRE(hostUsage(scene.instance) - base == 2u * plainCharge);
    Launched      first        = launch(scene, plain, spec.family, rows, nullptr);
    Launched      second       = launch(scene, mapped, spec.family, rows, nullptr);
    const Columns plainResult  = completedColumns(scene, spec.family, first.operation.op_id);
    const Columns mappedResult = completedColumns(scene, spec.family, second.operation.op_id);
    REQUIRE(sameDevice(mappedResult.common.execution_device, kCpu));
    for (size_t index = 0u; index < plainResult.tensors.size(); ++index)
    {
        const DLTensor& a = plainResult.tensors[index];
        const DLTensor& b = mappedResult.tensors[index];
        REQUIRE((a.shape == nullptr) == (b.shape == nullptr));
        if (a.shape == nullptr)
        {
            continue;
        }
        REQUIRE(sameDevice(b.device, kCpu) && tensorBytes(a) == tensorBytes(b));
        REQUIRE(std::memcmp(hostData(a), hostData(b), tensorBytes(a)) == 0);
    }
    REQUIRE(hostValue<uint64_t>(storedHitCount(mappedResult), 0u) > kRows);
    REQUIRE(ovspatialquery_destroy_op(scene.instance, &first.operation.op_id).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_op(scene.instance, &second.operation.op_id).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_batch_plan(scene.instance, plain).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_batch_plan(scene.instance, mapped).status == OVSPATIALQUERY_API_SUCCESS);

    phase("bit 256 is refused by every family on both devices");
    for (const DLDevice device : {kCpu, kCuda})
    {
        for (const Family family : {Family::raycast, Family::sphereSweep, Family::boxSweep, Family::overlap})
        {
            PlanSpec unknown;
            unknown.family = family;
            unknown.mode =
                family == Family::overlap ? OVSPATIALQUERY_HIT_MODE_MULTIPLE : OVSPATIALQUERY_HIT_MODE_CLOSEST;
            unknown.device                          = device;
            unknown.fields                          = OVSPATIALQUERY_BATCH_RESULT_HOST_MAPPED | 256u;
            ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
            REQUIRE(createPlan(scene, unknown, plan) == OVSPATIALQUERY_API_INVALID_ARGUMENT);
            REQUIRE(plan == OVSPATIALQUERY_INVALID_BATCH_PLAN);
            unknown.fields = OVSPATIALQUERY_BATCH_RESULT_HOST_MAPPED;
            plan           = requirePlan(scene, unknown);
            REQUIRE(ovspatialquery_destroy_batch_plan(scene.instance, plan).status == OVSPATIALQUERY_API_SUCCESS);
        }
    }
}

} // namespace

int main()
{
    Scene                    scene;
    CudaDriver               cuda(CudaDriver::Ownership::tracked);
    const CudaDriver::Stream stream = cuda.createStream();
    Rows                     rows(&cuda, stream);
    const struct
    {
        Family                           family;
        ovspatialquery_hit_mode_t        mode;
        ovspatialquery_query_precision_t precision;
        const char*                      name;
    } cases[] = {
        {Family::raycast, OVSPATIALQUERY_HIT_MODE_CLOSEST, OVSPATIALQUERY_QUERY_EXACT, "raycast closest"},
        {Family::raycast, OVSPATIALQUERY_HIT_MODE_ANY, OVSPATIALQUERY_QUERY_EXACT, "raycast any"},
        {Family::raycast, OVSPATIALQUERY_HIT_MODE_MULTIPLE, OVSPATIALQUERY_QUERY_EXACT, "raycast multiple"},
        {Family::raycast, OVSPATIALQUERY_HIT_MODE_MULTIPLE, OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE,
         "raycast multiple, bounds candidates"},
        {Family::sphereSweep, OVSPATIALQUERY_HIT_MODE_CLOSEST, OVSPATIALQUERY_QUERY_EXACT, "sphere sweep closest"},
        {Family::sphereSweep, OVSPATIALQUERY_HIT_MODE_ANY, OVSPATIALQUERY_QUERY_EXACT, "sphere sweep any"},
        {Family::sphereSweep, OVSPATIALQUERY_HIT_MODE_MULTIPLE, OVSPATIALQUERY_QUERY_EXACT, "sphere sweep multiple"},
        {Family::boxSweep, OVSPATIALQUERY_HIT_MODE_CLOSEST, OVSPATIALQUERY_QUERY_EXACT, "box sweep closest"},
        {Family::boxSweep, OVSPATIALQUERY_HIT_MODE_ANY, OVSPATIALQUERY_QUERY_EXACT, "box sweep any"},
        {Family::boxSweep, OVSPATIALQUERY_HIT_MODE_MULTIPLE, OVSPATIALQUERY_QUERY_EXACT, "box sweep multiple"},
        {Family::overlap, OVSPATIALQUERY_HIT_MODE_ANY, OVSPATIALQUERY_QUERY_EXACT, "overlap any"},
        {Family::overlap, OVSPATIALQUERY_HIT_MODE_MULTIPLE, OVSPATIALQUERY_QUERY_EXACT, "overlap multiple"},
    };
    for (const auto& entry : cases)
    {
        phase(entry.name);
        PlanSpec spec;
        spec.family    = entry.family;
        spec.mode      = entry.mode;
        spec.precision = entry.precision;
        testMappedRowsEqualDeviceRows(scene, cuda, stream, rows, spec);
    }
    testMappedFailureAndSlotReuse(scene, cuda, stream, rows);
    testCpuPlansIgnoreTheBitAndUnknownBitsStayRefused(scene);
    scene.close();
    testMappedColumnsAreChargedToTheHostBudget();
    testMappedLaunchBehindFailingPublication(cuda);
    testPollErrorWaitsForTheDevice(cuda, false);
    testPollErrorWaitsForTheDevice(cuda, true);
    std::fprintf(stderr, "host mapped CUDA plan test passed\n");
    return 0;
}
