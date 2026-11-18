/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <ovspatialquery/ovspatialquery_explicit.h>

#include "../../support/native/cuda_public_test_helpers.h"
#include "../../support/native/far_sphere_sweep_scene.h"
#include "../../support/native/float64_box_row_scene.h"
#include "../../support/native/hull_box_gap_scene.h"
#include "../../support/native/public_api_inputs.h"
#include "cuda_driver.h"
#include "ovspatialquery_cuda_failure_points.h"
#include "ovspatialquery_family_batch_plan_operation.h"
#include "ovspatialquery_instance_operation_runtime.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <limits>
#include <string>
#include <thread>
#include <utility>
#include <vector>

extern "C" void ovspatialquery_test_set_cuda_failure_point(uint32_t point);
extern "C" bool ovspatialquery_test_hold_cuda_signal_rings(bool held);

namespace
{

const char* gPhase = "initialization";

using namespace ovspatialquery::detail;

void requireAt(bool condition, int line)
{
    if (!condition)
    {
        std::fprintf(stderr, "public CUDA family API failure at line %d: %s\n", line, gPhase);
        std::fflush(stderr);
        std::_Exit(EXIT_FAILURE);
    }
}

#undef REQUIRE
#define REQUIRE(condition) requireAt((condition), __LINE__)

constexpr DLDevice kCudaDevice{kDLCUDA, 0};

// The element pointer of a device column, for an InputArray on kCudaDevice;
// T names the element type the array describes.
template <typename T>
const T* devicePointer(CudaDriver::DevicePointer pointer)
{
    return reinterpret_cast<const T*>(static_cast<uintptr_t>(pointer));
}

// Allocates device memory for `values` and queues their upload on `stream`.
template <typename T>
CudaDriver::DevicePointer upload(CudaDriver& cuda, const T& values, CudaDriver::Stream stream)
{
    const CudaDriver::DevicePointer pointer = cuda.allocate(sizeof(values));
    cuda.uploadAsync(pointer, &values, sizeof(values), stream);
    return pointer;
}

// Retires a CUDA operation once every read queued on `consumer` is done: the
// consumer_done event passes to the library, which destroys it, so the
// tracked driver forgets it.
void retireAfterConsumer(CudaDriver& cuda, CudaDriver::Stream consumer, ovspatialquery_instance_handle_t instance,
                         ovspatialquery_op_id_t& operation)
{
    const CudaDriver::Event consumerDone = cuda.createEvent();
    cuda.record(consumerDone, consumer);
    cuda.synchronize(consumerDone);
    ovspatialquery_cuda_event_t done{reinterpret_cast<uintptr_t>(consumerDone)};
    REQUIRE(ovspatialquery_destroy_op_after_cuda(instance, &operation, &done).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(operation == OVSPATIALQUERY_INVALID_OP && done.value == 0u);
    REQUIRE(cuda.forgetTransferredEvent(consumerDone));
}

template <typename Result, typename Fetch>
void verifyRecoverableCudaFailure(CudaDriver& cuda, CudaDriver::Stream consumer,
                                  ovspatialquery_instance_handle_t instance, ovspatialquery_op_id_t& operation,
                                  ovspatialquery_cuda_event_t              resultReady,
                                  const ovspatialquery_batch_hit_ranges_t& ranges, Fetch fetch)
{
    ovspatialquery_op_wait_result_t waited{};
    REQUIRE(ovspatialquery_wait_op(instance, operation, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(waited.state == OVSPATIALQUERY_OP_FAILURE);
    cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(resultReady.value));
    uint32_t rowStatus = UINT32_MAX;
    uint64_t stored    = UINT64_MAX;
    uint64_t offset    = UINT64_MAX;
    uint64_t count     = UINT64_MAX;
    uint8_t  truncated = UINT8_MAX;
    cuda.downloadAsync(&rowStatus, deviceAddress(ranges.query_status), sizeof(rowStatus), consumer);
    cuda.downloadAsync(&stored, deviceAddress(ranges.stored_hit_count), sizeof(stored), consumer);
    cuda.downloadAsync(&offset, deviceAddress(ranges.hit_offsets), sizeof(offset), consumer);
    cuda.downloadAsync(&count, deviceAddress(ranges.hit_counts), sizeof(count), consumer);
    cuda.downloadAsync(&truncated, deviceAddress(ranges.truncated), sizeof(truncated), consumer);

    Result unavailable{};
    std::memset(&unavailable, 0xa5, sizeof(unavailable));
    const Result sentinel = unavailable;
    REQUIRE(fetch(&unavailable).status == OVSPATIALQUERY_API_OPERATION_FAILED);
    REQUIRE(std::memcmp(&unavailable, &sentinel, sizeof(unavailable)) == 0);
    REQUIRE(fetch(&unavailable).status == OVSPATIALQUERY_API_OPERATION_FAILED);
    REQUIRE(std::memcmp(&unavailable, &sentinel, sizeof(unavailable)) == 0);
    REQUIRE(ovspatialquery_destroy_op(instance, &operation).status == OVSPATIALQUERY_API_BUSY);
    retireAfterConsumer(cuda, consumer, instance, operation);
    REQUIRE(rowStatus == OVSPATIALQUERY_BATCH_ROW_FAILED);
    REQUIRE(stored == 0u && offset == 0u && count == 0u && truncated == 0u);
    REQUIRE(transferredEventDestroysWereInvoked());
}

struct ExactMeshFixture
{
    // A budget of 64 MiB on each of the first `cudaDevices` CUDA devices.
    explicit ExactMeshFixture(uint64_t hostByteBudget = UINT64_C(64) << 20, uint32_t cudaDevices = 1u,
                              uint32_t maxCpuConcurrency = 1u)
    {
        gPhase = "create instance";
        const ovspatialquery_device_budget_t budgets[2]{{{kDLCUDA, 0}, UINT64_C(64) << 20},
                                                        {{kDLCUDA, 1}, UINT64_C(64) << 20}};
        const ovspatialquery_instance_desc_t desc{4u, maxCpuConcurrency, hostByteBudget, budgets, cudaDevices};
        REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
        registerMesh(meshValue, 0.0f);

        gPhase = "publish exact mesh population";
        const ovspatialquery_population_desc_t populationDesc{2u, 1u};
        REQUIRE(ovspatialquery_create_explicit_population(instance, &populationDesc, &population).status ==
                OVSPATIALQUERY_API_SUCCESS);
        int32_t                                        geometryType   = OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH;
        double                                         translation[3] = {4.0, 0.0, 0.0};
        float                                          rotation[4]    = {0.0f, 0.0f, 0.0f, 1.0f};
        uint64_t                                       bits[2]        = {0u, UINT64_C(0x20)}; // bit 101
        InputArray                                     objects(&objectValue, 1, kDLUInt);
        InputArray                                     types(&geometryType, 1, kDLInt);
        InputArray                                     geometryValues(&meshValue, 1, kDLUInt);
        InputArray                                     positions(translation, 1, 3, kDLFloat);
        InputArray                                     rotations(rotation, 1, 4, kDLFloat);
        InputArray                                     participation(bits, 1, 2, kDLUInt);
        const ovspatialquery_explicit_population_set_t set{sourceOrdinal,
                                                           {&objects.value, &types.value, &geometryValues.value,
                                                            &positions.value, &rotations.value, nullptr,
                                                            &participation.value, nullptr},
                                                           nullptr,
                                                           0u};
        waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));
    }

    // A double-sided triangle in the local plane x = `x`, spanning y and z in [-1, 1].
    void registerMesh(uint64_t value, float x)
    {
        gPhase                 = "register exact mesh";
        uint64_t zero          = 0u;
        uint64_t vertexCount   = 3u;
        uint64_t triangleCount = 1u;
        float    vertices[9]   = {
            x, -1.0f, -1.0f, x, 1.0f, -1.0f, x, 0.0f, 1.0f,
        };
        uint32_t                                              triangles[3] = {0u, 1u, 2u};
        uint8_t                                               doubleSided  = 1u;
        InputArray                                            values(&value, 1, kDLUInt);
        InputArray                                            vertexOffsets(&zero, 1, kDLUInt);
        InputArray                                            vertexCounts(&vertexCount, 1, kDLUInt);
        InputArray                                            triangleOffsets(&zero, 1, kDLUInt);
        InputArray                                            triangleCounts(&triangleCount, 1, kDLUInt);
        InputArray                                            vertexData(vertices, 3, 3, kDLFloat);
        InputArray                                            triangleData(triangles, 1, 3, kDLUInt);
        InputArray                                            sidedness(&doubleSided, 1, kDLUInt);
        const ovspatialquery_triangle_mesh_definition_table_t meshes{&values.value,
                                                                     &vertexOffsets.value,
                                                                     &vertexCounts.value,
                                                                     &triangleOffsets.value,
                                                                     &triangleCounts.value,
                                                                     &vertexData.value,
                                                                     &triangleData.value,
                                                                     &sidedness.value,
                                                                     nullptr,
                                                                     nullptr,
                                                                     nullptr};
        waitAndDestroy(instance, ovspatialquery_enqueue_triangle_mesh_registration(instance, &meshes));
        registeredMeshes.push_back(value);
    }

    // Moves the object to x, and places mesh `geometry` instead when it is nonzero.
    void publish(ovspatialquery_ordinal_t ordinal, double x, uint64_t geometry = 0u)
    {
        uint64_t                                          placed         = geometry != 0u ? geometry : meshValue;
        int32_t                                           geometryType   = OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH;
        double                                            translation[3] = {x, 0.0, 0.0};
        float                                             rotation[4]    = {0.0f, 0.0f, 0.0f, 1.0f};
        uint64_t                                          bits[2]        = {0u, UINT64_C(0x20)};
        InputArray                                        objects(&objectValue, 1, kDLUInt);
        InputArray                                        types(&geometryType, 1, kDLInt);
        InputArray                                        geometryValues(&placed, 1, kDLUInt);
        InputArray                                        positions(translation, 1, 3, kDLFloat);
        InputArray                                        rotations(rotation, 1, 4, kDLFloat);
        InputArray                                        participation(bits, 1, 2, kDLUInt);
        uint64_t                                          emptyRemoved = 0u;
        InputArray                                        removed(&emptyRemoved, 0, kDLUInt);
        const ovspatialquery_explicit_population_update_t update{
            ordinal - 1,
            ordinal,
            OVSPATIALQUERY_EXPLICIT_UPDATE_COMPLETE_ROWS,
            {&objects.value, &types.value, &geometryValues.value, &positions.value, &rotations.value, nullptr,
             &participation.value, nullptr},
            &removed.value,
            nullptr,
        };
        gPhase = "publish retained generation while CUDA plan exists";
        waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_update(instance, population, &update));
    }

    void close()
    {
        REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
        for (const uint64_t value : registeredMeshes)
        {
            REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, value})
                        .status == OVSPATIALQUERY_API_SUCCESS);
        }
        REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
        instance = OVSPATIALQUERY_INVALID_INSTANCE;
    }

    ovspatialquery_instance_handle_t   instance      = OVSPATIALQUERY_INVALID_INSTANCE;
    ovspatialquery_population_handle_t population    = OVSPATIALQUERY_INVALID_POPULATION;
    const uint64_t                     objectValue   = UINT64_C(0xabcdef0123456789);
    uint64_t                           meshValue     = 77u;
    const ovspatialquery_ordinal_t     sourceOrdinal = 9;
    std::vector<uint64_t>              registeredMeshes;
};

void testPublicExactCudaRaycast()
{
    ExactMeshFixture         fixture;
    CudaDriver               cuda(CudaDriver::Ownership::tracked);
    const CudaDriver::Stream producer             = cuda.createStream();
    const CudaDriver::Stream consumer             = cuda.createStream();
    const CudaDriver::Event  queryReady           = cuda.createEvent();
    const CudaDriver::Event  rejectedConsumerDone = cuda.createEvent();
    cuda.record(rejectedConsumerDone, consumer);
    cuda.synchronize(rejectedConsumerDone);

    gPhase = "create public exact CUDA raycast plan";
    ovspatialquery_raycast_plan_desc_t planDesc{};
    planDesc.target                 = populationRef(fixture.population);
    planDesc.hit_mode               = OVSPATIALQUERY_HIT_MODE_CLOSEST;
    planDesc.precision              = OVSPATIALQUERY_QUERY_EXACT;
    planDesc.device                 = {kDLCUDA, 0};
    planDesc.maximum_query_count    = 1u;
    planDesc.maximum_hits_per_query = 1u;
    planDesc.hit_capacity_per_slot  = 1u;
    planDesc.result_slot_count      = 2u; // the empty batch below runs while the first retires
    planDesc.result_fields          = OVSPATIALQUERY_BATCH_RESULT_POSITION | OVSPATIALQUERY_BATCH_RESULT_NORMAL |
                             OVSPATIALQUERY_BATCH_RESULT_DISTANCE | OVSPATIALQUERY_BATCH_RESULT_SOURCE_FEATURE |
                             OVSPATIALQUERY_BATCH_RESULT_GEOMETRY_IDENTITY;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_raycast_plan(fixture.instance, &planDesc, &plan).status ==
            OVSPATIALQUERY_API_SUCCESS);
    fixture.publish(10, 20.0);

    double                          origin[3]             = {0.0, 0.0, 0.0};
    float                           direction[3]          = {1.0f, 0.0f, 0.0f};
    double                          maximumDistance       = 10.0;
    uint64_t                        queryBits[2]          = {0u, UINT64_C(0x20)};
    const CudaDriver::DevicePointer deviceOrigin          = upload(cuda, origin, producer);
    const CudaDriver::DevicePointer deviceDirection       = upload(cuda, direction, producer);
    const CudaDriver::DevicePointer deviceMaximumDistance = upload(cuda, maximumDistance, producer);
    const CudaDriver::DevicePointer deviceQueryBits       = upload(cuda, queryBits, producer);
    // Keep the producer pending until both host-side pending checks finish.
    // The callback uses no CUDA calls; the completed consumer event above also
    // lets the host release it without waiting for other outstanding CUDA work.
    std::promise<void> releaseProducer;
    std::future<void>  producerReleased = releaseProducer.get_future();
    cuda.enqueueHostFunction(
        producer, [](void* state) { static_cast<std::future<void>*>(state)->wait(); }, &producerReleased);
    cuda.record(queryReady, producer);
    InputArray origins(devicePointer<double>(deviceOrigin), 1, 3, kDLFloat, kCudaDevice);
    InputArray directions(devicePointer<float>(deviceDirection), 1, 3, kDLFloat, kCudaDevice);
    InputArray maximumDistances(devicePointer<double>(deviceMaximumDistance), 1, kDLFloat, kCudaDevice);
    InputArray bits(devicePointer<uint64_t>(deviceQueryBits), 1, 2, kDLUInt, kCudaDevice);
    ovspatialquery_raycast_batch_t batch{};
    batch.ordinal.kind           = OVSPATIALQUERY_ORDINAL_RETAINED_EXACT;
    batch.ordinal.source_ordinal = fixture.sourceOrdinal;
    batch.query_count            = 1u;
    batch.query_bits             = &bits.value;
    batch.origins                = &origins.value;
    batch.unit_directions        = &directions.value;
    batch.max_distances          = &maximumDistances.value;
    batch.cuda_sync.has_stream   = 1u;
    batch.cuda_sync.stream       = reinterpret_cast<uintptr_t>(producer);
    batch.cuda_sync.wait_event   = reinterpret_cast<uintptr_t>(queryReady);
    ovspatialquery_raycast_batch_reservation_t reservation{};
    std::memset(&reservation, 0xa5, sizeof(reservation));
    gPhase = "enqueue public exact CUDA raycast";
    ovspatialquery_enqueue_result_t operation =
        ovspatialquery_enqueue_raycast_plan(fixture.instance, plan, &batch, &reservation);
    REQUIRE(operation.status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(operation.op_id != OVSPATIALQUERY_INVALID_OP);
    REQUIRE(reservation.common.source_ordinal == fixture.sourceOrdinal);
    REQUIRE(reservation.common.truth == OVSPATIALQUERY_HIT_EXACT_SURFACE);
    REQUIRE(reservation.common.execution_device.device_type == kDLCUDA);
    REQUIRE(reservation.common.result_ready.value != 0u);
    REQUIRE(reservation.columns.hit_ranges.query_status.shape != nullptr);
    REQUIRE(reservation.columns.hit_ranges.query_status.shape[0] == 1);
    REQUIRE(reservation.columns.object_ids.shape != nullptr);
    REQUIRE(reservation.columns.object_ids.shape[0] == 1);
    REQUIRE(reservation.columns.contact.exact.features.kinds.shape != nullptr);
    const ovspatialquery_raycast_batch_columns_t absentColumns{};
    REQUIRE(std::memcmp(reservation.columns.reserved, absentColumns.reserved, sizeof(absentColumns.reserved)) == 0);
    const ovspatialquery_op_id_t pendingOperation = operation.op_id;
    ovspatialquery_cuda_event_t  rejectedDone{reinterpret_cast<uintptr_t>(rejectedConsumerDone)};
    const uintptr_t              pendingDone               = rejectedDone.value;
    const uint64_t               destroyAttemptsBeforeBusy = transferredEventDestroyAttempts();
    gPhase                                                 = "preserve caller ownership on pending CUDA retirement";
    REQUIRE(ovspatialquery_destroy_op_after_cuda(fixture.instance, &operation.op_id, &rejectedDone).status ==
            OVSPATIALQUERY_API_BUSY);
    REQUIRE(operation.op_id == pendingOperation && rejectedDone.value == pendingDone &&
            transferredEventDestroyAttempts() == destroyAttemptsBeforeBusy);
    cuda.destroyEvent(rejectedConsumerDone);
    REQUIRE(reservation.columns.contact.exact.features.kinds.shape[0] == 1);

    gPhase = "pending public exact CUDA raycast fetch";
    ovspatialquery_raycast_batch_result_t pending{};
    std::memset(&pending, 0xa5, sizeof(pending));
    const ovspatialquery_raycast_batch_result_t pendingSentinel = pending;
    REQUIRE(ovspatialquery_fetch_raycast_batch(fixture.instance, operation.op_id, &pending).status ==
            OVSPATIALQUERY_API_NOT_READY);
    REQUIRE(std::memcmp(&pending, &pendingSentinel, sizeof(pending)) == 0);
    releaseProducer.set_value();

    gPhase = "fetch public exact CUDA raycast";
    ovspatialquery_op_wait_result_t waited{};
    REQUIRE(
        ovspatialquery_wait_op(fixture.instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
        OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(waited.state == OVSPATIALQUERY_OP_COMPLETED);
    ovspatialquery_raycast_batch_result_t result{};
    std::memset(&result, 0xa5, sizeof(result));
    REQUIRE(ovspatialquery_fetch_raycast_batch(fixture.instance, operation.op_id, &result).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.common.source_ordinal == fixture.sourceOrdinal);
    REQUIRE(result.common.result_ready.value == reservation.common.result_ready.value);
    REQUIRE(std::memcmp(result.columns.reserved, absentColumns.reserved, sizeof(absentColumns.reserved)) == 0);
    REQUIRE(result.columns.contact.exact.features.source_face_indices.dtype.code == kDLUInt);
    REQUIRE(result.columns.contact.exact.features.source_face_indices.dtype.bits == 64u);
    REQUIRE(result.columns.object_ids.data == reservation.columns.object_ids.data);
    REQUIRE(result.columns.object_ids.shape != reservation.columns.object_ids.shape);
    REQUIRE(result.columns.object_ids.strides != reservation.columns.object_ids.strides);
    REQUIRE(result.columns.object_ids.shape[0] == 1);

    cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(result.common.result_ready.value));
    uint32_t rowStatus        = UINT32_MAX;
    uint64_t stored           = UINT64_MAX;
    uint64_t offset           = UINT64_MAX;
    uint64_t count            = UINT64_MAX;
    uint8_t  truncated        = UINT8_MAX;
    uint64_t objectId         = 0u;
    uint32_t featureKind      = UINT32_MAX;
    int32_t  geometryType     = 0;
    uint64_t geometryValue    = 0u;
    uint64_t sourceFace       = UINT64_MAX;
    uint32_t sourceCorners[3] = {UINT32_MAX, UINT32_MAX, UINT32_MAX};
    float    barycentrics[3]  = {};
    cuda.downloadAsync(&rowStatus, deviceAddress(result.columns.hit_ranges.query_status), sizeof(rowStatus), consumer);
    cuda.downloadAsync(&stored, deviceAddress(result.columns.hit_ranges.stored_hit_count), sizeof(stored), consumer);
    cuda.downloadAsync(&offset, deviceAddress(result.columns.hit_ranges.hit_offsets), sizeof(offset), consumer);
    cuda.downloadAsync(&count, deviceAddress(result.columns.hit_ranges.hit_counts), sizeof(count), consumer);
    cuda.downloadAsync(&truncated, deviceAddress(result.columns.hit_ranges.truncated), sizeof(truncated), consumer);
    cuda.downloadAsync(&objectId, deviceAddress(result.columns.object_ids), sizeof(objectId), consumer);
    cuda.downloadAsync(&featureKind, deviceAddress(result.columns.contact.exact.features.kinds), sizeof(featureKind),
                       consumer);
    cuda.downloadAsync(&geometryType, deviceAddress(result.columns.geometry.types), sizeof(geometryType), consumer);
    cuda.downloadAsync(&geometryValue, deviceAddress(result.columns.geometry.values), sizeof(geometryValue), consumer);
    cuda.downloadAsync(&sourceFace, deviceAddress(result.columns.contact.exact.features.source_face_indices),
                       sizeof(sourceFace), consumer);
    cuda.downloadAsync(sourceCorners, deviceAddress(result.columns.contact.exact.features.source_corner_indices),
                       sizeof(sourceCorners), consumer);
    cuda.downloadAsync(barycentrics, deviceAddress(result.columns.contact.exact.features.barycentrics),
                       sizeof(barycentrics), consumer);
    gPhase = "retire public exact CUDA raycast";
    retireAfterConsumer(cuda, consumer, fixture.instance, operation.op_id);
    REQUIRE(rowStatus == OVSPATIALQUERY_BATCH_ROW_SUCCESS);
    REQUIRE(stored == 1u && offset == 0u && count == 1u && truncated == 0u);
    REQUIRE(objectId == fixture.objectValue);
    REQUIRE(featureKind == OVSPATIALQUERY_GEOMETRY_FEATURE_TRIANGLE);
    REQUIRE(geometryType == OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH);
    REQUIRE(geometryValue == fixture.meshValue);
    REQUIRE(sourceFace == 0u);
    REQUIRE(sourceCorners[0] == 0u && sourceCorners[1] == 1u && sourceCorners[2] == 2u);
    REQUIRE(barycentrics[0] == 0.25f && barycentrics[1] == 0.25f && barycentrics[2] == 0.5f);

    // DLPack producers report any strides for a dimension with no elements
    // (NumPy reports 0); an empty batch is accepted whatever they are.
    gPhase = "empty public CUDA raycast with producer strides";
    InputArray emptyOrigins(static_cast<double*>(nullptr), 0, 3, kDLFloat, kCudaDevice);
    InputArray emptyDirections(static_cast<float*>(nullptr), 0, 3, kDLFloat, kCudaDevice);
    InputArray emptyMaximums(static_cast<double*>(nullptr), 0, kDLFloat, kCudaDevice);
    for (InputArray* empty : {&emptyOrigins, &emptyDirections, &emptyMaximums})
    {
        empty->strides = {0, 0};
    }
    ovspatialquery_raycast_batch_t emptyBatch = batch;
    emptyBatch.query_count                    = 0u;
    emptyBatch.query_bits                     = nullptr;
    emptyBatch.origins                        = &emptyOrigins.value;
    emptyBatch.unit_directions                = &emptyDirections.value;
    emptyBatch.max_distances                  = &emptyMaximums.value;
    ovspatialquery_raycast_batch_reservation_t emptyReservation{};
    // An empty column still has zero byte_offset.
    emptyMaximums.value.tensor.byte_offset = sizeof(double);
    REQUIRE(ovspatialquery_enqueue_raycast_plan(fixture.instance, plan, &emptyBatch, &emptyReservation).status ==
            OVSPATIALQUERY_API_INVALID_ARGUMENT);
    emptyMaximums.value.tensor.byte_offset = 0u;
    ovspatialquery_enqueue_result_t emptyOperation =
        ovspatialquery_enqueue_raycast_plan(fixture.instance, plan, &emptyBatch, &emptyReservation);
    REQUIRE(emptyOperation.status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_wait_op(fixture.instance, emptyOperation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited)
                .status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(waited.state == OVSPATIALQUERY_OP_COMPLETED);
    retireAfterConsumer(cuda, consumer, fixture.instance, emptyOperation.op_id);

    REQUIRE(destroyPlanWhenRetired(fixture.instance, plan));
    fixture.close();

    REQUIRE(transferredEventDestroysWereInvoked(2u));
}

// The fixture triangle lies in the plane x = 4, so its world AABB has zero
// thickness in X. A BOUNDS_CANDIDATE row reports the face it enters through:
// +X from the far side, -X from the near side, although both faces sit at x = 4.
void testCudaBoundsCandidateNormalOfFlatAabbFacesTheRay()
{
    ExactMeshFixture         fixture;
    CudaDriver               cuda(CudaDriver::Ownership::tracked);
    const CudaDriver::Stream producer = cuda.createStream();
    const CudaDriver::Stream consumer = cuda.createStream();

    gPhase = "create CUDA bounds-candidate raycast plan";
    ovspatialquery_raycast_plan_desc_t desc{};
    desc.target                             = populationRef(fixture.population);
    desc.hit_mode                           = OVSPATIALQUERY_HIT_MODE_CLOSEST;
    desc.precision                          = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    desc.device                             = {kDLCUDA, 0};
    desc.maximum_query_count                = 2u;
    desc.maximum_hits_per_query             = 1u;
    desc.hit_capacity_per_slot              = 2u;
    desc.result_slot_count                  = 1u;
    desc.result_fields                      = OVSPATIALQUERY_BATCH_RESULT_NORMAL | OVSPATIALQUERY_BATCH_RESULT_DISTANCE;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_raycast_plan(fixture.instance, &desc, &plan).status == OVSPATIALQUERY_API_SUCCESS);

    double                          origins[2][3]         = {{10.0, 0.0, -0.5}, {-2.0, 0.0, -0.5}};
    float                           directions[2][3]      = {{-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}};
    double                          maximumDistances[2]   = {10.0, 10.0};
    const CudaDriver::DevicePointer deviceOrigins         = upload(cuda, origins, producer);
    const CudaDriver::DevicePointer deviceDirections      = upload(cuda, directions, producer);
    const CudaDriver::DevicePointer deviceMaximumDistance = upload(cuda, maximumDistances, producer);
    InputArray                      originColumn(devicePointer<double>(deviceOrigins), 2, 3, kDLFloat, kCudaDevice);
    InputArray directionColumn(devicePointer<float>(deviceDirections), 2, 3, kDLFloat, kCudaDevice);
    InputArray maximumColumn(devicePointer<double>(deviceMaximumDistance), 2, kDLFloat, kCudaDevice);
    ovspatialquery_raycast_batch_t batch{};
    batch.ordinal.kind         = OVSPATIALQUERY_ORDINAL_HEAD;
    batch.query_count          = 2u;
    batch.origins              = &originColumn.value;
    batch.unit_directions      = &directionColumn.value;
    batch.max_distances        = &maximumColumn.value;
    batch.cuda_sync.has_stream = 1u;
    batch.cuda_sync.stream     = reinterpret_cast<uintptr_t>(producer);
    ovspatialquery_raycast_batch_reservation_t reservation{};
    gPhase = "enqueue CUDA bounds-candidate raycast";
    ovspatialquery_enqueue_result_t operation =
        ovspatialquery_enqueue_raycast_plan(fixture.instance, plan, &batch, &reservation);
    REQUIRE(operation.status == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_op_wait_result_t waited{};
    REQUIRE(
        ovspatialquery_wait_op(fixture.instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
        OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(waited.state == OVSPATIALQUERY_OP_COMPLETED);
    ovspatialquery_raycast_batch_result_t result{};
    REQUIRE(ovspatialquery_fetch_raycast_batch(fixture.instance, operation.op_id, &result).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.common.truth == OVSPATIALQUERY_HIT_BOUNDS_CANDIDATE);

    cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(result.common.result_ready.value));
    uint64_t offsets[2]    = {UINT64_MAX, UINT64_MAX};
    uint64_t counts[2]     = {};
    float    normals[2][3] = {};
    double   distances[2]  = {};
    cuda.downloadAsync(offsets, deviceAddress(result.columns.hit_ranges.hit_offsets), sizeof(offsets), consumer);
    cuda.downloadAsync(counts, deviceAddress(result.columns.hit_ranges.hit_counts), sizeof(counts), consumer);
    cuda.downloadAsync(normals, deviceAddress(result.columns.contact.candidate.bounds_normals), sizeof(normals),
                       consumer);
    cuda.downloadAsync(distances, deviceAddress(result.columns.contact.candidate.distances), sizeof(distances),
                       consumer);
    gPhase = "retire CUDA bounds-candidate raycast";
    retireAfterConsumer(cuda, consumer, fixture.instance, operation.op_id);
    for (uint32_t row = 0u; row < 2u; ++row)
    {
        REQUIRE(counts[row] == 1u && offsets[row] < 2u);
        const float* normal = normals[offsets[row]];
        REQUIRE(distances[offsets[row]] == 6.0);
        REQUIRE(normal[0] == -directions[row][0] && normal[1] == 0.0f && normal[2] == 0.0f);
    }

    REQUIRE(destroyPlanWhenRetired(fixture.instance, plan));
    REQUIRE(transferredEventDestroysWereInvoked());
    fixture.close();
}

// A retired slot is free once its consumer is done, even while the host
// function that wakes the reaper has not run: a driver that runs host
// functions late (WDDM) must not keep a one-slot plan at RESULT_CAPACITY until
// it does. Every ring here waits on the driver's thread until the end, or for
// 30 s: a ring must not wait on CUDA work queued after it, so a driver that
// needs the held thread to finish the launches below fails the test instead
// of hanging it.
void testCudaSlotReturnsWhileReaperRingsWait()
{
    ExactMeshFixture         fixture;
    CudaDriver               cuda(CudaDriver::Ownership::tracked);
    const CudaDriver::Stream producer = cuda.createStream();
    const CudaDriver::Stream consumer = cuda.createStream();

    gPhase = "create one-slot CUDA raycast plan";
    ovspatialquery_raycast_plan_desc_t desc{};
    desc.target                             = populationRef(fixture.population);
    desc.hit_mode                           = OVSPATIALQUERY_HIT_MODE_CLOSEST;
    desc.precision                          = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    desc.device                             = {kDLCUDA, 0};
    desc.maximum_query_count                = 1u;
    desc.maximum_hits_per_query             = 1u;
    desc.hit_capacity_per_slot              = 1u;
    desc.result_slot_count                  = 1u;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_raycast_plan(fixture.instance, &desc, &plan).status == OVSPATIALQUERY_API_SUCCESS);

    double                          origin[3]       = {10.0, 0.0, -0.5};
    float                           direction[3]    = {-1.0f, 0.0f, 0.0f};
    double                          maximumDistance = 10.0;
    const CudaDriver::DevicePointer deviceOrigin    = upload(cuda, origin, producer);
    const CudaDriver::DevicePointer deviceDirection = upload(cuda, direction, producer);
    const CudaDriver::DevicePointer deviceMaximum   = upload(cuda, maximumDistance, producer);
    InputArray                      origins(devicePointer<double>(deviceOrigin), 1, 3, kDLFloat, kCudaDevice);
    InputArray                      directions(devicePointer<float>(deviceDirection), 1, 3, kDLFloat, kCudaDevice);
    InputArray                      maximums(devicePointer<double>(deviceMaximum), 1, kDLFloat, kCudaDevice);
    ovspatialquery_raycast_batch_t  batch{};
    batch.ordinal.kind         = OVSPATIALQUERY_ORDINAL_HEAD;
    batch.query_count          = 1u;
    batch.origins              = &origins.value;
    batch.unit_directions      = &directions.value;
    batch.max_distances        = &maximums.value;
    batch.cuda_sync.has_stream = 1u;
    batch.cuda_sync.stream     = reinterpret_cast<uintptr_t>(producer);

    REQUIRE(!ovspatialquery_test_hold_cuda_signal_rings(true));
    for (uint32_t launch = 0u; launch != 3u; ++launch)
    {
        gPhase = "launch on the slot retired while its reaper ring waits";
        // The previous consumer event completes on the device shortly after
        // its retirement queued it.
        const std::chrono::steady_clock::time_point deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(10);
        ovspatialquery_raycast_batch_reservation_t reservation{};
        ovspatialquery_enqueue_result_t            operation{};
        for (;;)
        {
            operation = ovspatialquery_enqueue_raycast_plan(fixture.instance, plan, &batch, &reservation);
            if (operation.status != OVSPATIALQUERY_API_RESULT_CAPACITY || std::chrono::steady_clock::now() > deadline)
            {
                break;
            }
            std::this_thread::yield();
        }
        REQUIRE(operation.status == OVSPATIALQUERY_API_SUCCESS);
        ovspatialquery_op_wait_result_t waited{};
        REQUIRE(ovspatialquery_wait_op(fixture.instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited)
                    .status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(waited.state == OVSPATIALQUERY_OP_COMPLETED);
        ovspatialquery_raycast_batch_result_t result{};
        REQUIRE(ovspatialquery_fetch_raycast_batch(fixture.instance, operation.op_id, &result).status ==
                OVSPATIALQUERY_API_SUCCESS);
        cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(result.common.result_ready.value));
        uint64_t stored = UINT64_MAX;
        cuda.downloadAsync(&stored, deviceAddress(result.columns.hit_ranges.stored_hit_count), sizeof(stored),
                           consumer);
        retireAfterConsumer(cuda, consumer, fixture.instance, operation.op_id);
        REQUIRE(stored == 1u);
    }
    gPhase = "release the held reaper rings";
    REQUIRE(!ovspatialquery_test_hold_cuda_signal_rings(false));

    REQUIRE(destroyPlanWhenRetired(fixture.instance, plan));
    REQUIRE(transferredEventDestroysWereInvoked(3u));
    fixture.close();
}

void testPublicExactSingleCudaRayModesAndBoundaries()
{
    ExactMeshFixture                fixture;
    CudaDriver                      cuda(CudaDriver::Ownership::tracked);
    const CudaDriver::Stream        producer              = cuda.createStream();
    const CudaDriver::Stream        consumer              = cuda.createStream();
    double                          origin[3]             = {0.0, 0.25, 0.0};
    float                           direction[3]          = {1.0f, 0.0f, 0.0f};
    double                          maximumDistance       = 10.0;
    uint64_t                        maximumHits           = 1u;
    const CudaDriver::DevicePointer deviceOrigin          = cuda.allocate(sizeof(origin));
    const CudaDriver::DevicePointer deviceDirection       = upload(cuda, direction, producer);
    const CudaDriver::DevicePointer deviceMaximumDistance = upload(cuda, maximumDistance, producer);
    const CudaDriver::DevicePointer deviceMaximumHits     = upload(cuda, maximumHits, producer);
    InputArray                      origins(devicePointer<double>(deviceOrigin), 1, 3, kDLFloat, kCudaDevice);
    InputArray                      directions(devicePointer<float>(deviceDirection), 1, 3, kDLFloat, kCudaDevice);
    InputArray maximumDistances(devicePointer<double>(deviceMaximumDistance), 1, kDLFloat, kCudaDevice);
    InputArray rowMaximumHits(devicePointer<uint64_t>(deviceMaximumHits), 1, kDLUInt, kCudaDevice);

    const auto run = [&](ovspatialquery_query_precision_t precision, ovspatialquery_hit_mode_t mode, double originZ,
                         double tolerance, uint64_t expectedStored)
    {
        origin[2] = originZ;
        cuda.uploadAsync(deviceOrigin, origin, sizeof(origin), producer);
        ovspatialquery_raycast_plan_desc_t desc{};
        desc.target                 = populationRef(fixture.population);
        desc.hit_mode               = mode;
        desc.precision              = precision;
        desc.device                 = {kDLCUDA, 0};
        desc.maximum_query_count    = 1u;
        desc.maximum_hits_per_query = 1u;
        desc.hit_capacity_per_slot  = 1u;
        desc.result_slot_count      = 1u;
        desc.result_fields          = OVSPATIALQUERY_BATCH_RESULT_DISTANCE | OVSPATIALQUERY_BATCH_RESULT_SOURCE_FEATURE;
        desc.contact_tolerance      = tolerance;
        ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
        gPhase                                  = "create CUDA raycast plan for precision boundary";
        const auto created                      = ovspatialquery_create_raycast_plan(fixture.instance, &desc, &plan);
        if (created.status != OVSPATIALQUERY_API_SUCCESS)
        {
            std::fprintf(stderr, "raycast plan status=%u precision=%u mode=%u tolerance=%g\n",
                         static_cast<unsigned>(created.status), static_cast<unsigned>(precision),
                         static_cast<unsigned>(mode), tolerance);
        }
        REQUIRE(created.status == OVSPATIALQUERY_API_SUCCESS);

        ovspatialquery_raycast_batch_t batch{};
        batch.ordinal.kind         = OVSPATIALQUERY_ORDINAL_HEAD;
        batch.query_count          = 1u;
        batch.origins              = &origins.value;
        batch.unit_directions      = &directions.value;
        batch.max_distances        = &maximumDistances.value;
        batch.cuda_sync.has_stream = 1u;
        batch.cuda_sync.stream     = reinterpret_cast<uintptr_t>(producer);
        ovspatialquery_raycast_batch_reservation_t reservation{};
        if (mode != OVSPATIALQUERY_HIT_MODE_MULTIPLE)
        {
            // A row hit cap belongs to MULTIPLE plans only.
            batch.maximum_hits = &rowMaximumHits.value;
            REQUIRE(ovspatialquery_enqueue_raycast_plan(fixture.instance, plan, &batch, &reservation).status ==
                    OVSPATIALQUERY_API_INVALID_ARGUMENT);
            batch.maximum_hits = nullptr;
        }
        ovspatialquery_enqueue_result_t operation =
            ovspatialquery_enqueue_raycast_plan(fixture.instance, plan, &batch, &reservation);
        REQUIRE(operation.status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(reservation.common.truth == OVSPATIALQUERY_HIT_EXACT_SURFACE);
        REQUIRE(reservation.columns.contact.exact.features.kinds.data != nullptr);

        ovspatialquery_op_wait_result_t waited{};
        REQUIRE(ovspatialquery_wait_op(fixture.instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited)
                    .status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(waited.state == OVSPATIALQUERY_OP_COMPLETED);
        ovspatialquery_raycast_batch_result_t result{};
        REQUIRE(ovspatialquery_fetch_raycast_batch(fixture.instance, operation.op_id, &result).status ==
                OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(result.common.truth == OVSPATIALQUERY_HIT_EXACT_SURFACE);
        cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(result.common.result_ready.value));
        uint32_t rowStatus = UINT32_MAX;
        uint64_t stored    = UINT64_MAX;
        uint64_t objectId  = 0u;
        uint32_t feature   = UINT32_MAX;
        double   distance  = 0.0;
        cuda.downloadAsync(&rowStatus, deviceAddress(result.columns.hit_ranges.query_status), sizeof(rowStatus),
                           consumer);
        cuda.downloadAsync(&stored, deviceAddress(result.columns.hit_ranges.stored_hit_count), sizeof(stored),
                           consumer);
        if (expectedStored != 0u)
        {
            cuda.downloadAsync(&objectId, deviceAddress(result.columns.object_ids), sizeof(objectId), consumer);
            cuda.downloadAsync(&feature, deviceAddress(result.columns.contact.exact.features.kinds), sizeof(feature),
                               consumer);
            cuda.downloadAsync(&distance, deviceAddress(result.columns.contact.exact.distances), sizeof(distance),
                               consumer);
        }
        retireAfterConsumer(cuda, consumer, fixture.instance, operation.op_id);
        REQUIRE(rowStatus == OVSPATIALQUERY_BATCH_ROW_SUCCESS);
        REQUIRE(stored == expectedStored);
        if (expectedStored != 0u)
        {
            REQUIRE(objectId == fixture.objectValue);
            REQUIRE(feature == OVSPATIALQUERY_GEOMETRY_FEATURE_TRIANGLE);
            REQUIRE(distance > 0.0 && distance <= 4.0);
        }
        REQUIRE(destroyPlanWhenRetired(fixture.instance, plan));
        REQUIRE(transferredEventDestroysWereInvoked());
    };

    // At y=0.25 the slanted triangle edge is z=0.5, and a point dz beyond it
    // lies dz/4 of the barycentric range outside. Both precisions accept a
    // crossing within 2^-22 of it, as the CPU's PhysX meshes do, so 2^-27
    // beyond (which binary32 rounds onto the edge) is a hit and 2^-18 beyond is
    // a miss, for EXACT and every admitted EXACT_SINGLE hit mode.
    const double insideMargin = 0.5 + std::ldexp(1.0, -27), outsideMargin = 0.5 + std::ldexp(1.0, -18);
    run(OVSPATIALQUERY_QUERY_EXACT, OVSPATIALQUERY_HIT_MODE_CLOSEST, insideMargin, 0.0, 1u);
    run(OVSPATIALQUERY_QUERY_EXACT, OVSPATIALQUERY_HIT_MODE_CLOSEST, outsideMargin, 0.0, 0u);
    // Keep the mode table in static storage: GCC 11's ARM Release build omits
    // initialization of the temporary initializer-list backing array here.
    static constexpr std::array<ovspatialquery_hit_mode_t, 3> modes{
        OVSPATIALQUERY_HIT_MODE_CLOSEST, OVSPATIALQUERY_HIT_MODE_ANY, OVSPATIALQUERY_HIT_MODE_MULTIPLE};
    for (ovspatialquery_hit_mode_t mode : modes)
    {
        run(OVSPATIALQUERY_QUERY_EXACT_SINGLE, mode, insideMargin, 0.0, 1u);
        run(OVSPATIALQUERY_QUERY_EXACT_SINGLE, mode, outsideMargin, 0.0, 0u);
    }
    // Nonzero tolerance uses the existing wider exact mesh path.
    run(OVSPATIALQUERY_QUERY_EXACT_SINGLE, OVSPATIALQUERY_HIT_MODE_CLOSEST, 0.55, 0.1, 1u);

    fixture.close();
}

void testPublicCudaSweep(ovspatialquery_query_precision_t precision, bool forceRecoverableFailure = false,
                         bool useRowRadius = false)
{
    ExactMeshFixture         fixture;
    CudaDriver               cuda(CudaDriver::Ownership::tracked);
    const CudaDriver::Stream stream   = cuda.createStream();
    const CudaDriver::Stream consumer = cuda.createStream();
    gPhase                            = "create public CUDA sphere-sweep plan";
    ovspatialquery_sphere_sweep_plan_desc_t planDesc{};
    planDesc.target              = populationRef(fixture.population);
    planDesc.precision           = precision;
    planDesc.radius              = 0.5;
    planDesc.device              = {kDLCUDA, 0};
    planDesc.maximum_query_count = 1u;
    planDesc.result_slot_count   = 1u;
    planDesc.result_fields       = OVSPATIALQUERY_BATCH_RESULT_POSITION | OVSPATIALQUERY_BATCH_RESULT_NORMAL |
                             OVSPATIALQUERY_BATCH_RESULT_DISTANCE | OVSPATIALQUERY_BATCH_RESULT_GEOMETRY_IDENTITY |
                             OVSPATIALQUERY_BATCH_RESULT_QUERY_POSITION;
    if (precision == OVSPATIALQUERY_QUERY_EXACT)
    {
        planDesc.result_fields |= OVSPATIALQUERY_BATCH_RESULT_SOURCE_FEATURE;
    }
    planDesc.hit_capacity_per_slot          = 1u;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_sphere_sweep_plan(fixture.instance, &planDesc, &plan).status ==
            OVSPATIALQUERY_API_SUCCESS);

    double                          center[3]             = {0.0, 0.0, 0.0};
    float                           direction[3]          = {1.0f, 0.0f, 0.0f};
    double                          maximumDistance       = 10.0;
    double                          rowRadius             = 0.25;
    uint64_t                        queryBits[2]          = {0u, UINT64_C(0x20)};
    const CudaDriver::DevicePointer deviceCenter          = upload(cuda, center, stream);
    const CudaDriver::DevicePointer deviceDirection       = upload(cuda, direction, stream);
    const CudaDriver::DevicePointer deviceMaximumDistance = upload(cuda, maximumDistance, stream);
    const CudaDriver::DevicePointer deviceQueryBits       = upload(cuda, queryBits, stream);
    const CudaDriver::DevicePointer deviceRadius          = useRowRadius ? upload(cuda, rowRadius, stream) : 0u;
    InputArray                      centers(devicePointer<double>(deviceCenter), 1, 3, kDLFloat, kCudaDevice);
    InputArray                      directions(devicePointer<float>(deviceDirection), 1, 3, kDLFloat, kCudaDevice);
    InputArray maximumDistances(devicePointer<double>(deviceMaximumDistance), 1, kDLFloat, kCudaDevice);
    InputArray bits(devicePointer<uint64_t>(deviceQueryBits), 1, 2, kDLUInt, kCudaDevice);
    InputArray radii(devicePointer<double>(deviceRadius), 1, kDLFloat, kCudaDevice);
    ovspatialquery_sphere_sweep_batch_t batch{};
    batch.ordinal.kind         = OVSPATIALQUERY_ORDINAL_HEAD;
    batch.query_count          = 1u;
    batch.query_bits           = &bits.value;
    batch.centers              = &centers.value;
    batch.unit_directions      = &directions.value;
    batch.max_distances        = &maximumDistances.value;
    batch.radii                = useRowRadius ? &radii.value : nullptr;
    batch.cuda_sync.has_stream = 1u;
    batch.cuda_sync.stream     = reinterpret_cast<uintptr_t>(stream);
    batch.cuda_sync.wait_event = 0u;
    ovspatialquery_sphere_sweep_batch_reservation_t reservation{};
    if (useRowRadius)
    {
        // Row radii are float64, and every sweep row needs a maximum distance.
        gPhase = "reject public CUDA sphere-sweep launch columns";
        InputArray                          narrowRadii(devicePointer<float>(deviceRadius), 1, kDLFloat, kCudaDevice);
        ovspatialquery_sphere_sweep_batch_t rejected = batch;
        rejected.radii                               = &narrowRadii.value;
        REQUIRE(ovspatialquery_enqueue_sphere_sweep_plan(fixture.instance, plan, &rejected, &reservation).status ==
                OVSPATIALQUERY_API_INVALID_ARGUMENT);
        rejected               = batch;
        rejected.max_distances = nullptr;
        REQUIRE(ovspatialquery_enqueue_sphere_sweep_plan(fixture.instance, plan, &rejected, &reservation).status ==
                OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    gPhase = "enqueue public CUDA sphere-sweep with zero wait event";
    if (forceRecoverableFailure)
    {
        ovspatialquery_test_set_cuda_failure_point(kCudaFailureFamilyLaunchAfterKernels);
    }
    ovspatialquery_enqueue_result_t operation =
        ovspatialquery_enqueue_sphere_sweep_plan(fixture.instance, plan, &batch, &reservation);
    REQUIRE(operation.status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(reservation.common.source_ordinal == fixture.sourceOrdinal);
    REQUIRE(reservation.common.truth == (precision == OVSPATIALQUERY_QUERY_EXACT
                                             ? OVSPATIALQUERY_HIT_EXACT_SURFACE
                                             : OVSPATIALQUERY_HIT_BOUNDS_CANDIDATE));
    REQUIRE(reservation.common.result_ready.value != 0u);
    const DLTensor& reservedQueryPositions = precision == OVSPATIALQUERY_QUERY_EXACT
                                                 ? reservation.columns.contact.exact.query_positions
                                                 : reservation.columns.contact.candidate.query_positions;
    REQUIRE(reservedQueryPositions.data != nullptr);
    REQUIRE(reservedQueryPositions.shape != nullptr);
    REQUIRE(reservedQueryPositions.shape[0] == 1);

    if (forceRecoverableFailure)
    {
        gPhase = "verify public CUDA sphere-sweep recoverable safe-empty result";
        verifyRecoverableCudaFailure<ovspatialquery_sphere_sweep_batch_result_t>(
            cuda, consumer, fixture.instance, operation.op_id, reservation.common.result_ready,
            reservation.columns.hit_ranges,
            [&](ovspatialquery_sphere_sweep_batch_result_t* output)
            { return ovspatialquery_fetch_sphere_sweep_batch(fixture.instance, operation.op_id, output); });
        REQUIRE(destroyPlanWhenRetired(fixture.instance, plan));
        fixture.close();
        return;
    }

    ovspatialquery_op_wait_result_t waited{};
    REQUIRE(
        ovspatialquery_wait_op(fixture.instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
        OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(waited.state == OVSPATIALQUERY_OP_COMPLETED);
    ovspatialquery_sphere_sweep_batch_result_t result{};
    REQUIRE(ovspatialquery_fetch_sphere_sweep_batch(fixture.instance, operation.op_id, &result).status ==
            OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_sphere_sweep_batch_result_t repeated{};
    REQUIRE(ovspatialquery_fetch_sphere_sweep_batch(fixture.instance, operation.op_id, &repeated).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(repeated.common.result_ready.value == result.common.result_ready.value);
    REQUIRE(repeated.columns.object_ids.data == result.columns.object_ids.data);
    REQUIRE(result.columns.object_ids.data == reservation.columns.object_ids.data);
    REQUIRE(result.columns.object_ids.shape != reservation.columns.object_ids.shape);
    const DLTensor& queryPositions = precision == OVSPATIALQUERY_QUERY_EXACT
                                         ? result.columns.contact.exact.query_positions
                                         : result.columns.contact.candidate.query_positions;
    REQUIRE(queryPositions.data != nullptr);
    REQUIRE((precision == OVSPATIALQUERY_QUERY_EXACT && result.columns.contact.exact.features.kinds.data != nullptr) ||
            (precision == OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE &&
             result.columns.contact.candidate.query_positions.data != nullptr));

    cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(result.common.result_ready.value));
    uint32_t rowStatus        = UINT32_MAX;
    uint64_t stored           = UINT64_MAX;
    uint64_t objectId         = 0u;
    double   queryPosition[3] = {};
    cuda.downloadAsync(&rowStatus, deviceAddress(result.columns.hit_ranges.query_status), sizeof(rowStatus), consumer);
    cuda.downloadAsync(&stored, deviceAddress(result.columns.hit_ranges.stored_hit_count), sizeof(stored), consumer);
    cuda.downloadAsync(&objectId, deviceAddress(result.columns.object_ids), sizeof(objectId), consumer);
    cuda.downloadAsync(queryPosition, deviceAddress(queryPositions), sizeof(queryPosition), consumer);
    retireAfterConsumer(cuda, consumer, fixture.instance, operation.op_id);
    REQUIRE(rowStatus == OVSPATIALQUERY_BATCH_ROW_SUCCESS);
    REQUIRE(stored == 1u && objectId == fixture.objectValue);
    REQUIRE(queryPosition[0] > 3.0 && queryPosition[0] < 4.0);
    if (useRowRadius && precision == OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE)
    {
        REQUIRE(std::abs(queryPosition[0] - 3.75) < 1.0e-9);
    }

    REQUIRE(destroyPlanWhenRetired(fixture.instance, plan));
    fixture.close();
    REQUIRE(transferredEventDestroysWereInvoked());
}

void testPublicCudaBoxSweep(ovspatialquery_query_precision_t precision)
{
    ExactMeshFixture                     fixture;
    CudaDriver                           cuda(CudaDriver::Ownership::tracked);
    const CudaDriver::Stream             stream   = cuda.createStream();
    const CudaDriver::Stream             consumer = cuda.createStream();
    ovspatialquery_box_sweep_plan_desc_t desc{};
    desc.target                = populationRef(fixture.population);
    desc.precision             = precision;
    desc.half_extents[0]       = 0.25;
    desc.half_extents[1]       = 0.25;
    desc.half_extents[2]       = 0.25;
    desc.device                = {kDLCUDA, 0};
    desc.maximum_query_count   = 1u;
    desc.result_slot_count     = 1u;
    desc.hit_capacity_per_slot = 1u;
    desc.result_fields         = OVSPATIALQUERY_BATCH_RESULT_DISTANCE | OVSPATIALQUERY_BATCH_RESULT_QUERY_POSITION;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_box_sweep_plan(fixture.instance, &desc, &plan).status == OVSPATIALQUERY_API_SUCCESS);

    double                           center[3]       = {0.0, 0.0, 0.0};
    float                            rotation[4]     = {0.0f, 0.0f, 0.0f, 1.0f};
    float                            direction[3]    = {1.0f, 0.0f, 0.0f};
    double                           maximumDistance = 10.0;
    const CudaDriver::DevicePointer  deviceCenter    = upload(cuda, center, stream);
    const CudaDriver::DevicePointer  deviceRotation  = upload(cuda, rotation, stream);
    const CudaDriver::DevicePointer  deviceDirection = upload(cuda, direction, stream);
    const CudaDriver::DevicePointer  deviceMaximum   = upload(cuda, maximumDistance, stream);
    InputArray                       centers(devicePointer<double>(deviceCenter), 1, 3, kDLFloat, kCudaDevice);
    InputArray                       rotations(devicePointer<float>(deviceRotation), 1, 4, kDLFloat, kCudaDevice);
    InputArray                       directions(devicePointer<float>(deviceDirection), 1, 3, kDLFloat, kCudaDevice);
    InputArray                       maximums(devicePointer<double>(deviceMaximum), 1, kDLFloat, kCudaDevice);
    ovspatialquery_box_sweep_batch_t batch{};
    batch.ordinal.kind         = OVSPATIALQUERY_ORDINAL_HEAD;
    batch.query_count          = 1u;
    batch.centers              = &centers.value;
    batch.rotations_xyzw       = &rotations.value;
    batch.unit_directions      = &directions.value;
    batch.max_distances        = &maximums.value;
    batch.cuda_sync.has_stream = 1u;
    batch.cuda_sync.stream     = reinterpret_cast<uintptr_t>(stream);
    ovspatialquery_box_sweep_batch_reservation_t reservation{};
    ovspatialquery_enqueue_result_t              admission =
        ovspatialquery_enqueue_box_sweep_plan(fixture.instance, plan, &batch, &reservation);
    REQUIRE(admission.status == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_op_wait_result_t waited{};
    REQUIRE(
        ovspatialquery_wait_op(fixture.instance, admission.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
        OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(waited.state == OVSPATIALQUERY_OP_COMPLETED);
    ovspatialquery_box_sweep_batch_result_t result{};
    REQUIRE(ovspatialquery_fetch_box_sweep_batch(fixture.instance, admission.op_id, &result).status ==
            OVSPATIALQUERY_API_SUCCESS);
    const DLTensor& distanceColumn = precision == OVSPATIALQUERY_QUERY_EXACT
                                         ? result.columns.contact.exact.distances
                                         : result.columns.contact.candidate.distances;
    cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(result.common.result_ready.value));
    uint32_t rowStatus = UINT32_MAX;
    double   distance  = -1.0;
    cuda.downloadAsync(&rowStatus, deviceAddress(result.columns.hit_ranges.query_status), sizeof(rowStatus), consumer);
    cuda.downloadAsync(&distance, deviceAddress(distanceColumn), sizeof(distance), consumer);
    retireAfterConsumer(cuda, consumer, fixture.instance, admission.op_id);
    REQUIRE(rowStatus == OVSPATIALQUERY_BATCH_ROW_SUCCESS);
    REQUIRE(std::abs(distance - 3.75) < 1.0e-5);
    REQUIRE(destroyPlanWhenRetired(fixture.instance, plan));
    fixture.close();
    REQUIRE(transferredEventDestroysWereInvoked());
}

// Rows that carry their own size on the device: box sweep half extents and
// SPHERE overlap radii. The mesh triangle lies in the plane x = 4.
void testPublicCudaRowSizes()
{
    ExactMeshFixture         fixture;
    CudaDriver               cuda(CudaDriver::Ownership::tracked);
    const CudaDriver::Stream stream   = cuda.createStream();
    const CudaDriver::Stream consumer = cuda.createStream();

    // Box sweeps from the origin along +x: the leading face sits 0.25 and 0.75
    // ahead of the center, so the rows reach the triangle at 3.75 and 3.25; a
    // zero extent is that row's INVALID_ARGUMENT. A float32 extent column or a
    // missing rotation column is the enqueue's.
    gPhase = "public CUDA box sweep with row half extents";
    ovspatialquery_box_sweep_plan_desc_t sweepDesc{};
    sweepDesc.target                             = populationRef(fixture.population);
    sweepDesc.precision                          = OVSPATIALQUERY_QUERY_EXACT;
    sweepDesc.half_extents[0]                    = 0.5;
    sweepDesc.half_extents[1]                    = 0.5;
    sweepDesc.half_extents[2]                    = 0.5;
    sweepDesc.device                             = {kDLCUDA, 0};
    sweepDesc.maximum_query_count                = 3u;
    sweepDesc.result_slot_count                  = 1u;
    sweepDesc.hit_capacity_per_slot              = 3u;
    sweepDesc.result_fields                      = OVSPATIALQUERY_BATCH_RESULT_DISTANCE;
    ovspatialquery_batch_plan_handle_t sweepPlan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_box_sweep_plan(fixture.instance, &sweepDesc, &sweepPlan).status ==
            OVSPATIALQUERY_API_SUCCESS);
    double centers[9] = {};
    // Five rows, so the overlap below can offer a rotation column of its row count.
    float                           rotations[20]     = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                                                         0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    float                           directions[9]     = {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    double                          maximums[3]       = {10.0, 10.0, 10.0};
    double                          halfExtents[9]    = {0.25, 0.25, 0.25, 0.75, 0.1, 0.1, 0.0, 0.1, 0.1};
    const CudaDriver::DevicePointer deviceCenters     = upload(cuda, centers, stream);
    const CudaDriver::DevicePointer deviceRotations   = upload(cuda, rotations, stream);
    const CudaDriver::DevicePointer deviceDirections  = upload(cuda, directions, stream);
    const CudaDriver::DevicePointer deviceMaximums    = upload(cuda, maximums, stream);
    const CudaDriver::DevicePointer deviceHalfExtents = upload(cuda, halfExtents, stream);
    InputArray                      centerColumn(devicePointer<double>(deviceCenters), 3, 3, kDLFloat, kCudaDevice);
    InputArray                      rotationColumn(devicePointer<float>(deviceRotations), 3, 4, kDLFloat, kCudaDevice);
    InputArray directionColumn(devicePointer<float>(deviceDirections), 3, 3, kDLFloat, kCudaDevice);
    InputArray maximumColumn(devicePointer<double>(deviceMaximums), 3, kDLFloat, kCudaDevice);
    InputArray halfExtentColumn(devicePointer<double>(deviceHalfExtents), 3, 3, kDLFloat, kCudaDevice);
    ovspatialquery_box_sweep_batch_t sweepBatch{};
    sweepBatch.ordinal.kind         = OVSPATIALQUERY_ORDINAL_HEAD;
    sweepBatch.query_count          = 3u;
    sweepBatch.centers              = &centerColumn.value;
    sweepBatch.rotations_xyzw       = &rotationColumn.value;
    sweepBatch.unit_directions      = &directionColumn.value;
    sweepBatch.max_distances        = &maximumColumn.value;
    sweepBatch.half_extents         = &halfExtentColumn.value;
    sweepBatch.cuda_sync.has_stream = 1u;
    sweepBatch.cuda_sync.stream     = reinterpret_cast<uintptr_t>(stream);
    ovspatialquery_box_sweep_batch_reservation_t sweepReservation{};
    InputArray narrowHalfExtentColumn(devicePointer<float>(deviceHalfExtents), 3, 3, kDLFloat, kCudaDevice);
    ovspatialquery_box_sweep_batch_t rejectedSweep = sweepBatch;
    rejectedSweep.half_extents                     = &narrowHalfExtentColumn.value;
    REQUIRE(
        ovspatialquery_enqueue_box_sweep_plan(fixture.instance, sweepPlan, &rejectedSweep, &sweepReservation).status ==
        OVSPATIALQUERY_API_INVALID_ARGUMENT);
    rejectedSweep                = sweepBatch;
    rejectedSweep.rotations_xyzw = nullptr;
    REQUIRE(
        ovspatialquery_enqueue_box_sweep_plan(fixture.instance, sweepPlan, &rejectedSweep, &sweepReservation).status ==
        OVSPATIALQUERY_API_INVALID_ARGUMENT);
    ovspatialquery_enqueue_result_t sweepOperation =
        ovspatialquery_enqueue_box_sweep_plan(fixture.instance, sweepPlan, &sweepBatch, &sweepReservation);
    REQUIRE(sweepOperation.status == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_op_wait_result_t waited{};
    REQUIRE(ovspatialquery_wait_op(fixture.instance, sweepOperation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited)
                .status == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_box_sweep_batch_result_t sweepResult{};
    REQUIRE(ovspatialquery_fetch_box_sweep_batch(fixture.instance, sweepOperation.op_id, &sweepResult).status ==
            OVSPATIALQUERY_API_SUCCESS);
    cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(sweepResult.common.result_ready.value));
    uint32_t sweepStatus[3]    = {};
    uint64_t sweepCounts[3]    = {};
    uint64_t sweepOffsets[3]   = {};
    double   sweepDistances[3] = {};
    cuda.downloadAsync(sweepStatus, deviceAddress(sweepResult.columns.hit_ranges.query_status), sizeof(sweepStatus),
                       consumer);
    cuda.downloadAsync(sweepCounts, deviceAddress(sweepResult.columns.hit_ranges.hit_counts), sizeof(sweepCounts),
                       consumer);
    cuda.downloadAsync(sweepOffsets, deviceAddress(sweepResult.columns.hit_ranges.hit_offsets), sizeof(sweepOffsets),
                       consumer);
    cuda.downloadAsync(sweepDistances, deviceAddress(sweepResult.columns.contact.exact.distances),
                       sizeof(sweepDistances), consumer);
    retireAfterConsumer(cuda, consumer, fixture.instance, sweepOperation.op_id);
    REQUIRE(sweepStatus[0] == OVSPATIALQUERY_BATCH_ROW_SUCCESS && sweepCounts[0] == 1u);
    REQUIRE(sweepStatus[1] == OVSPATIALQUERY_BATCH_ROW_SUCCESS && sweepCounts[1] == 1u);
    REQUIRE(sweepStatus[2] == OVSPATIALQUERY_BATCH_ROW_INVALID_ARGUMENT && sweepCounts[2] == 0u);
    REQUIRE(std::abs(sweepDistances[sweepOffsets[0]] - 3.75) < 1.0e-5);
    REQUIRE(std::abs(sweepDistances[sweepOffsets[1]] - 3.25) < 1.0e-5);
    REQUIRE(destroyPlanWhenRetired(fixture.instance, sweepPlan));
    REQUIRE(transferredEventDestroysWereInvoked());

    // Sphere overlaps one unit in front of the triangle: a 0.5 row misses it,
    // a 1.5 row reaches it, and a zero radius is that row's INVALID_ARGUMENT,
    // as are a radius and a center beyond the input domain's 2^60.
    // A float32 radius column, or a rotation column on a SPHERE, is the
    // enqueue's.
    gPhase = "public CUDA sphere overlap with row radii";
    ovspatialquery_overlap_plan_desc_t overlapDesc{};
    overlapDesc.target                             = populationRef(fixture.population);
    overlapDesc.hit_mode                           = OVSPATIALQUERY_HIT_MODE_ANY;
    overlapDesc.precision                          = OVSPATIALQUERY_QUERY_EXACT;
    overlapDesc.shape.kind                         = OVSPATIALQUERY_OVERLAP_SPHERE;
    overlapDesc.shape.data.sphere.radius           = 0.5;
    overlapDesc.device                             = {kDLCUDA, 0};
    overlapDesc.maximum_query_count                = 5u;
    overlapDesc.maximum_hits_per_query             = 1u;
    overlapDesc.hit_capacity_per_slot              = 5u;
    overlapDesc.result_slot_count                  = 1u;
    ovspatialquery_batch_plan_handle_t overlapPlan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_overlap_plan(fixture.instance, &overlapDesc, &overlapPlan).status ==
            OVSPATIALQUERY_API_SUCCESS);
    double translations[15] = {3.0, 0.0, 0.0, 3.0, 0.0, 0.0, 3.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0x1p61, 0.0, 0.0};
    double radii[5]         = {0.5, 1.5, 0.0, 0x1p61, 0.5};
    const CudaDriver::DevicePointer deviceTranslations = upload(cuda, translations, stream);
    const CudaDriver::DevicePointer deviceRadii        = upload(cuda, radii, stream);
    InputArray translationColumn(devicePointer<double>(deviceTranslations), 5, 3, kDLFloat, kCudaDevice);
    InputArray radiusColumn(devicePointer<double>(deviceRadii), 5, kDLFloat, kCudaDevice);
    InputArray narrowRadiusColumn(devicePointer<float>(deviceRadii), 5, kDLFloat, kCudaDevice);
    InputArray overlapRotationColumn(devicePointer<float>(deviceRotations), 5, 4, kDLFloat, kCudaDevice);
    ovspatialquery_overlap_batch_t overlapBatch{};
    overlapBatch.ordinal.kind         = OVSPATIALQUERY_ORDINAL_HEAD;
    overlapBatch.query_count          = 5u;
    overlapBatch.translations         = &translationColumn.value;
    overlapBatch.radii                = &narrowRadiusColumn.value;
    overlapBatch.cuda_sync.has_stream = 1u;
    overlapBatch.cuda_sync.stream     = reinterpret_cast<uintptr_t>(stream);
    ovspatialquery_overlap_batch_reservation_t overlapReservation{};
    REQUIRE(
        ovspatialquery_enqueue_overlap_plan(fixture.instance, overlapPlan, &overlapBatch, &overlapReservation).status ==
        OVSPATIALQUERY_API_INVALID_ARGUMENT);
    overlapBatch.radii          = &radiusColumn.value;
    overlapBatch.rotations_xyzw = &overlapRotationColumn.value;
    REQUIRE(
        ovspatialquery_enqueue_overlap_plan(fixture.instance, overlapPlan, &overlapBatch, &overlapReservation).status ==
        OVSPATIALQUERY_API_INVALID_ARGUMENT);
    overlapBatch.rotations_xyzw = nullptr;
    ovspatialquery_enqueue_result_t overlapOperation =
        ovspatialquery_enqueue_overlap_plan(fixture.instance, overlapPlan, &overlapBatch, &overlapReservation);
    REQUIRE(overlapOperation.status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_wait_op(fixture.instance, overlapOperation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited)
                .status == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_overlap_batch_result_t overlapResult{};
    REQUIRE(ovspatialquery_fetch_overlap_batch(fixture.instance, overlapOperation.op_id, &overlapResult).status ==
            OVSPATIALQUERY_API_SUCCESS);
    cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(overlapResult.common.result_ready.value));
    uint32_t overlapStatus[5] = {};
    uint64_t overlapCounts[5] = {};
    cuda.downloadAsync(overlapStatus, deviceAddress(overlapResult.columns.hit_ranges.query_status),
                       sizeof(overlapStatus), consumer);
    cuda.downloadAsync(overlapCounts, deviceAddress(overlapResult.columns.hit_ranges.hit_counts), sizeof(overlapCounts),
                       consumer);
    retireAfterConsumer(cuda, consumer, fixture.instance, overlapOperation.op_id);
    REQUIRE(overlapStatus[0] == OVSPATIALQUERY_BATCH_ROW_SUCCESS && overlapCounts[0] == 0u);
    REQUIRE(overlapStatus[1] == OVSPATIALQUERY_BATCH_ROW_SUCCESS && overlapCounts[1] == 1u);
    REQUIRE(overlapStatus[2] == OVSPATIALQUERY_BATCH_ROW_INVALID_ARGUMENT && overlapCounts[2] == 0u);
    REQUIRE(overlapStatus[3] == OVSPATIALQUERY_BATCH_ROW_INVALID_ARGUMENT && overlapCounts[3] == 0u);
    REQUIRE(overlapStatus[4] == OVSPATIALQUERY_BATCH_ROW_INVALID_ARGUMENT && overlapCounts[4] == 0u);
    REQUIRE(destroyPlanWhenRetired(fixture.instance, overlapPlan));
    fixture.close();
    REQUIRE(transferredEventDestroysWereInvoked());
}

void testPublicCudaOverlap(ovspatialquery_overlap_shape_kind_t shapeKind, ovspatialquery_query_precision_t precision,
                           bool matchingQueryBits = true, bool forceTruncation = false,
                           bool forceRecoverableFailure = false)
{
    ExactMeshFixture         fixture;
    CudaDriver               cuda(CudaDriver::Ownership::tracked);
    const CudaDriver::Stream stream   = cuda.createStream();
    const CudaDriver::Stream consumer = cuda.createStream();
    gPhase                            = "create public CUDA overlap plan";
    ovspatialquery_overlap_plan_desc_t planDesc{};
    planDesc.target     = populationRef(fixture.population);
    planDesc.hit_mode   = forceTruncation ? OVSPATIALQUERY_HIT_MODE_MULTIPLE : OVSPATIALQUERY_HIT_MODE_ANY;
    planDesc.precision  = precision;
    planDesc.shape.kind = shapeKind;
    if (shapeKind == OVSPATIALQUERY_OVERLAP_SPHERE)
    {
        planDesc.shape.data.sphere.radius = 0.5;
    }
    else
    {
        double* const extents = shapeKind == OVSPATIALQUERY_OVERLAP_BOX ? planDesc.shape.data.box.half_extents
                                                                        : planDesc.shape.data.aabb.half_extents;
        extents[0]            = 0.5;
        extents[1]            = 0.5;
        extents[2]            = 0.5;
    }
    planDesc.device                 = {kDLCUDA, 0};
    planDesc.maximum_query_count    = 1u;
    planDesc.maximum_hits_per_query = 1u;
    planDesc.hit_capacity_per_slot  = 1u;
    planDesc.result_slot_count      = 1u;
    planDesc.result_fields =
        matchingQueryBits ? static_cast<uint32_t>(OVSPATIALQUERY_BATCH_RESULT_GEOMETRY_IDENTITY) : 0u;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_overlap_plan(fixture.instance, &planDesc, &plan).status ==
            OVSPATIALQUERY_API_SUCCESS);

    double                          translation[3]    = {4.0, 0.0, 0.0};
    float                           rotation[4]       = {0.0f, 0.0f, 0.0f, 1.0f};
    uint64_t                        queryBits[2]      = {0u, matchingQueryBits ? UINT64_C(0x20) : UINT64_C(0x40)};
    uint64_t                        maximumHits       = forceTruncation ? 0u : 1u;
    const CudaDriver::DevicePointer deviceTranslation = upload(cuda, translation, stream);
    const CudaDriver::DevicePointer deviceRotation    = upload(cuda, rotation, stream);
    const CudaDriver::DevicePointer deviceQueryBits   = upload(cuda, queryBits, stream);
    const CudaDriver::DevicePointer deviceMaximumHits = upload(cuda, maximumHits, stream);
    InputArray                      translations(devicePointer<double>(deviceTranslation), 1, 3, kDLFloat, kCudaDevice);
    InputArray                      rotations(devicePointer<float>(deviceRotation), 1, 4, kDLFloat, kCudaDevice);
    InputArray                      bits(devicePointer<uint64_t>(deviceQueryBits), 1, 2, kDLUInt, kCudaDevice);
    InputArray                      rowMaximumHits(devicePointer<uint64_t>(deviceMaximumHits), 1, kDLUInt, kCudaDevice);
    ovspatialquery_overlap_batch_t  batch{};
    batch.ordinal.kind         = OVSPATIALQUERY_ORDINAL_HEAD;
    batch.query_count          = 1u;
    batch.query_bits           = &bits.value;
    batch.translations         = &translations.value;
    batch.rotations_xyzw       = shapeKind == OVSPATIALQUERY_OVERLAP_BOX ? &rotations.value : nullptr;
    batch.maximum_hits         = forceTruncation ? &rowMaximumHits.value : nullptr;
    batch.cuda_sync.has_stream = 1u;
    batch.cuda_sync.stream     = reinterpret_cast<uintptr_t>(stream);
    ovspatialquery_overlap_batch_reservation_t reservation{};
    if (shapeKind == OVSPATIALQUERY_OVERLAP_AABB)
    {
        // Row radii are SPHERE-only, and every overlap row needs a translation.
        // The translation buffer serves as a well-formed float64 row column.
        gPhase = "reject public CUDA AABB overlap launch columns";
        InputArray                     rowRadii(devicePointer<double>(deviceTranslation), 1, kDLFloat, kCudaDevice);
        ovspatialquery_overlap_batch_t rejected = batch;
        rejected.radii                          = &rowRadii.value;
        REQUIRE(ovspatialquery_enqueue_overlap_plan(fixture.instance, plan, &rejected, &reservation).status ==
                OVSPATIALQUERY_API_INVALID_ARGUMENT);
        rejected              = batch;
        rejected.translations = nullptr;
        REQUIRE(ovspatialquery_enqueue_overlap_plan(fixture.instance, plan, &rejected, &reservation).status ==
                OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    gPhase = "enqueue public CUDA overlap";
    if (forceRecoverableFailure)
    {
        ovspatialquery_test_set_cuda_failure_point(kCudaFailureFamilyLaunchAfterKernels);
    }
    ovspatialquery_enqueue_result_t operation =
        ovspatialquery_enqueue_overlap_plan(fixture.instance, plan, &batch, &reservation);
    REQUIRE(operation.status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(reservation.common.source_ordinal == fixture.sourceOrdinal);
    REQUIRE(reservation.common.truth == (precision == OVSPATIALQUERY_QUERY_EXACT
                                             ? OVSPATIALQUERY_HIT_EXACT_SURFACE
                                             : OVSPATIALQUERY_HIT_BOUNDS_CANDIDATE));
    REQUIRE(reservation.common.result_ready.value != 0u);
    REQUIRE(reservation.columns.object_ids.data != nullptr);
    REQUIRE(reservation.columns.object_ids.shape != nullptr);
    REQUIRE(reservation.columns.object_ids.shape[0] == 1);
    REQUIRE((matchingQueryBits && reservation.columns.geometry.types.shape != nullptr) ||
            (!matchingQueryBits && reservation.columns.geometry.types.shape == nullptr));

    if (forceRecoverableFailure)
    {
        gPhase = "verify public CUDA overlap recoverable safe-empty result";
        verifyRecoverableCudaFailure<ovspatialquery_overlap_batch_result_t>(
            cuda, consumer, fixture.instance, operation.op_id, reservation.common.result_ready,
            reservation.columns.hit_ranges,
            [&](ovspatialquery_overlap_batch_result_t* output)
            { return ovspatialquery_fetch_overlap_batch(fixture.instance, operation.op_id, output); });
        REQUIRE(destroyPlanWhenRetired(fixture.instance, plan));
        fixture.close();
        return;
    }

    ovspatialquery_op_wait_result_t waited{};
    REQUIRE(
        ovspatialquery_wait_op(fixture.instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
        OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(waited.state == OVSPATIALQUERY_OP_COMPLETED);
    ovspatialquery_overlap_batch_result_t result{};
    REQUIRE(ovspatialquery_fetch_overlap_batch(fixture.instance, operation.op_id, &result).status ==
            OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_overlap_batch_result_t repeated{};
    REQUIRE(ovspatialquery_fetch_overlap_batch(fixture.instance, operation.op_id, &repeated).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(repeated.columns.object_ids.data == result.columns.object_ids.data);
    REQUIRE(result.columns.object_ids.data == reservation.columns.object_ids.data);
    REQUIRE(result.columns.object_ids.shape != reservation.columns.object_ids.shape);
    cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(result.common.result_ready.value));
    uint32_t rowStatus     = UINT32_MAX;
    uint64_t stored        = UINT64_MAX;
    uint64_t offset        = UINT64_MAX;
    uint64_t count         = UINT64_MAX;
    uint8_t  truncated     = UINT8_MAX;
    uint64_t objectId      = 0u;
    int32_t  geometryType  = 0;
    uint64_t geometryValue = 0u;
    cuda.downloadAsync(&rowStatus, deviceAddress(result.columns.hit_ranges.query_status), sizeof(rowStatus), consumer);
    cuda.downloadAsync(&stored, deviceAddress(result.columns.hit_ranges.stored_hit_count), sizeof(stored), consumer);
    cuda.downloadAsync(&offset, deviceAddress(result.columns.hit_ranges.hit_offsets), sizeof(offset), consumer);
    cuda.downloadAsync(&count, deviceAddress(result.columns.hit_ranges.hit_counts), sizeof(count), consumer);
    cuda.downloadAsync(&truncated, deviceAddress(result.columns.hit_ranges.truncated), sizeof(truncated), consumer);
    cuda.downloadAsync(&objectId, deviceAddress(result.columns.object_ids), sizeof(objectId), consumer);
    if (matchingQueryBits && !forceTruncation)
    {
        cuda.downloadAsync(&geometryType, deviceAddress(result.columns.geometry.types), sizeof(geometryType), consumer);
        cuda.downloadAsync(&geometryValue, deviceAddress(result.columns.geometry.values), sizeof(geometryValue),
                           consumer);
    }
    // The fetched shapes are the operation's; read them before it retires.
    const uint64_t expectedStored = matchingQueryBits && !forceTruncation ? 1u : 0u;
    REQUIRE(result.columns.object_ids.shape[0] == static_cast<int64_t>(expectedStored));
    REQUIRE((matchingQueryBits && result.columns.geometry.types.shape != nullptr &&
             result.columns.geometry.types.shape[0] == static_cast<int64_t>(expectedStored)) ||
            (!matchingQueryBits && result.columns.geometry.types.shape == nullptr));
    retireAfterConsumer(cuda, consumer, fixture.instance, operation.op_id);
    REQUIRE(rowStatus == OVSPATIALQUERY_BATCH_ROW_SUCCESS);
    REQUIRE(stored == expectedStored && offset == 0u && count == expectedStored &&
            truncated == (matchingQueryBits && forceTruncation ? 1u : 0u));
    REQUIRE(expectedStored == 0u || objectId == fixture.objectValue);
    REQUIRE(expectedStored == 0u ||
            (geometryType == OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH && geometryValue == fixture.meshValue));
    REQUIRE(destroyPlanWhenRetired(fixture.instance, plan));
    fixture.close();
    REQUIRE(transferredEventDestroysWereInvoked());
}

// A float64 BOX row orients the device's exact test in double precision
// (ADR-0026), as the core batch test checks on the CPU, on the scene
// float64_box_row_scene.h describes. Rows 2e-9 beyond an upright box's +X face
// or a capsule's side miss with the float64 column and hit with the float32
// one; rows 2e-9 inside hit. The convex hull and the triangle-mesh quad must
// also miss and hit with float64 rows.
void testPublicCudaFloat64BoxRows()
{
    gPhase = "create float64 BOX row instance";
    const ovspatialquery_device_budget_t budget{{kDLCUDA, 0}, UINT64_C(64) << 20};
    const ovspatialquery_instance_desc_t desc{4u, 1u, UINT64_C(64) << 20, &budget, 1u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    gPhase = "publish float64 BOX row scene";
    Float64BoxRowScene scene(instance);

    gPhase = "create float64 BOX row CUDA plan";
    ovspatialquery_overlap_plan_desc_t planDesc{};
    planDesc.target     = populationRef(scene.population);
    planDesc.hit_mode   = OVSPATIALQUERY_HIT_MODE_MULTIPLE;
    planDesc.precision  = OVSPATIALQUERY_QUERY_EXACT;
    planDesc.shape.kind = OVSPATIALQUERY_OVERLAP_BOX;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        planDesc.shape.data.box.half_extents[axis] = Float64BoxRowScene::kHalfExtents[axis];
    }
    planDesc.device                         = {kDLCUDA, 0};
    planDesc.maximum_query_count            = 8u;
    planDesc.maximum_hits_per_query         = 2u;
    planDesc.hit_capacity_per_slot          = 16u;
    planDesc.result_slot_count              = 2u;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_overlap_plan(instance, &planDesc, &plan).status == OVSPATIALQUERY_API_SUCCESS);

    CudaDriver                      cuda(CudaDriver::Ownership::tracked);
    const CudaDriver::Stream        stream            = cuda.createStream();
    const CudaDriver::DevicePointer deviceTranslation = upload(cuda, scene.translations, stream);
    const CudaDriver::DevicePointer deviceRotation64  = upload(cuda, scene.rotations64, stream);
    const CudaDriver::DevicePointer deviceRotation32  = upload(cuda, scene.rotations32, stream);
    InputArray translationColumn(devicePointer<double>(deviceTranslation), 8, 3, kDLFloat, kCudaDevice);
    InputArray rotation64Column(devicePointer<double>(deviceRotation64), 8, 4, kDLFloat, kCudaDevice);
    InputArray rotation32Column(devicePointer<float>(deviceRotation32), 8, 4, kDLFloat, kCudaDevice);
    const auto hitCounts = [&](const InputArray& rotations)
    {
        gPhase = "enqueue float64 BOX row CUDA overlap";
        ovspatialquery_overlap_batch_t batch{};
        batch.ordinal.kind         = OVSPATIALQUERY_ORDINAL_HEAD;
        batch.query_count          = 8u;
        batch.translations         = &translationColumn.value;
        batch.rotations_xyzw       = &rotations.value;
        batch.cuda_sync.has_stream = 1u;
        batch.cuda_sync.stream     = reinterpret_cast<uintptr_t>(stream);
        ovspatialquery_overlap_batch_reservation_t reservation{};
        ovspatialquery_enqueue_result_t            operation =
            ovspatialquery_enqueue_overlap_plan(instance, plan, &batch, &reservation);
        REQUIRE(operation.status == OVSPATIALQUERY_API_SUCCESS);
        ovspatialquery_op_wait_result_t waited{};
        REQUIRE(ovspatialquery_wait_op(instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
                    OVSPATIALQUERY_API_SUCCESS &&
                waited.state == OVSPATIALQUERY_OP_COMPLETED);
        ovspatialquery_overlap_batch_result_t result{};
        REQUIRE(ovspatialquery_fetch_overlap_batch(instance, operation.op_id, &result).status ==
                OVSPATIALQUERY_API_SUCCESS);
        cuda.wait(stream, reinterpret_cast<CudaDriver::Event>(result.common.result_ready.value));
        std::array<uint64_t, 8u> counts{};
        cuda.downloadAsync(counts.data(), deviceAddress(result.columns.hit_ranges.hit_counts), sizeof(counts), stream);
        retireAfterConsumer(cuda, stream, instance, operation.op_id);
        REQUIRE(transferredEventDestroysWereInvoked());
        return counts;
    };
    // The hull and mesh rows of the float32 column are not checked: 2e-9 is
    // below their float32 resolution.
    const std::array<uint64_t, 8u> kFloat64Hits{0u, 1u, 0u, 1u, 0u, 1u, 0u, 1u};
    REQUIRE(hitCounts(rotation64Column) == kFloat64Hits);
    const std::array<uint64_t, 8u> float32Hits = hitCounts(rotation32Column);
    REQUIRE(float32Hits[0] == 1u && float32Hits[1] == 1u && float32Hits[2] == 1u && float32Hits[3] == 1u);

    REQUIRE(destroyPlanWhenRetired(instance, plan));
    scene.destroy();
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

// CUDA box overlaps of a convex hull agree with the CPU's on
// hull_box_gap_scene.h's rows: boxes 1e-3 and 1e-4 from the hull overlap
// nothing and boxes 1e-4 deep overlap it, with either rotation column.
void testCudaHullBoxOverlapsResolveSmallGaps()
{
    gPhase = "publish hull box-gap scene";
    const ovspatialquery_device_budget_t budget{{kDLCUDA, 0}, UINT64_C(64) << 20};
    const ovspatialquery_instance_desc_t desc{4u, 1u, UINT64_C(64) << 20, &budget, 1u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    HullBoxGapScene                    scene(instance);
    constexpr uint32_t                 kRows = HullBoxGapScene::kRows;
    ovspatialquery_overlap_plan_desc_t planDesc{};
    planDesc.target     = populationRef(scene.population);
    planDesc.hit_mode   = OVSPATIALQUERY_HIT_MODE_ANY;
    planDesc.precision  = OVSPATIALQUERY_QUERY_EXACT;
    planDesc.shape.kind = OVSPATIALQUERY_OVERLAP_BOX;
    for (double& extent : planDesc.shape.data.box.half_extents)
    {
        extent = HullBoxGapScene::kHalfExtent;
    }
    planDesc.device                         = kCudaDevice;
    planDesc.maximum_query_count            = kRows;
    planDesc.maximum_hits_per_query         = 1u;
    planDesc.hit_capacity_per_slot          = kRows;
    planDesc.result_slot_count              = 1u;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_overlap_plan(instance, &planDesc, &plan).status == OVSPATIALQUERY_API_SUCCESS);

    CudaDriver                      cuda(CudaDriver::Ownership::tracked);
    const CudaDriver::Stream        stream            = cuda.createStream();
    const CudaDriver::DevicePointer deviceTranslation = upload(cuda, scene.translations, stream);
    const CudaDriver::DevicePointer deviceRotation64  = upload(cuda, scene.rotations64, stream);
    const CudaDriver::DevicePointer deviceRotation32  = upload(cuda, scene.rotations32, stream);
    InputArray translationColumn(devicePointer<double>(deviceTranslation), kRows, 3, kDLFloat, kCudaDevice);
    InputArray rotation64Column(devicePointer<double>(deviceRotation64), kRows, 4, kDLFloat, kCudaDevice);
    InputArray rotation32Column(devicePointer<float>(deviceRotation32), kRows, 4, kDLFloat, kCudaDevice);
    const InputArray* const rotationColumns[2]{&rotation32Column, &rotation64Column};
    for (const InputArray* rotations : rotationColumns)
    {
        gPhase = "CUDA box overlap of a hull across a small gap";
        ovspatialquery_overlap_batch_t batch{};
        batch.ordinal.kind         = OVSPATIALQUERY_ORDINAL_HEAD;
        batch.query_count          = kRows;
        batch.translations         = &translationColumn.value;
        batch.rotations_xyzw       = &rotations->value;
        batch.cuda_sync.has_stream = 1u;
        batch.cuda_sync.stream     = reinterpret_cast<uintptr_t>(stream);
        ovspatialquery_overlap_batch_reservation_t reservation{};
        ovspatialquery_enqueue_result_t            operation =
            ovspatialquery_enqueue_overlap_plan(instance, plan, &batch, &reservation);
        REQUIRE(operation.status == OVSPATIALQUERY_API_SUCCESS);
        ovspatialquery_op_wait_result_t waited{};
        REQUIRE(ovspatialquery_wait_op(instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
                    OVSPATIALQUERY_API_SUCCESS &&
                waited.state == OVSPATIALQUERY_OP_COMPLETED);
        ovspatialquery_overlap_batch_result_t result{};
        REQUIRE(ovspatialquery_fetch_overlap_batch(instance, operation.op_id, &result).status ==
                OVSPATIALQUERY_API_SUCCESS);
        cuda.wait(stream, reinterpret_cast<CudaDriver::Event>(result.common.result_ready.value));
        std::array<uint64_t, kRows> counts{};
        cuda.downloadAsync(counts.data(), deviceAddress(result.columns.hit_ranges.hit_counts), sizeof(counts), stream);
        retireAfterConsumer(cuda, stream, instance, operation.op_id);
        REQUIRE(transferredEventDestroysWereInvoked());
        REQUIRE(std::memcmp(counts.data(), scene.overlaps, sizeof(scene.overlaps)) == 0);
    }
    REQUIRE(destroyPlanWhenRetired(instance, plan));
    cuda.free(deviceTranslation);
    cuda.free(deviceRotation64);
    cuda.free(deviceRotation32);
    scene.destroy();
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

// An exact BOX overlap reports primitives it only touches, as the CPU paths do
// (issue #74). The query box spans x 7..9.5 about y = 14; a box's -X face rests
// on its +X face and a sphere touches its -X face. Each is a pair whose bounds
// test, scaled by the y coordinate, rounds one ulp apart.
void testCudaExactBoxOverlapKeepsTouchingPrimitives()
{
    gPhase = "publish touching primitives";
    const ovspatialquery_device_budget_t budget{{kDLCUDA, 0}, UINT64_C(64) << 20};
    const ovspatialquery_instance_desc_t desc{4u, 1u, UINT64_C(64) << 20, &budget, 1u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    constexpr uint64_t                          kBox          = 41u;
    constexpr uint64_t                          kSphere       = 42u;
    uint64_t                                    boxValue      = kBox;
    double                                      boxExtents[3] = {0.5, 0.375, 0.5};
    InputArray                                  boxValues(&boxValue, 1, kDLUInt);
    InputArray                                  boxExtentArray(boxExtents, 1, 3, kDLFloat);
    const ovspatialquery_box_definition_table_t boxes{&boxValues.value, &boxExtentArray.value};
    waitAndDestroy(instance, ovspatialquery_enqueue_box_registration(instance, &boxes));
    registerSphere(instance, kSphere, 0.5);
    const ovspatialquery_population_handle_t population = createExplicitPopulation(instance, 1u);
    InstanceRows                             rows;
    rows.add(kBox, OVSPATIALQUERY_GEOMETRY_BOX, kBox, 10.0, 14.0, 0.5);
    rows.add(kSphere, OVSPATIALQUERY_GEOMETRY_SPHERE, kSphere, 6.5, 14.0, 0.5);
    const ovspatialquery_explicit_population_set_t set{3, rows.table(), nullptr, 0u};
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));

    gPhase = "create touching BOX overlap CUDA plan";
    ovspatialquery_overlap_plan_desc_t planDesc{};
    planDesc.target                         = populationRef(population);
    planDesc.hit_mode                       = OVSPATIALQUERY_HIT_MODE_MULTIPLE;
    planDesc.precision                      = OVSPATIALQUERY_QUERY_EXACT;
    planDesc.shape.kind                     = OVSPATIALQUERY_OVERLAP_BOX;
    planDesc.shape.data.box.half_extents[0] = 1.25;
    planDesc.shape.data.box.half_extents[1] = 1.25;
    planDesc.shape.data.box.half_extents[2] = 0.625;
    planDesc.device                         = {kDLCUDA, 0};
    planDesc.maximum_query_count            = 1u;
    planDesc.maximum_hits_per_query         = 2u;
    planDesc.hit_capacity_per_slot          = 2u;
    planDesc.result_slot_count              = 1u;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_overlap_plan(instance, &planDesc, &plan).status == OVSPATIALQUERY_API_SUCCESS);

    gPhase = "enqueue touching BOX overlap";
    CudaDriver                      cuda(CudaDriver::Ownership::tracked);
    const CudaDriver::Stream        stream            = cuda.createStream();
    double                          translation[3]    = {8.25, 14.0, 0.5};
    float                           rotation[4]       = {0.0f, 0.0f, 0.0f, 1.0f};
    const CudaDriver::DevicePointer deviceTranslation = upload(cuda, translation, stream);
    const CudaDriver::DevicePointer deviceRotation    = upload(cuda, rotation, stream);
    InputArray                      translations(devicePointer<double>(deviceTranslation), 1, 3, kDLFloat, kCudaDevice);
    InputArray                      rotations(devicePointer<float>(deviceRotation), 1, 4, kDLFloat, kCudaDevice);
    ovspatialquery_overlap_batch_t  batch{};
    batch.ordinal.kind         = OVSPATIALQUERY_ORDINAL_HEAD;
    batch.query_count          = 1u;
    batch.translations         = &translations.value;
    batch.rotations_xyzw       = &rotations.value;
    batch.cuda_sync.has_stream = 1u;
    batch.cuda_sync.stream     = reinterpret_cast<uintptr_t>(stream);
    ovspatialquery_overlap_batch_reservation_t reservation{};
    ovspatialquery_enqueue_result_t            operation =
        ovspatialquery_enqueue_overlap_plan(instance, plan, &batch, &reservation);
    REQUIRE(operation.status == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_op_wait_result_t waited{};
    REQUIRE(ovspatialquery_wait_op(instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
                OVSPATIALQUERY_API_SUCCESS &&
            waited.state == OVSPATIALQUERY_OP_COMPLETED);
    ovspatialquery_overlap_batch_result_t result{};
    REQUIRE(ovspatialquery_fetch_overlap_batch(instance, operation.op_id, &result).status ==
            OVSPATIALQUERY_API_SUCCESS);
    cuda.wait(stream, reinterpret_cast<CudaDriver::Event>(result.common.result_ready.value));
    uint64_t stored       = UINT64_MAX;
    uint64_t objectIds[2] = {};
    cuda.downloadAsync(&stored, deviceAddress(result.columns.hit_ranges.stored_hit_count), sizeof(stored), stream);
    cuda.downloadAsync(objectIds, deviceAddress(result.columns.object_ids), sizeof(objectIds), stream);
    retireAfterConsumer(cuda, stream, instance, operation.op_id);
    REQUIRE(transferredEventDestroysWereInvoked());
    REQUIRE(stored == 2u && objectIds[0] + objectIds[1] == kBox + kSphere && objectIds[0] != objectIds[1]);

    REQUIRE(destroyPlanWhenRetired(instance, plan));
    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_BOX, kBox}).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_SPHERE, kSphere}).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

// Two sphere objects on the X axis, far enough apart that "closest along the
// ray" and "closest object the view includes" disagree unless a plan
// actually honors view membership rather than just scanning the population.
constexpr uint64_t kNearObject        = 1u;
constexpr uint64_t kFarObject         = 2u;
constexpr uint64_t kSphereDefinition  = 9u;
constexpr uint64_t kParticipationBits = UINT64_C(0x1);

struct TwoSphereFixture
{
    TwoSphereFixture()
    {
        gPhase = "publish two-sphere population";
        const ovspatialquery_device_budget_t budget{{kDLCUDA, 0}, UINT64_C(64) << 20};
        const ovspatialquery_instance_desc_t desc{4u, 1u, UINT64_C(64) << 20, &budget, 1u};
        REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_SUCCESS);

        double                                         radius          = 0.5;
        uint64_t                                       definitionValue = kSphereDefinition;
        InputArray                                     definitionValues(&definitionValue, 1, kDLUInt);
        InputArray                                     radii(&radius, 1, kDLFloat);
        const ovspatialquery_sphere_definition_table_t spheres{&definitionValues.value, &radii.value};
        waitAndDestroy(instance, ovspatialquery_enqueue_sphere_registration(instance, &spheres));

        const ovspatialquery_population_desc_t populationDesc{0u, 1u};
        REQUIRE(ovspatialquery_create_explicit_population(instance, &populationDesc, &population).status ==
                OVSPATIALQUERY_API_SUCCESS);

        uint64_t   objects[2]       = {kNearObject, kFarObject};
        int32_t    types[2]         = {OVSPATIALQUERY_GEOMETRY_SPHERE, OVSPATIALQUERY_GEOMETRY_SPHERE};
        uint64_t   geometry[2]      = {kSphereDefinition, kSphereDefinition};
        double     translations[6]  = {2.0, 0.0, 0.0, 6.0, 0.0, 0.0};
        float      rotations[8]     = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
        uint64_t   participation[4] = {kParticipationBits, 0u, kParticipationBits, 0u};
        InputArray objectArray(objects, 2, kDLUInt);
        InputArray typeArray(types, 2, kDLInt);
        InputArray geometryArray(geometry, 2, kDLUInt);
        InputArray translationArray(translations, 2, 3, kDLFloat);
        InputArray rotationArray(rotations, 2, 4, kDLFloat);
        InputArray participationArray(participation, 2, 2, kDLUInt);
        const ovspatialquery_explicit_population_set_t set{sourceOrdinal,
                                                           {&objectArray.value, &typeArray.value, &geometryArray.value,
                                                            &translationArray.value, &rotationArray.value, nullptr,
                                                            &participationArray.value, nullptr},
                                                           nullptr,
                                                           0u};
        waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));
    }

    void close()
    {
        REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(
            ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_SPHERE, kSphereDefinition})
                .status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
        instance = OVSPATIALQUERY_INVALID_INSTANCE;
    }

    ovspatialquery_instance_handle_t   instance      = OVSPATIALQUERY_INVALID_INSTANCE;
    ovspatialquery_population_handle_t population    = OVSPATIALQUERY_INVALID_POPULATION;
    const ovspatialquery_ordinal_t     sourceOrdinal = 5;
};

// A plan launched against a VIEW filters by the view's membership: the near
// object is the physically closer candidate but the view excludes it, so the
// far object the view names must be reported instead.
std::promise<void>       gRetirementPaused;
std::shared_future<void> gResumeRetirement;

void pauseAfterRetiredSlot()
{
    gRetirementPaused.set_value();
    if (gResumeRetirement.wait_for(std::chrono::seconds(30)) != std::future_status::ready)
    {
        std::fprintf(stderr, "timed out waiting to resume CUDA retirement\n");
        std::_Exit(EXIT_FAILURE);
    }
}

void testCudaPlanHonorsViewMembership()
{
    TwoSphereFixture fixture;

    gPhase                                               = "create CUDA raycast plan on a view";
    uint64_t                                  includedId = kFarObject;
    InputArray                                includedArray(&includedId, 1, kDLUInt);
    const ovspatialquery_explicit_view_desc_t viewDesc{&includedArray.value};
    ovspatialquery_view_handle_t              view = OVSPATIALQUERY_INVALID_VIEW;
    REQUIRE(ovspatialquery_create_explicit_view(fixture.instance, fixture.population, &viewDesc, &view).status ==
            OVSPATIALQUERY_API_SUCCESS);

    CudaDriver               cuda(CudaDriver::Ownership::tracked);
    const CudaDriver::Stream producer   = cuda.createStream();
    const CudaDriver::Stream consumer   = cuda.createStream();
    const CudaDriver::Event  queryReady = cuda.createEvent();

    ovspatialquery_raycast_plan_desc_t planDesc{};
    planDesc.target                         = viewRef(view);
    planDesc.hit_mode                       = OVSPATIALQUERY_HIT_MODE_CLOSEST;
    planDesc.precision                      = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    planDesc.device                         = {kDLCUDA, 0};
    planDesc.maximum_query_count            = 1u;
    planDesc.maximum_hits_per_query         = 1u;
    planDesc.hit_capacity_per_slot          = 1u;
    planDesc.result_slot_count              = 1u;
    planDesc.result_fields                  = OVSPATIALQUERY_BATCH_RESULT_DISTANCE;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_raycast_plan(fixture.instance, &planDesc, &plan).status ==
            OVSPATIALQUERY_API_SUCCESS);

    double                          origin[3]             = {0.0, 0.0, 0.0};
    float                           direction[3]          = {1.0f, 0.0f, 0.0f};
    double                          maximumDistance       = 100.0;
    uint64_t                        queryBits[2]          = {kParticipationBits, 0u};
    const CudaDriver::DevicePointer deviceOrigin          = upload(cuda, origin, producer);
    const CudaDriver::DevicePointer deviceDirection       = upload(cuda, direction, producer);
    const CudaDriver::DevicePointer deviceMaximumDistance = upload(cuda, maximumDistance, producer);
    const CudaDriver::DevicePointer deviceQueryBits       = upload(cuda, queryBits, producer);
    cuda.record(queryReady, producer);

    InputArray origins(devicePointer<double>(deviceOrigin), 1, 3, kDLFloat, kCudaDevice);
    InputArray directions(devicePointer<float>(deviceDirection), 1, 3, kDLFloat, kCudaDevice);
    InputArray maximumDistances(devicePointer<double>(deviceMaximumDistance), 1, kDLFloat, kCudaDevice);
    InputArray bits(devicePointer<uint64_t>(deviceQueryBits), 1, 2, kDLUInt, kCudaDevice);

    ovspatialquery_raycast_batch_t batch{};
    batch.ordinal.kind         = OVSPATIALQUERY_ORDINAL_HEAD;
    batch.query_count          = 1u;
    batch.query_bits           = &bits.value;
    batch.origins              = &origins.value;
    batch.unit_directions      = &directions.value;
    batch.max_distances        = &maximumDistances.value;
    batch.cuda_sync.has_stream = 1u;
    batch.cuda_sync.stream     = reinterpret_cast<uintptr_t>(producer);
    batch.cuda_sync.wait_event = reinterpret_cast<uintptr_t>(queryReady);
    ovspatialquery_raycast_batch_reservation_t reservation{};
    gPhase = "enqueue CUDA raycast on a view";
    ovspatialquery_enqueue_result_t operation =
        ovspatialquery_enqueue_raycast_plan(fixture.instance, plan, &batch, &reservation);
    REQUIRE(operation.status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(operation.op_id != OVSPATIALQUERY_INVALID_OP);

    ovspatialquery_op_wait_result_t waited{};
    REQUIRE(
        ovspatialquery_wait_op(fixture.instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
        OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(waited.state == OVSPATIALQUERY_OP_COMPLETED);
    ovspatialquery_raycast_batch_result_t result{};
    REQUIRE(ovspatialquery_fetch_raycast_batch(fixture.instance, operation.op_id, &result).status ==
            OVSPATIALQUERY_API_SUCCESS);

    cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(result.common.result_ready.value));
    uint64_t stored   = UINT64_MAX;
    uint64_t objectId = 0u;
    double   distance = -1.0;
    cuda.downloadAsync(&stored, deviceAddress(result.columns.hit_ranges.stored_hit_count), sizeof(stored), consumer);
    cuda.downloadAsync(&objectId, deviceAddress(result.columns.object_ids), sizeof(objectId), consumer);
    cuda.downloadAsync(&distance, deviceAddress(result.columns.contact.candidate.distances), sizeof(distance),
                       consumer);
    // Keep the reaper's owner alive after its result slot becomes free. Plan
    // destruction must still release cached geometry before reporting success.
    std::promise<void> resumeRetirement;
    gResumeRetirement     = resumeRetirement.get_future().share();
    auto retirementPaused = gRetirementPaused.get_future();
    setFamilyBatchPlanRetiredSlotHookForTest(pauseAfterRetiredSlot);
    retireAfterConsumer(cuda, consumer, fixture.instance, operation.op_id);
    REQUIRE(retirementPaused.wait_for(std::chrono::seconds(10)) == std::future_status::ready);
    setFamilyBatchPlanRetiredSlotHookForTest(nullptr);

    REQUIRE(stored == 1u && objectId == kFarObject && distance > 5.0 && distance < 6.0);

    REQUIRE(transferredEventDestroysWereInvoked());
    REQUIRE(destroyPlanWhenRetired(fixture.instance, plan));
    REQUIRE(ovspatialquery_destroy_view(fixture.instance, view).status == OVSPATIALQUERY_API_SUCCESS);
    gPhase = "destroy geometry while the reaper still holds the retired plan owner";
    REQUIRE(ovspatialquery_destroy_population(fixture.instance, fixture.population).status ==
            OVSPATIALQUERY_API_SUCCESS);
    const auto geometryStatus = ovspatialquery_destroy_geometry_definition(
                                    fixture.instance, {OVSPATIALQUERY_GEOMETRY_SPHERE, kSphereDefinition})
                                    .status;
    if (geometryStatus != OVSPATIALQUERY_API_SUCCESS)
        std::fprintf(stderr, "retired plan geometry destruction status=%d\n", static_cast<int>(geometryStatus));
    REQUIRE(geometryStatus == OVSPATIALQUERY_API_SUCCESS);
    resumeRetirement.set_value();
    REQUIRE(ovspatialquery_destroy_instance(fixture.instance).status == OVSPATIALQUERY_API_SUCCESS);
}

// An oriented-BOX overlap plan dispatches on CUDA at BOUNDS_CANDIDATE and
// reports the sphere it is centered on, both with a float32 rotation and with
// a float64 rotation at an aligned non-zero byte offset plus row half extents
// (ADR-0026, ADR-0025). The kernels load rotations at their natural width, so
// a misaligned float64 rotation column, like a missing one, is the enqueue's.
void testCudaBoxOverlapCandidateDispatch()
{
    TwoSphereFixture fixture;

    CudaDriver               cuda(CudaDriver::Ownership::tracked);
    const CudaDriver::Stream producer   = cuda.createStream();
    const CudaDriver::Stream consumer   = cuda.createStream();
    const CudaDriver::Event  queryReady = cuda.createEvent();

    gPhase = "create CUDA candidate BOX overlap plan";
    ovspatialquery_overlap_plan_desc_t planDesc{};
    planDesc.target                         = populationRef(fixture.population);
    planDesc.hit_mode                       = OVSPATIALQUERY_HIT_MODE_ANY;
    planDesc.precision                      = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    planDesc.shape.kind                     = OVSPATIALQUERY_OVERLAP_BOX;
    planDesc.shape.data.box.half_extents[0] = 0.6;
    planDesc.shape.data.box.half_extents[1] = 0.6;
    planDesc.shape.data.box.half_extents[2] = 0.6;
    planDesc.device                         = {kDLCUDA, 0};
    planDesc.maximum_query_count            = 1u;
    planDesc.maximum_hits_per_query         = 1u;
    planDesc.hit_capacity_per_slot          = 1u;
    planDesc.result_slot_count              = 1u;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_overlap_plan(fixture.instance, &planDesc, &plan).status ==
            OVSPATIALQUERY_API_SUCCESS);

    // The query box is centered on the near object with identity rotation. The
    // float64 rotation starts one element in, after a NaN it must not read.
    double                          translation[3]    = {2.0, 0.0, 0.0};
    float                           rotation[4]       = {0.0f, 0.0f, 0.0f, 1.0f};
    uint64_t                        queryBits[2]      = {kParticipationBits, 0u};
    double                          rotation64[5]     = {std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0, 1.0};
    double                          extents[3]        = {0.6, 0.6, 0.6};
    const CudaDriver::DevicePointer deviceTranslation = upload(cuda, translation, producer);
    const CudaDriver::DevicePointer deviceRotation    = upload(cuda, rotation, producer);
    const CudaDriver::DevicePointer deviceQueryBits   = upload(cuda, queryBits, producer);
    const CudaDriver::DevicePointer deviceRotation64  = upload(cuda, rotation64, producer);
    const CudaDriver::DevicePointer deviceExtents     = upload(cuda, extents, producer);
    cuda.record(queryReady, producer);

    InputArray translations(devicePointer<double>(deviceTranslation), 1, 3, kDLFloat, kCudaDevice);
    InputArray rotations(devicePointer<float>(deviceRotation), 1, 4, kDLFloat, kCudaDevice);
    InputArray bits(devicePointer<uint64_t>(deviceQueryBits), 1, 2, kDLUInt, kCudaDevice);
    InputArray rotations64(devicePointer<double>(deviceRotation64), 1, 4, kDLFloat, kCudaDevice);
    rotations64.value.tensor.byte_offset = sizeof(double);
    InputArray rowExtents(devicePointer<double>(deviceExtents), 1, 3, kDLFloat, kCudaDevice);

    ovspatialquery_overlap_batch_t batch{};
    batch.ordinal.kind         = OVSPATIALQUERY_ORDINAL_HEAD;
    batch.query_count          = 1u;
    batch.query_bits           = &bits.value;
    batch.translations         = &translations.value;
    batch.rotations_xyzw       = &rotations.value;
    batch.cuda_sync.has_stream = 1u;
    batch.cuda_sync.stream     = reinterpret_cast<uintptr_t>(producer);
    batch.cuda_sync.wait_event = reinterpret_cast<uintptr_t>(queryReady);
    ovspatialquery_overlap_batch_reservation_t reservation{};
    {
        gPhase = "reject misaligned or missing CUDA BOX rotations";
        InputArray misaligned(devicePointer<double>(deviceRotation64), 1, 4, kDLFloat, kCudaDevice);
        misaligned.value.tensor.byte_offset          = 4u;
        ovspatialquery_overlap_batch_t rejectedBatch = batch;
        rejectedBatch.rotations_xyzw                 = &misaligned.value;
        ovspatialquery_overlap_batch_reservation_t rejectedReservation{};
        REQUIRE(
            ovspatialquery_enqueue_overlap_plan(fixture.instance, plan, &rejectedBatch, &rejectedReservation).status ==
            OVSPATIALQUERY_API_INVALID_ARGUMENT);
        rejectedBatch.rotations_xyzw = nullptr;
        REQUIRE(
            ovspatialquery_enqueue_overlap_plan(fixture.instance, plan, &rejectedBatch, &rejectedReservation).status ==
            OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    {
        gPhase                               = "CUDA candidate BOX overlap with float64 rotation and row extents";
        ovspatialquery_overlap_batch_t sized = batch;
        sized.rotations_xyzw                 = &rotations64.value;
        sized.half_extents                   = &rowExtents.value;
        ovspatialquery_overlap_batch_reservation_t sizedReservation{};
        ovspatialquery_enqueue_result_t            sizedOperation =
            ovspatialquery_enqueue_overlap_plan(fixture.instance, plan, &sized, &sizedReservation);
        REQUIRE(sizedOperation.status == OVSPATIALQUERY_API_SUCCESS);
        ovspatialquery_op_wait_result_t sizedWaited{};
        REQUIRE(ovspatialquery_wait_op(fixture.instance, sizedOperation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE,
                                       &sizedWaited)
                    .status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(sizedWaited.state == OVSPATIALQUERY_OP_COMPLETED);
        ovspatialquery_overlap_batch_result_t sizedResult{};
        REQUIRE(ovspatialquery_fetch_overlap_batch(fixture.instance, sizedOperation.op_id, &sizedResult).status ==
                OVSPATIALQUERY_API_SUCCESS);
        cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(sizedResult.common.result_ready.value));
        uint64_t sizedStored = UINT64_MAX;
        uint64_t sizedObject = 0u;
        cuda.downloadAsync(&sizedStored, deviceAddress(sizedResult.columns.hit_ranges.stored_hit_count),
                           sizeof(sizedStored), consumer);
        cuda.downloadAsync(&sizedObject, deviceAddress(sizedResult.columns.object_ids), sizeof(sizedObject), consumer);
        // Hold the reaper back: the next launch must reclaim the slot itself.
        REQUIRE(setInstanceCudaReaperPausedForTest(fixture.instance, true) == OVSPATIALQUERY_API_SUCCESS);
        retireAfterConsumer(cuda, consumer, fixture.instance, sizedOperation.op_id);
        REQUIRE(sizedStored == 1u && sizedObject == kNearObject);
    }
    // The plan has one slot. The sized row's consumer finished before its
    // retirement, so the slot is free for the very next launch.
    gPhase = "CUDA candidate BOX overlap with float32 rotation";
    ovspatialquery_enqueue_result_t operation =
        ovspatialquery_enqueue_overlap_plan(fixture.instance, plan, &batch, &reservation);
    REQUIRE(operation.status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(transferredEventDestroysWereInvoked());
    REQUIRE(setInstanceCudaReaperPausedForTest(fixture.instance, false) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(reservation.common.truth == OVSPATIALQUERY_HIT_BOUNDS_CANDIDATE);

    ovspatialquery_op_wait_result_t waited{};
    REQUIRE(
        ovspatialquery_wait_op(fixture.instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
        OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(waited.state == OVSPATIALQUERY_OP_COMPLETED);
    ovspatialquery_overlap_batch_result_t result{};
    REQUIRE(ovspatialquery_fetch_overlap_batch(fixture.instance, operation.op_id, &result).status ==
            OVSPATIALQUERY_API_SUCCESS);

    cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(result.common.result_ready.value));
    uint64_t stored   = UINT64_MAX;
    uint64_t objectId = 0u;
    cuda.downloadAsync(&stored, deviceAddress(result.columns.hit_ranges.stored_hit_count), sizeof(stored), consumer);
    cuda.downloadAsync(&objectId, deviceAddress(result.columns.object_ids), sizeof(objectId), consumer);
    retireAfterConsumer(cuda, consumer, fixture.instance, operation.op_id);

    REQUIRE(stored == 1u && objectId == kNearObject);

    REQUIRE(transferredEventDestroysWereInvoked());
    REQUIRE(destroyPlanWhenRetired(fixture.instance, plan));
    fixture.close();
}

void testPublicCudaMalformedStatusAtomicity()
{
    ExactMeshFixture fixture;
    gPhase = "reject malformed CUDA family capacities";
    ovspatialquery_raycast_plan_desc_t ray{};
    ray.target                 = populationRef(fixture.population);
    ray.hit_mode               = OVSPATIALQUERY_HIT_MODE_CLOSEST;
    ray.precision              = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    ray.device                 = {kDLCUDA, 0};
    ray.maximum_query_count    = 1u;
    ray.maximum_hits_per_query = 1u;
    ray.hit_capacity_per_slot  = 1u;
    ray.result_slot_count      = 1u;
    ovspatialquery_sphere_sweep_plan_desc_t sweep{};
    sweep.target                = ray.target;
    sweep.precision             = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    sweep.radius                = 0.5;
    sweep.device                = {kDLCUDA, 0};
    sweep.maximum_query_count   = 1u;
    sweep.result_slot_count     = 1u;
    sweep.hit_capacity_per_slot = 1u;
    ovspatialquery_overlap_plan_desc_t overlap{};
    overlap.target                   = ray.target;
    overlap.hit_mode                 = OVSPATIALQUERY_HIT_MODE_ANY;
    overlap.precision                = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    overlap.shape.kind               = OVSPATIALQUERY_OVERLAP_SPHERE;
    overlap.shape.data.sphere.radius = 0.5;
    overlap.device                   = {kDLCUDA, 0};
    overlap.maximum_query_count      = 1u;
    overlap.maximum_hits_per_query   = 1u;
    overlap.hit_capacity_per_slot    = 1u;
    overlap.result_slot_count        = 1u;

    ovspatialquery_batch_plan_handle_t rejectedPlan = ovspatialquery_batch_plan_handle_t{UINT64_C(0xa5a5a5a5)};
    ovspatialquery_raycast_plan_desc_t invalidRay   = ray;
    invalidRay.maximum_query_count                  = 0u;
    REQUIRE(ovspatialquery_create_raycast_plan(fixture.instance, &invalidRay, &rejectedPlan).status ==
            OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(rejectedPlan == OVSPATIALQUERY_INVALID_BATCH_PLAN);
    rejectedPlan                                         = ovspatialquery_batch_plan_handle_t{UINT64_C(0xa5a5a5a5)};
    ovspatialquery_sphere_sweep_plan_desc_t invalidSweep = sweep;
    invalidSweep.hit_capacity_per_slot = static_cast<uint64_t>((std::numeric_limits<int64_t>::max)()) + 1u;
    REQUIRE(ovspatialquery_create_sphere_sweep_plan(fixture.instance, &invalidSweep, &rejectedPlan).status ==
            OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(rejectedPlan == OVSPATIALQUERY_INVALID_BATCH_PLAN);
    rejectedPlan                                      = ovspatialquery_batch_plan_handle_t{UINT64_C(0xa5a5a5a5)};
    ovspatialquery_overlap_plan_desc_t invalidOverlap = overlap;
    invalidOverlap.maximum_query_count                = UINT64_MAX;
    REQUIRE(ovspatialquery_create_overlap_plan(fixture.instance, &invalidOverlap, &rejectedPlan).status ==
            OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(rejectedPlan == OVSPATIALQUERY_INVALID_BATCH_PLAN);

    // An in-domain capacity whose device byte charge does not fit in 64 bits
    // exceeds any device budget.
    gPhase                                     = "refuse a CUDA plan whose byte size overflows";
    ovspatialquery_raycast_plan_desc_t hugeRay = ray;
    hugeRay.maximum_query_count                = UINT64_C(1) << 62;
    REQUIRE(ovspatialquery_create_raycast_plan(fixture.instance, &hugeRay, &rejectedPlan).status ==
            OVSPATIALQUERY_API_DEVICE_BUDGET);
    REQUIRE(rejectedPlan == OVSPATIALQUERY_INVALID_BATCH_PLAN);

    ovspatialquery_batch_plan_handle_t rayPlan     = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    ovspatialquery_batch_plan_handle_t sweepPlan   = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    ovspatialquery_batch_plan_handle_t overlapPlan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_raycast_plan(fixture.instance, &ray, &rayPlan).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_create_sphere_sweep_plan(fixture.instance, &sweep, &sweepPlan).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_create_overlap_plan(fixture.instance, &overlap, &overlapPlan).status ==
            OVSPATIALQUERY_API_SUCCESS);

    gPhase = "reject zero CUDA stream atomically for every family";
    ovspatialquery_raycast_batch_t rayBatch{};
    rayBatch.ordinal.kind = OVSPATIALQUERY_ORDINAL_HEAD;
    rayBatch.query_count  = 1u;
    ovspatialquery_raycast_batch_reservation_t rayReservation{};
    std::memset(&rayReservation, 0xa5, sizeof(rayReservation));
    const ovspatialquery_raycast_batch_reservation_t raySentinel = rayReservation;
    const ovspatialquery_enqueue_result_t            rejectedRay =
        ovspatialquery_enqueue_raycast_plan(fixture.instance, rayPlan, &rayBatch, &rayReservation);
    REQUIRE(rejectedRay.status == OVSPATIALQUERY_API_INVALID_ARGUMENT &&
            rejectedRay.op_id == OVSPATIALQUERY_INVALID_OP);
    REQUIRE(std::memcmp(&rayReservation, &raySentinel, sizeof(rayReservation)) == 0);

    ovspatialquery_sphere_sweep_batch_t sweepBatch{};
    sweepBatch.ordinal.kind = OVSPATIALQUERY_ORDINAL_HEAD;
    sweepBatch.query_count  = 1u;
    ovspatialquery_sphere_sweep_batch_reservation_t sweepReservation{};
    std::memset(&sweepReservation, 0xa5, sizeof(sweepReservation));
    const ovspatialquery_sphere_sweep_batch_reservation_t sweepSentinel = sweepReservation;
    const ovspatialquery_enqueue_result_t                 rejectedSweep =
        ovspatialquery_enqueue_sphere_sweep_plan(fixture.instance, sweepPlan, &sweepBatch, &sweepReservation);
    REQUIRE(rejectedSweep.status == OVSPATIALQUERY_API_INVALID_ARGUMENT &&
            rejectedSweep.op_id == OVSPATIALQUERY_INVALID_OP);
    REQUIRE(std::memcmp(&sweepReservation, &sweepSentinel, sizeof(sweepReservation)) == 0);

    ovspatialquery_overlap_batch_t overlapBatch{};
    overlapBatch.ordinal.kind = OVSPATIALQUERY_ORDINAL_HEAD;
    overlapBatch.query_count  = 1u;
    ovspatialquery_overlap_batch_reservation_t overlapReservation{};
    std::memset(&overlapReservation, 0xa5, sizeof(overlapReservation));
    const ovspatialquery_overlap_batch_reservation_t overlapSentinel = overlapReservation;
    const ovspatialquery_enqueue_result_t            rejectedOverlap =
        ovspatialquery_enqueue_overlap_plan(fixture.instance, overlapPlan, &overlapBatch, &overlapReservation);
    REQUIRE(rejectedOverlap.status == OVSPATIALQUERY_API_INVALID_ARGUMENT &&
            rejectedOverlap.op_id == OVSPATIALQUERY_INVALID_OP);
    REQUIRE(std::memcmp(&overlapReservation, &overlapSentinel, sizeof(overlapReservation)) == 0);

    REQUIRE(ovspatialquery_destroy_batch_plan(fixture.instance, overlapPlan).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_batch_plan(fixture.instance, sweepPlan).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_batch_plan(fixture.instance, rayPlan).status == OVSPATIALQUERY_API_SUCCESS);
    fixture.close();
}

void testPublicCudaHostBudgetAccounting()
{
    gPhase = "calibrate CUDA family fixed host charge";
    ovspatialquery_raycast_plan_desc_t ray{};
    ray.target.kind            = OVSPATIALQUERY_TARGET_POPULATION;
    ray.hit_mode               = OVSPATIALQUERY_HIT_MODE_CLOSEST;
    ray.precision              = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    ray.device                 = {kDLCUDA, 0};
    ray.maximum_query_count    = 1u;
    ray.maximum_hits_per_query = 1u;
    ray.hit_capacity_per_slot  = 1u;
    ray.result_slot_count      = 1u;
    ExactMeshFixture calibration;
    ray.target.handle.population                 = calibration.population;
    const uint64_t                     baseUsage = hostUsage(calibration.instance);
    ovspatialquery_batch_plan_handle_t plan      = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_raycast_plan(calibration.instance, &ray, &plan).status == OVSPATIALQUERY_API_SUCCESS);
    const uint64_t planUsage = hostUsage(calibration.instance);
    REQUIRE(planUsage > baseUsage);
    REQUIRE(ovspatialquery_destroy_batch_plan(calibration.instance, plan).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(hostUsage(calibration.instance) == baseUsage);
    constexpr uint32_t                 paddingSlotCount = 4096u;
    ovspatialquery_raycast_plan_desc_t padding          = ray;
    padding.result_slot_count                           = paddingSlotCount;
    ovspatialquery_batch_plan_handle_t paddingPlan      = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_raycast_plan(calibration.instance, &padding, &paddingPlan).status ==
            OVSPATIALQUERY_API_SUCCESS);
    const uint64_t paddingUsage = hostUsage(calibration.instance);
    REQUIRE(paddingUsage > planUsage);
    REQUIRE(ovspatialquery_destroy_batch_plan(calibration.instance, paddingPlan).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(hostUsage(calibration.instance) == baseUsage);
    calibration.close();

    const uint64_t planCharge    = planUsage - baseUsage;
    const uint64_t paddingCharge = paddingUsage - baseUsage;

    gPhase                       = "reject one-byte-short CUDA family fixed host charge";
    const uint64_t   fixedBudget = baseUsage + paddingCharge + planCharge - 1u;
    ExactMeshFixture fixedBoundary(fixedBudget);
    REQUIRE(hostUsage(fixedBoundary.instance) == baseUsage);
    ray.target.handle.population     = fixedBoundary.population;
    padding.target.handle.population = fixedBoundary.population;
    paddingPlan                      = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    const ovspatialquery_result_t fixedPaddingCreated =
        ovspatialquery_create_raycast_plan(fixedBoundary.instance, &padding, &paddingPlan);
    if (fixedPaddingCreated.status != OVSPATIALQUERY_API_SUCCESS)
    {
        std::fprintf(stderr, "fixed budget=%llu base=%llu padding=%llu plan=%llu status=%u\n",
                     static_cast<unsigned long long>(fixedBudget), static_cast<unsigned long long>(baseUsage),
                     static_cast<unsigned long long>(paddingCharge), static_cast<unsigned long long>(planCharge),
                     fixedPaddingCreated.status);
    }
    REQUIRE(fixedPaddingCreated.status == OVSPATIALQUERY_API_SUCCESS);
    plan = ovspatialquery_batch_plan_handle_t{UINT64_C(0xa5a5a5a5)};
    REQUIRE(ovspatialquery_create_raycast_plan(fixedBoundary.instance, &ray, &plan).status ==
            OVSPATIALQUERY_API_HOST_BUDGET);
    REQUIRE(plan == OVSPATIALQUERY_INVALID_BATCH_PLAN);
    REQUIRE(hostUsage(fixedBoundary.instance) == baseUsage + paddingCharge);
    REQUIRE(ovspatialquery_destroy_batch_plan(fixedBoundary.instance, paddingPlan).status ==
            OVSPATIALQUERY_API_SUCCESS);
    fixedBoundary.close();

    gPhase                         = "reject one-byte CUDA catalogue staging atomically";
    const uint64_t   stagingBudget = baseUsage + paddingCharge + planCharge;
    ExactMeshFixture stagingBoundary(stagingBudget);
    ray.target.handle.population     = stagingBoundary.population;
    padding.target.handle.population = stagingBoundary.population;
    paddingPlan                      = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_raycast_plan(stagingBoundary.instance, &padding, &paddingPlan).status ==
            OVSPATIALQUERY_API_SUCCESS);
    plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_raycast_plan(stagingBoundary.instance, &ray, &plan).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(hostUsage(stagingBoundary.instance) == stagingBudget);
    CudaDriver                      cuda(CudaDriver::Ownership::tracked);
    const CudaDriver::Stream        stream                = cuda.createStream();
    double                          origin[3]             = {};
    float                           direction[3]          = {1.0f, 0.0f, 0.0f};
    double                          maximumDistance       = 10.0;
    const CudaDriver::DevicePointer deviceOrigin          = upload(cuda, origin, stream);
    const CudaDriver::DevicePointer deviceDirection       = upload(cuda, direction, stream);
    const CudaDriver::DevicePointer deviceMaximumDistance = upload(cuda, maximumDistance, stream);
    InputArray                      origins(devicePointer<double>(deviceOrigin), 1, 3, kDLFloat, kCudaDevice);
    InputArray                      directions(devicePointer<float>(deviceDirection), 1, 3, kDLFloat, kCudaDevice);
    InputArray maximumDistances(devicePointer<double>(deviceMaximumDistance), 1, kDLFloat, kCudaDevice);
    ovspatialquery_raycast_batch_t batch{};
    batch.ordinal.kind         = OVSPATIALQUERY_ORDINAL_HEAD;
    batch.query_count          = 1u;
    batch.origins              = &origins.value;
    batch.unit_directions      = &directions.value;
    batch.max_distances        = &maximumDistances.value;
    batch.cuda_sync.has_stream = 1u;
    batch.cuda_sync.stream     = reinterpret_cast<uintptr_t>(stream);
    ovspatialquery_raycast_batch_reservation_t reservation{};
    std::memset(&reservation, 0xa5, sizeof(reservation));
    const ovspatialquery_raycast_batch_reservation_t sentinel = reservation;
    const ovspatialquery_enqueue_result_t            rejected =
        ovspatialquery_enqueue_raycast_plan(stagingBoundary.instance, plan, &batch, &reservation);
    REQUIRE(rejected.status == OVSPATIALQUERY_API_HOST_BUDGET && rejected.op_id == OVSPATIALQUERY_INVALID_OP);
    REQUIRE(std::memcmp(&reservation, &sentinel, sizeof(reservation)) == 0);
    REQUIRE(hostUsage(stagingBoundary.instance) == stagingBudget);
    REQUIRE(ovspatialquery_destroy_batch_plan(stagingBoundary.instance, plan).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_batch_plan(stagingBoundary.instance, paddingPlan).status ==
            OVSPATIALQUERY_API_SUCCESS);
    stagingBoundary.close();
}

// A supplied stream 0 is CUDA's legacy default stream, not "no stream"
// (ADR-0060): a launch on it runs behind the work already queued there. A plan
// refuses a launch without a stream or with a has_stream other than 0 or 1.
void testCudaPlanRunsOnTheLegacyDefaultStream()
{
    ExactMeshFixture fixture;
    gPhase = "create CUDA raycast plan for the legacy default stream";
    ovspatialquery_raycast_plan_desc_t planDesc{};
    planDesc.target                         = populationRef(fixture.population);
    planDesc.hit_mode                       = OVSPATIALQUERY_HIT_MODE_CLOSEST;
    planDesc.precision                      = OVSPATIALQUERY_QUERY_EXACT;
    planDesc.device                         = {kDLCUDA, 0};
    planDesc.maximum_query_count            = 1u;
    planDesc.maximum_hits_per_query         = 1u;
    planDesc.hit_capacity_per_slot          = 1u;
    planDesc.result_slot_count              = 1u;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_raycast_plan(fixture.instance, &planDesc, &plan).status ==
            OVSPATIALQUERY_API_SUCCESS);

    CudaDriver                      cuda(CudaDriver::Ownership::tracked);
    const CudaDriver::Stream        legacy                = nullptr;
    const CudaDriver::Stream        consumer              = cuda.createStream();
    double                          origin[3]             = {0.0, 0.0, 0.0};
    float                           direction[3]          = {1.0f, 0.0f, 0.0f};
    double                          maximumDistance       = 10.0;
    const CudaDriver::DevicePointer deviceOrigin          = upload(cuda, origin, legacy);
    const CudaDriver::DevicePointer deviceDirection       = upload(cuda, direction, legacy);
    const CudaDriver::DevicePointer deviceMaximumDistance = upload(cuda, maximumDistance, legacy);
    InputArray                      origins(devicePointer<double>(deviceOrigin), 1, 3, kDLFloat, kCudaDevice);
    InputArray                      directions(devicePointer<float>(deviceDirection), 1, 3, kDLFloat, kCudaDevice);
    InputArray maximumDistances(devicePointer<double>(deviceMaximumDistance), 1, kDLFloat, kCudaDevice);
    ovspatialquery_raycast_batch_t batch{};
    batch.ordinal.kind     = OVSPATIALQUERY_ORDINAL_HEAD;
    batch.query_count      = 1u;
    batch.origins          = &origins.value;
    batch.unit_directions  = &directions.value;
    batch.max_distances    = &maximumDistances.value;
    batch.cuda_sync.stream = 0u;
    ovspatialquery_raycast_batch_reservation_t reservation{};
    for (const uint32_t hasStream : {0u, 2u})
    {
        gPhase                     = "refuse a CUDA launch without a supplied stream";
        batch.cuda_sync.has_stream = hasStream;
        std::memset(&reservation, 0xa5, sizeof(reservation));
        const ovspatialquery_raycast_batch_reservation_t sentinel = reservation;
        const ovspatialquery_enqueue_result_t            refused =
            ovspatialquery_enqueue_raycast_plan(fixture.instance, plan, &batch, &reservation);
        REQUIRE(refused.status == OVSPATIALQUERY_API_INVALID_ARGUMENT && refused.op_id == OVSPATIALQUERY_INVALID_OP);
        REQUIRE(std::memcmp(&reservation, &sentinel, sizeof(reservation)) == 0);
    }

    gPhase                       = "launch behind held work on the legacy default stream";
    const CudaDriver::Event held = cuda.createEvent();
    cuda.enqueueDelay(legacy, UINT64_C(400000000));
    cuda.record(held, legacy);
    batch.cuda_sync.has_stream = 1u;
    ovspatialquery_enqueue_result_t operation =
        ovspatialquery_enqueue_raycast_plan(fixture.instance, plan, &batch, &reservation);
    REQUIRE(operation.status == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_op_wait_result_t waited{};
    REQUIRE(
        ovspatialquery_wait_op(fixture.instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
        OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(waited.state == OVSPATIALQUERY_OP_COMPLETED);
    // Run on any other stream, the launch would finish while the delay still holds the legacy stream.
    REQUIRE(cuda.isComplete(held));

    gPhase = "read the hit of the legacy-stream launch";
    ovspatialquery_raycast_batch_result_t result{};
    REQUIRE(ovspatialquery_fetch_raycast_batch(fixture.instance, operation.op_id, &result).status ==
            OVSPATIALQUERY_API_SUCCESS);
    cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(result.common.result_ready.value));
    uint64_t count    = UINT64_MAX;
    uint64_t objectId = 0u;
    cuda.downloadAsync(&count, deviceAddress(result.columns.hit_ranges.hit_counts), sizeof(count), consumer);
    cuda.downloadAsync(&objectId, deviceAddress(result.columns.object_ids), sizeof(objectId), consumer);
    retireAfterConsumer(cuda, consumer, fixture.instance, operation.op_id);
    REQUIRE(count == 1u && objectId == fixture.objectValue);
    REQUIRE(destroyPlanWhenRetired(fixture.instance, plan));
    fixture.close();
    REQUIRE(transferredEventDestroysWereInvoked());
}

void testCudaPlanRefusesACapturingStream()
{
    ExactMeshFixture fixture;
    gPhase = "create CUDA raycast plan for a capturing stream";
    ovspatialquery_raycast_plan_desc_t planDesc{};
    planDesc.target                         = populationRef(fixture.population);
    planDesc.hit_mode                       = OVSPATIALQUERY_HIT_MODE_CLOSEST;
    planDesc.precision                      = OVSPATIALQUERY_QUERY_EXACT;
    planDesc.device                         = {kDLCUDA, 0};
    planDesc.maximum_query_count            = 1u;
    planDesc.maximum_hits_per_query         = 1u;
    planDesc.hit_capacity_per_slot          = 1u;
    planDesc.result_slot_count              = 1u;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_raycast_plan(fixture.instance, &planDesc, &plan).status ==
            OVSPATIALQUERY_API_SUCCESS);

    CudaDriver                      cuda(CudaDriver::Ownership::tracked);
    const CudaDriver::Stream        stream                = cuda.createStream();
    double                          origin[3]             = {0.0, 0.0, 0.0};
    float                           direction[3]          = {1.0f, 0.0f, 0.0f};
    double                          maximumDistance       = 10.0;
    const CudaDriver::DevicePointer deviceOrigin          = upload(cuda, origin, stream);
    const CudaDriver::DevicePointer deviceDirection       = upload(cuda, direction, stream);
    const CudaDriver::DevicePointer deviceMaximumDistance = upload(cuda, maximumDistance, stream);
    InputArray                      origins(devicePointer<double>(deviceOrigin), 1, 3, kDLFloat, kCudaDevice);
    InputArray                      directions(devicePointer<float>(deviceDirection), 1, 3, kDLFloat, kCudaDevice);
    InputArray maximumDistances(devicePointer<double>(deviceMaximumDistance), 1, kDLFloat, kCudaDevice);
    ovspatialquery_raycast_batch_t batch{};
    batch.ordinal.kind         = OVSPATIALQUERY_ORDINAL_HEAD;
    batch.query_count          = 1u;
    batch.origins              = &origins.value;
    batch.unit_directions      = &directions.value;
    batch.max_distances        = &maximumDistances.value;
    batch.cuda_sync.has_stream = 1u;
    batch.cuda_sync.stream     = reinterpret_cast<uintptr_t>(stream);

    gPhase = "refuse a launch on a stream that is capturing a CUDA graph";
    cuda.beginCapture(stream);
    ovspatialquery_raycast_batch_reservation_t reservation{};
    std::memset(&reservation, 0xa5, sizeof(reservation));
    const ovspatialquery_raycast_batch_reservation_t sentinel = reservation;
    const ovspatialquery_enqueue_result_t            refused =
        ovspatialquery_enqueue_raycast_plan(fixture.instance, plan, &batch, &reservation);
    const size_t capturedNodes = cuda.endCapture(stream);
    REQUIRE(refused.status == OVSPATIALQUERY_API_NOT_SUPPORTED && refused.op_id == OVSPATIALQUERY_INVALID_OP);
    REQUIRE(std::memcmp(&reservation, &sentinel, sizeof(reservation)) == 0);
    REQUIRE(capturedNodes == 0u);

    gPhase = "launch on the same stream once its capture has ended";
    ovspatialquery_enqueue_result_t operation =
        ovspatialquery_enqueue_raycast_plan(fixture.instance, plan, &batch, &reservation);
    REQUIRE(operation.status == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_op_wait_result_t waited{};
    REQUIRE(
        ovspatialquery_wait_op(fixture.instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
        OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(waited.state == OVSPATIALQUERY_OP_COMPLETED);
    retireAfterConsumer(cuda, stream, fixture.instance, operation.op_id);
    REQUIRE(destroyPlanWhenRetired(fixture.instance, plan));
    fixture.close();
    REQUIRE(transferredEventDestroysWereInvoked());
}

void testPublicCudaValidationAndBudget()
{
    ExactMeshFixture fixture;
    gPhase = "reject invalid CUDA family plan descriptors";
    ovspatialquery_raycast_plan_desc_t ray{};
    ray.target                                      = populationRef(fixture.population);
    ray.hit_mode                                    = OVSPATIALQUERY_HIT_MODE_CLOSEST;
    ray.precision                                   = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    ray.device                                      = {kDLCUDA, 0};
    ray.maximum_query_count                         = 1u;
    ray.maximum_hits_per_query                      = 1u;
    ray.hit_capacity_per_slot                       = 1u;
    ray.result_slot_count                           = 1u;
    ray.result_fields                               = OVSPATIALQUERY_BATCH_RESULT_SOURCE_FEATURE;
    ovspatialquery_batch_plan_handle_t rejectedPlan = ovspatialquery_batch_plan_handle_t{UINT64_C(0xa5a5a5a5)};
    REQUIRE(ovspatialquery_create_raycast_plan(fixture.instance, &ray, &rejectedPlan).status ==
            OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(rejectedPlan == OVSPATIALQUERY_INVALID_BATCH_PLAN);

    // CUDA availability is probed on the requested device, so a device the
    // driver does not have is named in the diagnostic.
    gPhase                                          = "reject a CUDA plan on a device the driver does not have";
    ovspatialquery_raycast_plan_desc_t absentDevice = ray;
    absentDevice.precision                          = OVSPATIALQUERY_QUERY_EXACT;
    absentDevice.device                             = {kDLCUDA, 1 << 20};
    REQUIRE(ovspatialquery_create_raycast_plan(fixture.instance, &absentDevice, &rejectedPlan).status ==
            OVSPATIALQUERY_API_NOT_SUPPORTED);
    REQUIRE(rejectedPlan == OVSPATIALQUERY_INVALID_BATCH_PLAN);
    const ovspatialquery_string_t absentError = ovspatialquery_get_last_error();
    REQUIRE(std::string(absentError.ptr, absentError.length).find("no CUDA device with the requested ID") !=
            std::string::npos);

    // A later refusal replaces that diagnostic instead of leaving it behind.
    gPhase = "replace the diagnostic on a refused explicit set";
    REQUIRE(ovspatialquery_enqueue_explicit_population_set(fixture.instance, OVSPATIALQUERY_INVALID_POPULATION, nullptr)
                .status == OVSPATIALQUERY_API_INVALID_ARGUMENT);
    const ovspatialquery_string_t setError = ovspatialquery_get_last_error();
    REQUIRE(setError.length != 0u);
    REQUIRE(std::string(setError.ptr, setError.length).find("no CUDA device") == std::string::npos);

    ovspatialquery_overlap_plan_desc_t overlap{};
    overlap.target                   = ray.target;
    overlap.hit_mode                 = OVSPATIALQUERY_HIT_MODE_ANY;
    overlap.precision                = OVSPATIALQUERY_QUERY_EXACT_SINGLE;
    overlap.shape.kind               = OVSPATIALQUERY_OVERLAP_SPHERE;
    overlap.shape.data.sphere.radius = 1.0;
    overlap.device                   = {kDLCUDA, 0};
    overlap.maximum_query_count      = 1u;
    overlap.maximum_hits_per_query   = 1u;
    overlap.hit_capacity_per_slot    = 1u;
    overlap.result_slot_count        = 1u;
    rejectedPlan                     = ovspatialquery_batch_plan_handle_t{UINT64_C(0xa5a5a5a5)};
    REQUIRE(ovspatialquery_create_overlap_plan(fixture.instance, &overlap, &rejectedPlan).status ==
            OVSPATIALQUERY_API_NOT_SUPPORTED);
    REQUIRE(rejectedPlan == OVSPATIALQUERY_INVALID_BATCH_PLAN);

    overlap.precision                      = OVSPATIALQUERY_QUERY_EXACT;
    overlap.shape.kind                     = OVSPATIALQUERY_OVERLAP_BOX;
    overlap.shape.data.box.half_extents[0] = 1.0;
    overlap.shape.data.box.half_extents[1] = -1.0;
    overlap.shape.data.box.half_extents[2] = 1.0;
    rejectedPlan                           = ovspatialquery_batch_plan_handle_t{UINT64_C(0xa5a5a5a5)};
    REQUIRE(ovspatialquery_create_overlap_plan(fixture.instance, &overlap, &rejectedPlan).status ==
            OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(rejectedPlan == OVSPATIALQUERY_INVALID_BATCH_PLAN);

    CudaDriver                      cuda(CudaDriver::Ownership::tracked);
    const CudaDriver::Stream        stream                = cuda.createStream();
    double                          center[3]             = {};
    float                           direction[3]          = {1.0f, 0.0f, 0.0f};
    double                          maximumDistance       = 10.0;
    const CudaDriver::DevicePointer deviceCenter          = upload(cuda, center, stream);
    const CudaDriver::DevicePointer deviceDirection       = upload(cuda, direction, stream);
    const CudaDriver::DevicePointer deviceMaximumDistance = upload(cuda, maximumDistance, stream);
    InputArray                      centers(devicePointer<double>(deviceCenter), 1, 3, kDLFloat, kCudaDevice);
    InputArray                      directions(devicePointer<float>(deviceDirection), 1, 3, kDLFloat, kCudaDevice);
    InputArray maximumDistances(devicePointer<double>(deviceMaximumDistance), 1, kDLFloat, kCudaDevice);
    ovspatialquery_sphere_sweep_plan_desc_t sweep{};
    sweep.target                                 = ray.target;
    sweep.precision                              = OVSPATIALQUERY_QUERY_EXACT;
    sweep.radius                                 = 0.5;
    sweep.device                                 = {kDLCUDA, 0};
    sweep.maximum_query_count                    = 1u;
    sweep.result_slot_count                      = 1u;
    sweep.hit_capacity_per_slot                  = 1u;
    ovspatialquery_batch_plan_handle_t sweepPlan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    sweep.precision                              = OVSPATIALQUERY_QUERY_EXACT_SINGLE;
    REQUIRE(ovspatialquery_create_sphere_sweep_plan(fixture.instance, &sweep, &sweepPlan).status ==
            OVSPATIALQUERY_API_NOT_SUPPORTED);
    REQUIRE(sweepPlan == OVSPATIALQUERY_INVALID_BATCH_PLAN);
    sweep.precision = OVSPATIALQUERY_QUERY_EXACT;
    REQUIRE(ovspatialquery_create_sphere_sweep_plan(fixture.instance, &sweep, &sweepPlan).status ==
            OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_sphere_sweep_batch_t oversized{};
    oversized.ordinal.kind         = OVSPATIALQUERY_ORDINAL_HEAD;
    oversized.query_count          = 2u;
    oversized.centers              = &centers.value;
    oversized.unit_directions      = &directions.value;
    oversized.max_distances        = &maximumDistances.value;
    oversized.cuda_sync.has_stream = 1u;
    oversized.cuda_sync.stream     = reinterpret_cast<uintptr_t>(stream);
    ovspatialquery_sphere_sweep_batch_reservation_t untouched{};
    std::memset(&untouched, 0xa5, sizeof(untouched));
    const ovspatialquery_sphere_sweep_batch_reservation_t sentinel = untouched;
    const ovspatialquery_enqueue_result_t                 rejected =
        ovspatialquery_enqueue_sphere_sweep_plan(fixture.instance, sweepPlan, &oversized, &untouched);
    REQUIRE(rejected.status == OVSPATIALQUERY_API_INVALID_ARGUMENT && rejected.op_id == OVSPATIALQUERY_INVALID_OP);
    REQUIRE(std::memcmp(&untouched, &sentinel, sizeof(untouched)) == 0);
    REQUIRE(ovspatialquery_destroy_batch_plan(fixture.instance, sweepPlan).status == OVSPATIALQUERY_API_SUCCESS);
    fixture.close();

    gPhase = "reject CUDA family plan at device budget boundary";
    const ovspatialquery_device_budget_t deviceBudget{{kDLCUDA, 0}, 1u};
    const ovspatialquery_instance_desc_t instanceDesc{1u, 1u, UINT64_C(64) << 20, &deviceBudget, 1u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&instanceDesc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    const ovspatialquery_population_desc_t populationDesc{1u, 1u};
    ovspatialquery_population_handle_t     population = OVSPATIALQUERY_INVALID_POPULATION;
    REQUIRE(ovspatialquery_create_explicit_population(instance, &populationDesc, &population).status ==
            OVSPATIALQUERY_API_SUCCESS);
    ray.target.handle.population = population;
    ray.result_fields            = 0u;
    rejectedPlan                 = ovspatialquery_batch_plan_handle_t{UINT64_C(0xa5a5a5a5)};
    REQUIRE(ovspatialquery_create_raycast_plan(instance, &ray, &rejectedPlan).status ==
            OVSPATIALQUERY_API_DEVICE_BUDGET);
    REQUIRE(rejectedPlan == OVSPATIALQUERY_INVALID_BATCH_PLAN);
    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

void testPublicCudaDeviceLossDiagnostic(uint32_t resultMetadataFailure = 0u)
{
    const bool                         duringResultMetadata = resultMetadataFailure != 0u;
    ExactMeshFixture                   fixture;
    CudaDriver                         cuda(CudaDriver::Ownership::tracked);
    const CudaDriver::Stream           stream = cuda.createStream();
    const CudaDriver::Event            ready  = cuda.createEvent();
    ovspatialquery_raycast_plan_desc_t planDesc{};
    planDesc.target                         = populationRef(fixture.population);
    planDesc.hit_mode                       = OVSPATIALQUERY_HIT_MODE_CLOSEST;
    planDesc.precision                      = OVSPATIALQUERY_QUERY_EXACT;
    planDesc.device                         = {kDLCUDA, 0};
    planDesc.maximum_query_count            = 1u;
    planDesc.maximum_hits_per_query         = 1u;
    planDesc.hit_capacity_per_slot          = 1u;
    planDesc.result_slot_count              = 1u;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_raycast_plan(fixture.instance, &planDesc, &plan).status ==
            OVSPATIALQUERY_API_SUCCESS);
    double                          origin[3]             = {};
    float                           direction[3]          = {1.0f, 0.0f, 0.0f};
    double                          maximumDistance       = 10.0;
    const CudaDriver::DevicePointer deviceOrigin          = upload(cuda, origin, stream);
    const CudaDriver::DevicePointer deviceDirection       = upload(cuda, direction, stream);
    const CudaDriver::DevicePointer deviceMaximumDistance = upload(cuda, maximumDistance, stream);
    cuda.record(ready, stream);
    InputArray origins(devicePointer<double>(deviceOrigin), 1, 3, kDLFloat, kCudaDevice);
    InputArray directions(devicePointer<float>(deviceDirection), 1, 3, kDLFloat, kCudaDevice);
    InputArray maximumDistances(devicePointer<double>(deviceMaximumDistance), 1, kDLFloat, kCudaDevice);
    ovspatialquery_raycast_batch_t batch{};
    batch.ordinal.kind         = OVSPATIALQUERY_ORDINAL_HEAD;
    batch.query_count          = 1u;
    batch.origins              = &origins.value;
    batch.unit_directions      = &directions.value;
    batch.max_distances        = &maximumDistances.value;
    batch.cuda_sync.has_stream = 1u;
    batch.cuda_sync.stream     = reinterpret_cast<uintptr_t>(stream);
    batch.cuda_sync.wait_event = reinterpret_cast<uintptr_t>(ready);
    ovspatialquery_raycast_batch_reservation_t reservation{};
    gPhase = "admit public CUDA device-loss operation";
    if (!duringResultMetadata)
    {
        ovspatialquery_test_set_cuda_failure_point(kCudaFailureFamilyLaunchDeviceLost);
    }
    ovspatialquery_enqueue_result_t operation =
        ovspatialquery_enqueue_raycast_plan(fixture.instance, plan, &batch, &reservation);
    REQUIRE(operation.status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE((reservation.common.result_ready.value != 0u) == duringResultMetadata);
    ovspatialquery_op_wait_result_t waited{};
    REQUIRE(
        ovspatialquery_wait_op(fixture.instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
        OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(waited.state == (duringResultMetadata ? OVSPATIALQUERY_OP_COMPLETED : OVSPATIALQUERY_OP_FAILURE));
    ovspatialquery_raycast_batch_result_t unavailable{};
    std::memset(&unavailable, 0xa5, sizeof(unavailable));
    const ovspatialquery_raycast_batch_result_t sentinel = unavailable;
    if (duringResultMetadata)
    {
        gPhase = "materialize public CUDA result metadata with device loss";
        ovspatialquery_test_set_cuda_failure_point(resultMetadataFailure);
        const ovspatialquery_api_status_t expected = resultMetadataFailure == kCudaFailureResultMetadataDeviceLost
                                                         ? OVSPATIALQUERY_API_DEVICE_LOST
                                                         : OVSPATIALQUERY_API_INTERNAL;
        REQUIRE(ovspatialquery_fetch_raycast_batch(fixture.instance, operation.op_id, &unavailable).status == expected);
        REQUIRE(std::memcmp(&unavailable, &sentinel, sizeof(unavailable)) == 0);
        REQUIRE(ovspatialquery_fetch_raycast_batch(fixture.instance, operation.op_id, &unavailable).status == expected);
        REQUIRE(std::memcmp(&unavailable, &sentinel, sizeof(unavailable)) == 0);
    }
    else
    {
        ovspatialquery_op_status_t status{};
        REQUIRE(ovspatialquery_query_op_status(fixture.instance, operation.op_id, &status).status ==
                OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(status.state == OVSPATIALQUERY_OP_FAILURE);
        REQUIRE(status.terminal_status == OVSPATIALQUERY_API_DEVICE_LOST);
        REQUIRE(status.progress == 1.0);
        const ovspatialquery_string_t diagnostic = ovspatialquery_get_last_op_error(fixture.instance, operation.op_id);
        REQUIRE(diagnostic.ptr != nullptr && diagnostic.length != 0u);
        REQUIRE(ovspatialquery_fetch_raycast_batch(fixture.instance, operation.op_id, &unavailable).status ==
                OVSPATIALQUERY_API_OPERATION_FAILED);
        REQUIRE(std::memcmp(&unavailable, &sentinel, sizeof(unavailable)) == 0);
        const CudaDriver::Event rejectedConsumerDone = cuda.createEvent();
        cuda.record(rejectedConsumerDone, stream);
        cuda.synchronize(rejectedConsumerDone);
        const ovspatialquery_op_id_t rejectedOperation = operation.op_id;
        ovspatialquery_cuda_event_t  rejectedDone{reinterpret_cast<uintptr_t>(rejectedConsumerDone)};
        const uintptr_t              rejectedDoneValue               = rejectedDone.value;
        const uint64_t               destroyAttemptsBeforeDeviceLoss = transferredEventDestroyAttempts();
        REQUIRE(ovspatialquery_destroy_op_after_cuda(fixture.instance, &operation.op_id, &rejectedDone).status ==
                OVSPATIALQUERY_API_INVALID_STATE);
        REQUIRE(operation.op_id == rejectedOperation && rejectedDone.value == rejectedDoneValue &&
                transferredEventDestroyAttempts() == destroyAttemptsBeforeDeviceLoss);
        cuda.destroyEvent(rejectedConsumerDone);
    }
    const bool afterConsumer = duringResultMetadata && resultMetadataFailure != kCudaFailureResultMetadataDeviceLost;
    if (afterConsumer)
    {
        REQUIRE(ovspatialquery_destroy_op(fixture.instance, &operation.op_id).status == OVSPATIALQUERY_API_BUSY);
        cuda.wait(stream, reinterpret_cast<CudaDriver::Event>(reservation.common.result_ready.value));
        retireAfterConsumer(cuda, stream, fixture.instance, operation.op_id);
    }
    else
    {
        REQUIRE(ovspatialquery_destroy_op(fixture.instance, &operation.op_id).status == OVSPATIALQUERY_API_SUCCESS);
    }
    REQUIRE(operation.op_id == OVSPATIALQUERY_INVALID_OP);
    REQUIRE(destroyPlanWhenRetired(fixture.instance, plan));
    fixture.close();
    if (afterConsumer)
    {
        REQUIRE(transferredEventDestroysWereInvoked());
    }
}

// Device loss while a device-prepared publication launches (DEVICE_LOST at
// admission) or completes (on the worker) fails that publication, and every
// later publication on the population is refused with DEVICE_LOST.
void testDevicePreparedPublicationDeviceLoss(uint32_t failurePoint)
{
    const bool atLaunch = failurePoint == kCudaFailureDevicePublicationLaunch;
    gPhase              = "create device-prepared publication instance";
    const ovspatialquery_device_budget_t budget{{kDLCUDA, 0}, UINT64_C(64) << 20};
    const ovspatialquery_instance_desc_t desc{2u, 1u, UINT64_C(64) << 20, &budget, 1u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    uint64_t                                       sphereValue  = 7001u;
    double                                         sphereRadius = 0.5;
    InputArray                                     definitionValues(&sphereValue, 1, kDLUInt);
    InputArray                                     radii(&sphereRadius, 1, kDLFloat);
    const ovspatialquery_sphere_definition_table_t spheres{&definitionValues.value, &radii.value};
    waitAndDestroy(instance, ovspatialquery_enqueue_sphere_registration(instance, &spheres));

    gPhase = "publish device-prepared set";
    CudaDriver                      cuda(CudaDriver::Ownership::tracked);
    const CudaDriver::Stream        producer       = cuda.createStream();
    const CudaDriver::Event         rowsReady      = cuda.createEvent();
    uint64_t                        objectValue    = 42u;
    int32_t                         geometryType   = OVSPATIALQUERY_GEOMETRY_SPHERE;
    double                          translation[3] = {1.5, 0.0, 0.0};
    float                           rotation[4]    = {0.0f, 0.0f, 0.0f, 1.0f};
    const CudaDriver::DevicePointer deviceIds      = upload(cuda, objectValue, producer);
    const CudaDriver::DevicePointer deviceTypes    = upload(cuda, geometryType, producer);
    const CudaDriver::DevicePointer deviceValues   = upload(cuda, sphereValue, producer);
    const CudaDriver::DevicePointer devicePosition = upload(cuda, translation, producer);
    const CudaDriver::DevicePointer deviceRotation = upload(cuda, rotation, producer);
    cuda.record(rowsReady, producer);
    InputArray ids(devicePointer<uint64_t>(deviceIds), 1, kDLUInt, kCudaDevice);
    InputArray types(devicePointer<int32_t>(deviceTypes), 1, kDLInt, kCudaDevice);
    InputArray values(devicePointer<uint64_t>(deviceValues), 1, kDLUInt, kCudaDevice);
    InputArray positions(devicePointer<double>(devicePosition), 1, 3, kDLFloat, kCudaDevice);
    InputArray rotations(devicePointer<float>(deviceRotation), 1, 4, kDLFloat, kCudaDevice);
    InputArray noRemovals(static_cast<uint64_t*>(nullptr), 0, kDLUInt, kCudaDevice);
    const ovspatialquery_dynamic_execution_t onDevice{
        kCudaDevice,
        {1u, reinterpret_cast<uintptr_t>(producer), reinterpret_cast<uintptr_t>(rowsReady)},
        OVSPATIALQUERY_PREPARE_ON_DEVICE};
    const ovspatialquery_population_desc_t populationDesc{0u, 1u};
    ovspatialquery_population_handle_t     population = OVSPATIALQUERY_INVALID_POPULATION;
    REQUIRE(ovspatialquery_create_explicit_population(instance, &populationDesc, &population).status ==
            OVSPATIALQUERY_API_SUCCESS);
    const ovspatialquery_explicit_population_set_t set{
        7,
        {&ids.value, &types.value, &values.value, &positions.value, &rotations.value, nullptr, nullptr, nullptr},
        &onDevice,
        0u};
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));

    gPhase = atLaunch ? "device loss while launching a device-prepared publication"
                      : "device loss while completing a device-prepared publication";
    ovspatialquery_explicit_population_update_t poses{
        7,
        8,
        OVSPATIALQUERY_EXPLICIT_UPDATE_POSES,
        {&ids.value, nullptr, nullptr, &positions.value, &rotations.value, nullptr, nullptr, nullptr},
        &noRemovals.value,
        &onDevice};
    ovspatialquery_test_set_cuda_failure_point(failurePoint);
    ovspatialquery_enqueue_result_t lost =
        ovspatialquery_enqueue_explicit_population_update(instance, population, &poses);
    if (atLaunch)
    {
        REQUIRE(lost.status == OVSPATIALQUERY_API_DEVICE_LOST && lost.op_id == OVSPATIALQUERY_INVALID_OP);
    }
    else
    {
        REQUIRE(lost.status == OVSPATIALQUERY_API_SUCCESS);
        ovspatialquery_op_wait_result_t waited{};
        REQUIRE(ovspatialquery_wait_op(instance, lost.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
                OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(waited.state == OVSPATIALQUERY_OP_FAILURE);
        ovspatialquery_op_status_t status{};
        REQUIRE(ovspatialquery_query_op_status(instance, lost.op_id, &status).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(status.terminal_status == OVSPATIALQUERY_API_DEVICE_LOST);
        REQUIRE(ovspatialquery_destroy_op(instance, &lost.op_id).status == OVSPATIALQUERY_API_SUCCESS);
    }

    gPhase               = "publish on a population whose device was lost";
    poses.target_ordinal = 9;
    const ovspatialquery_enqueue_result_t later =
        ovspatialquery_enqueue_explicit_population_update(instance, population, &poses);
    if (later.status != OVSPATIALQUERY_API_DEVICE_LOST || later.op_id != OVSPATIALQUERY_INVALID_OP)
    {
        std::fprintf(stderr, "publication failure point %u: later status=%d, op=%llu\n", failurePoint,
                     static_cast<int>(later.status), static_cast<unsigned long long>(later.op_id.raw_id));
    }
    REQUIRE(later.status == OVSPATIALQUERY_API_DEVICE_LOST && later.op_id == OVSPATIALQUERY_INVALID_OP);

    gPhase = "destroy the device-loss population";
    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    gPhase = "destroy the device-loss geometry definition";
    const ovspatialquery_api_status_t destroyed =
        ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_SPHERE, sphereValue}).status;
    if (destroyed != OVSPATIALQUERY_API_SUCCESS)
    {
        std::fprintf(stderr, "publication failure point %u: geometry destruction status=%d\n", failurePoint,
                     static_cast<int>(destroyed));
    }
    REQUIRE(destroyed == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

// A MULTIPLE raycast plan reports every triangle a mesh crosses by default and
// one row per object with MESH_HITS_NEAREST; a non-mesh object always gives one
// row, and a truncated row keeps its nearest crossings. Two meshes of two
// double-sided plates (local x 0 and 1) sit at x 2 and 6 and a sphere of radius
// 0.5 at x 10, so a +x ray crosses x 2, 3, 6, 7 and enters the sphere. The
// CUDA rows must match the synchronous CPU query as sets.
void testCudaRaycastReportsEveryMeshCrossing(ovspatialquery_query_precision_t precision)
{
    gPhase = "publish mesh crossing scene";
    const ovspatialquery_device_budget_t budget{{kDLCUDA, 0}, UINT64_C(64) << 20};
    const ovspatialquery_instance_desc_t desc{4u, 1u, UINT64_C(64) << 20, &budget, 1u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    uint64_t meshValue = 71u, zero = 0u, vertexCount = 8u, triangleCount = 4u;
    float    vertices[24]{};
    for (uint32_t plate = 0u; plate < 2u; ++plate)
    {
        const float corners[4][2] = {{-1.0f, -1.0f}, {1.0f, -1.0f}, {1.0f, 1.0f}, {-1.0f, 1.0f}};
        for (uint32_t corner = 0u; corner < 4u; ++corner)
        {
            float* vertex = vertices + (plate * 4u + corner) * 3u;
            vertex[0]     = float(plate);
            vertex[1]     = corners[corner][0];
            vertex[2]     = corners[corner][1];
        }
    }
    uint32_t   triangles[12] = {0u, 1u, 2u, 0u, 2u, 3u, 4u, 5u, 6u, 4u, 6u, 7u};
    uint8_t    doubleSided   = 1u;
    InputArray values(&meshValue, 1, kDLUInt), vertexOffsets(&zero, 1, kDLUInt), vertexCounts(&vertexCount, 1, kDLUInt),
        triangleOffsets(&zero, 1, kDLUInt), triangleCounts(&triangleCount, 1, kDLUInt),
        vertexData(vertices, 8, 3, kDLFloat), triangleData(triangles, 4, 3, kDLUInt),
        sidedness(&doubleSided, 1, kDLUInt);
    const ovspatialquery_triangle_mesh_definition_table_t meshes{&values.value,
                                                                 &vertexOffsets.value,
                                                                 &vertexCounts.value,
                                                                 &triangleOffsets.value,
                                                                 &triangleCounts.value,
                                                                 &vertexData.value,
                                                                 &triangleData.value,
                                                                 &sidedness.value,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr};
    waitAndDestroy(instance, ovspatialquery_enqueue_triangle_mesh_registration(instance, &meshes));
    constexpr uint64_t kSphere = 72u;
    registerSphere(instance, kSphere, 0.5);
    const ovspatialquery_population_handle_t population = createExplicitPopulation(instance, 1u);
    InstanceRows                             rows;
    rows.add(1u, OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, meshValue, 2.0, 0.0, 0.0);
    rows.add(2u, OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, meshValue, 6.0, 0.0, 0.0);
    rows.add(3u, OVSPATIALQUERY_GEOMETRY_SPHERE, kSphere, 10.0, 0.0, 0.0);
    const ovspatialquery_explicit_population_set_t set{1, rows.table(), nullptr, 0u};
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));

    // Row 0 sees everything, row 1 stops at x 5, and row 2 keeps three rows. The
    // rays run at y 0, z 0.125: off the plates' shared diagonal, and into the
    // sphere short of its x 9.5 pole.
    const double             kSphereEntry = 10.0 - std::sqrt(0.25 - 0.125 * 0.125);
    constexpr uint64_t       kRows        = 3u;
    std::array<double, 9>    origins{0.0, 0.0, 0.125, 0.0, 0.0, 0.125, 0.0, 0.0, 0.125};
    std::array<float, 9>     directions{1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    std::array<double, 3>    maximumDistances{20.0, 5.0, 20.0};
    std::array<uint64_t, 3>  maximumHits{8u, 8u, 3u};
    CudaDriver               cuda(CudaDriver::Ownership::tracked);
    const CudaDriver::Stream stream           = cuda.createStream();
    const auto               deviceOrigins    = upload(cuda, origins, stream);
    const auto               deviceDirections = upload(cuda, directions, stream);
    const auto               deviceMaximum    = upload(cuda, maximumDistances, stream);
    const auto               deviceHits       = upload(cuda, maximumHits, stream);
    InputArray               originColumn(devicePointer<double>(deviceOrigins), 3, 3, kDLFloat, kCudaDevice);
    InputArray               directionColumn(devicePointer<float>(deviceDirections), 3, 3, kDLFloat, kCudaDevice);
    InputArray               maximumColumn(devicePointer<double>(deviceMaximum), 3, kDLFloat, kCudaDevice);
    InputArray               hitColumn(devicePointer<uint64_t>(deviceHits), 3, kDLUInt, kCudaDevice);

    using Row = std::pair<uint64_t, double>; // object, distance
    static constexpr std::array kMeshHitModes{OVSPATIALQUERY_MESH_HITS_ALL, OVSPATIALQUERY_MESH_HITS_NEAREST};
    for (const ovspatialquery_mesh_hits_t meshHits : kMeshHitModes)
    {
        const bool all = meshHits == OVSPATIALQUERY_MESH_HITS_ALL;
        gPhase         = all ? "CUDA raycast every mesh crossing" : "CUDA raycast nearest per object";
        ovspatialquery_raycast_plan_desc_t planDesc{};
        planDesc.target                         = populationRef(population);
        planDesc.hit_mode                       = OVSPATIALQUERY_HIT_MODE_MULTIPLE;
        planDesc.mesh_hits                      = meshHits;
        planDesc.precision                      = precision;
        planDesc.device                         = {kDLCUDA, 0};
        planDesc.maximum_query_count            = kRows;
        planDesc.maximum_hits_per_query         = 8u;
        planDesc.hit_capacity_per_slot          = 24u;
        planDesc.result_slot_count              = 1u;
        planDesc.result_fields                  = OVSPATIALQUERY_BATCH_RESULT_DISTANCE;
        ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
        REQUIRE(ovspatialquery_create_raycast_plan(instance, &planDesc, &plan).status == OVSPATIALQUERY_API_SUCCESS);
        ovspatialquery_raycast_batch_t batch{};
        batch.ordinal.kind         = OVSPATIALQUERY_ORDINAL_HEAD;
        batch.query_count          = kRows;
        batch.origins              = &originColumn.value;
        batch.unit_directions      = &directionColumn.value;
        batch.max_distances        = &maximumColumn.value;
        batch.maximum_hits         = &hitColumn.value;
        batch.cuda_sync.has_stream = 1u;
        batch.cuda_sync.stream     = reinterpret_cast<uintptr_t>(stream);
        ovspatialquery_raycast_batch_reservation_t reservation{};
        ovspatialquery_enqueue_result_t            operation =
            ovspatialquery_enqueue_raycast_plan(instance, plan, &batch, &reservation);
        REQUIRE(operation.status == OVSPATIALQUERY_API_SUCCESS);
        ovspatialquery_op_wait_result_t waited{};
        REQUIRE(ovspatialquery_wait_op(instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
                    OVSPATIALQUERY_API_SUCCESS &&
                waited.state == OVSPATIALQUERY_OP_COMPLETED);
        ovspatialquery_raycast_batch_result_t result{};
        REQUIRE(ovspatialquery_fetch_raycast_batch(instance, operation.op_id, &result).status ==
                OVSPATIALQUERY_API_SUCCESS);
        cuda.wait(stream, reinterpret_cast<CudaDriver::Event>(result.common.result_ready.value));
        std::array<uint64_t, kRows> offsets{}, counts{};
        std::array<uint8_t, kRows>  truncated{};
        std::array<uint64_t, 24>    ids{};
        std::array<double, 24>      distances{};
        const auto&                 ranges = result.columns.hit_ranges;
        cuda.downloadAsync(offsets.data(), deviceAddress(ranges.hit_offsets), sizeof(offsets), stream);
        cuda.downloadAsync(counts.data(), deviceAddress(ranges.hit_counts), sizeof(counts), stream);
        cuda.downloadAsync(truncated.data(), deviceAddress(ranges.truncated), sizeof(truncated), stream);
        cuda.downloadAsync(ids.data(), deviceAddress(result.columns.object_ids),
                           size_t(result.columns.object_ids.shape[0]) * sizeof(uint64_t), stream);
        cuda.downloadAsync(distances.data(), deviceAddress(result.columns.contact.exact.distances),
                           size_t(result.columns.contact.exact.distances.shape[0]) * sizeof(double), stream);
        retireAfterConsumer(cuda, stream, instance, operation.op_id);
        REQUIRE(transferredEventDestroysWereInvoked());

        const std::array<std::vector<Row>, kRows> expected =
            all ? std::array<std::vector<Row>, kRows>{std::vector<Row>{{1u, 2.0},
                                                                       {1u, 3.0},
                                                                       {2u, 6.0},
                                                                       {2u, 7.0},
                                                                       {3u, kSphereEntry}},
                                                      std::vector<Row>{{1u, 2.0}, {1u, 3.0}},
                                                      std::vector<Row>{{1u, 2.0}, {1u, 3.0}, {2u, 6.0}}}
                : std::array<std::vector<Row>, kRows>{std::vector<Row>{{1u, 2.0}, {2u, 6.0}, {3u, kSphereEntry}},
                                                      std::vector<Row>{{1u, 2.0}},
                                                      std::vector<Row>{{1u, 2.0}, {2u, 6.0}, {3u, kSphereEntry}}};
        for (uint64_t row = 0u; row < kRows; ++row)
        {
            std::vector<Row> found;
            for (uint64_t index = offsets[row]; index < offsets[row] + counts[row]; ++index)
            {
                found.emplace_back(ids[index], distances[index]);
            }
            std::sort(found.begin(), found.end());
            REQUIRE(found.size() == expected[row].size());
            REQUIRE(truncated[row] == (all && row == 2u ? 1u : 0u));
            for (size_t index = 0u; index < found.size(); ++index)
            {
                REQUIRE(found[index].first == expected[row][index].first &&
                        std::abs(found[index].second - expected[row][index].second) < 1.0e-6);
            }

            // The synchronous CPU query stores the same rows.
            ovspatialquery_raycast_desc_t ray{};
            ray.common.target    = {populationRef(population), {OVSPATIALQUERY_ORDINAL_HEAD, 0}};
            ray.common.precision = precision;
            for (uint32_t axis = 0u; axis < 3u; ++axis)
            {
                ray.ray.origin[axis]         = origins[row * 3u + axis];
                ray.ray.unit_direction[axis] = directions[row * 3u + axis];
            }
            ray.ray.max_distance = maximumDistances[row];
            ray.mesh_hits        = meshHits;
            std::array<ovspatialquery_raycast_hit_t, 8u> cpuRows{};
            ovspatialquery_raycast_hit_buffer_t          cpuBuffer{cpuRows.data(), maximumHits[row]};
            ovspatialquery_multiple_result_t             cpuResult{};
            REQUIRE(ovspatialquery_raycast_multiple_sync(instance, &ray, &cpuBuffer, &cpuResult).status ==
                    OVSPATIALQUERY_API_SUCCESS);
            std::vector<Row> cpu;
            for (uint64_t index = 0u; index < cpuResult.stored_hit_count; ++index)
            {
                cpu.emplace_back(cpuRows[index].data.exact.identity.object_id.value,
                                 cpuRows[index].data.exact.distance);
            }
            std::sort(cpu.begin(), cpu.end());
            REQUIRE(cpu.size() == found.size() && cpuResult.truncated == truncated[row]);
            for (size_t index = 0u; index < cpu.size(); ++index)
            {
                REQUIRE(cpu[index].first == found[index].first &&
                        std::abs(cpu[index].second - found[index].second) < 1.0e-5);
            }
        }
        REQUIRE(destroyPlanWhenRetired(instance, plan));
    }

    // A tolerance tube has no crossing count: an ALL plan refuses it.
    ovspatialquery_raycast_plan_desc_t tube{};
    tube.target                                = populationRef(population);
    tube.hit_mode                              = OVSPATIALQUERY_HIT_MODE_MULTIPLE;
    tube.precision                             = precision;
    tube.device                                = {kDLCUDA, 0};
    tube.maximum_query_count                   = 1u;
    tube.maximum_hits_per_query                = 1u;
    tube.hit_capacity_per_slot                 = 1u;
    tube.result_slot_count                     = 1u;
    tube.contact_tolerance                     = 0.01;
    ovspatialquery_batch_plan_handle_t refused = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_raycast_plan(instance, &tube, &refused).status == OVSPATIALQUERY_API_NOT_SUPPORTED);

    cuda.free(deviceOrigins);
    cuda.free(deviceDirections);
    cuda.free(deviceMaximum);
    cuda.free(deviceHits);
    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, meshValue})
                .status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_SPHERE, kSphere}).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

// The box sweep kernel needs more than the driver's default 1 KiB per-thread
// stack. Unless the plan reserves it when created, the first launch grows the
// stack and waits for the whole device, so an enqueue behind a caller stream
// that is still held would block until the caller lets it go.
void testCudaPlanEnqueueDoesNotWaitForTheDevice()
{
    ExactMeshFixture                     fixture;
    CudaDriver                           cuda(CudaDriver::Ownership::tracked);
    const CudaDriver::Stream             stream   = cuda.createStream();
    const CudaDriver::Stream             consumer = cuda.createStream();
    ovspatialquery_box_sweep_plan_desc_t desc{};
    desc.target                             = populationRef(fixture.population);
    desc.precision                          = OVSPATIALQUERY_QUERY_EXACT;
    desc.half_extents[0]                    = 0.25;
    desc.half_extents[1]                    = 0.25;
    desc.half_extents[2]                    = 0.25;
    desc.device                             = {kDLCUDA, 0};
    desc.maximum_query_count                = 1u;
    desc.result_slot_count                  = 1u;
    desc.hit_capacity_per_slot              = 1u;
    desc.result_fields                      = OVSPATIALQUERY_BATCH_RESULT_DISTANCE;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_box_sweep_plan(fixture.instance, &desc, &plan).status == OVSPATIALQUERY_API_SUCCESS);

    double                          center[3]       = {0.0, 0.0, 0.0};
    float                           rotation[4]     = {0.0f, 0.0f, 0.0f, 1.0f};
    float                           direction[3]    = {1.0f, 0.0f, 0.0f};
    double                          maximumDistance = 10.0;
    const CudaDriver::DevicePointer deviceCenter    = upload(cuda, center, stream);
    const CudaDriver::DevicePointer deviceRotation  = upload(cuda, rotation, stream);
    const CudaDriver::DevicePointer deviceDirection = upload(cuda, direction, stream);
    const CudaDriver::DevicePointer deviceMaximum   = upload(cuda, maximumDistance, stream);
    gPhase                                          = "enqueue a box sweep behind a held caller stream";
    // Hold the caller's stream until enqueue returns; a watchdog lets it go after
    // ten seconds so a regression fails the timing check instead of hanging.
    std::promise<void> release, enqueued;
    std::future<void>  released = release.get_future();
    cuda.enqueueHostFunction(
        stream, [](void* state) { static_cast<std::future<void>*>(state)->wait(); }, &released);
    std::thread watchdog(
        [&release, returned = enqueued.get_future()]
        {
            returned.wait_for(std::chrono::seconds(10));
            release.set_value();
        });
    InputArray                       centers(devicePointer<double>(deviceCenter), 1, 3, kDLFloat, kCudaDevice);
    InputArray                       rotations(devicePointer<float>(deviceRotation), 1, 4, kDLFloat, kCudaDevice);
    InputArray                       directions(devicePointer<float>(deviceDirection), 1, 3, kDLFloat, kCudaDevice);
    InputArray                       maximums(devicePointer<double>(deviceMaximum), 1, kDLFloat, kCudaDevice);
    ovspatialquery_box_sweep_batch_t batch{};
    batch.ordinal.kind         = OVSPATIALQUERY_ORDINAL_HEAD;
    batch.query_count          = 1u;
    batch.centers              = &centers.value;
    batch.rotations_xyzw       = &rotations.value;
    batch.unit_directions      = &directions.value;
    batch.max_distances        = &maximums.value;
    batch.cuda_sync.has_stream = 1u;
    batch.cuda_sync.stream     = reinterpret_cast<uintptr_t>(stream);
    ovspatialquery_box_sweep_batch_reservation_t reservation{};
    const std::chrono::steady_clock::time_point  start = std::chrono::steady_clock::now();
    ovspatialquery_enqueue_result_t              admission =
        ovspatialquery_enqueue_box_sweep_plan(fixture.instance, plan, &batch, &reservation);
    const bool returnedWhileHeld = std::chrono::steady_clock::now() - start < std::chrono::seconds(5);
    enqueued.set_value();
    watchdog.join();
    REQUIRE(admission.status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(returnedWhileHeld);
    ovspatialquery_op_wait_result_t waited{};
    REQUIRE(
        ovspatialquery_wait_op(fixture.instance, admission.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
        OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(waited.state == OVSPATIALQUERY_OP_COMPLETED);
    ovspatialquery_box_sweep_batch_result_t result{};
    REQUIRE(ovspatialquery_fetch_box_sweep_batch(fixture.instance, admission.op_id, &result).status ==
            OVSPATIALQUERY_API_SUCCESS);
    cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(result.common.result_ready.value));
    double distance = -1.0;
    cuda.downloadAsync(&distance, deviceAddress(result.columns.contact.exact.distances), sizeof(distance), consumer);
    retireAfterConsumer(cuda, consumer, fixture.instance, admission.op_id);
    REQUIRE(std::abs(distance - 3.75) < 1.0e-5);
    REQUIRE(destroyPlanWhenRetired(fixture.instance, plan));
    fixture.close();
    REQUIRE(transferredEventDestroysWereInvoked());
}

// One exact closest ray from the origin along +x, on one CUDA device: plans for it
// over a population, and launches that report the distance of their one hit.
struct OriginRay
{
    OriginRay(CudaDriver& driver, int32_t ordinal)
        : cuda(driver), device{kDLCUDA, ordinal}, stream(driver.createStream()), consumer(driver.createStream())
    {
        origins.set(const_cast<double*>(devicePointer<double>(upload(cuda, origin, stream))), 1, 3, kDLFloat, device);
        directions.set(const_cast<float*>(devicePointer<float>(upload(cuda, direction, stream))), 1, 3, kDLFloat,
                       device);
        maximums.set(const_cast<double*>(devicePointer<double>(upload(cuda, maximum, stream))), 1, kDLFloat, device);
    }

    ovspatialquery_batch_plan_handle_t createPlan(ovspatialquery_instance_handle_t   instance,
                                                  ovspatialquery_population_handle_t population) const
    {
        ovspatialquery_raycast_plan_desc_t desc{};
        desc.target                             = populationRef(population);
        desc.hit_mode                           = OVSPATIALQUERY_HIT_MODE_CLOSEST;
        desc.precision                          = OVSPATIALQUERY_QUERY_EXACT;
        desc.device                             = device;
        desc.maximum_query_count                = 1u;
        desc.maximum_hits_per_query             = 1u;
        desc.hit_capacity_per_slot              = 1u;
        desc.result_slot_count                  = 2u;
        desc.result_fields                      = OVSPATIALQUERY_BATCH_RESULT_DISTANCE;
        ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
        REQUIRE(ovspatialquery_create_raycast_plan(instance, &desc, &plan).status == OVSPATIALQUERY_API_SUCCESS);
        return plan;
    }

    // Admits a launch on `ordinal`, or on the head when it is negative.
    ovspatialquery_op_id_t enqueue(ovspatialquery_instance_handle_t instance, ovspatialquery_batch_plan_handle_t plan,
                                   ovspatialquery_ordinal_t ordinal = -1)
    {
        ovspatialquery_raycast_batch_t batch{};
        batch.ordinal.kind = ordinal < 0 ? OVSPATIALQUERY_ORDINAL_HEAD : OVSPATIALQUERY_ORDINAL_RETAINED_EXACT;
        batch.ordinal.source_ordinal = ordinal < 0 ? 0 : ordinal;
        batch.query_count            = 1u;
        batch.origins                = &origins.value;
        batch.unit_directions        = &directions.value;
        batch.max_distances          = &maximums.value;
        batch.cuda_sync.has_stream   = 1u;
        batch.cuda_sync.stream       = reinterpret_cast<uintptr_t>(stream);
        ovspatialquery_raycast_batch_reservation_t reservation{};
        const ovspatialquery_enqueue_result_t      admitted =
            ovspatialquery_enqueue_raycast_plan(instance, plan, &batch, &reservation);
        REQUIRE(admitted.status == OVSPATIALQUERY_API_SUCCESS);
        return admitted.op_id;
    }

    // Waits for the launch and retires it; the hit's distance, or -1 for a miss.
    double finish(ovspatialquery_instance_handle_t instance, ovspatialquery_op_id_t operation)
    {
        ovspatialquery_op_wait_result_t waited{};
        REQUIRE(ovspatialquery_wait_op(instance, operation, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
                OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(waited.state == OVSPATIALQUERY_OP_COMPLETED);
        ovspatialquery_raycast_batch_result_t result{};
        REQUIRE(ovspatialquery_fetch_raycast_batch(instance, operation, &result).status == OVSPATIALQUERY_API_SUCCESS);
        cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(result.common.result_ready.value));
        uint64_t count    = 0u;
        double   distance = -1.0;
        cuda.downloadAsync(&count, deviceAddress(result.columns.hit_ranges.hit_counts), sizeof(count), consumer);
        cuda.downloadAsync(&distance, deviceAddress(result.columns.contact.exact.distances), sizeof(distance),
                           consumer);
        retireAfterConsumer(cuda, consumer, instance, operation);
        return count == 1u ? distance : -1.0;
    }

    double launch(ovspatialquery_instance_handle_t instance, ovspatialquery_batch_plan_handle_t plan,
                  ovspatialquery_ordinal_t ordinal = -1)
    {
        return finish(instance, enqueue(instance, plan, ordinal));
    }

    CudaDriver&              cuda;
    const DLDevice           device;
    const CudaDriver::Stream stream;
    const CudaDriver::Stream consumer;
    const double             origin[3]    = {0.0, 0.0, 0.0};
    const float              direction[3] = {1.0f, 0.0f, 0.0f};
    const double             maximum      = 10.0;
    InputArray               origins, directions, maximums;
};

uint64_t deviceCharge(ovspatialquery_instance_handle_t instance, DLDevice device)
{
    uint64_t bytes = 0u;
    REQUIRE(ovspatialquery_get_instance_memory_usage(instance, device, &bytes).status == OVSPATIALQUERY_API_SUCCESS);
    return bytes;
}

// An instance's CUDA plans share the device copy of a mesh definition: the launch
// that first needs it uploads it and charges the device budget, later plans read it,
// and it stays charged while any plan's catalogue holds it, whichever plan uploaded it.
void testCudaPlansShareMeshDefinitions()
{
    constexpr uint64_t kNearMesh = 78u; // the triangle one unit nearer the origin
    // A second worker publishes while the first waits for the held launch.
    ExactMeshFixture fixture(UINT64_C(64) << 20, 1u, 2u);
    fixture.registerMesh(kNearMesh, -1.0f);
    CudaDriver cuda(CudaDriver::Ownership::tracked);
    OriginRay  first(cuda, 0);
    OriginRay  second(cuda, 0);
    const auto charge = [&fixture]() { return deviceCharge(fixture.instance, kCudaDevice); };

    gPhase                                       = "the second plan shares the definition the first uploaded";
    const uint64_t                     noPlans   = charge();
    ovspatialquery_batch_plan_handle_t uploader  = first.createPlan(fixture.instance, fixture.population);
    ovspatialquery_batch_plan_handle_t sharer    = second.createPlan(fixture.instance, fixture.population);
    const uint64_t                     created   = charge();
    const uint64_t                     planBytes = (created - noPlans) / 2u;
    REQUIRE(std::abs(first.launch(fixture.instance, uploader) - 4.0) < 1.0e-9);
    const uint64_t uploaded = charge();
    REQUIRE(std::abs(second.launch(fixture.instance, sharer) - 4.0) < 1.0e-9);
    const uint64_t shared = charge();
    // Both plans publish the same tables; only the first also uploads the definition.
    const uint64_t tableBytes = shared - uploaded;
    REQUIRE(tableBytes != 0u && uploaded - created > tableBytes);

    gPhase = "the population changes its mesh while the sharer's launch is in flight";
    std::promise<void> release;
    std::future<void>  released = release.get_future();
    cuda.enqueueHostFunction(
        second.stream, [](void* state) { static_cast<std::future<void>*>(state)->wait(); }, &released);
    const ovspatialquery_op_id_t inFlight = second.enqueue(fixture.instance, sharer);
    fixture.publish(10, 4.0, kNearMesh);
    const ovspatialquery_op_id_t nearLaunch = first.enqueue(fixture.instance, uploader);
    release.set_value();
    REQUIRE(std::abs(second.finish(fixture.instance, inFlight) - 4.0) < 1.0e-9);
    REQUIRE(std::abs(first.finish(fixture.instance, nearLaunch) - 3.0) < 1.0e-9);
    const uint64_t bothMeshes = charge();
    REQUIRE(std::abs(second.launch(fixture.instance, sharer) - 3.0) < 1.0e-9);
    REQUIRE(charge() == bothMeshes + tableBytes);

    gPhase = "destroying the uploader leaves both definitions charged to the sharer";
    REQUIRE(destroyPlanWhenRetired(fixture.instance, uploader));
    REQUIRE(charge() == bothMeshes + tableBytes - planBytes - 2u * tableBytes);
    REQUIRE(std::abs(second.launch(fixture.instance, sharer, fixture.sourceOrdinal) - 4.0) < 1.0e-9);
    REQUIRE(std::abs(second.launch(fixture.instance, sharer) - 3.0) < 1.0e-9);
    REQUIRE(destroyPlanWhenRetired(fixture.instance, sharer));
    REQUIRE(charge() == noPlans);
    fixture.close();
    REQUIRE(transferredEventDestroysWereInvoked(7u));
}

// Plans on two devices of one instance each upload the definition to their own
// device and charge it there. Skipped with one CUDA device.
void testCudaPlansOnTwoDevicesKeepTheirOwnDefinitions()
{
    ExactMeshFixture fixture(UINT64_C(64) << 20, 2u);
    const DLDevice   secondDevice{kDLCUDA, 1};
    if (ovspatialquery_query_capability(fixture.instance, secondDevice, OVSPATIALQUERY_CAPABILITY_RAYCAST_PLAN_EXACT)
            .status != OVSPATIALQUERY_API_SUCCESS)
    {
        std::printf("skipped: two CUDA plans on two devices need a second CUDA device\n");
        fixture.close();
        return;
    }
    gPhase = "a plan on each of two devices";
    ovspatialquery_batch_plan_handle_t plans[2]{};
    uint64_t                           firstLaunchBytes[2]{};
    for (int32_t ordinal = 0; ordinal < 2; ++ordinal)
    {
        CudaDriver     cuda(CudaDriver::Ownership::tracked, ordinal);
        OriginRay      ray(cuda, ordinal);
        const DLDevice device{kDLCUDA, ordinal};
        plans[ordinal]         = ray.createPlan(fixture.instance, fixture.population);
        const uint64_t created = deviceCharge(fixture.instance, device);
        REQUIRE(std::abs(ray.launch(fixture.instance, plans[ordinal]) - 4.0) < 1.0e-9);
        firstLaunchBytes[ordinal] = deviceCharge(fixture.instance, device) - created;
    }
    // The first device's plan still holds its definition, and the second device's plan
    // uploaded its own: it charged as much as the first.
    REQUIRE(firstLaunchBytes[0] != 0u && firstLaunchBytes[1] == firstLaunchBytes[0]);
    REQUIRE(destroyPlanWhenRetired(fixture.instance, plans[0]) && destroyPlanWhenRetired(fixture.instance, plans[1]));
    fixture.close();
    REQUIRE(transferredEventDestroysWereInvoked(2u));
}

// ADR-0081: CUDA rays meet a convex where its instance's uniform scale and pose
// put the hull cooked from its local points. A box collider of a centimetre stage,
// 100 by 60 by 40 cm, registers under a hull vertex limit of 8 from its corners in
// centimetres and is placed with scale 0.01, a quarter turn about z and a
// translation, so in metres it spans 0.6 by 1.0 by 0.4 about (3, 1, 2). The fourth
// ray passes 0.1 m beside it; a reader that ignored the scale would meet a box
// a hundred times larger.
void testCudaRaysMeetAConvexWhereItsUniformScalePlacesIt()
{
    gPhase = "publish a scaled convex";
    const ovspatialquery_device_budget_t budget{{kDLCUDA, 0}, UINT64_C(64) << 20};
    const ovspatialquery_instance_desc_t desc{4u, 1u, UINT64_C(64) << 20, &budget, 1u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    uint64_t convexValue = 82u, vertexOffset = 0u, vertexCount = 8u;
    float    corners[24]{};
    for (uint32_t corner = 0u; corner < 8u; ++corner)
    {
        corners[corner * 3u]      = (corner & 1u) != 0u ? 50.0f : -50.0f;
        corners[corner * 3u + 1u] = (corner & 2u) != 0u ? 30.0f : -30.0f;
        corners[corner * 3u + 2u] = (corner & 4u) != 0u ? 20.0f : -20.0f;
    }
    InputArray                                     values(&convexValue, 1, kDLUInt);
    InputArray                                     offsets(&vertexOffset, 1, kDLUInt);
    InputArray                                     counts(&vertexCount, 1, kDLUInt);
    InputArray                                     vertices(corners, 8, 3, kDLFloat);
    const ovspatialquery_convex_definition_table_t convexes{
        &values.value, &offsets.value, &counts.value, nullptr, nullptr, &vertices.value,
        nullptr,       nullptr,        nullptr,       0.0,     8u};
    waitAndDestroy(instance, ovspatialquery_enqueue_convex_registration(instance, &convexes));
    uint64_t   objectValue  = 1u;
    int32_t    geometryType = OVSPATIALQUERY_GEOMETRY_CONVEX;
    double     translation[3]{3.0, 1.0, 2.0}, rotation[4]{0.0, 0.0, std::sqrt(0.5), std::sqrt(0.5)}, scale = 0.01;
    InputArray objects(&objectValue, 1, kDLUInt);
    InputArray types(&geometryType, 1, kDLInt);
    InputArray geometry(&convexValue, 1, kDLUInt);
    InputArray translations(translation, 1, 3, kDLFloat);
    InputArray rotations(rotation, 1, 4, kDLFloat);
    InputArray scales(&scale, 1, kDLFloat);
    const ovspatialquery_population_handle_t       population = createExplicitPopulation(instance, 1u);
    const ovspatialquery_explicit_population_set_t set{1,
                                                       {&objects.value, &types.value, &geometry.value,
                                                        &translations.value, &rotations.value, &scales.value, nullptr,
                                                        nullptr},
                                                       nullptr,
                                                       0u};
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));

    // Along +x onto the face at x = 2.7, along -y onto y = 1.5, along +z onto z = 1.8,
    // and along +x at y = 1.6, beside the box.
    constexpr uint64_t             kRows = 4u;
    std::array<double, kRows * 3u> origins{0.0, 1.0, 2.0, 3.0, 5.0, 2.0, 3.0, 1.0, 0.0, 0.0, 1.6, 2.0};
    std::array<float, kRows * 3u>  directions{1.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f};
    std::array<double, kRows>      maximumDistances{10.0, 10.0, 10.0, 10.0};
    const std::array<double, 3u>   expected{2.7, 3.5, 1.8};
    CudaDriver                     cuda(CudaDriver::Ownership::tracked);
    const CudaDriver::Stream       stream           = cuda.createStream();
    const auto                     deviceOrigins    = upload(cuda, origins, stream);
    const auto                     deviceDirections = upload(cuda, directions, stream);
    const auto                     deviceMaximum    = upload(cuda, maximumDistances, stream);
    InputArray                     originColumn(devicePointer<double>(deviceOrigins), kRows, 3, kDLFloat, kCudaDevice);
    InputArray directionColumn(devicePointer<float>(deviceDirections), kRows, 3, kDLFloat, kCudaDevice);
    InputArray maximumColumn(devicePointer<double>(deviceMaximum), kRows, kDLFloat, kCudaDevice);

    gPhase = "CUDA raycast a scaled convex";
    ovspatialquery_raycast_plan_desc_t planDesc{};
    planDesc.target                         = populationRef(population);
    planDesc.hit_mode                       = OVSPATIALQUERY_HIT_MODE_CLOSEST;
    planDesc.precision                      = OVSPATIALQUERY_QUERY_EXACT;
    planDesc.device                         = {kDLCUDA, 0};
    planDesc.maximum_query_count            = kRows;
    planDesc.maximum_hits_per_query         = 1u;
    planDesc.hit_capacity_per_slot          = kRows;
    planDesc.result_slot_count              = 1u;
    planDesc.result_fields                  = OVSPATIALQUERY_BATCH_RESULT_DISTANCE;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_raycast_plan(instance, &planDesc, &plan).status == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_raycast_batch_t batch{};
    batch.ordinal.kind         = OVSPATIALQUERY_ORDINAL_HEAD;
    batch.query_count          = kRows;
    batch.origins              = &originColumn.value;
    batch.unit_directions      = &directionColumn.value;
    batch.max_distances        = &maximumColumn.value;
    batch.cuda_sync.has_stream = 1u;
    batch.cuda_sync.stream     = reinterpret_cast<uintptr_t>(stream);
    ovspatialquery_raycast_batch_reservation_t reservation{};
    ovspatialquery_enqueue_result_t            operation =
        ovspatialquery_enqueue_raycast_plan(instance, plan, &batch, &reservation);
    REQUIRE(operation.status == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_op_wait_result_t waited{};
    REQUIRE(ovspatialquery_wait_op(instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
                OVSPATIALQUERY_API_SUCCESS &&
            waited.state == OVSPATIALQUERY_OP_COMPLETED);
    ovspatialquery_raycast_batch_result_t result{};
    REQUIRE(ovspatialquery_fetch_raycast_batch(instance, operation.op_id, &result).status ==
            OVSPATIALQUERY_API_SUCCESS);
    cuda.wait(stream, reinterpret_cast<CudaDriver::Event>(result.common.result_ready.value));
    std::array<uint64_t, kRows> hitOffsets{}, hitCounts{};
    std::array<double, kRows>   distances{};
    const auto&                 ranges = result.columns.hit_ranges;
    cuda.downloadAsync(hitOffsets.data(), deviceAddress(ranges.hit_offsets), sizeof(hitOffsets), stream);
    cuda.downloadAsync(hitCounts.data(), deviceAddress(ranges.hit_counts), sizeof(hitCounts), stream);
    cuda.downloadAsync(distances.data(), deviceAddress(result.columns.contact.exact.distances),
                       size_t(result.columns.contact.exact.distances.shape[0]) * sizeof(double), stream);
    retireAfterConsumer(cuda, stream, instance, operation.op_id);
    REQUIRE(transferredEventDestroysWereInvoked());
    for (uint64_t row = 0u; row < 3u; ++row)
    {
        REQUIRE(hitCounts[row] == 1u && std::abs(distances[hitOffsets[row]] - expected[row]) < 1.0e-9);
    }
    REQUIRE(hitCounts[3] == 0u);
    REQUIRE(destroyPlanWhenRetired(instance, plan));
    cuda.free(deviceOrigins);
    cuda.free(deviceDirections);
    cuda.free(deviceMaximum);
    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(
        ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_CONVEX, convexValue}).status ==
        OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

// A ray that starts far from a capsule reaches it at its analytic entry on CUDA
// (BH-027): the root of a far path loses eps |origin - capsule|^2 to cancellation,
// so the contact must not be judged by a radius-relative residual.
void testCudaRaycastReachesFarCapsule()
{
    gPhase = "publish far capsule";
    const ovspatialquery_device_budget_t budget{{kDLCUDA, 0}, UINT64_C(64) << 20};
    const ovspatialquery_instance_desc_t desc{4u, 1u, UINT64_C(64) << 20, &budget, 1u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    uint64_t   capsuleValue = 81u;
    double     radius = 0.5, halfSegment = 1.0;
    InputArray values(&capsuleValue, 1, kDLUInt), radii(&radius, 1, kDLFloat), halves(&halfSegment, 1, kDLFloat);
    const ovspatialquery_capsule_definition_table_t capsules{&values.value, &radii.value, &halves.value};
    waitAndDestroy(instance, ovspatialquery_enqueue_capsule_registration(instance, &capsules));
    const ovspatialquery_population_handle_t population = createExplicitPopulation(instance, 1u);
    InstanceRows                             rows;
    rows.add(1u, OVSPATIALQUERY_GEOMETRY_CAPSULE, capsuleValue, 0.0, 0.0, 0.0);
    const ovspatialquery_explicit_population_set_t set{1, rows.table(), nullptr, 0u};
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));

    // Each ray runs along +x from x = -distance at height y, offset z, across the
    // segment's span, so it enters the side at distance - sqrt(r^2 - z^2).
    constexpr uint64_t             kRows = 16u;
    std::array<double, kRows * 3u> origins{};
    std::array<float, kRows * 3u>  directions{};
    std::array<double, kRows>      maximumDistances{}, expected{};
    for (uint64_t row = 0u; row < kRows; ++row)
    {
        const double distance = 1000.0 + 37.25 * double(row), z = 0.45 * double(row) / double(kRows);
        origins[row * 3u]      = -distance;
        origins[row * 3u + 1u] = 0.0625 * double(row) - 0.5;
        origins[row * 3u + 2u] = z;
        directions[row * 3u]   = 1.0f;
        maximumDistances[row]  = 2.0 * distance;
        expected[row]          = distance - std::sqrt(radius * radius - z * z);
    }
    CudaDriver               cuda(CudaDriver::Ownership::tracked);
    const CudaDriver::Stream stream           = cuda.createStream();
    const auto               deviceOrigins    = upload(cuda, origins, stream);
    const auto               deviceDirections = upload(cuda, directions, stream);
    const auto               deviceMaximum    = upload(cuda, maximumDistances, stream);
    InputArray               originColumn(devicePointer<double>(deviceOrigins), kRows, 3, kDLFloat, kCudaDevice);
    InputArray               directionColumn(devicePointer<float>(deviceDirections), kRows, 3, kDLFloat, kCudaDevice);
    InputArray               maximumColumn(devicePointer<double>(deviceMaximum), kRows, kDLFloat, kCudaDevice);

    gPhase = "CUDA raycast far capsule";
    ovspatialquery_raycast_plan_desc_t planDesc{};
    planDesc.target                         = populationRef(population);
    planDesc.hit_mode                       = OVSPATIALQUERY_HIT_MODE_CLOSEST;
    planDesc.precision                      = OVSPATIALQUERY_QUERY_EXACT;
    planDesc.device                         = {kDLCUDA, 0};
    planDesc.maximum_query_count            = kRows;
    planDesc.maximum_hits_per_query         = 1u;
    planDesc.hit_capacity_per_slot          = kRows;
    planDesc.result_slot_count              = 1u;
    planDesc.result_fields                  = OVSPATIALQUERY_BATCH_RESULT_DISTANCE;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_raycast_plan(instance, &planDesc, &plan).status == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_raycast_batch_t batch{};
    batch.ordinal.kind         = OVSPATIALQUERY_ORDINAL_HEAD;
    batch.query_count          = kRows;
    batch.origins              = &originColumn.value;
    batch.unit_directions      = &directionColumn.value;
    batch.max_distances        = &maximumColumn.value;
    batch.cuda_sync.has_stream = 1u;
    batch.cuda_sync.stream     = reinterpret_cast<uintptr_t>(stream);
    ovspatialquery_raycast_batch_reservation_t reservation{};
    ovspatialquery_enqueue_result_t            operation =
        ovspatialquery_enqueue_raycast_plan(instance, plan, &batch, &reservation);
    REQUIRE(operation.status == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_op_wait_result_t waited{};
    REQUIRE(ovspatialquery_wait_op(instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
                OVSPATIALQUERY_API_SUCCESS &&
            waited.state == OVSPATIALQUERY_OP_COMPLETED);
    ovspatialquery_raycast_batch_result_t result{};
    REQUIRE(ovspatialquery_fetch_raycast_batch(instance, operation.op_id, &result).status ==
            OVSPATIALQUERY_API_SUCCESS);
    cuda.wait(stream, reinterpret_cast<CudaDriver::Event>(result.common.result_ready.value));
    std::array<uint64_t, kRows> offsets{}, counts{};
    std::array<double, kRows>   distances{};
    const auto&                 ranges = result.columns.hit_ranges;
    cuda.downloadAsync(offsets.data(), deviceAddress(ranges.hit_offsets), sizeof(offsets), stream);
    cuda.downloadAsync(counts.data(), deviceAddress(ranges.hit_counts), sizeof(counts), stream);
    cuda.downloadAsync(distances.data(), deviceAddress(result.columns.contact.exact.distances),
                       size_t(result.columns.contact.exact.distances.shape[0]) * sizeof(double), stream);
    retireAfterConsumer(cuda, stream, instance, operation.op_id);
    REQUIRE(transferredEventDestroysWereInvoked());
    for (uint64_t row = 0u; row < kRows; ++row)
    {
        REQUIRE(counts[row] == 1u && std::abs(distances[offsets[row]] - expected[row]) < 1.0e-9);
    }

    // The same rays through the layouts producers hand over (ADR-0053): origins
    // sliced from one (Q, 6) ray array, directions in column-major order, and
    // one maximum distance repeated by a zero stride.
    gPhase = "CUDA raycast far capsule from strided columns";
    std::array<double, kRows * 6u> rays{};
    std::array<float, kRows * 3u>  columnMajorDirections{};
    for (uint64_t row = 0u; row < kRows; ++row)
    {
        for (uint64_t axis = 0u; axis < 3u; ++axis)
        {
            rays[row * 6u + axis]                     = origins[row * 3u + axis];
            rays[row * 6u + 3u + axis]                = double(directions[row * 3u + axis]);
            columnMajorDirections[axis * kRows + row] = directions[row * 3u + axis];
        }
    }
    const double sharedMaximum       = 5000.0;
    const auto   deviceRays          = upload(cuda, rays, stream);
    const auto   deviceColumnMajor   = upload(cuda, columnMajorDirections, stream);
    const auto   deviceSharedMaximum = upload(cuda, sharedMaximum, stream);
    InputArray   rayOrigins(devicePointer<double>(deviceRays), kRows, 3, kDLFloat, kCudaDevice);
    InputArray   columnMajor(devicePointer<float>(deviceColumnMajor), kRows, 3, kDLFloat, kCudaDevice);
    InputArray   repeatedMaximum(devicePointer<double>(deviceSharedMaximum), kRows, kDLFloat, kCudaDevice);
    rayOrigins.strides      = {6, 1};
    columnMajor.strides     = {1, int64_t(kRows)};
    repeatedMaximum.strides = {0, 0};
    batch.origins           = &rayOrigins.value;
    batch.unit_directions   = &columnMajor.value;
    batch.max_distances     = &repeatedMaximum.value;
    rayOrigins.strides[0]   = -6;
    REQUIRE(ovspatialquery_enqueue_raycast_plan(instance, plan, &batch, &reservation).status ==
            OVSPATIALQUERY_API_INVALID_ARGUMENT);
    const ovspatialquery_string_t refusal = ovspatialquery_get_last_error();
    REQUIRE(std::string(refusal.ptr, refusal.length).find("origins") != std::string::npos);
    rayOrigins.strides[0] = 6;
    // The one-slot plan's first operation was retired after a finished
    // consumer, so its slot is free without waiting for the reaper.
    operation = ovspatialquery_enqueue_raycast_plan(instance, plan, &batch, &reservation);
    REQUIRE(operation.status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_wait_op(instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
                OVSPATIALQUERY_API_SUCCESS &&
            waited.state == OVSPATIALQUERY_OP_COMPLETED);
    result = {};
    REQUIRE(ovspatialquery_fetch_raycast_batch(instance, operation.op_id, &result).status ==
            OVSPATIALQUERY_API_SUCCESS);
    cuda.wait(stream, reinterpret_cast<CudaDriver::Event>(result.common.result_ready.value));
    offsets   = {};
    counts    = {};
    distances = {};
    cuda.downloadAsync(offsets.data(), deviceAddress(result.columns.hit_ranges.hit_offsets), sizeof(offsets), stream);
    cuda.downloadAsync(counts.data(), deviceAddress(result.columns.hit_ranges.hit_counts), sizeof(counts), stream);
    cuda.downloadAsync(distances.data(), deviceAddress(result.columns.contact.exact.distances),
                       size_t(result.columns.contact.exact.distances.shape[0]) * sizeof(double), stream);
    retireAfterConsumer(cuda, stream, instance, operation.op_id);
    REQUIRE(transferredEventDestroysWereInvoked());
    for (uint64_t row = 0u; row < kRows; ++row)
    {
        REQUIRE(counts[row] == 1u && std::abs(distances[offsets[row]] - expected[row]) < 1.0e-9);
    }

    REQUIRE(destroyPlanWhenRetired(instance, plan));
    cuda.free(deviceOrigins);
    cuda.free(deviceDirections);
    cuda.free(deviceMaximum);
    cuda.free(deviceRays);
    cuda.free(deviceColumnMajor);
    cuda.free(deviceSharedMaximum);
    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(
        ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_CAPSULE, capsuleValue}).status ==
        OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

// Rows for one EXACT CLOSEST CUDA plan and what it reported for each of them.
struct ClosestRows
{
    std::vector<double>   origins;    // [Q, 3]
    std::vector<float>    directions; // [Q, 3]
    std::vector<double>   maximumDistances;
    std::vector<uint64_t> counts;
    std::vector<double>   distances; // valid where counts[row] != 0
};

// Runs the rows through one CUDA plan on population, EXACT CLOSEST unless given:
// a raycast when radius is 0, a sphere sweep of that radius otherwise. A MULTIPLE
// row keeps up to four hits; `distances` holds its first.
void runCudaClosest(ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population,
                    double radius, ClosestRows& rows,
                    ovspatialquery_query_precision_t precision = OVSPATIALQUERY_QUERY_EXACT,
                    ovspatialquery_hit_mode_t        mode      = OVSPATIALQUERY_HIT_MODE_CLOSEST)
{
    const uint64_t           perRow = mode == OVSPATIALQUERY_HIT_MODE_MULTIPLE ? 4u : 1u;
    const uint64_t           count  = rows.maximumDistances.size();
    CudaDriver               cuda(CudaDriver::Ownership::tracked);
    const CudaDriver::Stream stream       = cuda.createStream();
    const auto               uploadVector = [&](const auto& values)
    {
        const size_t                    bytes   = values.size() * sizeof(values[0]);
        const CudaDriver::DevicePointer pointer = cuda.allocate(bytes);
        cuda.uploadAsync(pointer, values.data(), bytes, stream);
        return pointer;
    };
    const auto     deviceOrigins    = uploadVector(rows.origins);
    const auto     deviceDirections = uploadVector(rows.directions);
    const auto     deviceMaximum    = uploadVector(rows.maximumDistances);
    InputArray     originColumn(devicePointer<double>(deviceOrigins), int64_t(count), 3, kDLFloat, kCudaDevice);
    InputArray     directionColumn(devicePointer<float>(deviceDirections), int64_t(count), 3, kDLFloat, kCudaDevice);
    InputArray     maximumColumn(devicePointer<double>(deviceMaximum), int64_t(count), kDLFloat, kCudaDevice);
    const uint32_t fields                   = OVSPATIALQUERY_BATCH_RESULT_DISTANCE;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    ovspatialquery_enqueue_result_t    operation{};
    ovspatialquery_op_wait_result_t    waited{};
    const auto                         download = [&](const auto& result)
    {
        const auto& columns = result.columns;
        cuda.wait(stream, reinterpret_cast<CudaDriver::Event>(result.common.result_ready.value));
        std::vector<uint64_t> offsets(count);
        rows.counts.assign(count, 0u);
        rows.distances.assign(count, 0.0);
        const uint64_t      stored = uint64_t(columns.contact.exact.distances.shape[0]);
        std::vector<double> distances(stored);
        cuda.downloadAsync(offsets.data(), deviceAddress(columns.hit_ranges.hit_offsets), count * 8u, stream);
        cuda.downloadAsync(rows.counts.data(), deviceAddress(columns.hit_ranges.hit_counts), count * 8u, stream);
        cuda.downloadAsync(distances.data(), deviceAddress(columns.contact.exact.distances), stored * 8u, stream);
        retireAfterConsumer(cuda, stream, instance, operation.op_id);
        REQUIRE(transferredEventDestroysWereInvoked());
        for (uint64_t row = 0u; row < count; ++row)
        {
            if (rows.counts[row] != 0u)
            {
                rows.distances[row] = distances[offsets[row]];
            }
        }
    };
    if (radius == 0.0)
    {
        ovspatialquery_raycast_plan_desc_t desc{};
        desc.target                 = populationRef(population);
        desc.hit_mode               = mode;
        desc.precision              = precision;
        desc.device                 = kCudaDevice;
        desc.maximum_query_count    = count;
        desc.maximum_hits_per_query = perRow;
        desc.hit_capacity_per_slot  = count * perRow;
        desc.result_slot_count      = 1u;
        desc.result_fields          = fields;
        REQUIRE(ovspatialquery_create_raycast_plan(instance, &desc, &plan).status == OVSPATIALQUERY_API_SUCCESS);
        ovspatialquery_raycast_batch_t batch{};
        batch.ordinal.kind         = OVSPATIALQUERY_ORDINAL_HEAD;
        batch.query_count          = count;
        batch.origins              = &originColumn.value;
        batch.unit_directions      = &directionColumn.value;
        batch.max_distances        = &maximumColumn.value;
        batch.cuda_sync.has_stream = 1u;
        batch.cuda_sync.stream     = reinterpret_cast<uintptr_t>(stream);
        ovspatialquery_raycast_batch_reservation_t reservation{};
        operation = ovspatialquery_enqueue_raycast_plan(instance, plan, &batch, &reservation);
        REQUIRE(operation.status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(ovspatialquery_wait_op(instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
                    OVSPATIALQUERY_API_SUCCESS &&
                waited.state == OVSPATIALQUERY_OP_COMPLETED);
        ovspatialquery_raycast_batch_result_t result{};
        REQUIRE(ovspatialquery_fetch_raycast_batch(instance, operation.op_id, &result).status ==
                OVSPATIALQUERY_API_SUCCESS);
        download(result);
    }
    else
    {
        ovspatialquery_sphere_sweep_plan_desc_t desc{};
        desc.target                 = populationRef(population);
        desc.hit_mode               = mode;
        desc.precision              = precision;
        desc.radius                 = radius;
        desc.device                 = kCudaDevice;
        desc.maximum_query_count    = count;
        desc.maximum_hits_per_query = perRow;
        desc.hit_capacity_per_slot  = count * perRow;
        desc.result_slot_count      = 1u;
        desc.result_fields          = fields;
        REQUIRE(ovspatialquery_create_sphere_sweep_plan(instance, &desc, &plan).status == OVSPATIALQUERY_API_SUCCESS);
        ovspatialquery_sphere_sweep_batch_t batch{};
        batch.ordinal.kind         = OVSPATIALQUERY_ORDINAL_HEAD;
        batch.query_count          = count;
        batch.centers              = &originColumn.value;
        batch.unit_directions      = &directionColumn.value;
        batch.max_distances        = &maximumColumn.value;
        batch.cuda_sync.has_stream = 1u;
        batch.cuda_sync.stream     = reinterpret_cast<uintptr_t>(stream);
        ovspatialquery_sphere_sweep_batch_reservation_t reservation{};
        operation = ovspatialquery_enqueue_sphere_sweep_plan(instance, plan, &batch, &reservation);
        REQUIRE(operation.status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(ovspatialquery_wait_op(instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
                    OVSPATIALQUERY_API_SUCCESS &&
                waited.state == OVSPATIALQUERY_OP_COMPLETED);
        ovspatialquery_sphere_sweep_batch_result_t result{};
        REQUIRE(ovspatialquery_fetch_sphere_sweep_batch(instance, operation.op_id, &result).status ==
                OVSPATIALQUERY_API_SUCCESS);
        download(result);
    }
    REQUIRE(destroyPlanWhenRetired(instance, plan));
    cuda.free(deviceOrigins);
    cuda.free(deviceDirections);
    cuda.free(deviceMaximum);
}

// A sphere of radius 0.5 swept at a box with half extents 1 first touches an
// edge or a corner. Each path passes the feature at distance h (below the
// radius), so the centre comes within the radius where the path is
// sqrt(0.25 - h^2) before its closest approach, while it is still in that
// feature's region. Paths start 6, 24 and 96 from the closest approach.
void testCudaSphereSweepReachesBoxEdgesAndCorners()
{
    gPhase = "publish unit box";
    const ovspatialquery_device_budget_t budget{{kDLCUDA, 0}, UINT64_C(64) << 20};
    const ovspatialquery_instance_desc_t desc{4u, 1u, UINT64_C(64) << 20, &budget, 1u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    uint64_t   boxValue      = 82u;
    double     boxExtents[3] = {1.0, 1.0, 1.0};
    InputArray boxValues(&boxValue, 1, kDLUInt), boxExtentArray(boxExtents, 1, 3, kDLFloat);
    const ovspatialquery_box_definition_table_t boxes{&boxValues.value, &boxExtentArray.value};
    waitAndDestroy(instance, ovspatialquery_enqueue_box_registration(instance, &boxes));
    const ovspatialquery_population_handle_t population = createExplicitPopulation(instance, 1u);
    InstanceRows                             instances;
    instances.add(1u, OVSPATIALQUERY_GEOMETRY_BOX, boxValue, 0.0, 0.0, 0.0);
    const ovspatialquery_explicit_population_set_t set{1, instances.table(), nullptr, 0u};
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));

    // Edge x = -1, y = 1 along z, passed along (1, -1, 0.5) / 1.5; corner
    // (-1, 1, 1), passed along (2, -2, -1) / 3 with offsets along (1, 2, -2) / 3.
    // The closed form takes the float32 direction the plan reads.
    constexpr double kRadius = 0.5;
    struct Feature
    {
        double point[3], direction[3], offset[3];
        bool   edge;
    };
    const Feature features[2] = {
        {{-1.0, 1.0, 0.0}, {1.0 / 1.5, -1.0 / 1.5, 0.5 / 1.5}, {1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0), 0.0}, true},
        {{-1.0, 1.0, 1.0}, {2.0 / 3.0, -2.0 / 3.0, -1.0 / 3.0}, {1.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0}, false},
    };
    ClosestRows         rows;
    std::vector<double> expected;
    for (const Feature& feature : features)
    {
        for (const double start : {6.0, 24.0, 96.0})
        {
            for (const double h : {0.0, 0.1, 0.2})
            {
                long double origin[3] = {}, direction[3] = {};
                for (unsigned int axis = 0u; axis < 3u; ++axis)
                {
                    const float narrowed = float(feature.direction[axis]);
                    direction[axis]      = narrowed;
                    origin[axis] = feature.point[axis] - start * feature.direction[axis] + h * feature.offset[axis];
                    rows.origins.push_back(double(origin[axis]));
                    rows.directions.push_back(narrowed);
                }
                // An edge's distance ignores the coordinate along it.
                const unsigned int axes  = feature.edge ? 2u : 3u;
                long double        speed = 0.0L, along = 0.0L;
                for (unsigned int axis = 0u; axis < axes; ++axis)
                {
                    speed += direction[axis] * direction[axis];
                    along += (feature.point[axis] - (long double)rows.origins[rows.origins.size() - 3u + axis]) *
                             direction[axis];
                }
                const long double closest = along / speed;
                long double       gap     = 0.0L;
                for (unsigned int axis = 0u; axis < axes; ++axis)
                {
                    const long double delta = (long double)rows.origins[rows.origins.size() - 3u + axis] +
                                              closest * direction[axis] - feature.point[axis];
                    gap += delta * delta;
                }
                expected.push_back(double(closest - std::sqrt((kRadius * kRadius - gap) / speed)));
                rows.maximumDistances.push_back(2.0 * start);
            }
        }
    }
    gPhase = "CUDA sphere sweep at box edges and corners";
    runCudaClosest(instance, population, kRadius, rows);
    for (size_t row = 0u; row < expected.size(); ++row)
    {
        REQUIRE(rows.counts[row] == 1u && std::abs(rows.distances[row] - expected[row]) < 1.0e-9);
    }
    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_BOX, boxValue}).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

// far_sphere_sweep_scene.h's rows on CUDA plans: spheres meeting a triangle's
// face or a large cube's near face more than a thousand units from their start
// must report that contact, not miss it or a later one, and spheres grazing the
// triangle from just beyond their radius must miss it.
void testCudaSphereSweepsMeetFarFaces()
{
    gPhase = "publish far triangle and cube";
    const ovspatialquery_device_budget_t budget{{kDLCUDA, 0}, UINT64_C(64) << 20};
    const ovspatialquery_instance_desc_t desc{4u, 1u, UINT64_C(64) << 20, &budget, 1u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    FarSphereSweepScene scene(instance);
    gPhase = "CUDA sphere sweeps meeting far faces";
    const std::array cases{std::make_pair(scene.meshPopulation, &scene.meshRows),
                           std::make_pair(scene.cubePopulation, &scene.cubeRows)};
    for (const auto& [population, expected] : cases)
    {
        ClosestRows rows;
        rows.origins          = expected->origins;
        rows.directions       = expected->directions;
        rows.maximumDistances = expected->maximumDistances;
        runCudaClosest(instance, population, FarSphereSweepScene::kRadius, rows);
        for (size_t row = 0u; row < expected->expected.size(); ++row)
        {
            REQUIRE(rows.counts[row] == 1u &&
                    std::abs(rows.distances[row] - expected->expected[row]) <= 1.0e-9 * expected->expected[row]);
        }
    }
    ClosestRows grazing;
    grazing.origins          = scene.grazingRows.origins;
    grazing.directions       = scene.grazingRows.directions;
    grazing.maximumDistances = scene.grazingRows.maximumDistances;
    runCudaClosest(instance, scene.meshPopulation, FarSphereSweepScene::kRadius, grazing);
    for (size_t row = 0u; row < grazing.maximumDistances.size(); ++row)
        REQUIRE(grazing.counts[row] == 0u);
    scene.destroy();
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

// Signed distance from a local point to a box core (negative inside), a
// segment on local y, or a point; the shape's own radius is subtracted later.
long double signedCoreDistance(int32_t type, const long double point[3], const double size[3])
{
    if (type == OVSPATIALQUERY_GEOMETRY_BOX)
    {
        long double outside = 0.0L, inside = -std::numeric_limits<long double>::infinity();
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            const long double excess = std::fabs(point[axis]) - size[axis];
            outside += excess > 0.0L ? excess * excess : 0.0L;
            inside = std::fmax(inside, excess);
        }
        return outside > 0.0L ? std::sqrt(outside) : inside;
    }
    const long double along = type == OVSPATIALQUERY_GEOMETRY_CAPSULE
                                  ? std::fmax(-(long double)size[1], std::fmin((long double)size[1], point[1]))
                                  : 0.0L;
    return std::sqrt(point[0] * point[0] + (point[1] - along) * (point[1] - along) + point[2] * point[2]);
}

// CUDA raycast and sphere-sweep plans against a box, a capsule and a sphere,
// from 4, 16 and 96 away in random directions aimed near each shape, against
// an independent first contact: the path's distance to a convex shape is
// convex in t, so a ternary search finds its closest approach and a bisection
// before it the first touch. Rows within 1e-6 of grazing or of starting in
// contact are left out. Edges, corners, caps and far starts are where
// each solver's rounding shows.
void testCudaPlansFindTheFirstContactOfEveryPrimitive()
{
    gPhase = "publish box, capsule and sphere";
    const ovspatialquery_device_budget_t budget{{kDLCUDA, 0}, UINT64_C(64) << 20};
    const ovspatialquery_instance_desc_t desc{4u, 1u, UINT64_C(64) << 20, &budget, 1u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    struct Shape
    {
        int32_t type;
        double  size[3]; // box half extents; capsule half segment in [1]
        double  radius;  // capsule and sphere radius
        double  center[3];
    };
    const Shape shapes[3] = {
        {OVSPATIALQUERY_GEOMETRY_BOX, {1.0, 0.5, 0.75}, 0.0, {0.0, 0.0, 0.0}},
        {OVSPATIALQUERY_GEOMETRY_CAPSULE, {0.0, 1.0, 0.0}, 0.5, {8.0, 1.0, -2.0}},
        {OVSPATIALQUERY_GEOMETRY_SPHERE, {0.0, 0.0, 0.0}, 0.75, {-8.0, -1.0, 2.0}},
    };
    uint64_t   boxValue = 1u, capsuleValue = 2u, sphereValue = 3u;
    double     capsuleRadius = shapes[1].radius, halfSegment = shapes[1].size[1], sphereRadius = shapes[2].radius;
    InputArray boxValues(&boxValue, 1, kDLUInt), boxExtents(shapes[0].size, 1, 3, kDLFloat);
    InputArray capsuleValues(&capsuleValue, 1, kDLUInt), capsuleRadii(&capsuleRadius, 1, kDLFloat),
        halves(&halfSegment, 1, kDLFloat);
    InputArray sphereValues(&sphereValue, 1, kDLUInt), sphereRadii(&sphereRadius, 1, kDLFloat);
    const ovspatialquery_box_definition_table_t     boxes{&boxValues.value, &boxExtents.value};
    const ovspatialquery_capsule_definition_table_t capsules{&capsuleValues.value, &capsuleRadii.value, &halves.value};
    const ovspatialquery_sphere_definition_table_t  spheres{&sphereValues.value, &sphereRadii.value};
    waitAndDestroy(instance, ovspatialquery_enqueue_box_registration(instance, &boxes));
    waitAndDestroy(instance, ovspatialquery_enqueue_capsule_registration(instance, &capsules));
    waitAndDestroy(instance, ovspatialquery_enqueue_sphere_registration(instance, &spheres));
    const ovspatialquery_population_handle_t population = createExplicitPopulation(instance, 1u);
    InstanceRows                             instances;
    for (uint64_t index = 0u; index < 3u; ++index)
    {
        instances.add(index + 1u, shapes[index].type, index + 1u, shapes[index].center[0], shapes[index].center[1],
                      shapes[index].center[2]);
    }
    const ovspatialquery_explicit_population_set_t set{1, instances.table(), nullptr, 0u};
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));

    uint64_t   state   = UINT64_C(0x9e3779b97f4a7c15);
    const auto uniform = [&state](double low, double high)
    {
        state ^= state << 13u;
        state ^= state >> 7u;
        state ^= state << 17u;
        return low + (high - low) * double(state >> 11u) * 0x1.0p-53;
    };
    ClosestRows rows;
    for (const Shape& shape : shapes)
    {
        for (const double start : {4.0, 16.0, 96.0})
        {
            for (unsigned int sample = 0u; sample < 512u; ++sample)
            {
                double direction[3] = {}, length = 0.0;
                do
                {
                    length = 0.0;
                    for (double& value : direction)
                    {
                        value = uniform(-1.0, 1.0);
                        length += value * value;
                    }
                } while (length > 1.0 || length < 0.01);
                for (unsigned int axis = 0u; axis < 3u; ++axis)
                {
                    const float narrowed = float(direction[axis] / std::sqrt(length));
                    rows.directions.push_back(narrowed);
                    rows.origins.push_back(shape.center[axis] + uniform(-1.6, 1.6) - start * double(narrowed));
                }
                rows.maximumDistances.push_back(2.0 * start);
            }
        }
    }
    const size_t count   = rows.maximumDistances.size();
    size_t       checked = 0u, hits = 0u;
    for (const double radius : {0.0, 0.5})
    {
        gPhase = radius == 0.0 ? "CUDA raycast at every primitive" : "CUDA sphere sweep at every primitive";
        runCudaClosest(instance, population, radius, rows);
        for (size_t row = 0u; row < count; ++row)
        {
            long double first = std::numeric_limits<long double>::infinity();
            bool        clear = true;
            for (const Shape& shape : shapes)
            {
                const auto excess = [&](long double t)
                {
                    long double local[3] = {};
                    for (unsigned int axis = 0u; axis < 3u; ++axis)
                    {
                        local[axis] = (long double)rows.origins[row * 3u + axis] +
                                      t * (long double)rows.directions[row * 3u + axis] - shape.center[axis];
                    }
                    return signedCoreDistance(shape.type, local, shape.size) - shape.radius - radius;
                };
                long double low = 0.0L, high = rows.maximumDistances[row];
                for (unsigned int step = 0u; step < 200u; ++step)
                {
                    const long double left = low + (high - low) / 3.0L, right = high - (high - low) / 3.0L;
                    if (excess(left) < excess(right))
                    {
                        high = right;
                    }
                    else
                    {
                        low = left;
                    }
                }
                const long double closest = low;
                if (std::fabs(excess(closest)) < 1.0e-6L || std::fabs(excess(0.0L)) < 1.0e-6L ||
                    std::fabs(excess(rows.maximumDistances[row])) < 1.0e-6L)
                {
                    clear = false;
                    continue;
                }
                if (excess(closest) > 0.0L)
                {
                    continue;
                }
                low = 0.0L, high = closest;
                for (unsigned int step = 0u; step < 200u && excess(0.0L) > 0.0L; ++step)
                {
                    const long double middle = (low + high) * 0.5L;
                    if (excess(middle) > 0.0L)
                    {
                        low = middle;
                    }
                    else
                    {
                        high = middle;
                    }
                }
                first = std::fmin(first, excess(0.0L) > 0.0L ? high : 0.0L);
            }
            if (!clear)
            {
                continue;
            }
            ++checked;
            if (first == std::numeric_limits<long double>::infinity())
            {
                REQUIRE(rows.counts[row] == 0u);
                continue;
            }
            ++hits;
            REQUIRE(rows.counts[row] == 1u && std::fabs((long double)rows.distances[row] - first) < 1.0e-6L);
        }
    }
    // Leaving rows out must not hollow the check.
    REQUIRE(checked > count * 2u * 9u / 10u && hits > checked / 3u);
    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    for (const auto& definition : {ovspatialquery_geometry_ref_t{OVSPATIALQUERY_GEOMETRY_BOX, boxValue},
                                   ovspatialquery_geometry_ref_t{OVSPATIALQUERY_GEOMETRY_CAPSULE, capsuleValue},
                                   ovspatialquery_geometry_ref_t{OVSPATIALQUERY_GEOMETRY_SPHERE, sphereValue}})
    {
        REQUIRE(ovspatialquery_destroy_geometry_definition(instance, definition).status == OVSPATIALQUERY_API_SUCCESS);
    }
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

// A ray through an edge two triangles share meets the surface there, as a
// terrain or level grid seen along its diagonals does. Quads of side 1 and 2
// in y = 0, split on their diagonal x = z and placed up to 1000 away, take rows
// whose origins and float directions have equal x and z, so each crosses its
// quad on that edge at the distance where it reaches y = 0. CUDA plans of both
// exact precisions and every hit mode must report that contact, as the CPU's
// PhysX meshes do with their margin of 2^-22 of the barycentric range.
void testCudaRaycastHitsSharedMeshEdges()
{
    gPhase = "publish quads split on their diagonal";
    const ovspatialquery_device_budget_t budget{{kDLCUDA, 0}, UINT64_C(64) << 20};
    const ovspatialquery_instance_desc_t desc{4u, 1u, UINT64_C(64) << 20, &budget, 1u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    uint64_t meshValues[2]{91u, 92u}, vertexOffsets[2]{0u, 4u}, vertexCounts[2]{4u, 4u}, triangleOffsets[2]{0u, 2u},
        triangleCounts[2]{2u, 2u};
    const float sides[2]{1.0f, 2.0f};
    float       vertices[24]{};
    for (uint32_t mesh = 0u; mesh < 2u; ++mesh)
    {
        const float corners[4][2]{{0, 0}, {1, 0}, {1, 1}, {0, 1}}; // (x, z)
        for (uint32_t corner = 0u; corner < 4u; ++corner)
        {
            vertices[(mesh * 4u + corner) * 3u]      = sides[mesh] * corners[corner][0];
            vertices[(mesh * 4u + corner) * 3u + 2u] = sides[mesh] * corners[corner][1];
        }
    }
    uint32_t   triangles[12]{0u, 2u, 1u, 0u, 3u, 2u, 0u, 2u, 1u, 0u, 3u, 2u}; // front +y
    uint8_t    singleSided[2]{0u, 0u};
    InputArray values(meshValues, 2, kDLUInt), vertexOffsetArray(vertexOffsets, 2, kDLUInt),
        vertexCountArray(vertexCounts, 2, kDLUInt), triangleOffsetArray(triangleOffsets, 2, kDLUInt),
        triangleCountArray(triangleCounts, 2, kDLUInt), vertexData(vertices, 8, 3, kDLFloat),
        triangleData(triangles, 4, 3, kDLUInt), sidedness(singleSided, 2, kDLUInt);
    const ovspatialquery_triangle_mesh_definition_table_t meshes{&values.value,
                                                                 &vertexOffsetArray.value,
                                                                 &vertexCountArray.value,
                                                                 &triangleOffsetArray.value,
                                                                 &triangleCountArray.value,
                                                                 &vertexData.value,
                                                                 &triangleData.value,
                                                                 &sidedness.value,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr};
    waitAndDestroy(instance, ovspatialquery_enqueue_triangle_mesh_registration(instance, &meshes));
    struct Placement
    {
        uint32_t mesh;
        double   offset; // along x and z
    };
    const Placement                          placements[4]{{0u, 0.0}, {0u, 100.0}, {1u, -128.0}, {1u, 1000.0}};
    const ovspatialquery_population_handle_t population = createExplicitPopulation(instance, 1u);
    InstanceRows                             instances;
    for (uint64_t index = 0u; index < 4u; ++index)
    {
        instances.add(index + 1u, OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, meshValues[placements[index].mesh],
                      placements[index].offset, 0.0, placements[index].offset);
    }
    const ovspatialquery_explicit_population_set_t set{1, instances.table(), nullptr, 0u};
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));

    // Round crossing points and slopes round exactly; arbitrary ones are where the
    // two triangles' rounding of the edge disagrees.
    uint64_t   state   = UINT64_C(0x9e3779b97f4a7c15);
    const auto uniform = [&state](double low, double high)
    {
        state ^= state << 13u;
        state ^= state >> 7u;
        state ^= state << 17u;
        return low + (high - low) * double(state >> 11u) * 0x1.0p-53;
    };
    ClosestRows         rows;
    std::vector<double> expected;
    for (const Placement& placement : placements)
    {
        for (uint32_t sample = 0u; sample < 64u; ++sample)
        {
            const double along = placement.offset + sides[placement.mesh] * uniform(0.02, 0.98);
            const double slope = uniform(0.05, 1.5), back = uniform(1.0, 60.0);
            const double norm   = std::sqrt(2.0 + slope * slope);
            const float  across = float(1.0 / norm), down = float(-slope / norm);
            const double origin[3]{along - back * double(across), -back * double(down), along - back * double(across)};
            rows.origins.insert(rows.origins.end(), origin, origin + 3);
            rows.directions.insert(rows.directions.end(), {across, down, across});
            rows.maximumDistances.push_back(2.0 * back + 1.0);
            expected.push_back(origin[1] / -double(down));
        }
    }
    const size_t                                                     count = expected.size();
    static constexpr std::array<ovspatialquery_query_precision_t, 2> kPrecisions{OVSPATIALQUERY_QUERY_EXACT,
                                                                                 OVSPATIALQUERY_QUERY_EXACT_SINGLE};
    static constexpr std::array<ovspatialquery_hit_mode_t, 3>        kModes{
        OVSPATIALQUERY_HIT_MODE_CLOSEST, OVSPATIALQUERY_HIT_MODE_ANY, OVSPATIALQUERY_HIT_MODE_MULTIPLE};
    for (const ovspatialquery_query_precision_t precision : kPrecisions)
    {
        // Single precision may move the contact by its rounding.
        const double tolerance = precision == OVSPATIALQUERY_QUERY_EXACT ? 1.0e-9 : 1.0e-4;
        for (const ovspatialquery_hit_mode_t mode : kModes)
        {
            gPhase = "CUDA raycast through a shared mesh edge";
            runCudaClosest(instance, population, 0.0, rows, precision, mode);
            for (size_t row = 0u; row < count; ++row)
            {
                REQUIRE(rows.counts[row] >= 1u && (mode == OVSPATIALQUERY_HIT_MODE_MULTIPLE || rows.counts[row] == 1u));
                REQUIRE(std::abs(rows.distances[row] - expected[row]) <= tolerance * expected[row]);
            }
        }
    }
    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    for (const uint64_t value : meshValues)
    {
        REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, value})
                    .status == OVSPATIALQUERY_API_SUCCESS);
    }
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

} // namespace

// ADR-0051 on CUDA and CPU alike: one floor triangle (0,0,0) (4,0,0) (0,4,0), front +z,
// single-sided at the origin (object 1) and double-sided at x 100 (object 2). A unit
// sphere ignores a single-sided triangle with n . d > 0, a box of half extent 1 one with
// n . d >= 0; every other triangle stops a sweep at its first touch from either side.
// A ray meets a single-sided triangle only through its front face.
// ADR-0065: OVSPATIALQUERY_QUERY_MESH_BOTH_SIDES makes the single-sided copy answer as
// the double-sided one, per query and per batch launch, except that a back-face ray hit
// keeps its front normal there, as PhysX's eMESH_BOTH_SIDES does.
// Each case runs at both copies through CUDA plans and the synchronous CPU calls.
void testCudaSweepsFollowMeshSidedness()
{
    gPhase = "publish sidedness floors";
    const ovspatialquery_device_budget_t budget{{kDLCUDA, 0}, UINT64_C(64) << 20};
    const ovspatialquery_instance_desc_t desc{4u, 1u, UINT64_C(64) << 20, &budget, 1u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    uint64_t   meshValues[2]{81u, 82u}, zero[2]{}, vertexCounts[2]{3u, 3u}, triangleCounts[2]{1u, 1u};
    float      vertices[9]{0, 0, 0, 4, 0, 0, 0, 4, 0};
    uint32_t   triangles[3]{0u, 1u, 2u};
    uint8_t    sides[2]{0u, 1u};
    InputArray values(meshValues, 2, kDLUInt), vertexOffsets(zero, 2, kDLUInt), vertexCount(vertexCounts, 2, kDLUInt),
        triangleOffsets(zero, 2, kDLUInt), triangleCount(triangleCounts, 2, kDLUInt),
        vertexData(vertices, 3, 3, kDLFloat), triangleData(triangles, 1, 3, kDLUInt), sidedness(sides, 2, kDLUInt);
    const ovspatialquery_triangle_mesh_definition_table_t meshes{&values.value,
                                                                 &vertexOffsets.value,
                                                                 &vertexCount.value,
                                                                 &triangleOffsets.value,
                                                                 &triangleCount.value,
                                                                 &vertexData.value,
                                                                 &triangleData.value,
                                                                 &sidedness.value,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr};
    waitAndDestroy(instance, ovspatialquery_enqueue_triangle_mesh_registration(instance, &meshes));
    const ovspatialquery_population_handle_t population = createExplicitPopulation(instance, 1u);
    InstanceRows                             rows;
    rows.add(1u, OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, meshValues[0], 0.0, 0.0, 0.0);
    rows.add(2u, OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, meshValues[1], 100.0, 0.0, 0.0);
    const ovspatialquery_explicit_population_set_t set{1, rows.table(), nullptr, 0u};
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));

    // Distances single-sided then double-sided; NaN is no hit. The unit sphere falling past
    // the rim meets the edge x = 0 where (s - 1.5)^2 + (0.2 - s)^2 = 1 for s = t / sqrt(2);
    // the box's face x = s - 0.5 reaches it at s = 0.5. A ray from the case's center meets
    // the plane z = 0 inside the triangle only from above and from below.
    struct Case
    {
        double center[3];
        float  direction[3];
        double sphere[2];
        double box[2];
        double ray[2];
    };
    const double kNone   = std::numeric_limits<double>::quiet_NaN();
    const float  kHalf   = float(std::sqrt(0.5));
    const double kRim    = std::sqrt(2.0) * (3.4 - std::sqrt(1.24)) / 4;
    const Case   cases[] = {
        {{-1.5, 1, 0.2}, {kHalf, 0, -kHalf}, {kRim, kRim}, {std::sqrt(0.5), std::sqrt(0.5)}, {kNone, kNone}}, // rim
        {{1, 1, 3}, {0, 0, -1}, {2, 2}, {2, 2}, {3, 3}},                                                      // above
        {{1, 1, -3}, {0, 0, 1}, {kNone, 2}, {kNone, 2}, {kNone, 3}},                                          // below
        {{1, 1, 0.9}, {0, 0, 1}, {kNone, 0}, {kNone, 0}, {kNone, kNone}}, // resting on it, leaving
        {{1, 1, -0.3}, {1, 0, 0}, {0, 0}, {kNone, 0}, {kNone, kNone}},    // overlapping from behind, sliding
        {{1, 1, -0.3}, {0, 0, -1}, {0, 0}, {0, 0}, {kNone, kNone}},       // overlapping from behind, sinking
    };
    constexpr uint64_t            kRows = 2u * (sizeof(cases) / sizeof(cases[0]));
    std::array<double, 3 * kRows> centers{};
    std::array<float, 3 * kRows>  directions{};
    std::array<float, 4 * kRows>  rotations{};
    std::array<double, kRows>     maximums{};
    for (uint64_t row = 0; row < kRows; ++row)
    {
        const Case& c = cases[row / 2];
        for (int a = 0; a < 3; ++a)
        {
            centers[3 * row + a]    = c.center[a] + (a == 0 ? 100.0 * double(row % 2) : 0.0);
            directions[3 * row + a] = c.direction[a];
        }
        rotations[4 * row + 3] = 1.0f;
        maximums[row]          = 5.0;
    }
    enum class Query
    {
        sphere,
        box,
        ray,
    };
    // A both-sides row expects what the double-sided copy reports.
    const auto expected = [&](Query query, uint64_t row, bool bothSides)
    {
        const Case&    c    = cases[row / 2];
        const uint64_t side = bothSides ? 1u : row % 2;
        return query == Query::box ? c.box[side] : query == Query::sphere ? c.sphere[side] : c.ray[side];
    };
    const auto check =
        [&](Query query, uint64_t row, bool bothSides, bool hit, uint64_t object, double distance, const char* device)
    {
        const double want = expected(query, row, bothSides);
        if (std::isnan(want) ? hit : !(hit && object == row % 2 + 1 && std::abs(distance - want) <= 1e-6))
        {
            std::fprintf(stderr,
                         "%s query %d MESH_BOTH_SIDES %d row %llu: hit %d object %llu distance %.9g, expected %.9g\n",
                         device, int(query), int(bothSides), static_cast<unsigned long long>(row), int(hit),
                         static_cast<unsigned long long>(object), distance, want);
            REQUIRE(false);
        }
    };
    // Row 4 rises from below into the single-sided copy: with MESH_BOTH_SIDES its back-face
    // ray normal stays the front normal +z, while the double-sided copy's (row 5) faces the ray.
    const auto requireVerticalNormal = [](const float* normal, float z)
    { REQUIRE(std::abs(normal[0]) <= 1e-6f && std::abs(normal[1]) <= 1e-6f && std::abs(normal[2] - z) <= 1e-6f); };

    gPhase = "CPU sidedness sweeps";
    for (const uint32_t bothSides : {0u, uint32_t(OVSPATIALQUERY_QUERY_MESH_BOTH_SIDES)})
    {
        for (uint64_t row = 0; row < kRows; ++row)
        {
            ovspatialquery_query_common_t common{};
            common.target.target = populationRef(population);
            common.precision     = OVSPATIALQUERY_QUERY_EXACT;
            common.query_flags   = bothSides;
            ovspatialquery_sphere_sweep_desc_t sphere{common, {}};
            ovspatialquery_box_sweep_desc_t    box{common, {}};
            ovspatialquery_raycast_desc_t      ray{common, {}, OVSPATIALQUERY_MESH_HITS_ALL};
            for (int a = 0; a < 3; ++a)
            {
                sphere.sweep.center[a] = box.sweep.center[a] = ray.ray.origin[a] = centers[3 * row + a];
                sphere.sweep.unit_direction[a] = box.sweep.unit_direction[a] = ray.ray.unit_direction[a] =
                    directions[3 * row + a];
                box.sweep.half_extents[a] = 1.0;
            }
            sphere.sweep.radius        = 1.0;
            box.sweep.rotation_xyzw[3] = 1.0f;
            sphere.sweep.max_distance = box.sweep.max_distance = ray.ray.max_distance = 5.0;
            ovspatialquery_sphere_sweep_closest_result_t result{};
            REQUIRE(ovspatialquery_sweep_sphere_closest_sync(instance, &sphere, &result).status ==
                    OVSPATIALQUERY_API_SUCCESS);
            check(Query::sphere, row, bothSides != 0u, result.has_hit, result.hit.data.exact.identity.object_id.value,
                  result.hit.data.exact.distance, "CPU");
            REQUIRE(ovspatialquery_sweep_box_closest_sync(instance, &box, &result).status ==
                    OVSPATIALQUERY_API_SUCCESS);
            check(Query::box, row, bothSides != 0u, result.has_hit, result.hit.data.exact.identity.object_id.value,
                  result.hit.data.exact.distance, "CPU");
            ovspatialquery_raycast_single_result_t single{};
            REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &single).status == OVSPATIALQUERY_API_SUCCESS);
            check(Query::ray, row, bothSides != 0u, single.has_hit, single.hit.data.exact.identity.object_id.value,
                  single.hit.data.exact.distance, "CPU");
            if (bothSides != 0u && (row == 4u || row == 5u))
            {
                requireVerticalNormal(single.hit.data.exact.normal, row == 4u ? 1.0f : -1.0f);
            }
        }
    }

    CudaDriver               cuda(CudaDriver::Ownership::tracked);
    const CudaDriver::Stream stream           = cuda.createStream();
    const CudaDriver::Stream consumer         = cuda.createStream();
    const auto               deviceCenters    = upload(cuda, centers, stream);
    const auto               deviceDirections = upload(cuda, directions, stream);
    const auto               deviceRotations  = upload(cuda, rotations, stream);
    const auto               deviceMaximums   = upload(cuda, maximums, stream);
    InputArray centerColumn(devicePointer<double>(deviceCenters), int64_t(kRows), 3, kDLFloat, kCudaDevice);
    InputArray directionColumn(devicePointer<float>(deviceDirections), int64_t(kRows), 3, kDLFloat, kCudaDevice);
    InputArray rotationColumn(devicePointer<float>(deviceRotations), int64_t(kRows), 4, kDLFloat, kCudaDevice);
    InputArray maximumColumn(devicePointer<double>(deviceMaximums), int64_t(kRows), kDLFloat, kCudaDevice);
    // Each plan runs two launches, the second while the first's slot may still be retiring.
    const ovspatialquery_batch_result_field_t fields = static_cast<ovspatialquery_batch_result_field_t>(
        OVSPATIALQUERY_BATCH_RESULT_DISTANCE | OVSPATIALQUERY_BATCH_RESULT_GEOMETRY_IDENTITY |
        OVSPATIALQUERY_BATCH_RESULT_NORMAL);
    // Downloads one completed launch's rows and checks them against `query`'s expectations.
    const auto checkLaunch = [&](Query query, const ovspatialquery_batch_hit_ranges_t& ranges,
                                 const DLTensor& objectIds, const DLTensor& distanceColumn,
                                 const DLTensor& normalColumn, uint64_t capacity, bool flagged,
                                 ovspatialquery_cuda_event_t ready, ovspatialquery_op_id_t& operation)
    {
        cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(ready.value));
        std::vector<uint32_t> statuses(kRows);
        std::vector<uint64_t> offsets(kRows), counts(kRows), ids(capacity);
        std::vector<double>   distances(capacity);
        std::vector<float>    normals(3 * capacity);
        cuda.downloadAsync(statuses.data(), deviceAddress(ranges.query_status), 4 * kRows, consumer);
        cuda.downloadAsync(offsets.data(), deviceAddress(ranges.hit_offsets), 8 * kRows, consumer);
        cuda.downloadAsync(counts.data(), deviceAddress(ranges.hit_counts), 8 * kRows, consumer);
        cuda.downloadAsync(ids.data(), deviceAddress(objectIds), 8 * capacity, consumer);
        cuda.downloadAsync(distances.data(), deviceAddress(distanceColumn), 8 * capacity, consumer);
        cuda.downloadAsync(normals.data(), deviceAddress(normalColumn), 12 * capacity, consumer);
        retireAfterConsumer(cuda, consumer, instance, operation);
        for (uint64_t row = 0; row < kRows; ++row)
        {
            REQUIRE(statuses[row] == OVSPATIALQUERY_BATCH_ROW_SUCCESS && counts[row] <= 1u);
            const bool hit = counts[row] == 1u && offsets[row] < capacity;
            check(query, row, flagged, hit, hit ? ids[offsets[row]] : 0u, hit ? distances[offsets[row]] : 0.0, "CUDA");
        }
        if (query == Query::sphere && !flagged)
        {
            // A sphere sinking from behind the single-sided floor (row 10) keeps the normal from the
            // contact toward its center.
            requireVerticalNormal(normals.data() + 3 * offsets[10], -1.0f);
        }
        if (query == Query::ray && flagged)
        {
            requireVerticalNormal(normals.data() + 3 * offsets[4], 1.0f);
            requireVerticalNormal(normals.data() + 3 * offsets[5], -1.0f);
        }
    };
    for (const bool box : {false, true})
    {
        gPhase                                  = box ? "CUDA sidedness box sweeps" : "CUDA sidedness sphere sweeps";
        ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
        if (box)
        {
            ovspatialquery_box_sweep_plan_desc_t planDesc{};
            planDesc.target          = populationRef(population);
            planDesc.precision       = OVSPATIALQUERY_QUERY_EXACT;
            planDesc.half_extents[0] = planDesc.half_extents[1] = planDesc.half_extents[2] = 1.0;
            planDesc.device                                                                = {kDLCUDA, 0};
            planDesc.maximum_query_count                                                   = kRows;
            planDesc.result_slot_count                                                     = 2u;
            planDesc.hit_capacity_per_slot                                                 = kRows;
            planDesc.result_fields                                                         = fields;
            REQUIRE(ovspatialquery_create_box_sweep_plan(instance, &planDesc, &plan).status ==
                    OVSPATIALQUERY_API_SUCCESS);
        }
        else
        {
            ovspatialquery_sphere_sweep_plan_desc_t planDesc{};
            planDesc.target                = populationRef(population);
            planDesc.precision             = OVSPATIALQUERY_QUERY_EXACT;
            planDesc.radius                = 1.0;
            planDesc.device                = {kDLCUDA, 0};
            planDesc.maximum_query_count   = kRows;
            planDesc.result_slot_count     = 2u;
            planDesc.hit_capacity_per_slot = kRows;
            planDesc.result_fields         = fields;
            REQUIRE(ovspatialquery_create_sphere_sweep_plan(instance, &planDesc, &plan).status ==
                    OVSPATIALQUERY_API_SUCCESS);
        }
        for (const bool flagged : {false, true})
        {
            ovspatialquery_enqueue_result_t                 operation{};
            ovspatialquery_sphere_sweep_batch_reservation_t reservation{};
            const uint32_t flags = flagged ? uint32_t(OVSPATIALQUERY_QUERY_MESH_BOTH_SIDES) : 0u;
            if (box)
            {
                ovspatialquery_box_sweep_batch_t batch{};
                batch.ordinal.kind         = OVSPATIALQUERY_ORDINAL_HEAD;
                batch.query_count          = kRows;
                batch.centers              = &centerColumn.value;
                batch.rotations_xyzw       = &rotationColumn.value;
                batch.unit_directions      = &directionColumn.value;
                batch.max_distances        = &maximumColumn.value;
                batch.query_flags          = flags;
                batch.cuda_sync.has_stream = 1u;
                batch.cuda_sync.stream     = reinterpret_cast<uintptr_t>(stream);
                operation = ovspatialquery_enqueue_box_sweep_plan(instance, plan, &batch, &reservation);
            }
            else
            {
                ovspatialquery_sphere_sweep_batch_t batch{};
                batch.ordinal.kind         = OVSPATIALQUERY_ORDINAL_HEAD;
                batch.query_count          = kRows;
                batch.centers              = &centerColumn.value;
                batch.unit_directions      = &directionColumn.value;
                batch.max_distances        = &maximumColumn.value;
                batch.query_flags          = flags;
                batch.cuda_sync.has_stream = 1u;
                batch.cuda_sync.stream     = reinterpret_cast<uintptr_t>(stream);
                operation = ovspatialquery_enqueue_sphere_sweep_plan(instance, plan, &batch, &reservation);
            }
            REQUIRE(operation.status == OVSPATIALQUERY_API_SUCCESS);
            ovspatialquery_op_wait_result_t waited{};
            REQUIRE(
                ovspatialquery_wait_op(instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
                    OVSPATIALQUERY_API_SUCCESS &&
                waited.state == OVSPATIALQUERY_OP_COMPLETED);
            ovspatialquery_sphere_sweep_batch_result_t result{};
            REQUIRE((box ? ovspatialquery_fetch_box_sweep_batch(instance, operation.op_id, &result)
                         : ovspatialquery_fetch_sphere_sweep_batch(instance, operation.op_id, &result))
                        .status == OVSPATIALQUERY_API_SUCCESS);
            checkLaunch(box ? Query::box : Query::sphere, result.columns.hit_ranges, result.columns.object_ids,
                        result.columns.contact.exact.distances, result.columns.contact.exact.normals, kRows, flagged,
                        result.common.result_ready, operation.op_id);
        }
        REQUIRE(destroyPlanWhenRetired(instance, plan));
    }
    for (const ovspatialquery_hit_mode_t mode : {OVSPATIALQUERY_HIT_MODE_CLOSEST, OVSPATIALQUERY_HIT_MODE_MULTIPLE})
    {
        gPhase =
            mode == OVSPATIALQUERY_HIT_MODE_CLOSEST ? "CUDA sidedness closest rays" : "CUDA sidedness multiple rays";
        const bool                         multiple = mode == OVSPATIALQUERY_HIT_MODE_MULTIPLE;
        ovspatialquery_raycast_plan_desc_t planDesc{};
        planDesc.target                         = populationRef(population);
        planDesc.hit_mode                       = mode;
        planDesc.precision                      = OVSPATIALQUERY_QUERY_EXACT;
        planDesc.device                         = {kDLCUDA, 0};
        planDesc.maximum_query_count            = kRows;
        planDesc.maximum_hits_per_query         = multiple ? 2u : 1u;
        planDesc.hit_capacity_per_slot          = multiple ? 2u * kRows : kRows;
        planDesc.result_slot_count              = 2u;
        planDesc.result_fields                  = fields;
        planDesc.mesh_hits                      = OVSPATIALQUERY_MESH_HITS_ALL;
        ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
        REQUIRE(ovspatialquery_create_raycast_plan(instance, &planDesc, &plan).status == OVSPATIALQUERY_API_SUCCESS);
        for (const bool flagged : {false, true})
        {
            ovspatialquery_raycast_batch_t batch{};
            batch.ordinal.kind         = OVSPATIALQUERY_ORDINAL_HEAD;
            batch.query_count          = kRows;
            batch.origins              = &centerColumn.value;
            batch.unit_directions      = &directionColumn.value;
            batch.max_distances        = &maximumColumn.value;
            batch.query_flags          = flagged ? uint32_t(OVSPATIALQUERY_QUERY_MESH_BOTH_SIDES) | 2u : 0u;
            batch.cuda_sync.has_stream = 1u;
            batch.cuda_sync.stream     = reinterpret_cast<uintptr_t>(stream);
            ovspatialquery_raycast_batch_reservation_t reservation{};
            if (flagged)
            {
                // An unknown bit is refused before anything is queued.
                REQUIRE(ovspatialquery_enqueue_raycast_plan(instance, plan, &batch, &reservation).status ==
                        OVSPATIALQUERY_API_NOT_SUPPORTED);
                batch.query_flags = OVSPATIALQUERY_QUERY_MESH_BOTH_SIDES;
            }
            ovspatialquery_enqueue_result_t operation =
                ovspatialquery_enqueue_raycast_plan(instance, plan, &batch, &reservation);
            REQUIRE(operation.status == OVSPATIALQUERY_API_SUCCESS);
            ovspatialquery_op_wait_result_t waited{};
            REQUIRE(
                ovspatialquery_wait_op(instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
                    OVSPATIALQUERY_API_SUCCESS &&
                waited.state == OVSPATIALQUERY_OP_COMPLETED);
            ovspatialquery_raycast_batch_result_t result{};
            REQUIRE(ovspatialquery_fetch_raycast_batch(instance, operation.op_id, &result).status ==
                    OVSPATIALQUERY_API_SUCCESS);
            checkLaunch(Query::ray, result.columns.hit_ranges, result.columns.object_ids,
                        result.columns.contact.exact.distances, result.columns.contact.exact.normals,
                        planDesc.hit_capacity_per_slot, flagged, result.common.result_ready, operation.op_id);
        }
        REQUIRE(destroyPlanWhenRetired(instance, plan));
    }
    {
        // A ray with a contact tolerance runs the kernel's sphere path; with MESH_BOTH_SIDES its
        // back-face normal still follows the ray rule: rising from below (rows 4 and 5) it
        // reports the single-sided floor's front normal +z and the double-sided floor's -z,
        // touching both one tolerance early. The CPU synchronous call agrees.
        gPhase = "CUDA sidedness rays with a contact tolerance";
        ovspatialquery_raycast_plan_desc_t planDesc{};
        planDesc.target                         = populationRef(population);
        planDesc.hit_mode                       = OVSPATIALQUERY_HIT_MODE_CLOSEST;
        planDesc.precision                      = OVSPATIALQUERY_QUERY_EXACT;
        planDesc.device                         = {kDLCUDA, 0};
        planDesc.maximum_query_count            = kRows;
        planDesc.maximum_hits_per_query         = 1u;
        planDesc.hit_capacity_per_slot          = kRows;
        planDesc.result_slot_count              = 1u;
        planDesc.result_fields                  = fields;
        planDesc.contact_tolerance              = 0.1;
        ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
        REQUIRE(ovspatialquery_create_raycast_plan(instance, &planDesc, &plan).status == OVSPATIALQUERY_API_SUCCESS);
        ovspatialquery_raycast_batch_t batch{};
        batch.ordinal.kind         = OVSPATIALQUERY_ORDINAL_HEAD;
        batch.query_count          = kRows;
        batch.origins              = &centerColumn.value;
        batch.unit_directions      = &directionColumn.value;
        batch.max_distances        = &maximumColumn.value;
        batch.query_flags          = OVSPATIALQUERY_QUERY_MESH_BOTH_SIDES;
        batch.cuda_sync.has_stream = 1u;
        batch.cuda_sync.stream     = reinterpret_cast<uintptr_t>(stream);
        ovspatialquery_raycast_batch_reservation_t reservation{};
        ovspatialquery_enqueue_result_t            operation =
            ovspatialquery_enqueue_raycast_plan(instance, plan, &batch, &reservation);
        REQUIRE(operation.status == OVSPATIALQUERY_API_SUCCESS);
        ovspatialquery_op_wait_result_t waited{};
        REQUIRE(ovspatialquery_wait_op(instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
                    OVSPATIALQUERY_API_SUCCESS &&
                waited.state == OVSPATIALQUERY_OP_COMPLETED);
        ovspatialquery_raycast_batch_result_t result{};
        REQUIRE(ovspatialquery_fetch_raycast_batch(instance, operation.op_id, &result).status ==
                OVSPATIALQUERY_API_SUCCESS);
        cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(result.common.result_ready.value));
        std::vector<uint64_t> offsets(kRows), counts(kRows);
        std::vector<double>   distances(kRows);
        std::vector<float>    normals(3 * kRows);
        const auto&           ranges = result.columns.hit_ranges;
        cuda.downloadAsync(offsets.data(), deviceAddress(ranges.hit_offsets), 8 * kRows, consumer);
        cuda.downloadAsync(counts.data(), deviceAddress(ranges.hit_counts), 8 * kRows, consumer);
        cuda.downloadAsync(distances.data(), deviceAddress(result.columns.contact.exact.distances), 8 * kRows,
                           consumer);
        cuda.downloadAsync(normals.data(), deviceAddress(result.columns.contact.exact.normals), 12 * kRows, consumer);
        retireAfterConsumer(cuda, consumer, instance, operation.op_id);
        for (const uint64_t row : {uint64_t(4), uint64_t(5)})
        {
            REQUIRE(counts[row] == 1u && offsets[row] < kRows);
            REQUIRE(std::abs(distances[offsets[row]] - 2.9) <= 1e-6);
            requireVerticalNormal(normals.data() + 3 * offsets[row], row == 4u ? 1.0f : -1.0f);

            ovspatialquery_query_common_t common{};
            common.target.target     = populationRef(population);
            common.precision         = OVSPATIALQUERY_QUERY_EXACT;
            common.query_flags       = OVSPATIALQUERY_QUERY_MESH_BOTH_SIDES;
            common.contact_tolerance = 0.1;
            ovspatialquery_raycast_desc_t ray{common, {}, OVSPATIALQUERY_MESH_HITS_ALL};
            for (int a = 0; a < 3; ++a)
            {
                ray.ray.origin[a]         = centers[3 * row + a];
                ray.ray.unit_direction[a] = directions[3 * row + a];
            }
            ray.ray.max_distance = 5.0;
            ovspatialquery_raycast_single_result_t single{};
            REQUIRE(ovspatialquery_raycast_closest_sync(instance, &ray, &single).status == OVSPATIALQUERY_API_SUCCESS);
            REQUIRE(single.has_hit == 1u && std::abs(single.hit.data.exact.distance - 2.9) <= 1e-6);
            requireVerticalNormal(single.hit.data.exact.normal, row == 4u ? 1.0f : -1.0f);
        }
        REQUIRE(destroyPlanWhenRetired(instance, plan));
    }
    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    for (const uint64_t value : meshValues)
        REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, value})
                    .status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(transferredEventDestroysWereInvoked(9u));
}

// Promise: on a CUDA device the instance configured, ovspatialquery_query_capability
// answers SUCCESS where this file's other tests create every family at every precision,
// and refuses a device the driver lacks with the diagnostic plan creation gives it.
void testCapabilityQueryAgreesWithCudaPlanCreation()
{
    ExactMeshFixture fixture;
    gPhase = "query every capability on the configured CUDA device";
    for (uint32_t key = OVSPATIALQUERY_CAPABILITY_RAYCAST_PLAN; key <= OVSPATIALQUERY_CAPABILITY_PAIR_REPORTER; ++key)
    {
        REQUIRE(ovspatialquery_query_capability(fixture.instance, {kDLCUDA, 0},
                                                static_cast<ovspatialquery_capability_t>(key))
                    .status == OVSPATIALQUERY_API_SUCCESS);
    }
    gPhase = "a configured exact raycast plan agrees";
    ovspatialquery_raycast_plan_desc_t ray{};
    ray.target                              = populationRef(fixture.population);
    ray.hit_mode                            = OVSPATIALQUERY_HIT_MODE_CLOSEST;
    ray.precision                           = OVSPATIALQUERY_QUERY_EXACT;
    ray.device                              = {kDLCUDA, 0};
    ray.maximum_query_count                 = 1u;
    ray.maximum_hits_per_query              = 1u;
    ray.hit_capacity_per_slot               = 1u;
    ray.result_slot_count                   = 1u;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_raycast_plan(fixture.instance, &ray, &plan).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_batch_plan(fixture.instance, plan).status == OVSPATIALQUERY_API_SUCCESS);

    gPhase     = "a device the driver does not have is refused with the plan's diagnostic";
    ray.device = {kDLCUDA, 1 << 20};
    REQUIRE(ovspatialquery_create_raycast_plan(fixture.instance, &ray, &plan).status ==
            OVSPATIALQUERY_API_NOT_SUPPORTED);
    const ovspatialquery_string_t planError = ovspatialquery_get_last_error();
    const std::string             planDiagnostic(planError.ptr, planError.length);
    REQUIRE(ovspatialquery_query_capability(fixture.instance, ray.device, OVSPATIALQUERY_CAPABILITY_RAYCAST_PLAN_EXACT)
                .status == OVSPATIALQUERY_API_NOT_SUPPORTED);
    const ovspatialquery_string_t queryError = ovspatialquery_get_last_error();
    REQUIRE(!planDiagnostic.empty() && std::string(queryError.ptr, queryError.length) == planDiagnostic);
    fixture.close();
}

// Materials on CUDA plan columns (ADR-0066): an object's material on every
// precision, and a triangle's own one on exact mesh contacts, as the CPU path
// reports them. Two triangles in the plane x = 5: triangle 0 (y < 0) carries
// material 2000, triangle 1 (y > 0) zero, so it reports its instance's 3000;
// the unit sphere at (0, 10, 0) carries 1000.
void testCudaColumnsCarryMaterials()
{
    gPhase = "create material scene";
    const ovspatialquery_device_budget_t budget{{kDLCUDA, 0}, UINT64_C(64) << 20};
    const ovspatialquery_instance_desc_t desc{8u, 1u, UINT64_C(64) << 20, &budget, 1u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    registerSphere(instance, 1u, 1.0);
    uint64_t meshValue     = 2u;
    uint64_t zero          = 0u;
    uint64_t vertexCount   = 6u;
    uint64_t triangleCount = 3u;
    float    vertices[18]  = {5.0f, -3.0f, -1.0f, 5.0f, -1.0f, -1.0f, 5.0f, -2.0f, 1.0f,
                              5.0f, 1.0f,  -1.0f, 5.0f, 3.0f,  -1.0f, 5.0f, 2.0f,  1.0f};
    // Triangle 1 repeats a corner, so registration drops it (ADR-0073); triangle 2 keeps
    // its own ordinal and so its material entry, zero, not triangle 1's 9999.
    uint32_t                                              triangles[9]     = {0u, 1u, 2u, 3u, 3u, 4u, 3u, 4u, 5u};
    uint8_t                                               doubleSided      = 1u;
    uint64_t                                              faceMaterials[3] = {2000u, 9999u, 0u};
    InputArray                                            values(&meshValue, 1, kDLUInt);
    InputArray                                            offsets(&zero, 1, kDLUInt);
    InputArray                                            vertexCounts(&vertexCount, 1, kDLUInt);
    InputArray                                            triangleCounts(&triangleCount, 1, kDLUInt);
    InputArray                                            vertexData(vertices, 6, 3, kDLFloat);
    InputArray                                            triangleData(triangles, 3, 3, kDLUInt);
    InputArray                                            sidedness(&doubleSided, 1, kDLUInt);
    InputArray                                            materialData(faceMaterials, 3, kDLUInt);
    const ovspatialquery_triangle_mesh_definition_table_t meshes{
        &values.value,     &offsets.value,      &vertexCounts.value, &offsets.value,      &triangleCounts.value,
        &vertexData.value, &triangleData.value, &sidedness.value,    &materialData.value, nullptr,
        nullptr,
    };
    waitAndDestroy(instance, ovspatialquery_enqueue_triangle_mesh_registration(instance, &meshes));
    uint64_t dropped = 0u;
    REQUIRE(ovspatialquery_get_dropped_triangle_count(instance, meshValue, &dropped).status ==
                OVSPATIALQUERY_API_SUCCESS &&
            dropped == 1u);
    // The sphere's value names no triangle-mesh definition; the output keeps its value.
    REQUIRE(ovspatialquery_get_dropped_triangle_count(instance, 1u, &dropped).status ==
                OVSPATIALQUERY_API_INVALID_ARGUMENT &&
            dropped == 1u);
    const ovspatialquery_population_handle_t population = createExplicitPopulation(instance, 1u);
    {
        InstanceRows rows;
        rows.add(10u, OVSPATIALQUERY_GEOMETRY_SPHERE, 1u, 0.0, 10.0, 0.0);
        rows.add(20u, OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, meshValue, 0.0, 0.0, 0.0);
        rows.materials = {1000u, 3000u};
        ovspatialquery_explicit_population_set_t set{};
        set.target_ordinal = 1;
        set.instances      = rows.table();
        waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));
    }

    CudaDriver                      cuda(CudaDriver::Ownership::tracked);
    const CudaDriver::Stream        producer            = cuda.createStream();
    const CudaDriver::Stream        consumer            = cuda.createStream();
    double                          origins[3][3]       = {{0.0, 10.0, -5.0}, {0.0, -2.0, 0.0}, {0.0, 2.0, 0.0}};
    float                           directions[3][3]    = {{0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}};
    double                          maximumDistances[3] = {20.0, 20.0, 20.0};
    const CudaDriver::DevicePointer deviceOrigins       = upload(cuda, origins, producer);
    const CudaDriver::DevicePointer deviceDirections    = upload(cuda, directions, producer);
    const CudaDriver::DevicePointer deviceMaximums      = upload(cuda, maximumDistances, producer);
    InputArray                      originColumn(devicePointer<double>(deviceOrigins), 3, 3, kDLFloat, kCudaDevice);
    InputArray directionColumn(devicePointer<float>(deviceDirections), 3, 3, kDLFloat, kCudaDevice);
    InputArray maximumColumn(devicePointer<double>(deviceMaximums), 3, kDLFloat, kCudaDevice);

    // One material per ray: the row's single stored hit.
    const auto rayMaterials = [&](ovspatialquery_query_precision_t precision, ovspatialquery_hit_mode_t mode)
    {
        ovspatialquery_raycast_plan_desc_t planDesc{};
        planDesc.target                         = populationRef(population);
        planDesc.hit_mode                       = mode;
        planDesc.precision                      = precision;
        planDesc.device                         = kCudaDevice;
        planDesc.maximum_query_count            = 3u;
        planDesc.maximum_hits_per_query         = mode == OVSPATIALQUERY_HIT_MODE_MULTIPLE ? 2u : 1u;
        planDesc.hit_capacity_per_slot          = 3u * planDesc.maximum_hits_per_query;
        planDesc.result_slot_count              = 1u;
        planDesc.result_fields                  = OVSPATIALQUERY_BATCH_RESULT_MATERIAL;
        ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
        REQUIRE(ovspatialquery_create_raycast_plan(instance, &planDesc, &plan).status == OVSPATIALQUERY_API_SUCCESS);
        ovspatialquery_raycast_batch_t batch{};
        batch.ordinal.kind         = OVSPATIALQUERY_ORDINAL_HEAD;
        batch.query_count          = 3u;
        batch.origins              = &originColumn.value;
        batch.unit_directions      = &directionColumn.value;
        batch.max_distances        = &maximumColumn.value;
        batch.cuda_sync.has_stream = 1u;
        batch.cuda_sync.stream     = reinterpret_cast<uintptr_t>(producer);
        ovspatialquery_raycast_batch_reservation_t reservation{};
        ovspatialquery_enqueue_result_t            operation =
            ovspatialquery_enqueue_raycast_plan(instance, plan, &batch, &reservation);
        REQUIRE(operation.status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(reservation.columns.material_ids.shape != nullptr);
        ovspatialquery_op_wait_result_t waited{};
        REQUIRE(ovspatialquery_wait_op(instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
                OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(waited.state == OVSPATIALQUERY_OP_COMPLETED);
        ovspatialquery_raycast_batch_result_t result{};
        REQUIRE(ovspatialquery_fetch_raycast_batch(instance, operation.op_id, &result).status ==
                OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(result.columns.material_ids.shape[0] == 3 && result.columns.material_ids.dtype.bits == 64u);
        cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(result.common.result_ready.value));
        uint64_t hitOffsets[3] = {};
        uint64_t hitCounts[3]  = {};
        uint64_t materials[3]  = {};
        cuda.downloadAsync(hitOffsets, deviceAddress(result.columns.hit_ranges.hit_offsets), sizeof(hitOffsets),
                           consumer);
        cuda.downloadAsync(hitCounts, deviceAddress(result.columns.hit_ranges.hit_counts), sizeof(hitCounts), consumer);
        cuda.downloadAsync(materials, deviceAddress(result.columns.material_ids), sizeof(materials), consumer);
        retireAfterConsumer(cuda, consumer, instance, operation.op_id);
        REQUIRE(destroyPlanWhenRetired(instance, plan));
        std::array<uint64_t, 3u> perRow{};
        for (uint32_t row = 0u; row < 3u; ++row)
        {
            REQUIRE(hitCounts[row] == 1u && hitOffsets[row] < 3u);
            perRow[row] = materials[hitOffsets[row]];
        }
        return perRow;
    };
    gPhase = "CUDA exact closest raycast materials";
    REQUIRE((rayMaterials(OVSPATIALQUERY_QUERY_EXACT, OVSPATIALQUERY_HIT_MODE_CLOSEST) ==
             std::array<uint64_t, 3u>{1000u, 2000u, 3000u}));
    gPhase = "CUDA exact multiple raycast materials";
    REQUIRE((rayMaterials(OVSPATIALQUERY_QUERY_EXACT, OVSPATIALQUERY_HIT_MODE_MULTIPLE) ==
             std::array<uint64_t, 3u>{1000u, 2000u, 3000u}));
    gPhase = "CUDA bounds-candidate raycast materials";
    REQUIRE((rayMaterials(OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE, OVSPATIALQUERY_HIT_MODE_CLOSEST) ==
             std::array<uint64_t, 3u>{1000u, 3000u, 3000u}));

    gPhase = "CUDA overlap materials";
    ovspatialquery_overlap_plan_desc_t overlapDesc{};
    overlapDesc.target                      = populationRef(population);
    overlapDesc.hit_mode                    = OVSPATIALQUERY_HIT_MODE_MULTIPLE;
    overlapDesc.precision                   = OVSPATIALQUERY_QUERY_EXACT;
    overlapDesc.shape.kind                  = OVSPATIALQUERY_OVERLAP_SPHERE;
    overlapDesc.shape.data.sphere.radius    = 0.5;
    overlapDesc.device                      = kCudaDevice;
    overlapDesc.maximum_query_count         = 1u;
    overlapDesc.maximum_hits_per_query      = 2u;
    overlapDesc.hit_capacity_per_slot       = 2u;
    overlapDesc.result_slot_count           = 1u;
    overlapDesc.result_fields               = OVSPATIALQUERY_BATCH_RESULT_MATERIAL;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_overlap_plan(instance, &overlapDesc, &plan).status == OVSPATIALQUERY_API_SUCCESS);
    double                          center[3]    = {0.0, 10.0, 0.0};
    const CudaDriver::DevicePointer deviceCenter = upload(cuda, center, producer);
    InputArray                      centerColumn(devicePointer<double>(deviceCenter), 1, 3, kDLFloat, kCudaDevice);
    ovspatialquery_overlap_batch_t  overlapBatch{};
    overlapBatch.ordinal.kind         = OVSPATIALQUERY_ORDINAL_HEAD;
    overlapBatch.query_count          = 1u;
    overlapBatch.translations         = &centerColumn.value;
    overlapBatch.cuda_sync.has_stream = 1u;
    overlapBatch.cuda_sync.stream     = reinterpret_cast<uintptr_t>(producer);
    ovspatialquery_overlap_batch_reservation_t overlapReservation{};
    ovspatialquery_enqueue_result_t            operation =
        ovspatialquery_enqueue_overlap_plan(instance, plan, &overlapBatch, &overlapReservation);
    REQUIRE(operation.status == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_op_wait_result_t waited{};
    REQUIRE(ovspatialquery_wait_op(instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
            OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_overlap_batch_result_t overlapResult{};
    REQUIRE(ovspatialquery_fetch_overlap_batch(instance, operation.op_id, &overlapResult).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(overlapResult.columns.material_ids.shape[0] == 1);
    cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(overlapResult.common.result_ready.value));
    uint64_t overlapMaterial = 0u;
    cuda.downloadAsync(&overlapMaterial, deviceAddress(overlapResult.columns.material_ids), sizeof(overlapMaterial),
                       consumer);
    retireAfterConsumer(cuda, consumer, instance, operation.op_id);
    REQUIRE(destroyPlanWhenRetired(instance, plan));
    REQUIRE(overlapMaterial == 1000u);
    // Three raycasts and the overlap each handed the library a consumer_done event.
    REQUIRE(transferredEventDestroysWereInvoked(4u));

    gPhase = "release material scene";
    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_SPHERE, 1u}).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, meshValue})
                .status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

// CUDA plan rows on the polygon mesh (public_api_inputs.h) report the caller's
// source face, corners and the face's material on every exact raycast mode and
// precision and on an exact sphere sweep, as the CPU path does: face 7 with
// material 2000 for the quad's two triangles, face 3 with the object's 3000 for
// the pentagon's three.
void testCudaHitsReportCallerSourceFaces()
{
    gPhase = "create polygon scene";
    const ovspatialquery_device_budget_t budget{{kDLCUDA, 0}, UINT64_C(64) << 20};
    const ovspatialquery_instance_desc_t desc{8u, 1u, UINT64_C(64) << 20, &budget, 1u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    PolygonMesh mesh;
    mesh.value                                                  = 2u;
    const ovspatialquery_triangle_mesh_definition_table_t table = mesh.table();
    waitAndDestroy(instance, ovspatialquery_enqueue_triangle_mesh_registration(instance, &table));
    const ovspatialquery_population_handle_t population = createExplicitPopulation(instance, 1u);
    {
        InstanceRows rows;
        rows.add(20u, OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, mesh.value, 0.0, 0.0, 0.0);
        rows.materials = {3000u};
        ovspatialquery_explicit_population_set_t set{};
        set.target_ordinal = 1;
        set.instances      = rows.table();
        waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));
    }

    constexpr uint32_t       kRows = PolygonMesh::kTriangleCount;
    CudaDriver               cuda(CudaDriver::Ownership::tracked);
    const CudaDriver::Stream producer = cuda.createStream();
    const CudaDriver::Stream consumer = cuda.createStream();
    double                   origins[kRows][3]{};
    float                    directions[kRows][3]{};
    double                   maximums[kRows]{};
    for (uint32_t row = 0u; row < kRows; ++row)
    {
        origins[row][1]    = kPolygonRayYz[row][0];
        origins[row][2]    = kPolygonRayYz[row][1];
        directions[row][0] = 1.0f;
        maximums[row]      = 20.0;
    }
    const CudaDriver::DevicePointer deviceOrigins    = upload(cuda, origins, producer);
    const CudaDriver::DevicePointer deviceDirections = upload(cuda, directions, producer);
    const CudaDriver::DevicePointer deviceMaximums   = upload(cuda, maximums, producer);
    InputArray                      originColumn(devicePointer<double>(deviceOrigins), kRows, 3, kDLFloat, kCudaDevice);
    InputArray     directionColumn(devicePointer<float>(deviceDirections), kRows, 3, kDLFloat, kCudaDevice);
    InputArray     maximumColumn(devicePointer<double>(deviceMaximums), kRows, kDLFloat, kCudaDevice);
    const uint32_t fields = OVSPATIALQUERY_BATCH_RESULT_SOURCE_FEATURE | OVSPATIALQUERY_BATCH_RESULT_MATERIAL;

    // Each row's single stored hit reports the face, corners and material of the
    // triangle the row crosses.
    const auto requireRows = [&](const ovspatialquery_batch_hit_ranges_t&      ranges,
                                 const ovspatialquery_exact_feature_columns_t& features, const DLTensor& materialColumn,
                                 uintptr_t ready, ovspatialquery_op_id_t& operation)
    {
        cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(ready));
        uint64_t hitOffsets[kRows] = {};
        uint64_t hitCounts[kRows]  = {};
        uint64_t faces[kRows]      = {};
        uint32_t corners[kRows][3] = {};
        uint64_t materials[kRows]  = {};
        cuda.downloadAsync(hitOffsets, deviceAddress(ranges.hit_offsets), sizeof(hitOffsets), consumer);
        cuda.downloadAsync(hitCounts, deviceAddress(ranges.hit_counts), sizeof(hitCounts), consumer);
        cuda.downloadAsync(faces, deviceAddress(features.source_face_indices), sizeof(faces), consumer);
        cuda.downloadAsync(corners, deviceAddress(features.source_corner_indices), sizeof(corners), consumer);
        cuda.downloadAsync(materials, deviceAddress(materialColumn), sizeof(materials), consumer);
        retireAfterConsumer(cuda, consumer, instance, operation);
        for (uint32_t row = 0u; row < kRows; ++row)
        {
            REQUIRE(hitCounts[row] == 1u && hitOffsets[row] < kRows);
            const uint64_t hit = hitOffsets[row];
            REQUIRE(faces[hit] == mesh.faces[row]);
            REQUIRE(std::memcmp(corners[hit], mesh.corners[row], sizeof(corners[hit])) == 0);
            REQUIRE(materials[hit] == (mesh.faces[row] == 7u ? 2000u : 3000u));
        }
    };

    const std::pair<ovspatialquery_query_precision_t, ovspatialquery_hit_mode_t> modes[] = {
        {OVSPATIALQUERY_QUERY_EXACT, OVSPATIALQUERY_HIT_MODE_CLOSEST},
        {OVSPATIALQUERY_QUERY_EXACT, OVSPATIALQUERY_HIT_MODE_ANY},
        {OVSPATIALQUERY_QUERY_EXACT, OVSPATIALQUERY_HIT_MODE_MULTIPLE},
        {OVSPATIALQUERY_QUERY_EXACT_SINGLE, OVSPATIALQUERY_HIT_MODE_CLOSEST}};
    for (const auto& mode : modes)
    {
        gPhase = "CUDA raycast source faces";
        ovspatialquery_raycast_plan_desc_t planDesc{};
        planDesc.target                         = populationRef(population);
        planDesc.precision                      = mode.first;
        planDesc.hit_mode                       = mode.second;
        planDesc.device                         = kCudaDevice;
        planDesc.maximum_query_count            = kRows;
        planDesc.maximum_hits_per_query         = mode.second == OVSPATIALQUERY_HIT_MODE_MULTIPLE ? 2u : 1u;
        planDesc.hit_capacity_per_slot          = kRows * planDesc.maximum_hits_per_query;
        planDesc.result_slot_count              = 1u;
        planDesc.result_fields                  = fields;
        ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
        REQUIRE(ovspatialquery_create_raycast_plan(instance, &planDesc, &plan).status == OVSPATIALQUERY_API_SUCCESS);
        ovspatialquery_raycast_batch_t batch{};
        batch.ordinal.kind         = OVSPATIALQUERY_ORDINAL_HEAD;
        batch.query_count          = kRows;
        batch.origins              = &originColumn.value;
        batch.unit_directions      = &directionColumn.value;
        batch.max_distances        = &maximumColumn.value;
        batch.cuda_sync.has_stream = 1u;
        batch.cuda_sync.stream     = reinterpret_cast<uintptr_t>(producer);
        ovspatialquery_raycast_batch_reservation_t reservation{};
        ovspatialquery_enqueue_result_t            operation =
            ovspatialquery_enqueue_raycast_plan(instance, plan, &batch, &reservation);
        REQUIRE(operation.status == OVSPATIALQUERY_API_SUCCESS);
        ovspatialquery_op_wait_result_t waited{};
        REQUIRE(ovspatialquery_wait_op(instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
                OVSPATIALQUERY_API_SUCCESS);
        ovspatialquery_raycast_batch_result_t result{};
        REQUIRE(ovspatialquery_fetch_raycast_batch(instance, operation.op_id, &result).status ==
                OVSPATIALQUERY_API_SUCCESS);
        requireRows(result.columns.hit_ranges, result.columns.contact.exact.features, result.columns.material_ids,
                    result.common.result_ready.value, operation.op_id);
        REQUIRE(destroyPlanWhenRetired(instance, plan));
    }

    gPhase = "CUDA sphere sweep source faces";
    ovspatialquery_sphere_sweep_plan_desc_t sweepDesc{};
    sweepDesc.target                        = populationRef(population);
    sweepDesc.precision                     = OVSPATIALQUERY_QUERY_EXACT;
    sweepDesc.radius                        = 0.05;
    sweepDesc.device                        = kCudaDevice;
    sweepDesc.maximum_query_count           = kRows;
    sweepDesc.result_slot_count             = 1u;
    sweepDesc.result_fields                 = fields;
    sweepDesc.hit_capacity_per_slot         = kRows;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    REQUIRE(ovspatialquery_create_sphere_sweep_plan(instance, &sweepDesc, &plan).status == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_sphere_sweep_batch_t sweep{};
    sweep.ordinal.kind         = OVSPATIALQUERY_ORDINAL_HEAD;
    sweep.query_count          = kRows;
    sweep.centers              = &originColumn.value;
    sweep.unit_directions      = &directionColumn.value;
    sweep.max_distances        = &maximumColumn.value;
    sweep.cuda_sync.has_stream = 1u;
    sweep.cuda_sync.stream     = reinterpret_cast<uintptr_t>(producer);
    ovspatialquery_sphere_sweep_batch_reservation_t sweepReservation{};
    ovspatialquery_enqueue_result_t                 operation =
        ovspatialquery_enqueue_sphere_sweep_plan(instance, plan, &sweep, &sweepReservation);
    REQUIRE(operation.status == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_op_wait_result_t waited{};
    REQUIRE(ovspatialquery_wait_op(instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status ==
            OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_sphere_sweep_batch_result_t sweepResult{};
    REQUIRE(ovspatialquery_fetch_sphere_sweep_batch(instance, operation.op_id, &sweepResult).status ==
            OVSPATIALQUERY_API_SUCCESS);
    requireRows(sweepResult.columns.hit_ranges, sweepResult.columns.contact.exact.features,
                sweepResult.columns.material_ids, sweepResult.common.result_ready.value, operation.op_id);
    REQUIRE(destroyPlanWhenRetired(instance, plan));
    // Four raycasts and the sweep each handed the library a consumer_done event.
    REQUIRE(transferredEventDestroysWereInvoked(5u));

    gPhase = "release polygon scene";
    REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, mesh.value})
                .status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

int main()
{
    // Run first: earlier plans can raise the shared primary-context stack limit
    // and hide a regression in initial stack reservation before enqueue.
    testCudaPlanEnqueueDoesNotWaitForTheDevice();
    testCapabilityQueryAgreesWithCudaPlanCreation();
    testCudaPlansShareMeshDefinitions();
    testCudaPlansOnTwoDevicesKeepTheirOwnDefinitions();
    testCudaRaycastReportsEveryMeshCrossing(OVSPATIALQUERY_QUERY_EXACT);
    testCudaRaycastReportsEveryMeshCrossing(OVSPATIALQUERY_QUERY_EXACT_SINGLE);
    testCudaRaycastReachesFarCapsule();
    testCudaRaysMeetAConvexWhereItsUniformScalePlacesIt();
    testCudaSphereSweepReachesBoxEdgesAndCorners();
    testCudaSphereSweepsMeetFarFaces();
    testCudaPlansFindTheFirstContactOfEveryPrimitive();
    testCudaRaycastHitsSharedMeshEdges();
    testCudaColumnsCarryMaterials();
    testCudaHitsReportCallerSourceFaces();
    testPublicExactCudaRaycast();
    testPublicExactSingleCudaRayModesAndBoundaries();
    testCudaBoundsCandidateNormalOfFlatAabbFacesTheRay();
    testCudaSlotReturnsWhileReaperRingsWait();
    testPublicCudaSweep(OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE);
    testPublicCudaSweep(OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE, false, true);
    testPublicCudaSweep(OVSPATIALQUERY_QUERY_EXACT);
    testPublicCudaBoxSweep(OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE);
    testPublicCudaBoxSweep(OVSPATIALQUERY_QUERY_EXACT);
    testCudaSweepsFollowMeshSidedness();
    testPublicCudaRowSizes();
    testPublicCudaOverlap(OVSPATIALQUERY_OVERLAP_SPHERE, OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE);
    testPublicCudaOverlap(OVSPATIALQUERY_OVERLAP_SPHERE, OVSPATIALQUERY_QUERY_EXACT);
    testPublicCudaOverlap(OVSPATIALQUERY_OVERLAP_BOX, OVSPATIALQUERY_QUERY_EXACT);
    testPublicCudaOverlap(OVSPATIALQUERY_OVERLAP_AABB, OVSPATIALQUERY_QUERY_EXACT);
    testPublicCudaOverlap(OVSPATIALQUERY_OVERLAP_SPHERE, OVSPATIALQUERY_QUERY_EXACT, false);
    testPublicCudaOverlap(OVSPATIALQUERY_OVERLAP_SPHERE, OVSPATIALQUERY_QUERY_EXACT, true, true);
    for (uint32_t repeat = 0u; repeat != 3u; ++repeat)
    {
        testPublicCudaSweep(OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE, true);
        testPublicCudaOverlap(OVSPATIALQUERY_OVERLAP_SPHERE, OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE, true, false, true);
    }
    testPublicCudaFloat64BoxRows();
    testCudaExactBoxOverlapKeepsTouchingPrimitives();
    testCudaHullBoxOverlapsResolveSmallGaps();
    testCudaPlanHonorsViewMembership();
    testCudaBoxOverlapCandidateDispatch();
    testPublicCudaMalformedStatusAtomicity();
    testPublicCudaHostBudgetAccounting();
    testPublicCudaValidationAndBudget();
    testCudaPlanRunsOnTheLegacyDefaultStream();
    testCudaPlanRefusesACapturingStream();
    testPublicCudaDeviceLossDiagnostic();
    testPublicCudaDeviceLossDiagnostic(kCudaFailureResultMetadataTransfer);
    testPublicCudaDeviceLossDiagnostic(kCudaFailureResultMetadataDeviceLost);
    testPublicCudaDeviceLossDiagnostic(kCudaFailureResultMetadataInvariant);
    testDevicePreparedPublicationDeviceLoss(kCudaFailureDevicePublicationLaunch);
    testDevicePreparedPublicationDeviceLoss(kCudaFailureDevicePublicationComplete);
    return 0;
}
