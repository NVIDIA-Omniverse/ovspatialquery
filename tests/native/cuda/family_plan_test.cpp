/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define NOMINMAX
#include "../../support/native/cuda_public_test_helpers.h"
#include "../../support/native/test_failure.h"
#include "cuda_driver.h"
#include "oracle_backend.h"
#include "ovspatialquery_box_sweep_geometry.h"
#include "ovspatialquery_box_sweep_math.h"
#include "ovspatialquery_cuda_backend.h"
#include "ovspatialquery_cuda_failure_points.h"
#include "ovspatialquery_cuda_family_plan.h"
#include "ovspatialquery_cuda_family_plan_types.h"
#include "ovspatialquery_family_batch_plan_operation.h"
#include "ovspatialquery_instance_operation_runtime.h"
#include "ovspatialquery_instance_source.h"
#include "ovspatialquery_target_resolver.h"
#include "ovspatialquery_typed_geometry.h"
#include "portable_convex_hull.h"
#include "sphere_sweep_oracle.h"
#include "stored_sphere_paths.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

extern "C" void     ovspatialquery_test_get_cuda_dispatch_evidence(uint64_t* deviceToHostBytes,
                                                                   uint64_t* deviceToDeviceBytes, uint64_t* moduleLoads,
                                                                   uint64_t* moduleUnloads);
extern "C" void     ovspatialquery_test_set_cuda_failure_point(uint32_t point);
extern "C" uint32_t ovspatialquery_test_get_cuda_failure_point();

namespace
{
using namespace ovspatialquery::detail;

constexpr uint64_t kMaximumUint32SourceFace = UINT64_C(0xffffffff);
constexpr uint64_t kAboveUint32SourceFace   = UINT64_C(0x100000000);

[[noreturn]] void fail(const char* message)
{
    std::fprintf(stderr, "FAIL: %s\n", message);
    OVSPATIALQUERY_TEST_EXIT_FAILURE();
}

void require(bool condition, const char* message)
{
    if (!condition)
    {
        fail(message);
    }
}

// Hands `done` to the operation as its consumer_done, waits for the consumer to finish and
// checks that the owner released and destroyed the event exactly once.
void retireConsumer(CudaFamilyOperation& operation, CudaDriver::Event done, const char* message)
{
    require(operation.transferConsumerDone({reinterpret_cast<uintptr_t>(done)}) == OVSPATIALQUERY_API_SUCCESS, message);
    const std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    CudaCompletionState                         state    = operation.consumerState();
    while (state == CudaCompletionState::pending && std::chrono::steady_clock::now() < deadline)
    {
        std::this_thread::yield();
        state = operation.consumerState();
    }
    operation.releaseReadyDependencies();
    require(state == CudaCompletionState::complete && operation.consumerDoneOwnershipReleased() &&
                transferredEventDestroysWereInvoked(),
            message);
}

uint64_t transferredEventPreDestroyFailures()
{
    uint64_t failures = 0u;
    ovspatialquery_test_get_cuda_transferred_event_evidence(nullptr, nullptr, nullptr, &failures);
    return failures;
}

class OperationInstance
{
  public:
    explicit OperationInstance(uint32_t operationCapacity = 4u, uint32_t deviceResultCapacity = 4u,
                               uint64_t hostByteBudget   = std::numeric_limits<uint64_t>::max(),
                               uint64_t deviceByteBudget = std::numeric_limits<uint64_t>::max())
    {
        const SourceDeviceBudget deviceBudget{
            DLDevice{kDLCUDA, 0},
            deviceByteBudget,
        };
        const SourceInstanceConfig config{
            operationCapacity, deviceResultCapacity, 1u, hostByteBudget, &deviceBudget, 1u,
        };
        require(createInstance(config, &mInstance) == OVSPATIALQUERY_API_SUCCESS,
                "CUDA family-plan instance creation failed");
    }

    ~OperationInstance()
    {
        ovspatialquery_api_status_t released = OVSPATIALQUERY_API_BUSY;
        for (uint32_t attempt = 0u; attempt != 1000u && released == OVSPATIALQUERY_API_BUSY; ++attempt)
        {
            released = ovspatialquery_destroy_instance(mInstance).status;
            if (released == OVSPATIALQUERY_API_BUSY)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        if (released != OVSPATIALQUERY_API_SUCCESS)
        {
            std::fprintf(stderr, "CUDA family-plan instance release status=%u\n", static_cast<unsigned int>(released));
        }
        require(released == OVSPATIALQUERY_API_SUCCESS, "CUDA family-plan instance release failed");
    }

    ovspatialquery_instance_handle_t get() const noexcept { return mInstance; }

  private:
    ovspatialquery_instance_handle_t mInstance = OVSPATIALQUERY_INVALID_INSTANCE;
};

// Two bounds-only candidates: a sphere at x 2..4 (bit 0x1) and a box at x 5..7 (bit 0x2).
std::vector<PopulationObject> candidateObjects()
{
    PopulationObject object{};
    object.id                = {17u};
    object.bounds            = {{2.0, -1.0, -1.0}, {4.0, 1.0, 1.0}};
    object.geometryReference = {ExactGeometryFamily::sphere, 101u};
    object.participationBits = {{UINT64_C(0x1), 0u}};
    PopulationObject second{};
    second.id                = {23u};
    second.bounds            = {{5.0, -1.0, -1.0}, {7.0, 1.0, 1.0}};
    second.geometryReference = {ExactGeometryFamily::box, 202u};
    second.participationBits = {{UINT64_C(0x2), 0u}};
    return {object, second};
}

// `count` bounds-only sphere candidates three units apart along x.
std::vector<PopulationObject> candidateObjects(uint32_t count)
{
    std::vector<PopulationObject> objects(count);
    for (uint32_t index = 0u; index < count; ++index)
    {
        const double x                   = static_cast<double>(index) * 3.0;
        objects[index].id                = {static_cast<uint64_t>(index) + 1u};
        objects[index].bounds            = {{x, -1.0, -1.0}, {x + 2.0, 1.0, 1.0}};
        objects[index].geometryReference = {ExactGeometryFamily::sphere, index + 1u};
        objects[index].participationBits = {{UINT64_C(0x1), 0u}};
    }
    return objects;
}

PopulationObject exactObject(uint64_t objectId, uint64_t geometryValue, std::shared_ptr<const ExactGeometry> geometry,
                             uint64_t participationBits)
{
    PopulationObject object{};
    object.id.value          = objectId;
    object.bounds            = geometry->worldBounds();
    object.geometryReference = {geometry->family(), geometryValue};
    object.exactGeometry     = std::move(geometry);
    object.participationBits = {{participationBits, 0u}};
    return object;
}

std::shared_ptr<const PreparedConvexHull> unitCubeHull()
{
    const std::vector<float> vertices{
        -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0,
        -1.0, -1.0, 1.0,  1.0, -1.0, 1.0,  1.0, 1.0, 1.0,  -1.0, 1.0, 1.0,
    };
    std::shared_ptr<const PreparedConvexHull> hull;
    require(preparePortableConvexHull(vertices, 8u, UINT64_MAX, hull) == ConvexHullPreparationOutcome::success &&
                hull != nullptr,
            "CUDA unit cube hull preparation failed");
    return hull;
}

std::shared_ptr<const ExactGeometry> meshGeometry()
{
    TriangleMeshSource mesh{};
    mesh.key      = 77u;
    mesh.vertices = {
        14.0, -1.0, -1.0, 14.0, -1.0, 1.0, 14.0, 1.0, 1.0, 14.0, 1.0, -1.0,
    };
    mesh.triangles   = {0u, 1u, 2u, 0u, 2u, 3u};
    mesh.sourceFaces = {
        kMaximumUint32SourceFace,
        kAboveUint32SourceFace,
    };
    mesh.sourceCorners = {0u, 1u, 2u, 0u, 2u, 3u};
    std::shared_ptr<const PreparedTriangleMesh>   preparedMesh;
    std::shared_ptr<const TriangleObjectGeometry> prepared;
    require(prepareTriangleMesh(mesh, preparedMesh) == OVSPATIALQUERY_API_SUCCESS &&
                bindPreparedTriangleGeometry(
                    preparedMesh, {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0},
                    true, prepared) == OVSPATIALQUERY_API_SUCCESS,
            "CUDA exact mesh preparation failed");
    std::shared_ptr<const ExactGeometry> geometry;
    require(createTriangleMeshExactGeometry(prepared, geometry) == OVSPATIALQUERY_API_SUCCESS,
            "CUDA exact mesh creation failed");
    return geometry;
}

std::vector<PopulationObject> fiveExactFamilies()
{
    std::vector<PopulationObject>        objects;
    std::shared_ptr<const ExactGeometry> geometry;

    SphereGeometrySource sphere{};
    sphere.center = {2.0, 0.0, 0.0};
    sphere.radius = 1.0;
    require(createSphereExactGeometry(sphere, geometry) == OVSPATIALQUERY_API_SUCCESS,
            "CUDA exact sphere creation failed");
    objects.push_back(exactObject(1u, 101u, geometry, UINT64_C(0x1)));

    BoxGeometrySource box{};
    box.center = {5.0, 0.0, 0.0};
    box.axes   = {
        std::array<double, 3u>{1.0, 0.0, 0.0},
        std::array<double, 3u>{0.0, 1.0, 0.0},
        std::array<double, 3u>{0.0, 0.0, 1.0},
    };
    box.halfExtents = {1.0, 1.0, 1.0};
    require(createBoxExactGeometry(box, geometry) == OVSPATIALQUERY_API_SUCCESS, "CUDA exact box creation failed");
    objects.push_back(exactObject(2u, 202u, geometry, UINT64_C(0x2)));

    CapsuleGeometrySource capsule{};
    capsule.endpointA = {8.0, -1.0, 0.0};
    capsule.endpointB = {8.0, 1.0, 0.0};
    capsule.radius    = 1.0;
    require(createCapsuleExactGeometry(capsule, geometry) == OVSPATIALQUERY_API_SUCCESS,
            "CUDA exact capsule creation failed");
    objects.push_back(exactObject(3u, 303u, geometry, UINT64_C(0x4)));

    const std::shared_ptr<const PreparedConvexHull> preparedHull = unitCubeHull();
    const std::array<double, 16u>                   convexWorldFromLocal{
        0.0, -2.0, 0.0, 11.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    };
    require(createConvexExactGeometry(preparedHull, convexWorldFromLocal, geometry) ==
                ConvexConstructionOutcome::success,
            "CUDA exact prepared convex creation failed");
    objects.push_back(exactObject(4u, 404u, geometry, UINT64_C(0x8)));
    objects.push_back(exactObject(5u, 505u, meshGeometry(), UINT64_C(0x10)));
    return objects;
}

DLTensor cudaTensor(CudaDriver::DevicePointer pointer, DLDataType type, int32_t rank, int64_t* shape, int64_t* strides)
{
    DLTensor tensor{};
    tensor.data    = reinterpret_cast<void*>(static_cast<uintptr_t>(pointer));
    tensor.device  = {kDLCUDA, 0};
    tensor.ndim    = rank;
    tensor.dtype   = type;
    tensor.shape   = shape;
    tensor.strides = strides;
    return tensor;
}

// Query bits as the plans read them, uint64[Q,2] with bits 0-63 first; `shape` is {Q, 2}.
int64_t kBitStrides[]{2, 1};

DLTensor cudaBitsTensor(CudaDriver::DevicePointer pointer, int64_t* shape)
{
    return cudaTensor(pointer, {kDLUInt, 64u, 1u}, 2, shape, kBitStrides);
}

ovspatialquery_api_status_t prepareAndLaunchCudaFamilyPlan(const std::shared_ptr<CudaFamilyPlan>& plan,
                                                           uint32_t resultSlot, CudaFamilyLaunch launch,
                                                           GenerationLease                       generation,
                                                           std::shared_ptr<CudaFamilyOperation>& outOperation)
{
    // The owner's order: measure the launch's staging, then prepare with it.
    CudaFamilyStagingCharge              charge{};
    std::shared_ptr<CudaFamilyOperation> prepared;
    ovspatialquery_api_status_t          status = plan->measure(launch, generation, charge);
    if (status == OVSPATIALQUERY_API_SUCCESS)
    {
        status = plan->prepare(launch, std::move(generation), std::move(charge.staging), prepared);
    }
    return status == OVSPATIALQUERY_API_SUCCESS ? plan->launch(resultSlot, std::move(prepared), outOperation) : status;
}

void requireTensor(const DLTensor& tensor, uint8_t code, uint8_t bits, int32_t rank, int64_t firstExtent,
                   int64_t secondExtent = 0)
{
    require(tensor.data != nullptr && tensor.device.device_type == kDLCUDA && tensor.device.device_id == 0 &&
                tensor.dtype.code == code && tensor.dtype.bits == bits && tensor.dtype.lanes == 1u &&
                tensor.ndim == rank && tensor.shape != nullptr && tensor.shape[0] == firstExtent &&
                (rank != 2 || tensor.shape[1] == secondExtent),
            "CUDA family result tensor descriptor changed");
}

uint64_t cudaDeviceToHostBytes()
{
    uint64_t deviceToHost   = 0u;
    uint64_t deviceToDevice = 0u;
    ovspatialquery_test_get_cuda_dispatch_evidence(&deviceToHost, &deviceToDevice, nullptr, nullptr);
    return deviceToHost;
}

// Closes `owner`, retrying for up to ten seconds while device-safe retirement keeps it BUSY.
void closeOwner(FamilyBatchPlanOperationOwner& owner, const char* message)
{
    const std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    ovspatialquery_api_status_t                 closed   = owner.close();
    while (closed == OVSPATIALQUERY_API_BUSY && std::chrono::steady_clock::now() < deadline)
    {
        std::this_thread::yield();
        closed = owner.close();
    }
    require(closed == OVSPATIALQUERY_API_SUCCESS, message);
}

void testPrivateCudaRaycastLaunchPublishesSemanticColumns()
{
    CudaDriver               cuda;
    const CudaDriver::Stream producer      = cuda.createStream();
    const CudaDriver::Stream consumer      = cuda.createStream();
    const CudaDriver::Event  producerReady = cuda.createEvent();
    const CudaDriver::Event  consumerDone  = cuda.createEvent();

    ovspatialquery_backend_runtime_t* runtime = nullptr;
    require(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS,
            "CUDA semantic-column oracle runtime creation failed");
    PopulationTable populations(runtime);
    PopulationKey   population;
    require(populations.create(2u, 1, candidateObjects(), population) == OVSPATIALQUERY_API_SUCCESS,
            "CUDA semantic-column population creation failed");

    TargetResolver resolver(populations);
    ResolvedTarget resolved;
    TargetRequest  request{};
    request.target.population = population;
    request.selection         = {TargetSelectionKind::head, 0};
    require(resolver.resolve(request, resolved) == TargetResolveOutcome::success && resolved.ordinal == 1,
            "CUDA semantic-column target resolution failed");

    CudaFamilyPlanConfig config{};
    config.family       = CudaFamilyKind::raycast;
    config.precision    = CudaFamilyPrecision::boundsCandidate;
    config.hitMode      = CudaFamilyHitMode::closest;
    config.resultFields = kCudaFamilyResultPosition | kCudaFamilyResultNormal | kCudaFamilyResultDistance |
                          kCudaFamilyResultGeometryIdentity;
    config.maximumQueryCount   = 1u;
    config.maximumHitsPerQuery = 1u;
    config.hitCapacityPerSlot  = 1u;
    config.resultSlotCount     = 1u;
    std::shared_ptr<CudaFamilyPlan> plan;
    require(createCudaFamilyPlan(config, plan) == OVSPATIALQUERY_API_SUCCESS && plan != nullptr &&
                plan->deviceByteCharge() != 0u,
            "private CUDA raycast plan creation failed");

    struct QueryData
    {
        double   origin[3]{0.0, 0.0, 0.0};
        float    direction[3]{1.0f, 0.0f, 0.0f};
        double   minimumDistance = 0.0;
        double   maximumDistance = 10.0;
        uint64_t queryBits[2]{UINT64_MAX, UINT64_MAX};
    } host;
    const CudaDriver::DevicePointer device = cuda.allocate(sizeof(host));
    cuda.uploadAsync(device, &host, sizeof(host), producer);
    cuda.record(producerReady, producer);

    const int64_t queryCount      = 1;
    int64_t       rowShape[]      = {queryCount};
    int64_t       vectorShape[]   = {queryCount, 3};
    int64_t       rowStrides[]    = {1};
    int64_t       vectorStrides[] = {3, 1};
    const auto    at              = [device](size_t offset)
    { return static_cast<CudaDriver::DevicePointer>(device + static_cast<CudaDriver::DevicePointer>(offset)); };
    CudaFamilyLaunch launch{};
    launch.queryCount = static_cast<uint64_t>(queryCount);
    int64_t bitRows[]{queryCount, 2};
    launch.queryBits = cudaBitsTensor(at(offsetof(QueryData, queryBits)), bitRows);
    launch.positions = cudaTensor(at(offsetof(QueryData, origin)), {kDLFloat, 64u, 1u}, 2, vectorShape, vectorStrides);
    launch.directionsOrRotations =
        cudaTensor(at(offsetof(QueryData, direction)), {kDLFloat, 32u, 1u}, 2, vectorShape, vectorStrides);
    launch.minimumDistances =
        cudaTensor(at(offsetof(QueryData, minimumDistance)), {kDLFloat, 64u, 1u}, 1, rowShape, rowStrides);
    launch.maximumDistances =
        cudaTensor(at(offsetof(QueryData, maximumDistance)), {kDLFloat, 64u, 1u}, 1, rowShape, rowStrides);
    launch.stream    = reinterpret_cast<uintptr_t>(producer);
    launch.waitEvent = reinterpret_cast<uintptr_t>(producerReady);

    std::shared_ptr<CudaFamilyOperation> operation;
    require(prepareAndLaunchCudaFamilyPlan(plan, 0u, launch, std::move(resolved.generation), operation) ==
                    OVSPATIALQUERY_API_SUCCESS &&
                operation != nullptr,
            "private CUDA raycast launch failed");
    CudaFamilyResultColumns reservation = operation->reservation();
    require(reservation.resultSlot == 0u && reservation.queryCount == static_cast<uint64_t>(queryCount) &&
                reservation.hitCapacity == 1u && reservation.truth == cuda_family::kPrecisionBoundsCandidate &&
                reservation.sourceOrdinal == 1 && reservation.executionDevice.device_type == kDLCUDA &&
                reservation.executionDevice.device_id == 0 && reservation.resultReady.value != 0u,
            "CUDA admission metadata was not final at launch return");
    requireTensor(reservation.queryStatus, kDLUInt, 32u, 1, queryCount);
    requireTensor(reservation.storedHitCount, kDLUInt, 64u, 1, 1);
    requireTensor(reservation.hitOffsets, kDLUInt, 64u, 1, queryCount);
    requireTensor(reservation.hitCounts, kDLUInt, 64u, 1, queryCount);
    requireTensor(reservation.truncated, kDLUInt, 8u, 1, queryCount);
    requireTensor(reservation.objectIds, kDLUInt, 64u, 1, 1);
    requireTensor(reservation.geometryTypes, kDLInt, 32u, 1, 1);
    requireTensor(reservation.geometryValues, kDLUInt, 64u, 1, 1);
    requireTensor(reservation.positions, kDLFloat, 64u, 2, 1, 3);
    requireTensor(reservation.normals, kDLFloat, 32u, 2, 1, 3);
    requireTensor(reservation.distances, kDLFloat, 64u, 1, 1);

    const uintptr_t reservedReady   = reservation.resultReady.value;
    const void*     reservedStatus  = reservation.queryStatus.data;
    const void*     reservedObjects = reservation.objectIds.data;
    reservation                     = {};

    const uint64_t attemptsBeforeRejectedReady = transferredEventDestroyAttempts();
    require(operation->transferConsumerDone({reservedReady}) == OVSPATIALQUERY_API_INVALID_ARGUMENT &&
                transferredEventDestroyAttempts() == attemptsBeforeRejectedReady,
            "CUDA raycast owner consumed its result_ready event as consumer_done");

    cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(reservedReady));
    cuda.synchronize(reinterpret_cast<CudaDriver::Event>(reservedReady));
    CudaFamilyResultColumns view{};
    require(operation->completedResult(view) == CudaFamilyCompletedResultOutcome::success,
            "terminal CUDA result was unavailable after result_ready");
    require(view.resultReady.value == reservedReady && view.queryStatus.data == reservedStatus &&
                view.objectIds.data == reservedObjects,
            "terminal CUDA fetch did not preserve admission-time descriptors");
    uint32_t queryStatus    = UINT32_MAX;
    uint64_t storedHitCount = UINT64_MAX;
    uint64_t hitOffset      = UINT64_MAX;
    uint64_t hitCount       = UINT64_MAX;
    uint8_t  truncated      = UINT8_MAX;
    uint64_t objectId       = UINT64_MAX;
    int32_t  geometryType   = 0;
    uint64_t geometryValue  = 0u;
    double   distance       = -1.0;
    cuda.downloadAsync(&queryStatus, deviceAddress(view.queryStatus), sizeof(queryStatus), consumer);
    cuda.downloadAsync(&storedHitCount, deviceAddress(view.storedHitCount), sizeof(storedHitCount), consumer);
    cuda.downloadAsync(&hitOffset, deviceAddress(view.hitOffsets), sizeof(hitOffset), consumer);
    cuda.downloadAsync(&hitCount, deviceAddress(view.hitCounts), sizeof(hitCount), consumer);
    cuda.downloadAsync(&truncated, deviceAddress(view.truncated), sizeof(truncated), consumer);
    cuda.downloadAsync(&objectId, deviceAddress(view.objectIds), sizeof(objectId), consumer);
    cuda.downloadAsync(&geometryType, deviceAddress(view.geometryTypes), sizeof(geometryType), consumer);
    cuda.downloadAsync(&geometryValue, deviceAddress(view.geometryValues), sizeof(geometryValue), consumer);
    cuda.downloadAsync(&distance, deviceAddress(view.distances), sizeof(distance), consumer);
    cuda.record(consumerDone, consumer);
    cuda.synchronize(consumerDone);
    require(queryStatus == cuda_family::kBatchRowSuccess && storedHitCount == 1u && hitOffset == 0u && hitCount == 1u &&
                truncated == 0u && objectId == 17u && geometryType == OVSPATIALQUERY_GEOMETRY_SPHERE &&
                geometryValue == 101u && distance >= 1.999 && distance <= 2.001,
            "CUDA raycast semantic columns changed");

    retireConsumer(*operation, consumerDone, "CUDA raycast consumer retirement failed");
    operation.reset();
    plan.reset();
    std::shared_ptr<void> retired;
    require(populations.release(population, retired) == OVSPATIALQUERY_API_SUCCESS,
            "CUDA semantic-column plan retained its population");
    ovspatialquery_test_oracle_destroy_runtime(runtime);
    cuda.free(device);
    cuda.destroyEvent(producerReady);
    cuda.destroyStream(consumer);
    cuda.destroyStream(producer);
}

bool absentTensor(const DLTensor& tensor)
{
    return tensor.data == nullptr && tensor.device.device_type == 0 && tensor.device.device_id == 0 &&
           tensor.ndim == 0 && tensor.dtype.code == 0u && tensor.dtype.bits == 0u && tensor.dtype.lanes == 0u &&
           tensor.shape == nullptr && tensor.strides == nullptr && tensor.byte_offset == 0u;
}

bool nearFloat(float value, float expected) { return value >= expected - 0.001f && value <= expected + 0.001f; }

void runExactFiveFamilyCase(CudaDriver& cuda, CudaDriver::Stream consumer, PopulationTable& populations,
                            PopulationKey population, const CudaFamilyPlanConfig& config,
                            const CudaFamilyLaunch& launch)
{
    TargetResolver resolver(populations);
    ResolvedTarget resolved;
    TargetRequest  request{};
    request.target.population = population;
    request.selection         = {TargetSelectionKind::head, 0};
    require(resolver.resolve(request, resolved) == TargetResolveOutcome::success && resolved.ordinal == 31,
            "CUDA exact-family target resolution failed");

    std::shared_ptr<CudaFamilyPlan> plan;
    require(createCudaFamilyPlan(config, plan) == OVSPATIALQUERY_API_SUCCESS && plan != nullptr,
            "CUDA exact-family plan creation failed");
    std::shared_ptr<CudaFamilyOperation> operation;
    require(prepareAndLaunchCudaFamilyPlan(plan, 0u, launch, std::move(resolved.generation), operation) ==
                    OVSPATIALQUERY_API_SUCCESS &&
                operation != nullptr,
            "CUDA exact-family launch failed");

    CudaFamilyResultColumns reserved = operation->reservation();
    require(reserved.queryCount == 5u && reserved.hitCapacity == 5u && reserved.truth == cuda_family::kPrecisionExact &&
                reserved.sourceOrdinal == 31 && reserved.resultReady.value != 0u,
            "CUDA exact-family reservation metadata changed");
    const uintptr_t ready           = reserved.resultReady.value;
    const void*     reservedStatus  = reserved.queryStatus.data;
    const void*     reservedObjects = reserved.objectIds.data;
    reserved                        = {};

    cuda.synchronize(reinterpret_cast<CudaDriver::Event>(ready));
    CudaFamilyResultColumns view{};
    require(operation->completedResult(view) == CudaFamilyCompletedResultOutcome::success,
            "CUDA exact-family completed result was unavailable");
    require(view.resultReady.value == ready && view.queryStatus.data == reservedStatus &&
                view.objectIds.data == reservedObjects,
            "CUDA exact-family completed descriptors changed after admission");

    uint32_t status[5]            = {};
    uint64_t stored               = UINT64_MAX;
    uint64_t offsets[5]           = {};
    uint64_t counts[5]            = {};
    uint8_t  truncated[5]         = {};
    uint64_t objectIds[5]         = {};
    int32_t  geometryTypes[5]     = {};
    uint64_t geometryValues[5]    = {};
    uint32_t featureKinds[5]      = {};
    uint64_t sourceFaces[5]       = {};
    uint32_t sourceCorners[5][3]  = {};
    float    barycentrics[5][3]   = {};
    double   queryPositions[5][3] = {};
    cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(ready));
    cuda.downloadAsync(status, deviceAddress(view.queryStatus), sizeof(status), consumer);
    cuda.downloadAsync(&stored, deviceAddress(view.storedHitCount), sizeof(stored), consumer);
    cuda.downloadAsync(offsets, deviceAddress(view.hitOffsets), sizeof(offsets), consumer);
    cuda.downloadAsync(counts, deviceAddress(view.hitCounts), sizeof(counts), consumer);
    cuda.downloadAsync(truncated, deviceAddress(view.truncated), sizeof(truncated), consumer);
    cuda.downloadAsync(objectIds, deviceAddress(view.objectIds), sizeof(objectIds), consumer);
    if (config.family != CudaFamilyKind::overlap)
    {
        requireTensor(view.geometryTypes, kDLInt, 32u, 1, 5);
        requireTensor(view.geometryValues, kDLUInt, 64u, 1, 5);
        requireTensor(view.sourceFaceIndices, kDLUInt, 64u, 1, 5);
        cuda.downloadAsync(geometryTypes, deviceAddress(view.geometryTypes), sizeof(geometryTypes), consumer);
        cuda.downloadAsync(geometryValues, deviceAddress(view.geometryValues), sizeof(geometryValues), consumer);
        cuda.downloadAsync(featureKinds, deviceAddress(view.sourceFeatureKinds), sizeof(featureKinds), consumer);
        cuda.downloadAsync(sourceFaces, deviceAddress(view.sourceFaceIndices), sizeof(sourceFaces), consumer);
        cuda.downloadAsync(sourceCorners, deviceAddress(view.sourceFaceCorners), sizeof(sourceCorners), consumer);
        cuda.downloadAsync(barycentrics, deviceAddress(view.sourceBarycentrics), sizeof(barycentrics), consumer);
        if (config.family == CudaFamilyKind::sphereSweep)
        {
            requireTensor(view.queryPositions, kDLFloat, 64u, 2, 5, 3);
            cuda.downloadAsync(queryPositions, deviceAddress(view.queryPositions), sizeof(queryPositions), consumer);
        }
    }
    const CudaDriver::Event consumerDone = cuda.createEvent();
    cuda.record(consumerDone, consumer);
    cuda.synchronize(consumerDone);

    require(stored == 5u, "CUDA exact-family stored count changed");
    for (uint64_t row = 0u; row != 5u; ++row)
    {
        require(status[row] == cuda_family::kBatchRowSuccess && offsets[row] < 5u && counts[row] == 1u &&
                    truncated[row] == 0u && objectIds[offsets[row]] == row + 1u,
                "CUDA exact-family row identity or partition changed");
    }
    if (config.family == CudaFamilyKind::overlap)
    {
        require(absentTensor(view.sourceFeatureKinds) && absentTensor(view.sourceFaceIndices) &&
                    absentTensor(view.sourceFaceCorners) && absentTensor(view.sourceBarycentrics),
                "CUDA overlap exposed ray/sweep-only columns");
    }
    else
    {
        const int32_t  expectedGeometryTypes[5]{OVSPATIALQUERY_GEOMETRY_SPHERE, OVSPATIALQUERY_GEOMETRY_BOX,
                                                OVSPATIALQUERY_GEOMETRY_CAPSULE, OVSPATIALQUERY_GEOMETRY_CONVEX,
                                                OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH};
        const uint64_t expectedGeometryValues[5]{101u, 202u, 303u, 404u, 505u};
        for (uint64_t row = 0u; row != 5u; ++row)
        {
            const uint64_t hit = offsets[row];
            require(geometryTypes[hit] == expectedGeometryTypes[row] &&
                        geometryValues[hit] == expectedGeometryValues[row],
                    "CUDA geometry identity column changed");
            if (config.family == CudaFamilyKind::sphereSweep)
            {
                require(std::isfinite(queryPositions[hit][0]) && std::isfinite(queryPositions[hit][1]) &&
                            std::isfinite(queryPositions[hit][2]),
                        "CUDA sweep query-contact column was not valid");
            }
        }
        for (uint64_t row = 0u; row != 4u; ++row)
        {
            const uint64_t hit = offsets[row];
            require(featureKinds[hit] == cuda_family::kFeatureNone, "CUDA analytic/convex source feature was not NONE");
        }
        const uint64_t triangle = offsets[4];
        const bool     sweep    = config.family == CudaFamilyKind::sphereSweep;
        require(featureKinds[triangle] == cuda_family::kFeatureTriangle &&
                    sourceFaces[triangle] == (sweep ? kAboveUint32SourceFace : kMaximumUint32SourceFace) &&
                    sourceCorners[triangle][0] == 0u && sourceCorners[triangle][1] == (sweep ? 2u : 1u) &&
                    sourceCorners[triangle][2] == (sweep ? 3u : 2u) &&
                    nearFloat(barycentrics[triangle][0], sweep ? 0.25f : 0.5f) &&
                    nearFloat(barycentrics[triangle][1], sweep ? 0.5f : 0.25f) &&
                    nearFloat(barycentrics[triangle][2], 0.25f),
                "CUDA uint64 triangle source feature payload changed");
    }

    operation->releaseReadyDependencies();
    retireConsumer(*operation, consumerDone, "CUDA exact-family consumer retirement failed");
    operation.reset();
    plan.reset();
}

void testExactFiveStoredGeometryFamiliesAcrossQueries()
{
    CudaDriver                        cuda;
    const CudaDriver::Stream          producer      = cuda.createStream();
    const CudaDriver::Stream          consumer      = cuda.createStream();
    const CudaDriver::Event           producerReady = cuda.createEvent();
    ovspatialquery_backend_runtime_t* runtime       = nullptr;
    require(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS,
            "CUDA exact-family oracle runtime creation failed");
    PopulationTable populations(runtime);
    PopulationKey   population;
    require(populations.create(2u, 31, fiveExactFamilies(), population) == OVSPATIALQUERY_API_SUCCESS,
            "CUDA exact-family population creation failed");

    struct QueryData
    {
        uint64_t bits[5][2]{{1u, 0u}, {2u, 0u}, {4u, 0u}, {8u, 0u}, {16u, 0u}};
        double   origins[5][3]{};
        float    directions[5][3]{};
        double   minimumDistances[5]{};
        double   maximumDistances[5]{};
        double   aabbCenters[5][3]{};
    } host;
    const double centers[5]{2.0, 5.0, 8.0, 11.0, 14.0};
    for (size_t row = 0u; row != 5u; ++row)
    {
        host.origins[row][1]       = -0.5;
        host.directions[row][0]    = 1.0f;
        host.maximumDistances[row] = 20.0;
        host.aabbCenters[row][0]   = centers[row];
    }
    host.origins[3][1]                     = 3.0;
    host.aabbCenters[3][1]                 = 3.0;
    const CudaDriver::DevicePointer device = cuda.allocate(sizeof(host));
    cuda.uploadAsync(device, &host, sizeof(host), producer);
    cuda.record(producerReady, producer);
    int64_t    rows[]          = {5};
    int64_t    vectors[]       = {5, 3};
    int64_t    rowStrides[]    = {1};
    int64_t    vectorStrides[] = {3, 1};
    const auto at              = [device](size_t offset)
    { return static_cast<CudaDriver::DevicePointer>(device + static_cast<CudaDriver::DevicePointer>(offset)); };
    int64_t        bitRows[]{5, 2};
    const DLTensor bits = cudaBitsTensor(at(offsetof(QueryData, bits)), bitRows);
    const DLTensor origins =
        cudaTensor(at(offsetof(QueryData, origins)), {kDLFloat, 64u, 1u}, 2, vectors, vectorStrides);
    const DLTensor directions =
        cudaTensor(at(offsetof(QueryData, directions)), {kDLFloat, 32u, 1u}, 2, vectors, vectorStrides);
    const DLTensor minimumDistances =
        cudaTensor(at(offsetof(QueryData, minimumDistances)), {kDLFloat, 64u, 1u}, 1, rows, rowStrides);
    const DLTensor maximumDistances =
        cudaTensor(at(offsetof(QueryData, maximumDistances)), {kDLFloat, 64u, 1u}, 1, rows, rowStrides);
    const DLTensor aabbCenters =
        cudaTensor(at(offsetof(QueryData, aabbCenters)), {kDLFloat, 64u, 1u}, 2, vectors, vectorStrides);

    CudaFamilyPlanConfig config{};
    config.precision    = CudaFamilyPrecision::exact;
    config.hitMode      = CudaFamilyHitMode::closest;
    config.resultFields = kCudaFamilyResultPosition | kCudaFamilyResultNormal | kCudaFamilyResultDistance |
                          kCudaFamilyResultSourceFeature | kCudaFamilyResultGeometryIdentity;
    config.maximumQueryCount   = 5u;
    config.maximumHitsPerQuery = 1u;
    config.hitCapacityPerSlot  = 5u;
    config.resultSlotCount     = 1u;
    CudaFamilyLaunch launch{};
    launch.queryCount            = 5u;
    launch.queryBits             = bits;
    launch.positions             = origins;
    launch.directionsOrRotations = directions;
    launch.minimumDistances      = minimumDistances;
    launch.maximumDistances      = maximumDistances;
    launch.stream                = reinterpret_cast<uintptr_t>(producer);
    launch.waitEvent             = reinterpret_cast<uintptr_t>(producerReady);
    config.family                = CudaFamilyKind::raycast;
    runExactFiveFamilyCase(cuda, consumer, populations, population, config, launch);
    config.precision = CudaFamilyPrecision::exactSingle;
    runExactFiveFamilyCase(cuda, consumer, populations, population, config, launch);

    host.origins[4][1] = 0.5;
    cuda.uploadAsync(device, &host, sizeof(host), producer);
    cuda.record(producerReady, producer);
    config.precision = CudaFamilyPrecision::exact;
    config.family    = CudaFamilyKind::sphereSweep;
    config.resultFields |= kCudaFamilyResultQueryPosition;
    config.planRadius = 0.1;
    runExactFiveFamilyCase(cuda, consumer, populations, population, config, launch);

    config.family             = CudaFamilyKind::overlap;
    config.hitMode            = CudaFamilyHitMode::any;
    config.overlapKind        = CudaOverlapKind::aabb;
    config.resultFields       = 0u;
    config.planHalfExtents[0] = 1.2;
    config.planHalfExtents[1] = 1.2;
    config.planHalfExtents[2] = 1.2;
    launch                    = {};
    launch.queryCount         = 5u;
    launch.queryBits          = bits;
    launch.positions          = aabbCenters;
    launch.stream             = reinterpret_cast<uintptr_t>(producer);
    launch.waitEvent          = reinterpret_cast<uintptr_t>(producerReady);
    runExactFiveFamilyCase(cuda, consumer, populations, population, config, launch);

    std::shared_ptr<void> retired;
    require(populations.release(population, retired) == OVSPATIALQUERY_API_SUCCESS,
            "CUDA exact-family tests retained their population");
    ovspatialquery_test_oracle_destroy_runtime(runtime);
    cuda.free(device);
    cuda.destroyEvent(producerReady);
    cuda.destroyStream(consumer);
    cuda.destroyStream(producer);
}

void testBoxSweepsAcrossFiveFamilies()
{
    CudaDriver                        cuda;
    const auto                        producer = cuda.createStream(), consumer = cuda.createStream();
    const auto                        producerReady = cuda.createEvent();
    ovspatialquery_backend_runtime_t* runtime       = nullptr;
    require(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS, "box CUDA runtime");
    PopulationTable    populations(runtime);
    auto               objects = fiveExactFamilies();
    TriangleMeshSource mesh{};
    mesh.key           = 983;
    mesh.vertices      = {0, -2, -2, 0, -2, 2, 0, 2, 2, 0, 2, -2};
    mesh.triangles     = {0, 1, 2, 0, 2, 3};
    mesh.sourceFaces   = {kAboveUint32SourceFace, kAboveUint32SourceFace + 1};
    mesh.sourceCorners = {3, 7, 9, 3, 9, 12};
    std::shared_ptr<const PreparedTriangleMesh> prepared;
    require(prepareTriangleMesh(mesh, prepared) == OVSPATIALQUERY_API_SUCCESS, "box CUDA mesh preparation");
    std::shared_ptr<const TriangleObjectGeometry> placement;
    const std::array<double, 16>                  transform = {2, 0, 0, 14, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1};
    require(bindPreparedTriangleGeometry(prepared, transform, true, placement) == OVSPATIALQUERY_API_SUCCESS,
            "box CUDA scaled mesh");
    std::shared_ptr<const ExactGeometry> geometry;
    require(createTriangleMeshExactGeometry(placement, geometry) == OVSPATIALQUERY_API_SUCCESS,
            "box CUDA mesh geometry");
    objects[4] = exactObject(5, 505, geometry, 16);
    PopulationKey population;
    require(populations.create(2, 31, objects, population) == OVSPATIALQUERY_API_SUCCESS, "box CUDA population");
    constexpr size_t count = 12;
    struct Queries
    {
        uint64_t bits[count][2]{};
        double   origins[count][3]{}, minimum[count]{}, maximum[count]{};
        float    directions[count][3]{}, rotations[count][4]{};
    } host;
    const unsigned family[count] = {0, 1, 2, 3, 4, 0, 1, 0, 0, 2, 3, 4};
    for (size_t row = 0; row < count; ++row)
    {
        host.bits[row][0]       = uint64_t{1} << family[row];
        host.directions[row][0] = 1;
        host.rotations[row][3]  = 1;
        host.maximum[row]       = 20;
    }
    host.origins[0][1]   = .75;
    host.origins[0][2]   = 1;
    host.rotations[1][2] = float(std::sin(.3926990816987241548));
    host.rotations[1][3] = float(std::cos(.3926990816987241548));
    host.origins[2][2]   = 1.2;
    host.origins[3][0]   = 11;
    host.origins[3][1]   = 3;
    host.maximum[3]      = 0;
    host.origins[4][1]   = .3;
    host.origins[4][2]   = .6;
    host.minimum[5] = host.maximum[5] = 1;
    host.maximum[6]                   = 3;
    host.rotations[7][3]              = 0;
    host.minimum[8]                   = 2;
    host.maximum[8]                   = 1;
    host.origins[9][0]                = 8;
    host.maximum[9]                   = 0;
    host.origins[10][1]               = 3;
    host.origins[11][1]               = 50;
    const auto device                 = cuda.allocate(sizeof(host));
    cuda.uploadAsync(device, &host, sizeof(host), producer);
    cuda.record(producerReady, producer);
    int64_t rows[] = {count}, vectors[] = {count, 3}, quaternions[] = {count, 4}, rowStride[] = {1},
            vectorStride[] = {3, 1}, rotationStride[] = {4, 1};
    const auto       at = [&](size_t offset) { return device + offset; };
    CudaFamilyLaunch launch{};
    launch.queryCount = count;
    launch.stream     = reinterpret_cast<uintptr_t>(producer);
    launch.waitEvent  = reinterpret_cast<uintptr_t>(producerReady);
    launch.positions  = cudaTensor(at(offsetof(Queries, origins)), {kDLFloat, 64, 1}, 2, vectors, vectorStride);
    launch.directionsOrRotations =
        cudaTensor(at(offsetof(Queries, directions)), {kDLFloat, 32, 1}, 2, vectors, vectorStride);
    launch.rotations = cudaTensor(at(offsetof(Queries, rotations)), {kDLFloat, 32, 1}, 2, quaternions, rotationStride);
    launch.minimumDistances = cudaTensor(at(offsetof(Queries, minimum)), {kDLFloat, 64, 1}, 1, rows, rowStride);
    launch.maximumDistances = cudaTensor(at(offsetof(Queries, maximum)), {kDLFloat, 64, 1}, 1, rows, rowStride);
    int64_t bitRows[]{rows[0], 2};
    launch.queryBits = cudaBitsTensor(at(offsetof(Queries, bits)), bitRows);
    for (auto precision : {CudaFamilyPrecision::exact, CudaFamilyPrecision::boundsCandidate})
    {
        CudaFamilyPlanConfig config{};
        config.family              = CudaFamilyKind::boxSweep;
        config.precision           = precision;
        config.hitMode             = CudaFamilyHitMode::closest;
        config.maximumQueryCount   = count;
        config.maximumHitsPerQuery = 1;
        config.hitCapacityPerSlot  = count;
        config.resultSlotCount     = 1;
        config.planHalfExtents[0]  = .5;
        config.planHalfExtents[1]  = .25;
        config.planHalfExtents[2]  = .5;
        config.resultFields        = kCudaFamilyResultPosition | kCudaFamilyResultNormal | kCudaFamilyResultDistance |
                              kCudaFamilyResultGeometryIdentity | kCudaFamilyResultQueryPosition;
        if (precision == CudaFamilyPrecision::exact)
        {
            config.resultFields |= kCudaFamilyResultSourceFeature;
        }
        bool             expected[count]{};
        ExactGeometryHit cpu[count]{};
        for (size_t row = 0; row < count; ++row)
        {
            if (row == 7 || row == 8)
                continue;
            ovspatialquery_overlap_box_t query{};
            std::copy_n(host.origins[row], 3, query.center);
            std::copy_n(host.rotations[row], 4, query.rotation_xyzw);
            std::copy_n(config.planHalfExtents, 3, query.half_extents);
            ExactGeometryDescription description;
            if (precision == CudaFamilyPrecision::exact)
                require(objects[family[row]].exactGeometry->exportDescription(description), "box CPU export");
            else
            {
                BoxExactGeometryDescription bounds{};
                for (size_t a = 0; a < 3; ++a)
                {
                    bounds.center[a] =
                        (objects[family[row]].bounds.minimum[a] + objects[family[row]].bounds.maximum[a]) * .5;
                    bounds.halfExtents[a] =
                        (objects[family[row]].bounds.maximum[a] - objects[family[row]].bounds.minimum[a]) * .5;
                    bounds.axes[a][a] = 1;
                }
                description = bounds;
            }
            require(sweepBoxGeometry(description, makeQueryBoxGeometry(query), host.directions[row], host.minimum[row],
                                     host.maximum[row], expected[row], cpu[row]) == OVSPATIALQUERY_API_SUCCESS,
                    "box CPU reference");
        }
        require(expected[0] && expected[1] && expected[2] && expected[3] && expected[4] && expected[5] &&
                    !expected[6] && expected[9] && expected[10] && !expected[11],
                "box analytic membership");
        if (precision == CudaFamilyPrecision::exact)
        {
            require(std::abs(cpu[0].distance - (1.5 - std::sqrt(.5))) < 1e-12, "sphere corner analytic time");
            require(std::abs(cpu[2].distance - (7.5 - std::sqrt(.51))) < 1e-12, "capsule interior analytic time");
            require(cpu[3].distance == 0 && cpu[5].distance == 1 && cpu[9].distance == 0 && cpu[4].distance == 13.5 &&
                        cpu[10].distance == 8.5,
                    "box clipped/initial/plane analytic time");
        }
        GenerationLease lease;
        require(populations.pin(population, lease) == OVSPATIALQUERY_API_SUCCESS, "box CUDA pin");
        std::shared_ptr<CudaFamilyPlan> plan;
        require(createCudaFamilyPlan(config, plan) == OVSPATIALQUERY_API_SUCCESS, "box CUDA plan");
        std::shared_ptr<CudaFamilyOperation> operation;
        require(prepareAndLaunchCudaFamilyPlan(plan, 0, launch, std::move(lease), operation) ==
                    OVSPATIALQUERY_API_SUCCESS,
                "box CUDA launch");
        const auto ready = operation->reservation().resultReady.value;
        cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(ready));
        cuda.synchronize(reinterpret_cast<CudaDriver::Event>(ready));
        CudaFamilyResultColumns result{};
        require(operation->completedResult(result) == CudaFamilyCompletedResultOutcome::success, "box CUDA completion");
        uint32_t statuses[count]{}, features[count]{}, corners[count][3]{};
        uint64_t offsets[count]{}, counts[count]{}, ids[count]{}, faces[count]{};
        double   distances[count]{}, positions[count][3]{}, queryPositions[count][3]{};
        float    normals[count][3]{}, bary[count][3]{};
        cuda.downloadAsync(statuses, deviceAddress(result.queryStatus), sizeof(statuses), consumer);
        cuda.downloadAsync(offsets, deviceAddress(result.hitOffsets), sizeof(offsets), consumer);
        cuda.downloadAsync(counts, deviceAddress(result.hitCounts), sizeof(counts), consumer);
        cuda.downloadAsync(ids, deviceAddress(result.objectIds), sizeof(ids), consumer);
        cuda.downloadAsync(distances, deviceAddress(result.distances), sizeof(distances), consumer);
        cuda.downloadAsync(positions, deviceAddress(result.positions), sizeof(positions), consumer);
        cuda.downloadAsync(normals, deviceAddress(result.normals), sizeof(normals), consumer);
        cuda.downloadAsync(queryPositions, deviceAddress(result.queryPositions), sizeof(queryPositions), consumer);
        if (precision == CudaFamilyPrecision::exact)
        {
            cuda.downloadAsync(features, deviceAddress(result.sourceFeatureKinds), sizeof(features), consumer);
            cuda.downloadAsync(faces, deviceAddress(result.sourceFaceIndices), sizeof(faces), consumer);
            cuda.downloadAsync(corners, deviceAddress(result.sourceFaceCorners), sizeof(corners), consumer);
            cuda.downloadAsync(bary, deviceAddress(result.sourceBarycentrics), sizeof(bary), consumer);
        }
        const auto done = cuda.createEvent();
        cuda.record(done, consumer);
        cuda.synchronize(done);
        for (size_t row = 0; row < count; ++row)
        {
            if (row == 7 || row == 8)
            {
                require(statuses[row] == cuda_family::kBatchRowInvalidArgument && counts[row] == 0,
                        "box malformed row isolation");
                continue;
            }
            require(statuses[row] == cuda_family::kBatchRowSuccess && counts[row] == unsigned(expected[row]),
                    "box CPU CUDA membership");
            if (!expected[row])
                continue;
            const auto h = offsets[row];
            require(h < count && ids[h] == family[row] + 1, "box CUDA identity");
            require(std::abs(distances[h] - cpu[row].distance) < 1e-9, "box CUDA analytic distance");
            for (int a = 0; a < 3; ++a)
            {
                require(std::isfinite(positions[h][a]) && std::isfinite(normals[h][a]), "box CUDA finite contact");
                require(std::abs(queryPositions[h][a] -
                                 (host.origins[row][a] + host.directions[row][a] * distances[h])) < 1e-9,
                        "box CUDA query position");
            }
            if (precision == CudaFamilyPrecision::exact && family[row] == 4)
            {
                require(features[h] == cuda_family::kFeatureTriangle && faces[h] == cpu[row].sourceFace,
                        "box CUDA authored face");
                for (int a = 0; a < 3; ++a)
                    require(corners[h][a] == cpu[row].sourceCorners[a], "box CUDA authored corners");
                require(std::abs(bary[h][1] - cpu[row].uv[0]) < 1e-6 && std::abs(bary[h][2] - cpu[row].uv[1]) < 1e-6,
                        "box CUDA scaled barycentrics");
            }
        }
        retireConsumer(*operation, done, "box CUDA consumer retirement failed");
        operation.reset();
        plan.reset();
    }
    std::shared_ptr<void> retired;
    require(populations.release(population, retired) == OVSPATIALQUERY_API_SUCCESS, "box CUDA population release");
    ovspatialquery_test_oracle_destroy_runtime(runtime);
    cuda.free(device);
    cuda.destroyEvent(producerReady);
    cuda.destroyStream(consumer);
    cuda.destroyStream(producer);
}

void testBoxSweepKeepsSmallFeaturesAtLargeTranslation()
{
    CudaDriver                        cuda;
    const auto                        producer = cuda.createStream(), consumer = cuda.createStream();
    const auto                        producerReady = cuda.createEvent();
    ovspatialquery_backend_runtime_t* runtime       = nullptr;
    require(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS,
            "translated box CUDA runtime");
    PopulationTable                      populations(runtime);
    constexpr double                     placement       = 0x1p59 + 0x1p9; // no float; a double ulp of 128
    const auto                           hull            = unitCubeHull();
    const std::array<double, 16>         convexTransform = {2, 0, 0, placement, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1};
    std::shared_ptr<const ExactGeometry> convex;
    require(createConvexExactGeometry(hull, convexTransform, convex) == ConvexConstructionOutcome::success,
            "translated box CUDA convex");
    TriangleMeshSource mesh{};
    mesh.key           = 984;
    mesh.vertices      = {-2, -2, 0, 2, -2, 0, 2, 2, 0, -2, 2, 0};
    mesh.triangles     = {0, 1, 2, 0, 2, 3};
    mesh.sourceFaces   = {17, 18};
    mesh.sourceCorners = {0, 1, 2, 0, 2, 3};
    std::shared_ptr<const PreparedTriangleMesh> preparedMesh;
    require(prepareTriangleMesh(mesh, preparedMesh) == OVSPATIALQUERY_API_SUCCESS, "translated box CUDA mesh");
    std::shared_ptr<const TriangleObjectGeometry> placementMesh;
    const std::array<double, 16> meshTransform = {1, 0, 0, placement, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    require(bindPreparedTriangleGeometry(preparedMesh, meshTransform, true, placementMesh) ==
                OVSPATIALQUERY_API_SUCCESS,
            "translated box CUDA mesh placement");
    std::shared_ptr<const ExactGeometry> exactMesh;
    require(createTriangleMeshExactGeometry(placementMesh, exactMesh) == OVSPATIALQUERY_API_SUCCESS,
            "translated box CUDA mesh geometry");
    PopulationKey population;
    require(populations.create(2, 31, {exactObject(1, 1, convex, 1), exactObject(2, 2, exactMesh, 2)}, population) ==
                OVSPATIALQUERY_API_SUCCESS,
            "translated box CUDA population");

    struct Queries
    {
        uint64_t bits[2][2]{{1, 0}, {2, 0}};
        double   centers[2][3]{{placement, 0, 5}, {placement, 0, 5}};
        float    directions[2][3]{{0, 0, -1}, {0, 0, -1}};
        float    rotations[2][4]{{0, float(std::sin(.3926990816987241548)), 0, float(std::cos(.3926990816987241548))},
                                 {0, 0, 0, 1}};
        double   maximum[2]{10, 10};
    } host;
    const auto device = cuda.allocate(sizeof(host));
    cuda.uploadAsync(device, &host, sizeof(host), producer);
    cuda.record(producerReady, producer);
    int64_t rows[]{2}, vectors[]{2, 3}, quaternions[]{2, 4}, rowStride[]{1}, vectorStride[]{3, 1},
        rotationStride[]{4, 1};
    const auto       at = [&](size_t offset) { return device + offset; };
    CudaFamilyLaunch launch{};
    launch.queryCount = 2;
    int64_t bitRows[]{rows[0], 2};
    launch.queryBits = cudaBitsTensor(at(offsetof(Queries, bits)), bitRows);
    launch.positions = cudaTensor(at(offsetof(Queries, centers)), {kDLFloat, 64, 1}, 2, vectors, vectorStride);
    launch.directionsOrRotations =
        cudaTensor(at(offsetof(Queries, directions)), {kDLFloat, 32, 1}, 2, vectors, vectorStride);
    launch.rotations = cudaTensor(at(offsetof(Queries, rotations)), {kDLFloat, 32, 1}, 2, quaternions, rotationStride);
    launch.maximumDistances = cudaTensor(at(offsetof(Queries, maximum)), {kDLFloat, 64, 1}, 1, rows, rowStride);
    launch.stream           = reinterpret_cast<uintptr_t>(producer);
    launch.waitEvent        = reinterpret_cast<uintptr_t>(producerReady);
    CudaFamilyPlanConfig config{};
    config.family              = CudaFamilyKind::boxSweep;
    config.precision           = CudaFamilyPrecision::exact;
    config.hitMode             = CudaFamilyHitMode::closest;
    config.resultFields        = kCudaFamilyResultDistance;
    config.maximumQueryCount   = 2;
    config.maximumHitsPerQuery = 1;
    config.hitCapacityPerSlot  = 2;
    config.resultSlotCount     = 1;
    config.planHalfExtents[0]  = .25;
    config.planHalfExtents[1]  = .25;
    config.planHalfExtents[2]  = .5;
    GenerationLease lease;
    require(populations.pin(population, lease) == OVSPATIALQUERY_API_SUCCESS, "translated box CUDA pin");
    std::shared_ptr<CudaFamilyPlan> plan;
    require(createCudaFamilyPlan(config, plan) == OVSPATIALQUERY_API_SUCCESS, "translated box CUDA plan");
    std::shared_ptr<CudaFamilyOperation> operation;
    require(prepareAndLaunchCudaFamilyPlan(plan, 0, launch, std::move(lease), operation) == OVSPATIALQUERY_API_SUCCESS,
            "translated box CUDA launch");
    const auto columns = operation->reservation();
    cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(columns.resultReady.value));
    uint32_t status[2]{};
    uint64_t offsets[2]{}, counts[2]{};
    double   distance[2]{};
    cuda.downloadAsync(status, deviceAddress(columns.queryStatus), sizeof(status), consumer);
    cuda.downloadAsync(offsets, deviceAddress(columns.hitOffsets), sizeof(offsets), consumer);
    cuda.downloadAsync(counts, deviceAddress(columns.hitCounts), sizeof(counts), consumer);
    cuda.downloadAsync(distance, deviceAddress(columns.distances), sizeof(distance), consumer);
    const auto done = cuda.createEvent();
    cuda.record(done, consumer);
    cuda.synchronize(done);
    const double convexExpected = 3.0 - (.25 + .5) * std::sqrt(.5);
    require(status[0] == cuda_family::kBatchRowSuccess && status[1] == cuda_family::kBatchRowSuccess &&
                counts[0] == 1 && counts[1] == 1 && offsets[0] < 2 && offsets[1] < 2 &&
                std::abs(distance[offsets[0]] - convexExpected) < 1e-5 && std::abs(distance[offsets[1]] - 4.5) < 1e-9,
            "translated box CUDA retained small convex or mesh feature");
    operation->releaseReadyDependencies();
    retireConsumer(*operation, done, "translated box CUDA consumer retirement failed");
    operation.reset();
    plan.reset();
    std::shared_ptr<void> retired;
    require(populations.release(population, retired) == OVSPATIALQUERY_API_SUCCESS,
            "translated box CUDA population release");
    ovspatialquery_test_oracle_destroy_runtime(runtime);
    cuda.free(device);
    cuda.destroyEvent(producerReady);
    cuda.destroyStream(consumer);
    cuda.destroyStream(producer);
}

// Xorshift64 draws in [low, high); each traversal test seeds its own stream.
struct Draw
{
    uint64_t state;

    double operator()(double low, double high)
    {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return low + (high - low) * (double(state >> 11) * 0x1.0p-53);
    }
};

constexpr uint32_t kSoupSide = 16;

// Numbers every triangle's source face 1000 + t with corners 3t, 3t + 1 and 3t + 2.
void numberSourceFeatures(TriangleMeshSource& mesh)
{
    for (uint32_t t = 0; t < mesh.triangles.size() / 3; ++t)
    {
        mesh.sourceFaces.push_back(1000 + t);
        mesh.sourceCorners.insert(mesh.sourceCorners.end(), {3 * t, 3 * t + 1, 3 * t + 2});
    }
}

// The traversal tests' meshes, keyed from `key`. A tessellated plane gives shared edges and
// equal-time ties and a random soup adds slivers, initial overlaps and near misses; together
// they span many BVH leaves. A tilted floor patch (z = 0.3 y) lies 1e4 from its mesh origin,
// so float projections of its corners round by about 1e-4, far beyond a 1 um contact. A fine
// float-exact plane is numbered from its far corner, so the lowest index of a tie lands in
// the last warp batch.
struct TraversalMeshes
{
    TriangleMeshSource soup, floor, fine;
};

TraversalMeshes traversalMeshes(uint64_t key, Draw& between)
{
    TraversalMeshes     meshes;
    TriangleMeshSource& soup = meshes.soup;
    TriangleMeshSource& fine = meshes.fine;
    soup.key                 = key;
    meshes.floor.key         = key + 1;
    fine.key                 = key + 2;
    constexpr uint32_t side  = kSoupSide;
    for (uint32_t y = 0; y <= side; ++y)
        for (uint32_t x = 0; x <= side; ++x)
            soup.vertices.insert(soup.vertices.end(), {float(-3 + 6.0 * x / side), float(-3 + 6.0 * y / side), 0.f});
    for (uint32_t y = 0; y < side; ++y)
        for (uint32_t x = 0; x < side; ++x)
        {
            const uint32_t corner = y * (side + 1) + x;
            soup.triangles.insert(soup.triangles.end(), {corner, corner + 1, corner + side + 2, corner,
                                                         corner + side + 2, corner + side + 1});
        }
    for (int t = 0; t < 500; ++t)
    {
        const auto   first   = uint32_t(soup.vertices.size() / 3);
        const double size    = t % 4 ? .5 : .002;
        const double base[3] = {between(-3, 3), between(-3, 3), between(-3, 3)};
        for (int v = 0; v < 3; ++v)
            for (int a = 0; a < 3; ++a)
                soup.vertices.push_back(float(base[a] + between(-size, size)));
        soup.triangles.insert(soup.triangles.end(), {first, first + 1, first + 2});
    }
    meshes.floor.vertices  = {-20, 9980, 2994, 20, 9980, 2994, 20, 10020, 3006, -20, 10020, 3006};
    meshes.floor.triangles = {0, 1, 2, 0, 2, 3};
    for (uint32_t y = 0; y <= 64; ++y)
        for (uint32_t x = 0; x <= 64; ++x)
            fine.vertices.insert(fine.vertices.end(), {float(-1 + x / 32.0), float(-1 + y / 32.0), 0.f});
    for (uint32_t y = 64; y-- > 0;)
        for (uint32_t x = 64; x-- > 0;)
        {
            const uint32_t corner = y * 65 + x;
            fine.triangles.insert(fine.triangles.end(),
                                  {corner, corner + 1, corner + 66, corner, corner + 66, corner + 65});
        }
    for (TriangleMeshSource* mesh : {&soup, &meshes.floor, &fine})
    {
        numberSourceFeatures(*mesh);
    }
    return meshes;
}

// Identity and rigid placements with uniform scale about each axis, one far away, for the
// soup; identity and rotated half scale for the floor; identity for the fine plane.
std::array<std::array<double, 16>, 8> traversalPlacements()
{
    const double c = std::cos(.7), s = std::sin(.7), c4 = std::cos(.4), s4 = std::sin(.4);
    return {{
        {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1},
        {.37 * c, -.37 * s, 0, 3, .37 * s, .37 * c, 0, -2, 0, 0, .37, 1, 0, 0, 0, 1},
        {2.5 * c, 0, 2.5 * s, 1e6, 0, 2.5, 0, -2e6, -2.5 * s, 0, 2.5 * c, 5e5, 0, 0, 0, 1},
        {2, 0, 0, 5, 0, 2 * c4, -2 * s4, 0, 0, 2 * s4, 2 * c4, 0, 0, 0, 0, 1},
        {1.5 * c, -1.5 * s, 0, 0, 1.5 * s, 1.5 * c, 0, 0, 0, 0, 1.5, 0, 0, 0, 0, 1},
        {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1},
        {.5 * c4, -.5 * s4, 0, 7, .5 * s4, .5 * c4, 0, -3, 0, 0, .5, 2, 0, 0, 0, 1},
        {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1},
    }};
}

// The device box traversal culls nodes and triangles with float face-axis bounds,
// visits them nearest first and computes only the winner's contact; the CPU's
// exhaustive SAT scan over every triangle is the reference. Presence and distance
// agree; since the device contracts multiply-adds, a different face is accepted only
// as a tie that the host SAT confirms on that face, and every contact lies on its
// face and on the moved box.
void testBoxSweepTraversalMatchesExhaustiveScan()
{
    CudaDriver                        cuda;
    const auto                        producer = cuda.createStream(), consumer = cuda.createStream();
    const auto                        producerReady = cuda.createEvent();
    ovspatialquery_backend_runtime_t* runtime       = nullptr;
    require(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS,
            "box traversal CUDA runtime");
    PopulationTable       populations(runtime);
    Draw                  between{0x9e3779b97f4a7c15ull};
    const TraversalMeshes meshes = traversalMeshes(985, between);
    const auto &          soup = meshes.soup, &floor = meshes.floor, &fine = meshes.fine;
    struct Target
    {
        const TriangleMeshSource* mesh;
        std::array<double, 16>    matrix;
        bool                      doubleSided;
        ExactGeometryDescription  exhaustive;
    };
    const auto          placements = traversalPlacements();
    const bool          sided[]    = {true, false, true, false, true, true, true, true};
    std::vector<Target> targets;
    for (size_t p = 0; p < placements.size(); ++p)
        targets.push_back({p < 5 ? &soup : p < 7 ? &floor : &fine, placements[p], sided[p], {}});
    std::shared_ptr<const PreparedTriangleMesh> preparedSoup, preparedFloor, preparedFine;
    require(prepareTriangleMesh(soup, preparedSoup) == OVSPATIALQUERY_API_SUCCESS &&
                prepareTriangleMesh(floor, preparedFloor) == OVSPATIALQUERY_API_SUCCESS &&
                prepareTriangleMesh(fine, preparedFine) == OVSPATIALQUERY_API_SUCCESS,
            "box traversal meshes");
    std::vector<PopulationObject> objects;
    for (size_t p = 0; p < targets.size(); ++p)
    {
        const auto&                                   prepared = targets[p].mesh == &soup    ? preparedSoup
                                                                 : targets[p].mesh == &floor ? preparedFloor
                                                                                             : preparedFine;
        std::shared_ptr<const TriangleObjectGeometry> placement;
        require(bindPreparedTriangleGeometry(prepared, targets[p].matrix, targets[p].doubleSided, placement) ==
                    OVSPATIALQUERY_API_SUCCESS,
                "box traversal placement");
        std::shared_ptr<const ExactGeometry> geometry;
        require(createTriangleMeshExactGeometry(placement, geometry) == OVSPATIALQUERY_API_SUCCESS,
                "box traversal geometry");
        require(geometry->exportDescription(targets[p].exhaustive), "box traversal export");
        auto* triangles = std::get_if<TriangleMeshExactGeometryDescription>(&targets[p].exhaustive);
        require(triangles != nullptr, "box traversal mesh description");
        triangles->owner = nullptr; // every triangle, without the gather or the bound
        objects.push_back(exactObject(p + 1, p + 1, geometry, uint64_t{1} << p));
    }
    PopulationKey population;
    require(populations.create(2, 31, objects, population) == OVSPATIALQUERY_API_SUCCESS, "box traversal population");

    // Rows 0..499 cycle through the soup placements; four floor rows per floor placement
    // and four rows resting on the fine plane follow.
    static constexpr size_t soupRows = 500, fineRows = soupRows + 8, count = fineRows + 4;
    struct Queries
    {
        uint64_t bits[count][2]{};
        double   origins[count][3]{}, minimum[count]{}, maximum[count]{};
        float    directions[count][3]{}, rotations[count][4]{};
    };
    const auto host     = std::make_unique<Queries>();
    const auto targetOf = [](size_t row) {
        return row < soupRows ? row % 5 : row < fineRows ? 5 + (row - soupRows) / 4 : size_t(7);
    };
    const std::array<std::array<double, 3>, 2> plans = {{{.3, .15, .45}, {.05, .6, .2}}};
    for (size_t row = 0; row < soupRows; ++row)
    {
        const size_t i       = row / 5;
        const bool   aligned = i % 4 == 0;
        const auto&  matrix  = targets[targetOf(row)].matrix;
        // Aligned rows start above the plane and slide or drop; the others aim from
        // outside the soup at a point inside it, so most make a moving first contact.
        const double local[3]  = {between(-4, 4), between(-4, 4), aligned ? between(.05, 1) : between(-5, 5)};
        const double target[3] = {between(-2.5, 2.5), between(-2.5, 2.5), between(-2.5, 2.5)};
        double       reach     = 0;
        for (int a = 0; a < 3; ++a)
            host->origins[row][a] = matrix[4 * a] * local[0] + matrix[4 * a + 1] * local[1] +
                                    matrix[4 * a + 2] * local[2] + matrix[4 * a + 3];
        if (aligned)
        {
            host->rotations[row][3]          = 1;
            host->directions[row][i / 4 % 3] = i % 8 ? -1.f : 1.f;
            reach                            = between(0, 6);
        }
        else
        {
            double q[4], d[3], qLength = 0;
            for (double& value : q)
            {
                value = between(-1, 1);
                qLength += value * value;
            }
            for (int a = 0; a < 3; ++a)
            {
                d[a] = 0;
                for (int b = 0; b < 3; ++b)
                    d[a] += matrix[4 * a + b] * (target[b] - local[b]);
                reach += d[a] * d[a];
            }
            reach = std::sqrt(reach);
            for (int a = 0; a < 4; ++a)
                host->rotations[row][a] = float(q[a] / std::sqrt(qLength));
            for (int a = 0; a < 3; ++a)
                host->directions[row][a] = float(d[a] / reach);
        }
        host->minimum[row] = i % 5 == 0 ? between(0, 1) : 0;
        host->maximum[row] = host->minimum[row] + reach * between(.5, 1.5);
    }
    // Under the first plan's box these rest exactly on the unplaced plane: touching at the start.
    const auto resting = [](size_t row) { return row < soupRows && row % 20 == 0 && row != 0 && row <= 160; };
    for (size_t row = 0; row < soupRows; ++row)
        if (resting(row))
        {
            host->origins[row][0] = between(-2, 2);
            host->origins[row][1] = between(-2, 2);
            host->origins[row][2] = .45;
            host->minimum[row]    = 0;
            host->maximum[row]    = 2;
        }
    // Under the first plan, floor rows sink 1 um (in z - 0.3 y) into the floor 2.5 box
    // widths back in mesh-local x, then slide along it or fall a lift onto it. Their
    // third face axis follows the floor's normal, so a face axis alone must keep them:
    // float projections of the floor's corners round by about 1e-4 of these units.
    const auto multiply = [](const double a[4], const double b[4], double out[4])
    {
        out[0] = a[3] * b[0] + b[3] * a[0] + a[1] * b[2] - a[2] * b[1];
        out[1] = a[3] * b[1] + b[3] * a[1] + a[2] * b[0] - a[0] * b[2];
        out[2] = a[3] * b[2] + b[3] * a[2] + a[0] * b[1] - a[1] * b[0];
        out[3] = a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2];
    };
    for (size_t row = soupRows; row < fineRows; ++row)
    {
        const auto&  matrix = targets[targetOf(row)].matrix;
        const double scale  = matrix[10];
        const size_t k      = row - soupRows;
        const double yaw = std::atan2(matrix[4], matrix[0]), tilt = std::atan(.3), spin = k % 2 ? .6 : 0;
        const double normal[3]      = {std::sin(yaw) * std::sin(tilt), -std::cos(yaw) * std::sin(tilt), std::cos(tilt)};
        const double aboutX[4]      = {std::sin(tilt / 2), 0, 0, std::cos(tilt / 2)};
        const double aboutZ[4]      = {0, 0, std::sin(yaw / 2), std::cos(yaw / 2)};
        const double aboutNormal[4] = {normal[0] * std::sin(spin / 2), normal[1] * std::sin(spin / 2),
                                       normal[2] * std::sin(spin / 2), std::cos(spin / 2)};
        double       tilted[4], turn[4];
        multiply(aboutZ, aboutX, tilted);
        multiply(aboutNormal, tilted, turn);
        ovspatialquery_overlap_box_t box{};
        for (int a = 0; a < 4; ++a)
            host->rotations[row][a] = box.rotation_xyzw[a] = float(turn[a]);
        std::copy_n(plans[0].data(), 3, box.half_extents);
        const auto query   = makeQueryBoxGeometry(box);
        double     support = 0;
        for (unsigned j = 0; j < 3; ++j)
        {
            double local[3]{};
            for (unsigned a = 0; a < 3; ++a)
                for (unsigned b = 0; b < 3; ++b)
                    local[a] += matrix[4 * b + a] * query.axes[j][b] / (scale * scale);
            support += plans[0][j] * std::abs(local[2] - .3 * local[1]);
        }
        const bool   falls    = k / 2 % 2 == 1;
        const double lift     = falls ? 1 : 0;
        const double local[3] = {-2.5, 1e4, 3e3 + support - 1e-6 + lift / scale};
        for (unsigned a = 0; a < 3; ++a)
            host->origins[row][a] = matrix[4 * a] * local[0] + matrix[4 * a + 1] * local[1] +
                                    matrix[4 * a + 2] * local[2] + matrix[4 * a + 3];
        host->directions[row][0] = falls ? 0.f : float(matrix[0] / scale);
        host->directions[row][1] = falls ? 0.f : float(matrix[4] / scale);
        host->directions[row][2] = falls ? -1.f : 0.f;
        host->maximum[row]       = falls ? 2 : 10;
    }
    for (size_t row = fineRows; row < count; ++row)
    {
        const size_t k               = row - fineRows;
        host->origins[row][0]        = between(-.5, .5);
        host->origins[row][1]        = between(-.5, .5);
        host->origins[row][2]        = .45;
        host->rotations[row][3]      = 1;
        host->directions[row][k % 2] = k / 2 ? -1.f : 1.f;
        host->maximum[row]           = 1;
    }
    for (size_t row = 0; row < count; ++row)
        host->bits[row][0] = uint64_t{1} << targetOf(row);
    const auto device = cuda.allocate(sizeof(Queries));
    cuda.uploadAsync(device, host.get(), sizeof(Queries), producer);
    cuda.record(producerReady, producer);
    int64_t rows[] = {count}, vectors[] = {count, 3}, quaternions[] = {count, 4}, rowStride[] = {1},
            vectorStride[] = {3, 1}, rotationStride[] = {4, 1};
    const auto       at = [&](size_t offset) { return device + offset; };
    CudaFamilyLaunch launch{};
    launch.queryCount = count;
    launch.stream     = reinterpret_cast<uintptr_t>(producer);
    launch.waitEvent  = reinterpret_cast<uintptr_t>(producerReady);
    launch.positions  = cudaTensor(at(offsetof(Queries, origins)), {kDLFloat, 64, 1}, 2, vectors, vectorStride);
    launch.directionsOrRotations =
        cudaTensor(at(offsetof(Queries, directions)), {kDLFloat, 32, 1}, 2, vectors, vectorStride);
    launch.rotations = cudaTensor(at(offsetof(Queries, rotations)), {kDLFloat, 32, 1}, 2, quaternions, rotationStride);
    launch.minimumDistances = cudaTensor(at(offsetof(Queries, minimum)), {kDLFloat, 64, 1}, 1, rows, rowStride);
    launch.maximumDistances = cudaTensor(at(offsetof(Queries, maximum)), {kDLFloat, 64, 1}, 1, rows, rowStride);
    int64_t bitRows[]{rows[0], 2};
    launch.queryBits = cudaBitsTensor(at(offsetof(Queries, bits)), bitRows);

    size_t hits = 0, starts = 0, ties = 0;
    for (size_t planIndex = 0; planIndex < plans.size(); ++planIndex)
    {
        const auto&          half = plans[planIndex];
        CudaFamilyPlanConfig config{};
        config.family              = CudaFamilyKind::boxSweep;
        config.precision           = CudaFamilyPrecision::exact;
        config.hitMode             = CudaFamilyHitMode::closest;
        config.maximumQueryCount   = count;
        config.maximumHitsPerQuery = 1;
        config.hitCapacityPerSlot  = count;
        config.resultSlotCount     = 1;
        std::copy(half.begin(), half.end(), config.planHalfExtents);
        config.resultFields = kCudaFamilyResultPosition | kCudaFamilyResultNormal | kCudaFamilyResultDistance |
                              kCudaFamilyResultGeometryIdentity | kCudaFamilyResultSourceFeature;
        GenerationLease lease;
        require(populations.pin(population, lease) == OVSPATIALQUERY_API_SUCCESS, "box traversal pin");
        std::shared_ptr<CudaFamilyPlan> plan;
        require(createCudaFamilyPlan(config, plan) == OVSPATIALQUERY_API_SUCCESS, "box traversal plan");
        std::shared_ptr<CudaFamilyOperation> operation;
        require(prepareAndLaunchCudaFamilyPlan(plan, 0, launch, std::move(lease), operation) ==
                    OVSPATIALQUERY_API_SUCCESS,
                "box traversal launch");
        const auto ready = operation->reservation().resultReady.value;
        cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(ready));
        cuda.synchronize(reinterpret_cast<CudaDriver::Event>(ready));
        CudaFamilyResultColumns result{};
        require(operation->completedResult(result) == CudaFamilyCompletedResultOutcome::success,
                "box traversal completion");
        std::vector<uint32_t> statuses(count), features(count), corners(3 * count);
        std::vector<uint64_t> offsets(count), counts(count), ids(count), faces(count);
        std::vector<double>   distances(count), positions(3 * count);
        std::vector<float>    normals(3 * count), bary(3 * count);
        cuda.downloadAsync(statuses.data(), deviceAddress(result.queryStatus), 4 * count, consumer);
        cuda.downloadAsync(offsets.data(), deviceAddress(result.hitOffsets), 8 * count, consumer);
        cuda.downloadAsync(counts.data(), deviceAddress(result.hitCounts), 8 * count, consumer);
        cuda.downloadAsync(ids.data(), deviceAddress(result.objectIds), 8 * count, consumer);
        cuda.downloadAsync(distances.data(), deviceAddress(result.distances), 8 * count, consumer);
        cuda.downloadAsync(positions.data(), deviceAddress(result.positions), 24 * count, consumer);
        cuda.downloadAsync(normals.data(), deviceAddress(result.normals), 12 * count, consumer);
        cuda.downloadAsync(features.data(), deviceAddress(result.sourceFeatureKinds), 4 * count, consumer);
        cuda.downloadAsync(faces.data(), deviceAddress(result.sourceFaceIndices), 8 * count, consumer);
        cuda.downloadAsync(corners.data(), deviceAddress(result.sourceFaceCorners), 12 * count, consumer);
        cuda.downloadAsync(bary.data(), deviceAddress(result.sourceBarycentrics), 12 * count, consumer);
        const auto done = cuda.createEvent();
        cuda.record(done, consumer);
        cuda.synchronize(done);
        for (size_t row = 0; row < count; ++row)
        {
            const size_t                 p      = targetOf(row);
            const Target&                target = targets[p];
            ovspatialquery_overlap_box_t box{};
            std::copy_n(host->origins[row], 3, box.center);
            std::copy_n(host->rotations[row], 4, box.rotation_xyzw);
            std::copy_n(half.data(), 3, box.half_extents);
            const auto       query    = makeQueryBoxGeometry(box);
            bool             expected = false;
            ExactGeometryHit cpu{};
            require(sweepBoxGeometry(target.exhaustive, query, host->directions[row], host->minimum[row],
                                     host->maximum[row], expected, cpu) == OVSPATIALQUERY_API_SUCCESS,
                    "box traversal exhaustive reference");
            require(statuses[row] == cuda_family::kBatchRowSuccess && counts[row] == unsigned(expected),
                    "box traversal presence");
            if (planIndex == 0 && (resting(row) || row >= soupRows))
            {
                // Resting and sunk boxes start in contact; falling ones meet the floor a lift less the sink.
                const bool   falls  = row >= soupRows && row < fineRows && (row - soupRows) / 2 % 2 == 1;
                const double expect = falls ? 1 - 1e-6 * target.matrix[10] : 0;
                require(expected && std::abs(cpu.distance - expect) <= 1e-9 && offsets[row] < count &&
                            std::abs(distances[offsets[row]] - expect) <= 1e-9,
                        "box traversal resting, sunk or falling contact");
            }
            if (!expected)
                continue;
            ++hits;
            starts += cpu.distance == host->minimum[row];
            const auto h = offsets[row];
            require(h < count && ids[h] == p + 1 && std::abs(distances[h] - cpu.distance) <= 1e-9,
                    "box traversal distance");
            const uint32_t triangleCount = uint32_t(target.mesh->triangles.size() / 3);
            require(features[h] == cuda_family::kFeatureTriangle && faces[h] >= 1000 && faces[h] < 1000 + triangleCount,
                    "box traversal face");
            const uint32_t triangle = uint32_t(faces[h] - 1000);
            for (uint32_t a = 0; a < 3; ++a)
                require(corners[3 * h + a] == 3 * triangle + a, "box traversal corners");
            // The chosen face on the host, relative to the placement's translation.
            const auto&     matrix = target.matrix;
            box_sweep::Box  relative{};
            box_sweep::Vec3 vertices[3]{}, point{}, direction{};
            for (int a = 0; a < 3; ++a)
            {
                relative.center[a] = query.center[a] - matrix[4 * a + 3];
                relative.half[a]   = query.halfExtents[a];
                for (int b = 0; b < 3; ++b)
                    relative.axis[b][a] = query.axes[b][a];
                point[a]     = positions[3 * h + a] - matrix[4 * a + 3];
                direction[a] = host->directions[row][a];
            }
            for (int corner = 0; corner < 3; ++corner)
            {
                const float* local = target.mesh->vertices.data() + 3 * target.mesh->triangles[3 * triangle + corner];
                for (int a = 0; a < 3; ++a)
                    vertices[corner][a] =
                        std::fma(matrix[4 * a], double(local[0]),
                                 std::fma(matrix[4 * a + 1], double(local[1]), matrix[4 * a + 2] * double(local[2])));
            }
            // Axis-aligned rows on the unplaced plane round identically on host and device
            // (products by 0 and 1 only), so their ties must keep the lower index there too.
            const bool exact =
                p == 7 || (p == 0 && row < soupRows && (row / 5) % 4 == 0 &&
                           cpu.generatedTriangle < 2 * kSoupSide * kSoupSide && triangle < 2 * kSoupSide * kSoupSide);
            require(!exact || triangle == cpu.generatedTriangle, "box traversal broke the lower-index tie rule");
            // On the same face both paths take the same separating axis here, so a normal that
            // came from another lane's triangle cannot pass.
            for (int a = 0; a < 3 && triangle == cpu.generatedTriangle; ++a)
                require(std::abs(double(normals[3 * h + a]) - cpu.normal[a]) <= 1e-6, "box traversal normal");
            if (triangle != cpu.generatedTriangle)
            {
                box_sweep::Hit tied{};
                require(box_sweep::boxTriangle(relative, vertices[0], vertices[1], vertices[2], target.doubleSided,
                                               direction, host->minimum[row], host->maximum[row], tied) &&
                            std::abs(tied.distance - cpu.distance) <= 1e-9,
                        "box traversal face is not a confirmed tie");
                ++ties;
            }
            // On the face: its closest point and the reported barycentrics.
            double     u = 0, v = 0, scale = 1;
            const auto closest = box_sweep::triangleClosest(point, vertices[0], vertices[1], vertices[2], u, v);
            for (int a = 0; a < 3; ++a)
                for (const auto& vertex : vertices)
                    scale = std::max({scale, std::abs(point[a]), std::abs(vertex[a])});
            const auto gap     = box_sweep::sub(point, closest);
            const auto uvPoint = box_sweep::add(
                vertices[0], box_sweep::add(box_sweep::mul(box_sweep::sub(vertices[1], vertices[0]), bary[3 * h + 1]),
                                            box_sweep::mul(box_sweep::sub(vertices[2], vertices[0]), bary[3 * h + 2])));
            const auto uvGap = box_sweep::sub(point, uvPoint);
            require(std::sqrt(box_sweep::dot(gap, gap)) <= 1e-9 * scale &&
                        std::sqrt(box_sweep::dot(uvGap, uvGap)) <= 1e-5 * scale,
                    "box traversal contact off its face");
            // On the moved box, and a unit normal against the motion.
            box_sweep::Box moved = relative;
            moved.center         = box_sweep::at(relative.center, direction, distances[h]);
            const auto boxPoint  = box_sweep::localPoint(moved, point);
            double     outside = 0, normalLength = 0, facing = 0;
            for (int a = 0; a < 3; ++a)
            {
                outside = std::max(outside, std::abs(boxPoint[a]) - moved.half[a]);
                normalLength += double(normals[3 * h + a]) * normals[3 * h + a];
                facing += double(normals[3 * h + a]) * direction[a];
            }
            require(outside <= 1e-9 * scale && std::abs(normalLength - 1) <= 1e-5 && facing <= 1e-6,
                    "box traversal contact off the moved box");
        }
        retireConsumer(*operation, done, "box traversal consumer retirement failed");
        operation.reset();
        plan.reset();
    }
    // Enough contacts, including touching and initial overlap, for the comparison to mean something.
    std::printf("box traversal: %zu hits, %zu at the minimum, %zu confirmed ties\n", hits, starts, ties);
    require(hits > 500 && starts > 30, "box traversal coverage");
    std::shared_ptr<void> retired;
    require(populations.release(population, retired) == OVSPATIALQUERY_API_SUCCESS, "box traversal population release");
    ovspatialquery_test_oracle_destroy_runtime(runtime);
    cuda.free(device);
    cuda.destroyEvent(producerReady);
    cuda.destroyStream(consumer);
    cuda.destroyStream(producer);
}

// Sweeps 2,000 m long through lanes of objects listed farthest first, so a row fills its
// candidate slots from the far end and must lower its reach round by round. Lane z = 0
// starts with two boxes the sweeps meet at one distance, the lower id listed after every
// other object; lane z = 10 starts with a mesh listed last. Each CLOSEST and ANY row equals,
// contact included, the MULTIPLE row of its object, and exact rows meet the CPU sweeps.
void testLongSweepsStopAtTheirFirstContact()
{
    CudaDriver                        cuda;
    const auto                        producer = cuda.createStream(), consumer = cuda.createStream();
    const auto                        producerReady = cuda.createEvent();
    ovspatialquery_backend_runtime_t* runtime       = nullptr;
    require(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS, "long sweep runtime");
    PopulationTable                      populations(runtime);
    std::vector<PopulationObject>        objects;
    std::shared_ptr<const ExactGeometry> geometry;
    const auto                           addBox = [&](uint64_t id, std::array<double, 3> center)
    {
        BoxGeometrySource box{};
        box.center = center;
        box.axes   = {std::array<double, 3>{1, 0, 0}, std::array<double, 3>{0, 1, 0}, std::array<double, 3>{0, 0, 1}};
        box.halfExtents = {1, 1, 1};
        require(createBoxExactGeometry(box, geometry) == OVSPATIALQUERY_API_SUCCESS, "long sweep box");
        objects.push_back(exactObject(id, id, geometry, 1));
    };
    addBox(9, {6, 1.2, 0});
    for (uint64_t k = 40; k-- > 0;)
        for (const double z : {0.0, 10.0})
        {
            const std::array<double, 3> center{10.0 + 4.0 * double(k), 0, z};
            if (k % 2 == 0)
            {
                addBox(100 + 2 * k + (z != 0), center);
                continue;
            }
            SphereGeometrySource sphere{};
            sphere.center = center;
            sphere.radius = 1;
            require(createSphereExactGeometry(sphere, geometry) == OVSPATIALQUERY_API_SUCCESS, "long sweep sphere");
            objects.push_back(exactObject(100 + 2 * k + (z != 0), 100 + 2 * k + (z != 0), geometry, 1));
        }
    addBox(8, {6, -1.2, 0});
    TriangleMeshSource quad{};
    quad.key       = 7;
    quad.vertices  = {6, -1, 9, 6, -1, 11, 6, 1, 11, 6, 1, 9};
    quad.triangles = {0, 1, 2, 0, 2, 3};
    numberSourceFeatures(quad);
    std::shared_ptr<const PreparedTriangleMesh>   preparedQuad;
    std::shared_ptr<const TriangleObjectGeometry> placedQuad;
    require(prepareTriangleMesh(quad, preparedQuad) == OVSPATIALQUERY_API_SUCCESS &&
                bindPreparedTriangleGeometry(preparedQuad, {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, true,
                                             placedQuad) == OVSPATIALQUERY_API_SUCCESS &&
                createTriangleMeshExactGeometry(placedQuad, geometry) == OVSPATIALQUERY_API_SUCCESS,
            "long sweep mesh");
    objects.push_back(exactObject(7, 7, geometry, 1));
    PopulationKey population;
    require(populations.create(2, 31, objects, population) == OVSPATIALQUERY_API_SUCCESS, "long sweep population");

    // Rows: the tied boxes; the mesh; nothing above the lanes; a start between two lane
    // objects; nothing behind; a box turned 0.4 rad about z, which meets one box first.
    constexpr size_t count = 6, capacity = 64;
    struct Queries
    {
        uint64_t bits[count][2]{};
        double   origins[count][3]{{0, 0, 0}, {0, 0, 10}, {0, 0, 20}, {32, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        double   minimum[count]{};
        double   maximum[count]{2000, 2000, 2000, 2000, 2000, 2000};
        float    directions[count][3]{{1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {-1, 0, 0}, {1, 0, 0}};
        float    rotations[count][4]{{0, 0, 0, 1}, {0, 0, 0, 1}, {0, 0, 0, 1},
                                     {0, 0, 0, 1}, {0, 0, 0, 1}, {0, 0, float(std::sin(.2)), float(std::cos(.2))}};
    } host;
    const auto device = cuda.allocate(sizeof(host));
    cuda.uploadAsync(device, &host, sizeof(host), producer);
    cuda.record(producerReady, producer);
    int64_t rows[] = {count}, vectors[] = {count, 3}, quaternions[] = {count, 4}, rowStride[] = {1},
            vectorStride[] = {3, 1}, rotationStride[] = {4, 1}, bitRows[] = {count, 2};
    const auto       at = [&](size_t offset) { return device + offset; };
    CudaFamilyLaunch launch{};
    launch.queryCount = count;
    launch.stream     = reinterpret_cast<uintptr_t>(producer);
    launch.waitEvent  = reinterpret_cast<uintptr_t>(producerReady);
    launch.queryBits  = cudaBitsTensor(at(offsetof(Queries, bits)), bitRows);
    launch.positions  = cudaTensor(at(offsetof(Queries, origins)), {kDLFloat, 64, 1}, 2, vectors, vectorStride);
    launch.directionsOrRotations =
        cudaTensor(at(offsetof(Queries, directions)), {kDLFloat, 32, 1}, 2, vectors, vectorStride);
    launch.minimumDistances = cudaTensor(at(offsetof(Queries, minimum)), {kDLFloat, 64, 1}, 1, rows, rowStride);
    launch.maximumDistances = cudaTensor(at(offsetof(Queries, maximum)), {kDLFloat, 64, 1}, 1, rows, rowStride);

    struct Rows
    {
        uint64_t offsets[count]{}, counts[count]{}, ids[capacity * count]{};
        double   distances[capacity * count]{}, positions[capacity * count][3]{};
        float    normals[capacity * count][3]{};
    };
    const auto run = [&](CudaFamilyKind family, CudaFamilyPrecision precision, CudaFamilyHitMode mode, Rows& out)
    {
        CudaFamilyPlanConfig config{};
        config.family              = family;
        config.precision           = precision;
        config.hitMode             = mode;
        config.maximumQueryCount   = count;
        config.maximumHitsPerQuery = mode == CudaFamilyHitMode::multiple ? capacity : 1;
        config.hitCapacityPerSlot  = count * config.maximumHitsPerQuery;
        config.resultSlotCount     = 1;
        config.planRadius          = .5;
        config.planHalfExtents[0] = config.planHalfExtents[1] = config.planHalfExtents[2] = .5;
        config.resultFields        = kCudaFamilyResultPosition | kCudaFamilyResultNormal | kCudaFamilyResultDistance;
        CudaFamilyLaunch rowLaunch = launch;
        if (family == CudaFamilyKind::boxSweep)
            rowLaunch.rotations =
                cudaTensor(at(offsetof(Queries, rotations)), {kDLFloat, 32, 1}, 2, quaternions, rotationStride);
        GenerationLease lease;
        require(populations.pin(population, lease) == OVSPATIALQUERY_API_SUCCESS, "long sweep pin");
        std::shared_ptr<CudaFamilyPlan> plan;
        require(createCudaFamilyPlan(config, plan) == OVSPATIALQUERY_API_SUCCESS, "long sweep plan");
        std::shared_ptr<CudaFamilyOperation> operation;
        require(prepareAndLaunchCudaFamilyPlan(plan, 0, rowLaunch, std::move(lease), operation) ==
                    OVSPATIALQUERY_API_SUCCESS,
                "long sweep launch");
        const auto ready = operation->reservation().resultReady.value;
        cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(ready));
        CudaFamilyResultColumns result{};
        cuda.synchronize(reinterpret_cast<CudaDriver::Event>(ready));
        require(operation->completedResult(result) == CudaFamilyCompletedResultOutcome::success, "long sweep result");
        const size_t hits = count * config.maximumHitsPerQuery;
        cuda.downloadAsync(out.offsets, deviceAddress(result.hitOffsets), sizeof(out.offsets), consumer);
        cuda.downloadAsync(out.counts, deviceAddress(result.hitCounts), sizeof(out.counts), consumer);
        cuda.downloadAsync(out.ids, deviceAddress(result.objectIds), hits * sizeof(uint64_t), consumer);
        cuda.downloadAsync(out.distances, deviceAddress(result.distances), hits * sizeof(double), consumer);
        cuda.downloadAsync(out.positions, deviceAddress(result.positions), hits * 3 * sizeof(double), consumer);
        cuda.downloadAsync(out.normals, deviceAddress(result.normals), hits * 3 * sizeof(float), consumer);
        const auto done = cuda.createEvent();
        cuda.record(done, consumer);
        cuda.synchronize(done);
        retireConsumer(*operation, done, "long sweep retirement");
    };
    for (const auto family : {CudaFamilyKind::sphereSweep, CudaFamilyKind::boxSweep})
        for (const auto precision : {CudaFamilyPrecision::exact, CudaFamilyPrecision::boundsCandidate})
        {
            const bool exact = precision == CudaFamilyPrecision::exact;
            Rows       multiple, closest, any;
            run(family, precision, CudaFamilyHitMode::multiple, multiple);
            run(family, precision, CudaFamilyHitMode::closest, closest);
            run(family, precision, CudaFamilyHitMode::any, any);
            for (size_t row = 0; row < count; ++row)
            {
                // The CPU's per-object sweeps, exact only: every object hit, and the nearest
                // contact with the lowest id among those at its distance.
                size_t           expectedHits = 0;
                uint64_t         nearestId    = 0;
                ExactGeometryHit nearest{};
                for (const auto& object : objects)
                {
                    if (!exact)
                        break;
                    bool             hit = false;
                    ExactGeometryHit cpu{};
                    if (family == CudaFamilyKind::sphereSweep)
                        hit = object.exactGeometry->sweepSphere(host.origins[row], .5, host.directions[row],
                                                                host.minimum[row], host.maximum[row], cpu);
                    else
                    {
                        ovspatialquery_overlap_box_t query{};
                        std::copy_n(host.origins[row], 3, query.center);
                        std::copy_n(host.rotations[row], 4, query.rotation_xyzw);
                        query.half_extents[0] = query.half_extents[1] = query.half_extents[2] = .5;
                        ExactGeometryDescription description;
                        require(object.exactGeometry->exportDescription(description), "long sweep CPU export");
                        require(sweepBoxGeometry(description, makeQueryBoxGeometry(query), host.directions[row],
                                                 host.minimum[row], host.maximum[row], hit,
                                                 cpu) == OVSPATIALQUERY_API_SUCCESS,
                                "long sweep CPU box");
                    }
                    if (!hit)
                        continue;
                    ++expectedHits;
                    if (nearestId == 0 || cpu.distance < nearest.distance ||
                        (cpu.distance == nearest.distance && object.id.value < nearestId))
                    {
                        nearestId = object.id.value;
                        nearest   = cpu;
                    }
                }
                const uint64_t* begin = multiple.ids + multiple.offsets[row];
                require(multiple.counts[row] < capacity && (!exact || multiple.counts[row] == expectedHits),
                        "long sweep MULTIPLE rows");
                require(closest.counts[row] == (multiple.counts[row] != 0) && any.counts[row] == closest.counts[row],
                        "long sweep has-hit");
                if (closest.counts[row] == 0)
                    continue;
                size_t first = 0;
                for (size_t k = 1; k < multiple.counts[row]; ++k)
                {
                    const double d = multiple.distances[multiple.offsets[row] + k],
                                 b = multiple.distances[multiple.offsets[row] + first];
                    if (d < b || (d == b && begin[k] < begin[first]))
                        first = k;
                }
                const auto same = [&](const Rows& single, size_t k)
                {
                    const size_t from = single.offsets[row], to = multiple.offsets[row] + k;
                    return single.ids[from] == multiple.ids[to] && single.distances[from] == multiple.distances[to] &&
                           std::equal(single.positions[from], single.positions[from] + 3, multiple.positions[to]) &&
                           std::equal(single.normals[from], single.normals[from] + 3, multiple.normals[to]);
                };
                require(same(closest, first), "long sweep CLOSEST is the nearest MULTIPLE row");
                const uint64_t* anyRow = std::find(begin, begin + multiple.counts[row], any.ids[any.offsets[row]]);
                require(anyRow != begin + multiple.counts[row] && same(any, size_t(anyRow - begin)),
                        "long sweep ANY is a MULTIPLE row");
                require(!exact || (closest.ids[closest.offsets[row]] == nearestId &&
                                   std::abs(closest.distances[closest.offsets[row]] - nearest.distance) < 1e-9),
                        "long sweep CLOSEST meets the CPU");
            }
            // The tied boxes keep the lower id; the mesh listed last comes first on its lane.
            require(closest.ids[closest.offsets[0]] == 8 && closest.ids[closest.offsets[1]] == 7 &&
                        closest.counts[2] == 0 && closest.counts[3] == 1 && closest.counts[4] == 0,
                    "long sweep nearest objects");
        }
    std::shared_ptr<void> retired;
    require(populations.release(population, retired) == OVSPATIALQUERY_API_SUCCESS, "long sweep release");
    ovspatialquery_test_oracle_destroy_runtime(runtime);
    cuda.free(device);
    cuda.destroyEvent(producerReady);
    cuda.destroyStream(consumer);
    cuda.destroyStream(producer);
}

void testBorrowedSphereSweepRowRadiiAndFallback()
{
    CudaDriver                        cuda;
    const auto                        producer = cuda.createStream(), consumer = cuda.createStream();
    const auto                        ready   = cuda.createEvent();
    ovspatialquery_backend_runtime_t* runtime = nullptr;
    require(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS,
            "row-radius CUDA runtime");
    PopulationTable      populations(runtime);
    SphereGeometrySource sphere{};
    sphere.center = {2, 0, 0};
    sphere.radius = 1;
    std::shared_ptr<const ExactGeometry> geometry;
    require(createSphereExactGeometry(sphere, geometry) == OVSPATIALQUERY_API_SUCCESS, "row-radius CUDA geometry");
    PopulationKey population;
    require(populations.create(2, 31, {exactObject(1, 1, geometry, 1)}, population) == OVSPATIALQUERY_API_SUCCESS,
            "row-radius CUDA population");
    constexpr size_t count = 4;
    struct Queries
    {
        double centers[count][3]{};
        float  directions[count][3]{{1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}};
        double maximum[count]{10, 10, 10, 10};
        double radii[count]{.25, .5, std::numeric_limits<double>::quiet_NaN(), -1};
    } host;
    const auto device = cuda.allocate(sizeof(host));
    cuda.uploadAsync(device, &host, sizeof(host), producer);
    cuda.record(ready, producer);
    int64_t          rows[]{count}, vectors[]{count, 3}, rowStride[]{1}, vectorStride[]{3, 1};
    const auto       at = [&](size_t offset) { return device + offset; };
    CudaFamilyLaunch launch{};
    launch.queryCount = count;
    launch.positions  = cudaTensor(at(offsetof(Queries, centers)), {kDLFloat, 64, 1}, 2, vectors, vectorStride);
    launch.directionsOrRotations =
        cudaTensor(at(offsetof(Queries, directions)), {kDLFloat, 32, 1}, 2, vectors, vectorStride);
    launch.maximumDistances = cudaTensor(at(offsetof(Queries, maximum)), {kDLFloat, 64, 1}, 1, rows, rowStride);
    launch.radii            = cudaTensor(at(offsetof(Queries, radii)), {kDLFloat, 64, 1}, 1, rows, rowStride);
    launch.stream           = reinterpret_cast<uintptr_t>(producer);
    launch.waitEvent        = reinterpret_cast<uintptr_t>(ready);
    CudaFamilyPlanConfig config{};
    config.family              = CudaFamilyKind::sphereSweep;
    config.precision           = CudaFamilyPrecision::exact;
    config.hitMode             = CudaFamilyHitMode::closest;
    config.resultFields        = kCudaFamilyResultDistance;
    config.maximumQueryCount   = count;
    config.maximumHitsPerQuery = 1;
    config.hitCapacityPerSlot  = count;
    config.resultSlotCount     = 1;
    config.planRadius          = .1;
    std::shared_ptr<CudaFamilyPlan> plan;
    require(createCudaFamilyPlan(config, plan) == OVSPATIALQUERY_API_SUCCESS, "row-radius CUDA plan");
    for (bool present : {true, false})
    {
        CudaFamilyLaunch current = launch;
        if (!present)
        {
            current.radii = CudaFamilyInput{};
        }
        GenerationLease lease;
        require(populations.pin(population, lease) == OVSPATIALQUERY_API_SUCCESS, "row-radius CUDA pin");
        std::shared_ptr<CudaFamilyOperation> operation;
        require(prepareAndLaunchCudaFamilyPlan(plan, 0, current, std::move(lease), operation) ==
                    OVSPATIALQUERY_API_SUCCESS,
                "row-radius CUDA launch");
        const auto columns = operation->reservation();
        cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(columns.resultReady.value));
        uint32_t status[count]{};
        uint64_t offsets[count]{}, counts[count]{};
        double   distance[count]{};
        cuda.downloadAsync(status, deviceAddress(columns.queryStatus), sizeof(status), consumer);
        cuda.downloadAsync(offsets, deviceAddress(columns.hitOffsets), sizeof(offsets), consumer);
        cuda.downloadAsync(counts, deviceAddress(columns.hitCounts), sizeof(counts), consumer);
        cuda.downloadAsync(distance, deviceAddress(columns.distances), sizeof(distance), consumer);
        const auto done = cuda.createEvent();
        cuda.record(done, consumer);
        cuda.synchronize(done);
        for (size_t row = 0; row < count; ++row)
        {
            const bool valid = !present || row < 2;
            require(status[row] == (valid ? cuda_family::kBatchRowSuccess : cuda_family::kBatchRowInvalidArgument) &&
                        counts[row] == (valid ? 1u : 0u),
                    "row-radius CUDA malformed row isolation");
            if (valid)
            {
                const double expected = 1.0 - (present ? host.radii[row] : config.planRadius);
                require(offsets[row] < count && std::abs(distance[offsets[row]] - expected) < 1e-9,
                        "row-radius CUDA override or plan fallback");
            }
        }
        operation->releaseReadyDependencies();
        retireConsumer(*operation, done, "row-radius CUDA consumer retirement failed");
        operation.reset();
    }
    plan.reset();
    std::shared_ptr<void> retired;
    require(populations.release(population, retired) == OVSPATIALQUERY_API_SUCCESS,
            "row-radius CUDA population release");
    ovspatialquery_test_oracle_destroy_runtime(runtime);
    cuda.free(device);
    cuda.destroyEvent(ready);
    cuda.destroyStream(consumer);
    cuda.destroyStream(producer);
}

// sweepTriangle's rule on the host for one world triangle: none for a single-sided
// triangle the sphere moves along (ADR-0051), else the least parameter in
// [minimum, maximum] among the start, the face contacts from either side and the vertex
// and edge entries at which the sphere touches the triangle within the squared-distance
// tolerance. The entries use the device's
// formulas (|d|^2 kept, solved around the path's closest approach), so both sides
// propose the same parameters; the independent oracle below judges them.
bool sphereTriangleReference(const box_sweep::Vec3 corner[3], const double center[3], double radius,
                             const float direction[3], double minimum, double maximum, bool doubleSided,
                             double& distance)
{
    using namespace box_sweep;
    const Vec3   origin{center[0] - corner[0].x, center[1] - corner[0].y, center[2] - corner[0].z};
    const Vec3   d{direction[0], direction[1], direction[2]};
    const Vec3   first = sub(corner[1], corner[0]), second = sub(corner[2], corner[0]);
    Vec3         normal = cross(first, second);
    const double length = std::sqrt(dot(normal, normal));
    if (!(length > 0) || !std::isfinite(length))
        return false;
    normal                 = mul(normal, 1 / length);
    const double squared   = radius * radius;
    const double tolerance = std::max(1.0, squared) * std::numeric_limits<double>::epsilon() * 256.0;
    bool         hit       = false;
    const auto   consider  = [&](double t)
    {
        if (!std::isfinite(t) || t < minimum || t > maximum || (hit && t >= distance))
            return;
        const Vec3 moved = at(origin, d, t);
        double     u = 0, v = 0;
        const Vec3 gap = sub(moved, triangleClosest(moved, Vec3{}, first, second, u, v));
        if (dot(gap, gap) > squared + tolerance)
            return;
        hit      = true;
        distance = t;
    };
    const double height = dot(origin, normal), speed = dot(d, normal);
    if (!doubleSided && speed > 0)
        return false;
    consider(minimum);
    for (int side = -1; side <= 1; side += 2)
        if (side * speed < 0)
            consider((side * radius - height) / speed);
    const double inverse     = 1 / dot(d, d);
    const Vec3   vertices[3] = {Vec3{}, first, second};
    for (int k = 0; k < 3; ++k)
    {
        const Vec3   offset  = sub(origin, vertices[k]);
        const double closest = -dot(offset, d) * inverse;
        const Vec3   gap     = at(offset, d, closest);
        const double reach   = squared - dot(gap, gap);
        if (reach >= 0)
            consider(closest - std::sqrt(reach * inverse));
        // Across the edge's line, scaled by |e|^2.
        const Vec3   edge = sub(vertices[(k + 1) % 3], vertices[k]);
        const double ee = dot(edge, edge), eo = dot(edge, offset), ed = dot(edge, d);
        const Vec3   across = sub(mul(offset, ee), mul(edge, eo)), motion = sub(mul(d, ee), mul(edge, ed));
        const double motionSquared = dot(motion, motion);
        if (!(motionSquared > 0))
            continue;
        const double edgeClosest = -dot(across, motion) / motionSquared;
        const Vec3   edgeGap     = at(across, motion, edgeClosest);
        const double edgeReach   = squared * ee * ee - dot(edgeGap, edgeGap);
        if (edgeReach < 0)
            continue;
        const double t = edgeClosest - std::sqrt(edgeReach / motionSquared), along = eo + t * ed;
        if (along >= 0 && along <= ee)
            consider(t);
    }
    return hit;
}

// Sphere sweeps and tolerance rays against the CUDA traversal: its path-frame bounds on
// nodes and triangles, near-first order and full-warp queue must find what
// sweepTriangle's rule finds on every triangle, and that contact must be the first the
// independent oracle finds. The meshes and placements are those of the box traversal
// check; launches cover the plan radius, per-row radii (some tiny enough that the
// predicate's tolerance dominates), a plan contact tolerance and a raycast plan's
// tolerance tube. Float directions are as rounded or up to 9e-6 off unit squared length,
// and some rows start eight times farther from their target.
void testSphereSweepTraversalMatchesExhaustiveScan()
{
    CudaDriver                        cuda;
    const auto                        producer = cuda.createStream(), consumer = cuda.createStream();
    const auto                        producerReady = cuda.createEvent();
    ovspatialquery_backend_runtime_t* runtime       = nullptr;
    require(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS,
            "sphere traversal CUDA runtime");
    PopulationTable       populations(runtime);
    Draw                  between{0x2545f4914f6cdd1dull};
    const TraversalMeshes meshes = traversalMeshes(990, between);
    const auto &          soup = meshes.soup, &floor = meshes.floor, &fine = meshes.fine;
    // A millimetre triangle: a tiny sphere passing 2e-7 from its corner meets it only
    // through the predicate's squared-distance tolerance, which the triangle bound must
    // include. A larger triangle below the sweeps' reach keeps the object and node bounds,
    // which take the nominal radius, from rejecting the mesh first.
    TriangleMeshSource speck{};
    speck.key       = 993;
    speck.vertices  = {0,      0,      0,        1e-3f, 0,      0,        0, 1e-3f, 0,
                       -1e-3f, -1e-3f, -1.5e-3f, 1e-3f, -1e-3f, -1.5e-3f, 0, 1e-3f, -1.5e-3f};
    speck.triangles = {0, 1, 2, 3, 4, 5};
    numberSourceFeatures(speck);
    struct Target
    {
        const TriangleMeshSource* mesh;
        std::array<double, 16>    matrix;
        bool                      doubleSided;
    };
    const auto          placements = traversalPlacements();
    const bool          sided[]    = {true, false, true, false, true, false, true, false};
    std::vector<Target> targets;
    for (size_t p = 0; p < placements.size(); ++p)
        targets.push_back({p < 5 ? &soup : p < 7 ? &floor : &fine, placements[p], sided[p]});
    targets.push_back({&speck, placements[0], false});
    std::shared_ptr<const PreparedTriangleMesh> preparedSoup, preparedFloor, preparedFine, preparedSpeck;
    require(prepareTriangleMesh(soup, preparedSoup) == OVSPATIALQUERY_API_SUCCESS &&
                prepareTriangleMesh(floor, preparedFloor) == OVSPATIALQUERY_API_SUCCESS &&
                prepareTriangleMesh(fine, preparedFine) == OVSPATIALQUERY_API_SUCCESS &&
                prepareTriangleMesh(speck, preparedSpeck) == OVSPATIALQUERY_API_SUCCESS,
            "sphere traversal meshes");
    std::vector<PopulationObject> objects;
    for (size_t p = 0; p < targets.size(); ++p)
    {
        const auto&                                   prepared = targets[p].mesh == &soup    ? preparedSoup
                                                                 : targets[p].mesh == &floor ? preparedFloor
                                                                 : targets[p].mesh == &fine  ? preparedFine
                                                                                             : preparedSpeck;
        std::shared_ptr<const TriangleObjectGeometry> placement;
        require(bindPreparedTriangleGeometry(prepared, targets[p].matrix, targets[p].doubleSided, placement) ==
                    OVSPATIALQUERY_API_SUCCESS,
                "sphere traversal placement");
        std::shared_ptr<const ExactGeometry> geometry;
        require(createTriangleMeshExactGeometry(placement, geometry) == OVSPATIALQUERY_API_SUCCESS,
                "sphere traversal geometry");
        objects.push_back(exactObject(p + 1, p + 1, geometry, uint64_t{1} << p));
    }
    PopulationKey population;
    require(populations.create(2, 31, objects, population) == OVSPATIALQUERY_API_SUCCESS,
            "sphere traversal population");
    // World triangles exactly as the device reconstructs them.
    const auto worldTriangle = [&targets](size_t p, uint32_t triangle, box_sweep::Vec3 corner[3])
    {
        const auto& matrix = targets[p].matrix;
        for (int k = 0; k < 3; ++k)
        {
            const float* local = targets[p].mesh->vertices.data() + 3 * targets[p].mesh->triangles[3 * triangle + k];
            for (int a = 0; a < 3; ++a)
                corner[k][a] = std::fma(matrix[4 * a], double(local[0]),
                                        std::fma(matrix[4 * a + 1], double(local[1]),
                                                 std::fma(matrix[4 * a + 2], double(local[2]), matrix[4 * a + 3])));
        }
    };

    // Rows 0..499 cycle through the soup placements; four floor rows per floor placement,
    // six rows on the fine plane and two at the speck follow.
    constexpr double        plan     = .3;
    static constexpr size_t soupRows = 500, fineRows = soupRows + 8, speckRows = fineRows + 6, count = speckRows + 2;
    struct Queries
    {
        uint64_t bits[count][2]{};
        double   origins[count][3]{}, minimum[count]{}, maximum[count]{}, radii[count]{};
        float    directions[count][3]{};
    };
    const auto host     = std::make_unique<Queries>();
    const auto targetOf = [](size_t row)
    {
        return row < soupRows    ? row % 5
               : row < fineRows  ? 5 + (row - soupRows) / 4
               : row < speckRows ? size_t(7)
                                 : size_t(8);
    };
    for (size_t row = 0; row < soupRows; ++row)
    {
        const size_t i       = row / 5;
        const bool   aligned = i % 4 == 0;
        const auto&  matrix  = targets[targetOf(row)].matrix;
        // Aligned rows start above the plane and slide or drop; the others aim from
        // outside the soup at a point inside it, so most make a moving first contact.
        double       local[3]  = {between(-4, 4), between(-4, 4), aligned ? between(.35, 1.2) : between(-5, 5)};
        const double target[3] = {between(-2.5, 2.5), between(-2.5, 2.5), between(-2.5, 2.5)};
        double       reach     = 0;
        if (!aligned && i % 10 == 7)
            for (int a = 0; a < 3; ++a)
                local[a] = target[a] + (local[a] - target[a]) * 8;
        for (int a = 0; a < 3; ++a)
            host->origins[row][a] = matrix[4 * a] * local[0] + matrix[4 * a + 1] * local[1] +
                                    matrix[4 * a + 2] * local[2] + matrix[4 * a + 3];
        if (aligned)
        {
            host->directions[row][i / 4 % 3] = i % 8 ? -1.f : 1.f;
            reach                            = between(0, 6);
        }
        else
        {
            double d[3];
            for (int a = 0; a < 3; ++a)
            {
                d[a] = 0;
                for (int b = 0; b < 3; ++b)
                    d[a] += matrix[4 * a + b] * (target[b] - local[b]);
                reach += d[a] * d[a];
            }
            reach                 = std::sqrt(reach);
            const double excess[] = {0, 1e-6, -1e-6, 9e-6, -9e-6};
            for (int a = 0; a < 3; ++a)
                host->directions[row][a] = float(d[a] / reach * std::sqrt(1 + excess[i / 10 % 5]));
        }
        host->minimum[row] = i % 5 == 0 ? between(0, 1) : 0;
        host->maximum[row] = host->minimum[row] + reach * between(.5, 1.5);
        host->radii[row]   = row % 7 == 3 ? 1e-7 : between(.05, .6);
    }
    // Under the plan radius these rest exactly on the unplaced plane: touching at the start.
    const auto resting = [](size_t row) { return row < soupRows && row % 20 == 0 && row != 0 && row <= 160; };
    for (size_t row = 0; row < soupRows; ++row)
        if (resting(row))
        {
            host->origins[row][0] = between(-2, 2);
            host->origins[row][1] = between(-2, 2);
            host->origins[row][2] = plan;
            host->minimum[row]    = 0;
            host->maximum[row]    = 2;
        }
    // Floor rows sink 1 um under the plan radius into the floor 2.5 units back in mesh-local
    // x, then slide along it or fall a lift onto it along -z: float projections of the
    // floor's corners round by about 1e-4 of these units, so only the allowance keeps them.
    const double floorNormal[3] = {0, -.3 / std::sqrt(1.09), 1 / std::sqrt(1.09)};
    const auto   floorFall = [&](size_t row) { return (row - soupRows) % 2 == 1 ? .7 - 1e-6 / floorNormal[2] : 0.0; };
    for (size_t row = soupRows; row < fineRows; ++row)
    {
        const auto&  matrix   = targets[targetOf(row)].matrix;
        const double scale    = matrix[10];
        const double local[3] = {-2.5, 1e4, 3e3};
        const bool   falls    = floorFall(row) != 0;
        for (int a = 0; a < 3; ++a)
        {
            double normal = 0;
            for (int b = 0; b < 3; ++b)
                normal += matrix[4 * a + b] * floorNormal[b] / scale;
            host->origins[row][a] = matrix[4 * a] * local[0] + matrix[4 * a + 1] * local[1] +
                                    matrix[4 * a + 2] * local[2] + matrix[4 * a + 3] + normal * (plan - 1e-6);
            host->directions[row][a] = falls ? 0.f : float(matrix[4 * a] / scale);
        }
        host->origins[row][2] += falls ? .7 : 0;
        host->directions[row][2] = falls ? -1.f : host->directions[row][2];
        host->maximum[row]       = falls ? 2 : 10;
        host->radii[row]         = plan;
    }
    // On the fine plane: sunk by 0.1 at the start (sliding or still), resting above a
    // vertex, and falling onto one.
    const double fineCenters[6][3] = {
        {.1015625, -.2265625, plan - .1}, {-.3046875, .4140625, plan - .1}, {.25, -.125, plan}, {-.5, .3125, plan},
        {.375, .1875, plan + .5},         {-.0625, -.4375, plan + .25}};
    for (size_t row = fineRows; row < speckRows; ++row)
    {
        const size_t k = row - fineRows;
        std::copy_n(fineCenters[k], 3, host->origins[row]);
        host->directions[row][k < 4 ? k % 2 : 2] = k < 4 ? (k / 2 ? -1.f : 1.f) : -1.f;
        host->maximum[row]                       = 1;
        host->radii[row]                         = plan;
    }
    // Falling onto the speck: past its corner at 2e-7 with a 1e-7 row radius, and through it.
    const double speckCenters[2][3] = {{-2e-7 / std::sqrt(2.0), -2e-7 / std::sqrt(2.0), 1e-3}, {2e-4, 2e-4, 1e-3}};
    for (size_t row = speckRows; row < count; ++row)
    {
        std::copy_n(speckCenters[row - speckRows], 3, host->origins[row]);
        host->directions[row][2] = -1.f;
        host->maximum[row]       = 2e-3;
        host->radii[row]         = 1e-7;
    }
    for (size_t row = 0; row < count; ++row)
        host->bits[row][0] = uint64_t{1} << targetOf(row);
    const auto device = cuda.allocate(sizeof(Queries));
    cuda.uploadAsync(device, host.get(), sizeof(Queries), producer);
    cuda.record(producerReady, producer);
    int64_t    rows[] = {count}, vectors[] = {count, 3}, rowStride[] = {1}, vectorStride[] = {3, 1};
    const auto at = [&](size_t offset) { return device + offset; };

    struct Variant
    {
        CudaFamilyKind family;
        double         tolerance;
        bool           rowRadii;
        const char*    name;
    };
    const Variant variants[] = {{CudaFamilyKind::sphereSweep, 0, false, "plan radius"},
                                {CudaFamilyKind::sphereSweep, 0, true, "row radii"},
                                {CudaFamilyKind::sphereSweep, .01, false, "plan tolerance"},
                                {CudaFamilyKind::raycast, .02, false, "ray tube"}};
    size_t        hits = 0, starts = 0, ties = 0, tiny = 0, speckGrazes = 0, offUnit = 0;
    for (const Variant& variant : variants)
    {
        CudaFamilyLaunch launch{};
        launch.queryCount = count;
        launch.stream     = reinterpret_cast<uintptr_t>(producer);
        launch.waitEvent  = reinterpret_cast<uintptr_t>(producerReady);
        launch.positions  = cudaTensor(at(offsetof(Queries, origins)), {kDLFloat, 64, 1}, 2, vectors, vectorStride);
        launch.directionsOrRotations =
            cudaTensor(at(offsetof(Queries, directions)), {kDLFloat, 32, 1}, 2, vectors, vectorStride);
        launch.minimumDistances = cudaTensor(at(offsetof(Queries, minimum)), {kDLFloat, 64, 1}, 1, rows, rowStride);
        launch.maximumDistances = cudaTensor(at(offsetof(Queries, maximum)), {kDLFloat, 64, 1}, 1, rows, rowStride);
        int64_t bitRows[]{rows[0], 2};
        launch.queryBits = cudaBitsTensor(at(offsetof(Queries, bits)), bitRows);
        if (variant.rowRadii)
            launch.radii = cudaTensor(at(offsetof(Queries, radii)), {kDLFloat, 64, 1}, 1, rows, rowStride);
        CudaFamilyPlanConfig config{};
        config.family              = variant.family;
        config.precision           = CudaFamilyPrecision::exact;
        config.hitMode             = CudaFamilyHitMode::closest;
        config.maximumQueryCount   = count;
        config.maximumHitsPerQuery = 1;
        config.hitCapacityPerSlot  = count;
        config.resultSlotCount     = 1;
        config.contactTolerance    = variant.tolerance;
        config.planRadius          = variant.family == CudaFamilyKind::sphereSweep ? plan : 0;
        config.resultFields        = kCudaFamilyResultPosition | kCudaFamilyResultNormal | kCudaFamilyResultDistance |
                              kCudaFamilyResultGeometryIdentity | kCudaFamilyResultSourceFeature;
        GenerationLease lease;
        require(populations.pin(population, lease) == OVSPATIALQUERY_API_SUCCESS, "sphere traversal pin");
        std::shared_ptr<CudaFamilyPlan> cudaPlan;
        require(createCudaFamilyPlan(config, cudaPlan) == OVSPATIALQUERY_API_SUCCESS, "sphere traversal plan");
        std::shared_ptr<CudaFamilyOperation> operation;
        require(prepareAndLaunchCudaFamilyPlan(cudaPlan, 0, launch, std::move(lease), operation) ==
                    OVSPATIALQUERY_API_SUCCESS,
                "sphere traversal launch");
        const auto ready = operation->reservation().resultReady.value;
        cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(ready));
        cuda.synchronize(reinterpret_cast<CudaDriver::Event>(ready));
        CudaFamilyResultColumns result{};
        require(operation->completedResult(result) == CudaFamilyCompletedResultOutcome::success,
                "sphere traversal completion");
        std::vector<uint32_t> statuses(count), features(count), corners(3 * count);
        std::vector<uint64_t> offsets(count), counts(count), ids(count), faces(count);
        std::vector<double>   distances(count), positions(3 * count);
        std::vector<float>    normals(3 * count), bary(3 * count);
        cuda.downloadAsync(statuses.data(), deviceAddress(result.queryStatus), 4 * count, consumer);
        cuda.downloadAsync(offsets.data(), deviceAddress(result.hitOffsets), 8 * count, consumer);
        cuda.downloadAsync(counts.data(), deviceAddress(result.hitCounts), 8 * count, consumer);
        cuda.downloadAsync(ids.data(), deviceAddress(result.objectIds), 8 * count, consumer);
        cuda.downloadAsync(distances.data(), deviceAddress(result.distances), 8 * count, consumer);
        cuda.downloadAsync(positions.data(), deviceAddress(result.positions), 24 * count, consumer);
        cuda.downloadAsync(normals.data(), deviceAddress(result.normals), 12 * count, consumer);
        cuda.downloadAsync(features.data(), deviceAddress(result.sourceFeatureKinds), 4 * count, consumer);
        cuda.downloadAsync(faces.data(), deviceAddress(result.sourceFaceIndices), 8 * count, consumer);
        cuda.downloadAsync(corners.data(), deviceAddress(result.sourceFaceCorners), 12 * count, consumer);
        cuda.downloadAsync(bary.data(), deviceAddress(result.sourceBarycentrics), 12 * count, consumer);
        const auto done = cuda.createEvent();
        cuda.record(done, consumer);
        cuda.synchronize(done);
        for (size_t row = 0; row < count; ++row)
        {
            const size_t  p         = targetOf(row);
            const Target& target    = targets[p];
            const double  radius    = (variant.rowRadii ? host->radii[row] : config.planRadius) + variant.tolerance;
            const auto    reference = [&](uint32_t triangle, double& distance)
            {
                box_sweep::Vec3 corner[3];
                worldTriangle(p, triangle, corner);
                return sphereTriangleReference(corner, host->origins[row], radius, host->directions[row],
                                               host->minimum[row], host->maximum[row], target.doubleSided, distance);
            };
            bool           expected      = false;
            double         best          = 0;
            uint32_t       winner        = 0;
            const uint32_t triangleCount = uint32_t(target.mesh->triangles.size() / 3);
            for (uint32_t triangle = 0; triangle < triangleCount; ++triangle)
            {
                double distance = 0;
                if (reference(triangle, distance) && (!expected || distance < best))
                {
                    expected = true;
                    best     = distance;
                    winner   = triangle;
                }
            }
            require(statuses[row] == cuda_family::kBatchRowSuccess && counts[row] == unsigned(expected),
                    "sphere traversal presence");
            // The independent oracle: no triangle that counts is clearly touched before the
            // reported contact, which touches its face and, once the sweep has moved, does not
            // penetrate it. A single-sided triangle the sphere moves along does not count, and
            // one within rounding of parallel may go either way (see the CPU test).
            {
                namespace oracle = sphere_sweep_oracle;
                using Wide       = oracle::Wide;
                const double tolerance =
                    std::max(1.0, radius * radius) * std::numeric_limits<double>::epsilon() * 256.0;
                const Wide     accept     = std::sqrt(Wide(radius) * radius + tolerance);
                const Wide     slack      = Wide(1e-9) * std::max(1.0, radius);
                const bool     reported   = counts[row] == 1 && offsets[row] < count;
                const Wide     reportedAt = reported ? Wide(distances[offsets[row]]) : 0;
                const uint32_t face       = reported ? uint32_t(faces[offsets[row]] - 1000) : 0;
                const Wide     parallel   = Wide(1e-12);
                bool           faceKept   = true;
                const auto     pathOf     = [&](uint32_t triangle)
                {
                    box_sweep::Vec3 corner[3];
                    worldTriangle(p, triangle, corner);
                    const double world[3][3] = {{corner[0].x, corner[0].y, corner[0].z},
                                                {corner[1].x, corner[1].y, corner[1].z},
                                                {corner[2].x, corner[2].y, corner[2].z}};
                    return oracle::makePath(world, host->origins[row], host->directions[row]);
                };
                for (uint32_t triangle = 0; triangle < triangleCount; ++triangle)
                {
                    const oracle::Path path  = pathOf(triangle);
                    Wide               touch = 0, clear = 0;
                    if (!oracle::mayReach(path, accept + slack, host->minimum[row], host->maximum[row]) ||
                        !oracle::firstWithin(path, accept + slack, host->minimum[row], host->maximum[row], touch))
                        continue;
                    const Wide approach = oracle::dot(path.normal, path.direction);
                    const bool kept     = target.doubleSided || approach <= parallel;
                    const bool required = target.doubleSided || approach < -parallel || approach == 0;
                    const bool reaches =
                        oracle::firstWithin(path, Wide(radius) - slack, host->minimum[row], host->maximum[row], clear);
                    if (reported && triangle == face)
                        faceKept = kept;
                    if (required && reaches)
                        require(reported && reportedAt <= clear + Wide(1e-12) * std::max(Wide(1), clear),
                                "sphere traversal missed a contact the oracle finds first");
                }
                if (reported)
                {
                    const Wide gap = oracle::distanceAt(pathOf(face), reportedAt);
                    require(gap <= accept + slack, "sphere traversal contact does not touch its face");
                    require(faceKept, "sphere traversal reported a single-sided triangle it moves along");
                    require(reportedAt == host->minimum[row] || gap >= Wide(radius) - slack,
                            "sphere traversal contact penetrates its face");
                    double squaredLength = 0;
                    for (float component : host->directions[row])
                        squaredLength += double(component) * component;
                    offUnit += std::abs(squaredLength - 1) > 1e-7;
                }
            }
            if (variant.family == CudaFamilyKind::sphereSweep && !variant.rowRadii && variant.tolerance == 0 &&
                (resting(row) || (row >= soupRows && row < speckRows)))
            {
                // Resting and sunk rows start in contact; falling ones meet the floor a lift less the
                // sink along -z, and the fine plane a lift above the radius.
                const double fall = row >= fineRows + 4                 ? host->origins[row][2] - plan
                                    : row >= soupRows && row < fineRows ? floorFall(row)
                                                                        : 0.0;
                require(expected && std::abs(best - fall) <= 1e-9 && offsets[row] < count &&
                            std::abs(distances[offsets[row]] - fall) <= 1e-9,
                        "sphere traversal resting, sunk or falling contact");
            }
            if (!expected)
                continue;
            ++hits;
            starts += best == host->minimum[row];
            tiny += variant.rowRadii && host->radii[row] < 1e-6;
            speckGrazes += variant.rowRadii && row == speckRows;
            const auto h = offsets[row];
            require(h < count && ids[h] == p + 1 && std::abs(distances[h] - best) <= 1e-9 * std::max(1.0, best),
                    "sphere traversal distance");
            require(features[h] == cuda_family::kFeatureTriangle && faces[h] >= 1000 && faces[h] < 1000 + triangleCount,
                    "sphere traversal face");
            const uint32_t triangle = uint32_t(faces[h] - 1000);
            for (uint32_t a = 0; a < 3; ++a)
                require(corners[3 * h + a] == 3 * triangle + a, "sphere traversal corners");
            // Starts on the fine plane tie exactly, so the lower index must win there.
            require(p != 7 || best != host->minimum[row] || triangle == winner,
                    "sphere traversal broke the lower-index tie rule");
            if (triangle != winner)
            {
                double tied = 0;
                require(reference(triangle, tied) && std::abs(tied - best) <= 1e-9 * std::max(1.0, best),
                        "sphere traversal face is not a confirmed tie");
                ++ties;
            }
            // On the face, at the moved sphere, with barycentrics and a unit normal that agree.
            box_sweep::Vec3 corner[3], point{}, moved{};
            worldTriangle(p, triangle, corner);
            double scale = 1;
            for (int a = 0; a < 3; ++a)
            {
                point[a] = positions[3 * h + a];
                moved[a] = std::fma(double(host->directions[row][a]), distances[h], host->origins[row][a]);
                scale    = std::max({scale, std::abs(point[a]), std::abs(moved[a])});
            }
            double     u = 0, v = 0;
            const auto closest = box_sweep::triangleClosest(point, corner[0], corner[1], corner[2], u, v);
            const auto gap     = box_sweep::sub(point, closest);
            const auto uvPoint = box_sweep::add(
                corner[0], box_sweep::add(box_sweep::mul(box_sweep::sub(corner[1], corner[0]), bary[3 * h + 1]),
                                          box_sweep::mul(box_sweep::sub(corner[2], corner[0]), bary[3 * h + 2])));
            const auto   uvGap      = box_sweep::sub(point, uvPoint);
            const auto   offset     = box_sweep::sub(moved, point);
            const double reach      = std::sqrt(box_sweep::dot(offset, offset));
            const double acceptance = std::sqrt(radius * radius + std::max(1.0, radius * radius) *
                                                                      std::numeric_limits<double>::epsilon() * 256.0);
            require(std::sqrt(box_sweep::dot(gap, gap)) <= 1e-9 * scale &&
                        std::sqrt(box_sweep::dot(uvGap, uvGap)) <= 1e-5 * scale,
                    "sphere traversal contact off its face");
            // Within the moved sphere; on it once the sweep has moved (the oracle above covers
            // one-sided faces first reached from behind).
            require(reach <= acceptance + 1e-9 * scale, "sphere traversal contact off the moved sphere");
            double normalLength = 0, along = 0;
            for (int a = 0; a < 3; ++a)
            {
                normalLength += double(normals[3 * h + a]) * normals[3 * h + a];
                along += double(normals[3 * h + a]) * offset[a];
            }
            require(std::abs(normalLength - 1) <= 1e-5 && (reach <= 1e-6 * scale || along >= reach * (1 - 1e-5)),
                    "sphere traversal normal");
        }
        retireConsumer(*operation, done, "sphere traversal consumer retirement failed");
        operation.reset();
        cudaPlan.reset();
    }
    // Enough contacts, including touching, initial overlap and tiny radii, for the comparison to mean something.
    std::printf("sphere traversal: %zu hits, %zu at the minimum, %zu confirmed ties, %zu tiny radii, %zu off-unit "
                "directions\n",
                hits, starts, ties, tiny, offUnit);
    require(hits > 1000 && starts > 40 && tiny > 10 && speckGrazes == 1 && offUnit > 300, "sphere traversal coverage");
    std::shared_ptr<void> retired;
    require(populations.release(population, retired) == OVSPATIALQUERY_API_SUCCESS,
            "sphere traversal population release");
    ovspatialquery_test_oracle_destroy_runtime(runtime);
    cuda.free(device);
    cuda.destroyEvent(producerReady);
    cuda.destroyStream(consumer);
    cuda.destroyStream(producer);
}

// Whether a triangle meets a box (centre, unit axes as rows, half extents), by separating
// axes in long double: the box's axes, the triangle's normal and their nine edge crosses.
bool triangleMeetsBoxReference(const box_sweep::Vec3 corner[3], const double center[3], const long double axes[3][3],
                               const double half[3])
{
    long double local[3][3];
    for (int k = 0; k < 3; ++k)
        for (int a = 0; a < 3; ++a)
            local[k][a] = axes[a][0] * ((long double)corner[k].x - center[0]) +
                          axes[a][1] * ((long double)corner[k].y - center[1]) +
                          axes[a][2] * ((long double)corner[k].z - center[2]);
    const auto separates = [&](const long double axis[3])
    {
        long double low = 0, high = 0, reach = 0;
        for (int k = 0; k < 3; ++k)
        {
            const long double along = axis[0] * local[k][0] + axis[1] * local[k][1] + axis[2] * local[k][2];
            low                     = k == 0 ? along : std::min(low, along);
            high                    = k == 0 ? along : std::max(high, along);
        }
        for (int a = 0; a < 3; ++a)
            reach += half[a] * std::fabs(axis[a]);
        return low > reach || high < -reach;
    };
    long double edges[3][3];
    for (int k = 0; k < 3; ++k)
        for (int a = 0; a < 3; ++a)
            edges[k][a] = local[(k + 1) % 3][a] - local[k][a];
    const long double normal[3] = {edges[0][1] * edges[1][2] - edges[0][2] * edges[1][1],
                                   edges[0][2] * edges[1][0] - edges[0][0] * edges[1][2],
                                   edges[0][0] * edges[1][1] - edges[0][1] * edges[1][0]};
    if (separates(normal))
        return false;
    for (int a = 0; a < 3; ++a)
    {
        const long double face[3] = {a == 0 ? 1.0L : 0.0L, a == 1 ? 1.0L : 0.0L, a == 2 ? 1.0L : 0.0L};
        if (separates(face))
            return false;
        for (int k = 0; k < 3; ++k)
        {
            const long double across[3] = {face[1] * edges[k][2] - face[2] * edges[k][1],
                                           face[2] * edges[k][0] - face[0] * edges[k][2],
                                           face[0] * edges[k][1] - face[1] * edges[k][0]};
            if (separates(across))
                return false;
        }
    }
    return true;
}

// Exact sphere, BOX and AABB overlaps reporting every placed mesh they touch, against a scan
// of every triangle: on the device a float gate drops objects whose bounds the shape misses
// and a float BVH screen reports a mesh at a node the shape holds and leaves one whose leaves
// it misses, so most meshes take no triangle test; every row's set of objects must still be
// the scan's. The meshes and placements are the traversal checks'; shapes are drawn about
// each placement, some holding the whole mesh, most cutting through it or stopping beside it.
// Every launch runs on two populations of the same placements: ids descending with catalogue
// order, so a window's hits are inserted one by one, and ascending, so they are appended at
// once; each hit's geometry identity names its placement. Pairs within 1e-9 of touching, relative to the
// coordinates, are left out. A slot holding two hits truncates the first of three giant rows,
// which writes its hits while they are counted, and leaves the others none.
void testMeshOverlapsMatchExhaustiveScan()
{
    CudaDriver                        cuda;
    const auto                        producer = cuda.createStream(), consumer = cuda.createStream();
    const auto                        producerReady = cuda.createEvent();
    ovspatialquery_backend_runtime_t* runtime       = nullptr;
    require(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS,
            "mesh overlap CUDA runtime");
    PopulationTable       populations(runtime);
    Draw                  between{0x9e3779b97f4a7c15ull};
    const TraversalMeshes meshes     = traversalMeshes(1990, between);
    const auto            placements = traversalPlacements();
    struct Target
    {
        const TriangleMeshSource* mesh;
        std::array<double, 16>    matrix;
        double                    local[3]; // the mesh's middle, in its own coordinates
        double                    size;     // and its half size there
    };
    std::vector<Target> targets;
    for (size_t p = 0; p < placements.size(); ++p)
    {
        if (p < 5)
            targets.push_back({&meshes.soup, placements[p], {0, 0, 0}, 3});
        else if (p < 7)
            targets.push_back({&meshes.floor, placements[p], {0, 1e4, 3e3}, 20});
        else
            targets.push_back({&meshes.fine, placements[p], {0, 0, 0}, 1});
    }
    std::shared_ptr<const PreparedTriangleMesh> soup, floor, fine;
    require(prepareTriangleMesh(meshes.soup, soup) == OVSPATIALQUERY_API_SUCCESS &&
                prepareTriangleMesh(meshes.floor, floor) == OVSPATIALQUERY_API_SUCCESS &&
                prepareTriangleMesh(meshes.fine, fine) == OVSPATIALQUERY_API_SUCCESS,
            "mesh overlap meshes");
    const size_t                       objectCount = targets.size();
    std::vector<PopulationObject>      objects, ascending;
    std::vector<ovspatialquery_aabb_t> bounds;
    for (size_t p = 0; p < objectCount; ++p)
    {
        const auto& prepared = targets[p].mesh == &meshes.soup ? soup : targets[p].mesh == &meshes.floor ? floor : fine;
        std::shared_ptr<const TriangleObjectGeometry> placement;
        require(bindPreparedTriangleGeometry(prepared, targets[p].matrix, false, placement) ==
                    OVSPATIALQUERY_API_SUCCESS,
                "mesh overlap placement");
        std::shared_ptr<const ExactGeometry> geometry;
        require(createTriangleMeshExactGeometry(placement, geometry) == OVSPATIALQUERY_API_SUCCESS,
                "mesh overlap geometry");
        bounds.push_back(geometry->worldBounds());
        objects.push_back(exactObject(100 - p, p + 1, geometry, 1));
        ascending.push_back(exactObject(200 + p, p + 1, geometry, 1));
    }
    PopulationKey descendingPopulation, ascendingPopulation;
    require(populations.create(2, 41, objects, descendingPopulation) == OVSPATIALQUERY_API_SUCCESS &&
                populations.create(3, 41, ascending, ascendingPopulation) == OVSPATIALQUERY_API_SUCCESS,
            "mesh overlap populations");

    // Rows 0..2 hold every mesh near the origin; then 40 rows per placement.
    static constexpr size_t perTarget = 40, count = 3 + 8 * perTarget;
    struct Queries
    {
        double centers[count][3]{}, half[count][3]{}, radii[count]{};
        float  rotations[count][4]{};
    };
    const auto host = std::make_unique<Queries>();
    for (size_t row = 0; row < count; ++row)
    {
        double local[3] = {}, size = 50, scale = 1;
        if (row >= 3)
        {
            const Target& target = targets[(row - 3) / perTarget];
            const double* m      = target.matrix.data();
            scale                = std::sqrt(m[0] * m[0] + m[4] * m[4] + m[8] * m[8]);
            size = target.size * (row % 8 == 0 ? between(1.8, 2.5) : row % 3 == 0 ? between(.3, 1) : between(.01, .3));
            for (int a = 0; a < 3; ++a)
                local[a] = target.local[a] + target.size * between(-1.4, 1.4);
            for (int a = 0; a < 3; ++a)
                host->centers[row][a] =
                    m[4 * a] * local[0] + m[4 * a + 1] * local[1] + m[4 * a + 2] * local[2] + m[4 * a + 3];
        }
        host->radii[row] = size * scale;
        for (int a = 0; a < 3; ++a)
            host->half[row][a] = size * scale * between(.3, 1);
        double quaternion[4], length = 0;
        for (double& component : quaternion)
        {
            component = between(-1, 1);
            length += component * component;
        }
        for (int c = 0; c < 4; ++c)
            host->rotations[row][c] = float(quaternion[c] / std::sqrt(length));
    }
    const auto device = cuda.allocate(sizeof(Queries));
    cuda.uploadAsync(device, host.get(), sizeof(Queries), producer);
    cuda.record(producerReady, producer);
    int64_t    rowStride[] = {1}, vectorStride[] = {3, 1}, quaternionStride[] = {4, 1};
    const auto at = [&](size_t offset) { return device + offset; };
    // World triangles exactly as the device reconstructs them.
    const auto worldTriangle = [&targets](size_t p, uint32_t triangle, box_sweep::Vec3 corner[3])
    {
        const auto& matrix = targets[p].matrix;
        for (int k = 0; k < 3; ++k)
        {
            const float* local = targets[p].mesh->vertices.data() + 3 * targets[p].mesh->triangles[3 * triangle + k];
            for (int a = 0; a < 3; ++a)
                corner[k][a] = std::fma(matrix[4 * a], double(local[0]),
                                        std::fma(matrix[4 * a + 1], double(local[1]),
                                                 std::fma(matrix[4 * a + 2], double(local[2]), matrix[4 * a + 3])));
        }
    };

    size_t held = 0, cut = 0, beside = 0, compared = 0;
    for (const CudaOverlapKind kind : {CudaOverlapKind::sphere, CudaOverlapKind::box, CudaOverlapKind::aabb})
        for (const bool truncating : {false, true})
        {
            if (truncating && kind != CudaOverlapKind::sphere)
                continue;
            const size_t rowCount = truncating ? 3 : count;
            // The scan: per row and object, touched (1), untouched (0) or within 1e-9 of touching (2).
            std::vector<uint8_t> expected(rowCount * objectCount);
            for (size_t row = 0; row < rowCount; ++row)
            {
                const double* center     = host->centers[row];
                long double   axes[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
                if (kind == CudaOverlapKind::box)
                {
                    const long double x = host->rotations[row][0], y = host->rotations[row][1],
                                      z = host->rotations[row][2], w = host->rotations[row][3];
                    const long double n           = 1 / std::sqrt(x * x + y * y + z * z + w * w);
                    const long double q[4]        = {x * n, y * n, z * n, w * n};
                    const long double rows3[3][3] = {{1 - 2 * (q[1] * q[1] + q[2] * q[2]),
                                                      2 * (q[0] * q[1] + q[2] * q[3]), 2 * (q[0] * q[2] - q[1] * q[3])},
                                                     {2 * (q[0] * q[1] - q[2] * q[3]),
                                                      1 - 2 * (q[0] * q[0] + q[2] * q[2]),
                                                      2 * (q[1] * q[2] + q[0] * q[3])},
                                                     {2 * (q[0] * q[2] + q[1] * q[3]), 2 * (q[1] * q[2] - q[0] * q[3]),
                                                      1 - 2 * (q[0] * q[0] + q[1] * q[1])}};
                    std::copy(&rows3[0][0], &rows3[0][0] + 9, &axes[0][0]);
                }
                const double slack =
                    1e-9 * (1 + std::abs(center[0]) + std::abs(center[1]) + std::abs(center[2]) + host->radii[row]);
                for (size_t p = 0; p < objectCount; ++p)
                {
                    const auto touches = [&](double grow)
                    {
                        double half[3];
                        for (int a = 0; a < 3; ++a)
                            half[a] = host->half[row][a] + grow;
                        const double radius = host->radii[row] + grow;
                        for (uint32_t t = 0; t < targets[p].mesh->triangles.size() / 3; ++t)
                        {
                            box_sweep::Vec3 corner[3];
                            worldTriangle(p, t, corner);
                            if (kind != CudaOverlapKind::sphere)
                            {
                                if (triangleMeetsBoxReference(corner, center, axes, half))
                                    return true;
                                continue;
                            }
                            double                u = 0, v = 0;
                            const box_sweep::Vec3 point{center[0], center[1], center[2]};
                            const box_sweep::Vec3 gap = box_sweep::sub(
                                point, box_sweep::triangleClosest(point, corner[0], corner[1], corner[2], u, v));
                            if (box_sweep::dot(gap, gap) <= radius * radius)
                                return true;
                        }
                        return false;
                    };
                    const bool outer = touches(slack), inner = touches(-slack);
                    expected[row * objectCount + p] = inner ? 1 : outer ? 2 : 0;
                    if (truncating || inner != outer)
                        continue;
                    // Coverage: a whole mesh inside the shape, one the shape cuts, and one whose
                    // bounds it reaches without touching a triangle.
                    bool inside = true, reaches = true;
                    for (int corner = 0; corner < 8; ++corner)
                    {
                        double point[3], offset[3];
                        for (int a = 0; a < 3; ++a)
                        {
                            point[a]  = corner & (1 << a) ? bounds[p].maximum[a] : bounds[p].minimum[a];
                            offset[a] = point[a] - center[a];
                        }
                        if (kind == CudaOverlapKind::sphere)
                            inside = inside && offset[0] * offset[0] + offset[1] * offset[1] + offset[2] * offset[2] <
                                                   host->radii[row] * host->radii[row];
                        else
                            for (int a = 0; a < 3; ++a)
                                inside = inside && std::abs(axes[a][0] * offset[0] + axes[a][1] * offset[1] +
                                                            axes[a][2] * offset[2]) < host->half[row][a];
                    }
                    if (kind == CudaOverlapKind::sphere)
                    {
                        double gap = 0;
                        for (int a = 0; a < 3; ++a)
                        {
                            const double d =
                                std::max({bounds[p].minimum[a] - center[a], center[a] - bounds[p].maximum[a], 0.0});
                            gap += d * d;
                        }
                        reaches = gap < host->radii[row] * host->radii[row];
                    }
                    else
                        for (int a = 0; a < 3; ++a)
                            reaches = reaches && bounds[p].minimum[a] < center[a] + host->half[row][a] * 2 &&
                                      bounds[p].maximum[a] > center[a] - host->half[row][a] * 2;
                    held += inside;
                    cut += inner && !inside;
                    beside += !inner && reaches;
                }
            }

            for (const bool rising : {false, true})
            {
                int64_t rows[] = {int64_t(rowCount)}, vectors[] = {int64_t(rowCount), 3},
                        quaternions[] = {int64_t(rowCount), 4};
                CudaFamilyLaunch launch{};
                launch.queryCount = rowCount;
                launch.stream     = reinterpret_cast<uintptr_t>(producer);
                launch.waitEvent  = reinterpret_cast<uintptr_t>(producerReady);
                launch.positions =
                    cudaTensor(at(offsetof(Queries, centers)), {kDLFloat, 64, 1}, 2, vectors, vectorStride);
                if (kind == CudaOverlapKind::sphere)
                    launch.radii = cudaTensor(at(offsetof(Queries, radii)), {kDLFloat, 64, 1}, 1, rows, rowStride);
                else
                    launch.halfExtents =
                        cudaTensor(at(offsetof(Queries, half)), {kDLFloat, 64, 1}, 2, vectors, vectorStride);
                if (kind == CudaOverlapKind::box)
                    launch.directionsOrRotations = cudaTensor(at(offsetof(Queries, rotations)), {kDLFloat, 32, 1}, 2,
                                                              quaternions, quaternionStride);
                CudaFamilyPlanConfig config{};
                config.family              = CudaFamilyKind::overlap;
                config.overlapKind         = kind;
                config.precision           = CudaFamilyPrecision::exact;
                config.hitMode             = CudaFamilyHitMode::multiple;
                config.maximumQueryCount   = count;
                config.maximumHitsPerQuery = objectCount;
                config.hitCapacityPerSlot  = truncating ? 2 : count * objectCount;
                config.resultSlotCount     = 1;
                config.planRadius          = 1;
                config.resultFields        = kCudaFamilyResultGeometryIdentity;
                std::fill_n(config.planHalfExtents, 3, 1.0);
                GenerationLease lease;
                require(populations.pin(rising ? ascendingPopulation : descendingPopulation, lease) ==
                            OVSPATIALQUERY_API_SUCCESS,
                        "mesh overlap pin");
                std::shared_ptr<CudaFamilyPlan> plan;
                require(createCudaFamilyPlan(config, plan) == OVSPATIALQUERY_API_SUCCESS, "mesh overlap plan");
                std::shared_ptr<CudaFamilyOperation> operation;
                require(prepareAndLaunchCudaFamilyPlan(plan, 0, launch, std::move(lease), operation) ==
                            OVSPATIALQUERY_API_SUCCESS,
                        "mesh overlap launch");
                const auto ready = operation->reservation().resultReady.value;
                cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(ready));
                CudaFamilyResultColumns result{};
                cuda.synchronize(reinterpret_cast<CudaDriver::Event>(ready));
                require(operation->completedResult(result) == CudaFamilyCompletedResultOutcome::success,
                        "mesh overlap completion");
                const size_t          capacity = size_t(config.hitCapacityPerSlot);
                std::vector<uint32_t> statuses(rowCount);
                std::vector<uint64_t> offsets(rowCount), counts(rowCount), ids(capacity), values(capacity);
                std::vector<int32_t>  types(capacity);
                std::vector<uint8_t>  truncated(rowCount);
                cuda.downloadAsync(statuses.data(), deviceAddress(result.queryStatus), 4 * rowCount, consumer);
                cuda.downloadAsync(offsets.data(), deviceAddress(result.hitOffsets), 8 * rowCount, consumer);
                cuda.downloadAsync(counts.data(), deviceAddress(result.hitCounts), 8 * rowCount, consumer);
                cuda.downloadAsync(truncated.data(), deviceAddress(result.truncated), rowCount, consumer);
                cuda.downloadAsync(ids.data(), deviceAddress(result.objectIds), 8 * capacity, consumer);
                cuda.downloadAsync(types.data(), deviceAddress(result.geometryTypes), 4 * capacity, consumer);
                cuda.downloadAsync(values.data(), deviceAddress(result.geometryValues), 8 * capacity, consumer);
                const auto done = cuda.createEvent();
                cuda.record(done, consumer);
                cuda.synchronize(done);
                for (size_t row = 0; row < rowCount; ++row)
                {
                    require(statuses[row] == cuda_family::kBatchRowSuccess && offsets[row] + counts[row] <= capacity,
                            "mesh overlap row status");
                    std::vector<bool> reported(objectCount);
                    for (uint64_t h = 0; h < counts[row]; ++h)
                    {
                        const uint64_t id = ids[offsets[row] + h];
                        const uint64_t p  = rising ? id - 200 : 100 - id;
                        require(p < objectCount && !reported[p] && values[offsets[row] + h] == p + 1 &&
                                    types[offsets[row] + h] == int32_t(OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH),
                                "mesh overlap object identity");
                        reported[p] = true;
                    }
                    size_t touched = 0;
                    for (size_t p = 0; p < objectCount; ++p)
                    {
                        const uint8_t want = expected[row * objectCount + p];
                        touched += want == 1;
                        if (want == 2 || (truncating && reported[p]))
                        {
                            require(!reported[p] || want != 0, "mesh overlap truncated row holds an untouched mesh");
                            continue;
                        }
                        if (!truncating && reported[p] != (want == 1))
                            std::fprintf(stderr, "mesh overlap kind=%d row=%zu object=%zu reported=%d expected=%u\n",
                                         int(kind), row, p, int(reported[p]), unsigned(want));
                        require(truncating || reported[p] == (want == 1), "mesh overlap set");
                        compared += !truncating && !rising;
                    }
                    if (truncating)
                        require(touched > 2 && counts[row] == (row == 0 ? 2u : 0u) && truncated[row] == 1,
                                "mesh overlap truncation");
                    else
                        require(truncated[row] == 0, "mesh overlap untruncated row");
                }
                retireConsumer(*operation, done, "mesh overlap consumer retirement");
                operation.reset();
                plan.reset();
            }
        }
    std::printf("mesh overlaps: %zu pairs compared, %zu meshes held whole, %zu cut, %zu beside\n", compared, held, cut,
                beside);
    require(compared > 7500 && held > 80 && cut > 800 && beside > 500, "mesh overlap coverage");
    std::shared_ptr<void> retiredDescending, retiredAscending;
    require(populations.release(descendingPopulation, retiredDescending) == OVSPATIALQUERY_API_SUCCESS &&
                populations.release(ascendingPopulation, retiredAscending) == OVSPATIALQUERY_API_SUCCESS,
            "mesh overlap population release");
    ovspatialquery_test_oracle_destroy_runtime(runtime);
    cuda.free(device);
    cuda.destroyEvent(producerReady);
    cuda.destroyStream(consumer);
    cuda.destroyStream(producer);
}

// Sphere overlaps at tiny scales. At s = 2^-75 the squares of the distances are subnormal
// floats: a triangle whose nearest corner is 1.125 * sqrt(2) s from the centre is touched by
// radius 1.625 s and missed by 1.5 s. Unscaled, the float gate's squared gap (two terms of
// about 1.27 * 2^-150) rounded above the squared radius and dropped the touched triangle.
// At r = 2^-600, beyond the float gate's range, a sphere of radius r centred 3r away misses
// one of radius r at the origin, and so does the same pair moved to x = 1, where the float
// gate's range holds but its margin dwarfs r; the exact sphere test squares their
// distances to zero, so only the double bounds gate keeps it from reporting either pair.
// Both spheres of scale s hold the tiny sphere at the origin.
void testTinySphereOverlapKeepsItsCandidate()
{
    CudaDriver                        cuda;
    const auto                        producer = cuda.createStream(), consumer = cuda.createStream();
    const auto                        producerReady = cuda.createEvent();
    ovspatialquery_backend_runtime_t* runtime       = nullptr;
    require(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS,
            "tiny overlap CUDA runtime");
    PopulationTable    populations(runtime);
    const double       s = std::ldexp(1.0, -75);
    TriangleMeshSource mesh{};
    mesh.key       = 2990;
    mesh.vertices  = {float(1.125 * s), float(1.125 * s), 0.f, float(1.375 * s), float(1.125 * s), 0.f,
                      float(1.125 * s), float(1.375 * s), 0.f};
    mesh.triangles = {0, 1, 2};
    numberSourceFeatures(mesh);
    std::shared_ptr<const PreparedTriangleMesh> prepared;
    require(prepareTriangleMesh(mesh, prepared) == OVSPATIALQUERY_API_SUCCESS, "tiny overlap mesh");
    std::shared_ptr<const TriangleObjectGeometry> placement;
    require(bindPreparedTriangleGeometry(prepared, {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, false,
                                         placement) == OVSPATIALQUERY_API_SUCCESS,
            "tiny overlap placement");
    std::shared_ptr<const ExactGeometry> geometry;
    require(createTriangleMeshExactGeometry(placement, geometry) == OVSPATIALQUERY_API_SUCCESS,
            "tiny overlap geometry");
    const double         r = std::ldexp(1.0, -600);
    SphereGeometrySource speck{};
    speck.center = {3 * r, 0.0, 0.0};
    speck.radius = r;
    std::shared_ptr<const ExactGeometry> sphere, moved;
    require(createSphereExactGeometry(speck, sphere) == OVSPATIALQUERY_API_SUCCESS, "tiny overlap sphere");
    speck.center = {1.0, 3 * r, 0.0};
    require(createSphereExactGeometry(speck, moved) == OVSPATIALQUERY_API_SUCCESS, "tiny overlap moved sphere");
    PopulationKey population;
    require(populations.create(
                2, 51, {exactObject(7, 1, geometry, 1), exactObject(8, 2, sphere, 1), exactObject(9, 3, moved, 1)},
                population) == OVSPATIALQUERY_API_SUCCESS,
            "tiny overlap population");
    struct Queries
    {
        double centers[4][3]{}, radii[4]{};
    } host;
    host.radii[0]      = 1.625 * s;
    host.radii[1]      = 1.5 * s;
    host.radii[2]      = r;
    host.radii[3]      = r;
    host.centers[3][0] = 1.0;
    const auto device  = cuda.allocate(sizeof(Queries));
    cuda.uploadAsync(device, &host, sizeof(Queries), producer);
    cuda.record(producerReady, producer);
    int64_t          rows[] = {4}, vectors[] = {4, 3}, rowStride[] = {1}, vectorStride[] = {3, 1};
    CudaFamilyLaunch launch{};
    launch.queryCount = 4;
    launch.stream     = reinterpret_cast<uintptr_t>(producer);
    launch.waitEvent  = reinterpret_cast<uintptr_t>(producerReady);
    launch.positions  = cudaTensor(device + offsetof(Queries, centers), {kDLFloat, 64, 1}, 2, vectors, vectorStride);
    launch.radii      = cudaTensor(device + offsetof(Queries, radii), {kDLFloat, 64, 1}, 1, rows, rowStride);
    CudaFamilyPlanConfig config{};
    config.family              = CudaFamilyKind::overlap;
    config.overlapKind         = CudaOverlapKind::sphere;
    config.precision           = CudaFamilyPrecision::exact;
    config.hitMode             = CudaFamilyHitMode::multiple;
    config.maximumQueryCount   = 4;
    config.maximumHitsPerQuery = 2;
    config.hitCapacityPerSlot  = 8;
    config.resultSlotCount     = 1;
    config.planRadius          = s;
    GenerationLease lease;
    require(populations.pin(population, lease) == OVSPATIALQUERY_API_SUCCESS, "tiny overlap pin");
    std::shared_ptr<CudaFamilyPlan> plan;
    require(createCudaFamilyPlan(config, plan) == OVSPATIALQUERY_API_SUCCESS, "tiny overlap plan");
    std::shared_ptr<CudaFamilyOperation> operation;
    require(prepareAndLaunchCudaFamilyPlan(plan, 0, launch, std::move(lease), operation) == OVSPATIALQUERY_API_SUCCESS,
            "tiny overlap launch");
    const auto ready = operation->reservation().resultReady.value;
    cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(ready));
    cuda.synchronize(reinterpret_cast<CudaDriver::Event>(ready));
    CudaFamilyResultColumns result{};
    require(operation->completedResult(result) == CudaFamilyCompletedResultOutcome::success, "tiny overlap completion");
    uint32_t statuses[4]{};
    uint64_t offsets[4]{}, counts[4]{}, ids[8]{};
    cuda.downloadAsync(statuses, deviceAddress(result.queryStatus), sizeof(statuses), consumer);
    cuda.downloadAsync(offsets, deviceAddress(result.hitOffsets), sizeof(offsets), consumer);
    cuda.downloadAsync(counts, deviceAddress(result.hitCounts), sizeof(counts), consumer);
    cuda.downloadAsync(ids, deviceAddress(result.objectIds), sizeof(ids), consumer);
    const auto done = cuda.createEvent();
    cuda.record(done, consumer);
    cuda.synchronize(done);
    const auto holds = [&](size_t row, uint64_t id)
    {
        for (uint64_t h = 0; h < counts[row]; ++h)
            if (ids[offsets[row] + h] == id)
                return true;
        return false;
    };
    for (size_t row = 0; row < 4; ++row)
        require(statuses[row] == cuda_family::kBatchRowSuccess, "tiny overlap status");
    require(counts[0] == 2 && holds(0, 7) && holds(0, 8), "tiny overlap: radius 1.625 s touches the triangle");
    require(counts[1] == 1 && holds(1, 8), "tiny overlap: radius 1.5 s misses the triangle");
    require(counts[2] == 0, "tiny overlap: disjoint spheres of radius 2^-600");
    require(counts[3] == 0, "tiny overlap: disjoint spheres of radius 2^-600 at x = 1");
    retireConsumer(*operation, done, "tiny overlap consumer retirement");
    operation.reset();
    plan.reset();
    std::shared_ptr<void> retired;
    require(populations.release(population, retired) == OVSPATIALQUERY_API_SUCCESS, "tiny overlap population release");
    ovspatialquery_test_oracle_destroy_runtime(runtime);
    cuda.free(device);
    cuda.destroyEvent(producerReady);
    cuda.destroyStream(consumer);
    cuda.destroyStream(producer);
}

// Closest-hit path rows for one CUDA raycast or sphere-sweep plan launch.
struct CudaPathRows
{
    std::vector<uint64_t> bits; // two words per row
    std::vector<double>   origins, minimum, maximum, radii;
    std::vector<float>    directions;

    void add(uint64_t rowBits, const double origin[3], const float direction[3], double rowMinimum, double rowMaximum,
             double radius)
    {
        bits.insert(bits.end(), {rowBits, 0u});
        origins.insert(origins.end(), origin, origin + 3);
        directions.insert(directions.end(), direction, direction + 3);
        minimum.push_back(rowMinimum);
        maximum.push_back(rowMaximum);
        radii.push_back(radius);
    }
};

struct CudaPathAnswers
{
    std::vector<uint8_t> found;
    std::vector<double>  distances, positions;
    std::vector<float>   normals;
};

// Launches `rows` as one exact closest-hit plan (row radii for a sphere sweep) against the
// population's head and reads the contacts back on the host, one answer per row.
CudaPathAnswers launchCudaPaths(CudaDriver& cuda, PopulationTable& populations, PopulationKey population,
                                CudaFamilyKind family, double contactTolerance, const CudaPathRows& rows)
{
    const size_t count    = rows.minimum.size();
    const auto   producer = cuda.createStream(), consumer = cuda.createStream();
    const auto   producerReady = cuda.createEvent();
    const auto   deviceBits = cuda.allocate(16 * count), deviceOrigins = cuda.allocate(24 * count),
               deviceDirections = cuda.allocate(12 * count), deviceMinimum = cuda.allocate(8 * count),
               deviceMaximum = cuda.allocate(8 * count), deviceRadii = cuda.allocate(8 * count);
    cuda.uploadAsync(deviceBits, rows.bits.data(), 16 * count, producer);
    cuda.uploadAsync(deviceOrigins, rows.origins.data(), 24 * count, producer);
    cuda.uploadAsync(deviceDirections, rows.directions.data(), 12 * count, producer);
    cuda.uploadAsync(deviceMinimum, rows.minimum.data(), 8 * count, producer);
    cuda.uploadAsync(deviceMaximum, rows.maximum.data(), 8 * count, producer);
    cuda.uploadAsync(deviceRadii, rows.radii.data(), 8 * count, producer);
    cuda.record(producerReady, producer);
    int64_t shape[] = {int64_t(count)}, vectors[] = {int64_t(count), 3}, stride[] = {1}, vectorStride[] = {3, 1},
            bitRows[] = {int64_t(count), 2};
    CudaFamilyLaunch launch{};
    launch.queryCount            = count;
    launch.stream                = reinterpret_cast<uintptr_t>(producer);
    launch.waitEvent             = reinterpret_cast<uintptr_t>(producerReady);
    launch.queryBits             = cudaBitsTensor(deviceBits, bitRows);
    launch.positions             = cudaTensor(deviceOrigins, {kDLFloat, 64, 1}, 2, vectors, vectorStride);
    launch.directionsOrRotations = cudaTensor(deviceDirections, {kDLFloat, 32, 1}, 2, vectors, vectorStride);
    launch.minimumDistances      = cudaTensor(deviceMinimum, {kDLFloat, 64, 1}, 1, shape, stride);
    launch.maximumDistances      = cudaTensor(deviceMaximum, {kDLFloat, 64, 1}, 1, shape, stride);
    CudaFamilyPlanConfig config{};
    config.family              = family;
    config.precision           = CudaFamilyPrecision::exact;
    config.hitMode             = CudaFamilyHitMode::closest;
    config.maximumQueryCount   = count;
    config.maximumHitsPerQuery = 1;
    config.hitCapacityPerSlot  = count;
    config.resultSlotCount     = 1;
    config.contactTolerance    = contactTolerance;
    config.resultFields        = kCudaFamilyResultPosition | kCudaFamilyResultNormal | kCudaFamilyResultDistance;
    if (family == CudaFamilyKind::sphereSweep)
    {
        config.planRadius = 1.0;
        launch.radii      = cudaTensor(deviceRadii, {kDLFloat, 64, 1}, 1, shape, stride);
    }
    GenerationLease lease;
    require(populations.pin(population, lease) == OVSPATIALQUERY_API_SUCCESS, "CUDA path pin");
    std::shared_ptr<CudaFamilyPlan> plan;
    require(createCudaFamilyPlan(config, plan) == OVSPATIALQUERY_API_SUCCESS, "CUDA path plan");
    std::shared_ptr<CudaFamilyOperation> operation;
    require(prepareAndLaunchCudaFamilyPlan(plan, 0, launch, std::move(lease), operation) == OVSPATIALQUERY_API_SUCCESS,
            "CUDA path launch");
    const auto columns = operation->reservation();
    cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(columns.resultReady.value));
    std::vector<uint32_t> status(count);
    std::vector<uint64_t> offsets(count), counts(count);
    CudaPathAnswers       stored;
    stored.distances.resize(count);
    stored.positions.resize(3 * count);
    stored.normals.resize(3 * count);
    cuda.downloadAsync(status.data(), deviceAddress(columns.queryStatus), 4 * count, consumer);
    cuda.downloadAsync(offsets.data(), deviceAddress(columns.hitOffsets), 8 * count, consumer);
    cuda.downloadAsync(counts.data(), deviceAddress(columns.hitCounts), 8 * count, consumer);
    cuda.downloadAsync(stored.distances.data(), deviceAddress(columns.distances), 8 * count, consumer);
    cuda.downloadAsync(stored.positions.data(), deviceAddress(columns.positions), 24 * count, consumer);
    cuda.downloadAsync(stored.normals.data(), deviceAddress(columns.normals), 12 * count, consumer);
    const auto done = cuda.createEvent();
    cuda.record(done, consumer);
    cuda.synchronize(done);
    CudaPathAnswers answers;
    answers.found.resize(count);
    answers.distances.resize(count);
    answers.positions.resize(3 * count);
    answers.normals.resize(3 * count);
    for (size_t row = 0; row < count; ++row)
    {
        require(status[row] == cuda_family::kBatchRowSuccess && counts[row] <= 1 &&
                    (!counts[row] || offsets[row] < count),
                "CUDA path row status");
        answers.found[row] = counts[row] == 1;
        if (!answers.found[row])
            continue;
        const size_t hit       = offsets[row];
        answers.distances[row] = stored.distances[hit];
        std::copy_n(&stored.positions[3 * hit], 3, &answers.positions[3 * row]);
        std::copy_n(&stored.normals[3 * hit], 3, &answers.normals[3 * row]);
    }
    retireConsumer(*operation, done, "CUDA path consumer retirement failed");
    operation.reset();
    plan.reset();
    for (const auto pointer : {deviceBits, deviceOrigins, deviceDirections, deviceMinimum, deviceMaximum, deviceRadii})
        cuda.free(pointer);
    cuda.destroyEvent(producerReady);
    cuda.destroyStream(consumer);
    cuda.destroyStream(producer);
    return answers;
}

// Stored-sphere rays, tubes and sweeps through CUDA plans, judged by the independent oracle
// (see stored_sphere_paths.h): a raycast plan, a raycast plan whose contact tolerance makes
// the tube, and a sphere-sweep plan with row radii, over one population whose participation
// bits select each row's sphere.
void testStoredSpherePathsMeetIndependentOracle()
{
    namespace paths = stored_sphere_paths;
    CudaDriver                        cuda;
    ovspatialquery_backend_runtime_t* runtime = nullptr;
    require(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS,
            "stored sphere CUDA runtime");
    PopulationTable               populations(runtime);
    std::vector<PopulationObject> objects;
    for (uint32_t target = 0; target < 3; ++target)
    {
        SphereGeometrySource sphere{};
        sphere.center = {paths::kTargets[target].center[0], paths::kTargets[target].center[1],
                         paths::kTargets[target].center[2]};
        sphere.radius = paths::kTargets[target].radius;
        std::shared_ptr<const ExactGeometry> geometry;
        require(createSphereExactGeometry(sphere, geometry) == OVSPATIALQUERY_API_SUCCESS,
                "stored sphere CUDA geometry");
        objects.push_back(exactObject(target + 1, target + 1, geometry, uint64_t{1} << target));
    }
    PopulationKey population;
    require(populations.create(2, 31, std::move(objects), population) == OVSPATIALQUERY_API_SUCCESS,
            "stored sphere CUDA population");
    const std::vector<paths::Row> rows = paths::rows();
    size_t                        hits = 0;
    for (int group = 0; group < 3; ++group)
    {
        // 0: plain rays, 1: tolerance tubes, 2: sphere sweeps.
        std::vector<size_t> members;
        CudaPathRows        launch;
        for (size_t index = 0; index < rows.size(); ++index)
        {
            const paths::Row& row = rows[index];
            if ((row.radius == 0 ? 0 : row.tube ? 1 : 2) != group)
                continue;
            members.push_back(index);
            launch.add(uint64_t{1} << row.target, row.origin, row.direction, row.minimum, row.maximum, row.radius);
        }
        const CudaPathAnswers answers = launchCudaPaths(
            cuda, populations, population, group == 2 ? CudaFamilyKind::sphereSweep : CudaFamilyKind::raycast,
            group == 1 ? paths::kTube : 0.0, launch);
        for (size_t k = 0; k < members.size(); ++k)
        {
            const char* failure = paths::verdict(rows[members[k]], answers.found[k], answers.distances[k],
                                                 &answers.positions[3 * k], &answers.normals[3 * k]);
            if (failure != nullptr)
                std::fprintf(stderr, "stored sphere CUDA path row %zu: %s\n", members[k], failure);
            require(failure == nullptr, "stored sphere CUDA path disagrees with the oracle");
            hits += answers.found[k];
        }
    }
    require(hits > rows.size() / 2, "stored sphere CUDA coverage");
    std::shared_ptr<void> retired;
    require(populations.release(population, retired) == OVSPATIALQUERY_API_SUCCESS,
            "stored sphere CUDA population release");
    ovspatialquery_test_oracle_destroy_runtime(runtime);
}

// Rays and sphere sweeps whose minimum lies past their first contact with each solid family
// (sphere, box, capsule, convex hull), so the clipped start already overlaps it: CUDA must
// report the minimum with the host's contact (ADR-0018), which the host test pins to the
// contact of a start moved there. Directions are 9e-6 off unit squared length. Rows starting on
// the capsule's segment, which has no unique nearest surface point, take the host's contact too.
void testSolidsReportInitialOverlapAtTheMinimum()
{
    CudaDriver                        cuda;
    ovspatialquery_backend_runtime_t* runtime = nullptr;
    require(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS,
            "initial-overlap CUDA runtime");
    PopulationTable                      populations(runtime);
    std::vector<PopulationObject>        objects = fiveExactFamilies();
    std::shared_ptr<const ExactGeometry> solids[4];
    for (size_t k = 0; k < 4; ++k)
        solids[k] = objects[k].exactGeometry;
    PopulationKey population;
    require(populations.create(2, 31, std::move(objects), population) == OVSPATIALQUERY_API_SUCCESS,
            "initial-overlap CUDA population");
    const float direction[3] = {float(std::sqrt(1 + 9e-6)), 0.0f, 0.0f};
    for (const bool sweep : {false, true})
    {
        const double                  radius = sweep ? 0.25 : 0.0;
        CudaPathRows                  rows;
        std::vector<ExactGeometryHit> expected(14);
        for (size_t k = 0; k < 8; ++k)
        {
            // Two heights through each solid: the sphere, box and capsule sit on the x axis
            // (x = 2, 5, 8) and the convex hull is centred at (11, 3, 0).
            const size_t     solid     = k % 4;
            const double     origin[3] = {-1.0, (solid == 3 ? 3.0 : 0.0) + (k < 4 ? -0.25 : 0.4), 0.05};
            ExactGeometryHit first{};
            const bool       touched = sweep ? solids[solid]->sweepSphere(origin, radius, direction, 0.0, 30.0, first)
                                             : solids[solid]->raycast(origin, direction, 0.0, 30.0, false, first);
            require(touched, "initial-overlap CUDA rows must reach their solid");
            const double minimum = first.distance + 0.1;
            require(sweep ? solids[solid]->sweepSphere(origin, radius, direction, minimum, 30.0, expected[k])
                          : solids[solid]->raycast(origin, direction, minimum, 30.0, false, expected[k]),
                    "initial-overlap host contact");
            require(expected[k].distance == minimum, "host solid did not report its overlapping start");
            rows.add(uint64_t{1} << solid, origin, direction, minimum, 30.0, sweep ? radius : 1.0);
        }
        // Inside the capsule's span and at its end, crossing the segment, along it and into it.
        const double onSegment[2][3] = {{8.0, 0.3, 0.0}, {8.0, 1.0, 0.0}};
        const float  motions[3][3]   = {{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, -1.0f, 0.0f}};
        for (size_t k = 8; k < 14; ++k)
        {
            const double* origin = onSegment[(k - 8) / 3];
            const float*  motion = motions[(k - 8) % 3];
            require(sweep ? solids[2]->sweepSphere(origin, radius, motion, 0.0, 30.0, expected[k])
                          : solids[2]->raycast(origin, motion, 0.0, 30.0, false, expected[k]),
                    "on-segment host contact");
            rows.add(uint64_t{1} << 2, origin, motion, 0.0, 30.0, sweep ? radius : 1.0);
        }
        const CudaPathAnswers answers = launchCudaPaths(
            cuda, populations, population, sweep ? CudaFamilyKind::sphereSweep : CudaFamilyKind::raycast, 0.0, rows);
        for (size_t k = 0; k < 14; ++k)
        {
            require(answers.found[k] && answers.distances[k] == rows.minimum[k],
                    "CUDA solid did not report its overlapping start at the minimum");
            for (int a = 0; a < 3; ++a)
                require(std::abs(answers.positions[3 * k + a] - expected[k].position[a]) <= 1e-9 &&
                            std::abs(answers.normals[3 * k + a] - expected[k].normal[a]) <= 1e-5,
                        "CUDA initial-overlap contact differs from the host");
        }
    }
    std::shared_ptr<void> retired;
    require(populations.release(population, retired) == OVSPATIALQUERY_API_SUCCESS,
            "initial-overlap CUDA population release");
    ovspatialquery_test_oracle_destroy_runtime(runtime);
}

void testPlacedConvexKeepsFeaturesAtLargeTranslations()
{
    CudaDriver                        cuda;
    const auto                        producer = cuda.createStream(), consumer = cuda.createStream();
    const auto                        producerReady = cuda.createEvent();
    ovspatialquery_backend_runtime_t* runtime       = nullptr;
    require(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS,
            "CUDA local convex runtime creation failed");
    PopulationTable  populations(runtime);
    const auto       hull  = unitCubeHull();
    constexpr size_t count = 7;
    struct Queries
    {
        double origins[count][3]{}, minimum[count]{}, maximum[count]{};
        float  directions[count][3]{}, rotations[count][4]{};
    } host;
    const auto device = cuda.allocate(sizeof(host));
    int64_t    rows[]{count}, vectors[]{count, 3}, quaternions[]{count, 4}, rowStrides[]{1}, vectorStrides[]{3, 1},
        rotationStrides[]{4, 1};
    const auto at         = [&](size_t offset) { return device + offset; };
    const auto origins    = cudaTensor(at(offsetof(Queries, origins)), {kDLFloat, 64, 1}, 2, vectors, vectorStrides);
    const auto directions = cudaTensor(at(offsetof(Queries, directions)), {kDLFloat, 32, 1}, 2, vectors, vectorStrides);
    const auto rotations =
        cudaTensor(at(offsetof(Queries, rotations)), {kDLFloat, 32, 1}, 2, quaternions, rotationStrides);
    const auto minimum = cudaTensor(at(offsetof(Queries, minimum)), {kDLFloat, 64, 1}, 1, rows, rowStrides);
    const auto maximum = cudaTensor(at(offsetof(Queries, maximum)), {kDLFloat, 64, 1}, 1, rows, rowStrides);
    for (size_t poseIndex = 0; poseIndex < 3; ++poseIndex)
    {
        const double                 translation = poseIndex == 0 ? 396 : poseIndex == 1 ? 0x1p59 : 0;
        const double                 scale       = poseIndex == 2 ? 0x1p-60 : 2;
        const double                 angle       = poseIndex == 0 ? .25 : 0;
        const double                 c = std::cos(angle), sn = std::sin(angle);
        const std::array<double, 16> pose{scale * c, -scale * sn, 0,     translation, scale * sn, scale * c, 0, 0,
                                          0,         0,           scale, 0,           0,          0,         0, 1};
        std::shared_ptr<const ExactGeometry> geometry;
        require(createConvexExactGeometry(hull, pose, geometry) == ConvexConstructionOutcome::success,
                "CUDA local convex placement failed");
        PopulationKey population;
        require(populations.create(2, 31, {exactObject(1, 404, geometry, UINT64_MAX)}, population) ==
                    OVSPATIALQUERY_API_SUCCESS,
                "CUDA local convex population failed");
        for (size_t family = 0; family < (poseIndex == 2 ? 1u : 4u); ++family)
        {
            host                = {};
            const double radius = .125 * scale;
            for (size_t row = 0; row < count; ++row)
            {
                host.origins[row][0]    = translation;
                host.origins[row][1]    = .17 * scale;
                host.origins[row][2]    = 3 * scale;
                host.directions[row][2] = -1;
                host.maximum[row]       = 10 * scale;
                host.rotations[row][2]  = float(std::sin(.17));
                host.rotations[row][3]  = float(std::cos(.17));
            }
            host.maximum[1]    = .25 * scale;
            host.origins[2][2] = .5 * scale;
            host.maximum[2]    = 0; // Unique nearest face at an initial contact.
            host.minimum[3] = host.maximum[3] = 2.5 * scale;
            host.minimum[4]                   = 4.5 * scale;
            host.origins[5][0] = poseIndex == 1 ? std::nextafter(translation, INFINITY) : translation + 4 * scale;
            // Far in local units, near enough that the world contact keeps its precision.
            host.origins[6][0]    = translation + 0x1p32 * scale;
            host.origins[6][2]    = .5 * scale;
            host.directions[6][0] = -1;
            host.directions[6][2] = 0;
            host.maximum[6]       = 0x1p33 * scale;
            if (family >= 2)
            {
                const double heights[count]{1.0625, 1.5, .5, 0, -1.0625, 0, 0};
                for (size_t row = 0; row < count; ++row)
                    host.origins[row][2] = heights[row] * scale;
            }
            CudaFamilyPlanConfig config{};
            config.family      = family == 0   ? CudaFamilyKind::raycast
                                 : family == 1 ? CudaFamilyKind::sphereSweep
                                               : CudaFamilyKind::overlap;
            config.precision   = CudaFamilyPrecision::exact;
            config.hitMode     = family < 2 ? CudaFamilyHitMode::closest : CudaFamilyHitMode::any;
            config.overlapKind = family == 2 ? CudaOverlapKind::sphere : CudaOverlapKind::box;
            config.planRadius  = radius;
            for (double& half : config.planHalfExtents)
                half = radius;
            config.resultFields =
                family < 2 ? kCudaFamilyResultPosition | kCudaFamilyResultNormal | kCudaFamilyResultDistance : 0;
            config.maximumQueryCount = config.hitCapacityPerSlot = count;
            config.maximumHitsPerQuery = config.resultSlotCount = 1;
            bool             expected[count]{};
            ExactGeometryHit contacts[count]{};
            for (size_t row = 0; row < count; ++row)
            {
                ovspatialquery_api_status_t status = OVSPATIALQUERY_API_SUCCESS;
                if (family == 0)
                    expected[row] = geometry->raycast(host.origins[row], host.directions[row], host.minimum[row],
                                                      host.maximum[row], false, contacts[row], nullptr, &status);
                else if (family == 1)
                    expected[row] =
                        geometry->sweepSphere(host.origins[row], radius, host.directions[row], host.minimum[row],
                                              host.maximum[row], contacts[row], nullptr, &status);
                else
                {
                    ovspatialquery_overlap_shape_t shape{};
                    shape.kind = family == 2 ? OVSPATIALQUERY_OVERLAP_SPHERE : OVSPATIALQUERY_OVERLAP_BOX;
                    if (family == 2)
                    {
                        std::copy_n(host.origins[row], 3, shape.data.sphere.center);
                        shape.data.sphere.radius = radius;
                    }
                    else
                    {
                        std::copy_n(host.origins[row], 3, shape.data.box.center);
                        std::copy_n(host.rotations[row], 4, shape.data.box.rotation_xyzw);
                        std::fill_n(shape.data.box.half_extents, 3, radius);
                    }
                    expected[row] = geometry->overlaps(shape, nullptr, nullptr, &status);
                }
                require(status == OVSPATIALQUERY_API_SUCCESS, "CUDA local convex CPU reference failed");
            }
            require(expected[0] && !expected[1] && expected[2] && expected[3] && !expected[5],
                    "CUDA local convex fixture lost hit/miss coverage");
            cuda.uploadAsync(device, &host, sizeof(host), producer);
            cuda.record(producerReady, producer);
            CudaFamilyLaunch launch{};
            launch.queryCount = count;
            launch.positions  = origins;
            if (family < 2)
            {
                launch.directionsOrRotations = directions;
                launch.minimumDistances      = minimum;
                launch.maximumDistances      = maximum;
            }
            else if (family == 3)
                launch.directionsOrRotations = rotations;
            launch.stream    = reinterpret_cast<uintptr_t>(producer);
            launch.waitEvent = reinterpret_cast<uintptr_t>(producerReady);
            GenerationLease generation;
            require(populations.pin(population, generation) == OVSPATIALQUERY_API_SUCCESS,
                    "CUDA local convex pin failed");
            std::shared_ptr<CudaFamilyPlan> plan;
            require(createCudaFamilyPlan(config, plan) == OVSPATIALQUERY_API_SUCCESS, "CUDA local convex plan failed");
            std::shared_ptr<CudaFamilyOperation> operation;
            require(prepareAndLaunchCudaFamilyPlan(plan, 0, launch, std::move(generation), operation) ==
                        OVSPATIALQUERY_API_SUCCESS,
                    "CUDA local convex launch failed");
            const auto view = operation->reservation();
            cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(view.resultReady.value));
            uint32_t statuses[count]{};
            uint64_t offsets[count]{}, counts[count]{}, ids[count]{};
            uint8_t  truncated[count]{};
            double   distances[count]{}, positions[count][3]{};
            float    normals[count][3]{};
            cuda.downloadAsync(statuses, deviceAddress(view.queryStatus), sizeof(statuses), consumer);
            cuda.downloadAsync(offsets, deviceAddress(view.hitOffsets), sizeof(offsets), consumer);
            cuda.downloadAsync(counts, deviceAddress(view.hitCounts), sizeof(counts), consumer);
            cuda.downloadAsync(ids, deviceAddress(view.objectIds), sizeof(ids), consumer);
            cuda.downloadAsync(truncated, deviceAddress(view.truncated), sizeof(truncated), consumer);
            if (family < 2)
            {
                cuda.downloadAsync(distances, deviceAddress(view.distances), sizeof(distances), consumer);
                cuda.downloadAsync(positions, deviceAddress(view.positions), sizeof(positions), consumer);
                cuda.downloadAsync(normals, deviceAddress(view.normals), sizeof(normals), consumer);
            }
            const auto done = cuda.createEvent();
            cuda.record(done, consumer);
            cuda.synchronize(done);
            for (size_t row = 0; row < count; ++row)
            {
                if (statuses[row] != cuda_family::kBatchRowSuccess || truncated[row] != 0 ||
                    counts[row] != uint64_t(expected[row]))
                    std::fprintf(stderr,
                                 "CUDA convex pose=%zu family=%zu row=%zu status=%u truncated=%u count=%llu "
                                 "expected=%u origin=(%.17g,%.17g,%.17g) range=(%.17g,%.17g)\n",
                                 poseIndex, family, row, statuses[row], unsigned(truncated[row]),
                                 static_cast<unsigned long long>(counts[row]), unsigned(expected[row]),
                                 host.origins[row][0], host.origins[row][1], host.origins[row][2], host.minimum[row],
                                 host.maximum[row]);
                require(statuses[row] == cuda_family::kBatchRowSuccess && truncated[row] == 0 &&
                            counts[row] == uint64_t(expected[row]),
                        "CUDA local convex membership/status mismatch");
                if (!expected[row])
                    continue;
                const auto hit = offsets[row];
                require(hit < count && ids[hit] == 1, "CUDA local convex identity mismatch");
                if (family >= 2)
                    continue;
                require(std::abs(distances[hit] - contacts[row].distance) <=
                            2e-6 * std::max(1.0, contacts[row].distance),
                        "CUDA local convex distance mismatch");
                for (size_t axis = 0; axis < 3; ++axis)
                    require(std::isfinite(positions[hit][axis]) &&
                                std::abs(positions[hit][axis] - contacts[row].position[axis]) <= 2e-6 &&
                                std::abs(normals[hit][axis] - contacts[row].normal[axis]) <= 2e-6,
                            "CUDA local convex contact mismatch");
            }
            operation->releaseReadyDependencies();
            retireConsumer(*operation, done, "CUDA local convex consumer retirement failed");
        }
        std::shared_ptr<void> retired;
        require(populations.release(population, retired) == OVSPATIALQUERY_API_SUCCESS,
                "CUDA local convex population release failed");
    }
    ovspatialquery_test_oracle_destroy_runtime(runtime);
    cuda.free(device);
    cuda.destroyEvent(producerReady);
    cuda.destroyStream(consumer);
    cuda.destroyStream(producer);
}

void testPlacedMeshBvhKeepsBoundaryHitsAndReusesCatalogue(CudaFamilyPrecision precision)
{
    CudaDriver                        cuda;
    const auto                        producer = cuda.createStream(), consumer = cuda.createStream();
    const auto                        producerReady = cuda.createEvent();
    ovspatialquery_backend_runtime_t* runtime       = nullptr;
    require(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS,
            "CUDA placed-mesh runtime creation failed");
    PopulationTable populations(runtime);

    TriangleMeshSource mesh{};
    mesh.key           = 88u;
    mesh.vertices      = {0.0f, -1.0f, -1.0f, 0.0f, -1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, -1.0f};
    mesh.triangles     = {0u, 1u, 2u, 0u, 2u, 3u};
    mesh.sourceFaces   = {900u, 901u};
    mesh.sourceCorners = {0u, 1u, 2u, 0u, 2u, 3u};
    std::shared_ptr<const PreparedTriangleMesh> preparedMesh;
    require(prepareTriangleMesh(mesh, preparedMesh) == OVSPATIALQUERY_API_SUCCESS,
            "CUDA placed-mesh preparation failed");
    const double                                  translations[2][3]{{17.0, 23.0, 31.0}, {1.0e9, -1.0e9, 1.0e9}};
    std::array<double, 16u>                       placements[2]{};
    std::shared_ptr<const TriangleObjectGeometry> prepared[2];
    std::vector<PopulationObject>                 objects;
    std::shared_ptr<const ExactGeometry>          geometries[2];
    for (size_t index = 0u; index < 2u; ++index)
    {
        const double* translation = translations[index];
        placements[index]         = {0.0, -2.0, 0.0, translation[0], 2.0, 0.0, 0.0, translation[1],
                                     0.0, 0.0,  2.0, translation[2], 0.0, 0.0, 0.0, 1.0};
        require(bindPreparedTriangleGeometry(preparedMesh, placements[index], true, prepared[index]) ==
                    OVSPATIALQUERY_API_SUCCESS,
                "CUDA placed-mesh preparation failed");
        require(createTriangleMeshExactGeometry(prepared[index], geometries[index]) == OVSPATIALQUERY_API_SUCCESS,
                "CUDA placed-mesh exact geometry creation failed");
        objects.push_back(exactObject(index + 1u, 808u, geometries[index], uint64_t{1u} << index));
    }
    PopulationKey population;
    require(populations.create(2u, 31, std::move(objects), population) == OVSPATIALQUERY_API_SUCCESS,
            "CUDA placed-mesh population creation failed");

    struct Queries
    {
        uint64_t bits[2][2]{{1u, 0u}, {2u, 0u}};
        double   origins[2][3]{};
        float    directions[2][3]{};
        double   maximum[2]{20.0, 20.0};
    } host;
    // The first ray lies on the shared triangle diagonal; the second reaches
    // the transformed corner at a large world translation.
    host.origins[0][0]    = translations[0][0];
    host.origins[0][1]    = translations[0][1] - 6.0;
    host.origins[0][2]    = translations[0][2];
    host.origins[1][0]    = translations[1][0] - 2.0;
    host.origins[1][1]    = translations[1][1] - 6.0;
    host.origins[1][2]    = translations[1][2] + 2.0;
    host.directions[0][1] = host.directions[1][1] = 1.0f;
    ExactGeometryHit cpuHits[2]{};
    for (size_t row = 0u; row < 2u; ++row)
    {
        ovspatialquery_api_status_t status = OVSPATIALQUERY_API_SUCCESS;
        require(geometries[row]->raycast(host.origins[row], host.directions[row], 0.0, host.maximum[row], false,
                                         cpuHits[row], nullptr, &status) &&
                    status == OVSPATIALQUERY_API_SUCCESS,
                "CUDA placed-mesh CPU boundary reference missed");
    }

    const auto device = cuda.allocate(sizeof(host));
    cuda.uploadAsync(device, &host, sizeof(host), producer);
    cuda.record(producerReady, producer);
    int64_t          rows[]{2}, vectors[]{2, 3}, rowStrides[]{1}, vectorStrides[]{3, 1};
    const auto       at = [&](size_t offset) { return device + offset; };
    CudaFamilyLaunch launch{};
    launch.queryCount = 2u;
    int64_t bitRows[]{rows[0], 2};
    launch.queryBits = cudaBitsTensor(at(offsetof(Queries, bits)), bitRows);
    launch.positions = cudaTensor(at(offsetof(Queries, origins)), {kDLFloat, 64u, 1u}, 2, vectors, vectorStrides);
    launch.directionsOrRotations =
        cudaTensor(at(offsetof(Queries, directions)), {kDLFloat, 32u, 1u}, 2, vectors, vectorStrides);
    launch.maximumDistances = cudaTensor(at(offsetof(Queries, maximum)), {kDLFloat, 64u, 1u}, 1, rows, rowStrides);
    launch.stream           = reinterpret_cast<uintptr_t>(producer);
    launch.waitEvent        = reinterpret_cast<uintptr_t>(producerReady);
    CudaFamilyPlanConfig config{};
    config.family              = CudaFamilyKind::raycast;
    config.precision           = precision;
    config.hitMode             = CudaFamilyHitMode::closest;
    config.resultFields        = kCudaFamilyResultDistance | kCudaFamilyResultSourceFeature;
    config.maximumQueryCount   = 2u;
    config.maximumHitsPerQuery = 1u;
    config.hitCapacityPerSlot  = 2u;
    config.resultSlotCount     = 1u;
    std::shared_ptr<CudaFamilyPlan> plan;
    require(createCudaFamilyPlan(config, plan) == OVSPATIALQUERY_API_SUCCESS, "CUDA placed-mesh plan failed");
    // One launch against a pinned generation; `references` supply the CPU exact
    // truth per row, and the measured charge must publish exactly the expected
    // catalogue tables and newly resident definitions.
    // `faceBases[row]` is the first authored face id of the row's definition, so
    // a hit through a stale or wrong definition is caught by its source feature.
    const auto runAttempt = [&](GenerationLease generation, const std::shared_ptr<const ExactGeometry>* references,
                                const uint64_t* faceBases, bool expectTables, size_t expectDefinitions,
                                const char* label)
    {
        CudaFamilyStagingCharge charge{};
        require(plan->measure(launch, generation, charge) == OVSPATIALQUERY_API_SUCCESS && charge.staging != nullptr &&
                    (charge.publishedDeviceBytes != 0u) == expectTables &&
                    charge.publishedDefinitionBytes.size() == expectDefinitions,
                label);
        std::shared_ptr<CudaFamilyOperation> preparedLaunch;
        require(plan->prepare(launch, std::move(generation), std::move(charge.staging), preparedLaunch) ==
                    OVSPATIALQUERY_API_SUCCESS,
                "CUDA placed-mesh launch preparation failed");
        std::shared_ptr<CudaFamilyOperation> operation;
        require(plan->launch(0u, std::move(preparedLaunch), operation) == OVSPATIALQUERY_API_SUCCESS,
                "CUDA placed-mesh launch failed");
        const auto view = operation->reservation();
        cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(view.resultReady.value));
        uint32_t statuses[2]{};
        uint64_t offsets[2]{}, counts[2]{}, ids[2]{}, faces[2]{};
        double   distances[2]{};
        cuda.downloadAsync(statuses, deviceAddress(view.queryStatus), sizeof(statuses), consumer);
        cuda.downloadAsync(offsets, deviceAddress(view.hitOffsets), sizeof(offsets), consumer);
        cuda.downloadAsync(counts, deviceAddress(view.hitCounts), sizeof(counts), consumer);
        cuda.downloadAsync(ids, deviceAddress(view.objectIds), sizeof(ids), consumer);
        cuda.downloadAsync(faces, deviceAddress(view.sourceFaceIndices), sizeof(faces), consumer);
        cuda.downloadAsync(distances, deviceAddress(view.distances), sizeof(distances), consumer);
        const auto done = cuda.createEvent();
        cuda.record(done, consumer);
        cuda.synchronize(done);
        for (size_t row = 0u; row < 2u; ++row)
        {
            ExactGeometryHit            reference{};
            ovspatialquery_api_status_t status = OVSPATIALQUERY_API_SUCCESS;
            const bool expected                = references[row]->raycast(host.origins[row], host.directions[row], 0.0,
                                                                          host.maximum[row], false, reference, nullptr, &status);
            require(status == OVSPATIALQUERY_API_SUCCESS && statuses[row] == cuda_family::kBatchRowSuccess &&
                        counts[row] == (expected ? 1u : 0u),
                    "CUDA placed-mesh row outcome diverged from CPU exact geometry");
            if (!expected)
            {
                continue;
            }
            const uint64_t hit = offsets[row];
            require(hit < 2u && ids[hit] == row + 1u &&
                        (faces[hit] == faceBases[row] || faces[hit] == faceBases[row] + 1u) &&
                        std::abs(distances[hit] - reference.distance) <= 1.0e-9,
                    "CUDA placed-mesh boundary/cache result diverged from CPU exact geometry");
        }
        operation->releaseReadyDependencies();
        retireConsumer(*operation, done, "CUDA placed-mesh consumer retirement failed");
    };
    const auto pinHead = [&]()
    {
        GenerationLease generation;
        require(populations.pin(population, generation) == OVSPATIALQUERY_API_SUCCESS,
                "CUDA placed-mesh generation pin failed");
        return generation;
    };
    const auto pinRetained = [&](ovspatialquery_ordinal_t ordinal)
    {
        GenerationLease generation;
        require(populations.pinExact(population, ordinal, generation) == OVSPATIALQUERY_API_SUCCESS,
                "CUDA placed-mesh retained generation pin failed");
        return generation;
    };
    // First use publishes the tables and the one definition both instances share;
    // the same generation then reuses the whole catalogue.
    const uint64_t sharedFaces[2]{900u, 900u};
    runAttempt(pinHead(), geometries, sharedFaces, true, 1u, "CUDA placed-mesh first catalogue measurement changed");
    runAttempt(pinHead(), geometries, sharedFaces, false, 0u, "CUDA placed-mesh catalogue cache measurement changed");

    // A transform-only republication keeps the same prepared mesh: only its
    // tables are published, the resident definition is shared.
    std::vector<PopulationObject>        movedObjects;
    std::shared_ptr<const ExactGeometry> movedGeometries[2];
    for (size_t index = 0u; index < 2u; ++index)
    {
        std::shared_ptr<TriangleObjectGeometry> moved = std::make_shared<TriangleObjectGeometry>(*prepared[index]);
        moved->worldFromLocal[3] += 5.0;
        moved->worldBounds.minimum[0] += 5.0;
        moved->worldBounds.maximum[0] += 5.0;
        require(createTriangleMeshExactGeometry(moved, movedGeometries[index]) == OVSPATIALQUERY_API_SUCCESS,
                "CUDA placed-mesh moved exact geometry creation failed");
        movedObjects.push_back(exactObject(index + 1u, 808u, movedGeometries[index], uint64_t{1u} << index));
    }
    const auto publish = [&](ovspatialquery_ordinal_t ordinal, std::vector<PopulationObject> published)
    {
        GenerationLease base;
        require(populations.beginUpdate(population, ordinal, nullptr, base) == OVSPATIALQUERY_API_SUCCESS,
                "CUDA placed-mesh republication did not begin");
        GenerationLease generation;
        require(populations.buildGeneration(ordinal, std::move(published), {}, generation) ==
                    OVSPATIALQUERY_API_SUCCESS,
                "CUDA placed-mesh republication build failed");
        require(populations.publishUpdate(population, generation) == OVSPATIALQUERY_API_SUCCESS,
                "CUDA placed-mesh republication failed");
        generation.reset();
        base.reset();
        populations.finishUpdate(population);
    };
    publish(32, std::move(movedObjects));
    host.origins[0][0] += 5.0;
    host.origins[1][0] += 5.0;
    cuda.uploadAsync(device, &host, sizeof(host), producer);
    cuda.record(producerReady, producer);
    runAttempt(pinHead(), movedGeometries, sharedFaces, true, 0u,
               "CUDA moved-mesh republication rebuilt its definition");
    // Alternating between the retained generation and the head reuses the catalogue the
    // plan kept for each: neither tables nor definitions are published again.
    runAttempt(pinRetained(31), geometries, sharedFaces, false, 0u, "CUDA retained generation rebuilt its catalogue");
    runAttempt(pinHead(), movedGeometries, sharedFaces, false, 0u, "CUDA head generation rebuilt its catalogue");
    runAttempt(pinHead(), movedGeometries, sharedFaces, false, 0u, "CUDA head generation lost its catalogue");

    // Changed authored content is a new definition; the unchanged one stays shared.
    TriangleMeshSource changedMesh{};
    changedMesh.key           = 89u;
    changedMesh.vertices      = {0.0f, -2.0f, -2.0f, 0.0f, -2.0f, 2.0f, 0.0f, 2.0f, 2.0f, 0.0f, 2.0f, -2.0f};
    changedMesh.triangles     = {0u, 1u, 2u, 0u, 2u, 3u};
    changedMesh.sourceFaces   = {902u, 903u};
    changedMesh.sourceCorners = {0u, 1u, 2u, 0u, 2u, 3u};
    std::shared_ptr<const PreparedTriangleMesh>   changedMeshPrepared;
    std::shared_ptr<const TriangleObjectGeometry> changedPrepared;
    require(prepareTriangleMesh(changedMesh, changedMeshPrepared) == OVSPATIALQUERY_API_SUCCESS &&
                bindPreparedTriangleGeometry(changedMeshPrepared, placements[0], true, changedPrepared) ==
                    OVSPATIALQUERY_API_SUCCESS,
            "CUDA changed-mesh preparation failed");
    std::shared_ptr<const ExactGeometry> changedGeometries[2];
    require(createTriangleMeshExactGeometry(changedPrepared, changedGeometries[0]) == OVSPATIALQUERY_API_SUCCESS,
            "CUDA changed-mesh exact geometry creation failed");
    changedGeometries[1] = movedGeometries[1];
    std::vector<PopulationObject> changedObjects;
    for (size_t index = 0u; index < 2u; ++index)
    {
        changedObjects.push_back(exactObject(index + 1u, 808u, changedGeometries[index], uint64_t{1u} << index));
    }
    publish(33, std::move(changedObjects));
    host.origins[0][0] -= 5.0;
    cuda.uploadAsync(device, &host, sizeof(host), producer);
    cuda.record(producerReady, producer);
    const uint64_t changedFaces[2]{902u, 900u};
    runAttempt(pinHead(), changedGeometries, changedFaces, true, 1u,
               "CUDA changed mesh did not become a new definition");
    runAttempt(pinHead(), changedGeometries, changedFaces, false, 0u,
               "CUDA changed-mesh generation lost its catalogue");

    // Two more transform-only republications of that content: the second evicts
    // the catalogue of ordinal 33, whose tables have the same size, and the new
    // generation's tables must replace them wherever they are stored.
    const std::shared_ptr<const TriangleObjectGeometry> changedBases[2]{changedPrepared, prepared[1]};
    double                                              shifts[2]{0.0, 5.0};
    for (const ovspatialquery_ordinal_t ordinal : {34, 35})
    {
        std::vector<PopulationObject>        steppedObjects;
        std::shared_ptr<const ExactGeometry> steppedGeometries[2];
        for (size_t index = 0u; index < 2u; ++index)
        {
            shifts[index] += 7.0;
            std::shared_ptr<TriangleObjectGeometry> stepped =
                std::make_shared<TriangleObjectGeometry>(*changedBases[index]);
            stepped->worldFromLocal[3] += shifts[index];
            stepped->worldBounds.minimum[0] += shifts[index];
            stepped->worldBounds.maximum[0] += shifts[index];
            require(createTriangleMeshExactGeometry(stepped, steppedGeometries[index]) == OVSPATIALQUERY_API_SUCCESS,
                    "CUDA stepped-mesh exact geometry creation failed");
            steppedObjects.push_back(exactObject(index + 1u, 808u, steppedGeometries[index], uint64_t{1u} << index));
            host.origins[index][0] += 7.0;
        }
        publish(ordinal, std::move(steppedObjects));
        cuda.uploadAsync(device, &host, sizeof(host), producer);
        cuda.record(producerReady, producer);
        runAttempt(pinHead(), steppedGeometries, changedFaces, true, 0u,
                   "CUDA stepped-mesh republication rebuilt a definition");
    }
    plan.reset();
    std::shared_ptr<void> retired;
    require(populations.release(population, retired) == OVSPATIALQUERY_API_SUCCESS,
            "CUDA placed-mesh cache retained its population after plan destruction");
    ovspatialquery_test_oracle_destroy_runtime(runtime);
    cuda.free(device);
    cuda.destroyEvent(producerReady);
    cuda.destroyStream(consumer);
    cuda.destroyStream(producer);
}

void testCandidateMixedRowsAndForcedCapacity(CudaFamilyKind family)
{
    CudaDriver                        cuda;
    const CudaDriver::Stream          producer      = cuda.createStream();
    const CudaDriver::Stream          consumer      = cuda.createStream();
    const CudaDriver::Event           producerReady = cuda.createEvent();
    const CudaDriver::Event           consumerDone  = cuda.createEvent();
    ovspatialquery_backend_runtime_t* runtime       = nullptr;
    require(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS,
            "CUDA mixed-row oracle runtime creation failed");
    PopulationTable populations(runtime);
    PopulationKey   population;
    require(populations.create(2u, 1, candidateObjects(), population) == OVSPATIALQUERY_API_SUCCESS,
            "CUDA mixed-row population creation failed");
    TargetResolver resolver(populations);
    ResolvedTarget resolved;
    TargetRequest  request{};
    request.target.population = population;
    require(resolver.resolve(request, resolved) == TargetResolveOutcome::success,
            "CUDA mixed-row target resolution failed");

    CudaFamilyPlanConfig config{};
    config.family    = family;
    config.precision = CudaFamilyPrecision::boundsCandidate;
    config.hitMode   = family == CudaFamilyKind::sphereSweep ? CudaFamilyHitMode::closest : CudaFamilyHitMode::multiple;
    config.overlapKind         = CudaOverlapKind::aabb;
    config.resultFields        = family == CudaFamilyKind::overlap ? 0u : kCudaFamilyResultDistance;
    config.maximumQueryCount   = 3u;
    config.maximumHitsPerQuery = 2u;
    config.hitCapacityPerSlot  = 1u;
    config.resultSlotCount     = 1u;
    config.planRadius          = 0.5;
    config.planHalfExtents[0]  = 4.0;
    config.planHalfExtents[1]  = 2.0;
    config.planHalfExtents[2]  = 2.0;
    std::shared_ptr<CudaFamilyPlan> plan;
    require(createCudaFamilyPlan(config, plan) == OVSPATIALQUERY_API_SUCCESS && plan != nullptr,
            "CUDA mixed-row plan creation failed");

    struct QueryData
    {
        uint64_t bits[3][2]{{0u, 0u}, {0u, 0u}, {2u, 0u}};
        double   origins[3][3]{};
        float    directions[3][3]{
            {1.0f, 0.0f, 0.0f},
            {0.0f, 0.0f, 0.0f},
            {1.0f, 0.0f, 0.0f},
        };
        double   maximumDistances[3]{10.0, 10.0, 10.0};
        uint64_t maximumHits[3]{2u, 2u, 1u};
        double   aabbCenters[3][3]{
            {4.0, 0.0, 0.0},
            {std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0},
            {8.0, 0.0, 0.0},
        };
    } host;
    const CudaDriver::DevicePointer device = cuda.allocate(sizeof(host));
    cuda.uploadAsync(device, &host, sizeof(host), producer);
    cuda.record(producerReady, producer);
    int64_t    rows[]          = {3};
    int64_t    vectors[]       = {3, 3};
    int64_t    rowStrides[]    = {1};
    int64_t    vectorStrides[] = {3, 1};
    const auto at              = [device](size_t offset)
    { return static_cast<CudaDriver::DevicePointer>(device + static_cast<CudaDriver::DevicePointer>(offset)); };
    CudaFamilyLaunch launch{};
    launch.queryCount = 3u;
    int64_t bitRows[]{rows[0], 2};
    launch.queryBits = cudaBitsTensor(at(offsetof(QueryData, bits)), bitRows);
    launch.positions = cudaTensor(at(offsetof(QueryData, origins)), {kDLFloat, 64u, 1u}, 2, vectors, vectorStrides);
    launch.directionsOrRotations =
        cudaTensor(at(offsetof(QueryData, directions)), {kDLFloat, 32u, 1u}, 2, vectors, vectorStrides);
    launch.maximumDistances =
        cudaTensor(at(offsetof(QueryData, maximumDistances)), {kDLFloat, 64u, 1u}, 1, rows, rowStrides);
    if (family != CudaFamilyKind::sphereSweep)
    {
        launch.rowMaximumHits =
            cudaTensor(at(offsetof(QueryData, maximumHits)), {kDLUInt, 64u, 1u}, 1, rows, rowStrides);
    }
    if (family == CudaFamilyKind::overlap)
    {
        launch.directionsOrRotations = CudaFamilyInput{};
        launch.maximumDistances      = CudaFamilyInput{};
        launch.positions =
            cudaTensor(at(offsetof(QueryData, aabbCenters)), {kDLFloat, 64u, 1u}, 2, vectors, vectorStrides);
    }
    launch.stream    = reinterpret_cast<uintptr_t>(producer);
    launch.waitEvent = reinterpret_cast<uintptr_t>(producerReady);
    std::shared_ptr<CudaFamilyOperation> operation;
    require(prepareAndLaunchCudaFamilyPlan(plan, 0u, launch, std::move(resolved.generation), operation) ==
                    OVSPATIALQUERY_API_SUCCESS &&
                operation != nullptr,
            "CUDA mixed-row launch failed");
    const uintptr_t ready = operation->reservation().resultReady.value;
    cuda.synchronize(reinterpret_cast<CudaDriver::Event>(ready));
    CudaFamilyResultColumns view{};
    require(operation->completedResult(view) == CudaFamilyCompletedResultOutcome::success,
            "CUDA mixed-row completion was unavailable");
    uint32_t statuses[3]  = {};
    uint64_t stored       = UINT64_MAX;
    uint64_t offsets[3]   = {};
    uint64_t counts[3]    = {};
    uint8_t  truncated[3] = {};
    uint64_t objectId     = UINT64_MAX;
    cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(ready));
    cuda.downloadAsync(statuses, deviceAddress(view.queryStatus), sizeof(statuses), consumer);
    cuda.downloadAsync(&stored, deviceAddress(view.storedHitCount), sizeof(stored), consumer);
    cuda.downloadAsync(offsets, deviceAddress(view.hitOffsets), sizeof(offsets), consumer);
    cuda.downloadAsync(counts, deviceAddress(view.hitCounts), sizeof(counts), consumer);
    cuda.downloadAsync(truncated, deviceAddress(view.truncated), sizeof(truncated), consumer);
    cuda.downloadAsync(&objectId, deviceAddress(view.objectIds), sizeof(objectId), consumer);
    cuda.record(consumerDone, consumer);
    cuda.synchronize(consumerDone);
    require(statuses[0] == cuda_family::kBatchRowSuccess && statuses[1] == cuda_family::kBatchRowInvalidArgument &&
                statuses[2] == cuda_family::kBatchRowSuccess && stored == 1u && offsets[0] == 0u && offsets[1] == 0u &&
                offsets[2] == 1u && counts[0] == 1u && counts[1] == 0u && counts[2] == 0u &&
                truncated[0] == (family == CudaFamilyKind::sphereSweep ? 0u : 1u) && truncated[1] == 0u &&
                truncated[2] == 1u && objectId == 17u,
            "CUDA mixed-row status/partition/order/capacity truth changed");

    operation->releaseReadyDependencies();
    retireConsumer(*operation, consumerDone, "CUDA mixed-row consumer retirement failed");
    operation.reset();
    plan.reset();
    std::shared_ptr<void> retired;
    require(populations.release(population, retired) == OVSPATIALQUERY_API_SUCCESS,
            "CUDA mixed-row test retained its population");
    ovspatialquery_test_oracle_destroy_runtime(runtime);
    cuda.free(device);
    cuda.destroyEvent(producerReady);
    cuda.destroyStream(consumer);
    cuda.destroyStream(producer);
}

void testCudaOwnerUsesSharedRuntimeAndDeviceSafeSlotReuse()
{
    CudaDriver                        cuda;
    const CudaDriver::Stream          producer           = cuda.createStream();
    const CudaDriver::Stream          consumer           = cuda.createStream();
    const CudaDriver::Event           producerReady      = cuda.createEvent();
    const CudaDriver::Event           firstConsumerDone  = cuda.createEvent();
    const CudaDriver::Event           secondConsumerDone = cuda.createEvent();
    OperationInstance                 instance(2u, 1u);
    ovspatialquery_backend_runtime_t* runtime = nullptr;
    require(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS,
            "CUDA owner oracle runtime creation failed");
    PopulationTable populations(runtime);
    PopulationKey   population;
    require(populations.create(2u, 1, candidateObjects(), population) == OVSPATIALQUERY_API_SUCCESS,
            "CUDA owner population creation failed");

    RaycastFamilyBatchPlanConfig config{};
    config.common.target.population = population;
    config.common.backend           = FamilyBatchPlanBackend::cuda;
    config.common.resultSlotCount   = 1u;
    config.common.cudaDevice        = {kDLCUDA, 0};
    config.common.cudaResultFields  = kCudaFamilyResultDistance;
    config.plan.hitMode             = HitMode::closest;
    config.plan.precision           = QueryPrecision::boundsCandidate;
    config.plan.maximumQueryCount   = 1u;
    config.plan.maximumHitsPerQuery = 1u;
    config.plan.hitCapacity         = 1u;
    std::unique_ptr<FamilyBatchPlanOperationOwner> owner;
    require(FamilyBatchPlanOperationOwner::create(populations, instance.get(), config, owner) ==
                    OVSPATIALQUERY_API_SUCCESS &&
                owner != nullptr,
            "CUDA owner creation failed");

    struct QueryData
    {
        double origin[3]{};
        float  direction[3]{1.0f, 0.0f, 0.0f};
        double maximumDistance = 10.0;
    } host;
    const CudaDriver::DevicePointer device = cuda.allocate(sizeof(host));
    cuda.uploadAsync(device, &host, sizeof(host), producer);
    cuda.record(producerReady, producer);
    int64_t    rows[]          = {1};
    int64_t    vectors[]       = {1, 3};
    int64_t    rowStrides[]    = {1};
    int64_t    vectorStrides[] = {3, 1};
    const auto at              = [device](size_t offset)
    { return static_cast<CudaDriver::DevicePointer>(device + static_cast<CudaDriver::DevicePointer>(offset)); };
    CudaFamilyLaunch launch{};
    launch.queryCount = 1u;
    launch.positions  = cudaTensor(at(offsetof(QueryData, origin)), {kDLFloat, 64u, 1u}, 2, vectors, vectorStrides);
    launch.directionsOrRotations =
        cudaTensor(at(offsetof(QueryData, direction)), {kDLFloat, 32u, 1u}, 2, vectors, vectorStrides);
    launch.maximumDistances =
        cudaTensor(at(offsetof(QueryData, maximumDistance)), {kDLFloat, 64u, 1u}, 1, rows, rowStrides);
    launch.stream    = reinterpret_cast<uintptr_t>(producer);
    launch.waitEvent = reinterpret_cast<uintptr_t>(producerReady);

    require(setInstanceOperationWorkersPausedForTest(instance.get(), true) == OVSPATIALQUERY_API_SUCCESS,
            "CUDA owner workers did not pause before pending-fetch proof");
    CudaFamilyResultColumns        firstReservation{};
    const uint64_t                 deviceToHostBeforeLaunch = cudaDeviceToHostBytes();
    const FamilyBatchPlanAdmission first = owner->enqueueCuda({TargetSelectionKind::head, 0}, launch, firstReservation);
    require(first.outcome == OVSPATIALQUERY_API_SUCCESS && first.operation.valid() && first.resultSlot == 0u &&
                firstReservation.resultReady.value != 0u,
            "CUDA owner launch was not admitted");
    require(cudaDeviceToHostBytes() == deviceToHostBeforeLaunch, "CUDA launch with rows queued a metadata read");
    CudaFamilyResultColumns rejectedReservation{};
    rejectedReservation.resultSlot = 77u;
    const FamilyBatchPlanAdmission rejected =
        owner->enqueueCuda({TargetSelectionKind::head, 0}, launch, rejectedReservation);
    require(rejected.outcome == OVSPATIALQUERY_API_RESULT_CAPACITY && !rejected.operation.valid() &&
                rejectedReservation.resultSlot == 77u,
            "CUDA result-slot rejection was not atomic");

    CudaFamilyResultColumns early{};
    early.sourceOrdinal = 999;
    require(owner->fetchCuda(first.operation, early) == OVSPATIALQUERY_API_NOT_READY && early.sourceOrdinal == 999,
            "pending CUDA fetch mutated its output");
    FamilyBatchPlanOperationId firstOperation = first.operation;
    require(owner->destroyOperation(firstOperation) == OVSPATIALQUERY_API_BUSY && firstOperation.valid(),
            "unobserved CUDA operation was destroyed normally");
    require(setInstanceOperationWorkersPausedForTest(instance.get(), false) == OVSPATIALQUERY_API_SUCCESS,
            "CUDA owner workers did not resume after pending-fetch proof");
    InstanceOperationState state = InstanceOperationState::pending;
    require(waitInstanceOperation(instance.get(), first.operation.value, OVSPATIALQUERY_TIMEOUT_INFINITE, state) ==
                    OVSPATIALQUERY_API_SUCCESS &&
                state == InstanceOperationState::completed,
            "CUDA owner operation did not become terminal");
    CudaFamilyResultColumns firstResult{};
    CudaFamilyResultColumns repeated{};
    require(owner->fetchCuda(first.operation, firstResult) == OVSPATIALQUERY_API_SUCCESS, "CUDA terminal fetch failed");
    require(cudaDeviceToHostBytes() == deviceToHostBeforeLaunch, "CUDA terminal fetch read device memory");
    require(owner->fetchCuda(first.operation, repeated) == OVSPATIALQUERY_API_SUCCESS &&
                cudaDeviceToHostBytes() == deviceToHostBeforeLaunch &&
                repeated.resultReady.value == firstResult.resultReady.value &&
                repeated.objectIds.data == firstResult.objectIds.data,
            "CUDA repeat fetch changed descriptors or repeated metadata read");
    require(owner->destroyOperation(firstOperation) == OVSPATIALQUERY_API_BUSY && firstOperation.valid(),
            "completed CUDA operation bypassed device-safe destruction");
    cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(firstResult.resultReady.value));
    uint64_t firstObject = UINT64_MAX;
    cuda.downloadAsync(&firstObject, deviceAddress(firstResult.objectIds), sizeof(firstObject), consumer);
    cuda.record(firstConsumerDone, consumer);
    cuda.synchronize(firstConsumerDone);
    ovspatialquery_cuda_event_t firstDone{reinterpret_cast<uintptr_t>(firstConsumerDone)};
    const uint64_t              attemptsBeforeContextSelectionFailure = transferredEventDestroyAttempts();
    const uint64_t              preDestroyFailuresBefore              = transferredEventPreDestroyFailures();
    ovspatialquery_test_set_cuda_failure_point(kCudaFailureConsumerDestroyContextSelection);
    require(owner->destroyOperationAfterCuda(firstOperation, firstDone) == OVSPATIALQUERY_API_SUCCESS &&
                !firstOperation.valid() && firstDone.value == 0u && firstObject == 17u,
            "CUDA device-safe operation destruction failed");
    const std::chrono::steady_clock::time_point failureDeadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (transferredEventPreDestroyFailures() == preDestroyFailuresBefore &&
           transferredEventDestroyAttempts() == attemptsBeforeContextSelectionFailure &&
           std::chrono::steady_clock::now() < failureDeadline)
    {
        std::this_thread::yield();
    }
    require(transferredEventPreDestroyFailures() > preDestroyFailuresBefore &&
                transferredEventDestroyAttempts() == attemptsBeforeContextSelectionFailure,
            "CUDA owner forgot consumer_done after context selection failed");

    CudaFamilyResultColumns        blockedReservation{};
    const FamilyBatchPlanAdmission blocked =
        owner->enqueueCuda({TargetSelectionKind::head, 0}, launch, blockedReservation);
    require(blocked.outcome == OVSPATIALQUERY_API_RESULT_CAPACITY && !blocked.operation.valid(),
            "CUDA context-selection failure released a device-unsafe result slot");
    ovspatialquery_test_set_cuda_failure_point(0u);
    require(transferredEventDestroysWereInvoked(),
            "CUDA owner did not retry consumer_done after context selection recovered");

    FamilyBatchPlanAdmission                    reused{};
    CudaFamilyResultColumns                     reuseReservation{};
    const std::chrono::steady_clock::time_point reuseDeadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    do
    {
        reused = owner->enqueueCuda({TargetSelectionKind::head, 0}, launch, reuseReservation);
        if (reused.outcome == OVSPATIALQUERY_API_SUCCESS)
        {
            break;
        }
        require(reused.outcome == OVSPATIALQUERY_API_RESULT_CAPACITY, "CUDA slot reuse returned an unexpected outcome");
        std::this_thread::yield();
    } while (std::chrono::steady_clock::now() < reuseDeadline);
    require(reused.outcome == OVSPATIALQUERY_API_SUCCESS && reused.operation.valid() && reused.resultSlot == 0u,
            "CUDA result slot was not reusable after consumer_done");
    require(waitInstanceOperation(instance.get(), reused.operation.value, OVSPATIALQUERY_TIMEOUT_INFINITE, state) ==
                    OVSPATIALQUERY_API_SUCCESS &&
                state == InstanceOperationState::completed,
            "reused CUDA operation did not complete");
    CudaFamilyResultColumns reusedResult{};
    require(owner->fetchCuda(reused.operation, reusedResult) == OVSPATIALQUERY_API_SUCCESS,
            "reused CUDA result did not fetch");
    cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(reusedResult.resultReady.value));
    cuda.record(secondConsumerDone, consumer);
    cuda.synchronize(secondConsumerDone);
    FamilyBatchPlanOperationId  reusedOperation = reused.operation;
    ovspatialquery_cuda_event_t secondDone{reinterpret_cast<uintptr_t>(secondConsumerDone)};
    require(owner->destroyOperationAfterCuda(reusedOperation, secondDone) == OVSPATIALQUERY_API_SUCCESS &&
                !reusedOperation.valid() && secondDone.value == 0u,
            "reused CUDA operation did not retire device-safely");
    require(transferredEventDestroysWereInvoked(), "CUDA reused owner did not issue one consumer_done destroy");

    std::unique_ptr<FamilyBatchPlanOperationOwner> outerDeviceLostOwner;
    require(FamilyBatchPlanOperationOwner::create(populations, instance.get(), config, outerDeviceLostOwner) ==
                    OVSPATIALQUERY_API_SUCCESS &&
                outerDeviceLostOwner != nullptr,
            "CUDA outer-context DEVICE_LOST owner creation failed");
    CudaFamilyResultColumns        outerDeviceLostReservation{};
    const FamilyBatchPlanAdmission outerDeviceLost =
        outerDeviceLostOwner->enqueueCuda({TargetSelectionKind::head, 0}, launch, outerDeviceLostReservation);
    require(outerDeviceLost.outcome == OVSPATIALQUERY_API_SUCCESS && outerDeviceLost.operation.valid(),
            "CUDA outer-context DEVICE_LOST operation was not admitted");
    require(waitInstanceOperation(instance.get(), outerDeviceLost.operation.value, OVSPATIALQUERY_TIMEOUT_INFINITE,
                                  state) == OVSPATIALQUERY_API_SUCCESS &&
                state == InstanceOperationState::completed,
            "CUDA outer-context DEVICE_LOST operation did not complete");
    CudaFamilyResultColumns outerDeviceLostResult{};
    require(outerDeviceLostOwner->fetchCuda(outerDeviceLost.operation, outerDeviceLostResult) ==
                OVSPATIALQUERY_API_SUCCESS,
            "CUDA outer-context DEVICE_LOST result did not fetch");
    cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(outerDeviceLostResult.resultReady.value));
    const CudaDriver::Event outerDeviceLostDone = cuda.createEvent();
    cuda.record(outerDeviceLostDone, consumer);
    cuda.synchronize(outerDeviceLostDone);
    const uint64_t attemptsBeforeOuterDeviceLost = transferredEventDestroyAttempts();
    ovspatialquery_test_set_cuda_failure_point(kCudaFailureConsumerDestroyOuterContextDeviceLost);
    FamilyBatchPlanOperationId  outerDeviceLostOperation = outerDeviceLost.operation;
    ovspatialquery_cuda_event_t outerDone{reinterpret_cast<uintptr_t>(outerDeviceLostDone)};
    require(outerDeviceLostOwner->destroyOperationAfterCuda(outerDeviceLostOperation, outerDone) ==
                    OVSPATIALQUERY_API_SUCCESS &&
                !outerDeviceLostOperation.valid() && outerDone.value == 0u,
            "CUDA outer-context DEVICE_LOST operation did not transfer ownership");
    closeOwner(*outerDeviceLostOwner, "CUDA outer-context DEVICE_LOST retirement remained busy");
    require(ovspatialquery_test_get_cuda_failure_point() == 0u &&
                transferredEventDestroyAttempts() == attemptsBeforeOuterDeviceLost,
            "CUDA outer-context DEVICE_LOST retirement destroyed the event");
    outerDeviceLostOwner.reset();

    closeOwner(*owner, "CUDA owner did not close after device-safe retirement");
    owner.reset();
    std::shared_ptr<void> retired;
    require(populations.release(population, retired) == OVSPATIALQUERY_API_SUCCESS,
            "CUDA owner retained its target after close");
    ovspatialquery_test_oracle_destroy_runtime(runtime);
    cuda.free(device);
    cuda.destroyEvent(producerReady);
    cuda.destroyStream(consumer);
    cuda.destroyStream(producer);
}

void testCudaOwnerStagingBudgetAdmissionIsAtomic()
{
    CudaDriver               cuda;
    const CudaDriver::Stream producer              = cuda.createStream();
    const CudaDriver::Event  producerReady         = cuda.createEvent();
    uint64_t                 oneSlotFixedHostBytes = 0u;
    uint64_t                 twoSlotFixedHostBytes = 0u;
    require(estimateCudaFamilyPlanFixedHostByteCharge(1u, oneSlotFixedHostBytes) &&
                estimateCudaFamilyPlanFixedHostByteCharge(2u, twoSlotFixedHostBytes) && oneSlotFixedHostBytes != 0u &&
                twoSlotFixedHostBytes > oneSlotFixedHostBytes,
            "CUDA fixed host estimator omitted backend plan/slot storage");
    constexpr uint64_t largeCatalogueObjectCount = 10000u;
    // The private owner selects the CudaFamily catalogue path.  A bounds-only
    // catalogue keeps ids, participation bits, and AABB extrema both for the
    // host staging vectors and their device-upload payloads, plus the fixed
    // allocation/staging pointer tables.
    constexpr uint64_t allFamilyHostCataloguePayloadBytes =
        largeCatalogueObjectCount * 2u * (sizeof(uint64_t) * 3u + sizeof(uint32_t) + sizeof(double) * 6u) +
        17u * (sizeof(CudaDriver::DevicePointer) + sizeof(void*));

    struct QueryData
    {
        double origin[3]{};
        float  direction[3]{1.0f, 0.0f, 0.0f};
        double maximumDistance = 10.0;
    } host;
    const CudaDriver::DevicePointer device = cuda.allocate(sizeof(host));
    cuda.uploadAsync(device, &host, sizeof(host), producer);
    cuda.record(producerReady, producer);
    int64_t    rows[]          = {1};
    int64_t    vectors[]       = {1, 3};
    int64_t    rowStrides[]    = {1};
    int64_t    vectorStrides[] = {3, 1};
    const auto at              = [device](size_t offset)
    { return static_cast<CudaDriver::DevicePointer>(device + static_cast<CudaDriver::DevicePointer>(offset)); };
    CudaFamilyLaunch launch{};
    launch.queryCount = 1u;
    launch.positions  = cudaTensor(at(offsetof(QueryData, origin)), {kDLFloat, 64u, 1u}, 2, vectors, vectorStrides);
    launch.directionsOrRotations =
        cudaTensor(at(offsetof(QueryData, direction)), {kDLFloat, 32u, 1u}, 2, vectors, vectorStrides);
    launch.maximumDistances =
        cudaTensor(at(offsetof(QueryData, maximumDistance)), {kDLFloat, 64u, 1u}, 1, rows, rowStrides);
    launch.stream    = reinterpret_cast<uintptr_t>(producer);
    launch.waitEvent = reinterpret_cast<uintptr_t>(producerReady);

    const auto exercise = [&launch, &cuda, producer](
                              uint64_t hostBudget, uint64_t deviceBudget, ovspatialquery_api_status_t expected,
                              const char* message, bool forceLengthError = false, uint32_t objectCount = 2u,
                              bool proveNoMaterialization = false, bool copyInputs = false,
                              uint64_t expectedDeviceToDeviceBytes = UINT64_MAX, uint64_t* observedHostCharge = nullptr)
    {
        CudaDriver::Event                 admittedConsumerDone = nullptr;
        OperationInstance                 instance(2u, 2u, hostBudget, deviceBudget);
        InstanceHostLedger                ledger(hostBudget);
        ovspatialquery_backend_runtime_t* runtime = nullptr;
        require(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS,
                "CUDA staging-budget oracle runtime creation failed");
        PopulationTable populations(runtime);
        PopulationKey   population;
        require(populations.create(2u, 1, objectCount == 2u ? candidateObjects() : candidateObjects(objectCount),
                                   population) == OVSPATIALQUERY_API_SUCCESS,
                "CUDA staging-budget population creation failed");
        RaycastFamilyBatchPlanConfig config{};
        config.common.target.population = population;
        config.common.backend           = FamilyBatchPlanBackend::cuda;
        config.common.resultSlotCount   = 1u;
        config.common.cudaDevice        = {kDLCUDA, 0};
        config.common.cudaResultFields  = kCudaFamilyResultDistance;
        config.common.hostLedger        = &ledger;
        config.plan.hitMode             = HitMode::closest;
        config.plan.precision           = QueryPrecision::boundsCandidate;
        config.plan.maximumQueryCount   = 1u;
        config.plan.maximumHitsPerQuery = 1u;
        config.plan.hitCapacity         = 1u;
        std::unique_ptr<FamilyBatchPlanOperationOwner> owner;
        require(FamilyBatchPlanOperationOwner::create(populations, instance.get(), config, owner) ==
                    OVSPATIALQUERY_API_SUCCESS,
                "CUDA staging-budget plan creation failed before admission");
        CudaFamilyResultColumns reservation{};
        reservation.resultSlot = 71u;
        if (forceLengthError)
        {
            ovspatialquery_test_set_cuda_failure_point(kCudaFailureFamilyPrepare);
        }
        CudaFamilyLaunch submitted = launch;
        if (copyInputs)
        {
            submitted.positions.ownership             = 1u;
            submitted.directionsOrRotations.ownership = 1u;
            submitted.maximumDistances.ownership      = 1u;
        }
        const auto deviceToDeviceBytes = []()
        {
            uint64_t deviceToHost   = 0u;
            uint64_t deviceToDevice = 0u;
            ovspatialquery_test_get_cuda_dispatch_evidence(&deviceToHost, &deviceToDevice, nullptr, nullptr);
            return deviceToDevice;
        };
        const uint64_t           deviceToDeviceBefore = deviceToDeviceBytes();
        FamilyBatchPlanAdmission admission = owner->enqueueCuda({TargetSelectionKind::head, 0}, submitted, reservation);
        if (expected == OVSPATIALQUERY_API_SUCCESS)
        {
            require(admission.operation.valid() && admission.resultSlot == 0u && reservation.resultSlot == 0u, message);
            if (observedHostCharge != nullptr)
            {
                *observedHostCharge = ledger.total();
            }
            InstanceOperationState state = InstanceOperationState::pending;
            require(waitInstanceOperation(instance.get(), admission.operation.value, OVSPATIALQUERY_TIMEOUT_INFINITE,
                                          state) == OVSPATIALQUERY_API_SUCCESS &&
                        state == InstanceOperationState::completed,
                    "exact host staging-budget admission did not complete");
            CudaFamilyResultColumns completed{};
            require(owner->fetchCuda(admission.operation, completed) == OVSPATIALQUERY_API_SUCCESS,
                    "exact host staging-budget admission did not fetch");
            admittedConsumerDone = cuda.createEvent();
            cuda.record(admittedConsumerDone, producer);
            cuda.synchronize(admittedConsumerDone);
            ovspatialquery_cuda_event_t consumerDoneHandle{
                reinterpret_cast<uintptr_t>(admittedConsumerDone),
            };
            require(owner->destroyOperationAfterCuda(admission.operation, consumerDoneHandle) ==
                        OVSPATIALQUERY_API_SUCCESS,
                    "exact host staging-budget admission did not destroy");
        }
        else
        {
            require(admission.outcome == expected && !admission.operation.valid() &&
                        admission.resultSlot == UINT32_MAX && reservation.resultSlot == 71u,
                    message);
        }
        if (expectedDeviceToDeviceBytes != UINT64_MAX)
        {
            require(deviceToDeviceBytes() - deviceToDeviceBefore == expectedDeviceToDeviceBytes,
                    "CUDA copied-input device-to-device evidence was not exact");
        }
        if (proveNoMaterialization)
        {
            require(ledger.total() == 0u, "hard host-budget rejection charged canonical CUDA staging");
        }
        closeOwner(*owner, "CUDA staging-budget rejection leaked plan lifecycle state");
        owner.reset();
        if (admittedConsumerDone != nullptr)
        {
            require(transferredEventDestroysWereInvoked(), "CUDA budget owner did not issue one consumer_done destroy");
        }
        std::shared_ptr<void> retired;
        require(populations.release(population, retired) == OVSPATIALQUERY_API_SUCCESS,
                "CUDA staging-budget plan retained its population");
        ovspatialquery_test_oracle_destroy_runtime(runtime);
    };
    // The white-box owner charges only launch staging against this budget; one byte short of the
    // two bounds-candidate objects' 168 staging bytes must reject before anything is charged.
    exercise(std::numeric_limits<uint64_t>::max(), 167u, OVSPATIALQUERY_API_DEVICE_BUDGET,
             "CUDA device staging-budget rejection was not atomic");
    // One bounds-candidate object requires 84 device staging bytes (ID 8, type 4, geometry
    // value 8, participation bits 16, bounds 48).  These
    // three copied query columns add exactly 44 bytes (24 + 12 + 8), while
    // borrowed descriptors add none.  The resident 49-byte result slot above
    // is a separate fixed plan charge and is not double-counted per launch.
    exercise(std::numeric_limits<uint64_t>::max(), 127u, OVSPATIALQUERY_API_DEVICE_BUDGET,
             "CUDA copied-input required-minus-one budget was not atomic", false, 1u, false, true, 0u);
    exercise(std::numeric_limits<uint64_t>::max(), 128u, OVSPATIALQUERY_API_SUCCESS,
             "CUDA copied-input exact staging budget was not admitted", false, 1u, false, true, 44u);
    exercise(std::numeric_limits<uint64_t>::max(), 84u, OVSPATIALQUERY_API_SUCCESS,
             "CUDA borrowed inputs were charged as copied staging", false, 1u, false, false, 0u);
    exercise(std::numeric_limits<uint64_t>::max(), std::numeric_limits<uint64_t>::max(),
             OVSPATIALQUERY_API_OUT_OF_MEMORY,
             "CUDA catalogue length-error rejection was not resource-exhaustion atomic", true);
    uint64_t allFamilyHostCharge = 0u;
    exercise(std::numeric_limits<uint64_t>::max(), std::numeric_limits<uint64_t>::max(), OVSPATIALQUERY_API_SUCCESS,
             "CUDA host staging-charge calibration did not succeed", false,
             static_cast<uint32_t>(largeCatalogueObjectCount), false, false, UINT64_MAX, &allFamilyHostCharge);
    require(allFamilyHostCharge > allFamilyHostCataloguePayloadBytes,
            "CUDA host staging charge omitted operation/control storage");
    exercise(allFamilyHostCharge, std::numeric_limits<uint64_t>::max(), OVSPATIALQUERY_API_SUCCESS,
             "CUDA exact host staging-budget admission did not succeed", false,
             static_cast<uint32_t>(largeCatalogueObjectCount));
    exercise(allFamilyHostCharge - 1u, std::numeric_limits<uint64_t>::max(), OVSPATIALQUERY_API_HOST_BUDGET,
             "CUDA one-byte-short host staging budget was not atomic", false,
             static_cast<uint32_t>(largeCatalogueObjectCount), true);
    cuda.free(device);
    cuda.destroyEvent(producerReady);
    cuda.destroyStream(producer);
}

void testCudaZeroRowReadinessUsesTheRecordedEvent()
{
    CudaDriver                        cuda;
    const CudaDriver::Stream          producer                   = cuda.createStream();
    const CudaDriver::Stream          execution                  = cuda.createStream();
    const CudaDriver::Event           producerReady              = cuda.createEvent();
    const CudaDriver::Event           zeroRowConsumerDone        = cuda.createEvent();
    const CudaDriver::Event           partialFailureConsumerDone = cuda.createEvent();
    ovspatialquery_backend_runtime_t* runtime                    = nullptr;
    require(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS,
            "CUDA zero-row oracle runtime creation failed");
    PopulationTable populations(runtime);
    PopulationKey   population;
    require(populations.create(2u, 1, {}, population) == OVSPATIALQUERY_API_SUCCESS,
            "CUDA zero-row population creation failed");
    TargetResolver resolver(populations);
    ResolvedTarget resolved;
    require(resolver.resolve({{population, nullptr}, {TargetSelectionKind::head, 0}}, resolved) ==
                TargetResolveOutcome::success,
            "CUDA zero-row target resolution failed");
    CudaFamilyPlanConfig config{};
    config.family              = CudaFamilyKind::raycast;
    config.precision           = CudaFamilyPrecision::boundsCandidate;
    config.hitMode             = CudaFamilyHitMode::closest;
    config.device              = {kDLCUDA, 0};
    config.resultFields        = kCudaFamilyResultDistance;
    config.maximumQueryCount   = 1u;
    config.maximumHitsPerQuery = 1u;
    config.hitCapacityPerSlot  = 1u;
    config.resultSlotCount     = 1u;
    std::shared_ptr<CudaFamilyPlan> plan;
    require(createCudaFamilyPlan(config, plan) == OVSPATIALQUERY_API_SUCCESS, "CUDA zero-row plan creation failed");
    cuda.enqueueDelay(producer, UINT64_C(100000000));
    cuda.record(producerReady, producer);
    int64_t          rows[]          = {0};
    int64_t          vectors[]       = {0, 3};
    int64_t          rowStrides[]    = {1};
    int64_t          vectorStrides[] = {3, 1};
    CudaFamilyLaunch launch{};
    launch.positions             = cudaTensor(0u, {kDLFloat, 64u, 1u}, 2, vectors, vectorStrides);
    launch.directionsOrRotations = cudaTensor(0u, {kDLFloat, 32u, 1u}, 2, vectors, vectorStrides);
    launch.maximumDistances      = cudaTensor(0u, {kDLFloat, 64u, 1u}, 1, rows, rowStrides);
    launch.stream                = reinterpret_cast<uintptr_t>(execution);
    launch.waitEvent             = reinterpret_cast<uintptr_t>(producerReady);
    std::shared_ptr<CudaFamilyOperation> operation;
    require(prepareAndLaunchCudaFamilyPlan(plan, 0u, launch, std::move(resolved.generation), operation) ==
                    OVSPATIALQUERY_API_SUCCESS &&
                operation != nullptr,
            "CUDA zero-row launch failed");
    require(operation->resultState() == CudaCompletionState::pending,
            "CUDA zero-row operation completed before its result-ready event");
    const uintptr_t ready = operation->reservation().resultReady.value;
    cuda.synchronize(reinterpret_cast<CudaDriver::Event>(ready));
    require(operation->resultState() == CudaCompletionState::complete,
            "CUDA zero-row operation did not complete after result-ready");
    operation->releaseReadyDependencies();
    cuda.wait(execution, reinterpret_cast<CudaDriver::Event>(ready));
    cuda.record(zeroRowConsumerDone, execution);
    cuda.synchronize(zeroRowConsumerDone);
    retireConsumer(*operation, zeroRowConsumerDone, "CUDA zero-row consumer retirement failed");
    operation.reset();
    ResolvedTarget partialFailureTarget;
    require(resolver.resolve({{population, nullptr}, {TargetSelectionKind::head, 0}}, partialFailureTarget) ==
                TargetResolveOutcome::success,
            "CUDA partial-enqueue target resolution failed");
    cuda.enqueueDelay(producer, UINT64_C(100000000));
    cuda.record(producerReady, producer);
    ovspatialquery_test_set_cuda_failure_point(kCudaFailureFamilyLaunchBeforeKernels);
    std::shared_ptr<CudaFamilyOperation> partialFailure;
    require(prepareAndLaunchCudaFamilyPlan(plan, 0u, launch, std::move(partialFailureTarget.generation),
                                           partialFailure) == OVSPATIALQUERY_API_SUCCESS &&
                partialFailure != nullptr,
            "CUDA partial enqueue failure did not retain a terminal operation");
    require(partialFailure->resultState() == CudaCompletionState::pending,
            "CUDA later enqueue failure erased earlier queued-work state");
    cuda.synchronize(producerReady);
    CudaCompletionState partialState = CudaCompletionState::pending;
    for (uint32_t attempt = 0u; attempt != 1000u; ++attempt)
    {
        partialState = partialFailure->resultState();
        if (partialState != CudaCompletionState::pending)
        {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    require(partialState == CudaCompletionState::error,
            "CUDA partial enqueue failure was not terminally observed by stream query");
    const uintptr_t partialReady = partialFailure->reservation().resultReady.value;
    cuda.wait(execution, reinterpret_cast<CudaDriver::Event>(partialReady));
    cuda.record(partialFailureConsumerDone, execution);
    cuda.synchronize(partialFailureConsumerDone);
    require(partialFailure->transferConsumerDone({reinterpret_cast<uintptr_t>(partialFailureConsumerDone)}) ==
                OVSPATIALQUERY_API_SUCCESS,
            "CUDA partial enqueue failure did not retain aggregate consumer lifetime");
    ovspatialquery_test_set_cuda_failure_point(kCudaFailureConsumerPollDeviceLost);
    const CudaCompletionState partialConsumer = partialFailure->consumerState();
    require(partialConsumer == CudaCompletionState::deviceLost && ovspatialquery_test_get_cuda_failure_point() == 0u,
            "CUDA consumer-poll device loss was not latched before context restore");
    const uint64_t attemptsBeforeDeviceLostDisposition = transferredEventDestroyAttempts();
    partialFailure->releaseReadyDependencies();
    require(partialFailure->consumerDoneOwnershipReleased() &&
                transferredEventDestroyAttempts() == attemptsBeforeDeviceLostDisposition,
            "CUDA DEVICE_LOST disposition attempted normal consumer retirement");
    partialFailure.reset();
    plan.reset();
    std::shared_ptr<void> retired;
    require(populations.release(population, retired) == OVSPATIALQUERY_API_SUCCESS,
            "CUDA zero-row plan retained its population");
    ovspatialquery_test_oracle_destroy_runtime(runtime);
    cuda.destroyEvent(producerReady);
    cuda.destroyStream(execution);
    cuda.destroyStream(producer);
}

} // namespace

int main()
{
    testPrivateCudaRaycastLaunchPublishesSemanticColumns();
    testCandidateMixedRowsAndForcedCapacity(CudaFamilyKind::raycast);
    testCandidateMixedRowsAndForcedCapacity(CudaFamilyKind::sphereSweep);
    testCandidateMixedRowsAndForcedCapacity(CudaFamilyKind::overlap);
    testExactFiveStoredGeometryFamiliesAcrossQueries();
    testBoxSweepsAcrossFiveFamilies();
    testBoxSweepKeepsSmallFeaturesAtLargeTranslation();
    testBoxSweepTraversalMatchesExhaustiveScan();
    testLongSweepsStopAtTheirFirstContact();
    testBorrowedSphereSweepRowRadiiAndFallback();
    testSphereSweepTraversalMatchesExhaustiveScan();
    testMeshOverlapsMatchExhaustiveScan();
    testTinySphereOverlapKeepsItsCandidate();
    testStoredSpherePathsMeetIndependentOracle();
    testSolidsReportInitialOverlapAtTheMinimum();
    testPlacedConvexKeepsFeaturesAtLargeTranslations();
    testPlacedMeshBvhKeepsBoundaryHitsAndReusesCatalogue(CudaFamilyPrecision::exact);
    testPlacedMeshBvhKeepsBoundaryHitsAndReusesCatalogue(CudaFamilyPrecision::exactSingle);
    testCudaOwnerUsesSharedRuntimeAndDeviceSafeSlotReuse();
    testCudaOwnerStagingBudgetAdmissionIsAtomic();
    testCudaZeroRowReadinessUsesTheRecordedEvent();
    return 0;
}
