/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../support/native/test_failure.h"
#include "oracle_backend.h"
#include "ovspatialquery_family_batch_plan_operation.h"
#include "ovspatialquery_instance_source.h"
#include "ovspatialquery_instance_state.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

namespace
{

#if defined(OVSPATIALQUERY_ENABLE_FAMILY_BATCH_PLAN_OPERATION_TEST_HOOK)
ovspatialquery::detail::InstanceHostLedger* gFamilyTaskDestroyLedger         = nullptr;
uint64_t                                    gFamilyTaskDestroyObservedCharge = UINT64_MAX;

void observeFamilyTaskPayloadDestroy() { gFamilyTaskDestroyObservedCharge = gFamilyTaskDestroyLedger->total(); }
#endif

using namespace ovspatialquery::detail;

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

// The first element of a fetched CPU column, which lives in host memory.
template <typename T>
const T* columnData(const DLTensor& column)
{
    return reinterpret_cast<const T*>(static_cast<const char*>(column.data) + column.byte_offset);
}

// Compares one fetched column against the exact rows a test expects.
template <typename T>
bool sameColumn(const DLTensor& column, const std::vector<T>& expected)
{
    if (column.shape == nullptr || column.shape[0] != static_cast<int64_t>(expected.size()))
    {
        return false;
    }
    for (size_t index = 0u; index < expected.size(); ++index)
    {
        if (!(columnData<T>(column)[index] == expected[index]))
        {
            return false;
        }
    }
    return true;
}

uint64_t storedHitCount(const ovspatialquery_batch_hit_ranges_t& ranges)
{
    return columnData<uint64_t>(ranges.stored_hit_count)[0];
}

class OperationInstance
{
  public:
    explicit OperationInstance(uint32_t operationCapacity = 8u, uint32_t maxCpuConcurrency = 1u)
    {
        const SourceInstanceConfig config{
            operationCapacity, 0u, maxCpuConcurrency, std::numeric_limits<uint64_t>::max(), nullptr, 0u,
        };
        require(createInstance(config, &mInstance) == OVSPATIALQUERY_API_SUCCESS, "operation instance creation failed");
    }

    ~OperationInstance()
    {
        require(ovspatialquery_destroy_instance(mInstance).status == OVSPATIALQUERY_API_SUCCESS,
                "operation instance release failed");
    }

    ovspatialquery_instance_handle_t handle() const noexcept { return mInstance; }

  private:
    ovspatialquery_instance_handle_t mInstance = OVSPATIALQUERY_INVALID_INSTANCE;
};

void setWorkersPaused(const OperationInstance& instance, bool paused)
{
    require(setInstanceOperationWorkersPausedForTest(instance.handle(), paused) == OVSPATIALQUERY_API_SUCCESS,
            "instance worker pause control failed");
}

void failNextScheduling(const OperationInstance& instance)
{
    require(failNextInstanceOperationSchedulingForTest(instance.handle()) == OVSPATIALQUERY_API_SUCCESS,
            "instance scheduling fault control failed");
}

void setWaiterPaused(const OperationInstance& instance, bool paused)
{
    require(setInstanceOperationWaiterPausedBeforeRevalidationForTest(instance.handle(), paused) ==
                OVSPATIALQUERY_API_SUCCESS,
            "instance waiter pause control failed");
}

void waitForWaiter(const OperationInstance& instance)
{
    require(waitForInstanceOperationWaiterAtRevalidationForTest(instance.handle()) == OVSPATIALQUERY_API_SUCCESS,
            "instance waiter did not reach revalidation");
}

// The terminal cause query_op_status reports for an operation.
ovspatialquery_api_status_t terminalStatus(const OperationInstance& instance, FamilyBatchPlanOperationId operation)
{
    ovspatialquery_op_status_t status{};
    return ovspatialquery_query_op_status(instance.handle(), operation.value, &status).status ==
                   OVSPATIALQUERY_API_SUCCESS
               ? status.terminal_status
               : OVSPATIALQUERY_API_INTERNAL;
}

uint64_t waiterEntryCount(const OperationInstance& instance)
{
    return instanceOperationWaiterEntryCountForTest(instance.handle());
}

PopulationObject object(uint64_t id, double minimumX, double maximumX)
{
    PopulationObject value{};
    value.id     = {id};
    value.bounds = {
        {minimumX, -1.0, -1.0},
        {maximumX, 1.0, 1.0},
    };
    value.geometryReference = {
        ExactGeometryFamily::sphere,
        id + 100u,
    };
    value.participationBits = {{UINT64_MAX, UINT64_MAX}};
    return value;
}

PopulationObject exactObject(uint64_t id, uint64_t geometryValue, std::shared_ptr<const ExactGeometry> geometry)
{
    PopulationObject value{};
    value.id                = {id};
    value.bounds            = geometry->worldBounds();
    value.geometryReference = {geometry->family(), geometryValue};
    value.exactGeometry     = std::move(geometry);
    value.participationBits = {{UINT64_MAX, UINT64_MAX}};
    return value;
}

std::shared_ptr<const ExactGeometry> meshGeometry(uint32_t sourceFace = UINT32_MAX)
{
    TriangleMeshSource mesh{};
    mesh.key      = 9001u;
    mesh.vertices = {
        14.0, -1.0, -1.0, 14.0, -1.0, 1.0, 14.0, 1.0, 1.0, 14.0, 1.0, -1.0,
    };
    mesh.triangles     = {0u, 1u, 2u, 0u, 2u, 3u};
    mesh.sourceFaces   = {sourceFace, sourceFace};
    mesh.sourceCorners = {0u, 1u, 2u, 0u, 2u, 3u};
    std::shared_ptr<const PreparedTriangleMesh>   preparedMesh;
    std::shared_ptr<const TriangleObjectGeometry> prepared;
    require(prepareTriangleMesh(mesh, preparedMesh) == OVSPATIALQUERY_API_SUCCESS &&
                bindPreparedTriangleGeometry(
                    preparedMesh, {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0},
                    true, prepared) == OVSPATIALQUERY_API_SUCCESS,
            "triangle geometry preparation failed");
    std::shared_ptr<const ExactGeometry> geometry;
    require(createTriangleMeshExactGeometry(prepared, geometry) == OVSPATIALQUERY_API_SUCCESS,
            "triangle exact geometry creation failed");
    return geometry;
}

void publish(PopulationTable& populations, PopulationKey population, ovspatialquery_ordinal_t ordinal,
             std::vector<PopulationObject> objects, std::vector<GenerationViewMembership> memberships = {})
{
    GenerationLease base;
    require(populations.beginUpdate(population, ordinal, nullptr, base) == OVSPATIALQUERY_API_SUCCESS,
            "population update did not begin");
    GenerationLease generation;
    require(populations.buildGeneration(ordinal, std::move(objects), std::move(memberships), generation) ==
                OVSPATIALQUERY_API_SUCCESS,
            "generation build failed");
    require(populations.publishUpdate(population, generation) == OVSPATIALQUERY_API_SUCCESS,
            "population publication failed");
    generation.reset();
    base.reset();
    populations.finishUpdate(population);
}

std::shared_ptr<const std::vector<uint64_t>> selected(std::initializer_list<uint64_t> values)
{
    return std::make_shared<const std::vector<uint64_t>>(values);
}

RaycastBatchRow ray(double originX)
{
    RaycastBatchRow row{};
    row.query.origin          = {originX, 0.0, 0.0};
    row.query.unitDirection   = {1.0f, 0.0f, 0.0f};
    row.query.maximumDistance = 100.0;
    return row;
}

SphereSweepBatchRow sweep(double originX)
{
    SphereSweepBatchRow row{};
    row.query.center          = {originX, 0.0, 0.0};
    row.query.radius          = 0.25;
    row.query.unitDirection   = {1.0f, 0.0f, 0.0f};
    row.query.maximumDistance = 100.0;
    return row;
}

OverlapBatchRow overlap(double minimumX, double maximumX)
{
    OverlapBatchRow row{};
    row.query.geometry.kind                    = OVSPATIALQUERY_OVERLAP_AABB;
    row.query.geometry.data.aabb.min_corner[0] = minimumX;
    row.query.geometry.data.aabb.min_corner[1] = -2.0;
    row.query.geometry.data.aabb.min_corner[2] = -2.0;
    row.query.geometry.data.aabb.max_corner[0] = maximumX;
    row.query.geometry.data.aabb.max_corner[1] = 2.0;
    row.query.geometry.data.aabb.max_corner[2] = 2.0;
    return row;
}

void testCpuPlanRetainsTargetUntilClose()
{
    OperationInstance                 operationInstance(1u);
    ovspatialquery_backend_runtime_t* runtime = nullptr;
    require(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS,
            "oracle runtime creation failed");
    PopulationTable populations(runtime);
    PopulationKey   population;
    require(populations.create(1u, population) == OVSPATIALQUERY_API_SUCCESS, "population creation failed");

    RaycastFamilyBatchPlanConfig config{};
    config.common.target.population = population;
    config.common.resultSlotCount   = 1u;
    config.plan.maximumQueryCount   = 1u;
    config.plan.hitCapacity         = 1u;

    std::unique_ptr<FamilyBatchPlanOperationOwner> owner;
    require(FamilyBatchPlanOperationOwner::create(populations, operationInstance.handle(), config, owner) ==
                OVSPATIALQUERY_API_SUCCESS,
            "CPU family plan creation failed");

    std::shared_ptr<void> retired;
    require(populations.release(population, retired) == OVSPATIALQUERY_API_BUSY,
            "live family plan did not retain its target");
    require(owner->close() == OVSPATIALQUERY_API_SUCCESS, "idle family plan did not close");
    require(populations.release(population, retired) == OVSPATIALQUERY_API_SUCCESS,
            "closed family plan did not release its target");

    owner.reset();
    ovspatialquery_test_oracle_destroy_runtime(runtime);
}

void testRayLaunchOwnsTypedResultUntilObservedDestroy()
{
    OperationInstance                 operationInstance(2u);
    ovspatialquery_backend_runtime_t* runtime = nullptr;
    require(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS,
            "oracle runtime creation failed");
    PopulationTable populations(runtime);
    PopulationKey   population;
    require(populations.create(2u, population) == OVSPATIALQUERY_API_SUCCESS, "population creation failed");
    publish(populations, population, 7, {object(41u, 2.0, 3.0)});

    RaycastFamilyBatchPlanConfig config{};
    config.common.target.population = population;
    config.common.resultSlotCount   = 1u;
    config.plan.maximumQueryCount   = 1u;
    config.plan.hitCapacity         = 1u;
    std::unique_ptr<FamilyBatchPlanOperationOwner> owner;
    require(FamilyBatchPlanOperationOwner::create(populations, operationInstance.handle(), config, owner) ==
                OVSPATIALQUERY_API_SUCCESS,
            "ray plan creation failed");
    setWorkersPaused(operationInstance, true);

    RaycastBatchLaunch launch;
    launch.rows.push_back(ray(0.0));
    ovspatialquery_raycast_batch_reservation_t reservation{};
    const FamilyBatchPlanAdmission             admission =
        owner->enqueueColumns({TargetSelectionKind::head, 0}, launch, reservation);
    require(admission.outcome == OVSPATIALQUERY_API_SUCCESS && admission.operation.valid() &&
                admission.resultSlot == 0u,
            "ray launch was not admitted into its fixed slot");
    ovspatialquery_raycast_batch_result_t result{};
    require(owner->fetchColumns(admission.operation, result) == OVSPATIALQUERY_API_NOT_READY,
            "pending ray result was readable");
    FamilyBatchPlanOperationId operation = admission.operation;
    require(owner->destroyOperation(operation) == OVSPATIALQUERY_API_BUSY && operation.valid(),
            "pending ray operation was destroyed");
    require(owner->close() == OVSPATIALQUERY_API_BUSY, "plan closed while its result slot was live");

    setWorkersPaused(operationInstance, false);
    InstanceOperationState state = InstanceOperationState::pending;
    require(waitInstanceOperation(operationInstance.handle(), operation.value, OVSPATIALQUERY_TIMEOUT_INFINITE,
                                  state) == OVSPATIALQUERY_API_SUCCESS &&
                state == InstanceOperationState::completed,
            "ray launch did not complete");
    require(owner->fetchColumns(operation, result) == OVSPATIALQUERY_API_SUCCESS, "ray result was not fetchable");
    require(result.common.source_ordinal == 7 && result.common.truth == OVSPATIALQUERY_HIT_BOUNDS_CANDIDATE &&
                storedHitCount(result.columns.hit_ranges) == 1u &&
                columnData<uint64_t>(result.columns.object_ids)[0] == 41u,
            "ray typed result lost immutable target truth");
    ovspatialquery_raycast_batch_result_t repeated{};
    require(owner->fetchColumns(operation, repeated) == OVSPATIALQUERY_API_SUCCESS &&
                repeated.common.source_ordinal == 7 && storedHitCount(repeated.columns.hit_ranges) == 1u,
            "ray fetch was not repeatable");
    require(owner->destroyOperation(operation) == OVSPATIALQUERY_API_SUCCESS && !operation.valid(),
            "observed ray operation did not release its slot");
    require(owner->close() == OVSPATIALQUERY_API_SUCCESS, "idle ray plan did not close");

    std::shared_ptr<void> retired;
    require(populations.release(population, retired) == OVSPATIALQUERY_API_SUCCESS,
            "closed ray plan retained its population");
    owner.reset();
    ovspatialquery_test_oracle_destroy_runtime(runtime);
}

void testSweepAndOverlapKeepStronglyTypedResultSurfaces()
{
    OperationInstance                 operationInstance(2u);
    ovspatialquery_backend_runtime_t* runtime = nullptr;
    require(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS,
            "oracle runtime creation failed");
    PopulationTable populations(runtime);
    PopulationKey   population;
    require(populations.create(2u, population) == OVSPATIALQUERY_API_SUCCESS, "population creation failed");
    publish(populations, population, 11, {object(52u, 4.0, 5.0)});

    SphereSweepFamilyBatchPlanConfig sweepConfig{};
    sweepConfig.common.target.population = population;
    sweepConfig.common.resultSlotCount   = 1u;
    sweepConfig.common.resultFields      = kBatchResultPosition | kBatchResultNormal | kBatchResultDistance |
                                      kBatchResultGeometryIdentity | kBatchResultQueryPosition;
    sweepConfig.plan.maximumQueryCount = 1u;
    sweepConfig.plan.hitCapacity       = 1u;
    std::unique_ptr<FamilyBatchPlanOperationOwner> sweepOwner;
    require(FamilyBatchPlanOperationOwner::create(populations, operationInstance.handle(), sweepConfig, sweepOwner) ==
                OVSPATIALQUERY_API_SUCCESS,
            "sweep plan creation failed");
    SphereSweepBatchLaunch sweepLaunch;
    sweepLaunch.rows.push_back(sweep(0.0));
    ovspatialquery_sphere_sweep_batch_reservation_t sweepReservation{};
    FamilyBatchPlanAdmission                        admission =
        sweepOwner->enqueueColumns({TargetSelectionKind::head, 0}, sweepLaunch, sweepReservation);
    require(admission.outcome == OVSPATIALQUERY_API_SUCCESS &&
                sweepReservation.common.result_slot == admission.resultSlot &&
                sweepReservation.columns.object_ids.shape != nullptr &&
                sweepReservation.columns.object_ids.shape[0] == 1 &&
                sweepReservation.columns.contact.candidate.query_positions.shape != nullptr &&
                sweepReservation.columns.contact.candidate.query_positions.shape[0] == 1,
            "sweep launch did not return its C-shaped typed reservation");
    InstanceOperationState state = InstanceOperationState::pending;
    require(waitInstanceOperation(operationInstance.handle(), admission.operation.value,
                                  OVSPATIALQUERY_TIMEOUT_INFINITE, state) == OVSPATIALQUERY_API_SUCCESS,
            "sweep launch did not complete");
    ovspatialquery_sphere_sweep_batch_result_t sweepColumns{};
    require(sweepOwner->fetchColumns(admission.operation, sweepColumns) == OVSPATIALQUERY_API_SUCCESS &&
                sweepColumns.common.source_ordinal == 11 &&
                sweepColumns.columns.object_ids.data == sweepReservation.columns.object_ids.data &&
                sweepColumns.columns.object_ids.shape != sweepReservation.columns.object_ids.shape &&
                sweepColumns.columns.object_ids.shape[0] == 1 &&
                static_cast<const uint64_t*>(sweepColumns.columns.object_ids.data)[0] == 52u &&
                sweepColumns.columns.contact.candidate.query_positions.shape[1] == 3,
            "sweep typed columns changed aliasing, H shape, or identity truth");
    ovspatialquery_raycast_batch_result_t wrongFamily{};
    wrongFamily.common.source_ordinal = 999;
    require(sweepOwner->fetchColumns(admission.operation, wrongFamily) == OVSPATIALQUERY_API_INVALID_ARGUMENT &&
                wrongFamily.common.source_ordinal == 999,
            "wrong-family fetch did not preserve its output");
    FamilyBatchPlanOperationId operation = admission.operation;
    require(sweepOwner->destroyOperation(operation) == OVSPATIALQUERY_API_SUCCESS, "sweep operation did not destroy");
    require(sweepOwner->close() == OVSPATIALQUERY_API_SUCCESS, "sweep plan did not close");

    OverlapFamilyBatchPlanConfig overlapConfig{};
    overlapConfig.common.target.population = population;
    overlapConfig.common.resultSlotCount   = 1u;
    overlapConfig.common.resultFields      = kBatchResultGeometryIdentity;
    overlapConfig.plan.maximumQueryCount   = 1u;
    overlapConfig.plan.hitCapacity         = 1u;
    std::unique_ptr<FamilyBatchPlanOperationOwner> overlapOwner;
    require(FamilyBatchPlanOperationOwner::create(populations, operationInstance.handle(), overlapConfig,
                                                  overlapOwner) == OVSPATIALQUERY_API_SUCCESS,
            "overlap plan creation failed");
    OverlapBatchLaunch overlapLaunch;
    overlapLaunch.rows.push_back(overlap(0.0, 10.0));
    ovspatialquery_overlap_batch_reservation_t overlapReservation{};
    admission = overlapOwner->enqueueColumns({TargetSelectionKind::head, 0}, overlapLaunch, overlapReservation);
    require(admission.outcome == OVSPATIALQUERY_API_SUCCESS &&
                overlapReservation.common.result_slot == admission.resultSlot &&
                overlapReservation.columns.object_ids.shape != nullptr &&
                overlapReservation.columns.object_ids.shape[0] == 1 &&
                overlapReservation.columns.geometry.types.shape != nullptr &&
                overlapReservation.columns.geometry.types.shape[0] == 1,
            "overlap launch did not return its C-shaped typed reservation");
    require(waitInstanceOperation(operationInstance.handle(), admission.operation.value,
                                  OVSPATIALQUERY_TIMEOUT_INFINITE, state) == OVSPATIALQUERY_API_SUCCESS,
            "overlap launch did not complete");
    ovspatialquery_overlap_batch_result_t overlapColumns{};
    require(overlapOwner->fetchColumns(admission.operation, overlapColumns) == OVSPATIALQUERY_API_SUCCESS &&
                overlapColumns.common.source_ordinal == 11 &&
                overlapColumns.columns.object_ids.data == overlapReservation.columns.object_ids.data &&
                overlapColumns.columns.object_ids.shape != overlapReservation.columns.object_ids.shape &&
                overlapColumns.columns.object_ids.shape[0] == 1 &&
                static_cast<const uint64_t*>(overlapColumns.columns.object_ids.data)[0] == 52u &&
                static_cast<const int32_t*>(overlapColumns.columns.geometry.types.data)[0] ==
                    OVSPATIALQUERY_GEOMETRY_SPHERE,
            "overlap typed columns changed aliasing, H shape, or geometry truth");
    operation = admission.operation;
    require(overlapOwner->destroyOperation(operation) == OVSPATIALQUERY_API_SUCCESS,
            "overlap operation did not destroy");
    require(overlapOwner->close() == OVSPATIALQUERY_API_SUCCESS, "overlap plan did not close");

    std::shared_ptr<void> retired;
    require(populations.release(population, retired) == OVSPATIALQUERY_API_SUCCESS,
            "typed family plans retained the population after close");
    sweepOwner.reset();
    overlapOwner.reset();
    ovspatialquery_test_oracle_destroy_runtime(runtime);
}

void testAdmissionPrecedenceBudgetAndFailureAtomicity()
{
    OperationInstance                 operationInstance(1u);
    ovspatialquery_backend_runtime_t* runtime = nullptr;
    require(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS,
            "oracle runtime creation failed");
    PopulationTable populations(runtime);
    PopulationKey   population;
    require(populations.create(1u, population) == OVSPATIALQUERY_API_SUCCESS, "population creation failed");

    RaycastFamilyBatchPlanConfig config{};
    config.common.target.population = population;
    config.common.resultSlotCount   = 1u;
    config.plan.maximumQueryCount   = 1u;
    config.plan.hitCapacity         = 1u;

    RaycastFamilyBatchPlanConfig rejected = config;
    rejected.common.resultSlotCount       = 0u;
    std::unique_ptr<FamilyBatchPlanOperationOwner> owner;
    require(FamilyBatchPlanOperationOwner::create(populations, operationInstance.handle(), config, owner) ==
                OVSPATIALQUERY_API_SUCCESS,
            "sentinel owner creation failed");
    FamilyBatchPlanOperationOwner* const sentinel = owner.get();
    require(FamilyBatchPlanOperationOwner::create(populations, operationInstance.handle(), rejected, owner) ==
                    OVSPATIALQUERY_API_INVALID_ARGUMENT &&
                owner.get() == sentinel,
            "invalid create mutated its output owner");
    rejected                = config;
    rejected.common.backend = FamilyBatchPlanBackend::cuda;
    // The all-family CUDA lane admits exact ray truth. Candidate truth cannot
    // request exact-only source features, so this remains an unsupported
    // configuration without depending on a physical CUDA device.
    rejected.common.cudaResultFields = kCudaFamilyResultSourceFeature;
    require(FamilyBatchPlanOperationOwner::create(populations, operationInstance.handle(), rejected, owner) ==
                    OVSPATIALQUERY_API_NOT_SUPPORTED &&
                owner.get() == sentinel,
            "unsupported backend did not fail truthfully and atomically");
    rejected                        = config;
    rejected.plan.maximumQueryCount = UINT64_MAX;
    require(FamilyBatchPlanOperationOwner::create(populations, operationInstance.handle(), rejected, owner) ==
                    OVSPATIALQUERY_API_OUT_OF_MEMORY &&
                owner.get() == sentinel,
            "unrepresentable fixed storage was not resource exhaustion");
    require(owner->close() == OVSPATIALQUERY_API_SUCCESS, "sentinel owner did not close");
    owner.reset();

    RaycastFamilyBatchPlanConfig invalidTarget = config;
    invalidTarget.common.target.population     = {UINT32_MAX, 0u};
    require(FamilyBatchPlanOperationOwner::create(populations, operationInstance.handle(), invalidTarget, owner) ==
                    OVSPATIALQUERY_API_INVALID_HANDLE &&
                owner == nullptr,
            "invalid target was not rejected");

    require(FamilyBatchPlanOperationOwner::create(populations, operationInstance.handle(), config, owner) ==
                OVSPATIALQUERY_API_SUCCESS,
            "admission owner creation failed");
    setWorkersPaused(operationInstance, true);
    ovspatialquery_raycast_batch_reservation_t      reservation{};
    ovspatialquery_sphere_sweep_batch_reservation_t wrongFamilyReservation{};
    RaycastBatchLaunch                              oversized;
    oversized.rows                     = {ray(0.0), ray(1.0)};
    FamilyBatchPlanAdmission admission = owner->enqueueColumns({TargetSelectionKind::head, 0}, oversized, reservation);
    require(admission.outcome == OVSPATIALQUERY_API_INVALID_ARGUMENT && !admission.operation.valid() &&
                admission.resultSlot == UINT32_MAX,
            "row-capacity rejection was not atomic or did not precede resolution");
    SphereSweepBatchLaunch         wrongFamilyLaunch;
    const FamilyBatchPlanAdmission unsupportedFamily =
        owner->enqueueColumns({TargetSelectionKind::head, 0}, wrongFamilyLaunch, wrongFamilyReservation);
    require(unsupportedFamily.outcome == OVSPATIALQUERY_API_NOT_SUPPORTED && !unsupportedFamily.operation.valid() &&
                unsupportedFamily.resultSlot == UINT32_MAX,
            "wrong-family launch occupied lifecycle state");
    RaycastBatchLaunch launch;
    launch.rows.push_back(ray(0.0));
    admission = owner->enqueueColumns({TargetSelectionKind::head, 0}, launch, reservation);
    require(admission.outcome == OVSPATIALQUERY_API_NOT_READY && !admission.operation.valid(),
            "empty population did not reject HEAD as not ready");
    publish(populations, population, 5, {object(61u, 2.0, 3.0)});
    admission = owner->enqueueColumns({TargetSelectionKind::retainedExact, -1}, launch, reservation);
    require(admission.outcome == OVSPATIALQUERY_API_INVALID_ARGUMENT && !admission.operation.valid(),
            "negative exact ordinal was not classified as invalid selection");
    admission = owner->enqueueColumns({TargetSelectionKind::retainedExact, 4}, launch, reservation);
    require(admission.outcome == OVSPATIALQUERY_API_ORDINAL_NOT_RETAINED && !admission.operation.valid(),
            "missing exact ordinal was admitted");
    admission = owner->enqueueColumns({static_cast<TargetSelectionKind>(255), 0}, launch, reservation);
    require(admission.outcome == OVSPATIALQUERY_API_INVALID_ARGUMENT && !admission.operation.valid(),
            "invalid ordinal selection was admitted");

    failNextScheduling(operationInstance);
    admission = owner->enqueueColumns({TargetSelectionKind::head, 0}, launch, reservation);
    require(admission.outcome == OVSPATIALQUERY_API_INTERNAL && !admission.operation.valid() &&
                admission.resultSlot == UINT32_MAX,
            "scheduling failure leaked an operation or result slot");
    admission = owner->enqueueColumns({TargetSelectionKind::head, 0}, launch, reservation);
    require(admission.outcome == OVSPATIALQUERY_API_SUCCESS && admission.resultSlot == 0u,
            "scheduling rollback did not restore the first fixed slot");
    const FamilyBatchPlanAdmission capacity =
        owner->enqueueColumns({TargetSelectionKind::head, 0}, launch, reservation);
    require(capacity.outcome == OVSPATIALQUERY_API_OPERATION_CAPACITY && !capacity.operation.valid(),
            "operation capacity did not precede result capacity");

    InstanceOperationState state = InstanceOperationState::completed;
    require(waitInstanceOperation(operationInstance.handle(), admission.operation.value, 0u, state) ==
                    OVSPATIALQUERY_API_TIMEOUT &&
                state == InstanceOperationState::completed,
            "poll timeout mutated its output or observed terminal state");
    setWorkersPaused(operationInstance, false);
    require(waitInstanceOperation(operationInstance.handle(), admission.operation.value,
                                  OVSPATIALQUERY_TIMEOUT_INFINITE, state) == OVSPATIALQUERY_API_SUCCESS,
            "admitted operation did not complete");
    ovspatialquery_raycast_batch_result_t fetched{};
    require(owner->fetchColumns(admission.operation, fetched) == OVSPATIALQUERY_API_SUCCESS,
            "completed result was not fetchable");
    FamilyBatchPlanOperationId operation = admission.operation;
    require(owner->destroyOperation(operation) == OVSPATIALQUERY_API_SUCCESS,
            "wait-observed operation did not destroy");

    setWorkersPaused(operationInstance, true);
    owner->failNextResourceForTest();
    admission = owner->enqueueColumns({TargetSelectionKind::head, 0}, launch, reservation);
    require(admission.outcome == OVSPATIALQUERY_API_SUCCESS, "resource-failure operation was not admitted");
    setWorkersPaused(operationInstance, false);
    state = InstanceOperationState::pending;
    require(waitInstanceOperation(operationInstance.handle(), admission.operation.value,
                                  OVSPATIALQUERY_TIMEOUT_INFINITE, state) == OVSPATIALQUERY_API_OPERATION_FAILED &&
                state == InstanceOperationState::failed,
            "resource failure did not become terminal operation failure");
    require(terminalStatus(operationInstance, admission.operation) == OVSPATIALQUERY_API_OUT_OF_MEMORY,
            "resource failure diagnostic was not repeatable and typed");
    fetched.common.source_ordinal = 1234;
    require(owner->fetchColumns(admission.operation, fetched) == OVSPATIALQUERY_API_OPERATION_FAILED &&
                fetched.common.source_ordinal == 1234,
            "failed fetch mutated its result output");
    operation = admission.operation;
    require(owner->destroyOperation(operation) == OVSPATIALQUERY_API_SUCCESS,
            "observed failed operation did not release its slot");

    for (const ovspatialquery_api_status_t expected :
         {OVSPATIALQUERY_API_OPERATION_FAILED, OVSPATIALQUERY_API_INTERNAL})
    {
        setWorkersPaused(operationInstance, true);
        if (expected == OVSPATIALQUERY_API_OPERATION_FAILED)
        {
            owner->failNextBackendForTest();
        }
        else
        {
            owner->failNextInternalForTest();
        }
        admission = owner->enqueueColumns({TargetSelectionKind::head, 0}, launch, reservation);
        require(admission.outcome == OVSPATIALQUERY_API_SUCCESS, "diagnostic failure launch was not admitted");
        setWorkersPaused(operationInstance, false);
        state = InstanceOperationState::pending;
        require(waitInstanceOperation(operationInstance.handle(), admission.operation.value,
                                      OVSPATIALQUERY_TIMEOUT_INFINITE, state) == OVSPATIALQUERY_API_OPERATION_FAILED &&
                    state == InstanceOperationState::failed,
                "diagnostic failure did not terminate");
        require(terminalStatus(operationInstance, admission.operation) == expected,
                "terminal diagnostic did not preserve its failure class");
        operation = admission.operation;
        require(owner->destroyOperation(operation) == OVSPATIALQUERY_API_SUCCESS,
                "diagnostic failure operation did not release its slot");
    }

    require(owner->close() == OVSPATIALQUERY_API_SUCCESS, "failure-test owner did not close");
    owner.reset();

    OperationInstance resultInstance(2u);
    config.common.resultSlotCount = 1u;
    require(FamilyBatchPlanOperationOwner::create(populations, resultInstance.handle(), config, owner) ==
                OVSPATIALQUERY_API_SUCCESS,
            "result-capacity owner creation failed");
    setWorkersPaused(resultInstance, true);
    admission = owner->enqueueColumns({TargetSelectionKind::head, 0}, launch, reservation);
    const FamilyBatchPlanAdmission resultCapacity =
        owner->enqueueColumns({TargetSelectionKind::head, 0}, launch, reservation);
    require(admission.outcome == OVSPATIALQUERY_API_SUCCESS &&
                resultCapacity.outcome == OVSPATIALQUERY_API_RESULT_CAPACITY && !resultCapacity.operation.valid(),
            "result capacity did not reject atomically after operation availability");
    setWorkersPaused(resultInstance, false);
    state = InstanceOperationState::pending;
    require(waitInstanceOperation(resultInstance.handle(), admission.operation.value, OVSPATIALQUERY_TIMEOUT_INFINITE,
                                  state) == OVSPATIALQUERY_API_SUCCESS,
            "result-capacity operation did not complete");
    operation = admission.operation;
    require(owner->destroyOperation(operation) == OVSPATIALQUERY_API_SUCCESS &&
                owner->close() == OVSPATIALQUERY_API_SUCCESS,
            "result-capacity operation did not release");
    owner.reset();

    std::shared_ptr<void> retired;
    require(populations.release(population, retired) == OVSPATIALQUERY_API_SUCCESS,
            "instance-style owner teardown leaked target or generation resources");
    ovspatialquery_test_oracle_destroy_runtime(runtime);
}

void testConcurrentLaunchesHoldResolvedGenerationTruthAndReuseSafely()
{
    OperationInstance                 operationInstance(2u, 2u);
    ovspatialquery_backend_runtime_t* runtime = nullptr;
    require(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS,
            "oracle runtime creation failed");
    PopulationTable populations(runtime);
    PopulationKey   population;
    require(populations.create(1u, population) == OVSPATIALQUERY_API_SUCCESS, "population creation failed");
    const ovspatialquery_view_handle_t viewHandle = ovspatialquery_view_handle_t{7001u};
    std::shared_ptr<ViewState>         view       = std::make_shared<ViewState>();
    view->handle                                  = viewHandle;
    view->parent                                  = population;
    view->automatic                               = true;
    PopulationObject first                        = object(71u, 2.0, 3.0);
    first.participationBits                       = {{UINT64_C(0x1), 0u}};
    PopulationObject second                       = object(72u, 4.0, 5.0);
    second.participationBits                      = {{UINT64_C(0x2), 0u}};
    publish(populations, population, 10, {first, second}, {{viewHandle, selected({71u})}});

    RaycastFamilyBatchPlanConfig config{};
    config.common.target          = {population, view};
    config.common.resultSlotCount = 2u;
    config.plan.maximumQueryCount = 1u;
    config.plan.hitCapacity       = 1u;
    std::unique_ptr<FamilyBatchPlanOperationOwner> owner;
    require(FamilyBatchPlanOperationOwner::create(populations, operationInstance.handle(), config, owner) ==
                OVSPATIALQUERY_API_SUCCESS,
            "retained-view plan creation failed");
    setWorkersPaused(operationInstance, true);

    ovspatialquery_raycast_batch_reservation_t reservation{};
    RaycastBatchLaunch                         firstLaunch;
    firstLaunch.rows.push_back(ray(0.0));
    firstLaunch.rows[0].queryBits = {{UINT64_C(0x1), 0u}};
    const FamilyBatchPlanAdmission firstAdmission =
        owner->enqueueColumns({TargetSelectionKind::retainedExact, 10}, firstLaunch, reservation);
    require(firstAdmission.outcome == OVSPATIALQUERY_API_SUCCESS && firstAdmission.resultSlot == 0u,
            "retained ordinal launch was not admitted");

    first.bounds.minimum[0] = 20.0;
    first.bounds.maximum[0] = 21.0;
    publish(populations, population, 11, {first, second}, {{viewHandle, selected({72u})}});
    publish(populations, population, 12, {first, second}, {{viewHandle, selected({72u})}});
    RaycastBatchLaunch secondLaunch;
    secondLaunch.rows.push_back(ray(0.0));
    secondLaunch.rows[0].queryBits = {{UINT64_C(0x2), 0u}};
    const FamilyBatchPlanAdmission secondAdmission =
        owner->enqueueColumns({TargetSelectionKind::head, 0}, secondLaunch, reservation);
    require(secondAdmission.outcome == OVSPATIALQUERY_API_SUCCESS && secondAdmission.resultSlot == 1u,
            "concurrent HEAD launch did not reserve the second fixed slot");

    ovspatialquery_api_status_t waiterOutcome = OVSPATIALQUERY_API_BUSY;
    InstanceOperationState      waiterState   = InstanceOperationState::pending;
    std::thread                 waiter(
        [&]()
        {
            waiterOutcome = waitInstanceOperation(operationInstance.handle(), firstAdmission.operation.value,
                                                                  OVSPATIALQUERY_TIMEOUT_INFINITE - 1u, waiterState);
        });
    setWorkersPaused(operationInstance, false);
    waiter.join();
    require(waiterOutcome == OVSPATIALQUERY_API_SUCCESS && waiterState == InstanceOperationState::completed,
            "large finite wait did not preserve or complete the retained launch");
    InstanceOperationState secondState = InstanceOperationState::pending;
    require(waitInstanceOperation(operationInstance.handle(), secondAdmission.operation.value,
                                  OVSPATIALQUERY_TIMEOUT_INFINITE, secondState) == OVSPATIALQUERY_API_SUCCESS,
            "second concurrent launch did not complete");

    ovspatialquery_raycast_batch_result_t retainedResult{};
    ovspatialquery_raycast_batch_result_t headResult{};
    require(owner->fetchColumns(firstAdmission.operation, retainedResult) == OVSPATIALQUERY_API_SUCCESS &&
                retainedResult.common.source_ordinal == 10 && storedHitCount(retainedResult.columns.hit_ranges) == 1u &&
                columnData<uint64_t>(retainedResult.columns.object_ids)[0] == 71u,
            "evicted retained launch lost generation membership or participation");
    require(owner->fetchColumns(secondAdmission.operation, headResult) == OVSPATIALQUERY_API_SUCCESS &&
                headResult.common.source_ordinal == 12 && storedHitCount(headResult.columns.hit_ranges) == 1u &&
                columnData<uint64_t>(headResult.columns.object_ids)[0] == 72u,
            "HEAD launch did not capture its publication membership/participation");

    const FamilyBatchPlanOperationId stale           = firstAdmission.operation;
    FamilyBatchPlanOperationId       firstOperation  = firstAdmission.operation;
    FamilyBatchPlanOperationId       secondOperation = secondAdmission.operation;
    require(owner->destroyOperation(firstOperation) == OVSPATIALQUERY_API_SUCCESS &&
                owner->destroyOperation(secondOperation) == OVSPATIALQUERY_API_SUCCESS,
            "concurrent operations did not release their slots");
    setWorkersPaused(operationInstance, true);
    const FamilyBatchPlanAdmission reused =
        owner->enqueueColumns({TargetSelectionKind::head, 0}, secondLaunch, reservation);
    require(reused.outcome == OVSPATIALQUERY_API_SUCCESS && reused.resultSlot == 0u &&
                reused.operation.value != stale.value,
            "released operation/result slot did not reuse generation-safely");
    InstanceOperationState untouched = InstanceOperationState::failed;
    require(queryInstanceOperation(operationInstance.handle(), stale.value, untouched) ==
                    OVSPATIALQUERY_API_INVALID_HANDLE &&
                untouched == InstanceOperationState::failed,
            "stale operation ID resolved to a reused slot");
    setWorkersPaused(operationInstance, false);
    InstanceOperationState reusedState = InstanceOperationState::pending;
    require(waitInstanceOperation(operationInstance.handle(), reused.operation.value, OVSPATIALQUERY_TIMEOUT_INFINITE,
                                  reusedState) == OVSPATIALQUERY_API_SUCCESS,
            "reused operation did not complete");
    FamilyBatchPlanOperationId reusedOperation = reused.operation;
    require(owner->destroyOperation(reusedOperation) == OVSPATIALQUERY_API_SUCCESS &&
                owner->close() == OVSPATIALQUERY_API_SUCCESS,
            "reused operation did not release");
    owner.reset();

    std::shared_ptr<void> retired;
    require(populations.release(population, retired) == OVSPATIALQUERY_API_SUCCESS,
            "instance teardown with pending work leaked target/generation resources");
    ovspatialquery_test_oracle_destroy_runtime(runtime);
}

void testExactPayloadZeroRowsNoHitsAndDeterministicStress()
{
    OperationInstance                 operationInstance(2u);
    ovspatialquery_backend_runtime_t* runtime = nullptr;
    require(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS,
            "oracle runtime creation failed");
    PopulationTable populations(runtime);
    PopulationKey   population;
    require(populations.create(2u, population) == OVSPATIALQUERY_API_SUCCESS, "population creation failed");
    publish(populations, population, 20, {exactObject(91u, 9191u, meshGeometry())});

    RaycastFamilyBatchPlanConfig config{};
    config.common.target.population = population;
    config.common.resultSlotCount   = 1u;
    config.common.resultFields      = kBatchResultPosition | kBatchResultNormal | kBatchResultDistance |
                                 kBatchResultGeometryIdentity | kBatchResultSourceFeature;
    config.plan.hitMode             = HitMode::multiple;
    config.plan.precision           = QueryPrecision::exact;
    config.plan.maximumQueryCount   = 1u;
    config.plan.maximumHitsPerQuery = 1u;
    config.plan.hitCapacity         = 1u;
    std::unique_ptr<FamilyBatchPlanOperationOwner> owner;
    require(FamilyBatchPlanOperationOwner::create(populations, operationInstance.handle(), config, owner) ==
                OVSPATIALQUERY_API_SUCCESS,
            "exact payload plan creation failed");

    ovspatialquery_raycast_batch_reservation_t reservation{};
    RaycastBatchLaunch                         exactLaunch;
    exactLaunch.rows.push_back(ray(0.0));
    exactLaunch.rows[0].query.origin[1] = 0.75;
    exactLaunch.rows[0].maximumHits     = 1u;
    FamilyBatchPlanAdmission admission =
        owner->enqueueColumns({TargetSelectionKind::head, 0}, exactLaunch, reservation);
    const ovspatialquery_raycast_batch_reservation_t exactReservation = reservation;
    require(admission.outcome == OVSPATIALQUERY_API_SUCCESS && exactReservation.common.query_count == 1u &&
                exactReservation.common.hit_capacity == 1u && exactReservation.columns.object_ids.shape != nullptr &&
                exactReservation.columns.object_ids.shape[0] == 1 &&
                exactReservation.columns.contact.exact.features.kinds.shape != nullptr &&
                exactReservation.columns.contact.exact.features.kinds.shape[0] == 1,
            "exact ray operation did not own its capacity-shaped reservation");
    InstanceOperationState state = InstanceOperationState::pending;
    require(admission.outcome == OVSPATIALQUERY_API_SUCCESS &&
                waitInstanceOperation(operationInstance.handle(), admission.operation.value,
                                      OVSPATIALQUERY_TIMEOUT_INFINITE, state) == OVSPATIALQUERY_API_SUCCESS,
            "exact payload launch did not complete");
    ovspatialquery_raycast_batch_result_t columnResult{};
    require(owner->fetchColumns(admission.operation, columnResult) == OVSPATIALQUERY_API_SUCCESS &&
                columnResult.common.source_ordinal == 20 &&
                columnResult.common.truth == OVSPATIALQUERY_HIT_EXACT_SURFACE &&
                sameColumn(columnResult.columns.hit_ranges.hit_offsets, std::vector<uint64_t>{0u}) &&
                sameColumn(columnResult.columns.hit_ranges.hit_counts, std::vector<uint64_t>{1u}) &&
                sameColumn(columnResult.columns.hit_ranges.truncated, std::vector<uint8_t>{0u}) &&
                storedHitCount(columnResult.columns.hit_ranges) == 1u &&
                columnResult.columns.object_ids.data == exactReservation.columns.object_ids.data &&
                columnResult.columns.object_ids.shape != exactReservation.columns.object_ids.shape &&
                columnResult.columns.object_ids.shape[0] == 1 &&
                columnData<uint64_t>(columnResult.columns.object_ids)[0] == 91u &&
                columnData<uint64_t>(columnResult.columns.geometry.values)[0] == 9191u &&
                columnData<uint32_t>(columnResult.columns.contact.exact.features.kinds)[0] ==
                    OVSPATIALQUERY_GEOMETRY_FEATURE_TRIANGLE,
            "exact ray operation changed column identity, partitions, feature truth, or aliasing");
    const ovspatialquery_exact_feature_columns_t& features    = columnResult.columns.contact.exact.features;
    const uint32_t*                               corners     = columnData<uint32_t>(features.source_corner_indices);
    const float*                                  barycentric = columnData<float>(features.barycentrics);
    require(columnData<uint64_t>(features.source_face_indices)[0] == UINT64_C(4294967295) && corners[0] == 0u &&
                corners[1] == 2u && corners[2] == 3u && std::abs(barycentric[0] - 0.125f) < 1.0e-6f &&
                std::abs(barycentric[1] - 0.5f) < 1.0e-6f && std::abs(barycentric[2] - 0.375f) < 1.0e-6f,
            "fixed result slot changed triangle source feature payload");
    FamilyBatchPlanOperationId operation = admission.operation;
    require(owner->destroyOperation(operation) == OVSPATIALQUERY_API_SUCCESS,
            "exact payload operation did not release its slot");

    RaycastBatchLaunch zeroRows;
    admission = owner->enqueueColumns({TargetSelectionKind::head, 0}, zeroRows, reservation);
    require(admission.outcome == OVSPATIALQUERY_API_SUCCESS, "zero-row launch was not admitted");
    bool                                  fetchedTerminal = false;
    ovspatialquery_raycast_batch_result_t result{};
    for (uint32_t attempt = 0u; attempt < 100000u; ++attempt)
    {
        const ovspatialquery_api_status_t outcome = owner->fetchColumns(admission.operation, result);
        if (outcome == OVSPATIALQUERY_API_SUCCESS)
        {
            fetchedTerminal = true;
            break;
        }
        require(outcome == OVSPATIALQUERY_API_NOT_READY, "zero-row fetch returned a non-terminal error");
        std::this_thread::yield();
    }
    require(fetchedTerminal && result.common.source_ordinal == 20 &&
                result.common.truth == OVSPATIALQUERY_HIT_EXACT_SURFACE && result.common.query_count == 0u &&
                storedHitCount(result.columns.hit_ranges) == 0u,
            "zero-row result lost target truth or gained partitions");
    operation = admission.operation;
    require(owner->destroyOperation(operation) == OVSPATIALQUERY_API_SUCCESS,
            "terminal typed fetch did not satisfy terminal observation");

    RaycastBatchLaunch noHit;
    noHit.rows.push_back(ray(0.0));
    noHit.rows[0].query.origin[1] = 100.0;
    for (uint32_t iteration = 0u; iteration < 128u; ++iteration)
    {
        admission = owner->enqueueColumns({TargetSelectionKind::head, 0}, noHit, reservation);
        require(admission.outcome == OVSPATIALQUERY_API_SUCCESS && admission.resultSlot == 0u,
                "stress launch did not deterministically reuse result slot zero");
        state = InstanceOperationState::pending;
        require(waitInstanceOperation(operationInstance.handle(), admission.operation.value,
                                      OVSPATIALQUERY_TIMEOUT_INFINITE, state) == OVSPATIALQUERY_API_SUCCESS,
                "stress launch did not complete");
        result = {};
        require(owner->fetchColumns(admission.operation, result) == OVSPATIALQUERY_API_SUCCESS &&
                    result.common.source_ordinal == 20 &&
                    sameColumn(result.columns.hit_ranges.hit_counts, std::vector<uint64_t>{0u}) &&
                    sameColumn(result.columns.hit_ranges.truncated, std::vector<uint8_t>{0u}) &&
                    storedHitCount(result.columns.hit_ranges) == 0u,
                "stress no-hit result changed semantic partitions");
        operation = admission.operation;
        require(owner->destroyOperation(operation) == OVSPATIALQUERY_API_SUCCESS,
                "stress operation did not release its slot");
    }
    require(owner->close() == OVSPATIALQUERY_API_SUCCESS, "stress plan did not close");
    owner.reset();
    std::shared_ptr<void> retired;
    require(populations.release(population, retired) == OVSPATIALQUERY_API_SUCCESS,
            "stress plan leaked its population");
    ovspatialquery_test_oracle_destroy_runtime(runtime);
}

void testMixedRowsPreservePartitionsAndPhysicalCapacity()
{
    OperationInstance                 operationInstance(1u);
    ovspatialquery_backend_runtime_t* runtime = nullptr;
    require(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS,
            "oracle runtime creation failed");
    PopulationTable populations(runtime);
    PopulationKey   population;
    require(populations.create(1u, population) == OVSPATIALQUERY_API_SUCCESS, "population creation failed");
    publish(populations, population, 30, {object(101u, 2.0, 3.0), object(102u, 4.0, 5.0)});

    RaycastFamilyBatchPlanConfig config{};
    config.common.target.population = population;
    config.common.resultSlotCount   = 1u;
    config.plan.hitMode             = HitMode::multiple;
    config.plan.maximumQueryCount   = 3u;
    config.plan.maximumHitsPerQuery = 2u;
    config.plan.hitCapacity         = 1u;
    std::unique_ptr<FamilyBatchPlanOperationOwner> owner;
    require(FamilyBatchPlanOperationOwner::create(populations, operationInstance.handle(), config, owner) ==
                OVSPATIALQUERY_API_SUCCESS,
            "mixed-row plan creation failed");

    ovspatialquery_raycast_batch_reservation_t reservation{};
    RaycastBatchLaunch                         launch;
    launch.rows                        = {ray(0.0), ray(0.0), ray(0.0)};
    launch.rows[0].maximumHits         = 2u;
    launch.rows[1].query.unitDirection = {0.0f, 0.0f, 0.0f};
    launch.rows[1].maximumHits         = 2u;
    launch.rows[2].maximumHits         = 2u;
    const FamilyBatchPlanAdmission admission =
        owner->enqueueColumns({TargetSelectionKind::head, 0}, launch, reservation);
    InstanceOperationState state = InstanceOperationState::pending;
    require(admission.outcome == OVSPATIALQUERY_API_SUCCESS &&
                waitInstanceOperation(operationInstance.handle(), admission.operation.value,
                                      OVSPATIALQUERY_TIMEOUT_INFINITE, state) == OVSPATIALQUERY_API_SUCCESS,
            "mixed-row launch did not complete");
    const std::vector<uint32_t> expectedStatus{
        static_cast<uint32_t>(OVSPATIALQUERY_BATCH_ROW_SUCCESS),
        static_cast<uint32_t>(OVSPATIALQUERY_BATCH_ROW_INVALID_ARGUMENT),
        static_cast<uint32_t>(OVSPATIALQUERY_BATCH_ROW_SUCCESS),
    };
    const std::vector<uint64_t> expectedCounts{1u, 0u, 0u};
    const std::vector<uint8_t>  expectedTruncated{1u, 0u, 1u};
    // Only the nonempty success row's offset is defined.
    const auto samePartitions = [&](const ovspatialquery_raycast_batch_result_t& value)
    {
        return sameColumn(value.columns.hit_ranges.query_status, expectedStatus) &&
               columnData<uint64_t>(value.columns.hit_ranges.hit_offsets)[0] == 0u &&
               sameColumn(value.columns.hit_ranges.hit_counts, expectedCounts) &&
               sameColumn(value.columns.hit_ranges.truncated, expectedTruncated);
    };
    ovspatialquery_raycast_batch_result_t result{};
    require(owner->fetchColumns(admission.operation, result) == OVSPATIALQUERY_API_SUCCESS &&
                result.common.source_ordinal == 30 && result.common.hit_capacity == 1u && samePartitions(result) &&
                storedHitCount(result.columns.hit_ranges) == 1u &&
                columnData<uint64_t>(result.columns.object_ids)[0] == 101u,
            "result-slot commit changed row isolation, ordering, or partitions");
    // Both fetches alias one slot's columns, so a repeat fetch is re-checked
    // against the literal partitions instead of against the first fetch.
    ovspatialquery_raycast_batch_result_t repeated{};
    require(owner->fetchColumns(admission.operation, repeated) == OVSPATIALQUERY_API_SUCCESS &&
                samePartitions(repeated),
            "repeat fetch changed the atomic mixed-row result");
    FamilyBatchPlanOperationId operation = admission.operation;
    require(owner->destroyOperation(operation) == OVSPATIALQUERY_API_SUCCESS &&
                owner->close() == OVSPATIALQUERY_API_SUCCESS,
            "mixed-row operation did not release");
    owner.reset();
    std::shared_ptr<void> retired;
    require(populations.release(population, retired) == OVSPATIALQUERY_API_SUCCESS,
            "mixed-row plan leaked its population");
    ovspatialquery_test_oracle_destroy_runtime(runtime);
}

void testWaitRevalidatesSlotGenerationAndPreservesLargeTimeouts()
{
    OperationInstance                 operationInstance(1u);
    ovspatialquery_backend_runtime_t* runtime = nullptr;
    require(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS,
            "oracle runtime creation failed");
    PopulationTable populations(runtime);
    PopulationKey   population;
    require(populations.create(2u, population) == OVSPATIALQUERY_API_SUCCESS, "population creation failed");
    publish(populations, population, 1, {object(81u, 2.0, 3.0)});

    RaycastFamilyBatchPlanConfig config{};
    config.common.target.population = population;
    config.common.resultSlotCount   = 1u;
    config.plan.maximumQueryCount   = 1u;
    config.plan.hitCapacity         = 1u;
    std::unique_ptr<FamilyBatchPlanOperationOwner> owner;
    require(FamilyBatchPlanOperationOwner::create(populations, operationInstance.handle(), config, owner) ==
                OVSPATIALQUERY_API_SUCCESS,
            "wait-race plan creation failed");
    ovspatialquery_raycast_batch_reservation_t reservation{};
    RaycastBatchLaunch                         launch;
    launch.rows.push_back(ray(0.0));

    setWorkersPaused(operationInstance, true);
    setWaiterPaused(operationInstance, true);
    const uint64_t                 staleEntriesBefore = waiterEntryCount(operationInstance);
    const FamilyBatchPlanAdmission first = owner->enqueueColumns({TargetSelectionKind::head, 0}, launch, reservation);
    require(first.outcome == OVSPATIALQUERY_API_SUCCESS, "wait-race first launch was not admitted");
    ovspatialquery_api_status_t staleOutcome = OVSPATIALQUERY_API_SUCCESS;
    InstanceOperationState      staleState   = InstanceOperationState::pending;
    std::thread                 staleWaiter(
        [&]()
        {
            staleOutcome = waitInstanceOperation(operationInstance.handle(), first.operation.value,
                                                                 OVSPATIALQUERY_TIMEOUT_INFINITE, staleState);
        });
    while (waiterEntryCount(operationInstance) == staleEntriesBefore)
    {
        std::this_thread::yield();
    }
    setWorkersPaused(operationInstance, false);
    waitForWaiter(operationInstance);

    InstanceOperationState observed = InstanceOperationState::pending;
    require(queryInstanceOperation(operationInstance.handle(), first.operation.value, observed) ==
                    OVSPATIALQUERY_API_SUCCESS &&
                observed == InstanceOperationState::completed,
            "wait-race operation was not terminal");
    FamilyBatchPlanOperationId firstOperation = first.operation;
    require(owner->destroyOperation(firstOperation) == OVSPATIALQUERY_API_SUCCESS,
            "wait-race operation was not destroyed after observation");
    setWorkersPaused(operationInstance, true);
    const FamilyBatchPlanAdmission reused = owner->enqueueColumns({TargetSelectionKind::head, 0}, launch, reservation);
    require(reused.outcome == OVSPATIALQUERY_API_SUCCESS && reused.operation.value != first.operation.value,
            "wait-race slot did not receive a new generation");

    setWaiterPaused(operationInstance, false);
    staleWaiter.join();
    require(staleOutcome == OVSPATIALQUERY_API_INVALID_HANDLE,
            "stale waiter crossed an operation-slot generation reuse");

    const uint64_t              entriesBefore = waiterEntryCount(operationInstance);
    ovspatialquery_api_status_t largeOutcome  = OVSPATIALQUERY_API_TIMEOUT;
    InstanceOperationState      largeState    = InstanceOperationState::pending;
    std::thread                 largeWaiter(
        [&]()
        {
            largeOutcome =
                waitInstanceOperation(operationInstance.handle(), reused.operation.value,
                                                      static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) + 1u, largeState);
        });
    while (waiterEntryCount(operationInstance) == entriesBefore)
    {
        std::this_thread::yield();
    }
    setWorkersPaused(operationInstance, false);
    largeWaiter.join();
    require(largeOutcome == OVSPATIALQUERY_API_SUCCESS && largeState == InstanceOperationState::completed,
            "finite timeout above INT64_MAX narrowed or timed out prematurely");
    FamilyBatchPlanOperationId reusedOperation = reused.operation;
    require(owner->destroyOperation(reusedOperation) == OVSPATIALQUERY_API_SUCCESS &&
                owner->close() == OVSPATIALQUERY_API_SUCCESS,
            "wait-race plan did not release its reused operation");
    owner.reset();
    std::shared_ptr<void> retired;
    require(populations.release(population, retired) == OVSPATIALQUERY_API_SUCCESS,
            "wait-race plan leaked its population");
    ovspatialquery_test_oracle_destroy_runtime(runtime);
}

void testExecutionPeakUsesAtomicHostLeaseAndReleasesAtTerminal()
{
    OperationInstance                 operationInstance(1u);
    ovspatialquery_backend_runtime_t* runtime = nullptr;
    require(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS,
            "ledger runtime creation failed");
    PopulationTable populations(runtime);
    PopulationKey   population;
    require(populations.create(1u, population) == OVSPATIALQUERY_API_SUCCESS, "ledger population creation failed");
    publish(populations, population, 17, {object(1u, 2.0, 3.0)});
    ovspatialquery_raycast_batch_reservation_t reservation{};
    RaycastBatchLaunch                         launch;
    launch.rows.push_back(ray(0.0));

    const auto makeOwner = [&](InstanceHostLedger& ledger)
    {
        RaycastFamilyBatchPlanConfig config{};
        config.common.target.population = population;
        config.common.resultSlotCount   = 1u;
        config.common.hostLedger        = &ledger;
        config.plan.maximumQueryCount   = 1u;
        config.plan.hitCapacity         = 1u;
        std::unique_ptr<FamilyBatchPlanOperationOwner> owner;
        require(FamilyBatchPlanOperationOwner::create(populations, operationInstance.handle(), config, owner) ==
                    OVSPATIALQUERY_API_SUCCESS,
                "ledger plan creation failed");
        return owner;
    };

    InstanceHostLedger                             calibrationLedger(UINT64_MAX);
    std::unique_ptr<FamilyBatchPlanOperationOwner> calibration = makeOwner(calibrationLedger);
    setWorkersPaused(operationInstance, true);
    const FamilyBatchPlanAdmission calibrationAdmission =
        calibration->enqueueColumns({TargetSelectionKind::head, 0}, launch, reservation);
    require(calibrationAdmission.outcome == OVSPATIALQUERY_API_SUCCESS, "ledger calibration launch was not admitted");
    const uint64_t executionCharge = calibrationLedger.total();
    require(executionCharge != 0u, "ledger execution peak was not charged");
    setWorkersPaused(operationInstance, false);
    InstanceOperationState state = InstanceOperationState::pending;
    require(waitInstanceOperation(operationInstance.handle(), calibrationAdmission.operation.value,
                                  OVSPATIALQUERY_TIMEOUT_INFINITE, state) == OVSPATIALQUERY_API_SUCCESS &&
                state == InstanceOperationState::completed && calibrationLedger.total() == executionCharge,
            "ledger live task charge changed before operation destroy");
    FamilyBatchPlanOperationId calibrationOperation = calibrationAdmission.operation;
    require(calibration->destroyOperation(calibrationOperation) == OVSPATIALQUERY_API_SUCCESS &&
                calibrationLedger.total() == 0u && calibration->close() == OVSPATIALQUERY_API_SUCCESS,
            "ledger calibration task charge did not release at destroy");
    calibration.reset();

    InstanceHostLedger                             rejectedLedger(executionCharge - 1u);
    std::unique_ptr<FamilyBatchPlanOperationOwner> rejected = makeOwner(rejectedLedger);
    const FamilyBatchPlanAdmission                 rejection =
        rejected->enqueueColumns({TargetSelectionKind::head, 0}, launch, reservation);
    require(rejection.outcome == OVSPATIALQUERY_API_HOST_BUDGET && !rejection.operation.valid() &&
                rejectedLedger.total() == 0u && rejected->close() == OVSPATIALQUERY_API_SUCCESS,
            "ledger charge-1 admission mutated plan/runtime state");
    rejected.reset();

    InstanceHostLedger                             exactLedger(executionCharge);
    std::unique_ptr<FamilyBatchPlanOperationOwner> exact = makeOwner(exactLedger);
    setWorkersPaused(operationInstance, true);
    const FamilyBatchPlanAdmission exactAdmission =
        exact->enqueueColumns({TargetSelectionKind::head, 0}, launch, reservation);
    require(exactAdmission.outcome == OVSPATIALQUERY_API_SUCCESS && exactLedger.total() == executionCharge,
            "ledger exact-boundary launch was not admitted atomically");
    setWorkersPaused(operationInstance, false);
    state = InstanceOperationState::pending;
    require(waitInstanceOperation(operationInstance.handle(), exactAdmission.operation.value,
                                  OVSPATIALQUERY_TIMEOUT_INFINITE, state) == OVSPATIALQUERY_API_SUCCESS &&
                state == InstanceOperationState::completed && exactLedger.total() == executionCharge,
            "ledger exact-boundary live task charge changed before destroy");
    FamilyBatchPlanOperationId exactOperation = exactAdmission.operation;
#if defined(OVSPATIALQUERY_ENABLE_FAMILY_BATCH_PLAN_OPERATION_TEST_HOOK)
    gFamilyTaskDestroyLedger         = &exactLedger;
    gFamilyTaskDestroyObservedCharge = UINT64_MAX;
    setFamilyBatchPlanTaskPayloadDestroyHookForTest(&observeFamilyTaskPayloadDestroy);
#endif
    require(exact->destroyOperation(exactOperation) == OVSPATIALQUERY_API_SUCCESS && exactLedger.total() == 0u &&
                exact->close() == OVSPATIALQUERY_API_SUCCESS,
            "ledger exact-boundary plan did not release its slot");
#if defined(OVSPATIALQUERY_ENABLE_FAMILY_BATCH_PLAN_OPERATION_TEST_HOOK)
    setFamilyBatchPlanTaskPayloadDestroyHookForTest(nullptr);
    gFamilyTaskDestroyLedger = nullptr;
    require(gFamilyTaskDestroyObservedCharge == executionCharge,
            "task payload charge released before charged task storage retired");
#endif
    exact.reset();

    std::shared_ptr<void> retired;
    require(populations.release(population, retired) == OVSPATIALQUERY_API_SUCCESS,
            "ledger plan leaked its population lease");
    ovspatialquery_test_oracle_destroy_runtime(runtime);
}

} // namespace

int main()
{
    testCpuPlanRetainsTargetUntilClose();
    testRayLaunchOwnsTypedResultUntilObservedDestroy();
    testSweepAndOverlapKeepStronglyTypedResultSurfaces();
    testAdmissionPrecedenceBudgetAndFailureAtomicity();
    testConcurrentLaunchesHoldResolvedGenerationTruthAndReuseSafely();
    testExactPayloadZeroRowsNoHitsAndDeterministicStress();
    testMixedRowsPreservePartitionsAndPhysicalCapacity();
    testWaitRevalidatesSlotGenerationAndPreservesLargeTimeouts();
    testExecutionPeakUsesAtomicHostLeaseAndReleasesAtTerminal();
    return 0;
}
