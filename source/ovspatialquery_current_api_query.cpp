/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_float_bounds.h"
#include "ovspatialquery_instance_state.h"

#include "ovspatialquery_family_batch_cpu.h"
#include "ovspatialquery_instance_operation_runtime.h"
#include "ovspatialquery_pair_materialization.h"
#include "ovspatialquery_target_resolver.h"

#include <ovspatialquery_query_evaluation.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iterator>
#include <type_traits>

using namespace ovspatialquery::detail;

namespace
{

template <typename T>
void apiWriteElement(void* destination, size_t index, const T& value)
{
    std::memcpy(static_cast<uint8_t*>(destination) + index * sizeof(T), &value, sizeof(T));
}

ovspatialquery_api_status_t apiTargetRequest(const InstanceReference&               instance,
                                             const ovspatialquery_query_target_t&   source,
                                             ovspatialquery::detail::TargetRequest& outRequest)
{
    ovspatialquery::detail::PopulationTarget target{};
    if (source.target.kind == 0u)
    {
        const auto status = instance->handles.resolvePopulation(source.target.handle.population, target.population);
        if (status != OVSPATIALQUERY_API_SUCCESS)
            return status;
    }
    else if (source.target.kind == 1u)
    {
        if (instance->handles.resolveView(source.target.handle.view, target.view) != OVSPATIALQUERY_API_SUCCESS)
        {
            return OVSPATIALQUERY_API_INVALID_HANDLE;
        }
        target.population = target.view->parent;
    }
    else
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }

    bool                                          valid     = false;
    const ovspatialquery::detail::TargetSelection selection = apiSelection(source.ordinal, valid);
    if (!valid || selection.kind == ovspatialquery::detail::TargetSelectionKind::admittedExact)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    outRequest = {target, selection};
    return OVSPATIALQUERY_API_SUCCESS;
}

void apiIdentity(const ovspatialquery::detail::SemanticHitIdentity& source, ovspatialquery_hit_identity_t& out)
{
    out.object_id      = source.objectId;
    out.geometry.type  = static_cast<ovspatialquery_geometry_type_t>(static_cast<uint32_t>(source.geometry.family) +
                                                                    kGeometryTypeFirstBuiltin);
    out.geometry.value = source.geometry.value;
    out.material_id    = source.materialId;
}

void apiFeature(const ovspatialquery::detail::GeometryFeature& source, ovspatialquery_geometry_feature_t& out)
{
    out      = {};
    out.kind = source.kind == ovspatialquery::detail::GeometryFeatureKind::triangle
                   ? OVSPATIALQUERY_GEOMETRY_FEATURE_TRIANGLE
                   : OVSPATIALQUERY_GEOMETRY_FEATURE_NONE;
    if (out.kind == OVSPATIALQUERY_GEOMETRY_FEATURE_TRIANGLE)
    {
        out.data.triangle.source_face_index = source.triangle.sourceFace;
        std::copy(source.triangle.sourceCorners.begin(), source.triangle.sourceCorners.end(),
                  out.data.triangle.source_face_corner_indices);
        std::copy(source.triangle.barycentrics.begin(), source.triangle.barycentrics.end(),
                  out.data.triangle.barycentrics);
    }
}

ovspatialquery_raycast_hit_t apiRaycastHit(const ovspatialquery::detail::RaycastSemanticHit& source)
{
    ovspatialquery_raycast_hit_t out{};
    out.truth = source.truth == ovspatialquery::detail::HitTruth::exactSurface ? OVSPATIALQUERY_HIT_EXACT_SURFACE
                                                                               : OVSPATIALQUERY_HIT_BOUNDS_CANDIDATE;
    if (out.truth == OVSPATIALQUERY_HIT_BOUNDS_CANDIDATE)
    {
        apiIdentity(source.boundsCandidate.identity, out.data.bounds_candidate.identity);
        std::copy(source.boundsCandidate.boundsPosition.begin(), source.boundsCandidate.boundsPosition.end(),
                  out.data.bounds_candidate.bounds_position);
        std::copy(source.boundsCandidate.boundsNormal.begin(), source.boundsCandidate.boundsNormal.end(),
                  out.data.bounds_candidate.bounds_normal);
        out.data.bounds_candidate.distance = source.boundsCandidate.distance;
    }
    else
    {
        apiIdentity(source.exact.identity, out.data.exact.identity);
        std::copy(source.exact.position.begin(), source.exact.position.end(), out.data.exact.position);
        std::copy(source.exact.normal.begin(), source.exact.normal.end(), out.data.exact.normal);
        out.data.exact.distance = source.exact.distance;
        apiFeature(source.exact.feature, out.data.exact.feature);
    }
    return out;
}

ovspatialquery_sphere_sweep_hit_t apiSphereSweepHit(const ovspatialquery::detail::SphereSweepSemanticHit& source)
{
    ovspatialquery_sphere_sweep_hit_t out{};
    out.truth = source.truth == ovspatialquery::detail::HitTruth::exactSurface ? OVSPATIALQUERY_HIT_EXACT_SURFACE
                                                                               : OVSPATIALQUERY_HIT_BOUNDS_CANDIDATE;
    if (out.truth == OVSPATIALQUERY_HIT_BOUNDS_CANDIDATE)
    {
        apiIdentity(source.boundsCandidate.identity, out.data.bounds_candidate.identity);
        std::copy(source.boundsCandidate.queryPosition.begin(), source.boundsCandidate.queryPosition.end(),
                  out.data.bounds_candidate.query_position);
        std::copy(source.boundsCandidate.boundsPosition.begin(), source.boundsCandidate.boundsPosition.end(),
                  out.data.bounds_candidate.bounds_position);
        std::copy(source.boundsCandidate.boundsNormal.begin(), source.boundsCandidate.boundsNormal.end(),
                  out.data.bounds_candidate.bounds_normal);
        out.data.bounds_candidate.distance = source.boundsCandidate.distance;
    }
    else
    {
        apiIdentity(source.exact.identity, out.data.exact.identity);
        std::copy(source.exact.queryPosition.begin(), source.exact.queryPosition.end(), out.data.exact.query_position);
        std::copy(source.exact.objectPosition.begin(), source.exact.objectPosition.end(),
                  out.data.exact.object_position);
        std::copy(source.exact.normal.begin(), source.exact.normal.end(), out.data.exact.normal);
        out.data.exact.distance = source.exact.distance;
        apiFeature(source.exact.feature, out.data.exact.feature);
    }
    return out;
}

ovspatialquery_overlap_hit_t apiOverlapHit(const ovspatialquery::detail::OverlapSemanticHit& source)
{
    ovspatialquery_overlap_hit_t out{};
    out.truth = source.truth == ovspatialquery::detail::HitTruth::exactSurface ? OVSPATIALQUERY_HIT_EXACT_SURFACE
                                                                               : OVSPATIALQUERY_HIT_BOUNDS_CANDIDATE;
    apiIdentity(source.identity, out.identity);
    return out;
}

// Fills the target and the plan's precision, tolerance and ledger of one
// synchronous query.
template <typename Plan>
ovspatialquery_api_status_t apiCommonRequest(const InstanceReference&               instance,
                                             const ovspatialquery_query_common_t&   source,
                                             ovspatialquery::detail::TargetRequest& outTarget, Plan& out,
                                             bool allowSingle)
{
    // A reserved per-query input or an unknown option belongs to a later version (ADR-0062, ADR-0065).
    if (source.reserved[0] != nullptr || source.reserved[1] != nullptr ||
        (source.query_flags & ~static_cast<uint32_t>(OVSPATIALQUERY_QUERY_MESH_BOTH_SIDES)) != 0u)
    {
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    if (static_cast<uint32_t>(source.precision) > OVSPATIALQUERY_QUERY_EXACT_SINGLE ||
        !inInputDomain(source.contact_tolerance) || source.contact_tolerance < 0.0)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    if (source.precision == OVSPATIALQUERY_QUERY_EXACT_SINGLE && !allowSingle)
    {
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    // CPU evaluators run EXACT_SINGLE through their EXACT path.
    const ovspatialquery_api_status_t status = apiTargetRequest(instance, source.target, outTarget);
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        return status;
    }
    out.precision        = source.precision == 0u ? ovspatialquery::detail::QueryPrecision::boundsCandidate
                                                  : ovspatialquery::detail::QueryPrecision::exact;
    out.contactTolerance = source.contact_tolerance;
    out.hostLedger       = &instance->hostLedger;
    return OVSPATIALQUERY_API_SUCCESS;
}

bool apiQuery(const ovspatialquery_raycast_desc_t& desc, ovspatialquery::detail::RaycastQuery& out)
{
    std::copy(std::begin(desc.ray.origin), std::end(desc.ray.origin), out.origin.begin());
    std::copy(std::begin(desc.ray.unit_direction), std::end(desc.ray.unit_direction), out.unitDirection.begin());
    out.minimumDistance = desc.ray.min_distance;
    out.maximumDistance = desc.ray.max_distance;
    out.bothSides       = (desc.common.query_flags & OVSPATIALQUERY_QUERY_MESH_BOTH_SIDES) != 0u;
    return true;
}

bool apiQuery(const ovspatialquery_sphere_sweep_desc_t& desc, ovspatialquery::detail::SphereSweepQuery& out)
{
    std::copy(std::begin(desc.sweep.center), std::end(desc.sweep.center), out.center.begin());
    std::copy(std::begin(desc.sweep.unit_direction), std::end(desc.sweep.unit_direction), out.unitDirection.begin());
    out.radius          = desc.sweep.radius;
    out.minimumDistance = desc.sweep.min_distance;
    out.maximumDistance = desc.sweep.max_distance;
    out.bothSides       = (desc.common.query_flags & OVSPATIALQUERY_QUERY_MESH_BOTH_SIDES) != 0u;
    return true;
}

bool apiQuery(const ovspatialquery_box_sweep_desc_t& desc, ovspatialquery::detail::SphereSweepQuery& out)
{
    std::copy(std::begin(desc.sweep.center), std::end(desc.sweep.center), out.center.begin());
    std::copy(std::begin(desc.sweep.unit_direction), std::end(desc.sweep.unit_direction), out.unitDirection.begin());
    out.box = true;
    std::copy(std::begin(desc.sweep.half_extents), std::end(desc.sweep.half_extents), out.halfExtents.begin());
    std::copy(std::begin(desc.sweep.rotation_xyzw), std::end(desc.sweep.rotation_xyzw), out.rotationXyzw.begin());
    out.minimumDistance = desc.sweep.min_distance;
    out.maximumDistance = desc.sweep.max_distance;
    out.bothSides       = (desc.common.query_flags & OVSPATIALQUERY_QUERY_MESH_BOTH_SIDES) != 0u;
    return true;
}

bool apiQuery(const ovspatialquery_overlap_desc_t& desc, ovspatialquery::detail::OverlapQuery& out)
{
    out = {};
    if (desc.shape.kind != OVSPATIALQUERY_OVERLAP_SPHERE && desc.shape.kind != OVSPATIALQUERY_OVERLAP_BOX &&
        desc.shape.kind != OVSPATIALQUERY_OVERLAP_AABB)
    {
        return false;
    }
    out.geometry = desc.shape;
    return true;
}

// The hit mode and the hits a synchronous query may keep.
template <typename Plan>
void apiSyncMode(Plan& plan, ovspatialquery::detail::HitMode mode, uint64_t capacity)
{
    plan.hitMode             = mode;
    plan.maximumHitsPerQuery = capacity;
}

// Options a synchronous descriptor adds to its one-row plan; only raycasts have one.
template <typename Plan, typename Desc>
void apiSyncOptions(Plan&, const Desc&)
{}

void apiSyncOptions(ovspatialquery::detail::RaycastBatchPlan& plan, const ovspatialquery_raycast_desc_t& desc)
{
    plan.allMeshHits = desc.mesh_hits == OVSPATIALQUERY_MESH_HITS_ALL;
}

// Serves one synchronous query entry point once its outputs are checked:
// resolves the instance, takes the lifecycle lock, fills a one-row plan, runs
// the row and passes its source ordinal and hits to write, which runs only on
// success. Only raycasts take EXACT_SINGLE.
template <typename Plan, typename Row, typename Hit, typename Desc, typename Write>
ovspatialquery_result_t apiRunSync(ovspatialquery_instance_handle_t instanceHandle, const Desc& desc,
                                   ovspatialquery::detail::HitMode mode, uint64_t capacity, Write write)
{
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE);
    }
    ovspatialquery::detail::TargetRequest target{};
    Plan                                  plan{};
    Row                                   row{};
    std::lock_guard<std::mutex>           lifecycle(instance->lifecycleMutex);
    ovspatialquery_api_status_t           status = apiCommonRequest(
        instance, desc.common, target, plan, std::is_same<Row, ovspatialquery::detail::RaycastBatchRow>::value);
    if (status == OVSPATIALQUERY_API_SUCCESS && !apiQuery(desc, row.query))
    {
        status = OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    if (status == OVSPATIALQUERY_API_SUCCESS)
    {
        apiSyncMode(plan, mode, capacity);
        apiSyncOptions(plan, desc);
        row.queryBits = desc.common.query_bits;
        ovspatialquery::detail::ResolvedTarget resolved;
        std::vector<Hit>                       hits;
        status = ovspatialquery::detail::executeFamilyRowCpu(
            ovspatialquery::detail::TargetResolver(instance->populations), target, plan, row, resolved, hits);
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            write(resolved.ordinal, hits);
        }
    }
    return apiResult(status);
}

template <typename Plan, typename Row, typename Hit, typename Desc, typename Projected, typename ProjectHit>
ovspatialquery_result_t apiSyncSingle(ovspatialquery_instance_handle_t instanceHandle, const Desc& desc,
                                      ovspatialquery::detail::HitMode mode, Projected* out, ProjectHit projectHit)
{
    return apiRunSync<Plan, Row, Hit>(instanceHandle, desc, mode, 1u,
                                      [&](ovspatialquery_ordinal_t ordinal, const std::vector<Hit>& hits)
                                      {
                                          Projected projected{};
                                          projected.source_ordinal = ordinal;
                                          projected.has_hit        = hits.empty() ? 0u : 1u;
                                          if (!hits.empty())
                                          {
                                              projected.hit = projectHit(hits.front());
                                          }
                                          *out = projected;
                                      });
}

template <typename Plan, typename Row, typename Hit, typename Desc, typename Buffer, typename ProjectHit>
ovspatialquery_result_t apiSyncMultiple(ovspatialquery_instance_handle_t instanceHandle, const Desc& desc,
                                        Buffer* inoutHits, ovspatialquery_multiple_result_t* out, ProjectHit projectHit)
{
    return apiRunSync<Plan, Row, Hit>(
        instanceHandle, desc, ovspatialquery::detail::HitMode::multiple, inoutHits->capacity,
        [&](ovspatialquery_ordinal_t ordinal, const std::vector<Hit>& hits)
        {
            const uint64_t stored = std::min(static_cast<uint64_t>(hits.size()), inoutHits->capacity);
            for (size_t index = 0u; index < static_cast<size_t>(stored); ++index)
            {
                apiWriteElement(inoutHits->hits, index, projectHit(hits[index]));
            }
            *out = ovspatialquery_multiple_result_t{ordinal, stored,
                                                    stored < static_cast<uint64_t>(hits.size()) ? 1u : 0u};
        });
}

} // namespace

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_destroy_population(
    ovspatialquery_instance_handle_t instance_handle, ovspatialquery_population_handle_t population)
{
    InstanceReference instance = instanceState(instance_handle);
    if (instance == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE, "instance handle is invalid");
    }
    const ovspatialquery_api_status_t status = ovspatialquery_release_population(instance_handle, population);
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        return apiResult(status);
    }
    std::shared_ptr<ovspatialquery::detail::HostLease> charge;
    {
        std::lock_guard<std::mutex> lock(instance->budgetMutex);
        const std::unordered_map<ovspatialquery_population_handle_t,
                                 std::shared_ptr<ovspatialquery::detail::HostLease>>::iterator found =
            instance->populationCharges.find(population);
        if (found != instance->populationCharges.end())
        {
            charge = found->second;
            instance->populationCharges.erase(found);
        }
    }
    charge.reset();
    return apiResult(OVSPATIALQUERY_API_SUCCESS);
}

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t
ovspatialquery_destroy_view(ovspatialquery_instance_handle_t instance_handle, ovspatialquery_view_handle_t view)
{
    InstanceReference instance = instanceState(instance_handle);
    if (instance == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE, "instance handle is invalid");
    }
    std::shared_ptr<ovspatialquery::detail::HostLease> charge;
    {
        const std::unique_lock<std::mutex> completionLock = instance->acquirePopulationCompletion();
        const ovspatialquery_api_status_t  status         = instance->handles.invalidateView(view);
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return apiResult(status);
        }
        const auto found = instance->viewCharges.find(view);
        if (found != instance->viewCharges.end())
        {
            charge = found->second;
            instance->viewCharges.erase(found);
        }
    }
    charge.reset();
    return apiResult(OVSPATIALQUERY_API_SUCCESS);
}

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_get_population_info(
    ovspatialquery_instance_handle_t instanceHandle, ovspatialquery_population_handle_t population,
    ovspatialquery_population_info_t* outInfo)
{
    if (outInfo == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE);
    }
    ovspatialquery::detail::PopulationKey key{};
    if (instance->handles.resolvePopulation(population, key) != OVSPATIALQUERY_API_SUCCESS)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE);
    }
    bool                              ready         = false;
    bool                              pending       = false;
    ovspatialquery_ordinal_t          ordinal       = 0;
    uint64_t                          objectCount   = 0u;
    uint32_t                          retainedCount = 0u;
    const ovspatialquery_api_status_t status =
        instance->populations.getInfoState(key, ready, ordinal, objectCount, retainedCount, pending);
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        return apiResult(status);
    }
    *outInfo =
        ovspatialquery_population_info_t{ready ? 1u : 0u, ordinal, objectCount, retainedCount, pending ? 1u : 0u};
    return apiResult(OVSPATIALQUERY_API_SUCCESS);
}

namespace ovspatialquery
{
namespace detail
{

ovspatialquery_api_status_t fetchDroppedTriangles(ovspatialquery_instance_handle_t instanceHandle,
                                                  ovspatialquery_op_id_t opId, DLTensor& outObjects,
                                                  DLTensor& outCounts) noexcept
{
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    uint32_t       slotIndex  = 0u;
    uint32_t       generation = 0u;
    OperationSlot* slot       = resolveOperation(instance, opId, slotIndex, generation);
    if (slot == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    std::lock_guard<std::mutex> lock(slot->mutex);
    if (slot->kind != OperationKind::populationUpdate || !slot->inUse || slot->generation != generation)
    {
        return OVSPATIALQUERY_API_INVALID_HANDLE;
    }
    if (!slot->terminal)
    {
        return OVSPATIALQUERY_API_NOT_READY;
    }
    slot->terminalObserved = true;
    if (slot->executionStatus != OVSPATIALQUERY_API_SUCCESS)
    {
        return OVSPATIALQUERY_API_OPERATION_FAILED;
    }
    PopulationUpdateTask& update = static_cast<PopulationUpdateTask&>(*slot->task);
    update.droppedTriangleShape  = static_cast<int64_t>(update.droppedTriangles.objects.size());
    outObjects                   = {update.droppedTriangles.objects.data(), {kDLCPU, 0}, 1, {kDLUInt, 64, 1},
                                    &update.droppedTriangleShape,           nullptr,     0u};
    outCounts                    = {update.droppedTriangles.counts.data(), {kDLCPU, 0}, 1, {kDLUInt, 64, 1},
                                    &update.droppedTriangleShape,          nullptr,     0u};
    return OVSPATIALQUERY_API_SUCCESS;
}

} // namespace detail
} // namespace ovspatialquery

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t
ovspatialquery_fetch_population_update(ovspatialquery_instance_handle_t instanceHandle, ovspatialquery_op_id_t opId,
                                       ovspatialquery_population_update_result_t* outResult)
{
    if (outResult == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE);
    }
    uint32_t       slotIndex  = 0u;
    uint32_t       generation = 0u;
    OperationSlot* slot       = resolveOperation(instance, opId, slotIndex, generation);
    if (slot == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE);
    }
    std::lock_guard<std::mutex> lock(slot->mutex);
    // Rechecked under the slot lock: a destroy_op since the resolve has
    // released the update this reads.
    if (slot->kind != OperationKind::populationUpdate || !slot->inUse || slot->generation != generation)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE);
    }
    if (!slot->terminal)
    {
        return apiResult(OVSPATIALQUERY_API_NOT_READY);
    }
    slot->terminalObserved = true;
    if (slot->executionStatus != OVSPATIALQUERY_API_SUCCESS)
    {
        return apiResult(OVSPATIALQUERY_API_OPERATION_FAILED);
    }
    *outResult = static_cast<const PopulationUpdateTask&>(*slot->task).result;
    return apiResult(OVSPATIALQUERY_API_SUCCESS);
}

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_destroy_geometry_definition(
    ovspatialquery_instance_handle_t instanceHandle, ovspatialquery_geometry_ref_t geometry)
{
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE);
    }
    if (geometry.type < kGeometryTypeFirst || geometry.type > kGeometryTypeLast || geometry.value == 0u)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    if (geometry.type == OVSPATIALQUERY_GEOMETRY_USER)
    {
        return apiResult(OVSPATIALQUERY_API_NOT_SUPPORTED); // reserved: no USER definition can exist
    }
    return apiResult(instance->explicitPreparationRegistry->destroyDefinition(
        {static_cast<ovspatialquery::detail::ExactGeometryFamily>(geometry.type - kGeometryTypeFirstBuiltin),
         geometry.value}));
}

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_get_dropped_triangle_count(
    ovspatialquery_instance_handle_t instanceHandle, uint64_t definitionValue, uint64_t* outCount)
{
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE);
    }
    uint64_t                          count = 0u;
    const ovspatialquery_api_status_t status =
        outCount == nullptr ? OVSPATIALQUERY_API_INVALID_ARGUMENT
                            : instance->explicitPreparationRegistry->droppedTriangleCount(definitionValue, count);
    if (status == OVSPATIALQUERY_API_SUCCESS)
    {
        *outCount = count;
    }
    return apiResult(status, status == OVSPATIALQUERY_API_INVALID_ARGUMENT && outCount != nullptr
                                 ? "the value names no registered explicit triangle-mesh definition"
                                 : nullptr);
}

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_raycast_closest_sync(
    ovspatialquery_instance_handle_t instanceHandle, const ovspatialquery_raycast_desc_t* desc,
    ovspatialquery_raycast_single_result_t* outResult)
{
    if (desc == nullptr || outResult == nullptr ||
        static_cast<uint32_t>(desc->mesh_hits) > OVSPATIALQUERY_MESH_HITS_NEAREST)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    return apiSyncSingle<ovspatialquery::detail::RaycastBatchPlan, ovspatialquery::detail::RaycastBatchRow,
                         ovspatialquery::detail::RaycastSemanticHit>(
        instanceHandle, *desc, ovspatialquery::detail::HitMode::closest, outResult, apiRaycastHit);
}

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_raycast_any_sync(
    ovspatialquery_instance_handle_t instanceHandle, const ovspatialquery_raycast_desc_t* desc,
    ovspatialquery_raycast_single_result_t* outResult)
{
    if (desc == nullptr || outResult == nullptr ||
        static_cast<uint32_t>(desc->mesh_hits) > OVSPATIALQUERY_MESH_HITS_NEAREST)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    return apiSyncSingle<ovspatialquery::detail::RaycastBatchPlan, ovspatialquery::detail::RaycastBatchRow,
                         ovspatialquery::detail::RaycastSemanticHit>(
        instanceHandle, *desc, ovspatialquery::detail::HitMode::any, outResult, apiRaycastHit);
}

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_raycast_multiple_sync(
    ovspatialquery_instance_handle_t instanceHandle, const ovspatialquery_raycast_desc_t* desc,
    ovspatialquery_raycast_hit_buffer_t* inoutHits, ovspatialquery_multiple_result_t* outResult)
{
    if (desc == nullptr || inoutHits == nullptr || outResult == nullptr ||
        (inoutHits->capacity != 0u && inoutHits->hits == nullptr) ||
        static_cast<uint32_t>(desc->mesh_hits) > OVSPATIALQUERY_MESH_HITS_NEAREST)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    if (desc->mesh_hits == OVSPATIALQUERY_MESH_HITS_ALL && desc->common.contact_tolerance != 0.0)
    {
        return apiResult(OVSPATIALQUERY_API_NOT_SUPPORTED,
                         "MESH_HITS_ALL needs a zero contact_tolerance; ask for MESH_HITS_NEAREST");
    }
    return apiSyncMultiple<ovspatialquery::detail::RaycastBatchPlan, ovspatialquery::detail::RaycastBatchRow,
                           ovspatialquery::detail::RaycastSemanticHit>(instanceHandle, *desc, inoutHits, outResult,
                                                                       apiRaycastHit);
}

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_sweep_sphere_closest_sync(
    ovspatialquery_instance_handle_t instanceHandle, const ovspatialquery_sphere_sweep_desc_t* desc,
    ovspatialquery_sphere_sweep_closest_result_t* outResult)
{
    if (desc == nullptr || outResult == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    return apiSyncSingle<ovspatialquery::detail::SphereSweepBatchPlan, ovspatialquery::detail::SphereSweepBatchRow,
                         ovspatialquery::detail::SphereSweepSemanticHit>(
        instanceHandle, *desc, ovspatialquery::detail::HitMode::closest, outResult, apiSphereSweepHit);
}

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_sweep_box_closest_sync(
    ovspatialquery_instance_handle_t instanceHandle, const ovspatialquery_box_sweep_desc_t* desc,
    ovspatialquery_box_sweep_closest_result_t* outResult)
{
    if (desc == nullptr || outResult == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    return apiSyncSingle<ovspatialquery::detail::SphereSweepBatchPlan, ovspatialquery::detail::SphereSweepBatchRow,
                         ovspatialquery::detail::SphereSweepSemanticHit>(
        instanceHandle, *desc, ovspatialquery::detail::HitMode::closest, outResult, apiSphereSweepHit);
}

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_sweep_sphere_any_sync(
    ovspatialquery_instance_handle_t instanceHandle, const ovspatialquery_sphere_sweep_desc_t* desc,
    ovspatialquery_sphere_sweep_closest_result_t* outResult)
{
    if (desc == nullptr || outResult == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    return apiSyncSingle<ovspatialquery::detail::SphereSweepBatchPlan, ovspatialquery::detail::SphereSweepBatchRow,
                         ovspatialquery::detail::SphereSweepSemanticHit>(
        instanceHandle, *desc, ovspatialquery::detail::HitMode::any, outResult, apiSphereSweepHit);
}

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_sweep_sphere_multiple_sync(
    ovspatialquery_instance_handle_t instanceHandle, const ovspatialquery_sphere_sweep_desc_t* desc,
    ovspatialquery_sphere_sweep_hit_buffer_t* inoutHits, ovspatialquery_multiple_result_t* outResult)
{
    if (desc == nullptr || inoutHits == nullptr || outResult == nullptr ||
        (inoutHits->capacity != 0u && inoutHits->hits == nullptr))
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    return apiSyncMultiple<ovspatialquery::detail::SphereSweepBatchPlan, ovspatialquery::detail::SphereSweepBatchRow,
                           ovspatialquery::detail::SphereSweepSemanticHit>(instanceHandle, *desc, inoutHits, outResult,
                                                                           apiSphereSweepHit);
}

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_sweep_box_any_sync(
    ovspatialquery_instance_handle_t instanceHandle, const ovspatialquery_box_sweep_desc_t* desc,
    ovspatialquery_box_sweep_closest_result_t* outResult)
{
    if (desc == nullptr || outResult == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    return apiSyncSingle<ovspatialquery::detail::SphereSweepBatchPlan, ovspatialquery::detail::SphereSweepBatchRow,
                         ovspatialquery::detail::SphereSweepSemanticHit>(
        instanceHandle, *desc, ovspatialquery::detail::HitMode::any, outResult, apiSphereSweepHit);
}

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_sweep_box_multiple_sync(
    ovspatialquery_instance_handle_t instanceHandle, const ovspatialquery_box_sweep_desc_t* desc,
    ovspatialquery_box_sweep_hit_buffer_t* inoutHits, ovspatialquery_multiple_result_t* outResult)
{
    if (desc == nullptr || inoutHits == nullptr || outResult == nullptr ||
        (inoutHits->capacity != 0u && inoutHits->hits == nullptr))
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    return apiSyncMultiple<ovspatialquery::detail::SphereSweepBatchPlan, ovspatialquery::detail::SphereSweepBatchRow,
                           ovspatialquery::detail::SphereSweepSemanticHit>(instanceHandle, *desc, inoutHits, outResult,
                                                                           apiSphereSweepHit);
}

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_overlap_any_sync(
    ovspatialquery_instance_handle_t instanceHandle, const ovspatialquery_overlap_desc_t* desc,
    ovspatialquery_overlap_any_result_t* outResult)
{
    if (desc == nullptr || outResult == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    return apiSyncSingle<ovspatialquery::detail::OverlapBatchPlan, ovspatialquery::detail::OverlapBatchRow,
                         ovspatialquery::detail::OverlapSemanticHit>(
        instanceHandle, *desc, ovspatialquery::detail::HitMode::any, outResult, apiOverlapHit);
}

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_overlap_multiple_sync(
    ovspatialquery_instance_handle_t instanceHandle, const ovspatialquery_overlap_desc_t* desc,
    ovspatialquery_overlap_hit_buffer_t* inoutHits, ovspatialquery_multiple_result_t* outResult)
{
    if (desc == nullptr || inoutHits == nullptr || outResult == nullptr ||
        (inoutHits->capacity != 0u && inoutHits->hits == nullptr))
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    return apiSyncMultiple<ovspatialquery::detail::OverlapBatchPlan, ovspatialquery::detail::OverlapBatchRow,
                           ovspatialquery::detail::OverlapSemanticHit>(instanceHandle, *desc, inoutHits, outResult,
                                                                       apiOverlapHit);
}

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_query_pairs_sync(
    ovspatialquery_instance_handle_t instanceHandle, const ovspatialquery_pair_snapshot_desc_t* desc,
    ovspatialquery_pair_buffer_t* inoutPairs, ovspatialquery_pair_snapshot_result_t* outResult)
{
    if (desc == nullptr || inoutPairs == nullptr || outResult == nullptr ||
        (inoutPairs->capacity != 0u && inoutPairs->pairs == nullptr) || !inInputDomain(desc->contact_tolerance) ||
        desc->contact_tolerance < 0.0)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE);
    }
    std::lock_guard<std::mutex>                           lifecycle(instance->lifecycleMutex);
    std::array<ovspatialquery::detail::TargetRequest, 2u> requests{};
    ovspatialquery_api_status_t                           status = apiTargetRequest(instance, desc->first, requests[0]);
    if (status == OVSPATIALQUERY_API_SUCCESS)
    {
        status = apiTargetRequest(instance, desc->second, requests[1]);
    }
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        return apiResult(status);
    }
    const ovspatialquery::detail::TargetResolver           resolver(instance->populations);
    std::array<ovspatialquery::detail::ResolvedTarget, 2u> resolved{};
    const ovspatialquery::detail::TargetResolveOutcome     resolveOutcome = resolver.resolvePair(requests, resolved);
    if (resolveOutcome != ovspatialquery::detail::TargetResolveOutcome::success)
    {
        using Outcome = ovspatialquery::detail::TargetResolveOutcome;
        switch (resolveOutcome)
        {
        case Outcome::invalid: status = OVSPATIALQUERY_API_INVALID_ARGUMENT; break;
        case Outcome::notReady: status = OVSPATIALQUERY_API_NOT_READY; break;
        case Outcome::ordinalNotRetained: status = OVSPATIALQUERY_API_ORDINAL_NOT_RETAINED; break;
        case Outcome::ordinalMismatch: status = OVSPATIALQUERY_API_ORDINAL_MISMATCH; break;
        case Outcome::resourceExhausted: status = OVSPATIALQUERY_API_OUT_OF_MEMORY; break;
        default: status = OVSPATIALQUERY_API_INTERNAL; break;
        }
        return apiResult(status);
    }
    for (const ovspatialquery::detail::ResolvedTarget& endpoint : resolved)
    {
        // A readable device version is copied to the host on first read, charged
        // here; plain ON_DEVICE has no host objects for pairs (ADR-0024).
        status = instance->populations.materializeHostCopy(endpoint.generation, &instance->hostLedger);
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return apiResult(status);
        }
    }
    const ovspatialquery::detail::PairEndpointRelation relation = ovspatialquery::detail::pairEndpointRelation(
        resolved[0].population == resolved[1].population, desc->first.target, desc->first_query_bits,
        desc->second.target, desc->second_query_bits);
    uint64_t pairPeak = 0u;
    if (!addHostCharge(inoutPairs->capacity, sizeof(ovspatialquery_pair_t), pairPeak) ||
        !addHostCharge(ovspatialquery::detail::pairTraversalScratchBytes(resolved[0], desc->first_query_bits,
                                                                         resolved[1], desc->second_query_bits, relation,
                                                                         desc->contact_tolerance),
                       1u, pairPeak))
    {
        return apiResult(OVSPATIALQUERY_API_OUT_OF_MEMORY);
    }
    ovspatialquery::detail::HostLease       pairLease;
    ovspatialquery::detail::HostTransaction pairTransaction(instance->hostLedger);
    if (!pairTransaction.reserve(pairPeak) || !pairTransaction.commit(pairLease))
    {
        return apiResult(OVSPATIALQUERY_API_HOST_BUDGET);
    }
    ovspatialquery::detail::PairMaterializationRequest request{};
    request.first              = {std::move(resolved[0]), desc->first_query_bits};
    request.second             = {std::move(resolved[1]), desc->second_query_bits};
    request.relation           = relation;
    request.contactTolerance   = desc->contact_tolerance;
    request.activePairCapacity = inoutPairs->capacity;
    std::vector<ovspatialquery_pair_t>                       pairs;
    bool                                                     truncated = false;
    const ovspatialquery::detail::PairMaterializationOutcome pairOutcome =
        ovspatialquery::detail::materializeBoundedPairSnapshot(request, pairs, truncated);
    if (pairOutcome != ovspatialquery::detail::PairMaterializationOutcome::success)
    {
        return apiResult(pairOutcome == ovspatialquery::detail::PairMaterializationOutcome::resourceExhausted
                             ? OVSPATIALQUERY_API_OUT_OF_MEMORY
                             : OVSPATIALQUERY_API_INTERNAL);
    }
    if (!pairs.empty())
    {
        std::memcpy(inoutPairs->pairs, pairs.data(), pairs.size() * sizeof(ovspatialquery_pair_t));
    }
    *outResult = ovspatialquery_pair_snapshot_result_t{request.first.target.ordinal, request.second.target.ordinal,
                                                       static_cast<uint64_t>(pairs.size()), truncated ? 1u : 0u};
    return apiResult(OVSPATIALQUERY_API_SUCCESS);
}
