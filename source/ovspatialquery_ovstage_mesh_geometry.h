/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_OVSTAGE_MESH_GEOMETRY_H
#define OVSPATIALQUERY_OVSTAGE_MESH_GEOMETRY_H

#include "ovspatialquery_ovstage_filter.h"

#include "ovspatialquery_backend.h"
#include <ovspatialquery_instance_source.h>

#include <ovstage/ovstage_api/ovstage_api.h>
#include <ovx/path_dictionary/path_dictionary.h>

#include <array>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ovspatialquery
{
namespace detail
{

/* Authored source faces are stored as uint32_t; a mesh with a face beyond that is NOT_SUPPORTED. */
inline bool ovstageSourceFaceFitsUint32(uint64_t sourceFace) noexcept { return sourceFace <= UINT32_MAX; }

#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
void rejectNextUnrepresentableOvstageSourceFaceFixture() noexcept;
#endif

struct OvstageTriangleRow
{
    std::vector<float>      points;
    std::vector<int32_t>    faceVertexCounts;
    std::vector<int32_t>    faceVertexIndices;
    std::vector<int32_t>    holeIndices;
    std::array<double, 16u> worldFromLocal{};
    ovx_token_t             orientation = OVX_INVALID_TOKEN;
    /* physxConvexHullCollision:hullVertexLimit, read for a convex; -1 when unauthored. */
    int64_t hullVertexLimit      = -1;
    bool    hasPoints            = false;
    bool    hasFaceVertexCounts  = false;
    bool    hasFaceVertexIndices = false;
    bool    hasOrientation       = false;
    bool    hasHoleIndices       = false;
    bool    doubleSided          = false;
};

struct AcquiredTypedMeshGeometry
{
    TriangleGeometrySource               triangles;
    std::vector<ConvexGeometrySource>    convexes;
    std::vector<GeometryDefinitionValue> triangleDefinitions;
    std::vector<GeometryDefinitionValue> convexDefinitions;
    std::vector<size_t>                  triangleObjectRows;
    std::vector<size_t>                  convexObjectRows;
};

bool tensorBytes(const DLTensor& tensor, uint8_t expectedCode, uint8_t expectedBits, uint16_t alternateLanes,
                 const uint8_t*& outBytes, size_t& outByteCount);

template <typename Consumer>
ovspatialquery_api_status_t consumeAttributeGroup(const ReadGroup& group, ovx_token_t expectedAttribute,
                                                  ovspatialquery_ordinal_t     targetOrdinal,
                                                  path_dictionary_instance_t*  dictionary,
                                                  ovstage_attribute_semantic_t expectedSemantic, uint8_t expectedCode,
                                                  uint8_t expectedBits, uint16_t alternateLanes, bool allowNoneSemantic,
                                                  Consumer consume, bool skipUnreadableRows = false)
{
    if (group.attribute != expectedAttribute || group.is_delete ||
        (group.semantic != expectedSemantic && !(allowNoneSemantic && group.semantic == OVSTAGE_SEMANTIC_NONE)) ||
        group.meta.attribute_write_floor_ordinal != static_cast<ovstage_ordinal_t>(targetOrdinal) ||
        group.ordinal > static_cast<ovstage_ordinal_t>(targetOrdinal) || group.data.tensors == nullptr ||
        group.data.tensor_count == 0u || (group.data.index_map != nullptr && group.data.mask != nullptr) ||
        group.data.cuda_sync.stream != 0u || group.data.cuda_sync.wait_event != 0u ||
        ((group.data.index_map != nullptr || group.data.mask != nullptr) && group.data.count != group.prims.count))
    {
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    std::vector<ovx_primpath_t>       paths;
    const ovspatialquery_api_status_t pathStatus = resolveGroupPaths(group, dictionary, paths);
    if (pathStatus != OVSPATIALQUERY_API_SUCCESS)
    {
        return pathStatus;
    }
    const bool      perRow           = group.is_array || group.data.tensor_count > 1u;
    const DLTensor* fixedTensor      = perRow ? nullptr : &group.data.tensors[0];
    const uint8_t*  fixedBytes       = nullptr;
    size_t          fixedByteCount   = 0u;
    size_t          fixedStorageRows = 0u;
    if (!perRow)
    {
        if (!tensorBytes(*fixedTensor, expectedCode, expectedBits, alternateLanes, fixedBytes, fixedByteCount))
        {
            return OVSPATIALQUERY_API_NOT_SUPPORTED;
        }
        fixedStorageRows = group.data.index_map == nullptr ? static_cast<size_t>(group.prims.count)
                                                           : static_cast<size_t>(fixedTensor->shape[0]);
        if (fixedStorageRows == 0u || fixedByteCount % fixedStorageRows != 0u)
        {
            return OVSPATIALQUERY_API_NOT_SUPPORTED;
        }
    }
    for (uint32_t logicalRow = 0u; logicalRow < group.prims.count; ++logicalRow)
    {
        if (group.data.mask != nullptr &&
            (group.data.mask[logicalRow / 64u] & (UINT64_C(1) << (logicalRow % 64u))) == 0u)
        {
            continue;
        }
        if (paths[logicalRow] == OVX_INVALID_PRIMPATH)
        {
            return OVSPATIALQUERY_API_INVALID_STATE;
        }
        const size_t   storageRow = group.data.index_map == nullptr ? logicalRow : group.data.index_map[logicalRow];
        const uint8_t* bytes      = nullptr;
        size_t         byteCount  = 0u;
        if (perRow)
        {
            if (storageRow >= group.data.tensor_count || !tensorBytes(group.data.tensors[storageRow], expectedCode,
                                                                      expectedBits, alternateLanes, bytes, byteCount))
            {
                if (skipUnreadableRows && storageRow < group.data.tensor_count)
                {
                    continue;
                }
                return OVSPATIALQUERY_API_NOT_SUPPORTED;
            }
        }
        else
        {
            if (storageRow >= fixedStorageRows)
            {
                return OVSPATIALQUERY_API_NOT_SUPPORTED;
            }
            byteCount = fixedByteCount / fixedStorageRows;
            bytes     = fixedBytes + storageRow * byteCount;
        }
        const ovspatialquery_api_status_t consumeStatus = consume(handleValue(paths[logicalRow]), bytes, byteCount);
        if (consumeStatus != OVSPATIALQUERY_API_SUCCESS)
        {
            return consumeStatus;
        }
    }
    return OVSPATIALQUERY_API_SUCCESS;
}

/* Reads one attribute through the query and hands every present row of every group
 * to `consume`; `afterGroup` runs once a group's rows have all been consumed, while the
 * group's storage is still pinned, so consumers may hold its pointers until then. An
 * exception from either ends the read with OUT_OF_MEMORY or INTERNAL after the group and
 * the read are released. `skipUnreadable` reads optional data: a row whose array the read
 * does not take is skipped, and so is a group whose layout it does not take (NOT_SUPPORTED),
 * keeping any rows the group handed over before; a read the stage refuses to enqueue for a
 * reason other than memory reads nothing (Ovstage refuses an attribute stored in two types
 * across the queried prims). */
template <typename Consumer, typename AfterGroup>
ovspatialquery_api_status_t readTriangleAttribute(ovstage_instance_t* stage, ovstage_query_handle_t queryHandle,
                                                  ovx_token_t attribute, ovspatialquery_ordinal_t targetOrdinal,
                                                  path_dictionary_instance_t*  dictionary,
                                                  ovstage_attribute_semantic_t expectedSemantic, uint8_t expectedCode,
                                                  uint8_t expectedBits, uint16_t alternateLanes, bool allowNoneSemantic,
                                                  bool optional, Consumer consume, AfterGroup afterGroup,
                                                  bool skipUnreadable = false)
{
    const ovstage_ordinal_range_t  range{0u, static_cast<ovstage_ordinal_t>(targetOrdinal), false};
    ovstage_read_handle_t          readHandle = OVSTAGE_INVALID_READ_HANDLE;
    const ovstage_enqueue_result_t read =
        stageVtable(stage)->read_attributes(stageContext(stage), queryHandle, &attribute, 1u, range, &readHandle);
    if ((read.status == OVSTAGE_ERROR_NOT_FOUND && optional) ||
        (skipUnreadable && read.status != OVSTAGE_OK && read.status != OVSTAGE_ERROR_OUT_OF_MEMORY))
    {
        return OVSPATIALQUERY_API_SUCCESS;
    }
    if (read.status != OVSTAGE_OK)
    {
        return mapStageStatus(read.status);
    }
    ovspatialquery_api_status_t status = waitAndRelease(stage, read.op_index);
    ReadCursor                  cursor(stage);
    while (status == OVSPATIALQUERY_API_SUCCESS)
    {
        ReadGroup                  group{};
        const ovstage_api_status_t fetchStatus = cursor.next(readHandle, OVSTAGE_TIMEOUT_INFINITE, group);
        if (fetchStatus == OVSTAGE_ERROR_END_OF_ITERATION)
        {
            break;
        }
        if (fetchStatus != OVSTAGE_OK)
        {
            status = mapStageStatus(fetchStatus);
            break;
        }
        ovspatialquery_api_status_t groupStatus = OVSPATIALQUERY_API_SUCCESS;
        try
        {
            groupStatus =
                consumeAttributeGroup(group, attribute, targetOrdinal, dictionary, expectedSemantic, expectedCode,
                                      expectedBits, alternateLanes, allowNoneSemantic, consume, skipUnreadable);
            if (groupStatus == OVSPATIALQUERY_API_SUCCESS)
            {
                groupStatus = afterGroup();
            }
        }
        catch (const std::bad_alloc&)
        {
            groupStatus = OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        catch (const std::length_error&)
        {
            groupStatus = OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        catch (...)
        {
            groupStatus = OVSPATIALQUERY_API_INTERNAL;
        }
        const ovstage_api_status_t releaseStatus = releaseReadGroup(stage, group);
        if (skipUnreadable && groupStatus == OVSPATIALQUERY_API_NOT_SUPPORTED)
        {
            groupStatus = OVSPATIALQUERY_API_SUCCESS;
        }
        if (groupStatus == OVSPATIALQUERY_API_SUCCESS)
        {
            groupStatus = mapStageStatus(releaseStatus);
        }
        status = groupStatus;
    }
    const ovspatialquery_api_status_t releaseStatus = releaseRead(stage, readHandle);
    return status == OVSPATIALQUERY_API_SUCCESS ? releaseStatus : status;
}

template <typename Consumer>
ovspatialquery_api_status_t readTriangleAttribute(ovstage_instance_t* stage, ovstage_query_handle_t queryHandle,
                                                  ovx_token_t attribute, ovspatialquery_ordinal_t targetOrdinal,
                                                  path_dictionary_instance_t*  dictionary,
                                                  ovstage_attribute_semantic_t expectedSemantic, uint8_t expectedCode,
                                                  uint8_t expectedBits, uint16_t alternateLanes, bool allowNoneSemantic,
                                                  bool optional, Consumer consume)
{
    return readTriangleAttribute(stage, queryHandle, attribute, targetOrdinal, dictionary, expectedSemantic,
                                 expectedCode, expectedBits, alternateLanes, allowNoneSemantic, optional, consume,
                                 []() { return OVSPATIALQUERY_API_SUCCESS; });
}

/* The adapter header's rule for a mesh or convex placement, applied to every publication,
 * including one that keeps or deforms an already prepared mesh: a finite right-handed rigid
 * pose with positive uniform scale; any other transform is NOT_SUPPORTED. */
ovspatialquery_api_status_t qualifyRigidMeshTransform(const std::array<double, 16u>& worldFromLocal);

/* `worldFromLocal[i]` is the pose the typed read already took for `objects[i]`;
 * `worldMatrixColumnRead` says whether that read also took the authored world matrix
 * column, which is otherwise read here only for its status. A mesh this refuses is noted
 * in `refusal` without its type, which the caller knows. */
ovspatialquery_api_status_t acquireAuthoredMeshGeometry(
    ovstage_instance_t* stage, ovstage_query_handle_t queryHandle, const OwnedFilter& filter,
    ovspatialquery_ordinal_t targetOrdinal, path_dictionary_instance_t* dictionary,
    const std::vector<PopulationObject>& objects, const std::vector<std::array<double, 16u>>& worldFromLocal,
    bool worldMatrixColumnRead, const std::unordered_set<uint64_t>& convexIds, AcquiredTypedMeshGeometry& outGeometry,
    PrimRefusal& refusal);

/* The hull vertex limit Omni Physics cooks a convexHull collider under: its authored
 * physxConvexHullCollision:hullVertexLimit, else PhysxSchema's default 64; a negative value
 * counts as unauthored, and the rest is clamped to 8 to 64 (ADR-0078). */
uint32_t ovstageHullVertexLimit(int64_t authored) noexcept;

/* A mesh whose authored points changed while its faces, indices, orientation,
 * holes and sidedness stayed those of the definition the base
 * generation retains: the new points, copied from the read group so the acquisition
 * need not read them again, and that definition, whose other arrays still hold. */
struct DeformedMeshPoints
{
    std::vector<float>         points;
    const MeshDefinitionValue* authored = nullptr;
};

/* Compares each candidate's authored content in place with the definition the
 * identity registry retains for its base-generation mesh, byte for byte. Identical
 * content goes to `outUnchanged`; with `outDeformed`, content that differs in points
 * only goes there with its new points. Each group's comparisons run one prim per task
 * on the caller and on the instance's idle workers (`filter.preparation.workers`); the
 * filter's workers may be absent. A failed read reports nothing unchanged and no
 * error; a failed comparison task reports its status. */
ovspatialquery_api_status_t findUnchangedAuthoredMeshes(
    ovstage_instance_t* stage, ovstage_query_handle_t queryHandle, const OwnedFilter& filter,
    ovspatialquery_ordinal_t targetOrdinal, path_dictionary_instance_t* dictionary,
    const std::unordered_map<uint64_t, const MeshDefinitionValue*>& candidates,
    std::unordered_set<uint64_t>&                                   outUnchanged,
    std::unordered_map<uint64_t, DeformedMeshPoints>*               outDeformed = nullptr);

ovspatialquery_api_status_t collectAuthoredConvexIds(ovstage_instance_t* stage, ovstage_query_handle_t queryHandle,
                                                     const OwnedFilter& filter, ovspatialquery_ordinal_t targetOrdinal,
                                                     path_dictionary_instance_t*   dictionary,
                                                     std::unordered_set<uint64_t>& outIds);

} // namespace detail
} // namespace ovspatialquery

#endif
