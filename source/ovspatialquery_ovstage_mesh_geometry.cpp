/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_ovstage_mesh_geometry.h"

#include "ovspatialquery_backend.h"
#include "ovspatialquery_float_bounds.h"
#include <ovspatialquery_instance_source.h>

#include <ovstage/ovstage_api/ovstage_api.h>
#include <ovx/path_dictionary/path_dictionary.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <new>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ovspatialquery
{
namespace detail
{

#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
namespace
{

std::atomic<bool> gRejectNextUnrepresentableSourceFace{false};

} // namespace

void rejectNextUnrepresentableOvstageSourceFaceFixture() noexcept
{
    gRejectNextUnrepresentableSourceFace.store(true, std::memory_order_release);
}
#endif

bool tensorBytes(const DLTensor& tensor, uint8_t expectedCode, uint8_t expectedBits, uint16_t alternateLanes,
                 const uint8_t*& outBytes, size_t& outByteCount)
{
    outBytes             = nullptr;
    outByteCount         = 0u;
    const uint16_t lanes = tensor.dtype.lanes == 0u ? 1u : tensor.dtype.lanes;
    if (tensor.device.device_type != kDLCPU || tensor.device.device_id != 0 || tensor.dtype.code != expectedCode ||
        tensor.dtype.bits != expectedBits || (lanes != 1u && lanes != alternateLanes) || tensor.ndim <= 0 ||
        tensor.shape == nullptr || tensor.byte_offset > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
    {
        return false;
    }
    size_t  elementCount   = 1u;
    int64_t expectedStride = 1;
    for (int32_t dimension = tensor.ndim - 1; dimension >= 0; --dimension)
    {
        if (tensor.shape[dimension] < 0 || (tensor.strides != nullptr && tensor.strides[dimension] != expectedStride))
        {
            return false;
        }
        const size_t extent = static_cast<size_t>(tensor.shape[dimension]);
        if (extent != 0u && elementCount > std::numeric_limits<size_t>::max() / extent)
        {
            return false;
        }
        elementCount *= extent;
        if (dimension != 0)
        {
            if (tensor.shape[dimension] != 0 &&
                expectedStride > std::numeric_limits<int64_t>::max() / tensor.shape[dimension])
            {
                return false;
            }
            expectedStride *= tensor.shape[dimension];
        }
    }
    const size_t bytesPerElement = static_cast<size_t>(expectedBits / 8u) * lanes;
    if (elementCount != 0u && bytesPerElement > std::numeric_limits<size_t>::max() / elementCount)
    {
        return false;
    }
    outByteCount = elementCount * bytesPerElement;
    if (outByteCount != 0u && tensor.data == nullptr)
    {
        return false;
    }
    if (tensor.data != nullptr)
    {
        const uintptr_t address = reinterpret_cast<uintptr_t>(tensor.data);
        if (address > std::numeric_limits<uintptr_t>::max() - static_cast<size_t>(tensor.byte_offset))
        {
            return false;
        }
        outBytes = static_cast<const uint8_t*>(tensor.data) + static_cast<size_t>(tensor.byte_offset);
    }
    return true;
}

uint32_t ovstageHullVertexLimit(int64_t authored) noexcept
{
    if (authored < 0)
        return kDefaultHullVertexLimit;
    return static_cast<uint32_t>(std::clamp<int64_t>(authored, kMinimumHullVertexLimit, kMaximumHullVertexLimit));
}

ovspatialquery_api_status_t qualifyRigidMeshTransform(const std::array<double, 16u>& worldFromLocal)
{
    QualifiedPrimitiveTransform       transform{};
    const ovspatialquery_api_status_t status =
        qualifyPrimitiveTransform(worldFromLocal, PrimitiveScaleRule::uniform, transform);
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        return status;
    }
    const double determinant =
        transform.axes[0][0] *
            (transform.axes[1][1] * transform.axes[2][2] - transform.axes[1][2] * transform.axes[2][1]) -
        transform.axes[1][0] *
            (transform.axes[0][1] * transform.axes[2][2] - transform.axes[0][2] * transform.axes[2][1]) +
        transform.axes[2][0] *
            (transform.axes[0][1] * transform.axes[1][2] - transform.axes[0][2] * transform.axes[1][1]);
    if (!std::isfinite(determinant) || determinant <= 0.0)
    {
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    const bool inDomain = inScaleDomain(transform.scales[0]) && inInputDomain(transform.center[0]) &&
                          inInputDomain(transform.center[1]) && inInputDomain(transform.center[2]);
    return inDomain ? OVSPATIALQUERY_API_SUCCESS : OVSPATIALQUERY_API_INVALID_ARGUMENT;
}

ovspatialquery_api_status_t acquireAuthoredMeshGeometry(
    ovstage_instance_t* stage, ovstage_query_handle_t queryHandle, const OwnedFilter& filter,
    ovspatialquery_ordinal_t targetOrdinal, path_dictionary_instance_t* dictionary,
    const std::vector<ovspatialquery::detail::PopulationObject>& objects,
    const std::vector<std::array<double, 16u>>& worldFromLocal, bool worldMatrixColumnRead,
    const std::unordered_set<uint64_t>& convexIds, AcquiredTypedMeshGeometry& outGeometry, PrimRefusal& refusal)
{
    try
    {
        std::unordered_map<uint64_t, size_t> rowById;
        rowById.reserve(objects.size());
        std::vector<OvstageTriangleRow> rows(objects.size());
        for (size_t row = 0u; row < objects.size(); ++row)
        {
            rowById.emplace(objects[row].id.value, row);
            rows[row].worldFromLocal = worldFromLocal[row];
        }
        const auto findRow = [&](uint64_t id) -> OvstageTriangleRow*
        {
            const std::unordered_map<uint64_t, size_t>::const_iterator found = rowById.find(id);
            return found == rowById.end() ? nullptr : &rows[found->second];
        };
        ovspatialquery_api_status_t status =
            readTriangleAttribute(stage, queryHandle, filter.points, targetOrdinal, dictionary, OVSTAGE_SEMANTIC_POINT,
                                  kDLFloat, 32u, 3u, false, false,
                                  [&](uint64_t id, const uint8_t* bytes, size_t byteCount)
                                  {
                                      OvstageTriangleRow* row = findRow(id);
                                      if (row == nullptr)
                                      {
                                          return OVSPATIALQUERY_API_SUCCESS;
                                      }
                                      if (row->hasPoints || byteCount == 0u || byteCount % (sizeof(float) * 3u) != 0u)
                                      {
                                          refusal.note(id, OVX_INVALID_TOKEN, "its points are not float3 values");
                                          return OVSPATIALQUERY_API_INVALID_ARGUMENT;
                                      }
                                      const size_t valueCount = byteCount / sizeof(float);
                                      row->points.resize(valueCount);
                                      for (size_t value = 0u; value < valueCount; ++value)
                                      {
                                          float component = 0.0f;
                                          std::memcpy(&component, bytes + value * sizeof(component), sizeof(component));
                                          row->points[value] = component;
                                      }
                                      row->hasPoints = true;
                                      return OVSPATIALQUERY_API_SUCCESS;
                                  });
        const auto readIntArray = [&](ovx_token_t attribute, bool OvstageTriangleRow::*present,
                                      std::vector<int32_t> OvstageTriangleRow::*values, const char* rule)
        {
            return readTriangleAttribute(stage, queryHandle, attribute, targetOrdinal, dictionary,
                                         OVSTAGE_SEMANTIC_NONE, kDLInt, 32u, 1u, false, true,
                                         [&](uint64_t id, const uint8_t* bytes, size_t byteCount)
                                         {
                                             OvstageTriangleRow* row = findRow(id);
                                             if (row == nullptr)
                                             {
                                                 return OVSPATIALQUERY_API_SUCCESS;
                                             }
                                             if (convexIds.find(id) != convexIds.end())
                                             {
                                                 return OVSPATIALQUERY_API_SUCCESS;
                                             }
                                             if (row->*present || byteCount == 0u || byteCount % sizeof(int32_t) != 0u)
                                             {
                                                 refusal.note(id, OVX_INVALID_TOKEN, rule);
                                                 return OVSPATIALQUERY_API_INVALID_ARGUMENT;
                                             }
                                             std::vector<int32_t>& destination = row->*values;
                                             destination.resize(byteCount / sizeof(int32_t));
                                             std::memcpy(destination.data(), bytes, byteCount);
                                             row->*present = true;
                                             return OVSPATIALQUERY_API_SUCCESS;
                                         });
        };
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            status = readIntArray(filter.faceVertexCounts, &OvstageTriangleRow::hasFaceVertexCounts,
                                  &OvstageTriangleRow::faceVertexCounts,
                                  "its faceVertexCounts is empty or not an int array");
        }
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            status = readIntArray(filter.faceVertexIndices, &OvstageTriangleRow::hasFaceVertexIndices,
                                  &OvstageTriangleRow::faceVertexIndices,
                                  "its faceVertexIndices is empty or not an int array");
        }
        const auto readRequiredToken =
            [&](ovx_token_t attribute, bool OvstageTriangleRow::*present, ovx_token_t OvstageTriangleRow::*value)
        {
            return readTriangleAttribute(stage, queryHandle, attribute, targetOrdinal, dictionary,
                                         OVSTAGE_SEMANTIC_TOKEN_ID, kDLUInt, 64u, 1u, false, true,
                                         [&](uint64_t id, const uint8_t* bytes, size_t byteCount)
                                         {
                                             OvstageTriangleRow* row = findRow(id);
                                             if (row == nullptr)
                                             {
                                                 return OVSPATIALQUERY_API_SUCCESS;
                                             }
                                             if (convexIds.find(id) != convexIds.end())
                                             {
                                                 return OVSPATIALQUERY_API_SUCCESS;
                                             }
                                             if (row->*present || byteCount != sizeof(ovx_token_t))
                                             {
                                                 refusal.note(id, OVX_INVALID_TOKEN,
                                                              "its orientation is not one token");
                                                 return OVSPATIALQUERY_API_INVALID_ARGUMENT;
                                             }
                                             std::memcpy(&(row->*value), bytes, sizeof(ovx_token_t));
                                             row->*present = true;
                                             return OVSPATIALQUERY_API_SUCCESS;
                                         });
        };
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            status = readRequiredToken(filter.orientation, &OvstageTriangleRow::hasOrientation,
                                       &OvstageTriangleRow::orientation);
        }
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            status = readTriangleAttribute(stage, queryHandle, filter.holeIndices, targetOrdinal, dictionary,
                                           OVSTAGE_SEMANTIC_NONE, kDLInt, 32u, 1u, false, true,
                                           [&](uint64_t id, const uint8_t* bytes, size_t byteCount)
                                           {
                                               OvstageTriangleRow* row = findRow(id);
                                               if (row == nullptr || convexIds.find(id) != convexIds.end())
                                               {
                                                   return OVSPATIALQUERY_API_SUCCESS;
                                               }
                                               if (row->hasHoleIndices || byteCount % sizeof(int32_t) != 0u)
                                               {
                                                   refusal.note(id, OVX_INVALID_TOKEN,
                                                                "its holeIndices is not one int array");
                                                   return OVSPATIALQUERY_API_INVALID_ARGUMENT;
                                               }
                                               row->holeIndices.resize(byteCount / sizeof(int32_t));
                                               if (byteCount != 0u)
                                               {
                                                   std::memcpy(row->holeIndices.data(), bytes, byteCount);
                                               }
                                               row->hasHoleIndices = true;
                                               return OVSPATIALQUERY_API_SUCCESS;
                                           });
        }
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            status = readTriangleAttribute(
                stage, queryHandle, filter.doubleSided, targetOrdinal, dictionary, OVSTAGE_SEMANTIC_NONE, kDLBool, 8u,
                1u, false, true,
                [&](uint64_t id, const uint8_t* bytes, size_t byteCount)
                {
                    OvstageTriangleRow* row = findRow(id);
                    if (row != nullptr && convexIds.find(id) == convexIds.end() && byteCount != 0u)
                    {
                        if (byteCount != 1u || bytes[0] > 1u)
                        {
                            refusal.note(id, OVX_INVALID_TOKEN, "its doubleSided is not one bool");
                            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
                        }
                        row->doubleSided = bytes[0] != 0u;
                    }
                    return OVSPATIALQUERY_API_SUCCESS;
                });
        }
        /* A convex's authored hull vertex limit, as Omni Physics reads it; a value that is not
         * one int, another type included, counts as unauthored (ADR-0078). */
        if (status == OVSPATIALQUERY_API_SUCCESS && !convexIds.empty())
        {
            status = readTriangleAttribute(
                stage, queryHandle, filter.hullVertexLimit, targetOrdinal, dictionary, OVSTAGE_SEMANTIC_NONE, kDLInt,
                32u, 1u, false, true,
                [&](uint64_t id, const uint8_t* bytes, size_t byteCount)
                {
                    OvstageTriangleRow* row = findRow(id);
                    if (row != nullptr && byteCount == sizeof(int32_t) && convexIds.find(id) != convexIds.end())
                    {
                        int32_t value = 0;
                        std::memcpy(&value, bytes, sizeof(value));
                        row->hullVertexLimit = value;
                    }
                    return OVSPATIALQUERY_API_SUCCESS;
                },
                []() { return OVSPATIALQUERY_API_SUCCESS; }, true);
        }
        /* The caller's typed read already took every row's matrix and the authored world
         * matrix column's status; when the Fabric matrix covered every row it skipped that
         * column, which still fails the publication here if it is malformed. */
        if (status == OVSPATIALQUERY_API_SUCCESS && !worldMatrixColumnRead)
        {
            status = readTriangleAttribute(stage, queryHandle, filter.worldMatrix, targetOrdinal, dictionary,
                                           OVSTAGE_SEMANTIC_MATRIX, kDLFloat, 64u, 16u, false, true,
                                           [](uint64_t, const uint8_t*, size_t) { return OVSPATIALQUERY_API_SUCCESS; });
        }
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return status;
        }

        AcquiredTypedMeshGeometry geometry;
        geometry.triangles.meshes.reserve(rows.size());
        geometry.triangles.objects.reserve(rows.size());
        geometry.convexes.reserve(convexIds.size());
        geometry.triangleDefinitions.reserve(rows.size());
        geometry.convexDefinitions.reserve(convexIds.size());
        geometry.triangleObjectRows.reserve(rows.size());
        geometry.convexObjectRows.reserve(convexIds.size());
        for (size_t rowIndex = 0u; rowIndex < rows.size(); ++rowIndex)
        {
            const OvstageTriangleRow& row      = rows[rowIndex];
            const uint64_t            prim     = objects[rowIndex].id.value;
            const bool                isConvex = convexIds.find(prim) != convexIds.end();
            const auto                refuse   = [&](ovspatialquery_api_status_t refusalStatus, const char* rule)
            {
                refusal.note(prim, OVX_INVALID_TOKEN, rule);
                return refusalStatus;
            };
            if (!row.hasPoints || row.points.empty() || row.points.size() % 3u != 0u)
            {
                return refuse(OVSPATIALQUERY_API_INVALID_ARGUMENT, "it has no points");
            }
            for (float point : row.points)
                if (!inInputDomain(point))
                    return refuse(OVSPATIALQUERY_API_INVALID_ARGUMENT, "its points lie outside the input domain");
            const char* missing = isConvex                    ? nullptr
                                  : !row.hasFaceVertexCounts  ? "it has no faceVertexCounts"
                                  : !row.hasFaceVertexIndices ? "it has no faceVertexIndices"
                                  : !row.hasOrientation       ? "it has no orientation"
                                                              : nullptr;
            if (missing != nullptr)
            {
                return refuse(OVSPATIALQUERY_API_INVALID_ARGUMENT, missing);
            }
            if (!isConvex && row.orientation != filter.rightHanded && row.orientation != filter.leftHanded)
            {
                return refuse(OVSPATIALQUERY_API_INVALID_ARGUMENT,
                              "its orientation is neither rightHanded nor leftHanded");
            }
            const ovspatialquery_api_status_t transformStatus = qualifyRigidMeshTransform(row.worldFromLocal);
            if (transformStatus != OVSPATIALQUERY_API_SUCCESS)
            {
                return refuse(transformStatus,
                              transformStatus == OVSPATIALQUERY_API_INVALID_ARGUMENT
                                  ? "its world transform is not finite or lies outside the input domain of "
                                    "ovspatialquery.h"
                                  : "ovstage populations publish a Mesh only under a right-handed rigid world "
                                    "transform with positive uniform scale");
            }
            if (isConvex)
            {
                ovspatialquery::detail::ConvexGeometrySource convex;
                convex.vertices       = row.points;
                convex.worldFromLocal = row.worldFromLocal;
                geometry.convexObjectRows.push_back(rowIndex);
                ovspatialquery::detail::ConvexDefinitionValue definition;
                definition.points          = convex.vertices;
                definition.hullVertexLimit = ovstageHullVertexLimit(row.hullVertexLimit);
                geometry.convexDefinitions.push_back(std::move(definition));
                geometry.convexes.push_back(std::move(convex));
                continue;
            }
            ovspatialquery::detail::TriangleMeshSource mesh;
            mesh.key                             = objects[rowIndex].id.value;
            mesh.vertices                        = row.points;
            bool rejectUnrepresentableSourceFace = false;
#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
            rejectUnrepresentableSourceFace =
                gRejectNextUnrepresentableSourceFace.exchange(false, std::memory_order_acq_rel);
#endif
            size_t validatedIndexOffset = 0u;
            for (size_t sourceFace = 0u; sourceFace < row.faceVertexCounts.size(); ++sourceFace)
            {
                const int32_t signedCount = row.faceVertexCounts[sourceFace];
                if (signedCount < 3)
                {
                    return refuse(OVSPATIALQUERY_API_INVALID_ARGUMENT, "a face in its faceVertexCounts has fewer than "
                                                                       "three vertices");
                }
                const size_t count = static_cast<size_t>(signedCount);
                if (validatedIndexOffset > row.faceVertexIndices.size() ||
                    count > row.faceVertexIndices.size() - validatedIndexOffset)
                {
                    return refuse(OVSPATIALQUERY_API_INVALID_ARGUMENT,
                                  "its faceVertexCounts sum to more than its faceVertexIndices hold");
                }
                for (size_t corner = 0u; corner < count; ++corner)
                {
                    const int32_t signedIndex = row.faceVertexIndices[validatedIndexOffset + corner];
                    if (signedIndex < 0 || static_cast<size_t>(signedIndex) >= row.points.size() / 3u)
                    {
                        return refuse(OVSPATIALQUERY_API_INVALID_ARGUMENT,
                                      "its faceVertexIndices name a point it does not have");
                    }
                }
                validatedIndexOffset += count;
            }
            if (validatedIndexOffset != row.faceVertexIndices.size() || row.faceVertexCounts.empty())
            {
                return refuse(OVSPATIALQUERY_API_INVALID_ARGUMENT,
                              "its faceVertexCounts do not sum to the length of its faceVertexIndices");
            }
            /* Topology validation above takes precedence over the source-face domain. */
            if (!ovstageSourceFaceFitsUint32(row.faceVertexCounts.size() - 1u) || rejectUnrepresentableSourceFace)
            {
                return refuse(OVSPATIALQUERY_API_NOT_SUPPORTED,
                              "ovstage populations publish only meshes whose face indices fit in 32 bits");
            }

            std::vector<uint8_t> holeFaces(row.holeIndices.empty() ? 0u : row.faceVertexCounts.size(), 0u);
            for (int32_t hole : row.holeIndices)
            {
                if (hole < 0 || static_cast<size_t>(hole) >= holeFaces.size())
                {
                    return refuse(OVSPATIALQUERY_API_INVALID_ARGUMENT, "its holeIndices name a face it does not have");
                }
                holeFaces[static_cast<size_t>(hole)] = 1u;
            }
            const bool leftHanded  = row.orientation == filter.leftHanded;
            size_t     indexOffset = 0u;
            for (size_t sourceFace = 0u; sourceFace < row.faceVertexCounts.size(); ++sourceFace)
            {
                const size_t count = static_cast<size_t>(row.faceVertexCounts[sourceFace]);
                if (!holeFaces.empty() && holeFaces[sourceFace] != 0u)
                {
                    indexOffset += count;
                    continue;
                }
                for (size_t corner = 1u; corner + 1u < count; ++corner)
                {
                    const size_t polygonCorners[] = {0u, leftHanded ? corner + 1u : corner,
                                                     leftHanded ? corner : corner + 1u};
                    for (size_t triangleCorner : polygonCorners)
                    {
                        const int32_t signedIndex = row.faceVertexIndices[indexOffset + triangleCorner];
                        mesh.triangles.push_back(static_cast<uint32_t>(signedIndex));
                    }
                    mesh.sourceFaces.push_back(static_cast<uint32_t>(sourceFace));
                    for (size_t triangleCorner : polygonCorners)
                    {
                        mesh.sourceCorners.push_back(static_cast<uint32_t>(triangleCorner));
                    }
                }
                indexOffset += count;
            }
            if (mesh.triangles.empty())
            {
                return refuse(OVSPATIALQUERY_API_INVALID_ARGUMENT, "all of its faces are holes");
            }
            const uint64_t meshIndex = static_cast<uint64_t>(geometry.triangles.meshes.size());
            geometry.triangles.meshes.push_back(std::move(mesh));
            ovspatialquery::detail::TriangleObjectSource object;
            object.meshIndex      = meshIndex;
            object.worldFromLocal = row.worldFromLocal;
            object.doubleSided    = row.doubleSided;
            geometry.triangles.objects.push_back(object);
            geometry.triangleObjectRows.push_back(rowIndex);
            ovspatialquery::detail::MeshDefinitionValue definition;
            definition.points = row.points;
            definition.faceVertexCounts.assign(row.faceVertexCounts.begin(), row.faceVertexCounts.end());
            definition.faceVertexIndices.assign(row.faceVertexIndices.begin(), row.faceVertexIndices.end());
            definition.holeIndices.assign(row.holeIndices.begin(), row.holeIndices.end());
            definition.leftHanded                                        = leftHanded;
            const ovspatialquery::detail::TriangleMeshSource& storedMesh = geometry.triangles.meshes.back();
            definition.sourceFaces                                       = storedMesh.sourceFaces;
            definition.sourceCorners                                     = storedMesh.sourceCorners;
            definition.doubleSided                                       = row.doubleSided;
            geometry.triangleDefinitions.push_back(std::move(definition));
        }
        outGeometry = std::move(geometry);
        return OVSPATIALQUERY_API_SUCCESS;
    }
    catch (const std::bad_alloc&)
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    catch (const std::length_error&)
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    catch (...)
    {
        return OVSPATIALQUERY_API_INTERNAL;
    }
}

ovspatialquery_api_status_t findUnchangedAuthoredMeshes(
    ovstage_instance_t* stage, ovstage_query_handle_t queryHandle, const OwnedFilter& filter,
    ovspatialquery_ordinal_t targetOrdinal, path_dictionary_instance_t* dictionary,
    const std::unordered_map<uint64_t, const MeshDefinitionValue*>& candidates,
    std::unordered_set<uint64_t>& outUnchanged, std::unordered_map<uint64_t, DeformedMeshPoints>* outDeformed)
{
    outUnchanged.clear();
    if (outDeformed != nullptr)
    {
        outDeformed->clear();
    }
    if (candidates.empty())
    {
        return OVSPATIALQUERY_API_SUCCESS;
    }
    struct Comparison
    {
        const MeshDefinitionValue* kept            = nullptr;
        bool                       samePoints      = false;
        bool                       sameCounts      = false;
        bool                       sameIndices     = false;
        bool                       sameOrientation = false;
        bool                       doubleSided     = false;
        bool                       sameHoles       = false;
        /* Set when the prim's bytes of an attribute are queued for comparison, so a
         * second tensor for the same prim is not compared alongside the first. */
        bool pointsQueued  = false;
        bool countsQueued  = false;
        bool indicesQueued = false;
        /* Filled while the points differ but the vertex count does not, for `outDeformed`. */
        std::vector<float> movedPoints;
        bool               moved = false;
    };
    /* One prim's bytes from the group being read, compared once the group is in hand. */
    struct Pending
    {
        Comparison*    comparison = nullptr;
        const uint8_t* bytes      = nullptr;
        size_t         byteCount  = 0u;
    };
    using Compare = std::function<void(Comparison&, const uint8_t*, size_t)>;
    struct Batch
    {
        const std::vector<Pending>* pending = nullptr;
        const Compare*              compare = nullptr;
        std::atomic<int32_t>*       failure = nullptr;
    };
    try
    {
        std::unordered_map<uint64_t, Comparison> comparisons;
        comparisons.reserve(candidates.size());
        for (const std::pair<const uint64_t, const MeshDefinitionValue*>& candidate : candidates)
        {
            if (candidate.second != nullptr && !candidate.second->faceVertexCounts.empty())
            {
                Comparison comparison{};
                comparison.kept      = candidate.second;
                comparison.sameHoles = candidate.second->holeIndices.empty();
                comparisons.emplace(candidate.first, std::move(comparison));
            }
        }
        const auto find = [&](uint64_t id) -> Comparison*
        {
            const std::unordered_map<uint64_t, Comparison>::iterator found = comparisons.find(id);
            return found == comparisons.end() ? nullptr : &found->second;
        };

        /* A group's comparisons run one prim per task on this thread and on the instance's
         * idle workers; prims are independent. A task's failure is kept and reported once
         * the reads are done. */
        const SharedWorkerLoan& workers = filter.preparation.workers;
        std::vector<Pending>    pending;
        pending.reserve(comparisons.size());
        std::atomic<int32_t> taskFailure{static_cast<int32_t>(OVSPATIALQUERY_API_SUCCESS)};
        const auto           runTask = [](void* state, size_t index) noexcept
        {
            const Batch&   batch = *static_cast<const Batch*>(state);
            const Pending& entry = (*batch.pending)[index];
            try
            {
                (*batch.compare)(*entry.comparison, entry.bytes, entry.byteCount);
            }
            catch (const std::bad_alloc&)
            {
                batch.failure->store(static_cast<int32_t>(OVSPATIALQUERY_API_OUT_OF_MEMORY));
            }
            catch (...)
            {
                batch.failure->store(static_cast<int32_t>(OVSPATIALQUERY_API_INTERNAL));
            }
        };
        const auto comparePending = [&](const Compare& compare)
        {
            Batch batch{&pending, &compare, &taskFailure};
            if (workers.run != nullptr && pending.size() > 1u)
            {
                workers.run(workers.state, pending.size(), runTask, &batch);
            }
            else
            {
                for (size_t index = 0u; index < pending.size(); ++index)
                {
                    runTask(&batch, index);
                }
            }
            pending.clear();
            return static_cast<ovspatialquery_api_status_t>(taskFailure.load());
        };
        // Every consumer queues its prim's bytes and never fails: the full read owns the
        // diagnosis of malformed content.
        const auto queue = [&](Comparison* comparison, bool Comparison::*queued, const uint8_t* bytes, size_t byteCount)
        {
            if (comparison != nullptr && !(comparison->*queued))
            {
                comparison->*queued = true;
                pending.push_back(Pending{comparison, bytes, byteCount});
            }
            return OVSPATIALQUERY_API_SUCCESS;
        };
        // Retained definitions normalize signed zero in the authored points. Their counts
        // and indices are the authored int32 values the full read
        // accepted, all non-negative, so they share their bytes with the uint32 copies and
        // no negative authored value can match them.
        const auto sameBytes = [](const uint8_t* bytes, size_t byteCount, const void* kept, size_t keptByteCount)
        { return byteCount == keptByteCount && std::memcmp(bytes, kept, byteCount) == 0; };
        const Compare comparePoints = [&](Comparison& comparison, const uint8_t* bytes, size_t byteCount)
        {
            const std::vector<float>& points = comparison.kept->points;
            comparison.samePoints = sameBytes(bytes, byteCount, points.data(), points.size() * sizeof(float));
            // Registry definitions normalize signed zero; authored tensors need not.
            if (!comparison.samePoints && byteCount == points.size() * sizeof(float))
            {
                comparison.samePoints = true;
                for (size_t index = 0u; index < points.size(); ++index)
                {
                    float point = 0.0f;
                    std::memcpy(&point, bytes + index * sizeof(float), sizeof(float));
                    if (point != points[index])
                    {
                        comparison.samePoints = false;
                        break;
                    }
                }
            }
            if (outDeformed != nullptr && !comparison.samePoints && byteCount == points.size() * sizeof(float))
            {
                comparison.movedPoints.resize(points.size());
                std::memcpy(comparison.movedPoints.data(), bytes, byteCount);
                // Invalid authored points take the full read's per-prim refusal,
                // before K can shrink them into the accepted input domain.
                comparison.moved = std::all_of(comparison.movedPoints.begin(), comparison.movedPoints.end(),
                                               [](float point) { return inInputDomain(point); });
            }
        };
        const Compare compareCounts = [&](Comparison& comparison, const uint8_t* bytes, size_t byteCount)
        {
            const std::vector<uint32_t>& counts = comparison.kept->faceVertexCounts;
            comparison.sameCounts = sameBytes(bytes, byteCount, counts.data(), counts.size() * sizeof(uint32_t));
        };
        const Compare compareIndices = [&](Comparison& comparison, const uint8_t* bytes, size_t byteCount)
        {
            const std::vector<uint32_t>& indices = comparison.kept->faceVertexIndices;
            comparison.sameIndices = sameBytes(bytes, byteCount, indices.data(), indices.size() * sizeof(uint32_t));
        };
        const auto readCompared = [&](ovx_token_t attribute, ovstage_attribute_semantic_t semantic, uint8_t code,
                                      uint16_t lanes, bool Comparison::*queued, const Compare& compare)
        {
            return readTriangleAttribute(
                stage, queryHandle, attribute, targetOrdinal, dictionary, semantic, code, 32u, lanes, false, true,
                [&, queued](uint64_t id, const uint8_t* bytes, size_t byteCount)
                { return queue(find(id), queued, bytes, byteCount); },
                [&]() { return comparePending(compare); });
        };
        ovspatialquery_api_status_t status =
            readCompared(filter.points, OVSTAGE_SEMANTIC_POINT, kDLFloat, 3u, &Comparison::pointsQueued, comparePoints);
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            status = readCompared(filter.faceVertexCounts, OVSTAGE_SEMANTIC_NONE, kDLInt, 1u, &Comparison::countsQueued,
                                  compareCounts);
        }
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            status = readCompared(filter.faceVertexIndices, OVSTAGE_SEMANTIC_NONE, kDLInt, 1u,
                                  &Comparison::indicesQueued, compareIndices);
        }
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            status = readTriangleAttribute(stage, queryHandle, filter.orientation, targetOrdinal, dictionary,
                                           OVSTAGE_SEMANTIC_TOKEN_ID, kDLUInt, 64u, 1u, false, true,
                                           [&](uint64_t id, const uint8_t* bytes, size_t byteCount)
                                           {
                                               Comparison* comparison = find(id);
                                               if (comparison != nullptr && byteCount == sizeof(ovx_token_t))
                                               {
                                                   ovx_token_t value = OVX_INVALID_TOKEN;
                                                   std::memcpy(&value, bytes, sizeof(value));
                                                   comparison->sameOrientation =
                                                       value == (comparison->kept->leftHanded ? filter.leftHanded
                                                                                              : filter.rightHanded);
                                               }
                                               return OVSPATIALQUERY_API_SUCCESS;
                                           });
        }
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            status = readTriangleAttribute(stage, queryHandle, filter.holeIndices, targetOrdinal, dictionary,
                                           OVSTAGE_SEMANTIC_NONE, kDLInt, 32u, 1u, false, true,
                                           [&](uint64_t id, const uint8_t* bytes, size_t byteCount)
                                           {
                                               Comparison* comparison = find(id);
                                               if (comparison != nullptr)
                                               {
                                                   const std::vector<uint32_t>& holes = comparison->kept->holeIndices;
                                                   comparison->sameHoles =
                                                       byteCount == 0u ? holes.empty()
                                                                       : sameBytes(bytes, byteCount, holes.data(),
                                                                                   holes.size() * sizeof(uint32_t));
                                               }
                                               return OVSPATIALQUERY_API_SUCCESS;
                                           });
        }
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            status = readTriangleAttribute(stage, queryHandle, filter.doubleSided, targetOrdinal, dictionary,
                                           OVSTAGE_SEMANTIC_NONE, kDLBool, 8u, 1u, false, true,
                                           [&](uint64_t id, const uint8_t* bytes, size_t byteCount)
                                           {
                                               Comparison* comparison = find(id);
                                               if (comparison != nullptr && byteCount == 1u)
                                               {
                                                   comparison->doubleSided = bytes[0] != 0u;
                                               }
                                               return OVSPATIALQUERY_API_SUCCESS;
                                           });
        }
        const ovspatialquery_api_status_t failed = static_cast<ovspatialquery_api_status_t>(taskFailure.load());
        if (failed != OVSPATIALQUERY_API_SUCCESS)
        {
            return failed;
        }
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return OVSPATIALQUERY_API_SUCCESS;
        }
        for (std::pair<const uint64_t, Comparison>& entry : comparisons)
        {
            Comparison& comparison   = entry.second;
            const bool  sameTopology = comparison.sameCounts && comparison.sameIndices && comparison.sameOrientation &&
                                      comparison.sameHoles && comparison.doubleSided == comparison.kept->doubleSided;
            if (!sameTopology)
            {
                continue;
            }
            if (comparison.samePoints)
            {
                outUnchanged.insert(entry.first);
            }
            else if (outDeformed != nullptr && comparison.moved)
            {
                outDeformed->emplace(entry.first,
                                     DeformedMeshPoints{std::move(comparison.movedPoints), comparison.kept});
            }
        }
        return OVSPATIALQUERY_API_SUCCESS;
    }
    catch (const std::bad_alloc&)
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    catch (...)
    {
        return OVSPATIALQUERY_API_INTERNAL;
    }
}

ovspatialquery_api_status_t collectAuthoredConvexIds(ovstage_instance_t* stage, ovstage_query_handle_t queryHandle,
                                                     const OwnedFilter& filter, ovspatialquery_ordinal_t targetOrdinal,
                                                     path_dictionary_instance_t*   dictionary,
                                                     std::unordered_set<uint64_t>& outIds)
{
    try
    {
        std::unordered_set<uint64_t> schemaIds;
        std::unordered_set<uint64_t> approximationIds;
        ovspatialquery_api_status_t  status = readTriangleAttribute(
            stage, queryHandle, filter.usdSchemas, targetOrdinal, dictionary, OVSTAGE_SEMANTIC_TOKEN_ID, kDLUInt, 64u,
            1u, true, /* The pinned built-in reports SEMANTIC_NONE. */
            true,
            [&](uint64_t id, const uint8_t* bytes, size_t byteCount)
            {
                if (byteCount % sizeof(ovx_token_t) != 0u)
                {
                    return OVSPATIALQUERY_API_INVALID_STATE;
                }
                bool hasSchema = false;
                for (size_t offset = 0u; offset < byteCount; offset += sizeof(ovx_token_t))
                {
                    ovx_token_t schema = OVX_INVALID_TOKEN;
                    std::memcpy(&schema, bytes + offset, sizeof(schema));
                    hasSchema = hasSchema || schema == filter.physicsMeshCollisionApi;
                }
                if (hasSchema && !schemaIds.insert(id).second)
                {
                    return OVSPATIALQUERY_API_INVALID_STATE;
                }
                return OVSPATIALQUERY_API_SUCCESS;
            });
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            status =
                readTriangleAttribute(stage, queryHandle, filter.physicsApproximation, targetOrdinal, dictionary,
                                      OVSTAGE_SEMANTIC_TOKEN_ID, kDLUInt, 64u, 1u, false, true,
                                      [&](uint64_t id, const uint8_t* bytes, size_t byteCount)
                                      {
                                          ovx_token_t approximation = OVX_INVALID_TOKEN;
                                          if (byteCount != sizeof(approximation))
                                          {
                                              return OVSPATIALQUERY_API_INVALID_STATE;
                                          }
                                          std::memcpy(&approximation, bytes, sizeof(approximation));
                                          if (approximation == filter.convexHull && !approximationIds.insert(id).second)
                                          {
                                              return OVSPATIALQUERY_API_INVALID_STATE;
                                          }
                                          return OVSPATIALQUERY_API_SUCCESS;
                                      });
        }
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            std::unordered_set<uint64_t> ids;
            ids.reserve(std::min(schemaIds.size(), approximationIds.size()));
            for (uint64_t id : schemaIds)
            {
                if (approximationIds.find(id) != approximationIds.end())
                {
                    ids.insert(id);
                }
            }
            outIds = std::move(ids);
        }
        return status;
    }
    catch (const std::bad_alloc&)
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    catch (const std::length_error&)
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    catch (...)
    {
        return OVSPATIALQUERY_API_INTERNAL;
    }
}

} // namespace detail
} // namespace ovspatialquery
