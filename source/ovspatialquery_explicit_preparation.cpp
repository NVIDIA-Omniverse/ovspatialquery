/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_explicit_preparation.h"

#include "ovspatialquery_float_bounds.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <set>
#include <string>
#include <type_traits>
#include <utility>

namespace ovspatialquery
{
namespace detail
{
namespace
{

constexpr double   kQuaternionLengthTolerance = 1.0e-5;
constexpr uint64_t kSharedOwnerAllowance      = 128u;

struct PreparedDefinition
{
    GeometryReference                                     reference{};
    ExplicitDefinitionValue                               value{};
    std::shared_ptr<const PreparedConvexHull>             preparedConvex;
    mutable std::shared_ptr<const TriangleObjectGeometry> preparedTriangle;
    mutable std::shared_ptr<void>                         memoryOwner;
};

using DefinitionMap = std::map<GeometryReference, std::shared_ptr<const PreparedDefinition>, GeometryReferenceLess>;

double normalizedScalar(double value) noexcept { return value == 0.0 ? 0.0 : value; }

void normalizeZeros(std::vector<float>& values) noexcept
{
    for (float& value : values)
    {
        if (value == 0.0f)
        {
            value = 0.0f;
        }
    }
}

// Why a triangle mesh with finite vertices was refused, naming its definition value and
// the first triangle that names a vertex outside it, or saying that none of its triangles
// has area (ADR-0073); empty for another reason. Runs only after a refusal.
std::string describeRefusedTriangle(uint64_t value, const ExplicitTriangleMeshDefinition& definition)
{
    const std::string name          = "triangle mesh definition " + std::to_string(value);
    const size_t      vertexCount   = definition.vertices.size() / 3u;
    const size_t      triangleCount = definition.triangleVertexIndices.size() / 3u;
    for (size_t face = 0u; face < triangleCount; ++face)
    {
        const uint32_t* triangle = definition.triangleVertexIndices.data() + face * 3u;
        if (triangle[0] >= vertexCount || triangle[1] >= vertexCount || triangle[2] >= vertexCount)
        {
            return name + ": its triangle " + std::to_string(face) + " names a vertex outside its definition";
        }
    }
    // Otherwise preparation refuses a mesh with indices in range and vertices in the input
    // domain only when every triangle has no area.
    if (triangleCount == 0u ||
        !std::all_of(definition.vertices.begin(), definition.vertices.end(),
                     [](float component) { return inInputDomain(static_cast<double>(component)); }))
    {
        return {};
    }
    return name + ": none of its " + std::to_string(triangleCount) +
           " triangles has area, each has three collinear or repeated corners";
}

// A CUDA plan keys a multi-hit row's face material by a 32-bit source face while it sorts
// the rows (cudaFamilyMaterialKey), all ones meaning none, so face values with materials
// stay below it.
constexpr uint64_t kLargestMaterialFace = UINT64_C(0xffffffff);

bool repeatsCorner(const uint32_t* corners) noexcept
{
    return corners[0] == corners[1] || corners[1] == corners[2] || corners[0] == corners[2];
}

// Why a triangle mesh's caller source faces were refused: a triangle naming one corner
// twice, or two triangles of one face carrying different materials. Empty for another
// reason. Runs only after a refusal, on the per-triangle materials marshalling read.
std::string describeRefusedSourceFaces(uint64_t value, const ExplicitTriangleMeshDefinition& definition)
{
    const std::string name = "triangle mesh definition " + std::to_string(value);
    for (size_t triangle = 0u; triangle < definition.sourceFaces.size(); ++triangle)
    {
        if (repeatsCorner(definition.sourceCorners.data() + triangle * 3u))
        {
            return name + ": its triangle " + std::to_string(triangle) + " names one source face corner twice";
        }
    }
    const std::vector<uint64_t>* materials = definition.sourceFaceMaterialIds.get();
    if (materials == nullptr || materials->size() != definition.sourceFaces.size())
    {
        return {};
    }
    for (size_t triangle = 0u; triangle < definition.sourceFaces.size(); ++triangle)
    {
        if (definition.sourceFaces[triangle] >= kLargestMaterialFace)
        {
            return name + ": its triangle " + std::to_string(triangle) + " names source face " +
                   std::to_string(definition.sourceFaces[triangle]) +
                   ", but a source face with materials is below 4294967295";
        }
    }
    std::map<uint64_t, size_t> firstTriangle;
    for (size_t triangle = 0u; triangle < definition.sourceFaces.size(); ++triangle)
    {
        const auto   inserted = firstTriangle.emplace(definition.sourceFaces[triangle], triangle);
        const size_t first    = inserted.first->second;
        if (!inserted.second && (*materials)[first] != (*materials)[triangle])
        {
            return name + ": its triangles " + std::to_string(first) + " and " + std::to_string(triangle) +
                   " belong to source face " + std::to_string(definition.sourceFaces[triangle]) +
                   " but carry different materials (" + std::to_string((*materials)[first]) + " and " +
                   std::to_string((*materials)[triangle]) + "); a material belongs to a source face";
        }
    }
    return {};
}

// Checks the caller's source faces and, with face materials, replaces the per-triangle
// materials by one per face value up to the largest, which the hit path reads by source
// face (hitMaterialId). INVALID_ARGUMENT for a triangle naming one corner twice or a face
// whose triangles carry different materials; HOST_BUDGET when the face table exceeds
// `tableBytesLeft`, what the registration's earlier face tables left of the host budget,
// so neither a stray large face value nor many definitions reach the allocator before
// admission charges them. Deducts the table it keeps.
ovspatialquery_api_status_t prepareSourceFaces(ExplicitTriangleMeshDefinition& definition, uint64_t& tableBytesLeft)
{
    const size_t triangleCount = definition.sourceFaces.size();
    if (definition.sourceCorners.size() != triangleCount * 3u ||
        triangleCount * 3u != definition.triangleVertexIndices.size())
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    for (size_t triangle = 0u; triangle < triangleCount; ++triangle)
    {
        if (repeatsCorner(definition.sourceCorners.data() + triangle * 3u))
        {
            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
        }
    }
    if (definition.sourceFaceMaterialIds == nullptr)
    {
        return OVSPATIALQUERY_API_SUCCESS;
    }
    const std::vector<uint64_t>& perTriangle = *definition.sourceFaceMaterialIds;
    const uint64_t largestFace = *std::max_element(definition.sourceFaces.begin(), definition.sourceFaces.end());
    if (largestFace >= kLargestMaterialFace)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    // Below 2^32, so the byte counts cannot overflow. The check also covers the one-bit
    // per face scratch below, which is freed before this returns.
    const uint64_t tableBytes   = (largestFace + 1u) * sizeof(uint64_t);
    const uint64_t scratchBytes = (largestFace + 8u) / 8u;
    if (tableBytes + scratchBytes > tableBytesLeft)
    {
        return OVSPATIALQUERY_API_HOST_BUDGET;
    }
    const size_t          faceCount = static_cast<size_t>(largestFace) + 1u;
    std::vector<uint64_t> perFace(faceCount, 0u);
    std::vector<bool>     assigned(faceCount, false);
    for (size_t triangle = 0u; triangle < triangleCount; ++triangle)
    {
        const size_t face = static_cast<size_t>(definition.sourceFaces[triangle]);
        if (assigned[face] && perFace[face] != perTriangle[triangle])
        {
            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
        }
        assigned[face] = true;
        perFace[face]  = perTriangle[triangle];
    }
    definition.sourceFaceMaterialIds = std::make_shared<const std::vector<uint64_t>>(std::move(perFace));
    tableBytesLeft -= tableBytes;
    return OVSPATIALQUERY_API_SUCCESS;
}

// A dropped triangle is named by the value its hits would report: its source face when
// the caller supplied source faces, otherwise its submitted ordinal.
const char* droppedNoun(const ExplicitDefinitionValue& value) noexcept
{
    const auto* mesh = std::get_if<ExplicitTriangleMeshDefinition>(&value);
    return mesh != nullptr && mesh->hasSourceFaces ? "source face" : "triangle";
}

bool explicitTriangleTopologyWellFormed(const ExplicitTriangleMeshDefinition& definition) noexcept
{
    if (definition.vertices.empty() || definition.vertices.size() % 3u != 0u ||
        definition.triangleVertexIndices.empty() || definition.triangleVertexIndices.size() % 3u != 0u)
    {
        return false;
    }
    // Preparation checks each triangle's vertex indices and drops those without area.
    return std::all_of(definition.vertices.begin(), definition.vertices.end(),
                       [](float component) { return std::isfinite(component); });
}

std::array<double, 16u> identityMatrix() noexcept
{
    return {
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    };
}

// Moves the definition's source faces and corners into the source: the prepared mesh's
// topology keeps them, so the definition does not hold a second copy.
TriangleMeshSource triangleMeshSource(GeometryReference reference, ExplicitTriangleMeshDefinition& definition)
{
    TriangleMeshSource source;
    source.key           = reference.value;
    source.vertices      = definition.vertices;
    source.triangles     = definition.triangleVertexIndices;
    source.sourceFaces   = std::move(definition.sourceFaces);
    source.sourceCorners = std::move(definition.sourceCorners);
    definition.sourceFaces.clear();
    definition.sourceCorners.clear();
    return source;
}

ovspatialquery_api_status_t validateAndNormalizeDefinition(
    ExplicitDefinitionInput& input, ovspatialquery_backend_runtime_t* backendRuntime,
    uint64_t maximumPreparedConvexResidentBytes, uint64_t& faceTableBytesLeft,
    std::shared_ptr<const PreparedConvexHull>&     outPreparedConvex,
    std::shared_ptr<const TriangleObjectGeometry>& outPreparedTriangle, ConvexHullPreparationOutcome& outConvexOutcome,
    double& outFarthestOutside) noexcept
{
    outFarthestOutside = 0.0;
    outPreparedConvex.reset();
    outPreparedTriangle.reset();
    outConvexOutcome = ConvexHullPreparationOutcome::success;
    // Values from kFirstAdapterGeometryValue up belong to the identity registry.
    if (input.reference.value == 0u || input.reference.value >= kFirstAdapterGeometryValue)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    return std::visit(
        [&input, backendRuntime, maximumPreparedConvexResidentBytes, &faceTableBytesLeft, &outPreparedConvex,
         &outPreparedTriangle, &outConvexOutcome,
         &outFarthestOutside](auto& definition) noexcept -> ovspatialquery_api_status_t
        {
            using Value = std::decay_t<decltype(definition)>;
            if constexpr (std::is_same_v<Value, ExplicitSphereDefinition>)
            {
                definition.radius = normalizedScalar(definition.radius);
                return inInputDomain(definition.radius) && definition.radius > 0.0
                           ? OVSPATIALQUERY_API_SUCCESS
                           : OVSPATIALQUERY_API_INVALID_ARGUMENT;
            }
            else if constexpr (std::is_same_v<Value, ExplicitBoxDefinition>)
            {
                for (double& extent : definition.halfExtents)
                {
                    extent = normalizedScalar(extent);
                    if (!inInputDomain(extent) || extent <= 0.0)
                    {
                        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
                    }
                }
                return OVSPATIALQUERY_API_SUCCESS;
            }
            else if constexpr (std::is_same_v<Value, ExplicitCapsuleDefinition>)
            {
                definition.radius      = normalizedScalar(definition.radius);
                definition.halfSegment = normalizedScalar(definition.halfSegment);
                return inInputDomain(definition.radius) && definition.radius > 0.0 &&
                               inInputDomain(definition.halfSegment) && definition.halfSegment >= 0.0
                           ? OVSPATIALQUERY_API_SUCCESS
                           : OVSPATIALQUERY_API_INVALID_ARGUMENT;
            }
            else if constexpr (std::is_same_v<Value, ExplicitConvexDefinition>)
            {
                normalizeZeros(definition.vertices);
                outConvexOutcome =
                    prepareConvexHull(backendRuntime, definition.vertices, maximumPreparedConvexResidentBytes,
                                      outPreparedConvex, nullptr, definition.options, &outFarthestOutside);
                return convexHullPreparationStatus(outConvexOutcome);
            }
            else if constexpr (std::is_same_v<Value, ExplicitTriangleMeshDefinition>)
            {
                normalizeZeros(definition.vertices);
                if (!explicitTriangleTopologyWellFormed(definition))
                {
                    return OVSPATIALQUERY_API_INVALID_ARGUMENT;
                }
                try
                {
                    if (definition.hasSourceFaces)
                    {
                        const ovspatialquery_api_status_t faceStatus =
                            prepareSourceFaces(definition, faceTableBytesLeft);
                        if (faceStatus != OVSPATIALQUERY_API_SUCCESS)
                        {
                            return faceStatus;
                        }
                    }
                    std::shared_ptr<const PreparedTriangleMesh> mesh;
                    ovspatialquery_api_status_t                 status =
                        prepareTriangleMesh(triangleMeshSource(input.reference, definition), mesh);
                    if (status == OVSPATIALQUERY_API_SUCCESS)
                    {
                        status = bindPreparedTriangleGeometry(mesh, identityMatrix(), definition.doubleSided,
                                                              outPreparedTriangle);
                    }
                    return status;
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
            else
            {
                return OVSPATIALQUERY_API_SUCCESS;
            }
        },
        input.definition);
}

template <typename Value>
uint64_t addCapacity(uint64_t total, const std::vector<Value>& values) noexcept
{
    return saturatingResidentByteAdd(total, static_cast<uint64_t>(values.capacity()),
                                     static_cast<uint64_t>(sizeof(Value)));
}

uint64_t definitionResidentByteCount(const PreparedDefinition& definition) noexcept
{
    uint64_t total =
        saturatingResidentByteAdd(static_cast<uint64_t>(sizeof(PreparedDefinition)), 1u, kSharedOwnerAllowance);
    total = std::visit(
        [total, &definition](const auto& value) mutable noexcept
        {
            using Value = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<Value, ExplicitConvexDefinition>)
            {
                total = addCapacity(total, value.vertices);
                total = addCapacity(total, value.faceVertexCounts);
                total = addCapacity(total, value.faceVertexIndices);
                if (definition.preparedConvex != nullptr)
                {
                    total = saturatingResidentByteAdd(total, definition.preparedConvex->residentByteCount(), 1u);
                    total = saturatingResidentByteAdd(total, 1u, kSharedOwnerAllowance);
                }
                return total;
            }
            else if constexpr (std::is_same_v<Value, ExplicitTriangleMeshDefinition>)
            {
                total = addCapacity(total, value.vertices);
                total = addCapacity(total, value.triangleVertexIndices);
                total = addCapacity(total, value.sourceFaces);
                total = addCapacity(total, value.sourceCorners);
                if (value.sourceFaceMaterialIds != nullptr)
                {
                    total = addCapacity(total, *value.sourceFaceMaterialIds);
                    total = saturatingResidentByteAdd(total, 1u, kSharedOwnerAllowance);
                }
                return total;
            }
            else
            {
                return total;
            }
        },
        definition.value);
    if (definition.preparedTriangle != nullptr)
    {
        total = saturatingResidentByteAdd(total, triangleGeometryResidentByteCount(*definition.preparedTriangle), 1u);
    }
    constexpr uint64_t retainedMapNodeAllowance =
        2u * (static_cast<uint64_t>(sizeof(GeometryReference)) + 6u * static_cast<uint64_t>(sizeof(void*)));
    return saturatingResidentByteAdd(total, retainedMapNodeAllowance, 1u);
}

uint64_t definitionStagingByteCount(const PreparedDefinition& definition, uint64_t retainedBytes) noexcept
{
    return definition.preparedTriangle == nullptr
               ? retainedBytes
               : saturatingResidentByteAdd(
                     retainedBytes, 1u, static_cast<uint64_t>(sizeof(TriangleObjectGeometry)) + kSharedOwnerAllowance);
}

ovspatialquery_api_status_t makeRigidTransform(const ExplicitInstanceInput& input, std::array<double, 16u>& outMatrix,
                                               std::array<std::array<double, 3u>, 3u>& outAxes) noexcept
{
    if (!inScaleDomain(input.uniformScale))
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    for (double component : input.translation)
    {
        if (!inInputDomain(component))
        {
            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
        }
    }
    double quaternion[4] = {};
    double lengthSquared = 0.0;
    for (uint32_t component = 0u; component < 4u; ++component)
    {
        quaternion[component] = static_cast<double>(input.rotationXyzw[component]);
        if (!std::isfinite(quaternion[component]))
        {
            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
        }
        lengthSquared = std::fma(quaternion[component], quaternion[component], lengthSquared);
    }
    if (!std::isfinite(lengthSquared) || lengthSquared == 0.0 ||
        std::abs(lengthSquared - 1.0) > kQuaternionLengthTolerance)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    const double inverseLength  = 1.0 / std::sqrt(lengthSquared);
    double       rotation[3][3] = {};
    quaternionRotationMatrix(quaternion, inverseLength, rotation);
    std::array<double, 16u> matrix = identityMatrix();
    for (uint32_t row = 0u; row < 3u; ++row)
    {
        for (uint32_t column = 0u; column < 3u; ++column)
        {
            const double component                         = rotation[row][column];
            outAxes[column][row]                           = component;
            matrix[static_cast<size_t>(row) * 4u + column] = component * input.uniformScale;
        }
        matrix[static_cast<size_t>(row) * 4u + 3u] = normalizedScalar(input.translation[row]);
    }
    outMatrix = matrix;
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t createInstanceGeometry(const ExplicitInstanceInput& input,
                                                   const PreparedDefinition&    definition,
                                                   const TriangleGeometryCache& meshCache, PopulationObject& outObject)
{
    std::array<double, 16u>                worldFromLocal{};
    std::array<std::array<double, 3u>, 3u> axes{};
    ovspatialquery_api_status_t            status = makeRigidTransform(input, worldFromLocal, axes);
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        return status;
    }
    PopulationObject object{};
    object.id                = objectIdFromValue(input.objectValue);
    object.geometryReference = input.geometry;
    object.participationBits = input.participationBits;
    object.uniformScale      = input.uniformScale;
    object.materialId        = input.materialId;
    status                   = std::visit(
        [&input, &definition, &meshCache, &object, &axes,
         &worldFromLocal](const auto& value) -> ovspatialquery_api_status_t
        {
            using Value = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<Value, ExplicitSphereDefinition>)
            {
                SphereGeometrySource source;
                source.center = input.translation;
                source.radius = value.radius * input.uniformScale;
                return createSphereExactGeometry(source, object.exactGeometry);
            }
            else if constexpr (std::is_same_v<Value, ExplicitBoxDefinition>)
            {
                BoxGeometrySource source;
                source.center = input.translation;
                source.axes   = axes;
                for (uint32_t axis = 0u; axis < 3u; ++axis)
                {
                    source.halfExtents[axis] = value.halfExtents[axis] * input.uniformScale;
                }
                return createBoxExactGeometry(source, object.exactGeometry);
            }
            else if constexpr (std::is_same_v<Value, ExplicitCapsuleDefinition>)
            {
                CapsuleGeometrySource source;
                source.radius = value.radius * input.uniformScale;
                for (uint32_t axis = 0u; axis < 3u; ++axis)
                {
                    const double offset    = axes[1u][axis] * value.halfSegment * input.uniformScale;
                    source.endpointA[axis] = input.translation[axis] - offset;
                    source.endpointB[axis] = input.translation[axis] + offset;
                }
                return createCapsuleExactGeometry(source, object.exactGeometry);
            }
            else if constexpr (std::is_same_v<Value, ExplicitConvexDefinition>)
            {
                const ConvexConstructionOutcome outcome =
                    createConvexExactGeometry(definition.preparedConvex, worldFromLocal, object.exactGeometry);
                if (outcome == ConvexConstructionOutcome::success)
                {
                    return OVSPATIALQUERY_API_SUCCESS;
                }
                if (outcome == ConvexConstructionOutcome::outOfMemory)
                {
                    return OVSPATIALQUERY_API_OUT_OF_MEMORY;
                }
                return outcome == ConvexConstructionOutcome::internalFailure ? OVSPATIALQUERY_API_INTERNAL
                                                                                               : OVSPATIALQUERY_API_INVALID_ARGUMENT;
            }
            else if constexpr (std::is_same_v<Value, ExplicitTriangleMeshDefinition>)
            {
                object.geometryKey                           = definition.reference.value;
                object.faceMaterialIds                       = value.sourceFaceMaterialIds;
                const ovspatialquery_api_status_t bindStatus = meshCache.bindDefinition(
                    definition.reference.value, worldFromLocal, value.doubleSided, object.triangleGeometry);
                if (bindStatus != OVSPATIALQUERY_API_SUCCESS)
                {
                    return bindStatus;
                }
                return createTriangleMeshExactGeometry(object.triangleGeometry, object.exactGeometry);
            }
        },
        definition.value);
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        return status;
    }
    if (object.exactGeometry != nullptr)
    {
        object.bounds = object.exactGeometry->worldBounds();
    }
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        if (!inInputDomain(object.bounds.minimum[axis]) || !inInputDomain(object.bounds.maximum[axis]))
        {
            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
        }
    }
    outObject = std::move(object);
    return OVSPATIALQUERY_API_SUCCESS;
}

// The new head's exact-geometry charge from the base head's recorded one.
// Explicit objects never share per-object allocations, so replacing or removing
// fewer than half of the base objects is cheaper to account by adjusting the
// base charge than by recounting every object. Otherwise, or on any failure,
// the admission recount measures the new head.
std::optional<uint64_t> adjustedExactGeometryCharge(const GenerationLease& base, const ObjectTable& baseObjects,
                                                    const std::vector<PopulationObject>& changedObjects,
                                                    const std::vector<size_t>&           changedPositions,
                                                    const std::vector<size_t>&           removedPositions) noexcept
{
    const auto replaces = [&baseObjects, &changedObjects, &changedPositions](size_t index)
    {
        const size_t position = changedPositions[index];
        return position != baseObjects.size() && baseObjects[position].id.value == changedObjects[index].id.value;
    };
    size_t retiredCount = removedPositions.size();
    for (size_t index = 0u; index < changedObjects.size(); ++index)
    {
        retiredCount += replaces(index) ? 1u : 0u;
    }
    const std::optional<uint64_t> baseCharge = base.exactGeometryCharge();
    if (!baseCharge || 2u * retiredCount >= baseObjects.size())
    {
        return std::nullopt;
    }
    try
    {
        ExactGeometryResidentAccumulator retired;
        ExactGeometryResidentAccumulator added;
        bool                             counted = true;
        for (size_t index = 0u; index < changedObjects.size() && counted; ++index)
        {
            counted = accumulateExactGeometryResidentAllocations(added, changedObjects[index]) &&
                      (!replaces(index) ||
                       accumulateExactGeometryResidentAllocations(retired, baseObjects[changedPositions[index]]));
        }
        for (size_t index = 0u; index < removedPositions.size() && counted; ++index)
        {
            counted = accumulateExactGeometryResidentAllocations(retired, baseObjects[removedPositions[index]]);
        }
        const uint64_t retiredBytes = retired.residentByteCount();
        const uint64_t addedBytes   = added.residentByteCount();
        if (!counted || retiredBytes > *baseCharge ||
            addedBytes >= std::numeric_limits<uint64_t>::max() - (*baseCharge - retiredBytes))
        {
            return std::nullopt;
        }
        return *baseCharge - retiredBytes + addedBytes;
    }
    catch (...)
    {
        return std::nullopt;
    }
}

// First index at or after `from` whose ID is not below `value` in ID-ordered
// objects. The probe grows geometrically, so ascending lookups cost the log of
// the distance travelled rather than of the whole vector.
template <typename Objects>
size_t lowerBoundObjectFrom(const Objects& objects, size_t from, uint64_t value) noexcept
{
    size_t low  = from;
    size_t high = from;
    for (size_t step = 1u; high < objects.size() && objects[high].id.value < value; step *= 2u)
    {
        low  = high + 1u;
        high = low + step;
    }
    const auto end = objects.begin() + static_cast<std::ptrdiff_t>(std::min(high + 1u, objects.size()));
    return static_cast<size_t>(
        std::lower_bound(objects.begin() + static_cast<std::ptrdiff_t>(std::min(low, objects.size())), end, value,
                         [](const PopulationObject& object, uint64_t id) { return object.id.value < id; }) -
        objects.begin());
}

} // namespace

struct ExplicitPreparationRegistry::State
{
    explicit State(ovspatialquery_backend_runtime_t*                backendRuntimeValue,
                   uint64_t                                         maximumPreparedConvexResidentBytesValue,
                   std::shared_ptr<GeometryReferenceClaimDirectory> claimsValue)
        : backendRuntime(backendRuntimeValue),
          maximumPreparedConvexResidentBytes(maximumPreparedConvexResidentBytesValue), claims(std::move(claimsValue))
    {}

    mutable std::mutex                               mutex;
    DefinitionMap                                    definitions;
    ovspatialquery_backend_runtime_t*                backendRuntime                     = nullptr;
    uint64_t                                         maximumPreparedConvexResidentBytes = 0u;
    std::shared_ptr<GeometryReferenceClaimDirectory> claims;
    TriangleGeometryCache                            meshCache;
};

struct ExplicitPreparedRegistration::Impl
{
    std::vector<std::shared_ptr<const PreparedDefinition>> definitions;
    std::vector<GeometryReference>                         references;
    GeometryReferenceClaimDirectory::Batch                 claimBatch;
    std::vector<uint64_t>                                  definitionResidentBytes;
    std::vector<uint64_t>                                  definitionRetainedBytes;
    uint64_t                                               claimResidentBytes     = 0u;
    uint64_t                                               transientResidentBytes = 0u;
};

struct ExplicitPreparedUpdate::Impl
{
    ObjectTable                        objects;
    std::optional<uint64_t>            exactGeometryCharge;
    std::optional<VersionChangeRecord> changes;
};

ExplicitPreparedRegistration::ExplicitPreparedRegistration()                                        = default;
ExplicitPreparedRegistration::~ExplicitPreparedRegistration()                                       = default;
ExplicitPreparedRegistration::ExplicitPreparedRegistration(ExplicitPreparedRegistration&&) noexcept = default;
ExplicitPreparedRegistration& ExplicitPreparedRegistration::operator=(ExplicitPreparedRegistration&&) noexcept =
    default;

bool ExplicitPreparedRegistration::initialized() const noexcept { return mImpl != nullptr; }

size_t ExplicitPreparedRegistration::definitionCount() const noexcept
{
    return mImpl == nullptr ? 0u : mImpl->definitions.size();
}

const std::vector<uint64_t>& ExplicitPreparedRegistration::definitionResidentByteCounts() const noexcept
{
    static const std::vector<uint64_t> empty;
    return mImpl == nullptr ? empty : mImpl->definitionResidentBytes;
}

std::vector<const PreparedTriangleMesh*> ExplicitPreparedRegistration::triangleMeshes() const
{
    std::vector<const PreparedTriangleMesh*> meshes;
    if (mImpl != nullptr)
    {
        for (const std::shared_ptr<const PreparedDefinition>& definition : mImpl->definitions)
        {
            collectTriangleMesh(definition->preparedTriangle.get(), meshes);
            if (definition->preparedConvex != nullptr)
                collectTriangleMesh(definition->preparedConvex->preparedSurface().get(), meshes);
        }
    }
    return meshes;
}

const std::vector<uint64_t>& ExplicitPreparedRegistration::definitionRetainedByteCounts() const noexcept
{
    static const std::vector<uint64_t> empty;
    return mImpl == nullptr ? empty : mImpl->definitionRetainedBytes;
}

uint64_t ExplicitPreparedRegistration::claimResidentByteCount() const noexcept
{
    return mImpl == nullptr ? 0u : mImpl->claimResidentBytes;
}

uint64_t ExplicitPreparedRegistration::transientResidentByteCount() const noexcept
{
    return mImpl == nullptr ? 0u : mImpl->transientResidentBytes;
}

ExplicitPreparedUpdate::ExplicitPreparedUpdate()                                             = default;
ExplicitPreparedUpdate::~ExplicitPreparedUpdate()                                            = default;
ExplicitPreparedUpdate::ExplicitPreparedUpdate(ExplicitPreparedUpdate&&) noexcept            = default;
ExplicitPreparedUpdate& ExplicitPreparedUpdate::operator=(ExplicitPreparedUpdate&&) noexcept = default;

const ObjectTable& ExplicitPreparedUpdate::objects() const noexcept
{
    static const ObjectTable empty;
    return mImpl == nullptr ? empty : mImpl->objects;
}

std::optional<uint64_t> ExplicitPreparedUpdate::exactGeometryCharge() const noexcept
{
    return mImpl == nullptr ? std::nullopt : mImpl->exactGeometryCharge;
}

std::optional<VersionChangeRecord> ExplicitPreparedUpdate::releaseChanges() noexcept
{
    return mImpl == nullptr ? std::nullopt : std::move(mImpl->changes);
}

ObjectTable ExplicitPreparedUpdate::releaseObjects() noexcept
{
    return mImpl == nullptr ? ObjectTable() : std::move(mImpl->objects);
}

ExplicitPreparationRegistry::ExplicitPreparationRegistry(
    ovspatialquery_backend_runtime_t* backendRuntime, uint64_t maximumPreparedConvexResidentBytes,
    std::shared_ptr<GeometryReferenceClaimDirectory> claimDirectory)
    : mState(std::make_shared<State>(backendRuntime, maximumPreparedConvexResidentBytes,
                                     claimDirectory == nullptr ? std::make_shared<GeometryReferenceClaimDirectory>()
                                                               : std::move(claimDirectory)))
{}

ExplicitPreparationRegistry::~ExplicitPreparationRegistry() = default;

uint64_t ExplicitPreparationRegistry::fixedResidentByteCount() noexcept
{
    return static_cast<uint64_t>(sizeof(ExplicitPreparationRegistry)) + static_cast<uint64_t>(sizeof(State)) +
           2u * kSharedOwnerAllowance;
}

ovspatialquery_api_status_t ExplicitPreparationRegistry::prepareRegistration(
    const std::vector<ExplicitDefinitionInput>& definitions, ExplicitPreparedRegistration& outRegistration,
    std::string* outDiagnostic, std::string* outDropReport, uint64_t hostBytesInUse) const noexcept
{
    try
    {
        constexpr size_t                                    kNamedMeshes   = 4u;
        size_t                                              droppingMeshes = 0u;
        std::string                                         dropReport;
        std::unique_ptr<ExplicitPreparedRegistration::Impl> prepared(new ExplicitPreparedRegistration::Impl());
        prepared->definitions.reserve(definitions.size());
        prepared->references.reserve(definitions.size());
        prepared->definitionResidentBytes.reserve(definitions.size());
        prepared->definitionRetainedBytes.reserve(definitions.size());
        std::set<GeometryReference, GeometryReferenceLess> references;
        // The call's face-material tables together stay within what the instance's host
        // budget has left until admission charges them with their definitions.
        uint64_t faceTableBytesLeft = mState->maximumPreparedConvexResidentBytes > hostBytesInUse
                                          ? mState->maximumPreparedConvexResidentBytes - hostBytesInUse
                                          : 0u;
        for (const ExplicitDefinitionInput& source : definitions)
        {
            ExplicitDefinitionInput                       normalized = source;
            std::shared_ptr<const PreparedConvexHull>     preparedConvex;
            std::shared_ptr<const TriangleObjectGeometry> preparedTriangle;
            ConvexHullPreparationOutcome                  convexOutcome   = ConvexHullPreparationOutcome::success;
            double                                        farthestOutside = 0.0;
            const ovspatialquery_api_status_t             status          = validateAndNormalizeDefinition(
                normalized, mState->backendRuntime, mState->maximumPreparedConvexResidentBytes, faceTableBytesLeft,
                preparedConvex, preparedTriangle, convexOutcome, farthestOutside);
            const auto* convexDefinition = std::get_if<ExplicitConvexDefinition>(&normalized.definition);
            if (outDiagnostic != nullptr && (convexOutcome == ConvexHullPreparationOutcome::beyondVertexLimit ||
                                             convexOutcome == ConvexHullPreparationOutcome::beyondFaceLimit))
            {
                const bool vertices = convexOutcome == ConvexHullPreparationOutcome::beyondVertexLimit;
                *outDiagnostic      = "convex definition " + std::to_string(normalized.reference.value) +
                                 ": its hull has more than " +
                                 std::to_string(vertices ? kMaximumConvexHullVertices : kMaximumConvexHullFaces) +
                                 (vertices ? " vertices" : " faces") + ", the most a hull may have";
            }
            else if (outDiagnostic != nullptr && convexOutcome == ConvexHullPreparationOutcome::noEnclosingHull &&
                     convexDefinition != nullptr && convexDefinition->options.vertexLimit != 0u)
                *outDiagnostic = "convex definition " + std::to_string(normalized.reference.value) +
                                 ": PhysX cooks no valid hull of its points under the vertex limit " +
                                 std::to_string(convexDefinition->options.vertexLimit);
            else if (outDiagnostic != nullptr && convexOutcome == ConvexHullPreparationOutcome::noEnclosingHull)
            {
                *outDiagnostic = "convex definition " + std::to_string(normalized.reference.value) +
                                 ": no hull with its points as vertices encloses them all, as when they lie "
                                 "nearly in one plane";
                // The farthest point and the tolerance in force tell the caller what would register it.
                if (farthestOutside > 0.0 && convexDefinition != nullptr)
                {
                    char distances[96];
                    std::snprintf(
                        distances, sizeof(distances),
                        "; its farthest point lies %.3g of the half extent outside, beyond the tolerance %.3g",
                        farthestOutside, convexDefinition->options.enclosureTolerance);
                    *outDiagnostic += distances;
                }
            }
            if (status != OVSPATIALQUERY_API_SUCCESS)
            {
                const auto* mesh = std::get_if<ExplicitTriangleMeshDefinition>(&normalized.definition);
                if (status == OVSPATIALQUERY_API_INVALID_ARGUMENT && mesh != nullptr && outDiagnostic != nullptr &&
                    mesh->hasSourceFaces)
                {
                    *outDiagnostic = describeRefusedSourceFaces(normalized.reference.value, *mesh);
                }
                if (status == OVSPATIALQUERY_API_INVALID_ARGUMENT && mesh != nullptr && outDiagnostic != nullptr &&
                    outDiagnostic->empty() && mesh->vertices.size() % 3u == 0u &&
                    mesh->triangleVertexIndices.size() % 3u == 0u &&
                    std::all_of(mesh->vertices.begin(), mesh->vertices.end(),
                                [](float component) { return std::isfinite(component); }))
                {
                    *outDiagnostic = describeRefusedTriangle(normalized.reference.value, *mesh);
                }
                return status;
            }
            if (!references.insert(normalized.reference).second)
            {
                return OVSPATIALQUERY_API_INVALID_ARGUMENT;
            }
            if (preparedTriangle != nullptr && outDropReport != nullptr &&
                preparedTriangleMeshDroppedCount(*preparedTriangle->mesh) != 0u && droppingMeshes++ < kNamedMeshes)
            {
                dropReport += (dropReport.empty() ? "triangle mesh definition " : "; triangle mesh definition ") +
                              std::to_string(normalized.reference.value) + " " +
                              describeDroppedTriangles(*preparedTriangle->mesh, droppedNoun(normalized.definition));
            }
            std::shared_ptr<PreparedDefinition> definition = std::make_shared<PreparedDefinition>();
            definition->reference                          = normalized.reference;
            definition->value                              = std::move(normalized.definition);
            definition->preparedConvex                     = std::move(preparedConvex);
            definition->preparedTriangle                   = std::move(preparedTriangle);
            const uint64_t retainedDefinitionBytes         = definitionResidentByteCount(*definition);
            const uint64_t stagingDefinitionBytes = definitionStagingByteCount(*definition, retainedDefinitionBytes);
            prepared->definitionResidentBytes.push_back(stagingDefinitionBytes);
            prepared->definitionRetainedBytes.push_back(retainedDefinitionBytes);
            prepared->definitions.push_back(std::move(definition));
            prepared->references.push_back(normalized.reference);
        }
        uint64_t transientResident =
            saturatingResidentByteAdd(static_cast<uint64_t>(sizeof(ExplicitPreparedRegistration::Impl)),
                                      static_cast<uint64_t>(prepared->definitions.capacity()),
                                      static_cast<uint64_t>(sizeof(std::shared_ptr<const PreparedDefinition>)));
        transientResident =
            saturatingResidentByteAdd(transientResident, static_cast<uint64_t>(prepared->references.capacity()),
                                      static_cast<uint64_t>(sizeof(GeometryReference)));
        transientResident = saturatingResidentByteAdd(
            transientResident, static_cast<uint64_t>(prepared->definitionResidentBytes.capacity()),
            static_cast<uint64_t>(sizeof(uint64_t)));
        transientResident = saturatingResidentByteAdd(
            transientResident, static_cast<uint64_t>(prepared->definitionRetainedBytes.capacity()),
            static_cast<uint64_t>(sizeof(uint64_t)));
        const GeometryIdentityOutcome claimOutcome = mState->claims->stage(prepared->references, prepared->claimBatch);
        if (claimOutcome != GeometryIdentityOutcome::success)
        {
            return claimOutcome == GeometryIdentityOutcome::invalidArgument
                       ? OVSPATIALQUERY_API_INVALID_ARGUMENT
                       : (claimOutcome == GeometryIdentityOutcome::outOfMemory
                              ? OVSPATIALQUERY_API_OUT_OF_MEMORY
                              : (claimOutcome == GeometryIdentityOutcome::resourceExhausted
                                     ? OVSPATIALQUERY_API_HOST_BUDGET
                                     : OVSPATIALQUERY_API_INTERNAL));
        }
        prepared->claimResidentBytes = prepared->claimBatch.additionalResidentByteCount();
        prepared->transientResidentBytes =
            saturatingResidentByteAdd(transientResident, prepared->claimBatch.stagingResidentByteCount(), 1u);
        if (droppingMeshes > kNamedMeshes)
        {
            dropReport += "; and " + std::to_string(droppingMeshes - kNamedMeshes) +
                          " more definitions dropped triangles without area";
        }
        if (outDropReport != nullptr)
        {
            *outDropReport = std::move(dropReport);
        }
        outRegistration.mImpl = std::move(prepared);
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

ovspatialquery_api_status_t ExplicitPreparationRegistry::commitRegistration(
    const ExplicitPreparedRegistration& registration, std::vector<std::shared_ptr<void>> definitionMemoryOwners,
    std::shared_ptr<void> claimMemoryOwner) noexcept
{
    if (registration.mImpl == nullptr || registration.mImpl->definitions.empty() ||
        (!definitionMemoryOwners.empty() && definitionMemoryOwners.size() != registration.mImpl->definitions.size()))
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    try
    {
        std::lock_guard<std::mutex>             lock(mState->mutex);
        DefinitionMap                           candidate = mState->definitions;
        std::vector<PreparedTriangleDefinition> meshes;
        for (size_t index = 0u; index < registration.mImpl->definitions.size(); ++index)
        {
            const std::shared_ptr<const PreparedDefinition>& definition = registration.mImpl->definitions[index];
            if (definition == nullptr || candidate.find(definition->reference) != candidate.end())
            {
                return OVSPATIALQUERY_API_INVALID_ARGUMENT;
            }
            definition->memoryOwner =
                definitionMemoryOwners.empty() ? std::shared_ptr<void>{} : definitionMemoryOwners[index];
            candidate.emplace(definition->reference, definition);
            if (std::get_if<ExplicitTriangleMeshDefinition>(&definition->value) != nullptr)
            {
                if (definition->preparedTriangle == nullptr)
                {
                    return OVSPATIALQUERY_API_INTERNAL;
                }
                meshes.push_back({definition->reference.value, definition->preparedTriangle});
            }
        }
        if (!meshes.empty())
        {
            const ovspatialquery_api_status_t status = mState->meshCache.adoptDefinitions(meshes);
            if (status != OVSPATIALQUERY_API_SUCCESS)
            {
                return status;
            }
        }
        for (const std::shared_ptr<const PreparedDefinition>& definition : registration.mImpl->definitions)
        {
            definition->preparedTriangle.reset();
        }
        mState->definitions.swap(candidate);
        mState->claims->commit(registration.mImpl->claimBatch, std::move(claimMemoryOwner));
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

ovspatialquery_api_status_t ExplicitPreparationRegistry::droppedTriangleCount(uint64_t  value,
                                                                              uint64_t& outCount) const noexcept
{
    try
    {
        std::lock_guard<std::mutex> lock(mState->mutex);
        if (mState->definitions.count(GeometryReference{ExactGeometryFamily::triangleMesh, value}) == 0u)
        {
            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
        }
        return mState->meshCache.droppedTriangleCount(value, outCount);
    }
    catch (...)
    {
        return OVSPATIALQUERY_API_INTERNAL;
    }
}

ovspatialquery_api_status_t ExplicitPreparationRegistry::destroyDefinition(GeometryReference reference) noexcept
{
    try
    {
        std::lock_guard<std::mutex>   lock(mState->mutex);
        const DefinitionMap::iterator found = mState->definitions.find(reference);
        if (found == mState->definitions.end())
        {
            return OVSPATIALQUERY_API_INVALID_HANDLE;
        }
        if (found->second.use_count() != 1)
        {
            return OVSPATIALQUERY_API_BUSY;
        }
        if (reference.family == ExactGeometryFamily::triangleMesh)
        {
            const ovspatialquery_api_status_t status = mState->meshCache.destroyDefinition(reference.value);
            if (status != OVSPATIALQUERY_API_SUCCESS)
            {
                return status;
            }
        }
        mState->definitions.erase(found);
        return OVSPATIALQUERY_API_SUCCESS;
    }
    catch (...)
    {
        return OVSPATIALQUERY_API_INTERNAL;
    }
}

ovspatialquery_api_status_t ExplicitPreparationRegistry::retainDefinitions(
    const std::vector<GeometryReference>&     references,
    std::vector<std::shared_ptr<const void>>& outOwners) const noexcept
{
    try
    {
        std::vector<std::shared_ptr<const void>> owners;
        owners.reserve(references.size());
        std::lock_guard<std::mutex> lock(mState->mutex);
        for (GeometryReference reference : references)
        {
            const DefinitionMap::const_iterator found = mState->definitions.find(reference);
            if (found == mState->definitions.end())
            {
                return OVSPATIALQUERY_API_INVALID_ARGUMENT;
            }
            owners.push_back(found->second);
        }
        outOwners = std::move(owners);
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

ovspatialquery_api_status_t ExplicitPreparationRegistry::describeDeviceShapes(
    const std::vector<GeometryReference>& references, std::vector<DeviceShapeDescription>& outShapes,
    std::vector<std::shared_ptr<const void>>& outOwners) const noexcept
{
    try
    {
        std::vector<std::shared_ptr<const PreparedDefinition>> definitions;
        definitions.reserve(references.size());
        std::vector<DeviceShapeDescription> shapes;
        shapes.reserve(references.size());
        std::lock_guard<std::mutex> lock(mState->mutex);
        for (GeometryReference reference : references)
        {
            const DefinitionMap::const_iterator found = mState->definitions.find(reference);
            if (found == mState->definitions.end())
            {
                return OVSPATIALQUERY_API_INVALID_ARGUMENT;
            }
            definitions.push_back(found->second);
        }
        for (const std::shared_ptr<const PreparedDefinition>& entry : definitions)
        {
            const PreparedDefinition& definition = *entry;
            const GeometryReference   reference  = definition.reference;
            DeviceShapeDescription    shape;
            shape.value                        = reference.value;
            ovspatialquery_api_status_t status = OVSPATIALQUERY_API_SUCCESS;
            std::visit(
                [&](const auto& value)
                {
                    using Value = std::decay_t<decltype(value)>;
                    if constexpr (std::is_same_v<Value, ExplicitSphereDefinition>)
                    {
                        shape.type          = OVSPATIALQUERY_GEOMETRY_SPHERE;
                        shape.parameters[0] = value.radius;
                    }
                    else if constexpr (std::is_same_v<Value, ExplicitBoxDefinition>)
                    {
                        shape.type       = OVSPATIALQUERY_GEOMETRY_BOX;
                        shape.parameters = value.halfExtents;
                    }
                    else if constexpr (std::is_same_v<Value, ExplicitCapsuleDefinition>)
                    {
                        shape.type          = OVSPATIALQUERY_GEOMETRY_CAPSULE;
                        shape.parameters[0] = value.radius;
                        shape.parameters[1] = value.halfSegment;
                    }
                    else if constexpr (std::is_same_v<Value, ExplicitConvexDefinition>)
                    {
                        // The device derives the bounds the placed hull derives on the
                        // host, which needs the hull's prepared surface.
                        shape.type        = OVSPATIALQUERY_GEOMETRY_CONVEX;
                        shape.hull        = definition.preparedConvex;
                        shape.memoryOwner = definition.memoryOwner;
                        if (shape.hull == nullptr || shape.hull->preparedSurface() == nullptr ||
                            shape.hull->localGeometry() == nullptr)
                        {
                            status = OVSPATIALQUERY_API_NOT_SUPPORTED;
                        }
                    }
                    else if constexpr (std::is_same_v<Value, ExplicitTriangleMeshDefinition>)
                    {
                        shape.type = OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH;
                        status     = mState->meshCache.localBounds(reference.value, shape.localBounds);
                    }
                },
                definition.value);
            if (status != OVSPATIALQUERY_API_SUCCESS)
            {
                return status;
            }
            shapes.push_back(std::move(shape));
        }
        std::vector<std::shared_ptr<const void>> owners(definitions.begin(), definitions.end());
        outShapes = std::move(shapes);
        outOwners = std::move(owners);
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

ovspatialquery_api_status_t completePoseRows(const GenerationLease& base, bool rowsCarryScale, bool rowsCarryBits,
                                             std::vector<ExplicitInstanceInput>& rows) noexcept
{
    if (!base.initialized())
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    if (base.preparedOnDevice())
    {
        return OVSPATIALQUERY_API_NOT_SUPPORTED; // as prepareUpdate, which both callers reach next (ADR-0024)
    }
    // Explicit heads hold their objects in strictly ascending ID order.
    const ObjectTable& objects = base.objects();
    for (ExplicitInstanceInput& row : rows)
    {
        const auto found =
            std::lower_bound(objects.begin(), objects.end(), row.objectValue,
                             [](const PopulationObject& object, uint64_t id) { return object.id.value < id; });
        if (found == objects.end() || found->id.value != row.objectValue)
        {
            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
        }
        row.geometry   = found->geometryReference;
        row.materialId = found->materialId;
        if (!rowsCarryBits)
        {
            row.participationBits = found->participationBits;
        }
        if (!rowsCarryScale)
        {
            row.uniformScale = found->uniformScale;
        }
    }
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t ExplicitPreparationRegistry::prepareObjects(
    const std::vector<ExplicitInstanceInput>& instances, std::vector<PopulationObject>& outObjects) const
{
    // Objects are created in ascending ID order. Rows usually arrive in that
    // order, which also proves their IDs unique; other orders sort compact keys
    // once rather than sorting the created objects. `order` stays empty for
    // rows that already ascend.
    bool ascending = true;
    for (size_t row = 1u; row < instances.size() && ascending; ++row)
    {
        ascending = instances[row - 1u].objectValue < instances[row].objectValue;
    }
    std::vector<size_t> order;
    if (ascending)
    {
        if (!instances.empty() && instances.front().objectValue == 0u)
        {
            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
        }
    }
    else
    {
        std::vector<std::pair<uint64_t, size_t>> keys;
        keys.reserve(instances.size());
        for (size_t row = 0u; row < instances.size(); ++row)
        {
            keys.emplace_back(instances[row].objectValue, row);
        }
        std::sort(keys.begin(), keys.end());
        order.reserve(keys.size());
        for (size_t index = 0u; index < keys.size(); ++index)
        {
            if (keys[index].first == 0u || (index != 0u && keys[index - 1u].first == keys[index].first))
            {
                return OVSPATIALQUERY_API_INVALID_ARGUMENT;
            }
            order.push_back(keys[index].second);
        }
    }
    // Consecutive rows usually share a definition; resolve each run once.
    std::vector<std::shared_ptr<const PreparedDefinition>> definitions;
    std::vector<size_t>                                    rowDefinitions(instances.size());
    {
        std::lock_guard<std::mutex> lock(mState->mutex);
        for (size_t row = 0u; row < instances.size(); ++row)
        {
            if (definitions.empty() || definitions.back()->reference != instances[row].geometry)
            {
                const DefinitionMap::const_iterator found = mState->definitions.find(instances[row].geometry);
                if (found == mState->definitions.end())
                {
                    return OVSPATIALQUERY_API_INVALID_ARGUMENT;
                }
                definitions.push_back(found->second);
            }
            rowDefinitions[row] = definitions.size() - 1u;
        }
    }
    std::vector<PopulationObject>& objects = outObjects;
    objects.clear();
    objects.reserve(instances.size());
    for (size_t index = 0u; index < instances.size(); ++index)
    {
        const size_t                                     row        = order.empty() ? index : order[index];
        const std::shared_ptr<const PreparedDefinition>& definition = definitions[rowDefinitions[row]];
        PopulationObject                                 object;
        const ovspatialquery_api_status_t                status =
            createInstanceGeometry(instances[row], *definition, mState->meshCache, object);
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return status;
        }
        object.definitionOwner = definition;
        objects.push_back(std::move(object));
    }
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t ExplicitPreparationRegistry::prepareSet(const std::vector<ExplicitInstanceInput>& instances,
                                                                    ExplicitPreparedUpdate& outUpdate) const noexcept
{
    try
    {
        std::vector<PopulationObject>     objects;
        const ovspatialquery_api_status_t status = prepareObjects(instances, objects);
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return status;
        }
        std::unique_ptr<ExplicitPreparedUpdate::Impl> prepared(new ExplicitPreparedUpdate::Impl());
        prepared->objects = ObjectTable(std::move(objects));
        outUpdate.mImpl   = std::move(prepared);
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

ovspatialquery_api_status_t ExplicitPreparationRegistry::prepareUpdate(
    const GenerationLease& base, ovspatialquery_ordinal_t requestedBaseOrdinal,
    const std::vector<ExplicitInstanceInput>& changedInstances, const std::vector<uint64_t>& removedObjectValues,
    ExplicitPreparedUpdate& outUpdate) const noexcept
{
    if (!base.initialized())
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    if (base.preparedOnDevice())
    {
        return OVSPATIALQUERY_API_NOT_SUPPORTED; // a version does not move from the device to the host (ADR-0024)
    }
    if (requestedBaseOrdinal != base.ordinal())
    {
        return OVSPATIALQUERY_API_ORDINAL_MISMATCH;
    }
    try
    {
        // prepareObjects validates unique, nonzero changed IDs and returns its
        // objects in ID order, so every lookup below walks forward through sorted IDs.
        std::vector<PopulationObject>     changedObjects;
        const ovspatialquery_api_status_t changedStatus = prepareObjects(changedInstances, changedObjects);
        if (changedStatus != OVSPATIALQUERY_API_SUCCESS)
        {
            return changedStatus;
        }
        // Explicit heads hold their objects in strictly ascending, nonzero ID order.
        const ObjectTable* baseObjects = &base.objects();
        // Where each changed object lands in the base; a new ID must not have been used before.
        const std::vector<uint64_t>& usedObjectValues = base.usedObjectValues();
        std::vector<size_t>          changedPositions(changedObjects.size());
        size_t                       insertedCount = 0u;
        size_t                       position      = 0u;
        for (size_t index = 0u; index < changedObjects.size(); ++index)
        {
            const uint64_t value    = changedObjects[index].id.value;
            position                = lowerBoundObjectFrom(*baseObjects, position, value);
            changedPositions[index] = position;
            if (position == baseObjects->size() || (*baseObjects)[position].id.value != value)
            {
                if (std::binary_search(usedObjectValues.begin(), usedObjectValues.end(), value))
                {
                    return OVSPATIALQUERY_API_INVALID_ARGUMENT;
                }
                ++insertedCount;
            }
        }
        // Removed IDs are unique, present in the base and absent from the changed rows.
        std::vector<uint64_t> removedValues(removedObjectValues);
        std::sort(removedValues.begin(), removedValues.end());
        std::vector<size_t> removedPositions(removedValues.size());
        position               = 0u;
        size_t changedPosition = 0u;
        for (size_t index = 0u; index < removedValues.size(); ++index)
        {
            const uint64_t value = removedValues[index];
            position             = lowerBoundObjectFrom(*baseObjects, position, value);
            changedPosition      = lowerBoundObjectFrom(changedObjects, changedPosition, value);
            if (value == 0u || (index != 0u && removedValues[index - 1u] == value) || position == baseObjects->size() ||
                (*baseObjects)[position].id.value != value ||
                (changedPosition != changedObjects.size() && changedObjects[changedPosition].id.value == value))
            {
                return OVSPATIALQUERY_API_INVALID_ARGUMENT;
            }
            removedPositions[index] = position;
        }
        const std::optional<uint64_t> exactGeometryCharge =
            adjustedExactGeometryCharge(base, *baseObjects, changedObjects, changedPositions, removedPositions);
        // Positions stay put unless a removal or an insertion before existing
        // objects renumbers them; new objects after the base's last one take
        // the next positions. A row equal to its base object is no change, so a
        // caller resending every row records only what differs; past a quarter
        // changed the record says everything.
        bool         renumbered = !removedValues.empty();
        const size_t limit      = VersionChangeRecord::positionLimit(baseObjects->size() + insertedCount);
        auto         slotOf     = [&changedPositions, baseObjects, next = baseObjects->size()](size_t index) mutable
        { return changedPositions[index] == baseObjects->size() ? next++ : changedPositions[index]; };
        VersionChangeRecord changes;
        if (!renumbered)
        {
            changes.everything  = false;
            changes.baseOrdinal = base.ordinal();
            auto recordSlotOf   = slotOf;
            for (size_t index = 0u; index < changedObjects.size(); ++index)
            {
                const size_t slot = recordSlotOf(index);
                if (slot < baseObjects->size() && (*baseObjects)[slot].id.value != changedObjects[index].id.value)
                {
                    renumbered = true;
                    changes    = VersionChangeRecord{};
                    break;
                }
                if (slot < baseObjects->size() && !objectChanged((*baseObjects)[slot], changedObjects[index]))
                {
                    continue;
                }
                if (changes.positions.size() == limit)
                {
                    changes = VersionChangeRecord{};
                    break;
                }
                changes.positions.push_back(static_cast<uint32_t>(slot));
            }
        }
        // A few rows over stable positions go into new chunks that share the
        // base's other chunks (ADR-0019); anything else is one contiguous
        // table, as a complete set is.
        ObjectTable objects;
        if (!renumbered && changedObjects.size() <= limit)
        {
            // At most `limit` rows, so the loop above checked every one.
            std::vector<ObjectTable::Row> rows;
            rows.reserve(changedObjects.size());
            for (size_t index = 0u; index < changedObjects.size(); ++index)
            {
                rows.push_back({static_cast<uint32_t>(slotOf(index)), std::move(changedObjects[index])});
            }
            objects = ObjectTable::derive(*baseObjects, std::move(rows));
        }
        else
        {
            std::vector<PopulationObject> merged;
            if (insertedCount == 0u && removedValues.empty() && changedObjects.size() == baseObjects->size())
            {
                // Every base object is replaced; the changed objects are the new head.
                merged = std::move(changedObjects);
            }
            else
            {
                // Copy unchanged runs of the base between changed and removed positions.
                merged.reserve(baseObjects->size() - removedValues.size() + insertedCount);
                size_t     baseIndex    = 0u;
                size_t     removedIndex = 0u;
                const auto copyBaseUntil =
                    [&merged, baseObjects, &baseIndex, &removedIndex, &removedPositions](size_t end)
                {
                    for (; removedIndex < removedPositions.size() && removedPositions[removedIndex] < end;
                         ++removedIndex)
                    {
                        merged.insert(merged.end(), baseObjects->begin() + static_cast<std::ptrdiff_t>(baseIndex),
                                      baseObjects->begin() +
                                          static_cast<std::ptrdiff_t>(removedPositions[removedIndex]));
                        baseIndex = removedPositions[removedIndex] + 1u;
                    }
                    merged.insert(merged.end(), baseObjects->begin() + static_cast<std::ptrdiff_t>(baseIndex),
                                  baseObjects->begin() + static_cast<std::ptrdiff_t>(end));
                    baseIndex = end;
                };
                for (size_t index = 0u; index < changedObjects.size(); ++index)
                {
                    copyBaseUntil(changedPositions[index]);
                    if (baseIndex < baseObjects->size() &&
                        (*baseObjects)[baseIndex].id.value == changedObjects[index].id.value)
                    {
                        ++baseIndex;
                    }
                    merged.push_back(std::move(changedObjects[index]));
                }
                copyBaseUntil(baseObjects->size());
            }
            objects = ObjectTable(std::move(merged));
        }
        std::unique_ptr<ExplicitPreparedUpdate::Impl> prepared(new ExplicitPreparedUpdate::Impl());
        prepared->objects             = std::move(objects);
        prepared->exactGeometryCharge = exactGeometryCharge;
        prepared->changes             = std::move(changes);
        outUpdate.mImpl               = std::move(prepared);
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

PopulationSourceBinding ExplicitPreparationRegistry::sourceBinding() const noexcept
{
    PopulationSourceBinding binding;
    binding.sourceState = mState;
    binding.preparation = PopulationPreparationStrategy::typedFiveFamily;
    return binding;
}

size_t ExplicitPreparationRegistry::definitionCount() const noexcept
{
    std::lock_guard<std::mutex> lock(mState->mutex);
    return mState->definitions.size();
}

} // namespace detail
} // namespace ovspatialquery
