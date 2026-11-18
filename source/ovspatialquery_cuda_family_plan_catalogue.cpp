/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_cuda_family_plan_internal.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace ovspatialquery
{
namespace detail
{

#if defined(OVSPATIALQUERY_HAS_CUDA_BACKEND)
namespace
{

struct CudaFamilyResultSlot : CudaResultSlotEvents
{
    std::vector<CuDevicePtr> allocations;
    CuDevicePtr              queryStatus    = 0u;
    CuDevicePtr              storedHitCount = 0u;
    CuDevicePtr              hitOffsets     = 0u;
    CuDevicePtr              hitCounts      = 0u;
    CuDevicePtr              truncated      = 0u;
    CuDevicePtr              objectIds      = 0u;
    CuDevicePtr              geometryTypes  = 0u;
    CuDevicePtr              geometryValues = 0u;
    CuDevicePtr              materialIds    = 0u;
    CuDevicePtr              positions      = 0u;
    CuDevicePtr              queryPositions = 0u;
    CuDevicePtr              normals        = 0u;
    CuDevicePtr              distances      = 0u;
    CuDevicePtr              featureKinds   = 0u;
    CuDevicePtr              faceIndices    = 0u;
    CuDevicePtr              corners        = 0u;
    CuDevicePtr              barycentrics   = 0u;
    CuDevicePtr              rowPlans       = 0u; // cuda_family::RowPlan per query
    CuDevicePtr              stash          = 0u; // kStashRowBytes per query for closest/any plans
    // Mapped pinned copy of storedHitCount: the scan writes it with H, so a fetch
    // reads H without a transfer. Its device address is storedHitCountMirror.
    uint64_t*   hostStoredHitCount   = nullptr;
    CuDevicePtr storedHitCountMirror = 0u;
    // A HOST_MAPPED plan's mapped block for this slot (ADR-0077), laid out by
    // the plan's CudaFamilyMappedLayout; the host and the device share its address.
    CuDevicePtr mapped = 0u;
};

struct CudaFamilyColumnMetadata
{
    void set(uint64_t queryCount, uint64_t hitCount) noexcept
    {
        queryShape[0] = static_cast<int64_t>(queryCount);
        hitShape[0]   = static_cast<int64_t>(hitCount);
        vec3Shape[0]  = static_cast<int64_t>(hitCount);
    }

    std::array<int64_t, 1u> scalarShape{1};
    std::array<int64_t, 1u> scalarStride{1};
    std::array<int64_t, 1u> queryShape{0};
    std::array<int64_t, 1u> queryStride{1};
    std::array<int64_t, 1u> hitShape{0};
    std::array<int64_t, 1u> hitStride{1};
    std::array<int64_t, 2u> vec3Shape{0, 3};
    std::array<int64_t, 2u> vec3Stride{3, 1};
};

class CudaFamilyFamilyPlan;

struct CudaFamilyCatalogueCounts
{
    uint64_t objectIds       = 0u;
    uint64_t geometryTypes   = 0u;
    uint64_t geometryValues  = 0u;
    uint64_t objectMaterials = 0u;
    uint64_t faceMaterials   = 0u;
    uint64_t objectBits      = 0u;
    uint64_t objectMinimums  = 0u;
    uint64_t objectMaximums  = 0u;
    uint64_t exactObjects    = 0u;
    uint64_t spheres         = 0u;
    uint64_t boxes           = 0u;
    uint64_t capsules        = 0u;
    uint64_t convexes        = 0u;
    uint64_t convexVertices  = 0u;
    uint64_t convexTriangles = 0u;
    uint64_t convexPlanes    = 0u;
    uint64_t convexEdges     = 0u;
    uint64_t meshes          = 0u;
    uint64_t meshDefinitions = 0u;
};

constexpr uint64_t kCudaMeshBvhLeafTriangles = 32u;
// Catalogues a plan keeps: the generation it last queried and the one before, so a
// consumer alternating between a new head and the retained previous head rebuilds
// neither, and the previous generation's mesh definitions stay resident for the next
// head's meshes to share their topology.
constexpr size_t kCudaRetainedCatalogues = 2u;
// Device arrays one catalogue owns for its tables (one packed allocation), and
// one resident mesh definition owns.
constexpr uint64_t kCatalogueTableAllocations = 1u;
constexpr uint64_t kDefinitionAllocations     = 6u;
// Device arrays one result slot owns at most (CudaFamilyResultSlot's columns).
constexpr uint64_t kResultSlotAllocations = 19u;

// A copied input column's device buffer and its size, as the plan's staging pool gave it.
using CopiedInputAllocation = std::pair<CuDevicePtr, size_t>;

// The launch columns an operation retains or copies, in staging order.
template <typename Launch>
auto cudaFamilyRetainedInputs(Launch& launch) noexcept
{
    return std::array{
        &launch.queryBits, &launch.positions,        &launch.directionsOrRotations,
        &launch.rotations, &launch.minimumDistances, &launch.maximumDistances,
        &launch.radii,     &launch.rowMaximumHits,   &launch.halfExtents,
    };
}

// A copied column, or a borrowed one the kernels cannot read in place (ADR-0053).
bool cudaFamilyInputCopied(const CudaFamilyInput& input) noexcept
{
    return input.present && (input.ownership == 1u || !input.layout.compact);
}

bool cudaFamilyInputBytes(const CudaFamilyInput& input, size_t& out) noexcept
{
    out = 0u;
    if (!input.present)
    {
        return true;
    }
    const DLTensor& tensor = input.tensor;
    if (tensor.shape == nullptr || tensor.ndim < 1 || tensor.ndim > 2 || tensor.dtype.bits == 0u ||
        (tensor.dtype.bits % 8u) != 0u)
    {
        return false;
    }
    size_t elements = 1u;
    for (int32_t dimension = 0; dimension < tensor.ndim; ++dimension)
    {
        if (tensor.shape[dimension] < 0 || (elements != 0u && static_cast<uint64_t>(tensor.shape[dimension]) >
                                                                  std::numeric_limits<size_t>::max() / elements))
        {
            return false;
        }
        elements *= static_cast<size_t>(tensor.shape[dimension]);
    }
    const size_t elementBytes = tensor.dtype.bits / 8u;
    if (elements != 0u && elementBytes > std::numeric_limits<size_t>::max() / elements)
    {
        return false;
    }
    out = elements * elementBytes;
    return true;
}

ovspatialquery_api_status_t measureCudaFamilyCopiedInputs(const CudaFamilyLaunch& launch, uint64_t& outBytes,
                                                          uint64_t& outAllocationCount) noexcept
{
    outBytes           = 0u;
    outAllocationCount = 0u;
    for (const CudaFamilyInput* input : cudaFamilyRetainedInputs(launch))
    {
        if (!cudaFamilyInputCopied(*input))
        {
            continue;
        }
        ++outAllocationCount;
        size_t bytes = 0u;
        if (!cudaFamilyInputBytes(*input, bytes) || static_cast<uint64_t>(bytes) > UINT64_MAX - outBytes)
        {
            outBytes = 0u;
            return OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        outBytes += static_cast<uint64_t>(bytes);
    }
    return OVSPATIALQUERY_API_SUCCESS;
}

struct MeshDefinitionKey
{
    uintptr_t vertices      = 0u;
    uintptr_t triangles     = 0u;
    uintptr_t faces         = 0u;
    uintptr_t corners       = 0u;
    uint64_t  vertexCount   = 0u;
    uint64_t  triangleCount = 0u;

    bool operator<(const MeshDefinitionKey& other) const noexcept
    {
        return std::tie(vertices, triangles, faces, corners, vertexCount, triangleCount) <
               std::tie(other.vertices, other.triangles, other.faces, other.corners, other.vertexCount,
                        other.triangleCount);
    }
    bool operator==(const MeshDefinitionKey& other) const noexcept
    {
        return std::tie(vertices, triangles, faces, corners, vertexCount, triangleCount) ==
               std::tie(other.vertices, other.triangles, other.faces, other.corners, other.vertexCount,
                        other.triangleCount);
    }
};

// Identity of a definition's topology: the index arrays a prepared mesh shares with the
// meshes derived from it under new vertices (see TriangleMeshSource::topologyBase).
struct MeshTopologyKey
{
    uintptr_t triangles     = 0u;
    uintptr_t faces         = 0u;
    uintptr_t corners       = 0u;
    uint64_t  vertexCount   = 0u;
    uint64_t  triangleCount = 0u;

    bool operator<(const MeshTopologyKey& other) const noexcept
    {
        return std::tie(triangles, faces, corners, vertexCount, triangleCount) <
               std::tie(other.triangles, other.faces, other.corners, other.vertexCount, other.triangleCount);
    }
};

MeshTopologyKey meshTopologyKey(const TriangleMeshExactGeometryDescription& mesh) noexcept
{
    return {reinterpret_cast<uintptr_t>(mesh.triangleVertexIndices), reinterpret_cast<uintptr_t>(mesh.sourceFaces),
            reinterpret_cast<uintptr_t>(mesh.sourceCorners), mesh.vertexCount, mesh.triangleCount};
}

MeshDefinitionKey meshDefinitionKey(const TriangleMeshExactGeometryDescription& mesh) noexcept
{
    return {reinterpret_cast<uintptr_t>(mesh.localVertices),
            reinterpret_cast<uintptr_t>(mesh.triangleVertexIndices),
            reinterpret_cast<uintptr_t>(mesh.sourceFaces),
            reinterpret_cast<uintptr_t>(mesh.sourceCorners),
            mesh.vertexCount,
            mesh.triangleCount};
}

bool addBvhNodeCount(uint64_t triangleCount, uint64_t& total) noexcept
{
    uint64_t pending[64] = {};
    uint32_t count       = 1u;
    pending[0]           = triangleCount;
    while (count != 0u)
    {
        const uint64_t triangles = pending[--count];
        if (total == UINT64_MAX)
        {
            return false;
        }
        ++total;
        if (triangles > kCudaMeshBvhLeafTriangles)
        {
            if (count > 61u)
            {
                return false;
            }
            const uint64_t left = triangles / 2u;
            pending[count++]    = left;
            pending[count++]    = triangles - left;
        }
    }
    return true;
}

bool invertMeshLinear(const std::array<double, 16u>& worldFromLocal, double output[9]) noexcept
{
    double linearScale = 0.0;
    for (const size_t index : {0u, 1u, 2u, 4u, 5u, 6u, 8u, 9u, 10u})
    {
        linearScale = std::max(linearScale, std::abs(worldFromLocal[index]));
    }
    if (linearScale == 0.0)
    {
        return false;
    }
    const double a = worldFromLocal[0] / linearScale, b = worldFromLocal[1] / linearScale;
    const double c = worldFromLocal[2] / linearScale, d = worldFromLocal[4] / linearScale;
    const double e = worldFromLocal[5] / linearScale, f = worldFromLocal[6] / linearScale;
    const double g = worldFromLocal[8] / linearScale, h = worldFromLocal[9] / linearScale;
    const double i           = worldFromLocal[10] / linearScale;
    const double determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if (!std::isfinite(determinant) || determinant == 0.0)
    {
        return false;
    }
    const double inverse = 1.0 / determinant / linearScale;
    output[0]            = (e * i - f * h) * inverse;
    output[1]            = (c * h - b * i) * inverse;
    output[2]            = (b * f - c * e) * inverse;
    output[3]            = (f * g - d * i) * inverse;
    output[4]            = (a * i - c * g) * inverse;
    output[5]            = (c * d - a * f) * inverse;
    output[6]            = (d * h - e * g) * inverse;
    output[7]            = (b * g - a * h) * inverse;
    output[8]            = (a * e - b * d) * inverse;
    return std::all_of(output, output + 9u, [](double value) { return std::isfinite(value); });
}

// Sets every node's bounds from the vertices of its triangles: a leaf's from its
// triangles in leaf order, rounded outward, and a parent's as the union of its
// children's. Children follow their parent in the node array, so one pass from the
// back sees every child before its parent. The outward rounding is increasing, so a
// parent's union equals its triangles' bounds rounded outward.
bool setMeshBvhBounds(const TriangleMeshExactGeometryDescription& mesh, const std::vector<uint32_t>& traversalTriangles,
                      std::vector<cuda_family::MeshBvhNode>& nodes)
{
    const auto component = [&mesh](uint32_t triangle, uint32_t corner, uint32_t axis)
    {
        const uint32_t vertex = mesh.triangleVertexIndices[static_cast<size_t>(triangle) * 3u + corner];
        return mesh.localVertices[static_cast<size_t>(vertex) * 3u + axis];
    };
    for (size_t index = nodes.size(); index-- > 0u;)
    {
        cuda_family::MeshBvhNode& node = nodes[index];
        float minimum[3] = {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::infinity()};
        float maximum[3] = {-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity()};
        if (node.count != 0u)
        {
            if (node.first > traversalTriangles.size() || node.count > traversalTriangles.size() - node.first)
            {
                return false;
            }
            for (uint64_t ordered = node.first; ordered < node.first + node.count; ++ordered)
            {
                const uint32_t triangle = traversalTriangles[static_cast<size_t>(ordered)];
                if (triangle >= mesh.triangleCount)
                {
                    return false;
                }
                for (uint32_t axis = 0u; axis < 3u; ++axis)
                {
                    const float first  = component(triangle, 0u, axis);
                    const float second = component(triangle, 1u, axis);
                    const float third  = component(triangle, 2u, axis);
                    minimum[axis]      = std::min(minimum[axis], std::min(first, std::min(second, third)));
                    maximum[axis]      = std::max(maximum[axis], std::max(first, std::max(second, third)));
                }
            }
            for (uint32_t axis = 0u; axis < 3u; ++axis)
            {
                node.minimum[axis] = std::nextafter(minimum[axis], -std::numeric_limits<float>::infinity());
                node.maximum[axis] = std::nextafter(maximum[axis], std::numeric_limits<float>::infinity());
            }
            continue;
        }
        if (node.first <= index || node.first + 1u >= nodes.size())
        {
            return false;
        }
        const cuda_family::MeshBvhNode& left  = nodes[static_cast<size_t>(node.first)];
        const cuda_family::MeshBvhNode& right = nodes[static_cast<size_t>(node.first) + 1u];
        for (uint32_t axis = 0u; axis < 3u; ++axis)
        {
            node.minimum[axis] = std::min(left.minimum[axis], right.minimum[axis]);
            node.maximum[axis] = std::max(left.maximum[axis], right.maximum[axis]);
        }
    }
    return true;
}

// One triangle as the builder partitions it: its index and the sum of its corners on
// each axis, three times its centroid, which orders the median splits. The records
// move with the partition, so each level reads them in order instead of looking up
// every corner again.
struct MeshBvhBuildRecord
{
    uint32_t triangle;
    float    sum[3];
};
constexpr uint64_t kMeshBvhBuildRecordBytes = sizeof(MeshBvhBuildRecord);

// One node still to split, its triangles' records [begin, end), and the index its
// first child takes when nodes are numbered in the builder's order.
struct MeshBvhBuildTask
{
    uint64_t node  = 0u;
    size_t   begin = 0u;
    size_t   end   = 0u;
    uint64_t next  = 0u;
};

// The node count of a median-split tree over `triangles`: the numbering, and so each
// subtree's block of node indices, follows from the triangle counts alone.
uint64_t meshBvhNodeCount(uint64_t triangles) noexcept
{
    uint64_t count = 0u;
    return addBvhNodeCount(triangles, count) ? count : 0u;
}

// Sorts the records of an internal node around its median along the widest extent
// of their centroids, ties by triangle index; returns the median's position.
// Dividing by three is increasing, so the sums' bounds divided by three are the
// centroids' bounds.
size_t splitMeshBvhNode(std::vector<MeshBvhBuildRecord>& records, size_t begin, size_t end) noexcept
{
    float minimum[3] = {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::infinity()};
    float maximum[3] = {-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity()};
    for (size_t ordered = begin; ordered < end; ++ordered)
    {
        for (uint32_t axis = 0u; axis < 3u; ++axis)
        {
            minimum[axis] = std::min(minimum[axis], records[ordered].sum[axis]);
            maximum[axis] = std::max(maximum[axis], records[ordered].sum[axis]);
        }
    }
    float extent[3] = {};
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        extent[axis] = maximum[axis] / 3.0f - minimum[axis] / 3.0f;
    }
    uint32_t splitAxis = 0u;
    for (uint32_t axis = 1u; axis < 3u; ++axis)
    {
        if (extent[axis] > extent[splitAxis])
        {
            splitAxis = axis;
        }
    }
    const size_t middle = begin + (end - begin) / 2u;
    std::nth_element(records.begin() + static_cast<ptrdiff_t>(begin), records.begin() + static_cast<ptrdiff_t>(middle),
                     records.begin() + static_cast<ptrdiff_t>(end),
                     [splitAxis](const MeshBvhBuildRecord& left, const MeshBvhBuildRecord& right)
                     {
                         return left.sum[splitAxis] < right.sum[splitAxis] ||
                                (left.sum[splitAxis] == right.sum[splitAxis] && left.triangle < right.triangle);
                     });
    return middle;
}

// Splits nodes depth first, left child first, numbering each node's children with the
// next two free indices. A node whose triangles fit `grain` is left in `deferred`
// with the block of indices its subtree takes in that order, so building it later,
// on any thread, numbers its nodes as building it in place would. A leaf's triangles
// are its records' range, which the leaf order keeps.
bool buildMeshBvhNodes(std::vector<MeshBvhBuildRecord>& records, std::vector<cuda_family::MeshBvhNode>& nodes,
                       MeshBvhBuildTask root, size_t grain, std::vector<MeshBvhBuildTask>* deferred)
{
    // Depth first holds one pending right sibling per level, and a level halves its triangles.
    std::array<MeshBvhBuildTask, 72u> tasks{};
    size_t                            pending = 0u;
    uint64_t                          next    = root.next;
    tasks[pending++]                          = root;
    while (pending != 0u)
    {
        const MeshBvhBuildTask task = tasks[--pending];
        if (task.begin >= task.end || task.node >= nodes.size())
        {
            return false;
        }
        cuda_family::MeshBvhNode& node          = nodes[static_cast<size_t>(task.node)];
        const size_t              triangleCount = task.end - task.begin;
        if (triangleCount <= kCudaMeshBvhLeafTriangles)
        {
            node.first = task.begin;
            node.count = static_cast<uint32_t>(triangleCount);
            continue;
        }
        if (deferred != nullptr && triangleCount <= grain)
        {
            deferred->push_back({task.node, task.begin, task.end, next});
            next += meshBvhNodeCount(triangleCount) - 1u;
            continue;
        }
        const size_t middle = splitMeshBvhNode(records, task.begin, task.end);
        if (next + 2u > nodes.size() || pending + 2u > tasks.size())
        {
            return false;
        }
        node.first = next;
        node.count = 0u;
        next += 2u;
        tasks[pending++] = {node.first + 1u, middle, task.end, 0u};
        tasks[pending++] = {node.first, task.begin, middle, 0u};
    }
    return true;
}

// Builds the median-split tree. The caller splits the top levels; the subtrees below
// them are independent and run on `workers` too when the pool has idle workers. The
// tree is the same whoever builds which subtree.
bool buildMeshBvh(const TriangleMeshExactGeometryDescription& mesh, std::vector<cuda_family::MeshBvhNode>& nodes,
                  std::vector<uint32_t>& traversalTriangles, const SharedWorkerLoan& workers)
{
    if (mesh.triangleCount == 0u || mesh.triangleCount - 1u > UINT32_MAX ||
        mesh.triangleCount > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) || !nodes.empty())
    {
        return false;
    }
    const uint64_t nodeCount = meshBvhNodeCount(mesh.triangleCount);
    if (nodeCount == 0u || nodeCount > nodes.capacity())
    {
        return false;
    }
    std::vector<MeshBvhBuildRecord> records(static_cast<size_t>(mesh.triangleCount));
    const auto                      component = [&mesh](size_t triangle, uint32_t corner, uint32_t axis)
    {
        const uint32_t vertex = mesh.triangleVertexIndices[triangle * 3u + corner];
        return mesh.localVertices[static_cast<size_t>(vertex) * 3u + axis];
    };
    for (size_t triangle = 0u; triangle < records.size(); ++triangle)
    {
        MeshBvhBuildRecord& record = records[triangle];
        record.triangle            = static_cast<uint32_t>(triangle);
        for (uint32_t axis = 0u; axis < 3u; ++axis)
        {
            record.sum[axis] =
                component(triangle, 0u, axis) + component(triangle, 1u, axis) + component(triangle, 2u, axis);
        }
    }
    nodes.resize(static_cast<size_t>(nodeCount));
    // About 64 subtrees spread the build over a pool; small meshes stay on the caller.
    constexpr size_t              kMinimumSubtreeTriangles = 4096u;
    const size_t                  grain                    = std::max(kMinimumSubtreeTriangles, records.size() / 64u);
    std::vector<MeshBvhBuildTask> subtrees;
    if (!buildMeshBvhNodes(records, nodes, {0u, 0u, records.size(), 1u}, grain, &subtrees))
    {
        return false;
    }
    struct SubtreeBuild
    {
        std::vector<MeshBvhBuildRecord>*       records;
        std::vector<cuda_family::MeshBvhNode>* nodes;
        const std::vector<MeshBvhBuildTask>*   subtrees;
        std::atomic<bool>                      failed{false};
    } build{&records, &nodes, &subtrees};
    const auto buildSubtree = [](void* state, size_t index)
    {
        SubtreeBuild& work = *static_cast<SubtreeBuild*>(state);
        if (!buildMeshBvhNodes(*work.records, *work.nodes, (*work.subtrees)[index], 0u, nullptr))
        {
            work.failed.store(true);
        }
    };
    if (workers.run != nullptr && subtrees.size() > 1u)
    {
        workers.run(workers.state, subtrees.size(), buildSubtree, &build);
    }
    else
    {
        for (size_t index = 0u; index < subtrees.size(); ++index)
        {
            buildSubtree(&build, index);
        }
    }
    if (build.failed.load())
    {
        return false;
    }
    traversalTriangles.resize(records.size());
    for (size_t ordered = 0u; ordered < records.size(); ++ordered)
    {
        traversalTriangles[ordered] = records[ordered].triangle;
    }
    return setMeshBvhBounds(mesh, traversalTriangles, nodes);
}

// The host side of one definition's tree: the nodes (structure and the bounds they
// were built with) and the leaf order. A launch builds it; the resident definition
// keeps it for derived definitions to refit.
struct MeshBvhStructure
{
    std::vector<cuda_family::MeshBvhNode> nodes;
    std::vector<uint32_t>                 traversalTriangles;
};

// Recomputes the node bounds of `structure` for the mesh's vertices, keeping its
// splits and leaf order. Large deformations leave the splits where the base mesh
// put them, and a query then visits more nodes.
bool refitMeshBvh(const TriangleMeshExactGeometryDescription& mesh, const MeshBvhStructure& structure,
                  std::vector<cuda_family::MeshBvhNode>& outNodes)
{
    if (structure.nodes.empty() || structure.traversalTriangles.size() != mesh.triangleCount)
    {
        return false;
    }
    outNodes = structure.nodes;
    return setMeshBvhBounds(mesh, structure.traversalTriangles, outNodes);
}

// Frees one owner's device arrays and readiness event on the plan's context. A
// lost device abandons them and records the loss for the plan.
void releaseCatalogueDeviceStorage(Driver* driver, CuContext context, std::atomic<bool>* deviceLost,
                                   const std::vector<CuDevicePtr>& allocations, CuEvent ready) noexcept
{
    if (driver == nullptr || deviceLost == nullptr || deviceLost->load(std::memory_order_relaxed))
    {
        return;
    }
    CudaCurrentContext scope(*driver, context, *deviceLost);
    CuResult           status = scope.status();
    for (size_t index = 0u; status == kCuSuccess && index < allocations.size(); ++index)
    {
        status = scope.observe(driver->memFree(allocations[index]));
    }
    if (status == kCuSuccess && ready != nullptr)
    {
        (void)scope.observe(driver->eventDestroy(ready));
    }
}

// One mesh definition resident on the device: the local arrays and BVH the
// kernel reads through cuda_family::MeshDefinition. Catalogues of successive
// generations, and of the instance's other plans on the device, share a
// residency while the authored mesh is unchanged, so a transform-only
// republication rebuilds only its instance tables and a second plan builds
// nothing. The owner keeps the host arrays behind the key alive, so a live key
// cannot alias another mesh at the same addresses; the residency lives as long
// as some catalogue holds it, and its budget reservation with it. It frees its
// arrays through the device's shared module, which may outlive the plan that
// uploaded them.
struct CudaMeshDefinitionResidency
{
    std::shared_ptr<CudaContextModule>   device;
    std::shared_ptr<const ExactGeometry> owner;
    std::vector<CuDevicePtr>             allocations;
    std::shared_ptr<void>                budgetReservation;
    CuEvent                              ready = nullptr;
    cuda_family::MeshDefinition          record{};
    // The tree's host structure, shared by every definition of one topology, and the
    // definition owning the device index arrays a refitted definition reads through its
    // record (null for that owner itself). A derived definition keeps the root alive,
    // never the derived definition it was refitted after, so a long run of
    // republications holds one root and the catalogues' own definitions.
    std::shared_ptr<const MeshBvhStructure>      structure;
    std::shared_ptr<CudaMeshDefinitionResidency> topologyRoot;

    ~CudaMeshDefinitionResidency()
    {
        if (device != nullptr)
        {
            releaseCatalogueDeviceStorage(&device->driver, device->context, &device->lost, allocations, ready);
        }
    }
};

// What every exact catalogue derived on the device from the versions of one
// membership shares (BH-065): each slot's header, the convex and mesh
// instances' templates, the hulls' local vertices, planes, triangles and edges,
// the face materials and the resident mesh definitions with their records,
// only of the definitions that membership binds. Written once per membership
// on a launch stream, before `ready`, and kept only by the catalogues that read
// it, so it holds a definition no longer than a retained catalogue's version does.
struct CudaExactLayout
{
    Driver*                                                   driver     = nullptr;
    CuContext                                                 context    = nullptr;
    std::atomic<bool>*                                        deviceLost = nullptr;
    std::weak_ptr<const void>                                 membership;
    uint64_t                                                  slots = 0u;
    std::vector<CuDevicePtr>                                  allocations;
    std::shared_ptr<void>                                     budgetReservation;
    CuEvent                                                   ready = nullptr;
    std::vector<std::shared_ptr<CudaMeshDefinitionResidency>> definitions;
    bool                                                      materials = false;
    uint64_t spheres = 0u, boxes = 0u, capsules = 0u, convexes = 0u, meshes = 0u, convexVertices = 0u,
             convexPlanes = 0u;
    CuDevicePtr headers = 0u, convexTemplates = 0u, convexSources = 0u, localVertices = 0u, localPlanes = 0u,
                meshTemplates = 0u, faceMaterials = 0u, meshDefinitions = 0u, convexTriangles = 0u, convexEdges = 0u;

    bool matches(const CudaExactMembership& version) const noexcept
    {
        const std::shared_ptr<const void> mine = membership.lock();
        return mine != nullptr && mine == version.membership.lock() && slots >= version.slots;
    }

    ~CudaExactLayout() { releaseCatalogueDeviceStorage(driver, context, deviceLost, allocations, ready); }
};

// A layout a launch builds. Its measurement takes the slot history's census and
// sizes every table from it; the launch, once its host charge is reserved,
// reads the history per slot and fills the host tables it uploads.
struct CudaExactLayoutBuild
{
    // Where each bound definition's shared arrays start, and its sizes.
    struct Placement
    {
        bool     convex        = false;
        bool     doubleSided   = false;
        uint64_t vertexCount   = 0u;
        uint64_t planeCount    = 0u;
        uint64_t triangleCount = 0u;
        uint64_t edgeCount     = 0u;
        uint64_t faceCount     = 0u; // face materials staged for a mesh, zero without
        uint64_t vertices = 0u, planes = 0u, triangles = 0u, edges = 0u, faces = 0u, mesh = 0u;
    };
    std::shared_ptr<CudaExactLayout>             layout;
    std::shared_ptr<const CudaPopulationVersion> version;
    CudaExactMembership                          membership{};
    std::vector<CudaExactDefinition>             definitions; // the census's
    std::vector<Placement>                       placements;  // parallel to `definitions`
    uint64_t localVertexCount = 0u, localPlaneCount = 0u, triangleCount = 0u, edgeCount = 0u, faceCount = 0u;
    std::vector<cuda_family::ExactObjectHeader>       headers;
    std::vector<cuda_family::ConvexGeometry>          convexTemplates;
    std::vector<cuda_family::ExactConvexSource>       convexSources;
    std::vector<double>                               localVertices;
    std::vector<double>                               localPlanes;
    std::vector<cuda_family::MeshGeometry>            meshTemplates;
    std::vector<uint64_t>                             faceMaterials;
    std::vector<uint32_t>                             convexTriangles;
    std::vector<uint32_t>                             convexEdges;
    std::vector<std::shared_ptr<const ExactGeometry>> meshOwners; // per mesh definition, in definition order
};

// The device tables one derived exact catalogue owns, back to back in one
// allocation of `bytes`, every offset eight-byte aligned.
struct CudaExactTables
{
    size_t objects = 0u, minimums = 0u, maximums = 0u, materials = 0u, spheres = 0u, boxes = 0u, capsules = 0u,
           convexes = 0u, convexVertices = 0u, convexPlanes = 0u, meshes = 0u, failed = 0u, bytes = 0u;
};

CudaExactTables exactTables(const CudaExactLayout& layout) noexcept
{
    CudaExactTables tables{};
    size_t          offset = 0u;
    const auto      place  = [&offset](size_t& at, uint64_t count, size_t width)
    {
        at = offset;
        offset += static_cast<size_t>(count) * width;
    };
    place(tables.objects, layout.slots, sizeof(cuda_family::ExactObjectHeader));
    place(tables.minimums, layout.slots, 3u * sizeof(double));
    place(tables.maximums, layout.slots, 3u * sizeof(double));
    place(tables.materials, layout.materials ? layout.slots : 0u, sizeof(uint64_t));
    place(tables.spheres, layout.spheres, sizeof(cuda_family::SphereGeometry));
    place(tables.boxes, layout.boxes, sizeof(cuda_family::BoxGeometry));
    place(tables.capsules, layout.capsules, sizeof(cuda_family::CapsuleGeometry));
    place(tables.convexes, layout.convexes, sizeof(cuda_family::ConvexGeometry));
    place(tables.convexVertices, layout.convexVertices, 3u * sizeof(double));
    place(tables.convexPlanes, layout.convexPlanes, 4u * sizeof(double));
    place(tables.meshes, layout.meshes, sizeof(cuda_family::MeshGeometry));
    place(tables.failed, 1u, sizeof(uint64_t));
    tables.bytes = offset;
    return tables;
}

struct CudaFamilyDeviceCatalogue
{
    Driver*            driver     = nullptr;
    CuContext          context    = nullptr;
    std::atomic<bool>* deviceLost = nullptr;
    GenerationLease    generation;
    // The generation owns these device columns and its readiness event.
    bool                                                      borrowedStorage = false;
    std::vector<CuDevicePtr>                                  allocations;
    std::vector<std::shared_ptr<CudaMeshDefinitionResidency>> definitions; // definition-index order
    std::shared_ptr<CudaExactLayout>                          exactLayout; // of a catalogue derived on the device
    std::shared_ptr<void>                                     budgetReservation;
    CuEvent                                                   ready           = nullptr;
    size_t                                                    tableBytes      = 0u; // Of the one table allocation.
    uint64_t                                                  objectCount     = 0u;
    CuDevicePtr                                               objectIds       = 0u;
    CuDevicePtr                                               geometryTypes   = 0u;
    CuDevicePtr                                               geometryValues  = 0u;
    CuDevicePtr                                               objectMaterials = 0u;
    CuDevicePtr                                               faceMaterials   = 0u;
    CuDevicePtr                                               objectBits      = 0u;
    CuDevicePtr                                               objectMinimums  = 0u;
    CuDevicePtr                                               objectMaximums  = 0u;
    CuDevicePtr                                               exactObjects    = 0u;
    CuDevicePtr                                               spheres         = 0u;
    CuDevicePtr                                               boxes           = 0u;
    CuDevicePtr                                               capsules        = 0u;
    CuDevicePtr                                               convexes        = 0u;
    CuDevicePtr                                               convexVertices  = 0u;
    CuDevicePtr                                               convexTriangles = 0u;
    CuDevicePtr                                               convexPlanes    = 0u;
    CuDevicePtr                                               convexEdges     = 0u;
    CuDevicePtr                                               meshes          = 0u;
    CuDevicePtr                                               meshDefinitions = 0u;
    // Nonzero device word once the borrowed version's preparation failed (ADR-0021).
    CuDevicePtr preparationFailed = 0u;
    // The borrowed version's grid (W4), or zero when the plan scans the bounds.
    CuDevicePtr grid = 0u;

    bool matches(const GenerationLease& candidate) const noexcept { return generation.sameSelection(candidate); }

    ~CudaFamilyDeviceCatalogue()
    {
        if (!borrowedStorage)
        {
            releaseCatalogueDeviceStorage(driver, context, deviceLost, allocations, ready);
        }
    }
};

} // namespace

// The definitions some live catalogue on one device keeps resident, for the
// instance's plans to share. Lookups and registrations hold `mutex`; entries are
// weak, so the index never keeps a definition alive.
struct CudaMeshResidencyIndex
{
    std::mutex                                                              mutex;
    std::map<MeshDefinitionKey, std::weak_ptr<CudaMeshDefinitionResidency>> definitions;
    // The definitions owning a resident tree per topology, for derived meshes to refit.
    std::map<MeshTopologyKey, std::weak_ptr<CudaMeshDefinitionResidency>> topologies;
};

std::shared_ptr<CudaMeshResidencyIndex> CudaMeshResidencyRegistry::index(int32_t deviceId) noexcept
{
    try
    {
        const std::lock_guard<std::mutex>       lock(mMutex);
        std::weak_ptr<CudaMeshResidencyIndex>&  entry = mIndices[deviceId];
        std::shared_ptr<CudaMeshResidencyIndex> index = entry.lock();
        if (index == nullptr)
        {
            index = std::make_shared<CudaMeshResidencyIndex>();
            entry = index;
        }
        return index;
    }
    catch (const std::bad_alloc&)
    {
        return nullptr;
    }
}

namespace
{

// Whether a plan over a device-prepared version borrows the version's device
// tables: bounds-candidate plans do; exact plans need the exact objects, which
// a readable publication allows (ADR-0024 decision 4): derived on the device
// from an explicit version's poses and slot history (derivesExactCatalogue),
// or else the host-prepared path over the version's host copy.
bool borrowsDeviceTables(const CudaFamilyPlanConfig& config, const CudaPopulationDescription& description) noexcept
{
    return description.devicePrepared && config.precision == CudaFamilyPrecision::boundsCandidate;
}

// An exact plan over a whole readable explicit device-prepared version on the
// plan's own device derives its catalogue there, in stream order after the
// publication, so its launch neither waits for the version nor copies it to the
// host (BH-065). A version on another device is read through its host copy,
// since its tables' addresses mean nothing to the plan's kernels.
bool derivesExactCatalogue(const CudaFamilyPlanConfig& config, const GenerationLease& generation,
                           CudaExactMembership& outMembership) noexcept
{
    if (config.precision == CudaFamilyPrecision::boundsCandidate || generation.hasSelection())
    {
        return false;
    }
    const std::shared_ptr<const CudaPopulationVersion> version = generation.cudaVersion();
    if (version == nullptr)
    {
        return false;
    }
    const CudaPopulationDescription description = version->description();
    return description.devicePrepared && description.device.device_type == kDLCUDA &&
           description.device.device_id == config.device.device_id && version->exactMembership(outMembership);
}

ovspatialquery_api_status_t validateDevicePreparedCatalogue(const CudaFamilyPlanConfig&      config,
                                                            const GenerationLease&           generation,
                                                            const CudaPopulationDescription& description) noexcept
{
    // Bounds-candidate overlaps, rays and sweeps of every shape and hit mode
    // read the version's tables through the grid; exact precision needs the
    // exact objects a device-prepared version does not hold yet.
    if (generation.hasSelection() || config.precision != CudaFamilyPrecision::boundsCandidate)
    {
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    if (description.device.device_type != kDLCUDA || description.device.device_id != config.device.device_id ||
        description.ready.value == 0u ||
        (description.objectCount != 0u &&
         (description.objectIds == nullptr || description.geometryTypes == nullptr ||
          description.geometryValues == nullptr || description.objectBits == nullptr ||
          description.objectMinimums == nullptr || description.objectMaximums == nullptr)))
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    return OVSPATIALQUERY_API_SUCCESS;
}

// What one launch stages for its generation, decided under the plan's
// catalogue mutex so a resident definition found there stays alive until the
// launch has recorded its own catalogue over it.
struct CudaMeshDefinitionStaging
{
    MeshDefinitionKey                            key;
    std::shared_ptr<CudaMeshDefinitionResidency> resident; // shared when set; built by this launch otherwise
    // A resident definition of the same topology: this launch refits its tree to the
    // new vertices and shares its device index arrays instead of building and
    // uploading everything.
    std::shared_ptr<CudaMeshDefinitionResidency> topologyBase;
    std::shared_ptr<const ExactGeometry>         owner;
    uint64_t                                     triangleCount = 0u;
    uint64_t                                     nodeCount     = 0u; // upper bound, see addBvhNodeCount
    uint64_t                                     deviceBytes   = 0u;
};

struct CudaFamilyCatalogueStaging
{
    std::shared_ptr<CudaFamilyDeviceCatalogue> catalogue;   // the plan's current catalogue when it already matches
    std::vector<CudaMeshDefinitionStaging>     definitions; // first-seen order over the selected objects
    CudaFamilyCatalogueCounts                  counts{};
    uint64_t                                   tableDeviceBytes = 0u;
    uint64_t                                   hostBytes        = 0u;
    // A catalogue derived on the device: the version's membership, and the layout
    // it reads, resident already or built by this launch (`exactBuild`).
    bool                                  exact = false;
    CudaExactMembership                   exactMembership{};
    std::shared_ptr<CudaExactLayout>      exactLayout;
    std::shared_ptr<CudaExactLayoutBuild> exactBuild;
    uint64_t                              layoutDeviceBytes = 0u; // of a layout this launch builds
};

// Stages one distinct mesh definition: shared when `resident` (the instance's
// index for the plan's device) holds it, refitted over a resident tree of its
// topology when it derives from one, built otherwise; expired entries are
// dropped. False when its device bytes overflow.
bool stageMeshDefinition(const TriangleMeshExactGeometryDescription& mesh, std::shared_ptr<const ExactGeometry> owner,
                         CudaMeshResidencyIndex& resident, CudaMeshDefinitionStaging& staging)
{
    const auto addProduct = [](uint64_t& total, uint64_t count, uint64_t width)
    {
        if (count != 0u && (width > UINT64_MAX / count || count * width > UINT64_MAX - total))
        {
            return false;
        }
        total += count * width;
        return true;
    };
    staging.key           = meshDefinitionKey(mesh);
    staging.owner         = std::move(owner);
    staging.triangleCount = mesh.triangleCount;
    std::unique_lock<std::mutex> residentLock(resident.mutex);
    const auto                   residentFound = resident.definitions.find(staging.key);
    if (residentFound != resident.definitions.end())
    {
        staging.resident = residentFound->second.lock();
        if (staging.resident == nullptr)
        {
            resident.definitions.erase(residentFound);
        }
    }
    if (staging.resident == nullptr && mesh.refitTree)
    {
        const auto found = resident.topologies.find(meshTopologyKey(mesh));
        if (found != resident.topologies.end())
        {
            staging.topologyBase = found->second.lock();
            if (staging.topologyBase == nullptr)
            {
                resident.topologies.erase(found);
            }
            else if (staging.topologyBase->structure == nullptr ||
                     staging.topologyBase->structure->traversalTriangles.size() != mesh.triangleCount)
            {
                staging.topologyBase.reset();
            }
        }
    }
    residentLock.unlock();
    if (staging.resident == nullptr && staging.topologyBase != nullptr)
    {
        // Refitted: new vertices and node bounds; the index arrays are the root's.
        staging.nodeCount = staging.topologyBase->structure->nodes.size();
        return addProduct(staging.deviceBytes, mesh.vertexCount, 3u * sizeof(float)) &&
               addProduct(staging.deviceBytes, staging.nodeCount, sizeof(cuda_family::MeshBvhNode));
    }
    if (staging.resident == nullptr)
    {
        return addBvhNodeCount(mesh.triangleCount, staging.nodeCount) &&
               addProduct(staging.deviceBytes, mesh.vertexCount, 3u * sizeof(float)) &&
               addProduct(staging.deviceBytes, mesh.triangleCount, 3u * sizeof(uint32_t)) &&
               addProduct(staging.deviceBytes, mesh.triangleCount, sizeof(uint64_t)) &&
               addProduct(staging.deviceBytes, mesh.triangleCount, 3u * sizeof(uint32_t)) &&
               addProduct(staging.deviceBytes, staging.nodeCount, sizeof(cuda_family::MeshBvhNode)) &&
               addProduct(staging.deviceBytes, mesh.triangleCount, sizeof(uint32_t));
    }
    return true;
}

// The host bytes of staging `definitions`: per definition its staging record and
// residency handle, and per new one the pinned staging of its arrays, the tree
// build (nodes, leaf order and the builder's triangle records) and the residency.
bool addDefinitionHostBytes(const std::vector<CudaMeshDefinitionStaging>& definitions, uint64_t& total)
{
    const auto add = [&total](uint64_t count)
    {
        if (count > UINT64_MAX - total)
        {
            return false;
        }
        total += count;
        return true;
    };
    const auto addProduct = [&add](uint64_t count, uint64_t width)
    { return count == 0u || (width <= UINT64_MAX / count && add(count * width)); };
    constexpr uint64_t pointerPair = sizeof(CuDevicePtr) + sizeof(void*);
    for (const CudaMeshDefinitionStaging& definition : definitions)
    {
        if (!add(sizeof(CudaMeshDefinitionStaging) + sizeof(MeshBvhStructure) +
                 sizeof(std::shared_ptr<CudaMeshDefinitionResidency>)))
        {
            return false;
        }
        if (definition.resident != nullptr)
        {
            continue;
        }
        // A new tree's leaf order and build records; a refit keeps the root's order.
        const uint64_t perTriangle =
            sizeof(uint32_t) + (definition.topologyBase == nullptr ? kMeshBvhBuildRecordBytes : sizeof(uint32_t));
        if (!add(definition.deviceBytes) || !addProduct(definition.nodeCount, sizeof(cuda_family::MeshBvhNode)) ||
            !addProduct(definition.triangleCount, perTriangle) ||
            !add(sizeof(CudaMeshDefinitionResidency) + kDefinitionAllocations * pointerPair))
        {
            return false;
        }
    }
    return true;
}

// Sizes a layout from the census of a readable explicit version's slot history
// at `membership.slots` slots: the slots of each family, and per bound
// definition its arrays and the slots that bind it. Nothing it allocates grows
// with the slots; the caller stages the mesh definitions in `meshOwners`.
ovspatialquery_api_status_t planExactLayout(const CudaFamilyPlanConfig&                  config,
                                            std::shared_ptr<const CudaPopulationVersion> version,
                                            const CudaExactMembership& membership, CudaExactLayoutBuild& build) noexcept
{
    try
    {
        CudaExactSlots                    census;
        const ovspatialquery_api_status_t status = version->exactSlots(membership.slots, false, census);
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return status;
        }
        if (census.definitionSlots.size() != census.definitions.size())
        {
            return OVSPATIALQUERY_API_INTERNAL;
        }
        CudaExactLayout& layout = *build.layout;
        layout.membership       = membership.membership;
        layout.slots            = membership.slots;
        layout.materials        = (config.resultFields & kCudaFamilyResultMaterial) != 0u;
        layout.spheres          = census.spheres;
        layout.boxes            = census.boxes;
        layout.capsules         = census.capsules;
        build.version           = std::move(version);
        build.membership        = membership;
        build.placements.resize(census.definitions.size());
        for (size_t index = 0u; index < census.definitions.size(); ++index)
        {
            const CudaExactDefinition&       definition  = census.definitions[index];
            const uint64_t                   uses        = census.definitionSlots[index];
            CudaExactLayoutBuild::Placement& place       = build.placements[index];
            ExactGeometryDescription         description = {};
            if (exportExactGeometryDescription(definition.local, description) != OVSPATIALQUERY_API_SUCCESS)
            {
                return OVSPATIALQUERY_API_NOT_SUPPORTED;
            }
            if (const ConvexExactGeometryDescription* convex =
                    std::get_if<ConvexExactGeometryDescription>(&description))
            {
                place.convex        = true;
                place.vertexCount   = convex->vertexCount;
                place.planeCount    = convex->planeCount;
                place.triangleCount = convex->triangleCount;
                place.edgeCount     = convex->edgeCount;
                place.vertices      = build.localVertexCount;
                place.planes        = build.localPlaneCount;
                place.triangles     = build.triangleCount;
                place.edges         = build.edgeCount;
                build.localVertexCount += convex->vertexCount;
                build.localPlaneCount += convex->planeCount;
                build.triangleCount += convex->triangleCount;
                build.edgeCount += convex->edgeCount;
                layout.convexes += uses;
                layout.convexVertices += uses * convex->vertexCount;
                layout.convexPlanes += uses * convex->planeCount;
            }
            else if (const TriangleMeshExactGeometryDescription* mesh =
                         std::get_if<TriangleMeshExactGeometryDescription>(&description))
            {
                place.doubleSided = mesh->doubleSided;
                place.mesh        = build.meshOwners.size();
                place.faces       = build.faceCount;
                place.faceCount =
                    layout.materials && definition.faceMaterialIds != nullptr ? definition.faceMaterialIds->size() : 0u;
                build.faceCount += place.faceCount;
                build.meshOwners.push_back(definition.local);
                layout.meshes += uses;
            }
            else
            {
                return OVSPATIALQUERY_API_NOT_SUPPORTED;
            }
        }
        build.definitions = std::move(census.definitions);
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

// The device bytes of a planned layout's tables, which its host tables match.
uint64_t exactLayoutBytes(const CudaExactLayoutBuild& build) noexcept
{
    const CudaExactLayout& layout = *build.layout;
    return layout.slots * sizeof(cuda_family::ExactObjectHeader) +
           layout.convexes * (sizeof(cuda_family::ConvexGeometry) + sizeof(cuda_family::ExactConvexSource)) +
           build.localVertexCount * 3u * sizeof(double) + build.localPlaneCount * 4u * sizeof(double) +
           layout.meshes * sizeof(cuda_family::MeshGeometry) + build.faceCount * sizeof(uint64_t) +
           build.triangleCount * 3u * sizeof(uint32_t) + build.edgeCount * 2u * sizeof(uint32_t) +
           build.meshOwners.size() * sizeof(cuda_family::MeshDefinition);
}

// Fills a planned layout's host tables from the slot history read per slot:
// the launch's work that grows with the slots, after its host charge.
ovspatialquery_api_status_t fillExactLayout(CudaExactLayoutBuild& build) noexcept
{
    namespace cudaFamily = cuda_family;
    try
    {
        CudaExactSlots                    slots;
        const ovspatialquery_api_status_t status = build.version->exactSlots(build.membership.slots, true, slots);
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return status;
        }
        const CudaExactLayout& layout = *build.layout;
        if (slots.types.size() != layout.slots || slots.definitionIndices.size() != layout.slots)
        {
            return OVSPATIALQUERY_API_INTERNAL;
        }
        build.localVertices.reserve(static_cast<size_t>(build.localVertexCount * 3u));
        build.localPlanes.reserve(static_cast<size_t>(build.localPlaneCount * 4u));
        build.convexTriangles.reserve(static_cast<size_t>(build.triangleCount * 3u));
        build.convexEdges.reserve(static_cast<size_t>(build.edgeCount * 2u));
        build.faceMaterials.reserve(static_cast<size_t>(build.faceCount));
        for (size_t index = 0u; index < build.definitions.size(); ++index)
        {
            const CudaExactDefinition&             definition  = build.definitions[index];
            const CudaExactLayoutBuild::Placement& place       = build.placements[index];
            ExactGeometryDescription               description = {};
            if (exportExactGeometryDescription(definition.local, description) != OVSPATIALQUERY_API_SUCCESS)
            {
                return OVSPATIALQUERY_API_INTERNAL;
            }
            if (const ConvexExactGeometryDescription* convex =
                    std::get_if<ConvexExactGeometryDescription>(&description);
                convex != nullptr && place.convex)
            {
                build.localVertices.insert(build.localVertices.end(), convex->vertices,
                                           convex->vertices + convex->vertexCount * 3u);
                build.convexTriangles.insert(build.convexTriangles.end(), convex->triangleVertexIndices,
                                             convex->triangleVertexIndices + convex->triangleCount * 3u);
                for (uint64_t plane = 0u; plane < convex->planeCount; ++plane)
                {
                    build.localPlanes.insert(build.localPlanes.end(), convex->planes[plane].normal.begin(),
                                             convex->planes[plane].normal.end());
                    build.localPlanes.push_back(convex->planes[plane].offset);
                }
                for (uint64_t edge = 0u; edge < convex->edgeCount; ++edge)
                {
                    build.convexEdges.push_back(convex->edges[edge].first);
                    build.convexEdges.push_back(convex->edges[edge].second);
                }
            }
            else if (place.faceCount != 0u)
            {
                build.faceMaterials.insert(build.faceMaterials.end(), definition.faceMaterialIds->begin(),
                                           definition.faceMaterialIds->end());
            }
        }
        build.headers.resize(static_cast<size_t>(layout.slots));
        build.convexTemplates.reserve(static_cast<size_t>(layout.convexes));
        build.convexSources.reserve(static_cast<size_t>(layout.convexes));
        build.meshTemplates.reserve(static_cast<size_t>(layout.meshes));
        uint64_t spheres = 0u, boxes = 0u, capsules = 0u, convexVertices = 0u, convexPlanes = 0u;
        for (size_t slot = 0u; slot < slots.types.size(); ++slot)
        {
            cudaFamily::ExactObjectHeader& header = build.headers[slot];
            const int32_t                  type   = slots.types[slot];
            const uint32_t                 index  = slots.definitionIndices[slot];
            if (type == OVSPATIALQUERY_GEOMETRY_SPHERE || type == OVSPATIALQUERY_GEOMETRY_BOX ||
                type == OVSPATIALQUERY_GEOMETRY_CAPSULE)
            {
                uint64_t& count    = type == OVSPATIALQUERY_GEOMETRY_SPHERE ? spheres
                                     : type == OVSPATIALQUERY_GEOMETRY_BOX  ? boxes
                                                                            : capsules;
                header.kind        = type == OVSPATIALQUERY_GEOMETRY_SPHERE ? cudaFamily::kGeometrySphere
                                     : type == OVSPATIALQUERY_GEOMETRY_BOX  ? cudaFamily::kGeometryBox
                                                                            : cudaFamily::kGeometryCapsule;
                header.familyIndex = count++;
                continue;
            }
            if (index == CudaExactSlots::kAbsent)
            {
                header.kind = cudaFamily::kGeometryAbsent; // removed, its definition no longer bound
                continue;
            }
            if (index >= build.placements.size())
            {
                return OVSPATIALQUERY_API_INTERNAL;
            }
            const CudaExactLayoutBuild::Placement& place = build.placements[index];
            if (place.convex && type == OVSPATIALQUERY_GEOMETRY_CONVEX)
            {
                // The instance's own vertices and planes; the triangles and edges are the hull's.
                cudaFamily::ConvexGeometry value{};
                value.vertexOffset   = convexVertices;
                value.vertexCount    = place.vertexCount;
                value.triangleOffset = place.triangles;
                value.triangleCount  = place.triangleCount;
                value.planeOffset    = convexPlanes;
                value.planeCount     = place.planeCount;
                value.edgeOffset     = place.edges;
                value.edgeCount      = place.edgeCount;
                convexVertices += place.vertexCount;
                convexPlanes += place.planeCount;
                header.kind        = cudaFamily::kGeometryConvex;
                header.familyIndex = build.convexTemplates.size();
                build.convexTemplates.push_back(value);
                build.convexSources.push_back({place.vertices, place.planes});
            }
            else if (!place.convex && type == OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH)
            {
                cudaFamily::MeshGeometry value{};
                value.definitionIndex = place.mesh;
                if (place.faceCount != 0u)
                {
                    value.faceMaterialOffset = place.faces;
                    value.faceMaterialCount  = place.faceCount;
                }
                header.kind        = cudaFamily::kGeometryTriangleMesh;
                header.flags       = place.doubleSided ? cudaFamily::kGeometryFlagDoubleSided : 0u;
                header.familyIndex = build.meshTemplates.size();
                build.meshTemplates.push_back(value);
            }
            else
            {
                return OVSPATIALQUERY_API_INTERNAL;
            }
        }
        // The census and the per-slot reading saw the same history and membership.
        const bool sized = spheres == layout.spheres && boxes == layout.boxes && capsules == layout.capsules &&
                           build.convexTemplates.size() == layout.convexes &&
                           build.meshTemplates.size() == layout.meshes && convexVertices == layout.convexVertices &&
                           convexPlanes == layout.convexPlanes;
        return sized ? OVSPATIALQUERY_API_SUCCESS : OVSPATIALQUERY_API_INTERNAL;
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

// Counts the catalogue tables and stages every distinct mesh definition of the
// generation. Definitions already on the device in `resident` (the instance's
// index for the plan's device) are shared and expired entries are dropped.
ovspatialquery_api_status_t measureCudaFamilyCatalogue(const CudaFamilyPlanConfig& config,
                                                       const GenerationLease&      generation,
                                                       CudaMeshResidencyIndex&     resident,
                                                       CudaFamilyCatalogueStaging& out) noexcept;

class CudaFamilyFamilyOperation final : public CudaFamilyOperation
{
  public:
    friend class CudaFamilyFamilyPlan;
    CudaFamilyFamilyOperation(std::shared_ptr<CudaFamilyFamilyPlan> owner, uint32_t slot, CudaFamilyLaunch launch,
                              GenerationLease generation) noexcept;
    ~CudaFamilyFamilyOperation() override;

    CudaFamilyResultColumns          reservation() const noexcept override;
    CudaFamilyCompletedResultOutcome completedResult(CudaFamilyResultColumns& outResult) const noexcept override;
    CudaCompletionState              resultState() const noexcept override;
    void                             waitForResult() const noexcept override;
    CudaCompletionState              consumerState() const noexcept override;
    bool                             consumerDoneOwnershipReleased() const noexcept override;
    void                             releaseReadyDependencies() noexcept override;
    ovspatialquery_api_status_t      transferConsumerDone(ovspatialquery_cuda_event_t event) noexcept override;
    void                             attachDeviceResultReservation(std::shared_ptr<void> reservation) noexcept override;
    void                             attachStagingReservation(std::shared_ptr<void> reservation) noexcept override;
    void attachHostStagingReservation(std::shared_ptr<HostLease> reservation) noexcept override;
    void attachPublishedReservations(std::shared_ptr<void>              catalogueReservation,
                                     std::vector<std::shared_ptr<void>> definitionReservations) noexcept override
    {
        mPublishedCatalogueReservation   = std::move(catalogueReservation);
        mPublishedDefinitionReservations = std::move(definitionReservations);
    }
    void ringWhenConsumed(const std::shared_ptr<DeviceSignal>& signal) const noexcept override;

    ovspatialquery_api_status_t materializeCatalogue() noexcept;
    CuResult                    stageCopiedInputs(CuStream stream) noexcept;

    struct UploadPiece
    {
        const void*  data;
        size_t       bytes;
        CuDevicePtr* output;
    };

    template <typename T>
    static UploadPiece piece(const std::vector<T>& values, CuDevicePtr& output) noexcept
    {
        return {values.data(), values.size() * sizeof(T), &output};
    }

    static size_t uploadBytes(std::initializer_list<UploadPiece> pieces) noexcept
    {
        size_t bytes = 0u;
        for (const UploadPiece& piece : pieces)
        {
            bytes += piece.bytes;
        }
        return bytes;
    }

    // Stages the pieces back to back through one pinned buffer into one device
    // allocation on the launch stream, which joins `allocations`, whose owner
    // frees it: `reuse` when nonzero (an idle allocation of uploadBytes(pieces)
    // bytes), a new one otherwise. Each piece's output is its device address, or
    // zero when empty; the caller orders the pieces so each starts aligned.
    CuResult upload(std::initializer_list<UploadPiece> pieces, std::vector<CuDevicePtr>& allocations,
                    CuDevicePtr reuse = 0u)
    {
        for (const UploadPiece& piece : pieces)
        {
            *piece.output = 0u;
        }
        const size_t bytes = uploadBytes(pieces);
        if (bytes == 0u)
        {
            return kCuSuccess;
        }
        CuDevicePtr base   = reuse;
        void*       host   = nullptr;
        CuResult    status = base != 0u ? kCuSuccess : mOwnerDriver().memAlloc(&base, bytes);
        if (status != kCuSuccess)
        {
            return status;
        }
        allocations.push_back(base);
        status = mOwnerDriver().memHostAlloc(&host, bytes, kCuMemHostAllocPortable);
        if (status != kCuSuccess)
        {
            return status;
        }
        mStaging.push_back(host);
        size_t offset = 0u;
        for (const UploadPiece& piece : pieces)
        {
            if (piece.bytes != 0u)
            {
                std::memcpy(static_cast<unsigned char*>(host) + offset, piece.data, piece.bytes);
                *piece.output = base + offset;
                offset += piece.bytes;
            }
        }
        status = mOwnerDriver().copyHtoDAsync(base, host, bytes, reinterpret_cast<CuStream>(mLaunch.stream));
        if (status == kCuSuccess)
        {
            mWorkEnqueued = true;
        }
        return status;
    }

    Driver&                          mOwnerDriver() const noexcept;
    CudaFamilyResultSlot&            slot() const noexcept;
    CudaFamilyResultColumns          columns(const CudaFamilyColumnMetadata& metadata) const noexcept;
    CudaFamilyCompletedResultOutcome prepareResultMetadata() const noexcept;
    void configure(bool workEnqueued, bool readyRecorded, bool wakeRecorded, bool failed, bool deviceLost) noexcept;
    ovspatialquery_api_status_t populateCatalogue() noexcept;
    ovspatialquery_api_status_t buildExactDefinitions() noexcept;
    CuResult                    stageDefinitions(CuStream stream) noexcept;
    CuResult                    stageExactCatalogue(CuStream stream, CuDevicePtr reuse) noexcept;
    bool                        reserveCatalogueStorage() noexcept;
    void                        releaseCatalogueStorage() noexcept;

  private:
    CudaCompletionState eventState(CuEvent event, bool queryStream) const noexcept;
    CudaCompletionState confirmDeviceDone() const noexcept;
    void                releaseAfterDeviceLostLocked() noexcept;

    std::shared_ptr<CudaFamilyFamilyPlan>      mOwner;
    std::shared_ptr<CudaFamilyDeviceCatalogue> mCatalogue;
    bool                                       mBuildCatalogue = false;
    uint32_t                                   mSlot           = UINT32_MAX;
    CudaFamilyLaunch                           mLaunch{};
    GenerationLease                            mGeneration;
    ovspatialquery_ordinal_t                   mSourceOrdinal = 0;
    // Copied input columns' device buffers and their sizes, taken from and
    // given back to the plan's staging pool.
    std::vector<CopiedInputAllocation>          mCopiedInputAllocations;
    std::vector<void*>                          mStaging;
    std::vector<uint64_t>                       mObjectIds;
    std::vector<uint32_t>                       mGeometryTypes;
    std::vector<uint64_t>                       mGeometryValues;
    std::vector<uint64_t>                       mObjectMaterials;
    std::vector<uint64_t>                       mFaceMaterials;
    std::vector<uint64_t>                       mObjectBits;
    std::vector<double>                         mObjectMinimums;
    std::vector<double>                         mObjectMaximums;
    std::vector<cuda_family::ExactObjectHeader> mExactObjects;
    std::vector<cuda_family::SphereGeometry>    mSpheres;
    std::vector<cuda_family::BoxGeometry>       mBoxes;
    std::vector<cuda_family::CapsuleGeometry>   mCapsules;
    std::vector<cuda_family::ConvexGeometry>    mConvexes;
    std::vector<double>                         mConvexVertices;
    std::vector<uint32_t>                       mConvexTriangles;
    std::vector<double>                         mConvexPlanes;
    std::vector<uint32_t>                       mConvexEdges;
    std::vector<cuda_family::MeshGeometry>      mMeshes;
    std::vector<cuda_family::MeshDefinition>    mMeshDefinitions;
    std::shared_ptr<CudaFamilyCatalogueStaging> mCatalogueStaging;
    std::vector<MeshBvhStructure>               mDefinitionBuilds; // parallel to the staged definitions
    std::vector<std::shared_ptr<void>>          mPublishedDefinitionReservations;
    mutable std::mutex                          mMutex;
    bool                                        mWorkEnqueued      = false;
    bool                                        mReadyRecorded     = false;
    bool                                        mWakeRecorded      = false;
    bool                                        mFailed            = false;
    mutable bool                                mDeviceLost        = false;
    bool                                        mConsumerRecorded  = false;
    CuEvent                                     mOwnedConsumerDone = nullptr;
    mutable CudaCompletionState                 mReadyState        = CudaCompletionState::pending;
    mutable CudaCompletionState                 mConsumerState     = CudaCompletionState::pending;
    CudaFamilyColumnMetadata                    mReservationMetadata;
    mutable CudaFamilyColumnMetadata            mResultMetadata;
    mutable bool                                mResultMetadataAttempted = false;
    mutable CudaFamilyCompletedResultOutcome    mResultMetadataOutcome = CudaFamilyCompletedResultOutcome::unavailable;
    std::shared_ptr<void>                       mDeviceResultReservation;
    std::shared_ptr<void>                       mStagingBudgetReservation;
    std::shared_ptr<void>                       mPublishedCatalogueReservation;
    std::shared_ptr<HostLease>                  mHostStagingReservation;
};

class CudaFamilyFamilyPlan final : public CudaFamilyPlan, public std::enable_shared_from_this<CudaFamilyFamilyPlan>
{
  public:
    Driver    driver;
    CuDevice  device            = 0;
    CuContext context           = nullptr;
    CuContext submissionContext = nullptr;
    // The device's module, which this library copy loads once per context.
    std::shared_ptr<CudaContextModule> contextModule;
    CuModule                           module        = nullptr;
    CuFunction                         function      = nullptr;
    CuFunction                         scanFunction  = nullptr;
    CuFunction                         exactFunction = nullptr; // derives an exact catalogue on the device
    // HOST_MAPPED only: the pass that copies the device columns into mapped memory.
    CuFunction                        publishFunction = nullptr;
    CudaFamilyPlanConfig              config{};
    std::vector<CudaFamilyResultSlot> slots;
    // Where a retirement queues the ring that wakes the reaper after the slot's
    // consumer; up to two per slot may be outstanding.
    CudaSignalRings   signalRings;
    uint64_t          byteCharge      = 0u;
    bool              driverOpened    = false;
    bool              primaryRetained = false;
    std::atomic<bool> deviceLost{false};
    std::mutex        catalogueMutex;
    // Set with deviceLost when a launch's ready poll and the wait meant to confirm it both
    // failed (confirmDeviceDone), so the device may still read or write any storage of the
    // plan: from then on the plan frees no page-locked memory and admits no device work.
    std::atomic<bool> workUnconfirmed{false};
    // Most recently used first; at most kCudaRetainedCatalogues.
    std::vector<std::shared_ptr<CudaFamilyDeviceCatalogue>> catalogues;
    // Definitions the instance's live catalogues on this device keep resident, for
    // the next generation and the instance's other plans to share.
    std::shared_ptr<CudaMeshResidencyIndex> meshResidency;
    // Device buffers copied input columns reuse across launches (ADR-0053).
    CudaStagingPool stagingPool;
    // Mapped pinned, one uint64_t per slot. Held for the plan's life, it also keeps the
    // driver's pinned pool mapped: without a live pinned allocation each new generation's
    // staging cuMemHostAlloc and cuMemFreeHost cost about 0.6 and 0.4 ms instead of 6 us.
    void* hostStoredHitCounts = nullptr;
    // HOST_MAPPED only: every slot's mapped block, one page-locked allocation.
    void*                  mappedColumns = nullptr;
    CudaFamilyMappedLayout mappedLayout{};

    ~CudaFamilyFamilyPlan() override
    {
        if (!driverOpened)
        {
            return;
        }
        // Context teardown owns a lost device's state, but page-locked memory is the host's:
        // whatever the paths below leave allocated is freed here, best effort, as a lost launch
        // frees its pinned staging, so a HOST_MAPPED plan does not keep its pinned columns
        // after the host budget has released their charge. Unconfirmed work may still write
        // the columns and the H mirror, so they are then left allocated, uncharged like the
        // device memory a lost plan leaves to its context, which the plan never releases.
        struct PinnedRelease
        {
            CudaFamilyFamilyPlan& plan;
            ~PinnedRelease()
            {
                if ((plan.mappedColumns == nullptr && plan.hostStoredHitCounts == nullptr) ||
                    plan.workUnconfirmed.load(std::memory_order_relaxed))
                {
                    return;
                }
                CuContext  restore  = nullptr;
                const bool selected = plan.primaryRetained && plan.driver.contextGetCurrent(&restore) == kCuSuccess &&
                                      plan.driver.contextSetCurrent(plan.context) == kCuSuccess;
                for (void** pinned : {&plan.mappedColumns, &plan.hostStoredHitCounts})
                {
                    if (*pinned != nullptr)
                    {
                        (void)plan.driver.memFreeHost(std::exchange(*pinned, nullptr));
                    }
                }
                if (selected)
                {
                    (void)plan.driver.contextSetCurrent(restore);
                }
            }
        } pinnedRelease{*this};
        if (deviceLost.load(std::memory_order_relaxed))
        {
            return;
        }
        CuContext  restore        = nullptr;
        bool       ok             = true;
        const auto observeCleanup = [this, &ok](CuResult status)
        {
            if (status == kCuDeviceLost)
            {
                deviceLost.store(true, std::memory_order_relaxed);
                return false;
            }
            ok = status == kCuSuccess && ok;
            return true;
        };
        if (primaryRetained)
        {
            const CuResult current = driver.contextGetCurrent(&restore);
            if (!observeCleanup(current))
            {
                return;
            }
            if (current != kCuSuccess)
            {
                return;
            }
            const CuResult selected = driver.contextSetCurrent(context);
            if (!observeCleanup(selected) || selected != kCuSuccess)
            {
                return;
            }
        }
        {
            std::lock_guard<std::mutex> lock(catalogueMutex);
            catalogues.clear();
        }
        stagingPool.clear();
        for (CudaFamilyResultSlot& slot : slots)
        {
            for (CuDevicePtr allocation : slot.allocations)
            {
                if (!observeCleanup(driver.memFree(allocation)))
                {
                    return;
                }
            }
            if (!slot.destroy(driver, observeCleanup))
            {
                return;
            }
        }
        if (signalRings.stream != nullptr &&
            !observeCleanup(driver.streamDestroy(std::exchange(signalRings.stream, nullptr))))
        {
            return;
        }
        if (hostStoredHitCounts != nullptr &&
            !observeCleanup(driver.memFreeHost(std::exchange(hostStoredHitCounts, nullptr))))
        {
            return;
        }
        if (mappedColumns != nullptr && !observeCleanup(driver.memFreeHost(std::exchange(mappedColumns, nullptr))))
        {
            return;
        }
        if (primaryRetained)
        {
            if (!observeCleanup(driver.contextSetCurrent(restore)) || !observeCleanup(driver.primaryRelease(device)))
            {
                return;
            }
        }
        assert(ok);
    }

    ovspatialquery_api_status_t initialize(const CudaFamilyPlanConfig& source) noexcept;
    ovspatialquery_api_status_t measure(const CudaFamilyLaunch& launch, const GenerationLease& generation,
                                        CudaFamilyStagingCharge& outCharge) noexcept override;
    bool                        needsHostObjects(const GenerationLease& generation) const noexcept override
    {
        CudaExactMembership membership{};
        return generation.preparedOnDevice() && config.precision != CudaFamilyPrecision::boundsCandidate &&
               !derivesExactCatalogue(config, generation, membership);
    }
    ovspatialquery_api_status_t prepare(CudaFamilyLaunch launch, GenerationLease generation,
                                        std::shared_ptr<void>                 staging,
                                        std::shared_ptr<CudaFamilyOperation>& outPrepared) noexcept override;
    // Under catalogueMutex: reuses the current catalogue or stages the generation's tables and definitions.
    ovspatialquery_api_status_t stageLocked(const GenerationLease&      generation,
                                            CudaFamilyCatalogueStaging& outStaging) noexcept;
    // Reuses a retained catalogue, or measures one derived on the device over the
    // current layout of the membership or over a layout this launch builds.
    ovspatialquery_api_status_t measureExact(const GenerationLease& generation, const CudaExactMembership& membership,
                                             CudaFamilyCatalogueStaging& outStaging) noexcept;
    ovspatialquery_api_status_t launch(uint32_t resultSlot, std::shared_ptr<CudaFamilyOperation> prepared,
                                       std::shared_ptr<CudaFamilyOperation>& outOperation) noexcept override;
    uint64_t                    deviceByteCharge() const noexcept override { return byteCharge; }

    // The device storage the kernels write for `column`.
    static CuDevicePtr deviceColumn(const CudaFamilyResultSlot& slot, CudaFamilyMappedColumn column) noexcept
    {
        using Column = CudaFamilyMappedColumn;
        switch (column)
        {
        case Column::queryStatus: return slot.queryStatus;
        case Column::storedHitCount: return slot.storedHitCount;
        case Column::hitOffsets: return slot.hitOffsets;
        case Column::hitCounts: return slot.hitCounts;
        case Column::truncated: return slot.truncated;
        case Column::objectIds: return slot.objectIds;
        case Column::geometryTypes: return slot.geometryTypes;
        case Column::geometryValues: return slot.geometryValues;
        case Column::materialIds: return slot.materialIds;
        case Column::positions: return slot.positions;
        case Column::queryPositions: return slot.queryPositions;
        case Column::normals: return slot.normals;
        case Column::distances: return slot.distances;
        case Column::featureKinds: return slot.featureKinds;
        case Column::faceIndices: return slot.faceIndices;
        case Column::corners: return slot.corners;
        case Column::barycentrics: return slot.barycentrics;
        case Column::count: break;
        }
        return 0u;
    }

    // Where the caller reads `column`: its mapped copy for a HOST_MAPPED plan,
    // otherwise its device storage; zero for a column the mapped plan does not publish.
    CuDevicePtr publishedColumn(const CudaFamilyResultSlot& slot, CudaFamilyMappedColumn column) const noexcept
    {
        if (!config.hostMapped)
        {
            return deviceColumn(slot, column);
        }
        const size_t index = static_cast<size_t>(column);
        return mappedLayout.rowBytes[index] == 0u ? 0u : slot.mapped + mappedLayout.offsets[index];
    }

  private:
    CuDevicePtr takeEvictedTables(size_t bytes) noexcept;

    CuResult allocate(CuDevicePtr& output, size_t bytes, CudaFamilyResultSlot& slot)
    {
        if (bytes == 0u)
        {
            output = 0u;
            return kCuSuccess;
        }
        const CuResult status = driver.memAlloc(&output, bytes);
        if (status == kCuSuccess)
        {
            slot.allocations.push_back(output);
        }
        return status;
    }
};

ovspatialquery_api_status_t CudaFamilyFamilyPlan::initialize(const CudaFamilyPlanConfig& source) noexcept
{
    uint64_t                          estimated = 0u;
    const ovspatialquery_api_status_t validated = validateCudaFamilyPlanConfig(source, estimated);
    if (validated != OVSPATIALQUERY_API_SUCCESS)
    {
        return validated;
    }
    config     = source;
    byteCharge = estimated;
    if (!driver.open())
    {
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    driverOpened = true;
    if (driver.deviceGet(&device, config.device.device_id) != kCuSuccess ||
        driver.contextGetCurrent(&submissionContext) != kCuSuccess ||
        driver.primaryRetain(&context, device) != kCuSuccess)
    {
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    primaryRetained                          = true;
    const ovspatialquery_api_status_t loaded = acquireCudaContextModule(config.device.device_id, contextModule);
    if (loaded != OVSPATIALQUERY_API_SUCCESS)
    {
        return loaded;
    }
    module = contextModule->module;
    try
    {
        meshResidency = config.meshResidency != nullptr ? config.meshResidency->index(config.device.device_id)
                                                        : std::make_shared<CudaMeshResidencyIndex>();
    }
    catch (const std::bad_alloc&)
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    if (meshResidency == nullptr)
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    // The registry was only for finding the index.
    config.meshResidency = nullptr;
    // MESH_HITS_ALL changes only exact mesh evaluation; bounds rows keep the plain entry.
    const bool crossings = config.family == CudaFamilyKind::raycast && config.allMeshHits &&
                           config.hitMode == CudaFamilyHitMode::multiple &&
                           config.precision != CudaFamilyPrecision::boundsCandidate;
    const bool  single = config.precision == CudaFamilyPrecision::exactSingle;
    const char* familyKernel =
        config.family == CudaFamilyKind::boxSweep ? "ovspatialquery_cuda_family_plan_box"
        : config.family == CudaFamilyKind::overlap
            ? (single ? "ovspatialquery_cuda_family_plan_overlap_single" : "ovspatialquery_cuda_family_plan_overlap")
        : crossings ? (single ? "ovspatialquery_cuda_family_plan_crossings_single"
                              : "ovspatialquery_cuda_family_plan_crossings")
                    : (single ? "ovspatialquery_cuda_family_plan_single" : "ovspatialquery_cuda_family_plan");
    if (driver.contextSetCurrent(context) != kCuSuccess ||
        driver.moduleGetFunction(&function, module, familyKernel) != kCuSuccess ||
        driver.moduleGetFunction(&scanFunction, module, "ovspatialquery_cuda_family_scan") != kCuSuccess ||
        driver.moduleGetFunction(&exactFunction, module, "ovspatialquery_cuda_exact_catalogue") != kCuSuccess)
    {
        driver.contextSetCurrent(submissionContext);
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    // The family kernels' stack is near or above the driver's 1 KiB default and
    // differs per architecture; growing it here keeps enqueue from waiting on the device.
    if (const CuResult stack = reserveStack(driver, module, function); stack != kCuSuccess)
    {
        driver.contextSetCurrent(submissionContext);
        return stack == kCuOutOfMemory ? OVSPATIALQUERY_API_OUT_OF_MEMORY : OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    try
    {
        slots.resize(config.resultSlotCount);
        if (slots.capacity() != config.resultSlotCount)
        {
            throw std::length_error("CUDA family result-slot storage exceeded its charge");
        }
        const uint64_t hitCapacity = config.hitCapacityPerSlot;
        CuDevicePtr    mirrors     = 0u;
        if (driver.memHostAlloc(&hostStoredHitCounts, slots.size() * sizeof(uint64_t),
                                kCuMemHostAllocPortable | kCuMemHostAllocMapped) != kCuSuccess)
        {
            hostStoredHitCounts = nullptr;
            driver.contextSetCurrent(submissionContext);
            return OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        if (driver.hostDevicePointer(&mirrors, hostStoredHitCounts, 0u) != kCuSuccess)
        {
            driver.contextSetCurrent(submissionContext);
            return OVSPATIALQUERY_API_NOT_SUPPORTED;
        }
        for (size_t index = 0u; index < slots.size(); ++index)
        {
            slots[index].hostStoredHitCount   = static_cast<uint64_t*>(hostStoredHitCounts) + index;
            slots[index].storedHitCountMirror = mirrors + index * sizeof(uint64_t);
        }
        if (config.hostMapped)
        {
            uint64_t mappedBytes = 0u;
            if (!layoutCudaFamilyMappedColumns(config.maximumQueryCount, hitCapacity, config.resultFields,
                                               mappedLayout) ||
                !addHostCharge(slots.size(), mappedLayout.bytes, mappedBytes) ||
                mappedBytes > std::numeric_limits<size_t>::max() ||
                driver.memHostAlloc(&mappedColumns, static_cast<size_t>(mappedBytes),
                                    kCuMemHostAllocPortable | kCuMemHostAllocMapped) != kCuSuccess)
            {
                mappedColumns = nullptr;
                driver.contextSetCurrent(submissionContext);
                return OVSPATIALQUERY_API_OUT_OF_MEMORY;
            }
            // The caller reads and the device writes one address, which unified
            // addressing makes the host address.
            CuDevicePtr mappedDevice = 0u;
            if (driver.hostDevicePointer(&mappedDevice, mappedColumns, 0u) != kCuSuccess ||
                mappedDevice != static_cast<CuDevicePtr>(reinterpret_cast<uintptr_t>(mappedColumns)) ||
                driver.moduleGetFunction(&publishFunction, module, "ovspatialquery_cuda_family_publish_mapped") !=
                    kCuSuccess)
            {
                driver.contextSetCurrent(submissionContext);
                return OVSPATIALQUERY_API_NOT_SUPPORTED;
            }
            for (size_t index = 0u; index < slots.size(); ++index)
            {
                slots[index].mapped = mappedDevice + index * mappedLayout.bytes;
            }
        }
        if (signalRings.create(driver, 2u * static_cast<uint32_t>(slots.size())) != kCuSuccess)
        {
            driver.contextSetCurrent(submissionContext);
            return OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        for (CudaFamilyResultSlot& slot : slots)
        {
            slot.allocations.reserve(kResultSlotAllocations);
            if (slot.allocations.capacity() != kResultSlotAllocations)
            {
                throw std::length_error("CUDA family allocation table exceeded its charge");
            }
            const size_t q = static_cast<size_t>(config.maximumQueryCount);
            const size_t c = static_cast<size_t>(hitCapacity);
            if (slot.create(driver) != kCuSuccess ||
                allocate(slot.queryStatus, q * sizeof(uint32_t), slot) != kCuSuccess ||
                allocate(slot.storedHitCount, sizeof(uint64_t), slot) != kCuSuccess ||
                allocate(slot.hitOffsets, q * sizeof(uint64_t), slot) != kCuSuccess ||
                allocate(slot.hitCounts, q * sizeof(uint64_t), slot) != kCuSuccess ||
                allocate(slot.truncated, q * sizeof(uint8_t), slot) != kCuSuccess ||
                allocate(slot.rowPlans, q * sizeof(cuda_family::RowPlan), slot) != kCuSuccess ||
                (config.hitMode != CudaFamilyHitMode::multiple &&
                 allocate(slot.stash, q * size_t(cuda_family::kStashRowBytes), slot) != kCuSuccess) ||
                allocate(slot.objectIds, c * sizeof(uint64_t), slot) != kCuSuccess ||
                ((config.resultFields & kCudaFamilyResultGeometryIdentity) != 0u &&
                 (allocate(slot.geometryTypes, c * sizeof(int32_t), slot) != kCuSuccess ||
                  allocate(slot.geometryValues, c * sizeof(uint64_t), slot) != kCuSuccess)) ||
                ((config.resultFields & kCudaFamilyResultMaterial) != 0u &&
                 allocate(slot.materialIds, c * sizeof(uint64_t), slot) != kCuSuccess) ||
                ((config.resultFields & kCudaFamilyResultPosition) != 0u &&
                 allocate(slot.positions, c * 3u * sizeof(double), slot) != kCuSuccess) ||
                ((config.resultFields & kCudaFamilyResultQueryPosition) != 0u &&
                 allocate(slot.queryPositions, c * 3u * sizeof(double), slot) != kCuSuccess) ||
                ((config.resultFields & kCudaFamilyResultNormal) != 0u &&
                 allocate(slot.normals, c * 3u * sizeof(float), slot) != kCuSuccess) ||
                (config.family != CudaFamilyKind::overlap &&
                 allocate(slot.distances, c * sizeof(double), slot) != kCuSuccess) ||
                ((config.resultFields & kCudaFamilyResultSourceFeature) != 0u &&
                 (allocate(slot.featureKinds, c * sizeof(uint32_t), slot) != kCuSuccess ||
                  allocate(slot.faceIndices, c * sizeof(uint64_t), slot) != kCuSuccess ||
                  allocate(slot.corners, c * 3u * sizeof(uint32_t), slot) != kCuSuccess ||
                  allocate(slot.barycentrics, c * 3u * sizeof(float), slot) != kCuSuccess)))
            {
                driver.contextSetCurrent(submissionContext);
                return OVSPATIALQUERY_API_OUT_OF_MEMORY;
            }
        }
    }
    catch (const std::bad_alloc&)
    {
        driver.contextSetCurrent(submissionContext);
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    catch (const std::length_error&)
    {
        driver.contextSetCurrent(submissionContext);
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    catch (...)
    {
        driver.contextSetCurrent(submissionContext);
        return OVSPATIALQUERY_API_INTERNAL;
    }
    if (driver.contextSetCurrent(submissionContext) != kCuSuccess)
    {
        return OVSPATIALQUERY_API_INTERNAL;
    }
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t CudaFamilyFamilyPlan::measure(const CudaFamilyLaunch&  launchInput,
                                                          const GenerationLease&   generation,
                                                          CudaFamilyStagingCharge& outCharge) noexcept
{
    outCharge = {};
    if (!generation.initialized() || !validateCudaFamilyLaunch(config, launchInput))
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    std::shared_ptr<CudaFamilyCatalogueStaging> staging;
    try
    {
        staging = std::make_shared<CudaFamilyCatalogueStaging>();
        CudaExactMembership membership{};
        if (derivesExactCatalogue(config, generation, membership))
        {
            const ovspatialquery_api_status_t stagedStatus = measureExact(generation, membership, *staging);
            if (stagedStatus != OVSPATIALQUERY_API_SUCCESS)
            {
                return stagedStatus;
            }
        }
        else
        {
            std::lock_guard<std::mutex>       lock(catalogueMutex);
            const ovspatialquery_api_status_t stagedStatus = stageLocked(generation, *staging);
            if (stagedStatus != OVSPATIALQUERY_API_SUCCESS)
            {
                return stagedStatus;
            }
        }
        for (const CudaMeshDefinitionStaging& definition : staging->definitions)
        {
            if (definition.resident == nullptr)
            {
                outCharge.publishedDefinitionBytes.push_back(definition.deviceBytes);
            }
        }
        // A layout this launch builds is reserved last, after its new definitions.
        if (staging->exactBuild != nullptr)
        {
            outCharge.publishedDefinitionBytes.push_back(staging->layoutDeviceBytes);
        }
    }
    catch (const std::bad_alloc&)
    {
        outCharge = {};
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    outCharge.publishedDeviceBytes                               = staging->tableDeviceBytes;
    outCharge.hostBytes                                          = staging->hostBytes;
    outCharge.staging                                            = std::move(staging);
    uint64_t                          copiedInputBytes           = 0u;
    uint64_t                          copiedInputAllocationCount = 0u;
    const ovspatialquery_api_status_t copiedStatus =
        measureCudaFamilyCopiedInputs(launchInput, copiedInputBytes, copiedInputAllocationCount);
    if (copiedStatus != OVSPATIALQUERY_API_SUCCESS)
    {
        outCharge = {};
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    outCharge.deviceBytes                     = copiedInputBytes;
    constexpr uint64_t sharedControlAllowance = 3u * sizeof(void*);
    const uint64_t     operationStorage       = sizeof(CudaFamilyFamilyOperation) + sharedControlAllowance;
    if (operationStorage > UINT64_MAX - outCharge.hostBytes ||
        copiedInputAllocationCount >
            (UINT64_MAX - outCharge.hostBytes - operationStorage) / sizeof(CopiedInputAllocation))
    {
        outCharge = {};
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    outCharge.hostBytes += operationStorage + copiedInputAllocationCount * sizeof(CopiedInputAllocation);
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t CudaFamilyFamilyPlan::stageLocked(const GenerationLease&      generation,
                                                              CudaFamilyCatalogueStaging& outStaging) noexcept
{
    for (const std::shared_ptr<CudaFamilyDeviceCatalogue>& retained : catalogues)
    {
        if (retained != nullptr && retained->matches(generation))
        {
            outStaging.catalogue = retained;
            return OVSPATIALQUERY_API_SUCCESS;
        }
    }
    return measureCudaFamilyCatalogue(config, generation, *meshResidency, outStaging);
}

ovspatialquery_api_status_t CudaFamilyFamilyPlan::measureExact(const GenerationLease&      generation,
                                                               const CudaExactMembership&  membership,
                                                               CudaFamilyCatalogueStaging& outStaging) noexcept
{
    outStaging.exact           = true;
    outStaging.exactMembership = membership;
    {
        std::lock_guard<std::mutex> lock(catalogueMutex);
        for (const std::shared_ptr<CudaFamilyDeviceCatalogue>& retained : catalogues)
        {
            if (retained != nullptr && retained->matches(generation))
            {
                outStaging.catalogue = retained;
                return OVSPATIALQUERY_API_SUCCESS;
            }
        }
        for (const std::shared_ptr<CudaFamilyDeviceCatalogue>& retained : catalogues)
        {
            if (retained != nullptr && retained->exactLayout != nullptr && retained->exactLayout->matches(membership))
            {
                outStaging.exactLayout = retained->exactLayout;
                break;
            }
        }
    }
    // Per launch the host only sizes the version's tables from the layout; the
    // slot history is read once per membership, when no retained catalogue's layout matches.
    constexpr uint64_t pointerPair = sizeof(CuDevicePtr) + sizeof(void*);
    outStaging.hostBytes           = sizeof(CudaFamilyCatalogueStaging) + 3u * sizeof(void*);
    if (outStaging.exactLayout == nullptr)
    {
        std::shared_ptr<CudaExactLayoutBuild> build;
        try
        {
            build                     = std::make_shared<CudaExactLayoutBuild>();
            build->layout             = std::make_shared<CudaExactLayout>();
            build->layout->driver     = &driver;
            build->layout->context    = context;
            build->layout->deviceLost = &deviceLost;
        }
        catch (const std::bad_alloc&)
        {
            return OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        const ovspatialquery_api_status_t built = planExactLayout(config, generation.cudaVersion(), membership, *build);
        if (built != OVSPATIALQUERY_API_SUCCESS)
        {
            return built;
        }
        try
        {
            outStaging.definitions.reserve(build->meshOwners.size());
            for (const std::shared_ptr<const ExactGeometry>& owner : build->meshOwners)
            {
                ExactGeometryDescription description{};
                if (exportExactGeometryDescription(owner, description) != OVSPATIALQUERY_API_SUCCESS)
                {
                    return OVSPATIALQUERY_API_NOT_SUPPORTED;
                }
                const TriangleMeshExactGeometryDescription* mesh =
                    std::get_if<TriangleMeshExactGeometryDescription>(&description);
                CudaMeshDefinitionStaging definition{};
                if (mesh == nullptr || !stageMeshDefinition(*mesh, owner, *meshResidency, definition))
                {
                    return mesh == nullptr ? OVSPATIALQUERY_API_NOT_SUPPORTED : OVSPATIALQUERY_API_OUT_OF_MEMORY;
                }
                outStaging.definitions.push_back(std::move(definition));
            }
        }
        catch (const std::bad_alloc&)
        {
            return OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        // The layout's host tables and their pinned staging, the slot history's
        // per-slot copy they are filled from, and the new definitions' builds:
        // all made by the launch, after this charge is reserved.
        outStaging.layoutDeviceBytes = exactLayoutBytes(*build);
        const uint64_t slotCopy      = membership.slots * (sizeof(int32_t) + sizeof(uint32_t));
        if (outStaging.layoutDeviceBytes > (UINT64_MAX - slotCopy) / 4u)
        {
            return OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        outStaging.hostBytes += 2u * outStaging.layoutDeviceBytes + slotCopy + sizeof(CudaExactLayoutBuild) +
                                sizeof(CudaExactLayout) + kCatalogueTableAllocations * pointerPair;
        if (!addDefinitionHostBytes(outStaging.definitions, outStaging.hostBytes))
        {
            return OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        outStaging.exactLayout = build->layout;
        outStaging.exactBuild  = std::move(build);
    }
    outStaging.tableDeviceBytes = exactTables(*outStaging.exactLayout).bytes;
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t CudaFamilyFamilyPlan::prepare(CudaFamilyLaunch launchInput, GenerationLease generation,
                                                          std::shared_ptr<void>                 staging,
                                                          std::shared_ptr<CudaFamilyOperation>& outPrepared) noexcept
{
    try
    {
        const std::shared_ptr<const CudaPopulationVersion> version = generation.cudaVersion();
        const CudaPopulationDescription                    description =
            version != nullptr ? version->description() : CudaPopulationDescription{};
        if (takeCudaFailurePoint(kCudaFailureFamilyPrepare))
        {
            throw std::length_error("CUDA family catalogue test boundary");
        }
        std::shared_ptr<CudaFamilyFamilyOperation> operation = std::make_shared<CudaFamilyFamilyOperation>(
            shared_from_this(), UINT32_MAX, std::move(launchInput), std::move(generation));
        std::shared_ptr<CudaFamilyCatalogueStaging> measured =
            std::static_pointer_cast<CudaFamilyCatalogueStaging>(std::move(staging));
        if (measured == nullptr)
        {
            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
        }
        operation->mCatalogueStaging = std::move(measured);
        if (operation->mCatalogueStaging->catalogue != nullptr)
        {
            operation->mCatalogue = operation->mCatalogueStaging->catalogue;
        }
        else
        {
            operation->mCatalogue             = std::make_shared<CudaFamilyDeviceCatalogue>();
            operation->mCatalogue->driver     = &driver;
            operation->mCatalogue->context    = context;
            operation->mCatalogue->deviceLost = &deviceLost;
            operation->mCatalogue->generation = operation->mGeneration;
            if (borrowsDeviceTables(config, description))
            {
                CudaFamilyDeviceCatalogue& catalogue = *operation->mCatalogue;
                catalogue.borrowedStorage            = true;
                catalogue.ready                      = reinterpret_cast<CuEvent>(description.ready.value);
                catalogue.objectCount                = description.objectCount;
                catalogue.objectIds                  = reinterpret_cast<CuDevicePtr>(description.objectIds);
                catalogue.geometryTypes              = reinterpret_cast<CuDevicePtr>(description.geometryTypes);
                catalogue.geometryValues             = reinterpret_cast<CuDevicePtr>(description.geometryValues);
                catalogue.objectBits                 = reinterpret_cast<CuDevicePtr>(description.objectBits);
                catalogue.objectMinimums             = reinterpret_cast<CuDevicePtr>(description.objectMinimums);
                catalogue.objectMaximums             = reinterpret_cast<CuDevicePtr>(description.objectMaximums);
                catalogue.preparationFailed          = reinterpret_cast<CuDevicePtr>(description.preparationFailed);
                catalogue.grid                       = reinterpret_cast<CuDevicePtr>(description.grid);
            }
            else
            {
                operation->mBuildCatalogue = true;
            }
        }
        outPrepared = std::move(operation);
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

ovspatialquery_api_status_t measureCudaFamilyCatalogue(const CudaFamilyPlanConfig& config,
                                                       const GenerationLease&      generation,
                                                       CudaMeshResidencyIndex&     resident,
                                                       CudaFamilyCatalogueStaging& out) noexcept
{
    out.counts           = {};
    out.tableDeviceBytes = 0u;
    out.hostBytes        = 0u;
    out.definitions.clear();
    CudaFamilyCatalogueCounts&                         counts  = out.counts;
    const std::shared_ptr<const CudaPopulationVersion> version = generation.cudaVersion();
    if (version != nullptr && !borrowsDeviceTables(config, version->description()) &&
        version->description().devicePrepared && !generation.hasHostCopy())
    {
        return OVSPATIALQUERY_API_NOT_SUPPORTED; // an exact plan needs the version's host objects
    }
    if (version != nullptr && borrowsDeviceTables(config, version->description()))
    {
        const CudaPopulationDescription   description = version->description();
        const ovspatialquery_api_status_t status = validateDevicePreparedCatalogue(config, generation, description);
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return status;
        }
        if (description.objectCount > UINT64_MAX / 3u)
        {
            return OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        counts.objectIds      = description.objectCount;
        counts.geometryTypes  = description.objectCount;
        counts.geometryValues = description.objectCount;
        counts.objectBits     = description.objectCount * 2u;
        counts.objectMinimums = description.objectCount * 3u;
        counts.objectMaximums = description.objectCount * 3u;
        // The plan's fixed host charge covers the catalogue wrapper. This launch
        // needs only its staging descriptor; the version already charges its arrays.
        out.hostBytes = sizeof(CudaFamilyCatalogueStaging) + 3u * sizeof(void*);
        return OVSPATIALQUERY_API_SUCCESS;
    }
    const ObjectTable& objects = generation.objects();
    const auto         add     = [](uint64_t& total, uint64_t count)
    {
        if (count > std::numeric_limits<uint64_t>::max() - total)
        {
            return false;
        }
        total += count;
        return true;
    };
    const auto addProduct = [&add](uint64_t& total, uint64_t count, uint64_t width)
    { return count == 0u || (width <= UINT64_MAX / count && add(total, count * width)); };
    uint64_t faceMaterialTableCount = 0u;
    try
    {
        std::map<MeshDefinitionKey, uint64_t> meshDefinitions;
        // Face material tables are staged once per table the selected objects share.
        std::set<const std::vector<uint64_t>*> faceMaterialTables;
        const bool                             materials = (config.resultFields & kCudaFamilyResultMaterial) != 0u;
        for (size_t objectIndex = 0u; objectIndex < objects.size(); ++objectIndex)
        {
            if (!generation.includesObject(objectIndex))
            {
                continue;
            }
            if (materials && !add(counts.objectMaterials, 1u))
            {
                return OVSPATIALQUERY_API_OUT_OF_MEMORY;
            }
            if (!add(counts.objectIds, 1u) || !add(counts.geometryTypes, 1u) || !add(counts.geometryValues, 1u) ||
                !addProduct(counts.objectBits, 1u, 2u) || !addProduct(counts.objectMinimums, 1u, 3u) ||
                !addProduct(counts.objectMaximums, 1u, 3u))
            {
                return OVSPATIALQUERY_API_OUT_OF_MEMORY;
            }
            if (config.precision == CudaFamilyPrecision::boundsCandidate)
            {
                continue;
            }
            ExactGeometryDescription description{};
            if (exportExactGeometryDescription(objects[objectIndex].exactGeometry, description) !=
                OVSPATIALQUERY_API_SUCCESS)
            {
                return OVSPATIALQUERY_API_NOT_SUPPORTED;
            }
            if (!add(counts.exactObjects, 1u))
            {
                return OVSPATIALQUERY_API_OUT_OF_MEMORY;
            }
            if (std::holds_alternative<SphereExactGeometryDescription>(description))
            {
                if (!add(counts.spheres, 1u))
                {
                    return OVSPATIALQUERY_API_OUT_OF_MEMORY;
                }
            }
            else if (std::holds_alternative<BoxExactGeometryDescription>(description))
            {
                if (!add(counts.boxes, 1u))
                {
                    return OVSPATIALQUERY_API_OUT_OF_MEMORY;
                }
            }
            else if (std::holds_alternative<CapsuleExactGeometryDescription>(description))
            {
                if (!add(counts.capsules, 1u))
                {
                    return OVSPATIALQUERY_API_OUT_OF_MEMORY;
                }
            }
            else if (const ConvexExactGeometryDescription* convex =
                         std::get_if<ConvexExactGeometryDescription>(&description))
            {
                if (!add(counts.convexes, 1u) || !addProduct(counts.convexVertices, convex->vertexCount, 3u) ||
                    !addProduct(counts.convexTriangles, convex->triangleCount, 3u) ||
                    !addProduct(counts.convexPlanes, convex->planeCount, 4u) ||
                    !addProduct(counts.convexEdges, convex->edgeCount, 2u))
                {
                    return OVSPATIALQUERY_API_OUT_OF_MEMORY;
                }
            }
            else if (const TriangleMeshExactGeometryDescription* mesh =
                         std::get_if<TriangleMeshExactGeometryDescription>(&description))
            {
                if (!add(counts.meshes, 1u))
                {
                    return OVSPATIALQUERY_API_OUT_OF_MEMORY;
                }
                const std::shared_ptr<const std::vector<uint64_t>>& faceTable = objects[objectIndex].faceMaterialIds;
                if (materials && faceTable != nullptr && faceMaterialTables.insert(faceTable.get()).second &&
                    !add(counts.faceMaterials, faceTable->size()))
                {
                    return OVSPATIALQUERY_API_OUT_OF_MEMORY;
                }
                const MeshDefinitionKey key = meshDefinitionKey(*mesh);
                if (!meshDefinitions.emplace(key, meshDefinitions.size()).second)
                {
                    continue;
                }
                if (!add(counts.meshDefinitions, 1u))
                {
                    return OVSPATIALQUERY_API_OUT_OF_MEMORY;
                }
                CudaMeshDefinitionStaging staging{};
                if (!stageMeshDefinition(*mesh, objects[objectIndex].exactGeometry, resident, staging))
                {
                    return OVSPATIALQUERY_API_OUT_OF_MEMORY;
                }
                out.definitions.push_back(std::move(staging));
            }
            else
            {
                return OVSPATIALQUERY_API_NOT_SUPPORTED;
            }
        }
        faceMaterialTableCount = faceMaterialTables.size();
    }
    catch (const std::bad_alloc&)
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }

    const auto charge = [&addProduct, &out](uint64_t count, uint64_t size)
    { return addProduct(out.tableDeviceBytes, count, size); };
    constexpr uint64_t pointerPair = sizeof(CuDevicePtr) + sizeof(void*);
    if (!charge(counts.objectIds, sizeof(uint64_t)) || !charge(counts.geometryTypes, sizeof(uint32_t)) ||
        !charge(counts.geometryValues, sizeof(uint64_t)) || !charge(counts.objectBits, sizeof(uint64_t)) ||
        !charge(counts.objectMaterials, sizeof(uint64_t)) || !charge(counts.faceMaterials, sizeof(uint64_t)) ||
        !charge(counts.objectMinimums, sizeof(double)) || !charge(counts.objectMaximums, sizeof(double)) ||
        !charge(counts.exactObjects, sizeof(cuda_family::ExactObjectHeader)) ||
        !charge(counts.spheres, sizeof(cuda_family::SphereGeometry)) ||
        !charge(counts.boxes, sizeof(cuda_family::BoxGeometry)) ||
        !charge(counts.capsules, sizeof(cuda_family::CapsuleGeometry)) ||
        !charge(counts.convexes, sizeof(cuda_family::ConvexGeometry)) ||
        !charge(counts.convexVertices, sizeof(double)) || !charge(counts.convexTriangles, sizeof(uint32_t)) ||
        !charge(counts.convexPlanes, sizeof(double)) || !charge(counts.convexEdges, sizeof(uint32_t)) ||
        !charge(counts.meshes, sizeof(cuda_family::MeshGeometry)) ||
        !charge(counts.meshDefinitions, sizeof(cuda_family::MeshDefinition)) ||
        out.tableDeviceBytes > (UINT64_MAX - kCatalogueTableAllocations * pointerPair) / 2u)
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    // Host vectors and pinned staging of the tables, then per definition the
    // pinned staging of its arrays, the BVH build (nodes, leaf order and the
    // builder's triangle records) and the residency record.
    out.hostBytes = out.tableDeviceBytes * 2u + kCatalogueTableAllocations * pointerPair;
    // Each distinct face material table also takes a node in this measurement's set and
    // in the population's offset map.
    constexpr uint64_t faceTableNodes = 2u * (sizeof(void*) + sizeof(uint64_t) + 6u * sizeof(void*));
    if (!addProduct(out.hostBytes, faceMaterialTableCount, faceTableNodes))
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    return addDefinitionHostBytes(out.definitions, out.hostBytes) ? OVSPATIALQUERY_API_SUCCESS
                                                                  : OVSPATIALQUERY_API_OUT_OF_MEMORY;
}

bool CudaFamilyFamilyOperation::reserveCatalogueStorage() noexcept
{
    const auto reserve = [](auto& values, uint64_t count)
    {
        if (count > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
        {
            return false;
        }
        values.reserve(static_cast<size_t>(count));
        return values.capacity() == static_cast<size_t>(count);
    };
    try
    {
        const CudaFamilyCatalogueStaging& staging        = *mCatalogueStaging;
        size_t                            newDefinitions = 0u;
        for (const CudaMeshDefinitionStaging& definition : staging.definitions)
        {
            newDefinitions += definition.resident == nullptr ? 1u : 0u;
        }
        const size_t stagingCount = static_cast<size_t>(kCatalogueTableAllocations) +
                                    static_cast<size_t>(kDefinitionAllocations) * newDefinitions;
        mCatalogue->allocations.reserve(static_cast<size_t>(kCatalogueTableAllocations));
        mCatalogue->definitions.reserve(staging.definitions.size());
        mStaging.reserve(stagingCount);
        mDefinitionBuilds.resize(staging.definitions.size());
        bool buildsReserved = mCatalogue->definitions.capacity() == staging.definitions.size();
        for (size_t index = 0u; index < staging.definitions.size(); ++index)
        {
            const CudaMeshDefinitionStaging& definition = staging.definitions[index];
            if (definition.resident != nullptr)
            {
                continue;
            }
            buildsReserved = buildsReserved && reserve(mDefinitionBuilds[index].nodes, definition.nodeCount) &&
                             (definition.topologyBase != nullptr ||
                              reserve(mDefinitionBuilds[index].traversalTriangles, definition.triangleCount));
        }
        return buildsReserved && mCatalogue->allocations.capacity() == kCatalogueTableAllocations &&
               mStaging.capacity() == stagingCount && reserve(mObjectIds, staging.counts.objectIds) &&
               reserve(mGeometryTypes, staging.counts.geometryTypes) &&
               reserve(mGeometryValues, staging.counts.geometryValues) &&
               reserve(mObjectMaterials, staging.counts.objectMaterials) &&
               reserve(mFaceMaterials, staging.counts.faceMaterials) &&
               reserve(mObjectBits, staging.counts.objectBits) &&
               reserve(mObjectMinimums, staging.counts.objectMinimums) &&
               reserve(mObjectMaximums, staging.counts.objectMaximums) &&
               reserve(mExactObjects, staging.counts.exactObjects) && reserve(mSpheres, staging.counts.spheres) &&
               reserve(mBoxes, staging.counts.boxes) && reserve(mCapsules, staging.counts.capsules) &&
               reserve(mConvexes, staging.counts.convexes) && reserve(mConvexVertices, staging.counts.convexVertices) &&
               reserve(mConvexTriangles, staging.counts.convexTriangles) &&
               reserve(mConvexPlanes, staging.counts.convexPlanes) &&
               reserve(mConvexEdges, staging.counts.convexEdges) && reserve(mMeshes, staging.counts.meshes) &&
               reserve(mMeshDefinitions, staging.counts.meshDefinitions);
    }
    catch (const std::bad_alloc&)
    {
        return false;
    }
    catch (const std::length_error&)
    {
        return false;
    }
}

ovspatialquery_api_status_t CudaFamilyFamilyOperation::materializeCatalogue() noexcept
{
    if (!mBuildCatalogue)
    {
        return OVSPATIALQUERY_API_SUCCESS;
    }
    if (mCatalogueStaging->exact)
    {
        const ovspatialquery_api_status_t built = buildExactDefinitions();
        if (built != OVSPATIALQUERY_API_SUCCESS)
        {
            releaseCatalogueStorage();
        }
        return built;
    }
    if (!reserveCatalogueStorage())
    {
        releaseCatalogueStorage();
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    const ovspatialquery_api_status_t status = populateCatalogue();
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        releaseCatalogueStorage();
    }
    return status;
}

CuResult CudaFamilyFamilyOperation::stageCopiedInputs(CuStream stream) noexcept
{
    try
    {
        size_t copiedCount = 0u;
        for (const CudaFamilyInput* input : cudaFamilyRetainedInputs(mLaunch))
        {
            copiedCount += cudaFamilyInputCopied(*input) ? 1u : 0u;
        }
        mCopiedInputAllocations.reserve(copiedCount);
        if (mCopiedInputAllocations.capacity() != copiedCount)
        {
            return kCuOutOfMemory;
        }
        for (CudaFamilyInput* input : cudaFamilyRetainedInputs(mLaunch))
        {
            if (!cudaFamilyInputCopied(*input))
            {
                continue;
            }
            size_t bytes = 0u;
            if (!cudaFamilyInputBytes(*input, bytes))
            {
                return kCuInvalidHandle;
            }
            if (bytes == 0u)
            {
                continue;
            }
            CuDevicePtr    allocation = 0u;
            size_t         capacity   = 0u;
            const CuResult allocated =
                mOwner->stagingPool.take(mOwnerDriver(), mOwner->context, bytes, allocation, capacity);
            if (allocated != kCuSuccess)
            {
                return allocated;
            }
            mCopiedInputAllocations.emplace_back(allocation, capacity);
            const CuDevicePtr source = static_cast<CuDevicePtr>(reinterpret_cast<uintptr_t>(input->tensor.data));
            if (input->tensor.byte_offset > UINT64_MAX - source)
            {
                return kCuInvalidHandle;
            }
            // A strided copy queues several transfers, and one may fail after
            // others were queued: count the work as enqueued before trying.
            mWorkEnqueued         = true;
            const CuResult copied = copyInputColumnAsync(mOwnerDriver(), source + input->tensor.byte_offset,
                                                         input->layout, allocation, stream);
            if (copied != kCuSuccess)
            {
                return copied;
            }
#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
            noteCudaDeviceToDevice(static_cast<uint64_t>(bytes));
#endif
            input->tensor.data         = reinterpret_cast<void*>(static_cast<uintptr_t>(allocation));
            input->tensor.byte_offset  = 0u;
            input->tensor.strides      = nullptr;
            input->layout.rowStride    = input->layout.width;
            input->layout.columnStride = 1u;
            input->layout.compact      = true;
        }
        return kCuSuccess;
    }
    catch (const std::bad_alloc&)
    {
        return kCuOutOfMemory;
    }
    catch (const std::length_error&)
    {
        return kCuOutOfMemory;
    }
    catch (...)
    {
        return kCuInvalidHandle;
    }
}

ovspatialquery_api_status_t CudaFamilyFamilyOperation::populateCatalogue() noexcept
{
    namespace cudaFamily = cuda_family;
    try
    {
        const CudaFamilyPlanConfig&                      config = mOwner->config;
        std::map<MeshDefinitionKey, uint64_t>            meshDefinitionIndices;
        std::map<const std::vector<uint64_t>*, uint64_t> faceMaterialOffsets;
        const bool                        materials = (config.resultFields & kCudaFamilyResultMaterial) != 0u;
        const CudaFamilyCatalogueStaging& staging   = *mCatalogueStaging;
        const ObjectTable&                objects   = mGeneration.objects();
        for (size_t objectIndex = 0u; objectIndex < objects.size(); ++objectIndex)
        {
            if (!mGeneration.includesObject(objectIndex))
            {
                continue;
            }
            const PopulationObject& object = objects[objectIndex];
            mObjectIds.push_back(object.id.value);
            mGeometryTypes.push_back(static_cast<uint32_t>(object.geometryReference.family) +
                                     kGeometryTypeFirstBuiltin);
            mGeometryValues.push_back(object.geometryReference.value);
            if (materials)
            {
                mObjectMaterials.push_back(object.materialId);
            }
            mObjectBits.insert(mObjectBits.end(), object.participationBits.words, object.participationBits.words + 2u);
            mObjectMinimums.insert(mObjectMinimums.end(), object.bounds.minimum, object.bounds.minimum + 3u);
            mObjectMaximums.insert(mObjectMaximums.end(), object.bounds.maximum, object.bounds.maximum + 3u);
            if (config.precision == CudaFamilyPrecision::boundsCandidate)
            {
                continue;
            }
            ExactGeometryDescription description{};
            if (exportExactGeometryDescription(object.exactGeometry, description) != OVSPATIALQUERY_API_SUCCESS)
            {
                return OVSPATIALQUERY_API_NOT_SUPPORTED;
            }
            cudaFamily::ExactObjectHeader header{};
            if (const SphereExactGeometryDescription* sphere =
                    std::get_if<SphereExactGeometryDescription>(&description))
            {
                header.kind        = cudaFamily::kGeometrySphere;
                header.familyIndex = static_cast<unsigned long long>(mSpheres.size());
                cudaFamily::SphereGeometry value{};
                std::copy(sphere->center.begin(), sphere->center.end(), value.center);
                value.radius = sphere->radius;
                mSpheres.push_back(value);
            }
            else if (const BoxExactGeometryDescription* box = std::get_if<BoxExactGeometryDescription>(&description))
            {
                header.kind        = cudaFamily::kGeometryBox;
                header.familyIndex = static_cast<unsigned long long>(mBoxes.size());
                cudaFamily::BoxGeometry value{};
                std::copy(box->center.begin(), box->center.end(), value.center);
                for (uint32_t axis = 0u; axis < 3u; ++axis)
                {
                    std::copy(box->axes[axis].begin(), box->axes[axis].end(), value.axes + axis * 3u);
                }
                std::copy(box->halfExtents.begin(), box->halfExtents.end(), value.halfExtents);
                mBoxes.push_back(value);
            }
            else if (const CapsuleExactGeometryDescription* capsule =
                         std::get_if<CapsuleExactGeometryDescription>(&description))
            {
                header.kind        = cudaFamily::kGeometryCapsule;
                header.familyIndex = static_cast<unsigned long long>(mCapsules.size());
                cudaFamily::CapsuleGeometry value{};
                std::copy(capsule->endpointA.begin(), capsule->endpointA.end(), value.endpointA);
                std::copy(capsule->endpointB.begin(), capsule->endpointB.end(), value.endpointB);
                value.radius = capsule->radius;
                mCapsules.push_back(value);
            }
            else if (const ConvexExactGeometryDescription* convex =
                         std::get_if<ConvexExactGeometryDescription>(&description))
            {
                header.kind        = cudaFamily::kGeometryConvex;
                header.familyIndex = static_cast<unsigned long long>(mConvexes.size());
                cudaFamily::ConvexGeometry value{};
                value.vertexOffset   = mConvexVertices.size() / 3u;
                value.vertexCount    = convex->vertexCount;
                value.triangleOffset = mConvexTriangles.size() / 3u;
                value.triangleCount  = convex->triangleCount;
                value.planeOffset    = mConvexPlanes.size() / 4u;
                value.planeCount     = convex->planeCount;
                value.edgeOffset     = mConvexEdges.size() / 2u;
                value.edgeCount      = convex->edgeCount;
                if (convex->worldFromLocal == nullptr)
                {
                    mConvexVertices.insert(mConvexVertices.end(), convex->vertices,
                                           convex->vertices + convex->vertexCount * 3u);
                }
                else
                {
                    for (uint32_t axis = 0; axis < 3; ++axis)
                        value.translation[axis] = convex->worldFromLocal[axis * 4u + 3u];
                    for (uint64_t vertex = 0u; vertex < convex->vertexCount; ++vertex)
                    {
                        const double* local = convex->vertices + vertex * 3u;
                        for (uint32_t axis = 0u; axis < 3u; ++axis)
                        {
                            const double* row = convex->worldFromLocal + axis * 4u;
                            mConvexVertices.push_back(
                                std::fma(row[0], local[0], std::fma(row[1], local[1], row[2] * local[2])));
                        }
                    }
                }
                mConvexTriangles.insert(mConvexTriangles.end(), convex->triangleVertexIndices,
                                        convex->triangleVertexIndices + convex->triangleCount * 3u);
                for (uint64_t plane = 0u; plane < convex->planeCount; ++plane)
                {
                    const ConvexPlaneDescription& localPlane = convex->planes[plane];
                    if (convex->worldFromLocal == nullptr)
                    {
                        mConvexPlanes.insert(mConvexPlanes.end(), localPlane.normal.begin(), localPlane.normal.end());
                        mConvexPlanes.push_back(localPlane.offset);
                    }
                    else
                    {
                        std::array<double, 3u> worldNormal{};
                        for (uint32_t axis = 0u; axis < 3u; ++axis)
                        {
                            const double* row = convex->worldFromLocal + axis * 4u;
                            worldNormal[axis] =
                                std::fma(row[0], localPlane.normal[0],
                                         std::fma(row[1], localPlane.normal[1], row[2] * localPlane.normal[2])) /
                                convex->uniformScale;
                        }
                        mConvexPlanes.insert(mConvexPlanes.end(), worldNormal.begin(), worldNormal.end());
                        mConvexPlanes.push_back(localPlane.offset * convex->uniformScale);
                    }
                }
                for (uint64_t edge = 0u; edge < convex->edgeCount; ++edge)
                {
                    mConvexEdges.push_back(convex->edges[edge].first);
                    mConvexEdges.push_back(convex->edges[edge].second);
                }
                mConvexes.push_back(value);
            }
            else if (const TriangleMeshExactGeometryDescription* mesh =
                         std::get_if<TriangleMeshExactGeometryDescription>(&description))
            {
                header.kind                      = cudaFamily::kGeometryTriangleMesh;
                header.flags                     = mesh->doubleSided ? cudaFamily::kGeometryFlagDoubleSided : 0u;
                header.familyIndex               = static_cast<unsigned long long>(mMeshes.size());
                const MeshDefinitionKey key      = meshDefinitionKey(*mesh);
                const auto              inserted = meshDefinitionIndices.emplace(key, meshDefinitionIndices.size());
                const uint64_t          definitionIndex = inserted.first->second;
                // The staging visited the selected objects in this same first-seen order.
                if (definitionIndex >= staging.definitions.size() || !(staging.definitions[definitionIndex].key == key))
                {
                    return OVSPATIALQUERY_API_INTERNAL;
                }
                if (inserted.second && staging.definitions[definitionIndex].resident == nullptr)
                {
                    MeshBvhStructure&                                   build = mDefinitionBuilds[definitionIndex];
                    const std::shared_ptr<CudaMeshDefinitionResidency>& topologyBase =
                        staging.definitions[definitionIndex].topologyBase;
                    if (topologyBase != nullptr
                            ? !refitMeshBvh(*mesh, *topologyBase->structure, build.nodes)
                            : !buildMeshBvh(*mesh, build.nodes, build.traversalTriangles, config.workers))
                    {
                        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
                    }
                }
                cudaFamily::MeshGeometry value{};
                value.definitionIndex = definitionIndex;
                for (uint32_t row = 0u; row < 3u; ++row)
                {
                    for (uint32_t column = 0u; column < 4u; ++column)
                    {
                        value.worldFromLocal[row * 4u + column] = mesh->worldFromLocal[row * 4u + column];
                    }
                }
                if (!invertMeshLinear(mesh->worldFromLocal, value.localFromWorldLinear))
                {
                    return OVSPATIALQUERY_API_NOT_SUPPORTED;
                }
                if (materials && object.faceMaterialIds != nullptr)
                {
                    const auto placed =
                        faceMaterialOffsets.emplace(object.faceMaterialIds.get(), mFaceMaterials.size());
                    if (placed.second)
                    {
                        mFaceMaterials.insert(mFaceMaterials.end(), object.faceMaterialIds->begin(),
                                              object.faceMaterialIds->end());
                    }
                    value.faceMaterialOffset = placed.first->second;
                    value.faceMaterialCount  = object.faceMaterialIds->size();
                }
                mMeshes.push_back(value);
            }
            else
            {
                return OVSPATIALQUERY_API_NOT_SUPPORTED;
            }
            mExactObjects.push_back(header);
        }

        // Records are filled once the definitions are resident.
        mMeshDefinitions.resize(staging.definitions.size());
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

CuResult CudaFamilyFamilyOperation::stageDefinitions(CuStream stream) noexcept
{
    CudaFamilyCatalogueStaging& staging   = *mCatalogueStaging;
    CudaFamilyDeviceCatalogue&  catalogue = *mCatalogue;
    Driver&                     driver    = mOwnerDriver();
    try
    {
        size_t newDefinitions = 0u;
        for (const CudaMeshDefinitionStaging& definition : staging.definitions)
        {
            newDefinitions += definition.resident == nullptr ? 1u : 0u;
        }
        // A caller that reserved the measured definitions must have reserved exactly these.
        if (!mPublishedDefinitionReservations.empty() && mPublishedDefinitionReservations.size() != newDefinitions)
        {
            return kCuInvalidHandle;
        }
        size_t reservationIndex = 0u;
        for (size_t index = 0u; index < staging.definitions.size(); ++index)
        {
            const CudaMeshDefinitionStaging&             definition = staging.definitions[index];
            std::shared_ptr<CudaMeshDefinitionResidency> residency  = definition.resident;
            if (residency == nullptr)
            {
                residency         = std::make_shared<CudaMeshDefinitionResidency>();
                residency->device = mOwner->contextModule;
                residency->owner  = definition.owner;
                residency->allocations.reserve(static_cast<size_t>(kDefinitionAllocations));
                if (reservationIndex < mPublishedDefinitionReservations.size())
                {
                    residency->budgetReservation = std::move(mPublishedDefinitionReservations[reservationIndex]);
                }
                ++reservationIndex;
            }
            // The catalogue owns the residency from here, so a failure below leaves
            // its device arrays alive with this operation until completion, like the
            // tables, instead of freeing the targets of enqueued copies.
            catalogue.definitions.push_back(residency);
            if (definition.resident != nullptr)
            {
                // Uploaded by an earlier launch, possibly on another stream.
                const CuResult waited = driver.streamWaitEvent(stream, residency->ready, 0u);
                if (waited != kCuSuccess)
                {
                    return waited;
                }
                mWorkEnqueued           = true;
                mMeshDefinitions[index] = residency->record;
                continue;
            }
            ExactGeometryDescription description{};
            if (exportExactGeometryDescription(definition.owner, description) != OVSPATIALQUERY_API_SUCCESS)
            {
                return kCuInvalidHandle;
            }
            const TriangleMeshExactGeometryDescription* mesh =
                std::get_if<TriangleMeshExactGeometryDescription>(&description);
            if (mesh == nullptr || !(meshDefinitionKey(*mesh) == definition.key))
            {
                return kCuInvalidHandle;
            }
            MeshBvhStructure&            build         = mDefinitionBuilds[index];
            cuda_family::MeshDefinition& record        = residency->record;
            const size_t                 triangleCount = static_cast<size_t>(mesh->triangleCount);
            const size_t                 vertexBytes   = static_cast<size_t>(mesh->vertexCount) * 3u * sizeof(float);
            const size_t                 indexBytes    = triangleCount * 3u * sizeof(uint32_t);
            const size_t                 faceBytes     = triangleCount * sizeof(uint64_t);
            const size_t                 nodeBytes     = build.nodes.size() * sizeof(cuda_family::MeshBvhNode);
            const size_t                 leafBytes     = build.traversalTriangles.size() * sizeof(uint32_t);
            CuResult status = upload({{mesh->localVertices, vertexBytes, &record.vertices}}, residency->allocations);
            if (definition.topologyBase != nullptr)
            {
                // Refitted over the root's device index arrays, which its readiness event
                // covers; the root stays alive through the residency.
                const std::shared_ptr<CudaMeshDefinitionResidency> root =
                    definition.topologyBase->topologyRoot != nullptr ? definition.topologyBase->topologyRoot
                                                                     : definition.topologyBase;
                if (status == kCuSuccess)
                {
                    status = driver.streamWaitEvent(stream, root->ready, 0u);
                }
                record.triangles          = root->record.triangles;
                record.sourceFaces        = root->record.sourceFaces;
                record.sourceCorners      = root->record.sourceCorners;
                record.traversalTriangles = root->record.traversalTriangles;
                residency->topologyRoot   = root;
                residency->structure      = root->structure;
            }
            else
            {
                if (status == kCuSuccess)
                {
                    status =
                        upload({{mesh->triangleVertexIndices, indexBytes, &record.triangles}}, residency->allocations);
                }
                if (status == kCuSuccess)
                {
                    status = upload({{mesh->sourceFaces, faceBytes, &record.sourceFaces}}, residency->allocations);
                }
                if (status == kCuSuccess)
                {
                    status = upload({{mesh->sourceCorners, indexBytes, &record.sourceCorners}}, residency->allocations);
                }
            }
            if (status == kCuSuccess)
            {
                status = upload({{build.nodes.data(), nodeBytes, &record.nodes}}, residency->allocations);
            }
            if (status == kCuSuccess && definition.topologyBase == nullptr)
            {
                status = upload({{build.traversalTriangles.data(), leafBytes, &record.traversalTriangles}},
                                residency->allocations);
            }
            if (status == kCuSuccess)
            {
                status = driver.eventCreate(&residency->ready, kCuEventDisableTiming);
            }
            if (status == kCuSuccess)
            {
                status = driver.eventRecord(residency->ready, stream);
            }
            if (status != kCuSuccess)
            {
                return status;
            }
            record.vertexCount   = mesh->vertexCount;
            record.triangleCount = mesh->triangleCount;
            record.nodeCount     = build.nodes.size();
            if (definition.topologyBase == nullptr)
            {
                // The uploads copied into pinned staging; the host arrays now serve refits.
                residency->structure = std::make_shared<MeshBvhStructure>(std::move(build));
            }
            mMeshDefinitions[index]              = record;
            CudaMeshResidencyIndex&     resident = *mOwner->meshResidency;
            std::lock_guard<std::mutex> lock(resident.mutex);
            // Definitions no catalogue holds any more leave the index here, so it
            // stays bounded by the distinct meshes some live catalogue still shares.
            for (auto entry = resident.definitions.begin(); entry != resident.definitions.end();)
            {
                entry = entry->second.expired() ? resident.definitions.erase(entry) : std::next(entry);
            }
            for (auto entry = resident.topologies.begin(); entry != resident.topologies.end();)
            {
                entry = entry->second.expired() ? resident.topologies.erase(entry) : std::next(entry);
            }
            resident.definitions[definition.key] = residency;
            if (definition.topologyBase == nullptr)
            {
                resident.topologies[meshTopologyKey(*mesh)] = residency;
            }
        }
        return kCuSuccess;
    }
    catch (const std::bad_alloc&)
    {
        return kCuOutOfMemory;
    }
    catch (const std::length_error&)
    {
        return kCuOutOfMemory;
    }
    catch (...)
    {
        return kCuInvalidHandle;
    }
}

// Builds or refits the trees of the mesh definitions a new layout stages; the
// records are filled once they are resident.
ovspatialquery_api_status_t CudaFamilyFamilyOperation::buildExactDefinitions() noexcept
{
    try
    {
        const CudaFamilyCatalogueStaging& staging        = *mCatalogueStaging;
        size_t                            newDefinitions = 0u;
        for (const CudaMeshDefinitionStaging& definition : staging.definitions)
        {
            newDefinitions += definition.resident == nullptr ? 1u : 0u;
        }
        // Reserved here so the launch's bookkeeping cannot throw once it owns a
        // device or pinned allocation: the table allocation, the definitions, and
        // the pinned staging of each new definition's arrays and of a new layout.
        mCatalogue->allocations.reserve(static_cast<size_t>(kCatalogueTableAllocations));
        mCatalogue->definitions.reserve(staging.definitions.size());
        mStaging.reserve(static_cast<size_t>(kDefinitionAllocations) * newDefinitions + 1u);
        if (staging.exactBuild != nullptr)
        {
            staging.exactLayout->allocations.reserve(static_cast<size_t>(kCatalogueTableAllocations));
            const ovspatialquery_api_status_t filled = fillExactLayout(*staging.exactBuild);
            if (filled != OVSPATIALQUERY_API_SUCCESS)
            {
                return filled;
            }
        }
        mDefinitionBuilds.resize(staging.definitions.size());
        mMeshDefinitions.resize(staging.definitions.size());
        for (size_t index = 0u; index < staging.definitions.size(); ++index)
        {
            const CudaMeshDefinitionStaging& definition = staging.definitions[index];
            if (definition.resident != nullptr)
            {
                continue;
            }
            ExactGeometryDescription description{};
            if (exportExactGeometryDescription(definition.owner, description) != OVSPATIALQUERY_API_SUCCESS)
            {
                return OVSPATIALQUERY_API_INTERNAL;
            }
            const TriangleMeshExactGeometryDescription* mesh =
                std::get_if<TriangleMeshExactGeometryDescription>(&description);
            MeshBvhStructure& build = mDefinitionBuilds[index];
            build.nodes.reserve(static_cast<size_t>(definition.nodeCount));
            if (definition.topologyBase == nullptr)
            {
                build.traversalTriangles.reserve(static_cast<size_t>(definition.triangleCount));
            }
            if (mesh == nullptr ||
                (definition.topologyBase != nullptr
                     ? !refitMeshBvh(*mesh, *definition.topologyBase->structure, build.nodes)
                     : !buildMeshBvh(*mesh, build.nodes, build.traversalTriangles, mOwner->config.workers)))
            {
                return OVSPATIALQUERY_API_OUT_OF_MEMORY;
            }
        }
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

// Enqueues a catalogue derived on the device (BH-065): a new layout's definitions
// and tables first, otherwise a wait for the resident layout; then, after the
// version's own preparation, the kernel that writes the version's tables into
// `reuse` (an evicted catalogue's allocation of the same size) or a new
// allocation. The host does no work that grows with the objects here.
CuResult CudaFamilyFamilyOperation::stageExactCatalogue(CuStream stream, CuDevicePtr reuse) noexcept
{
    namespace cudaFamily                  = cuda_family;
    CudaFamilyCatalogueStaging& staging   = *mCatalogueStaging;
    CudaFamilyDeviceCatalogue&  catalogue = *mCatalogue;
    CudaExactLayout&            layout    = *staging.exactLayout;
    Driver&                     driver    = mOwnerDriver();
    // The catalogue owns a taken allocation before anything can fail; its
    // capacity was reserved when the catalogue was materialized.
    if (reuse != 0u)
    {
        catalogue.allocations.push_back(reuse);
    }
    try
    {
        CuResult status = kCuSuccess;
        if (staging.exactBuild != nullptr)
        {
            CudaExactLayoutBuild& build = *staging.exactBuild;
            if (!mPublishedDefinitionReservations.empty())
            {
                layout.budgetReservation = std::move(mPublishedDefinitionReservations.back());
                mPublishedDefinitionReservations.pop_back();
            }
            status = stageDefinitions(stream);
            if (status == kCuSuccess)
            {
                layout.definitions = catalogue.definitions;
                status             = upload(
                    {piece(build.headers, layout.headers), piece(build.convexTemplates, layout.convexTemplates),
                                 piece(build.convexSources, layout.convexSources), piece(build.localVertices, layout.localVertices),
                                 piece(build.localPlanes, layout.localPlanes), piece(build.meshTemplates, layout.meshTemplates),
                                 piece(build.faceMaterials, layout.faceMaterials), piece(mMeshDefinitions, layout.meshDefinitions),
                                 piece(build.convexTriangles, layout.convexTriangles),
                                 piece(build.convexEdges, layout.convexEdges)},
                    layout.allocations);
            }
            if (status == kCuSuccess)
            {
                status = driver.eventCreate(&layout.ready, kCuEventDisableTiming);
            }
            if (status == kCuSuccess)
            {
                status = driver.eventRecord(layout.ready, stream);
            }
        }
        else
        {
            // Uploaded by an earlier launch, possibly on another stream.
            status        = driver.streamWaitEvent(stream, layout.ready, 0u);
            mWorkEnqueued = mWorkEnqueued || status == kCuSuccess;
        }
        catalogue.exactLayout                       = staging.exactLayout;
        const CudaPopulationDescription description = mGeneration.cudaVersion()->description();
        if (status == kCuSuccess)
        {
            status        = driver.streamWaitEvent(stream, reinterpret_cast<CuEvent>(description.ready.value), 0u);
            mWorkEnqueued = mWorkEnqueued || status == kCuSuccess;
        }
        const CudaExactTables tables = exactTables(layout);
        CuDevicePtr           base   = reuse;
        if (status == kCuSuccess && base == 0u)
        {
            status = driver.memAlloc(&base, tables.bytes);
            if (status == kCuSuccess)
            {
                catalogue.allocations.push_back(base);
            }
        }
        if (status != kCuSuccess)
        {
            return status;
        }
        catalogue.tableBytes      = tables.bytes;
        catalogue.objectCount     = staging.exactMembership.slots;
        catalogue.objectIds       = reinterpret_cast<CuDevicePtr>(description.objectIds);
        catalogue.geometryTypes   = reinterpret_cast<CuDevicePtr>(description.geometryTypes);
        catalogue.geometryValues  = reinterpret_cast<CuDevicePtr>(description.geometryValues);
        catalogue.objectBits      = reinterpret_cast<CuDevicePtr>(description.objectBits);
        catalogue.objectMinimums  = base + tables.minimums;
        catalogue.objectMaximums  = base + tables.maximums;
        catalogue.objectMaterials = layout.materials ? base + tables.materials : 0u;
        catalogue.faceMaterials   = layout.faceMaterials;
        catalogue.exactObjects    = base + tables.objects;
        catalogue.spheres         = base + tables.spheres;
        catalogue.boxes           = base + tables.boxes;
        catalogue.capsules        = base + tables.capsules;
        catalogue.convexes        = base + tables.convexes;
        catalogue.convexVertices  = base + tables.convexVertices;
        catalogue.convexTriangles = layout.convexTriangles;
        catalogue.convexPlanes    = base + tables.convexPlanes;
        catalogue.convexEdges     = layout.convexEdges;
        catalogue.meshes          = base + tables.meshes;
        catalogue.meshDefinitions = layout.meshDefinitions;
        if (catalogue.objectCount == 0u)
        {
            // Nothing to derive: the launch reads the version's own status.
            catalogue.preparationFailed = reinterpret_cast<CuDevicePtr>(description.preparationFailed);
            return kCuSuccess;
        }
        catalogue.preparationFailed = base + tables.failed;
        status                      = driver.memsetD32Async(catalogue.preparationFailed, 0u, 2u, stream);
        mWorkEnqueued               = true;
        if (status != kCuSuccess)
        {
            return status;
        }
        const auto onDevice = [](auto pointer, auto deviceAddress)
        { return reinterpret_cast<decltype(pointer)>(static_cast<uintptr_t>(deviceAddress)); };
        cudaFamily::ExactCatalogueRecord record{};
        record.slotCount       = catalogue.objectCount;
        record.poses           = static_cast<const double*>(description.objectPoses);
        record.versionMinimums = static_cast<const double*>(description.objectMinimums);
        record.versionFailed   = static_cast<const unsigned int*>(description.preparationFailed);
        record.params          = static_cast<const double*>(staging.exactMembership.params);
        record.definitions     = static_cast<const unsigned int*>(staging.exactMembership.definitions);
        record.records         = static_cast<const cudaFamily::DeviceShapeRecord*>(staging.exactMembership.records);
        record.headers         = onDevice(record.headers, layout.headers);
        record.convexTemplates = onDevice(record.convexTemplates, layout.convexTemplates);
        record.convexSources   = onDevice(record.convexSources, layout.convexSources);
        record.localVertices   = onDevice(record.localVertices, layout.localVertices);
        record.localPlanes     = onDevice(record.localPlanes, layout.localPlanes);
        record.meshTemplates   = onDevice(record.meshTemplates, layout.meshTemplates);
        record.objects         = onDevice(record.objects, catalogue.exactObjects);
        record.minimums        = onDevice(record.minimums, catalogue.objectMinimums);
        record.maximums        = onDevice(record.maximums, catalogue.objectMaximums);
        record.materials       = onDevice(record.materials, catalogue.objectMaterials);
        record.spheres         = onDevice(record.spheres, catalogue.spheres);
        record.boxes           = onDevice(record.boxes, catalogue.boxes);
        record.capsules        = onDevice(record.capsules, catalogue.capsules);
        record.convexes        = onDevice(record.convexes, catalogue.convexes);
        record.convexVertices  = onDevice(record.convexVertices, catalogue.convexVertices);
        record.convexPlanes    = onDevice(record.convexPlanes, catalogue.convexPlanes);
        record.meshes          = onDevice(record.meshes, catalogue.meshes);
        record.failed          = onDevice(record.failed, catalogue.preparationFailed);
        constexpr unsigned int   threads = 128u;
        const unsigned long long blocks  = (catalogue.objectCount + threads - 1u) / threads;
        if (blocks > static_cast<unsigned long long>(std::numeric_limits<int32_t>::max()))
        {
            return kCuInvalidHandle;
        }
        void* parameters[] = {&record};
        return driver.launchKernel(mOwner->exactFunction, static_cast<unsigned int>(blocks), 1u, 1u, threads, 1u, 1u,
                                   0u, stream, parameters, nullptr);
    }
    catch (const std::bad_alloc&)
    {
        return kCuOutOfMemory;
    }
    catch (...)
    {
        return kCuInvalidHandle;
    }
}

// A new generation's catalogue evicts the least recently used of the retained
// ones. When nothing else holds that catalogue, its upload has completed and
// its tables took `bytes`, as they do when a frame-rate update republishes the
// same population, it is evicted now and its table allocation returned for the
// new tables. Freeing it instead waits for the whole device, including the host
// function a retired operation queued to wake the reaper, which is still pending
// when the launch reclaimed that operation's result slot itself (BH-006). Zero
// otherwise; the catalogue is then evicted, and its tables freed, as the new one
// is retained.
CuDevicePtr CudaFamilyFamilyPlan::takeEvictedTables(size_t bytes) noexcept
{
    std::shared_ptr<CudaFamilyDeviceCatalogue> evicted;
    CuDevicePtr                                tables = 0u;
    {
        std::lock_guard<std::mutex> lock(catalogueMutex);
        if (bytes == 0u || catalogues.size() < kCudaRetainedCatalogues)
        {
            return 0u;
        }
        // References to a retained catalogue are taken only under this mutex.
        const std::shared_ptr<CudaFamilyDeviceCatalogue>& oldest = catalogues.back();
        if (oldest == nullptr || oldest.use_count() != 1 || oldest->borrowedStorage ||
            oldest->allocations.size() != kCatalogueTableAllocations || oldest->tableBytes != bytes ||
            oldest->ready == nullptr || driver.eventQuery(oldest->ready) != kCuSuccess)
        {
            return 0u;
        }
        tables = oldest->allocations.front();
        oldest->allocations.clear();
        evicted = std::move(catalogues.back());
        catalogues.pop_back();
    }
    return tables;
}

ovspatialquery_api_status_t CudaFamilyFamilyPlan::launch(uint32_t                              resultSlot,
                                                         std::shared_ptr<CudaFamilyOperation>  prepared,
                                                         std::shared_ptr<CudaFamilyOperation>& outOperation) noexcept
{
    namespace cudaFamily = cuda_family;
    std::shared_ptr<CudaFamilyFamilyOperation> operation =
        std::dynamic_pointer_cast<CudaFamilyFamilyOperation>(std::move(prepared));
    if (resultSlot >= slots.size() || operation == nullptr || operation->mOwner.get() != this ||
        operation->mSlot != UINT32_MAX)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    if (workUnconfirmed.load(std::memory_order_relaxed))
    {
        // Earlier work may still write this slot's storage, so the launch ends with device
        // loss, which promises no storage, without touching the device.
        operation->mSlot = resultSlot;
        operation->configure(operation->mWorkEnqueued, false, false, false, true);
        outOperation = std::move(operation);
        return OVSPATIALQUERY_API_SUCCESS;
    }
    const ovspatialquery_api_status_t materializeStatus = operation->materializeCatalogue();
    if (materializeStatus != OVSPATIALQUERY_API_SUCCESS)
    {
        return materializeStatus;
    }
    operation->mSlot              = resultSlot;
    CudaFamilyLaunch& launchInput = operation->mLaunch;
    try
    {

        CudaCurrentContext scope(driver, context, deviceLost);
        CuResult           status = scope.status();
        CuStream           stream = reinterpret_cast<CuStream>(launchInput.stream);
        if (status == kCuSuccess && launchInput.waitEvent != 0u)
        {
            status = driver.streamWaitEvent(stream, reinterpret_cast<CuEvent>(launchInput.waitEvent), 0u);
            operation->mWorkEnqueued = operation->mWorkEnqueued || status == kCuSuccess;
        }
        if (status == kCuSuccess)
        {
            status = operation->stageCopiedInputs(stream);
        }
        CudaFamilyDeviceCatalogue& deviceCatalogue = *operation->mCatalogue;
        const bool                 exact           = operation->mBuildCatalogue && operation->mCatalogueStaging->exact;
        if (status == kCuSuccess && operation->mBuildCatalogue)
        {
            deviceCatalogue.budgetReservation = std::move(operation->mPublishedCatalogueReservation);
            status                            = driver.eventCreate(&deviceCatalogue.ready, kCuEventDisableTiming);
            if (status == kCuSuccess && exact)
            {
                status = operation->stageExactCatalogue(
                    stream, takeEvictedTables(exactTables(*operation->mCatalogueStaging->exactLayout).bytes));
            }
            else if (status == kCuSuccess)
            {
                // Shared definitions are waited for, new ones uploaded, before the tables that index them.
                status = operation->stageDefinitions(stream);
            }
        }
        if (status == kCuSuccess && operation->mBuildCatalogue && !exact)
        {
            // One allocation holds every table: the eight-byte tables first and
            // the three four-byte ones last, so none needs padding and the
            // device bytes are the measured table bytes.
            CudaFamilyFamilyOperation&                                          o = *operation;
            CudaFamilyDeviceCatalogue&                                          c = deviceCatalogue;
            const std::initializer_list<CudaFamilyFamilyOperation::UploadPiece> tables{
                o.piece(o.mObjectIds, c.objectIds),
                o.piece(o.mGeometryValues, c.geometryValues),
                o.piece(o.mObjectMaterials, c.objectMaterials),
                o.piece(o.mFaceMaterials, c.faceMaterials),
                o.piece(o.mObjectBits, c.objectBits),
                o.piece(o.mObjectMinimums, c.objectMinimums),
                o.piece(o.mObjectMaximums, c.objectMaximums),
                o.piece(o.mExactObjects, c.exactObjects),
                o.piece(o.mSpheres, c.spheres),
                o.piece(o.mBoxes, c.boxes),
                o.piece(o.mCapsules, c.capsules),
                o.piece(o.mConvexes, c.convexes),
                o.piece(o.mConvexVertices, c.convexVertices),
                o.piece(o.mConvexPlanes, c.convexPlanes),
                o.piece(o.mMeshes, c.meshes),
                o.piece(o.mMeshDefinitions, c.meshDefinitions),
                o.piece(o.mGeometryTypes, c.geometryTypes),
                o.piece(o.mConvexTriangles, c.convexTriangles),
                o.piece(o.mConvexEdges, c.convexEdges)};
            c.tableBytes = CudaFamilyFamilyOperation::uploadBytes(tables);
            status       = o.upload(tables, c.allocations, takeEvictedTables(c.tableBytes));
        }
        if (status == kCuSuccess && operation->mBuildCatalogue)
        {
            if (!exact)
            {
                deviceCatalogue.objectCount = operation->mObjectIds.size();
            }
            status                   = driver.eventRecord(deviceCatalogue.ready, stream);
            operation->mWorkEnqueued = operation->mWorkEnqueued || status == kCuSuccess;
            if (status == kCuSuccess)
            {
                // The newest catalogue goes first; one older generation stays queryable
                // without a rebuild, anything older is released with its tables.
                std::lock_guard<std::mutex> lock(catalogueMutex);
                catalogues.erase(std::remove_if(catalogues.begin(), catalogues.end(),
                                                [&](const std::shared_ptr<CudaFamilyDeviceCatalogue>& retained) {
                                                    return retained == nullptr ||
                                                           retained->matches(operation->mGeneration);
                                                }),
                                 catalogues.end());
                catalogues.insert(catalogues.begin(), operation->mCatalogue);
                if (catalogues.size() > kCudaRetainedCatalogues)
                {
                    catalogues.resize(kCudaRetainedCatalogues);
                }
            }
        }
        if (status == kCuSuccess)
        {
            status                   = driver.streamWaitEvent(stream, deviceCatalogue.ready, 0u);
            operation->mWorkEnqueued = operation->mWorkEnqueued || status == kCuSuccess;
        }
        CudaFamilyResultSlot&    output         = slots[resultSlot];
        const unsigned long long queryCount     = launchInput.queryCount;
        const unsigned long long resultCapacity = config.hitCapacityPerSlot;
        const auto               onDevice       = [](auto pointer, auto deviceAddress)
        { return reinterpret_cast<decltype(pointer)>(static_cast<uintptr_t>(deviceAddress)); };
        const auto column = [&onDevice](auto pointer, const CudaFamilyInput& input)
        {
            const DLTensor& tensor = input.tensor;
            return tensor.data == nullptr
                       ? decltype(pointer){nullptr}
                       : onDevice(pointer, reinterpret_cast<CuDevicePtr>(tensor.data) + tensor.byte_offset);
        };
        cudaFamily::PlanLaunchRecord launch{};
        cudaFamily::FamilyArguments& arguments = launch.arguments;
        arguments.family                       = static_cast<unsigned int>(config.family);
        arguments.precision                    = static_cast<unsigned int>(config.precision);
        arguments.overlapKind                  = static_cast<unsigned int>(config.overlapKind);
        arguments.hitMode                      = static_cast<unsigned int>(config.hitMode);
        arguments.resultFields                 = config.resultFields;
        for (size_t axis = 0u; axis < 3u; ++axis)
        {
            arguments.planHalfExtents[axis] = config.planHalfExtents[axis];
        }
        arguments.objectCount           = deviceCatalogue.objectCount;
        arguments.objectIds             = onDevice(arguments.objectIds, deviceCatalogue.objectIds);
        arguments.geometryTypes         = onDevice(arguments.geometryTypes, deviceCatalogue.geometryTypes);
        arguments.geometryValues        = onDevice(arguments.geometryValues, deviceCatalogue.geometryValues);
        arguments.objectMaterials       = onDevice(arguments.objectMaterials, deviceCatalogue.objectMaterials);
        arguments.faceMaterials         = onDevice(arguments.faceMaterials, deviceCatalogue.faceMaterials);
        arguments.outputMaterialIds     = onDevice(arguments.outputMaterialIds, output.materialIds);
        arguments.objectBits            = onDevice(arguments.objectBits, deviceCatalogue.objectBits);
        arguments.objectMinimums        = onDevice(arguments.objectMinimums, deviceCatalogue.objectMinimums);
        arguments.objectMaximums        = onDevice(arguments.objectMaximums, deviceCatalogue.objectMaximums);
        arguments.exactObjects          = onDevice(arguments.exactObjects, deviceCatalogue.exactObjects);
        arguments.spheres               = onDevice(arguments.spheres, deviceCatalogue.spheres);
        arguments.boxes                 = onDevice(arguments.boxes, deviceCatalogue.boxes);
        arguments.capsules              = onDevice(arguments.capsules, deviceCatalogue.capsules);
        arguments.convexes              = onDevice(arguments.convexes, deviceCatalogue.convexes);
        arguments.convexVertices        = onDevice(arguments.convexVertices, deviceCatalogue.convexVertices);
        arguments.convexTriangleIndices = onDevice(arguments.convexTriangleIndices, deviceCatalogue.convexTriangles);
        arguments.convexPlanes          = onDevice(arguments.convexPlanes, deviceCatalogue.convexPlanes);
        arguments.convexEdges           = onDevice(arguments.convexEdges, deviceCatalogue.convexEdges);
        arguments.meshes                = onDevice(arguments.meshes, deviceCatalogue.meshes);
        arguments.meshDefinitions       = onDevice(arguments.meshDefinitions, deviceCatalogue.meshDefinitions);
        arguments.outputObjectIds       = onDevice(arguments.outputObjectIds, output.objectIds);
        arguments.outputGeometryTypes   = onDevice(arguments.outputGeometryTypes, output.geometryTypes);
        arguments.outputGeometryValues  = onDevice(arguments.outputGeometryValues, output.geometryValues);
        arguments.outputPositions       = onDevice(arguments.outputPositions, output.positions);
        arguments.outputQueryPositions  = onDevice(arguments.outputQueryPositions, output.queryPositions);
        arguments.outputNormals         = onDevice(arguments.outputNormals, output.normals);
        arguments.outputDistances       = onDevice(arguments.outputDistances, output.distances);
        arguments.outputFeatureKinds    = onDevice(arguments.outputFeatureKinds, output.featureKinds);
        arguments.outputFaceIndices     = onDevice(arguments.outputFaceIndices, output.faceIndices);
        arguments.outputFaceCorners     = onDevice(arguments.outputFaceCorners, output.corners);
        arguments.outputBarycentrics    = onDevice(arguments.outputBarycentrics, output.barycentrics);
        arguments.stash                 = onDevice(arguments.stash, output.stash);
        arguments.grid                  = onDevice(arguments.grid, deviceCatalogue.grid);
        launch.pass                     = cudaFamily::kPassEvaluate;
        launch.rotationBits =
            launchInput.directionsOrRotations.present ? launchInput.directionsOrRotations.tensor.dtype.bits : 32u;
        launch.contactTolerance    = config.contactTolerance;
        launch.planRadius          = config.planRadius;
        launch.maximumHitsPerQuery = config.maximumHitsPerQuery;
        launch.queryCount          = queryCount;
        launch.resultCapacity      = resultCapacity;
        launch.queryBits           = column(launch.queryBits, launchInput.queryBits);
        launch.queryPositions      = column(launch.queryPositions, launchInput.positions);
        launch.queryDirectionsOrRotations =
            column(launch.queryDirectionsOrRotations, launchInput.directionsOrRotations);
        launch.queryRotations    = column(launch.queryRotations, launchInput.rotations);
        launch.minimumDistances  = column(launch.minimumDistances, launchInput.minimumDistances);
        launch.maximumDistances  = column(launch.maximumDistances, launchInput.maximumDistances);
        launch.radii             = column(launch.radii, launchInput.radii);
        launch.rowMaximumHits    = column(launch.rowMaximumHits, launchInput.rowMaximumHits);
        launch.queryHalfExtents  = column(launch.queryHalfExtents, launchInput.halfExtents);
        launch.queryFlags        = launchInput.queryFlags;
        launch.queryStatus       = onDevice(launch.queryStatus, output.queryStatus);
        launch.hitOffsets        = onDevice(launch.hitOffsets, output.hitOffsets);
        launch.rowPlans          = onDevice(launch.rowPlans, output.rowPlans);
        launch.preparationFailed = onDevice(launch.preparationFailed, deviceCatalogue.preparationFailed);
        void* parameters[]       = {&launch};
        void* scanParameters[]   = {
            const_cast<unsigned long long*>(&queryCount),
            const_cast<unsigned long long*>(&resultCapacity),
            &output.rowPlans,
            &output.storedHitCount,
            &output.storedHitCountMirror,
            &output.hitOffsets,
            &output.hitCounts,
            &output.truncated,
        };
        const uint32_t failurePoint = takeCudaFailurePoint();
        if (status == kCuSuccess && failurePoint == kCudaFailureFamilyLaunchBeforeKernels)
        {
            status = kCuInvalidHandle;
        }
        else if (status == kCuSuccess && failurePoint == kCudaFailureFamilyLaunchDeviceLost)
        {
            status = kCuDeviceLost;
        }
        // H's host copy where the scan does not write it, ordered before `ready`.
        const auto mirrorStoredHitCount = [&]()
        {
#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
            noteCudaDeviceToHost(sizeof(uint64_t));
#endif
            return driver.copyDtoHAsync(output.hostStoredHitCount, output.storedHitCount, sizeof(uint64_t), stream);
        };
        // A HOST_MAPPED plan's published H, cleared where the device H is.
        const auto clearMappedStoredHitCount = [&]()
        {
            return config.hostMapped
                       ? driver.memsetD32Async(publishedColumn(output, CudaFamilyMappedColumn::storedHitCount), 0u, 2u,
                                               stream)
                       : kCuSuccess;
        };
        // A HOST_MAPPED plan's one pass behind the write pass (ADR-0077): a grid
        // row per published column copies its live rows into mapped memory.
        const auto publishMapped = [&]()
        {
            cudaFamily::MappedPublishRecord record{};
            unsigned int                    columnCount = 0u;
            uint64_t                        largest     = 0u;
            for (size_t index = 0u; index < CudaFamilyMappedLayout::kCount; ++index)
            {
                if (mappedLayout.rowBytes[index] == 0u)
                {
                    continue;
                }
                const auto                    published = static_cast<CudaFamilyMappedColumn>(index);
                cudaFamily::MappedColumnCopy& copy      = record.columns[columnCount++];
                copy.source                             = deviceColumn(output, published);
                copy.destination                        = publishedColumn(output, published);
                copy.rowBytes                           = mappedLayout.rowBytes[index];
                copy.rows                               = mappedLayout.rows[index];
                const uint64_t rows                     = copy.rows == cudaFamily::kMappedRowsQuery ? queryCount
                                                          : copy.rows == cudaFamily::kMappedRowsHit ? resultCapacity
                                                                                                    : 1u;
                largest                                 = (std::max)(largest, rows * copy.rowBytes);
            }
            record.storedHitCount = onDevice(record.storedHitCount, output.storedHitCount);
            record.queryCount     = queryCount;
            // Enough blocks that each thread copies a few words of the largest column.
            constexpr uint64_t blockBytes = uint64_t(cudaFamily::kMappedThreads) * 16u * 4u;
            const unsigned int blocks =
                static_cast<unsigned int>((std::min)(uint64_t{64u}, (largest + blockBytes - 1u) / blockBytes));
            void* publishParameters[] = {&record};
            return driver.launchKernel(publishFunction, (std::max)(blocks, 1u), columnCount, 1u,
                                       cudaFamily::kMappedThreads, 1u, 1u, 0u, stream, publishParameters, nullptr);
        };
        // The evaluation writes every row's status and the scan writes H and its host
        // copy, so only a launch without rows clears H before `ready`.
        if (status == kCuSuccess && queryCount == 0u)
        {
            status                   = driver.memsetD32Async(output.storedHitCount, 0u, 2u, stream);
            operation->mWorkEnqueued = operation->mWorkEnqueued || status == kCuSuccess;
            if (status == kCuSuccess)
            {
                status = mirrorStoredHitCount();
            }
            if (status == kCuSuccess)
            {
                status = clearMappedStoredHitCount();
            }
        }
        if (status == kCuSuccess && queryCount != 0u)
        {
            // One block per query evaluates; one block packs the rows; one block per
            // query writes its granted hits. The grid dimension bounds a launch.
            if (queryCount > static_cast<unsigned long long>(std::numeric_limits<int32_t>::max()))
            {
                status = kCuInvalidHandle;
            }
            const unsigned int grid = static_cast<unsigned int>(queryCount);
            if (status == kCuSuccess)
            {
                status = driver.launchKernel(function, grid, 1u, 1u, cudaFamily::kEvaluateThreads, 1u, 1u, 0u, stream,
                                             parameters, nullptr);
                operation->mWorkEnqueued = true;
            }
            if (status == kCuSuccess)
            {
                status = driver.launchKernel(scanFunction, 1u, 1u, 1u, cudaFamily::kScanThreads, 1u, 1u, 0u, stream,
                                             scanParameters, nullptr);
            }
            if (status == kCuSuccess)
            {
                launch.pass = cudaFamily::kPassWrite;
                status = driver.launchKernel(function, grid, 1u, 1u, cudaFamily::kEvaluateThreads, 1u, 1u, 0u, stream,
                                             parameters, nullptr);
            }
            if (status == kCuSuccess && config.hostMapped)
            {
                status = publishMapped();
            }
        }
        if (status == kCuSuccess && failurePoint == kCudaFailureFamilyLaunchAfterKernels)
        {
            // Candidate-ray compatibility: fail only after the CudaFamily kernel is
            // accepted so the operation remains terminal-observable and its
            // staging/result leases cannot roll back early.
            status = kCuInvalidHandle;
        }
        bool readyRecorded = false;
        if (status == kCuSuccess)
        {
            status        = driver.eventRecord(output.ready, stream);
            readyRecorded = status == kCuSuccess;
        }
        bool failed = false;
        if (status != kCuSuccess && operation->mWorkEnqueued)
        {
            failed = true;
            if (status != kCuDeviceLost)
            {
                // Publish the authority-defined safe empty result after any
                // previously accepted work and before signaling readiness, in
                // the columns the caller reads: a HOST_MAPPED plan's are mapped
                // memory, which the memsets write directly. Payload rows need no
                // clearing because H becomes zero.
                using Column           = CudaFamilyMappedColumn;
                CuResult publishStatus = kCuSuccess;
                if (queryCount != 0u)
                {
                    publishStatus =
                        driver.memsetD32Async(publishedColumn(output, Column::queryStatus), cudaFamily::kBatchRowFailed,
                                              static_cast<size_t>(queryCount), stream);
                    if (publishStatus == kCuSuccess)
                    {
                        publishStatus = driver.memsetD32Async(publishedColumn(output, Column::hitOffsets), 0u,
                                                              static_cast<size_t>(queryCount) * 2u, stream);
                    }
                    if (publishStatus == kCuSuccess)
                    {
                        publishStatus = driver.memsetD32Async(publishedColumn(output, Column::hitCounts), 0u,
                                                              static_cast<size_t>(queryCount) * 2u, stream);
                    }
                    if (publishStatus == kCuSuccess)
                    {
                        publishStatus = driver.memsetD8Async(publishedColumn(output, Column::truncated), 0u,
                                                             static_cast<size_t>(queryCount), stream);
                    }
                }
                if (publishStatus == kCuSuccess)
                {
                    publishStatus = driver.memsetD32Async(output.storedHitCount, 0u, 2u, stream);
                }
                if (publishStatus == kCuSuccess)
                {
                    publishStatus = mirrorStoredHitCount();
                }
                if (publishStatus == kCuSuccess)
                {
                    publishStatus = clearMappedStoredHitCount();
                }
                if (publishStatus == kCuSuccess)
                {
                    publishStatus = driver.eventRecord(output.ready, stream);
                }
                status        = publishStatus;
                readyRecorded = status == kCuSuccess;
            }
        }
        // Without the wake event the worker polls `ready` instead.
        const bool     wakeRecorded = readyRecorded && driver.eventRecord(output.wake, stream) == kCuSuccess;
        const CuResult restored     = scope.restore();
        operation->configure(operation->mWorkEnqueued, readyRecorded, wakeRecorded, failed || restored != kCuSuccess,
                             status == kCuDeviceLost || restored == kCuDeviceLost);
        if (status == kCuDeviceLost && operation->mWorkEnqueued)
        {
            // Device loss terminalizes admitted work without a producer event.
            // The private lifecycle path will publish its dedicated failure;
            // do not erase the operation or synthesize result_ready.
            outOperation = std::move(operation);
            return OVSPATIALQUERY_API_SUCCESS;
        }
        if (status != kCuSuccess || (!operation->mWorkEnqueued && restored != kCuSuccess))
        {
            return status == kCuOutOfMemory ? OVSPATIALQUERY_API_OUT_OF_MEMORY
                                            : (status == kCuInvalidHandle ? OVSPATIALQUERY_API_INVALID_ARGUMENT
                                                                          : OVSPATIALQUERY_API_INTERNAL);
        }
        outOperation = std::move(operation);
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

CudaFamilyFamilyOperation::CudaFamilyFamilyOperation(std::shared_ptr<CudaFamilyFamilyPlan> owner, uint32_t slotIndex,
                                                     CudaFamilyLaunch launch, GenerationLease generation) noexcept
    : mOwner(std::move(owner)), mSlot(slotIndex), mLaunch(launch), mGeneration(std::move(generation)),
      mSourceOrdinal(mGeneration.ordinal())
{
    mReservationMetadata.set(mLaunch.queryCount, mOwner->config.hitCapacityPerSlot);
    mResultMetadata.set(mLaunch.queryCount, 0u);
}

void CudaFamilyFamilyOperation::releaseCatalogueStorage() noexcept
{
    decltype(mStaging){}.swap(mStaging);
    decltype(mObjectIds){}.swap(mObjectIds);
    decltype(mGeometryTypes){}.swap(mGeometryTypes);
    decltype(mGeometryValues){}.swap(mGeometryValues);
    decltype(mObjectMaterials){}.swap(mObjectMaterials);
    decltype(mFaceMaterials){}.swap(mFaceMaterials);
    decltype(mObjectBits){}.swap(mObjectBits);
    decltype(mObjectMinimums){}.swap(mObjectMinimums);
    decltype(mObjectMaximums){}.swap(mObjectMaximums);
    decltype(mExactObjects){}.swap(mExactObjects);
    decltype(mSpheres){}.swap(mSpheres);
    decltype(mBoxes){}.swap(mBoxes);
    decltype(mCapsules){}.swap(mCapsules);
    decltype(mConvexes){}.swap(mConvexes);
    decltype(mConvexVertices){}.swap(mConvexVertices);
    decltype(mConvexTriangles){}.swap(mConvexTriangles);
    decltype(mConvexPlanes){}.swap(mConvexPlanes);
    decltype(mConvexEdges){}.swap(mConvexEdges);
    decltype(mMeshes){}.swap(mMeshes);
    decltype(mMeshDefinitions){}.swap(mMeshDefinitions);
    decltype(mDefinitionBuilds){}.swap(mDefinitionBuilds);
    decltype(mPublishedDefinitionReservations){}.swap(mPublishedDefinitionReservations);
    mCatalogueStaging.reset();
}

Driver&               CudaFamilyFamilyOperation::mOwnerDriver() const noexcept { return mOwner->driver; }
CudaFamilyResultSlot& CudaFamilyFamilyOperation::slot() const noexcept { return mOwner->slots[mSlot]; }

CudaCompletionState CudaFamilyFamilyOperation::eventState(CuEvent event, bool queryStream) const noexcept
{
    CudaCurrentContext scope(mOwner->driver, mOwner->context, mOwner->deviceLost);
    if (scope.status() != kCuSuccess)
    {
        return scope.status() == kCuDeviceLost ? CudaCompletionState::deviceLost : CudaCompletionState::error;
    }
    if (event == slot().ready && (takeCudaFailurePoint(kCudaFailureResultPollContextSelection) ||
                                  cudaFailurePointArmed(kCudaFailureResultConfirmation)))
    {
        return CudaCompletionState::error;
    }
    const bool injectConsumerDeviceLost =
        !queryStream && event == slot().consumer && takeCudaFailurePoint(kCudaFailureConsumerPollDeviceLost);
    const CuResult status = injectConsumerDeviceLost
                                ? kCuDeviceLost
                                : (queryStream ? mOwner->driver.streamQuery(reinterpret_cast<CuStream>(mLaunch.stream))
                                               : mOwner->driver.eventQuery(event));
    if (status == kCuDeviceLost)
    {
        mOwner->deviceLost.store(true, std::memory_order_relaxed);
        mDeviceLost = true;
        return CudaCompletionState::deviceLost;
    }
    const CuResult restored = scope.restore();
    if (restored != kCuSuccess)
    {
        return restored == kCuDeviceLost ? CudaCompletionState::deviceLost : CudaCompletionState::error;
    }
    return status == kCuSuccess ? CudaCompletionState::complete
                                : (status == kCuNotReady ? CudaCompletionState::pending : CudaCompletionState::error);
}

// A terminal state releases the launch's staging and slot and lets a HOST_MAPPED plan's host
// reader read the columns (ADR-0077), and an error of the poll itself, a failed context switch
// or query, does not say the device is done with them. Waits for the ready event, or the
// stream when none was recorded: the launch is then complete, and its own failure, if any,
// applies as usual. Device loss when the plan's context cannot be reached or the wait fails,
// because nothing can then be promised; CUDA may report an earlier launch's error without
// having stopped the work, so the plan keeps everything the device may still use.
CudaCompletionState CudaFamilyFamilyOperation::confirmDeviceDone() const noexcept
{
    CudaCurrentContext scope(mOwner->driver, mOwner->context, mOwner->deviceLost);
    CuResult           status = scope.status();
    if (status == kCuSuccess && takeCudaFailurePoint(kCudaFailureResultConfirmation))
    {
        status = kCuInvalidContext;
    }
    if (status == kCuSuccess)
    {
        status = mReadyRecorded ? mOwner->driver.eventSynchronize(slot().ready)
                                : mOwner->driver.streamSynchronize(reinterpret_cast<CuStream>(mLaunch.stream));
    }
    (void)scope.restore();
    if (status != kCuSuccess)
    {
        mOwner->workUnconfirmed.store(true, std::memory_order_relaxed);
        mOwner->deviceLost.store(true, std::memory_order_relaxed);
        return CudaCompletionState::deviceLost;
    }
    return CudaCompletionState::complete;
}

CudaCompletionState CudaFamilyFamilyOperation::resultState() const noexcept
{
    std::lock_guard<std::mutex> lock(mMutex);
    if (mDeviceLost)
    {
        return CudaCompletionState::deviceLost;
    }
    if (!mWorkEnqueued)
    {
        return mFailed ? CudaCompletionState::error : CudaCompletionState::complete;
    }
    if (mReadyState != CudaCompletionState::pending)
    {
        return mReadyState;
    }
    CudaCompletionState state = eventState(slot().ready, !mReadyRecorded);
    if (state == CudaCompletionState::error)
    {
        state = confirmDeviceDone();
    }
    if (state == CudaCompletionState::complete && mFailed)
    {
        state = CudaCompletionState::error;
    }
    if (state == CudaCompletionState::complete)
    {
        // A publication selected before it completed (ADR-0021) ran before this
        // launch on the stream, so its outcome is known once ours is; its
        // failure is this launch's failure, and the rows already say FAILED.
        const std::shared_ptr<const CudaPopulationVersion> version = mGeneration.cudaVersion();
        if (version != nullptr && version->description().preparationFailed != nullptr)
        {
            const CudaCompletionState prepared = version->resultState();
            if (prepared == CudaCompletionState::pending)
            {
                state = CudaCompletionState::pending;
            }
            else if (prepared == CudaCompletionState::deviceLost)
            {
                state = CudaCompletionState::deviceLost;
            }
            else if (prepared != CudaCompletionState::complete)
            {
                state = CudaCompletionState::error;
            }
        }
    }
    if (state != CudaCompletionState::pending)
    {
        mReadyState = state;
    }
    return state;
}

void CudaFamilyFamilyOperation::waitForResult() const noexcept
{
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (!mWakeRecorded || mDeviceLost || mReadyState != CudaCompletionState::pending)
        {
            return;
        }
    }
    // Outside the mutex: resultState() and consumerState() stay available to
    // other threads while this one sleeps. A failure here is left to
    // resultState() to observe; device loss from the context switch is also
    // recorded for the plan.
    CudaCurrentContext scope(mOwner->driver, mOwner->context, mOwner->deviceLost);
    if (scope.status() == kCuSuccess)
    {
        (void)mOwner->driver.eventSynchronize(slot().wake);
    }
}

CudaCompletionState CudaFamilyFamilyOperation::consumerState() const noexcept
{
    std::lock_guard<std::mutex> lock(mMutex);
    if (!mConsumerRecorded)
    {
        return CudaCompletionState::complete;
    }
    if (mConsumerState == CudaCompletionState::pending)
    {
        mConsumerState = eventState(slot().consumer, false);
        if (mConsumerState == CudaCompletionState::deviceLost)
        {
            mDeviceLost = true;
        }
    }
    return mConsumerState;
}

bool CudaFamilyFamilyOperation::consumerDoneOwnershipReleased() const noexcept
{
    std::lock_guard<std::mutex> lock(mMutex);
    return mOwnedConsumerDone == nullptr;
}

void CudaFamilyFamilyOperation::releaseAfterDeviceLostLocked() noexcept
{
    (void)releaseTransferredConsumerDone(mOwner->driver, mOwner->context, mOwnedConsumerDone, true);
    // A lost device has no producer event and therefore no consumer-event
    // retirement path. Forget the device-owned bookkeeping and release the
    // host-side leases; context teardown owns the unavailable device state.
    mDeviceLost = true;
    mOwner->deviceLost.store(true, std::memory_order_relaxed);
    // Unconfirmed work may still read the staging; it is then left allocated, as the plan
    // leaves its own page-locked memory.
    if (!mOwner->workUnconfirmed.load(std::memory_order_relaxed))
    {
        for (void* pointer : mStaging)
        {
            (void)mOwner->driver.memFreeHost(pointer);
        }
    }
    decltype(mStaging){}.swap(mStaging);
    releaseCatalogueStorage();
    decltype(mCopiedInputAllocations){}.swap(mCopiedInputAllocations);
    mDeviceResultReservation.reset();
    mStagingBudgetReservation.reset();
    mHostStagingReservation.reset();
    mGeneration.reset();
}

void CudaFamilyFamilyOperation::releaseReadyDependencies() noexcept
{
    const CudaCompletionState state    = resultState();
    const CudaCompletionState consumer = consumerState();
    if (state == CudaCompletionState::deviceLost || consumer == CudaCompletionState::deviceLost)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        releaseAfterDeviceLostLocked();
        return;
    }
    if (state != CudaCompletionState::complete && state != CudaCompletionState::error)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(mMutex);
    // The failure point stands for a switch to the plan's context that reports
    // device loss, so the plan's context is never made current.
    if (takeCudaFailurePoint(kCudaFailureConsumerDestroyOuterContextDeviceLost))
    {
        releaseAfterDeviceLostLocked();
        return;
    }
    CudaCurrentContext scope(mOwner->driver, mOwner->context, mOwner->deviceLost);
    if (scope.status() != kCuSuccess)
    {
        if (scope.status() == kCuDeviceLost)
        {
            releaseAfterDeviceLostLocked();
        }
        return;
    }
    if (consumer != CudaCompletionState::pending)
    {
        if (releaseTransferredConsumerDone(mOwner->driver, mOwner->context, mOwnedConsumerDone, false))
        {
            releaseAfterDeviceLostLocked();
            return;
        }
    }
    for (void* pointer : mStaging)
    {
        mOwner->driver.memFreeHost(pointer);
    }
    decltype(mStaging){}.swap(mStaging);
    // The result and the consumer are done, so nothing reads these buffers.
    for (const auto& [pointer, bytes] : mCopiedInputAllocations)
    {
        mOwner->stagingPool.give(mOwner->driver, mOwner->context, pointer, bytes);
    }
    decltype(mCopiedInputAllocations){}.swap(mCopiedInputAllocations);
    releaseCatalogueStorage();
    mStagingBudgetReservation.reset();
    mHostStagingReservation.reset();
    mGeneration.reset();
}

ovspatialquery_api_status_t CudaFamilyFamilyOperation::transferConsumerDone(ovspatialquery_cuda_event_t event) noexcept
{
    std::lock_guard<std::mutex>       lock(mMutex);
    const ovspatialquery_api_status_t status = slot().transferConsumerDone(
        mOwner->driver, mOwner->context, mOwner->deviceLost, event, mDeviceLost, mConsumerRecorded);
    if (status == OVSPATIALQUERY_API_NOT_SUPPORTED)
    {
        mDeviceLost = true;
    }
    else if (status == OVSPATIALQUERY_API_SUCCESS)
    {
        mOwnedConsumerDone = reinterpret_cast<CuEvent>(event.value);
        mConsumerRecorded  = true;
    }
    return status;
}

void CudaFamilyFamilyOperation::ringWhenConsumed(const std::shared_ptr<DeviceSignal>& signal) const noexcept
{
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (!mConsumerRecorded || mDeviceLost)
        {
            return;
        }
    }
    CudaCurrentContext scope(mOwner->driver, mOwner->context, mOwner->deviceLost);
    if (scope.status() == kCuSuccess)
    {
        (void)slot().ringWhenConsumed(mOwner->driver, mOwner->signalRings, signal);
    }
}

void CudaFamilyFamilyOperation::attachDeviceResultReservation(std::shared_ptr<void> reservation) noexcept
{
    std::lock_guard<std::mutex> lock(mMutex);
    mDeviceResultReservation = std::move(reservation);
}

void CudaFamilyFamilyOperation::attachStagingReservation(std::shared_ptr<void> reservation) noexcept
{
    std::lock_guard<std::mutex> lock(mMutex);
    mStagingBudgetReservation = std::move(reservation);
}

void CudaFamilyFamilyOperation::attachHostStagingReservation(std::shared_ptr<HostLease> reservation) noexcept
{
    std::lock_guard<std::mutex> lock(mMutex);
    mHostStagingReservation = std::move(reservation);
}

void CudaFamilyFamilyOperation::configure(bool workEnqueued, bool readyRecorded, bool wakeRecorded, bool failed,
                                          bool deviceLost) noexcept
{
    std::lock_guard<std::mutex> lock(mMutex);
    mWorkEnqueued  = workEnqueued;
    mReadyRecorded = readyRecorded;
    mWakeRecorded  = wakeRecorded;
    mFailed        = failed;
    mDeviceLost    = deviceLost;
}

CudaFamilyCompletedResultOutcome CudaFamilyFamilyOperation::prepareResultMetadata() const noexcept
{
    std::lock_guard<std::mutex> lock(mMutex);
    if (mResultMetadataAttempted)
    {
        return mResultMetadataOutcome;
    }
    if (mDeviceLost)
    {
        return CudaFamilyCompletedResultOutcome::deviceLost;
    }
    mResultMetadataAttempted = true;
    // Another operation may have seen the plan's device lost since `ready` completed;
    // the columns are then unusable, as a device read would have reported.
    if (mOwner->deviceLost.load(std::memory_order_relaxed))
    {
        mDeviceLost            = true;
        mResultMetadataOutcome = CudaFamilyCompletedResultOutcome::deviceLost;
        return mResultMetadataOutcome;
    }
    // `ready` completed, so the scan, or the copy that stands in for it, has written
    // H's host copy; the device writes it, so it is read through a volatile pointer.
    uint64_t       storedHitCount = *static_cast<const volatile uint64_t*>(slot().hostStoredHitCount);
    const uint32_t failurePoint   = takeCudaFailurePoint();
    if (failurePoint == kCudaFailureResultMetadataDeviceLost)
    {
        mDeviceLost = true;
        mOwner->deviceLost.store(true);
        mResultMetadataOutcome = CudaFamilyCompletedResultOutcome::deviceLost;
        return mResultMetadataOutcome;
    }
    if (failurePoint == kCudaFailureResultMetadataTransfer)
    {
        mResultMetadataOutcome = CudaFamilyCompletedResultOutcome::transferFailure;
        return mResultMetadataOutcome;
    }
    if (failurePoint == kCudaFailureResultMetadataInvariant)
    {
        storedHitCount = mOwner->config.hitCapacityPerSlot + 1u;
    }
    if (storedHitCount > mOwner->config.hitCapacityPerSlot)
    {
        mResultMetadataOutcome = CudaFamilyCompletedResultOutcome::invariantFailure;
        return mResultMetadataOutcome;
    }
    mResultMetadata.set(mLaunch.queryCount, storedHitCount);
    mResultMetadataOutcome = CudaFamilyCompletedResultOutcome::success;
    return mResultMetadataOutcome;
}

CudaFamilyResultColumns CudaFamilyFamilyOperation::columns(const CudaFamilyColumnMetadata& metadata) const noexcept
{
    CudaFamilyResultSlot&   resultSlot = slot();
    CudaFamilyResultColumns result{};
    result.resultSlot      = mSlot;
    result.queryCount      = mLaunch.queryCount;
    result.hitCapacity     = mOwner->config.hitCapacityPerSlot;
    result.truth           = mOwner->config.precision == CudaFamilyPrecision::boundsCandidate
                                 ? cuda_family::kPrecisionBoundsCandidate
                                 : cuda_family::kPrecisionExact;
    result.sourceOrdinal   = mSourceOrdinal;
    result.executionDevice = mOwner->config.device;
    if (mDeviceLost)
    {
        return result;
    }
    result.resultReady.value = reinterpret_cast<uintptr_t>(resultSlot.ready);
    // A HOST_MAPPED plan's columns are pinned host memory the device also
    // addresses, which DLPack names { kDLCUDAHost, 0 } (ADR-0077).
    using Column                       = CudaFamilyMappedColumn;
    const CudaFamilyFamilyPlan& plan   = *mOwner;
    const DLDevice              device = plan.config.hostMapped ? DLDevice{kDLCUDAHost, 0} : plan.config.device;
    const auto                  tensor =
        [&](Column column, uint8_t code, uint8_t bits, int32_t rank, const int64_t* shape, const int64_t* strides)
    {
        DLTensor value = makeCudaTensor(plan.publishedColumn(resultSlot, column), device.device_id, code, bits, rank,
                                        shape, strides);
        value.device   = device;
        return value;
    };
    const auto     published   = [&](Column column) { return plan.publishedColumn(resultSlot, column) != 0u; };
    const int64_t* queryShape  = metadata.queryShape.data();
    const int64_t* queryStride = metadata.queryStride.data();
    const int64_t* hitShape    = metadata.hitShape.data();
    const int64_t* hitStride   = metadata.hitStride.data();
    const int64_t* vec3Shape   = metadata.vec3Shape.data();
    const int64_t* vec3Stride  = metadata.vec3Stride.data();
    result.queryStatus         = tensor(Column::queryStatus, kDLUInt, 32u, 1, queryShape, queryStride);
    result.storedHitCount =
        tensor(Column::storedHitCount, kDLUInt, 64u, 1, metadata.scalarShape.data(), metadata.scalarStride.data());
    result.hitOffsets = tensor(Column::hitOffsets, kDLUInt, 64u, 1, queryShape, queryStride);
    result.hitCounts  = tensor(Column::hitCounts, kDLUInt, 64u, 1, queryShape, queryStride);
    result.truncated  = tensor(Column::truncated, kDLUInt, 8u, 1, queryShape, queryStride);
    result.objectIds  = tensor(Column::objectIds, kDLUInt, 64u, 1, hitShape, hitStride);
    if (published(Column::geometryTypes))
    {
        result.geometryTypes  = tensor(Column::geometryTypes, kDLInt, 32u, 1, hitShape, hitStride);
        result.geometryValues = tensor(Column::geometryValues, kDLUInt, 64u, 1, hitShape, hitStride);
    }
    if (published(Column::materialIds))
    {
        result.materialIds = tensor(Column::materialIds, kDLUInt, 64u, 1, hitShape, hitStride);
    }
    if (published(Column::positions))
    {
        result.positions = tensor(Column::positions, kDLFloat, 64u, 2, vec3Shape, vec3Stride);
    }
    if (published(Column::queryPositions))
    {
        result.queryPositions = tensor(Column::queryPositions, kDLFloat, 64u, 2, vec3Shape, vec3Stride);
    }
    if (published(Column::normals))
    {
        result.normals = tensor(Column::normals, kDLFloat, 32u, 2, vec3Shape, vec3Stride);
    }
    // A raycast or sweep keeps device distances for its own ordering; they are
    // published only when the caller asked for them.
    if (published(Column::distances) && (plan.config.resultFields & kCudaFamilyResultDistance) != 0u)
    {
        result.distances = tensor(Column::distances, kDLFloat, 64u, 1, hitShape, hitStride);
    }
    if (published(Column::featureKinds))
    {
        result.sourceFeatureKinds = tensor(Column::featureKinds, kDLUInt, 32u, 1, hitShape, hitStride);
        result.sourceFaceIndices  = tensor(Column::faceIndices, kDLUInt, 64u, 1, hitShape, hitStride);
        result.sourceFaceCorners  = tensor(Column::corners, kDLUInt, 32u, 2, vec3Shape, vec3Stride);
        result.sourceBarycentrics = tensor(Column::barycentrics, kDLFloat, 32u, 2, vec3Shape, vec3Stride);
    }
    return result;
}

CudaFamilyResultColumns CudaFamilyFamilyOperation::reservation() const noexcept
{
    return columns(mReservationMetadata);
}

CudaFamilyCompletedResultOutcome CudaFamilyFamilyOperation::completedResult(
    CudaFamilyResultColumns& outResult) const noexcept
{
    const CudaCompletionState state = resultState();
    if (state == CudaCompletionState::deviceLost)
    {
        return CudaFamilyCompletedResultOutcome::deviceLost;
    }
    if (state != CudaCompletionState::complete)
    {
        return CudaFamilyCompletedResultOutcome::unavailable;
    }
    const CudaFamilyCompletedResultOutcome metadata = prepareResultMetadata();
    if (metadata != CudaFamilyCompletedResultOutcome::success)
    {
        return metadata;
    }
    outResult = columns(mResultMetadata);
    return CudaFamilyCompletedResultOutcome::success;
}

CudaFamilyFamilyOperation::~CudaFamilyFamilyOperation()
{
    const CudaCompletionState ready    = resultState();
    const CudaCompletionState consumer = consumerState();
    const bool                safe     = (!mWorkEnqueued || ready == CudaCompletionState::complete ||
                       ready == CudaCompletionState::error || ready == CudaCompletionState::deviceLost) &&
                      (consumer == CudaCompletionState::complete || consumer == CudaCompletionState::error ||
                       consumer == CudaCompletionState::deviceLost);
    assert(safe);
    if (!safe)
    {
        std::terminate();
    }
    releaseReadyDependencies();
    const bool ownershipReleased = consumerDoneOwnershipReleased();
    assert(ownershipReleased);
    if (!ownershipReleased)
    {
        std::terminate();
    }
}

} // namespace

bool estimateCudaFamilyPlanFixedHostByteCharge(uint32_t resultSlotCount, uint64_t& outHostByteCharge) noexcept
{
    outHostByteCharge = 0u;
    if (resultSlotCount == 0u)
    {
        return false;
    }
    constexpr uint64_t sharedControlAllowance = 3u * sizeof(void*);
    constexpr uint64_t catalogueHostBytes =
        sizeof(CudaFamilyDeviceCatalogue) + sharedControlAllowance + kCatalogueTableAllocations * sizeof(CuDevicePtr);
    uint64_t total = sizeof(CudaFamilyFamilyPlan) + sharedControlAllowance;
    // One plan-current catalogue plus one replaced catalogue per live result
    // slot bounds every generation retained until consumer completion.
    if (!addHostCharge(static_cast<uint64_t>(resultSlotCount) + 1u, catalogueHostBytes, total) ||
        !addHostCharge(resultSlotCount, sizeof(CudaFamilyResultSlot), total) ||
        !addHostCharge(resultSlotCount, kResultSlotAllocations * sizeof(CuDevicePtr), total) ||
        !addHostCharge(resultSlotCount, sizeof(uint64_t), total))
    {
        return false;
    }
    outHostByteCharge = total;
    return true;
}

#endif // defined(OVSPATIALQUERY_HAS_CUDA_BACKEND)

#if !defined(OVSPATIALQUERY_HAS_CUDA_BACKEND)
bool estimateCudaFamilyPlanFixedHostByteCharge(uint32_t resultSlotCount, uint64_t& outHostByteCharge) noexcept
{
    outHostByteCharge = 0u;
    (void)resultSlotCount;
    return false;
}
#endif

ovspatialquery_api_status_t createCudaFamilyPlan(const CudaFamilyPlanConfig&      config,
                                                 std::shared_ptr<CudaFamilyPlan>& outPlan) noexcept
{
#if defined(OVSPATIALQUERY_HAS_CUDA_BACKEND)
    try
    {
        const std::shared_ptr<CudaFamilyFamilyPlan> family = std::make_shared<CudaFamilyFamilyPlan>();
        const ovspatialquery_api_status_t           status = family->initialize(config);
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return status;
        }
        outPlan = family;
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
#else
    (void)config;
    (void)outPlan;
    return OVSPATIALQUERY_API_NOT_SUPPORTED;
#endif
}

} // namespace detail
} // namespace ovspatialquery
