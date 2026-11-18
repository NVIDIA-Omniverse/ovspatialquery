/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_geometry.h"
#include "ovspatialquery_instance_host_ledger.h"
#include "ovspatialquery_physx_backend.h"
#include "ovspatialquery_typed_geometry.h"
#include <PxPhysicsAPI.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

namespace ovspatialquery
{
namespace detail
{
struct PhysXInfrastructure
{
    physx::PxDefaultAllocator     allocator;
    physx::PxDefaultErrorCallback errorCallback;
    physx::PxFoundation*          foundation = nullptr;
    physx::PxPhysics*             physics    = nullptr;

    PhysXInfrastructure()
    {
        foundation = PxCreateFoundation(PX_PHYSICS_VERSION, allocator, errorCallback);
        if (foundation)
            physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, physx::PxTolerancesScale(), false, nullptr);
    }

    ~PhysXInfrastructure()
    {
        if (physics != nullptr)
        {
            physics->release();
        }
        if (foundation != nullptr)
        {
            foundation->release();
        }
    }
};

ovspatialquery_api_status_t acquirePhysXInfrastructure(std::shared_ptr<PhysXInfrastructure>& outInfrastructure)
{
    // A module lease avoids racing destruction against a new acquisition. Mesh
    // and runtime leases also make teardown safe regardless of static order.
    try
    {
        static const auto infrastructure = std::make_shared<PhysXInfrastructure>();
        if (!infrastructure->physics)
            return OVSPATIALQUERY_API_INTERNAL;
        outInfrastructure = infrastructure;
        return OVSPATIALQUERY_API_SUCCESS;
    }
    catch (const std::bad_alloc&)
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
}

physx::PxInsertionCallback* physxInsertionCallback(const std::shared_ptr<PhysXInfrastructure>& infrastructure)
{
    return infrastructure && infrastructure->physics ? &infrastructure->physics->getPhysicsInsertionCallback()
                                                     : nullptr;
}

physx::PxPhysics* physxPhysics(const std::shared_ptr<PhysXInfrastructure>& infrastructure)
{
    return infrastructure ? infrastructure->physics : nullptr;
}

namespace
{
// The cooked mesh's bounds, read once its vertices are final; a ray whose origin lies more
// than `farGap` outside them starts its float ray near them (makeLocalRay).
struct CookedBounds
{
    double minimum[3]{};
    double maximum[3]{};
    double extent = 0; // the largest extent
    double farGap = 0;

    void read(const physx::PxTriangleMesh& mesh)
    {
        const physx::PxBounds3 bounds = mesh.getLocalBounds();
        extent                        = 0;
        for (uint32_t axis = 0; axis < 3; ++axis)
        {
            minimum[axis] = double(bounds.minimum[axis]);
            maximum[axis] = double(bounds.maximum[axis]);
            extent        = std::max(extent, maximum[axis] - minimum[axis]);
        }
        farGap = 16.0 * (extent + 1.0e-3);
    }
};
} // namespace

struct PhysXTriangleMesh::Impl
{
    std::shared_ptr<PhysXInfrastructure> infrastructure;

    physx::PxTriangleMesh* mesh = nullptr;
    // The power of two the cooked vertices are the local ones times (cookedVertices).
    double       cookedFromLocal = 1.0;
    CookedBounds bounds;

    ~Impl()
    {
        if (mesh)
            mesh->release();
    }
};
PhysXTriangleMesh::PhysXTriangleMesh(std::unique_ptr<Impl> impl) : mImpl(std::move(impl)) {}
PhysXTriangleMesh::~PhysXTriangleMesh() = default;

namespace
{
// PhysX tests a ray against each triangle in float, with every barycentric bound widened
// by the mesh's geomEpsilon. Its float barycentrics err by about 2^-24 times the ray
// origin's distance from the triangle over the triangle's height and the cosine of
// incidence; at the 2^-22 a unit mesh gets, a ray could pass between two triangles that
// share an edge (3 % of rays 10 units from a unit square aimed at its diagonal) and
// report the surface behind or nothing. The cooked mesh is therefore widened by
// kRayCandidateMargin, which admits every crossing as a candidate, and a candidate
// within that margin of an edge is decided by exactCrossing.
constexpr float kRayCandidateMargin = 0x1p-10f;
// The barycentric margin of the exact decision, the CUDA plans' kTriangleEdgeMargin.
constexpr double kTriangleEdgeMargin = 0x1p-22;

bool describeTriangleMesh(const TriangleGeometryDescriptionView& source, std::vector<physx::PxVec3>& cooked,
                          double& cookedFromLocal, physx::PxTriangleMeshDesc& outDesc);
} // namespace

ovspatialquery_api_status_t PhysXTriangleMesh::create(const TriangleGeometryDescriptionView& source,
                                                      std::unique_ptr<PhysXTriangleMesh>&    outMesh)
{
    auto       impl   = std::make_unique<Impl>();
    const auto status = acquirePhysXInfrastructure(impl->infrastructure);
    if (status != OVSPATIALQUERY_API_SUCCESS)
        return status;
    physx::PxTriangleMeshDesc  desc;
    std::vector<physx::PxVec3> cooked;
    if (!describeTriangleMesh(source, cooked, impl->cookedFromLocal, desc))
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    auto* insertion = physxInsertionCallback(impl->infrastructure);
    if (insertion == nullptr)
        return OVSPATIALQUERY_API_INTERNAL;
    physx::PxCookingParams params{physx::PxTolerancesScale()};
    // Source triangles were validated above. Preserve their count and source remap.
    params.meshPreprocessParams |= physx::PxMeshPreprocessingFlag::eDISABLE_CLEAN_MESH;
    impl->mesh = PxCreateTriangleMesh(params, desc, *insertion);
    if (!impl->mesh || impl->mesh->getNbTriangles() != source.triangleCount)
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    impl->bounds.read(*impl->mesh);
    outMesh = std::unique_ptr<PhysXTriangleMesh>(new PhysXTriangleMesh(std::move(impl)));
    return OVSPATIALQUERY_API_SUCCESS;
}

namespace
{
// PhysX's float ray and overlap tests multiply coordinates together, so a mesh authored
// around 1e14 overflows and one around 1e-8 falls under PhysX's absolute determinant
// limit. The cooked copy is scaled by the power of two that brings its largest coordinate
// into [0.5, 1), which is exact, and every query is scaled the same way.
double cookedScale(const float* vertices, uint64_t vertexCount) noexcept
{
    float largest = 0.0f;
    for (uint64_t index = 0; index < vertexCount * 3u; ++index)
        largest = std::max(largest, std::abs(vertices[index]));
    int exponent = 0;
    std::frexp(double(largest), &exponent);
    return largest > 0.0f ? std::ldexp(1.0, -exponent) : 1.0;
}

void scaleVertices(const float* vertices, uint64_t vertexCount, double scale, physx::PxVec3* outVertices) noexcept
{
    for (uint64_t vertex = 0; vertex < vertexCount; ++vertex)
    {
        const float* local  = vertices + vertex * 3u;
        outVertices[vertex] = physx::PxVec3(float(local[0] * scale), float(local[1] * scale), float(local[2] * scale));
    }
}

bool describeTriangleMesh(const TriangleGeometryDescriptionView& source, std::vector<physx::PxVec3>& cooked,
                          double& cookedFromLocal, physx::PxTriangleMeshDesc& outDesc)
{
    if (!source.localVertices || !source.triangleVertexIndices || !source.vertexCount || !source.triangleCount ||
        source.vertexCount > UINT32_MAX || source.triangleCount > UINT32_MAX)
        return false;
    cookedFromLocal = cookedScale(source.localVertices, source.vertexCount);
    cooked.resize(static_cast<size_t>(source.vertexCount));
    scaleVertices(source.localVertices, source.vertexCount, cookedFromLocal, cooked.data());
    outDesc.points.count     = static_cast<physx::PxU32>(source.vertexCount);
    outDesc.points.stride    = sizeof(physx::PxVec3);
    outDesc.points.data      = cooked.data();
    outDesc.triangles.count  = static_cast<physx::PxU32>(source.triangleCount);
    outDesc.triangles.stride = 3 * sizeof(uint32_t);
    outDesc.triangles.data   = source.triangleVertexIndices;
    // PhysX widens every triangle's barycentric bounds by geomEpsilon in its ray tests. Left
    // at 0 it is derived from the largest coordinate (about 2^-22 of it), a length used as a
    // fraction, so a mesh authored at 1e6 accepted rays crossing its plane half a triangle
    // outside. A fixed margin keeps the test independent of the size and offset a definition
    // is authored at, as a scaled placement is; the hits it admits beside a triangle are
    // decided exactly (kRayCandidateMargin).
    outDesc.geomEpsilon = kRayCandidateMargin;
    return true;
}
} // namespace

ovspatialquery_api_status_t PhysXTriangleMesh::cookTemplate(const TriangleGeometryDescriptionView& source,
                                                            std::vector<uint8_t>&                  outCooked)
{
    physx::PxTriangleMeshDesc  desc;
    std::vector<physx::PxVec3> cooked;
    double                     cookedFromLocal = 1.0;
    try
    {
        if (!describeTriangleMesh(source, cooked, cookedFromLocal, desc))
            return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    catch (const std::bad_alloc&)
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    physx::PxCookingParams params{physx::PxTolerancesScale()};
    // Source triangles were validated by the preparation; keeping the authored order and
    // the one-to-one vertex mapping is what getVerticesForModification requires, and a
    // quantized tree cannot be refitted.
    params.meshPreprocessParams |= physx::PxMeshPreprocessingFlag::eDISABLE_CLEAN_MESH;
    params.midphaseDesc.mBVH34Desc.quantized = false;
    physx::PxDefaultMemoryOutputStream stream;
    if (!PxCookTriangleMesh(params, desc, stream))
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    try
    {
        outCooked.assign(stream.getData(), stream.getData() + stream.getSize());
    }
    catch (const std::bad_alloc&)
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t PhysXTriangleMesh::createFromTemplate(const std::vector<uint8_t>&            cooked,
                                                                  const TriangleGeometryDescriptionView& source,
                                                                  std::unique_ptr<PhysXTriangleMesh>&    outMesh)
{
    if (cooked.empty() || cooked.size() > UINT32_MAX || !source.localVertices || !source.vertexCount)
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    auto       impl   = std::make_unique<Impl>();
    const auto status = acquirePhysXInfrastructure(impl->infrastructure);
    if (status != OVSPATIALQUERY_API_SUCCESS)
        return status;
    physx::PxDefaultMemoryInputData input(const_cast<physx::PxU8*>(cooked.data()),
                                          static_cast<physx::PxU32>(cooked.size()));
    impl->mesh = impl->infrastructure->physics->createTriangleMesh(input);
    if (!impl->mesh || impl->mesh->getNbTriangles() != source.triangleCount ||
        impl->mesh->getNbVertices() != source.vertexCount)
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    physx::PxVec3* vertices = impl->mesh->getVerticesForModification();
    if (vertices == nullptr)
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    impl->cookedFromLocal = cookedScale(source.localVertices, source.vertexCount);
    scaleVertices(source.localVertices, source.vertexCount, impl->cookedFromLocal, vertices);
    impl->mesh->refitBVH();
    impl->bounds.read(*impl->mesh);
    outMesh = std::unique_ptr<PhysXTriangleMesh>(new PhysXTriangleMesh(std::move(impl)));
    return OVSPATIALQUERY_API_SUCCESS;
}

namespace
{
// The local float ray PhysX evaluates for a double query; raycast and
// raycastAll share it so both see the same input domain and rounding.
struct LocalRay
{
    physx::PxVec3 origin;
    physx::PxVec3 direction;
    double        length          = 0;
    float         range           = 0;
    float         traversalRange  = 0;
    double        cookedFromLocal = 1.0;
    // A PhysX distance plus `advance`, times this, is the caller's ray parameter.
    double distanceScale = 1.0;
    // 1 / cookedFromLocal, exact: both are powers of two.
    double localFromCooked = 1.0;
    // How far, in cooked units along the unit direction, the float ray starts past the
    // caller's origin, `start` in double; PhysX's distances are measured from `origin`, its
    // rounding (rayParameter).
    double advance = 0;
    double start[3]{};
    // How far before the caller's range ends a PhysX distance is decided exactly, in the
    // caller's ray parameter: its float rounding can move a crossing across the end.
    double endAllowance = 0;
};

// A rare path of a function every ray runs stays out of line, so it does not crowd the code
// that does run; the steps every ray runs are inlined into it. Left to the compiler, the
// closest ray's steps stayed calls and cost CPU rays up to 6 % (KP_RT, MeshSurface).
#if defined(_MSC_VER)
#define OVS_RAY_COLD   __declspec(noinline)
#define OVS_RAY_INLINE __forceinline
#else
#define OVS_RAY_COLD   __attribute__((noinline, cold))
#define OVS_RAY_INLINE inline __attribute__((always_inline))
#endif

// The interval of the ray from the cooked-frame `origin`, in cooked units along the unit
// direction, that lies within `bounds` grown by `pad`; `entry` > `exit` when it misses them.
void boundsInterval(const double origin[3], const double direction[3], double length, const CookedBounds& bounds,
                    double pad, double& entry, double& exit)
{
    entry = -std::numeric_limits<double>::infinity();
    exit  = std::numeric_limits<double>::infinity();
    for (uint32_t axis = 0; axis < 3; ++axis)
    {
        const double minimum = bounds.minimum[axis] - pad;
        const double maximum = bounds.maximum[axis] + pad;
        if (direction[axis] == 0)
        {
            if (origin[axis] < minimum || origin[axis] > maximum)
            {
                entry = std::numeric_limits<double>::infinity();
                exit  = -std::numeric_limits<double>::infinity();
                return;
            }
            continue;
        }
        double near = (minimum - origin[axis]) * length / direction[axis];
        double far  = (maximum - origin[axis]) * length / direction[axis];
        if (near > far)
            std::swap(near, far);
        entry = std::max(entry, near);
        exit  = std::min(exit, far);
    }
}

// Moves the cooked-frame `origin` to where the ray enters `bounds` grown by a quarter of their
// largest extent, within [0, range], and returns how far it moved; 0, leaving it, when the ray
// does not enter them there. The new start is computed from coordinates much larger than the
// mesh, whose rounding can exceed the growth, so it is checked from where it lands, in
// coordinates the size of the mesh; a start inside or past the bounds moves back toward the
// origin by margins of that rounding, and keeps the ray's own start if none clears them.
OVS_RAY_COLD double advanceToBounds(double origin[3], const double direction[3], double length, double range,
                                    const CookedBounds& bounds)
{
    double entry = 0, exit = 0;
    boundsInterval(origin, direction, length, bounds, bounds.extent * 0.25 + 1.0e-3, entry, exit);
    entry = std::max(entry, 0.0);
    exit  = std::min(exit, range);
    if (!(entry > 0 && entry <= exit) || !std::isfinite(entry))
        return 0;
    double magnitude = entry;
    for (uint32_t axis = 0; axis < 3; ++axis)
        magnitude = std::max(magnitude, std::abs(origin[axis]));
    double margin = 0;
    for (uint32_t attempt = 0; attempt < 8 && margin < entry; ++attempt)
    {
        double start[3]{};
        for (uint32_t axis = 0; axis < 3; ++axis)
            start[axis] = origin[axis] + (entry - margin) / length * direction[axis];
        double inside = 0, leaves = 0;
        boundsInterval(start, direction, length, bounds, 0.0, inside, leaves);
        if (!(inside <= leaves && inside < 0))
        {
            std::copy_n(start, 3, origin);
            return entry - margin;
        }
        margin = margin == 0 ? 16.0 * std::numeric_limits<double>::epsilon() * magnitude : 4.0 * margin;
    }
    return 0;
}

// NOT_SUPPORTED for a ray float cannot hold. With `clampRange` a ray longer
// than the largest float range stops there: the stored vertices are float, so
// no crossing lies beyond it.
OVS_RAY_INLINE ovspatialquery_api_status_t makeLocalRay(const double origin[3], const double direction[3],
                                                        double maximumDistance, double cookedFromLocal,
                                                        const CookedBounds& cookedBounds, bool clampRange,
                                                        LocalRay& ray)
{
    for (uint32_t axis = 0; axis < 3; ++axis)
    {
        if (!std::isfinite(origin[axis]) || !std::isfinite(direction[axis]))
            return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    ray.cookedFromLocal = cookedFromLocal;
    ray.length   = std::sqrt(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]);
    double range = maximumDistance * cookedFromLocal * ray.length;
    if (clampRange)
        range = std::min(range, double(std::numeric_limits<float>::max()));
    if (!std::isfinite(range) || range < 0 || range > std::numeric_limits<float>::max() || !(ray.length > 0))
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    // PhysX's float barycentrics err in proportion to the ray origin's distance from a
    // triangle, so a ray from more than 16 mesh extents away starts where it nears the
    // mesh's bounds, in double.
    double cookedOrigin[3]{}, outside = 0;
    for (uint32_t axis = 0; axis < 3; ++axis)
    {
        cookedOrigin[axis]  = origin[axis] * cookedFromLocal;
        ray.direction[axis] = static_cast<float>(direction[axis]);
        outside             = std::max(outside, std::max(cookedBounds.minimum[axis] - cookedOrigin[axis],
                                                         cookedOrigin[axis] - cookedBounds.maximum[axis]));
    }
    if (outside > cookedBounds.farGap)
        ray.advance = advanceToBounds(cookedOrigin, direction, ray.length, range, cookedBounds);
    for (uint32_t axis = 0; axis < 3; ++axis)
    {
        ray.start[axis]  = cookedOrigin[axis];
        ray.origin[axis] = static_cast<float>(cookedOrigin[axis]);
    }
    ray.localFromCooked = 1.0 / cookedFromLocal;
    ray.distanceScale   = ray.localFromCooked / ray.length;
    if (!ray.origin.isFinite() || !ray.direction.isFinite() || ray.direction.isZero())
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    ray.direction.normalize();
    range     = std::max(range - ray.advance, 0.0);
    ray.range = static_cast<float>(range);
    // PhysX's float distance errs by ulps of the coordinates and, at grazing incidence, by
    // about its barycentrics' error times the triangle's size: less than kRayCandidateMargin of
    // the mesh's extent wherever that margin admits every crossing. A crossing that near the
    // range's end is decided on its exact distance, and PhysX is asked for hits that far
    // beyond it. The extension stays within the float range PhysX accepts as a distance.
    const float  magnitude = std::max(std::abs(ray.origin.x), std::max(std::abs(ray.origin.y), std::abs(ray.origin.z)));
    const double extension = 16.0 * double(std::numeric_limits<float>::epsilon()) * (double(magnitude) + range + 1.0) +
                             double(kRayCandidateMargin) * cookedBounds.extent;
    ray.endAllowance   = extension * ray.distanceScale;
    ray.traversalRange = static_cast<float>(std::min(range + extension, double(std::numeric_limits<float>::max())));
    return OVSPATIALQUERY_API_SUCCESS;
}

// The caller's ray parameter of a PhysX distance. PhysX measures from the float origin; its
// rounding along the ray is known exactly, and a face it moves behind the caller's origin
// comes out at a negative distance.
OVS_RAY_INLINE double rayParameter(const LocalRay& ray, const double direction[3], float physxDistance)
{
    double shift = 0;
    for (uint32_t axis = 0; axis < 3; ++axis)
        shift += (double(ray.origin[axis]) - ray.start[axis]) * direction[axis];
    return (ray.advance + shift / ray.length + double(physxDistance)) * ray.distanceScale;
}

enum class LocalHitOutcome
{
    hit,       // inside the interval; `hit` is filled
    beyond,    // past the caller's range
    outside,   // float origin rounding moved it out of the interval
    malformed, // PhysX left out a requested field
    missed,    // admitted only by the candidate margin; the ray misses the triangle
};

void cookedTriangle(const physx::PxTriangleMesh& mesh, uint32_t face, double corners[3][3])
{
    const physx::PxVec3* vertices = mesh.getVertices();
    for (uint32_t k = 0; k < 3; ++k)
    {
        const uint32_t index = mesh.getTriangleMeshFlags() & physx::PxTriangleMeshFlag::e16_BIT_INDICES
                                   ? static_cast<const physx::PxU16*>(mesh.getTriangles())[3u * face + k]
                                   : static_cast<const physx::PxU32*>(mesh.getTriangles())[3u * face + k];
        for (uint32_t axis = 0; axis < 3; ++axis)
            corners[k][axis] = double(vertices[index][axis]);
    }
}

// Whether the local ray crosses the cooked mesh's triangle `face`, decided in double as the
// CUDA plans' rayTriangle does: in the triangle's own frame, centred and scaled by a power
// of two, with kTriangleEdgeMargin of the barycentric range. Two triangles sharing an edge
// round it alike to well within that margin, so a ray on it crosses at least one.
// The ray starts at `start`, in the cooked frame: the caller's origin, or for a ray from far
// away where makeLocalRay moved it near the mesh, since the far origin's coordinates would
// swamp the triangle's in this frame. The distance is measured from `start`.
OVS_RAY_COLD bool exactCrossing(const physx::PxTriangleMesh& mesh, uint32_t face, const double start[3],
                                const double direction[3], double cookedFromLocal, bool bothFaces, double& outDistance,
                                float outUv[2], double outPoint[3])
{
    double corners[3][3]{};
    cookedTriangle(mesh, face, corners);
    double center[3]{}, scale = 0;
    for (uint32_t axis = 0; axis < 3; ++axis)
    {
        const double low  = std::min(corners[0][axis], std::min(corners[1][axis], corners[2][axis]));
        const double high = std::max(corners[0][axis], std::max(corners[1][axis], corners[2][axis]));
        center[axis]      = low * 0.5 + high * 0.5;
        scale             = std::max(scale, std::max(high - center[axis], center[axis] - low));
    }
    if (!(scale > 0) || !std::isfinite(scale))
        return false;
    int exponent = 0;
    std::frexp(scale, &exponent);
    const double inverse = std::ldexp(1.0, -exponent);
    double       a[3]{}, e1[3]{}, e2[3]{}, delta[3]{}, d[3]{};
    for (uint32_t axis = 0; axis < 3; ++axis)
    {
        a[axis]     = (corners[0][axis] - center[axis]) * inverse;
        e1[axis]    = (corners[1][axis] - center[axis]) * inverse - a[axis];
        e2[axis]    = (corners[2][axis] - center[axis]) * inverse - a[axis];
        delta[axis] = (start[axis] - center[axis]) * inverse - a[axis];
        d[axis]     = direction[axis] * inverse;
    }
    const double p[3]        = {d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2], d[0] * e2[1] - d[1] * e2[0]};
    const double determinant = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
    const double tolerance   = (std::abs(e1[0] * p[0]) + std::abs(e1[1] * p[1]) + std::abs(e1[2] * p[2])) * 8.0 *
                             std::numeric_limits<double>::epsilon();
    if (!std::isfinite(determinant) || (bothFaces ? std::abs(determinant) <= tolerance : determinant <= tolerance))
        return false;
    const double u = (delta[0] * p[0] + delta[1] * p[1] + delta[2] * p[2]) / determinant;
    if (u < -kTriangleEdgeMargin || u > 1.0 + kTriangleEdgeMargin)
        return false;
    const double q[3] = {delta[1] * e1[2] - delta[2] * e1[1], delta[2] * e1[0] - delta[0] * e1[2],
                         delta[0] * e1[1] - delta[1] * e1[0]};
    const double v    = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) / determinant;
    if (v < -kTriangleEdgeMargin || u + v > 1.0 + kTriangleEdgeMargin)
        return false;
    // The parameter along `direction` in the cooked frame, cookedFromLocal times the local one.
    const double distance = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) / determinant / cookedFromLocal;
    if (!std::isfinite(distance))
        return false;
    outDistance = distance;
    outUv[0]    = float(u);
    outUv[1]    = float(v);
    // The crossing on the triangle, from its barycentrics: the point at its distance along the
    // ray loses the surface when the origin lies far beyond it.
    for (uint32_t axis = 0; axis < 3; ++axis)
        outPoint[axis] =
            (corners[0][axis] + u * (corners[1][axis] - corners[0][axis]) + v * (corners[2][axis] - corners[0][axis])) /
            cookedFromLocal;
    return true;
}

// With `decideAll` every crossing is decided exactly; otherwise only one within the candidate
// margin of an edge or within the allowance of the range's end, which sets `decided`.
OVS_RAY_INLINE LocalHitOutcome makeLocalHit(const physx::PxGeomRaycastHit& result, const double direction[3],
                                            double maximumDistance, const LocalRay& ray,
                                            const physx::PxTriangleMesh& mesh, bool bothFaces, bool decideAll,
                                            PreciseTriangleHit& hit, bool& decided)
{
    decided = false;
    if (!(result.flags & physx::PxHitFlag::ePOSITION) || !(result.flags & physx::PxHitFlag::eNORMAL) ||
        !(result.flags & physx::PxHitFlag::eUV) || !(result.flags & physx::PxHitFlag::eFACE_INDEX) ||
        result.faceIndex >= mesh.getNbTriangles())
        return LocalHitOutcome::malformed;
    float  uv[2]    = {result.u, result.v};
    double distance = rayParameter(ray, direction, result.distance);
    double point[3]{};
    // Within the candidate margin of an edge the ray may miss the triangle PhysX admitted.
    if (decideAll || result.u < kRayCandidateMargin || result.v < kRayCandidateMargin ||
        1.0f - result.u - result.v < kRayCandidateMargin || distance > maximumDistance - ray.endAllowance)
    {
        if (!exactCrossing(mesh, result.faceIndex, ray.start, direction, ray.cookedFromLocal, bothFaces, distance, uv,
                           point))
        {
            hit          = PreciseTriangleHit{};
            hit.distance = distance;
            return LocalHitOutcome::missed;
        }
        distance += ray.advance * ray.distanceScale;
        // PhysX ranked the candidates by its float distance, which the exact one replaces;
        // another crossing within that float distance's error may lie nearer.
        decided = true;
    }
    // The interval is closed and decided on the double distance, with no allowance at either
    // end: any would admit a face beyond a short enough ray or behind a small enough mesh.
    if (distance > maximumDistance)
        return LocalHitOutcome::beyond;
    // Float origin rounding can admit a face just behind the origin; the contact path decides
    // it exactly.
    if (!(distance >= 0))
        return LocalHitOutcome::outside;
    const auto* remap     = mesh.getTrianglesRemap();
    hit                   = PreciseTriangleHit{};
    hit.generatedTriangle = remap ? remap[result.faceIndex] : result.faceIndex;
    hit.distance          = distance;
    // PhysX's contact, unless the crossing is decided exactly; then its point on the triangle.
    for (uint32_t axis = 0; axis < 3; ++axis)
    {
        hit.position[axis] = decided ? point[axis] : double(result.position[axis]) * ray.localFromCooked;
        hit.normal[axis]   = result.normal[axis];
    }
    hit.uv[0] = uv[0];
    hit.uv[1] = uv[1];
    return LocalHitOutcome::hit;
}

// eMESH_BOTH_SIDES counts back faces of a single-sided mesh and keeps its front normal (ADR-0065).
physx::PxHitFlags raycastFields(bool bothSides)
{
    physx::PxHitFlags fields =
        physx::PxHitFlag::ePOSITION | physx::PxHitFlag::eNORMAL | physx::PxHitFlag::eFACE_INDEX | physx::PxHitFlag::eUV;
    if (bothSides)
        fields |= physx::PxHitFlag::eMESH_BOTH_SIDES;
    return fields;
}
} // namespace

namespace
{
// The nearest crossing a raycastAll reports, and whether one needs the contact path.
struct NearestCrossing
{
    PreciseTriangleHit hit;
    bool               found   = false;
    bool               recheck = false;

    static bool take(void* context, const PreciseTriangleHit& candidate, bool recheck)
    {
        auto& state = *static_cast<NearestCrossing*>(context);
        if (recheck)
            state.recheck = true;
        else if (!state.found || candidate.distance < state.hit.distance)
        {
            state.hit   = candidate;
            state.found = true;
        }
        return true;
    }
};

// How many times a closest ray restarts past a candidate the exact decision rejected
// before it lists every crossing instead.
constexpr uint32_t kRayRestarts = 16u;

// What a closest query on the cooked mesh needs.
struct ClosestQuery
{
    const physx::PxTriangleMesh&         mesh;
    const physx::PxTriangleMeshGeometry& geometry;
    double                               cookedFromLocal;
    const CookedBounds&                  bounds;
    bool                                 bothSides;
    bool                                 bothFaces;
};
// PhysX's closest candidate of the local ray within maximumDistance, decided by
// makeLocalHit into `outcome`; `any` is false when PhysX reports none.
OVS_RAY_INLINE ovspatialquery_api_status_t closestCandidate(const ClosestQuery& query, const double origin[3],
                                                            const double direction[3], double maximumDistance,
                                                            LocalRay& ray, LocalHitOutcome& outcome,
                                                            PreciseTriangleHit& candidate, bool& any, bool& decided)
{
    any = false;
    const auto status =
        makeLocalRay(origin, direction, maximumDistance, query.cookedFromLocal, query.bounds, false, ray);
    if (status != OVSPATIALQUERY_API_SUCCESS)
        return status;
    // Ask for the closest local hit even for an ANY scene query.
    physx::PxGeomRaycastHit result{};
    if (!physx::PxGeometryQuery::raycast(ray.origin, ray.direction, query.geometry,
                                         physx::PxTransform(physx::PxIdentity), ray.traversalRange,
                                         raycastFields(query.bothSides), 1, &result))
        return OVSPATIALQUERY_API_SUCCESS;
    any = true;
    outcome =
        makeLocalHit(result, direction, maximumDistance, ray, query.mesh, query.bothFaces, false, candidate, decided);
    return OVSPATIALQUERY_API_SUCCESS;
}

// The local hit `nearest` found from `start` along the caller's ray, as seen from its origin.
void hitFrom(const PreciseTriangleHit& nearest, double start, PreciseTriangleHit& hit)
{
    hit = nearest;
    hit.distance += start;
}

// The nearest crossing within `window` of `origin`, which lies `start` along the caller's ray,
// from every crossing listed there and decided exactly: the closest candidate's exact decision
// moved it from PhysX's float distance, or past the range's end, so another crossing, even one
// whose float distance lies beyond it, may be nearer.
OVS_RAY_COLD ovspatialquery_api_status_t nearestListed(const PhysXTriangleMesh& self, const ClosestQuery& query,
                                                       const double origin[3], const double direction[3], double window,
                                                       double start, bool doubleSided, InstanceHostLedger* ledger,
                                                       bool& hasHit, PreciseTriangleHit& hit)
{
    NearestCrossing nearest;
    const auto      listed = self.raycastAll(origin, direction, window, doubleSided, query.bothSides, ledger,
                                             &NearestCrossing::take, &nearest);
    if (listed != OVSPATIALQUERY_API_SUCCESS)
        return listed;
    if (nearest.recheck)
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    if (nearest.found)
    {
        hitFrom(nearest.hit, start, hit);
        hasHit = true;
    }
    return OVSPATIALQUERY_API_SUCCESS;
}

// The nearest candidate, at `rejected`, lies beside its triangle. Every crossing is a
// candidate, so none lies before it; the crossings just past it, such as the triangle
// across its edge, are listed, and beyond them the closest query starts again.
OVS_RAY_COLD ovspatialquery_api_status_t raycastPastRejected(const PhysXTriangleMesh& self, const ClosestQuery& query,
                                                             const double origin[3], const double direction[3],
                                                             double maximumDistance, bool doubleSided, double rejected,
                                                             InstanceHostLedger* ledger, bool& hasHit,
                                                             PreciseTriangleHit& hit)
{
    // The mesh's largest extent in units of the ray's parameter.
    const double length =
        std::sqrt(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]);
    const double extent = query.bounds.extent / (query.cookedFromLocal * length);
    double       start  = 0;
    for (uint32_t attempt = 0; attempt < kRayRestarts; ++attempt)
    {
        double shifted[3]{};
        for (uint32_t axis = 0; axis < 3; ++axis)
            shifted[axis] = origin[axis] + start * direction[axis];
        const double reach = maximumDistance - start;
        if (attempt > 0)
        {
            LocalRay           ray;
            LocalHitOutcome    outcome{};
            PreciseTriangleHit candidate;
            bool               any = false, decided = false;
            const auto         status =
                closestCandidate(query, shifted, direction, reach, ray, outcome, candidate, any, decided);
            if (status != OVSPATIALQUERY_API_SUCCESS || !any)
                return status;
            switch (outcome)
            {
            case LocalHitOutcome::hit:
                if (decided)
                    return nearestListed(self, query, shifted, direction, candidate.distance, start, doubleSided,
                                         ledger, hasHit, hit);
                hitFrom(candidate, start, hit);
                hasHit = true;
                return OVSPATIALQUERY_API_SUCCESS;
            case LocalHitOutcome::beyond:
                return nearestListed(self, query, shifted, direction, reach, start, doubleSided, ledger, hasHit, hit);
            case LocalHitOutcome::outside: return OVSPATIALQUERY_API_NOT_SUPPORTED;
            case LocalHitOutcome::missed: break;
            default: return OVSPATIALQUERY_API_INTERNAL;
            }
            rejected = candidate.distance;
        }
        const double    windowEnd = std::min(reach, rejected + double(kRayCandidateMargin) * (extent + rejected));
        NearestCrossing nearest;
        const auto      listed = self.raycastAll(shifted, direction, windowEnd, doubleSided, query.bothSides, ledger,
                                                 &NearestCrossing::take, &nearest);
        if (listed != OVSPATIALQUERY_API_SUCCESS)
            return listed;
        if (nearest.recheck)
            return OVSPATIALQUERY_API_NOT_SUPPORTED;
        if (nearest.found)
        {
            hitFrom(nearest.hit, start, hit);
            hasHit = true;
            return OVSPATIALQUERY_API_SUCCESS;
        }
        if (!(windowEnd < reach))
            return OVSPATIALQUERY_API_SUCCESS;
        start += windowEnd;
    }
    NearestCrossing nearest;
    const auto      listed = self.raycastAll(origin, direction, maximumDistance, doubleSided, query.bothSides, ledger,
                                             &NearestCrossing::take, &nearest);
    if (listed != OVSPATIALQUERY_API_SUCCESS)
        return listed;
    if (nearest.recheck)
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    if (nearest.found)
    {
        hit    = nearest.hit;
        hasHit = true;
    }
    return OVSPATIALQUERY_API_SUCCESS;
}
#undef OVS_RAY_COLD
#undef OVS_RAY_INLINE
} // namespace

ovspatialquery_api_status_t PhysXTriangleMesh::raycast(const double origin[3], const double direction[3],
                                                       double maximumDistance, bool doubleSided, bool bothSides,
                                                       bool& hasHit, PreciseTriangleHit& hit,
                                                       InstanceHostLedger* ledger) const
{
    hasHit = false;
    physx::PxMeshGeometryFlags meshFlags;
    if (doubleSided)
        meshFlags |= physx::PxMeshGeometryFlag::eDOUBLE_SIDED;
    const physx::PxTriangleMeshGeometry geometry(mImpl->mesh, physx::PxMeshScale(), meshFlags);
    const ClosestQuery                  query{*mImpl->mesh,  geometry,  mImpl->cookedFromLocal,
                             mImpl->bounds, bothSides, doubleSided || bothSides};
    LocalRay                            ray;
    LocalHitOutcome                     outcome{};
    bool                                any = false, decided = false;
    const auto status = closestCandidate(query, origin, direction, maximumDistance, ray, outcome, hit, any, decided);
    if (status != OVSPATIALQUERY_API_SUCCESS || !any)
        return status;
    switch (outcome)
    {
    case LocalHitOutcome::hit:
        if (decided)
            return nearestListed(*this, query, origin, direction, hit.distance, 0, doubleSided, ledger, hasHit, hit);
        hasHit = true;
        return OVSPATIALQUERY_API_SUCCESS;
    // Only a crossing decided exactly lies beyond the range, which then may hide one within it.
    case LocalHitOutcome::beyond:
        return nearestListed(*this, query, origin, direction, maximumDistance, 0, doubleSided, ledger, hasHit, hit);
    // Float origin rounding can move a clipped face back into the traversal.
    // Use the existing contact path so rejecting it does not hide a later face.
    case LocalHitOutcome::outside: return OVSPATIALQUERY_API_NOT_SUPPORTED;
    case LocalHitOutcome::missed:
        return raycastPastRejected(*this, query, origin, direction, maximumDistance, doubleSided, hit.distance, ledger,
                                   hasHit, hit);
    default: return OVSPATIALQUERY_API_INTERNAL;
    }
}

ovspatialquery_api_status_t PhysXTriangleMesh::raycastAll(const double origin[3], const double direction[3],
                                                          double maximumDistance, bool doubleSided, bool bothSides,
                                                          InstanceHostLedger* ledger, TriangleHitSink emit,
                                                          void* context) const
{
    LocalRay   ray;
    const auto status =
        makeLocalRay(origin, direction, maximumDistance, mImpl->cookedFromLocal, mImpl->bounds, true, ray);
    if (status != OVSPATIALQUERY_API_SUCCESS)
        return status;
    physx::PxMeshGeometryFlags meshFlags;
    if (doubleSided)
        meshFlags |= physx::PxMeshGeometryFlag::eDOUBLE_SIDED;
    const physx::PxTriangleMeshGeometry geometry(mImpl->mesh, physx::PxMeshScale(), meshFlags);
    const physx::PxHitFlags             fields = raycastFields(bothSides) | physx::PxHitFlag::eMESH_MULTIPLE;
    // PhysX stops at a full buffer without saying whether more crossed, and a
    // query cannot resume, so a full buffer is retried four times larger, up to
    // one entry per triangle (a ray crosses each at most once). Each larger
    // buffer is charged to `ledger`.
    constexpr uint32_t                         kInlineHits = 64u;
    physx::PxGeomRaycastHit                    inlineHits[kInlineHits];
    std::unique_ptr<physx::PxGeomRaycastHit[]> large;
    HostLease                                  charge;
    physx::PxGeomRaycastHit*                   hits          = inlineHits;
    uint32_t                                   capacity      = kInlineHits;
    const uint32_t                             triangleCount = mImpl->mesh->getNbTriangles();
    uint32_t                                   count =
        physx::PxGeometryQuery::raycast(ray.origin, ray.direction, geometry, physx::PxTransform(physx::PxIdentity),
                                        ray.traversalRange, fields, capacity, hits);
    while (count == capacity && capacity < triangleCount)
    {
        capacity = uint32_t(std::min<uint64_t>(uint64_t(capacity) * 4u, triangleCount));
        charge.reset();
        large.reset();
        if (ledger)
        {
            HostTransaction transaction(*ledger);
            if (!transaction.reserve(uint64_t(capacity) * sizeof(physx::PxGeomRaycastHit)) ||
                !transaction.commit(charge))
                return OVSPATIALQUERY_API_HOST_BUDGET;
        }
        large.reset(new physx::PxGeomRaycastHit[capacity]);
        hits = large.get();
        count =
            physx::PxGeometryQuery::raycast(ray.origin, ray.direction, geometry, physx::PxTransform(physx::PxIdentity),
                                            ray.traversalRange, fields, capacity, hits);
    }
    const auto* remap = mImpl->mesh->getTrianglesRemap();
    for (uint32_t index = 0; index < count; ++index)
    {
        PreciseTriangleHit hit;
        bool               decided = false; // every crossing is listed, so their order does not matter
        // Every crossing is decided on its exact distance, which orders them and clips them to the
        // range as the CUDA plans do; PhysX's float distances can tie or swap nearby crossings.
        switch (makeLocalHit(hits[index], direction, maximumDistance, ray, *mImpl->mesh, doubleSided || bothSides, true,
                             hit, decided))
        {
        case LocalHitOutcome::hit:
            if (!emit(context, hit, false))
                return OVSPATIALQUERY_API_OUT_OF_MEMORY;
            break;
        // The rounded contact position says nothing about the exact crossing.
        case LocalHitOutcome::outside:
            hit                   = PreciseTriangleHit{};
            hit.generatedTriangle = remap ? remap[hits[index].faceIndex] : hits[index].faceIndex;
            if (!emit(context, hit, true))
                return OVSPATIALQUERY_API_OUT_OF_MEMORY;
            break;
        case LocalHitOutcome::beyond:
        case LocalHitOutcome::missed: break;
        default: return OVSPATIALQUERY_API_INTERNAL;
        }
    }
    return OVSPATIALQUERY_API_SUCCESS;
}

uint32_t PhysXTriangleMesh::overlapTriangles(const double localMinimum[3], const double localMaximum[3],
                                             uint32_t* results, uint32_t capacity, bool& overflow) const
{
    physx::PxVec3 center, half;
    for (uint32_t a = 0; a < 3; ++a)
    {
        const double minimum = localMinimum[a] * mImpl->cookedFromLocal;
        const double maximum = localMaximum[a] * mImpl->cookedFromLocal;
        center[a]            = float(minimum * 0.5 + maximum * 0.5);
        const double extent  = std::max(double(center[a]) - minimum, maximum - double(center[a]));
        half[a]              = std::nextafter(float(extent), std::numeric_limits<float>::infinity());
        half[a]              = std::max(half[a], std::numeric_limits<float>::epsilon());
    }
    // Inside the input domain the local query box is finite in float (ADR-0032).
    const uint32_t count = physx::PxMeshQuery::findOverlapTriangleMesh(
        physx::PxBoxGeometry(half), physx::PxTransform(center), physx::PxTriangleMeshGeometry(mImpl->mesh),
        physx::PxTransform(physx::PxIdentity), results, capacity, 0, overflow);
    const auto* remap = mImpl->mesh->getTrianglesRemap();
    if (remap)
        for (uint32_t a = 0; a < count; ++a)
            results[a] = remap[results[a]];
    return count;
}

uint32_t PhysXTriangleMesh::overlapTrianglesCapsule(const double localStart[3], const double localEnd[3],
                                                    double localRadius, uint32_t* results, uint32_t capacity,
                                                    bool& overflow) const
{
    const double scale  = mImpl->cookedFromLocal;
    const double radius = localRadius * scale;
    double       start[3]{}, end[3]{}, center[3]{}, axis[3]{};
    for (uint32_t a = 0; a < 3; ++a)
    {
        start[a] = localStart[a] * scale;
        end[a]   = localEnd[a] * scale;
    }
    double lengthSquared = 0, magnitude = 0;
    for (uint32_t a = 0; a < 3; ++a)
    {
        center[a]     = start[a] * 0.5 + end[a] * 0.5;
        axis[a]       = end[a] - start[a];
        lengthSquared = std::fma(axis[a], axis[a], lengthSquared);
        magnitude     = std::max(magnitude, std::max(std::abs(start[a]), std::abs(end[a])));
    }
    const double length = std::sqrt(lengthSquared);
    // The float capsule test can misplace a boundary triangle by ULPs of the
    // coordinates involved; inflating the radius by more than that keeps the
    // gather conservative, as the outward-rounded box does.
    const double inflated =
        radius + (magnitude + length + radius) * 64.0 * static_cast<double>(std::numeric_limits<float>::epsilon());
    const physx::PxVec3 position{float(center[0]), float(center[1]), float(center[2])};
    const float         halfHeight  = std::nextafter(float(length * 0.5), std::numeric_limits<float>::infinity());
    const float         floatRadius = std::nextafter(float(inflated), std::numeric_limits<float>::infinity());
    physx::PxQuat       rotation(physx::PxIdentity);
    if (length > 0)
    {
        const physx::PxVec3 direction(float(axis[0] / length), float(axis[1] / length), float(axis[2] / length));
        rotation = physx::PxShortestRotation(physx::PxVec3(1.0f, 0.0f, 0.0f), direction.getNormalized());
    }
    // Inside the input domain the float capsule and its pose are valid (ADR-0032).
    const physx::PxCapsuleGeometry capsule(floatRadius, halfHeight);
    const physx::PxTransform       pose(position, rotation);
    const uint32_t                 count = physx::PxMeshQuery::findOverlapTriangleMesh(
        capsule, pose, physx::PxTriangleMeshGeometry(mImpl->mesh), physx::PxTransform(physx::PxIdentity), results,
        capacity, 0, overflow);
    const auto* remap = mImpl->mesh->getTrianglesRemap();
    if (remap)
        for (uint32_t a = 0; a < count; ++a)
            results[a] = remap[results[a]];
    return count;
}

uint32_t PhysXTriangleMesh::overlapTrianglesBox(const double localCenter[3], const double axes[3][3],
                                                const double localHalf[3], uint32_t* results, uint32_t capacity,
                                                bool& overflow) const
{
    double center[3]{}, half[3]{};
    for (uint32_t a = 0; a < 3; ++a)
    {
        center[a] = localCenter[a] * mImpl->cookedFromLocal;
        half[a]   = localHalf[a] * mImpl->cookedFromLocal;
    }
    // The float center, axes and triangle test misplace a boundary triangle by
    // ULPs of the coordinates involved: the box's, and the vertices' relative to
    // it. Unlike a circumscribed capsule, the box has no spare room at a face.
    const physx::PxBounds3 bounds    = mImpl->mesh->getLocalBounds();
    double                 magnitude = 0, reach = 0, extent = 0;
    for (uint32_t a = 0; a < 3; ++a)
    {
        magnitude = std::max(magnitude, std::abs(center[a]));
        reach     = std::max(reach, std::max(std::abs(double(bounds.minimum[a])), std::abs(double(bounds.maximum[a]))));
        extent += half[a];
    }
    const double inflation =
        (magnitude + reach + extent) * 64.0 * static_cast<double>(std::numeric_limits<float>::epsilon());
    physx::PxVec3 floatHalf;
    for (uint32_t a = 0; a < 3; ++a)
        floatHalf[a] = std::nextafter(float(half[a] + inflation), std::numeric_limits<float>::infinity());
    physx::PxVec3 columns[3];
    for (uint32_t i = 0; i < 3; ++i)
        columns[i] = physx::PxVec3(static_cast<float>(axes[i][0]), static_cast<float>(axes[i][1]),
                                   static_cast<float>(axes[i][2]));
    const physx::PxMat33 rotation(columns[0], columns[1], columns[2]);
    physx::PxQuat        orientation(rotation);
    orientation.normalize();
    const physx::PxVec3      position{float(center[0]), float(center[1]), float(center[2])};
    const physx::PxTransform pose(position, orientation);
    // Inside the input domain the float box and its pose are valid (ADR-0032).
    const physx::PxBoxGeometry box(floatHalf);
    const uint32_t             count = physx::PxMeshQuery::findOverlapTriangleMesh(
        box, pose, physx::PxTriangleMeshGeometry(mImpl->mesh), physx::PxTransform(physx::PxIdentity), results, capacity,
        0, overflow);
    const auto* remap = mImpl->mesh->getTrianglesRemap();
    if (remap)
        for (uint32_t a = 0; a < count; ++a)
            results[a] = remap[results[a]];
    return count;
}

ovspatialquery_api_status_t PhysXTriangleMesh::overlap(const TriangleObjectGeometry&         placement,
                                                       const ovspatialquery_overlap_shape_t& query,
                                                       bool&                                 hasOverlap) const
{
    return overlapPhysXGeometry(physx::PxTriangleMeshGeometry(mImpl->mesh), placement, query, hasOverlap,
                                CookedFrame{mImpl->cookedFromLocal, {}});
}

ovspatialquery_api_status_t overlapPhysXGeometry(const physx::PxGeometry& mesh, const TriangleObjectGeometry& placement,
                                                 const ovspatialquery_overlap_shape_t& query, bool& hasOverlap,
                                                 const CookedFrame& cooked)
{
    const double      cookedFromLocal = cooked.scale;
    const double      scale           = placement.uniformScale;
    double            center[3]{}, half[3]{}, axes[3][3]{};
    double            radius      = 0;
    const double*     worldCenter = nullptr;
    BoxGeometrySource box{};
    if (query.kind == OVSPATIALQUERY_OVERLAP_SPHERE)
    {
        worldCenter = query.data.sphere.center;
        radius      = query.data.sphere.radius / scale * cookedFromLocal;
        if (!(radius > 0))
            return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    else
    {
        if (!queryBoxGeometry(query, nullptr, box))
            return OVSPATIALQUERY_API_NOT_SUPPORTED;
        worldCenter = box.center.data();
        for (uint32_t a = 0; a < 3; ++a)
        {
            half[a] = box.halfExtents[a] / scale * cookedFromLocal;
            for (uint32_t b = 0; b < 3; ++b)
                for (uint32_t c = 0; c < 3; ++c)
                    axes[a][b] += placement.localFromWorldLinear[a * 3 + c] * scale * box.axes[b][c];
        }
    }
    for (uint32_t a = 0; a < 3; ++a)
    {
        center[a] = -cooked.origin[a] * cookedFromLocal;
        for (uint32_t b = 0; b < 3; ++b)
            center[a] = std::fma(placement.localFromWorldLinear[a * 3 + b] * cookedFromLocal,
                                 worldCenter[b] - placement.worldFromLocal[b * 4 + 3], center[a]);
    }
    const physx::PxVec3 position{float(center[0]), float(center[1]), float(center[2])};
    if (!position.isFinite())
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    if (radius > 0)
    {
        const physx::PxSphereGeometry sphere{float(radius)};
        if (!sphere.isValid())
            return OVSPATIALQUERY_API_NOT_SUPPORTED;
        hasOverlap = physx::PxGeometryQuery::overlap(sphere, physx::PxTransform(position), mesh,
                                                     physx::PxTransform(physx::PxIdentity));
    }
    else
    {
        const physx::PxBoxGeometry localBox{float(half[0]), float(half[1]), float(half[2])};
        if (!localBox.isValid())
            return OVSPATIALQUERY_API_NOT_SUPPORTED;
        const physx::PxMat33 rotation{physx::PxVec3(float(axes[0][0]), float(axes[1][0]), float(axes[2][0])),
                                      physx::PxVec3(float(axes[0][1]), float(axes[1][1]), float(axes[2][1])),
                                      physx::PxVec3(float(axes[0][2]), float(axes[1][2]), float(axes[2][2]))};
        hasOverlap = physx::PxGeometryQuery::overlap(
            localBox, physx::PxTransform(position, physx::PxQuat(rotation).getNormalized()), mesh,
            physx::PxTransform(physx::PxIdentity));
    }
    return OVSPATIALQUERY_API_SUCCESS;
}

} // namespace detail
} // namespace ovspatialquery
