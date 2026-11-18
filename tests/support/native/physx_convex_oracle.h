/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_TEST_PHYSX_CONVEX_ORACLE_H
#define OVSPATIALQUERY_TEST_PHYSX_CONVEX_ORACLE_H

// The hull a PhysX simulation of a stage collides with for a convexHull collider (ADR-0078),
// cooked with PhysX directly as Omni Physics cooks it: the mesh's points as given, every point in
// order at its own coordinates, eCOMPUTE_CONVEX under the vertex limit, GPU-compatible data and
// the stage's tolerance scale, 1 / metersPerUnit (a stage in metres by default). The oracle solid
// is the hull of the cooked vertices, which PhysX's contacts use; its merged polygons' planes can
// lie off those vertices, so the faces come from every plane through three vertices with all of
// them on one side. The caller has created PhysX's foundation, as acquiring the library's backend
// does.

#include <PxPhysicsAPI.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace physx_convex_oracle
{

using Vector = std::array<double, 3u>;

inline double dot(const Vector& a, const Vector& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

inline Vector cross(const Vector& a, const Vector& b)
{
    return Vector{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

inline Vector unit(const Vector& v)
{
    const double length = std::sqrt(dot(v, v));
    return Vector{v[0] / length, v[1] / length, v[2] / length};
}

// The cooked vertices, or none when PhysX cooks no hull.
inline std::vector<Vector> cookAsOmniPhysics(const float* points, size_t count, uint32_t vertexLimit,
                                             double metersPerUnit = 1.0)
{
    physx::PxConvexMeshDesc cooking{};
    cooking.points.count          = static_cast<physx::PxU32>(count);
    cooking.points.stride         = sizeof(float) * 3u;
    cooking.points.data           = points;
    cooking.flags                 = physx::PxConvexFlag::eCOMPUTE_CONVEX;
    cooking.vertexLimit           = static_cast<physx::PxU16>(vertexLimit);
    const float            length = static_cast<float>(1.0 / metersPerUnit);
    physx::PxCookingParams parameters{physx::PxTolerancesScale(length, 10.0f * length)};
    parameters.buildGPUData     = true;
    physx::PxConvexMesh* cooked = PxCreateConvexMesh(parameters, cooking, *PxGetStandaloneInsertionCallback());
    std::vector<Vector>  corners;
    if (cooked == nullptr)
        return corners;
    for (physx::PxU32 vertex = 0u; vertex < cooked->getNbVertices(); ++vertex)
    {
        const physx::PxVec3& corner = cooked->getVertices()[vertex];
        corners.push_back({double(corner.x), double(corner.y), double(corner.z)});
    }
    cooked->release();
    return corners;
}

struct Face
{
    Vector              normal;
    double              offset;
    std::vector<Vector> corners;
};

// The faces of the corners' hull, outward; `half` is the size the tolerances are relative to.
inline std::vector<Face> hullFaces(const std::vector<Vector>& corners, double half)
{
    std::vector<Face> faces;
    for (size_t a = 0u; a < corners.size(); ++a)
        for (size_t b = a + 1u; b < corners.size(); ++b)
            for (size_t c = b + 1u; c < corners.size(); ++c)
            {
                const Vector ab{corners[b][0] - corners[a][0], corners[b][1] - corners[a][1],
                                corners[b][2] - corners[a][2]};
                const Vector ac{corners[c][0] - corners[a][0], corners[c][1] - corners[a][1],
                                corners[c][2] - corners[a][2]};
                Vector       normal = cross(ab, ac);
                if (dot(normal, normal) < 1e-12 * half * half * half * half)
                    continue;
                normal       = unit(normal);
                double above = -INFINITY, below = INFINITY;
                for (const Vector& corner : corners)
                {
                    above = std::max(above, dot(normal, corner) - dot(normal, corners[a]));
                    below = std::min(below, dot(normal, corner) - dot(normal, corners[a]));
                }
                if (above > 1e-9 * half && below < -1e-9 * half)
                    continue;
                if (above > 1e-9 * half)
                    normal = {-normal[0], -normal[1], -normal[2]};
                bool known = false;
                for (const Face& face : faces)
                    known = known || dot(face.normal, normal) > 1.0 - 1e-12;
                if (known)
                    continue;
                Face face{normal, dot(normal, corners[a]), {}};
                for (const Vector& corner : corners)
                    if (std::abs(dot(normal, corner) - face.offset) <= 1e-9 * half)
                        face.corners.push_back(corner);
                faces.push_back(std::move(face));
            }
    return faces;
}

// The entry and exit distances of a ray through the faces' solid, clipped to [0, reach].
inline bool clipRay(const std::vector<Face>& faces, const Vector& origin, const Vector& direction, double reach,
                    double& entry, double& exit)
{
    entry = 0.0, exit = reach;
    for (const Face& face : faces)
    {
        const double rate = dot(face.normal, direction), height = dot(face.normal, origin) - face.offset;
        if (rate == 0.0)
        {
            if (height > 0.0)
                return false;
        }
        else if (rate < 0.0)
            entry = std::max(entry, -height / rate);
        else
            exit = std::min(exit, -height / rate);
    }
    return entry <= exit;
}

} // namespace physx_convex_oracle

#endif
