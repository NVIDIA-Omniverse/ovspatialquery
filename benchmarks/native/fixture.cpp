/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Fixture generation, movement replay and parity comparison.
#include "fixture.h"
#include "sphere_sweep_oracle.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>

namespace bench
{
const char* kindName(QueryKind kind)
{
    switch (kind)
    {
    case QueryKind::rays: return "rays";
    case QueryKind::sweep: return "sweep";
    case QueryKind::boxSweep: return "box-sweep";
    case QueryKind::occupancy: return "occupancy";
    case QueryKind::zones: return "zones";
    case QueryKind::sphereOverlap: return "sphere-overlap";
    case QueryKind::sphereAny: return "sphere-any";
    }
    return "unknown";
}
Fixture fixture(const Options& o)
{
    Fixture f;
    f.kind             = o.workload == "occupancy" ? QueryKind::occupancy
                         : o.workload == "zones"   ? QueryKind::zones
                                                   : QueryKind::rays;
    f.even             = o.subset == "even";
    f.zoneHitCapacity  = o.zoneHitCapacity;
    const size_t width = size_t(std::ceil(std::sqrt(double(o.objects))));
    for (size_t i = 0; i < o.objects; ++i)
    {
        f.objects.push_back({i + 1,
                             i / 2 + 1,
                             i % 3 == 1,
                             {double(i % width) * 2, double(i / width) * 2, 0.5},
                             i % 4 == 0 ? rotation(0.5f) : identity,
                             i % 2 ? 2u : 1u});
    }
    const size_t grid = size_t(std::sqrt(double(o.queries)));
    const double span = double(width) * 2;
    const double cell = span / double(std::max(size_t(1), grid));
    // MapGenerator: thin height bands still use at least one cell of box height.
    if (o.workload == "occupancy")
        f.half = {cell / 2, cell / 2, std::max(0.25 / 2, cell / 2)};
    for (size_t i = 0; i < o.queries; ++i)
    {
        Query       q;
        const auto& object = f.objects[(i * 37) % o.objects];
        q.p                = object.p;
        if (o.workload == "rays")
        {
            q.p[0] += i % 5 == 4 ? 0.875 : 0.125;
            q.p[1] += 0.0625;
            q.p[2] = 4;
        }
        else if (o.workload == "occupancy")
        {
            q.p = {(double(i % grid) + 0.5) * cell - 1, (double(i / grid) + 0.5) * cell - 1, 0.125};
        }
        else
        {
            q.q = i % 2 ? rotation(0.375f) : identity;
            q.p[0] += 0.25;
        }
        f.queries.push_back(q);
    }
    return f;
}
std::array<std::array<double, 3>, 3> rotationMatrix(const Quat& q)
{
    return rotationMatrix(std::array<double, 4>{q[0], q[1], q[2], q[3]});
}
std::array<std::array<double, 3>, 3> rotationMatrix(const std::array<double, 4>& q)
{
    const double x = q[0], y = q[1], z = q[2], w = q[3];
    return {{{1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)},
             {2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)},
             {2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)}}};
}
Aabb worldBounds(const Object& ob)
{
    require(!ob.mesh && !ob.convex && !ob.capsule, "world bounds oracle covers spheres and boxes only");
    Aabb b{ob.p, ob.p};
    if (ob.sphere)
    {
        for (size_t axis = 0; axis < 3; ++axis)
        {
            b.minimum[axis] -= sphereRadius;
            b.maximum[axis] += sphereRadius;
        }
        return b;
    }
    const auto m = rotationMatrix(ob.q);
    for (size_t axis = 0; axis < 3; ++axis)
    {
        const double extent =
            std::abs(m[axis][0]) * boxHalf[0] + std::abs(m[axis][1]) * boxHalf[1] + std::abs(m[axis][2]) * boxHalf[2];
        b.minimum[axis] -= extent;
        b.maximum[axis] += extent;
    }
    return b;
}
std::vector<uint64_t> identities(const Row& row)
{
    std::vector<uint64_t> ids;
    for (const auto& hit : row.hits)
        ids.push_back(hit.id);
    std::sort(ids.begin(), ids.end());
    return ids;
}
std::vector<uint64_t> owners(const Fixture& f, const Row& row, uint64_t excluded)
{
    std::vector<uint64_t> ids;
    for (const auto& hit : row.hits)
    {
        require(hit.id && hit.id <= f.objects.size(), "unknown application identity");
        const uint64_t owner = f.objects[size_t(hit.id - 1)].owner;
        if (owner != excluded)
            ids.push_back(owner);
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return ids;
}
void validateMeshFace(const Fixture& f, const Hit& h, const std::string& where)
{
    // A direct PhysX initial-overlap sweep has no defined contact position.
    if (h.geometryType != OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH || !h.positionValid)
        return;
    const auto& mesh = f.meshes.at(size_t(h.geometryValue - 1));
    require(h.sourceFace < mesh.triangles.size(), where + " mesh source face out of range");
    const auto& object = f.objects.at(size_t(h.id - 1));
    const auto& tri    = mesh.triangles[size_t(h.sourceFace)];
    const auto  point  = pose(object.p, object.q).transformInv(px(h.position));
    const auto  a = meshVertex(mesh.vertices[tri[0]]), u = meshVertex(mesh.vertices[tri[1]]) - a,
               v      = meshVertex(mesh.vertices[tri[2]]) - a;
    const auto w      = point - a;
    const auto normal = u.cross(v);
    const auto detail = [&]()
    {
        return where + " object " + std::to_string(h.id) + " face " + std::to_string(h.sourceFace) + " distance " +
               std::to_string(h.distance) + " position (" + std::to_string(h.position[0]) + ", " +
               std::to_string(h.position[1]) + ", " + std::to_string(h.position[2]) + ")";
    };
    require(std::abs(w.dot(normal)) <= 2e-4 * normal.magnitude(),
            "contact outside reported source face plane: " + detail());
    // Absolute tolerance in mesh units: dense consumer meshes contain slivers whose barycentric
    // coordinates amplify a float contact position by orders of magnitude, so the contact must lie
    // within 0.2 mm of the reported triangle rather than within a barycentric fraction of it.
    const double uu = u.dot(u), uv = u.dot(v), vv = v.dot(v), wu = w.dot(u), wv = w.dot(v), denom = uu * vv - uv * uv;
    const double b = (wu * vv - wv * uv) / denom, c = (wv * uu - wu * uv) / denom;
    auto         clampToTriangle = [&]()
    {
        // Closest point on the triangle to `point`: clamp the barycentric projection to each edge.
        const auto candidates   = {a, a + u, a + v};
        auto       best         = point;
        double     bestDistance = -1;
        const auto consider     = [&](const PxVec3& candidate)
        {
            const double distance = (point - candidate).magnitude();
            if (bestDistance < 0 || distance < bestDistance)
            {
                bestDistance = distance;
                best         = candidate;
            }
        };
        if (b >= 0 && c >= 0 && b + c <= 1)
            return 0.0;
        for (const auto& corner : candidates)
            consider(corner);
        const auto segment = [&](const PxVec3& start, const PxVec3& end)
        {
            const auto   direction = end - start;
            const double t =
                std::min(1.0, std::max(0.0, double((point - start).dot(direction)) / direction.magnitudeSquared()));
            consider(start + direction * float(t));
        };
        segment(a, a + u);
        segment(a, a + v);
        segment(a + u, a + v);
        return bestDistance;
    };
    const double offset = clampToTriangle();
    require(offset <= 2e-4, "contact outside reported source triangle: " + detail() + " barycentric (" +
                                std::to_string(b) + ", " + std::to_string(c) + ") offset " + std::to_string(offset));
}
namespace
{
// World-space corners of the triangle a mesh hit reports.
bool hitTriangle(const Fixture& f, const Hit& h, std::array<Vec, 3>& corners)
{
    if (h.geometryType != OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH || h.sourceFace == UINT64_MAX)
        return false;
    const auto& mesh   = f.meshes.at(size_t(h.geometryValue - 1));
    const auto& object = f.objects.at(size_t(h.id - 1));
    const auto& tri    = mesh.triangles.at(size_t(h.sourceFace));
    const auto  m      = rotationMatrix(object.q);
    for (size_t c = 0; c < 3; ++c)
    {
        const auto& v = mesh.vertices[tri[c]];
        for (size_t k = 0; k < 3; ++k)
            corners[c][k] = object.p[k] + m[k][0] * v[0] + m[k][1] * v[1] + m[k][2] * v[2];
    }
    return true;
}
Vec    sub(const Vec& a, const Vec& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
double dot(const Vec& a, const Vec& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Vec    cross(const Vec& a, const Vec& b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
// Moller-Trumbore in double; negative when the ray misses the face.
double rayFaceDistance(const Vec& origin, const Vec& direction, bool doubleSided, const std::array<Vec, 3>& c)
{
    const Vec    e1 = sub(c[1], c[0]), e2 = sub(c[2], c[0]), p = cross(direction, e2);
    const double det = dot(e1, p);
    // Back faces of one-sided meshes (det < 0, the ray leaving the counter-clockwise front) are not hits; the callers
    // only evaluate reported faces.
    if (std::abs(det) < 1e-300 || (!doubleSided && det < 0))
        return -1;
    const Vec    t = sub(origin, c[0]);
    const double u = dot(t, p) / det;
    const Vec    q = cross(t, e1);
    const double v = dot(direction, q) / det, distance = dot(e2, q) / det;
    const double slack = 1e-9;
    if (u < -slack || v < -slack || u + v > 1 + slack)
        return -1;
    return distance;
}
// The convex-search oracle the native tests share (support/native/sphere_sweep_oracle.h) for a
// query and a face: the center moves along p + t d with the float direction as given, as the
// library's distances do, and no formula is shared with the library's face, edge and vertex roots.
template <typename Corners>
sphere_sweep_oracle::Path sweepPath(const Query& q, const Corners& c)
{
    double corner[3][3] = {};
    for (size_t k = 0; k < 3; ++k)
        for (size_t axis = 0; axis < 3; ++axis)
            corner[k][axis] = double(c[k][axis]);
    const double center[3] = {q.p[0], q.p[1], q.p[2]};
    return sphere_sweep_oracle::makePath(corner, center, q.direction.data());
}
// The largest center distance the library's sphere rule accepts as touching.
double sphereAcceptance(double radius)
{
    return std::sqrt(radius * radius + std::max(1.0, radius * radius) * std::numeric_limits<double>::epsilon() * 256);
}
// Returned by the sphere-sweep face distance for a face the library never meets: one-sided, with
// its front normal along the motion.
constexpr double kIgnoredFace = -2;
// First parameter in [near, far] at which a sphere of `radius` touches the face; -1 when it never
// does or the face is degenerate. As in the library (ADR-0051), a one-sided face whose front normal
// n = e1 x e2 points along the motion, n . d > 0, is ignored (kIgnoredFace); every other face is
// met from either side.
template <typename Corners>
double sweepFaceDistance(const Query& q, double radius, bool doubleSided, const Corners& c)
{
    const auto path = sweepPath(q, c);
    if (sphere_sweep_oracle::dot(path.normal, path.normal) == 0)
        return -1;
    if (!doubleSided && sphere_sweep_oracle::dot(path.normal, path.direction) > 0)
        return kIgnoredFace;
    sphere_sweep_oracle::Wide first = 0;
    if (!sphere_sweep_oracle::firstWithin(path, radius, q.nearDistance, q.farDistance, first))
        return -1;
    return double(first);
}
// The box oracle works in long double. The library normalizes float quaternions, which sit up to
// ~1e-7 off unit length, so these rotations are normalized too.
using Wide    = long double;
using WideVec = std::array<Wide, 3>;
using WideTri = std::array<WideVec, 3>;
Wide    wideDot(const WideVec& a, const WideVec& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
WideVec wideSub(const WideVec& a, const WideVec& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
WideVec wideCross(const WideVec& a, const WideVec& b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
// Column-vector rotation m[row][column] of a normalized quaternion.
std::array<WideVec, 3> wideRotation(const Quat& q)
{
    Wide       x = q[0], y = q[1], z = q[2], w = q[3];
    const Wide length = std::sqrt(x * x + y * y + z * z + w * w);
    x /= length;
    y /= length;
    z /= length;
    w /= length;
    return {{{1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)},
             {2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)},
             {2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)}}};
}
// World corners of the triangle a mesh hit reports; false for any other hit.
bool wideHitTriangle(const Fixture& f, const Hit& h, WideTri& corners)
{
    if (h.geometryType != OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH || h.id == 0 || h.id > f.objects.size() ||
        h.geometryValue == 0 || h.geometryValue > f.meshes.size() ||
        h.sourceFace >= f.meshes[size_t(h.geometryValue - 1)].triangles.size())
        return false;
    const auto& mesh   = f.meshes[size_t(h.geometryValue - 1)];
    const auto& object = f.objects[size_t(h.id - 1)];
    const auto& tri    = mesh.triangles[size_t(h.sourceFace)];
    const auto  m      = wideRotation(object.q);
    for (size_t c = 0; c < 3; ++c)
    {
        const auto& v = mesh.vertices[tri[c]];
        for (size_t k = 0; k < 3; ++k)
            corners[c][k] = Wide(object.p[k]) + m[k][0] * v[0] + m[k][1] * v[1] + m[k][2] * v[2];
    }
    return true;
}
Wide wideDistanceToTriangle(const WideVec& p, const WideTri& c)
{
    const WideVec u = wideSub(c[1], c[0]), v = wideSub(c[2], c[0]), w = wideSub(p, c[0]);
    const WideVec normal        = wideCross(u, v);
    const Wide    normalSquared = wideDot(normal, normal);
    if (normalSquared > 0)
    {
        // The Gram determinant uu*vv-uv*uv loses digits on long, thin triangles when MSVC's
        // long double is double. Cross products preserve the small perpendicular component.
        const Wide b  = wideDot(wideCross(w, v), normal) / normalSquared;
        const Wide cc = wideDot(wideCross(u, w), normal) / normalSquared;
        if (b >= 0 && cc >= 0 && b + cc <= 1)
        {
            const WideVec foot = {c[0][0] + u[0] * b + v[0] * cc, c[0][1] + u[1] * b + v[1] * cc,
                                  c[0][2] + u[2] * b + v[2] * cc};
            return std::sqrt(wideDot(wideSub(p, foot), wideSub(p, foot)));
        }
    }
    Wide best = -1;
    for (size_t k = 0; k < 3; ++k)
    {
        const WideVec e  = wideSub(c[(k + 1) % 3], c[k]);
        const Wide    ee = wideDot(e, e);
        const Wide    t  = ee > 0 ? std::min<Wide>(1, std::max<Wide>(0, wideDot(wideSub(p, c[k]), e) / ee)) : 0;
        const WideVec q  = {c[k][0] + e[0] * t, c[k][1] + e[1] * t, c[k][2] + e[2] * t};
        const Wide    d  = std::sqrt(wideDot(wideSub(p, q), wideSub(p, q)));
        if (best < 0 || d < best)
            best = d;
    }
    return best;
}
// First distance in [near, far] at which the query's box, translated along its direction, meets
// the triangle. It is the least t over triangle points a + u e1 + v e2 (u, v >= 0, u + v <= 1)
// with |A_k . (point - center - t d)| <= h_k on each box axis A_k: a three-variable linear program
// whose minimum lies at a vertex where three constraints are active. Enumerating those vertices
// in long double shares nothing with the library's separating-axis intervals. As in the library
// (ADR-0051), a one-sided face counts only when the motion runs against its front normal,
// n . d < 0 with n = e1 x e2, and is then met from either side. Negative when never met or
// degenerate, which the library also never meets.
Wide boxSweepFaceDistance(const Query& q, const Vec& half, const WideTri& c, bool doubleSided)
{
    const auto    axes      = wideRotation(q.q);
    const WideVec center    = {q.p[0], q.p[1], q.p[2]};
    const WideVec direction = {q.direction[0], q.direction[1], q.direction[2]};
    const WideVec e1 = wideSub(c[1], c[0]), e2 = wideSub(c[2], c[0]), offset = wideSub(c[0], center);
    const WideVec normal = wideCross(e1, e2);
    if (wideDot(normal, normal) == 0 || (!doubleSided && !(wideDot(normal, direction) < 0)))
        return -1;
    // Rows g . (u, v, t) <= h, stored as {g, h}.
    std::array<std::array<Wide, 4>, 11> rows{};
    size_t                              count = 0;
    for (size_t k = 0; k < 3; ++k)
    {
        const WideVec axis = {axes[0][k], axes[1][k], axes[2][k]};
        const Wide    u = wideDot(axis, e1), v = wideDot(axis, e2), along = wideDot(axis, direction);
        const Wide    from = wideDot(axis, offset);
        rows[count++]      = {u, v, -along, Wide(half[k]) - from};
        rows[count++]      = {-u, -v, along, Wide(half[k]) + from};
    }
    rows[count++] = {-1, 0, 0, 0};
    rows[count++] = {0, -1, 0, 0};
    rows[count++] = {1, 1, 0, 1};
    rows[count++] = {0, 0, -1, -Wide(q.nearDistance)};
    rows[count++] = {0, 0, 1, Wide(q.farDistance)};
    // A face-to-face contact is a degenerate vertex where more than three constraints are active,
    // and rounding leaves its solution just outside one of them; a tighter slack rejected such
    // vertices on the rack lattices and returned a later one. This one is about 1e-15 of a metre
    // plus of the terms in x87 long double, and proportionally more where long double is double.
    const Wide slack = 1e4L * std::numeric_limits<Wide>::epsilon();
    Wide       best  = 0;
    bool       found = false;
    for (size_t i = 0; i < count; ++i)
        for (size_t j = i + 1; j < count; ++j)
            for (size_t k = j + 1; k < count; ++k)
            {
                const auto &a = rows[i], &b = rows[j], &g = rows[k];
                const Wide  m0 = b[1] * g[2] - b[2] * g[1], m1 = b[0] * g[2] - b[2] * g[0],
                           m2  = b[0] * g[1] - b[1] * g[0];
                const Wide det = a[0] * m0 - a[1] * m1 + a[2] * m2;
                if (det == 0)
                    continue; // Parallel constraints; nearly parallel ones fail the feasibility test.
                const Wide t =
                    (a[0] * (b[1] * g[3] - b[3] * g[1]) - a[1] * (b[0] * g[3] - b[3] * g[0]) + a[3] * m2) / det;
                if (found && t >= best)
                    continue;
                const Wide u =
                    (a[3] * m0 - a[1] * (b[3] * g[2] - b[2] * g[3]) + a[2] * (b[3] * g[1] - b[1] * g[3])) / det;
                const Wide v =
                    (a[0] * (b[3] * g[2] - b[2] * g[3]) - a[3] * m1 + a[2] * (b[0] * g[3] - b[3] * g[0])) / det;
                bool feasible = true;
                for (size_t n = 0; n < count && feasible; ++n)
                {
                    const auto& row = rows[n];
                    feasible =
                        row[0] * u + row[1] * v + row[2] * t - row[3] <=
                        slack * (1 + std::abs(row[3]) + std::abs(row[0]) + std::abs(row[1]) + std::abs(row[2] * t));
                }
                if (feasible)
                {
                    best  = t;
                    found = true;
                }
            }
    if (!found)
        return -1;
    // The slack admits a start a few roundoffs before the range; the range bounds the time.
    return std::max(best, Wide(q.nearDistance));
}
} // namespace
namespace
{
// Distance from the fixture's moving sphere center at `distance` to the face a mesh hit reports,
// or a negative value for any other hit.
double sweepHitFaceGap(const Fixture& f, size_t i, const Hit& h, double distance)
{
    WideTri corners{};
    if (!wideHitTriangle(f, h, corners))
        return -1;
    return double(sphere_sweep_oracle::distanceAt(sweepPath(f.queries[i], corners), distance));
}
} // namespace
double analyticFaceDistance(const Fixture& f, size_t i, const Hit& h)
{
    std::array<Vec, 3> corners{};
    if (!hitTriangle(f, h, corners))
        return -1;
    const auto& q           = f.queries[i];
    const bool  doubleSided = f.meshes.at(size_t(h.geometryValue - 1)).doubleSided;
    if (f.kind == QueryKind::rays)
        return rayFaceDistance(q.p, {double(q.direction[0]), double(q.direction[1]), double(q.direction[2])},
                               doubleSided, corners);
    // Sweep contacts are held to rounding, so their corners come from the normalized rotation the
    // library applies, not the float quaternion's slightly skewed matrix.
    WideTri wide{};
    if (f.kind == QueryKind::sweep && wideHitTriangle(f, h, wide))
        return sweepFaceDistance(q, f.rowRadii ? q.radius : f.radius, doubleSided, wide);
    return -1;
}
bool near(double a, double b, double tolerance)
{
    return std::abs(a - b) <= std::max(tolerance, 2e-5 + 2e-6 * std::max(std::abs(a), std::abs(b)));
}
void validateBoxSweepContact(const Fixture& f, size_t queryIndex, const Hit& hit, const std::string& where,
                             double slack)
{
    if (!hit.positionValid)
        return; // Initial overlaps have no unique surface contact.
    const auto& q        = f.queries[queryIndex];
    const auto  rotation = rotationMatrix(q.q);
    const Vec center = {q.p[0] + double(q.direction[0]) * hit.distance, q.p[1] + double(q.direction[1]) * hit.distance,
                        q.p[2] + double(q.direction[2]) * hit.distance};
    const Vec relative = sub(hit.position, center);
    double    boundary = 1e30;
    for (size_t axis = 0; axis < 3; ++axis)
    {
        const double coordinate =
            rotation[0][axis] * relative[0] + rotation[1][axis] * relative[1] + rotation[2][axis] * relative[2];
        const double gap = std::abs(std::abs(coordinate) - f.half[axis]);
        boundary         = std::min(boundary, gap);
        require(std::abs(coordinate) <= f.half[axis] + slack, where + " contact outside swept box at axis " +
                                                                  std::to_string(axis) + " by " +
                                                                  std::to_string(std::abs(coordinate) - f.half[axis]));
    }
    require(boundary <= slack,
            where + " contact inside swept box rather than on its surface; gap " + std::to_string(boundary));
    const Vec    normal       = {hit.normal[0], hit.normal[1], hit.normal[2]};
    const double normalLength = std::sqrt(dot(normal, normal));
    require(std::abs(normalLength - 1) <= 2e-3,
            where + " contact normal is not unit length: " + std::to_string(normalLength));
    require(dot(normal, {q.direction[0], q.direction[1], q.direction[2]}) <= 2e-3,
            where + " contact normal points along sweep travel");
}
namespace
{
std::string precise(double value)
{
    char text[32];
    std::snprintf(text, sizeof(text), "%.17g", value);
    return text;
}
// Installed box contacts are double evaluations of float geometry: over the mesh and rack
// lattices they match the long-double solve of their face within 2e-15 of max(1 m, distance)
// and lie within 7e-13 of their coordinates' size from it. This bound keeps at least three
// orders of magnitude above both.
constexpr double kBoxExact = 1e-9;
// Installed sphere-sweep contacts against the convex-search oracle, as a fraction of max(1 m, distance).
constexpr double kSphereExact = 1e-9;
// Double-precision mesh ray contacts against the analytic point of their face, as a fraction of
// max(1 m, distance): CUDA's lie within 1.3e-14 of it 3 m away on the camera blob.
constexpr double kRayExact = 1e-9;
// Whether a mesh ray contact is, to rounding, the point at which the ray meets its reported face
// by the double-precision evaluation (`rayFaceDistance`).
bool atAnalyticRayPoint(const Fixture& f, size_t i, const Hit& h)
{
    if (f.kind != QueryKind::rays || h.geometryType != OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH)
        return false;
    const double analytic = analyticFaceDistance(f, i, h);
    const double bound    = kRayExact * std::max(1.0, analytic);
    if (analytic < 0 || std::abs(h.distance - analytic) > bound)
        return false;
    const auto& q = f.queries[i];
    for (size_t k = 0; k < 3; ++k)
        if (std::abs(h.position[k] - (q.p[k] + double(q.direction[k]) * analytic)) > bound)
            return false;
    return true;
}
// Checks an installed box contact against its own face and returns the distance at which that face
// first meets the swept box: the reported distance, with the contact on the face and on the box
// there. The face and edge contacts of a box are patches and segments, so two correct evaluators
// can report different points or, when several triangles are met at once, different faces.
double validateInstalledBoxContact(const Fixture& f, size_t i, const Hit& h, const std::string& where)
{
    validateBoxSweepContact(f, i, h, where, 1e-7);
    if (h.geometryType == OVSPATIALQUERY_GEOMETRY_BOX && h.id != 0 && h.id <= f.objects.size())
    {
        // A primitive box is met first where the swept box first meets one of its faces, each two
        // triangles of the same long-double solve.
        const auto&       object  = f.objects[size_t(h.id - 1)];
        const auto        m       = wideRotation(object.q);
        const std::string subject = where + " box object " + std::to_string(h.id);
        // Every primitive box of a fixture has the half extents boxHalf.
        require(object.geometryType() == OVSPATIALQUERY_GEOMETRY_BOX, subject + " is not a fixture box");
        Wide first = -1, off = -1;
        for (size_t axis = 0; axis < 3; ++axis)
            for (const Wide side : {Wide(-1), Wide(1)})
            {
                // The face's corners, walking its two other axes around a square.
                std::array<WideVec, 4> quad{};
                const size_t           u = (axis + 1) % 3, v = (axis + 2) % 3;
                const Wide             su[4] = {-1, 1, 1, -1}, sv[4] = {-1, -1, 1, 1};
                for (size_t c = 0; c < 4; ++c)
                {
                    Wide local[3]{};
                    local[axis] = side * Wide(boxHalf[axis]);
                    local[u]    = su[c] * Wide(boxHalf[u]);
                    local[v]    = sv[c] * Wide(boxHalf[v]);
                    for (size_t k = 0; k < 3; ++k)
                        quad[c][k] = Wide(object.p[k]) + m[k][0] * local[0] + m[k][1] * local[1] + m[k][2] * local[2];
                }
                for (const WideTri& tri : {WideTri{quad[0], quad[1], quad[2]}, WideTri{quad[0], quad[2], quad[3]}})
                {
                    const Wide t = boxSweepFaceDistance(f.queries[i], f.half, tri, true);
                    if (t >= 0 && (first < 0 || t < first))
                        first = t;
                    const Wide d = wideDistanceToTriangle({h.position[0], h.position[1], h.position[2]}, tri);
                    if (off < 0 || d < off)
                        off = d;
                }
            }
        require(first >= 0 && std::abs(double(first) - h.distance) <= kBoxExact * std::max(1.0, h.distance),
                subject + " reported at distance " + precise(h.distance) + " first meets the swept box at " +
                    precise(double(first)));
        const double scale = std::max({1.0, std::abs(h.position[0]), std::abs(h.position[1]), std::abs(h.position[2])});
        require(double(off) <= kBoxExact * scale,
                subject + " contact lies " + precise(double(off)) + " off its surface");
        return double(first);
    }
    WideTri corners{};
    require(wideHitTriangle(f, h, corners),
            where + " reports no face of a fixture mesh; box sweep parity covers triangle-mesh and box contacts");
    const std::string subject = where + " object " + std::to_string(h.id) + " face " + std::to_string(h.sourceFace);
    const double      first   = double(
        boxSweepFaceDistance(f.queries[i], f.half, corners, f.meshes.at(size_t(h.geometryValue - 1)).doubleSided));
    require(first >= 0 && std::abs(first - h.distance) <= kBoxExact * std::max(1.0, h.distance),
            subject + " reported at distance " + precise(h.distance) + " first meets the swept box at " +
                precise(first));
    const double scale = std::max({1.0, std::abs(h.position[0]), std::abs(h.position[1]), std::abs(h.position[2])});
    const double off   = double(wideDistanceToTriangle({h.position[0], h.position[1], h.position[2]}, corners));
    require(off <= kBoxExact * scale, subject + " contact lies " + precise(off) + " off its face");
    return first;
}
// Box sweep rows: installed contacts are checked independently, then compared on distance. They
// must report the same object and face unless the independent solve shows that both faces meet
// the box together; rounding decides such ties differently on CPU and CUDA. Normals are compared
// on a common face. A direct PhysX baseline follows its named limitation, and its reported face
// is solved too.
void compareBoxSweepRow(const Fixture& f, size_t i, const Hit& ah, const Hit& bh, const std::string& where, bool ties,
                        CompareStats* stats, bool firstIsPhysx)
{
    const double second = validateInstalledBoxContact(f, i, bh, where + " second");
    if (stats)
        stats->maxDistanceErrorSecond = std::max(stats->maxDistanceErrorSecond, std::abs(second - bh.distance));
    const std::string pair = " (objects " + std::to_string(ah.id) + " vs " + std::to_string(bh.id) + ", faces " +
                             std::to_string(ah.sourceFace) + " vs " + std::to_string(bh.sourceFace) + ")";
    if (firstIsPhysx)
    {
        // PhysX's distance is not trusted, but its face is: if the box meets that face before the
        // installed first contact, the installed path missed it. A face met at that contact is a
        // tie and may belong to another object; back faces and missed slivers never meet it.
        WideTri      corners{};
        const Wide   baseline = wideHitTriangle(f, ah, corners)
                                    ? boxSweepFaceDistance(f.queries[i], f.half, corners,
                                                           f.meshes[size_t(ah.geometryValue - 1)].doubleSided)
                                    : -1;
        const double bound    = kBoxExact * std::max(1.0, bh.distance);
        require(baseline < 0 || double(baseline) >= bh.distance - bound,
                where + " baseline face is met at " + precise(double(baseline)) +
                    ", before the installed first contact " + precise(bh.distance) + pair);
        const bool tie = baseline >= 0 && double(baseline) <= bh.distance + bound;
        require(ties || tie || ah.id == bh.id, where + " closest identity differs" + pair);
        const bool   close = near(ah.distance, bh.distance);
        const double early = f.boxSweepBaselineEarlyTolerance * std::max(1.0, bh.distance);
        require(close || ah.distance <= bh.distance, where + " baseline distance " + precise(ah.distance) +
                                                         " is after the installed first contact " +
                                                         precise(bh.distance) + pair);
        require(close || ah.distance >= bh.distance - early,
                where + " baseline distance " + precise(ah.distance) + " is before the installed first contact " +
                    precise(bh.distance) + " beyond the declared limitation" + pair);
        if (stats)
            stats->maxBaselineEarly = std::max(stats->maxBaselineEarly, bh.distance - ah.distance);
    }
    else
    {
        const double first = validateInstalledBoxContact(f, i, ah, where + " first");
        if (stats)
            stats->maxDistanceErrorFirst = std::max(stats->maxDistanceErrorFirst, std::abs(first - ah.distance));
        require(near(ah.distance, bh.distance),
                where + " distance " + precise(ah.distance) + " vs " + precise(bh.distance) + pair);
        const bool sameFace = ah.id == bh.id && ah.sourceFace == bh.sourceFace;
        require(ties || sameFace || std::abs(first - second) <= kBoxExact * std::max(1.0, bh.distance),
                where + " closest faces differ and do not meet the box together: first at " + precise(first) +
                    ", second at " + precise(second) + pair);
        if (sameFace)
            for (size_t k = 0; k < 3; ++k)
                require(near(ah.normal[k], bh.normal[k]), where + " normal axis " + std::to_string(k) + " " +
                                                              std::to_string(ah.normal[k]) + " vs " +
                                                              std::to_string(bh.normal[k]));
    }
    if (ah.id == bh.id)
        require(ah.geometryType == bh.geometryType && ah.geometryValue == bh.geometryValue,
                where + " geometry identity differs");
}
} // namespace
void compare(const Fixture& f, const Results& a, const Results& b, const std::string& label, bool ties,
             bool contactFields, CompareStats* stats, bool firstIsPhysx)
{
    const double declaredTolerance = f.contactTolerance;
    require(a.size() == b.size(), label + " result row count");
    for (size_t i = 0; i < a.size(); ++i)
    {
        const std::string where = label + " query " + std::to_string(i);
        require(a[i].hits.size() == b[i].hits.size(),
                where + " hit count " + std::to_string(a[i].hits.size()) + " vs " + std::to_string(b[i].hits.size()));
        if (f.kind != QueryKind::rays && f.kind != QueryKind::sweep && f.kind != QueryKind::boxSweep)
        {
            require(identities(a[i]) == identities(b[i]), where + " unordered shape identities differ");
            if (f.kind == QueryKind::zones)
                require(owners(f, a[i]) == owners(f, b[i]), where + " deduplicated owners differ");
        }
        else if (!a[i].hits.empty())
        {
            const auto& ah = a[i].hits[0];
            const auto& bh = b[i].hits[0];
            if (f.kind == QueryKind::boxSweep)
            {
                compareBoxSweepRow(f, i, ah, bh, where, ties, stats, firstIsPhysx);
                continue;
            }
            validateMeshFace(f, ah, where + " first");
            validateMeshFace(f, bh, where + " second");
            // Independent analytic check of each reported distance against the reported triangle.
            for (const auto* side : {&ah, &bh})
            {
                if (side == &ah && !contactFields)
                    continue;
                const double analytic     = analyticFaceDistance(f, i, *side);
                const bool   baselineSide = firstIsPhysx && side == &ah;
                // The library ignores a one-sided face it moves along; an installed contact on one is wrong.
                require(baselineSide || analytic != kIgnoredFace,
                        where + (side == &ah ? " first" : " second") + " contact at " + precise(side->distance) +
                            " is on one-sided object " + std::to_string(side->id) + " face " +
                            std::to_string(side->sourceFace) + ", whose front normal points along the sweep");
                if (analytic < 0)
                {
                    // Not the face's first touch at the radius (a grazing touch within the rule's
                    // acceptance): an installed contact must still touch its face at the reported distance.
                    if (f.kind == QueryKind::sweep && !baselineSide &&
                        side->geometryType == OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH)
                    {
                        const double radius = f.rowRadii ? f.queries[i].radius : f.radius;
                        require(sweepHitFaceGap(f, i, *side, side->distance) <=
                                    sphereAcceptance(radius) + kSphereExact * std::max(1.0, radius),
                                where + (side == &ah ? " first" : " second") + " contact at " +
                                    precise(side->distance) + " does not touch object " + std::to_string(side->id) +
                                    " face " + std::to_string(side->sourceFace));
                    }
                    continue;
                }
                const double error = std::abs(analytic - side->distance);
                // Installed sphere sweeps meet the oracle to rounding; a direct PhysX sweep follows its
                // declared relative limitation, and ray contacts the declared contact tolerance.
                const double allowed =
                    f.kind != QueryKind::sweep ? std::max(declaredTolerance, 2e-5 + 2e-6 * side->distance)
                    : baselineSide             ? std::max(f.sweepRelativeTolerance * std::max(1.0, side->distance),
                                                          2e-5 + 2e-6 * side->distance)
                                               : kSphereExact * std::max(1.0, side->distance);
                if (stats)
                {
                    double& worst = side == &ah ? stats->maxDistanceErrorFirst : stats->maxDistanceErrorSecond;
                    if (error > worst && error > std::min(allowed, 2e-5 + 2e-6 * side->distance))
                        std::cerr << where << (side == &ah ? " first" : " second") << " analytic " << precise(analytic)
                                  << " reported " << precise(side->distance) << " object " << side->id << " face "
                                  << side->sourceFace << '\n';
                    worst = std::max(worst, error);
                    ++(side == &ah ? stats->analyticFirst : stats->analyticSecond);
                }
                // A direct PhysX sphere sweep can name a neighbouring face of the object that it meets
                // later than the distance it reports (Fixture::sweepRelativeTolerance); that distance is
                // compared with the installed contact below. A face met before it is still an error.
                const bool laterBaselineFace = baselineSide && f.kind == QueryKind::sweep && analytic > side->distance;
                require(laterBaselineFace || error <= allowed,
                        where + (side == &ah ? " first" : " second") + " distance " + precise(side->distance) +
                            " differs from the analytic " + precise(analytic) + " for object " +
                            std::to_string(side->id) + " face " + std::to_string(side->sourceFace));
            }
            // Independent analytic check for mesh ray contacts: the contact must lie on the ray at the
            // reported distance. Primitive probes with interior origins define their contact
            // differently and are covered by the semantic smoke. An approximate baseline's own contact
            // fields are not checked; the installed side always is.
            if (f.kind == QueryKind::rays)
                for (const auto* side : {&ah, &bh})
                {
                    if ((side == &ah && !contactFields) || side->geometryType != OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH)
                        continue;
                    const auto& q   = f.queries[i];
                    double      off = 0;
                    for (size_t k = 0; k < 3; ++k)
                    {
                        const double expected = q.p[k] + double(q.direction[k]) * side->distance;
                        off += (side->position[k] - expected) * (side->position[k] - expected);
                    }
                    if (stats)
                        (side == &ah ? stats->maxOffRayFirst : stats->maxOffRaySecond) =
                            std::max(side == &ah ? stats->maxOffRayFirst : stats->maxOffRaySecond, std::sqrt(off));
                    require(std::sqrt(off) <= std::max(declaredTolerance, 1e-4 + 1e-6 * side->distance),
                            where + (side == &ah ? " first" : " second") + " contact " +
                                std::to_string(std::sqrt(off)) + " off the ray at distance " +
                                std::to_string(side->distance) + " (object " + std::to_string(side->id) + ", face " +
                                std::to_string(side->sourceFace) + ")");
                }
            // A direct PhysX sphere distance is approximate, but its face is real: if the sphere
            // clearly touches that face before the installed first contact, the installed path
            // missed it (CPU and CUDA share the rule, so only this catches a shared late contact).
            if (firstIsPhysx && f.kind == QueryKind::sweep && ah.geometryType == OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH)
            {
                const double baselineFace = analyticFaceDistance(f, i, ah);
                require(baselineFace < 0 || baselineFace >= bh.distance - kSphereExact * std::max(1.0, bh.distance),
                        where + " baseline face " + std::to_string(ah.sourceFace) + " of object " +
                            std::to_string(ah.id) + " is met at " + precise(baselineFace) +
                            ", before the installed first contact " + precise(bh.distance));
            }
            // Installed sphere sweeps on triangle meshes are compared at the oracle's rounding level,
            // without `near`'s floor, so a face chosen a few micrometres late is still adjudicated below.
            const bool exactSweep = f.kind == QueryKind::sweep && !firstIsPhysx &&
                                    ah.geometryType == OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH &&
                                    bh.geometryType == OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH;
            const double tolerance    = f.kind == QueryKind::sweep
                                            ? (exactSweep ? kSphereExact : f.sweepRelativeTolerance) *
                                               std::max(1.0, std::max(std::abs(ah.distance), std::abs(bh.distance)))
                                            : 0;
            const auto   sameDistance = [&](double x, double y)
            { return exactSweep ? std::abs(x - y) <= tolerance : near(x, y, tolerance); };
            // Within a declared sweep tolerance two objects can be touched at indistinguishable
            // distances; the closest identity is then ambiguous and only the distance is compared.
            const bool ambiguousIdentity = tolerance > 0 && ah.id != bh.id && sameDistance(ah.distance, bh.distance);
            if (!ties && !ambiguousIdentity && (ah.id != bh.id || ah.sourceFace != bh.sourceFace) &&
                f.kind == QueryKind::sweep && tolerance > 0 && !sameDistance(ah.distance, bh.distance))
            {
                // Adjudicate with the analytic oracle: when the second side's earlier contact is real
                // on its reported face, whichever object or face the first side reported, the first
                // side missed it. The second side may never miss.
                const double analyticSecond = analyticFaceDistance(f, i, bh);
                const bool   secondReal     = analyticSecond >= 0 && sameDistance(analyticSecond, bh.distance);
                if (secondReal && bh.distance < ah.distance)
                {
                    validateMeshFace(f, bh, where + " second");
                    if (stats)
                        ++stats->missedByFirst;
                    std::cerr << where << " first missed object " << bh.id << " face " << bh.sourceFace << " at "
                              << bh.distance << " (reported object " << ah.id << " face " << ah.sourceFace << " at "
                              << ah.distance << ")\n";
                    continue;
                }
            }
            require(ties || ambiguousIdentity || ah.id == bh.id,
                    where + " closest identity differs: objects " + std::to_string(ah.id) + " vs " +
                        std::to_string(bh.id) + " at distances " + std::to_string(ah.distance) + " vs " +
                        std::to_string(bh.distance) + " (faces " + std::to_string(ah.sourceFace) + " vs " +
                        std::to_string(bh.sourceFace) + ")");
            if (!ambiguousIdentity)
                require(ah.geometryType == bh.geometryType && ah.geometryValue == bh.geometryValue,
                        where + " geometry identity differs");
            // Ray distances of PhysX float contacts on dense meshes deviate from the analytic value
            // by up to the declared contact tolerance; the per-side analytic check above bounds each.
            require(f.kind == QueryKind::rays ? near(ah.distance, bh.distance, declaredTolerance)
                                              : sameDistance(ah.distance, bh.distance),
                    where + " distance " + precise(ah.distance) + " vs " + precise(bh.distance) + " (objects " +
                        std::to_string(ah.id) + " vs " + std::to_string(bh.id) + ", faces " +
                        std::to_string(ah.sourceFace) + " vs " + std::to_string(bh.sourceFace) + ")");
            // Within a declared sweep tolerance two implementations may touch different triangles
            // of the same object first; the contact is then ambiguous and only its distance and
            // identity are comparable. Each contact still has to lie on the face it reports. A direct
            // PhysX sphere sweep that names a later face places its contact on that face.
            const bool physxSweep    = firstIsPhysx && f.kind == QueryKind::sweep;
            const bool ambiguousFace = (tolerance > 0 || declaredTolerance > 0 || physxSweep) &&
                                       ah.sourceFace != bh.sourceFace && ah.sourceFace != UINT64_MAX &&
                                       bh.sourceFace != UINT64_MAX;
            // A direct PhysX initial-overlap sweep leaves its contact position undefined.
            if (!contactFields || ambiguousIdentity || ambiguousFace || !ah.positionValid || !bh.positionValid)
                continue;
            const double contactTolerance = std::max(tolerance, declaredTolerance);
            // The CPU path returns PhysX's float contact, evaluated in the object's frame; on a face
            // the ray grazes it can sit farther from the double-precision contact than `near` allows,
            // 2.9e-5 m at a cosine of 0.0024 on the camera blob, and the header promises no common
            // bound between implementations. When the second contact is the analytic point of its
            // face, the first is held to the analytic distance, on-ray and face checks above.
            const bool analyticSecond = atAnalyticRayPoint(f, i, bh);
            for (size_t k = 0; k < 3; ++k)
            {
                require(analyticSecond || near(ah.position[k], bh.position[k], contactTolerance),
                        where + " contact position axis " + std::to_string(k) + " " + std::to_string(ah.position[k]) +
                            " vs " + std::to_string(bh.position[k]));
                require(near(ah.normal[k], bh.normal[k], contactTolerance),
                        where + " normal axis " + std::to_string(k) + " " + std::to_string(ah.normal[k]) + " vs " +
                            std::to_string(bh.normal[k]));
            }
        }
    }
}
void move(Fixture& f, size_t count, size_t frame)
{
    // Prescribed replay, not simulated dynamics. Absolute alternating offsets avoid drift.
    const double delta = frame % 2 ? -0.25 : 0.25;
    for (size_t i = 0; i < count; ++i)
        f.objects[i].p[0] += delta;
}
size_t changedCount(const Fixture& f, size_t percent)
{
    return percent ? std::max(size_t(1), f.objects.size() * percent / 100) : 0;
}
void printHit(const Hit& h)
{
    std::cout << "{\"id\":" << h.id << ",\"distance\":" << h.distance
              << ",\"position_valid\":" << (h.positionValid ? "true" : "false") << ",\"position\":[" << h.position[0]
              << ',' << h.position[1] << ',' << h.position[2] << "],\"normal\":[" << h.normal[0] << ',' << h.normal[1]
              << ',' << h.normal[2] << "],\"geometry_type\":" << h.geometryType
              << ",\"geometry_value\":" << h.geometryValue << '}';
}
} // namespace bench
