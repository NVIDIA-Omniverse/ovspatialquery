/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_BOX_SWEEP_MATH_H
#define OVSPATIALQUERY_BOX_SWEEP_MATH_H

#if !defined(__CUDACC__)
#include <math.h>
#endif

#if defined(__CUDACC__)
#define OVS_BOX_HD __host__ __device__
#else
#define OVS_BOX_HD
#endif

// The CPU path solvers each have one caller per evaluator; inlining them there,
// as a file-local function would be, keeps the ray and sweep hot paths call-free.
#if defined(__CUDACC__)
#define OVS_BOX_INLINE __forceinline__
#elif defined(_MSC_VER)
#define OVS_BOX_INLINE __forceinline
#else
#define OVS_BOX_INLINE inline __attribute__((always_inline))
#endif
// A rare branch of an inlined hot path stays out of line on the host, so it does not crowd
// the code that runs for every hit.
#if defined(__CUDACC__)
#define OVS_BOX_COLD inline
#elif defined(_MSC_VER)
#define OVS_BOX_COLD __declspec(noinline) inline
#else
#define OVS_BOX_COLD inline __attribute__((noinline, cold))
#endif

namespace ovspatialquery
{
namespace detail
{
namespace box_sweep
{
struct Vec3
{
    double             x, y, z;
    OVS_BOX_HD double& operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }
    OVS_BOX_HD double  operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
};
struct Box
{
    Vec3 center, axis[3], half;
};
struct Hit
{
    double distance;
    Vec3   position, normal;
    double u, v;
};
#if defined(_MSC_VER) && !defined(__CUDACC__)
// MSVC compiles fmax and fmin to C runtime calls, about 50 per triangle a box sweep tests.
// These hide them in this namespace with the runtime's exact results: a NaN operand yields
// the other, and -0 orders below +0.
inline double fmax(double a, double b)
{
    if (a < b)
        return b;
    if (b < a)
        return a;
    if (a != a)
        return b;
    if (b != b)
        return a;
    return a == 0 ? a + b : a;
}
inline double fmin(double a, double b)
{
    if (a < b)
        return a;
    if (b < a)
        return b;
    if (a != a)
        return b;
    if (b != b)
        return a;
    return a == 0 ? -(-a - b) : a;
}
#endif
OVS_BOX_HD inline Vec3   add(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
OVS_BOX_HD inline Vec3   sub(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
OVS_BOX_HD inline Vec3   mul(Vec3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
OVS_BOX_HD inline double maximumComponent(Vec3 a) { return fmax(fabs(a.x), fmax(fabs(a.y), fabs(a.z))); }
OVS_BOX_HD inline Vec3   divided(Vec3 a, double scale) { return {a.x / scale, a.y / scale, a.z / scale}; }
OVS_BOX_HD inline Vec3   scaledAxis(Vec3 a)
{
    double scale = maximumComponent(a);
    return scale > 0 ? divided(a, scale) : a;
}
OVS_BOX_HD inline double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
OVS_BOX_HD inline Vec3   cross(Vec3 a, Vec3 b)
{
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
OVS_BOX_HD inline Vec3 at(Vec3 a, Vec3 d, double t) { return {fma(d.x, t, a.x), fma(d.y, t, a.y), fma(d.z, t, a.z)}; }
OVS_BOX_HD inline Vec3 unit(Vec3 a, Vec3 fallback)
{
    a        = scaledAxis(a);
    double l = dot(a, a);
    return l > 0 ? mul(a, 1 / sqrt(l)) : fallback;
}
OVS_BOX_HD inline Vec3 localVector(const Box& b, Vec3 v)
{
    return {dot(b.axis[0], v), dot(b.axis[1], v), dot(b.axis[2], v)};
}
OVS_BOX_HD inline Vec3 localPoint(const Box& b, Vec3 v) { return localVector(b, sub(v, b.center)); }
OVS_BOX_HD inline Vec3 worldVector(const Box& b, Vec3 v)
{
    return add(add(mul(b.axis[0], v.x), mul(b.axis[1], v.y)), mul(b.axis[2], v.z));
}
OVS_BOX_HD inline Vec3 clampBox(Vec3 p, Vec3 h)
{
    return {fmax(-h.x, fmin(h.x, p.x)), fmax(-h.y, fmin(h.y, p.y)), fmax(-h.z, fmin(h.z, p.z))};
}
OVS_BOX_HD inline Vec3 vertex(const Box& b, int i)
{
    return add(b.center, worldVector(b, {(i & 1) ? b.half.x : -b.half.x, (i & 2) ? b.half.y : -b.half.y,
                                         (i & 4) ? b.half.z : -b.half.z}));
}
OVS_BOX_HD inline double radius(const Box& b, Vec3 n)
{
    return b.half.x * fabs(dot(n, b.axis[0])) + b.half.y * fabs(dot(n, b.axis[1])) + b.half.z * fabs(dot(n, b.axis[2]));
}
// Each separating axis contributes a closed time interval. Their intersection
// is the exact continuous SAT solution for fixed-orientation polytopes.
struct Interval
{
    double enter, exit;
    Vec3   normal;
};
OVS_BOX_HD inline bool axisInterval(Vec3 axis, double low, double high, double center, double extent, double velocity,
                                    Interval& s)
{
    if (dot(axis, axis) == 0)
        return true;
    if (velocity == 0)
        return center + extent >= low && center - extent <= high;
    double a = (low - extent - center) / velocity, b = (high + extent - center) / velocity;
    Vec3   n = mul(axis, velocity > 0 ? -1 : 1);
    if (a > b)
    {
        double t = a;
        a        = b;
        b        = t;
    }
    if (a > s.enter)
    {
        s.enter  = a;
        s.normal = n;
    }
    s.exit = fmin(s.exit, b);
    return s.enter <= s.exit;
}
OVS_BOX_HD inline Vec3 triangleClosest(Vec3 p, Vec3 a, Vec3 b, Vec3 c, double& u, double& v)
{
    const Vec3   rawAb = sub(b, a), rawAc = sub(c, a);
    const double scale = fmax(maximumComponent(rawAb), maximumComponent(rawAc));
    if (scale == 0)
    {
        u = v = 0;
        return a;
    }
    Vec3   ab = divided(rawAb, scale), ac = divided(rawAc, scale), ap = divided(sub(p, a), scale);
    double d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0 && d2 <= 0)
    {
        u = v = 0;
        return a;
    }
    Vec3   bp = sub(ap, ab);
    double d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0 && d4 <= d3)
    {
        u = 1;
        v = 0;
        return b;
    }
    double vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0)
    {
        u = d1 / (d1 - d3);
        v = 0;
        return at(a, rawAb, u);
    }
    Vec3   cp = sub(ap, ac);
    double d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0 && d5 <= d6)
    {
        u = 0;
        v = 1;
        return c;
    }
    double vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0)
    {
        u = 0;
        v = d2 / (d2 - d6);
        return at(a, rawAc, v);
    }
    double va = d3 * d6 - d5 * d4;
    if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0)
    {
        v = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        u = 1 - v;
        return at(b, sub(c, b), v);
    }
    double sum = va + vb + vc;
    if (sum == 0)
    {
        u = v = 0;
        return a;
    }
    u = vb / sum;
    v = vc / sum;
    return add(a, add(mul(rawAb, u), mul(rawAc, v)));
}
OVS_BOX_HD inline Vec3 closestSegmentPair(Vec3 a, Vec3 b, Vec3 c, Vec3 d, Vec3& first)
{
    Vec3         u = sub(b, a), v = sub(d, c), r = sub(a, c);
    const double scale = fmax(maximumComponent(u), fmax(maximumComponent(v), maximumComponent(r)));
    if (scale == 0)
    {
        first = a;
        return c;
    }
    u               = divided(u, scale);
    v               = divided(v, scale);
    r               = divided(r, scale);
    const double aa = dot(u, u), ee = dot(v, v), f = dot(v, r);
    double       x = 0, y = 0;
    if (aa == 0)
        y = ee == 0 ? 0 : fmax(0, fmin(1, f / ee));
    else
    {
        const double cc = dot(u, r);
        if (ee == 0)
            x = fmax(0, fmin(1, -cc / aa));
        else
        {
            const double bb = dot(u, v), den = aa * ee - bb * bb;
            if (den > 0)
                x = fmax(0, fmin(1, (bb * f - cc * ee) / den));
            y = (bb * x + f) / ee;
            if (y < 0)
            {
                y = 0;
                x = fmax(0, fmin(1, -cc / aa));
            }
            else if (y > 1)
            {
                y = 1;
                x = fmax(0, fmin(1, (bb - cc) / aa));
            }
        }
    }
    first = at(a, sub(b, a), x);
    return at(c, sub(d, c), y);
}
OVS_BOX_HD inline Vec3 triangleContact(const Box& moved, Vec3 a, Vec3 b, Vec3 c, double& u, double& v)
{
    // Clip the target triangle against the query box to get a target-surface
    // witness, including edge/edge contacts; barycentrics retain authored corners.
    Vec3 p[12], q[12];
    p[0]                   = localPoint(moved, a);
    p[1]                   = localPoint(moved, b);
    p[2]                   = localPoint(moved, c);
    const double clipScale = fmax(maximumComponent(moved.half),
                                  fmax(maximumComponent(p[0]), fmax(maximumComponent(p[1]), maximumComponent(p[2]))));
    const double clipSlack = 256.0 * 2.2204460492503131e-16 * clipScale;
    int          count     = 3;
    for (int axis = 0; axis < 3 && count; ++axis)
        for (int side = -1; side <= 1; side += 2)
        {
            int next = 0;
            for (int i = 0; i < count; ++i)
            {
                Vec3   x = p[i], y = p[(i + 1) % count];
                double dx = side * x[axis] - moved.half[axis], dy = side * y[axis] - moved.half[axis];
                // SAT time is rounded once and transformed corners a second
                // time. Keep their roundoff-sized boundary points so a grazing
                // polygon does not vanish and fall back to an unrelated witness.
                const bool insideX = dx <= clipSlack, insideY = dy <= clipSlack;
                if (insideX && next < 12)
                    q[next++] = x;
                if (insideX != insideY && next < 12)
                    q[next++] = at(x, sub(y, x), fmax(0.0, fmin(1.0, dx / (dx - dy))));
            }
            count = next;
            for (int i = 0; i < count; ++i)
                p[i] = q[i];
        }
    Vec3 witness = moved.center;
    if (count)
    {
        Vec3 sum = {0, 0, 0};
        for (int i = 0; i < count; ++i)
            sum = add(sum, p[i]);
        witness = add(moved.center, worldVector(moved, mul(sum, 1.0 / count)));
    }
    if (count == 0)
    {
        // A rounded grazing time can leave no polygon. Recover the actual
        // closest pair of boundary features rather than projecting the center,
        // which need not lie near the box/triangle contact.
        double     best        = 1.7976931348623157e308;
        const Vec3 triangle[3] = {a, b, c};
        for (int i = 0; i < 3; ++i)
        {
            Vec3   local    = localPoint(moved, triangle[i]);
            Vec3   gap      = divided(sub(local, clampBox(local, moved.half)), clipScale);
            double distance = dot(gap, gap);
            if (distance < best)
            {
                best    = distance;
                witness = triangle[i];
            }
        }
        for (int i = 0; i < 8; ++i)
        {
            Vec3   point = vertex(moved, i);
            double bu, bv;
            Vec3   closest  = triangleClosest(point, a, b, c, bu, bv);
            Vec3   gap      = divided(sub(point, closest), clipScale);
            double distance = dot(gap, gap);
            if (distance < best)
            {
                best    = distance;
                witness = closest;
            }
            for (int axis = 0; axis < 3; ++axis)
                if (!(i & (1 << axis)))
                    for (int edge = 0; edge < 3; ++edge)
                    {
                        Vec3 onBox;
                        Vec3 onTriangle = closestSegmentPair(point, vertex(moved, i | (1 << axis)), triangle[edge],
                                                             triangle[(edge + 1) % 3], onBox);
                        gap             = divided(sub(onBox, onTriangle), clipScale);
                        distance        = dot(gap, gap);
                        if (distance < best)
                        {
                            best    = distance;
                            witness = onTriangle;
                        }
                    }
        }
    }
    return triangleClosest(witness, a, b, c, u, v);
}
// The triangle's interval on a scaled separating axis against the box's extent on it.
OVS_BOX_HD inline bool boxTriangleProjection(const Box& query, Vec3 axis, double extent, double velocity, Vec3 a,
                                             Vec3 b, Vec3 c, Interval& s)
{
    double x = dot(axis, sub(a, query.center)), y = dot(axis, sub(b, query.center)),
           z = dot(axis, sub(c, query.center));
    return axisInterval(axis, fmin(x, fmin(y, z)), fmax(x, fmax(y, z)), 0, extent, velocity, s);
}
// One of boxTriangle's separating axes.
OVS_BOX_HD inline bool boxTriangleAxis(const Box& query, Vec3 axis, Vec3 a, Vec3 b, Vec3 c, Vec3 d, Interval& s)
{
    axis = scaledAxis(axis);
    return boxTriangleProjection(query, axis, radius(query, axis), dot(axis, d), a, b, c, s);
}
// The box's face axes as boxTriangleAxis scales them, with the box's extent and
// velocity on each: the same for every triangle a box is tested against.
struct BoxFaceAxes
{
    Vec3   axis[3];
    double extent[3], velocity[3];
};
OVS_BOX_HD inline BoxFaceAxes boxFaceAxes(const Box& query, Vec3 d)
{
    BoxFaceAxes faces;
    for (int i = 0; i < 3; ++i)
    {
        faces.axis[i]     = scaledAxis(query.axis[i]);
        faces.extent[i]   = radius(query, faces.axis[i]);
        faces.velocity[i] = dot(faces.axis[i], d);
    }
    return faces;
}
// boxTriangleInterval in two parts. The first is the box's three face axes, the
// second the sidedness test and the triangle's ten axes, its normal then the nine
// across a box axis and an edge, in boxTriangleInterval's order. Each axis only
// raises s.enter or lowers s.exit, so the complete entry is never earlier than the
// first part's.
OVS_BOX_HD inline bool boxTriangleBoxAxes(const Box& query, const BoxFaceAxes& faces, Vec3 a, Vec3 b, Vec3 c, Vec3 d,
                                          double minimum, double maximum, Interval& s)
{
    s = {minimum, maximum, mul(d, -1)};
    for (int i = 0; i < 3; ++i)
        if (!boxTriangleProjection(query, faces.axis[i], faces.extent[i], faces.velocity[i], a, b, c, s))
            return false;
    return true;
}
OVS_BOX_HD inline bool boxTriangleTriangleAxes(const Box& query, Vec3 a, Vec3 b, Vec3 c, bool doubleSided, Vec3 d,
                                               Interval& s)
{
    const Vec3 edges[3] = {scaledAxis(sub(b, a)), scaledAxis(sub(c, b)), scaledAxis(sub(a, c))};
    const Vec3 n        = scaledAxis(cross(edges[0], scaledAxis(sub(c, a))));
    if (dot(n, n) == 0 || (!doubleSided && !(dot(n, d) < 0)))
        return false;
    if (!boxTriangleAxis(query, n, a, b, c, d, s))
        return false;
    for (int i = 4; i < 13; ++i)
        if (!boxTriangleAxis(query, cross(query.axis[(i - 4) / 3], edges[(i - 4) % 3]), a, b, c, d, s))
            return false;
    return true;
}
// The complete 13-axis continuous SAT of boxTriangle and its sidedness test,
// without the contact witness, which a caller keeping only the closest of many
// triangles computes once from the interval. A single-sided triangle is ignored
// whole unless the motion runs against its front normal, n . d < 0 (ADR-0051, as
// PhysX's box sweeps cull n . d >= 0); every other triangle is met from either
// side at its first SAT entry.
OVS_BOX_HD inline bool boxTriangleInterval(const Box& query, Vec3 a, Vec3 b, Vec3 c, bool doubleSided, Vec3 d,
                                           double minimum, double maximum, Interval& s)
{
    return boxTriangleBoxAxes(query, boxFaceAxes(query, d), a, b, c, d, minimum, maximum, s) &&
           boxTriangleTriangleAxes(query, a, b, c, doubleSided, d, s);
}
OVS_BOX_HD inline void boxTriangleContact(const Box& query, Vec3 a, Vec3 b, Vec3 c, Vec3 d, const Interval& s, Hit& hit)
{
    Box moved    = query;
    moved.center = at(query.center, d, s.enter);
    hit.distance = s.enter;
    hit.normal   = unit(s.normal, mul(d, -1));
    hit.position = triangleContact(moved, a, b, c, hit.u, hit.v);
}
OVS_BOX_HD inline bool boxTriangle(const Box& query, Vec3 a, Vec3 b, Vec3 c, bool doubleSided, Vec3 d, double minimum,
                                   double maximum, Hit& hit)
{
    Interval s{};
    if (!boxTriangleInterval(query, a, b, c, doubleSided, d, minimum, maximum, s))
        return false;
    boxTriangleContact(query, a, b, c, d, s, hit);
    return true;
}
// The complete 15-axis continuous SAT of two boxes, without a contact witness.
OVS_BOX_HD inline bool boxBoxInterval(const Box& query, const Box& target, Vec3 d, double minimum, double maximum,
                                      Interval& s)
{
    s          = {minimum, maximum, mul(d, -1)};
    Vec3 delta = sub(query.center, target.center);
    for (int i = 0; i < 15; ++i)
    {
        Vec3   axis = i < 3 ? query.axis[i]
                            : (i < 6 ? target.axis[i - 3] : cross(query.axis[(i - 6) / 3], target.axis[(i - 6) % 3]));
        double r    = radius(target, axis);
        if (!axisInterval(axis, -r, r, dot(axis, delta), radius(query, axis), dot(axis, d), s))
            return false;
    }
    return true;
}
// A lower bound on the squared gap boxBox measures between the moved query box
// and the witness it finds on one target face, whose corners p, q and r are in
// the box's frame: the face's separation from the box along a box axis, less
// `slack`, which lies far above the rounding of the witness's local
// coordinates. Zero when the face may touch the box.
OVS_BOX_HD inline double faceGapBound(const Box& moved, Vec3 p, Vec3 q, Vec3 r, double slack)
{
    double separation = 0;
    for (int axis = 0; axis < 3; ++axis)
    {
        const double low = fmin(p[axis], fmin(q[axis], r[axis])), high = fmax(p[axis], fmax(q[axis], r[axis]));
        separation = fmax(separation, fmax(low - moved.half[axis], -moved.half[axis] - high));
    }
    separation -= slack;
    return separation > 0 ? separation * separation : 0;
}
// boxBox's contact for the interval boxBoxInterval found, which a caller keeping only
// the closest of many boxes computes once, for the winner.
OVS_BOX_HD inline void boxBoxContact(const Box& query, const Box& target, Vec3 d, const Interval& s, Hit& hit)
{
    hit.distance = s.enter;
    hit.normal   = unit(s.normal, mul(d, -1));
    hit.u = hit.v = 0;
    // A target face triangle supplies a surface witness for edge/edge as well as
    // face contacts. Admission above remains the complete 15-axis SAT test. The
    // witness is the first face, in this order, of least gap to the moved box.
    const int faces[12][3] = {{0, 2, 3}, {0, 3, 1}, {4, 5, 7}, {4, 7, 6}, {0, 1, 5}, {0, 5, 4},
                              {2, 6, 7}, {2, 7, 3}, {0, 4, 6}, {0, 6, 2}, {1, 3, 7}, {1, 7, 5}};
    Box       moved        = query;
    moved.center           = at(query.center, d, s.enter);
    double best            = 1.7976931348623157e308;
    hit.position           = target.center;
    // Faces that do not touch the box take triangleContact's costly fallback,
    // so the face of least gap bound goes first and a face whose bound exceeds
    // the best gap so far is skipped: it can neither beat nor tie it.
    const double slack    = 1e-9 * (maximumComponent(moved.center) + maximumComponent(target.center) +
                                 maximumComponent(moved.half) + maximumComponent(target.half));
    int          bestFace = 12;
    int          first    = 0;
    Vec3         corners[8];
    for (int i = 0; i < 8; ++i)
        corners[i] = localPoint(moved, vertex(target, i));
    double bounds[12];
    for (int i = 0; i < 12; ++i)
    {
        bounds[i] = faceGapBound(moved, corners[faces[i][0]], corners[faces[i][1]], corners[faces[i][2]], slack);
        if (bounds[i] < bounds[first])
            first = i;
    }
    for (int k = -1; k < 12; ++k)
    {
        const int i = k < 0 ? first : k;
        if (k == first || (k >= 0 && bounds[i] > best))
            continue;
        Vec3   a = vertex(target, faces[i][0]), b = vertex(target, faces[i][1]), c = vertex(target, faces[i][2]);
        double u, v;
        Vec3   p     = triangleContact(moved, a, b, c, u, v);
        Vec3   local = localPoint(moved, p);
        Vec3   gap   = sub(local, clampBox(local, moved.half));
        double dist  = dot(gap, gap);
        if (dist < best || (bestFace < 12 && dist == best && i < bestFace))
        {
            best         = dist;
            bestFace     = i;
            hit.position = p;
        }
    }
}
OVS_BOX_HD inline bool boxBox(const Box& query, const Box& target, Vec3 d, double minimum, double maximum, Hit& hit)
{
    Interval s{};
    if (!boxBoxInterval(query, target, d, minimum, maximum, s))
        return false;
    boxBoxContact(query, target, d, s, hit);
    return true;
}
// Solve |offset+t*velocity|<=radius over one feature's valid interval. Centering
// at the quadratic minimum avoids subtracting large squared approach distances.
OVS_BOX_HD inline bool quadraticEntry(Vec3 offset, Vec3 velocity, double r, double low, double high, double& t)
{
    Vec3 p = at(offset, velocity, low);
    if (dot(p, p) <= r * r)
    {
        t = low;
        return true;
    }
    double a = dot(velocity, velocity);
    if (a == 0)
        return false;
    double center  = -dot(offset, velocity) / a;
    Vec3   closest = at(offset, velocity, center);
    double radial  = r * r - dot(closest, closest);
    if (radial < 0)
        return false;
    double entry = center - sqrt(radial / a);
    if (entry < low || entry > high)
        return false;
    t = entry;
    return true;
}
OVS_BOX_HD inline void sort(double* a, int n)
{
    for (int i = 1; i < n; ++i)
    {
        double v = a[i];
        int    j = i;
        while (j && a[j - 1] > v)
        {
            a[j] = a[j - 1];
            --j;
        }
        a[j] = v;
    }
}
// Path and distance math of the CPU sphere, box and capsule evaluators. The box
// sweeps below keep their own entry solvers (roundedBoxRay, capsuleRay), which
// have no residual guard. Clamps keep std::max(low, std::min(high, value)) order,
// so every compiler picks the same signed zero.
OVS_BOX_HD inline double clamp(double value, double low, double high)
{
    const double upper = value < high ? value : high;
    return low < upper ? upper : low;
}
OVS_BOX_HD inline double pointBoxDistanceSquared(Vec3 p, Vec3 half, Vec3* closest)
{
    double result = 0;
    Vec3   c      = {0, 0, 0};
    for (int i = 0; i < 3; ++i)
    {
        c[i]               = clamp(p[i], -half[i], half[i]);
        const double delta = p[i] - c[i];
        result             = fma(delta, delta, result);
    }
    if (closest)
        *closest = c;
    return result;
}
OVS_BOX_HD inline double boxDistanceAt(Vec3 origin, Vec3 direction, Vec3 half, double t)
{
    return pointBoxDistanceSquared(at(origin, direction, t), half, nullptr);
}
OVS_BOX_HD inline Vec3 closestOnSegment(Vec3 a, Vec3 b, Vec3 p)
{
    const Vec3   edge          = sub(b, a);
    const double lengthSquared = dot(edge, edge);
    return lengthSquared > 0 ? at(a, edge, clamp(dot(sub(p, a), edge) / lengthSquared, 0, 1)) : a;
}
OVS_BOX_HD inline double segmentDistanceSquared(Vec3 a, Vec3 b, Vec3 p)
{
    const Vec3 gap = sub(p, closestOnSegment(a, b, p));
    return dot(gap, gap);
}
// `v` without its part along `segment`, whose squared length is `lengthSquared`.
OVS_BOX_HD inline Vec3 acrossSegment(Vec3 v, Vec3 segment, double lengthSquared)
{
    return lengthSquared > 0 ? sub(v, mul(segment, dot(v, segment) / lengthSquared)) : v;
}
// The normal of a capsule contact at a center on the segment itself, which has no nearest
// direction: the motion reversed, kept from pointing into the segment, and then any direction
// across the segment. `s` is the center's clamped parameter along `segment`.
OVS_BOX_HD OVS_BOX_COLD Vec3 capsuleOnSegmentNormal(Vec3 segment, double lengthSquared, double s, bool span,
                                                    Vec3 direction)
{
    Vec3         normal = mul(direction, -1);
    const double along  = dot(normal, segment);
    if (span || (s == 0 && along > 0) || (s == 1 && along < 0))
        normal = acrossSegment(normal, segment, lengthSquared);
    if (maximumComponent(normal) == 0)
    {
        const Vec3 least = fabs(segment.x) <= fabs(segment.y) && fabs(segment.x) <= fabs(segment.z) ? Vec3{1, 0, 0}
                           : fabs(segment.y) <= fabs(segment.z)                                     ? Vec3{0, 1, 0}
                                                                                                    : Vec3{0, 0, 1};
        normal           = lengthSquared > 0 ? cross(segment, least) : least;
    }
    return normal;
}
// The contact of a capsule path at `center`, which touches or is inside the capsule: the
// surface point nearest the center (ADR-0018) is `closest` + radius * the returned unit normal.
// Within the segment's span only the part across the segment counts, so a center on the
// segment, whose foot rounds a little along it, still leaves sideways. Every capsule hit runs
// this; on the CPU a call to it cost more than its arithmetic, so it is inlined.
OVS_BOX_HD OVS_BOX_INLINE Vec3 capsuleContactNormal(Vec3 a, Vec3 b, Vec3 center, Vec3 direction, Vec3& closest)
{
    const Vec3   segment       = sub(b, a);
    const double lengthSquared = dot(segment, segment);
    const double s             = lengthSquared > 0 ? clamp(dot(sub(center, a), segment) / lengthSquared, 0, 1) : 0;
    const bool   span          = s > 0 && s < 1;
    closest                    = at(a, segment, s);
    Vec3 normal                = sub(center, closest);
    if (span)
        normal = acrossSegment(normal, segment, lengthSquared);
    if (maximumComponent(normal) == 0)
        normal = capsuleOnSegmentNormal(segment, lengthSquared, s, span, direction);
    return unit(normal, normal);
}
// Entry of |offset + t velocity| <= radius within [left, right], where the path
// keeps one closest feature. The root is solved around the path's closest approach
// with the direction's own length: the textbook discriminant loses eps L^2.
OVS_BOX_HD inline bool featureEntry(Vec3 offset, Vec3 velocity, double quadratic, double linear, double radiusSquared,
                                    double left, double right, double& t)
{
    t = left;
    if (quadratic == 0)
        return dot(offset, offset) <= radiusSquared;
    const double closest = -linear / quadratic;
    const Vec3   gap     = at(offset, velocity, closest);
    const double radial  = radiusSquared - dot(gap, gap);
    if (!(radial >= 0))
        return false;
    const double half = sqrt(radial / quadratic);
    t                 = closest - half;
    // A root rounded just behind the interval start, with the path still inside
    // there, is a contact at the start.
    if (t < left && closest + half >= left)
        t = left;
    return t >= left && t <= right;
}
// A residual guard rejects roots created by cancellation. A genuine contact's
// distance is evaluated from coordinates of magnitude up to scale (the contact
// point, the origin and the shape), so its roundoff is relative to them, not to
// the radius: a radius-relative bound rejects real contacts of small spheres far
// from the origin. Callers compute scale only for a candidate root.
OVS_BOX_HD inline bool touches(double distanceSquared, double radius, double scale)
{
    const double reach = radius + 64.0 * 2.2204460492503131e-16 * scale;
    return distanceSquared <= reach * reach;
}
// First distance in [minimum, maximum] at which origin + t direction comes within
// radius of the box [-half, half]; the path keeps one closest feature between face
// crossings.
OVS_BOX_HD OVS_BOX_INLINE bool roundedBoxPath(Vec3 origin, Vec3 direction, Vec3 half, double radius, double minimum,
                                              double maximum, double& t)
{
    const double rr = radius * radius;
    t               = minimum;
    if (boxDistanceAt(origin, direction, half, minimum) <= rr)
        return true;
    double cuts[8] = {minimum, maximum};
    int    count   = 2;
    for (int i = 0; i < 3; ++i)
        for (int side = -1; side <= 1 && direction[i] != 0; side += 2)
        {
            const double cut = (side * half[i] - origin[i]) / direction[i];
            if (cut > minimum && cut < maximum)
                cuts[count++] = cut;
        }
    sort(cuts, count);
    for (int i = 0; i + 1 < count; ++i)
    {
        const double left = cuts[i], mid = left * 0.5 + cuts[i + 1] * 0.5;
        t = left;
        if (boxDistanceAt(origin, direction, half, left) <= rr)
            return true;
        double quadratic = 0, linear = 0;
        Vec3   offset = {0, 0, 0}, velocity = {0, 0, 0};
        for (int j = 0; j < 3; ++j)
        {
            const double coordinate = fma(direction[j], mid, origin[j]);
            if (!(coordinate < -half[j]) && !(coordinate > half[j]))
                continue;
            offset[j]   = coordinate < -half[j] ? origin[j] + half[j] : origin[j] - half[j];
            velocity[j] = direction[j];
            quadratic   = fma(direction[j], direction[j], quadratic);
            linear      = fma(direction[j], offset[j], linear);
        }
        if (featureEntry(offset, velocity, quadratic, linear, rr, left, cuts[i + 1], t) &&
            (quadratic == 0 || touches(boxDistanceAt(origin, direction, half, t), radius,
                                       fmax(maximumComponent(at(origin, direction, t)),
                                            fmax(maximumComponent(origin), maximumComponent(half))))))
            return true;
    }
    t = maximum;
    return boxDistanceAt(origin, direction, half, maximum) <= rr;
}
// The same first contact with the capsule of radius around segment a-b; the path
// keeps one closest feature where its projection on the segment stays below 0,
// within [0, 1] or above 1.
OVS_BOX_HD OVS_BOX_INLINE bool capsulePath(Vec3 a, Vec3 b, Vec3 origin, Vec3 direction, double radius, double minimum,
                                           double maximum, double& t)
{
    const Vec3   segment = sub(b, a), originOffset = sub(origin, a);
    const double lengthSquared = dot(segment, segment), rr = radius * radius;
    const double projectionOrigin = dot(originOffset, segment), projectionDirection = dot(direction, segment);
    t = minimum;
    if (segmentDistanceSquared(a, b, at(origin, direction, minimum)) <= rr)
        return true;
    double cuts[4] = {minimum, maximum};
    int    count   = 2;
    for (int i = 0; i < 2 && projectionDirection != 0; ++i)
    {
        const double cut = ((i ? lengthSquared : 0.0) - projectionOrigin) / projectionDirection;
        if (cut > minimum && cut < maximum)
            cuts[count++] = cut;
    }
    sort(cuts, count);
    for (int i = 0; i + 1 < count; ++i)
    {
        const double left = cuts[i], mid = left * 0.5 + cuts[i + 1] * 0.5;
        t = left;
        if (segmentDistanceSquared(a, b, at(origin, direction, left)) <= rr)
            return true;
        const double projection = fma(projectionDirection, mid, projectionOrigin);
        Vec3         offset = originOffset, velocity = direction;
        if (!(projection <= 0) && projection >= lengthSquared)
            offset = sub(origin, b);
        else if (!(projection <= 0))
        {
            // Within the segment's span only the part across the segment counts.
            const Vec3 back = mul(segment, -1);
            offset          = at(originOffset, back, projectionOrigin / lengthSquared);
            velocity        = at(direction, back, projectionDirection / lengthSquared);
        }
        const double quadratic = dot(velocity, velocity);
        if (featureEntry(offset, velocity, quadratic, dot(offset, velocity), rr, left, cuts[i + 1], t) &&
            (quadratic == 0 || touches(segmentDistanceSquared(a, b, at(origin, direction, t)), radius,
                                       fmax(fmax(maximumComponent(at(origin, direction, t)), maximumComponent(origin)),
                                            fmax(maximumComponent(a), maximumComponent(b))))))
            return true;
    }
    t = maximum;
    return segmentDistanceSquared(a, b, at(origin, direction, maximum)) <= rr;
}
// Squared distance between segment a-b and the box [-half, half], minimized
// piecewise between the parameters where the segment crosses a face plane.
OVS_BOX_HD OVS_BOX_INLINE double segmentBoxDistanceSquared(Vec3 a, Vec3 b, Vec3 half)
{
    const Vec3 direction = sub(b, a);
    double     cuts[8]   = {0, 1};
    int        count     = 2;
    for (int i = 0; i < 3; ++i)
        for (int side = -1; side <= 1 && direction[i] != 0; side += 2)
        {
            const double cut = (side * half[i] - a[i]) / direction[i];
            if (cut > 0 && cut < 1)
                cuts[count++] = cut;
        }
    sort(cuts, count);
    const double first = boxDistanceAt(a, direction, half, 0), last = boxDistanceAt(a, direction, half, 1);
    double       best = last < first ? last : first;
    for (int i = 0; i + 1 < count; ++i)
    {
        const double left = cuts[i], right = cuts[i + 1], mid = left * 0.5 + right * 0.5;
        double       quadratic = 0, linear = 0;
        for (int j = 0; j < 3; ++j)
        {
            const double coordinate = fma(direction[j], mid, a[j]);
            if (!(coordinate < -half[j]) && !(coordinate > half[j]))
                continue;
            quadratic = fma(direction[j], direction[j], quadratic);
            linear    = fma(direction[j], coordinate < -half[j] ? a[j] + half[j] : a[j] - half[j], linear);
        }
        const double distance =
            boxDistanceAt(a, direction, half, quadratic == 0 ? left : clamp(-linear / quadratic, left, right));
        best = distance < best ? distance : best;
    }
    return best;
}
OVS_BOX_HD inline bool roundedBoxRay(Vec3 p, Vec3 d, Vec3 half, double r, double minimum, double maximum, double& t)
{
    double cuts[8] = {minimum, maximum};
    int    count   = 2;
    for (int i = 0; i < 3; ++i)
        if (d[i] != 0)
            for (int side = -1; side <= 1; side += 2)
            {
                double a = (side * half[i] - p[i]) / d[i];
                if (a > minimum && a < maximum)
                    cuts[count++] = a;
            }
    sort(cuts, count);
    for (int i = 0; i < count - 1; ++i)
    {
        double mid = cuts[i] + (cuts[i + 1] - cuts[i]) * 0.5;
        Vec3   q = at(p, d, mid), o = {0, 0, 0}, v = {0, 0, 0};
        for (int j = 0; j < 3; ++j)
            if (q[j] < -half[j] || q[j] > half[j])
            {
                o[j] = p[j] - (q[j] < 0 ? -half[j] : half[j]);
                v[j] = d[j];
            }
        if (quadraticEntry(o, v, r, cuts[i], cuts[i + 1], t))
            return true;
    }
    Vec3 end = at(p, d, maximum), gap = sub(end, clampBox(end, half));
    if (dot(gap, gap) <= r * r)
    {
        t = maximum;
        return true;
    }
    return false;
}
OVS_BOX_HD inline bool boxSphere(const Box& query, Vec3 center, double r, Vec3 d, double minimum, double maximum,
                                 Hit& hit)
{
    Vec3   p = localPoint(query, center), v = localVector(query, mul(d, -1));
    double t;
    if (!roundedBoxRay(p, v, query.half, r, minimum, maximum, t))
        return false;
    Vec3 local = at(p, v, t), closest = clampBox(local, query.half);
    Vec3 normal = unit(worldVector(query, sub(closest, local)), mul(d, -1));
    hit         = {t, at(center, normal, r), normal, 0, 0};
    return true;
}
OVS_BOX_HD inline Vec3 segmentClosest(Vec3 a, Vec3 b, Vec3 p)
{
    Vec3   edge = sub(b, a);
    double den  = dot(edge, edge);
    return at(a, edge, den > 0 ? fmax(0, fmin(1, dot(sub(p, a), edge) / den)) : 0);
}
OVS_BOX_HD inline bool capsuleRay(Vec3 p, Vec3 d, Vec3 a, Vec3 b, double r, double minimum, double maximum, double& t)
{
    Vec3   edge = sub(b, a), o = sub(p, a);
    double den = dot(edge, edge);
    if (den == 0)
        return quadraticEntry(o, d, r, minimum, maximum, t);
    double start = dot(o, edge) / den, speed = dot(d, edge) / den, cuts[4] = {minimum, maximum};
    int    count = 2;
    if (speed != 0)
        for (int i = 0; i < 2; ++i)
        {
            double x = (i - start) / speed;
            if (x > minimum && x < maximum)
                cuts[count++] = x;
        }
    sort(cuts, count);
    for (int i = 0; i < count - 1; ++i)
    {
        double mid = cuts[i] + (cuts[i + 1] - cuts[i]) * 0.5, along = start + mid * speed;
        Vec3   q = o, v = d;
        if (along > 1)
            q = sub(p, b);
        else if (along >= 0)
        {
            q = sub(o, mul(edge, start));
            v = sub(d, mul(edge, speed));
        }
        if (quadraticEntry(q, v, r, cuts[i], cuts[i + 1], t))
            return true;
    }
    return false;
}
OVS_BOX_HD inline double segmentBoxDistance(Vec3 a, Vec3 b, Vec3 half, Vec3& segmentPoint, Vec3& boxPoint)
{
    Vec3   d       = sub(b, a);
    double cuts[8] = {0, 1};
    int    n       = 2;
    for (int i = 0; i < 3; ++i)
        if (d[i] != 0)
            for (int side = -1; side <= 1; side += 2)
            {
                double t = (side * half[i] - a[i]) / d[i];
                if (t > 0 && t < 1)
                    cuts[n++] = t;
            }
    sort(cuts, n);
    double best = 1.7976931348623157e308;
    for (int i = 0; i < n - 1; ++i)
    {
        double mid = (cuts[i] + cuts[i + 1]) * 0.5;
        Vec3   q = at(a, d, mid), o = {0, 0, 0}, v = {0, 0, 0};
        for (int j = 0; j < 3; ++j)
            if (q[j] < -half[j] || q[j] > half[j])
            {
                o[j] = a[j] - (q[j] < 0 ? -half[j] : half[j]);
                v[j] = d[j];
            }
        double den = dot(v, v), t = den > 0 ? fmax(cuts[i], fmin(cuts[i + 1], -dot(o, v) / den)) : cuts[i];
        q        = at(a, d, t);
        Vec3   c = clampBox(q, half), gap = sub(q, c);
        double dist = dot(gap, gap);
        if (dist < best)
        {
            best         = dist;
            segmentPoint = q;
            boxPoint     = c;
        }
    }
    return best;
}
OVS_BOX_HD inline bool boxCapsule(const Box& query, Vec3 a, Vec3 b, double r, Vec3 d, double minimum, double maximum,
                                  Hit& hit)
{
    const Vec3 translation   = a;
    Box        relativeQuery = query;
    relativeQuery.center     = sub(query.center, translation);
    b                        = sub(b, translation);
    a                        = {0, 0, 0};
    Box moved                = relativeQuery;
    moved.center             = at(relativeQuery.center, d, minimum);
    Vec3 segmentPoint, boxPoint;
    if (segmentBoxDistance(localPoint(moved, a), localPoint(moved, b), relativeQuery.half, segmentPoint, boxPoint) <=
        r * r)
    {
        Vec3 normal  = unit(worldVector(relativeQuery, sub(boxPoint, segmentPoint)), mul(d, -1));
        Vec3 center  = add(moved.center, worldVector(relativeQuery, segmentPoint));
        hit          = {minimum, at(center, normal, r), normal, 0, 0};
        hit.position = add(hit.position, translation);
        return true;
    }
    bool   found = false;
    Hit    candidate{};
    double best = maximum;
    for (int i = 0; i < 2; ++i)
        if (boxSphere(relativeQuery, i ? b : a, r, d, minimum, best, candidate))
        {
            hit   = candidate;
            best  = hit.distance;
            found = true;
        }
    for (int i = 0; i < 8; ++i)
    {
        Vec3   p = vertex(relativeQuery, i);
        double t;
        if (capsuleRay(p, d, a, b, r, minimum, best, t))
        {
            Vec3 point = at(p, d, t), on = segmentClosest(a, b, point), normal = unit(sub(point, on), mul(d, -1));
            hit   = {t, at(on, normal, r), normal, 0, 0};
            best  = t;
            found = true;
        }
    }
    // The remaining closest-feature case is two segment interiors. Their fixed
    // directions make signed line separation linear in time; test both roots.
    Vec3   targetEdge = sub(b, a);
    double ee         = dot(targetEdge, targetEdge);
    for (int axis = 0; axis < 3; ++axis)
        for (int bits = 0; bits < 8; ++bits)
            if (!(bits & (1 << axis)))
            {
                Vec3 p = vertex(relativeQuery, bits), edge = sub(vertex(relativeQuery, bits | (1 << axis)), p),
                     normal = cross(edge, targetEdge);
                double nn = dot(normal, normal), dd = dot(edge, edge), de = dot(edge, targetEdge);
                if (nn == 0 || ee == 0)
                    continue;
                double speed = dot(normal, d);
                if (speed == 0)
                    continue;
                double offset = dot(normal, sub(p, a)), reach = r * sqrt(nn);
                for (int sign = -1; sign <= 1; sign += 2)
                {
                    double t = (sign * reach - offset) / speed;
                    if (t < minimum || t > best)
                        continue;
                    Vec3   delta = sub(at(p, d, t), a);
                    double w = dot(delta, edge), z = dot(delta, targetEdge), den = dd * ee - de * de;
                    if (den <= 0)
                        continue;
                    double s = (de * z - ee * w) / den, u = (dd * z - de * w) / den;
                    if (s < 0 || s > 1 || u < 0 || u > 1)
                        continue;
                    Vec3 x = at(at(p, d, t), edge, s), y = at(a, targetEdge, u), n = unit(sub(x, y), mul(d, -1));
                    hit   = {t, at(y, n, r), n, 0, 0};
                    best  = t;
                    found = true;
                }
            }
    if (found)
        hit.position = add(hit.position, translation);
    return found;
}
} // namespace box_sweep
} // namespace detail
} // namespace ovspatialquery
#undef OVS_BOX_HD
#undef OVS_BOX_INLINE
#undef OVS_BOX_COLD
#endif
