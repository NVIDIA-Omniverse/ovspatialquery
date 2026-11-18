/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_box_sweep_math.h"
#include "ovspatialquery_cuda_family_plan_types.h"

namespace
{

namespace cudaFamily = ovspatialquery::detail::cuda_family;
namespace boxSweep   = ovspatialquery::detail::box_sweep;

// The input domain of ovspatialquery.h (ADR-0032): a row value outside
// +/-2^60 caller units, NaN included, makes the row INVALID_ARGUMENT.
__device__ bool cudaFamilyInDomain(double value) { return fabs(value) <= 1152921504606846976.0; }

__device__ double dotVector(const double left[3], const double right[3])
{
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

// Normalises an xyzw quaternion and writes the row-major rotation axes, where
// axes[local * 3u + world] is the world component of the local axis.
template <typename Source>
__device__ bool cudaFamilyQuaternionAxes(const Source* rotation, double axes[9])
{
    double quaternion[4] = {};
    double lengthSquared = 0.0;
    for (unsigned int component = 0u; component < 4u; ++component)
    {
        quaternion[component] = static_cast<double>(rotation[component]);
        if (!isfinite(quaternion[component]))
        {
            return false;
        }
        lengthSquared = fma(quaternion[component], quaternion[component], lengthSquared);
    }
    if (!isfinite(lengthSquared) || fabs(lengthSquared - 1.0) > 1.0e-5)
    {
        return false;
    }
    const double inverseLength = 1.0 / sqrt(lengthSquared);
    for (unsigned int component = 0u; component < 4u; ++component)
    {
        quaternion[component] *= inverseLength;
    }
    const double x = quaternion[0];
    const double y = quaternion[1];
    const double z = quaternion[2];
    const double w = quaternion[3];
    axes[0]        = 1.0 - 2.0 * (y * y + z * z);
    axes[1]        = 2.0 * (x * y + z * w);
    axes[2]        = 2.0 * (x * z - y * w);
    axes[3]        = 2.0 * (x * y - z * w);
    axes[4]        = 1.0 - 2.0 * (x * x + z * z);
    axes[5]        = 2.0 * (y * z + x * w);
    axes[6]        = 2.0 * (x * z + y * w);
    axes[7]        = 2.0 * (y * z - x * w);
    axes[8]        = 1.0 - 2.0 * (x * x + y * y);
    return true;
}

__device__ bool validPath(const double* origin, const float* direction, double minimum_distance,
                          double maximum_distance, double radius)
{
    double length_squared = 0.0;
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        if (!cudaFamilyInDomain(origin[axis]) || !isfinite(direction[axis]))
        {
            return false;
        }
        length_squared += static_cast<double>(direction[axis]) * direction[axis];
    }
    return fabs(length_squared - 1.0) <= 1.0e-5 && cudaFamilyInDomain(minimum_distance) &&
           cudaFamilyInDomain(maximum_distance) && minimum_distance >= 0.0 && maximum_distance >= minimum_distance &&
           isfinite(radius) && radius >= 0.0;
}

__device__ bool pathAabbHit(const double* origin, const float* direction, double minimum_distance,
                            double maximum_distance, double radius, const double* object_minimum,
                            const double* object_maximum, double& distance, double position[3], float normal[3])
{
    double entry          = minimum_distance;
    double exit           = maximum_distance;
    int    entry_axis     = -1;
    double entry_boundary = 0.0;
    float  entry_normal   = 0.0f;
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        const double minimum   = object_minimum[axis] - radius;
        const double maximum   = object_maximum[axis] + radius;
        const double component = static_cast<double>(direction[axis]);
        if (component == 0.0)
        {
            if (origin[axis] < minimum || origin[axis] > maximum)
            {
                return false;
            }
            continue;
        }
        double near_distance = (minimum - origin[axis]) / component;
        double far_distance  = (maximum - origin[axis]) / component;
        double near_boundary = minimum;
        float  near_normal   = -1.0f;
        if (component < 0.0) // by sign: a zero-thickness slab has equal distances
        {
            const double temporary = near_distance;
            near_distance          = far_distance;
            far_distance           = temporary;
            near_boundary          = maximum;
            near_normal            = 1.0f;
        }
        if (near_distance > entry || (near_distance == entry && entry_axis < 0))
        {
            entry          = near_distance;
            entry_axis     = static_cast<int>(axis);
            entry_boundary = near_boundary;
            entry_normal   = near_normal;
        }
        exit = fmin(exit, far_distance);
        if (entry > exit)
        {
            return false;
        }
    }
    if (!isfinite(entry) || entry < minimum_distance || entry > maximum_distance)
    {
        return false;
    }
    distance = entry;
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        position[axis] = fma(static_cast<double>(direction[axis]), entry, origin[axis]);
        normal[axis]   = 0.0f;
    }
    if (entry_axis >= 0)
    {
        position[entry_axis] = entry_boundary;
        normal[entry_axis]   = entry_normal;
    }
    if (entry_axis < 0 || (radius > 0.0 && entry == 0.0))
    {
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            normal[axis] = -direction[axis];
        }
    }
    return true;
}

__device__ bool aabbsOverlap(const double* query_minimum, const double* query_maximum, const double* object_minimum,
                             const double* object_maximum)
{
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        if (query_maximum[axis] < object_minimum[axis] || query_minimum[axis] > object_maximum[axis])
        {
            return false;
        }
    }
    return true;
}

__device__ bool sphereOverlapsAabb(const double* center, double radius, const double* object_minimum,
                                   const double* object_maximum)
{
    if (!isfinite(radius) || radius <= 0.0)
    {
        return false;
    }
    double normalized_squared = 0.0;
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        if (!isfinite(center[axis]))
        {
            return false;
        }
        double distance = 0.0;
        if (center[axis] < object_minimum[axis])
        {
            distance = object_minimum[axis] - center[axis];
        }
        else if (center[axis] > object_maximum[axis])
        {
            distance = center[axis] - object_maximum[axis];
        }
        if (distance > radius)
        {
            return false;
        }
        const double normalized = distance / radius;
        normalized_squared += normalized * normalized;
        if (normalized_squared > 1.0)
        {
            return false;
        }
    }
    return true;
}

constexpr double kDoubleEpsilon = 2.2204460492503131e-16;
// The barycentric margin a ray-triangle test accepts outside each edge: the geomEpsilon
// (2^-22) the CPU cooks its PhysX meshes with. Without one, a ray through an edge two
// triangles share rounds outside both of them a quarter of the time and passes through.
constexpr double kTriangleEdgeMargin = 2.384185791015625e-07;

// `slack` widens the separation boundary, in the scaled units of boxOverlapsAabb.
__device__ bool separatedOnAxis(const double axis[3], const double center_delta[3], const double aabb_extents[3],
                                const double box_extents[3], const double box_axes[9], double slack)
{
    if (dotVector(axis, axis) < 1.0e-30)
    {
        return false;
    }
    const double center_distance = fabs(dotVector(center_delta, axis));
    double       aabb_radius     = 0.0;
    double       box_radius      = 0.0;
    for (unsigned int world = 0u; world < 3u; ++world)
    {
        aabb_radius += aabb_extents[world] * fabs(axis[world]);
    }
    for (unsigned int local = 0u; local < 3u; ++local)
    {
        box_radius += box_extents[local] * fabs(dotVector(box_axes + local * 3u, axis));
    }
    return center_distance > aabb_radius + box_radius + slack;
}

// A BOX query against an object's bounds, the bounds-precision result. As the
// exact test's candidate gate (`exact`) it must not reject a pair the exact test
// accepts: rounding the scaled terms can put an exactly touching pair one ulp
// apart, so the gate separates only beyond a few ulps of the unit scale.
__device__ bool boxOverlapsAabb(const double* center, const double* quaternion_source, const double* box_extents_source,
                                const double* object_minimum, const double* object_maximum, bool exact)
{
    double axes[9] = {};
    if (!cudaFamilyQuaternionAxes(quaternion_source, axes))
    {
        return false;
    }
    double broad_minimum[3] = {};
    double broad_maximum[3] = {};
    for (unsigned int world = 0u; world < 3u; ++world)
    {
        if (!isfinite(center[world]) || !isfinite(box_extents_source[world]) || box_extents_source[world] <= 0.0)
        {
            return false;
        }
        double world_extent = 0.0;
        for (unsigned int local = 0u; local < 3u; ++local)
        {
            world_extent += fabs(axes[local * 3u + world]) * box_extents_source[local];
        }
        broad_minimum[world] = center[world] - world_extent;
        broad_maximum[world] = center[world] + world_extent;
    }
    double clipped_minimum[3] = {};
    double clipped_maximum[3] = {};
    double aabb_center[3]     = {};
    double aabb_extents[3]    = {};
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        clipped_minimum[axis] = fmax(object_minimum[axis], broad_minimum[axis]);
        clipped_maximum[axis] = fmin(object_maximum[axis], broad_maximum[axis]);
        if (clipped_minimum[axis] > clipped_maximum[axis])
        {
            return false;
        }
        aabb_center[axis]  = clipped_minimum[axis] * 0.5 + clipped_maximum[axis] * 0.5;
        aabb_extents[axis] = clipped_maximum[axis] * 0.5 - clipped_minimum[axis] * 0.5;
    }
    double scale = 1.0;
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        scale = fmax(scale, fabs(aabb_center[axis]));
        scale = fmax(scale, fabs(center[axis]));
        scale = fmax(scale, aabb_extents[axis]);
        scale = fmax(scale, box_extents_source[axis]);
    }
    double center_delta[3]        = {};
    double scaled_aabb_extents[3] = {};
    double scaled_box_extents[3]  = {};
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        center_delta[axis]        = center[axis] / scale - aabb_center[axis] / scale;
        scaled_aabb_extents[axis] = aabb_extents[axis] / scale;
        scaled_box_extents[axis]  = box_extents_source[axis] / scale;
    }
    const double slack            = exact ? 64.0 * kDoubleEpsilon : 0.0;
    const double world_axes[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    for (unsigned int world = 0u; world < 3u; ++world)
    {
        if (separatedOnAxis(world_axes[world], center_delta, scaled_aabb_extents, scaled_box_extents, axes, slack))
        {
            return false;
        }
    }
    for (unsigned int local = 0u; local < 3u; ++local)
    {
        const double* const box_axis = axes + local * 3u;
        if (separatedOnAxis(box_axis, center_delta, scaled_aabb_extents, scaled_box_extents, axes, slack))
        {
            return false;
        }
        for (unsigned int world = 0u; world < 3u; ++world)
        {
            const double cross_axis[3] = {world_axes[world][1] * box_axis[2] - world_axes[world][2] * box_axis[1],
                                          world_axes[world][2] * box_axis[0] - world_axes[world][0] * box_axis[2],
                                          world_axes[world][0] * box_axis[1] - world_axes[world][1] * box_axis[0]};
            if (separatedOnAxis(cross_axis, center_delta, scaled_aabb_extents, scaled_box_extents, axes, slack))
            {
                return false;
            }
        }
    }
    return true;
}

struct TriangleHit
{
    double             distance;
    double             position[3];
    float              normal[3];
    unsigned long long face;
    float              uv[2];
};

// World triangle corners; with Relative the placement translation is left out:
// the box sweep subtracts it from its query once, so the mesh's small linear
// features stay in the same relative frame until contact.
template <bool Relative = false>
__device__ void cudaFamilyLoadMeshTriangle(const cudaFamily::MeshGeometry&   mesh,
                                           const cudaFamily::MeshDefinition& definition, unsigned long long triangle,
                                           double output[3][3])
{
    const float*        vertices = reinterpret_cast<const float*>(definition.vertices);
    const unsigned int* indices  = reinterpret_cast<const unsigned int*>(definition.triangles);
    for (unsigned int corner = 0u; corner < 3u; ++corner)
    {
        const unsigned int vertex = indices[triangle * 3u + corner];
        const float*       local  = vertices + vertex * 3u;
        for (unsigned int world = 0u; world < 3u; ++world)
        {
            const double* row = mesh.worldFromLocal + world * 4u;
            const double  z   = static_cast<double>(local[2]);
            output[corner][world] =
                fma(row[0], static_cast<double>(local[0]),
                    fma(row[1], static_cast<double>(local[1]), Relative ? row[2] * z : fma(row[2], z, row[3])));
        }
    }
}

// The reciprocal of the power of two that maps `value` (finite, positive) into [1, 2).
// Scaling a computation's inputs by it is exact, so the ratios of the scaled
// Moller-Trumbore below are bitwise those of the unscaled one and stay in range,
// while the divisions a general scale needs would round. The double pipe of consumer
// and workstation GPUs runs at 1/64 of the float rate and a division is a dozen of its
// instructions, so the normalizing divisions cost more than the rest of the test.
__device__ double inversePowerOfTwo(double value)
{
    const long long exponent = (__double_as_longlong(value) >> 52) & 0x7ffll;
    const long long inverse  = 2046ll - exponent;
    return __longlong_as_double((inverse < 1ll ? 1ll : inverse) << 52);
}

__device__ bool triangleNormal(const double triangle[3][3], double normal[3])
{
    double first[3]   = {};
    double second[3]  = {};
    double edge_scale = 0.0;
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        first[axis]  = triangle[1][axis] - triangle[0][axis];
        second[axis] = triangle[2][axis] - triangle[0][axis];
        edge_scale   = fmax(edge_scale, fabs(first[axis]));
        edge_scale   = fmax(edge_scale, fabs(second[axis]));
    }
    if (!isfinite(edge_scale) || edge_scale == 0.0)
    {
        return false;
    }
    const double inverse_edge_scale = inversePowerOfTwo(edge_scale);
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        first[axis] *= inverse_edge_scale;
        second[axis] *= inverse_edge_scale;
    }
    normal[0]           = first[1] * second[2] - first[2] * second[1];
    normal[1]           = first[2] * second[0] - first[0] * second[2];
    normal[2]           = first[0] * second[1] - first[1] * second[0];
    const double length = sqrt(dotVector(normal, normal));
    if (!isfinite(length) || length == 0.0)
    {
        return false;
    }
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        normal[axis] /= length;
    }
    return true;
}

__device__ void closestPointOnTriangle(const double triangle[3][3], const double point[3], double closest[3],
                                       double& out_u, double& out_v)
{
    double first[3]      = {};
    double second[3]     = {};
    double from_first[3] = {};
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        first[axis]      = triangle[1][axis] - triangle[0][axis];
        second[axis]     = triangle[2][axis] - triangle[0][axis];
        from_first[axis] = point[axis] - triangle[0][axis];
    }
    const double first_projection  = dotVector(first, from_first);
    const double second_projection = dotVector(second, from_first);
    if (first_projection <= 0.0 && second_projection <= 0.0)
    {
        out_u = 0.0;
        out_v = 0.0;
    }
    else
    {
        double from_second[3] = {};
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            from_second[axis] = point[axis] - triangle[1][axis];
        }
        const double third_projection  = dotVector(first, from_second);
        const double fourth_projection = dotVector(second, from_second);
        if (third_projection >= 0.0 && fourth_projection <= third_projection)
        {
            out_u = 1.0;
            out_v = 0.0;
        }
        else
        {
            const double first_edge_region =
                first_projection * fourth_projection - third_projection * second_projection;
            if (first_edge_region <= 0.0 && first_projection >= 0.0 && third_projection <= 0.0)
            {
                out_u = first_projection / (first_projection - third_projection);
                out_v = 0.0;
            }
            else
            {
                double from_third[3] = {};
                for (unsigned int axis = 0u; axis < 3u; ++axis)
                {
                    from_third[axis] = point[axis] - triangle[2][axis];
                }
                const double fifth_projection = dotVector(first, from_third);
                const double sixth_projection = dotVector(second, from_third);
                if (sixth_projection >= 0.0 && fifth_projection <= sixth_projection)
                {
                    out_u = 0.0;
                    out_v = 1.0;
                }
                else
                {
                    const double second_edge_region =
                        fifth_projection * second_projection - first_projection * sixth_projection;
                    if (second_edge_region <= 0.0 && second_projection >= 0.0 && sixth_projection <= 0.0)
                    {
                        out_u = 0.0;
                        out_v = second_projection / (second_projection - sixth_projection);
                    }
                    else
                    {
                        const double third_edge_region =
                            third_projection * sixth_projection - fifth_projection * fourth_projection;
                        if (third_edge_region <= 0.0 && fourth_projection - third_projection >= 0.0 &&
                            fifth_projection - sixth_projection >= 0.0)
                        {
                            out_v = (fourth_projection - third_projection) /
                                    ((fourth_projection - third_projection) + (fifth_projection - sixth_projection));
                            out_u = 1.0 - out_v;
                        }
                        else
                        {
                            const double inverse = 1.0 / (first_edge_region + second_edge_region + third_edge_region);
                            out_u                = second_edge_region * inverse;
                            out_v                = first_edge_region * inverse;
                        }
                    }
                }
            }
        }
    }
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        closest[axis] = fma(out_v, second[axis], fma(out_u, first[axis], triangle[0][axis]));
    }
}

__device__ bool rayTriangle(const double origin[3], const float direction_source[3], double minimum_distance,
                            double maximum_distance, const double triangle[3][3], bool double_sided,
                            bool flip_back_faces, TriangleHit& output)
{
    double normalized[3][3] = {};
    double center[3]        = {};
    double scale            = 0.0;
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        double minimum = triangle[0][axis];
        double maximum = minimum;
        for (unsigned int corner = 1u; corner < 3u; ++corner)
        {
            minimum = fmin(minimum, triangle[corner][axis]);
            maximum = fmax(maximum, triangle[corner][axis]);
        }
        center[axis] = minimum * 0.5 + maximum * 0.5;
    }
    for (unsigned int corner = 0u; corner < 3u; ++corner)
    {
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            normalized[corner][axis] = triangle[corner][axis] - center[axis];
            scale                    = fmax(scale, fabs(normalized[corner][axis]));
        }
    }
    if (!isfinite(scale) || scale == 0.0)
    {
        return false;
    }
    // One division, by the determinant, remains below.
    const double inverse_scale = inversePowerOfTwo(scale);
    for (unsigned int corner = 0u; corner < 3u; ++corner)
    {
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            normalized[corner][axis] *= inverse_scale;
        }
    }
    double first_edge[3]             = {};
    double second_edge[3]            = {};
    double origin_delta[3]           = {};
    double direction[3]              = {};
    double cross_direction_second[3] = {};
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        first_edge[axis]   = normalized[1][axis] - normalized[0][axis];
        second_edge[axis]  = normalized[2][axis] - normalized[0][axis];
        origin_delta[axis] = (origin[axis] - center[axis]) * inverse_scale - normalized[0][axis];
        direction[axis]    = static_cast<double>(direction_source[axis]) * inverse_scale;
    }
    cross_direction_second[0]          = direction[1] * second_edge[2] - direction[2] * second_edge[1];
    cross_direction_second[1]          = direction[2] * second_edge[0] - direction[0] * second_edge[2];
    cross_direction_second[2]          = direction[0] * second_edge[1] - direction[1] * second_edge[0];
    const double determinant           = dotVector(first_edge, cross_direction_second);
    const double determinant_magnitude = fabs(first_edge[0] * cross_direction_second[0]) +
                                         fabs(first_edge[1] * cross_direction_second[1]) +
                                         fabs(first_edge[2] * cross_direction_second[2]);
    const double tolerance = determinant_magnitude * kDoubleEpsilon * 8.0;
    if (!isfinite(determinant) || (double_sided ? fabs(determinant) <= tolerance : determinant <= tolerance))
    {
        return false;
    }
    const double inverse_determinant = 1.0 / determinant;
    const double u                   = dotVector(origin_delta, cross_direction_second) * inverse_determinant;
    if (u < -kTriangleEdgeMargin || u > 1.0 + kTriangleEdgeMargin)
    {
        return false;
    }
    const double cross_origin_first[3] = {
        origin_delta[1] * first_edge[2] - origin_delta[2] * first_edge[1],
        origin_delta[2] * first_edge[0] - origin_delta[0] * first_edge[2],
        origin_delta[0] * first_edge[1] - origin_delta[1] * first_edge[0],
    };
    const double v = dotVector(direction, cross_origin_first) * inverse_determinant;
    if (v < -kTriangleEdgeMargin || u + v > 1.0 + kTriangleEdgeMargin)
    {
        return false;
    }
    const double distance = dotVector(second_edge, cross_origin_first) * inverse_determinant;
    if (!isfinite(distance) || distance < minimum_distance || distance > maximum_distance)
    {
        return false;
    }
    double normal[3] = {};
    if (!triangleNormal(triangle, normal))
    {
        return false;
    }
    double direction_dot_normal = 0.0;
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        direction_dot_normal += normal[axis] * static_cast<double>(direction_source[axis]);
    }
    // A mesh stored double-sided reports the normal facing the ray; one the query counts
    // from both sides keeps its front normal, as PhysX's eMESH_BOTH_SIDES does (ADR-0065).
    if (flip_back_faces && direction_dot_normal > 0.0)
    {
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            normal[axis] = -normal[axis];
        }
    }
    output.distance = distance;
    output.uv[0]    = static_cast<float>(u);
    output.uv[1]    = static_cast<float>(v);
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        output.position[axis] = fma(static_cast<double>(direction_source[axis]), distance, origin[axis]);
        output.normal[axis]   = static_cast<float>(normal[axis]);
    }
    return true;
}

// The center moves to origin + t d with the caller's float direction as given, unit only
// to within rounding. As on the host, the vertex and edge entries keep |d|^2 and solve
// around the path's closest approach, so neither the direction's length nor the rounding
// of a long path's squared distances leaves the root outside the contact check's tolerance.
__device__ bool raySphereEntry(const double origin[3], const float direction[3], double inverse_direction_square,
                               const double point[3], double radius, double& out_distance)
{
    double offset[3]  = {};
    double projection = 0.0;
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        offset[axis] = origin[axis] - point[axis];
        projection   = fma(offset[axis], static_cast<double>(direction[axis]), projection);
    }
    const double closest     = -projection * inverse_direction_square;
    double       gap_squared = 0.0;
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        const double gap = fma(static_cast<double>(direction[axis]), closest, offset[axis]);
        gap_squared      = fma(gap, gap, gap_squared);
    }
    const double reach = radius * radius - gap_squared;
    if (!isfinite(reach) || reach < 0.0)
    {
        return false;
    }
    out_distance = closest - sqrt(reach * inverse_direction_square);
    return isfinite(out_distance);
}

__device__ bool rayCapsuleEdgeEntry(const double origin[3], const float direction[3], const double first[3],
                                    const double second[3], double radius, double& out_distance)
{
    double edge[3]        = {};
    double offset[3]      = {};
    double edge_squared   = 0.0;
    double edge_offset    = 0.0;
    double edge_direction = 0.0;
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        edge[axis]     = second[axis] - first[axis];
        offset[axis]   = origin[axis] - first[axis];
        edge_squared   = fma(edge[axis], edge[axis], edge_squared);
        edge_offset    = fma(edge[axis], offset[axis], edge_offset);
        edge_direction = fma(edge[axis], static_cast<double>(direction[axis]), edge_direction);
    }
    // Offset and motion across the edge's line, both scaled by |e|^2, then their
    // closest approach; the scale cancels from the parameter and the gap is |e|^2 too.
    double across[3]      = {};
    double motion[3]      = {};
    double motion_squared = 0.0;
    double approach       = 0.0;
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        across[axis]   = fma(-edge[axis], edge_offset, edge_squared * offset[axis]);
        motion[axis]   = fma(-edge[axis], edge_direction, edge_squared * static_cast<double>(direction[axis]));
        motion_squared = fma(motion[axis], motion[axis], motion_squared);
        approach       = fma(across[axis], motion[axis], approach);
    }
    if (!isfinite(motion_squared) || motion_squared <= 0.0)
    {
        return false;
    }
    const double inverse_motion = 1.0 / motion_squared;
    const double closest        = -approach * inverse_motion;
    double       gap_squared    = 0.0;
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        const double gap = fma(motion[axis], closest, across[axis]);
        gap_squared      = fma(gap, gap, gap_squared);
    }
    const double scaled_radius = radius * edge_squared;
    const double reach         = scaled_radius * scaled_radius - gap_squared;
    if (!isfinite(reach) || reach < 0.0)
    {
        return false;
    }
    const double distance       = closest - sqrt(reach * inverse_motion);
    const double edge_parameter = edge_offset + distance * edge_direction;
    if (!isfinite(distance) || edge_parameter < 0.0 || edge_parameter > edge_squared)
    {
        return false;
    }
    out_distance = distance;
    return true;
}

// As on CPU: the squared-distance slack of a contact rebuilt from coordinates far larger
// than its radius.
__device__ double sphereContactRoundingSlack(const double center[3], const double anchor[3],
                                             const double relative[3][3], double radius, double distance)
{
    double magnitude = 0.0;
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        magnitude = fmax(magnitude, fabs(center[axis]) + fabs(anchor[axis]));
        for (unsigned int corner = 1u; corner < 3u; ++corner)
            magnitude = fmax(magnitude, fabs(relative[corner][axis]));
    }
    const double spread = (magnitude + fabs(distance)) * kDoubleEpsilon * 256.0;
    return spread * (2.0 * radius + spread);
}

__device__ void considerSweepContact(const double triangle[3][3], const double face_normal[3], const double center[3],
                                     double radius, const float direction[3], double minimum_distance,
                                     double maximum_distance, double distance, unsigned int face, bool& has_hit,
                                     TriangleHit& output)
{
    if (!isfinite(distance) || distance < minimum_distance || distance > maximum_distance ||
        (has_hit && distance > output.distance))
    {
        return;
    }
    // Match CPU contact reconstruction without rounding a large query center
    // before evaluating the radius residual.
    double relative_triangle[3][3]{};
    for (unsigned int corner = 0; corner < 3; ++corner)
        for (unsigned int axis = 0; axis < 3; ++axis)
            relative_triangle[corner][axis] = triangle[corner][axis] - triangle[0][axis];
    double sphere_center[3] = {};
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        sphere_center[axis] = fma(static_cast<double>(direction[axis]), distance, center[axis] - triangle[0][axis]);
    }
    double contact[3] = {};
    double u          = 0.0;
    double v          = 0.0;
    closestPointOnTriangle(relative_triangle, sphere_center, contact, u, v);
    double delta[3]         = {};
    double distance_squared = 0.0;
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        delta[axis]      = sphere_center[axis] - contact[axis];
        distance_squared = fma(delta[axis], delta[axis], distance_squared);
    }
    const double radius_squared = radius * radius;
    const double tolerance      = fmax(1.0, radius_squared) * kDoubleEpsilon * 256.0;
    // As on CPU, only a solved contact time takes the coordinate slack, not the start's overlap.
    if (!isfinite(distance_squared) ||
        (distance_squared > radius_squared + tolerance &&
         (distance == minimum_distance ||
          distance_squared >
              radius_squared + sphereContactRoundingSlack(center, triangle[0], relative_triangle, radius, distance))) ||
        (has_hit && distance == output.distance && face >= output.face))
    {
        return;
    }
    double       contact_normal[3] = {};
    const double contact_distance  = sqrt(fmax(0.0, distance_squared));
    if (contact_distance > sqrt(tolerance))
    {
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            contact_normal[axis] = delta[axis] / contact_distance;
        }
    }
    else
    {
        double direction_dot_normal = 0.0;
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            contact_normal[axis] = face_normal[axis];
            direction_dot_normal = fma(static_cast<double>(direction[axis]), face_normal[axis], direction_dot_normal);
        }
        // Only a double-sided triangle survives with the motion along its front normal.
        if (direction_dot_normal > 0.0)
        {
            for (unsigned int axis = 0u; axis < 3u; ++axis)
            {
                contact_normal[axis] = -contact_normal[axis];
            }
        }
    }
    has_hit         = true;
    output.distance = distance;
    output.face     = face;
    output.uv[0]    = static_cast<float>(u);
    output.uv[1]    = static_cast<float>(v);
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        output.position[axis] = contact[axis] + triangle[0][axis];
        output.normal[axis]   = static_cast<float>(contact_normal[axis]);
    }
}

__device__ bool sweepTriangle(const double center[3], double radius, const float direction[3], double minimum_distance,
                              double maximum_distance, const double triangle[3][3], bool double_sided,
                              bool flip_back_faces, unsigned int face, TriangleHit& output)
{
    double face_normal[3] = {};
    if (!triangleNormal(triangle, face_normal))
    {
        return false;
    }
    double signed_origin   = 0.0;
    double normal_velocity = 0.0;
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        signed_origin   = fma(center[axis] - triangle[0][axis], face_normal[axis], signed_origin);
        normal_velocity = fma(static_cast<double>(direction[axis]), face_normal[axis], normal_velocity);
    }
    // ADR-0051, as on CPU: a single-sided triangle whose front normal points along the motion
    // is ignored; every other one is a two-sided surface, met first at its least contact time.
    if (!double_sided && normal_velocity > 0.0)
    {
        return false;
    }
    bool has_hit = false;
    considerSweepContact(triangle, face_normal, center, radius, direction, minimum_distance, maximum_distance,
                         minimum_distance, face, has_hit, output);
    for (int side = -1; side <= 1; side += 2)
    {
        const double oriented_velocity = static_cast<double>(side) * normal_velocity;
        if (oriented_velocity < 0.0)
        {
            const double distance = (static_cast<double>(side) * radius - signed_origin) / normal_velocity;
            considerSweepContact(triangle, face_normal, center, radius, direction, minimum_distance, maximum_distance,
                                 distance, face, has_hit, output);
        }
    }
    double direction_square = 0.0;
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        direction_square =
            fma(static_cast<double>(direction[axis]), static_cast<double>(direction[axis]), direction_square);
    }
    const double inverse_direction_square = 1.0 / direction_square;
    for (unsigned int corner = 0u; corner < 3u; ++corner)
    {
        double distance = 0.0;
        if (raySphereEntry(center, direction, inverse_direction_square, triangle[corner], radius, distance))
        {
            considerSweepContact(triangle, face_normal, center, radius, direction, minimum_distance, maximum_distance,
                                 distance, face, has_hit, output);
        }
        const unsigned int next = (corner + 1u) % 3u;
        if (rayCapsuleEdgeEntry(center, direction, triangle[corner], triangle[next], radius, distance))
        {
            considerSweepContact(triangle, face_normal, center, radius, direction, minimum_distance, maximum_distance,
                                 distance, face, has_hit, output);
        }
    }
    // A ray with a contact tolerance that reaches a single-sided triangle only because the
    // query counts it from both sides reports its contact normal mirrored to the front side,
    // as a ray reports the front normal there (ADR-0065); sweeps keep theirs.
    if (has_hit && !flip_back_faces && normal_velocity > 0.0)
    {
        double along_normal = 0.0;
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            along_normal = fma(static_cast<double>(output.normal[axis]), face_normal[axis], along_normal);
        }
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            output.normal[axis] = static_cast<float>(
                fma(-2.0 * along_normal, face_normal[axis], static_cast<double>(output.normal[axis])));
        }
    }
    return has_hit;
}

__device__ bool separatesTriangleAndBox(const double axis[3], const double triangle[3][3], const double half_extents[3])
{
    double axis_scale = 0.0;
    for (unsigned int component = 0u; component < 3u; ++component)
    {
        axis_scale = fmax(axis_scale, fabs(axis[component]));
    }
    if (!isfinite(axis_scale) || axis_scale == 0.0)
    {
        return false;
    }
    const double normalized_axis[3] = {
        axis[0] / axis_scale,
        axis[1] / axis_scale,
        axis[2] / axis_scale,
    };
    double minimum = dotVector(triangle[0], normalized_axis);
    double maximum = minimum;
    for (unsigned int corner = 1u; corner < 3u; ++corner)
    {
        const double projection = dotVector(triangle[corner], normalized_axis);
        minimum                 = fmin(minimum, projection);
        maximum                 = fmax(maximum, projection);
    }
    const double radius = half_extents[0] * fabs(normalized_axis[0]) + half_extents[1] * fabs(normalized_axis[1]) +
                          half_extents[2] * fabs(normalized_axis[2]);
    return minimum > radius || maximum < -radius;
}

__device__ bool triangleOverlapsCenteredBox(const double triangle[3][3], const double half_extents[3])
{
    const double box_axes[3][3] = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    };
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        if (separatesTriangleAndBox(box_axes[axis], triangle, half_extents))
        {
            return false;
        }
    }
    double edges[3][3] = {};
    for (unsigned int edge = 0u; edge < 3u; ++edge)
    {
        const unsigned int next = (edge + 1u) % 3u;
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            edges[edge][axis] = triangle[next][axis] - triangle[edge][axis];
        }
        for (unsigned int box_axis = 0u; box_axis < 3u; ++box_axis)
        {
            const double cross_axis[3] = {
                edges[edge][1] * box_axes[box_axis][2] - edges[edge][2] * box_axes[box_axis][1],
                edges[edge][2] * box_axes[box_axis][0] - edges[edge][0] * box_axes[box_axis][2],
                edges[edge][0] * box_axes[box_axis][1] - edges[edge][1] * box_axes[box_axis][0],
            };
            if (separatesTriangleAndBox(cross_axis, triangle, half_extents))
            {
                return false;
            }
        }
    }
    const double triangle_normal[3] = {
        edges[0][1] * edges[1][2] - edges[0][2] * edges[1][1],
        edges[0][2] * edges[1][0] - edges[0][0] * edges[1][2],
        edges[0][0] * edges[1][1] - edges[0][1] * edges[1][0],
    };
    return !separatesTriangleAndBox(triangle_normal, triangle, half_extents);
}

__device__ bool preciseOverlap(unsigned int overlap_type, const double* position, double radius,
                               const double* rotation_source, const double* extents_source, const double* aabb_maximum,
                               const double triangle[3][3])
{
    if (overlap_type == cudaFamily::kOverlapSphere)
    {
        if (!isfinite(radius) || radius <= 0.0)
        {
            return false;
        }
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            if (!isfinite(position[axis]))
            {
                return false;
            }
        }
        double closest[3] = {};
        double u          = 0.0;
        double v          = 0.0;
        closestPointOnTriangle(triangle, position, closest, u, v);
        double normalized_squared = 0.0;
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            const double distance = position[axis] - closest[axis];
            if (fabs(distance) > radius)
            {
                return false;
            }
            const double normalized = distance / radius;
            normalized_squared      = fma(normalized, normalized, normalized_squared);
        }
        return normalized_squared <= 1.0;
    }
    double local_triangle[3][3] = {};
    double half_extents[3]      = {};
    if (overlap_type == cudaFamily::kOverlapAabb)
    {
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            if (!isfinite(position[axis]) || !isfinite(aabb_maximum[axis]) || position[axis] > aabb_maximum[axis])
            {
                return false;
            }
            const double center = position[axis] * 0.5 + aabb_maximum[axis] * 0.5;
            half_extents[axis]  = aabb_maximum[axis] * 0.5 - position[axis] * 0.5;
            for (unsigned int corner = 0u; corner < 3u; ++corner)
            {
                local_triangle[corner][axis] = triangle[corner][axis] - center;
            }
        }
    }
    else if (overlap_type == cudaFamily::kOverlapBox)
    {
        double axes[9] = {};
        if (!cudaFamilyQuaternionAxes(rotation_source, axes))
        {
            return false;
        }
        for (unsigned int local_axis = 0u; local_axis < 3u; ++local_axis)
        {
            if (!isfinite(position[local_axis]) || !isfinite(extents_source[local_axis]) ||
                extents_source[local_axis] <= 0.0)
            {
                return false;
            }
            half_extents[local_axis] = extents_source[local_axis];
            for (unsigned int corner = 0u; corner < 3u; ++corner)
            {
                for (unsigned int world_axis = 0u; world_axis < 3u; ++world_axis)
                {
                    local_triangle[corner][local_axis] =
                        fma(axes[local_axis * 3u + world_axis], triangle[corner][world_axis] - position[world_axis],
                            local_triangle[corner][local_axis]);
                }
            }
        }
    }
    else
    {
        return false;
    }
    return triangleOverlapsCenteredBox(local_triangle, half_extents);
}

struct CudaFamilyHit
{
    double             distance;
    double             position[3];
    float              normal[3];
    bool               triangleValid;
    unsigned long long sourceFace;
    unsigned int       sourceCorners[3];
    float              barycentrics[3];
};

__device__ boxSweep::Vec3 cudaFamilySweepVec(const double value[3]) { return {value[0], value[1], value[2]}; }

__device__ boxSweep::Vec3 cudaFamilySweepDirection(const float value[3])
{
    return {static_cast<double>(value[0]), static_cast<double>(value[1]), static_cast<double>(value[2])};
}

__device__ boxSweep::Box cudaFamilySweepBox(const cudaFamily::BoxGeometry& value)
{
    return {cudaFamilySweepVec(value.center),
            {{value.axes[0], value.axes[1], value.axes[2]},
             {value.axes[3], value.axes[4], value.axes[5]},
             {value.axes[6], value.axes[7], value.axes[8]}},
            cudaFamilySweepVec(value.halfExtents)};
}

__device__ void cudaFamilySweepHit(const boxSweep::Hit& source, CudaFamilyHit& destination)
{
    destination             = CudaFamilyHit{};
    destination.distance    = source.distance;
    destination.position[0] = source.position.x;
    destination.position[1] = source.position.y;
    destination.position[2] = source.position.z;
    destination.normal[0]   = static_cast<float>(source.normal.x);
    destination.normal[1]   = static_cast<float>(source.normal.y);
    destination.normal[2]   = static_cast<float>(source.normal.z);
}

// boxBox's contact, out of line: inlined into each box-box test, its face loop would
// enlarge the family kernel, which every lane pays for.
__device__ __noinline__ void cudaFamilyBoxBoxContact(const boxSweep::Box& query, const boxSweep::Box& target,
                                                     boxSweep::Vec3 direction, const boxSweep::Interval& interval,
                                                     CudaFamilyHit& hit)
{
    boxSweep::Hit contact{};
    boxSweep::boxBoxContact(query, target, direction, interval, contact);
    cudaFamilySweepHit(contact, hit);
}

// A box-box hit, with its contact only when `contact`; a single-hit row computes the
// contact once, for its winner (cudaFamilyBoxSweepWinnerContact).
__device__ bool cudaFamilyBoxBoxHit(const boxSweep::Box& query, const boxSweep::Box& target, boxSweep::Vec3 direction,
                                    double minimumDistance, double maximumDistance, bool contact, CudaFamilyHit& hit)
{
    boxSweep::Interval interval{};
    if (!boxSweep::boxBoxInterval(query, target, direction, minimumDistance, maximumDistance, interval))
    {
        return false;
    }
    if (contact)
    {
        cudaFamilyBoxBoxContact(query, target, direction, interval, hit);
        return true;
    }
    hit          = CudaFamilyHit{};
    hit.distance = interval.enter;
    return true;
}

// An object's bounds as an axis-aligned box.
__device__ cudaFamily::BoxGeometry cudaFamilyBoundsBox(const double boundsMinimum[3], const double boundsMaximum[3])
{
    cudaFamily::BoxGeometry box{};
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        box.center[axis]      = boundsMinimum[axis] + (boundsMaximum[axis] - boundsMinimum[axis]) * 0.5;
        box.halfExtents[axis] = (boundsMaximum[axis] - boundsMinimum[axis]) * 0.5;
    }
    box.axes[0] = box.axes[4] = box.axes[8] = 1.0;
    return box;
}

__device__ void cudaFamilyCopy3(const double* source, double output[3])
{
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        output[axis] = source[axis];
    }
}

__device__ void cudaFamilySubtract3(const double first[3], const double second[3], double output[3])
{
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        output[axis] = first[axis] - second[axis];
    }
}

__device__ void cudaFamilyCross3(const double first[3], const double second[3], double output[3])
{
    output[0] = first[1] * second[2] - first[2] * second[1];
    output[1] = first[2] * second[0] - first[0] * second[2];
    output[2] = first[0] * second[1] - first[1] * second[0];
}

__device__ bool cudaFamilyNormalize(double value[3])
{
    const double length = sqrt(dotVector(value, value));
    if (!isfinite(length) || length == 0.0)
    {
        return false;
    }
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        value[axis] /= length;
    }
    return true;
}

__device__ double cudaFamilyAddTolerance(double value, double tolerance)
{
    const double maximum = 1.7976931348623157e+308;
    return value > maximum - tolerance ? maximum : value + tolerance;
}

__device__ void cudaFamilySort(double* values, unsigned int count)
{
    for (unsigned int index = 1u; index < count; ++index)
    {
        const double value       = values[index];
        unsigned int destination = index;
        while (destination != 0u && value < values[destination - 1u])
        {
            values[destination] = values[destination - 1u];
            --destination;
        }
        values[destination] = value;
    }
}

__device__ void cudaFamilyBoxLocalPoint(const cudaFamily::BoxGeometry& box, const double world[3], double local[3])
{
    double delta[3] = {};
    cudaFamilySubtract3(world, box.center, delta);
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        local[axis] = dotVector(delta, box.axes + axis * 3u);
    }
}

__device__ void cudaFamilyBoxLocalVector(const cudaFamily::BoxGeometry& box, const double world[3], double local[3])
{
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        local[axis] = dotVector(world, box.axes + axis * 3u);
    }
}

__device__ void cudaFamilyBoxWorldPoint(const cudaFamily::BoxGeometry& box, const double local[3], double world[3])
{
    cudaFamilyCopy3(box.center, world);
    for (unsigned int localAxis = 0u; localAxis < 3u; ++localAxis)
    {
        for (unsigned int worldAxis = 0u; worldAxis < 3u; ++worldAxis)
        {
            world[worldAxis] = fma(box.axes[localAxis * 3u + worldAxis], local[localAxis], world[worldAxis]);
        }
    }
}

__device__ double cudaFamilyPointBoxDistanceSquared(const double point[3], const double extents[3], double* closest)
{
    double result = 0.0;
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        const double value = fmax(-extents[axis], fmin(extents[axis], point[axis]));
        if (closest != nullptr)
        {
            closest[axis] = value;
        }
        const double delta = point[axis] - value;
        result             = fma(delta, delta, result);
    }
    return result;
}

__device__ double cudaFamilyRoundedBoxDistanceAt(const double origin[3], const double direction[3],
                                                 const double extents[3], double distance)
{
    double point[3] = {};
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        point[axis] = fma(direction[axis], distance, origin[axis]);
    }
    return cudaFamilyPointBoxDistanceSquared(point, extents, nullptr);
}

__device__ bool cudaFamilyFirstRoundedBoxPath(const double origin[3], const double direction[3],
                                              const double extents[3], double radius, double minimumDistance,
                                              double maximumDistance, double& output)
{
    const double radiusSquared = radius * radius;
    if (cudaFamilyRoundedBoxDistanceAt(origin, direction, extents, minimumDistance) <= radiusSquared)
    {
        output = minimumDistance;
        return true;
    }
    double       breakpoints[8] = {};
    unsigned int count          = 0u;
    breakpoints[count++]        = minimumDistance;
    breakpoints[count++]        = maximumDistance;
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        if (direction[axis] == 0.0)
        {
            continue;
        }
        for (unsigned int side = 0u; side < 2u; ++side)
        {
            const double face     = side == 0u ? -extents[axis] : extents[axis];
            const double distance = (face - origin[axis]) / direction[axis];
            if (distance > minimumDistance && distance < maximumDistance)
            {
                breakpoints[count++] = distance;
            }
        }
    }
    cudaFamilySort(breakpoints, count);
    for (unsigned int interval = 0u; interval + 1u < count; ++interval)
    {
        const double left  = breakpoints[interval];
        const double right = breakpoints[interval + 1u];
        if (cudaFamilyRoundedBoxDistanceAt(origin, direction, extents, left) <= radiusSquared)
        {
            output = left;
            return true;
        }
        const double   midpoint  = left * 0.5 + right * 0.5;
        double         quadratic = 0.0;
        double         linear    = 0.0;
        boxSweep::Vec3 offset    = {0.0, 0.0, 0.0};
        boxSweep::Vec3 velocity  = {0.0, 0.0, 0.0};
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            const double coordinate = fma(direction[axis], midpoint, origin[axis]);
            if (coordinate < -extents[axis])
            {
                offset[axis] = origin[axis] + extents[axis];
            }
            else if (coordinate > extents[axis])
            {
                offset[axis] = origin[axis] - extents[axis];
            }
            else
            {
                continue;
            }
            velocity[axis] = direction[axis];
            quadratic      = fma(direction[axis], direction[axis], quadratic);
            linear         = fma(direction[axis], offset[axis], linear);
        }
        // The CPU's root and residual guard (box_sweep::roundedBoxPath). At an edge or a
        // corner the textbook discriminant is a small difference of large terms, and an
        // exact re-check then rejects the true contact a few ulps outside the radius.
        double candidate = left;
        if (!boxSweep::featureEntry(offset, velocity, quadratic, linear, radiusSquared, left, right, candidate))
        {
            continue;
        }
        double scale = 0.0;
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            scale = fmax(scale, fmax(fmax(fabs(fma(direction[axis], candidate, origin[axis])), fabs(origin[axis])),
                                     extents[axis]));
        }
        if (quadratic == 0.0 ||
            boxSweep::touches(cudaFamilyRoundedBoxDistanceAt(origin, direction, extents, candidate), radius, scale))
        {
            output = candidate;
            return true;
        }
    }
    if (cudaFamilyRoundedBoxDistanceAt(origin, direction, extents, maximumDistance) <= radiusSquared)
    {
        output = maximumDistance;
        return true;
    }
    return false;
}

__device__ void cudaFamilySetBoxHit(const cudaFamily::BoxGeometry& box, const double localOrigin[3],
                                    const double localDirection[3], double distance, CudaFamilyHit& hit)
{
    double localPoint[3] = {};
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        localPoint[axis] = fma(localDirection[axis], distance, localOrigin[axis]);
    }
    double       localClosest[3] = {};
    const double distanceSquared = cudaFamilyPointBoxDistanceSquared(localPoint, box.halfExtents, localClosest);
    double       localNormal[3]  = {};
    if (distanceSquared > 0.0)
    {
        const double inverseLength = 1.0 / sqrt(distanceSquared);
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            localNormal[axis] = (localPoint[axis] - localClosest[axis]) * inverseLength;
        }
    }
    else
    {
        unsigned int nearestAxis = 0u;
        double       nearestFace = 1.7976931348623157e+308;
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            const double faceDistance = fabs(box.halfExtents[axis] - fabs(localPoint[axis]));
            if (faceDistance < nearestFace)
            {
                nearestFace = faceDistance;
                nearestAxis = axis;
            }
        }
        localNormal[nearestAxis]  = localPoint[nearestAxis] < 0.0 ? -1.0 : 1.0;
        localClosest[nearestAxis] = localNormal[nearestAxis] * box.halfExtents[nearestAxis];
    }
    cudaFamilyBoxWorldPoint(box, localClosest, hit.position);
    hit.distance = distance;
    for (unsigned int worldAxis = 0u; worldAxis < 3u; ++worldAxis)
    {
        double normal = 0.0;
        for (unsigned int localAxis = 0u; localAxis < 3u; ++localAxis)
        {
            normal = fma(box.axes[localAxis * 3u + worldAxis], localNormal[localAxis], normal);
        }
        hit.normal[worldAxis] = static_cast<float>(normal);
    }
}

__device__ bool cudaFamilyBoxPathHit(const cudaFamily::BoxGeometry& box, const double origin[3],
                                     const float directionSource[3], double radius, double minimumDistance,
                                     double maximumDistance, CudaFamilyHit& hit)
{
    double direction[3] = {
        static_cast<double>(directionSource[0]),
        static_cast<double>(directionSource[1]),
        static_cast<double>(directionSource[2]),
    };
    double localOrigin[3]    = {};
    double localDirection[3] = {};
    cudaFamilyBoxLocalPoint(box, origin, localOrigin);
    cudaFamilyBoxLocalVector(box, direction, localDirection);
    double distance = 0.0;
    if (!cudaFamilyFirstRoundedBoxPath(localOrigin, localDirection, box.halfExtents, radius, minimumDistance,
                                       maximumDistance, distance))
    {
        return false;
    }
    cudaFamilySetBoxHit(box, localOrigin, localDirection, distance, hit);
    return true;
}

__device__ void cudaFamilyClosestPointOnSegment(const cudaFamily::CapsuleGeometry& capsule, const double point[3],
                                                double closest[3])
{
    double segment[3] = {};
    double offset[3]  = {};
    cudaFamilySubtract3(capsule.endpointB, capsule.endpointA, segment);
    cudaFamilySubtract3(point, capsule.endpointA, offset);
    // An empty segment divides 0 by 0; fmin and fmax drop that NaN, so the result is endpoint A.
    // Guarding the division with a branch instead made EXACT sphere sweeps about 12% slower.
    const double denominator = dotVector(segment, segment);
    const double parameter   = fmax(0.0, fmin(1.0, dotVector(offset, segment) / denominator));
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        closest[axis] = fma(segment[axis], parameter, capsule.endpointA[axis]);
    }
}

__device__ double cudaFamilyPointCapsuleDistanceSquared(const cudaFamily::CapsuleGeometry& capsule,
                                                        const double                       point[3])
{
    double closest[3] = {};
    double delta[3]   = {};
    cudaFamilyClosestPointOnSegment(capsule, point, closest);
    cudaFamilySubtract3(point, closest, delta);
    return dotVector(delta, delta);
}

__device__ double cudaFamilyCapsuleDistanceAt(const cudaFamily::CapsuleGeometry& capsule, const double origin[3],
                                              const double direction[3], double distance)
{
    double point[3] = {};
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        point[axis] = fma(direction[axis], distance, origin[axis]);
    }
    return cudaFamilyPointCapsuleDistanceSquared(capsule, point);
}

__device__ bool cudaFamilyFirstCapsulePath(const cudaFamily::CapsuleGeometry& capsule, const double origin[3],
                                           const double direction[3], double radius, double minimumDistance,
                                           double maximumDistance, double& output)
{
    double segment[3]      = {};
    double originOffset[3] = {};
    cudaFamilySubtract3(capsule.endpointB, capsule.endpointA, segment);
    cudaFamilySubtract3(origin, capsule.endpointA, originOffset);
    const double segmentLengthSquared = dotVector(segment, segment);
    const double projectionOrigin     = dotVector(originOffset, segment);
    const double projectionDirection  = dotVector(direction, segment);
    const double radiusSquared        = radius * radius;
    if (cudaFamilyCapsuleDistanceAt(capsule, origin, direction, minimumDistance) <= radiusSquared)
    {
        output = minimumDistance;
        return true;
    }
    double       breakpoints[4] = {minimumDistance, maximumDistance, 0.0, 0.0};
    unsigned int count          = 2u;
    if (projectionDirection != 0.0)
    {
        for (unsigned int endpoint = 0u; endpoint < 2u; ++endpoint)
        {
            const double projection = endpoint == 0u ? 0.0 : segmentLengthSquared;
            const double distance   = (projection - projectionOrigin) / projectionDirection;
            if (distance > minimumDistance && distance < maximumDistance)
            {
                breakpoints[count++] = distance;
            }
        }
    }
    cudaFamilySort(breakpoints, count);
    for (unsigned int interval = 0u; interval + 1u < count; ++interval)
    {
        const double left  = breakpoints[interval];
        const double right = breakpoints[interval + 1u];
        if (cudaFamilyCapsuleDistanceAt(capsule, origin, direction, left) <= radiusSquared)
        {
            output = left;
            return true;
        }
        const double midpoint              = left * 0.5 + right * 0.5;
        const double projection            = fma(projectionDirection, midpoint, projectionOrigin);
        double       quadraticOrigin[3]    = {};
        double       quadraticDirection[3] = {};
        if (projection <= 0.0)
        {
            cudaFamilyCopy3(originOffset, quadraticOrigin);
            cudaFamilyCopy3(direction, quadraticDirection);
        }
        else if (projection >= segmentLengthSquared)
        {
            cudaFamilySubtract3(origin, capsule.endpointB, quadraticOrigin);
            cudaFamilyCopy3(direction, quadraticDirection);
        }
        else
        {
            const double originParameter    = projectionOrigin / segmentLengthSquared;
            const double directionParameter = projectionDirection / segmentLengthSquared;
            for (unsigned int axis = 0u; axis < 3u; ++axis)
            {
                quadraticOrigin[axis]    = fma(-segment[axis], originParameter, originOffset[axis]);
                quadraticDirection[axis] = fma(-segment[axis], directionParameter, direction[axis]);
            }
        }
        // The CPU's root and residual guard (box_sweep::capsulePath): a textbook
        // discriminant loses eps |origin - capsule|^2, and a radius-relative guard
        // then rejects true contacts of paths that start far from the capsule.
        const boxSweep::Vec3 offset    = cudaFamilySweepVec(quadraticOrigin);
        const boxSweep::Vec3 velocity  = cudaFamilySweepVec(quadraticDirection);
        const double         quadratic = boxSweep::dot(velocity, velocity);
        double               candidate = left;
        if (!boxSweep::featureEntry(offset, velocity, quadratic, boxSweep::dot(offset, velocity), radiusSquared, left,
                                    right, candidate))
        {
            continue;
        }
        double scale = 0.0;
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            scale = fmax(scale, fmax(fmax(fabs(fma(direction[axis], candidate, origin[axis])), fabs(origin[axis])),
                                     fmax(fabs(capsule.endpointA[axis]), fabs(capsule.endpointB[axis]))));
        }
        if (quadratic == 0.0 ||
            boxSweep::touches(cudaFamilyCapsuleDistanceAt(capsule, origin, direction, candidate), radius, scale))
        {
            output = candidate;
            return true;
        }
    }
    if (cudaFamilyCapsuleDistanceAt(capsule, origin, direction, maximumDistance) <= radiusSquared)
    {
        output = maximumDistance;
        return true;
    }
    return false;
}

__device__ bool cudaFamilyCapsulePathHit(const cudaFamily::CapsuleGeometry& capsule, const double origin[3],
                                         const float directionSource[3], double expansionRadius, double minimumDistance,
                                         double maximumDistance, CudaFamilyHit& hit)
{
    double direction[3] = {
        static_cast<double>(directionSource[0]),
        static_cast<double>(directionSource[1]),
        static_cast<double>(directionSource[2]),
    };
    double distance = 0.0;
    if (!cudaFamilyFirstCapsulePath(capsule, origin, direction, capsule.radius + expansionRadius, minimumDistance,
                                    maximumDistance, distance))
    {
        return false;
    }
    double center[3] = {};
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        center[axis] = fma(direction[axis], distance, origin[axis]);
    }
    boxSweep::Vec3       closest{};
    const boxSweep::Vec3 normal =
        boxSweep::capsuleContactNormal(cudaFamilySweepVec(capsule.endpointA), cudaFamilySweepVec(capsule.endpointB),
                                       cudaFamilySweepVec(center), cudaFamilySweepVec(direction), closest);
    hit.distance    = distance;
    hit.normal[0]   = static_cast<float>(normal.x);
    hit.normal[1]   = static_cast<float>(normal.y);
    hit.normal[2]   = static_cast<float>(normal.z);
    hit.position[0] = fma(normal.x, capsule.radius, closest.x);
    hit.position[1] = fma(normal.y, capsule.radius, closest.y);
    hit.position[2] = fma(normal.z, capsule.radius, closest.z);
    return true;
}

// As on the host, the roots keep the float direction's |d|^2 and take the half chord from the
// path's closest approach, and a path already touching or inside the sphere at the minimum
// reports the minimum.
__device__ bool cudaFamilySpherePathHit(const cudaFamily::SphereGeometry& sphere, const double origin[3],
                                        const float directionSource[3], double expansionRadius, double minimumDistance,
                                        double maximumDistance, CudaFamilyHit& hit)
{
    const double radius       = sphere.radius + expansionRadius;
    double       offset[3]    = {};
    double       direction[3] = {};
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        offset[axis]    = origin[axis] - sphere.center[axis];
        direction[axis] = static_cast<double>(directionSource[axis]);
    }
    const double inverseDirectionSquare = 1.0 / dotVector(direction, direction);
    const double closest                = -dotVector(offset, direction) * inverseDirectionSquare;
    double       gapSquared             = 0.0;
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        const double gap = fma(direction[axis], closest, offset[axis]);
        gapSquared       = fma(gap, gap, gapSquared);
    }
    const double halfChordSquared = (radius * radius - gapSquared) * inverseDirectionSquare;
    if (!(halfChordSquared >= 0.0))
    {
        return false;
    }
    const double halfChord = sqrt(halfChordSquared);
    if (!(closest + halfChord >= minimumDistance))
    {
        return false;
    }
    const double distance = fmax(minimumDistance, closest - halfChord);
    if (!isfinite(distance) || distance > maximumDistance)
    {
        return false;
    }
    double movingCenter[3] = {};
    double normal[3]       = {};
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        movingCenter[axis] = fma(direction[axis], distance, origin[axis]);
        normal[axis]       = movingCenter[axis] - sphere.center[axis];
    }
    if (!cudaFamilyNormalize(normal))
    {
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            normal[axis] = -direction[axis];
        }
    }
    hit.distance = distance;
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        hit.normal[axis]   = static_cast<float>(normal[axis]);
        hit.position[axis] = fma(normal[axis], sphere.radius, sphere.center[axis]);
    }
    return true;
}

template <typename Source>
__device__ bool cudaFamilyMakeQueryBox(const double translation[3], const Source rotationSource[4],
                                       const double extentsSource[3], double tolerance, cudaFamily::BoxGeometry& box)
{
    double axes[9] = {};
    if (!cudaFamilyQuaternionAxes(rotationSource, axes))
    {
        return false;
    }
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        if (!cudaFamilyInDomain(translation[axis]) || !cudaFamilyInDomain(extentsSource[axis]) ||
            extentsSource[axis] <= 0.0)
        {
            return false;
        }
        box.center[axis]      = translation[axis];
        box.halfExtents[axis] = cudaFamilyAddTolerance(extentsSource[axis], tolerance);
    }
    for (unsigned int value = 0u; value < 9u; ++value)
    {
        box.axes[value] = axes[value];
    }
    return true;
}

__device__ bool cudaFamilySeparatedOnAxis(const double axis[3], const double centerDelta[3],
                                          const cudaFamily::BoxGeometry& first, const cudaFamily::BoxGeometry& second)
{
    const double center       = fabs(dotVector(centerDelta, axis));
    double       firstRadius  = 0.0;
    double       secondRadius = 0.0;
    for (unsigned int local = 0u; local < 3u; ++local)
    {
        firstRadius  = fma(fabs(dotVector(axis, first.axes + local * 3u)), first.halfExtents[local], firstRadius);
        secondRadius = fma(fabs(dotVector(axis, second.axes + local * 3u)), second.halfExtents[local], secondRadius);
    }
    return center > firstRadius + secondRadius;
}

__device__ bool cudaFamilyObbsOverlap(const cudaFamily::BoxGeometry& first, const cudaFamily::BoxGeometry& second)
{
    double centerDelta[3] = {};
    cudaFamilySubtract3(second.center, first.center, centerDelta);
    for (unsigned int index = 0u; index < 3u; ++index)
    {
        if (cudaFamilySeparatedOnAxis(first.axes + index * 3u, centerDelta, first, second))
        {
            return false;
        }
    }
    for (unsigned int index = 0u; index < 3u; ++index)
    {
        if (cudaFamilySeparatedOnAxis(second.axes + index * 3u, centerDelta, first, second))
        {
            return false;
        }
    }
    for (unsigned int firstAxis = 0u; firstAxis < 3u; ++firstAxis)
    {
        for (unsigned int secondAxis = 0u; secondAxis < 3u; ++secondAxis)
        {
            double axis[3] = {};
            cudaFamilyCross3(first.axes + firstAxis * 3u, second.axes + secondAxis * 3u, axis);
            if (dotVector(axis, axis) == 0.0)
            {
                continue;
            }
            if (cudaFamilySeparatedOnAxis(axis, centerDelta, first, second))
            {
                return false;
            }
        }
    }
    return true;
}

__device__ double cudaFamilySegmentBoxDistanceSquared(const double localA[3], const double localB[3],
                                                      const double extents[3])
{
    double direction[3] = {};
    cudaFamilySubtract3(localB, localA, direction);
    double       breakpoints[8] = {0.0, 1.0};
    unsigned int count          = 2u;
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        if (direction[axis] == 0.0)
        {
            continue;
        }
        for (unsigned int side = 0u; side < 2u; ++side)
        {
            const double face      = side == 0u ? -extents[axis] : extents[axis];
            const double parameter = (face - localA[axis]) / direction[axis];
            if (parameter > 0.0 && parameter < 1.0)
            {
                breakpoints[count++] = parameter;
            }
        }
    }
    cudaFamilySort(breakpoints, count);
    double minimum = fmin(cudaFamilyRoundedBoxDistanceAt(localA, direction, extents, 0.0),
                          cudaFamilyRoundedBoxDistanceAt(localA, direction, extents, 1.0));
    for (unsigned int interval = 0u; interval + 1u < count; ++interval)
    {
        const double left      = breakpoints[interval];
        const double right     = breakpoints[interval + 1u];
        const double midpoint  = left * 0.5 + right * 0.5;
        double       quadratic = 0.0;
        double       linear    = 0.0;
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            const double coordinate = fma(direction[axis], midpoint, localA[axis]);
            double       offset     = 0.0;
            if (coordinate < -extents[axis])
            {
                offset = localA[axis] + extents[axis];
            }
            else if (coordinate > extents[axis])
            {
                offset = localA[axis] - extents[axis];
            }
            else
            {
                continue;
            }
            quadratic = fma(direction[axis], direction[axis], quadratic);
            linear    = fma(direction[axis], offset, linear);
        }
        const double candidate = quadratic == 0.0 ? left : fmax(left, fmin(right, -linear / quadratic));
        minimum                = fmin(minimum, cudaFamilyRoundedBoxDistanceAt(localA, direction, extents, candidate));
    }
    return minimum;
}

__device__ bool cudaFamilySphereOverlapsExact(const cudaFamily::SphereGeometry& sphere, const double center[3],
                                              double radius)
{
    double delta[3] = {};
    cudaFamilySubtract3(center, sphere.center, delta);
    const double sum = sphere.radius + radius;
    return dotVector(delta, delta) <= sum * sum;
}

__device__ bool cudaFamilyBoxOverlapsSphere(const cudaFamily::BoxGeometry& box, const double center[3], double radius)
{
    double local[3] = {};
    cudaFamilyBoxLocalPoint(box, center, local);
    return cudaFamilyPointBoxDistanceSquared(local, box.halfExtents, nullptr) <= radius * radius;
}

__device__ bool cudaFamilyCapsuleOverlapsSphere(const cudaFamily::CapsuleGeometry& capsule, const double center[3],
                                                double radius)
{
    const double sum = capsule.radius + radius;
    return cudaFamilyPointCapsuleDistanceSquared(capsule, center) <= sum * sum;
}

__device__ bool cudaFamilyCapsuleOverlapsBox(const cudaFamily::CapsuleGeometry& capsule,
                                             const cudaFamily::BoxGeometry&     box)
{
    double localA[3] = {};
    double localB[3] = {};
    cudaFamilyBoxLocalPoint(box, capsule.endpointA, localA);
    cudaFamilyBoxLocalPoint(box, capsule.endpointB, localB);
    return cudaFamilySegmentBoxDistanceSquared(localA, localB, box.halfExtents) <= capsule.radius * capsule.radius;
}

__device__ bool cudaFamilyConvexContains(const cudaFamily::ConvexGeometry& convex, const double* planes,
                                         const double point[3])
{
    for (unsigned long long index = 0u; index < convex.planeCount; ++index)
    {
        const double* plane = planes + (convex.planeOffset + index) * 4u;
        if (dotVector(plane, point) > plane[3])
        {
            return false;
        }
    }
    return true;
}

__device__ void cudaFamilyLoadConvexTriangle(const cudaFamily::ConvexGeometry& convex, const double* vertices,
                                             const unsigned int* indices, unsigned long long triangleIndex,
                                             double triangle[3][3])
{
    const unsigned long long indexBase = (convex.triangleOffset + triangleIndex) * 3u;
    for (unsigned int corner = 0u; corner < 3u; ++corner)
    {
        const unsigned int vertex = indices[indexBase + corner];
        cudaFamilyCopy3(vertices + (convex.vertexOffset + vertex) * 3u, triangle[corner]);
    }
}

__device__ bool cudaFamilyConvexSphereOverlap(const cudaFamily::ConvexGeometry& convex, const double* vertices,
                                              const unsigned int* indices, const double* planes, const double center[3],
                                              double radius, double* outClosest, double* outNormal)
{
    double closestDistance  = 1.7976931348623157e+308;
    double closest[3]       = {};
    double closestNormal[3] = {};
    for (unsigned long long index = 0u; index < convex.triangleCount; ++index)
    {
        double triangle[3][3] = {};
        double candidate[3]   = {};
        double u              = 0.0;
        double v              = 0.0;
        cudaFamilyLoadConvexTriangle(convex, vertices, indices, index, triangle);
        closestPointOnTriangle(triangle, center, candidate, u, v);
        double delta[3] = {};
        cudaFamilySubtract3(center, candidate, delta);
        const double distance = dotVector(delta, delta);
        if (distance < closestDistance)
        {
            closestDistance = distance;
            cudaFamilyCopy3(candidate, closest);
            double first[3]  = {};
            double second[3] = {};
            cudaFamilySubtract3(triangle[1], triangle[0], first);
            cudaFamilySubtract3(triangle[2], triangle[0], second);
            cudaFamilyCross3(first, second, closestNormal);
            (void)cudaFamilyNormalize(closestNormal);
            if (distance > 0.0 && !cudaFamilyConvexContains(convex, planes, center))
            {
                cudaFamilyCopy3(delta, closestNormal);
                (void)cudaFamilyNormalize(closestNormal);
            }
        }
    }
    const bool inside = cudaFamilyConvexContains(convex, planes, center);
    if (!inside && closestDistance > radius * radius)
    {
        return false;
    }
    if (outClosest != nullptr)
    {
        cudaFamilyCopy3(closest, outClosest);
    }
    if (outNormal != nullptr)
    {
        cudaFamilyCopy3(closestNormal, outNormal);
    }
    return true;
}

__device__ bool cudaFamilyConvexRayHit(const cudaFamily::ConvexGeometry& convex, const double* planes,
                                       const double origin[3], const float directionSource[3], double minimumDistance,
                                       double maximumDistance, CudaFamilyHit& hit)
{
    double direction[3] = {
        static_cast<double>(directionSource[0]),
        static_cast<double>(directionSource[1]),
        static_cast<double>(directionSource[2]),
    };
    double             entry      = minimumDistance;
    double             exit       = maximumDistance;
    unsigned long long entryPlane = ~0ull;
    for (unsigned long long index = 0u; index < convex.planeCount; ++index)
    {
        const double* plane        = planes + (convex.planeOffset + index) * 4u;
        const double  signedOrigin = dotVector(plane, origin) - plane[3];
        const double  velocity     = dotVector(plane, direction);
        if (velocity == 0.0)
        {
            if (signedOrigin > 0.0)
            {
                return false;
            }
            continue;
        }
        const double distance = -signedOrigin / velocity;
        if (velocity < 0.0)
        {
            if (distance > entry)
            {
                entry      = distance;
                entryPlane = index;
            }
        }
        else
        {
            exit = fmin(exit, distance);
        }
        if (entry > exit)
        {
            return false;
        }
    }
    double point[3] = {};
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        point[axis] = fma(direction[axis], entry, origin[axis]);
    }
    if (entryPlane == ~0ull)
    {
        double smallestGap = 1.7976931348623157e+308;
        for (unsigned long long index = 0u; index < convex.planeCount; ++index)
        {
            const double* plane = planes + (convex.planeOffset + index) * 4u;
            const double  gap   = plane[3] - dotVector(plane, point);
            if (gap < smallestGap)
            {
                smallestGap = gap;
                entryPlane  = index;
            }
        }
        if (entryPlane == ~0ull || !isfinite(smallestGap))
        {
            return false;
        }
        const double* plane = planes + (convex.planeOffset + entryPlane) * 4u;
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            point[axis] = fma(plane[axis], smallestGap, point[axis]);
        }
    }
    const double* plane = planes + (convex.planeOffset + entryPlane) * 4u;
    hit.distance        = entry;
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        hit.position[axis] = point[axis];
        hit.normal[axis]   = static_cast<float>(plane[axis]);
    }
    return true;
}

__device__ bool cudaFamilyConvexSweepHit(const cudaFamily::ConvexGeometry& convex, const double* vertices,
                                         const unsigned int* indices, const double* planes, const double center[3],
                                         double radius, const float direction[3], double minimumDistance,
                                         double maximumDistance, CudaFamilyHit& hit)
{
    double start[3] = {};
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        start[axis] = fma(static_cast<double>(direction[axis]), minimumDistance, center[axis]);
    }
    double closest[3] = {};
    double normal[3]  = {};
    if (cudaFamilyConvexSphereOverlap(convex, vertices, indices, planes, start, radius, closest, normal))
    {
        hit.distance = minimumDistance;
        cudaFamilyCopy3(closest, hit.position);
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            hit.normal[axis] = static_cast<float>(normal[axis]);
        }
        return true;
    }
    bool               found = false;
    TriangleHit        closestTriangle{};
    unsigned long long bestTriangle = 0u;
    for (unsigned long long index = 0u; index < convex.triangleCount; ++index)
    {
        double      triangle[3][3] = {};
        TriangleHit candidate{};
        cudaFamilyLoadConvexTriangle(convex, vertices, indices, index, triangle);
        if (sweepTriangle(center, radius, direction, minimumDistance,
                          found ? closestTriangle.distance : maximumDistance, triangle, true, true,
                          static_cast<unsigned int>(index), candidate) &&
            (!found || candidate.distance < closestTriangle.distance ||
             (candidate.distance == closestTriangle.distance && index < bestTriangle)))
        {
            found           = true;
            closestTriangle = candidate;
            bestTriangle    = index;
        }
    }
    if (!found)
    {
        return false;
    }
    hit.distance = closestTriangle.distance;
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        hit.position[axis] = closestTriangle.position[axis];
        hit.normal[axis]   = closestTriangle.normal[axis];
    }
    return true;
}

__device__ bool cudaFamilyConvexSeparated(const cudaFamily::ConvexGeometry& convex, const double* vertices,
                                          const cudaFamily::BoxGeometry& box, const double axis[3])
{
    double minimum = 1.7976931348623157e+308;
    double maximum = -1.7976931348623157e+308;
    for (unsigned long long index = 0u; index < convex.vertexCount; ++index)
    {
        const double projection = dotVector(axis, vertices + (convex.vertexOffset + index) * 3u);
        minimum                 = fmin(minimum, projection);
        maximum                 = fmax(maximum, projection);
    }
    const double center = dotVector(axis, box.center);
    double       radius = 0.0;
    for (unsigned int local = 0u; local < 3u; ++local)
    {
        radius = fma(fabs(dotVector(axis, box.axes + local * 3u)), box.halfExtents[local], radius);
    }
    return maximum < center - radius || minimum > center + radius;
}

__device__ bool cudaFamilyConvexBoxOverlap(const cudaFamily::ConvexGeometry& convex, const double* vertices,
                                           const double* planes, const unsigned int* edges,
                                           const cudaFamily::BoxGeometry& box)
{
    for (unsigned long long index = 0u; index < convex.planeCount; ++index)
    {
        const double* plane = planes + (convex.planeOffset + index) * 4u;
        if (cudaFamilyConvexSeparated(convex, vertices, box, plane))
        {
            return false;
        }
    }
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        if (cudaFamilyConvexSeparated(convex, vertices, box, box.axes + axis * 3u))
        {
            return false;
        }
    }
    for (unsigned long long index = 0u; index < convex.edgeCount; ++index)
    {
        const unsigned int* edge         = edges + (convex.edgeOffset + index) * 2u;
        double              direction[3] = {};
        cudaFamilySubtract3(vertices + (convex.vertexOffset + edge[1]) * 3u,
                            vertices + (convex.vertexOffset + edge[0]) * 3u, direction);
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            double crossAxis[3] = {};
            cudaFamilyCross3(direction, box.axes + axis * 3u, crossAxis);
            if (dotVector(crossAxis, crossAxis) != 0.0 && cudaFamilyConvexSeparated(convex, vertices, box, crossAxis))
            {
                return false;
            }
        }
    }
    return true;
}

// Node indices on the stack are relative to the definition's node offset. A path
// traversal pops up to a warp of nodes per step; once the stack holds
// kMeshStackSoftLimit entries it pops one node per step, so a depth-first descent
// (at most 62 levels, see addBvhNodeCount) stays within kMeshStackCapacity. A sphere
// sweep queues its exact tests in the tail of the same array and switches to one node
// per step at kSphereStackSoftLimit, so no kernel's shared memory grows for the queue.
// A box sweep queues in its own BoxSweepQueue of kBoxQueueCapacity entries.
enum : unsigned int
{
    kMeshStackCapacity    = 512u,
    kMeshStackSoftLimit   = 448u,
    kSphereQueueCapacity  = 2u * cudaFamily::kEvaluateThreads,
    kBoxQueueCapacity     = 2u * cudaFamily::kEvaluateThreads,
    kSphereStackCapacity  = kMeshStackCapacity - 2u * kSphereQueueCapacity,
    kSphereStackSoftLimit = kSphereStackCapacity - 64u,
};

struct MeshBvhTraversal
{
    union
    {
        unsigned int stack[kMeshStackCapacity];
        struct
        {
            unsigned int stack[kSphereStackCapacity];
            unsigned int triangle[kSphereQueueCapacity]; // triangles waiting for the exact test
            float        entry[kSphereQueueCapacity];    // and the least parameter at which each can touch
        } sphere;
    };
    unsigned int       stackCount;
    unsigned int       leafCount;
    unsigned long long leafFirst[cudaFamily::kEvaluateThreads];       // first traversal triangle per popped leaf
    unsigned int       leafPrefix[cudaFamily::kEvaluateThreads + 1u]; // exclusive prefix of their triangle counts
    double             bestDistance;
    unsigned int       done;
};

__device__ void cudaFamilyMeshLocalPoint(const cudaFamily::MeshGeometry& mesh, const double world[3], double local[3])
{
    const double translated[3] = {world[0] - mesh.worldFromLocal[3], world[1] - mesh.worldFromLocal[7],
                                  world[2] - mesh.worldFromLocal[11]};
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        const double* row = mesh.localFromWorldLinear + axis * 3u;
        local[axis]       = fma(row[0], translated[0], fma(row[1], translated[1], row[2] * translated[2]));
    }
}

__device__ void cudaFamilyMeshLocalVector(const cudaFamily::MeshGeometry& mesh, const float world[3], double local[3])
{
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        const double* row = mesh.localFromWorldLinear + axis * 3u;
        local[axis]       = fma(row[0], static_cast<double>(world[0]),
                                fma(row[1], static_cast<double>(world[1]), row[2] * static_cast<double>(world[2])));
    }
}

__device__ void cudaFamilyMeshLocalExpansion(const cudaFamily::MeshGeometry& mesh, double radius, double expansion[3])
{
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        const double* row = mesh.localFromWorldLinear + axis * 3u;
        expansion[axis]   = radius * sqrt(fma(row[0], row[0], fma(row[1], row[1], row[2] * row[2])));
    }
}

__device__ void cudaFamilyMeshLocalQueryBounds(const cudaFamily::MeshGeometry& mesh, unsigned int overlapKind,
                                               const double translation[3], double sphereRadius,
                                               const cudaFamily::BoxGeometry& queryBox, double minimum[3],
                                               double maximum[3])
{
    double center[3] = {};
    cudaFamilyMeshLocalPoint(mesh, translation, center);
    if (overlapKind == cudaFamily::kOverlapSphere)
    {
        double expansion[3] = {};
        cudaFamilyMeshLocalExpansion(mesh, sphereRadius, expansion);
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            minimum[axis] = center[axis] - expansion[axis];
            maximum[axis] = center[axis] + expansion[axis];
        }
        return;
    }
    for (unsigned int localAxis = 0u; localAxis < 3u; ++localAxis)
    {
        const double* inverseRow = mesh.localFromWorldLinear + localAxis * 3u;
        double        extent     = 0.0;
        for (unsigned int boxAxis = 0u; boxAxis < 3u; ++boxAxis)
        {
            const double* worldAxis = queryBox.axes + boxAxis * 3u;
            const double  projection =
                fma(inverseRow[0], worldAxis[0], fma(inverseRow[1], worldAxis[1], inverseRow[2] * worldAxis[2]));
            extent = fma(fabs(projection), queryBox.halfExtents[boxAxis], extent);
        }
        minimum[localAxis] = center[localAxis] - extent;
        maximum[localAxis] = center[localAxis] + extent;
    }
}

__device__ bool cudaFamilyMeshBoundsOverlap(const cudaFamily::MeshBvhNode& node, const double minimum[3],
                                            const double maximum[3])
{
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        if (maximum[axis] < static_cast<double>(node.minimum[axis]) ||
            minimum[axis] > static_cast<double>(node.maximum[axis]))
        {
            return false;
        }
    }
    return true;
}

// Conservative single-precision culling in front of the exact predicates. The exact
// tests below stay in double and decide every reported contact; these tests only
// decide what they never need to see, on the 1/64-rate double pipe of consumer and
// workstation GPUs. Every float quantity carries an allowance of kCullRelative times
// the magnitudes of the terms that produced it, about 128 float ULPs against chains
// of a dozen roundings, and a test that cannot decide conservatively passes its
// node or triangle on. Nothing the double predicate accepts is culled.
constexpr float kCullRelative = 1.52587890625e-05f; // 2^-16

struct CullRay
{
    float origin[3];
    float direction[3];
    float inverseDirection[3];
    float expansion[3];
    float minimumDistance;
    bool  parallel[3]; // the double component is exactly zero or its inverse overflows a float
};

__device__ void cudaFamilyMakeCullRay(const double origin[3], const double direction[3], const double expansion[3],
                                      double minimumDistance, CullRay& ray)
{
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        ray.origin[axis]           = static_cast<float>(origin[axis]);
        ray.direction[axis]        = static_cast<float>(direction[axis]);
        const float inverse        = static_cast<float>(direction[axis] == 0.0 ? 0.0 : 1.0 / direction[axis]);
        ray.inverseDirection[axis] = inverse;
        ray.parallel[axis]         = direction[axis] == 0.0 || !isfinite(inverse);
        ray.expansion[axis]        = __double2float_ru(expansion[axis]);
    }
    ray.minimumDistance = __double2float_rd(minimumDistance);
}

// Narrows [entry, exit] to the float slab over bounds expanded by the ray's radius;
// false when the double slab test over the same bounds could accept no path in it.
__device__ bool cudaFamilyNarrowSlab(const CullRay& ray, const float boundsMinimum[3], const float boundsMaximum[3],
                                     float& entry, float& exit)
{
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        const float magnitude = fabsf(ray.origin[axis]) +
                                fmaxf(fabsf(boundsMinimum[axis]), fabsf(boundsMaximum[axis])) + ray.expansion[axis];
        const float allowance = kCullRelative * magnitude;
        const float minimum   = boundsMinimum[axis] - ray.expansion[axis] - allowance;
        const float maximum   = boundsMaximum[axis] + ray.expansion[axis] + allowance;
        if (ray.parallel[axis])
        {
            if (ray.origin[axis] < minimum || ray.origin[axis] > maximum)
            {
                return false;
            }
            continue;
        }
        const float first  = (minimum - ray.origin[axis]) * ray.inverseDirection[axis];
        const float second = (maximum - ray.origin[axis]) * ray.inverseDirection[axis];
        float       near   = fminf(first, second);
        float       far    = fmaxf(first, second);
        near -= kCullRelative * fabsf(near);
        far += kCullRelative * fabsf(far);
        entry = fmaxf(entry, near);
        exit  = fminf(exit, far);
        if (entry > exit)
        {
            return false;
        }
    }
    return true;
}

// True whenever the double slab test over the same bounds could accept a path within [minimum, maximum].
__device__ bool cudaFamilyCullSlab(const CullRay& ray, const float boundsMinimum[3], const float boundsMaximum[3],
                                   double maximumDistance)
{
    float entry = ray.minimumDistance;
    float exit  = __double2float_ru(maximumDistance);
    return cudaFamilyNarrowSlab(ray, boundsMinimum, boundsMaximum, entry, exit);
}

__device__ float cudaFamilyCullDot(const float first[3], const float second[3])
{
    return fmaf(first[0], second[0], fmaf(first[1], second[1], first[2] * second[2]));
}

// True whenever rayTriangle could accept this triangle of the definition, evaluated
// in the mesh's local float coordinates (the ray parameter is the same under the
// affine placement). Mirrors the exact predicate's normalization; the allowances
// follow the magnitudes of each dot product's terms, including the origin's own
// rounding relative to the triangle centre, so a far origin or a tiny triangle widens
// the acceptance instead of narrowing it.
__device__ bool cudaFamilyCullRayTriangle(const cudaFamily::MeshDefinition& definition, unsigned long long triangle,
                                          const CullRay& ray, double maximumDistance, bool doubleSided)
{
    const float*        vertices = reinterpret_cast<const float*>(definition.vertices);
    const unsigned int* indices  = reinterpret_cast<const unsigned int*>(definition.triangles);
    float               corners[3][3];
    float               center[3];
    float               scale = 0.0f;
    for (unsigned int corner = 0u; corner < 3u; ++corner)
    {
        const float* local = vertices + indices[triangle * 3u + corner] * 3u;
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            corners[corner][axis] = local[axis];
        }
    }
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        const float minimum = fminf(corners[0][axis], fminf(corners[1][axis], corners[2][axis]));
        const float maximum = fmaxf(corners[0][axis], fmaxf(corners[1][axis], corners[2][axis]));
        center[axis]        = 0.5f * minimum + 0.5f * maximum;
        for (unsigned int corner = 0u; corner < 3u; ++corner)
        {
            corners[corner][axis] -= center[axis];
            scale = fmaxf(scale, fabsf(corners[corner][axis]));
        }
    }
    if (!(scale > 0.0f) || !isfinite(scale))
    {
        return true;
    }
    const float inverseScale = 1.0f / scale;
    float       firstEdge[3], secondEdge[3], originDelta[3], direction[3], originMagnitude[3];
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        for (unsigned int corner = 0u; corner < 3u; ++corner)
        {
            corners[corner][axis] *= inverseScale;
        }
        firstEdge[axis]   = corners[1][axis] - corners[0][axis];
        secondEdge[axis]  = corners[2][axis] - corners[0][axis];
        originDelta[axis] = (ray.origin[axis] - center[axis]) * inverseScale - corners[0][axis];
        direction[axis]   = ray.direction[axis] * inverseScale;
        // |delta| + 1 covers the corner; the origin and centre magnitudes cover their
        // subtraction's rounding, which is relative to them and not to the difference.
        originMagnitude[axis] =
            fabsf(originDelta[axis]) + 1.0f + (fabsf(ray.origin[axis]) + fabsf(center[axis])) * inverseScale;
    }
    float crossDirectionSecond[3], crossMagnitude[3];
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        const unsigned int next     = axis == 2u ? 0u : axis + 1u;
        const unsigned int previous = axis == 0u ? 2u : axis - 1u;
        crossDirectionSecond[axis]  = direction[next] * secondEdge[previous] - direction[previous] * secondEdge[next];
        crossMagnitude[axis] =
            fabsf(direction[next]) * fabsf(secondEdge[previous]) + fabsf(direction[previous]) * fabsf(secondEdge[next]);
    }
    float       determinant          = cudaFamilyCullDot(firstEdge, crossDirectionSecond);
    const float absFirst[3]          = {fabsf(firstEdge[0]), fabsf(firstEdge[1]), fabsf(firstEdge[2])};
    const float determinantAllowance = kCullRelative * cudaFamilyCullDot(absFirst, crossMagnitude);
    if (!(fabsf(determinant) > determinantAllowance))
    {
        return true; // too close to degenerate or parallel to decide in float
    }
    const float sign = determinant < 0.0f ? -1.0f : 1.0f;
    if (!doubleSided && sign < 0.0f)
    {
        return false; // the exact predicate rejects a negative determinant of a single-sided face
    }
    determinant *= sign;
    const float numeratorU = sign * cudaFamilyCullDot(originDelta, crossDirectionSecond);
    const float allowanceU = kCullRelative * cudaFamilyCullDot(originMagnitude, crossMagnitude);
    if (numeratorU + allowanceU < 0.0f || numeratorU - allowanceU > determinant + determinantAllowance)
    {
        return false;
    }
    float crossOriginFirst[3], crossOriginMagnitude[3];
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        const unsigned int next     = axis == 2u ? 0u : axis + 1u;
        const unsigned int previous = axis == 0u ? 2u : axis - 1u;
        crossOriginFirst[axis]      = originDelta[next] * firstEdge[previous] - originDelta[previous] * firstEdge[next];
        crossOriginMagnitude[axis] =
            originMagnitude[next] * absFirst[previous] + originMagnitude[previous] * absFirst[next];
    }
    const float absDirection[3] = {fabsf(direction[0]), fabsf(direction[1]), fabsf(direction[2])};
    const float numeratorV      = sign * cudaFamilyCullDot(direction, crossOriginFirst);
    const float allowanceV      = kCullRelative * cudaFamilyCullDot(absDirection, crossOriginMagnitude);
    if (numeratorV + allowanceV < 0.0f ||
        numeratorU + numeratorV - allowanceU - allowanceV > determinant + determinantAllowance)
    {
        return false;
    }
    const float absSecond[3]    = {fabsf(secondEdge[0]), fabsf(secondEdge[1]), fabsf(secondEdge[2])};
    const float numeratorT      = sign * cudaFamilyCullDot(secondEdge, crossOriginFirst);
    const float allowanceT      = kCullRelative * cudaFamilyCullDot(absSecond, crossOriginMagnitude);
    const float lowDeterminant  = determinant - determinantAllowance; // positive: see the gate above
    const float highDeterminant = determinant + determinantAllowance;
    const float lowNumerator    = numeratorT - allowanceT;
    const float highNumerator   = numeratorT + allowanceT;
    const float lowT            = fminf(lowNumerator / highDeterminant, lowNumerator / lowDeterminant);
    const float highT           = fmaxf(highNumerator / highDeterminant, highNumerator / lowDeterminant);
    return !(highT < ray.minimumDistance || lowT > __double2float_ru(maximumDistance));
}

// Float bounds on boxTriangle's first three separating axes, the query box's face
// axes A_i, over one mesh traversal: the CPU's ordered-refinement bound. Mesh-local
// vertices project onto A_i through the rows A_i W, so nodes and triangles are tested
// without their world transform. Axes, extents and velocities are the double values
// boxTriangle computes. With S_i bounding the magnitude of a projection's terms over
// the mesh (the placement's linear part on the root bounds, plus the query centre
// relative to the placement), boxTriangle's double projections and these float ones
// differ by a few float roundoffs of S_i, far inside the kCullRelative allowance
// (256 of them) that widens every interval. An axis whose velocity is zero only tests
// static separation, as boxTriangle does. An axis gives no bound when its inverse
// velocity overflows a float (boxTriangle still moves the box along it) or when its
// magnitudes leave the range where float projections neither overflow nor go
// subnormal. What is culled is either rejected by boxTriangle or enters beyond the clip.
struct BoxSweepCull
{
    float        row[3][3];
    float        offset[3];
    float        reach[3]; // extent plus allowance
    float        inverse[3];
    unsigned int active; // bit i: face axis i bounds the test
    unsigned int moving; // bit i: it bounds times; otherwise it tests static separation
};

__device__ void cudaFamilyMakeBoxSweepCull(const boxSweep::Box& relativeQuery, boxSweep::Vec3 direction,
                                           const double worldFromLocal[12], const float rootMinimum[3],
                                           const float rootMaximum[3], BoxSweepCull& cull)
{
    cull            = BoxSweepCull{};
    double reach[3] = {};
    for (unsigned int local = 0u; local < 3u; ++local)
    {
        reach[local] =
            fmax(fabs(static_cast<double>(rootMinimum[local])), fabs(static_cast<double>(rootMaximum[local])));
    }
    for (unsigned int face = 0u; face < 3u; ++face)
    {
        const boxSweep::Vec3 axis = boxSweep::scaledAxis(relativeQuery.axis[face]);
        if (boxSweep::dot(axis, axis) == 0.0)
        {
            continue;
        }
        const double extent    = boxSweep::radius(relativeQuery, axis);
        const double velocity  = boxSweep::dot(axis, direction);
        double       magnitude = 0.0;
        bool         finite    = true;
        for (unsigned int world = 0u; world < 3u; ++world)
        {
            double placed = fabs(relativeQuery.center[world]);
            for (unsigned int local = 0u; local < 3u; ++local)
            {
                placed += fabs(worldFromLocal[world * 4u + local]) * reach[local];
            }
            magnitude += fabs(axis[world]) * placed;
        }
        for (unsigned int local = 0u; local < 3u; ++local)
        {
            cull.row[face][local] =
                static_cast<float>(axis.x * worldFromLocal[local] + axis.y * worldFromLocal[4u + local] +
                                   axis.z * worldFromLocal[8u + local]);
            finite = finite && isfinite(cull.row[face][local]);
        }
        cull.offset[face]  = static_cast<float>(-boxSweep::dot(axis, relativeQuery.center));
        cull.reach[face]   = __double2float_ru(extent + static_cast<double>(kCullRelative) * (magnitude + extent));
        cull.inverse[face] = velocity == 0.0 ? 0.0f : static_cast<float>(1.0 / velocity);
        const double scale = magnitude + extent;
        if (finite && isfinite(cull.offset[face]) && scale >= 0x1p-100 && scale <= 0x1p100 &&
            (velocity == 0.0 || isfinite(cull.inverse[face])))
        {
            cull.active |= 1u << face;
            if (velocity != 0.0)
            {
                cull.moving |= 1u << face;
            }
        }
    }
}

// Narrows [entry, exit] by the face axes, given every point's projections within
// [low, high]; false when boxTriangle rejects every such triangle within them.
__device__ bool cudaFamilyNarrowBoxSweepFaces(const BoxSweepCull& cull, const float low[3], const float high[3],
                                              float& entry, float& exit)
{
    for (unsigned int face = 0u; face < 3u; ++face)
    {
        if ((cull.active & (1u << face)) == 0u)
        {
            continue;
        }
        const float lower = low[face] - cull.reach[face];
        const float upper = high[face] + cull.reach[face];
        if ((cull.moving & (1u << face)) == 0u)
        {
            if (lower > 0.0f || upper < 0.0f)
            {
                return false;
            }
            continue;
        }
        const float first  = lower * cull.inverse[face];
        const float second = upper * cull.inverse[face];
        float       near   = fminf(first, second);
        float       far    = fmaxf(first, second);
        near -= kCullRelative * fabsf(near);
        far += kCullRelative * fabsf(far);
        entry = fmaxf(entry, near);
        exit  = fminf(exit, far);
    }
    return entry <= exit;
}

// A node's local bounds projected on the rows of a sweep cull: low and high
// bound row . p + offset over every point p in the node.
__device__ void cudaFamilyProjectNode(const float row[3][3], const float offset[3], const cudaFamily::MeshBvhNode& node,
                                      float low[3], float high[3])
{
    for (unsigned int k = 0u; k < 3u; ++k)
    {
        low[k]  = offset[k];
        high[k] = offset[k];
        for (unsigned int local = 0u; local < 3u; ++local)
        {
            const float value = row[k][local];
            low[k]            = fmaf(value, value >= 0.0f ? node.minimum[local] : node.maximum[local], low[k]);
            high[k]           = fmaf(value, value >= 0.0f ? node.maximum[local] : node.minimum[local], high[k]);
        }
    }
}

// A node's local bounds against the local slab of the box's circumsphere and its
// face axes; `entry` is a lower bound on any contact within the node.
__device__ bool cudaFamilyCullBoxSweepNode(const CullRay& ray, const BoxSweepCull& cull,
                                           const cudaFamily::MeshBvhNode& node, float clip, float& entry)
{
    entry      = ray.minimumDistance;
    float exit = clip;
    if (!cudaFamilyNarrowSlab(ray, node.minimum, node.maximum, entry, exit))
    {
        return false;
    }
    float low[3], high[3];
    cudaFamilyProjectNode(cull.row, cull.offset, node, low, high);
    return cudaFamilyNarrowBoxSweepFaces(cull, low, high, entry, exit);
}

// One triangle's local vertices on the face axes; the node tests have already
// applied the local slab, which costs a triangle about as much as it culls.
__device__ bool cudaFamilyCullBoxSweepTriangle(const BoxSweepCull& cull, const cudaFamily::MeshDefinition& definition,
                                               unsigned long long triangle, float minimum, float clip, float& entry)
{
    const float*        vertices = reinterpret_cast<const float*>(definition.vertices);
    const unsigned int* indices  = reinterpret_cast<const unsigned int*>(definition.triangles);
    float               low[3], high[3];
    for (unsigned int corner = 0u; corner < 3u; ++corner)
    {
        const float* local = vertices + indices[triangle * 3u + corner] * 3u;
        for (unsigned int face = 0u; face < 3u; ++face)
        {
            const float projection =
                fmaf(cull.row[face][0], local[0],
                     fmaf(cull.row[face][1], local[1], fmaf(cull.row[face][2], local[2], cull.offset[face])));
            low[face]  = corner == 0u ? projection : fminf(low[face], projection);
            high[face] = corner == 0u ? projection : fmaxf(high[face], projection);
        }
    }
    entry      = minimum;
    float exit = clip;
    return cudaFamilyNarrowBoxSweepFaces(cull, low, high, entry, exit);
}

// Float bounds of a swept sphere in the frame of its path, the sphere's counterpart of
// the box's face axes: e0 along the direction, e1 and e2 across it. A point at s along
// e0 and h from the path's axis is within R of the center c + t d only while
// h <= R and |d| t >= s - sqrt(R^2 - h^2), where R is the largest center distance
// sweepTriangle accepts (the radius and its squared-distance tolerance). A triangle or
// node therefore needs no exact test unless its projection across the path comes within
// R of the axis, and it touches the sphere no earlier than its least s less the half
// chord at that distance. Mesh-local vertices project through the rows e_k W, so any
// affine placement keeps the bound valid. Coordinates are scaled by the power of two
// that brings S + R into [1, 2), with S bounding every projection's terms over the mesh
// (the root bounds placed relative to the center); `slack` is kCullRelative of that,
// 256 float roundoffs against the dozen of a projection, the axis distance and the half
// chord, plus the double rounding of the world triangles sweepTriangle reads, which
// follows their magnitude when mesh and sphere are both far from the origin. What is
// culled is either rejected by sweepTriangle or touches beyond the clip.
struct SphereSweepCull
{
    float row[3][3];
    float offset[3];
    float reach;     // R, scaled
    float slack;     // allowance on every scaled projection and axis distance
    float parameter; // path parameter per scaled unit along e0
    bool  active;
};

__device__ void cudaFamilyMakeSphereSweepCull(const double center[3], const float direction[3], double radius,
                                              const double worldFromLocal[12], const float rootMinimum[3],
                                              const float rootMaximum[3], SphereSweepCull& cull)
{
    cull = SphereSweepCull{};
    double axes[3][3]{};
    double length = 0.0;
    for (unsigned int world = 0u; world < 3u; ++world)
    {
        axes[0][world] = static_cast<double>(direction[world]);
        length         = fma(axes[0][world], axes[0][world], length);
    }
    length = sqrt(length);
    // e1 is perpendicular to e0 and to the world axis least aligned with it; e2 completes the frame.
    unsigned int least = 0u;
    for (unsigned int world = 0u; world < 3u; ++world)
    {
        axes[0][world] /= length;
        least = fabs(axes[0][world]) < fabs(axes[0][least]) ? world : least;
    }
    const unsigned int next = least == 2u ? 0u : least + 1u, previous = least == 0u ? 2u : least - 1u;
    const double       perpendicular = hypot(axes[0][next], axes[0][previous]);
    axes[1][next]                    = axes[0][previous] / perpendicular;
    axes[1][previous]                = -axes[0][next] / perpendicular;
    for (unsigned int world = 0u; world < 3u; ++world)
    {
        const unsigned int second = world == 2u ? 0u : world + 1u, third = world == 0u ? 2u : world - 1u;
        axes[2][world] = axes[0][second] * axes[1][third] - axes[0][third] * axes[1][second];
    }
    double reach[3] = {};
    for (unsigned int local = 0u; local < 3u; ++local)
    {
        reach[local] =
            fmax(fabs(static_cast<double>(rootMinimum[local])), fabs(static_cast<double>(rootMaximum[local])));
    }
    double relative[3] = {}, placed[3] = {}, worldMagnitude = 0.0;
    for (unsigned int world = 0u; world < 3u; ++world)
    {
        relative[world] = worldFromLocal[world * 4u + 3u] - center[world];
        double linear   = 0.0;
        for (unsigned int local = 0u; local < 3u; ++local)
        {
            linear += fabs(worldFromLocal[world * 4u + local]) * reach[local];
        }
        placed[world]  = fabs(relative[world]) + linear;
        worldMagnitude = fmax(worldMagnitude, fabs(worldFromLocal[world * 4u + 3u]) + linear);
    }
    double magnitude = 0.0, row[3][3] = {}, offset[3] = {};
    for (unsigned int k = 0u; k < 3u; ++k)
    {
        double axisMagnitude = 0.0;
        for (unsigned int world = 0u; world < 3u; ++world)
        {
            axisMagnitude += fabs(axes[k][world]) * placed[world];
            offset[k] = fma(axes[k][world], relative[world], offset[k]);
            for (unsigned int local = 0u; local < 3u; ++local)
            {
                row[k][local] = fma(axes[k][world], worldFromLocal[world * 4u + local], row[k][local]);
            }
        }
        magnitude = fmax(magnitude, axisMagnitude);
    }
    const double squared    = radius * radius;
    const double acceptance = sqrt(squared + fmax(1.0, squared) * kDoubleEpsilon * 256.0);
    const double total      = magnitude + acceptance;
    if (!isfinite(total) || !isfinite(worldMagnitude) || !(length > 0.0) || !(perpendicular > 0.0))
    {
        return;
    }
    const double scale  = inversePowerOfTwo(total);
    bool         finite = total * scale >= 1.0 && total * scale < 2.0;
    for (unsigned int k = 0u; k < 3u; ++k)
    {
        cull.offset[k] = static_cast<float>(offset[k] * scale);
        finite         = finite && isfinite(cull.offset[k]);
        for (unsigned int local = 0u; local < 3u; ++local)
        {
            cull.row[k][local] = static_cast<float>(row[k][local] * scale);
            finite             = finite && isfinite(cull.row[k][local]);
        }
    }
    cull.reach = __double2float_ru(acceptance * scale);
    cull.slack =
        __double2float_ru((static_cast<double>(kCullRelative) * total + 8.0 * kDoubleEpsilon * worldMagnitude) * scale);
    cull.parameter = static_cast<float>(1.0 / (scale * length));
    cull.active    = finite && isfinite(cull.slack) && isfinite(cull.parameter) && cull.parameter > 0.0f;
}

// Narrows [entry, exit] to the parameters at which a sphere can touch an item whose
// scaled projections along the path lie in [low, high] and whose distance from the
// path's axis is at least `across` less the slack; false when it cannot touch within them.
__device__ bool cudaFamilyNarrowSphereSweep(const SphereSweepCull& cull, float low, float high, float across,
                                            float& entry, float& exit)
{
    const float gap = fmaxf(0.0f, across - cull.slack);
    if (gap > cull.reach)
    {
        return false;
    }
    const float chord = sqrtf((cull.reach - gap) * (cull.reach + gap));
    float       near  = (low - cull.slack - chord) * cull.parameter;
    float       far   = (high + cull.slack + chord) * cull.parameter;
    near -= kCullRelative * fabsf(near);
    far += kCullRelative * fabsf(far);
    entry = fmaxf(entry, near);
    exit  = fminf(exit, far);
    return entry <= exit;
}

// A node's local bounds against the local slab of the sphere and its path frame;
// `entry` is a lower bound on any contact within the node.
__device__ bool cudaFamilyCullSphereSweepNode(const CullRay& ray, const SphereSweepCull& cull,
                                              const cudaFamily::MeshBvhNode& node, float clip, float& entry)
{
    entry      = ray.minimumDistance;
    float exit = clip;
    if (!cudaFamilyNarrowSlab(ray, node.minimum, node.maximum, entry, exit))
    {
        return false;
    }
    if (!cull.active)
    {
        return true;
    }
    float low[3], high[3];
    cudaFamilyProjectNode(cull.row, cull.offset, node, low, high);
    const float acrossX = fmaxf(0.0f, fmaxf(low[1], -high[1]));
    const float acrossY = fmaxf(0.0f, fmaxf(low[2], -high[2]));
    return cudaFamilyNarrowSphereSweep(cull, low[0], high[0], sqrtf(acrossX * acrossX + acrossY * acrossY), entry,
                                       exit);
}

// Distance from the path's axis to a triangle's projection across it: zero when the axis
// passes through the projection, else the least distance to its edges. Each edge is
// taken relative to its start, so an axis within a few roundoffs of an edge may be
// judged outside, and then its edge distance is as small. An edge shorter than 2^-40
// (the coordinates are scaled near one) counts as its start, within the slack of it.
__device__ float cudaFamilyAxisTriangleDistance(const float x[3], const float y[3])
{
    float nearest  = 3.402823466e+38f;
    bool  positive = true;
    bool  negative = true;
    for (unsigned int corner = 0u; corner < 3u; ++corner)
    {
        const unsigned int next    = corner == 2u ? 0u : corner + 1u;
        const float        edgeX   = x[next] - x[corner];
        const float        edgeY   = y[next] - y[corner];
        const float        side    = edgeX * -y[corner] - edgeY * -x[corner];
        const float        along   = edgeX * -x[corner] + edgeY * -y[corner];
        const float        squared = edgeX * edgeX + edgeY * edgeY;
        positive                   = positive && side >= 0.0f;
        negative                   = negative && side <= 0.0f;
        float distance             = x[corner] * x[corner] + y[corner] * y[corner];
        if (along >= squared && squared > 0x1p-80f)
        {
            distance = x[next] * x[next] + y[next] * y[next];
        }
        else if (along > 0.0f && squared > 0x1p-80f)
        {
            distance = side * side / squared;
        }
        nearest = fminf(nearest, distance);
    }
    return positive || negative ? 0.0f : sqrtf(nearest);
}

// One triangle's local vertices in the path frame; `entry` bounds its first contact.
__device__ bool cudaFamilyCullSphereSweepTriangle(const SphereSweepCull&            cull,
                                                  const cudaFamily::MeshDefinition& definition,
                                                  unsigned long long triangle, float minimum, float clip, float& entry)
{
    entry = minimum;
    if (!cull.active)
    {
        return true;
    }
    const float*        vertices = reinterpret_cast<const float*>(definition.vertices);
    const unsigned int* indices  = reinterpret_cast<const unsigned int*>(definition.triangles);
    float               along[3], x[3], y[3];
    for (unsigned int corner = 0u; corner < 3u; ++corner)
    {
        const float* local = vertices + indices[triangle * 3u + corner] * 3u;
        float        projection[3];
        for (unsigned int k = 0u; k < 3u; ++k)
        {
            projection[k] = fmaf(cull.row[k][0], local[0],
                                 fmaf(cull.row[k][1], local[1], fmaf(cull.row[k][2], local[2], cull.offset[k])));
        }
        along[corner] = projection[0];
        x[corner]     = projection[1];
        y[corner]     = projection[2];
    }
    float exit = clip;
    return cudaFamilyNarrowSphereSweep(cull, fminf(along[0], fminf(along[1], along[2])),
                                       fmaxf(along[0], fmaxf(along[1], along[2])), cudaFamilyAxisTriangleDistance(x, y),
                                       entry, exit);
}

enum SingleTriangleResult : unsigned int
{
    kSingleTriangleMiss,
    kSingleTriangleHit,
    kSingleTriangleFallback,
};

__device__ float cudaFamilySingleDot(const float first[3], const float second[3])
{
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2];
}

// The reduced-precision mode evaluates authored float vertices in mesh-local
// coordinates. Values which cannot be normalized without float overflow,
// underflow, an ambiguous near-degenerate determinant, or a crossing within float
// rounding of an edge use the existing world-double predicate instead.
__device__ SingleTriangleResult cudaFamilySingleRayTriangle(const cudaFamily::MeshDefinition& definition,
                                                            unsigned long long triangleIndex, const float origin[3],
                                                            const float directionSource[3], double minimumDistance,
                                                            double maximumDistance, bool doubleSided,
                                                            double orientation, TriangleHit& output)
{
    constexpr float     kFloatMinimumNormal = 1.1754943508222875e-38f;
    constexpr float     kFloatEpsilon       = 1.1920928955078125e-7f;
    const float*        vertices            = reinterpret_cast<const float*>(definition.vertices);
    const unsigned int* indices             = reinterpret_cast<const unsigned int*>(definition.triangles);
    float               triangle[3][3]{};
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        for (unsigned int corner = 0u; corner < 3u; ++corner)
        {
            triangle[corner][axis] = vertices[indices[triangleIndex * 3u + corner] * 3u + axis];
            if (!isfinite(triangle[corner][axis]))
            {
                return kSingleTriangleFallback;
            }
        }
    }

    float normalized[3][3]{};
    float center[3]{};
    float scale = 0.0f;
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        float minimum = triangle[0][axis];
        float maximum = minimum;
        for (unsigned int corner = 1u; corner < 3u; ++corner)
        {
            minimum = fminf(minimum, triangle[corner][axis]);
            maximum = fmaxf(maximum, triangle[corner][axis]);
        }
        center[axis] = minimum * 0.5f + maximum * 0.5f;
    }
    for (unsigned int corner = 0u; corner < 3u; ++corner)
    {
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            normalized[corner][axis] = triangle[corner][axis] - center[axis];
            if (!isfinite(normalized[corner][axis]))
            {
                return kSingleTriangleFallback;
            }
            scale = fmaxf(scale, fabsf(normalized[corner][axis]));
        }
    }
    if (!(scale >= kFloatMinimumNormal) || !isfinite(scale))
    {
        return kSingleTriangleFallback;
    }
    const int exponent = (__float_as_int(scale) >> 23) & 0xff;
    if (exponent >= 254)
    {
        return kSingleTriangleFallback;
    }
    // An exact power-of-two scale keeps the ratios of the qualified predicate
    // while avoiding a general float division before the determinant.
    const float inverseScale = __int_as_float((254 - exponent) << 23);
    for (unsigned int corner = 0u; corner < 3u; ++corner)
    {
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            normalized[corner][axis] *= inverseScale;
        }
    }

    float firstEdge[3]{};
    float secondEdge[3]{};
    float originDelta[3]{};
    float direction[3]{};
    float crossDirectionSecond[3]{};
    bool  hasDirection = false;
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        firstEdge[axis]   = normalized[1][axis] - normalized[0][axis];
        secondEdge[axis]  = normalized[2][axis] - normalized[0][axis];
        originDelta[axis] = (origin[axis] - center[axis]) * inverseScale - normalized[0][axis];
        direction[axis]   = directionSource[axis] * inverseScale;
        if (!isfinite(originDelta[axis]) || !isfinite(direction[axis]))
        {
            return kSingleTriangleFallback;
        }
        hasDirection = hasDirection || direction[axis] != 0.0f;
    }
    if (!hasDirection)
    {
        return kSingleTriangleFallback;
    }
    crossDirectionSecond[0]          = direction[1] * secondEdge[2] - direction[2] * secondEdge[1];
    crossDirectionSecond[1]          = direction[2] * secondEdge[0] - direction[0] * secondEdge[2];
    crossDirectionSecond[2]          = direction[0] * secondEdge[1] - direction[1] * secondEdge[0];
    const float determinant          = cudaFamilySingleDot(firstEdge, crossDirectionSecond);
    const float determinantMagnitude = fabsf(firstEdge[0] * crossDirectionSecond[0]) +
                                       fabsf(firstEdge[1] * crossDirectionSecond[1]) +
                                       fabsf(firstEdge[2] * crossDirectionSecond[2]);
    const float tolerance = determinantMagnitude * kFloatEpsilon * 8.0f;
    if (!isfinite(determinant) || !isfinite(tolerance) || fabsf(determinant) <= tolerance)
    {
        return kSingleTriangleFallback;
    }
    if (!doubleSided && static_cast<double>(determinant) * orientation <= static_cast<double>(tolerance))
    {
        return kSingleTriangleMiss;
    }
    // Bounds on the rounding of u and v from the magnitudes of their terms, as in
    // cudaFamilyCullRayTriangle: |originDelta| + 1 covers the delta's subtraction, and the
    // origin and centre magnitudes cover the origin's rounding to float, which moves this ray
    // off the one the double predicate tests. Two triangles that share an edge round it
    // differently, so a row within these bounds of the margin goes to the double predicate,
    // which decides it the same way for both.
    float originMagnitude[3]{};
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        originMagnitude[axis] =
            fabsf(originDelta[axis]) + 1.0f + (fabsf(origin[axis]) + fabsf(center[axis])) * inverseScale;
    }
    float determinantTerms = 0.0f;
    float uTerms           = 0.0f;
    float vTerms           = 0.0f;
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        const unsigned int next     = axis == 2u ? 0u : axis + 1u;
        const unsigned int previous = axis == 0u ? 2u : axis - 1u;
        const float        crossMagnitude =
            fabsf(direction[next]) * fabsf(secondEdge[previous]) + fabsf(direction[previous]) * fabsf(secondEdge[next]);
        const float crossOriginMagnitude =
            originMagnitude[next] * fabsf(firstEdge[previous]) + originMagnitude[previous] * fabsf(firstEdge[next]);
        determinantTerms += fabsf(firstEdge[axis]) * crossMagnitude;
        uTerms += originMagnitude[axis] * crossMagnitude;
        vTerms += fabsf(direction[axis]) * crossOriginMagnitude;
    }
    const float roundingScale = 16.0f * kFloatEpsilon / fabsf(determinant);
    const float uRounding     = (uTerms + determinantTerms) * roundingScale;
    const float vRounding     = (vTerms + determinantTerms) * roundingScale;
    const float margin        = static_cast<float>(kTriangleEdgeMargin);
    if (!isfinite(uRounding) || !isfinite(vRounding))
    {
        return kSingleTriangleFallback;
    }
    const float inverseDeterminant = 1.0f / determinant;
    const float u                  = cudaFamilySingleDot(originDelta, crossDirectionSecond) * inverseDeterminant;
    if (!isfinite(u))
    {
        return kSingleTriangleFallback;
    }
    if (u < -margin - uRounding || u > 1.0f + margin + uRounding)
    {
        return kSingleTriangleMiss;
    }
    const float crossOriginFirst[3] = {
        originDelta[1] * firstEdge[2] - originDelta[2] * firstEdge[1],
        originDelta[2] * firstEdge[0] - originDelta[0] * firstEdge[2],
        originDelta[0] * firstEdge[1] - originDelta[1] * firstEdge[0],
    };
    const float v = cudaFamilySingleDot(direction, crossOriginFirst) * inverseDeterminant;
    if (!isfinite(v))
    {
        return kSingleTriangleFallback;
    }
    if (v < -margin - vRounding || u + v > 1.0f + margin + uRounding + vRounding)
    {
        return kSingleTriangleMiss;
    }
    if (u < -margin + uRounding || u > 1.0f + margin - uRounding || v < -margin + vRounding ||
        u + v > 1.0f + margin - uRounding - vRounding)
    {
        return kSingleTriangleFallback;
    }
    const float distance = cudaFamilySingleDot(secondEdge, crossOriginFirst) * inverseDeterminant;
    if (!isfinite(distance))
    {
        return kSingleTriangleFallback;
    }
    if (static_cast<double>(distance) < minimumDistance || static_cast<double>(distance) > maximumDistance)
    {
        // Let the double predicate decide inclusive interval boundaries when
        // float rounding moves the computed parameter beyond either bound.
        return kSingleTriangleFallback;
    }
    output.distance = static_cast<double>(distance);
    output.uv[0]    = u;
    output.uv[1]    = v;
    return kSingleTriangleHit;
}

// The lane holding the warp's winning triangle: the closest key, then the lowest
// triangle index, over the lanes that found one; -1 when none did. The winner's
// key and triangle are returned uniform across the warp.
__device__ int cudaFamilyWarpWinner(bool found, double key, unsigned long long triangle, double& winnerKey,
                                    unsigned long long& winnerTriangle)
{
    constexpr unsigned long long kNone     = ~0ull;
    constexpr unsigned int       kFullMask = 0xffffffffu;
    winnerKey                              = found ? key : 0.0;
    winnerTriangle                         = found ? triangle : kNone;
    for (unsigned int offset = blockDim.x / 2u; offset > 0u; offset >>= 1u)
    {
        const double             otherKey      = __shfl_xor_sync(kFullMask, winnerKey, offset);
        const unsigned long long otherTriangle = __shfl_xor_sync(kFullMask, winnerTriangle, offset);
        if (otherTriangle != kNone && (winnerTriangle == kNone || otherKey < winnerKey ||
                                       (otherKey == winnerKey && otherTriangle < winnerTriangle)))
        {
            winnerKey      = otherKey;
            winnerTriangle = otherTriangle;
        }
    }
    if (winnerTriangle == kNone)
    {
        return -1;
    }
    // Exactly one lane tested the winning triangle.
    return __ffs(__ballot_sync(kFullMask, found && triangle == winnerTriangle)) - 1;
}

// The source face, corners and barycentrics of a contact on a traversal triangle.
__device__ void cudaFamilySetTriangleSource(const cudaFamily::MeshDefinition& definition, unsigned long long triangle,
                                            float u, float v, CudaFamilyHit& hit)
{
    const unsigned long long* sourceFaces   = reinterpret_cast<const unsigned long long*>(definition.sourceFaces);
    const unsigned int*       sourceCorners = reinterpret_cast<const unsigned int*>(definition.sourceCorners);
    hit.triangleValid                       = true;
    hit.sourceFace                          = sourceFaces[triangle];
    hit.barycentrics[0]                     = 1.0f - u - v;
    hit.barycentrics[1]                     = u;
    hit.barycentrics[2]                     = v;
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        hit.sourceCorners[axis] = sourceCorners[triangle * 3u + axis];
    }
}

// Cooperative closest ray over the definition's local BVH. The query transform
// is double; nodes and leaf triangles are culled by the conservative float tests
// above, and only the triangles they pass are reconstructed in world doubles for
// the existing predicate. Every lane of the block tests one popped node per step
// and one candidate triangle at a time; the block is one warp (kEvaluateThreads),
// so the reductions use warp shuffles. The result does not depend on the traversal
// order: a triangle at the closest distance is never culled, and equal distances
// keep the lower triangle index.
__device__ bool cudaFamilyMeshPathHitPass(const cudaFamily::MeshGeometry&   mesh,
                                          const cudaFamily::MeshDefinition* definitions, bool doubleSided,
                                          bool flipBackFaces, bool exactSingle, const double origin[3],
                                          const float direction[3], double minimumDistance, double maximumDistance,
                                          MeshBvhTraversal* sharedTraversal, CudaFamilyHit& hit, bool& retryExact)
{
    constexpr unsigned long long      kNone      = ~0ull;
    constexpr unsigned int            kFullMask  = 0xffffffffu;
    const cudaFamily::MeshDefinition& definition = definitions[mesh.definitionIndex];
    const cudaFamily::MeshBvhNode*    meshNodes  = reinterpret_cast<const cudaFamily::MeshBvhNode*>(definition.nodes);
    const unsigned int* traversalTriangles       = reinterpret_cast<const unsigned int*>(definition.traversalTriangles);
    const unsigned int  lane                     = threadIdx.x;
    double              localOrigin[3]           = {};
    double              localDirection[3]        = {};
    double              localExpansion[3]        = {};
    cudaFamilyMeshLocalPoint(mesh, origin, localOrigin);
    cudaFamilyMeshLocalVector(mesh, direction, localDirection);
    bool   singleTransformSafe = true;
    double orientation         = 1.0;
    if (exactSingle)
    {
        const double* transform = mesh.worldFromLocal;
        const double  orientationDeterminant =
            transform[0] * (transform[5] * transform[10] - transform[6] * transform[9]) -
            transform[1] * (transform[4] * transform[10] - transform[6] * transform[8]) +
            transform[2] * (transform[4] * transform[9] - transform[5] * transform[8]);
        singleTransformSafe = isfinite(orientationDeterminant) && orientationDeterminant != 0.0;
        orientation         = orientationDeterminant < 0.0 ? -1.0 : 1.0;
    }
    CullRay cullRay{};
    cudaFamilyMakeCullRay(localOrigin, localDirection, localExpansion, minimumDistance, cullRay);
    bool singleRaySafe = true;
    if (exactSingle)
    {
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            singleRaySafe = singleRaySafe && isfinite(cullRay.origin[axis]) && isfinite(cullRay.direction[axis]) &&
                            (localDirection[axis] == 0.0 || cullRay.direction[axis] != 0.0f);
        }
    }
    retryExact = false;
    if (lane == 0u)
    {
        sharedTraversal->stack[0]     = 0u;
        sharedTraversal->stackCount   = 1u;
        sharedTraversal->bestDistance = maximumDistance;
    }
    __syncthreads();
    // Each lane keeps the best of the triangles it evaluated; the block reduces once at the end.
    bool               found = false;
    TriangleHit        best{};
    unsigned long long bestTriangle = kNone;
    for (;;)
    {
        const unsigned int count = sharedTraversal->stackCount;
        if (count == 0u)
        {
            break;
        }
        // Popping `take` nodes pushes at most 2 * take children, so the count grows by at
        // most `take`; the soft limit keeps the frontier within the stack, one node at a time
        // once it is nearly full.
        unsigned int take = count < blockDim.x ? count : blockDim.x;
        if (count >= kMeshStackSoftLimit)
        {
            take = 1u;
        }
        else if (take > kMeshStackSoftLimit - count)
        {
            take = kMeshStackSoftLimit - count;
        }
        const unsigned int base         = count - take;
        const double       clip         = sharedTraversal->bestDistance;
        bool               nodeHit      = false;
        bool               nodeLeaf     = false;
        unsigned long long nodeFirst    = 0u;
        unsigned int       nodeTriangle = 0u;
        if (lane < take)
        {
            const cudaFamily::MeshBvhNode& node = meshNodes[sharedTraversal->stack[base + lane]];
            nodeHit                             = cudaFamilyCullSlab(cullRay, node.minimum, node.maximum, clip);
            nodeLeaf                            = node.count != 0u;
            nodeFirst                           = node.first;
            nodeTriangle                        = node.count;
        }
        __syncthreads();
        const unsigned int interiorMask = __ballot_sync(kFullMask, nodeHit && !nodeLeaf);
        const unsigned int leafMask     = __ballot_sync(kFullMask, nodeHit && nodeLeaf);
        const unsigned int below        = (1u << lane) - 1u;
        if (nodeHit && !nodeLeaf)
        {
            // Children are stored as node indices local to the definition; the left child goes on top.
            const unsigned int rank                       = __popc(interiorMask & below);
            const unsigned int child                      = static_cast<unsigned int>(nodeFirst);
            sharedTraversal->stack[base + 2u * rank]      = child + 1u;
            sharedTraversal->stack[base + 2u * rank + 1u] = child;
        }
        if (nodeHit && nodeLeaf)
        {
            const unsigned int rank                = __popc(leafMask & below);
            sharedTraversal->leafFirst[rank]       = nodeFirst;
            sharedTraversal->leafPrefix[rank + 1u] = nodeTriangle;
        }
        __syncthreads();
        const unsigned int leaves = __popc(leafMask);
        if (lane == 0u)
        {
            sharedTraversal->stackCount    = base + 2u * __popc(interiorMask);
            sharedTraversal->leafPrefix[0] = 0u;
            for (unsigned int leaf = 1u; leaf <= leaves; ++leaf)
            {
                sharedTraversal->leafPrefix[leaf] += sharedTraversal->leafPrefix[leaf - 1u];
            }
        }
        __syncthreads();
        const unsigned int candidates = leaves == 0u ? 0u : sharedTraversal->leafPrefix[leaves];
        for (unsigned int slot = lane; slot < candidates; slot += blockDim.x)
        {
            unsigned int leaf = 0u;
            while (slot >= sharedTraversal->leafPrefix[leaf + 1u])
            {
                ++leaf;
            }
            const unsigned long long triangleIndex =
                traversalTriangles[sharedTraversal->leafFirst[leaf] + (slot - sharedTraversal->leafPrefix[leaf])];
            const double laneClip = found ? fmin(clip, best.distance) : clip;
            if (!cudaFamilyCullRayTriangle(definition, triangleIndex, cullRay, laneClip, doubleSided))
            {
                continue;
            }
            // The float predicate where it is safe; the world-double one otherwise
            // and wherever the float one cannot decide.
            TriangleHit          candidate{};
            SingleTriangleResult single = kSingleTriangleFallback;
            if (exactSingle && singleTransformSafe && singleRaySafe)
            {
                single = cudaFamilySingleRayTriangle(definition, triangleIndex, cullRay.origin, cullRay.direction,
                                                     minimumDistance, laneClip, doubleSided, orientation, candidate);
            }
            bool accepted = single == kSingleTriangleHit;
            if (single == kSingleTriangleFallback)
            {
                double triangle[3][3]{};
                cudaFamilyLoadMeshTriangle(mesh, definition, triangleIndex, triangle);
                accepted = rayTriangle(origin, direction, minimumDistance, laneClip, triangle, doubleSided,
                                       flipBackFaces, candidate);
            }
            if (accepted && (!found || candidate.distance < best.distance ||
                             (candidate.distance == best.distance && triangleIndex < bestTriangle)))
            {
                found        = true;
                best         = candidate;
                bestTriangle = triangleIndex;
            }
        }
        // Tighten the culling distance for the next step.
        double tightest = found ? best.distance : clip;
        for (unsigned int offset = blockDim.x / 2u; offset > 0u; offset >>= 1u)
        {
            tightest = fmin(tightest, __shfl_xor_sync(kFullMask, tightest, offset));
        }
        if (lane == 0u)
        {
            sharedTraversal->bestDistance = fmin(sharedTraversal->bestDistance, tightest);
        }
        __syncthreads();
    }
    // Move the winning contact to lane 0.
    double             winnerDistance = 0.0;
    unsigned long long winnerTriangle = 0u;
    const int winner = cudaFamilyWarpWinner(found, best.distance, bestTriangle, winnerDistance, winnerTriangle);
    if (winner < 0)
    {
        return false;
    }
    TriangleHit chosen{};
    chosen.distance = __shfl_sync(kFullMask, best.distance, winner);
    chosen.uv[0]    = __shfl_sync(kFullMask, best.uv[0], winner);
    chosen.uv[1]    = __shfl_sync(kFullMask, best.uv[1], winner);
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        chosen.position[axis] = __shfl_sync(kFullMask, best.position[axis], winner);
        chosen.normal[axis]   = __shfl_sync(kFullMask, best.normal[axis], winner);
    }
    bool contactValid = true;
    if (lane == 0u)
    {
        if (exactSingle)
        {
            double triangle[3][3]{};
            double normal[3]{};
            cudaFamilyLoadMeshTriangle(mesh, definition, winnerTriangle, triangle);
            contactValid = triangleNormal(triangle, normal);
            if (contactValid)
            {
                double directionDotNormal = 0.0;
                for (unsigned int axis = 0u; axis < 3u; ++axis)
                {
                    directionDotNormal += normal[axis] * static_cast<double>(direction[axis]);
                }
                if (flipBackFaces && directionDotNormal > 0.0)
                {
                    for (unsigned int axis = 0u; axis < 3u; ++axis)
                    {
                        normal[axis] = -normal[axis];
                    }
                }
                for (unsigned int axis = 0u; axis < 3u; ++axis)
                {
                    chosen.position[axis] = fma(static_cast<double>(direction[axis]), chosen.distance, origin[axis]);
                    chosen.normal[axis]   = static_cast<float>(normal[axis]);
                }
            }
        }
        if (contactValid)
        {
            hit.distance = chosen.distance;
            cudaFamilySetTriangleSource(definition, winnerTriangle, chosen.uv[0], chosen.uv[1], hit);
            for (unsigned int axis = 0u; axis < 3u; ++axis)
            {
                hit.position[axis] = chosen.position[axis];
                hit.normal[axis]   = chosen.normal[axis];
            }
        }
    }
    contactValid = __shfl_sync(kFullMask, contactValid, 0);
    retryExact   = exactSingle && !contactValid;
    return contactValid;
}

// Sorts one (key, value) pair per lane across the warp, larger keys first and
// equal keys by the smaller value: a bitonic network over shuffles.
__device__ void cudaFamilyWarpSortDescending(float& key, unsigned int& value)
{
    const unsigned int lane = threadIdx.x;
    for (unsigned int size = 2u; size <= cudaFamily::kEvaluateThreads; size <<= 1u)
    {
        for (unsigned int stride = size >> 1u; stride != 0u; stride >>= 1u)
        {
            const float        otherKey   = __shfl_xor_sync(0xffffffffu, key, stride);
            const unsigned int otherValue = __shfl_xor_sync(0xffffffffu, value, stride);
            const bool         otherFirst = otherKey > key || (otherKey == key && otherValue < value);
            // In an ascending run the lower lane of a pair keeps the earlier element.
            if (otherFirst == (((lane & stride) == 0u) == ((lane & size) == 0u)))
            {
                key   = otherKey;
                value = otherValue;
            }
        }
    }
}

// The key a sweep's exact test orders its contacts by.
__device__ double cudaFamilySweepKey(const TriangleHit& hit) { return hit.distance; }

__device__ double cudaFamilySweepKey(const boxSweep::Interval& hit) { return hit.enter; }

// Cooperative closest sweep over the definition's local BVH, shared by the sphere
// and box sweeps; each instantiation lives only in its own kernel. Nodes pass
// `cullNode` and triangles `cullTriangle` before `exactTest`, which runs on full
// warps from a queue of QueueCapacity triangles. Each step pushes the children of
// its nearest passing nodes last, so they are popped next, and queues the
// triangles of its nearest leaves first; the running contact then clips the rest.
// Once the stack holds SoftLimit nodes a step pops one. Every triangle whose bound
// admits the closest contact still reaches the exact test, and equal keys keep the
// lower triangle index. Each lane keeps in `best` the contact of the best triangle
// it tested; returns the winning lane (see cudaFamilyWarpWinner), or -1.
template <unsigned int SoftLimit, unsigned int QueueCapacity, typename Hit, typename NodeCull, typename TriangleCull,
          typename ExactTest>
__device__ int cudaFamilyMeshQueuedSweep(const cudaFamily::MeshDefinition& definition, double maximumDistance,
                                         MeshBvhTraversal* sharedTraversal, unsigned int* stack,
                                         unsigned int* queueTriangle, float* queueEntry, NodeCull cullNode,
                                         TriangleCull cullTriangle, ExactTest exactTest, Hit& best, double& winnerKey,
                                         unsigned long long& winnerTriangle)
{
    constexpr unsigned long long   kNone     = ~0ull;
    constexpr unsigned int         kFullMask = 0xffffffffu;
    const cudaFamily::MeshBvhNode* meshNodes = reinterpret_cast<const cudaFamily::MeshBvhNode*>(definition.nodes);
    const unsigned int* traversalTriangles   = reinterpret_cast<const unsigned int*>(definition.traversalTriangles);
    const unsigned int  lane                 = threadIdx.x;
    const unsigned int  below                = (1u << lane) - 1u;
    if (lane == 0u)
    {
        stack[0]                      = 0u;
        sharedTraversal->stackCount   = 1u;
        sharedTraversal->bestDistance = maximumDistance;
    }
    __syncthreads();
    bool               found        = false;
    unsigned long long bestTriangle = kNone;
    unsigned int       head         = 0u; // queue range [head, tail), uniform across the warp
    unsigned int       tail         = 0u;
    for (;;)
    {
        const unsigned int count = sharedTraversal->stackCount;
        if (count == 0u)
        {
            break;
        }
        unsigned int take = count < blockDim.x ? count : blockDim.x;
        if (count >= SoftLimit)
        {
            take = 1u;
        }
        else if (take > SoftLimit - count)
        {
            take = SoftLimit - count;
        }
        const unsigned int base         = count - take;
        const double       clip         = sharedTraversal->bestDistance;
        bool               nodeHit      = false;
        bool               nodeLeaf     = false;
        unsigned long long nodeFirst    = 0u;
        unsigned int       nodeTriangle = 0u;
        float              nodeEntry    = 0.0f;
        if (lane < take)
        {
            const cudaFamily::MeshBvhNode& node = meshNodes[stack[base + lane]];
            nodeHit                             = cullNode(node, __double2float_ru(clip), nodeEntry);
            nodeLeaf                            = node.count != 0u;
            nodeFirst                           = node.first;
            nodeTriangle                        = node.count;
        }
        __syncthreads();
        // Farthest passing node first: its children go deepest on the stack, and
        // the nearest leaf's triangles are queued first.
        float        key   = nodeHit ? nodeEntry : -__int_as_float(0x7f800000); // negative infinity
        unsigned int owner = lane;
        cudaFamilyWarpSortDescending(key, owner);
        const bool               sortedHit      = __shfl_sync(kFullMask, nodeHit, owner);
        const bool               sortedLeaf     = __shfl_sync(kFullMask, nodeLeaf, owner);
        const unsigned long long sortedFirst    = __shfl_sync(kFullMask, nodeFirst, owner);
        const unsigned int       sortedTriangle = __shfl_sync(kFullMask, nodeTriangle, owner);
        const unsigned int       interiorMask   = __ballot_sync(kFullMask, sortedHit && !sortedLeaf);
        const unsigned int       leafMask       = __ballot_sync(kFullMask, sortedHit && sortedLeaf);
        if (sortedHit && !sortedLeaf)
        {
            const unsigned int rank      = __popc(interiorMask & below);
            const unsigned int child     = static_cast<unsigned int>(sortedFirst);
            stack[base + 2u * rank]      = child + 1u;
            stack[base + 2u * rank + 1u] = child;
        }
        if (sortedHit && sortedLeaf)
        {
            const unsigned int rank                = __popc(leafMask & ~(below | (1u << lane)));
            sharedTraversal->leafFirst[rank]       = sortedFirst;
            sharedTraversal->leafPrefix[rank + 1u] = sortedTriangle;
        }
        __syncthreads();
        const unsigned int leaves = __popc(leafMask);
        if (lane == 0u)
        {
            sharedTraversal->stackCount    = base + 2u * __popc(interiorMask);
            sharedTraversal->leafPrefix[0] = 0u;
            for (unsigned int leaf = 1u; leaf <= leaves; ++leaf)
            {
                sharedTraversal->leafPrefix[leaf] += sharedTraversal->leafPrefix[leaf - 1u];
            }
        }
        __syncthreads();
        const unsigned int candidates = leaves == 0u ? 0u : sharedTraversal->leafPrefix[leaves];
        const bool         lastStep   = sharedTraversal->stackCount == 0u;
        double             stepClip   = clip; // uniform: every lane's best after each batch
        // A partial batch waits for the next step's triangles until the traversal ends.
        for (unsigned int first = 0u; first < candidates || (lastStep && tail != head); first += blockDim.x)
        {
            const unsigned int slot     = first + lane;
            bool               pass     = false;
            unsigned int       triangle = 0u;
            float              bound    = 0.0f;
            if (slot < candidates)
            {
                unsigned int leaf = 0u;
                while (slot >= sharedTraversal->leafPrefix[leaf + 1u])
                {
                    ++leaf;
                }
                triangle =
                    traversalTriangles[sharedTraversal->leafFirst[leaf] + (slot - sharedTraversal->leafPrefix[leaf])];
                pass = cullTriangle(triangle, __double2float_ru(stepClip), bound);
            }
            const unsigned int passMask = __ballot_sync(kFullMask, pass);
            if (pass)
            {
                const unsigned int position = (tail + __popc(passMask & below)) % QueueCapacity;
                queueTriangle[position]     = triangle;
                queueEntry[position]        = bound;
            }
            tail += __popc(passMask);
            __syncwarp();
            const bool lastRound = lastStep && first + blockDim.x >= candidates;
            while (tail - head >= blockDim.x || (lastRound && tail != head))
            {
                const unsigned int batch = tail - head < blockDim.x ? tail - head : blockDim.x;
                if (lane < batch)
                {
                    const unsigned int       position  = (head + lane) % QueueCapacity;
                    const unsigned long long candidate = queueTriangle[position];
                    // A bound beyond the contact another lane found since it was queued cannot beat or tie it.
                    if (static_cast<double>(queueEntry[position]) <= stepClip)
                    {
                        Hit trial{};
                        if (exactTest(candidate, stepClip, trial) &&
                            (!found || cudaFamilySweepKey(trial) < cudaFamilySweepKey(best) ||
                             (cudaFamilySweepKey(trial) == cudaFamilySweepKey(best) && candidate < bestTriangle)))
                        {
                            found        = true;
                            best         = trial;
                            bestTriangle = candidate;
                        }
                    }
                }
                head += batch;
                __syncwarp();
                double tightest = found ? cudaFamilySweepKey(best) : stepClip;
                for (unsigned int offset = blockDim.x / 2u; offset > 0u; offset >>= 1u)
                {
                    tightest = fmin(tightest, __shfl_xor_sync(kFullMask, tightest, offset));
                }
                stepClip = fmin(stepClip, tightest);
            }
        }
        if (lane == 0u)
        {
            sharedTraversal->bestDistance = stepClip;
        }
        __syncthreads();
    }
    return cudaFamilyWarpWinner(found, cudaFamilySweepKey(best), bestTriangle, winnerKey, winnerTriangle);
}

// Cooperative closest sphere sweep over the definition's local BVH. Nodes pass the
// local slab of the sphere and its path-frame bound, triangles the path-frame bound,
// before the exact double sweepTriangle; the queue lives in the stack's tail. The
// winning lane's contact moves to lane 0.
__device__ bool cudaFamilyMeshSphereSweepHit(const cudaFamily::MeshGeometry&   mesh,
                                             const cudaFamily::MeshDefinition* definitions, bool doubleSided,
                                             bool flipBackFaces, const double center[3], const float direction[3],
                                             double radius, double minimumDistance, double maximumDistance,
                                             MeshBvhTraversal* sharedTraversal, CudaFamilyHit& hit)
{
    constexpr unsigned int            kFullMask  = 0xffffffffu;
    const cudaFamily::MeshDefinition& definition = definitions[mesh.definitionIndex];
    const cudaFamily::MeshBvhNode*    meshNodes  = reinterpret_cast<const cudaFamily::MeshBvhNode*>(definition.nodes);
    const unsigned int                lane       = threadIdx.x;
    double                            localOrigin[3] = {}, localDirection[3] = {}, localExpansion[3] = {};
    cudaFamilyMeshLocalPoint(mesh, center, localOrigin);
    cudaFamilyMeshLocalVector(mesh, direction, localDirection);
    cudaFamilyMeshLocalExpansion(mesh, radius, localExpansion);
    CullRay cullRay{};
    cudaFamilyMakeCullRay(localOrigin, localDirection, localExpansion, minimumDistance, cullRay);
    SphereSweepCull cull{};
    cudaFamilyMakeSphereSweepCull(center, direction, radius, mesh.worldFromLocal, meshNodes[0].minimum,
                                  meshNodes[0].maximum, cull);
    TriangleHit        best{};
    double             winnerDistance = 0.0;
    unsigned long long winnerTriangle = 0u;
    const int          winner         = cudaFamilyMeshQueuedSweep<kSphereStackSoftLimit, kSphereQueueCapacity>(
        definition, maximumDistance, sharedTraversal, sharedTraversal->sphere.stack, sharedTraversal->sphere.triangle,
        sharedTraversal->sphere.entry,
        [&](const cudaFamily::MeshBvhNode& node, float clip, float& entry)
        { return cudaFamilyCullSphereSweepNode(cullRay, cull, node, clip, entry); },
        [&](unsigned int triangle, float clip, float& bound)
        { return cudaFamilyCullSphereSweepTriangle(cull, definition, triangle, cullRay.minimumDistance, clip, bound); },
        [&](unsigned long long candidate, double clip, TriangleHit& trial)
        {
            double triangleWorld[3][3]{};
            cudaFamilyLoadMeshTriangle(mesh, definition, candidate, triangleWorld);
            return sweepTriangle(center, radius, direction, minimumDistance, clip, triangleWorld, doubleSided,
                                                  flipBackFaces, static_cast<unsigned int>(candidate), trial);
        },
        best, winnerDistance, winnerTriangle);
    if (winner < 0)
    {
        return false;
    }
    TriangleHit chosen{};
    chosen.distance = winnerDistance;
    chosen.uv[0]    = __shfl_sync(kFullMask, best.uv[0], winner);
    chosen.uv[1]    = __shfl_sync(kFullMask, best.uv[1], winner);
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        chosen.position[axis] = __shfl_sync(kFullMask, best.position[axis], winner);
        chosen.normal[axis]   = __shfl_sync(kFullMask, best.normal[axis], winner);
    }
    if (lane == 0u)
    {
        hit.distance = chosen.distance;
        cudaFamilySetTriangleSource(definition, winnerTriangle, chosen.uv[0], chosen.uv[1], hit);
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            hit.position[axis] = chosen.position[axis];
            hit.normal[axis]   = chosen.normal[axis];
        }
    }
    return true;
}

__device__ bool cudaFamilyMeshPathHit(const cudaFamily::MeshGeometry&   mesh,
                                      const cudaFamily::MeshDefinition* definitions, bool doubleSided,
                                      bool flipBackFaces, bool exactSingle, const double origin[3],
                                      const float direction[3], double radius, double minimumDistance,
                                      double maximumDistance, MeshBvhTraversal* sharedTraversal, CudaFamilyHit& hit)
{
    if (radius != 0.0)
    {
        return cudaFamilyMeshSphereSweepHit(mesh, definitions, doubleSided, flipBackFaces, origin, direction, radius,
                                            minimumDistance, maximumDistance, sharedTraversal, hit);
    }
    bool       retryExact = false;
    const bool found =
        cudaFamilyMeshPathHitPass(mesh, definitions, doubleSided, flipBackFaces, exactSingle, origin, direction,
                                  minimumDistance, maximumDistance, sharedTraversal, hit, retryExact);
    if (!retryExact)
    {
        return found;
    }
    // A float winner whose transformed world triangle has no normal may have
    // hidden a farther valid contact, so repeat the cooperative traversal with
    // the original predicate instead of dropping the whole mesh.
    return cudaFamilyMeshPathHitPass(mesh, definitions, doubleSided, flipBackFaces, false, origin, direction,
                                     minimumDistance, maximumDistance, sharedTraversal, hit, retryExact);
}

// Every triangle crossing of the ray, for MESH_HITS_ALL: the cooperative
// traversal of cudaFamilyMeshPathHitPass without its running clip, so no node
// or triangle in [minimumDistance, maximumDistance] is culled. Each round every
// lane calls sink(accepted, hit, object) once, so the sink may use warp
// collectives; `hit` is meaningful only where accepted. A float crossing whose
// world triangle has no normal is decided again by the double predicate, as
// the closest pass retries.
template <typename Sink>
__device__ void cudaFamilyMeshPathCrossings(const cudaFamily::MeshGeometry&   mesh,
                                            const cudaFamily::MeshDefinition* definitions, bool doubleSided,
                                            bool flipBackFaces, bool exactSingle, const double origin[3],
                                            const float direction[3], double minimumDistance, double maximumDistance,
                                            MeshBvhTraversal* sharedTraversal, unsigned long long object, Sink& sink)
{
    constexpr unsigned int            kFullMask  = 0xffffffffu;
    const cudaFamily::MeshDefinition& definition = definitions[mesh.definitionIndex];
    const cudaFamily::MeshBvhNode*    meshNodes  = reinterpret_cast<const cudaFamily::MeshBvhNode*>(definition.nodes);
    const unsigned int* traversalTriangles       = reinterpret_cast<const unsigned int*>(definition.traversalTriangles);
    const unsigned int  lane                     = threadIdx.x;
    double              localOrigin[3]           = {};
    double              localDirection[3]        = {};
    double              localExpansion[3]        = {};
    cudaFamilyMeshLocalPoint(mesh, origin, localOrigin);
    cudaFamilyMeshLocalVector(mesh, direction, localDirection);
    bool   singleTransformSafe = true;
    double orientation         = 1.0;
    if (exactSingle)
    {
        const double* transform = mesh.worldFromLocal;
        const double  orientationDeterminant =
            transform[0] * (transform[5] * transform[10] - transform[6] * transform[9]) -
            transform[1] * (transform[4] * transform[10] - transform[6] * transform[8]) +
            transform[2] * (transform[4] * transform[9] - transform[5] * transform[8]);
        singleTransformSafe = isfinite(orientationDeterminant) && orientationDeterminant != 0.0;
        orientation         = orientationDeterminant < 0.0 ? -1.0 : 1.0;
    }
    CullRay cullRay{};
    cudaFamilyMakeCullRay(localOrigin, localDirection, localExpansion, minimumDistance, cullRay);
    bool singleRaySafe = true;
    if (exactSingle)
    {
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            singleRaySafe = singleRaySafe && isfinite(cullRay.origin[axis]) && isfinite(cullRay.direction[axis]) &&
                            (localDirection[axis] == 0.0 || cullRay.direction[axis] != 0.0f);
        }
    }
    if (lane == 0u)
    {
        sharedTraversal->stack[0]   = 0u;
        sharedTraversal->stackCount = 1u;
    }
    __syncthreads();
    for (;;)
    {
        const unsigned int count = sharedTraversal->stackCount;
        if (count == 0u)
        {
            break;
        }
        unsigned int take = count < blockDim.x ? count : blockDim.x;
        if (count >= kMeshStackSoftLimit)
        {
            take = 1u;
        }
        else if (take > kMeshStackSoftLimit - count)
        {
            take = kMeshStackSoftLimit - count;
        }
        const unsigned int base         = count - take;
        bool               nodeHit      = false;
        bool               nodeLeaf     = false;
        unsigned long long nodeFirst    = 0u;
        unsigned int       nodeTriangle = 0u;
        if (lane < take)
        {
            const cudaFamily::MeshBvhNode& node = meshNodes[sharedTraversal->stack[base + lane]];
            nodeHit      = cudaFamilyCullSlab(cullRay, node.minimum, node.maximum, maximumDistance);
            nodeLeaf     = node.count != 0u;
            nodeFirst    = node.first;
            nodeTriangle = node.count;
        }
        __syncthreads();
        const unsigned int interiorMask = __ballot_sync(kFullMask, nodeHit && !nodeLeaf);
        const unsigned int leafMask     = __ballot_sync(kFullMask, nodeHit && nodeLeaf);
        const unsigned int below        = (1u << lane) - 1u;
        if (nodeHit && !nodeLeaf)
        {
            const unsigned int rank                       = __popc(interiorMask & below);
            const unsigned int child                      = static_cast<unsigned int>(nodeFirst);
            sharedTraversal->stack[base + 2u * rank]      = child + 1u;
            sharedTraversal->stack[base + 2u * rank + 1u] = child;
        }
        if (nodeHit && nodeLeaf)
        {
            const unsigned int rank                = __popc(leafMask & below);
            sharedTraversal->leafFirst[rank]       = nodeFirst;
            sharedTraversal->leafPrefix[rank + 1u] = nodeTriangle;
        }
        __syncthreads();
        const unsigned int leaves = __popc(leafMask);
        if (lane == 0u)
        {
            sharedTraversal->stackCount    = base + 2u * __popc(interiorMask);
            sharedTraversal->leafPrefix[0] = 0u;
            for (unsigned int leaf = 1u; leaf <= leaves; ++leaf)
            {
                sharedTraversal->leafPrefix[leaf] += sharedTraversal->leafPrefix[leaf - 1u];
            }
        }
        __syncthreads();
        const unsigned int candidates = leaves == 0u ? 0u : sharedTraversal->leafPrefix[leaves];
        for (unsigned int first = 0u; first < candidates; first += blockDim.x)
        {
            const unsigned int slot     = first + lane;
            bool               accepted = false;
            CudaFamilyHit      hit{};
            if (slot < candidates)
            {
                unsigned int leaf = 0u;
                while (slot >= sharedTraversal->leafPrefix[leaf + 1u])
                {
                    ++leaf;
                }
                const unsigned long long triangleIndex =
                    traversalTriangles[sharedTraversal->leafFirst[leaf] + (slot - sharedTraversal->leafPrefix[leaf])];
                if (cudaFamilyCullRayTriangle(definition, triangleIndex, cullRay, maximumDistance, doubleSided))
                {
                    TriangleHit          candidate{};
                    SingleTriangleResult single = kSingleTriangleFallback;
                    if (exactSingle && singleTransformSafe && singleRaySafe)
                    {
                        single = cudaFamilySingleRayTriangle(definition, triangleIndex, cullRay.origin,
                                                             cullRay.direction, minimumDistance, maximumDistance,
                                                             doubleSided, orientation, candidate);
                    }
                    // The world triangle gives a float crossing its normal or decides a
                    // fallback; a float miss needs neither.
                    double triangle[3][3]{};
                    if (single != kSingleTriangleMiss)
                    {
                        cudaFamilyLoadMeshTriangle(mesh, definition, triangleIndex, triangle);
                    }
                    accepted = single == kSingleTriangleHit;
                    if (accepted)
                    {
                        double normal[3]{};
                        if (triangleNormal(triangle, normal))
                        {
                            double directionDotNormal = 0.0;
                            for (unsigned int axis = 0u; axis < 3u; ++axis)
                            {
                                directionDotNormal += normal[axis] * static_cast<double>(direction[axis]);
                            }
                            const double sign = flipBackFaces && directionDotNormal > 0.0 ? -1.0 : 1.0;
                            for (unsigned int axis = 0u; axis < 3u; ++axis)
                            {
                                candidate.position[axis] =
                                    fma(static_cast<double>(direction[axis]), candidate.distance, origin[axis]);
                                candidate.normal[axis] = static_cast<float>(sign * normal[axis]);
                            }
                        }
                        else
                        {
                            single = kSingleTriangleFallback;
                        }
                    }
                    if (single == kSingleTriangleFallback)
                    {
                        accepted = rayTriangle(origin, direction, minimumDistance, maximumDistance, triangle,
                                               doubleSided, flipBackFaces, candidate);
                    }
                    if (accepted)
                    {
                        hit.distance = candidate.distance;
                        cudaFamilySetTriangleSource(definition, triangleIndex, candidate.uv[0], candidate.uv[1], hit);
                        for (unsigned int axis = 0u; axis < 3u; ++axis)
                        {
                            hit.position[axis] = candidate.position[axis];
                            hit.normal[axis]   = candidate.normal[axis];
                        }
                    }
                }
            }
            sink(accepted, hit, object);
        }
        __syncthreads();
    }
}

// Exact closest path for the stored primitive and convex families; meshes take
// the cooperative path above.
__device__ bool cudaFamilyExactPathHit(const cudaFamily::ExactObjectHeader& object,
                                       const cudaFamily::SphereGeometry* spheres, const cudaFamily::BoxGeometry* boxes,
                                       const cudaFamily::CapsuleGeometry* capsules,
                                       const cudaFamily::ConvexGeometry* convexes, const double* convexVertices,
                                       const unsigned int* convexIndices, const double* convexPlanes,
                                       const double origin[3], const float direction[3], double radius,
                                       double minimumDistance, double maximumDistance, CudaFamilyHit& hit)
{
    hit = CudaFamilyHit{};
    if (object.kind == cudaFamily::kGeometrySphere)
    {
        return cudaFamilySpherePathHit(spheres[object.familyIndex], origin, direction, radius, minimumDistance,
                                       maximumDistance, hit);
    }
    if (object.kind == cudaFamily::kGeometryBox)
    {
        return cudaFamilyBoxPathHit(boxes[object.familyIndex], origin, direction, radius, minimumDistance,
                                    maximumDistance, hit);
    }
    if (object.kind == cudaFamily::kGeometryCapsule)
    {
        return cudaFamilyCapsulePathHit(capsules[object.familyIndex], origin, direction, radius, minimumDistance,
                                        maximumDistance, hit);
    }
    if (object.kind == cudaFamily::kGeometryConvex)
    {
        const cudaFamily::ConvexGeometry& convex = convexes[object.familyIndex];
        // Keep small convex features independent of a large world translation.
        // Raw affine convexes have zero translation and retain their world arrays.
        const double relativeOrigin[3]{origin[0] - convex.translation[0], origin[1] - convex.translation[1],
                                       origin[2] - convex.translation[2]};
        const bool   found = radius == 0.0 ? cudaFamilyConvexRayHit(convex, convexPlanes, relativeOrigin, direction,
                                                                    minimumDistance, maximumDistance, hit)
                                           : cudaFamilyConvexSweepHit(convex, convexVertices, convexIndices, convexPlanes,
                                                                      relativeOrigin, radius, direction, minimumDistance,
                                                                      maximumDistance, hit);
        if (found)
            for (unsigned int axis = 0; axis < 3; ++axis)
                hit.position[axis] += convex.translation[axis];
        return found;
    }
    return false;
}

__device__ bool cudaFamilyExactBoxSweepHit(
    const cudaFamily::ExactObjectHeader& object, const cudaFamily::SphereGeometry* spheres,
    const cudaFamily::BoxGeometry* boxes, const cudaFamily::CapsuleGeometry* capsules,
    const cudaFamily::ConvexGeometry* convexes, const double* convexVertices, const unsigned int* convexIndices,
    const double* convexPlanes, const cudaFamily::BoxGeometry& queryBox, const float direction[3],
    double minimumDistance, double maximumDistance, bool contact, CudaFamilyHit& hit)
{
    const boxSweep::Box  query    = cudaFamilySweepBox(queryBox);
    const boxSweep::Vec3 movement = cudaFamilySweepDirection(direction);
    boxSweep::Hit        candidate{};
    bool                 found = false;
    if (object.kind == cudaFamily::kGeometryBox)
    {
        return cudaFamilyBoxBoxHit(query, cudaFamilySweepBox(boxes[object.familyIndex]), movement, minimumDistance,
                                   maximumDistance, contact, hit);
    }
    if (object.kind == cudaFamily::kGeometrySphere)
    {
        const auto& sphere = spheres[object.familyIndex];
        found = boxSweep::boxSphere(query, cudaFamilySweepVec(sphere.center), sphere.radius, movement, minimumDistance,
                                    maximumDistance, candidate);
    }
    else if (object.kind == cudaFamily::kGeometryCapsule)
    {
        const auto& capsule = capsules[object.familyIndex];
        found =
            boxSweep::boxCapsule(query, cudaFamilySweepVec(capsule.endpointA), cudaFamilySweepVec(capsule.endpointB),
                                 capsule.radius, movement, minimumDistance, maximumDistance, candidate);
    }
    else if (object.kind == cudaFamily::kGeometryConvex)
    {
        const auto&   convex        = convexes[object.familyIndex];
        boxSweep::Box relativeQuery = query;
        relativeQuery.center        = boxSweep::sub(query.center, cudaFamilySweepVec(convex.translation));
        unsigned long long winner   = ~0ull;
        for (unsigned long long index = 0u; index < convex.triangleCount; ++index)
        {
            double triangle[3][3] = {};
            cudaFamilyLoadConvexTriangle(convex, convexVertices, convexIndices, index, triangle);
            boxSweep::Hit trial{};
            if (boxSweep::boxTriangle(relativeQuery, cudaFamilySweepVec(triangle[0]), cudaFamilySweepVec(triangle[1]),
                                      cudaFamilySweepVec(triangle[2]), true, movement, minimumDistance,
                                      found ? candidate.distance : maximumDistance, trial) &&
                (!found || trial.distance < candidate.distance ||
                 (trial.distance == candidate.distance && index < winner)))
            {
                found     = true;
                winner    = index;
                candidate = trial;
            }
        }
        if (convex.triangleCount != 0u)
        {
            boxSweep::Box moved = relativeQuery;
            moved.center        = boxSweep::at(relativeQuery.center, movement, minimumDistance);
            bool contained      = true;
            for (int vertex = 0; vertex < 8 && contained; ++vertex)
            {
                const boxSweep::Vec3 vertexPosition = boxSweep::vertex(moved, vertex);
                const double         relative[3]    = {vertexPosition.x, vertexPosition.y, vertexPosition.z};
                contained                           = cudaFamilyConvexContains(convex, convexPlanes, relative);
            }
            if (contained)
            {
                double triangle[3][3] = {};
                cudaFamilyLoadConvexTriangle(convex, convexVertices, convexIndices, 0u, triangle);
                candidate.distance = minimumDistance;
                candidate.position = cudaFamilySweepVec(triangle[0]);
                candidate.normal   = boxSweep::unit(boxSweep::mul(movement, -1.0), {1.0, 0.0, 0.0});
                found              = true;
            }
        }
    }
    if (found)
    {
        if (object.kind == cudaFamily::kGeometryConvex)
        {
            candidate.position =
                boxSweep::add(candidate.position, cudaFamilySweepVec(convexes[object.familyIndex].translation));
        }
        cudaFamilySweepHit(candidate, hit);
    }
    return found;
}

// Triangles that passed the bounds of one traversal step, waiting for the exact test
// in full warps. A round appends at most a warp's worth before full batches drain.
struct BoxSweepQueue
{
    unsigned int triangle[kBoxQueueCapacity];
    float        entry[kBoxQueueCapacity];
};

// Cooperative closest box sweep over the definition's local BVH. Nodes pass the
// local slab of the box's circumsphere and the face-axis bounds above, triangles
// the face-axis bounds, before the exact double SAT; the queue is the kernel's
// BoxSweepQueue. The order is the CPU refinement's rule with a warp in place of its
// heap. NVRTC contracts multiply-adds that the host evaluates separately, so at
// ties the CPU and CUDA plans can select different, equally valid faces or contact
// witnesses. Triangles are relative to the placement's translation, which is
// restored only for the contact.
__device__ bool cudaFamilyMeshBoxSweepHit(const cudaFamily::MeshGeometry&   mesh,
                                          const cudaFamily::MeshDefinition* definitions, bool doubleSided,
                                          const cudaFamily::BoxGeometry& queryBox, const float direction[3],
                                          double minimumDistance, double maximumDistance,
                                          MeshBvhTraversal* sharedTraversal, BoxSweepQueue* sharedQueue,
                                          CudaFamilyHit& hit)
{
    constexpr unsigned int            kFullMask  = 0xffffffffu;
    const cudaFamily::MeshDefinition& definition = definitions[mesh.definitionIndex];
    const cudaFamily::MeshBvhNode*    meshNodes  = reinterpret_cast<const cudaFamily::MeshBvhNode*>(definition.nodes);
    const unsigned int                lane       = threadIdx.x;
    boxSweep::Box                     relativeQuery = cudaFamilySweepBox(queryBox);
    relativeQuery.center.x -= mesh.worldFromLocal[3];
    relativeQuery.center.y -= mesh.worldFromLocal[7];
    relativeQuery.center.z -= mesh.worldFromLocal[11];
    const boxSweep::Vec3 movement = cudaFamilySweepDirection(direction);
    // The local slab bounds every orientation of the box by its circumsphere.
    double localOrigin[3] = {}, localDirection[3] = {}, localExpansion[3] = {};
    cudaFamilyMeshLocalPoint(mesh, queryBox.center, localOrigin);
    cudaFamilyMeshLocalVector(mesh, direction, localDirection);
    cudaFamilyMeshLocalExpansion(
        mesh, hypot(hypot(queryBox.halfExtents[0], queryBox.halfExtents[1]), queryBox.halfExtents[2]), localExpansion);
    CullRay cullRay{};
    cudaFamilyMakeCullRay(localOrigin, localDirection, localExpansion, minimumDistance, cullRay);
    BoxSweepCull cull{};
    cudaFamilyMakeBoxSweepCull(relativeQuery, movement, mesh.worldFromLocal, meshNodes[0].minimum, meshNodes[0].maximum,
                               cull);
    // Each lane keeps the SAT interval of the best triangle it tested; only the
    // winner's contact witness is computed, from that interval, after the traversal.
    boxSweep::Interval best{};
    double             winnerDistance = 0.0;
    unsigned long long winnerTriangle = 0u;
    const int          winner         = cudaFamilyMeshQueuedSweep<kMeshStackSoftLimit, kBoxQueueCapacity>(
        definition, maximumDistance, sharedTraversal, sharedTraversal->stack, sharedQueue->triangle, sharedQueue->entry,
        [&](const cudaFamily::MeshBvhNode& node, float clip, float& entry)
        { return cudaFamilyCullBoxSweepNode(cullRay, cull, node, clip, entry); },
        [&](unsigned int triangle, float clip, float& bound)
        { return cudaFamilyCullBoxSweepTriangle(cull, definition, triangle, cullRay.minimumDistance, clip, bound); },
        [&](unsigned long long candidate, double clip, boxSweep::Interval& trial)
        {
            double corners[3][3]{};
            cudaFamilyLoadMeshTriangle<true>(mesh, definition, candidate, corners);
            return boxSweep::boxTriangleInterval(relativeQuery, cudaFamilySweepVec(corners[0]),
                                                                  cudaFamilySweepVec(corners[1]), cudaFamilySweepVec(corners[2]),
                                                                  doubleSided, movement, minimumDistance, clip, trial);
        },
        best, winnerDistance, winnerTriangle);
    if (winner < 0)
    {
        return false;
    }
    boxSweep::Interval chosen{};
    chosen.enter    = winnerDistance;
    chosen.normal.x = __shfl_sync(kFullMask, best.normal.x, winner);
    chosen.normal.y = __shfl_sync(kFullMask, best.normal.y, winner);
    chosen.normal.z = __shfl_sync(kFullMask, best.normal.z, winner);
    if (lane == 0u)
    {
        double corners[3][3]{};
        cudaFamilyLoadMeshTriangle<true>(mesh, definition, winnerTriangle, corners);
        boxSweep::Hit contact{};
        boxSweep::boxTriangleContact(relativeQuery, cudaFamilySweepVec(corners[0]), cudaFamilySweepVec(corners[1]),
                                     cudaFamilySweepVec(corners[2]), movement, chosen, contact);
        hit.distance = contact.distance;
        cudaFamilySetTriangleSource(definition, winnerTriangle, static_cast<float>(contact.u),
                                    static_cast<float>(contact.v), hit);
        hit.position[0] = contact.position.x + mesh.worldFromLocal[3];
        hit.position[1] = contact.position.y + mesh.worldFromLocal[7];
        hit.position[2] = contact.position.z + mesh.worldFromLocal[11];
        hit.normal[0]   = static_cast<float>(contact.normal.x);
        hit.normal[1]   = static_cast<float>(contact.normal.y);
        hit.normal[2]   = static_cast<float>(contact.normal.z);
    }
    return true;
}

// An exact overlap row's shape in float for its candidate gate, converted once per row.
struct OverlapGate
{
    float center[3];
    float axes[9];   // the box's axes, rows in world coordinates (an AABB row's are the world axes)
    float extent[3]; // the box's half extents, or the sphere's radius first
    float reach[3];  // the box's half extent along each world axis
    float magnitude; // the query's coordinates and extents, its share of the margin's magnitudes
    bool  sphere;
};

__device__ void cudaFamilyMakeOverlapGate(unsigned int overlapKind, const double translation[3], double sphereRadius,
                                          const cudaFamily::BoxGeometry& queryBox, OverlapGate& gate)
{
    gate.sphere          = overlapKind == cudaFamily::kOverlapSphere;
    const double* center = gate.sphere ? translation : queryBox.center;
    gate.magnitude       = gate.sphere ? static_cast<float>(sphereRadius) : 0.0f;
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        gate.center[axis] = static_cast<float>(center[axis]);
        gate.extent[axis] = gate.sphere ? (axis == 0u ? static_cast<float>(sphereRadius) : 0.0f)
                                        : static_cast<float>(queryBox.halfExtents[axis]);
        gate.magnitude += fabsf(gate.center[axis]) + (gate.sphere ? 0.0f : gate.extent[axis]);
        for (unsigned int world = 0u; world < 3u; ++world)
        {
            gate.axes[axis * 3u + world] = static_cast<float>(queryBox.axes[axis * 3u + world]);
        }
    }
    for (unsigned int world = 0u; world < 3u; ++world)
    {
        gate.reach[world] = 0.0f;
        for (unsigned int local = 0u; local < 3u; ++local)
        {
            gate.reach[world] = fmaf(fabsf(gate.axes[local * 3u + world]), gate.extent[local], gate.reach[world]);
        }
    }
}

// Whether an exact overlap row's shape misses an object's bounds by a margin of 2^-16 of
// their magnitudes, decided in float on the box and world axes (or by distance from the
// sphere's centre). The terms are scaled by the power of two that maps the magnitudes into
// [1, 2), so the sphere's squares neither overflow nor round as subnormals, and the float
// terms err by about 2^-21 of the magnitudes; the object's geometry then lies outside the
// shape and the exact tests report no touch. Magnitudes outside [2^-100, 2^100], where
// the conversion to float could lose more than the margin, are left to the double bounds
// test (kGateUndecided), as are coordinates the exact primitive tests could square to
// zero. Within that range this replaces the double test as an exact row's candidate gate:
// an object it keeps reaches the exact tests, which decide, so results are those of the
// double gate, whose fifteen scaled axis tests cost more than a giant overlap's other work.
enum : unsigned int
{
    kGateKeeps     = 0u,
    kGateMisses    = 1u,
    kGateUndecided = 2u,
};

__device__ unsigned int cudaFamilyOverlapGateBounds(const OverlapGate& gate, const double boundsMinimum[3],
                                                    const double boundsMaximum[3])
{
    float delta[3] = {}, half[3] = {};
    float magnitude = gate.magnitude;
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        const float low  = static_cast<float>(boundsMinimum[axis]);
        const float high = static_cast<float>(boundsMaximum[axis]);
        delta[axis]      = (low * 0.5f + high * 0.5f) - gate.center[axis];
        half[axis]       = high * 0.5f - low * 0.5f;
        magnitude += fmaxf(fabsf(low), fabsf(high));
    }
    // 2^-100 and 2^100.
    if (!(magnitude >= 7.888609e-31f && magnitude <= 1.2676506e30f))
    {
        return kGateUndecided;
    }
    const float margin = 1.52587890625e-05f * magnitude; // 2^-16
    if (gate.sphere)
    {
        // The power of two that maps the magnitude into [1, 2).
        const float scale = __int_as_float((254 - ((__float_as_int(magnitude) >> 23) & 0xff)) << 23);
        float       near  = 0.0f;
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            const float gap = fmaxf(fabsf(delta[axis]) - half[axis], 0.0f) * scale;
            near            = fmaf(gap, gap, near);
        }
        const float outer = (gate.extent[0] + margin) * scale;
        return near > outer * outer ? kGateMisses : kGateKeeps;
    }
    for (unsigned int local = 0u; local < 3u; ++local)
    {
        const float* axis   = gate.axes + local * 3u;
        const float  offset = fmaf(axis[0], delta[0], fmaf(axis[1], delta[1], axis[2] * delta[2]));
        const float  reach  = fmaf(fabsf(axis[0]), half[0], fmaf(fabsf(axis[1]), half[1], fabsf(axis[2]) * half[2]));
        if (fabsf(offset) - reach > gate.extent[local] + margin ||
            fabsf(delta[local]) - half[local] > gate.reach[local] + margin)
        {
            return kGateMisses;
        }
    }
    return kGateKeeps;
}

enum : unsigned int
{
    kShapeStraddles = 0u,
    kShapeHolds     = 1u,
    kShapeMisses    = 2u,
};

// An exact overlap row's shape seen from one mesh's BVH nodes in float, which decides
// most meshes without a triangle test: a node the shape holds is an overlap and a node it
// misses holds none, each by the CPU's containment margin (overlapShapeContainsObject),
// 2^-16 of the magnitudes the exact test works with (the placement's translation, the
// mesh's world bounds and the query's coordinates). Every kept triangle has area and lies
// in its leaf's bounds, and the exact test's double rounding is about 2^-50 of those
// magnitudes, so it reports a touch for each triangle of a held node and none in a missed
// one. Coordinates are scaled by the power of two that maps the magnitudes into [1, 2),
// so the float terms below stay in range and err by about 2^-20 of them, within the
// margin. Row r of `linear` takes a local point, and `offset[r]` the placement, to the
// point's world offset from the sphere's centre, or to its coordinate along box axis r
// (an AABB row's box has the world axes).
struct OverlapScreen
{
    float linear[9];
    float offset[3];
    float extent[3]; // the box's half extents, or the sphere's radius first
    float margin;
    bool  sphere;
};

// False when the mesh's local coordinates lie outside [2^-24, 2^64] or the magnitudes
// outside [2^-40, 2^100], where its float terms could leave their range or lose more
// than the margin; the mesh then takes the triangle test.
__device__ bool cudaFamilyMakeOverlapScreen(const OverlapGate& gate, const cudaFamily::MeshGeometry& mesh,
                                            const double boundsMinimum[3], const double boundsMaximum[3],
                                            const cudaFamily::MeshBvhNode& root, OverlapScreen& screen)
{
    float local = 0.0f, magnitude = gate.magnitude, linear[9] = {}, placement[3] = {};
    for (unsigned int row = 0u; row < 3u; ++row)
    {
        const float translation = static_cast<float>(mesh.worldFromLocal[row * 4u + 3u]);
        local                   = fmaxf(local, fmaxf(fabsf(root.minimum[row]), fabsf(root.maximum[row])));
        magnitude += fabsf(translation) + fmaxf(fabsf(static_cast<float>(boundsMinimum[row])),
                                                fabsf(static_cast<float>(boundsMaximum[row])));
        placement[row] = translation - gate.center[row];
        for (unsigned int column = 0u; column < 3u; ++column)
        {
            linear[row * 3u + column] = static_cast<float>(mesh.worldFromLocal[row * 4u + column]);
        }
    }
    // 2^-24, 2^64, 2^-40 and 2^100.
    if (!(local >= 5.9604645e-08f && local <= 1.8446744e19f && magnitude >= 9.094947e-13f &&
          magnitude <= 1.2676506e30f))
    {
        return false;
    }
    // The power of two that maps the magnitude into [1, 2).
    const float scale = __int_as_float((254 - ((__float_as_int(magnitude) >> 23) & 0xff)) << 23);
    screen.sphere     = gate.sphere;
    screen.margin     = 1.52587890625e-05f * magnitude * scale; // 2^-16
    for (unsigned int row = 0u; row < 3u; ++row)
    {
        screen.extent[row] = gate.extent[row] * scale;
        const float* axis  = gate.axes + row * 3u;
        for (unsigned int column = 0u; column < 3u; ++column)
        {
            screen.linear[row * 3u + column] =
                gate.sphere
                    ? linear[row * 3u + column] * scale
                    : fmaf(axis[0], linear[column], fmaf(axis[1], linear[3u + column], axis[2] * linear[6u + column])) *
                          scale;
        }
        screen.offset[row] =
            (gate.sphere ? placement[row]
                         : fmaf(axis[0], placement[0], fmaf(axis[1], placement[1], axis[2] * placement[2]))) *
            scale;
    }
    return true;
}

// `excess` orders straddling nodes: the lower, the nearer the node is to being held.
__device__ unsigned int cudaFamilyScreenNode(const OverlapScreen& screen, const cudaFamily::MeshBvhNode& node,
                                             float& excess)
{
    float middle[3] = {}, half[3] = {};
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        middle[axis] = node.minimum[axis] * 0.5f + node.maximum[axis] * 0.5f;
        half[axis]   = node.maximum[axis] * 0.5f - node.minimum[axis] * 0.5f;
    }
    float near = 0.0f, far = 0.0f;
    bool  holds = true;
    excess      = 0.0f;
    for (unsigned int row = 0u; row < 3u; ++row)
    {
        const float* linear = screen.linear + row * 3u;
        const float  at     = fabsf(
            fmaf(linear[0], middle[0], fmaf(linear[1], middle[1], fmaf(linear[2], middle[2], screen.offset[row]))));
        const float reach =
            fmaf(fabsf(linear[0]), half[0], fmaf(fabsf(linear[1]), half[1], fabsf(linear[2]) * half[2]));
        if (screen.sphere)
        {
            const float gap = fmaxf(at - reach, 0.0f);
            far             = fmaf(at + reach, at + reach, far);
            near            = fmaf(gap, gap, near);
            continue;
        }
        if (at - reach > screen.extent[row] + screen.margin)
        {
            return kShapeMisses;
        }
        holds  = holds && at + reach <= screen.extent[row] - screen.margin;
        excess = fmaxf(excess, at + reach - screen.extent[row]);
    }
    if (!screen.sphere)
    {
        return holds ? kShapeHolds : kShapeStraddles;
    }
    excess            = far;
    const float inner = screen.extent[0] - screen.margin;
    const float outer = screen.extent[0] + screen.margin;
    if (inner > 0.0f && far <= inner * inner)
    {
        return kShapeHolds;
    }
    return near > outer * outer ? kShapeMisses : kShapeStraddles;
}

// Cooperative overlap over one mesh: threads refine candidate BVH leaves and
// stop once any thread finds a precise overlap. The result is uniform across the block.
// Thread 0 skips a node the screen finds missed and stops at one it finds held.
__device__ bool cudaFamilyMeshOverlap(const cudaFamily::MeshGeometry&   mesh,
                                      const cudaFamily::MeshDefinition* definitions, unsigned int overlapKind,
                                      const double translation[3], double sphereRadius, const double* rotation,
                                      const cudaFamily::BoxGeometry& queryBox, const OverlapGate& gate,
                                      const double boundsMinimum[3], const double boundsMaximum[3],
                                      unsigned int* sharedFound, MeshBvhTraversal* sharedTraversal)
{
    const cudaFamily::MeshDefinition& definition = definitions[mesh.definitionIndex];
    const cudaFamily::MeshBvhNode*    nodes      = reinterpret_cast<const cudaFamily::MeshBvhNode*>(definition.nodes);
    const unsigned int* traversalTriangles       = reinterpret_cast<const unsigned int*>(definition.traversalTriangles);
    double              localMinimum[3]          = {};
    double              localMaximum[3]          = {};
    cudaFamilyMeshLocalQueryBounds(mesh, overlapKind, translation, sphereRadius, queryBox, localMinimum, localMaximum);
    if (threadIdx.x == 0u)
    {
        *sharedFound                = 0u;
        sharedTraversal->stack[0]   = 0u;
        sharedTraversal->stackCount = 1u;
        sharedTraversal->done       = 0u;
    }
    __syncthreads();
    double aabbMinimum[3] = {};
    double aabbMaximum[3] = {};
    if (overlapKind == cudaFamily::kOverlapAabb)
    {
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            aabbMinimum[axis] = translation[axis] - queryBox.halfExtents[axis];
            aabbMaximum[axis] = translation[axis] + queryBox.halfExtents[axis];
        }
    }
    OverlapScreen screen{};
    const bool    screened =
        threadIdx.x == 0u && cudaFamilyMakeOverlapScreen(gate, mesh, boundsMinimum, boundsMaximum, nodes[0], screen);
    while (sharedTraversal->done == 0u && *sharedFound == 0u)
    {
        if (threadIdx.x == 0u)
        {
            sharedTraversal->leafCount = 0u;
            while (sharedTraversal->stackCount != 0u && sharedTraversal->leafCount == 0u)
            {
                const unsigned int             nodeIndex = sharedTraversal->stack[--sharedTraversal->stackCount];
                const cudaFamily::MeshBvhNode& node      = nodes[nodeIndex];
                float                          excess    = 0.0f;
                const unsigned int shape = screened ? cudaFamilyScreenNode(screen, node, excess) : kShapeStraddles;
                if (shape == kShapeHolds)
                {
                    *sharedFound = 1u;
                    break;
                }
                if (shape == kShapeMisses || !cudaFamilyMeshBoundsOverlap(node, localMinimum, localMaximum))
                {
                    continue;
                }
                if (node.count != 0u)
                {
                    sharedTraversal->leafFirst[0] = node.first;
                    sharedTraversal->leafCount    = node.count;
                }
                else
                {
                    const unsigned int child                              = static_cast<unsigned int>(node.first);
                    sharedTraversal->stack[sharedTraversal->stackCount++] = child + 1u;
                    sharedTraversal->stack[sharedTraversal->stackCount++] = child;
                }
            }
            sharedTraversal->done = sharedTraversal->leafCount == 0u ? 1u : 0u;
        }
        __syncthreads();
        if (sharedTraversal->done != 0u)
        {
            break;
        }
        for (unsigned int leaf = threadIdx.x; leaf < sharedTraversal->leafCount; leaf += blockDim.x)
        {
            const unsigned long long triangleIndex  = traversalTriangles[sharedTraversal->leafFirst[0] + leaf];
            double                   triangle[3][3] = {};
            cudaFamilyLoadMeshTriangle(mesh, definition, triangleIndex, triangle);
            if (preciseOverlap(overlapKind, overlapKind == cudaFamily::kOverlapAabb ? aabbMinimum : translation,
                               sphereRadius, overlapKind == cudaFamily::kOverlapBox ? rotation : nullptr,
                               overlapKind == cudaFamily::kOverlapBox ? queryBox.halfExtents : nullptr,
                               overlapKind == cudaFamily::kOverlapAabb ? aabbMaximum : nullptr, triangle))
            {
                atomicExch(sharedFound, 1u);
            }
        }
        __syncthreads();
    }
    __syncthreads();
    return *sharedFound != 0u;
}

// Exact overlap for the stored primitive and convex families; meshes take the
// cooperative path above.
__device__ bool cudaFamilyExactOverlap(const cudaFamily::ExactObjectHeader& object,
                                       const cudaFamily::SphereGeometry* spheres, const cudaFamily::BoxGeometry* boxes,
                                       const cudaFamily::CapsuleGeometry* capsules,
                                       const cudaFamily::ConvexGeometry* convexes, const double* convexVertices,
                                       const unsigned int* convexIndices, const double* convexPlanes,
                                       const unsigned int* convexEdges, unsigned int overlapKind,
                                       const double translation[3], double sphereRadius,
                                       const cudaFamily::BoxGeometry* queryBox)
{
    if (object.kind == cudaFamily::kGeometrySphere)
    {
        const cudaFamily::SphereGeometry& sphere = spheres[object.familyIndex];
        return overlapKind == cudaFamily::kOverlapSphere
                   ? cudaFamilySphereOverlapsExact(sphere, translation, sphereRadius)
                   : cudaFamilyBoxOverlapsSphere(*queryBox, sphere.center, sphere.radius);
    }
    if (object.kind == cudaFamily::kGeometryBox)
    {
        const cudaFamily::BoxGeometry& box = boxes[object.familyIndex];
        return overlapKind == cudaFamily::kOverlapSphere ? cudaFamilyBoxOverlapsSphere(box, translation, sphereRadius)
                                                         : cudaFamilyObbsOverlap(box, *queryBox);
    }
    if (object.kind == cudaFamily::kGeometryCapsule)
    {
        const cudaFamily::CapsuleGeometry& capsule = capsules[object.familyIndex];
        return overlapKind == cudaFamily::kOverlapSphere
                   ? cudaFamilyCapsuleOverlapsSphere(capsule, translation, sphereRadius)
                   : cudaFamilyCapsuleOverlapsBox(capsule, *queryBox);
    }
    if (object.kind == cudaFamily::kGeometryConvex)
    {
        const cudaFamily::ConvexGeometry& convex = convexes[object.familyIndex];
        const double relativeCenter[3]{translation[0] - convex.translation[0], translation[1] - convex.translation[1],
                                       translation[2] - convex.translation[2]};
        if (overlapKind == cudaFamily::kOverlapSphere)
            return cudaFamilyConvexSphereOverlap(convex, convexVertices, convexIndices, convexPlanes, relativeCenter,
                                                 sphereRadius, nullptr, nullptr);
        cudaFamily::BoxGeometry relativeBox = *queryBox;
        cudaFamilyCopy3(relativeCenter, relativeBox.center);
        return cudaFamilyConvexBoxOverlap(convex, convexVertices, convexPlanes, convexEdges, relativeBox);
    }
    return false;
}

__device__ void cudaFamilyCopyOutputRow(unsigned long long destination, unsigned long long source,
                                        unsigned long long* objectIds, int* geometryTypes,
                                        unsigned long long* geometryValues, double* positions, double* queryPositions,
                                        float* normals, double* distances, unsigned int* featureKinds,
                                        unsigned long long* faceIndices, unsigned int* corners, float* barycentrics)
{
    objectIds[destination] = objectIds[source];
    if (distances != nullptr)
    {
        distances[destination] = distances[source];
    }
    if (geometryTypes != nullptr)
    {
        geometryTypes[destination] = geometryTypes[source];
    }
    if (geometryValues != nullptr)
    {
        geometryValues[destination] = geometryValues[source];
    }
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        if (positions != nullptr)
        {
            positions[destination * 3u + axis] = positions[source * 3u + axis];
        }
        if (queryPositions != nullptr)
        {
            queryPositions[destination * 3u + axis] = queryPositions[source * 3u + axis];
        }
        if (normals != nullptr)
        {
            normals[destination * 3u + axis] = normals[source * 3u + axis];
        }
        if (corners != nullptr)
        {
            corners[destination * 3u + axis] = corners[source * 3u + axis];
        }
        if (barycentrics != nullptr)
        {
            barycentrics[destination * 3u + axis] = barycentrics[source * 3u + axis];
        }
    }
    if (featureKinds != nullptr)
    {
        featureKinds[destination] = featureKinds[source];
    }
    if (faceIndices != nullptr)
    {
        faceIndices[destination] = faceIndices[source];
    }
}

__device__ void cudaFamilyWriteOutputRow(unsigned long long row, unsigned long long objectId, unsigned int geometryType,
                                         unsigned long long geometryValue, const CudaFamilyHit& hit, bool exact,
                                         const double* queryOrigin, const float* queryDirection,
                                         unsigned int resultFields, unsigned long long* objectIds, int* geometryTypes,
                                         unsigned long long* geometryValues, double* positions, double* queryPositions,
                                         float* normals, double* distances, unsigned int* featureKinds,
                                         unsigned long long* faceIndices, unsigned int* corners, float* barycentrics)
{
    objectIds[row] = objectId;
    if ((resultFields & cudaFamily::kResultGeometryIdentity) != 0u && geometryTypes != nullptr &&
        geometryValues != nullptr)
    {
        geometryTypes[row]  = static_cast<int>(geometryType);
        geometryValues[row] = geometryValue;
    }
    if ((resultFields & cudaFamily::kResultPosition) != 0u && positions != nullptr)
    {
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            positions[row * 3u + axis] = hit.position[axis];
        }
    }
    if ((resultFields & cudaFamily::kResultNormal) != 0u && normals != nullptr)
    {
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            normals[row * 3u + axis] = hit.normal[axis];
        }
    }
    if (distances != nullptr)
    {
        distances[row] = hit.distance;
    }
    if ((resultFields & cudaFamily::kResultQueryPosition) != 0u && queryOrigin != nullptr &&
        queryDirection != nullptr && queryPositions != nullptr)
    {
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            queryPositions[row * 3u + axis] =
                queryOrigin[axis] + static_cast<double>(queryDirection[axis]) * hit.distance;
        }
    }
    if (exact && (resultFields & cudaFamily::kResultSourceFeature) != 0u && featureKinds != nullptr &&
        faceIndices != nullptr && corners != nullptr && barycentrics != nullptr)
    {
        featureKinds[row] = hit.triangleValid ? cudaFamily::kFeatureTriangle : cudaFamily::kFeatureNone;
        if (hit.triangleValid)
        {
            faceIndices[row] = hit.sourceFace;
            for (unsigned int axis = 0u; axis < 3u; ++axis)
            {
                corners[row * 3u + axis]      = hit.sourceCorners[axis];
                barycentrics[row * 3u + axis] = hit.barycentrics[axis];
            }
        }
    }
}

__device__ bool cudaFamilyCandidatePathHit(const double boundsMinimum[3], const double boundsMaximum[3],
                                           const double origin[3], const float direction[3], double radius,
                                           double minimumDistance, double maximumDistance, CudaFamilyHit& hit)
{
    double expandedPosition[3] = {};
    float  normal[3]           = {};
    double distance            = 0.0;
    if (!pathAabbHit(origin, direction, minimumDistance, maximumDistance, radius, boundsMinimum, boundsMaximum,
                     distance, expandedPosition, normal))
    {
        return false;
    }
    hit          = CudaFamilyHit{};
    hit.distance = distance;
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        const double queryPosition = fma(static_cast<double>(direction[axis]), distance, origin[axis]);
        hit.position[axis]         = fmax(boundsMinimum[axis], fmin(boundsMaximum[axis], queryPosition));
        hit.normal[axis]           = normal[axis];
    }
    return true;
}

// A box sweep against an object's bounds, the BOUNDS_CANDIDATE result; its contact
// only when `contact`.
__device__ bool cudaFamilyCandidateBoxSweepHit(const cudaFamily::BoxGeometry& queryBox, const double boundsMinimum[3],
                                               const double boundsMaximum[3], const float direction[3],
                                               double minimumDistance, double maximumDistance, bool contact,
                                               CudaFamilyHit& hit)
{
    return cudaFamilyBoxBoxHit(cudaFamilySweepBox(queryBox),
                               cudaFamilySweepBox(cudaFamilyBoundsBox(boundsMinimum, boundsMaximum)),
                               cudaFamilySweepDirection(direction), minimumDistance, maximumDistance, contact, hit);
}

// The float slab over an object's double bounds, rounded outward, ahead of the
// double candidate test that produces the entry distance and the bounds contact;
// `entry` is a lower bound on the double slab's entry.
__device__ bool cudaFamilyCullObjectPath(const CullRay& ray, const double boundsMinimum[3],
                                         const double boundsMaximum[3], double maximumDistance, float& entry)
{
    const float minimum[3] = {__double2float_rd(boundsMinimum[0]), __double2float_rd(boundsMinimum[1]),
                              __double2float_rd(boundsMinimum[2])};
    const float maximum[3] = {__double2float_ru(boundsMaximum[0]), __double2float_ru(boundsMaximum[1]),
                              __double2float_ru(boundsMaximum[2])};
    entry                  = ray.minimumDistance;
    float exit             = __double2float_ru(maximumDistance);
    return cudaFamilyNarrowSlab(ray, minimum, maximum, entry, exit);
}

__device__ bool cudaFamilyCandidateOverlap(unsigned int overlapKind, const double translation[3], double sphereRadius,
                                           const double* rotation, const double* extents, const double boundsMinimum[3],
                                           const double boundsMaximum[3], bool exact)
{
    if (overlapKind == cudaFamily::kOverlapSphere)
    {
        return sphereOverlapsAabb(translation, sphereRadius, boundsMinimum, boundsMaximum);
    }
    if (overlapKind == cudaFamily::kOverlapBox)
    {
        return boxOverlapsAabb(translation, rotation, extents, boundsMinimum, boundsMaximum, exact);
    }
    double minimum[3] = {};
    double maximum[3] = {};
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        minimum[axis] = translation[axis] - extents[axis];
        maximum[axis] = translation[axis] + extents[axis];
    }
    return aabbsOverlap(minimum, maximum, boundsMinimum, boundsMaximum);
}

// An exact overlap row's candidate gate. The float test rejects what it can; a mesh it
// keeps goes to the triangle test, whose terms are normalized. A primitive it keeps, or
// bounds it leaves undecided, also take the double bounds test as before: the exact
// primitive tests square unscaled distances, so a primitive far smaller than its
// coordinates (radius 2^-600 at x = 1, three radii from the query) would round to a
// touch that the double test, whose terms are normalized, rejects.
__device__ bool cudaFamilyExactOverlapCandidate(const OverlapGate& gate, bool mesh, unsigned int overlapKind,
                                                const double translation[3], double sphereRadius,
                                                const double* rotation, const double* extents,
                                                const double boundsMinimum[3], const double boundsMaximum[3])
{
    const unsigned int verdict = cudaFamilyOverlapGateBounds(gate, boundsMinimum, boundsMaximum);
    if (verdict == kGateMisses)
    {
        return false;
    }
    return (verdict == kGateKeeps && mesh) ||
           cudaFamilyCandidateOverlap(overlapKind, translation, sphereRadius, rotation, extents, boundsMinimum,
                                      boundsMaximum, true);
}

__device__ bool cudaFamilyOverlapRowIsValid(unsigned int overlapKind, const double* translation, const double* rotation,
                                            const double* halfExtents, double radius, double tolerance,
                                            cudaFamily::BoxGeometry& queryBox)
{
    if (overlapKind == cudaFamily::kOverlapSphere)
    {
        if (translation == nullptr)
        {
            return false;
        }
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            if (!cudaFamilyInDomain(translation[axis]))
            {
                return false;
            }
        }
        return cudaFamilyInDomain(radius) && radius > 0.0 && isfinite(tolerance) && tolerance >= 0.0;
    }
    if (!isfinite(tolerance) || tolerance < 0.0)
    {
        return false;
    }
    if (overlapKind == cudaFamily::kOverlapBox)
    {
        return translation != nullptr && rotation != nullptr && halfExtents != nullptr &&
               cudaFamilyMakeQueryBox(translation, rotation, halfExtents, tolerance, queryBox);
    }
    if (overlapKind != cudaFamily::kOverlapAabb || translation == nullptr)
    {
        return false;
    }
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        // The row's corners, as the CPU path builds them, lie inside the domain:
        // the farther corner's magnitude rounds as |translation| + half does.
        if (!isfinite(halfExtents[axis]) || halfExtents[axis] <= 0.0 ||
            !(fabs(translation[axis]) + halfExtents[axis] <= 1152921504606846976.0))
        {
            return false;
        }
        queryBox.center[axis]      = translation[axis];
        queryBox.halfExtents[axis] = cudaFamilyAddTolerance(halfExtents[axis], tolerance);
    }
    for (unsigned int value = 0u; value < 9u; ++value)
    {
        queryBox.axes[value] = value == 0u || value == 4u || value == 8u ? 1.0 : 0.0;
    }
    return true;
}

// The material a hit on catalogue object `object` reports (ADR-0066): an exact
// triangle-mesh contact takes its source face's nonzero material, every other
// hit the object's; zero on a catalogue without material tables (a borrowed
// device-prepared version). Callers write it only into a requested material
// column, so it makes no other check: every check here is code the shared
// kernel body carries, and the multi-row paths that inline it sit at the
// register limit.
__device__ unsigned long long cudaFamilyHitMaterial(const cudaFamily::FamilyArguments& a, unsigned long long object,
                                                    const CudaFamilyHit& hit, bool exact)
{
    if (a.objectMaterials == nullptr)
    {
        return 0u;
    }
    if (exact && hit.triangleValid && a.faceMaterials != nullptr &&
        a.exactObjects[object].kind == cudaFamily::kGeometryTriangleMesh)
    {
        const cudaFamily::MeshGeometry& mesh = a.meshes[a.exactObjects[object].familyIndex];
        if (hit.sourceFace < mesh.faceMaterialCount)
        {
            const unsigned long long faceMaterial = a.faceMaterials[mesh.faceMaterialOffset + hit.sourceFace];
            if (faceMaterial != 0u)
            {
                return faceMaterial;
            }
        }
    }
    return a.objectMaterials[object];
}

// A multi-row hit's material is resolved after its query's rows are written
// (cudaFamilyResolveMaterialKeys): while rows are inserted and shifted, the
// material column holds this key, the catalogue object in the low 32 bits and
// the source face (all ones for none) in the high 32, which costs the hot
// candidate paths one store, as the object ID does.
constexpr unsigned long long kMaterialKeyNoFace = 0xffffffffull;

__device__ unsigned long long cudaFamilyMaterialKey(unsigned long long object, const CudaFamilyHit& hit, bool exact)
{
    const unsigned long long face =
        exact && hit.triangleValid && hit.sourceFace < kMaterialKeyNoFace ? hit.sourceFace : kMaterialKeyNoFace;
    return (face << 32u) | (object & 0xffffffffull);
}

// Replaces the keys of `count` rows from `first` by their materials, one row per lane.
__device__ void cudaFamilyResolveMaterialKeys(const cudaFamily::FamilyArguments& a, unsigned long long first,
                                              unsigned long long count, bool exact)
{
    for (unsigned long long row = threadIdx.x; row < count; row += blockDim.x)
    {
        const unsigned long long key  = a.outputMaterialIds[first + row];
        const unsigned long long face = key >> 32u;
        CudaFamilyHit            hit{};
        hit.triangleValid                = face != kMaterialKeyNoFace;
        hit.sourceFace                   = face;
        a.outputMaterialIds[first + row] = cudaFamilyHitMaterial(a, key & 0xffffffffull, hit, exact);
    }
}

// The closest or first hit a query keeps between the evaluation pass and the
// output pass; one row per query lives in the plan's stash allocation.
struct StashRow
{
    CudaFamilyHit      hit;
    unsigned long long objectId;
    unsigned long long geometryValue;
    unsigned long long materialId;
    unsigned int       geometryType;
};
static_assert(sizeof(StashRow) <= cudaFamily::kStashRowBytes, "CUDA stash row exceeds its reserved bytes");
// The block reduction and scan halve the thread count each step.
static_assert((cudaFamily::kEvaluateThreads & (cudaFamily::kEvaluateThreads - 1u)) == 0u &&
                  (cudaFamily::kScanThreads & (cudaFamily::kScanThreads - 1u)) == 0u,
              "CUDA family block sizes must be powers of two");
// The mesh traversal reduces with warp shuffles and ballots over the whole block.
static_assert(cudaFamily::kEvaluateThreads == 32u, "CUDA family evaluation block must be one warp");

// Zero query bits accept every object; otherwise the object's two participation
// words must share a set bit with the query's (ADR-0003). A zero query reads no
// object bits.
__device__ bool cudaFamilyParticipates(const unsigned long long* objectBits, unsigned long long object,
                                       const unsigned long long query[2])
{
    return (query[0] | query[1]) == 0u ||
           ((objectBits[object * 2u] & query[0]) | (objectBits[object * 2u + 1u] & query[1])) != 0u;
}

// One object of the current window that passed the participation and bounds
// tests, with the hit its evaluation produced.
struct CandidateSlot
{
    CudaFamilyHit      hit;
    unsigned long long object;
    unsigned int       state; // kSlotMiss, kSlotHit, kSlotPrimitive (to evaluate) or kSlotMesh (cooperative)
};

enum : unsigned int
{
    kSlotMiss      = 0u,
    kSlotHit       = 1u,
    kSlotPrimitive = 2u,
    kSlotMesh      = 3u,
};

// Thread 0's bookkeeping for the query its block evaluates.
struct RowState
{
    unsigned long long semanticCount;
    unsigned long long rowStored;
    unsigned long long closestId;
    // The closest hit's catalogue object: its geometry identity and material are
    // looked up once, when the row is stashed, not on every closer candidate.
    unsigned long long closestObject;
    unsigned int       closestFound;
    unsigned int       stop;
    CudaFamilyHit      closestHit;
};

// The grid cell of one coordinate (a power-of-two cell, so the product is
// exact), or false beyond the cell index range the keys hold.
__device__ bool boxGridCell(double value, double inverseCell, int& outCell)
{
    const double cell = floor(value * inverseCell);
    if (!(cell > -1073741824.0 && cell < 1073741824.0))
    {
        return false;
    }
    outCell = static_cast<int>(cell);
    return true;
}

__device__ unsigned int boxGridHash(int x, int y, int z, unsigned int mask)
{
    unsigned int hash = static_cast<unsigned int>(x) * 73856093u ^ static_cast<unsigned int>(y) * 19349663u ^
                        static_cast<unsigned int>(z) * 83492791u;
    hash ^= hash >> 16u;
    return hash & mask;
}

using FamilyArguments = cudaFamily::FamilyArguments;

// The per-row query description every thread of the block derives redundantly.
struct RowQuery
{
    const double* translation;
    const float*  directionOrRotation;
    // A BOX overlap row's quaternion, read from its float32 or float64 column.
    double                  rotation[4];
    double                  pathRadius;
    double                  overlapRadius;
    double                  minimumDistance;
    double                  maximumDistance;
    unsigned long long      acceptedBits[2];
    unsigned long long      semanticCapacity;
    cudaFamily::BoxGeometry queryBox;
    // The launch counts every mesh triangle from both sides (ADR-0065).
    bool bothSides;
};

enum : unsigned int
{
    kScreenDepth     = 16u, // stack entries per thread in the block's traversal stack
    kScreenExpansion = 32u, // straddling nodes a thread expands before leaving the mesh to the triangle test
};

// One thread's verdict on an exact overlap row's candidate mesh from the BVH alone
// (OverlapScreen): kSlotHit when the shape holds a node, kSlotMiss when it misses every
// leaf, and kSlotMesh, for the cooperative triangle test, when a straddling leaf remains
// or the walk exceeds its stack or expansion budget. Both children of a straddling
// node are screened before either is expanded, so a held child ends the walk early.
// Each thread takes one candidate, so a window's meshes are screened at once; `stack`
// is this thread's column of the block's traversal stack.
__device__ unsigned int cudaFamilyScreenMeshOverlap(const FamilyArguments& a, const OverlapGate& gate,
                                                    unsigned long long object, unsigned int* stack)
{
    // A node fits in the shape only once it is no larger than the shape's reach along every
    // axis. When the mesh's bounds would need more than about four halvings of the BVH for
    // that, the shape holds nothing above the deepest levels and the walk would reach only
    // straddling leaves, which the triangle test then walks to again; such a mesh goes to
    // that test directly.
    float halvings = 1.0f;
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        const float half =
            static_cast<float>(a.objectMaximums[object * 3u + axis] * 0.5 - a.objectMinimums[object * 3u + axis] * 0.5);
        const float reach = gate.sphere ? gate.extent[0] : gate.reach[axis];
        halvings *= fmaxf(1.0f, half / reach);
    }
    if (!(halvings <= 16.0f))
    {
        return kSlotMesh;
    }
    const cudaFamily::MeshGeometry&   mesh       = a.meshes[a.exactObjects[object].familyIndex];
    const cudaFamily::MeshDefinition& definition = a.meshDefinitions[mesh.definitionIndex];
    const cudaFamily::MeshBvhNode*    nodes      = reinterpret_cast<const cudaFamily::MeshBvhNode*>(definition.nodes);
    OverlapScreen                     screen{};
    if (!cudaFamilyMakeOverlapScreen(gate, mesh, a.objectMinimums + object * 3u, a.objectMaximums + object * 3u,
                                     nodes[0], screen))
    {
        return kSlotMesh;
    }
    float              excess = 0.0f;
    const unsigned int root   = cudaFamilyScreenNode(screen, nodes[0], excess);
    if (root != kShapeStraddles)
    {
        return root == kShapeHolds ? kSlotHit : kSlotMiss;
    }
    if (nodes[0].count != 0u)
    {
        return kSlotMesh;
    }
    // The stack holds the first child of each straddling node still to expand; of two
    // straddling children the one nearer to being held is expanded first.
    unsigned int depth = 1u, expanded = 0u;
    bool         open = false; // a straddling leaf or a subtree beyond the stack remains
    stack[0]          = static_cast<unsigned int>(nodes[0].first);
    while (depth != 0u)
    {
        if (++expanded > kScreenExpansion)
        {
            return kSlotMesh;
        }
        const unsigned int             first       = stack[--depth * cudaFamily::kEvaluateThreads];
        const cudaFamily::MeshBvhNode& left        = nodes[first];
        const cudaFamily::MeshBvhNode& right       = nodes[first + 1u];
        float                          leftExcess  = 0.0f;
        float                          rightExcess = 0.0f;
        const unsigned int             leftShape   = cudaFamilyScreenNode(screen, left, leftExcess);
        const unsigned int             rightShape  = cudaFamilyScreenNode(screen, right, rightExcess);
        if (leftShape == kShapeHolds || rightShape == kShapeHolds)
        {
            return kSlotHit;
        }
        const bool leftNearer = leftExcess <= rightExcess;
        for (unsigned int turn = 0u; turn < 2u; ++turn)
        {
            const bool                     takeLeft = (turn == 0u) != leftNearer;
            const cudaFamily::MeshBvhNode& node     = takeLeft ? left : right;
            if ((takeLeft ? leftShape : rightShape) == kShapeMisses)
            {
                continue;
            }
            if (node.count != 0u || depth == kScreenDepth)
            {
                open = true;
                continue;
            }
            stack[depth++ * cudaFamily::kEvaluateThreads] = static_cast<unsigned int>(node.first);
        }
    }
    return open ? kSlotMesh : kSlotMiss;
}

// A block's walk over the grid cells one query box covers, then the oversize
// list: thread t takes cells t, t + 32, ... and oversize entries t, t + 32, ...
// and follows one bucket chain at a time, skipping the entries of other cells
// that hash to the same bucket. The pointers are the grid's arrays.
struct GridCursor
{
    const unsigned int* heads;
    const unsigned int* next;
    const int*          keys;
    const unsigned int* oversize;
    unsigned int        bucketMask;
    unsigned int        oversizeCount;
    int                 low[3];
    unsigned int        span[3];
    unsigned int        cells;
    unsigned int        cell;  // next cell of this thread
    unsigned int        chain; // current chain entry, kGridEmpty between cells
    unsigned int        over;  // next oversize entry of this thread
    int                 key[3];
};

// The cells a query box covers, or false when the box is not finite or covers
// more cells than the grid has slots (at least kGridMinimumScanCells), in
// which case the row scans every object.
// An object linked in cell c spans less than two cells from c's lower edge, so
// one whose box reaches the query's lower edge lies at most one cell below it.
__device__ bool cudaFamilyGridRange(const FamilyArguments& a, const RowQuery& q, GridCursor& c)
{
    const cudaFamily::DeviceBoxGrid grid  = *a.grid;
    unsigned long long              cells = 1u;
    const unsigned long long        most =
        grid.capacity > cudaFamily::kGridMinimumScanCells ? grid.capacity : cudaFamily::kGridMinimumScanCells;
    // The query shape's own world bounds: the sphere's radius, the oriented
    // box's projected half extents, or the AABB's half extents (tolerance
    // included in each), so the cells hold every object the predicate accepts.
    double extent[3];
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        if (a.overlapKind == cudaFamily::kOverlapSphere)
        {
            extent[axis] = q.overlapRadius;
        }
        else if (a.overlapKind == cudaFamily::kOverlapBox)
        {
            double projected = 0.0;
            for (unsigned int local = 0u; local < 3u; ++local)
            {
                projected = fma(fabs(q.queryBox.axes[local * 3u + axis]), q.queryBox.halfExtents[local], projected);
            }
            extent[axis] = projected;
        }
        else
        {
            extent[axis] = q.queryBox.halfExtents[axis];
        }
    }
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        int        high    = 0;
        const bool inRange = boxGridCell(q.translation[axis] - extent[axis], grid.inverseCell, c.low[axis]) &&
                             boxGridCell(q.translation[axis] + extent[axis], grid.inverseCell, high);
        if (!inRange)
        {
            return false;
        }
        c.low[axis] -= 1;
        const long long span = static_cast<long long>(high) - static_cast<long long>(c.low[axis]) + 1ll;
        if (span <= 0ll || span > static_cast<long long>(most))
        {
            return false;
        }
        c.span[axis] = static_cast<unsigned int>(span);
        cells *= static_cast<unsigned long long>(span);
        if (cells > most)
        {
            return false;
        }
    }
    const unsigned char* base = reinterpret_cast<const unsigned char*>(a.grid);
    c.heads                   = reinterpret_cast<const unsigned int*>(base + cudaFamily::kGridHeaderBytes);
    c.next                    = c.heads + (grid.bucketMask + 1u);
    c.keys                    = reinterpret_cast<const int*>(c.next + grid.capacity);
    c.oversize                = reinterpret_cast<const unsigned int*>(c.keys + 3u * grid.capacity);
    c.bucketMask              = grid.bucketMask;
    c.oversizeCount           = ~grid.oversizeComplement;
    c.cells                   = static_cast<unsigned int>(cells);
    c.cell                    = threadIdx.x;
    c.chain                   = cudaFamily::kGridEmpty;
    c.over                    = threadIdx.x;
    return true;
}

// The next entry of this thread's current chain that belongs to the cell `c.key`;
// false once the chain ends.
__device__ bool cudaFamilyGridChainNext(GridCursor& c, unsigned long long& object)
{
    for (; c.chain != cudaFamily::kGridEmpty; c.chain = c.next[c.chain])
    {
        if (c.keys[3u * c.chain] == c.key[0] && c.keys[3u * c.chain + 1u] == c.key[1] &&
            c.keys[3u * c.chain + 2u] == c.key[2])
        {
            object = c.chain;
            return true;
        }
    }
    return false;
}

// This thread's next object, without consuming it: the next entry of its
// current chain that belongs to the cell, else the first entry of its next
// cell, else its next oversize entry.
__device__ bool cudaFamilyGridPeek(GridCursor& c, unsigned long long& object, bool& chained)
{
    for (;;)
    {
        if (cudaFamilyGridChainNext(c, object))
        {
            chained = true;
            return true;
        }
        if (c.cell < c.cells)
        {
            c.key[0] = c.low[0] + static_cast<int>(c.cell % c.span[0]);
            c.key[1] = c.low[1] + static_cast<int>((c.cell / c.span[0]) % c.span[1]);
            c.key[2] = c.low[2] + static_cast<int>(c.cell / (c.span[0] * c.span[1]));
            c.chain  = c.heads[boxGridHash(c.key[0], c.key[1], c.key[2], c.bucketMask)];
            c.cell += blockDim.x;
            continue;
        }
        if (c.over < c.oversizeCount)
        {
            object  = c.oversize[c.over];
            chained = false;
            return true;
        }
        return false;
    }
}

__device__ void cudaFamilyGridConsume(GridCursor& c, bool chained)
{
    if (chained)
    {
        c.chain = c.next[c.chain];
    }
    else
    {
        c.over += blockDim.x;
    }
}

// A block's walk along a path (a ray or a sweep) over the grid: the path is cut
// into segments one cell long along the parameter, thread t takes segments t,
// t + 32, ... and walks every cell the segment's swept box covers (the box
// around the path's points in the segment, inflated by the path's expansion),
// then the oversize list once. An object counts in the segment its entry
// distance falls in, so it is accepted exactly once whatever cells it is seen
// from; the entry distance is a pure function of the object and the row. The
// base cursor's cells are those of the current segment, walked one by one.
struct PathGridCursor : GridCursor
{
    double       inverseCell;
    double       width;
    double       origin[3];
    double       direction[3];
    double       expansion[3];
    double       begin;       // the path's minimum distance
    double       end;         // the path's maximum distance
    unsigned int segments;    // along the path
    unsigned int segment;     // this thread's next segment
    double       windowBegin; // the current segment's parametric window
    double       windowEnd;
    bool         lastWindow;
    bool         inSegment;
};

// The parameter where segment k begins; segment `segments` begins at the end,
// so adjacent windows share one boundary value exactly.
__device__ double cudaFamilyPathBoundary(const PathGridCursor& c, unsigned int k)
{
    return k >= c.segments ? c.end : fma(static_cast<double>(k), c.width, c.begin);
}

// Whether the grid serves this path: a finite path whose segments, each one
// cell long, cover few cells, and whose cells all lie in the key range. The
// cell range of a window is computed as for an overlap box, from the box
// around the window's two endpoints inflated by the expansion and a rounding
// slack, so an object whose entry lies in the window is in its cells.
__device__ bool cudaFamilyPathGridBegin(const FamilyArguments& a, const RowQuery& q, const double expansion[3],
                                        PathGridCursor& c)
{
    const cudaFamily::DeviceBoxGrid grid = *a.grid;
    const unsigned long long        most =
        grid.capacity > cudaFamily::kGridMinimumScanCells ? grid.capacity : cudaFamily::kGridMinimumScanCells;
    c.inverseCell = grid.inverseCell;
    c.width       = 1.0 / grid.inverseCell;
    c.begin       = q.minimumDistance;
    c.end         = q.maximumDistance;
    if (!isfinite(c.begin) || !isfinite(c.end) || c.end < c.begin)
    {
        return false;
    }
    const double length   = c.end - c.begin;
    const double segments = ceil(length * grid.inverseCell);
    if (!(segments >= 0.0) || segments > static_cast<double>(most))
    {
        return false;
    }
    c.segments = segments < 1.0 ? 1u : static_cast<unsigned int>(segments);
    // Cells a window can cover: its extent along an axis is at most the cell
    // plus twice the expansion, plus the one cell objects reach back.
    unsigned long long cellsPerWindow = 1u;
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        c.origin[axis]    = q.translation[axis];
        c.direction[axis] = static_cast<double>(q.directionOrRotation[axis]);
        c.expansion[axis] = expansion[axis];
        if (!isfinite(c.expansion[axis]) || c.expansion[axis] < 0.0)
        {
            return false;
        }
        const double extentCells = ceil((c.width + 2.0 * c.expansion[axis]) * grid.inverseCell) + 3.0;
        if (!(extentCells <= static_cast<double>(most)))
        {
            return false;
        }
        cellsPerWindow *= static_cast<unsigned long long>(extentCells);
        if (cellsPerWindow > most)
        {
            return false;
        }
        // Every window's cells lie between the path's two ends, which must be in range.
        int          first = 0, second = 0, third = 0, fourth = 0;
        const double startPoint = fma(c.direction[axis], c.begin, c.origin[axis]);
        const double endPoint   = fma(c.direction[axis], c.end, c.origin[axis]);
        const double slack      = (fabs(startPoint) + fabs(endPoint) + c.expansion[axis] + c.width) * 0x1p-40;
        if (!boxGridCell(startPoint - c.expansion[axis] - slack, c.inverseCell, first) ||
            !boxGridCell(startPoint + c.expansion[axis] + slack, c.inverseCell, second) ||
            !boxGridCell(endPoint - c.expansion[axis] - slack, c.inverseCell, third) ||
            !boxGridCell(endPoint + c.expansion[axis] + slack, c.inverseCell, fourth))
        {
            return false;
        }
    }
    const unsigned char* base = reinterpret_cast<const unsigned char*>(a.grid);
    c.heads                   = reinterpret_cast<const unsigned int*>(base + cudaFamily::kGridHeaderBytes);
    c.next                    = c.heads + (grid.bucketMask + 1u);
    c.keys                    = reinterpret_cast<const int*>(c.next + grid.capacity);
    c.oversize                = reinterpret_cast<const unsigned int*>(c.keys + 3u * grid.capacity);
    c.bucketMask              = grid.bucketMask;
    c.oversizeCount           = ~grid.oversizeComplement;
    c.segment                 = threadIdx.x;
    c.inSegment               = false;
    c.chain                   = cudaFamily::kGridEmpty;
    c.over                    = threadIdx.x;
    return true;
}

// This thread's next object without consuming it: the next entry of its
// current chain in the cell, else the next cell of its current segment, else
// the first cell of its next segment that begins within `reach`, else its next
// oversize entry. Its objects enter in that segment or later.
__device__ bool cudaFamilyPathGridPeek(PathGridCursor& c, double reach, unsigned long long& object, bool& chained)
{
    for (;;)
    {
        if (cudaFamilyGridChainNext(c, object))
        {
            chained = true;
            return true;
        }
        if (c.inSegment && c.cell < c.cells)
        {
            c.key[0] = c.low[0] + static_cast<int>(c.cell % c.span[0]);
            c.key[1] = c.low[1] + static_cast<int>((c.cell / c.span[0]) % c.span[1]);
            c.key[2] = c.low[2] + static_cast<int>(c.cell / (c.span[0] * c.span[1]));
            c.chain  = c.heads[boxGridHash(c.key[0], c.key[1], c.key[2], c.bucketMask)];
            ++c.cell;
            continue;
        }
        c.inSegment = false;
        if (c.segment < c.segments && (reach >= c.end || cudaFamilyPathBoundary(c, c.segment) <= reach))
        {
            c.windowBegin = cudaFamilyPathBoundary(c, c.segment);
            c.windowEnd   = cudaFamilyPathBoundary(c, c.segment + 1u);
            c.lastWindow  = c.segment + 1u >= c.segments;
            c.cells       = 1u;
            for (unsigned int axis = 0u; axis < 3u; ++axis)
            {
                const double first  = fma(c.direction[axis], c.windowBegin, c.origin[axis]);
                const double second = fma(c.direction[axis], c.windowEnd, c.origin[axis]);
                const double slack  = (fabs(first) + fabs(second) + c.expansion[axis] + c.width) * 0x1p-40;
                int          high   = 0;
                (void)boxGridCell(fmin(first, second) - c.expansion[axis] - slack, c.inverseCell, c.low[axis]);
                (void)boxGridCell(fmax(first, second) + c.expansion[axis] + slack, c.inverseCell, high);
                c.low[axis] -= 1;
                c.span[axis] = static_cast<unsigned int>(high - c.low[axis] + 1);
                c.cells *= c.span[axis];
            }
            c.cell      = 0u;
            c.inSegment = true;
            c.segment += blockDim.x;
            continue;
        }
        if (c.over < c.oversizeCount)
        {
            object  = c.oversize[c.over];
            chained = false;
            return true;
        }
        return false;
    }
}

// Whether a candidate seen from a segment counts there: its entry distance
// lies in the segment's window (the last window includes its end).
__device__ bool cudaFamilyPathGridOwns(const PathGridCursor& c, bool chained, double entry)
{
    return !chained || (entry >= c.windowBegin && (entry < c.windowEnd || (c.lastWindow && entry <= c.windowEnd)));
}

// Whether a path row's walk accepts an object within `reach`, and the distance at which
// it enters the object's bounds. With `floatEntry` (an exact sweep's scan) that is the
// float cull's lower bound: it only orders and clips the exact tests, which reject what
// the double bounds test would, and that test costs a box sweep a 15-axis SAT per
// candidate. Otherwise the double bounds test decides: its hit is a bounds plan's result,
// and the path grid places each object in the segment its entry lies in.
template <bool BoxSweep>
__device__ bool cudaFamilyPathCandidate(const FamilyArguments& a, const RowQuery& q, const CullRay& rowCull,
                                        unsigned long long object, double reach, bool floatEntry, bool boxContacts,
                                        CudaFamilyHit& hit)
{
    const double* minimum = a.objectMinimums + object * 3u;
    const double* maximum = a.objectMaximums + object * 3u;
    float         entry   = 0.0f;
    if (!cudaFamilyCullObjectPath(rowCull, minimum, maximum, reach, entry))
    {
        return false;
    }
    if (floatEntry)
    {
        hit          = CudaFamilyHit{};
        hit.distance = static_cast<double>(entry);
        return true;
    }
    if constexpr (BoxSweep)
    {
        return cudaFamilyCandidateBoxSweepHit(q.queryBox, minimum, maximum, q.directionOrRotation, q.minimumDistance,
                                              reach, boxContacts, hit);
    }
    else
    {
        return cudaFamilyCandidatePathHit(minimum, maximum, q.translation, q.directionOrRotation, q.pathRadius,
                                          q.minimumDistance, reach, hit);
    }
}

// The contact of a single-hit box sweep's winner when its test left it out: a box
// in an exact plan, any object's bounds in a bounds plan. The interval is solved
// again over the whole path; its entry does not depend on the path's end.
__device__ void cudaFamilyBoxSweepWinnerContact(const FamilyArguments& a, const RowQuery& q, bool exact,
                                                unsigned long long object, CudaFamilyHit& hit)
{
    cudaFamily::BoxGeometry target{};
    if (exact)
    {
        const cudaFamily::ExactObjectHeader& header = a.exactObjects[object];
        if (header.kind != cudaFamily::kGeometryBox)
        {
            return;
        }
        target = a.boxes[header.familyIndex];
    }
    else
    {
        target = cudaFamilyBoundsBox(a.objectMinimums + object * 3u, a.objectMaximums + object * 3u);
    }
    const boxSweep::Box  query     = cudaFamilySweepBox(q.queryBox);
    const boxSweep::Box  box       = cudaFamilySweepBox(target);
    const boxSweep::Vec3 direction = cudaFamilySweepDirection(q.directionOrRotation);
    boxSweep::Interval   interval{};
    (void)boxSweep::boxBoxInterval(query, box, direction, q.minimumDistance, q.maximumDistance, interval);
    cudaFamilyBoxBoxContact(query, box, direction, interval, hit);
}

// The CPU's closestSweepClipBound: a few float roundoffs beyond a closest sweep's
// running best, so an exact test whose contact rounding places just past the best
// solves it, rather than meeting the reach first, and it can still tie or beat it.
__device__ double cudaFamilySweepClipBound(double best) { return best + 8.0 * 0x1p-23 * fmax(1.0, best); }

// Records one accepted candidate for thread 0: closest and any modes keep one
// hit, multiple mode insertion-sorts into the row's granted range when `writing`.
template <bool ExactSingle>
__device__ void cudaFamilyEmit(const FamilyArguments& a, const RowQuery& q, const CandidateSlot& slot, bool writing,
                               unsigned long long rowBase, unsigned long long rowCapacity, RowState& state)
{
    const unsigned long long objectId = a.objectIds[slot.object];
    const bool               exact    = ExactSingle || a.precision == cudaFamily::kPrecisionExact;
    if (a.hitMode != cudaFamily::kHitModeMultiple)
    {
        state.semanticCount = 1u;
        // An overlap's single hit is the lowest object id among the overlapping
        // objects, which the scan in object order also finds first. On a grid
        // the visit order is free, so an ANY row walks to the end and keeps the
        // same hit CLOSEST would (overlaps: the lowest id; paths: the nearest
        // entry, then the lowest id), and histories stay identical; the scan
        // keeps stopping at its first hit.
        const bool anyFirst = a.hitMode == cudaFamily::kHitModeAny && a.grid == nullptr;
        if (anyFirst)
        {
            state.stop = 1u;
        }
        const bool better =
            a.family == cudaFamily::kFamilyOverlap
                ? (state.closestFound == 0u || objectId < state.closestId)
                : (anyFirst ? state.closestFound == 0u
                            : (state.closestFound == 0u || slot.hit.distance < state.closestHit.distance ||
                               (slot.hit.distance == state.closestHit.distance && objectId < state.closestId)));
        if (better)
        {
            state.closestFound  = 1u;
            state.closestId     = objectId;
            state.closestObject = slot.object;
            state.closestHit    = slot.hit;
        }
        return;
    }
    ++state.semanticCount;
    if (!writing || rowCapacity == 0u)
    {
        return;
    }
    if (a.family == cudaFamily::kFamilyOverlap)
    {
        if (state.rowStored < rowCapacity || objectId < a.outputObjectIds[rowBase + state.rowStored - 1u])
        {
            unsigned long long insertion = state.rowStored < rowCapacity ? state.rowStored : rowCapacity - 1u;
            while (insertion != 0u && objectId < a.outputObjectIds[rowBase + insertion - 1u])
            {
                a.outputObjectIds[rowBase + insertion] = a.outputObjectIds[rowBase + insertion - 1u];
                if (a.outputMaterialIds != nullptr)
                {
                    a.outputMaterialIds[rowBase + insertion] = a.outputMaterialIds[rowBase + insertion - 1u];
                }
                if ((a.resultFields & cudaFamily::kResultGeometryIdentity) != 0u && a.outputGeometryTypes != nullptr &&
                    a.outputGeometryValues != nullptr)
                {
                    a.outputGeometryTypes[rowBase + insertion]  = a.outputGeometryTypes[rowBase + insertion - 1u];
                    a.outputGeometryValues[rowBase + insertion] = a.outputGeometryValues[rowBase + insertion - 1u];
                }
                --insertion;
            }
            a.outputObjectIds[rowBase + insertion] = objectId;
            if (a.outputMaterialIds != nullptr)
            {
                a.outputMaterialIds[rowBase + insertion] = slot.object | (kMaterialKeyNoFace << 32u);
            }
            if ((a.resultFields & cudaFamily::kResultGeometryIdentity) != 0u && a.outputGeometryTypes != nullptr &&
                a.outputGeometryValues != nullptr)
            {
                a.outputGeometryTypes[rowBase + insertion]  = static_cast<int>(a.geometryTypes[slot.object]);
                a.outputGeometryValues[rowBase + insertion] = a.geometryValues[slot.object];
            }
            if (state.rowStored < rowCapacity)
            {
                ++state.rowStored;
            }
        }
        return;
    }
    const bool beforeLast =
        state.rowStored < rowCapacity || (slot.hit.distance < a.outputDistances[rowBase + state.rowStored - 1u] ||
                                          (slot.hit.distance == a.outputDistances[rowBase + state.rowStored - 1u] &&
                                           objectId < a.outputObjectIds[rowBase + state.rowStored - 1u]));
    if (!beforeLast)
    {
        return;
    }
    unsigned long long insertion = state.rowStored < rowCapacity ? state.rowStored : rowCapacity - 1u;
    while (insertion != 0u && (slot.hit.distance < a.outputDistances[rowBase + insertion - 1u] ||
                               (slot.hit.distance == a.outputDistances[rowBase + insertion - 1u] &&
                                objectId < a.outputObjectIds[rowBase + insertion - 1u])))
    {
        cudaFamilyCopyOutputRow(rowBase + insertion, rowBase + insertion - 1u, a.outputObjectIds, a.outputGeometryTypes,
                                a.outputGeometryValues, a.outputPositions, a.outputQueryPositions, a.outputNormals,
                                a.outputDistances, a.outputFeatureKinds, a.outputFaceIndices, a.outputFaceCorners,
                                a.outputBarycentrics);
        if (a.outputMaterialIds != nullptr)
        {
            a.outputMaterialIds[rowBase + insertion] = a.outputMaterialIds[rowBase + insertion - 1u];
        }
        --insertion;
    }
    cudaFamilyWriteOutputRow(rowBase + insertion, objectId, a.geometryTypes[slot.object], a.geometryValues[slot.object],
                             slot.hit, exact, q.translation, q.directionOrRotation, a.resultFields, a.outputObjectIds,
                             a.outputGeometryTypes, a.outputGeometryValues, a.outputPositions, a.outputQueryPositions,
                             a.outputNormals, a.outputDistances, a.outputFeatureKinds, a.outputFaceIndices,
                             a.outputFaceCorners, a.outputBarycentrics);
    if (a.outputMaterialIds != nullptr)
    {
        a.outputMaterialIds[rowBase + insertion] = cudaFamilyMaterialKey(slot.object, slot.hit, exact);
    }
    if (state.rowStored < rowCapacity)
    {
        ++state.rowStored;
    }
}

// Records a MULTIPLE overlap window's hits (slot `threadIdx.x` per thread) when that
// equals thread 0's insertions in cudaFamilyEmit: a pass that does not write only counts
// them, and one that writes appends them at their ranks when their ids do not descend,
// from the last id stored, and all fit the row's grant. False, with nothing recorded,
// otherwise.
__device__ bool cudaFamilyAppendOverlaps(const FamilyArguments& a, const CandidateSlot* sharedSlots, unsigned int count,
                                         bool writing, unsigned long long rowBase, unsigned long long rowCapacity,
                                         RowState& state)
{
    constexpr unsigned int   kFullMask = 0xffffffffu;
    const unsigned int       lane      = threadIdx.x;
    const bool               hit       = lane < count && sharedSlots[lane].state == kSlotHit;
    const unsigned int       mask      = __ballot_sync(kFullMask, hit);
    const unsigned int       total     = __popc(mask);
    const unsigned int       before    = mask & ((1u << lane) - 1u);
    const unsigned long long stored    = state.rowStored;
    bool                     fits      = !writing || rowCapacity == 0u;
    unsigned long long       objectId  = 0u;
    if (!fits && stored + total <= rowCapacity)
    {
        const unsigned long long object   = hit ? sharedSlots[lane].object : 0u;
        objectId                          = hit ? a.objectIds[object] : 0u;
        const unsigned long long previous = __shfl_sync(kFullMask, objectId, before != 0u ? 31 - __clz(before) : 0);
        const bool               ascending =
            !hit || (before != 0u ? previous <= objectId
                                  : stored == 0u || a.outputObjectIds[rowBase + stored - 1u] <= objectId);
        fits = __all_sync(kFullMask, ascending);
        if (fits && hit)
        {
            const unsigned long long row = rowBase + stored + __popc(before);
            a.outputObjectIds[row]       = objectId;
            if (a.outputMaterialIds != nullptr)
            {
                a.outputMaterialIds[row] = object | (kMaterialKeyNoFace << 32u);
            }
            if ((a.resultFields & cudaFamily::kResultGeometryIdentity) != 0u && a.outputGeometryTypes != nullptr &&
                a.outputGeometryValues != nullptr)
            {
                a.outputGeometryTypes[row]  = static_cast<int>(a.geometryTypes[object]);
                a.outputGeometryValues[row] = a.geometryValues[object];
            }
        }
    }
    __syncwarp();
    if (fits && lane == 0u)
    {
        state.semanticCount += total;
        if (writing && rowCapacity != 0u)
        {
            state.rowStored += total;
        }
    }
    return fits;
}

#ifndef OVSPATIALQUERY_CUDA_CROSSINGS_APPEND
#define OVSPATIALQUERY_CUDA_CROSSINGS_APPEND 1
#endif

// Records MESH_HITS_ALL crossings, one warp-wide call per round. The evaluation
// pass only counts them. The write pass appends them in traversal order, every
// lane at once, when the evaluation found the row's crossings fit its grant
// (`append`); otherwise thread 0 keeps the nearest by insertion as cudaFamilyEmit
// does, so a truncated row holds the nearest crossings.
template <bool ExactSingle>
struct CrossingSink
{
    const FamilyArguments& a;
    const RowQuery&        q;
    RowState&              state;
    CandidateSlot*         stage;
    bool                   writing;
    bool                   append;
    unsigned long long     rowBase;
    unsigned long long     rowCapacity;

    __device__ void operator()(bool accepted, const CudaFamilyHit& hit, unsigned long long object)
    {
        constexpr unsigned int kFullMask = 0xffffffffu;
        const unsigned int     mask      = __ballot_sync(kFullMask, accepted);
        if (mask == 0u)
        {
            return;
        }
        const unsigned int lane  = threadIdx.x;
        const unsigned int total = __popc(mask);
        const unsigned int rank  = __popc(mask & ((1u << lane) - 1u));
        if (!writing)
        {
            if (lane == 0u)
            {
                state.semanticCount += total;
            }
            __syncwarp();
            return;
        }
        if (append)
        {
            const unsigned long long row = state.rowStored + rank;
            if (accepted && row < rowCapacity)
            {
                cudaFamilyWriteOutputRow(rowBase + row, a.objectIds[object], a.geometryTypes[object],
                                         a.geometryValues[object], hit, true, q.translation, q.directionOrRotation,
                                         a.resultFields, a.outputObjectIds, a.outputGeometryTypes,
                                         a.outputGeometryValues, a.outputPositions, a.outputQueryPositions,
                                         a.outputNormals, a.outputDistances, a.outputFeatureKinds, a.outputFaceIndices,
                                         a.outputFaceCorners, a.outputBarycentrics);
                if (a.outputMaterialIds != nullptr)
                {
                    a.outputMaterialIds[rowBase + row] = cudaFamilyMaterialKey(object, hit, true);
                }
            }
            __syncwarp();
            if (lane == 0u)
            {
                state.rowStored += total;
                state.semanticCount += total;
            }
            __syncwarp();
            return;
        }
        if (accepted)
        {
            stage[rank].hit    = hit;
            stage[rank].object = object;
            stage[rank].state  = kSlotHit;
        }
        __syncwarp();
        if (lane == 0u)
        {
            for (unsigned int index = 0u; index < total; ++index)
            {
                cudaFamilyEmit<ExactSingle>(a, q, stage[index], true, rowBase, rowCapacity, state);
            }
        }
        __syncwarp();
    }
};

// Evaluates one query against every object with the whole block: windows of
// blockDim.x objects pass the participation and bounds tests in parallel and the
// accepted objects are appended to the candidate slots by ballot rank; once the
// slots are full or the objects are exhausted, the accepted primitives are
// evaluated one per thread, each accepted mesh is traversed cooperatively, and
// thread 0 records the results in object order. A query with few candidates, the
// usual case over many instances, therefore evaluates one slot window instead of
// one window per blockDim.x objects, and its meshes are ordered nearest first
// across the whole population rather than within one object window.
// Exact closest paths traverse the accepted meshes nearest first and clip every
// later traversal to the running best contact, as the CPU's
// evaluateOrderedClosestSweep does: a mesh entered beyond the best contact cannot
// beat or tie it, and a traversal bounded by it visits far fewer nodes. The emitted
// candidates, the exact predicates and the tie rule (distance, then lower object
// id) are unchanged. An ANY sweep clips the same way: any contact qualifies, so a
// mesh entered beyond one already found need not be traversed (ADR-0067).
template <bool ExactSingle, bool BoxSweep, bool Crossings = false, bool Overlap = false>
__device__ void cudaFamilyEvaluateRow(const FamilyArguments& a, const RowQuery& q, bool writing,
                                      unsigned long long rowBase, unsigned long long rowCapacity, RowState& state,
                                      CandidateSlot* sharedSlots, unsigned int* sharedOrder, double* sharedClip,
                                      unsigned int* sharedFound, MeshBvhTraversal* sharedTraversal,
                                      BoxSweepQueue* sharedQueue, CandidateSlot* sharedStage = nullptr,
                                      bool append = false)
{
    CrossingSink<ExactSingle> crossings{a, q, state, sharedStage, writing, append, rowBase, rowCapacity};
    // Overlap rows run in their own kernel entries, so the others carry none of their code.
    const bool overlapRow = Overlap && a.family == cudaFamily::kFamilyOverlap;
    const bool exact      = ExactSingle || a.precision == cudaFamily::kPrecisionExact;
    const bool clipMeshes = exact && !overlapRow &&
                            (a.hitMode == cudaFamily::kHitModeClosest ||
                             (a.hitMode == cudaFamily::kHitModeAny && a.family != cudaFamily::kFamilyRaycast));
    // Once a single-hit sweep holds a contact (MULTIPLE rows keep none), each round lowers
    // the reach of the walk, its candidate tests and its primitive tests to the running
    // best's clip bound, as the CPU's closestSweepReach does: an object entered beyond it can
    // neither beat nor tie the best. Rays keep their whole path.
    const bool sweep = a.family == cudaFamily::kFamilySphereSweep || a.family == cudaFamily::kFamilyBoxSweep;
    // Only the rows a MULTIPLE write pass stores need each box-box contact; a single-hit
    // row computes its winner's after the walk (cudaFamilyBoxSweepWinnerContact).
    const bool boxContacts = a.hitMode == cudaFamily::kHitModeMultiple && writing;
    CullRay    rowCull{};
    double     expansion[3] = {};
    if (!overlapRow)
    {
        double direction[3] = {};
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            direction[axis] = static_cast<double>(q.directionOrRotation[axis]);
            expansion[axis] = BoxSweep ? fabs(q.queryBox.axes[axis]) * q.queryBox.halfExtents[0] +
                                             fabs(q.queryBox.axes[3u + axis]) * q.queryBox.halfExtents[1] +
                                             fabs(q.queryBox.axes[6u + axis]) * q.queryBox.halfExtents[2]
                                       : q.pathRadius;
        }
        cudaFamilyMakeCullRay(q.translation, direction, expansion, q.minimumDistance, rowCull);
    }
    if (threadIdx.x == 0u)
    {
        state.semanticCount = 0u;
        state.rowStored     = 0u;
        state.closestFound  = 0u;
        state.stop          = 0u;
    }
    __syncthreads();
    // A device-prepared version's grid replaces the scan for every overlap kind
    // and for path rows: the block walks the cells the query (or each segment of
    // the path) covers and the oversize list, and every object is linked in one
    // cell or listed once, so it is visited at most once. The emit orders hits by
    // object id, so the visit order does not change a history. The scan remains
    // without a grid and for a query the grid cannot serve (not finite, too many
    // cells, or beyond the key range).
    GridCursor     cursor{};
    PathGridCursor path{};
    bool           gridded     = false;
    bool           pathGridded = false;
    if constexpr (!BoxSweep)
    {
        gridded = a.grid != nullptr && overlapRow && cudaFamilyGridRange(a, q, cursor);
    }
    if (a.grid != nullptr && !overlapRow)
    {
        pathGridded = cudaFamilyPathGridBegin(a, q, expansion, path);
    }
    OverlapGate gate{};
    if (!BoxSweep && overlapRow && exact)
    {
        cudaFamilyMakeOverlapGate(a.overlapKind, q.translation, q.overlapRadius, q.queryBox, gate);
    }
    unsigned long long base      = 0u;
    bool               exhausted = !gridded && !pathGridded && a.objectCount == 0u;
    while (!exhausted && state.stop == 0u)
    {
        const double reach = sweep && state.closestFound != 0u
                                 ? fmin(q.maximumDistance, cudaFamilySweepClipBound(state.closestHit.distance))
                                 : q.maximumDistance;
        unsigned int count = 0u; // slots filled; uniform across the block
        while (pathGridded)
        {
            unsigned long long object  = 0u;
            bool               chained = false;
            const bool         has     = cudaFamilyPathGridPeek(path, reach, object, chained);
            if (__ballot_sync(0xffffffffu, has) == 0u)
            {
                exhausted = true;
                break;
            }
            CudaFamilyHit boundsHit{};
            const bool    accepted =
                has && cudaFamilyParticipates(a.objectBits, object, q.acceptedBits) &&
                cudaFamilyPathCandidate<BoxSweep>(a, q, rowCull, object, reach, false, boxContacts, boundsHit) &&
                cudaFamilyPathGridOwns(path, chained, boundsHit.distance);
            const unsigned int mask  = __ballot_sync(0xffffffffu, accepted);
            const unsigned int total = __popc(mask);
            if (count + total > blockDim.x)
            {
                break; // the full slots are evaluated first; nothing is consumed
            }
            if (accepted)
            {
                CandidateSlot& slot = sharedSlots[count + __popc(mask & ((1u << threadIdx.x) - 1u))];
                slot.object         = object;
                slot.hit            = boundsHit;
                slot.state          = !exact                                                             ? kSlotHit
                                      : a.exactObjects[object].kind == cudaFamily::kGeometryTriangleMesh ? kSlotMesh
                                                                                                         : kSlotPrimitive;
            }
            if (has)
            {
                cudaFamilyGridConsume(path, chained);
            }
            count += total;
        }
        while (gridded)
        {
            unsigned long long object  = 0u;
            bool               chained = false;
            const bool         has     = cudaFamilyGridPeek(cursor, object, chained);
            if (__ballot_sync(0xffffffffu, has) == 0u)
            {
                exhausted = true;
                break;
            }
            const bool accepted = has && cudaFamilyParticipates(a.objectBits, object, q.acceptedBits) &&
                                  cudaFamilyCandidateOverlap(a.overlapKind, q.translation, q.overlapRadius, q.rotation,
                                                             q.queryBox.halfExtents, a.objectMinimums + object * 3u,
                                                             a.objectMaximums + object * 3u, exact);
            const unsigned int mask  = __ballot_sync(0xffffffffu, accepted);
            const unsigned int total = __popc(mask);
            if (count + total > blockDim.x)
            {
                break; // the full slots are evaluated first; nothing is consumed
            }
            if (accepted)
            {
                CandidateSlot& slot = sharedSlots[count + __popc(mask & ((1u << threadIdx.x) - 1u))];
                slot.object         = object;
                slot.hit            = CudaFamilyHit{};
                slot.state          = !exact                                                             ? kSlotHit
                                      : a.exactObjects[object].kind == cudaFamily::kGeometryTriangleMesh ? kSlotMesh
                                                                                                         : kSlotPrimitive;
            }
            if (has)
            {
                cudaFamilyGridConsume(cursor, chained);
            }
            count += total;
        }
        for (; !gridded && !pathGridded && base < a.objectCount; base += blockDim.x)
        {
            const unsigned long long object   = base + threadIdx.x;
            bool                     accepted = false;
            CudaFamilyHit            boundsHit{};
            if (object < a.objectCount && cudaFamilyParticipates(a.objectBits, object, q.acceptedBits))
            {
                accepted =
                    !BoxSweep && overlapRow
                        ? (exact
                               ? cudaFamilyExactOverlapCandidate(
                                     gate, a.exactObjects[object].kind == cudaFamily::kGeometryTriangleMesh,
                                     a.overlapKind, q.translation, q.overlapRadius, q.rotation, q.queryBox.halfExtents,
                                     a.objectMinimums + object * 3u, a.objectMaximums + object * 3u)
                               : cudaFamilyCandidateOverlap(a.overlapKind, q.translation, q.overlapRadius, q.rotation,
                                                            q.queryBox.halfExtents, a.objectMinimums + object * 3u,
                                                            a.objectMaximums + object * 3u, false))
                        : cudaFamilyPathCandidate<BoxSweep>(a, q, rowCull, object, reach, exact && sweep, boxContacts,
                                                            boundsHit);
            }
            const unsigned int mask  = __ballot_sync(0xffffffffu, accepted);
            const unsigned int total = __popc(mask);
            if (count + total > blockDim.x)
            {
                // The full slots are evaluated first; this window is tested again afterwards.
                break;
            }
            if (accepted)
            {
                CandidateSlot& slot = sharedSlots[count + __popc(mask & ((1u << threadIdx.x) - 1u))];
                slot.object         = object;
                // The bounds entry stays in the slot until the exact evaluation replaces it.
                slot.hit   = boundsHit;
                slot.state = !exact                                                             ? kSlotHit
                             : a.exactObjects[object].kind == cudaFamily::kGeometryTriangleMesh ? kSlotMesh
                                                                                                : kSlotPrimitive;
            }
            count += total;
        }
        if (!gridded && !pathGridded)
        {
            exhausted = base >= a.objectCount;
        }
        __syncthreads();
        for (unsigned int index = threadIdx.x; index < count; index += blockDim.x)
        {
            CandidateSlot& slot = sharedSlots[index];
            if (!BoxSweep && overlapRow && slot.state == kSlotMesh)
            {
                slot.state = cudaFamilyScreenMeshOverlap(a, gate, slot.object, sharedTraversal->stack + threadIdx.x);
                continue;
            }
            if (slot.state != kSlotPrimitive)
            {
                continue;
            }
            const cudaFamily::ExactObjectHeader& header = a.exactObjects[slot.object];
            bool                                 hit    = false;
            if constexpr (BoxSweep)
            {
                hit = cudaFamilyExactBoxSweepHit(
                    header, a.spheres, a.boxes, a.capsules, a.convexes, a.convexVertices, a.convexTriangleIndices,
                    a.convexPlanes, q.queryBox, q.directionOrRotation, q.minimumDistance, reach, boxContacts, slot.hit);
            }
            else if (overlapRow)
            {
                hit      = cudaFamilyExactOverlap(header, a.spheres, a.boxes, a.capsules, a.convexes, a.convexVertices,
                                                  a.convexTriangleIndices, a.convexPlanes, a.convexEdges, a.overlapKind,
                                                  q.translation, q.overlapRadius, &q.queryBox);
                slot.hit = CudaFamilyHit{};
            }
            else
            {
                hit = cudaFamilyExactPathHit(header, a.spheres, a.boxes, a.capsules, a.convexes, a.convexVertices,
                                             a.convexTriangleIndices, a.convexPlanes, q.translation,
                                             q.directionOrRotation, q.pathRadius, q.minimumDistance, reach, slot.hit);
            }
            slot.state = hit ? kSlotHit : kSlotMiss;
        }
        __syncthreads();
        // A window without meshes left to traverse needs neither their order nor the clip.
        const bool meshes = __any_sync(0xffffffffu, threadIdx.x < count && sharedSlots[threadIdx.x].state == kSlotMesh);
        if (meshes && threadIdx.x == 0u)
        {
            // Order the accepted meshes by bounds entry (then slot order) and seed the clip
            // with the best contact so far, including this window's primitive hits.
            unsigned int meshCount = 0u;
            double       clip      = q.maximumDistance;
            if (clipMeshes && state.closestFound != 0u)
            {
                clip = fmin(clip, state.closestHit.distance);
            }
            for (unsigned int index = 0u; index < count; ++index)
            {
                const CandidateSlot& slot = sharedSlots[index];
                if (slot.state == kSlotHit && clipMeshes)
                {
                    clip = fmin(clip, slot.hit.distance);
                }
                if (slot.state != kSlotMesh)
                {
                    continue;
                }
                unsigned int insertion = meshCount;
                while (insertion != 0u && sharedSlots[sharedOrder[insertion - 1u]].hit.distance > slot.hit.distance)
                {
                    sharedOrder[insertion] = sharedOrder[insertion - 1u];
                    --insertion;
                }
                sharedOrder[insertion] = index;
                ++meshCount;
            }
            sharedOrder[blockDim.x] = meshCount;
            *sharedClip             = clip;
        }
        __syncthreads();
        const unsigned int meshCount = meshes ? sharedOrder[blockDim.x] : 0u;
        for (unsigned int ordered = 0u; ordered < meshCount; ++ordered)
        {
            const unsigned int                   index  = sharedOrder[ordered];
            const cudaFamily::ExactObjectHeader& header = a.exactObjects[sharedSlots[index].object];
            const cudaFamily::MeshGeometry&      mesh   = a.meshes[header.familyIndex];
            const bool    storedDoubleSided             = (header.flags & cudaFamily::kGeometryFlagDoubleSided) != 0u;
            const double  clip                          = *sharedClip;
            bool          hit                           = false;
            CudaFamilyHit meshHit{};
            if constexpr (Crossings)
            {
                cudaFamilyMeshPathCrossings(mesh, a.meshDefinitions, q.bothSides || storedDoubleSided,
                                            storedDoubleSided, ExactSingle, q.translation, q.directionOrRotation,
                                            q.minimumDistance, q.maximumDistance, sharedTraversal,
                                            sharedSlots[index].object, crossings);
            }
            else if (clipMeshes && sharedSlots[index].hit.distance > clip)
            {
                // Entered beyond the best contact: it cannot beat or tie it.
            }
            else if constexpr (BoxSweep)
            {
                hit = cudaFamilyMeshBoxSweepHit(
                    mesh, a.meshDefinitions, q.bothSides || storedDoubleSided, q.queryBox, q.directionOrRotation,
                    q.minimumDistance, clipMeshes ? clip : q.maximumDistance, sharedTraversal, sharedQueue, meshHit);
            }
            else if (overlapRow)
            {
                const unsigned long long object = sharedSlots[index].object;
                hit = cudaFamilyMeshOverlap(mesh, a.meshDefinitions, a.overlapKind, q.translation, q.overlapRadius,
                                            q.rotation, q.queryBox, gate, a.objectMinimums + object * 3u,
                                            a.objectMaximums + object * 3u, sharedFound, sharedTraversal);
            }
            else
            {
                // A sphere sweep keeps its normal; only a ray follows the back-face normal rule.
                hit = cudaFamilyMeshPathHit(mesh, a.meshDefinitions, q.bothSides || storedDoubleSided,
                                            storedDoubleSided || a.family != cudaFamily::kFamilyRaycast, ExactSingle,
                                            q.translation, q.directionOrRotation, q.pathRadius, q.minimumDistance,
                                            clipMeshes ? clip : q.maximumDistance, sharedTraversal, meshHit);
            }
            if (threadIdx.x == 0u)
            {
                sharedSlots[index].state = hit ? kSlotHit : kSlotMiss;
                sharedSlots[index].hit   = meshHit;
                if (clipMeshes && hit)
                {
                    *sharedClip = fmin(clip, meshHit.distance);
                }
            }
            __syncthreads();
        }
        if constexpr (Crossings)
        {
            // The meshes recorded their crossings during traversal; their slots hold no hit.
            for (unsigned int first = 0u; first < count; first += blockDim.x)
            {
                const unsigned int index = first + threadIdx.x;
                const bool         accepted =
                    index < count && sharedSlots[index].state == kSlotHit &&
                    a.exactObjects[sharedSlots[index].object].kind != cudaFamily::kGeometryTriangleMesh;
                crossings(accepted, index < count ? sharedSlots[index].hit : CudaFamilyHit{},
                          index < count ? sharedSlots[index].object : 0u);
            }
        }
        else
        {
            // Every overlap a scan finds comes in object order, which is usually id order:
            // when this window's hits ascend from the last stored id and fit the grant, each
            // thread writes its own at its rank, as thread 0's insertions would have.
            bool appended = false;
            if (!BoxSweep && overlapRow && a.hitMode == cudaFamily::kHitModeMultiple)
            {
                appended = cudaFamilyAppendOverlaps(a, sharedSlots, count, writing, rowBase, rowCapacity, state);
            }
            if (!appended && threadIdx.x == 0u)
            {
                for (unsigned int index = 0u; index < count && state.stop == 0u; ++index)
                {
                    if (sharedSlots[index].state == kSlotHit)
                    {
                        cudaFamilyEmit<ExactSingle>(a, q, sharedSlots[index], writing, rowBase, rowCapacity, state);
                    }
                }
            }
        }
        __syncthreads();
    }
}

} // namespace

// One block per query. The evaluation pass records each row's hit count (and the
// closest or first hit) in the row plans; the scan kernel below turns those into
// dense offsets under the slot capacity; the output pass writes the granted rows.
template <bool ExactSingle, bool BoxSweep, bool Crossings = false, bool Overlap = false>
__device__ __forceinline__ void cudaFamilyPlanBody(const cudaFamily::PlanLaunchRecord& launch,
                                                   CandidateSlot* sharedSlots, unsigned int* sharedOrder,
                                                   double* sharedClip, unsigned int* sharedFound, RowState& state,
                                                   MeshBvhTraversal* sharedTraversal, BoxSweepQueue* sharedQueue,
                                                   CandidateSlot* sharedStage = nullptr)
{
    // The record's fields are read into values, never pointed at: see rowHalfExtents below.
    const unsigned int        pass                          = launch.pass;
    const unsigned int        family                        = launch.arguments.family;
    const bool                overlapRow                    = Overlap && family == cudaFamily::kFamilyOverlap;
    const unsigned int        precision                     = launch.arguments.precision;
    const unsigned int        overlap_kind                  = launch.arguments.overlapKind;
    const unsigned int        hit_mode                      = launch.arguments.hitMode;
    const unsigned int        result_fields                 = launch.arguments.resultFields;
    const double              contact_tolerance             = launch.contactTolerance;
    const unsigned long long  maximum_hits_per_query        = launch.maximumHitsPerQuery;
    const unsigned long long  query_count                   = launch.queryCount;
    const unsigned long long  result_capacity               = launch.resultCapacity;
    const double              plan_radius                   = launch.planRadius;
    const unsigned long long* query_bits                    = launch.queryBits;
    const double*             query_positions               = launch.queryPositions;
    const float*              query_directions_or_rotations = launch.queryDirectionsOrRotations;
    const float*              query_rotations               = launch.queryRotations;
    const double*             minimum_distances             = launch.minimumDistances;
    const double*             maximum_distances             = launch.maximumDistances;
    const double*             radii                         = launch.radii;
    const unsigned long long* row_maximum_hits              = launch.rowMaximumHits;
    unsigned int*             query_status                  = launch.queryStatus;
    unsigned long long*       hit_offsets                   = launch.hitOffsets;
    cudaFamily::RowPlan*      row_plans                     = launch.rowPlans;
    const unsigned int*       preparation_failed            = launch.preparationFailed;
    const double*             query_half_extents            = launch.queryHalfExtents;
    const unsigned int        query_flags                   = launch.queryFlags;
    const unsigned int        rotation_bits                 = launch.rotationBits;
    const unsigned long long  query                         = blockIdx.x;
    if (query >= query_count)
    {
        return;
    }
    // The selected publication was admitted but failed before this launch ran
    // (ADR-0021): every row reports FAILED with zero hits, the safe empty result.
    if (preparation_failed != nullptr && *preparation_failed != 0u)
    {
        if (pass == cudaFamily::kPassEvaluate && threadIdx.x == 0u)
        {
            query_status[query] = cudaFamily::kBatchRowFailed;
            row_plans[query]    = cudaFamily::RowPlan{};
        }
        return;
    }
    // Copied field by field: written as one aggregate assignment, the copy
    // spilled 64 more bytes to local memory in the plan entry.
    FamilyArguments a{};
    a.family                = BoxSweep ? cudaFamily::kFamilyBoxSweep : family;
    a.precision             = ExactSingle ? cudaFamily::kPrecisionExactSingle : precision;
    a.overlapKind           = launch.arguments.overlapKind;
    a.hitMode               = launch.arguments.hitMode;
    a.resultFields          = launch.arguments.resultFields;
    a.planHalfExtents[0]    = launch.arguments.planHalfExtents[0];
    a.planHalfExtents[1]    = launch.arguments.planHalfExtents[1];
    a.planHalfExtents[2]    = launch.arguments.planHalfExtents[2];
    a.objectCount           = launch.arguments.objectCount;
    a.objectIds             = launch.arguments.objectIds;
    a.geometryTypes         = launch.arguments.geometryTypes;
    a.geometryValues        = launch.arguments.geometryValues;
    a.objectMaterials       = launch.arguments.objectMaterials;
    a.faceMaterials         = launch.arguments.faceMaterials;
    a.outputMaterialIds     = launch.arguments.outputMaterialIds;
    a.objectBits            = launch.arguments.objectBits;
    a.objectMinimums        = launch.arguments.objectMinimums;
    a.objectMaximums        = launch.arguments.objectMaximums;
    a.exactObjects          = launch.arguments.exactObjects;
    a.spheres               = launch.arguments.spheres;
    a.boxes                 = launch.arguments.boxes;
    a.capsules              = launch.arguments.capsules;
    a.convexes              = launch.arguments.convexes;
    a.convexVertices        = launch.arguments.convexVertices;
    a.convexTriangleIndices = launch.arguments.convexTriangleIndices;
    a.convexPlanes          = launch.arguments.convexPlanes;
    a.convexEdges           = launch.arguments.convexEdges;
    a.meshes                = launch.arguments.meshes;
    a.meshDefinitions       = launch.arguments.meshDefinitions;
    a.outputObjectIds       = launch.arguments.outputObjectIds;
    a.outputGeometryTypes   = launch.arguments.outputGeometryTypes;
    a.outputGeometryValues  = launch.arguments.outputGeometryValues;
    a.outputPositions       = launch.arguments.outputPositions;
    a.outputQueryPositions  = launch.arguments.outputQueryPositions;
    a.outputNormals         = launch.arguments.outputNormals;
    a.outputDistances       = launch.arguments.outputDistances;
    a.outputFeatureKinds    = launch.arguments.outputFeatureKinds;
    a.outputFaceIndices     = launch.arguments.outputFaceIndices;
    a.outputFaceCorners     = launch.arguments.outputFaceCorners;
    a.outputBarycentrics    = launch.arguments.outputBarycentrics;
    a.stash                 = launch.arguments.stash;
    a.grid                  = launch.arguments.grid;

    // Every thread derives the row description; the validation is deterministic.
    RowQuery q{};
    q.translation = query_positions == nullptr ? nullptr : query_positions + query * 3u;
    q.directionOrRotation =
        query_directions_or_rotations == nullptr
            ? nullptr
            : query_directions_or_rotations + query * (overlapRow && overlap_kind == cudaFamily::kOverlapBox ? 4u : 3u);
    // A BOX overlap row's rotation in double, from a float64 or float32 column (ADR-0026).
    const bool boxRow =
        overlapRow && overlap_kind == cudaFamily::kOverlapBox && query_directions_or_rotations != nullptr;
    for (unsigned int component = 0u; component < 4u; ++component)
    {
        q.rotation[component] =
            !boxRow ? (component == 3u ? 1.0 : 0.0)
            : rotation_bits == 64u
                ? reinterpret_cast<const double*>(query_directions_or_rotations)[query * 4u + component]
                : static_cast<double>(q.directionOrRotation[component]);
    }
    // A sphere sweep or SPHERE overlap row may carry its own radius; the
    // launch validation admits the column only for those two.
    const double rowRadius = radii != nullptr ? radii[query] : plan_radius;
    q.bothSides            = (query_flags & 1u) != 0u; // OVSPATIALQUERY_QUERY_MESH_BOTH_SIDES
    q.pathRadius =
        family == cudaFamily::kFamilyRaycast
            ? contact_tolerance
            : (family == cudaFamily::kFamilySphereSweep ? cudaFamilyAddTolerance(rowRadius, contact_tolerance) : 0.0);
    q.overlapRadius   = cudaFamilyAddTolerance(rowRadius, contact_tolerance);
    q.minimumDistance = minimum_distances == nullptr ? 0.0 : minimum_distances[query];
    q.maximumDistance = maximum_distances == nullptr ? 0.0 : maximum_distances[query];
    q.acceptedBits[0] = query_bits == nullptr ? 0u : query_bits[query * 2u];
    q.acceptedBits[1] = query_bits == nullptr ? 0u : query_bits[query * 2u + 1u];
    // A box sweep, AABB or BOX row may carry its own half extents; the plan's
    // otherwise. Raycast and sphere-sweep launches never set the half-extents
    // column (the public enqueue sets it only for box sweeps and overlaps), so
    // their rows take the plan's values here and never use them. The three
    // values are copied, not pointed at: a pointer into the arguments record
    // would spill it to local memory, whose growth makes the first launch
    // synchronize the device, and a producer the caller holds pending would
    // then deadlock the host in enqueue.
    double rowHalfExtents[3];
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        rowHalfExtents[axis] =
            query_half_extents != nullptr ? query_half_extents[query * 3u + axis] : a.planHalfExtents[axis];
    }
    bool valid = false;
    if constexpr (BoxSweep)
    {
        valid = q.translation != nullptr && q.directionOrRotation != nullptr && maximum_distances != nullptr &&
                validPath(q.translation, q.directionOrRotation, q.minimumDistance, q.maximumDistance, 0.0) &&
                query_rotations != nullptr &&
                cudaFamilyMakeQueryBox(q.translation, query_rotations + query * 4u, rowHalfExtents, contact_tolerance,
                                       q.queryBox);
    }
    else if (!overlapRow)
    {
        valid = q.translation != nullptr && q.directionOrRotation != nullptr &&
                (family != cudaFamily::kFamilySphereSweep || (cudaFamilyInDomain(rowRadius) && rowRadius > 0.0)) &&
                maximum_distances != nullptr &&
                validPath(q.translation, q.directionOrRotation, q.minimumDistance, q.maximumDistance, q.pathRadius);
    }
    else
    {
        valid = cudaFamilyOverlapRowIsValid(overlap_kind, q.translation, q.rotation, rowHalfExtents, rowRadius,
                                            contact_tolerance, q.queryBox);
    }
    q.semanticCapacity = 1u;
    if (hit_mode == cudaFamily::kHitModeMultiple)
    {
        q.semanticCapacity = row_maximum_hits == nullptr ? maximum_hits_per_query : row_maximum_hits[query];
        q.semanticCapacity = q.semanticCapacity > maximum_hits_per_query ? maximum_hits_per_query : q.semanticCapacity;
    }
    const bool single = hit_mode != cudaFamily::kHitModeMultiple;
    if (!valid)
    {
        if (pass == cudaFamily::kPassEvaluate && threadIdx.x == 0u)
        {
            query_status[query] = cudaFamily::kBatchRowInvalidArgument;
            row_plans[query]    = cudaFamily::RowPlan{};
        }
        return;
    }
    unsigned long long rowBase     = 0u;
    unsigned long long rowCapacity = 0u;
    bool               append      = false;
    // Row 0's hits start at offset 0 and its grant is the lesser of its capacity and the
    // slot's, so a MULTIPLE overlap row 0 writes them while it counts them, as the write
    // pass would, and the write pass leaves it alone: a giant overlap, usually a launch's
    // only row, is evaluated once. Other families keep the write pass, whose path rows
    // append crossings only once their count is known to fit.
    const bool countWrites =
        !BoxSweep && !Crossings && overlapRow && hit_mode == cudaFamily::kHitModeMultiple && query == 0u;
    if (pass == cudaFamily::kPassWrite && countWrites)
    {
        return;
    }
    if (countWrites)
    {
        rowCapacity = q.semanticCapacity < result_capacity ? q.semanticCapacity : result_capacity;
    }
    if (pass == cudaFamily::kPassWrite)
    {
        const cudaFamily::RowPlan plan = row_plans[query];
        if (plan.granted == 0u)
        {
            return;
        }
        rowBase     = hit_offsets[query];
        rowCapacity = plan.granted;
        append      = OVSPATIALQUERY_CUDA_CROSSINGS_APPEND != 0 && plan.semanticCount <= plan.granted;
        if (single)
        {
            if (threadIdx.x == 0u)
            {
                const StashRow& row = reinterpret_cast<const StashRow*>(a.stash)[query];
                cudaFamilyWriteOutputRow(rowBase, row.objectId, row.geometryType, row.geometryValue, row.hit,
                                         ExactSingle || precision == cudaFamily::kPrecisionExact, q.translation,
                                         q.directionOrRotation, result_fields, a.outputObjectIds, a.outputGeometryTypes,
                                         a.outputGeometryValues, a.outputPositions, a.outputQueryPositions,
                                         a.outputNormals, a.outputDistances, a.outputFeatureKinds, a.outputFaceIndices,
                                         a.outputFaceCorners, a.outputBarycentrics);
                if (a.outputMaterialIds != nullptr)
                {
                    a.outputMaterialIds[rowBase] = row.materialId;
                }
            }
            return;
        }
    }
    else if (threadIdx.x == 0u)
    {
        query_status[query] = cudaFamily::kBatchRowSuccess;
    }
    const bool writing = pass == cudaFamily::kPassWrite || countWrites;
    cudaFamilyEvaluateRow<ExactSingle, BoxSweep, Crossings, Overlap>(a, q, writing, rowBase, rowCapacity, state,
                                                                     sharedSlots, sharedOrder, sharedClip, sharedFound,
                                                                     sharedTraversal, sharedQueue, sharedStage, append);
    if (writing && !single && a.outputMaterialIds != nullptr)
    {
        __syncwarp();
        const unsigned long long stored = state.rowStored < rowCapacity ? state.rowStored : rowCapacity;
        cudaFamilyResolveMaterialKeys(a, rowBase, stored, ExactSingle || precision == cudaFamily::kPrecisionExact);
    }
    if (pass == cudaFamily::kPassEvaluate && threadIdx.x == 0u)
    {
        cudaFamily::RowPlan plan{};
        plan.semanticCount = state.semanticCount;
        plan.desired       = single ? (state.closestFound != 0u ? 1u : 0u)
                                    : (state.semanticCount < q.semanticCapacity ? state.semanticCount : q.semanticCapacity);
        plan.valid         = 1u;
        row_plans[query]   = plan;
        if (single)
        {
            if constexpr (BoxSweep)
            {
                if (state.closestFound != 0u)
                {
                    cudaFamilyBoxSweepWinnerContact(a, q, precision == cudaFamily::kPrecisionExact, state.closestObject,
                                                    state.closestHit);
                }
            }
            StashRow row{};
            row.hit = state.closestHit;
            if (state.closestFound != 0u)
            {
                const unsigned long long object = state.closestObject;
                row.objectId                    = state.closestId;
                row.geometryType                = a.geometryTypes[object];
                row.geometryValue               = a.geometryValues[object];
                row.materialId                  = cudaFamilyHitMaterial(a, object, state.closestHit,
                                                                        ExactSingle || precision == cudaFamily::kPrecisionExact);
            }
            reinterpret_cast<StashRow*>(a.stash)[query] = row;
        }
    }
}

extern "C" __global__ void __launch_bounds__(cudaFamily::kEvaluateThreads, cudaFamily::kEvaluateBlocksPerMultiprocessor)
    ovspatialquery_cuda_family_plan(const cudaFamily::PlanLaunchRecord launch)
{
    __shared__ CandidateSlot    sharedSlots[cudaFamily::kEvaluateThreads];
    __shared__ unsigned int     sharedOrder[cudaFamily::kEvaluateThreads + 1u];
    __shared__ double           sharedClip;
    __shared__ unsigned int     sharedFound;
    __shared__ RowState         state;
    __shared__ MeshBvhTraversal sharedTraversal;
    cudaFamilyPlanBody<false, false>(launch, sharedSlots, sharedOrder, &sharedClip, &sharedFound, state,
                                     &sharedTraversal, nullptr);
}

extern "C" __global__ void __launch_bounds__(cudaFamily::kEvaluateThreads, cudaFamily::kEvaluateBlocksPerMultiprocessor)
    ovspatialquery_cuda_family_plan_single(const cudaFamily::PlanLaunchRecord launch)
{
    __shared__ CandidateSlot    sharedSlots[cudaFamily::kEvaluateThreads];
    __shared__ unsigned int     sharedOrder[cudaFamily::kEvaluateThreads + 1u];
    __shared__ double           sharedClip;
    __shared__ unsigned int     sharedFound;
    __shared__ RowState         state;
    __shared__ MeshBvhTraversal sharedTraversal;
    cudaFamilyPlanBody<true, false>(launch, sharedSlots, sharedOrder, &sharedClip, &sharedFound, state,
                                    &sharedTraversal, nullptr);
}

// Overlap rows. Their own entries keep the overlap gate, screen and append code, and the
// registers it holds, out of the path entries, which every family shared until the gate
// spilled the ray traversal's state to local memory.
extern "C" __global__ void __launch_bounds__(cudaFamily::kEvaluateThreads, cudaFamily::kEvaluateBlocksPerMultiprocessor)
    ovspatialquery_cuda_family_plan_overlap(const cudaFamily::PlanLaunchRecord launch)
{
    __shared__ CandidateSlot    sharedSlots[cudaFamily::kEvaluateThreads];
    __shared__ unsigned int     sharedOrder[cudaFamily::kEvaluateThreads + 1u];
    __shared__ double           sharedClip;
    __shared__ unsigned int     sharedFound;
    __shared__ RowState         state;
    __shared__ MeshBvhTraversal sharedTraversal;
    cudaFamilyPlanBody<false, false, false, true>(launch, sharedSlots, sharedOrder, &sharedClip, &sharedFound, state,
                                                  &sharedTraversal, nullptr);
}

extern "C" __global__ void __launch_bounds__(cudaFamily::kEvaluateThreads, cudaFamily::kEvaluateBlocksPerMultiprocessor)
    ovspatialquery_cuda_family_plan_overlap_single(const cudaFamily::PlanLaunchRecord launch)
{
    __shared__ CandidateSlot    sharedSlots[cudaFamily::kEvaluateThreads];
    __shared__ unsigned int     sharedOrder[cudaFamily::kEvaluateThreads + 1u];
    __shared__ double           sharedClip;
    __shared__ unsigned int     sharedFound;
    __shared__ RowState         state;
    __shared__ MeshBvhTraversal sharedTraversal;
    cudaFamilyPlanBody<true, false, false, true>(launch, sharedSlots, sharedOrder, &sharedClip, &sharedFound, state,
                                                 &sharedTraversal, nullptr);
}

// MESH_HITS_ALL raycasts: every triangle crossing is a hit. Their own entries
// keep the closest and per-object paths' code unchanged.
extern "C" __global__ void __launch_bounds__(cudaFamily::kEvaluateThreads, cudaFamily::kEvaluateBlocksPerMultiprocessor)
    ovspatialquery_cuda_family_plan_crossings(const cudaFamily::PlanLaunchRecord launch)
{
    __shared__ CandidateSlot    sharedSlots[cudaFamily::kEvaluateThreads];
    __shared__ unsigned int     sharedOrder[cudaFamily::kEvaluateThreads + 1u];
    __shared__ double           sharedClip;
    __shared__ unsigned int     sharedFound;
    __shared__ RowState         state;
    __shared__ MeshBvhTraversal sharedTraversal;
    __shared__ CandidateSlot    sharedStage[cudaFamily::kEvaluateThreads];
    cudaFamilyPlanBody<false, false, true>(launch, sharedSlots, sharedOrder, &sharedClip, &sharedFound, state,
                                           &sharedTraversal, nullptr, sharedStage);
}

extern "C" __global__ void __launch_bounds__(cudaFamily::kEvaluateThreads, cudaFamily::kEvaluateBlocksPerMultiprocessor)
    ovspatialquery_cuda_family_plan_crossings_single(const cudaFamily::PlanLaunchRecord launch)
{
    __shared__ CandidateSlot    sharedSlots[cudaFamily::kEvaluateThreads];
    __shared__ unsigned int     sharedOrder[cudaFamily::kEvaluateThreads + 1u];
    __shared__ double           sharedClip;
    __shared__ unsigned int     sharedFound;
    __shared__ RowState         state;
    __shared__ MeshBvhTraversal sharedTraversal;
    __shared__ CandidateSlot    sharedStage[cudaFamily::kEvaluateThreads];
    cudaFamilyPlanBody<true, false, true>(launch, sharedSlots, sharedOrder, &sharedClip, &sharedFound, state,
                                          &sharedTraversal, nullptr, sharedStage);
}

extern "C" __global__ void __launch_bounds__(cudaFamily::kEvaluateThreads, cudaFamily::kEvaluateBlocksPerMultiprocessor)
    ovspatialquery_cuda_family_plan_box(const cudaFamily::PlanLaunchRecord launch)
{
    __shared__ CandidateSlot    sharedSlots[cudaFamily::kEvaluateThreads];
    __shared__ unsigned int     sharedOrder[cudaFamily::kEvaluateThreads + 1u];
    __shared__ double           sharedClip;
    __shared__ unsigned int     sharedFound;
    __shared__ RowState         state;
    __shared__ MeshBvhTraversal sharedTraversal;
    __shared__ BoxSweepQueue    sharedQueue;
    cudaFamilyPlanBody<false, true>(launch, sharedSlots, sharedOrder, &sharedClip, &sharedFound, state,
                                    &sharedTraversal, &sharedQueue);
}

// One block. Turns the evaluation pass's desired counts into dense hit offsets,
// counts and truncation flags: rows are granted capacity in query order until
// the slot capacity is spent.
extern "C" __global__ void ovspatialquery_cuda_family_scan(unsigned long long query_count,
                                                           unsigned long long result_capacity,
                                                           ovspatialquery::detail::cuda_family::RowPlan* row_plans,
                                                           unsigned long long* stored_hit_count,
                                                           unsigned long long* stored_hit_count_mirror,
                                                           unsigned long long* hit_offsets,
                                                           unsigned long long* hit_counts, unsigned char* truncated)
{
    __shared__ unsigned long long values[cudaFamily::kScanThreads];
    __shared__ unsigned long long carry;
    if (threadIdx.x == 0u)
    {
        carry = 0u;
    }
    __syncthreads();
    for (unsigned long long base = 0u; base < query_count; base += blockDim.x)
    {
        const unsigned long long  query   = base + threadIdx.x;
        const bool                inRange = query < query_count;
        const cudaFamily::RowPlan plan    = inRange ? row_plans[query] : cudaFamily::RowPlan{};
        const unsigned long long  desired = inRange && plan.valid != 0u ? plan.desired : 0u;
        values[threadIdx.x]               = desired;
        __syncthreads();
        for (unsigned int offset = 1u; offset < blockDim.x; offset <<= 1u)
        {
            const unsigned long long addend = threadIdx.x >= offset ? values[threadIdx.x - offset] : 0u;
            __syncthreads();
            values[threadIdx.x] += addend;
            __syncthreads();
        }
        const unsigned long long exclusive = values[threadIdx.x] - desired;
        if (inRange)
        {
            const unsigned long long offset = carry + exclusive < result_capacity ? carry + exclusive : result_capacity;
            const unsigned long long room   = result_capacity - offset;
            const unsigned long long granted = desired < room ? desired : room;
            row_plans[query].granted         = granted;
            hit_offsets[query]               = plan.valid != 0u ? offset : 0u;
            hit_counts[query]                = granted;
            truncated[query]                 = plan.semanticCount > granted ? 1u : 0u;
        }
        __syncthreads();
        if (threadIdx.x == 0u)
        {
            carry += values[blockDim.x - 1u];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u)
    {
        // The mirror is mapped host memory: the host reads H there once `ready` completes.
        const unsigned long long stored = carry < result_capacity ? carry : result_capacity;
        *stored_hit_count               = stored;
        *stored_hit_count_mirror        = stored;
    }
}

// One grid row per column copies that column's live rows from device storage
// into mapped host memory. Consecutive threads write consecutive 16-byte words,
// so every write that crosses the bus is part of one contiguous run; scattered
// field stores into host memory were several times slower than pageable copies.
// The bases are 256-byte aligned, so only a run's last partial word is copied
// by bytes.
extern "C" __global__ void ovspatialquery_cuda_family_publish_mapped(cudaFamily::MappedPublishRecord record)
{
    const cudaFamily::MappedColumnCopy column = record.columns[blockIdx.y];
    const unsigned long long           rows   = column.rows == cudaFamily::kMappedRowsQuery ? record.queryCount
                                                : column.rows == cudaFamily::kMappedRowsHit ? *record.storedHitCount
                                                                                            : 1ull;
    const unsigned long long           bytes  = rows * column.rowBytes;
    const unsigned long long           words  = bytes / 16u;
    const unsigned long long           first  = blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
    const unsigned long long           stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;
    const uint4*                       source = reinterpret_cast<const uint4*>(column.source);
    uint4*                             destination = reinterpret_cast<uint4*>(column.destination);
    for (unsigned long long word = first; word < words; word += stride)
    {
        destination[word] = source[word];
    }
    const unsigned char* sourceBytes      = reinterpret_cast<const unsigned char*>(source);
    unsigned char*       destinationBytes = reinterpret_cast<unsigned char*>(destination);
    for (unsigned long long byte = words * 16u + first; byte < bytes; byte += stride)
    {
        destinationBytes[byte] = sourceBytes[byte];
    }
}

// reserveStack launches this once after it raises the stack limit: the driver
// finishes applying a new limit at the next launch, and until then a
// legacy-stream synchronization waits for every stream on the device.
extern "C" __global__ void ovspatialquery_cuda_apply_stack_limit() {}

#include "ovspatialquery_cuda_device_boxes.cuh"
#include "ovspatialquery_cuda_pair_kernels.cuh"
