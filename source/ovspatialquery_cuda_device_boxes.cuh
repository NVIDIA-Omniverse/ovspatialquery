// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Device validation and bounds for an explicit population prepared on the
// device (ADR-0024), of every exact geometry family. Complete rows are checked
// against the binding of their slot; poses-only rows carry no binding columns
// and keep each slot's scale when the scale column is absent. Participation bits belong to the version:
// a row with a bits column writes them, a complete row without one writes all
// bits set, and a pose row without one keeps the bits copied from the base. An
// excluded row writes its bits too, so a later pose row restores them with the slot.
// Explicit rounding here reproduces the CPU BOX matrix expression; bounds and
// quaternion-length reductions retain their explicit fused multiply-adds.
#include "ovspatialquery_ovstage_transform.h"

namespace device_boxes
{
__device__ double real(const void* data, unsigned int bits, unsigned long long i)
{
    return bits == 32u ? static_cast<double>(static_cast<const float*>(data)[i]) : static_cast<const double*>(data)[i];
}
__device__ double mul(double a, double b) { return __dmul_rn(a, b); }
__device__ double add(double a, double b) { return __dadd_rn(a, b); }
__device__ double sub(double a, double b) { return __dsub_rn(a, b); }
// The input domain of ovspatialquery.h (ADR-0032): positions and world bounds
// within +/-2^60 caller units, uniform scales within [2^-60, 2^60].
constexpr double kDomain = 1152921504606846976.0;
__device__ bool  inDomain(double value) { return fabs(value) <= kDomain; }
__device__ bool  withinDomain(double low, double high) { return inDomain(low) && inDomain(high); }
__device__ void  exclude(unsigned long long id, int reason, unsigned int* count, unsigned long long* ids, int* reasons)
{
    const unsigned int index = atomicAdd(count, 1u);
    ids[index]               = id;
    reasons[index]           = reason;
}
// A row's participation bits, written as for an accepted row, so an excluded
// slot keeps them for a later pose row that makes it valid again.
__device__ void writeBits(unsigned long long slot, unsigned long long row, const unsigned long long* bits,
                          unsigned long long* version_bits, unsigned int poses_only)
{
    if (!poses_only || bits)
    {
        version_bits[slot * 2]     = bits ? bits[row * 2] : ~0ull;
        version_bits[slot * 2 + 1] = bits ? bits[row * 2 + 1] : ~0ull;
    }
}
// Inverted bounds hide an excluded slot from every query and pair reporter.
__device__ void tombstone(unsigned long long slot, unsigned long long row, double* minimums, double* maximums,
                          double* poses, const unsigned long long* bits, unsigned long long* version_bits,
                          unsigned int poses_only)
{
    writeBits(slot, row, bits, version_bits, poses_only);
    for (unsigned int axis = 0; axis < 3; ++axis)
    {
        minimums[slot * 3 + axis] = 1.0 / 0.0;
        maximums[slot * 3 + axis] = -1.0 / 0.0;
    }
    for (unsigned int component = 0; component < 8; ++component)
        poses[slot * 8 + component] = 0.0;
}
} // namespace device_boxes
// Retired and unassigned slots hold inverted bounds that no query overlaps:
// slots [begin, end) and the listed slots.
extern "C" __global__ void ovspatialquery_cuda_tombstone_boxes(unsigned long long begin, unsigned long long end,
                                                               const unsigned int* slots, unsigned long long slot_count,
                                                               double* minimums, double* maximums)
{
    const unsigned long long work  = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long range = end - begin;
    unsigned long long       slot;
    if (work < range)
        slot = begin + work;
    else if (work - range < slot_count)
        slot = slots[work - range];
    else
        return;
    for (unsigned int axis = 0; axis < 3; ++axis)
    {
        minimums[slot * 3 + axis] = 1.0 / 0.0;
        maximums[slot * 3 + axis] = -1.0 / 0.0;
    }
}
// Rows resolve their object through the sorted index of live objects
// (`sorted_ids` ascending, `sorted_slots` the slot of each); a retired or
// unknown ID is not in it and fails the row. Slot arrays are capacity-sized.
// Each family derives the world AABB the host derives for the same row
// (ADR-0023): the sphere and capsule analytically, the box from its rotated
// half extents, the convex hull from every transformed hull vertex, the mesh
// from the eight corners of its local bounds, with the host's operation order.
extern "C" __global__ void ovspatialquery_cuda_prepare_boxes(
    unsigned long long live_count, const unsigned long long* sorted_ids, const unsigned int* sorted_slots,
    unsigned long long rows, const unsigned long long* fixed_refs, const double* params, const int* fixed_types,
    const unsigned int* defs, const ovspatialquery::detail::cuda_family::DeviceShapeRecord* records,
    const unsigned long long* ids, const int* kinds, const unsigned long long* refs, const void* translations,
    unsigned int translation_bits, const void* rotations, unsigned int rotation_bits, const void* scales,
    unsigned int scale_bits, const unsigned long long* bits, double* poses, unsigned long long* version_bits,
    double* minimums, double* maximums, unsigned int* seen, unsigned int* error, unsigned int poses_only,
    unsigned int* changed, unsigned int ovstage, const double* current_box_extents, double* version_box_extents,
    unsigned int* exclusion_count, unsigned long long* excluded_ids, int* exclusion_reasons)
{
    using namespace device_boxes;
    const unsigned long long row = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows)
        return;
    const unsigned long long id  = ids[row];
    unsigned long long       low = 0, high = live_count;
    while (low < high)
    {
        const auto mid = low + (high - low) / 2;
        if (sorted_ids[mid] < id)
            low = mid + 1;
        else
            high = mid;
    }
    if (!id || low == live_count || sorted_ids[low] != id)
    {
        atomicOr(error, 1u);
        return;
    }
    const unsigned long long slot = sorted_slots[low];
    // The slot this row rewrites, for readers that patch from the version's
    // change list instead of comparing whole versions (ADR-0019 decision 3).
    changed[row]   = static_cast<unsigned int>(slot);
    const int type = fixed_types[slot];
    if (!poses_only && (kinds[row] != type || refs[row] != fixed_refs[slot]))
    {
        atomicOr(error, 1u);
        return;
    }
    if (atomicExch(seen + slot, 1u) != 0)
    {
        atomicOr(error, 1u);
        return;
    }
    const double scale = scales ? real(scales, scale_bits, row) : poses_only ? poses[slot * 8 + 7] : 1.0;
    if (ovstage && scale < 0.0)
    {
        tombstone(slot, row, minimums, maximums, poses, bits, version_bits, poses_only);
        return; // the gather kernel already recorded this row's content exclusion
    }
    double q[4], p[3], length = 0.0;
    for (unsigned int i = 0; i < 4; ++i)
    {
        q[i] = real(rotations, rotation_bits, row * 4 + i);
        if (!isfinite(q[i]))
        {
            if (ovstage)
            {
                exclude(id, 4, exclusion_count, excluded_ids, exclusion_reasons);
                tombstone(slot, row, minimums, maximums, poses, bits, version_bits, poses_only);
            }
            else
                atomicOr(error, 1u);
            return;
        }
        length = fma(q[i], q[i], length);
    }
    if (!isfinite(length) || length == 0.0 || fabs(length - 1.0) > 1.0e-5 || !(scale >= 1.0 / kDomain) ||
        !(scale <= kDomain))
    {
        if (ovstage)
        {
            exclude(id, 4, exclusion_count, excluded_ids, exclusion_reasons);
            tombstone(slot, row, minimums, maximums, poses, bits, version_bits, poses_only);
        }
        else
            atomicOr(error, 1u);
        return;
    }
    const double inv = __ddiv_rn(1.0, sqrt(length));
    for (unsigned int i = 0; i < 4; ++i)
        q[i] = mul(q[i], inv);
    const double x = q[0], y = q[1], z = q[2], w = q[3];
    // The host's quaternionRotationMatrix, m[world][local].
    const double m[3][3] = {{sub(1.0, mul(2.0, add(mul(y, y), mul(z, z)))), mul(2.0, sub(mul(x, y), mul(z, w))),
                             mul(2.0, add(mul(x, z), mul(y, w)))},
                            {mul(2.0, add(mul(x, y), mul(z, w))), sub(1.0, mul(2.0, add(mul(x, x), mul(z, z)))),
                             mul(2.0, sub(mul(y, z), mul(x, w)))},
                            {mul(2.0, sub(mul(x, z), mul(y, w))), mul(2.0, add(mul(y, z), mul(x, w))),
                             sub(1.0, mul(2.0, add(mul(x, x), mul(y, y))))}};
    for (unsigned int i = 0; i < 3; ++i)
    {
        p[i] = real(translations, translation_bits, row * 3 + i);
        if (!inDomain(p[i]))
        {
            if (ovstage)
            {
                exclude(id, 4, exclusion_count, excluded_ids, exclusion_reasons);
                tombstone(slot, row, minimums, maximums, poses, bits, version_bits, poses_only);
            }
            else
                atomicOr(error, 1u);
            return;
        }
    }
    const double* shape = params + slot * 3;
    double        lo[3], hi[3];
    bool          valid = true;
    if (type == OVSPATIALQUERY_GEOMETRY_SPHERE) // sphere: createSphereExactGeometry
    {
        const double radius = ovstage ? current_box_extents[row * 3] : mul(shape[0], scale);
        valid               = isfinite(radius) && radius > 0.0;
        for (unsigned int axis = 0; valid && axis < 3; ++axis)
        {
            lo[axis] = sub(p[axis], radius);
            hi[axis] = add(p[axis], radius);
            valid    = withinDomain(lo[axis], hi[axis]);
        }
    }
    else if (type == OVSPATIALQUERY_GEOMETRY_BOX) // box: createBoxExactGeometry's orthonormality and bounds
    {
        double extents[3];
        for (unsigned int i = 0; valid && i < 3; ++i)
        {
            extents[i] = ovstage ? current_box_extents[row * 3 + i] : mul(shape[i], scale);
            valid      = isfinite(extents[i]) && extents[i] > 0.0;
        }
        for (unsigned int i = 0; valid && i < 3; ++i)
            for (unsigned int j = i; valid && j < 3; ++j)
            {
                const double dot = add(add(mul(m[0][i], m[0][j]), mul(m[1][i], m[1][j])), mul(m[2][i], m[2][j]));
                valid            = isfinite(dot) && fabs(dot - (i == j ? 1.0 : 0.0)) <= 1.0e-12;
            }
        for (unsigned int axis = 0; valid && axis < 3; ++axis)
        {
            double radius = 0.0;
            for (unsigned int local = 0; local < 3; ++local)
                radius = fma(fabs(m[axis][local]), extents[local], radius);
            lo[axis] = sub(p[axis], radius);
            hi[axis] = add(p[axis], radius);
            valid    = isfinite(radius) && withinDomain(lo[axis], hi[axis]);
        }
    }
    else if (type == OVSPATIALQUERY_GEOMETRY_CAPSULE) // capsule: the segment on local Y, createCapsuleExactGeometry
    {
        const double radius     = ovstage ? current_box_extents[row * 3] : mul(shape[0], scale);
        const double halfHeight = ovstage ? current_box_extents[row * 3 + 1] : mul(shape[1], scale);
        double       segment    = 0.0;
        valid                   = isfinite(radius) && radius > 0.0;
        for (unsigned int axis = 0; valid && axis < 3; ++axis)
        {
            const double offset = ovstage ? mul(m[axis][1], halfHeight) : mul(mul(m[axis][1], shape[1]), scale);
            const double a = sub(p[axis], offset), b = add(p[axis], offset);
            const double delta = sub(b, a);
            segment            = fma(delta, delta, segment);
            lo[axis]           = sub(fmin(a, b), radius);
            hi[axis]           = add(fmax(a, b), radius);
            valid              = isfinite(a) && isfinite(b) && withinDomain(lo[axis], hi[axis]);
        }
        valid = valid && isfinite(segment);
    }
    else if (type == OVSPATIALQUERY_GEOMETRY_CONVEX ||
             type == OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH) // convex hull and triangle mesh through the pose matrix
    {
        const ovspatialquery::detail::cuda_family::DeviceShapeRecord record = records[defs[slot]];
        double                                                       t[3], mc[3][3];
        for (unsigned int r = 0; r < 3; ++r)
        {
            t[r] = p[r] == 0.0 ? 0.0 : p[r]; // normalizedScalar
            for (unsigned int c = 0; c < 3; ++c)
                mc[r][c] = mul(m[r][c], scale);
        }
        for (unsigned int axis = 0; axis < 3; ++axis)
        {
            lo[axis] = 1.0 / 0.0;
            hi[axis] = -1.0 / 0.0;
        }
        if (type == OVSPATIALQUERY_GEOMETRY_CONVEX)
        {
            // placedConvexBounds: every hull vertex, fma(m0, l0, fma(m1, l1, fma(m2, l2, t))).
            const double* vertices = reinterpret_cast<const double*>(record.vertices);
            for (unsigned int v = 0; valid && v < record.vertexCount; ++v)
            {
                const double l0 = vertices[v * 3], l1 = vertices[v * 3 + 1], l2 = vertices[v * 3 + 2];
                for (unsigned int r = 0; valid && r < 3; ++r)
                {
                    const double world = fma(mc[r][0], l0, fma(mc[r][1], l1, fma(mc[r][2], l2, t[r])));
                    valid              = isfinite(world);
                    lo[r]              = fmin(lo[r], world);
                    hi[r]              = fmax(hi[r], world);
                }
            }
            valid = valid && record.vertexCount != 0u;
        }
        else
        {
            // The mesh's eight local corners, fma(m2, l2, fma(m1, l1, fma(m0, l0, t))).
            for (unsigned int corner = 0; valid && corner < 8; ++corner)
            {
                double local[3];
                for (unsigned int axis = 0; axis < 3; ++axis)
                    local[axis] =
                        (corner & (1u << axis)) != 0u ? record.localBounds[3 + axis] : record.localBounds[axis];
                for (unsigned int r = 0; valid && r < 3; ++r)
                {
                    const double world =
                        fma(mc[r][2], local[2], fma(mc[r][1], local[1], fma(mc[r][0], local[0], t[r])));
                    valid = isfinite(world);
                    lo[r] = fmin(lo[r], world);
                    hi[r] = fmax(hi[r], world);
                }
            }
        }
    }
    else
    {
        valid = false;
    }
    for (unsigned int axis = 0;
         valid && (type == OVSPATIALQUERY_GEOMETRY_CONVEX || type == OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH) && axis < 3;
         ++axis)
        valid = withinDomain(lo[axis], hi[axis]);
    if (!valid)
    {
        if (ovstage)
        {
            exclude(id, 4, exclusion_count, excluded_ids, exclusion_reasons);
            tombstone(slot, row, minimums, maximums, poses, bits, version_bits, poses_only);
        }
        else
            atomicOr(error, 1u);
        return;
    }
    if (poses_only)
    {
        // `poses` and `version_bits` still hold the base's: a row that rewrites
        // them bit for bit has not moved its slot and is not listed (ADR-0052).
        bool same =
            !bits || (bits[row * 2] == version_bits[slot * 2] && bits[row * 2 + 1] == version_bits[slot * 2 + 1]);
        for (unsigned int i = 0; same && i < 3; ++i)
            same = __double_as_longlong(poses[slot * 8 + i]) == __double_as_longlong(p[i]);
        for (unsigned int i = 0; same && i < 4; ++i)
            same = __double_as_longlong(poses[slot * 8 + 3 + i]) == __double_as_longlong(q[i]);
        if (ovstage)
            for (unsigned int i = 0; same && i < 3; ++i)
                same = __double_as_longlong(version_box_extents[slot * 3 + i]) ==
                       __double_as_longlong(current_box_extents[row * 3 + i]);
        if (same && __double_as_longlong(poses[slot * 8 + 7]) == __double_as_longlong(scale))
            changed[row] = ~0u;
    }
    for (unsigned int axis = 0; axis < 3; ++axis)
    {
        minimums[slot * 3 + axis] = lo[axis];
        maximums[slot * 3 + axis] = hi[axis];
        poses[slot * 8 + axis]    = p[axis];
    }
    for (unsigned int i = 0; i < 4; ++i)
        poses[slot * 8 + 3 + i] = q[i];
    poses[slot * 8 + 7] = scale;
    if (ovstage)
        for (unsigned int i = 0; i < 3; ++i)
            version_box_extents[slot * 3 + i] = current_box_extents[row * 3 + i];
    writeBits(slot, row, bits, version_bits, poses_only);
}
// Builds the version's grid over its final bounds: every slot whose box ends
// short of two cells past its minimum corner's cell along each axis is linked
// into that cell's bucket, every other finite box is listed as oversize, and
// tombstoned slots are skipped.
// The header and `heads` are filled with ones before the launch: an empty
// bucket is kGridEmpty and the oversize count is kept complemented.
extern "C" __global__ void ovspatialquery_cuda_grid_boxes(unsigned long long slots, const double* minimums,
                                                          const double* maximums, unsigned int bucket_mask,
                                                          double inverse_cell,
                                                          ovspatialquery::detail::cuda_family::DeviceBoxGrid* grid,
                                                          unsigned int* heads, unsigned int* next, int* cell_keys,
                                                          unsigned int* oversize)
{
    const unsigned long long slot = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (slot == 0u)
    {
        grid->bucketMask  = bucket_mask;
        grid->inverseCell = inverse_cell;
        grid->capacity    = static_cast<unsigned int>(slots);
    }
    if (slot >= slots)
        return;
    const double* minimum = minimums + slot * 3u;
    const double* maximum = maximums + slot * 3u;
    if (!(minimum[0] <= maximum[0]))
        return; // tombstoned, or never written
    int  cell[3];
    bool regular = true;
    for (unsigned int axis = 0; axis < 3 && regular; ++axis)
    {
        regular = boxGridCell(minimum[axis], inverse_cell, cell[axis]) &&
                  maximum[axis] * inverse_cell < static_cast<double>(cell[axis]) + 2.0;
    }
    if (!regular)
    {
        oversize[~atomicSub(&grid->oversizeComplement, 1u)] = static_cast<unsigned int>(slot);
        return;
    }
    cell_keys[3u * slot]      = cell[0];
    cell_keys[3u * slot + 1u] = cell[1];
    cell_keys[3u * slot + 2u] = cell[2];
    next[slot] =
        atomicExch(&heads[boxGridHash(cell[0], cell[1], cell[2], bucket_mask)], static_cast<unsigned int>(slot));
}
// Gathers world matrices into version pose rows. Content exclusions carry a
// negative reason in scales and append one row to the pinned exclusion list.
extern "C" __global__ void ovspatialquery_cuda_gather_ovstage_poses(
    unsigned long long rows, const unsigned int* pairs, const double* matrices,
    const ovspatialquery::detail::cuda_family::OvstageSlotRecord* slots, unsigned long long* ids, double* translations,
    double* rotations, double* scales, unsigned int* exclusion_count, unsigned long long* excluded_ids,
    int* exclusion_reasons, double* box_extents)
{
    namespace family             = ovspatialquery::detail::cuda_family;
    namespace transform          = ovspatialquery::detail::ovstage_transform;
    const unsigned long long row = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows)
        return;
    double m[16];
    for (unsigned int i = 0; i < 16; ++i)
        m[i] = matrices[static_cast<unsigned long long>(pairs[2 * row]) * 16u + i];
    const family::OvstageSlotRecord& slot = slots[pairs[2 * row + 1]];
    ids[row]                              = slot.id;
    for (unsigned int i = 0; i < 3; ++i)
        box_extents[row * 3 + i] = 0.0;
    int  reason = 0;
    bool valid  = transform::normalizeAffine(m, true);
    for (unsigned int i = 0; i < 16; ++i)
        valid = valid && isfinite(m[i]) && device_boxes::inDomain(m[i]);
    double l[9];
    for (unsigned int axis = 0; axis < 3; ++axis)
        for (unsigned int column = 0; column < 3; ++column)
            l[axis * 3 + column] = m[column * 4 + axis];
    const double determinant = transform::determinant(l);
    valid                    = valid && isfinite(determinant) && determinant != 0.0;
    double scale             = cbrt(fabs(determinant));
    valid = valid && isfinite(scale) && scale >= 1.0 / device_boxes::kDomain && scale <= device_boxes::kDomain;
    double rotation[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    if (!valid)
        reason = 4; // MALFORMED_GEOMETRY
    else if (slot.rule == family::kOvstageRuleBox)
    {
        double length[3];
        for (unsigned int column = 0; column < 3; ++column)
        {
            length[column] =
                sqrt((l[column] * l[column] + l[3 + column] * l[3 + column]) + l[6 + column] * l[6 + column]);
            for (unsigned int axis = 0; axis < 3; ++axis)
                rotation[axis * 3 + column] = l[axis * 3 + column] / length[column];
            box_extents[row * 3 + column] = slot.size * 0.5 * length[column];
        }
        if (determinant < 0.0)
            for (unsigned int axis = 0; axis < 3; ++axis)
                rotation[axis * 3] = -rotation[axis * 3];
        if (!transform::rigid(rotation))
            reason = 5; // RESHAPE_PENDING
    }
    else if (slot.rule == family::kOvstageRuleUniform)
    {
        // Spheres and capsules are symmetric under a mirror; use a proper frame.
        const double sign = determinant > 0.0 ? 1.0 : -1.0;
        for (unsigned int i = 0; i < 9; ++i)
            rotation[i] = sign * l[i] / scale;
        if (!transform::rigid(rotation))
            reason = 3; // UNSUPPORTED_SHAPE
        else
        {
            box_extents[row * 3]    = slot.radius * transform::largestSingular(l, scale);
            const unsigned int axis = slot.frame == family::kOvstageFrameFromX   ? 0u
                                      : slot.frame == family::kOvstageFrameFromZ ? 2u
                                                                                 : 1u;
            const double axisLength = sqrt((l[axis] * l[axis] + l[3 + axis] * l[3 + axis]) + l[6 + axis] * l[6 + axis]);
            box_extents[row * 3 + 1] = slot.halfHeight * axisLength;
            if (slot.halfHeight > 0.0)
            {
                // Keep the capsule's transformed segment, including its tilt.
                // The device capsule's segment lies on local Y.
                double y[3];
                for (unsigned int i = 0; i < 3; ++i)
                    y[i] = l[i * 3 + axis] / axisLength;
                unsigned int reference = 0;
                for (unsigned int i = 1; i < 3; ++i)
                    if (fabs(y[i]) < fabs(y[reference]))
                        reference = i;
                double unit[3]       = {0.0, 0.0, 0.0};
                unit[reference]      = 1.0;
                double       x[3]    = {y[1] * unit[2] - y[2] * unit[1], y[2] * unit[0] - y[0] * unit[2],
                                        y[0] * unit[1] - y[1] * unit[0]};
                const double xLength = sqrt((x[0] * x[0] + x[1] * x[1]) + x[2] * x[2]);
                for (unsigned int i = 0; i < 3; ++i)
                {
                    x[i] /= xLength;
                    rotation[i * 3]     = x[i];
                    rotation[i * 3 + 1] = y[i];
                }
                rotation[2] = x[1] * y[2] - x[2] * y[1];
                rotation[5] = x[2] * y[0] - x[0] * y[2];
                rotation[8] = x[0] * y[1] - x[1] * y[0];
            }
        }
    }
    else
    {
        transform::candidate(l, slot.inverse, scale, rotation);
        if (!transform::rigid(rotation))
            reason = 5; // RESHAPE_PENDING
    }
    double quaternion[4] = {0.0, 0.0, 0.0, 1.0};
    if (reason == 0)
        transform::quaternion(rotation, quaternion);
    else
        device_boxes::exclude(slot.id, reason, exclusion_count, excluded_ids, exclusion_reasons);
    for (unsigned int i = 0; i < 4; ++i)
        rotations[row * 4 + i] = quaternion[i];
    for (unsigned int axis = 0; axis < 3; ++axis)
        translations[row * 3 + axis] = m[12 + axis];
    scales[row] = reason == 0 ? scale : -static_cast<double>(reason);
}
namespace device_boxes
{
__device__ double quotient(double a, double b) { return __ddiv_rn(a, b); }
// std::min and std::max, which keep the first argument unless the second compares beyond it.
__device__ double lesser(double a, double b) { return b < a ? b : a; }
__device__ double greater(double a, double b) { return a < b ? b : a; }
// invertMeshLinear of the catalogue: the inverse of the linear part, scaled by its largest entry.
__device__ bool invertMeshLinear(const double* w, double* output)
{
    double linearScale = 0.0;
    for (unsigned int index = 0; index < 11; ++index)
        if (index % 4 != 3)
            linearScale = greater(linearScale, fabs(w[index]));
    if (linearScale == 0.0)
        return false;
    const double a = quotient(w[0], linearScale), b = quotient(w[1], linearScale), c = quotient(w[2], linearScale);
    const double d = quotient(w[4], linearScale), e = quotient(w[5], linearScale), f = quotient(w[6], linearScale);
    const double g = quotient(w[8], linearScale), h = quotient(w[9], linearScale), i = quotient(w[10], linearScale);
    const double determinant = add(sub(mul(a, sub(mul(e, i), mul(f, h))), mul(b, sub(mul(d, i), mul(f, g)))),
                                   mul(c, sub(mul(d, h), mul(e, g))));
    // The host's mesh placement also refuses a mirrored pose.
    if (!isfinite(determinant) || !(determinant > 0.0))
        return false;
    const double inverse  = quotient(quotient(1.0, determinant), linearScale);
    const double terms[9] = {sub(mul(e, i), mul(f, h)), sub(mul(c, h), mul(b, i)), sub(mul(b, f), mul(c, e)),
                             sub(mul(f, g), mul(d, i)), sub(mul(a, i), mul(c, g)), sub(mul(c, d), mul(a, f)),
                             sub(mul(d, h), mul(e, g)), sub(mul(b, g), mul(a, h)), sub(mul(a, e), mul(b, d))};
    bool         finite   = true;
    for (unsigned int index = 0; index < 9; ++index)
    {
        output[index] = mul(terms[index], inverse);
        finite        = finite && isfinite(output[index]);
    }
    return finite;
}
} // namespace device_boxes
// Derives a readable explicit version's exact catalogue (BH-065): per slot what
// the version's host copy holds and the catalogue upload would stage from it.
// The host copy rebuilds each object from its stored pose through
// makeRigidTransform, so the rotation is normalized again here; each family then
// follows its host constructor and the catalogue's transforms in their operation
// order, with a fused multiply-add exactly where the host calls std::fma and every
// other operation rounded on its own. A slot the version tombstoned, or never
// wrote, keeps inverted bounds and is marked absent, as the host copy leaves it out. A pose the host
// copy would refuse (a mirrored mesh, a hull whose placed frame is not
// orthonormal) sets `failed`, so the launch reports the failed version.
extern "C" __global__ void ovspatialquery_cuda_exact_catalogue(
    const ovspatialquery::detail::cuda_family::ExactCatalogueRecord r)
{
    using namespace device_boxes;
    namespace family              = ovspatialquery::detail::cuda_family;
    const unsigned long long slot = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (slot == 0u && *r.versionFailed != 0u)
        atomicOr(r.failed, 1u);
    if (slot >= r.slotCount)
        return;
    if (r.materials)
        r.materials[slot] = 0u; // the host copy's objects carry no material
    double*                         lo     = r.minimums + slot * 3u;
    double*                         hi     = r.maximums + slot * 3u;
    const family::ExactObjectHeader header = r.headers[slot];
    if (!isfinite(r.versionMinimums[slot * 3u]) || header.kind == family::kGeometryAbsent)
    {
        for (unsigned int axis = 0; axis < 3; ++axis)
        {
            lo[axis] = 1.0 / 0.0;
            hi[axis] = -1.0 / 0.0;
        }
        r.objects[slot] = {family::kGeometryAbsent, 0u, 0u};
        return;
    }
    r.objects[slot]      = header;
    const double* pose   = r.poses + slot * 8u;
    const double  p[3]   = {pose[0], pose[1], pose[2]};
    const double  scale  = pose[7];
    double        length = 0.0;
    for (unsigned int i = 0; i < 4; ++i)
        length = fma(pose[3 + i], pose[3 + i], length);
    const double inv = quotient(1.0, __dsqrt_rn(length));
    const double x = mul(pose[3], inv), y = mul(pose[4], inv), z = mul(pose[5], inv), w = mul(pose[6], inv);
    // quaternionRotationMatrix, m[world][local].
    const double  m[3][3] = {{sub(1.0, mul(2.0, add(mul(y, y), mul(z, z)))), mul(2.0, sub(mul(x, y), mul(z, w))),
                              mul(2.0, add(mul(x, z), mul(y, w)))},
                             {mul(2.0, add(mul(x, y), mul(z, w))), sub(1.0, mul(2.0, add(mul(x, x), mul(z, z)))),
                              mul(2.0, sub(mul(y, z), mul(x, w)))},
                             {mul(2.0, sub(mul(x, z), mul(y, w))), mul(2.0, add(mul(y, z), mul(x, w))),
                              sub(1.0, mul(2.0, add(mul(x, x), mul(y, y))))}};
    const double* shape   = r.params + slot * 3u;
    bool          valid   = true;
    if (header.kind == family::kGeometrySphere) // createSphereExactGeometry, SphereExactGeometry
    {
        family::SphereGeometry& sphere = r.spheres[header.familyIndex];
        sphere.radius                  = mul(shape[0], scale);
        for (unsigned int axis = 0; axis < 3; ++axis)
        {
            sphere.center[axis] = p[axis];
            lo[axis]            = sub(p[axis], sphere.radius);
            hi[axis]            = add(p[axis], sphere.radius);
        }
    }
    else if (header.kind == family::kGeometryBox) // createBoxExactGeometry, BoxExactGeometry
    {
        family::BoxGeometry& box = r.boxes[header.familyIndex];
        for (unsigned int local = 0; local < 3; ++local)
        {
            box.halfExtents[local] = mul(shape[local], scale);
            for (unsigned int axis = 0; axis < 3; ++axis)
                box.axes[local * 3 + axis] = m[axis][local];
        }
        for (unsigned int axis = 0; axis < 3; ++axis)
        {
            box.center[axis] = p[axis];
            double radius    = 0.0;
            for (unsigned int local = 0; local < 3; ++local)
                radius = fma(fabs(box.axes[local * 3 + axis]), box.halfExtents[local], radius);
            lo[axis] = sub(p[axis], radius);
            hi[axis] = add(p[axis], radius);
        }
    }
    else if (header.kind == family::kGeometryCapsule) // createCapsuleExactGeometry, CapsuleExactGeometry
    {
        family::CapsuleGeometry& capsule = r.capsules[header.familyIndex];
        capsule.radius                   = mul(shape[0], scale);
        for (unsigned int axis = 0; axis < 3; ++axis)
        {
            const double offset     = mul(mul(m[axis][1], shape[1]), scale);
            capsule.endpointA[axis] = sub(p[axis], offset);
            capsule.endpointB[axis] = add(p[axis], offset);
            lo[axis]                = sub(lesser(capsule.endpointA[axis], capsule.endpointB[axis]), capsule.radius);
            hi[axis]                = add(greater(capsule.endpointA[axis], capsule.endpointB[axis]), capsule.radius);
        }
    }
    else
    {
        // makeRigidTransform's worldFromLocal, three rows of four.
        double t[12];
        for (unsigned int row = 0; row < 3; ++row)
        {
            for (unsigned int column = 0; column < 3; ++column)
                t[row * 4 + column] = mul(m[row][column], scale);
            t[row * 4 + 3] = p[row] == 0.0 ? 0.0 : p[row]; // normalizedScalar
        }
        for (unsigned int axis = 0; axis < 3; ++axis)
        {
            lo[axis] = 1.0 / 0.0;
            hi[axis] = -1.0 / 0.0;
        }
        const family::DeviceShapeRecord& record = r.records[r.definitions[slot]];
        if (header.kind == family::kGeometryConvex) // the placed hull: placedConvexBounds and the catalogue
        {
            // The prepared surface's uniform scale, from the first column.
            const double     uniform   = __dsqrt_rn(add(add(mul(t[0], t[0]), mul(t[4], t[4])), mul(t[8], t[8])));
            constexpr double tolerance = 32.0 * 2.220446049250313e-16;
            for (unsigned int a = 0; valid && a < 3; ++a)
                for (unsigned int b = a; valid && b < 3; ++b)
                {
                    double product = 0.0;
                    for (unsigned int row = 0; row < 3; ++row)
                        product = fma(quotient(t[row * 4 + a], uniform), quotient(t[row * 4 + b], uniform), product);
                    valid = !(fabs(sub(product, a == b ? 1.0 : 0.0)) > tolerance);
                }
            const double* hull = reinterpret_cast<const double*>(record.vertices);
            for (unsigned int v = 0; valid && v < record.vertexCount; ++v)
            {
                for (unsigned int row = 0; row < 3; ++row)
                {
                    const double world =
                        fma(t[row * 4], hull[v * 3],
                            fma(t[row * 4 + 1], hull[v * 3 + 1], fma(t[row * 4 + 2], hull[v * 3 + 2], t[row * 4 + 3])));
                    valid   = valid && isfinite(world);
                    lo[row] = lesser(lo[row], world);
                    hi[row] = greater(hi[row], world);
                }
            }
            family::ConvexGeometry          convex = r.convexTemplates[header.familyIndex];
            const family::ExactConvexSource source = r.convexSources[header.familyIndex];
            for (unsigned int axis = 0; axis < 3; ++axis)
                convex.translation[axis] = t[axis * 4 + 3];
            for (unsigned long long v = 0; valid && v < convex.vertexCount; ++v)
            {
                const double* local  = r.localVertices + (source.localVertexOffset + v) * 3u;
                double*       placed = r.convexVertices + (convex.vertexOffset + v) * 3u;
                for (unsigned int axis = 0; axis < 3; ++axis)
                {
                    const double* row = t + axis * 4;
                    placed[axis]      = fma(row[0], local[0], fma(row[1], local[1], mul(row[2], local[2])));
                }
            }
            for (unsigned long long plane = 0; valid && plane < convex.planeCount; ++plane)
            {
                const double* local  = r.localPlanes + (source.localPlaneOffset + plane) * 4u;
                double*       placed = r.convexPlanes + (convex.planeOffset + plane) * 4u;
                for (unsigned int axis = 0; axis < 3; ++axis)
                {
                    const double* row = t + axis * 4;
                    placed[axis] =
                        quotient(fma(row[0], local[0], fma(row[1], local[1], mul(row[2], local[2]))), uniform);
                }
                placed[3] = mul(local[3], uniform);
            }
            r.convexes[header.familyIndex] = convex;
        }
        else // a triangle mesh: prepareObject's world bounds, the catalogue's matrices
        {
            for (unsigned int corner = 0; corner < 8; ++corner)
            {
                double local[3];
                for (unsigned int axis = 0; axis < 3; ++axis)
                    local[axis] =
                        (corner & (1u << axis)) != 0u ? record.localBounds[3 + axis] : record.localBounds[axis];
                for (unsigned int row = 0; row < 3; ++row)
                {
                    const double world = fma(t[row * 4 + 2], local[2],
                                             fma(t[row * 4 + 1], local[1], fma(t[row * 4], local[0], t[row * 4 + 3])));
                    valid              = valid && isfinite(world);
                    lo[row]            = lesser(lo[row], world);
                    hi[row]            = greater(hi[row], world);
                }
            }
            family::MeshGeometry mesh = r.meshTemplates[header.familyIndex];
            for (unsigned int index = 0; index < 12; ++index)
                mesh.worldFromLocal[index] = t[index];
            valid                        = invertMeshLinear(t, mesh.localFromWorldLinear) && valid;
            r.meshes[header.familyIndex] = mesh;
        }
    }
    if (!valid)
        atomicOr(r.failed, 1u);
}
