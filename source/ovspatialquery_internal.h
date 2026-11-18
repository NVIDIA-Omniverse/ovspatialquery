/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_INTERNAL_H
#define OVSPATIALQUERY_INTERNAL_H

#include "ovspatialquery_private_api.h"
#include <ovspatialquery_geometry.h>
#include <ovspatialquery_geometry_identity.h>
#include <ovspatialquery_typed_geometry.h>

#include <memory>
#include <stdint.h>
#include <vector>

#ifdef __cplusplus
namespace ovspatialquery
{
namespace detail
{

constexpr uint64_t kViewHandleKind       = UINT64_C(1) << 62u;
constexpr uint64_t kPopulationHandleKind = UINT64_C(2) << 62u;
constexpr uint64_t kHandlePayloadLimit   = UINT64_C(1) << 62u;

inline ovspatialquery_object_id_t objectIdFromValue(uint64_t value) { return ovspatialquery_object_id_t{value}; }

inline bool objectIdEqual(ovspatialquery_object_id_t first, ovspatialquery_object_id_t second)
{
    return first.value == second.value;
}

inline bool objectIdLess(ovspatialquery_object_id_t first, ovspatialquery_object_id_t second)
{
    return first.value < second.value;
}

struct OvstageShapeTransform
{
    std::array<double, 9> k{1, 0, 0, 0, 1, 0, 0, 0, 1};
    std::array<double, 9> inverse{1, 0, 0, 0, 1, 0, 0, 0, 1};
    std::array<double, 9> metric{1, 0, 0, 0, 1, 0, 0, 0, 1};
    bool                  mirrored = false;
};
struct PopulationObject
{
    ovspatialquery_object_id_t                    id;
    ovspatialquery_aabb_t                         bounds;
    std::shared_ptr<const TriangleObjectGeometry> triangleGeometry;
    std::shared_ptr<const ExactGeometry>          exactGeometry;
    // Explicit definitions retain their instance-owned immutable registration
    // independently of the per-instance exact query kernel.
    std::shared_ptr<const void> definitionOwner;
    GeometryReference           geometryReference{};
    ParticipationBits           participationBits = kAllParticipationBits;
    // The explicit adapter's positive uniform scale, so a poses-only update can keep it.
    double                                       uniformScale = 1.0;
    uint64_t                                     geometryKey  = 0u;
    std::shared_ptr<const OvstageShapeTransform> ovstageShape{};
    double                                       ovstageCubeSize = 0.0;
    // The material the object's hits report (ADR-0066), zero for none, and the
    // definition's per-source-face materials when a triangle mesh assigns any:
    // a nonzero entry replaces `materialId` for an exact contact on that face.
    uint64_t                                     materialId = 0u;
    std::shared_ptr<const std::vector<uint64_t>> faceMaterialIds{};
};

// The material a hit on `object` reports: its source face's when the hit has
// one and the definition assigns that face a nonzero material, else the object's.
inline uint64_t hitMaterialId(const PopulationObject& object, bool hasSourceFace, uint64_t sourceFace) noexcept
{
    if (hasSourceFace && object.faceMaterialIds != nullptr && sourceFace < object.faceMaterialIds->size())
    {
        const uint64_t faceMaterial = (*object.faceMaterialIds)[static_cast<size_t>(sourceFace)];
        if (faceMaterial != 0u)
        {
            return faceMaterial;
        }
    }
    return object.materialId;
}

uint64_t exactGeometryResidentByteCount(const std::vector<PopulationObject>& objects) noexcept;
// Adds one object's exact-geometry allocations; false once the total saturates.
bool accumulateExactGeometryResidentAllocations(ExactGeometryResidentAccumulator& accumulator,
                                                const PopulationObject&           object) noexcept;

} // namespace detail
} // namespace ovspatialquery
#endif

#endif
