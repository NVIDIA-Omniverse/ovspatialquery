/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_BOX_SWEEP_GEOMETRY_H
#define OVSPATIALQUERY_BOX_SWEEP_GEOMETRY_H

#include "ovspatialquery_typed_geometry.h"

namespace ovspatialquery
{
namespace detail
{
ovspatialquery_api_status_t sweepBoxTriangleObject(const TriangleObjectGeometry& target, const BoxGeometrySource& query,
                                                   const float unitDirection[3], double minimumDistance,
                                                   double maximumDistance, bool& hasHit, ExactGeometryHit& outHit,
                                                   InstanceHostLedger* ledger, bool doubleSided);

// The caller has already added contact tolerance to the query half extents.
// Fixed-orientation continuous tests preserve the published world geometry.
// A triangle mesh's sidedness is the description's `doubleSided`, which a
// both-sides query sets (ADR-0065).
ovspatialquery_api_status_t sweepBoxGeometry(const ExactGeometryDescription& target, const BoxGeometrySource& query,
                                             const float unitDirection[3], double minimumDistance,
                                             double maximumDistance, bool& hasHit, ExactGeometryHit& outHit,
                                             InstanceHostLedger* ledger = nullptr) noexcept;

// sweepBoxGeometry against a box target in two steps. The interval alone decides
// whether and where the sweep meets the box; a closest sweep over many boxes
// keeps the interval of its running best and completes the contact once, for the
// winner. Together they return exactly what sweepBoxGeometry returns.
struct BoxSweepInterval
{
    double enter = 0.0, exit = 0.0;
    double normal[3] = {};
};
bool sweepBoxBoxInterval(const BoxExactGeometryDescription& target, const BoxGeometrySource& query,
                         const float unitDirection[3], double minimumDistance, double maximumDistance,
                         BoxSweepInterval& out) noexcept;
void sweepBoxBoxContact(const BoxExactGeometryDescription& target, const BoxGeometrySource& query,
                        const float unitDirection[3], const BoxSweepInterval& interval,
                        ExactGeometryHit& outHit) noexcept;
} // namespace detail
} // namespace ovspatialquery
#endif
