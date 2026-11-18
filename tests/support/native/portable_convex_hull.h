/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_PORTABLE_CONVEX_HULL_H
#define OVSPATIALQUERY_PORTABLE_CONVEX_HULL_H

#include "ovspatialquery_typed_geometry.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace ovspatialquery
{
namespace detail
{

/*
 * The test oracle's convex hull, which needs no PhysX: an exact double quickhull of the
 * validated points, refused as beyondVertexLimit when it has more than `maximumVertexCount`
 * vertices and as noEnclosingHull when it misses an authored point, as the product refuses
 * (ADR-0074, ADR-0076). Under a hull vertex limit it keeps its hull when that has at most
 * the limit's vertices and refuses a larger one, which the product's PhysX reduces (ADR-0078).
 * The product cooks hulls with PhysX instead (ADR-0030).
 */
ConvexHullPreparationOutcome preparePortableConvexHull(const std::vector<float>& authoredPoints,
                                                       uint32_t maximumVertexCount, uint64_t maximumResidentBytes,
                                                       std::shared_ptr<const PreparedConvexHull>& outHull,
                                                       const ConvexHullPreparationBudget*         budget  = nullptr,
                                                       const ConvexHullCookingOptions&            options = {},
                                                       double* outFarthestOutside                         = nullptr);

} // namespace detail
} // namespace ovspatialquery

#endif
