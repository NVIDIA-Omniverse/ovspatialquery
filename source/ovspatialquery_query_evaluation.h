/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_QUERY_EVALUATION_H
#define OVSPATIALQUERY_QUERY_EVALUATION_H

#include <ovspatialquery_population_table.h>

#include <array>
#include <cstdint>
#include <vector>

namespace ovspatialquery
{
namespace detail
{

enum class HitTruth : uint8_t
{
    boundsCandidate,
    exactSurface,
};

struct SemanticHitIdentity
{
    ovspatialquery_object_id_t objectId{};
    GeometryReference          geometry{};
    uint64_t                   materialId = 0u;
};

enum class GeometryFeatureKind : uint8_t
{
    none,
    triangle,
};

struct TriangleSourceFeature
{
    // The semantic result is uint64. Current producer evidence is limited to
    // validated uint32 source ordinals, which are represented here by exact
    // zero-extension; this private slice makes no full-domain claim.
    uint64_t                 sourceFace = 0u;
    std::array<uint32_t, 3u> sourceCorners{};
    std::array<float, 3u>    barycentrics{};
};

struct GeometryFeature
{
    GeometryFeatureKind   kind = GeometryFeatureKind::none;
    TriangleSourceFeature triangle{};
};

struct RaycastBoundsCandidate
{
    SemanticHitIdentity    identity{};
    std::array<double, 3u> boundsPosition{};
    std::array<float, 3u>  boundsNormal{};
    double                 distance = 0.0;
};

struct RaycastExactHit
{
    SemanticHitIdentity    identity{};
    std::array<double, 3u> position{};
    std::array<float, 3u>  normal{};
    double                 distance = 0.0;
    GeometryFeature        feature{};
};

struct RaycastSemanticHit
{
    HitTruth               truth = HitTruth::boundsCandidate;
    RaycastBoundsCandidate boundsCandidate{};
    RaycastExactHit        exact{};
};

struct SphereSweepBoundsCandidate
{
    SemanticHitIdentity    identity{};
    std::array<double, 3u> queryPosition{};
    std::array<double, 3u> boundsPosition{};
    std::array<float, 3u>  boundsNormal{};
    double                 distance = 0.0;
};

struct SphereSweepExactHit
{
    SemanticHitIdentity    identity{};
    std::array<double, 3u> queryPosition{};
    std::array<double, 3u> objectPosition{};
    std::array<float, 3u>  normal{};
    double                 distance = 0.0;
    GeometryFeature        feature{};
};

struct SphereSweepSemanticHit
{
    HitTruth                   truth = HitTruth::boundsCandidate;
    SphereSweepBoundsCandidate boundsCandidate{};
    SphereSweepExactHit        exact{};
};

struct OverlapSemanticHit
{
    HitTruth            truth = HitTruth::boundsCandidate;
    SemanticHitIdentity identity{};
};

struct RaycastQuery
{
    std::array<double, 3u> origin{};
    std::array<float, 3u>  unitDirection{};
    double                 minimumDistance = 0.0;
    double                 maximumDistance = 0.0;
    // Every mesh triangle counts from both sides for this query (ADR-0065).
    bool bothSides = false;
};

struct SphereSweepQuery
{
    std::array<double, 3u> center{};
    double                 radius = 0.0;
    std::array<float, 3u>  unitDirection{};
    double                 minimumDistance = 0.0;
    double                 maximumDistance = 0.0;
    // Box and sphere sweeps share contact storage and the plan lifecycle.
    bool                   box = false;
    std::array<double, 3u> halfExtents{};
    std::array<float, 4u>  rotationXyzw{0.0f, 0.0f, 0.0f, 1.0f};
    // Every mesh triangle counts from both sides for this query (ADR-0065).
    bool bothSides = false;
};

struct OverlapQuery
{
    ovspatialquery_overlap_shape_t geometry{};
    // A BOX row's rotation in double precision, used instead of the shape's
    // float32 quaternion when set (ADR-0026); the synchronous API leaves it unset.
    bool                   hasBoxRotation = false;
    std::array<double, 4u> boxRotationXyzw{0.0, 0.0, 0.0, 1.0};
};

enum class QueryPrecision : uint8_t
{
    boundsCandidate,
    exact,
};

enum class HitMode : uint8_t
{
    closest,
    any,
    multiple,
};

class InstanceHostLedger;

enum class SemanticEvaluationOutcome : uint8_t
{
    success,
    invalidArgument,
    notSupported,
    outOfMemory,
    hostBudget,
    internalFailure,
};

// The callers validate precision, mode and contact tolerance and resolve the
// generation before evaluating. With allMeshHits a MULTIPLE raycast reports
// every triangle a mesh's surface crosses, nearest first by distance, object id
// and source face, and keeps at most `retainedHits` of them.
SemanticEvaluationOutcome evaluateSemanticRaycast(const GenerationLease& generation, const RaycastQuery& query,
                                                  QueryPrecision precision, HitMode mode,
                                                  const ParticipationBits&         queryBits,
                                                  std::vector<RaycastSemanticHit>& outHits, double contactTolerance,
                                                  InstanceHostLedger* hostLedger, bool allMeshHits = false,
                                                  uint64_t retainedHits = UINT64_MAX) noexcept;

// A MULTIPLE sweep reports one hit per object reached, its first contact,
// nearest first by distance, then object id (ADR-0067), keeping the nearest
// retainedHits; ANY reports the first object the traversal finds.
SemanticEvaluationOutcome evaluateSemanticSphereSweep(const GenerationLease& generation, const SphereSweepQuery& query,
                                                      QueryPrecision precision, HitMode mode,
                                                      const ParticipationBits&             queryBits,
                                                      std::vector<SphereSweepSemanticHit>& outHits,
                                                      double contactTolerance, InstanceHostLedger* hostLedger,
                                                      uint64_t retainedHits = UINT64_MAX) noexcept;

SemanticEvaluationOutcome evaluateSemanticOverlap(const GenerationLease& generation, const OverlapQuery& query,
                                                  QueryPrecision precision, HitMode mode,
                                                  const ParticipationBits&         queryBits,
                                                  std::vector<OverlapSemanticHit>& outHits, double contactTolerance,
                                                  InstanceHostLedger* hostLedger) noexcept;

// A row that reports every mesh crossing charges its crossings as it finds
// them, so this per-object peak covers both modes.
bool estimateSemanticRaycastPeakBytes(uint64_t objectCount, uint64_t& outBytes) noexcept;
bool estimateSemanticSphereSweepPeakBytes(uint64_t objectCount, HitMode mode, uint64_t& outBytes) noexcept;
bool estimateSemanticOverlapPeakBytes(uint64_t objectCount, uint64_t& outBytes) noexcept;

} // namespace detail
} // namespace ovspatialquery

#endif
