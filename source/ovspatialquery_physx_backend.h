/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_PHYSX_BACKEND_H
#define OVSPATIALQUERY_PHYSX_BACKEND_H

#include "ovspatialquery_private_api.h"
#include <array>
#include <memory>
#include <vector>

namespace physx
{
class PxInsertionCallback;
class PxGeometry;
class PxPhysics;
} // namespace physx

struct ovspatialquery_backend_runtime_t;

namespace ovspatialquery
{
namespace detail
{
struct PhysXInfrastructure;
ovspatialquery_api_status_t acquirePhysXInfrastructure(std::shared_ptr<PhysXInfrastructure>& outInfrastructure);
physx::PxInsertionCallback* physxInsertionCallback(const std::shared_ptr<PhysXInfrastructure>& infrastructure);
physx::PxPhysics*           physxPhysics(const std::shared_ptr<PhysXInfrastructure>& infrastructure);
class InstanceHostLedger;
struct TriangleGeometryDescriptionView;
struct PreciseTriangleHit;
struct TriangleObjectGeometry;

/* Where a cooked PhysX mesh sits in its geometry's local frame: cooked = (local - origin) *
 * scale. The scale is a power of two and the origin is chosen so that the map is exact for
 * the cooked points. */
struct CookedFrame
{
    double                scale = 1.0;
    std::array<double, 3> origin{};
};

// Shared local sphere/box conversion for triangle and convex geometry queries.
ovspatialquery_api_status_t overlapPhysXGeometry(const physx::PxGeometry&              geometry,
                                                 const TriangleObjectGeometry&         placement,
                                                 const ovspatialquery_overlap_shape_t& query, bool& hasOverlap,
                                                 const CookedFrame& cooked = {});

// One immutable cooked mesh; no backend type crosses the installed interface.
class PhysXTriangleMesh
{
  public:
    ~PhysXTriangleMesh();
    // Builds charge nothing: the admission that created the mesh reserved its structure
    // and the scratch of building it (ADR-0075).
    static ovspatialquery_api_status_t create(const TriangleGeometryDescriptionView& source,
                                              std::unique_ptr<PhysXTriangleMesh>&    outMesh);
    // Cooks the source once into the bytes of a non-quantized BVH34 mesh whose vertices
    // may later be replaced and whose tree refitted (the pinned SDK's deformable-mesh path).
    static ovspatialquery_api_status_t cookTemplate(const TriangleGeometryDescriptionView& source,
                                                    std::vector<uint8_t>&                  outCooked);
    // A fresh mesh loaded from `cooked`, its vertices replaced by the source's and its
    // tree refitted; the source's triangles must be the template's.
    static ovspatialquery_api_status_t createFromTemplate(const std::vector<uint8_t>&            cooked,
                                                          const TriangleGeometryDescriptionView& source,
                                                          std::unique_ptr<PhysXTriangleMesh>&    outMesh);
    // `bothSides` asks PhysX for eMESH_BOTH_SIDES: back faces count, and a single-sided
    // mesh keeps its front normal, while a `doubleSided` one reports the normal facing the ray.
    // A hit within PhysX's widened float margin of a triangle's edge is decided in double, so
    // a ray on an edge two triangles share crosses one of them; `ledger` is charged for the
    // buffer of every crossing when the nearest candidate is not one.
    ovspatialquery_api_status_t raycast(const double origin[3], const double direction[3], double maximumDistance,
                                        bool doubleSided, bool bothSides, bool& hasHit, PreciseTriangleHit& hit,
                                        InstanceHostLedger* ledger) const;
    // Every triangle the local ray crosses within maximumDistance, as PhysX's
    // eMESH_MULTIPLE reports them: unordered, and a ray through a shared edge or
    // vertex may report each triangle sharing it. `emit` receives each hit in
    // the same local form as raycast's and returns false to stop. A crossing
    // that float origin rounding moved out of the interval arrives with
    // `recheck` set and only `generatedTriangle` valid, for the caller to decide
    // exactly, as raycast's NOT_SUPPORTED sends the nearest path to its contact
    // evaluation. A full buffer is retried four times larger, charged to `ledger`.
    using TriangleHitSink = bool (*)(void* context, const PreciseTriangleHit& hit, bool recheck);
    ovspatialquery_api_status_t raycastAll(const double origin[3], const double direction[3], double maximumDistance,
                                           bool doubleSided, bool bothSides, InstanceHostLedger* ledger,
                                           TriangleHitSink emit, void* context) const;

    ovspatialquery_api_status_t overlap(const TriangleObjectGeometry&         placement,
                                        const ovspatialquery_overlap_shape_t& query, bool& hasOverlap) const;

    uint32_t overlapTriangles(const double minimum[3], const double maximum[3], uint32_t* results, uint32_t capacity,
                              bool& overflow) const;
    // Triangles within radius of the local segment [start, end], gathered
    // conservatively (float rounding inflates the capsule outward). Same
    // overflow contract as overlapTriangles.
    uint32_t overlapTrianglesCapsule(const double start[3], const double end[3], double radius, uint32_t* results,
                                     uint32_t capacity, bool& overflow) const;

    // Triangles overlapping the local box with orthonormal right-handed axis
    // rows, gathered conservatively (float rounding inflates the box). Same
    // overflow contract as overlapTriangles.
    uint32_t overlapTrianglesBox(const double center[3], const double axes[3][3], const double half[3],
                                 uint32_t* results, uint32_t capacity, bool& overflow) const;

  private:
    struct Impl;
    explicit PhysXTriangleMesh(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> mImpl;
};
} // namespace detail
} // namespace ovspatialquery

#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
void ovspatialquery_physx_test_block_rays(ovspatialquery_backend_runtime_t* runtime, uint32_t expected_rays);

uint32_t ovspatialquery_physx_test_wait_for_blocked_rays(ovspatialquery_backend_runtime_t* runtime);

void ovspatialquery_physx_test_release_rays(ovspatialquery_backend_runtime_t* runtime);
#endif

#endif
