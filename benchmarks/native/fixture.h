/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Deterministic scene and query fixtures shared by both adapters, result records and parity comparison.
#ifndef BENCH_FIXTURE_H
#define BENCH_FIXTURE_H
#include "common.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace bench
{
// The query family a fixture exercises; the CLI workload name selects both
// the fixture and one of these families.
enum class QueryKind : uint8_t
{
    rays,
    sweep,
    boxSweep,
    occupancy,
    zones,
    sphereOverlap,
    sphereAny,
};
const char* kindName(QueryKind kind);
// Overlaps that ask only whether anything is touched, and overlaps by a sphere.
inline bool anyOverlap(QueryKind kind) { return kind == QueryKind::occupancy || kind == QueryKind::sphereAny; }
inline bool sphereShape(QueryKind kind) { return kind == QueryKind::sphereOverlap || kind == QueryKind::sphereAny; }

struct Object
{
    uint64_t id, owner;
    bool     sphere;
    Vec      p;
    Quat     q;
    uint64_t bits    = 1;
    uint64_t mesh    = 0; // One-based definition index; zero selects the primitive.
    uint64_t convex  = 0; // One-based convex definition index.
    bool     capsule = false;
    int32_t  geometryType() const
    {
        return capsule  ? OVSPATIALQUERY_GEOMETRY_CAPSULE
               : convex ? OVSPATIALQUERY_GEOMETRY_CONVEX
               : mesh   ? OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH
                        : (sphere ? OVSPATIALQUERY_GEOMETRY_SPHERE : OVSPATIALQUERY_GEOMETRY_BOX);
    }
    uint64_t geometryValue() const { return convex ? convex : mesh ? mesh : 1; }
};
struct Query
{
    Vec                  p{0, 0, 0};
    Quat                 q = identity;
    std::array<float, 3> direction{0, 0, -1};
    double               nearDistance = 0.125, farDistance = 8;
    uint64_t             bits   = 0;
    double               radius = 0.5;
};
struct Mesh
{
    std::vector<std::array<float, 3>>    vertices;
    std::vector<std::array<uint32_t, 3>> triangles;
    bool                                 doubleSided = false;
};
struct Fixture
{
    std::vector<Mesh>                              meshes;
    std::vector<std::vector<std::array<float, 3>>> convexes;
    std::vector<Object>                            objects;
    std::vector<Query>                             queries;
    bool                                           pointCloud = false;
    Vec                                            half{1.25, 1.25, 0.625};
    QueryKind                                      kind            = QueryKind::rays;
    bool                                           even            = false;
    size_t                                         zoneHitCapacity = 0;
    double                                         radius          = 0.5;
    // Sphere overlaps and sweeps take each query's own radius (a per-row column in batches).
    bool rowRadii = false;
    // Pose updates publish POSES rows (identity and pose only) instead of complete rows.
    bool posesOnly = false;
    // Installed rays report bounds candidates; the PhysX baseline then needs axis-aligned boxes,
    // whose float AABB is the box itself.
    bool boundsCandidates = false;
    // Named limitation: direct PhysX scene sweeps against dense meshes. On the installed face their
    // distance agrees with the independent oracle to 3.2e-5 of max(1 m, distance), but they often
    // report another face of the object that the sphere meets later: up to about 1e-3 on the rack
    // meshes and 1.8e-3 on 224-cell terrain tiles, where PhysX and installed distances still agree
    // to 3e-6. A baseline face met later is accepted, and its distance is compared with the installed
    // contact instead; a fixture may declare this relative tolerance for that comparison (a fraction
    // of the reported distance, at least that fraction of a metre); zero keeps `near`. Installed CPU
    // and CUDA sphere sweeps meet the oracle to rounding whatever this is. A later object or face
    // beyond it, when the oracle confirms the other side's earlier contact, is counted as a miss
    // (`CompareStats::missedByFirst`).
    double sweepRelativeTolerance = 0;
    // Named limitation: PhysX float mesh contacts (which the installed CPU path returns verbatim)
    // sit up to 0.44 mm off the ray at 4.7 m on the dense rack meshes, while the CUDA kernel places
    // its double-precision contact on the ray; a fixture may declare an absolute contact tolerance
    // in metres for position and normal comparisons and the analytic on-ray check. Zero keeps `near`.
    double contactTolerance = 0;
    // Named limitation: direct PhysX scene queries return float contact positions and normals that
    // sit up to millimetres off the ray on dense consumer meshes, so a fixture may compare the raw
    // baseline on identity and distance only; the analytic on-ray check and the installed CPU/CUDA
    // comparison still cover every contact field.
    bool approximateBaselineContacts = false;
    // Named limitation: direct PhysX box sweeps against triangle meshes find the installed object
    // but not its contact. Checked against an independent first-contact solve, their points lie
    // up to 0.35 m outside the box on the reported face, or on the box up to 79 mm off that face;
    // the face can be one the box touches only from behind or a sliver it misses, and on the rack
    // meshes the distance is up to 2e-4 of max(1 m, distance) early, never later. PhysX 5.11's
    // ePRECISE_SWEEP returns the same hits. A box baseline is therefore compared on identity and
    // on distance, up to this fraction of max(1 m, distance) early (zero keeps `near`), and its
    // reported face must not meet the box before the installed first contact.
    double boxSweepBaselineEarlyTolerance = 0;
};
struct Hit
{
    uint64_t             id       = 0;
    double               distance = 0;
    Vec                  position{};
    std::array<float, 3> normal{};
    int32_t              geometryType  = 0;
    uint64_t             geometryValue = 0;
    uint64_t             sourceFace    = UINT64_MAX;
    bool                 positionValid = true;
};
struct Row
{
    std::vector<Hit> hits;
};
using Results = std::vector<Row>;
struct Aabb
{
    Vec minimum, maximum;
};

Fixture fixture(const Options& o);
// Column-vector rotation matrix of a unit quaternion, indexed m[row][column]. The double
// overload lets a caller renormalize first; a float quaternion leaves ~1e-7 non-orthogonality.
std::array<std::array<double, 3>, 3> rotationMatrix(const std::array<double, 4>& q);
std::array<std::array<double, 3>, 3> rotationMatrix(const Quat& q);
// World AABB of a primitive object in double precision: the independent oracle adapters are checked against.
Aabb                  worldBounds(const Object& ob);
std::vector<uint64_t> identities(const Row& row);
std::vector<uint64_t> owners(const Fixture& f, const Row& row, uint64_t excluded = 0);
void                  validateMeshFace(const Fixture& f, const Hit& h, const std::string& where = "");
bool                  near(double a, double b, double tolerance = 0);
// Largest deviation of each side's contact from the independent analytic evaluation of the
// reported triangle (metres): the ray or sphere-sweep distance against that face for `distance`,
// and the point on the ray at the reported distance for `offRay` (rays only).
struct CompareStats
{
    double maxOffRayFirst = 0, maxOffRaySecond = 0, maxDistanceErrorFirst = 0, maxDistanceErrorSecond = 0;
    // Ray and sphere sweep rows whose reported distance met the analytic face distance, per side.
    size_t analyticFirst = 0, analyticSecond = 0;
    // Rows where the first side reported a later object or face although the second side's contact
    // is analytically real and closer: the first side missed a contact (a named CPU/PhysX finding).
    size_t missedByFirst = 0;
    // Box sweeps: how far a PhysX baseline distance came before the installed first contact.
    double maxBaselineEarly = 0;
};
// Independent distance of the fixture's query `i` to the triangle a hit reports, or a negative
// value when the face is not hit: Moller-Trumbore in double for rays, the long-double convex search
// of support/native/sphere_sweep_oracle.h for sphere sweeps (-2 for a one-sided face whose front
// normal points along the sweep, which the library ignores, ADR-0051). Box sweeps use their own
// long-double first-contact solve inside `compare`.
double analyticFaceDistance(const Fixture& f, size_t i, const Hit& h);
void   compare(const Fixture& f, const Results& a, const Results& b, const std::string& label, bool ties = false,
               bool contactFields = true, CompareStats* stats = nullptr, bool firstIsPhysx = false);
void   move(Fixture& f, size_t count, size_t frame);
size_t changedCount(const Fixture& f, size_t percent);
void   printHit(const Hit& h);
// Output assembly consumes all requested contact fields and full owner sets.
// Sensor misses are encoded as max range with zero position/normal; hit mask is retained.
struct Observation
{
    std::vector<double>                ranges;
    std::vector<Vec>                   contacts;
    std::vector<std::array<float, 3>>  normals;
    std::vector<uint8_t>               active;
    std::vector<std::vector<uint64_t>> memberships;
    std::vector<std::vector<uint64_t>> previous;
    double                             checksum = 0;
    explicit Observation(size_t n) : ranges(n), contacts(n), normals(n), active(n), memberships(n), previous(n) {}
    void consume(const Fixture& f, const Results& results, bool transitions)
    {
        double value = 0;
        for (size_t i = 0; i < results.size(); ++i)
        {
            const auto& hits = results[i].hits;
            active[i]        = uint8_t(!hits.empty());
            value += double(i + 1) * active[i];
            if (f.kind == QueryKind::rays || f.kind == QueryKind::sweep || f.kind == QueryKind::boxSweep)
            {
                ranges[i]   = hits.empty() ? f.queries[i].farDistance : hits[0].distance;
                contacts[i] = hits.empty() ? Vec{} : hits[0].position;
                normals[i]  = hits.empty() ? std::array<float, 3>{} : hits[0].normal;
                value +=
                    ranges[i] + (hits.empty() ? 0 : double(hits[0].id + hits[0].geometryValue) + hits[0].geometryType);
                for (size_t j = 0; j < 3; ++j)
                    value += contacts[i][j] + normals[i][j];
            }
            else if (f.kind == QueryKind::zones || f.kind == QueryKind::sphereOverlap)
            {
                auto& ownersNow = memberships[i];
                ownersNow.clear();
                for (const auto& hit : hits)
                    ownersNow.push_back(f.objects[size_t(hit.id - 1)].owner);
                std::sort(ownersNow.begin(), ownersNow.end());
                ownersNow.erase(std::unique(ownersNow.begin(), ownersNow.end()), ownersNow.end());
                for (auto id : ownersNow)
                    value += double(id);
                if (transitions)
                {
                    for (auto id : ownersNow)
                        if (!std::binary_search(previous[i].begin(), previous[i].end(), id))
                            value += double(id) * 0.125;
                    for (auto id : previous[i])
                        if (!std::binary_search(ownersNow.begin(), ownersNow.end(), id))
                            value += double(id) * 0.25;
                    previous[i] = ownersNow;
                }
            }
        }
        checksum += value;
    }
};
} // namespace bench
#endif
