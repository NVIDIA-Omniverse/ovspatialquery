/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Pair scene replay, direct PhysX tree-pair and broadphase baselines, installed snapshot and reporter lanes,
// parity and timing.
#include "pairs.h"

#include "../../../support/native/cuda_driver.h"
#include "../installed_adapter.h"
#include "../measurement.h"

#include <ovspatialquery/ovspatialquery_pairs.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <thread>
#include <unordered_set>

namespace bench
{
namespace
{
// The published predicate: first_min <= second_max + tolerance on every axis, and symmetrically.
bool overlaps(const Aabb& a, const Aabb& b, double tolerance)
{
    for (size_t axis = 0; axis < 3; ++axis)
        if (a.minimum[axis] > b.maximum[axis] + tolerance || b.minimum[axis] > a.maximum[axis] + tolerance)
            return false;
    return true;
}
size_t gridWidth(size_t objects) { return size_t(std::ceil(std::sqrt(double(objects)))); }
// Both sides share one grid; the second side sits 0.6 along x and y from its
// first-side twin, so every second object pairs with exactly one first object
// at zero tolerance while row neighbours stay 0.03 or more apart.
Vec basePosition(size_t i, size_t width, bool second)
{
    const double offset = second ? .6 : 0;
    return {double(i % width) * 2 + offset, double(i / width) * 2 + offset, .5};
}
// Cells 6 m apart hold 2, 4, 16 or 64 axis-aligned boxes packed 2x1x1 up to
// 4x4x4, each overlapping its face neighbours by 0.05 m: every body pairs
// with its (up to 26) neighbours at zero tolerance and with nothing else,
// even at 0.1 m tolerance, so the tolerance changes the traversal, not the set.
constexpr double cellPitch = 6, bodyOverlap = .05, liftHeight = 6, shift = .02;
Vec              cellPosition(size_t i, size_t bodies, size_t cellsWide)
{
    const size_t nx = std::min<size_t>(bodies, 4), ny = bodies >= 16 ? 4 : 1;
    const size_t cell = i / bodies, local = i % bodies;
    return {double(cell % cellsWide) * cellPitch + double(local % nx) * (2 * boxHalf[0] - bodyOverlap),
            double(cell / cellsWide) * cellPitch + double(local / nx % ny) * (2 * boxHalf[1] - bodyOverlap),
            boxHalf[2] + double(local / (nx * ny)) * (2 * boxHalf[2] - bodyOverlap)};
}
size_t cellObjects(const PairScene& s) { return s.first.objects.size() + s.second.objects.size(); }
size_t cellsWide(const PairScene& s) { return gridWidth((cellObjects(s) + s.cellBodies - 1) / s.cellBodies); }
// Global cell-body index to its object: split alternates the bodies between the two populations.
Object& cellObject(PairScene& s, size_t i)
{
    if (s.topology == PairTopology::split)
        return (i % 2 ? s.second : s.first).objects[i / 2];
    return s.first.objects[i];
}
PairTopology topologyOf(const std::string& workload)
{
    return workload == "pairs-self"    ? PairTopology::self
           : workload == "pairs-views" ? PairTopology::views
           : workload == "pairs-split" ? PairTopology::split
                                       : PairTopology::sides;
}
const char* topologyName(PairTopology topology)
{
    switch (topology)
    {
    case PairTopology::sides: return "sides";
    case PairTopology::self: return "self";
    case PairTopology::views: return "views";
    case PairTopology::split: return "split";
    }
    return "unknown";
}

// One object of the oracle: its double world bounds, identity and roles (bit 0
// first target, bit 1 second target). Every entry is a distinct object.
struct OracleEntry
{
    Aabb     bounds;
    uint64_t id;
    uint8_t  roles;
};
std::vector<OracleEntry> oracleEntries(const PairScene& s)
{
    std::vector<OracleEntry> entries;
    for (size_t i = 0; i < s.first.objects.size(); ++i)
    {
        const auto& ob = s.first.objects[i];
        entries.push_back({worldBounds(ob), ob.id,
                           uint8_t(s.topology == PairTopology::views  ? s.roles[i]
                                   : s.topology == PairTopology::self ? 3
                                                                      : 1)});
    }
    for (const auto& ob : s.second.objects)
        entries.push_back({worldBounds(ob), ob.id, 2});
    return entries;
}
// Unordered identities for one population against itself; otherwise every
// admitted (first role, second role) direction of an overlapping object pair.
void emitOraclePair(const OracleEntry& a, const OracleEntry& b, bool unordered, PairSet& pairs)
{
    if (unordered)
    {
        pairs.emplace_back(std::min(a.id, b.id), std::max(a.id, b.id));
        return;
    }
    if ((a.roles & 1) && (b.roles & 2))
        pairs.emplace_back(a.id, b.id);
    if ((b.roles & 1) && (a.roles & 2))
        pairs.emplace_back(b.id, a.id);
}
// Sweep along x in double: an independent algorithm from the PhysX tree traversal
// and the ABP broadphase; brute force checks it on small scenes.
PairSet sweepPairs(std::vector<OracleEntry> entries, bool unordered, double tolerance)
{
    std::sort(entries.begin(), entries.end(),
              [](const OracleEntry& a, const OracleEntry& b) { return a.bounds.minimum[0] < b.bounds.minimum[0]; });
    PairSet pairs;
    for (size_t i = 0; i < entries.size(); ++i)
        for (size_t j = i + 1;
             j < entries.size() && !(entries[j].bounds.minimum[0] > entries[i].bounds.maximum[0] + tolerance); ++j)
            if (overlaps(entries[i].bounds, entries[j].bounds, tolerance))
                emitOraclePair(entries[i], entries[j], unordered, pairs);
    std::sort(pairs.begin(), pairs.end());
    return pairs;
}
PairSet bruteForcePairs(const std::vector<OracleEntry>& entries, bool unordered, double tolerance)
{
    PairSet pairs;
    for (size_t i = 0; i < entries.size(); ++i)
        for (size_t j = i + 1; j < entries.size(); ++j)
            if (overlaps(entries[i].bounds, entries[j].bounds, tolerance))
                emitOraclePair(entries[i], entries[j], unordered, pairs);
    std::sort(pairs.begin(), pairs.end());
    return pairs;
}
double consume(const PairSet& pairs)
{
    double value = double(pairs.size());
    for (const auto& p : pairs)
        value += double(p.first) * 1.5 + double(p.second);
    return value;
}
double consume(const ovspatialquery_pair_t* rows, size_t count)
{
    double value = double(count);
    for (size_t i = 0; i < count; ++i)
        value += double(rows[i].first.value) * 1.5 + double(rows[i].second.value);
    return value;
}
double consume(const DLTensor& t)
{
    const size_t count = t.shape ? size_t(t.shape[0]) : 0;
    double       value = double(count);
    for (size_t i = 0; i < count; ++i)
        value += double(element<uint64_t>(t, i, 0)) * 1.5 + double(element<uint64_t>(t, i, 1));
    return value;
}
PairSet sorted(PairSet set)
{
    std::sort(set.begin(), set.end());
    return set;
}
} // namespace

PairSet toSet(const ovspatialquery_pair_t* rows, size_t count)
{
    PairSet set;
    set.reserve(count);
    for (size_t i = 0; i < count; ++i)
        set.emplace_back(rows[i].first.value, rows[i].second.value);
    std::sort(set.begin(), set.end());
    return set;
}
PairSet toSet(const DLTensor& t)
{
    require(t.ndim == 2 && t.shape && t.shape[1] == 2 && t.dtype.code == kDLUInt && t.dtype.bits == 64,
            "reporter delta tensor shape/dtype");
    PairSet set;
    for (int64_t i = 0; i < t.shape[0]; ++i)
        set.emplace_back(element<uint64_t>(t, size_t(i), 0), element<uint64_t>(t, size_t(i), 1));
    std::sort(set.begin(), set.end());
    return set;
}
void requireEqual(const PairSet& expected, const PairSet& actual, const std::string& label)
{
    if (expected == actual)
        return;
    PairSet missing, extra;
    std::set_difference(expected.begin(), expected.end(), actual.begin(), actual.end(), std::back_inserter(missing));
    std::set_difference(actual.begin(), actual.end(), expected.begin(), expected.end(), std::back_inserter(extra));
    std::string detail =
        label + ": expected " + std::to_string(expected.size()) + " pairs, got " + std::to_string(actual.size());
    for (const auto* set : {&missing, &extra})
    {
        detail += set == &missing ? "; missing" : "; extra";
        for (size_t i = 0; i < std::min(set->size(), size_t(6)); ++i)
            detail += " (" + std::to_string((*set)[i].first) + "," + std::to_string((*set)[i].second) + ")";
    }
    throw std::runtime_error(detail);
}

namespace
{
const char* outcomeName(ovspatialquery_pair_reporter_outcome_t outcome)
{
    switch (outcome)
    {
    case OVSPATIALQUERY_PAIR_REPORTER_ADVANCED: return "advanced";
    case OVSPATIALQUERY_PAIR_REPORTER_ACTIVE_CAPACITY: return "active_capacity";
    case OVSPATIALQUERY_PAIR_REPORTER_DELTA_CAPACITY: return "delta_capacity";
    case OVSPATIALQUERY_PAIR_REPORTER_FAILED: return "failed";
    }
    return "unknown";
}

// PhysX allows one foundation per process; the parity run holds the tree lane and ABP.
struct PairFoundation
{
    physx::PxDefaultAllocator allocator;
    Errors                    errors;
    physx::PxFoundation*      foundation = nullptr;
    PairFoundation()
    {
        foundation = PxCreateFoundation(PX_PHYSICS_VERSION, allocator, errors);
        require(foundation != nullptr, "PxCreateFoundation failed");
    }
    ~PairFoundation() { foundation->release(); }
};
Errors& pairErrors()
{
    static PairFoundation shared;
    return shared.errors;
}

// Direct pinned PhysX across two populations: PxBounds3 per object, one PxBVH
// per population rebuilt on publication and PxFindOverlap across the two
// trees. The contact tolerance is a bounds offset on the second tree. No exact
// filter: the fixtures keep every bound far from a float rounding boundary.
class DirectPairs
{
    struct Tree
    {
        std::vector<physx::PxBounds3> bounds;
        physx::PxBVH*                 bvh = nullptr;
    };
    struct Collector final : physx::PxReportCallback<physx::PxGeomIndexPair>
    {
        const PairScene& scene;
        PairSet&         out;
        size_t           reports = 0;
        Collector(const PairScene& s, PairSet& pairs) : scene(s), out(pairs) {}
        bool flushResults(physx::PxU32 count, const physx::PxGeomIndexPair* items) override
        {
            reports += count;
            for (physx::PxU32 i = 0; i < count; ++i)
                out.emplace_back(scene.first.objects.at(items[i].id0).id, scene.second.objects.at(items[i].id1).id);
            return true;
        }
    };
    Errors&          errors = pairErrors();
    const PairScene& scene;
    Tree             first, second;
    static void      bounds(const Object& ob, physx::PxBounds3& out, float offset)
    {
        const bool ok =
            ob.sphere ? physx::PxGeometryQuery::computeGeomBounds(out, physx::PxSphereGeometry(float(sphereRadius)),
                                                                  pose(ob.p, identity), offset, 1)
                      : physx::PxGeometryQuery::computeGeomBounds(out, physx::PxBoxGeometry(px(boxHalf)),
                                                                  pose(ob.p, ob.q), offset, 1);
        require(ok, "PhysX bounds computation failed");
    }
    void rebuild(Tree& t, const Fixture& f, size_t changed, float offset)
    {
        if (t.bounds.size() != f.objects.size())
        {
            t.bounds.resize(f.objects.size());
            changed = f.objects.size();
        }
        for (size_t i = 0; i < changed; ++i)
            bounds(f.objects[i], t.bounds[i], offset);
        physx::PxBVHDesc desc;
        desc.enlargement   = 0;
        desc.bounds.count  = uint32_t(t.bounds.size());
        desc.bounds.stride = sizeof(physx::PxBounds3);
        desc.bounds.data   = t.bounds.data();
        physx::PxBVH* bvh  = PxCreateBVH(desc, *PxGetStandaloneInsertionCallback());
        require(bvh != nullptr, "PxCreateBVH failed");
        if (t.bvh)
            t.bvh->release();
        t.bvh = bvh;
    }

  public:
    double publicationMs = 0;
    size_t reports       = 0; // Candidate index pairs PxFindOverlap reported in the last find.
    explicit DirectPairs(const PairScene& s) : scene(s)
    {
        require(s.twoPopulations(), "the PhysX tree lane crosses two populations");
        const auto start = Clock::now();
        rebuild(first, s.first, 0, 0);
        rebuild(second, s.second, 0, float(s.tolerance));
        publicationMs = elapsed(start);
        require(!errors.failed, "PhysX reported setup error");
    }
    ~DirectPairs()
    {
        for (Tree* t : {&first, &second})
            if (t->bvh)
                t->bvh->release();
    }
    void update(size_t step)
    {
        if (scene.firstPublishes(step) && scene.firstMoved())
            rebuild(first, scene.first, scene.firstMoved(), 0);
        if (scene.secondMoved())
            rebuild(second, scene.second, scene.secondMoved(), float(scene.tolerance));
        require(!errors.failed, "PhysX reported rebuild error");
    }
    void find(PairSet& out)
    {
        out.clear();
        Collector collector(scene, out);
        (void)physx::PxFindOverlap(collector, *first.bvh, *second.bvh);
        reports = collector.reports;
        std::sort(out.begin(), out.end());
        require(!errors.failed, "PhysX reported pair query error");
    }
};

// Direct pinned PhysX ABP broadphase through PxAABBManager: every
// object is added once and only changed objects are updated. Float bounds
// round the double world bounds outward and each object carries half the
// tolerance, so the broadphase candidates are a superset of the exact pairs.
// The lane keeps each candidate's neighbours, evaluates the double predicate
// for created candidates and for every candidate of a changed object (an
// existing candidate can start or stop overlapping without a broadphase
// event), and maps entered/exited index pairs to the targets' identities.
// It is mutable and latest-only: no publication, retained versions or
// capacity outcomes.
class BroadphasePairs
{
    Errors&                            errors     = pairErrors();
    physx::PxBroadPhase*               broadphase = nullptr;
    physx::PxAABBManager*              manager    = nullptr;
    const PairScene&                   scene;
    size_t                             secondOffset = 0;
    float                              distance     = 0;
    std::vector<Aabb>                  exact;
    std::vector<uint64_t>              ids;
    std::vector<uint8_t>               roles;
    std::vector<std::vector<uint32_t>> neighbours;
    std::unordered_set<uint64_t>       active; // Unordered index pairs whose exact bounds overlap.
    std::vector<uint32_t>              changed;
    physx::PxBroadPhaseResults         results;

    static uint64_t key(uint32_t a, uint32_t b) { return uint64_t(std::min(a, b)) << 32 | std::max(a, b); }
    static float    down(double v)
    {
        float f = float(v);
        return double(f) > v ? std::nextafter(f, -INFINITY) : f;
    }
    static float up(double v)
    {
        float f = float(v);
        return double(f) < v ? std::nextafter(f, INFINITY) : f;
    }
    const Object& object(size_t index) const
    {
        return index < secondOffset ? scene.first.objects[index] : scene.second.objects[index - secondOffset];
    }
    physx::PxBounds3 floatBounds(size_t index)
    {
        exact[index]  = worldBounds(object(index));
        const Aabb& b = exact[index];
        return {{down(b.minimum[0]), down(b.minimum[1]), down(b.minimum[2])},
                {up(b.maximum[0]), up(b.maximum[1]), up(b.maximum[2])}};
    }
    void emit(uint32_t a, uint32_t b, PairSet& out) const
    {
        if (scene.topology == PairTopology::self)
        {
            out.emplace_back(std::min(ids[a], ids[b]), std::max(ids[a], ids[b]));
            return;
        }
        if ((roles[a] & 1) && (roles[b] & 2))
            out.emplace_back(ids[a], ids[b]);
        if ((roles[b] & 1) && (roles[a] & 2))
            out.emplace_back(ids[b], ids[a]);
    }
    void unlink(uint32_t a, uint32_t b)
    {
        auto& list = neighbours[a];
        auto  it   = std::find(list.begin(), list.end(), b);
        require(it != list.end(), "broadphase lost an unknown candidate");
        *it = list.back();
        list.pop_back();
    }
    // Reconciles one candidate with the exact predicate; idempotent.
    void evaluate(uint32_t a, uint32_t b, PairSet& entered, PairSet& exited)
    {
        if (overlaps(exact[a], exact[b], scene.tolerance))
        {
            if (active.insert(key(a, b)).second)
                emit(a, b, entered);
        }
        else if (active.erase(key(a, b)))
            emit(a, b, exited);
    }

  public:
    double publicationMs = 0;
    size_t candidates    = 0; // Current broadphase candidate pairs.
    size_t pairs         = 0; // Current exact pairs in the targets' identities.
    explicit BroadphasePairs(const PairScene& s) : scene(s)
    {
        const auto start = Clock::now();
        broadphase       = physx::PxCreateBroadPhase(physx::PxBroadPhaseDesc(physx::PxBroadPhaseType::eABP));
        require(broadphase != nullptr, "PxCreateBroadPhase failed");
        manager = physx::PxCreateAABBManager(*broadphase);
        require(manager != nullptr, "PxCreateAABBManager failed");
        // Two ulps above half the tolerance keep the sum of both inflations at or above it.
        distance           = std::nextafter(up(s.tolerance / 2), INFINITY);
        secondOffset       = s.first.objects.size();
        const size_t count = secondOffset + s.second.objects.size();
        exact.resize(count);
        neighbours.resize(count);
        for (size_t i = 0; i < count; ++i)
        {
            ids.push_back(object(i).id);
            roles.push_back(s.twoPopulations()                  ? (i < secondOffset ? 1 : 2)
                            : s.topology == PairTopology::views ? s.roles[i]
                                                                : 3);
            // Two populations: one dynamic group per side filters same-side pairs.
            const physx::PxBpFilterGroup group = physx::PxGetBroadPhaseDynamicFilterGroup(
                s.twoPopulations() ? (i < secondOffset ? 0u : 1u) : uint32_t(i));
            manager->addObject(uint32_t(i), floatBounds(i), group, distance);
        }
        publicationMs = elapsed(start);
        require(!errors.failed, "PhysX reported broadphase setup error");
    }
    ~BroadphasePairs()
    {
        if (manager)
            manager->release();
        if (broadphase)
            broadphase->release();
    }
    // Bounds of the changed objects of each publishing side.
    void update(size_t step)
    {
        changed.clear();
        if (scene.firstPublishes(step))
            for (size_t i = 0; i < scene.firstMoved(); ++i)
                changed.push_back(uint32_t(i));
        if (scene.twoPopulations())
            for (size_t i = 0; i < scene.secondMoved(); ++i)
                changed.push_back(uint32_t(secondOffset + i));
        for (uint32_t index : changed)
        {
            const physx::PxBounds3 b = floatBounds(index);
            manager->updateObject(index, &b);
        }
    }
    void broadphaseUpdate()
    {
        manager->update();
        manager->fetchResults(results);
        require(!errors.failed, "PhysX reported broadphase update error");
    }
    // Candidate maintenance, exact filtering and entered/exited identities.
    void reduce(PairSet& entered, PairSet& exited)
    {
        entered.clear();
        exited.clear();
        for (physx::PxU32 i = 0; i < results.mNbDeletedPairs; ++i)
        {
            const auto& p = results.mDeletedPairs[i];
            unlink(p.mID0, p.mID1);
            unlink(p.mID1, p.mID0);
            if (active.erase(key(p.mID0, p.mID1)))
                emit(p.mID0, p.mID1, exited);
        }
        for (physx::PxU32 i = 0; i < results.mNbCreatedPairs; ++i)
        {
            const auto& p = results.mCreatedPairs[i];
            neighbours[p.mID0].push_back(p.mID1);
            neighbours[p.mID1].push_back(p.mID0);
            evaluate(p.mID0, p.mID1, entered, exited);
        }
        for (uint32_t a : changed)
            for (uint32_t b : neighbours[a])
                evaluate(a, b, entered, exited);
        candidates += results.mNbCreatedPairs;
        candidates -= results.mNbDeletedPairs;
        pairs += entered.size();
        pairs -= exited.size();
    }
    PairSet activePairs() const
    {
        PairSet out;
        for (uint64_t k : active)
            emit(uint32_t(k >> 32), uint32_t(k), out);
        std::sort(out.begin(), out.end());
        return out;
    }
};

// Installed library: one instance, the first side through the shared adapter;
// two populations publish the second side as a second explicit population on
// the same instance, views select two explicit views of the first.
class InstalledPairs
{
  public:
    Spatial                               spatial;
    const PairScene&                      scene;
    ovspatialquery_population_handle_t    second         = OVSPATIALQUERY_INVALID_POPULATION;
    ovspatialquery_ordinal_t              secondOrdinal  = 0;
    ovspatialquery_view_handle_t          firstView      = OVSPATIALQUERY_INVALID_VIEW;
    ovspatialquery_view_handle_t          secondView     = OVSPATIALQUERY_INVALID_VIEW;
    ovspatialquery_pair_reporter_handle_t reporter       = OVSPATIALQUERY_INVALID_PAIR_REPORTER;
    uint64_t                              activeCapacity = 0, deltaCapacity = 0;
    std::vector<ovspatialquery_pair_t>    rows;
    ovspatialquery_pair_snapshot_result_t last{};
    // A CUDA reporter (ADR-0019 stage 3): its delta rows are mirrored to the
    // host on this stream after the result is ready, and each advance retires
    // after that copy, the host consumer's event.
    bool                                 cudaResults = false;
    std::unique_ptr<CudaDriver>          cuda;
    CudaDriver::Stream                   stream = nullptr;
    std::array<std::vector<uint64_t>, 2> mirrors;
    size_t                               resultCapacityRetries = 0;
    // A GPU client (`reporter-gpu`): each step's changed rows already sit in
    // device arrays, uploaded by produce() before the timed update, which
    // publishes them as CUDA input read on `stream` after `produced`.
    bool cudaInput = false;
    struct DeviceSide
    {
        std::unique_ptr<InstanceRows>            host;
        std::array<CudaDriver::DevicePointer, 6> columns{};
        size_t                                   capacity = 0;
        std::array<std::unique_ptr<Input>, 7>    inputs; // Six instance columns, then the empty removed column.
        ovspatialquery_explicit_instance_table_t table{};
    };
    std::array<DeviceSide, 2> deviceSides;
    CudaDriver::Event         produced = nullptr;
    struct Advance
    {
        ovspatialquery_pair_reporter_result_t result{};
        ovspatialquery_op_id_t                op       = OVSPATIALQUERY_INVALID_OP;
        CudaDriver::Event                     consumer = nullptr;
    };
    InstalledPairs(const PairScene& s, bool cudaReporter, bool cudaPublication = false)
        : spatial(s.first), scene(s), cudaResults(cudaReporter), cudaInput(cudaPublication)
    {
        if (cudaResults || cudaInput)
        {
            cuda   = std::make_unique<CudaDriver>(CudaDriver::Ownership::tracked);
            stream = cuda->createStream();
        }
        if (cudaInput)
            produced = cuda->createEvent();
        if (s.twoPopulations())
            second = spatial.publishPopulation(s.second, secondOrdinal);
        if (s.topology == PairTopology::views)
        {
            firstView  = createView(1);
            secondView = createView(2);
        }
    }
    ~InstalledPairs()
    {
        destroyReporter();
        for (auto view : {firstView, secondView})
            if (view != OVSPATIALQUERY_INVALID_VIEW)
                Spatial::checkRelease(ovspatialquery_destroy_view(spatial.instance, view), "destroy pair view");
        if (second != OVSPATIALQUERY_INVALID_POPULATION)
            Spatial::checkRelease(ovspatialquery_destroy_population(spatial.instance, second),
                                  "destroy second population");
    }
    ovspatialquery_view_handle_t createView(uint8_t role)
    {
        std::vector<uint64_t> selected;
        for (size_t i = 0; i < scene.first.objects.size(); ++i)
            if (scene.roles[i] & role)
                selected.push_back(scene.first.objects[i].id);
        Input                               subset(selected.data(), selected.size(), 0, kDLUInt, 64);
        ovspatialquery_explicit_view_desc_t vd{&subset.value};
        ovspatialquery_view_handle_t        view = OVSPATIALQUERY_INVALID_VIEW;
        check(ovspatialquery_create_explicit_view(spatial.instance, spatial.population, &vd, &view),
              "create pair view");
        return view;
    }
    static ovspatialquery_target_ref_t viewRef(ovspatialquery_view_handle_t view)
    {
        ovspatialquery_target_ref_t ref{};
        ref.kind        = OVSPATIALQUERY_TARGET_VIEW;
        ref.handle.view = view;
        return ref;
    }
    ovspatialquery_target_ref_t firstRef() const
    {
        return scene.topology == PairTopology::views ? viewRef(firstView) : spatial.target(false);
    }
    ovspatialquery_target_ref_t secondRef() const
    {
        if (scene.topology == PairTopology::views)
            return viewRef(secondView);
        if (!scene.twoPopulations())
            return spatial.target(false);
        ovspatialquery_target_ref_t ref{};
        ref.kind              = OVSPATIALQUERY_TARGET_POPULATION;
        ref.handle.population = second;
        return ref;
    }
    ovspatialquery_ordinal_t secondHead() const { return scene.twoPopulations() ? secondOrdinal : spatial.ordinal; }
    // The GPU client's producer: this step's changed rows in device arrays, complete before timing starts.
    void produce(size_t step)
    {
        if (!cudaInput)
            return;
        if (scene.firstPublishes(step) && scene.firstMoved())
            produceSide(deviceSides[0], scene.first, scene.firstMoved());
        if (scene.twoPopulations() && scene.secondMoved())
            produceSide(deviceSides[1], scene.second, scene.secondMoved());
        cuda->record(produced, stream);
        cuda->synchronize(produced);
    }
    void update(size_t step)
    {
        if (scene.firstPublishes(step) && scene.firstMoved())
        {
            if (cudaInput)
                publishDevice(deviceSides[0], spatial.population, spatial.ordinal);
            else
                spatial.update(scene.first, scene.firstMoved());
        }
        if (scene.twoPopulations() && scene.secondMoved())
        {
            if (cudaInput)
                publishDevice(deviceSides[1], second, secondOrdinal);
            else
                spatial.update(scene.second, scene.secondMoved(), second, secondOrdinal);
        }
    }
    void produceSide(DeviceSide& d, const Fixture& f, size_t count)
    {
        d.host                                                = std::make_unique<InstanceRows>(f, count);
        const InstanceRows&                                 h = *d.host;
        const std::array<std::pair<const void*, size_t>, 6> sources{{{h.ids.data(), sizeof(uint64_t)},
                                                                     {h.types.data(), sizeof(int32_t)},
                                                                     {h.definitions.data(), sizeof(uint64_t)},
                                                                     {h.bits.data(), 2u * sizeof(uint64_t)},
                                                                     {h.positions.data(), sizeof(Vec)},
                                                                     {h.rotations.data(), sizeof(Quat)}}};
        if (d.capacity < count)
        {
            for (size_t i = 0; i < sources.size(); ++i)
            {
                if (d.columns[i])
                    cuda->free(d.columns[i]);
                d.columns[i] = cuda->allocate(count * sources[i].second);
            }
            d.capacity = count;
        }
        for (size_t i = 0; i < sources.size(); ++i)
            cuda->uploadAsync(d.columns[i], sources[i].first, count * sources[i].second, stream);
        const DLDevice device{kDLCUDA, 0};
        const auto     column      = [&d](size_t i) { return reinterpret_cast<void*>(uintptr_t(d.columns[i])); };
        d.inputs[0]                = std::make_unique<Input>(column(0), count, 0, kDLUInt, 64, device);
        d.inputs[1]                = std::make_unique<Input>(column(1), count, 0, kDLInt, 32, device);
        d.inputs[2]                = std::make_unique<Input>(column(2), count, 0, kDLUInt, 64, device);
        d.inputs[3]                = std::make_unique<Input>(column(3), count, 2, kDLUInt, 64, device);
        d.inputs[4]                = std::make_unique<Input>(column(4), count, 3, kDLFloat, 64, device);
        d.inputs[5]                = std::make_unique<Input>(column(5), count, 4, kDLFloat, 32, device);
        d.inputs[6]                = std::make_unique<Input>(nullptr, 0, 0, kDLUInt, 64, device);
        d.table.object_values      = &d.inputs[0]->value;
        d.table.geometry_types     = &d.inputs[1]->value;
        d.table.geometry_values    = &d.inputs[2]->value;
        d.table.participation_bits = &d.inputs[3]->value;
        d.table.translations       = &d.inputs[4]->value;
        d.table.rotations_xyzw     = &d.inputs[5]->value;
    }
    void publishDevice(DeviceSide& d, ovspatialquery_population_handle_t target, ovspatialquery_ordinal_t& ordinal)
    {
        ovspatialquery_dynamic_execution_t execution{};
        execution.device               = {kDLCUDA, 0};
        execution.cuda_sync.has_stream = 1u;
        execution.cuda_sync.stream     = reinterpret_cast<uintptr_t>(stream);
        execution.cuda_sync.wait_event = reinterpret_cast<uintptr_t>(produced);
        ovspatialquery_explicit_population_update_t change{};
        change.base_ordinal          = ordinal;
        change.target_ordinal        = ordinal + 1;
        change.changed_instances     = d.table;
        change.removed_object_values = &d.inputs[6]->value;
        change.dynamic_execution     = &execution;
        ordinal = spatial.finish(ovspatialquery_enqueue_explicit_population_update(spatial.instance, target, &change),
                                 true, "CUDA-input population update");
    }
    // Stateless snapshot at both heads into the first `capacity` rows; only
    // snapshot lanes hold the row buffer.
    const ovspatialquery_pair_snapshot_result_t& snapshot(uint64_t capacity)
    {
        if (rows.size() < capacity)
            rows.resize(capacity);
        ovspatialquery_pair_snapshot_desc_t desc{};
        desc.first.target      = firstRef();
        desc.second.target     = secondRef();
        desc.contact_tolerance = scene.tolerance;
        ovspatialquery_pair_buffer_t buffer{capacity ? rows.data() : nullptr, capacity};
        last = {};
        check(ovspatialquery_query_pairs_sync(spatial.instance, &desc, &buffer, &last), "pair snapshot");
        require(last.first_source_ordinal == spatial.ordinal && last.second_source_ordinal == secondHead(),
                "snapshot ordinals differ from the published heads");
        return last;
    }
    void createReporter(uint64_t active, uint64_t delta)
    {
        destroyReporter();
        ovspatialquery_pair_reporter_desc_t desc{};
        desc.first                = firstRef();
        desc.second               = secondRef();
        desc.active_pair_capacity = active;
        desc.delta_pair_capacity  = delta;
        // A CUDA reporter's result slot waits for its consumer: two let the
        // next advance start while the previous one retires.
        desc.result_slot_count = cudaResults ? 2 : 1;
        desc.contact_tolerance = scene.tolerance;
        desc.result_device     = cudaResults ? DLDevice{kDLCUDA, 0} : DLDevice{kDLCPU, 0};
        check(ovspatialquery_create_pair_reporter(spatial.instance, &desc, &reporter), "create pair reporter");
        activeCapacity = active;
        deltaCapacity  = delta;
    }
    void destroyReporter()
    {
        if (reporter != OVSPATIALQUERY_INVALID_PAIR_REPORTER)
        {
            // A CUDA reporter is busy until its last advance retired.
            ovspatialquery_result_t released{};
            for (size_t attempt = 0; attempt < 100000; ++attempt)
            {
                released = ovspatialquery_destroy_pair_reporter(spatial.instance, reporter);
                if (released.status != OVSPATIALQUERY_API_BUSY || !cudaResults)
                    break;
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            Spatial::checkRelease(released, "destroy pair reporter");
        }
        reporter = OVSPATIALQUERY_INVALID_PAIR_REPORTER;
    }
    void grow(uint64_t active, uint64_t delta)
    {
        ovspatialquery_result_t grown{};
        for (size_t attempt = 0; attempt < 100000; ++attempt)
        {
            grown = ovspatialquery_grow_pair_reporter_capacity(spatial.instance, reporter, active, delta);
            if (grown.status != OVSPATIALQUERY_API_BUSY || !cudaResults)
                break;
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        check(grown, "grow pair reporter");
        activeCapacity = active;
        deltaCapacity  = delta;
    }
    // Enqueue at both heads, wait and fetch; the caller consumes the tensors, then releases.
    Advance advance()
    {
        ovspatialquery_pair_reporter_advance_t     selection{};
        ovspatialquery_pair_reporter_reservation_t reservation{};
        auto                                       admitted =
            ovspatialquery_enqueue_pair_reporter_advance(spatial.instance, reporter, &selection, &reservation);
        // Both slots can still wait for their consumers' retirement.
        for (size_t attempt = 0; cudaResults && admitted.status == OVSPATIALQUERY_API_RESULT_CAPACITY; ++attempt)
        {
            require(attempt < 1000000, "CUDA reporter slots never retired");
            ++resultCapacityRetries;
            std::this_thread::yield();
            admitted =
                ovspatialquery_enqueue_pair_reporter_advance(spatial.instance, reporter, &selection, &reservation);
        }
        check({admitted.status}, "enqueue reporter advance");
        ovspatialquery_op_wait_result_t waited{};
        check(ovspatialquery_wait_op(spatial.instance, admitted.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited),
              "wait reporter advance");
        if (waited.state != OVSPATIALQUERY_OP_COMPLETED)
        {
            const auto error = ovspatialquery_get_last_op_error(spatial.instance, admitted.op_id);
            throw std::runtime_error("reporter advance failed: " +
                                     std::string(error.ptr ? error.ptr : "", size_t(error.length)));
        }
        Advance a;
        a.op = admitted.op_id;
        check(ovspatialquery_fetch_pair_reporter(spatial.instance, a.op, &a.result), "fetch reporter result");
        require(a.result.result_slot == reservation.result_slot && a.result.first_source_ordinal == spatial.ordinal &&
                    a.result.second_source_ordinal == secondHead(),
                "reporter result slot/ordinals differ from the reservation and heads");
        return a;
    }
    // Copies a CUDA result's delta rows to the host once the device signals them,
    // so the same consumer reads them; the copy's event is the consumer event.
    void mirror(Advance& a)
    {
        if (!cudaResults)
            return;
        require(a.result.result_ready.value != 0 && a.result.entered_pairs.device.device_type == kDLCUDA,
                "CUDA reporter result is not on the device");
        cuda->wait(stream, reinterpret_cast<CudaDriver::Event>(a.result.result_ready.value));
        size_t index = 0;
        for (DLTensor* tensor : {&a.result.entered_pairs, &a.result.exited_pairs})
        {
            auto&        host  = mirrors[index++];
            const size_t count = size_t(tensor->shape[0]);
            host.resize(std::max<size_t>(2 * count, 2));
            if (count)
                cuda->downloadAsync(host.data(),
                                    static_cast<CudaDriver::DevicePointer>(reinterpret_cast<uintptr_t>(tensor->data)),
                                    2 * count * sizeof(uint64_t), stream);
            tensor->data   = host.data();
            tensor->device = {kDLCPU, 0};
        }
        a.consumer = cuda->createEvent();
        cuda->record(a.consumer, stream);
        cuda->synchronize(a.consumer);
    }
    void release(Advance& a)
    {
        if (!cudaResults)
        {
            check(ovspatialquery_destroy_op(spatial.instance, &a.op), "destroy advance op");
            return;
        }
        ovspatialquery_cuda_event_t done{reinterpret_cast<uintptr_t>(a.consumer)};
        check(ovspatialquery_destroy_op_after_cuda(spatial.instance, &a.op, &done), "retire CUDA advance");
        cuda->forgetTransferredEvent(a.consumer);
    }
};

// Default snapshot buffer and reporter capacity: the pre-existing object-based
// sizes for `pairs`, twice the initial pair count for the cell workloads.
size_t pairCapacity(const Options& o, const PairScene& scene, size_t initialPairs)
{
    if (o.pairCapacity)
        return o.pairCapacity;
    if (scene.topology == PairTopology::sides)
        return (scene.tolerance > 0 ? 4 : 2) * o.objects;
    return 2 * initialPairs + 64;
}

// Complete unordered parity of every lane against the double oracle over five
// steps, including truncation, the first-use capacity outcomes and deltas.
void pairsParity(const Options& o, PairScene& scene)
{
    const PairSet initial  = pairOracle(scene);
    const size_t  capacity = std::max(8 * o.objects + 8, 2 * initial.size() + 64);
    // The PhysX tree lane crosses two populations only.
    std::unique_ptr<DirectPairs> direct = scene.twoPopulations() ? std::make_unique<DirectPairs>(scene) : nullptr;
    BroadphasePairs              abp(scene);
    InstalledPairs               installed(scene, false);
    require(initial.size() >= 2, "pair fixtures need at least two initial pairs");
    installed.createReporter(initial.size() / 2, 2 * scene.toggled + 8);
    // The CUDA reporter (ADR-0019 stage 3) on its own instance, where a device exists.
    // A GPU client publishing its changed rows from device arrays feeds another.
    std::string                     cudaMessage;
    std::unique_ptr<InstalledPairs> cudaInstalled, gpuInstalled;
    if (cudaPlansAvailable(cudaMessage))
    {
        cudaInstalled = std::make_unique<InstalledPairs>(scene, true);
        cudaInstalled->createReporter(initial.size() / 2, 2 * scene.toggled + 8);
        gpuInstalled = std::make_unique<InstalledPairs>(scene, true, true);
        gpuInstalled->createReporter(initial.size() / 2, 2 * scene.toggled + 8);
    }
    PairSet                  previous;
    std::vector<size_t>      counts;
    std::vector<std::string> outcomes;
    for (size_t step = 0; step < 5; ++step)
    {
        if (step)
        {
            pairStep(scene, step);
            if (direct)
                direct->update(step);
            abp.update(step);
            installed.update(step);
            if (cudaInstalled)
                cudaInstalled->update(step);
            if (gpuInstalled)
            {
                gpuInstalled->produce(step);
                gpuInstalled->update(step);
            }
        }
        const std::string where  = "pairs step " + std::to_string(step) + " ";
        const PairSet     oracle = pairOracle(scene);
        counts.push_back(oracle.size());
        const auto entries = oracleEntries(scene);
        if (entries.size() <= 2000)
            requireEqual(bruteForcePairs(entries, scene.topology == PairTopology::self, scene.tolerance), oracle,
                         where + "sweep oracle versus brute force");
        if (scene.tolerance > 0 && step == 0 && scene.topology == PairTopology::sides)
        {
            PairScene zero = scene;
            zero.tolerance = 0;
            require(pairOracle(zero).size() < oracle.size(), where + "tolerance does not add any pair");
        }
        if (direct)
        {
            PairSet physx;
            direct->find(physx);
            requireEqual(oracle, physx, where + "PhysX tree pairs versus the double oracle");
        }
        PairSet entered, exited;
        std::set_difference(oracle.begin(), oracle.end(), previous.begin(), previous.end(),
                            std::back_inserter(entered));
        std::set_difference(previous.begin(), previous.end(), oracle.begin(), oracle.end(), std::back_inserter(exited));
        if (step)
            require(!entered.empty() || !exited.empty(), where + "replay produced no enter/exit transition");
        PairSet abpEntered, abpExited;
        abp.broadphaseUpdate();
        abp.reduce(abpEntered, abpExited);
        requireEqual(oracle, abp.activePairs(), where + "ABP active pairs versus the double oracle");
        requireEqual(entered, sorted(abpEntered), where + "ABP entered pairs");
        requireEqual(exited, sorted(abpExited), where + "ABP exited pairs");
        const auto& full = installed.snapshot(capacity);
        require(!full.truncated, where + "installed snapshot truncated below the parity capacity");
        requireEqual(oracle, toSet(installed.rows.data(), full.stored_pair_count),
                     where + "installed snapshot versus the double oracle");
        const auto& probe = installed.snapshot(0);
        require(probe.stored_pair_count == 0 && (probe.truncated != 0) == !oracle.empty(),
                where + "zero-capacity probe truncation");
        if (!oracle.empty())
        {
            const auto& partial = installed.snapshot(oracle.size() - 1);
            require(partial.truncated && partial.stored_pair_count == oracle.size() - 1, where + "short buffer");
            for (size_t i = 0; i < partial.stored_pair_count; ++i)
                require(
                    std::binary_search(oracle.begin(), oracle.end(),
                                       std::make_pair(installed.rows[i].first.value, installed.rows[i].second.value)),
                    where + "short buffer row outside the complete set");
        }
        for (InstalledPairs* lane : {&installed, cudaInstalled.get(), gpuInstalled.get()})
        {
            if (lane == nullptr)
                continue;
            const std::string name = lane->cudaInput     ? "CUDA-input CUDA reporter "
                                     : lane->cudaResults ? "CUDA reporter "
                                                         : "reporter ";
            for (size_t attempt = 0;; ++attempt)
            {
                require(attempt < 4, where + name + "did not advance within three capacity retries");
                auto a = lane->advance();
                lane->mirror(a);
                if (lane == &installed)
                    outcomes.push_back(outcomeName(a.result.outcome));
                if (a.result.outcome == OVSPATIALQUERY_PAIR_REPORTER_ADVANCED)
                {
                    require(a.result.accepted_active_pair_count == oracle.size(), where + name + "accepted count");
                    requireEqual(entered, toSet(a.result.entered_pairs), where + name + "entered pairs");
                    requireEqual(exited, toSet(a.result.exited_pairs), where + name + "exited pairs");
                    lane->release(a);
                    break;
                }
                require(a.result.accepted_active_pair_count == previous.size() &&
                            a.result.entered_pairs.shape[0] == 0 && a.result.exited_pairs.shape[0] == 0,
                        where + name + "capacity outcome exposed a partial delta or moved the baseline");
                const bool active = a.result.outcome == OVSPATIALQUERY_PAIR_REPORTER_ACTIVE_CAPACITY;
                lane->release(a);
                lane->grow(active ? capacity : lane->activeCapacity, active ? lane->deltaCapacity : capacity);
            }
        }
        previous = oracle;
    }
    // The first-use sequence on these fixtures: active capacity of half the pairs,
    // then a delta capacity sized for steady deltas, then the complete advance.
    require(outcomes.size() >= 3 && outcomes[0] == "active_capacity" && outcomes[1] == "delta_capacity" &&
                outcomes[2] == "advanced",
            "first-use capacity outcomes did not occur in the expected order");
    for (size_t i = 3; i < outcomes.size(); ++i)
        require(outcomes[i] == "advanced", "steady advance reported a capacity outcome");
    std::cout << "{\"backend\":\"parity\",\"workload\":\"" << o.workload << "\",\"parity\":true,\"topology\":\""
              << topologyName(scene.topology) << "\",\"objects\":" << o.objects
              << ",\"cell_bodies\":" << scene.cellBodies << ",\"steps\":5,\"contact_tolerance\":" << scene.tolerance
              << ",\"moved_per_step\":" << scene.moved << ",\"toggled_per_step\":" << scene.toggled
              << ",\"first_cadence\":" << scene.firstCadence
              << ",\"cuda_reporter\":" << (cudaInstalled ? "true" : "false")
              << ",\"cuda_input_reporter\":" << (gpuInstalled ? "true" : "false") << ",\"pair_counts\":[";
    for (size_t i = 0; i < counts.size(); ++i)
        std::cout << (i ? "," : "") << counts[i];
    std::cout << "],\"reporter_outcomes\":[";
    for (size_t i = 0; i < outcomes.size(); ++i)
        std::cout << (i ? ",\"" : "\"") << outcomes[i] << '"';
    std::cout << "],\"checks\":[\"sweep_oracle_equal_brute_force_up_to_2000_objects\","
              << (direct ? "\"physx_tree_pairs_equal_double_oracle\"," : "")
              << "\"abp_active_and_deltas_equal_double_oracle\","
                 "\"snapshot_equal_double_oracle\",\"zero_capacity_probe\",\"short_buffer_subset\","
                 "\"reporter_delta_equal_set_difference\",\"capacity_outcomes_preserve_baseline\"],\"physx_version\":"
              << PX_PHYSICS_VERSION << ",\"physx_config\":\"" << BENCH_PHYSX_CONFIG
              << "\",\"synthetic_analogue\":true}\n";
}
} // namespace

PairScene pairScene(const Options& o)
{
    PairScene s;
    s.topology  = topologyOf(o.workload);
    s.moved     = std::max(size_t(1), o.objects * o.updatePercent / 100);
    s.tolerance = double(o.pairToleranceMilli) / 1000;
    if (s.topology == PairTopology::sides)
    {
        s.toggled          = std::max(size_t(1), s.moved / 4);
        const size_t width = gridWidth(o.objects);
        for (size_t i = 0; i < o.objects; ++i)
        {
            s.first.objects.push_back(
                {i + 1, i + 1, i % 3 == 1, basePosition(i, width, false), i % 4 == 0 ? rotation(0.5f) : identity, 1});
            s.second.objects.push_back(
                {i + 1, i + 1, i % 3 == 1, basePosition(i, width, true), i % 4 == 0 ? rotation(0.5f) : identity, 1});
        }
        pairStep(s, 0);
        return s;
    }
    s.cellBodies      = o.queries;
    s.firstCadence    = 1;
    s.toggled         = (s.moved + 3) / 4;
    const size_t wide = gridWidth((o.objects + s.cellBodies - 1) / s.cellBodies);
    for (size_t i = 0; i < o.objects; ++i)
    {
        const Object ob{i + 1, i + 1, false, cellPosition(i, s.cellBodies, wide), identity, 1};
        (s.topology == PairTopology::split && i % 2 ? s.second : s.first).objects.push_back(ob);
        // Views: a third first-only, a third in both views, a third second-only.
        if (s.topology == PairTopology::views)
            s.roles.push_back(uint8_t(i % 3 == 0 ? 1 : i % 3 == 1 ? 3 : 2));
    }
    pairStep(s, 0);
    return s;
}
// Prescribed replay with absolute positions, so it never drifts.
// `pairs`: the first `moved` second-side objects republish each step. Toggled
// objects alternate between paired and lifted 2.25 above the scene, in two
// interleaved phases so every step has both entering and exiting pairs; the
// rest shift 0.25 along x and stay paired. The first side, on its cadence,
// shifts 0.1 along y and keeps its pairs.
// Cell workloads: the first `moved` bodies (whole cells first) republish every
// step. Every fourth lifts 6 m out of its cell or returns, in two interleaved
// phases; the rest shift 0.02 m along x and keep their neighbours.
void pairStep(PairScene& s, size_t step)
{
    if (s.topology != PairTopology::sides)
    {
        const size_t wide = cellsWide(s);
        for (size_t i = 0; i < s.moved; ++i)
        {
            auto& ob = cellObject(s, i);
            ob.p     = cellPosition(i, s.cellBodies, wide);
            if (i % 4 == 0)
                ob.p[2] += (step + i / 4) % 2 ? liftHeight : 0;
            else
                ob.p[0] += step % 2 ? -shift : shift;
        }
        return;
    }
    const size_t width = gridWidth(s.second.objects.size());
    for (size_t i = 0; i < s.moved; ++i)
    {
        auto& ob = s.second.objects[i];
        ob.p     = basePosition(i, width, true);
        if (i < s.toggled)
            ob.p[2] = (step + i) % 2 ? 2.75 : .5;
        else
            ob.p[0] += step % 2 ? -.25 : .25;
    }
    if (s.firstPublishes(step))
        for (size_t i = 0; i < s.moved; ++i)
        {
            auto& ob = s.first.objects[i];
            ob.p     = basePosition(i, width, false);
            ob.p[1] += (step / s.firstCadence) % 2 ? -.1 : .1;
        }
}
PairSet pairOracle(const PairScene& s)
{
    return sweepPairs(oracleEntries(s), s.topology == PairTopology::self, s.tolerance);
}
int pairsRun(const Options& o)
{
    PairScene scene = pairScene(o);
    if (o.backend == "parity")
    {
        pairsParity(o, scene);
        return 0;
    }
    const size_t                     initialPairs   = pairOracle(scene).size();
    const size_t                     capacity       = pairCapacity(o, scene, initialPairs);
    const uint64_t                   steadyDelta    = 2 * scene.toggled + 8;
    const auto                       memoryBaseline = memorySnapshot(OVSPATIALQUERY_INVALID_INSTANCE);
    const auto                       setupStart     = Clock::now();
    std::unique_ptr<DirectPairs>     direct;
    std::unique_ptr<BroadphasePairs> bp;
    std::unique_ptr<InstalledPairs>  installed;
    if (o.backend == "physx")
        direct = std::make_unique<DirectPairs>(scene);
    else if (o.backend == "abp")
        bp = std::make_unique<BroadphasePairs>(scene);
    else
        installed = std::make_unique<InstalledPairs>(scene, o.backend == "reporter-cuda" || o.backend == "reporter-gpu",
                                                     o.backend == "reporter-gpu");
    const double setupMs     = elapsed(setupStart);
    const auto   instance    = installed ? installed->spatial.instance : OVSPATIALQUERY_INVALID_INSTANCE;
    const auto   memoryReady = memorySnapshot(instance);
    // First use: the cold tree-pair query, the first broadphase update, the cold
    // snapshot, or reporter creation with the capacity outcomes a coordinator
    // meets before its first accepted baseline.
    std::vector<std::pair<std::string, double>> firstUse;
    PairSet                                     physxPairs, previous, entered, exited;
    double                                      checksum      = 0;
    const auto                                  firstUseStart = Clock::now();
    if (direct)
    {
        direct->find(physxPairs);
        checksum += consume(physxPairs);
        previous = physxPairs;
        firstUse.emplace_back("tree_pair_query", elapsed(firstUseStart));
    }
    else if (bp)
    {
        bp->broadphaseUpdate();
        bp->reduce(entered, exited);
        checksum += consume(entered);
        firstUse.emplace_back("broadphase_update", elapsed(firstUseStart));
    }
    else if (o.backend == "snapshot")
    {
        const auto& r = installed->snapshot(capacity);
        require(!r.truncated, "first snapshot truncated at pair capacity " + std::to_string(capacity));
        checksum += consume(installed->rows.data(), r.stored_pair_count);
        firstUse.emplace_back("snapshot", elapsed(firstUseStart));
    }
    else
    {
        installed->createReporter(
            std::max<uint64_t>(1, scene.topology == PairTopology::sides ? o.objects / 2 : initialPairs / 2),
            steadyDelta);
        for (size_t attempt = 0;; ++attempt)
        {
            require(attempt < 4, "reporter did not advance within three capacity retries");
            const auto start = Clock::now();
            auto       a     = installed->advance();
            installed->mirror(a);
            checksum += consume(a.result.entered_pairs) + consume(a.result.exited_pairs);
            const auto outcome = a.result.outcome;
            installed->release(a);
            firstUse.emplace_back(outcomeName(outcome), elapsed(start));
            if (outcome == OVSPATIALQUERY_PAIR_REPORTER_ADVANCED)
                break;
            const bool active = outcome == OVSPATIALQUERY_PAIR_REPORTER_ACTIVE_CAPACITY;
            installed->grow(active ? capacity : installed->activeCapacity,
                            active ? installed->deltaCapacity : capacity);
        }
    }
    const double firstUseCycleMs = elapsed(firstUseStart);
    const auto   memoryFirstUse  = memorySnapshot(instance);
    struct PairSample
    {
        double update = 0, query = 0, reduction = 0, observation = 0;
        size_t pairs = 0, entered = 0, exited = 0, candidates = 0;
    };
    std::vector<PairSample> samples;
    for (size_t step = 1; step <= o.frames + o.warmup; ++step)
    {
        PairSample s;
        pairStep(scene, step);
        if (installed)
            installed->produce(step);
        auto start = Clock::now();
        if (direct)
            direct->update(step);
        else if (bp)
            bp->update(step);
        else
            installed->update(step);
        s.update = elapsed(start);
        if (direct)
        {
            start = Clock::now();
            direct->find(physxPairs);
            s.query = elapsed(start);
            start   = Clock::now();
            entered.clear();
            exited.clear();
            std::set_difference(physxPairs.begin(), physxPairs.end(), previous.begin(), previous.end(),
                                std::back_inserter(entered));
            std::set_difference(previous.begin(), previous.end(), physxPairs.begin(), physxPairs.end(),
                                std::back_inserter(exited));
            s.reduction = elapsed(start);
            start       = Clock::now();
            checksum += consume(entered) + consume(exited);
            s.observation = elapsed(start);
            s.pairs       = physxPairs.size();
            s.entered     = entered.size();
            s.exited      = exited.size();
            s.candidates  = direct->reports;
            previous.swap(physxPairs);
        }
        else if (bp)
        {
            start = Clock::now();
            bp->broadphaseUpdate();
            s.query = elapsed(start);
            start   = Clock::now();
            bp->reduce(entered, exited);
            s.reduction = elapsed(start);
            start       = Clock::now();
            checksum += consume(entered) + consume(exited);
            s.observation = elapsed(start);
            s.pairs       = bp->pairs;
            s.entered     = entered.size();
            s.exited      = exited.size();
            s.candidates  = bp->candidates;
        }
        else if (o.backend == "snapshot")
        {
            start         = Clock::now();
            const auto& r = installed->snapshot(capacity);
            s.query       = elapsed(start);
            require(!r.truncated, "snapshot truncated at pair capacity " + std::to_string(capacity));
            start = Clock::now();
            checksum += consume(installed->rows.data(), r.stored_pair_count);
            s.observation = elapsed(start);
            s.pairs       = r.stored_pair_count;
        }
        else
        {
            start   = Clock::now();
            auto a  = installed->advance();
            s.query = elapsed(start);
            require(a.result.outcome == OVSPATIALQUERY_PAIR_REPORTER_ADVANCED,
                    "steady reporter advance reported a capacity outcome");
            start = Clock::now();
            installed->mirror(a);
            checksum += consume(a.result.entered_pairs) + consume(a.result.exited_pairs);
            s.entered = size_t(a.result.entered_pairs.shape[0]);
            s.exited  = size_t(a.result.exited_pairs.shape[0]);
            s.pairs   = a.result.accepted_active_pair_count;
            installed->release(a);
            s.observation = elapsed(start);
        }
        if (step > o.warmup)
            samples.push_back(s);
    }
    const auto memorySteady = memorySnapshot(instance);
    uint32_t   major = 0, minor = 0, patch = 0;
    check(ovspatialquery_get_version(&major, &minor, &patch), "runtime version");
    std::cout << std::setprecision(12) << "{\"backend\":\"" << o.backend << "\",\"workload\":\"" << o.workload
              << "\",\"topology\":\"" << topologyName(scene.topology) << "\",\"objects\":" << o.objects
              << ",\"objects_per_side\":" << (scene.topology == PairTopology::sides ? o.objects : 0)
              << ",\"cell_bodies\":" << scene.cellBodies << ",\"queries\":" << o.queries << ",\"frames\":" << o.frames
              << ",\"warmup\":" << o.warmup << ",\"update_percent\":" << o.updatePercent
              << ",\"moved_per_step\":" << scene.moved << ",\"toggled_per_step\":" << scene.toggled
              << ",\"first_cadence\":" << scene.firstCadence << ",\"contact_tolerance\":" << scene.tolerance
              << ",\"initial_pairs\":" << initialPairs << ",\"pair_capacity\":" << capacity
              << ",\"steady_delta_capacity\":" << steadyDelta
              << ",\"reporter_active_capacity\":" << (installed ? installed->activeCapacity : 0)
              << ",\"reporter_delta_capacity\":" << (installed ? installed->deltaCapacity : 0)
              << ",\"result_capacity_retries\":" << (installed ? installed->resultCapacityRetries : 0)
              << ",\"synthetic_analogue\":true,\"physx_version\":" << PX_PHYSICS_VERSION << ",\"physx_config\":\""
              << BENCH_PHYSX_CONFIG << "\",\"spatial_version\":[" << major << ',' << minor << ',' << patch
              << "],\"host_budget_bytes\":" << hostBudget << ",\"query_threads\":1,\"setup_ms\":" << setupMs
              << ",\"publication_ms\":"
              << (direct ? direct->publicationMs
                  : bp   ? bp->publicationMs
                         : installed->spatial.publicationMs)
              << ",\"first_use_cycle_ms\":" << firstUseCycleMs << ",\"first_use\":[";
    for (size_t i = 0; i < firstUse.size(); ++i)
        std::cout << (i ? "," : "") << "{\"phase\":\"" << firstUse[i].first << "\",\"ms\":" << firstUse[i].second
                  << '}';
    std::cout << "],\"query_scope\":\""
              << (direct ? "PxFindOverlap over both populations' trees, identity mapping and sort; "
                           "reduction_ms is the sorted set difference against the previous step"
                  : bp   ? "PxAABBManager update and fetch; update_ms packs changed bounds; reduction_ms maintains "
                           "candidates, filters them exactly and maps entered/exited identities"
                  : o.backend == "snapshot" ? "ovspatialquery_query_pairs_sync at both heads into the caller buffer"
                  : o.backend == "reporter-cuda"
                      ? "enqueue, wait and fetch of one CUDA reporter advance at both heads; observation_ms copies "
                        "the delta rows to the host after result_ready and retires the operation after that copy"
                  : o.backend == "reporter-gpu"
                      ? "update_ms publishes changed rows from CUDA arrays filled before timing (enqueue through "
                        "completion); then enqueue, wait and fetch of one CUDA reporter advance at both heads; "
                        "observation_ms copies the delta rows to the host after result_ready and retires the operation"
                      : "enqueue, wait and fetch of one reporter advance at both heads; observation_ms includes "
                        "operation destruction")
              << "\",\"memory\":";
    printMemoryRecord(
        installed != nullptr, nullptr, memoryBaseline,
        {{"ready", memoryReady, true}, {"first_use_ready", memoryFirstUse, true}, {"steady", memorySteady, true}});
    std::cout << ",\"checksum\":" << checksum << ",\"samples\":[";
    for (size_t i = 0; i < samples.size(); ++i)
    {
        const auto& s = samples[i];
        std::cout << (i ? "," : "") << "{\"update_ms\":" << s.update << ",\"query_ms\":" << s.query
                  << ",\"reduction_ms\":" << s.reduction << ",\"observation_ms\":" << s.observation
                  << ",\"frame_ms\":" << s.update + s.query + s.reduction + s.observation
                  << ",\"pair_count\":" << s.pairs << ",\"entered_count\":" << s.entered
                  << ",\"exited_count\":" << s.exited << ",\"candidate_count\":" << s.candidates << '}';
    }
    std::cout << "]}\n";
    return 0;
}
} // namespace bench
