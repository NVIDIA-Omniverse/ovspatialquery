/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Actor fixture selection.
#include "actor.h"

#include "mesh.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace bench
{
namespace
{
bool frameWorkload(const Options& o) { return o.workload.rfind("actor-frame", 0) == 0; }
bool avoidanceWorkload(const Options& o) { return o.workload.find("avoidance") == 0 || frameWorkload(o); }
// Participation bits of the actor-frame scene. The animation system publishes none; `actor-frame-bits`
// gives each avoidance row the dynamic bit so the ground and table cannot answer it.
constexpr uint64_t kStaticBits = 1, kDynamicBits = 2;
constexpr double   kTableHeight = 1;
// Each stage's input fixture carries the scene for the parity lane's mesh and identity checks.
Fixture withScene(const Fixture& input, const Fixture& scene)
{
    Fixture f  = input;
    f.objects  = scene.objects;
    f.convexes = scene.convexes;
    return f;
}
} // namespace
double ActorCycle::queryGroup(Group& group, ovspatialquery_ordinal_t ordinal,
                              std::unique_ptr<Spatial::BatchPlan>* shared)
{
    if (group.input.queries.empty())
        return 0;
    auto&      plan  = shared ? *shared : group.plan;
    const auto start = Clock::now();
    group.results.resize(group.input.queries.size());
    if (direct)
        direct->query(group.input, group.results);
    else if (options.backend == "scalar")
        spatial->scalar(group.input, group.results, ordinal);
    else
    {
        if (!options.reusePlans)
            plan.reset();
        if (!plan || plan->capacity < group.input.queries.size())
            plan = std::make_unique<Spatial::BatchPlan>(*spatial, group.input, group.input.queries.size());
        else
            plan->pack(group.input);
        bool truncated = false;
        plan->batch(group.input, group.results, ordinal, &truncated);
        while (truncated)
        {
            require(group.input.zoneHitCapacity && group.input.zoneHitCapacity < population,
                    "full population overlap capacity truncated");
            group.input.zoneHitCapacity = std::min(population, group.input.zoneHitCapacity * 2);
            plan.reset();
            plan = std::make_unique<Spatial::BatchPlan>(*spatial, group.input, group.input.queries.size());
            plan->batch(group.input, group.results, ordinal, &truncated);
            ++retries;
        }
    }
    return elapsed(start);
}
uint64_t ActorCycle::body(const Fixture& f, uint64_t id)
{
    const auto& object = f.objects.at(size_t(id - 1));
    return object.owner ? object.owner : object.id;
}
ActorCycle::ActorCycle(const Options& o, Direct* d, Spatial* s)
    : options(o), direct(d), spatial(s), previous(o.queries), next(o.queries), bestTime(o.queries),
      decisions(o.queries), ground(o.queries), population(o.objects)
{
    // Body keys are collider pairs; the actor-frame scene's two static rows key by their own IDs.
    const size_t moving = (o.objects + 1) / 2;
    bodies.resize(frameWorkload(o) ? o.objects + 3 : moving + 1);
    for (size_t key = 1; key < bodies.size(); ++key)
    {
        if (key > moving)
            continue;
        bodies[key].velocity = {key % 7 == 0 ? .02 : 2.0, 0, 0};
        if (key % 3 != 0)
            bodies[key].mass = key % 5 == 0 ? .1 : 10;
    }
}
void ActorCycle::prepare(Fixture& f, size_t frame)
{
    if (options.workload == "grounding-mesh")
    {
        meshQueries(f, frame);
        return;
    }
    if (frameWorkload(options))
    {
        prepareFrame(f, frame);
        return;
    }
    if (!avoidanceWorkload(options))
    {
        for (size_t i = 0; i < f.queries.size(); ++i)
        {
            auto&       q      = f.queries[i];
            const auto& object = f.objects[((i / 4) * 37) % f.objects.size()];
            q.p                = object.p;
            q.p[0] += (i % 2 ? .20 : -.20) + .015 * std::sin(double(frame) * .3);
            q.p[1] += (i % 4 < 2 ? -.12 : .12);
            if ((i / 4) % 7 == 6)
                q.p[0] += 1.1; // A foot pair over a gap.
            q.p[2]         = object.p[2] + .9;
            q.nearDistance = 0;
            q.farDistance  = options.workload == "support" ? 2 : .8;
            if (options.workload == "support")
            {
                q.p[2] = object.p[2] + boxHalf[2] + .1;
                if (frame % 3 == 2)
                    q.p[0] += 1.1;
            }
        }
        return;
    }
    // The animation system submits every actor's sphere in one launch with its own radius (per-row radii).
    const size_t groups = options.radiusGroups ? options.radiusGroups : (options.workload == "avoidance-mixed" ? 4 : 1);
    overlaps.input.queries.clear();
    overlaps.rows.clear();
    overlaps.input.kind     = QueryKind::sphereOverlap;
    overlaps.input.rowRadii = true;
    overlaps.input.even     = f.even;
    if (!overlaps.plan)
        overlaps.input.zoneHitCapacity = std::min(options.objects, options.actorHitCapacity);
    for (size_t i = 0; i < f.queries.size(); ++i)
    {
        auto&       q      = f.queries[i];
        const auto& object = f.objects[(2 * ((i * 7 + frame * 13) % ((f.objects.size() + 1) / 2))) % f.objects.size()];
        q.p                = object.p;
        q.p[0] += 1.3;
        q.p[1] -= .15;
        q.radius       = 1.6 + .125 * double(i % groups);
        q.nearDistance = 0;
        overlaps.input.queries.push_back(q);
        overlaps.rows.push_back(i);
    }
}
// One actor step for every actor (animation-system task updates and the pose phase):
// a walking actor contributes one avoidance sphere at its root and four foot probes; a seated actor
// one Sit support ray. Seated actors are every eighth; props move between actors across frames.
void ActorCycle::prepareFrame(Fixture& f, size_t frame)
{
    const size_t groups = options.radiusGroups ? options.radiusGroups : 4;
    const size_t props = f.objects.size() - 2, pairs = (props + 1) / 2;
    const auto&  table = f.objects.back();
    population         = f.objects.size();
    for (auto* group : {&overlaps, &sweeps, &feet, &sits})
    {
        group->input.queries.clear();
        group->rows.clear();
        group->input.even = f.even;
        if (group->input.meshes.empty())
            group->input.meshes = f.meshes; // Source faces are requested and checked on the dense table.
    }
    overlaps.input.kind     = QueryKind::sphereOverlap;
    overlaps.input.rowRadii = true;
    if (!overlaps.plan)
        overlaps.input.zoneHitCapacity = std::min(population, options.actorHitCapacity);
    feet.input.kind = sits.input.kind = QueryKind::rays;
    for (size_t i = 0; i < f.queries.size(); ++i)
    {
        auto& q = f.queries[i];
        q       = Query{};
        if (i % 8 == 7)
        {
            // Bounds top + 10 cm, two metres down, at interior points of the table's cells.
            Query s;
            s.p            = {table.p[0] - 3.483 + double((i * 17 + frame) % 61) / 10,
                              table.p[1] - 3.487 + double((i * 29) % 59) / 10, kTableHeight + .375 + .1};
            s.nearDistance = 0;
            s.farDistance  = 2;
            sits.input.queries.push_back(s);
            sits.rows.push_back(i);
            continue;
        }
        const auto& prop = f.objects[(2 * ((i * 7 + frame * 13) % pairs)) % props];
        q.p              = {prop.p[0] + 1.3, prop.p[1] - .15, 0}; // The actor root stands on the ground.
        q.radius         = 1.6 + .125 * double(i % groups);       // Authored avoidance trigger radius.
        q.nearDistance   = 0;
        q.bits           = options.workload == "actor-frame-bits" ? kDynamicBits : 0;
        overlaps.input.queries.push_back(q);
        overlaps.rows.push_back(i);
        for (size_t k = 0; k < 4; ++k)
        {
            // Heel and toe of each foot: root up + 30 cm, 80 cm down.
            Query ray;
            ray.p = {q.p[0] + (k % 2 ? .12 : -.12) + .015 * std::sin(double(frame) * .3), q.p[1] + (k < 2 ? -.1 : .1),
                     .3};
            ray.nearDistance = 0;
            ray.farDistance  = .8;
            feet.input.queries.push_back(ray);
        }
    }
}
void ActorCycle::execute(Fixture& f, bool record)
{
    trace.clear();
    overlapMs          = 0;
    sweepMs            = 0;
    reductionMs        = 0;
    rayMs              = 0;
    overlapHits        = 0;
    sweepCount         = 0;
    decisionCount      = 0;
    retries            = 0;
    const auto ordinal = spatial ? spatial->ordinal : 0;
    if (!avoidanceWorkload(options))
    {
        const auto start = Clock::now();
        if (direct)
            direct->query(f, ground);
        else if (options.backend == "scalar")
            spatial->scalar(f, ground, ordinal);
        else
        {
            if (!options.reusePlans)
                groundPlan.reset();
            if (!groundPlan)
                groundPlan = std::make_unique<Spatial::BatchPlan>(*spatial, f, f.queries.size());
            else
                groundPlan->pack(f);
            groundPlan->batch(f, ground, ordinal);
        }
        overlapMs         = elapsed(start);
        const auto reduce = Clock::now();
        reduceGround(ground, (options.workload == "grounding" || options.workload == "grounding-mesh") ? 2 : 1);
        reductionMs = elapsed(reduce);
        if (record)
            trace.push_back({f, ground});
        return;
    }
    candidates.clear();
    sweeps.input.queries.clear();
    sweeps.rows.clear();
    sweeps.input.kind     = QueryKind::sweep;
    sweeps.input.rowRadii = true;
    sweeps.input.even     = f.even;
    std::fill(bestTime.begin(), bestTime.end(), std::numeric_limits<double>::infinity());
    std::fill(decisions.begin(), decisions.end(), 0);
    overlapMs += queryGroup(overlaps, ordinal);
    auto reduce = Clock::now();
    {
        auto& group = overlaps;
        if (record)
            trace.push_back({withScene(group.input, f), group.results});
        for (size_t row = 0; row < group.rows.size(); ++row)
        {
            const size_t actor   = group.rows[row];
            const auto   present = owners(f, group.results[row]);
            auto&        latched = next[actor];
            latched.clear();
            for (auto key : previous[actor])
                if (std::binary_search(present.begin(), present.end(), key))
                    latched.push_back(key);
            overlapHits += group.results[row].hits.size();
            for (uint64_t key : present)
            {
                // Deterministic caller-owned metadata, identical for both adapters.
                const auto&  metadata = bodies.at(size_t(key));
                const double speed    = std::sqrt(metadata.velocity[0] * metadata.velocity[0] +
                                                  metadata.velocity[1] * metadata.velocity[1] +
                                                  metadata.velocity[2] * metadata.velocity[2]);
                if (key % 11 == 0 || std::find(latched.begin(), latched.end(), key) != latched.end() || speed < .05 ||
                    (metadata.mass && *metadata.mass < 1))
                    continue;
                Query q       = f.queries[actor];
                q.farDistance = q.radius;
                for (size_t k = 0; k < 3; ++k)
                    q.direction[k] = float(-metadata.velocity[k] / speed);
                // The animation system sweeps navigation radius + 0.25 m; the frame scene takes a 0.3 m agent.
                const double clearance = (frameWorkload(options) ? .55 : .35) +
                                         .0002 * double(actor % (options.radiusGroups ? options.radiusGroups
                                                                 : options.workload == "avoidance" ? 1
                                                                                                   : 4));
                q.radius = clearance;
                sweeps.input.queries.push_back(q);
                sweeps.rows.push_back(candidates.size());
                candidates.push_back({actor, key, speed});
            }
        }
    }
    reductionMs += elapsed(reduce);
    sweepMs += queryGroup(sweeps, ordinal);
    reduce = Clock::now();
    if (!sweeps.input.queries.empty())
    {
        auto& group = sweeps;
        if (record)
            trace.push_back({withScene(group.input, f), group.results});
        for (size_t row = 0; row < group.rows.size(); ++row)
        {
            ++sweepCount;
            const auto& candidate = candidates[group.rows[row]];
            const auto& hits      = group.results[row].hits;
            if (hits.empty())
                continue;
            const auto& hit = hits[0];
            checksum += hit.distance + double(hit.id);
            if (body(f, hit.id) != candidate.body || hit.distance <= 1e-5)
                continue;
            const double t = hit.distance / candidate.speed;
            if (t <= .5 && t < bestTime[candidate.actor])
            {
                bestTime[candidate.actor]  = t;
                decisions[candidate.actor] = candidate.body;
            }
        }
    }
    for (size_t i = 0; i < decisions.size(); ++i)
        if (decisions[i])
        {
            next[i].push_back(decisions[i]);
            ++decisionCount;
            checksum += double(decisions[i]);
        }
    checksum += double(overlapHits + sweepCount);
    reductionMs += elapsed(reduce);
    if (!frameWorkload(options))
        return;
    // Sit support flushes each seated actor's ray alone through the shared ray plan, then the
    // pose phase flushes every walking actor's foot probes in one launch.
    for (size_t row = 0; row < sits.rows.size(); ++row)
    {
        sit.input = sits.input;
        sit.input.queries.assign(1, sits.input.queries[row]);
        rayMs += queryGroup(sit, ordinal, &feet.plan);
        if (record)
            trace.push_back({withScene(sit.input, f), sit.results});
        reduce = Clock::now();
        if (!sit.results[0].hits.empty())
            checksum += sit.results[0].hits[0].position[2];
        reductionMs += elapsed(reduce);
    }
    rayMs += queryGroup(feet, ordinal);
    if (record)
        trace.push_back({withScene(feet.input, f), feet.results});
    reduce = Clock::now();
    reduceGround(feet.results, 2);
    reductionMs += elapsed(reduce);
}
// Heel and toe pairs: the higher contact and the normalized mean normal, rejecting initial overlaps.
void ActorCycle::reduceGround(const Results& rows, size_t stride)
{
    for (size_t i = 0; i < rows.size(); i += stride)
    {
        double height = -1e6;
        Vec    normal{};
        size_t hits = 0;
        for (size_t j = i; j < std::min(i + stride, rows.size()); ++j)
            if (!rows[j].hits.empty() && rows[j].hits[0].distance > .0001)
            {
                const auto& h = rows[j].hits[0];
                height        = std::max(height, h.position[2]);
                ++hits;
                for (size_t k = 0; k < 3; ++k)
                    normal[k] += h.normal[k];
                checksum += h.distance + h.position[0] + h.position[1];
            }
        if (hits)
        {
            const double length = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
            checksum += height;
            for (double n : normal)
                checksum += length > 1e-8 ? n / length : 0;
        }
    }
}
void ActorCycle::commitFrame() { previous = next; }
void ActorCycle::compareWith(const ActorCycle& other) const
{
    require(trace.size() == other.trace.size(), "actor stage/group count differs");
    // This cycle is the direct PhysX baseline (main.cpp); the other is an installed path.
    for (size_t i = 0; i < trace.size(); ++i)
        compare(trace[i].first, trace[i].second, other.trace[i].second,
                "actor " + other.options.backend + " stage " + std::to_string(i), false, true, nullptr, true);
    require(decisions == other.decisions && next == other.next, "actor decisions/latches differ");
    require(near(checksum, other.checksum), "actor observation checksum differs");
}
bool actorWorkload(const Options& o)
{
    return o.workload == "grounding" || o.workload == "grounding-mesh" || o.workload == "support" ||
           avoidanceWorkload(o);
}
// A pickup-like animation scene: `objects` convex props, two colliders per rigid body, resting on a
// two-triangle ground quad (finite ground), and one dense static mesh raised as a table.
// Props come first so pose updates move them; at 100% every row is republished, as an animation system does.
Fixture actorFrameFixture(const Options& o)
{
    Fixture f;
    f.kind      = QueryKind::sphereOverlap;
    f.even      = o.subset == "even";
    f.posesOnly = true; // The animation-system publisher sends POSES rows for every collider each frame.
    f.convexes.emplace_back();
    for (float z : {-.2f, .2f})
        for (float y : {-.25f, .25f})
            for (float x : {-.3f, .3f})
                f.convexes[0].push_back({x, y, z});
    f.convexes.emplace_back();
    for (float z : {-.2f, .2f})
        for (size_t side = 0; side < 16; ++side)
        {
            const double angle = double(side) * std::acos(-1.0) / 8;
            f.convexes[1].push_back({float(.25 * std::cos(angle)), float(.25 * std::sin(angle)), z});
        }
    const size_t pairs = (o.objects + 1) / 2, width = size_t(std::ceil(std::sqrt(double(pairs))));
    for (size_t i = 0; i < o.objects; ++i)
    {
        const size_t body = i / 2;
        f.objects.push_back({i + 1,
                             body + 1,
                             false,
                             {double(body % width) * 3.0, double(body / width) * 3.0 + (i % 2 ? .65 : 0), .2},
                             i % 4 == 3 ? rotation(.25f) : identity,
                             kDynamicBits,
                             0,
                             i % 2 + 1});
    }
    const double span = double(width) * 3.0 + 10.5;
    Mesh         quad;
    quad.vertices = {
        {-10.5f, -10.5f, 0}, {float(span), -10.5f, 0}, {float(span), float(span), 0}, {-10.5f, float(span), 0}};
    quad.triangles      = {{0, 1, 2}, {0, 2, 3}};
    f.meshes            = {quad, terrain(o.meshSide)};
    const double middle = double(width - 1) * 1.5;
    f.objects.push_back({o.objects + 1, o.objects + 1, false, {0, 0, 0}, identity, kStaticBits, 1});
    f.objects.push_back(
        {o.objects + 2, o.objects + 2, false, {middle, middle, kTableHeight}, identity, kStaticBits, 2});
    f.queries.resize(o.queries);
    return f;
}
Fixture actorFixture(const Options& o)
{
    if (frameWorkload(o))
        return actorFrameFixture(o);
    auto       f         = o.workload == "grounding-mesh" ? meshFixture(o) : fixture(o);
    const bool avoidance = o.workload.find("avoidance") == 0;
    f.kind               = avoidance ? QueryKind::sphereOverlap : QueryKind::rays;
    const size_t width   = size_t(std::ceil(std::sqrt(double((o.objects + 1) / 2))));
    if (avoidance)
        for (size_t i = 0; i < f.objects.size(); ++i)
        {
            auto&        ob   = f.objects[i];
            const size_t body = i / 2;
            ob.p              = {double(body % width) * 3.0, double(body / width) * 3.0 + (i % 2 ? .65 : 0), .5};
            // A single rigid body owns two separately registered colliders.
            ob.owner = body + 1;
            ob.q     = identity;
        }
    return f;
}
} // namespace bench
