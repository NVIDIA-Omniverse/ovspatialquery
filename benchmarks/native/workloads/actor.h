/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Animation-system consumer replay: grounding, support and dependent avoidance over the shared adapters.
#ifndef BENCH_WORKLOADS_ACTOR_H
#define BENCH_WORKLOADS_ACTOR_H
#include "../installed_adapter.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace bench
{
bool    actorWorkload(const Options& o);
Fixture actorFixture(const Options& o);
// Declared here (defined in workloads/mesh.h/.cpp) so prepare() below can call it.
void meshQueries(Fixture& f, size_t frame);
// A caller-side replay of the query consumers. No animation runtime or dynamics are linked.
class ActorCycle
{
    struct Group
    {
        Fixture                             input;
        std::vector<size_t>                 rows;
        Results                             results;
        std::unique_ptr<Spatial::BatchPlan> plan;
    };
    struct Candidate
    {
        size_t   actor;
        uint64_t body;
        double   speed;
    };
    struct BodyState
    {
        Vec                   velocity;
        std::optional<double> mass;
    };
    std::vector<BodyState>              bodies;
    const Options&                      options;
    Direct*                             direct;
    Spatial*                            spatial;
    Group                               overlaps, sweeps, feet, sits, sit;
    std::vector<Candidate>              candidates;
    std::vector<std::vector<uint64_t>>  previous, next;
    std::vector<double>                 bestTime;
    std::vector<uint64_t>               decisions;
    Results                             ground;
    std::unique_ptr<Spatial::BatchPlan> groundPlan;
    // Rows past the population's size are never truncated; set from the scene in prepare.
    size_t          population;
    double          queryGroup(Group& group, ovspatialquery_ordinal_t ordinal,
                               std::unique_ptr<Spatial::BatchPlan>* shared = nullptr);
    void            prepareFrame(Fixture& f, size_t frame);
    void            reduceGround(const Results& rows, size_t stride);
    static uint64_t body(const Fixture& f, uint64_t id);

  public:
    double checksum = 0, overlapMs = 0, sweepMs = 0, reductionMs = 0, rayMs = 0;
    size_t overlapHits = 0, sweepCount = 0, decisionCount = 0, retries = 0;
    // Retained only in the untimed parity lane.
    std::vector<std::pair<Fixture, Results>> trace;
    ActorCycle(const Options& o, Direct* d, Spatial* s);
    void prepare(Fixture& f, size_t frame);
    void execute(Fixture& f, bool record = false);
    void commitFrame();
    void compareWith(const ActorCycle& other) const;
};
} // namespace bench
#endif
