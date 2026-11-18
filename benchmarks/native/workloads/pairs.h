/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// co-simulation pair workload and engine-style cell populations: AABB pair snapshots and reporter advances.
#ifndef BENCH_WORKLOADS_PAIRS_H
#define BENCH_WORKLOADS_PAIRS_H
#include "../fixture.h"

#include <ovspatialquery/ovspatialquery_pairs.h>

#include <string>
#include <utility>
#include <vector>

namespace bench
{
// Sorted (first, second) object identities: the parity currency of every pair lane.
using PairSet = std::vector<std::pair<uint64_t, uint64_t>>;
// How the reporter's two targets select objects.
enum class PairTopology : uint8_t
{
    sides, // `pairs`: two populations on the twin grid, the first on a slower cadence.
    self,  // `pairs-self`: one population against itself; unordered pairs without self-pairs.
    views, // `pairs-views`: two partially overlapping explicit views of one population.
    split, // `pairs-split`: the cell bodies alternate between two populations (disjoint control).
};
// `first` stands for a slow-cadence scene population and `second` for robots
// republished every step (`pairs`), or `first` holds every cell body and
// `second` stays empty (self and views), or the bodies alternate between them
// (split). Synthetic analogues: no cosim source revision was available, and
// the cell layout only borrows the engine interaction's shape (packed boxes
// per cell, one self reporter, a contact margin), not its counts or motion.
struct PairScene
{
    Fixture              first, second;
    PairTopology         topology   = PairTopology::sides;
    size_t               cellBodies = 0;         // Bodies per cell; zero selects the twin grid of `pairs`.
    std::vector<uint8_t> roles;                  // Views: bit 0 first view, bit 1 second view, per first-side object.
    size_t               moved = 0, toggled = 0; // Per publishing step: shifted objects and the enter/exit subset.
    size_t               firstCadence = 4;       // The first side publishes every N steps.
    double               tolerance    = 0;
    bool                 firstPublishes(size_t step) const { return step % firstCadence == 0; }
    bool twoPopulations() const { return topology == PairTopology::sides || topology == PairTopology::split; }
    // Leading objects of each side republished on a publishing step.
    size_t firstMoved() const { return topology == PairTopology::split ? (moved + 1) / 2 : moved; }
    size_t secondMoved() const { return topology == PairTopology::split ? moved / 2 : moved; }
};
PairScene pairScene(const Options& o);
void      pairStep(PairScene& s, size_t step);
PairSet   pairOracle(const PairScene& s); // Double-precision sweep over both sides.
int       pairsRun(const Options& o);
// Sorted pairs of reporter rows or of an [n, 2] uint64 reporter delta tensor.
PairSet toSet(const ovspatialquery_pair_t* rows, size_t count);
PairSet toSet(const DLTensor& t);
// Fails with the first differing pairs of both sets so a parity break names its rows.
void requireEqual(const PairSet& expected, const PairSet& actual, const std::string& label);
} // namespace bench
#endif
