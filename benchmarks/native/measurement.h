/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Process and library memory snapshots, timing samples and their JSON printing.
#ifndef BENCH_MEASUREMENT_H
#define BENCH_MEASUREMENT_H
#include "common.h"

#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <optional>

namespace bench
{
struct ProcessMemory
{
    std::optional<uint64_t> rssBytes;
    std::optional<uint64_t> privateBytes;
    std::optional<uint64_t> pssBytes;
};
struct MemorySnapshot
{
    ProcessMemory           process;
    std::optional<uint64_t> ovspatialqueryHostChargedBytes;
    // Bytes charged to CUDA device 0's budget, where the instance has one.
    std::optional<uint64_t> ovspatialqueryDeviceChargedBytes;
};
struct MemoryDelta
{
    std::optional<int64_t> rssBytes;
    std::optional<int64_t> privateBytes;
    std::optional<int64_t> pssBytes;
    std::optional<int64_t> ovspatialqueryHostChargedBytes;
};

MemoryDelta    memoryDelta(const MemorySnapshot& snapshot, const MemorySnapshot& baseline);
MemorySnapshot memorySnapshot(ovspatialquery_instance_handle_t instance);
// One named phase of a memory record; a phase the run did not reach prints null.
struct MemoryPhase
{
    const char*                   name;
    std::optional<MemorySnapshot> snapshot;
    bool                          fromBaseline; // Also listed under deltas_from_baseline.
};
// Prints a workload's "memory" object: the process measurement source, the host budget (null unless the
// run has a budgeted instance), the process note where given, the baseline and the phases, their deltas
// from the baseline. fromBeforeFirstUse, when present, adds deltas_from_before_first_use.first_use_ready
// (null when empty). The whole-process high-water RSS follows the object as the record's peak_rss_bytes.
void printMemoryRecord(bool budgeted, const char* processNote, const MemorySnapshot& baseline,
                       std::initializer_list<MemoryPhase>        phases,
                       std::optional<std::optional<MemoryDelta>> fromBeforeFirstUse = std::nullopt);
template <class T>
void printOptionalValue(const std::optional<T>& value)
{
    if (value)
        std::cout << *value;
    else
        std::cout << "null";
}
struct Sample
{
    double update = 0, query = 0, observation = 0;
    double overlap = 0, sweep = 0, ray = 0, reduction = 0, pack = 0;
    double beamGeneration = 0, poseTransform = 0, completeFrame = 0;
    // CUDA plan launches only: admission call, completion wait, typed fetch, stream wait plus readback.
    double cudaEnqueue = 0, cudaWait = 0, cudaFetch = 0, cudaReadback = 0;
    size_t overlapHits = 0, sweeps = 0, decisions = 0, retries = 0;
};
} // namespace bench
#endif
