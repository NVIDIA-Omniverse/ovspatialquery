/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_CUDA_PAIR_TYPES_H
#define OVSPATIALQUERY_CUDA_PAIR_TYPES_H

// Records shared by the CUDA pair reporter's host code and its kernels
// (ADR-0019 stage 3), and the bounds predicate it shares with CPU pair
// materialization. Plain types only: NVRTC compiles this header without the
// standard library.

#if defined(__CUDACC__)
#define OVS_PAIR_HD __host__ __device__
#else
#define OVS_PAIR_HD
#endif

namespace ovspatialquery
{
namespace detail
{

// Whether two boxes overlap within the tolerance, on the CPU and the device
// alike. Only a positive separating gap is compared with the tolerance;
// subtraction may become +infinity for extreme finite endpoints, which is the
// correct overflow-safe result for a gap larger than finite tolerance. Plain
// subtractions and comparisons leave nothing for contraction to fuse.
inline OVS_PAIR_HD bool pairBoundsOverlap(const double* firstMinimum, const double* firstMaximum,
                                          const double* secondMinimum, const double* secondMaximum, double tolerance)
{
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        if ((firstMinimum[axis] > secondMaximum[axis] && firstMinimum[axis] - secondMaximum[axis] > tolerance) ||
            (secondMinimum[axis] > firstMaximum[axis] && secondMinimum[axis] - firstMaximum[axis] > tolerance))
        {
            return false;
        }
    }
    return true;
}

namespace cuda_pairs
{

// PairEndpointRelation, as the kernels see it.
enum : unsigned int
{
    kDistinctPopulations    = 0u,
    kDirectedSamePopulation = 1u,
    kIdenticalSelection     = 2u,
};

// ovspatialquery_pair_reporter_outcome_t values.
enum : unsigned int
{
    kOutcomeAdvanced       = 0u,
    kOutcomeActiveCapacity = 1u,
    kOutcomeDeltaCapacity  = 2u,
    kOutcomeFailed         = 3u,
};

constexpr unsigned int kThreads = 256u;
constexpr unsigned int kEmpty   = 0xffffffffu;
// A query that would visit more grid cells than this tests every object instead.
constexpr unsigned int kMaximumQueryCells = 4096u;
// Discovery gives each query object one warp.
constexpr unsigned int kWarpSize = 32u;

// One pair of an accepted or new set: the identities the caller sees and each
// object's position in its version's table, the slot its change records name.
struct Entry
{
    unsigned long long firstId;
    unsigned long long secondId;
    unsigned int       firstSlot;
    unsigned int       secondSlot;
};

// A uniform grid over one endpoint's admitted objects, rebuilt each advance.
// Objects are bucketed by the cell of their minimum corner; an object wider
// than a cell, outside the float range or beyond the cell index range is
// oversize and every query tests it.
struct Grid
{
    double       inverseCell;
    double       extentSum;
    unsigned int objectCount;
    unsigned int oversizeCount;
};

// Counters of one advance, one per reporter. `currentBuffer` names the ring
// buffer holding the accepted baseline and its count in the reporter's ring
// counts; only `finish` moves it, when an advance is accepted, so advances
// pending behind a capacity outcome or a failure compare with the preserved
// baseline (ADR-0022 decision 5) and a result the host does not commit never
// changes it. Everything from `nextCount` on is cleared per advance.
//
// Entered and exited pairs compare two sets by identity. A patched advance
// compares the accepted pairs with a changed object (set 0, by index into the
// baseline) with the new pairs discovered for changed objects (set 1, from
// `newStart`, after the carried pairs); a recomputation compares the whole
// baseline with the whole new set. `clear` sizes both identity indexes for
// their counts.
struct Control
{
    unsigned long long currentBuffer;
    unsigned long long nextCount;
    unsigned long long enteredCount;
    unsigned long long exitedCount;
    unsigned long long candidateCount;
    unsigned long long newStart;
    unsigned long long setCount[2];
    unsigned int       setMask[2];
    unsigned int       changedCount[2];
    Grid               grid[2];
};

// What an advance reports, written to the result slot and copied to the host
// before its ready event; the reservation's `counts` tensor is this record.
struct Header
{
    unsigned long long outcome;
    unsigned long long enteredCount;
    unsigned long long exitedCount;
    unsigned long long acceptedCount;
};

// The two sets an advance compares, each indexed by identity in `hash`: the
// baseline, the ring buffer `Control::currentBuffer` names (`ringStride`
// entries apart), through `indices` into it when patching; and the new set,
// `next`.
struct Sets
{
    const Entry*        next;
    const Entry*        ring;
    unsigned long long  ringStride;
    const unsigned int* indices;
    unsigned int*       hash[2];
};

// One object of a version's device table, as the host stages it.
struct ObjectRow
{
    unsigned long long id;
    unsigned long long bits[2];
    double             minimum[3];
    double             maximum[3];
};

// One endpoint of a version table: an object passes when its selection
// includes it and, unless the query bits are zero, its participation bits
// intersect them.
struct Side
{
    unsigned int              objectCount;
    const unsigned long long* ids;
    const unsigned long long* bits;
    const double*             minimums;
    const double*             maximums;
    const unsigned char*      membership;
    unsigned long long        queryBits[2];
};

// The arguments of one discovery pass.
struct Discovery
{
    Side                query;
    Side                build;
    const unsigned int* queryList;
    // A patched pass reads its deduplicated changed count here; `queryCount`
    // bounds it.
    const unsigned int* queryListCount;
    unsigned int        queryCount;
    unsigned int        relation;
    unsigned int        queryIsFirst;
    // Changed objects of the first endpoint's population, null when
    // recomputing: a patched pass over second objects skips pairs whose first
    // object changed, which the first pass found; an identical selection skips
    // a changed partner at a lower position, which found the pair itself.
    const unsigned int* firstChanged;
    double              tolerance;
    const Grid*         grid;
    const unsigned int* heads;
    const unsigned int* next;
    const int*          cellKeys;
    const unsigned int* oversize;
    unsigned int        bucketMask;
    Entry*              entries;
    unsigned long long  capacity;
    Control*            control;
};

} // namespace cuda_pairs
} // namespace detail
} // namespace ovspatialquery

#endif
