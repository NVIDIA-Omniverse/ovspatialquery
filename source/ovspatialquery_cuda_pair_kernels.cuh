/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Kernels of the CUDA pair reporter (ADR-0019 stage 3), included by
// ovspatialquery_cuda_kernel.cu so they share its PTX module.
//
// One advance runs on its stream, behind the advance before it, after the
// host clears its counters: `mark` turns the change records of each endpoint population into
// a changed bitset and list; `grid_stats` and `grid_link` bucket one
// endpoint's admitted objects; `carry` keeps the accepted pairs whose objects
// did not change and lists the others; `discover` finds, with one warp per
// query object, the pairs of every object when recomputing and of changed
// objects when patching; `clear` and `index` build identity indexes of the
// two sets to compare; `difference` reports the pairs of each set missing from
// the other, entered and exited; `finish` decides the outcome as the CPU
// transition does. Every candidate passes pairBoundsOverlap, the
// subtraction-form double predicate CPU pair materialization also calls, so
// the device reports the pairs a CPU snapshot reports.

#include "ovspatialquery_cuda_pair_types.h"

namespace
{

namespace cudaPairs = ovspatialquery::detail::cuda_pairs;

__device__ bool pairAdmitted(const cudaPairs::Side& side, unsigned int index)
{
    return (side.membership == nullptr || side.membership[index] != 0u) &&
           ((side.queryBits[0] | side.queryBits[1]) == 0ull ||
            ((side.bits[2u * index] & side.queryBits[0]) | (side.bits[2u * index + 1u] & side.queryBits[1])) != 0ull);
}

__device__ bool pairBitSet(const unsigned int* bits, unsigned int index)
{
    return bits != nullptr && ((bits[index >> 5u] >> (index & 31u)) & 1u) != 0u;
}

// The float box around an object, rounded outward, and whether it is finite
// and inside the float range.
__device__ bool pairFloatBox(const double* minimum, const double* maximum, float lower[3], float upper[3])
{
    bool finite = true;
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        lower[axis] = __double2float_rd(minimum[axis]);
        upper[axis] = __double2float_ru(maximum[axis]);
        finite =
            finite && isfinite(lower[axis]) && isfinite(upper[axis]) && lower[axis] > -3.0e38f && upper[axis] < 3.0e38f;
    }
    return finite;
}

__device__ unsigned int pairHash(unsigned long long first, unsigned long long second, unsigned int mask)
{
    unsigned long long hash = first * 0x9E3779B97F4A7C15ull ^ (second + 0x632BE59BD9B4E019ull) * 0xC2B2AE3D27D4EB4Full;
    hash ^= hash >> 29u;
    return static_cast<unsigned int>(hash) & mask;
}

// The entry at a position of one compared set: set 0 is the baseline, in the
// ring buffer `currentBuffer` names and through its list of indices when
// patching; set 1 is the new set, after its carried pairs.
__device__ const cudaPairs::Entry& pairSetEntry(const cudaPairs::Sets& sets, const cudaPairs::Control* control,
                                                unsigned int set, unsigned long long position)
{
    if (set == 0u)
    {
        return sets.ring[control->currentBuffer * sets.ringStride +
                         (sets.indices != nullptr ? sets.indices[position] : position)];
    }
    return sets.next[control->newStart + position];
}

// Whether a set, indexed by its hash, holds the pair.
__device__ bool pairFind(const cudaPairs::Sets& sets, const cudaPairs::Control* control, unsigned int set,
                         unsigned long long first, unsigned long long second)
{
    const unsigned int mask = control->setMask[set];
    unsigned int       slot = pairHash(first, second, mask);
    for (unsigned int probe = 0u; probe <= mask; ++probe)
    {
        const unsigned int position = sets.hash[set][slot];
        if (position == cudaPairs::kEmpty)
        {
            return false;
        }
        const cudaPairs::Entry& entry = pairSetEntry(sets, control, set, position);
        if (entry.firstId == first && entry.secondId == second)
        {
            return true;
        }
        slot = (slot + 1u) & mask;
    }
    return false;
}

__device__ void pairAppend(cudaPairs::Control* control, cudaPairs::Entry* entries, unsigned long long capacity,
                           const cudaPairs::Entry& entry)
{
    const unsigned long long index = atomicAdd(&control->nextCount, 1ull);
    if (index < capacity)
    {
        entries[index] = entry;
    }
}

// The CPU's visitCandidate for one query object and one build object: the
// relation's self-pair and ordering rules, then the predicate. An identical
// selection queries and builds one side; its pair is ordered by ID.
__device__ void pairConsider(const cudaPairs::Discovery& d, unsigned int query, unsigned int build)
{
    const bool             identical  = d.relation == cudaPairs::kIdenticalSelection;
    const cudaPairs::Side& first      = d.queryIsFirst != 0u ? d.query : d.build;
    const cudaPairs::Side& second     = d.queryIsFirst != 0u ? d.build : d.query;
    const unsigned int     firstSlot  = d.queryIsFirst != 0u ? query : build;
    const unsigned int     secondSlot = d.queryIsFirst != 0u ? build : query;
    if ((identical &&
         (d.firstChanged == nullptr ? build <= query
                                    : build == query || (pairBitSet(d.firstChanged, build) && build < query))) ||
        (d.relation == cudaPairs::kDirectedSamePopulation && firstSlot == secondSlot) ||
        (d.queryIsFirst == 0u && pairBitSet(d.firstChanged, firstSlot)) ||
        !ovspatialquery::detail::pairBoundsOverlap(first.minimums + 3u * firstSlot, first.maximums + 3u * firstSlot,
                                                   second.minimums + 3u * secondSlot, second.maximums + 3u * secondSlot,
                                                   d.tolerance))
    {
        return;
    }
    cudaPairs::Entry entry{first.ids[firstSlot], second.ids[secondSlot], firstSlot, secondSlot};
    if (identical && entry.secondId < entry.firstId)
    {
        entry = {entry.secondId, entry.firstId, secondSlot, firstSlot};
    }
    pairAppend(d.control, d.entries, d.capacity, entry);
}

// Finds the pairs of one query object with its warp: the lanes share the
// grid cells the query box covers, the oversize objects, or every object
// when the box covers too many cells.
__device__ void pairDiscoverWarp(const cudaPairs::Discovery& d, unsigned int query, unsigned int lane)
{
    if (query >= d.query.objectCount || !pairAdmitted(d.query, query))
    {
        return;
    }
    // The query box, inflated by the tolerance and by a slack of a few double
    // steps of the largest operand, which covers the rounding of the double
    // predicate's subtraction, then rounded outward to float: the cells it
    // covers hold every object the double predicate can accept.
    const double* minimum = d.query.minimums + 3u * query;
    const double* maximum = d.query.maximums + 3u * query;
    double        inflatedMinimum[3];
    double        inflatedMaximum[3];
    for (unsigned int axis = 0u; axis < 3u; ++axis)
    {
        const double slack    = (fabs(minimum[axis]) + fabs(maximum[axis]) + d.tolerance) * 0x1p-50;
        inflatedMinimum[axis] = minimum[axis] - d.tolerance - slack;
        inflatedMaximum[axis] = maximum[axis] + d.tolerance + slack;
    }
    float lower[3];
    float upper[3];
    bool  gridded = pairFloatBox(inflatedMinimum, inflatedMaximum, lower, upper);
    int   low[3]  = {0, 0, 0};
    int   high[3] = {0, 0, 0};
    for (unsigned int axis = 0u; gridded && axis < 3u; ++axis)
    {
        lower[axis] = nextafterf(lower[axis], -3.4e38f);
        upper[axis] = nextafterf(upper[axis], 3.4e38f);
        gridded     = boxGridCell(lower[axis], d.grid->inverseCell, low[axis]) &&
                  boxGridCell(upper[axis], d.grid->inverseCell, high[axis]);
        // An object of the grid overlaps the box only if its minimum corner
        // lies at most one cell below it.
        low[axis] -= 1;
    }
    unsigned long long cells = 1ull;
    for (unsigned int axis = 0u; gridded && axis < 3u; ++axis)
    {
        const long long          span    = static_cast<long long>(high[axis]) - static_cast<long long>(low[axis]) + 1ll;
        const unsigned long long product = cells * static_cast<unsigned long long>(span);
        cells = product > cudaPairs::kMaximumQueryCells ? cudaPairs::kMaximumQueryCells + 1ull : product;
    }
    if (!gridded || cells > cudaPairs::kMaximumQueryCells)
    {
        for (unsigned int build = lane; build < d.build.objectCount; build += cudaPairs::kWarpSize)
        {
            if (pairAdmitted(d.build, build))
            {
                pairConsider(d, query, build);
            }
        }
        return;
    }
    const unsigned int spanX = static_cast<unsigned int>(high[0] - low[0] + 1);
    const unsigned int spanY = static_cast<unsigned int>(high[1] - low[1] + 1);
    for (unsigned int cell = lane; cell < cells; cell += cudaPairs::kWarpSize)
    {
        const int x = low[0] + static_cast<int>(cell % spanX);
        const int y = low[1] + static_cast<int>((cell / spanX) % spanY);
        const int z = low[2] + static_cast<int>(cell / (spanX * spanY));
        for (unsigned int build = d.heads[boxGridHash(x, y, z, d.bucketMask)]; build != cudaPairs::kEmpty;
             build              = d.next[build])
        {
            if (d.cellKeys[3u * build] == x && d.cellKeys[3u * build + 1u] == y && d.cellKeys[3u * build + 2u] == z)
            {
                pairConsider(d, query, build);
            }
        }
    }
    for (unsigned int index = lane; index < d.grid->oversizeCount; index += cudaPairs::kWarpSize)
    {
        pairConsider(d, query, d.oversize[index]);
    }
}

} // namespace

// Marks the positions one change record lists in the population's changed
// bitset and appends each newly marked one to the changed list.
extern "C" __global__ void ovspatialquery_cuda_pairs_mark(unsigned int count, const unsigned int* positions,
                                                          unsigned int objectCount, unsigned int* changedBits,
                                                          unsigned int* changedList, unsigned int* changedCount)
{
    for (unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += gridDim.x * blockDim.x)
    {
        const unsigned int position = positions[index];
        if (position >= objectCount)
        {
            continue;
        }
        const unsigned int bit      = 1u << (position & 31u);
        const unsigned int previous = atomicOr(&changedBits[position >> 5u], bit);
        if ((previous & bit) == 0u)
        {
            changedList[atomicAdd(changedCount, 1u)] = position;
        }
    }
}

// Sums and bounds the extents of one endpoint's admitted finite objects.
extern "C" __global__ void ovspatialquery_cuda_pairs_grid_stats(cudaPairs::Side side, cudaPairs::Grid* grid)
{
    double       sum   = 0.0;
    unsigned int count = 0u;
    for (unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; index < side.objectCount;
         index += gridDim.x * blockDim.x)
    {
        float lower[3];
        float upper[3];
        if (!pairAdmitted(side, index) ||
            !pairFloatBox(side.minimums + 3u * index, side.maximums + 3u * index, lower, upper))
        {
            continue;
        }
        double extent = 0.0;
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            extent = fmax(extent, static_cast<double>(upper[axis]) - static_cast<double>(lower[axis]));
        }
        sum += extent;
        ++count;
    }
    for (unsigned int offset = 16u; offset > 0u; offset >>= 1u)
    {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
        count += __shfl_down_sync(0xffffffffu, count, offset);
    }
    if ((threadIdx.x & 31u) == 0u && count != 0u)
    {
        atomicAdd(&grid->extentSum, sum);
        atomicAdd(&grid->objectCount, count);
    }
}

// Links each admitted object into the bucket of its minimum corner's cell, or
// appends it to the oversize list. The cell is a power of two twice the mean
// extent, and wide enough for the tolerance, so that most objects fit one cell
// and a query visits few; every thread derives it and the first records it.
extern "C" __global__ void ovspatialquery_cuda_pairs_grid_link(cudaPairs::Side side, cudaPairs::Grid* grid,
                                                               double tolerance, unsigned int bucketMask,
                                                               unsigned int* heads, unsigned int* next, int* cellKeys,
                                                               unsigned int* oversize)
{
    const double mean   = grid->objectCount == 0u ? 0.0 : grid->extentSum / static_cast<double>(grid->objectCount);
    double       target = fmax(2.0 * mean, mean + 2.0 * tolerance);
    if (!(target > 0.0) || !isfinite(target))
    {
        target = 1.0;
    }
    int exponent = 0;
    (void)frexp(target, &exponent);
    exponent                 = exponent < -1000 ? -1000 : (exponent > 1000 ? 1000 : exponent);
    const double       width = ldexp(1.0, exponent);
    const double       scale = ldexp(1.0, -exponent);
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index == 0u)
    {
        grid->inverseCell = scale;
    }
    if (index >= side.objectCount || !pairAdmitted(side, index))
    {
        return;
    }
    float lower[3];
    float upper[3];
    bool  regular = pairFloatBox(side.minimums + 3u * index, side.maximums + 3u * index, lower, upper);
    int   cell[3] = {0, 0, 0};
    for (unsigned int axis = 0u; regular && axis < 3u; ++axis)
    {
        regular = static_cast<double>(upper[axis]) - static_cast<double>(lower[axis]) <= width &&
                  boxGridCell(lower[axis], scale, cell[axis]);
    }
    if (!regular)
    {
        oversize[atomicAdd(&grid->oversizeCount, 1u)] = index;
        return;
    }
    cellKeys[3u * index]      = cell[0];
    cellKeys[3u * index + 1u] = cell[1];
    cellKeys[3u * index + 2u] = cell[2];
    next[index]               = atomicExch(&heads[boxGridHash(cell[0], cell[1], cell[2], bucketMask)], index);
}

// Keeps the accepted pairs neither of whose objects changed, and lists the
// others as candidates to exit.
extern "C" __global__ void ovspatialquery_cuda_pairs_carry(const cudaPairs::Entry* ring, unsigned long long ringStride,
                                                           const unsigned long long* ringCounts,
                                                           const unsigned int*       firstChanged,
                                                           const unsigned int* secondChanged, cudaPairs::Entry* entries,
                                                           unsigned long long capacity, unsigned int* candidates,
                                                           cudaPairs::Control* control)
{
    const cudaPairs::Entry*  accepted = ring + control->currentBuffer * ringStride;
    const unsigned long long count    = ringCounts[control->currentBuffer];
    for (unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
         index += static_cast<unsigned long long>(gridDim.x) * blockDim.x)
    {
        const cudaPairs::Entry entry = accepted[index];
        if (pairBitSet(firstChanged, entry.firstSlot) || pairBitSet(secondChanged, entry.secondSlot))
        {
            candidates[atomicAdd(&control->candidateCount, 1ull)] = static_cast<unsigned int>(index);
        }
        else
        {
            pairAppend(control, entries, capacity, entry);
        }
    }
}

extern "C" __global__ void __launch_bounds__(cudaPairs::kThreads)
    ovspatialquery_cuda_pairs_discover(cudaPairs::Discovery d)
{
    const unsigned int index = (blockIdx.x * blockDim.x + threadIdx.x) / cudaPairs::kWarpSize;
    if (index < d.queryCount && (d.queryListCount == nullptr || index < *d.queryListCount))
    {
        pairDiscoverWarp(d, d.queryList != nullptr ? d.queryList[index] : index, threadIdx.x % cudaPairs::kWarpSize);
    }
}

// Sizes the identity indexes of the two sets an advance compares, once the
// new set is complete, and clears them: a new set beyond the capacity reports
// ACTIVE_CAPACITY and compares nothing. Every thread sizes both indexes; the
// first records the sets for `index` and `difference`.
extern "C" __global__ void ovspatialquery_cuda_pairs_clear(cudaPairs::Sets sets, cudaPairs::Control* control,
                                                           const unsigned long long* ringCounts,
                                                           unsigned long long capacity, unsigned int patched,
                                                           unsigned int maximumMask)
{
    if (control->nextCount > capacity)
    {
        return;
    }
    const unsigned long long accepted  = ringCounts[control->currentBuffer];
    const unsigned long long carried   = patched != 0u ? accepted - control->candidateCount : 0ull;
    const unsigned long long counts[2] = {patched != 0u ? control->candidateCount : accepted,
                                          control->nextCount - carried};
    unsigned long long slots[2] = {2ull, 2ull};
    for (unsigned int set = 0u; set < 2u; ++set)
    {
        while (slots[set] < 2ull * counts[set] && slots[set] <= maximumMask)
        {
            slots[set] <<= 1u;
        }
    }
    if (blockIdx.x == 0u && threadIdx.x == 0u)
    {
        control->newStart = carried;
        for (unsigned int set = 0u; set < 2u; ++set)
        {
            control->setCount[set] = counts[set];
            control->setMask[set]  = static_cast<unsigned int>(slots[set] - 1ull);
        }
    }
    for (unsigned int set = 0u; set < 2u; ++set)
    {
        const unsigned long long used = counts[set] != 0ull ? slots[set] : 0ull;
        for (unsigned long long index = blockIdx.x * blockDim.x + threadIdx.x; index < used;
             index += static_cast<unsigned long long>(gridDim.x) * blockDim.x)
        {
            sets.hash[set][index] = cudaPairs::kEmpty;
        }
    }
}

// Indexes both sets by identity, by position within the set.
extern "C" __global__ void ovspatialquery_cuda_pairs_index(cudaPairs::Sets sets, const cudaPairs::Control* control)
{
    for (unsigned int set = 0u; set < 2u; ++set)
    {
        const unsigned long long count = control->setCount[set];
        const unsigned int       mask  = control->setMask[set];
        for (unsigned long long position = blockIdx.x * blockDim.x + threadIdx.x; position < count;
             position += static_cast<unsigned long long>(gridDim.x) * blockDim.x)
        {
            const cudaPairs::Entry& entry = pairSetEntry(sets, control, set, position);
            unsigned int            slot  = pairHash(entry.firstId, entry.secondId, mask);
            while (atomicCAS(&sets.hash[set][slot], cudaPairs::kEmpty, static_cast<unsigned int>(position)) !=
                   cudaPairs::kEmpty)
            {
                slot = (slot + 1u) & mask;
            }
        }
    }
}

// The pairs of each set missing from the other: new pairs missing from the
// baseline entered, baseline pairs missing from the new set exited.
extern "C" __global__ void ovspatialquery_cuda_pairs_difference(cudaPairs::Sets sets, unsigned long long* entered,
                                                                unsigned long long* exited,
                                                                unsigned long long  deltaCapacity,
                                                                cudaPairs::Control* control)
{
    for (unsigned int from = 0u; from < 2u; ++from)
    {
        const unsigned int       against = 1u - from;
        const unsigned long long count   = control->setCount[from];
        const bool               empty   = control->setCount[against] == 0ull;
        unsigned long long*      counter = from == 1u ? &control->enteredCount : &control->exitedCount;
        unsigned long long*      rows    = from == 1u ? entered : exited;
        for (unsigned long long position = blockIdx.x * blockDim.x + threadIdx.x; position < count;
             position += static_cast<unsigned long long>(gridDim.x) * blockDim.x)
        {
            const cudaPairs::Entry& entry = pairSetEntry(sets, control, from, position);
            if (!empty && pairFind(sets, control, against, entry.firstId, entry.secondId))
            {
                continue;
            }
            const unsigned long long row = atomicAdd(counter, 1ull);
            if (row < deltaCapacity)
            {
                rows[2u * row]      = entry.firstId;
                rows[2u * row + 1u] = entry.secondId;
            }
        }
    }
}

// The CPU transition's outcome: active capacity first, then the delta. Only
// an advanced result records the new set's count and makes its buffer the
// baseline the next advance compares with. `firstFailed` and `secondFailed` are the endpoint versions' preparation
// status words when an endpoint was admitted before its publication completed
// (ADR-0021), null otherwise: a failed endpoint makes the advance FAILED with
// zero counts and an unchanged baseline, whatever the kernels before computed
// from its tables (ADR-0022 decision 3). A nonzero `failed` does the same for
// an advance whose host enqueue failed after some of its work was enqueued.
extern "C" __global__ void ovspatialquery_cuda_pairs_finish(cudaPairs::Control* control, unsigned long long* ringCounts,
                                                            unsigned int writeBuffer, unsigned long long capacity,
                                                            unsigned long long deltaCapacity, cudaPairs::Header* header,
                                                            const unsigned int* firstFailed,
                                                            const unsigned int* secondFailed, unsigned int failed)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u)
    {
        return;
    }
    cudaPairs::Header result{};
    result.acceptedCount = ringCounts[control->currentBuffer];
    if (failed != 0u || (firstFailed != nullptr && *firstFailed != 0u) ||
        (secondFailed != nullptr && *secondFailed != 0u))
    {
        result.outcome = cudaPairs::kOutcomeFailed;
    }
    else if (control->nextCount > capacity)
    {
        result.outcome = cudaPairs::kOutcomeActiveCapacity;
    }
    else if (control->enteredCount > deltaCapacity || control->exitedCount > deltaCapacity - control->enteredCount)
    {
        result.outcome = cudaPairs::kOutcomeDeltaCapacity;
    }
    else
    {
        result.outcome          = cudaPairs::kOutcomeAdvanced;
        result.enteredCount     = control->enteredCount;
        result.exitedCount      = control->exitedCount;
        result.acceptedCount    = control->nextCount;
        ringCounts[writeBuffer] = control->nextCount;
        control->currentBuffer  = writeBuffer;
    }
    *header = result;
}

// Writes staged object rows into a version's device table, at the listed
// positions or, without a list, at their own index.
extern "C" __global__ void ovspatialquery_cuda_pairs_scatter_rows(unsigned int count, const unsigned int* positions,
                                                                  const cudaPairs::ObjectRow* rows,
                                                                  unsigned long long* ids, unsigned long long* bits,
                                                                  double* minimums, double* maximums)
{
    for (unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += gridDim.x * blockDim.x)
    {
        const unsigned int         position = positions != nullptr ? positions[index] : index;
        const cudaPairs::ObjectRow row      = rows[index];
        ids[position]                       = row.id;
        bits[2u * position]                 = row.bits[0];
        bits[2u * position + 1u]            = row.bits[1];
        for (unsigned int axis = 0u; axis < 3u; ++axis)
        {
            minimums[3u * position + axis] = row.minimum[axis];
            maximums[3u * position + axis] = row.maximum[axis];
        }
    }
}

// Writes staged membership flags at the listed positions.
extern "C" __global__ void ovspatialquery_cuda_pairs_scatter_flags(unsigned int count, const unsigned int* positions,
                                                                   const unsigned char* values, unsigned char* flags)
{
    for (unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += gridDim.x * blockDim.x)
    {
        flags[positions[index]] = values[index];
    }
}
