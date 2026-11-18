/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_BACKEND_H
#define OVSPATIALQUERY_BACKEND_H

#include "ovspatialquery_internal.h"
#include "ovspatialquery_object_table.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

struct ovspatialquery_backend_runtime_t;

namespace ovspatialquery
{
namespace detail
{

struct FloatAabb
{
    float minimum[3];
    float maximum[3];
};

struct FloatRaycastQuery
{
    float origin[3];
    float unitDirection[3];
    float maxDistance;
};

struct FloatSphereSweepQuery
{
    float center[3];
    float unitDirection[3];
    float radius;
    float maxDistance;
};

// An axis-aligned box, centred at `center`, swept along the path.
struct FloatBoxSweepQuery
{
    float center[3];
    float halfExtents[3];
    float unitDirection[3];
    float maxDistance;
};

class BackendCandidateSink
{
  public:
    virtual ~BackendCandidateSink() = default;

    // Return false to stop traversal successfully. Implementations expose any
    // failure through status().
    virtual bool                        reportCandidate(uint32_t objectIndex) = 0;
    virtual ovspatialquery_api_status_t status() const                        = 0;
    // Read by the path traversals after each candidate: candidates the path
    // first reaches beyond this distance along the traversal's own float path
    // are no longer needed, so the backend may skip them. Infinity keeps the
    // whole path.
    float traversalReach() const { return mReach; }

  protected:
    // Lowers the reach; it never grows during a traversal.
    // A comparison rather than std::min, which a translation unit that includes
    // <windows.h> without NOMINMAX turns into its min macro.
    void lowerReach(float reach)
    {
        if (reach < mReach)
        {
            mReach = reach;
        }
    }

  private:
    float mReach = std::numeric_limits<float>::infinity();
};

// Receives candidate object-index pairs whose backend bounds overlap; the
// caller applies the exact predicate, selection and tolerance rules.
class BackendPairSink
{
  public:
    virtual ~BackendPairSink() = default;

    // Return false to stop traversal successfully. Implementations expose any
    // failure through status().
    virtual bool                        reportPair(uint32_t objectIndex, uint32_t otherObjectIndex) = 0;
    virtual ovspatialquery_api_status_t status() const                                              = 0;
};

// A traversal that a sink stopped with a failure returns that sink's status().
class BackendPopulation
{
  public:
    virtual ~BackendPopulation() = default;

    // Strong guarantee: failure leaves this backend's observable state unchanged.
    virtual ovspatialquery_api_status_t update(const ObjectTable& objects) = 0;
    // `update` for objects that equal `base`'s objects except at the sorted
    // `changedPositions` (ADR-0019), which a backend may use to reuse `base`'s
    // structures; `base` stays unchanged and usable. Every query then answers
    // as after `update`, up to conservative candidates.
    virtual ovspatialquery_api_status_t derive(const BackendPopulation& base, const ObjectTable& objects,
                                               const std::vector<uint32_t>& changedPositions)
    {
        (void)base;
        (void)changedPositions;
        return update(objects);
    }
    // Every candidate pair of this population's objects with `other`'s objects
    // whose zero-tolerance backend bounds overlap: conservative, never a subset,
    // and unlike the single-object candidate traversals each pair is reported
    // at most once.
    virtual ovspatialquery_api_status_t findPairs(const BackendPopulation& other, BackendPairSink& sink) const = 0;
    virtual ovspatialquery_api_status_t raycastCandidates(const FloatRaycastQuery& query,
                                                          BackendCandidateSink&    sink) const                    = 0;
    virtual ovspatialquery_api_status_t sphereSweepCandidates(const FloatSphereSweepQuery& query,
                                                              BackendCandidateSink&        sink) const                = 0;
    virtual ovspatialquery_api_status_t boxSweepCandidates(const FloatBoxSweepQuery& query,
                                                           BackendCandidateSink&     sink) const                   = 0;
    virtual ovspatialquery_api_status_t aabbOverlapCandidates(const FloatAabb&      query,
                                                              BackendCandidateSink& sink) const                = 0;
    // Largest per-axis distance by which a float ray origin may differ from
    // the exact double origin while raycastCandidates still reports every
    // object the exact ray can touch. Zero: only exactly representable origins.
    virtual double rayOriginTolerance() const noexcept { return 0.0; }
};

ovspatialquery_api_status_t createBackendPopulation(ovspatialquery_backend_runtime_t*   backendRuntime,
                                                    std::unique_ptr<BackendPopulation>& population);

ovspatialquery_api_status_t acquireDefaultBackendRuntime(ovspatialquery_backend_runtime_t** outRuntime);
void                        releaseDefaultBackendRuntime(ovspatialquery_backend_runtime_t* runtime);

#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
// Source-private deterministic test checkpoints. The oracle backend can hold
// an update immediately before or after its non-fallible publication step.
void reachPopulationBeforePublish(ovspatialquery_backend_runtime_t* backendRuntime, ovspatialquery_ordinal_t ordinal);
void reachPopulationAfterPublish(ovspatialquery_backend_runtime_t* backendRuntime, ovspatialquery_ordinal_t ordinal);
void reachOperationBeforeCompletion(ovspatialquery_backend_runtime_t* backendRuntime, ovspatialquery_ordinal_t ordinal);
#endif

} // namespace detail
} // namespace ovspatialquery

#endif
