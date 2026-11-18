/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The engine's backend interface first: it carries the internal status
// vocabulary the vtable below must return.
#include "oracle_backend.h"
#include "ovspatialquery_backend.h"
#include "ovspatialquery_test_gate.h"
#include "portable_convex_hull.h"

#include <memory>
#include <mutex>
#include <new>
#include <utility>
#include <vector>

using ovspatialquery::detail::TestGate;

struct OracleOrdinalGate
{
    TestGate                 gate;
    ovspatialquery_ordinal_t ordinal = 0u;
};

struct ovspatialquery_backend_runtime_t
{
    uint32_t                                             oracleRuntimeTag;
    TestGate                                             buildGate;
    TestGate                                             rayGate;
    OracleOrdinalGate                                    beforePublishGate;
    OracleOrdinalGate                                    afterPublishGate;
    OracleOrdinalGate                                    beforeCompletionGate;
    std::mutex                                           failureMutex;
    ovspatialquery_api_status_t                          nextBuildFailure = OVSPATIALQUERY_API_SUCCESS;
    ovspatialquery::detail::ConvexHullPreparationOutcome nextConvexPreparationFailure =
        ovspatialquery::detail::ConvexHullPreparationOutcome::success;
    ovspatialquery_api_status_t nextRayFailure = OVSPATIALQUERY_API_SUCCESS;
};

namespace
{

constexpr uint32_t                kOracleRuntimeTag = 0x4f565351u;
std::mutex                        gDefaultRuntimeMutex;
ovspatialquery_backend_runtime_t* gDefaultRuntime               = nullptr;
uint64_t                          gDefaultRuntimeReferenceCount = 0u;

void enterOrdinalGate(OracleOrdinalGate& ordinalGate, ovspatialquery_ordinal_t ordinal)
{
    TestGate&                    gate = ordinalGate.gate;
    std::unique_lock<std::mutex> lock(gate.mutex);
    if (ordinalGate.ordinal != ordinal || gate.expected == 0u || gate.entered >= gate.expected)
    {
        return;
    }
    ++gate.entered;
    gate.condition.notify_all();
    gate.condition.wait(lock, [&gate]() { return gate.released; });
}

void configureOrdinalGate(OracleOrdinalGate& ordinalGate, ovspatialquery_ordinal_t ordinal)
{
    TestGate&                   gate = ordinalGate.gate;
    std::lock_guard<std::mutex> lock(gate.mutex);
    ordinalGate.ordinal = ordinal;
    gate.expected       = 1u;
    gate.entered        = 0u;
    gate.released       = false;
}

void configureOrdinalGate(OracleOrdinalGate& ordinalGate, ovspatialquery_ordinal_t ordinal, uint32_t expected)
{
    TestGate&                   gate = ordinalGate.gate;
    std::lock_guard<std::mutex> lock(gate.mutex);
    ordinalGate.ordinal = ordinal;
    gate.expected       = expected;
    gate.entered        = 0u;
    gate.released       = expected == 0u;
}

bool overlaps(const ovspatialquery_aabb_t& left, const ovspatialquery_aabb_t& right)
{
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        if (left.maximum[axis] < right.minimum[axis] || right.maximum[axis] < left.minimum[axis])
        {
            return false;
        }
    }
    return true;
}

class OraclePopulation final : public ovspatialquery::detail::BackendPopulation
{
  public:
    explicit OraclePopulation(ovspatialquery_backend_runtime_t* runtime) : mRuntime(runtime) {}

    ovspatialquery_api_status_t update(const ovspatialquery::detail::ObjectTable& objects) override
    {
        ovspatialquery::detail::enterTestGate(mRuntime->buildGate);
        {
            std::lock_guard<std::mutex> lock(mRuntime->failureMutex);
            if (mRuntime->nextBuildFailure != OVSPATIALQUERY_API_SUCCESS)
            {
                const ovspatialquery_api_status_t failure = mRuntime->nextBuildFailure;
                mRuntime->nextBuildFailure                = OVSPATIALQUERY_API_SUCCESS;
                return failure;
            }
        }
        std::vector<ovspatialquery::detail::PopulationObject> candidate(objects.begin(), objects.end());
        m_objects.swap(candidate);
        return OVSPATIALQUERY_API_SUCCESS;
    }

    ovspatialquery_api_status_t findPairs(const ovspatialquery::detail::BackendPopulation& other,
                                          ovspatialquery::detail::BackendPairSink&         sink) const override
    {
        const OraclePopulation* otherPopulation = dynamic_cast<const OraclePopulation*>(&other);
        if (otherPopulation == nullptr)
        {
            return OVSPATIALQUERY_API_INTERNAL;
        }
        for (uint32_t populationAIndex = 0u; populationAIndex < m_objects.size(); ++populationAIndex)
        {
            for (uint32_t populationBIndex = 0u; populationBIndex < otherPopulation->m_objects.size();
                 ++populationBIndex)
            {
                if (overlaps(m_objects[populationAIndex].bounds, otherPopulation->m_objects[populationBIndex].bounds) &&
                    !sink.reportPair(populationAIndex, populationBIndex))
                {
                    return sink.status();
                }
            }
        }
        return sink.status();
    }

    ovspatialquery_api_status_t raycastCandidates(const ovspatialquery::detail::FloatRaycastQuery& query,
                                                  ovspatialquery::detail::BackendCandidateSink&    sink) const override
    {
        ovspatialquery::detail::enterTestGate(mRuntime->rayGate);
        {
            std::lock_guard<std::mutex> lock(mRuntime->failureMutex);
            if (mRuntime->nextRayFailure != OVSPATIALQUERY_API_SUCCESS)
            {
                const ovspatialquery_api_status_t failure = mRuntime->nextRayFailure;
                mRuntime->nextRayFailure                  = OVSPATIALQUERY_API_SUCCESS;
                return failure;
            }
        }
        (void)query;
        return reportAllCandidates(sink);
    }

    ovspatialquery_api_status_t sphereSweepCandidates(const ovspatialquery::detail::FloatSphereSweepQuery& query,
                                                      ovspatialquery::detail::BackendCandidateSink& sink) const override
    {
        (void)query;
        return reportAllCandidates(sink);
    }

    ovspatialquery_api_status_t boxSweepCandidates(const ovspatialquery::detail::FloatBoxSweepQuery& query,
                                                   ovspatialquery::detail::BackendCandidateSink& sink) const override
    {
        (void)query;
        return reportAllCandidates(sink);
    }

    ovspatialquery_api_status_t aabbOverlapCandidates(const ovspatialquery::detail::FloatAabb&      query,
                                                      ovspatialquery::detail::BackendCandidateSink& sink) const override
    {
        (void)query;
        return reportAllCandidates(sink);
    }

  private:
    ovspatialquery_api_status_t reportAllCandidates(ovspatialquery::detail::BackendCandidateSink& sink) const
    {
        for (uint32_t objectIndex = 0u; objectIndex < m_objects.size(); ++objectIndex)
        {
            if (!sink.reportCandidate(objectIndex))
            {
                break;
            }
            // Candidate-producing backends may conservatively report the same
            // population object more than once. Exercise that contract in every
            // oracle traversal without adding a separate control surface.
            if (!sink.reportCandidate(objectIndex))
            {
                break;
            }
        }
        return sink.status();
    }

    ovspatialquery_backend_runtime_t*                     mRuntime;
    std::vector<ovspatialquery::detail::PopulationObject> m_objects;
};

} // namespace

namespace ovspatialquery
{
namespace detail
{

ovspatialquery_api_status_t createBackendPopulation(ovspatialquery_backend_runtime_t*   backendRuntime,
                                                    std::unique_ptr<BackendPopulation>& population)
{
    if (backendRuntime == nullptr || backendRuntime->oracleRuntimeTag != kOracleRuntimeTag)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    try
    {
        std::unique_ptr<BackendPopulation> candidate(new OraclePopulation(backendRuntime));
        population = std::move(candidate);
        return OVSPATIALQUERY_API_SUCCESS;
    }
    catch (const std::bad_alloc&)
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
}

ConvexHullPreparationOutcome prepareConvexHull(ovspatialquery_backend_runtime_t* backendRuntime,
                                               const std::vector<float>& authoredPoints, uint64_t maximumResidentBytes,
                                               std::shared_ptr<const PreparedConvexHull>& outHull,
                                               const ConvexHullPreparationBudget*         budget,
                                               const ConvexHullCookingOptions& options, double* outFarthestOutside)
{
    if (backendRuntime == nullptr || backendRuntime->oracleRuntimeTag != kOracleRuntimeTag)
    {
        return ConvexHullPreparationOutcome::invalidArgument;
    }
    {
        std::lock_guard<std::mutex>        lock(backendRuntime->failureMutex);
        const ConvexHullPreparationOutcome failure   = backendRuntime->nextConvexPreparationFailure;
        backendRuntime->nextConvexPreparationFailure = ConvexHullPreparationOutcome::success;
        if (failure != ConvexHullPreparationOutcome::success)
        {
            return failure;
        }
    }
    return preparePortableConvexHull(authoredPoints, kMaximumConvexHullVertices, maximumResidentBytes, outHull, budget,
                                     options, outFarthestOutside);
}

void failNextConvexHullPreparationForTest(ovspatialquery_backend_runtime_t* backendRuntime,
                                          ConvexHullPreparationOutcome      failure)
{
    std::lock_guard<std::mutex> lock(backendRuntime->failureMutex);
    backendRuntime->nextConvexPreparationFailure = failure;
}

ovspatialquery_api_status_t acquireDefaultBackendRuntime(ovspatialquery_backend_runtime_t** outRuntime)
{
    if (outRuntime == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    *outRuntime = nullptr;
    std::lock_guard<std::mutex> lock(gDefaultRuntimeMutex);
    if (gDefaultRuntime == nullptr)
    {
        try
        {
            gDefaultRuntime                   = new ovspatialquery_backend_runtime_t();
            gDefaultRuntime->oracleRuntimeTag = kOracleRuntimeTag;
        }
        catch (const std::bad_alloc&)
        {
            return OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
    }
    ++gDefaultRuntimeReferenceCount;
    *outRuntime = gDefaultRuntime;
    return OVSPATIALQUERY_API_SUCCESS;
}

void releaseDefaultBackendRuntime(ovspatialquery_backend_runtime_t* runtime)
{
    std::lock_guard<std::mutex> lock(gDefaultRuntimeMutex);
    if (runtime == nullptr || runtime != gDefaultRuntime || gDefaultRuntimeReferenceCount == 0u)
    {
        return;
    }
    --gDefaultRuntimeReferenceCount;
    if (gDefaultRuntimeReferenceCount == 0u)
    {
        delete gDefaultRuntime;
        gDefaultRuntime = nullptr;
    }
}

void reachPopulationBeforePublish(ovspatialquery_backend_runtime_t* backendRuntime, ovspatialquery_ordinal_t ordinal)
{
    enterOrdinalGate(backendRuntime->beforePublishGate, ordinal);
}

void reachPopulationAfterPublish(ovspatialquery_backend_runtime_t* backendRuntime, ovspatialquery_ordinal_t ordinal)
{
    enterOrdinalGate(backendRuntime->afterPublishGate, ordinal);
}

void reachOperationBeforeCompletion(ovspatialquery_backend_runtime_t* backendRuntime, ovspatialquery_ordinal_t ordinal)
{
    enterOrdinalGate(backendRuntime->beforeCompletionGate, ordinal);
}

} // namespace detail
} // namespace ovspatialquery

namespace
{

// Tests inject failures in the public status vocabulary; the backend vtable
// still speaks the engine's internal codes. This switch disappears once the
// engine returns the public enum itself.
} // namespace

extern "C" ovspatialquery_api_status_t ovspatialquery_test_oracle_create_runtime(
    ovspatialquery_backend_runtime_t** out_runtime)
{
    if (out_runtime == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    *out_runtime = nullptr;
    try
    {
        std::unique_ptr<ovspatialquery_backend_runtime_t> runtime(new ovspatialquery_backend_runtime_t());
        runtime->oracleRuntimeTag = kOracleRuntimeTag;
        *out_runtime              = runtime.release();
        return OVSPATIALQUERY_API_SUCCESS;
    }
    catch (const std::bad_alloc&)
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
}

extern "C" void ovspatialquery_test_oracle_destroy_runtime(ovspatialquery_backend_runtime_t* runtime)
{
    delete runtime;
}

extern "C" ovspatialquery_backend_runtime_t* ovspatialquery_test_oracle_get_default_runtime()
{
    std::lock_guard<std::mutex> lock(gDefaultRuntimeMutex);
    return gDefaultRuntime;
}

extern "C" void ovspatialquery_test_oracle_block_builds(ovspatialquery_backend_runtime_t* runtime,
                                                        uint32_t                          expected_builds)
{
    ovspatialquery::detail::configureTestGate(runtime->buildGate, expected_builds);
}

extern "C" uint32_t ovspatialquery_test_oracle_wait_for_blocked_builds(ovspatialquery_backend_runtime_t* runtime)
{
    return ovspatialquery::detail::waitForTestGate(runtime->buildGate);
}

extern "C" void ovspatialquery_test_oracle_release_builds(ovspatialquery_backend_runtime_t* runtime)
{
    ovspatialquery::detail::releaseTestGate(runtime->buildGate);
}

extern "C" void ovspatialquery_test_oracle_block_rays(ovspatialquery_backend_runtime_t* runtime, uint32_t expected_rays)
{
    ovspatialquery::detail::configureTestGate(runtime->rayGate, expected_rays);
}

extern "C" uint32_t ovspatialquery_test_oracle_wait_for_blocked_rays(ovspatialquery_backend_runtime_t* runtime)
{
    return ovspatialquery::detail::waitForTestGate(runtime->rayGate);
}

extern "C" void ovspatialquery_test_oracle_release_rays(ovspatialquery_backend_runtime_t* runtime)
{
    ovspatialquery::detail::releaseTestGate(runtime->rayGate);
}

extern "C" void ovspatialquery_test_oracle_fail_next_build(ovspatialquery_backend_runtime_t* runtime,
                                                           ovspatialquery_api_status_t       failure)
{
    std::lock_guard<std::mutex> lock(runtime->failureMutex);
    runtime->nextBuildFailure = failure;
}

extern "C" void ovspatialquery_test_oracle_fail_next_ray(ovspatialquery_backend_runtime_t* runtime,
                                                         ovspatialquery_api_status_t       failure)
{
    std::lock_guard<std::mutex> lock(runtime->failureMutex);
    runtime->nextRayFailure = failure;
}

extern "C" void ovspatialquery_test_oracle_block_before_publish(ovspatialquery_backend_runtime_t* runtime,
                                                                ovspatialquery_ordinal_t          ordinal)
{
    configureOrdinalGate(runtime->beforePublishGate, ordinal);
}

extern "C" uint32_t ovspatialquery_test_oracle_wait_before_publish(ovspatialquery_backend_runtime_t* runtime)
{
    return ovspatialquery::detail::waitForTestGate(runtime->beforePublishGate.gate);
}

extern "C" void ovspatialquery_test_oracle_release_before_publish(ovspatialquery_backend_runtime_t* runtime)
{
    ovspatialquery::detail::releaseTestGate(runtime->beforePublishGate.gate);
}

extern "C" void ovspatialquery_test_oracle_block_after_publish(ovspatialquery_backend_runtime_t* runtime,
                                                               ovspatialquery_ordinal_t          ordinal)
{
    configureOrdinalGate(runtime->afterPublishGate, ordinal);
}

extern "C" uint32_t ovspatialquery_test_oracle_wait_after_publish(ovspatialquery_backend_runtime_t* runtime)
{
    return ovspatialquery::detail::waitForTestGate(runtime->afterPublishGate.gate);
}

extern "C" void ovspatialquery_test_oracle_release_after_publish(ovspatialquery_backend_runtime_t* runtime)
{
    ovspatialquery::detail::releaseTestGate(runtime->afterPublishGate.gate);
}

extern "C" void ovspatialquery_test_oracle_block_before_completion(ovspatialquery_backend_runtime_t* runtime,
                                                                   ovspatialquery_ordinal_t          ordinal,
                                                                   uint32_t expected_operations)
{
    configureOrdinalGate(runtime->beforeCompletionGate, ordinal, expected_operations);
}

extern "C" uint32_t ovspatialquery_test_oracle_wait_before_completion(ovspatialquery_backend_runtime_t* runtime)
{
    return ovspatialquery::detail::waitForTestGate(runtime->beforeCompletionGate.gate);
}

extern "C" void ovspatialquery_test_oracle_release_before_completion(ovspatialquery_backend_runtime_t* runtime)
{
    ovspatialquery::detail::releaseTestGate(runtime->beforeCompletionGate.gate);
}
