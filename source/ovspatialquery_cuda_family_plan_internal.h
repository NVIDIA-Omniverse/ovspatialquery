/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_CUDA_FAMILY_PLAN_INTERNAL_H
#define OVSPATIALQUERY_CUDA_FAMILY_PLAN_INTERNAL_H

#include "ovspatialquery_cuda_driver.h"
#include "ovspatialquery_cuda_family_plan.h"

#include <atomic>
#include <cstdint>
#include <memory>

namespace ovspatialquery
{
namespace detail
{

// The transferred consumer-done event release and its test counters, shared by
// ovspatialquery_cuda_dispatch.cpp, the CudaFamilyPlan implementation
// (ovspatialquery_cuda_family_plan_catalogue.cpp) and the CUDA pair reporter.
// The concrete plan class stays inside its own translation unit.

#if defined(OVSPATIALQUERY_HAS_CUDA_BACKEND)

// Releases a transferred consumer-done event. Returns true when the release
// observed a lost device; the caller then abandons the device-side state.
// Defined in ovspatialquery_cuda_dispatch.cpp next to the test hook that
// reports its evidence counters.
bool releaseTransferredConsumerDone(Driver& driver, CuContext ownerContext, CuEvent& ownedEvent,
                                    bool terminalDeviceLost) noexcept;

#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
// Evidence counters for the transferred consumer-done event destroy path,
// updated by releaseTransferredConsumerDone and reported by the
// ovspatialquery_test_get_cuda_transferred_event_evidence test hook in
// ovspatialquery_cuda_dispatch.cpp.
extern std::atomic<uint64_t> gCudaTransferredEventDestroyAttempts;
extern std::atomic<uint64_t> gCudaTransferredEventDestroySuccesses;
extern std::atomic<uint64_t> gCudaTransferredEventPreDestroyFailures;
extern std::atomic<int>      gCudaTransferredEventDestroyLastStatus;
#endif

#endif // defined(OVSPATIALQUERY_HAS_CUDA_BACKEND)

} // namespace detail
} // namespace ovspatialquery

#endif
