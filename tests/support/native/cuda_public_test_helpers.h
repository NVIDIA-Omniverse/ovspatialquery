/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_TEST_CUDA_PUBLIC_TEST_HELPERS_H
#define OVSPATIALQUERY_TEST_CUDA_PUBLIC_TEST_HELPERS_H

// Helpers shared by the CUDA native tests: a plan destroy that waits for the
// plan to retire, the device address of a result tensor, and the library's
// evidence that a transferred consumer_done event was destroyed.

#include "cuda_driver.h"

#include <ovspatialquery/ovspatialquery.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>

extern "C" void ovspatialquery_test_get_cuda_transferred_event_evidence(uint64_t* attempts, uint64_t* successes,
                                                                        int* lastStatus, uint64_t* preDestroyFailures);

// Destroys a batch plan, retrying while the library reports it busy for up to
// ten seconds. Returns whether the destroy succeeded.
inline bool destroyPlanWhenRetired(ovspatialquery_instance_handle_t instance, ovspatialquery_batch_plan_handle_t plan)
{
    const std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    do
    {
        const ovspatialquery_api_status_t status = ovspatialquery_destroy_batch_plan(instance, plan).status;
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            return true;
        }
        if (status != OVSPATIALQUERY_API_BUSY)
        {
            return false;
        }
        std::this_thread::yield();
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
}

// Device address of a result column, including its byte_offset.
inline CudaDriver::DevicePointer deviceAddress(const DLTensor& tensor)
{
    return static_cast<CudaDriver::DevicePointer>(reinterpret_cast<uintptr_t>(tensor.data)) + tensor.byte_offset;
}

// The transferred-event evidence already accounted for by earlier checks.
inline uint64_t gObservedTransferredEventDestroyAttempts  = 0u;
inline uint64_t gObservedTransferredEventDestroySuccesses = 0u;

inline uint64_t transferredEventDestroyAttempts()
{
    uint64_t attempts = 0u;
    ovspatialquery_test_get_cuda_transferred_event_evidence(&attempts, nullptr, nullptr, nullptr);
    return attempts;
}

// Waits up to ten seconds for new destroy attempts, then returns whether
// exactly expectedCount more transferred events were destroyed successfully.
inline bool transferredEventDestroysWereInvoked(uint64_t expectedCount = 1u)
{
    const std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    do
    {
        uint64_t attempts   = 0u;
        uint64_t successes  = 0u;
        int      lastStatus = 0;
        ovspatialquery_test_get_cuda_transferred_event_evidence(&attempts, &successes, &lastStatus, nullptr);
        if (attempts >= gObservedTransferredEventDestroyAttempts + expectedCount)
        {
            const bool exact = attempts == gObservedTransferredEventDestroyAttempts + expectedCount &&
                               successes == gObservedTransferredEventDestroySuccesses + expectedCount &&
                               lastStatus == 0;
            if (!exact)
            {
                std::fprintf(
                    stderr,
                    "consumer_done evidence attempts=%llu observed=%llu expected=%llu successes=%llu status=%d\n",
                    static_cast<unsigned long long>(attempts),
                    static_cast<unsigned long long>(gObservedTransferredEventDestroyAttempts),
                    static_cast<unsigned long long>(expectedCount), static_cast<unsigned long long>(successes),
                    lastStatus);
            }
            gObservedTransferredEventDestroyAttempts  = attempts;
            gObservedTransferredEventDestroySuccesses = successes;
            return exact;
        }
        std::this_thread::yield();
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
}

#endif
