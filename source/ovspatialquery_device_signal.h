/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_DEVICE_SIGNAL_H
#define OVSPATIALQUERY_DEVICE_SIGNAL_H

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>

namespace ovspatialquery
{
namespace detail
{

// Wakes a thread that waits for progress it cannot poll cheaply: device work
// (rung by a CUDA host function queued behind it) or another operation. A
// waiter reads count() before checking its condition and waits past that
// count, so a ring in between is not lost. Never held across a CUDA call, so
// a host function can always take the mutex. Shared ownership lets a host
// function that runs after its instance is gone ring a signal nobody waits on.
class DeviceSignal
{
  public:
    uint64_t count()
    {
        std::lock_guard<std::mutex> lock(mMutex);
        return mCount;
    }

    void ring()
    {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            ++mCount;
        }
        mCondition.notify_all();
    }

    // The timeout covers a ring that never comes; on Windows a wait shorter
    // than the timer tick (15.6 ms by default) lasts a whole tick.
    void waitPast(uint64_t seen, std::chrono::milliseconds timeout)
    {
        std::unique_lock<std::mutex> lock(mMutex);
        mCondition.wait_for(lock, timeout, [this, seen]() { return mCount != seen; });
    }

  private:
    std::mutex              mMutex;
    std::condition_variable mCondition;
    uint64_t                mCount = 0u;
};

// An instance's operation condition: every notification also rings `signal`, so
// a thread waiting on the signal sees everything a waiter on the condition sees.
class OperationCondition
{
  public:
    void notify_all()
    {
        mCondition.notify_all();
        signal->ring();
    }

    // Wakes one thread waiting on the condition: an operation that one idle worker
    // can run must not wake them all, which then contend for the queue's mutex
    // with the caller that will often run the operation itself (wait_op).
    void notify_one()
    {
        mCondition.notify_one();
        signal->ring();
    }

    template <typename Predicate>
    void wait(std::unique_lock<std::mutex>& lock, Predicate predicate)
    {
        mCondition.wait(lock, predicate);
    }

    template <typename Rep, typename Period>
    std::cv_status wait_for(std::unique_lock<std::mutex>& lock, const std::chrono::duration<Rep, Period>& timeout)
    {
        return mCondition.wait_for(lock, timeout);
    }

    template <typename Rep, typename Period, typename Predicate>
    bool wait_for(std::unique_lock<std::mutex>& lock, const std::chrono::duration<Rep, Period>& timeout,
                  Predicate predicate)
    {
        return mCondition.wait_for(lock, timeout, predicate);
    }

    const std::shared_ptr<DeviceSignal> signal = std::make_shared<DeviceSignal>();

  private:
    std::condition_variable mCondition;
};

} // namespace detail
} // namespace ovspatialquery

#endif
