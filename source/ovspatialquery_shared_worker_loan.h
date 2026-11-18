/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_SHARED_WORKER_LOAN_H
#define OVSPATIALQUERY_SHARED_WORKER_LOAN_H

#include <cstddef>
#include <cstdint>

namespace ovspatialquery
{
namespace detail
{

/* The instance's fixed worker pool, lent to a caller for independent tasks. `run` executes
 * `task(taskState, index)` once for every index below `count`, on the calling thread and on
 * whichever of the pool's workers are idle, and returns once all have run; it never waits
 * for a busy worker and creates no thread. A null `run` leaves the caller to run the tasks
 * itself. `idleWorkers` reports how many workers are not running an operation now. */
struct SharedWorkerLoan
{
    void* state                                   = nullptr;
    void (*run)(void* state, size_t count, void (*task)(void* taskState, size_t index),
                void* taskState) noexcept         = nullptr;
    uint32_t (*idleWorkers)(void* state) noexcept = nullptr;
};

} // namespace detail
} // namespace ovspatialquery

#endif
