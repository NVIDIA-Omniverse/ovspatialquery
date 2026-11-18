/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_TEST_GATE_H
#define OVSPATIALQUERY_TEST_GATE_H

// Test-build checkpoint shared by the PhysX and oracle backends: the first
// `expected` calls to enterTestGate block until the test releases the gate.

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>

namespace ovspatialquery
{
namespace detail
{

struct TestGate
{
    std::mutex              mutex;
    std::condition_variable condition;
    uint32_t                expected = 0u;
    uint32_t                entered  = 0u;
    bool                    released = true;
};

inline void enterTestGate(TestGate& gate)
{
    std::unique_lock<std::mutex> lock(gate.mutex);
    if (gate.expected == 0u || gate.entered >= gate.expected)
    {
        return;
    }
    ++gate.entered;
    gate.condition.notify_all();
    gate.condition.wait(lock, [&gate]() { return gate.released; });
}

inline void configureTestGate(TestGate& gate, uint32_t expected)
{
    std::lock_guard<std::mutex> lock(gate.mutex);
    gate.expected = expected;
    gate.entered  = 0u;
    gate.released = expected == 0u;
}

// 1 when every expected call is blocked within ten seconds, else 0.
inline uint32_t waitForTestGate(TestGate& gate)
{
    std::unique_lock<std::mutex> lock(gate.mutex);
    return gate.condition.wait_for(lock, std::chrono::seconds(10), [&gate]() { return gate.entered == gate.expected; })
               ? 1u
               : 0u;
}

inline void releaseTestGate(TestGate& gate)
{
    std::lock_guard<std::mutex> lock(gate.mutex);
    gate.released = true;
    gate.expected = 0u;
    gate.condition.notify_all();
}

} // namespace detail
} // namespace ovspatialquery

#endif
