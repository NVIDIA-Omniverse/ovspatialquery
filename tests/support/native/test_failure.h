/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_TEST_FAILURE_H
#define OVSPATIALQUERY_TEST_FAILURE_H

#include <cstdio>
#include <cstdlib>

// A failure can fire while an instance still owns live or test-paused worker
// threads; std::exit would run static destructors that join them and hang.
// Flush what the test printed, then terminate without any cleanup.
[[noreturn]] inline void ovspatialquery_test_exit_failure(const char* file, int line)
{
    std::fprintf(stderr, "native test failure exits nonzero at %s:%d\n", file, line);
    std::fflush(stdout);
    std::fflush(stderr);
    std::_Exit(EXIT_FAILURE);
}

#define OVSPATIALQUERY_TEST_EXIT_FAILURE() ovspatialquery_test_exit_failure(__FILE__, __LINE__)

// Exits nonzero at the caller's file and line when the condition is false.
// Variadic so brace-initialized literals with commas pass through.
#define REQUIRE(...) ((__VA_ARGS__) ? static_cast<void>(0) : ovspatialquery_test_exit_failure(__FILE__, __LINE__))

#endif
