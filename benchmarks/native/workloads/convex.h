/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Shared cube and prism convex fixtures for rays, sweeps and complete box overlaps.
#ifndef BENCH_WORKLOADS_CONVEX_H
#define BENCH_WORKLOADS_CONVEX_H
#include "../fixture.h"

namespace bench
{
Fixture convexFixture(const Options& o);
} // namespace bench
#endif
