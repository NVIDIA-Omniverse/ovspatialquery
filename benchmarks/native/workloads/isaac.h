/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Isaac-derived semantic smoke: bounded ray, occupancy and zone probes with inside-origin diagnostics.
#ifndef BENCH_WORKLOADS_ISAAC_H
#define BENCH_WORKLOADS_ISAAC_H
#include "../fixture.h"

#include <vector>

namespace bench
{
struct InsideProbe
{
    Hit direct, spatial, shifted, composed;
};
extern std::vector<InsideProbe> insideProbes;
void                            semanticSmoke();
} // namespace bench
#endif
