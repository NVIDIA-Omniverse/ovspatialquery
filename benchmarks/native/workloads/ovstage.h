/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Ovstage publication analogue: one sealed stage republished per step through the public Ovstage
// population API, with explicit registration of the same scene and direct PhysX as controls.
#ifndef BENCH_WORKLOADS_OVSTAGE_H
#define BENCH_WORKLOADS_OVSTAGE_H
#include "../common.h"

namespace bench
{
int ovstageRun(const Options& o);
} // namespace bench
#endif
