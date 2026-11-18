/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Fixed-topology deforming sheets: one flexible triangle sheet per workcell republished through the
// public Ovstage population API (unchanged, rigid, vertex-only, crumpled and occasional topology
// updates), queried by a ray batch on the new head, the warm head and the retained head, and paired
// against a tool population; direct pinned PhysX recooking and BVH refit are the raw baselines.
#ifndef BENCH_WORKLOADS_DEFORM_H
#define BENCH_WORKLOADS_DEFORM_H
#include "../common.h"

namespace bench
{
bool deformWorkload(const Options& o);
int  deformRun(const Options& o);
} // namespace bench
#endif
