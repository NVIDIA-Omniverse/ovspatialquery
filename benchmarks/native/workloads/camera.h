/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Camera-style coherent ray batches: one pinhole image of rays against an empty scene, one box
// reported as bounds candidates, or one dense closed mesh about the size of a scanned bunny.
#ifndef BENCH_WORKLOADS_CAMERA_H
#define BENCH_WORKLOADS_CAMERA_H
#include "../fixture.h"

namespace bench
{
bool    cameraWorkload(const Options& o);
Fixture cameraFixture(const Options& o);
} // namespace bench
#endif
