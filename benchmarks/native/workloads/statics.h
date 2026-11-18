/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Static scenes where the query reaches far beyond its first contact: long sweeps over a flat
// sea of boxes, sweeps through one densely tessellated mesh, and giant overlaps of a mesh level.
#ifndef BENCH_WORKLOADS_STATICS_H
#define BENCH_WORKLOADS_STATICS_H
#include "../fixture.h"

namespace bench
{
bool    staticsWorkload(const Options& o);
Fixture staticsFixture(const Options& o);
} // namespace bench
#endif
