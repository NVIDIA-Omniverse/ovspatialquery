/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Consumer-grounded warehouse rack mesh fixture ingested from the exported binary
// (benchmarks/warehouse_meshes.py): angular ray/sweep lattices and candidate-driven overlaps.
#ifndef BENCH_WORKLOADS_WAREHOUSE_H
#define BENCH_WORKLOADS_WAREHOUSE_H
#include "../fixture.h"

namespace bench
{
bool    warehouseWorkload(const Options& o);
Fixture warehouseFixture(const Options& o);
// Sum over objects of the triangle count of their mesh definition (instanced, not unique, triangles).
size_t warehouseInstancedTriangles(const Fixture& f);
} // namespace bench
#endif
