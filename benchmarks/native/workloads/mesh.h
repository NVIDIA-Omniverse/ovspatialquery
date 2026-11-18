/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Procedural terrain tiles, mesh grounding/sweep fixtures and the mesh semantic smoke.
#ifndef BENCH_WORKLOADS_MESH_H
#define BENCH_WORKLOADS_MESH_H
#include "../fixture.h"

namespace bench
{
double  terrainHeight(double x);
Mesh    terrain(size_t side);
Fixture meshFixture(const Options& o);
void    meshQueries(Fixture& f, size_t frame);
void    meshSemanticSmoke();
} // namespace bench
#endif
