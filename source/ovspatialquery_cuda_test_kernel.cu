/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Test-only CUDA source: a scheduling gate the native tests launch to hold a
// stream busy while they observe pending work. It is compiled to
// ovspatialquery_cuda_test.ptx, copied next to the CUDA test executables, and
// never shipped or installed.

extern "C" __global__ void ovspatialquery_cuda_scheduling_delay(unsigned long long clocks)
{
    const unsigned long long start = clock64();
    while (clock64() - start < clocks) {}
}
