/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_CUDA_FAILURE_POINTS_H
#define OVSPATIALQUERY_CUDA_FAILURE_POINTS_H

#include <cstdint>

namespace ovspatialquery
{
namespace detail
{

// The ids ovspatialquery_test_set_cuda_failure_point arms in a test runtime:
// one list for the library branches that take them and the tests that arm
// them. It needs no CUDA backend, so every build sees the same names.

// A CUDA family plan launch fails with an invalid handle after its kernels
// are accepted or before they are enqueued, or reports device loss before them.
constexpr uint32_t kCudaFailureFamilyLaunchAfterKernels  = 4u;
constexpr uint32_t kCudaFailureFamilyLaunchBeforeKernels = 5u;
constexpr uint32_t kCudaFailureFamilyLaunchDeviceLost    = 7u;
// Preparing a CUDA family plan launch throws std::length_error (OUT_OF_MEMORY).
constexpr uint32_t kCudaFailureFamilyPrepare                         = 8u;
constexpr uint32_t kCudaFailureConsumerPollDeviceLost                = 9u;
constexpr uint32_t kCudaFailureResultMetadataTransfer                = 10u;
constexpr uint32_t kCudaFailureResultMetadataDeviceLost              = 11u;
constexpr uint32_t kCudaFailureResultMetadataInvariant               = 12u;
constexpr uint32_t kCudaFailureConsumerDestroyContextSelection       = 13u;
constexpr uint32_t kCudaFailureConsumerDestroyOuterContextDeviceLost = 14u;
// A CUDA pair reporter admission fails after planning its device copies.
constexpr uint32_t kCudaFailurePairReporterAdmission = 15u;
// Device loss injected on the device-prepared publication path (ADR-0021).
constexpr uint32_t kCudaFailureDevicePublicationLaunch   = 16u;
constexpr uint32_t kCudaFailureDevicePublicationComplete = 17u;
// Device loss injected on a CUDA pair reporter advance, after its kernels are
// enqueued or when its ready event is queried.
constexpr uint32_t kCudaFailurePairReporterLaunch   = 18u;
constexpr uint32_t kCudaFailurePairReporterComplete = 19u;
// Restoring the caller's context after a consumer-event transfer fails, or
// reports device loss.
constexpr uint32_t kCudaFailureConsumerTransferRestore           = 20u;
constexpr uint32_t kCudaFailureConsumerTransferRestoreDeviceLost = 21u;
// A gather returns an error after accepting its first kernel, before recording completion.
constexpr uint32_t kCudaFailureOvstageGatherAfterKernel = 22u;
// Polling a CUDA family launch's ready event fails as a failed switch to the plan's context
// does, reporting an error while the launch's work may still be queued.
constexpr uint32_t kCudaFailureResultPollContextSelection = 23u;
// That poll fails, and so does the wait that should then confirm the device finished,
// as an error CUDA reports from earlier work does, while the launch's work may still run.
constexpr uint32_t kCudaFailureResultConfirmation = 24u;

} // namespace detail
} // namespace ovspatialquery

#endif
