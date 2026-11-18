/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_CUDA_MESH_RESIDENCY_H
#define OVSPATIALQUERY_CUDA_MESH_RESIDENCY_H

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>

namespace ovspatialquery
{
namespace detail
{

// The triangle mesh definitions one CUDA device holds for one instance's plans,
// by host identity (ovspatialquery_cuda_family_plan_catalogue.cpp). A plan that
// finds a definition here reads the same device arrays instead of building and
// uploading its own; the device bytes stay charged to the instance once, from
// the launch that uploaded them until no plan's catalogue holds them.
struct CudaMeshResidencyIndex;

// One instance's indices, one per CUDA device. Plans hold their device's index;
// the registry only finds it for the next plan, so it lives while some plan on
// that device does.
class CudaMeshResidencyRegistry
{
  public:
    // The index of CUDA device `deviceId`, made when no live plan holds one;
    // null when it cannot be allocated. Defined with the CUDA family plan.
    std::shared_ptr<CudaMeshResidencyIndex> index(int32_t deviceId) noexcept;

  private:
    std::mutex                                               mMutex;
    std::map<int32_t, std::weak_ptr<CudaMeshResidencyIndex>> mIndices;
};

} // namespace detail
} // namespace ovspatialquery

#endif
