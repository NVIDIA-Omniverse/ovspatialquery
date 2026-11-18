/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_FAMILY_BATCH_COLUMNS_H
#define OVSPATIALQUERY_FAMILY_BATCH_COLUMNS_H

#include "ovspatialquery_family_batch_cpu.h"
#include <ovspatialquery/ovspatialquery.h>

#include <cstdint>
#include <memory>

namespace ovspatialquery
{
namespace detail
{

constexpr uint32_t kBatchResultPosition         = 1u;
constexpr uint32_t kBatchResultNormal           = 2u;
constexpr uint32_t kBatchResultDistance         = 4u;
constexpr uint32_t kBatchResultSourceFeature    = 8u;
constexpr uint32_t kBatchResultGeometryIdentity = 16u;
constexpr uint32_t kBatchResultQueryPosition    = 32u;
constexpr uint32_t kBatchResultMaterial         = 64u;
// Not a column: a CUDA plan keeps its columns in mapped host memory (ADR-0077);
// a CPU plan ignores it.
constexpr uint32_t kBatchResultHostMapped = 128u;

enum class FamilyBatchColumnsOutcome : uint8_t
{
    success,
    invalidArgument,
};

struct FamilyBatchColumnConfig
{
    uint64_t maximumQueryCount = 0u;
    uint64_t hitCapacity       = 0u;
    uint32_t resultFields      = 0u;
    HitTruth truth             = HitTruth::boundsCandidate;
};

struct RaycastBatchColumnPolicy
{
    using Result      = RaycastBatchResult;
    using Hit         = RaycastSemanticHit;
    using Reservation = ovspatialquery_raycast_batch_reservation_t;
    using Fetched     = ovspatialquery_raycast_batch_result_t;
    using Columns     = ovspatialquery_raycast_batch_columns_t;
};

struct SphereSweepBatchColumnPolicy
{
    using Result      = SphereSweepBatchResult;
    using Hit         = SphereSweepSemanticHit;
    using Reservation = ovspatialquery_sphere_sweep_batch_reservation_t;
    using Fetched     = ovspatialquery_sphere_sweep_batch_result_t;
    using Columns     = ovspatialquery_sphere_sweep_batch_columns_t;
};

struct OverlapBatchColumnPolicy
{
    using Result      = OverlapBatchResult;
    using Hit         = OverlapSemanticHit;
    using Reservation = ovspatialquery_overlap_batch_reservation_t;
    using Fetched     = ovspatialquery_overlap_batch_result_t;
    using Columns     = ovspatialquery_overlap_batch_columns_t;
};

/** Fixed-capacity host columns owned by one private CPU family result slot. */
template <typename Policy>
class FamilyBatchColumnStorage
{
  public:
    using Config      = FamilyBatchColumnConfig;
    using Result      = typename Policy::Result;
    using Reservation = typename Policy::Reservation;
    using Fetched     = typename Policy::Fetched;

    static bool estimateResidentBytes(Config config, uint64_t& outBytes) noexcept;
    // Throws std::bad_alloc or std::length_error when the columns cannot be allocated.
    static std::unique_ptr<FamilyBatchColumnStorage> create(Config config);

    ~FamilyBatchColumnStorage();

    FamilyBatchColumnStorage(const FamilyBatchColumnStorage&)            = delete;
    FamilyBatchColumnStorage& operator=(const FamilyBatchColumnStorage&) = delete;

    // queryCount is at most the configured maximum; admission checks it.
    void                      begin(uint32_t resultSlot, uint64_t queryCount, ovspatialquery_ordinal_t sourceOrdinal,
                                    Reservation& outReservation) noexcept;
    FamilyBatchColumnsOutcome publish(const Result& semantic) noexcept;
    // The snapshot of the last publish.
    const Fetched& result() const noexcept;

  private:
    struct Impl;
    explicit FamilyBatchColumnStorage(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> mImpl;
};

using RaycastBatchColumnStorage     = FamilyBatchColumnStorage<RaycastBatchColumnPolicy>;
using SphereSweepBatchColumnStorage = FamilyBatchColumnStorage<SphereSweepBatchColumnPolicy>;
using OverlapBatchColumnStorage     = FamilyBatchColumnStorage<OverlapBatchColumnPolicy>;

extern template class FamilyBatchColumnStorage<RaycastBatchColumnPolicy>;
extern template class FamilyBatchColumnStorage<SphereSweepBatchColumnPolicy>;
extern template class FamilyBatchColumnStorage<OverlapBatchColumnPolicy>;

} // namespace detail
} // namespace ovspatialquery

#endif
