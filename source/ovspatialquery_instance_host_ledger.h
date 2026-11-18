/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_INSTANCE_HOST_LEDGER_H
#define OVSPATIALQUERY_INSTANCE_HOST_LEDGER_H

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <set>
#include <vector>

namespace ovspatialquery
{
namespace detail
{

// Adds count * size to total; false, leaving total unchanged, if the product or the sum overflows.
inline bool addHostCharge(uint64_t count, uint64_t size, uint64_t& total) noexcept
{
    if (count != 0u && size > UINT64_MAX / count)
    {
        return false;
    }
    const uint64_t bytes = count * size;
    if (bytes > UINT64_MAX - total)
    {
        return false;
    }
    total += bytes;
    return true;
}

class HostLease;
class HostTransaction;
class BuildScratchClaim;

enum class BuildScratchOutcome
{
    success,
    budget,
    outOfMemory,
};

// One synchronized byte total against the instance's host budget.
class InstanceHostLedger
{
  public:
    explicit InstanceHostLedger(uint64_t byteBudget) noexcept;

    InstanceHostLedger(const InstanceHostLedger&)            = delete;
    InstanceHostLedger& operator=(const InstanceHostLedger&) = delete;

    bool reserve(uint64_t bytes) noexcept;
    void release(uint64_t bytes) noexcept;

    uint64_t total() const noexcept;

    // The one build-scratch reservation (ADR-0075): a charge sized for the largest claim
    // still held, so structures built one at a time count their scratch once. Adds one
    // claim per entry of `bytes`, growing the charge to the largest claim when needed, or
    // none of them.
    BuildScratchOutcome claimBuildScratch(const std::vector<uint64_t>&    bytes,
                                          std::vector<BuildScratchClaim>& outClaims) noexcept;
    // Held while a claimed structure is built, so builds share the reservation.
    std::unique_lock<std::mutex> lockBuild();

  private:
    friend class BuildScratchClaim;
    // Ends one claim and shrinks the charge to the largest claim left.
    void releaseBuildScratch(uint64_t bytes) noexcept;

    const uint64_t          mByteBudget;
    mutable std::mutex      mMutex;
    uint64_t                mTotal        = 0u;
    uint64_t                mBuildScratch = 0u; // part of mTotal
    std::multiset<uint64_t> mBuildClaims;
    std::mutex              mBuildMutex;
};

// One structure's share of the build-scratch reservation, held until the structure is
// built or destroyed.
class BuildScratchClaim
{
  public:
    BuildScratchClaim() noexcept = default;
    ~BuildScratchClaim();

    BuildScratchClaim(const BuildScratchClaim&)            = delete;
    BuildScratchClaim& operator=(const BuildScratchClaim&) = delete;
    BuildScratchClaim(BuildScratchClaim&& other) noexcept;
    BuildScratchClaim& operator=(BuildScratchClaim&& other) noexcept;

    InstanceHostLedger* ledger() const noexcept { return mLedger; }
    void                reset() noexcept;

  private:
    friend class InstanceHostLedger;
    BuildScratchClaim(InstanceHostLedger& ledger, uint64_t bytes) noexcept : mLedger(&ledger), mBytes(bytes) {}

    InstanceHostLedger* mLedger = nullptr;
    uint64_t            mBytes  = 0u;
};

// Holds a byte charge on a ledger until reset or destroyed.
class HostLease
{
  public:
    HostLease() noexcept = default;
    ~HostLease();

    HostLease(const HostLease&)            = delete;
    HostLease& operator=(const HostLease&) = delete;
    HostLease(HostLease&& other) noexcept;
    HostLease& operator=(HostLease&& other) noexcept;

    explicit operator bool() const noexcept;
    uint64_t bytes() const noexcept;
    // Releases the part of the charge above `bytes`; false, leaving the
    // lease unchanged, when it holds no charge or less than `bytes`.
    bool shrink(uint64_t bytes) noexcept;
    // Moves `bytes` of the charge into `outLease` without releasing them;
    // false, leaving both unchanged, when this lease holds less or
    // `outLease` already holds a charge.
    bool split(uint64_t bytes, HostLease& outLease) noexcept;
    // Takes over the charge of `other` on the same ledger; false, leaving
    // both unchanged, when either holds no charge or their ledgers differ.
    bool absorb(HostLease& other) noexcept;
    void reset() noexcept;

  private:
    friend class HostTransaction;
    HostLease(InstanceHostLedger& ledger, uint64_t bytes) noexcept;

    InstanceHostLedger* mLedger = nullptr;
    uint64_t            mBytes  = 0u;
};

// Sums several charges and admits them as one lease, or none of them.
class HostTransaction
{
  public:
    explicit HostTransaction(InstanceHostLedger& ledger) noexcept;

    HostTransaction(const HostTransaction&)            = delete;
    HostTransaction& operator=(const HostTransaction&) = delete;

    bool reserve(uint64_t bytes) noexcept;
    bool commit(HostLease& outLease) noexcept;

  private:
    InstanceHostLedger& mLedger;
    uint64_t            mBytes     = 0u;
    bool                mValid     = true;
    bool                mCommitted = false;
};

} // namespace detail
} // namespace ovspatialquery

#endif
