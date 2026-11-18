/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_instance_host_ledger.h"

#include <algorithm>
#include <limits>

namespace ovspatialquery
{
namespace detail
{

InstanceHostLedger::InstanceHostLedger(uint64_t byteBudget) noexcept : mByteBudget(byteBudget) {}

bool InstanceHostLedger::reserve(uint64_t bytes) noexcept
{
    std::lock_guard<std::mutex> lock(mMutex);
    if (bytes > mByteBudget - std::min(mByteBudget, mTotal))
    {
        return false;
    }
    mTotal += bytes;
    return true;
}

void InstanceHostLedger::release(uint64_t bytes) noexcept
{
    std::lock_guard<std::mutex> lock(mMutex);
    mTotal -= std::min(bytes, mTotal);
}

uint64_t InstanceHostLedger::total() const noexcept
{
    std::lock_guard<std::mutex> lock(mMutex);
    return mTotal;
}

BuildScratchOutcome InstanceHostLedger::claimBuildScratch(const std::vector<uint64_t>&    bytes,
                                                          std::vector<BuildScratchClaim>& outClaims) noexcept
{
    std::vector<BuildScratchClaim> claims;
    try
    {
        claims.reserve(bytes.size());
    }
    catch (...)
    {
        return BuildScratchOutcome::outOfMemory;
    }
    {
        std::lock_guard<std::mutex> lock(mMutex);
        const uint64_t              largest = bytes.empty() ? 0u : *std::max_element(bytes.begin(), bytes.end());
        const uint64_t              growth  = largest > mBuildScratch ? largest - mBuildScratch : 0u;
        if (growth > mByteBudget - std::min(mByteBudget, mTotal))
        {
            return BuildScratchOutcome::budget;
        }
        try
        {
            for (uint64_t claim : bytes)
            {
                mBuildClaims.insert(claim);
                claims.push_back(BuildScratchClaim(*this, claim));
            }
        }
        catch (...)
        {
            for (BuildScratchClaim& claim : claims)
            {
                mBuildClaims.erase(mBuildClaims.find(claim.mBytes));
                claim.mLedger = nullptr;
            }
            return BuildScratchOutcome::outOfMemory;
        }
        mTotal += growth;
        mBuildScratch += growth;
    }
    outClaims = std::move(claims);
    return BuildScratchOutcome::success;
}

std::unique_lock<std::mutex> InstanceHostLedger::lockBuild() { return std::unique_lock<std::mutex>(mBuildMutex); }

void InstanceHostLedger::releaseBuildScratch(uint64_t bytes) noexcept
{
    std::lock_guard<std::mutex> lock(mMutex);
    const auto                  found = mBuildClaims.find(bytes);
    if (found != mBuildClaims.end())
    {
        mBuildClaims.erase(found);
    }
    const uint64_t largest = mBuildClaims.empty() ? 0u : *mBuildClaims.rbegin();
    if (largest < mBuildScratch)
    {
        mTotal -= std::min(mTotal, mBuildScratch - largest);
        mBuildScratch = largest;
    }
}

BuildScratchClaim::~BuildScratchClaim() { reset(); }

BuildScratchClaim::BuildScratchClaim(BuildScratchClaim&& other) noexcept : mLedger(other.mLedger), mBytes(other.mBytes)
{
    other.mLedger = nullptr;
}

BuildScratchClaim& BuildScratchClaim::operator=(BuildScratchClaim&& other) noexcept
{
    if (this != &other)
    {
        reset();
        mLedger       = other.mLedger;
        mBytes        = other.mBytes;
        other.mLedger = nullptr;
    }
    return *this;
}

void BuildScratchClaim::reset() noexcept
{
    if (mLedger != nullptr)
    {
        mLedger->releaseBuildScratch(mBytes);
        mLedger = nullptr;
    }
}

HostLease::HostLease(InstanceHostLedger& ledger, uint64_t bytes) noexcept : mLedger(&ledger), mBytes(bytes) {}

HostLease::~HostLease() { reset(); }

HostLease::HostLease(HostLease&& other) noexcept : mLedger(other.mLedger), mBytes(other.mBytes)
{
    other.mLedger = nullptr;
    other.mBytes  = 0u;
}

HostLease& HostLease::operator=(HostLease&& other) noexcept
{
    if (this != &other)
    {
        reset();
        mLedger       = other.mLedger;
        mBytes        = other.mBytes;
        other.mLedger = nullptr;
        other.mBytes  = 0u;
    }
    return *this;
}

HostLease::operator bool() const noexcept { return mLedger != nullptr; }

uint64_t HostLease::bytes() const noexcept { return mBytes; }

bool HostLease::shrink(uint64_t bytes) noexcept
{
    if (mLedger == nullptr || bytes > mBytes)
    {
        return false;
    }
    mLedger->release(mBytes - bytes);
    mBytes = bytes;
    return true;
}

bool HostLease::split(uint64_t bytes, HostLease& outLease) noexcept
{
    if (mLedger == nullptr || bytes > mBytes || outLease)
    {
        return false;
    }
    outLease = HostLease(*mLedger, bytes);
    mBytes -= bytes;
    return true;
}

bool HostLease::absorb(HostLease& other) noexcept
{
    if (mLedger == nullptr || other.mLedger != mLedger || &other == this)
    {
        return false;
    }
    mBytes += other.mBytes;
    other.mLedger = nullptr;
    other.mBytes  = 0u;
    return true;
}

void HostLease::reset() noexcept
{
    if (mLedger != nullptr)
    {
        mLedger->release(mBytes);
        mLedger = nullptr;
        mBytes  = 0u;
    }
}

HostTransaction::HostTransaction(InstanceHostLedger& ledger) noexcept : mLedger(ledger) {}

bool HostTransaction::reserve(uint64_t bytes) noexcept
{
    if (!mValid || mCommitted || bytes > std::numeric_limits<uint64_t>::max() - mBytes)
    {
        mValid = false;
        return false;
    }
    mBytes += bytes;
    return true;
}

bool HostTransaction::commit(HostLease& outLease) noexcept
{
    if (!mValid || mCommitted || outLease || !mLedger.reserve(mBytes))
    {
        return false;
    }
    outLease   = HostLease(mLedger, mBytes);
    mCommitted = true;
    return true;
}

} // namespace detail
} // namespace ovspatialquery
