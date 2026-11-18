/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../support/native/test_failure.h"
#include "ovspatialquery_instance_host_ledger.h"

#include <atomic>
#include <cstdlib>
#include <thread>
#include <vector>

namespace
{

void testAtomicTransactionAndLease()
{
    using namespace ovspatialquery::detail;
    InstanceHostLedger ledger(100u);
    HostTransaction    first(ledger);
    REQUIRE(first.reserve(40u) && first.reserve(50u));
    HostLease firstLease;
    REQUIRE(first.commit(firstLease));
    REQUIRE(ledger.total() == 90u);

    HostTransaction rejected(ledger);
    REQUIRE(rejected.reserve(11u));
    HostLease rejectedLease;
    REQUIRE(!rejected.commit(rejectedLease));
    REQUIRE(!rejectedLease && ledger.total() == 90u);
    firstLease.reset();
    REQUIRE(ledger.total() == 0u);
}

void testConcurrentAdmissionHasOneWinner()
{
    using namespace ovspatialquery::detail;
    InstanceHostLedger    ledger(100u);
    std::atomic<uint32_t> ready{0u};
    std::atomic<bool>     start{false};
    HostLease             leases[2];
    bool                  admitted[2] = {false, false};
    const auto            compete     = [&](uint32_t index)
    {
        HostTransaction transaction(ledger);
        REQUIRE(transaction.reserve(80u));
        ready.fetch_add(1u, std::memory_order_release);
        while (!start.load(std::memory_order_acquire))
        {
            std::this_thread::yield();
        }
        admitted[index] = transaction.commit(leases[index]);
    };
    std::thread first(compete, 0u);
    std::thread second(compete, 1u);
    while (ready.load(std::memory_order_acquire) != 2u)
    {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    first.join();
    second.join();
    REQUIRE(admitted[0] != admitted[1] && ledger.total() == 80u);
    leases[0].reset();
    leases[1].reset();
    REQUIRE(ledger.total() == 0u);
}

// Build scratch counts once, for the largest claim held (ADR-0075): claims grow the charge
// only past it, and ending a claim shrinks it to the largest one left.
void testBuildScratchFollowsTheLargestClaim()
{
    using namespace ovspatialquery::detail;
    InstanceHostLedger             ledger(100u);
    std::vector<BuildScratchClaim> first, second, refused;
    REQUIRE(ledger.claimBuildScratch({30u, 60u}, first) == BuildScratchOutcome::success);
    REQUIRE(first.size() == 2u && ledger.total() == 60u);
    REQUIRE(ledger.claimBuildScratch({60u, 20u}, second) == BuildScratchOutcome::success);
    REQUIRE(ledger.total() == 60u);
    REQUIRE(ledger.claimBuildScratch({101u}, refused) == BuildScratchOutcome::budget);
    REQUIRE(refused.empty() && ledger.total() == 60u);
    first[1].reset();
    REQUIRE(ledger.total() == 60u); // the second 60-byte claim remains
    second[0].reset();
    REQUIRE(ledger.total() == 30u);
    first.clear();
    second.clear();
    REQUIRE(ledger.total() == 0u);
}

} // namespace

int main()
{
    testAtomicTransactionAndLease();
    testConcurrentAdmissionHasOneWinner();
    testBuildScratchFollowsTheLargestClaim();
    return 0;
}
