/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "oracle_backend.h"
#include "ovspatialquery_target_resolver.h"
#include "test_failure.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

namespace
{

#if defined(OVSPATIALQUERY_ENABLE_TARGET_RESOLVER_TEST_HOOK)
struct ResolverBarrier
{
    std::atomic<bool> firstPinReached{false};
    std::atomic<bool> releaseResolver{false};
};

ResolverBarrier* gResolverBarrier = nullptr;

void blockAfterFirstPin()
{
    gResolverBarrier->firstPinReached.store(true, std::memory_order_release);
    while (!gResolverBarrier->releaseResolver.load(std::memory_order_acquire))
    {
        std::this_thread::yield();
    }
}
#endif

void testEmptyPopulationHeadIsNotReady(ovspatialquery_backend_runtime_t* runtime)
{
    ovspatialquery::detail::PopulationTable populations(runtime);
    ovspatialquery::detail::PopulationKey   population;
    REQUIRE(populations.create(2u, population) == OVSPATIALQUERY_API_SUCCESS);

    const ovspatialquery::detail::TargetResolver resolver(populations);
    ovspatialquery::detail::ResolvedTarget       resolved;
    const ovspatialquery::detail::TargetRequest  request{{population, {}},
                                                         {ovspatialquery::detail::TargetSelectionKind::head, 0}};
    REQUIRE(resolver.resolve(request, resolved) == ovspatialquery::detail::TargetResolveOutcome::notReady);
    REQUIRE(!resolved.generation.initialized());
    REQUIRE(resolved.ordinal == 0);

    const ovspatialquery::detail::TargetRequest exactRequest{
        {population, {}}, {ovspatialquery::detail::TargetSelectionKind::retainedExact, 17}};
    REQUIRE(resolver.resolve(exactRequest, resolved) ==
            ovspatialquery::detail::TargetResolveOutcome::ordinalNotRetained);
    REQUIRE(!resolved.generation.initialized());
    REQUIRE(resolved.ordinal == 0);

    std::shared_ptr<void> retired;
    REQUIRE(populations.release(population, retired) == OVSPATIALQUERY_API_SUCCESS);
}

std::vector<ovspatialquery::detail::PopulationObject> objects(uint64_t firstId, uint64_t secondId)
{
    return {
        {{firstId}, {{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}, {}, {}, {}, {}},
        {{secondId}, {{2.0, 0.0, 0.0}, {3.0, 1.0, 1.0}}, {}, {}, {}, {}},
    };
}

std::shared_ptr<const std::vector<uint64_t>> selected(std::initializer_list<uint64_t> ids)
{
    return std::make_shared<const std::vector<uint64_t>>(ids);
}

void publish(ovspatialquery::detail::PopulationTable& populations, ovspatialquery::detail::PopulationKey population,
             ovspatialquery_ordinal_t ordinal, std::vector<ovspatialquery::detail::PopulationObject> snapshot,
             std::vector<ovspatialquery::detail::GenerationViewMembership> memberships)
{
    ovspatialquery::detail::GenerationLease base;
    REQUIRE(populations.beginUpdate(population, ordinal, nullptr, base) == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery::detail::GenerationLease generation;
    REQUIRE(populations.buildGeneration(ordinal, std::move(snapshot), std::move(memberships), generation, {}, {}) ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(populations.publishUpdate(population, generation) == OVSPATIALQUERY_API_SUCCESS);
    generation.reset();
    base.reset();
    populations.finishUpdate(population);
}

void testExactSelectionAndGenerationOwnedMembership(ovspatialquery_backend_runtime_t* runtime)
{
    ovspatialquery::detail::PopulationTable populations(runtime);
    ovspatialquery::detail::PopulationKey   population;
    REQUIRE(populations.create(1u, population) == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery::detail::HandleDirectory handles;
    ovspatialquery_population_handle_t      populationHandle = OVSPATIALQUERY_INVALID_POPULATION;
    REQUIRE(handles.createPopulation(population, populationHandle, ovspatialquery::detail::PopulationSourceBinding{}) ==
            OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_view_handle_t viewHandle = OVSPATIALQUERY_INVALID_VIEW;
    REQUIRE(handles.createSourceView(population, std::make_shared<uint32_t>(1u), viewHandle) ==
            OVSPATIALQUERY_API_SUCCESS);
    std::shared_ptr<ovspatialquery::detail::ViewState> view;
    REQUIRE(handles.resolveView(viewHandle, view) == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_view_handle_t unreadyViewHandle = OVSPATIALQUERY_INVALID_VIEW;
    REQUIRE(handles.createSourceView(population, std::make_shared<uint32_t>(2u), unreadyViewHandle) ==
            OVSPATIALQUERY_API_SUCCESS);
    std::shared_ptr<ovspatialquery::detail::ViewState> unreadyView;
    REQUIRE(handles.resolveView(unreadyViewHandle, unreadyView) == OVSPATIALQUERY_API_SUCCESS);

    publish(populations, population, 10, objects(101u, 102u), {{viewHandle, selected({101u})}});
    publish(populations, population, 20, objects(201u, 202u), {{viewHandle, selected({202u})}});

    const ovspatialquery::detail::TargetResolver resolver(populations);
    ovspatialquery::detail::ResolvedTarget       head;
    REQUIRE(resolver.resolve({{population, view}, {ovspatialquery::detail::TargetSelectionKind::head, 0}}, head) ==
            ovspatialquery::detail::TargetResolveOutcome::success);

    ovspatialquery::detail::ResolvedTarget unready;
    REQUIRE(
        resolver.resolve({{population, unreadyView}, {ovspatialquery::detail::TargetSelectionKind::retainedExact, 20}},
                         unready) == ovspatialquery::detail::TargetResolveOutcome::ordinalNotRetained);
    REQUIRE(!unready.generation.initialized());
    REQUIRE(resolver.resolve({{population, unreadyView}, {ovspatialquery::detail::TargetSelectionKind::head, 0}},
                             unready) == ovspatialquery::detail::TargetResolveOutcome::notReady);

    // A held head lease keeps its ordinal and view membership across a later
    // publication. Retained membership and eviction are proven through query
    // results by family_batch_semantics_test.
    publish(populations, population, 30, objects(301u, 302u), {{viewHandle, selected({301u, 302u})}});
    REQUIRE(head.ordinal == 20 && !head.generation.includesObject(0u) && head.generation.includesObject(1u));

    head = {};
    REQUIRE(handles.invalidateView(unreadyViewHandle) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(handles.invalidateView(viewHandle) == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery::detail::PopulationSourceBinding retiredSource;
    REQUIRE(handles.invalidatePopulation(populationHandle, retiredSource) == OVSPATIALQUERY_API_SUCCESS);
    std::shared_ptr<void> retiredPopulation;
    REQUIRE(populations.release(population, retiredPopulation) == OVSPATIALQUERY_API_SUCCESS);
}

void testMembershipCapacityIsResident()
{
    std::shared_ptr<std::vector<uint64_t>> ids = std::make_shared<std::vector<uint64_t>>();
    ids->reserve(4096u);
    ids->push_back(17u);
    const ovspatialquery::detail::GenerationViewMembership membership{{101u}, ids};
    REQUIRE(membership.residentByteCount() >= static_cast<uint64_t>(ids->capacity()) * sizeof(uint64_t));
    REQUIRE(membership.residentByteCount() > static_cast<uint64_t>(ids->size()) * sizeof(uint64_t));
}

void testAtomicMultiTargetAdmission(ovspatialquery_backend_runtime_t* runtime)
{
    ovspatialquery::detail::PopulationTable populations(runtime);
    ovspatialquery::detail::PopulationKey   first;
    ovspatialquery::detail::PopulationKey   second;
    REQUIRE(populations.create(2u, first) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(populations.create(2u, second) == OVSPATIALQUERY_API_SUCCESS);

    const ovspatialquery_view_handle_t                       firstViewHandle = ovspatialquery_view_handle_t{101u};
    const std::shared_ptr<ovspatialquery::detail::ViewState> firstView =
        std::make_shared<ovspatialquery::detail::ViewState>();
    firstView->handle    = firstViewHandle;
    firstView->parent    = first;
    firstView->automatic = true;

    publish(populations, first, 10, objects(101u, 102u), {{firstViewHandle, selected({101u})}});
    publish(populations, first, 20, objects(201u, 202u), {{firstViewHandle, selected({202u})}});
    publish(populations, second, 7, objects(701u, 702u), {});

    const ovspatialquery::detail::TargetResolver  resolver(populations);
    const ovspatialquery::detail::TargetSelection head{ovspatialquery::detail::TargetSelectionKind::head, 0};
    const ovspatialquery::detail::TargetSelection exact10{ovspatialquery::detail::TargetSelectionKind::retainedExact,
                                                          10};
    const ovspatialquery::detail::TargetSelection exact20{ovspatialquery::detail::TargetSelectionKind::retainedExact,
                                                          20};

    using ovspatialquery::detail::TargetRequest;
    using RequestPair       = std::array<TargetRequest, 2u>;
    const auto resolvedNone = [](const std::array<ovspatialquery::detail::ResolvedTarget, 2u>& targets)
    { return !targets[0].generation.initialized() && !targets[1].generation.initialized(); };

    std::array<ovspatialquery::detail::ResolvedTarget, 2u> resolved{};
    REQUIRE(
        resolver.resolvePair(RequestPair{TargetRequest{{first, {}}, head}, TargetRequest{{first, firstView}, exact20}},
                             resolved) == ovspatialquery::detail::TargetResolveOutcome::success);
    REQUIRE(resolved[0].ordinal == 20 && resolved[1].ordinal == 20);
    REQUIRE(resolved[1].generation.includesObject(1u));

    REQUIRE(resolver.resolvePair(RequestPair{TargetRequest{{first, {}}, head}, TargetRequest{{second, {}}, head}},
                                 resolved) == ovspatialquery::detail::TargetResolveOutcome::success);
    REQUIRE(resolved[0].ordinal == 20 && resolved[1].ordinal == 7);

    REQUIRE(
        resolver.resolvePair(RequestPair{TargetRequest{{first, {}}, head}, TargetRequest{{first, firstView}, exact10}},
                             resolved) == ovspatialquery::detail::TargetResolveOutcome::ordinalMismatch);
    REQUIRE(resolvedNone(resolved));

    REQUIRE(
        resolver.resolvePair(
            RequestPair{TargetRequest{{first, {}}, exact20},
                        TargetRequest{{second, {}}, {ovspatialquery::detail::TargetSelectionKind::retainedExact, 99}}},
            resolved) == ovspatialquery::detail::TargetResolveOutcome::ordinalNotRetained);
    REQUIRE(resolvedNone(resolved));

    ovspatialquery::detail::ResolvedTarget single;
    REQUIRE(resolver.resolve({{first, {}}, head}, single) == ovspatialquery::detail::TargetResolveOutcome::success);
    REQUIRE(resolver.resolve({{ovspatialquery::detail::PopulationKey{}, {}}, head}, single) ==
            ovspatialquery::detail::TargetResolveOutcome::invalid);
    REQUIRE(!single.generation.initialized());

    std::atomic<bool> concurrentSuccess{true};
    const auto        admit = [&resolver, &first, &second, &head, &concurrentSuccess](bool reverse)
    {
        for (uint32_t iteration = 0u; iteration < 1000u; ++iteration)
        {
            RequestPair requests{TargetRequest{{first, {}}, head}, TargetRequest{{second, {}}, head}};
            if (reverse)
            {
                std::swap(requests[0], requests[1]);
            }
            std::array<ovspatialquery::detail::ResolvedTarget, 2u> admitted{};
            if (resolver.resolvePair(requests, admitted) != ovspatialquery::detail::TargetResolveOutcome::success ||
                !admitted[0].generation.initialized() || !admitted[1].generation.initialized())
            {
                concurrentSuccess.store(false);
                return;
            }
        }
    };
    std::thread forward(admit, false);
    std::thread reverse(admit, true);
    forward.join();
    reverse.join();
    REQUIRE(concurrentSuccess.load());

#if defined(OVSPATIALQUERY_ENABLE_TARGET_RESOLVER_TEST_HOOK)
    ResolverBarrier resolverBarrier;
    gResolverBarrier = &resolverBarrier;
    ovspatialquery::detail::setTargetResolverAfterFirstPinHook(&blockAfterFirstPin);
    std::array<ovspatialquery::detail::ResolvedTarget, 2u> atomicAdmission{};
    std::atomic<bool>                                      admissionComplete{false};
    std::thread                                            admitting(
        [&]()
        {
            REQUIRE(
                resolver.resolvePair(RequestPair{TargetRequest{{first, {}}, head}, TargetRequest{{first, {}}, head}},
                                                                                atomicAdmission) == ovspatialquery::detail::TargetResolveOutcome::success);
            admissionComplete.store(true, std::memory_order_release);
        });
    while (!resolverBarrier.firstPinReached.load(std::memory_order_acquire))
    {
        std::this_thread::yield();
    }

    std::atomic<bool> publicationAttempted{false};
    std::atomic<bool> publicationComplete{false};
    std::thread       publisher(
        [&]()
        {
            publicationAttempted.store(true, std::memory_order_release);
            publish(populations, first, 21, objects(211u, 212u), {});
            publicationComplete.store(true, std::memory_order_release);
        });
    while (!publicationAttempted.load(std::memory_order_acquire))
    {
        std::this_thread::yield();
    }
    // The resolver holds the table between its two pins, so the publication
    // stays blocked for as long as the resolver is parked; a window this long
    // lets an unblocked publication finish. Spun, not slept, so Windows timer
    // granularity does not stretch it.
    const std::chrono::steady_clock::time_point parkedUntil =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(50);
    while (std::chrono::steady_clock::now() < parkedUntil)
    {
        REQUIRE(!publicationComplete.load(std::memory_order_acquire));
        std::this_thread::yield();
    }
    REQUIRE(!publicationComplete.load(std::memory_order_acquire));
    REQUIRE(!admissionComplete.load(std::memory_order_acquire));

    resolverBarrier.releaseResolver.store(true, std::memory_order_release);
    admitting.join();
    publisher.join();
    ovspatialquery::detail::setTargetResolverAfterFirstPinHook(nullptr);
    gResolverBarrier = nullptr;
    REQUIRE(publicationComplete.load(std::memory_order_acquire));
    REQUIRE(atomicAdmission[0].generation.initialized() && atomicAdmission[1].generation.initialized());
    REQUIRE(atomicAdmission[0].ordinal == 20 && atomicAdmission[1].ordinal == 20);
#endif

    resolved = {};
    single   = {};
    std::shared_ptr<void> retired;
    REQUIRE(populations.release(first, retired) == OVSPATIALQUERY_API_SUCCESS);
    retired.reset();
    REQUIRE(populations.release(second, retired) == OVSPATIALQUERY_API_SUCCESS);
}

} // namespace

int main()
{
    ovspatialquery_backend_runtime_t* runtime = nullptr;
    REQUIRE(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS);
    testEmptyPopulationHeadIsNotReady(runtime);
    testExactSelectionAndGenerationOwnedMembership(runtime);
    testMembershipCapacityIsResident();
    testAtomicMultiTargetAdmission(runtime);
    ovspatialquery_test_oracle_destroy_runtime(runtime);
    return 0;
}
