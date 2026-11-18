/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../support/native/test_failure.h"
#include "oracle_backend.h"
#include "ovspatialquery_pair_materialization.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

namespace
{

using ovspatialquery::detail::PairEndpointRelation;
using ovspatialquery::detail::PairMaterializationOutcome;
using ovspatialquery::detail::PairMaterializationRequest;
using ovspatialquery::detail::PopulationKey;
using ovspatialquery::detail::PopulationObject;
using ovspatialquery::detail::ResolvedTarget;

[[noreturn]] void fail(const char* message)
{
    std::fprintf(stderr, "FAIL: %s\n", message);
    OVSPATIALQUERY_TEST_EXIT_FAILURE();
}

void require(bool condition, const char* message)
{
    if (!condition)
    {
        fail(message);
    }
}

ovspatialquery_pair_t pair(uint64_t first, uint64_t second) { return ovspatialquery_pair_t{{first}, {second}}; }

PopulationObject object(uint64_t id, double minimumX, double maximumX, uint64_t participationBits = UINT64_MAX)
{
    PopulationObject value{};
    value.id     = {id};
    value.bounds = {
        {minimumX, 0.0, 0.0},
        {maximumX, 1.0, 1.0},
    };
    value.participationBits = {{participationBits, 0u}};
    return value;
}

class Fixture
{
  public:
    Fixture()
        : runtime(createRuntime()), populations(std::make_unique<ovspatialquery::detail::PopulationTable>(runtime))
    {}

    ~Fixture()
    {
        populations.reset();
        ovspatialquery_test_oracle_destroy_runtime(runtime);
    }

    ResolvedTarget resolved(PopulationKey population, ovspatialquery_ordinal_t ordinal,
                            std::vector<PopulationObject> objects)
    {
        ovspatialquery::detail::GenerationLease generation;
        require(populations->buildGeneration(ordinal, std::move(objects), {}, generation) == OVSPATIALQUERY_API_SUCCESS,
                "generation build failed");
        return ResolvedTarget{population, ordinal, std::move(generation)};
    }

  private:
    static ovspatialquery_backend_runtime_t* createRuntime()
    {
        ovspatialquery_backend_runtime_t* result = nullptr;
        require(ovspatialquery_test_oracle_create_runtime(&result) == OVSPATIALQUERY_API_SUCCESS,
                "oracle runtime creation failed");
        return result;
    }

    ovspatialquery_backend_runtime_t*                        runtime;
    std::unique_ptr<ovspatialquery::detail::PopulationTable> populations;
};

void selectIds(ResolvedTarget& target, std::initializer_list<uint64_t> ids)
{
    target.generation.selectIds(std::make_shared<const std::vector<uint64_t>>(ids));
}

void testSamePopulationDistinctSelectionsPreserveDirection()
{
    Fixture                             fixture;
    const std::vector<PopulationObject> objects{
        object(1u, 0.0, 1.0),
        object(2u, 0.5, 1.5),
    };
    PairMaterializationRequest forward{};
    forward.first.target  = fixture.resolved({3u, 1u}, 8, objects);
    forward.second.target = fixture.resolved({3u, 1u}, 8, objects);
    selectIds(forward.first.target, {1u});
    selectIds(forward.second.target, {2u});
    forward.relation           = PairEndpointRelation::samePopulationDistinctSelections;
    forward.activePairCapacity = 1u;

    std::vector<ovspatialquery_pair_t> result;
    require(ovspatialquery::detail::materializePairSet(forward, result) == PairMaterializationOutcome::success &&
                result == std::vector<ovspatialquery_pair_t>{pair(1u, 2u)},
            "same-population A/B selection lost endpoint direction");

    PairMaterializationRequest reverse{};
    reverse.first.target  = fixture.resolved({3u, 1u}, 8, objects);
    reverse.second.target = fixture.resolved({3u, 1u}, 8, objects);
    selectIds(reverse.first.target, {2u});
    selectIds(reverse.second.target, {1u});
    reverse.relation           = PairEndpointRelation::samePopulationDistinctSelections;
    reverse.activePairCapacity = 1u;
    require(ovspatialquery::detail::materializePairSet(reverse, result) == PairMaterializationOutcome::success &&
                result == std::vector<ovspatialquery_pair_t>{pair(2u, 1u)},
            "same-population B/A selection was incorrectly canonicalized");

    PairMaterializationRequest overlapping{};
    overlapping.first.target  = fixture.resolved({3u, 1u}, 8, objects);
    overlapping.second.target = fixture.resolved({3u, 1u}, 8, objects);
    selectIds(overlapping.first.target, {1u, 2u});
    selectIds(overlapping.second.target, {1u, 2u});
    overlapping.relation           = PairEndpointRelation::samePopulationDistinctSelections;
    overlapping.activePairCapacity = 2u;
    require(ovspatialquery::detail::materializePairSet(overlapping, result) == PairMaterializationOutcome::success &&
                result == std::vector<ovspatialquery_pair_t>{pair(1u, 2u), pair(2u, 1u)},
            "overlapping directed selections lost reverse pair or kept self-pairs");
}

void testIdenticalSelectionProducesOneSortedUnorderedPairSet()
{
    Fixture                             fixture;
    const std::vector<PopulationObject> objects{
        object(3u, 0.0, 1.0),
        object(1u, 0.0, 1.0),
        object(2u, 0.0, 1.0),
        object(2u, 0.0, 1.0),
    };
    PairMaterializationRequest request{};
    request.first.target       = fixture.resolved({4u, 1u}, 9, objects);
    request.second.target      = fixture.resolved({4u, 1u}, 9, objects);
    request.first.queryBits    = {{UINT64_C(0x4), 0u}};
    request.second.queryBits   = {{UINT64_C(0x4), 0u}};
    request.relation           = PairEndpointRelation::identicalSelection;
    request.activePairCapacity = 8u;

    std::vector<ovspatialquery_pair_t> result;
    require(ovspatialquery::detail::materializePairSet(request, result) == PairMaterializationOutcome::success,
            "identical selection did not materialize");
    require(result == std::vector<ovspatialquery_pair_t>{pair(1u, 2u), pair(1u, 3u), pair(2u, 3u)},
            "identical selection was not sorted, unique, and unordered despite a "
            "duplicate source entry");
}

// Objects need not ascend by ID: endpoint admission falls back from one
// ordered walk over the selection to a search per object once IDs stop
// ascending (a selected ID absent from the generation is skipped either way),
// and pairs a traversal reports out of identity order take the complete sort.
void testUnorderedGenerationsAdmitAndSortLikeOrderedOnes()
{
    Fixture                             fixture;
    const std::vector<PopulationObject> objects{
        object(2u, 0.0, 1.0), object(5u, 0.5, 1.5, UINT64_C(0x1)), object(9u, 0.0, 1.0),
        object(3u, 0.8, 2.0), object(7u, 0.9, 1.1, UINT64_C(0x2)), object(11u, 1.3, 1.4),
    };
    for (const double tolerance : {0.0, 0.25})
    {
        PairMaterializationRequest identical{};
        identical.first.target  = fixture.resolved({5u, 1u}, 3, objects);
        identical.second.target = fixture.resolved({5u, 1u}, 3, objects);
        selectIds(identical.first.target, {3u, 5u, 6u, 7u, 9u, 11u});
        selectIds(identical.second.target, {3u, 5u, 6u, 7u, 9u, 11u});
        identical.relation           = PairEndpointRelation::identicalSelection;
        identical.contactTolerance   = tolerance;
        identical.activePairCapacity = 16u;
        std::vector<ovspatialquery_pair_t> expected{pair(3u, 5u), pair(3u, 7u), pair(3u, 9u),  pair(3u, 11u),
                                                    pair(5u, 7u), pair(5u, 9u), pair(5u, 11u), pair(7u, 9u)};
        if (tolerance > 0.0)
        {
            expected.push_back(pair(7u, 11u));
        }
        std::vector<ovspatialquery_pair_t> result;
        require(ovspatialquery::detail::materializePairSet(identical, result) == PairMaterializationOutcome::success &&
                    result == expected,
                "an unordered generation changed the identical-selection pair set or its order");

        PairMaterializationRequest directed{};
        directed.first.target  = fixture.resolved({5u, 1u}, 3, objects);
        directed.second.target = fixture.resolved({5u, 1u}, 3, objects);
        selectIds(directed.first.target, {3u, 5u, 6u, 7u, 9u, 11u});
        directed.first.queryBits    = {{UINT64_C(0x1), 0u}};
        directed.second.queryBits   = {{UINT64_C(0x2), 0u}};
        directed.relation           = PairEndpointRelation::samePopulationDistinctSelections;
        directed.contactTolerance   = tolerance;
        directed.activePairCapacity = 16u;
        expected = {pair(3u, 2u), pair(3u, 7u),  pair(3u, 9u), pair(3u, 11u), pair(5u, 2u), pair(5u, 3u), pair(5u, 7u),
                    pair(5u, 9u), pair(5u, 11u), pair(9u, 2u), pair(9u, 3u),  pair(9u, 7u), pair(11u, 3u)};
        if (tolerance > 0.0)
        {
            expected.push_back(pair(11u, 7u));
        }
        require(ovspatialquery::detail::materializePairSet(directed, result) == PairMaterializationOutcome::success &&
                    result == expected,
                "an unordered generation changed the directed selection pair set or its order");
    }
}

PairMaterializationRequest crossPopulationRequest(Fixture& fixture, std::vector<PopulationObject> firstObjects,
                                                  std::vector<PopulationObject> secondObjects, uint64_t capacity)
{
    PairMaterializationRequest request{};
    request.first.target       = fixture.resolved({10u, 1u}, 12, std::move(firstObjects));
    request.second.target      = fixture.resolved({11u, 1u}, 50, std::move(secondObjects));
    request.relation           = PairEndpointRelation::distinctPopulations;
    request.activePairCapacity = capacity;
    return request;
}

void testGenerationMembershipAndCapturedParticipationBitsAreEndpointTruth()
{
    Fixture                             fixture;
    const std::vector<PopulationObject> retainedObjects{
        object(10u, 0.0, 1.0, UINT64_C(0x1)),
        object(20u, 0.0, 1.0, UINT64_C(0x2)),
    };
    const std::vector<PopulationObject> headObjects{
        object(10u, 0.0, 1.0, UINT64_C(0x4)),
        object(20u, 0.0, 1.0, UINT64_C(0x1)),
    };
    const std::vector<PopulationObject> other{
        object(99u, 0.0, 1.0, UINT64_C(0x8)),
    };

    PairMaterializationRequest retained = crossPopulationRequest(fixture, retainedObjects, other, 1u);
    retained.first.queryBits            = {{UINT64_C(0x1), 0u}};
    selectIds(retained.first.target, {10u});
    std::vector<ovspatialquery_pair_t> result;
    require(ovspatialquery::detail::materializePairSet(retained, result) == PairMaterializationOutcome::success &&
                result == std::vector<ovspatialquery_pair_t>{pair(10u, 99u)},
            "retained membership or captured bits were not honored");

    PairMaterializationRequest head = crossPopulationRequest(fixture, headObjects, other, 1u);
    head.first.queryBits            = {{UINT64_C(0x1), 0u}};
    selectIds(head.first.target, {20u});
    require(ovspatialquery::detail::materializePairSet(head, result) == PairMaterializationOutcome::success &&
                result == std::vector<ovspatialquery_pair_t>{pair(20u, 99u)},
            "head membership or newly captured bits were not honored");
}

void testSealedMembershipIsAuthoritativeOverParticipationBits()
{
    Fixture                    fixture;
    PairMaterializationRequest request = crossPopulationRequest(fixture,
                                                                {
                                                                    object(1u, 0.0, 1.0, UINT64_C(0x1)),
                                                                    object(2u, 0.0, 1.0, UINT64_C(0x2)),
                                                                    object(3u, 0.0, 1.0, UINT64_C(0x4)),
                                                                },
                                                                {
                                                                    object(8u, 0.0, 1.0, UINT64_C(0x8)),
                                                                    object(9u, 0.0, 1.0, UINT64_C(0x10)),
                                                                },
                                                                2u);
    selectIds(request.first.target, {1u, 2u});
    selectIds(request.second.target, {8u});
    request.first.queryBits  = {{UINT64_C(0x2), 0u}};
    request.second.queryBits = {{UINT64_C(0x8), 0u}};

    std::vector<ovspatialquery_pair_t> result;
    require(ovspatialquery::detail::materializePairSet(request, result) == PairMaterializationOutcome::success &&
                result == std::vector<ovspatialquery_pair_t>{pair(2u, 8u)},
            "sealed memberships or role-specific participation bits were ignored");

    request.first.queryBits  = {{0u, 0u}};
    request.second.queryBits = {{UINT64_C(0x10), 0u}};
    require(ovspatialquery::detail::materializePairSet(request, result) == PairMaterializationOutcome::success &&
                result.empty(),
            "second-role membership was replaced by participation truth");
}

void testZeroAndIntersectingQueryBitsFilterEachEndpoint()
{
    Fixture                             fixture;
    const std::vector<PopulationObject> first{
        object(3u, 0.0, 1.0, 0u),
        object(2u, 0.0, 1.0, UINT64_C(0x2)),
        object(1u, 0.0, 1.0, UINT64_C(0x4)),
    };
    const std::vector<PopulationObject> second{
        object(8u, 0.0, 1.0, UINT64_C(0x8)),
    };
    PairMaterializationRequest         request = crossPopulationRequest(fixture, first, second, 3u);
    std::vector<ovspatialquery_pair_t> result;
    require(ovspatialquery::detail::materializePairSet(request, result) == PairMaterializationOutcome::success &&
                result == std::vector<ovspatialquery_pair_t>{pair(1u, 8u), pair(2u, 8u), pair(3u, 8u)},
            "zero query bits did not accept all endpoint objects");

    request.first.queryBits  = {{UINT64_C(0x2), 0u}};
    request.second.queryBits = {{UINT64_C(0x8), 0u}};
    require(ovspatialquery::detail::materializePairSet(request, result) == PairMaterializationOutcome::success &&
                result == std::vector<ovspatialquery_pair_t>{pair(2u, 8u)},
            "intersecting query bits did not filter both endpoints");

    request.second.queryBits = {{UINT64_C(0x10), 0u}};
    require(ovspatialquery::detail::materializePairSet(request, result) == PairMaterializationOutcome::success &&
                result.empty(),
            "nonintersecting second-endpoint bits admitted an object");
}

void testToleranceIsInclusiveFiniteAndAppliedExactlyOnce()
{
    Fixture                             fixture;
    const std::vector<PopulationObject> first{
        object(1u, 0.0, 1.0),
    };
    std::vector<ovspatialquery_pair_t> result;

    PairMaterializationRequest boundary = crossPopulationRequest(fixture, first, {object(2u, 1.25, 2.0)}, 1u);
    boundary.contactTolerance           = 0.25;
    require(ovspatialquery::detail::materializePairSet(boundary, result) == PairMaterializationOutcome::success &&
                result == std::vector<ovspatialquery_pair_t>{pair(1u, 2u)},
            "inclusive tolerance boundary did not pair");

    PairMaterializationRequest near = crossPopulationRequest(fixture, first, {object(2u, 1.249999, 2.0)}, 1u);
    near.contactTolerance           = 0.25;
    require(ovspatialquery::detail::materializePairSet(near, result) == PairMaterializationOutcome::success &&
                result.size() == 1u,
            "near tolerance hit did not pair");

    PairMaterializationRequest miss = crossPopulationRequest(fixture, first, {object(2u, 1.250001, 2.0)}, 1u);
    miss.contactTolerance           = 0.25;
    require(ovspatialquery::detail::materializePairSet(miss, result) == PairMaterializationOutcome::success &&
                result.empty(),
            "near tolerance miss paired");

    PairMaterializationRequest once = crossPopulationRequest(fixture, first, {object(2u, 2.5, 3.0)}, 1u);
    once.contactTolerance           = 1.0;
    require(ovspatialquery::detail::materializePairSet(once, result) == PairMaterializationOutcome::success &&
                result.empty(),
            "contact tolerance was applied more than once");

    PairMaterializationRequest extreme = crossPopulationRequest(fixture, first, {object(2u, 2.5, 3.0)}, 1u);
    extreme.contactTolerance           = std::numeric_limits<double>::max();
    require(ovspatialquery::detail::materializePairSet(extreme, result) == PairMaterializationOutcome::success &&
                result.size() == 1u,
            "extreme finite contact tolerance was not overflow-safe");
}

void testCapacityFailureIsAtomicAndRetryIsDeterministic()
{
    Fixture                             fixture;
    const std::vector<PopulationObject> first{
        object(2u, 0.0, 1.0),
        object(1u, 0.0, 1.0),
    };
    const std::vector<PopulationObject> second{
        object(9u, 0.0, 1.0),
    };
    PairMaterializationRequest               request = crossPopulationRequest(fixture, first, second, 0u);
    const std::vector<ovspatialquery_pair_t> sentinel{pair(70u, 71u)};
    std::vector<ovspatialquery_pair_t>       result = sentinel;
    require(ovspatialquery::detail::materializePairSet(request, result) == PairMaterializationOutcome::activeCapacity &&
                result == sentinel,
            "zero active capacity exposed a partial result");

    request.activePairCapacity = 1u;
    require(ovspatialquery::detail::materializePairSet(request, result) == PairMaterializationOutcome::activeCapacity &&
                result == sentinel,
            "insufficient active capacity changed caller output");

    request.activePairCapacity = 2u;
    require(ovspatialquery::detail::materializePairSet(request, result) == PairMaterializationOutcome::success &&
                result == std::vector<ovspatialquery_pair_t>{pair(1u, 9u), pair(2u, 9u)},
            "increased-capacity retry was incomplete or unsorted");
    const std::vector<ovspatialquery_pair_t> firstRetry = result;
    result                                              = sentinel;
    require(ovspatialquery::detail::materializePairSet(request, result) == PairMaterializationOutcome::success &&
                result == firstRetry,
            "increased-capacity retry was nondeterministic");
}

void testEmptyInputProducesEmptySuccess()
{
    Fixture                            fixture;
    PairMaterializationRequest         request = crossPopulationRequest(fixture, {}, {object(1u, 0.0, 1.0)}, 0u);
    std::vector<ovspatialquery_pair_t> result{pair(50u, 51u)};
    require(ovspatialquery::detail::materializePairSet(request, result) == PairMaterializationOutcome::success &&
                result.empty(),
            "empty endpoint did not produce an empty complete set");
}

void testResourceFailureIsAtomicAndRetryable()
{
    Fixture                    fixture;
    PairMaterializationRequest request =
        crossPopulationRequest(fixture, {object(1u, 0.0, 1.0)}, {object(2u, 0.0, 1.0)}, 1u);
    const std::vector<ovspatialquery_pair_t> sentinel{pair(80u, 81u)};
    std::vector<ovspatialquery_pair_t>       result = sentinel;
    ovspatialquery::detail::failNextPairMaterializationAllocationForTest();
    request.activePairCapacity = 0u;
    require(ovspatialquery::detail::materializePairSet(request, result) == PairMaterializationOutcome::activeCapacity &&
                result == sentinel,
            "zero capacity did not precede candidate allocation");
    request.activePairCapacity = 1u;
    require(ovspatialquery::detail::materializePairSet(request, result) ==
                    PairMaterializationOutcome::resourceExhausted &&
                result == sentinel,
            "resource exhaustion exposed or replaced a partial result");
    require(ovspatialquery::detail::materializePairSet(request, result) == PairMaterializationOutcome::success &&
                result == std::vector<ovspatialquery_pair_t>{pair(1u, 2u)},
            "resource retry did not deterministically succeed");
}

void publish(ovspatialquery::detail::PopulationTable& populations, PopulationKey population,
             ovspatialquery_ordinal_t ordinal, std::vector<PopulationObject> objects)
{
    ovspatialquery::detail::GenerationLease base;
    require(populations.beginUpdate(population, ordinal, nullptr, base) == OVSPATIALQUERY_API_SUCCESS,
            "population update did not begin");
    ovspatialquery::detail::GenerationLease generation;
    require(populations.buildGeneration(ordinal, std::move(objects), {}, generation) == OVSPATIALQUERY_API_SUCCESS,
            "published generation did not build");
    require(populations.publishUpdate(population, generation) == OVSPATIALQUERY_API_SUCCESS,
            "generation publication failed");
    generation.reset();
    base.reset();
    populations.finishUpdate(population);
}

ResolvedTarget resolve(const ovspatialquery::detail::TargetResolver& resolver, PopulationKey population,
                       ovspatialquery::detail::TargetSelectionKind kind, ovspatialquery_ordinal_t ordinal)
{
    ResolvedTarget target;
    require(resolver.resolve({{population, {}}, {kind, ordinal}}, target) ==
                ovspatialquery::detail::TargetResolveOutcome::success,
            "target resolution failed");
    return target;
}

void testImmutableHeadAndRetainedLeasesSurviveConcurrentPublication()
{
    ovspatialquery_backend_runtime_t* runtime = nullptr;
    require(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS,
            "oracle runtime creation failed");
    {
        ovspatialquery::detail::PopulationTable populations(runtime);
        PopulationKey                           firstPopulation;
        PopulationKey                           secondPopulation;
        require(populations.create(1u, firstPopulation) == OVSPATIALQUERY_API_SUCCESS,
                "first population creation failed");
        require(populations.create(1u, secondPopulation) == OVSPATIALQUERY_API_SUCCESS,
                "second population creation failed");
        publish(populations, firstPopulation, 1, {object(1u, 0.0, 1.0, UINT64_C(0x1))});
        publish(populations, firstPopulation, 2, {object(3u, 0.0, 1.0, UINT64_C(0x2))});
        publish(populations, secondPopulation, 20, {object(9u, 0.0, 1.0)});

        const ovspatialquery::detail::TargetResolver resolver(populations);
        PairMaterializationRequest                   retained{};
        retained.first.target =
            resolve(resolver, firstPopulation, ovspatialquery::detail::TargetSelectionKind::retainedExact, 1);
        retained.second.target =
            resolve(resolver, secondPopulation, ovspatialquery::detail::TargetSelectionKind::head, 0);
        retained.relation               = PairEndpointRelation::distinctPopulations;
        retained.activePairCapacity     = 1u;
        PairMaterializationRequest head = retained;
        head.first.target = resolve(resolver, firstPopulation, ovspatialquery::detail::TargetSelectionKind::head, 0);

        std::atomic<bool> stable{true};
        std::thread       reader(
            [&]()
            {
                for (uint32_t iteration = 0u; iteration < 500u; ++iteration)
                {
                    std::vector<ovspatialquery_pair_t> retainedPairs;
                    std::vector<ovspatialquery_pair_t> headPairs;
                    if (ovspatialquery::detail::materializePairSet(retained, retainedPairs) !=
                            PairMaterializationOutcome::success ||
                        retainedPairs != std::vector<ovspatialquery_pair_t>{pair(1u, 9u)} ||
                        ovspatialquery::detail::materializePairSet(head, headPairs) !=
                            PairMaterializationOutcome::success ||
                        headPairs != std::vector<ovspatialquery_pair_t>{pair(3u, 9u)})
                    {
                        stable.store(false, std::memory_order_release);
                        return;
                    }
                }
            });
        publish(populations, firstPopulation, 3, {object(4u, 4.0, 5.0, UINT64_C(0x4))});
        publish(populations, firstPopulation, 4, {object(5u, 5.0, 6.0, UINT64_C(0x8))});
        reader.join();
        require(stable.load(std::memory_order_acquire), "immutable materialization leases changed during publication");

        PairMaterializationRequest later = retained;
        later.first.target = resolve(resolver, firstPopulation, ovspatialquery::detail::TargetSelectionKind::head, 0);
        std::vector<ovspatialquery_pair_t> laterPairs;
        require(ovspatialquery::detail::materializePairSet(later, laterPairs) == PairMaterializationOutcome::success &&
                    laterPairs.empty(),
                "later head did not remain distinct from held immutable truth");

        retained = {};
        head     = {};
        later    = {};
        std::shared_ptr<void> retiredFirst;
        std::shared_ptr<void> retiredSecond;
        require(populations.release(firstPopulation, retiredFirst) == OVSPATIALQUERY_API_SUCCESS,
                "first population release failed");
        require(populations.release(secondPopulation, retiredSecond) == OVSPATIALQUERY_API_SUCCESS,
                "second population release failed");
    }
    ovspatialquery_test_oracle_destroy_runtime(runtime);
}

} // namespace

int main()
{
    testSamePopulationDistinctSelectionsPreserveDirection();
    testIdenticalSelectionProducesOneSortedUnorderedPairSet();
    testUnorderedGenerationsAdmitAndSortLikeOrderedOnes();
    testGenerationMembershipAndCapturedParticipationBitsAreEndpointTruth();
    testSealedMembershipIsAuthoritativeOverParticipationBits();
    testZeroAndIntersectingQueryBitsFilterEachEndpoint();
    testToleranceIsInclusiveFiniteAndAppliedExactlyOnce();
    testCapacityFailureIsAtomicAndRetryIsDeterministic();
    testEmptyInputProducesEmptySuccess();
    testResourceFailureIsAtomicAndRetryable();
    testImmutableHeadAndRetainedLeasesSurviveConcurrentPublication();
    return 0;
}
