/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../../source/ovspatialquery_instance_source.h"
#include "../../support/native/oracle_backend.h"
#include "../../support/native/public_api_inputs.h"
#include "../../support/native/test_failure.h"
#include <ovspatialquery/ovspatialquery.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace
{

using Outcome = ovspatialquery::detail::ParticipationOutcome;
using Table   = ovspatialquery::detail::ParticipationTableDesc;

ovspatialquery_op_wait_result_t waitPublicOperation(ovspatialquery_instance_handle_t instance,
                                                    ovspatialquery_op_id_t           operation)
{
    ovspatialquery_op_wait_result_t wait{};
    REQUIRE(ovspatialquery_wait_op(instance, operation, OVSPATIALQUERY_TIMEOUT_INFINITE, &wait).status ==
            OVSPATIALQUERY_API_SUCCESS);
    return wait;
}

void releasePublicOperation(ovspatialquery_instance_handle_t instance, ovspatialquery_op_id_t operation)
{
    REQUIRE(ovspatialquery_destroy_op(instance, &operation).status == OVSPATIALQUERY_API_SUCCESS);
}

Outcome install(ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population,
                const Table* table)
{
    return ovspatialquery::detail::installParticipationTable(instance, population, table);
}

void testInvalidInstanceHasNoSideEffect()
{
    const Table table{};
    REQUIRE(install(OVSPATIALQUERY_INVALID_INSTANCE, OVSPATIALQUERY_INVALID_POPULATION, &table) ==
            Outcome::invalidHandle);
}

struct SourceSnapshot
{
    std::vector<ovspatialquery::detail::PopulationObject> objects;
};

struct SourceProbe
{
    std::map<ovspatialquery_ordinal_t, SourceSnapshot> snapshots;
};

ovspatialquery::detail::PopulationObject object(uint64_t id, double minimumX)
{
    ovspatialquery::detail::PopulationObject result{};
    result.id                = {id};
    result.bounds.minimum[0] = minimumX;
    result.bounds.minimum[1] = -1.0;
    result.bounds.minimum[2] = -1.0;
    result.bounds.maximum[0] = minimumX + 1.0;
    result.bounds.maximum[1] = 1.0;
    result.bounds.maximum[2] = 1.0;
    return result;
}

ovspatialquery_api_status_t acquireSnapshot(const void*, const void*,
                                            const std::shared_ptr<void>& populationSourceState,
                                            const std::vector<ovspatialquery::detail::SourceViewRequest>&,
                                            ovspatialquery_ordinal_t targetOrdinal,
                                            const ovspatialquery::detail::SourceAcquisitionBase*,
                                            ovspatialquery::detail::CopiedSourceSnapshot& outSnapshot)
{
    const std::shared_ptr<SourceProbe> source = std::static_pointer_cast<SourceProbe>(populationSourceState);
    const std::map<ovspatialquery_ordinal_t, SourceSnapshot>::const_iterator found =
        source->snapshots.find(targetOrdinal);
    if (found == source->snapshots.end())
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    outSnapshot              = {};
    outSnapshot.objects      = found->second.objects;
    outSnapshot.indexedCount = outSnapshot.objects.size();
    return OVSPATIALQUERY_API_SUCCESS;
}

constexpr ovspatialquery_ordinal_t kHead = -1;

// Each object's two participation words, bits 0-63 first.
using ObjectBits = std::map<uint64_t, std::pair<uint64_t, uint64_t>>;

struct Fixture
{
    Fixture(uint64_t hostBudget = std::numeric_limits<uint64_t>::max())
    {
        const ovspatialquery::detail::SourceInstanceConfig instanceConfig{
            8u, 0u, 1u, hostBudget, nullptr, 0u,
        };
        REQUIRE(ovspatialquery::detail::createInstance(instanceConfig, &instance) == OVSPATIALQUERY_API_SUCCESS);
        source     = std::make_shared<SourceProbe>();
        population = createPopulation(OVSPATIALQUERY_SOURCE_OVSTAGE);
    }

    ~Fixture()
    {
        if (population != OVSPATIALQUERY_INVALID_POPULATION)
        {
            (void)ovspatialquery_destroy_population(instance, population);
        }
        if (instance != OVSPATIALQUERY_INVALID_INSTANCE)
        {
            (void)ovspatialquery_destroy_instance(instance);
        }
    }

    ovspatialquery_population_handle_t createPopulation(ovspatialquery_source_kind_t kind)
    {
        ovspatialquery::detail::PopulationSourceBinding binding{
            kind, this, source.get(), source, acquireSnapshot,
        };
        ovspatialquery_population_handle_t created = OVSPATIALQUERY_INVALID_POPULATION;
        REQUIRE(ovspatialquery::detail::createEmptyPopulationForSource(instance, std::move(binding), 3u, 0u, created) ==
                OVSPATIALQUERY_API_SUCCESS);
        return created;
    }

    void publish(ovspatialquery_ordinal_t ordinal)
    {
        waitAndDestroy(instance, {OVSPATIALQUERY_API_SUCCESS, submit(ordinal)});
    }

    ovspatialquery_op_id_t submit(ovspatialquery_ordinal_t ordinal)
    {
        ovspatialquery_op_id_t operation = OVSPATIALQUERY_INVALID_OP;
        REQUIRE(ovspatialquery::detail::submitPopulationSnapshotForSource(instance, OVSPATIALQUERY_SOURCE_OVSTAGE,
                                                                          population, nullptr, ordinal,
                                                                          operation) == OVSPATIALQUERY_API_SUCCESS);
        return operation;
    }

    // Raycasts through every object of the population at the head or at one retained ordinal,
    // and stores the sorted IDs that `queryBits` admits.
    ovspatialquery_api_status_t tryQuery(ovspatialquery_participation_bits_t queryBits,
                                         ovspatialquery_ordinal_t ordinal, std::vector<uint64_t>& outIds)
    {
        ovspatialquery_raycast_desc_t ray{};
        ray.common.target = populationTarget(population);
        if (ordinal != kHead)
        {
            ray.common.target.ordinal = {OVSPATIALQUERY_ORDINAL_RETAINED_EXACT, ordinal};
        }
        ray.common.precision      = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
        ray.common.query_bits     = queryBits;
        ray.ray.origin[0]         = -2.0;
        ray.ray.unit_direction[0] = 1.0f;
        ray.ray.max_distance      = 20.0;
        ovspatialquery_raycast_hit_t        storage[8]{};
        ovspatialquery_raycast_hit_buffer_t buffer{storage, 8u};
        ovspatialquery_multiple_result_t    result{};
        const ovspatialquery_api_status_t   status =
            ovspatialquery_raycast_multiple_sync(instance, &ray, &buffer, &result).status;
        outIds.clear();
        for (uint64_t index = 0u; status == OVSPATIALQUERY_API_SUCCESS && index < result.stored_hit_count; ++index)
        {
            outIds.push_back(storage[index].data.bounds_candidate.identity.object_id.value);
        }
        std::sort(outIds.begin(), outIds.end());
        return status;
    }

    std::vector<uint64_t> query(ovspatialquery_participation_bits_t queryBits, ovspatialquery_ordinal_t ordinal = kHead)
    {
        std::vector<uint64_t> ids;
        REQUIRE(tryQuery(queryBits, ordinal, ids) == OVSPATIALQUERY_API_SUCCESS);
        return ids;
    }

    std::vector<uint64_t> query(uint64_t lowBits, ovspatialquery_ordinal_t ordinal = kHead)
    {
        return query(participationBits(lowBits), ordinal);
    }

    // The participation bits each object of `ordinal` captured, read one bit at a time:
    // query_bits holding a single bit admits exactly the objects whose captured bits include it.
    ObjectBits bits(ovspatialquery_ordinal_t ordinal)
    {
        ObjectBits result;
        for (const uint64_t id : query(0u, ordinal))
        {
            result.emplace(id, std::make_pair(0u, 0u));
        }
        for (uint32_t bit = 0u; bit < 128u; ++bit)
        {
            const uint64_t word = UINT64_C(1) << (bit % 64u);
            for (const uint64_t id : query(bit < 64u ? participationBits(word) : participationBits(0u, word), ordinal))
            {
                REQUIRE(result.count(id) == 1u);
                (bit < 64u ? result[id].first : result[id].second) |= word;
            }
        }
        return result;
    }

    ovspatialquery_instance_handle_t   instance   = OVSPATIALQUERY_INVALID_INSTANCE;
    ovspatialquery_population_handle_t population = OVSPATIALQUERY_INVALID_POPULATION;
    std::shared_ptr<SourceProbe>       source;
};

void testInstalledTableAffectsOnlyFuturePublications()
{
    Fixture fixture;
    for (ovspatialquery_ordinal_t ordinal = 1; ordinal <= 3; ++ordinal)
    {
        fixture.source->snapshots.emplace(ordinal, SourceSnapshot{{object(11u, 0.0), object(22u, 3.0)}});
    }
    fixture.publish(1);

    // 22 is listed, so its bits replace the default and drop bit 0x1; 33 is absent from every
    // publication and contributes nothing. Bits 64 to 127 travel with the entries like bits 0 to 63.
    uint64_t                            paths[] = {22u, 33u};
    ovspatialquery_participation_bits_t bits[]  = {participationBits(UINT64_C(0x6), UINT64_C(1) << 63u),
                                                   participationBits(UINT64_C(0x8))};
    Table                               table{participationBits(UINT64_C(0x1), UINT64_C(0x2)), paths, bits, 2u};
    REQUIRE(install(fixture.instance, fixture.population, &table) == Outcome::success);
    // The call deep-copied the table, so later edits of the host's arrays change nothing.
    table.defaultBits = participationBits(UINT64_C(0x80));
    paths[0]          = 11u;
    bits[0]           = participationBits(UINT64_C(0x100));

    fixture.publish(2);

    const ObjectBits untabled{{11u, {UINT64_MAX, UINT64_MAX}}, {22u, {UINT64_MAX, UINT64_MAX}}};
    const ObjectBits tabled{{11u, {UINT64_C(0x1), UINT64_C(0x2)}}, {22u, {UINT64_C(0x6), UINT64_C(1) << 63u}}};
    REQUIRE(fixture.bits(1) == untabled);
    REQUIRE(fixture.bits(2) == tabled);
    REQUIRE(fixture.query(0u) == std::vector<uint64_t>({11u, 22u}));
    REQUIRE(fixture.query(UINT64_C(0x1)) == std::vector<uint64_t>{11u});
    REQUIRE(fixture.query(UINT64_C(0x8)).empty());
    REQUIRE(fixture.query(participationBits(0u, UINT64_C(1) << 63u)) == std::vector<uint64_t>{22u});
    REQUIRE(fixture.query(participationBits(UINT64_C(0x1), UINT64_C(1) << 63u)) == std::vector<uint64_t>({11u, 22u}));

    REQUIRE(install(fixture.instance, fixture.population, nullptr) == Outcome::success);
    fixture.publish(3);
    REQUIRE(fixture.bits(3) == untabled);
    REQUIRE(fixture.bits(2) == tabled);
}

void testRejectedTablesPreserveInstalledTable()
{
    Fixture     fixture;
    const Table original{participationBits(UINT64_C(0x11)), nullptr, nullptr, 0u};
    REQUIRE(install(fixture.instance, fixture.population, &original) == Outcome::success);

    const ovspatialquery_population_handle_t explicitPopulation =
        fixture.createPopulation(OVSPATIALQUERY_SOURCE_EXPLICIT);
    REQUIRE(install(fixture.instance, explicitPopulation, &original) == Outcome::invalidTargetSource);
    const ovspatialquery_population_handle_t destroyedPopulation =
        fixture.createPopulation(OVSPATIALQUERY_SOURCE_OVSTAGE);
    REQUIRE(ovspatialquery_destroy_population(fixture.instance, destroyedPopulation).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(install(fixture.instance, destroyedPopulation, &original) == Outcome::invalidHandle);

    const uint64_t                            paths[]    = {22u, 11u, 22u};
    const uint64_t                            zeroPath[] = {11u, 0u};
    const ovspatialquery_participation_bits_t bits[]     = {
        participationBits(UINT64_C(0x2)), participationBits(UINT64_C(0x4)), participationBits(UINT64_C(0x8))};
    const Table rejected[] = {
        {{}, zeroPath, bits, 2u}, // the invalid path
        {{}, paths, bits, 3u},    // 22 twice
    };
    for (const Table& table : rejected)
    {
        REQUIRE(install(fixture.instance, fixture.population, &table) == Outcome::invalidArgument);
    }
    const Table impossible{{}, paths, bits, std::numeric_limits<size_t>::max()};
    REQUIRE(install(fixture.instance, fixture.population, &impossible) == Outcome::outOfMemory);

    fixture.source->snapshots.emplace(1, SourceSnapshot{{object(11u, 0.0), object(22u, 3.0)}});
    fixture.publish(1);
    REQUIRE(fixture.bits(1) == ObjectBits({{11u, {UINT64_C(0x11), 0u}}, {22u, {UINT64_C(0x11), 0u}}}));

    // Entry order is free: the first two entries, listed 22 then 11, still pair each path with its bits.
    const Table unordered{{}, paths, bits, 2u};
    REQUIRE(install(fixture.instance, fixture.population, &unordered) == Outcome::success);
    fixture.source->snapshots.emplace(2, SourceSnapshot{{object(11u, 0.0), object(22u, 3.0)}});
    fixture.publish(2);
    REQUIRE(fixture.bits(2) == ObjectBits({{11u, {UINT64_C(0x4), 0u}}, {22u, {UINT64_C(0x2), 0u}}}));

    REQUIRE(ovspatialquery_destroy_population(fixture.instance, explicitPopulation).status ==
            OVSPATIALQUERY_API_SUCCESS);
}

// A table of 1024 distinct paths. It charges far more host bytes than one small publication,
// and clearing it returns them.
Outcome installBallastTable(ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population)
{
    std::vector<uint64_t> paths(1024u);
    for (size_t index = 0u; index < paths.size(); ++index)
    {
        paths[index] = index + 1u;
    }
    const std::vector<ovspatialquery_participation_bits_t> bits(paths.size(), participationBits(0u));
    const Table                                            table{{}, paths.data(), bits.data(), paths.size()};
    return install(instance, population, &table);
}

// Installs an empty table in an instance of `budget` host bytes that also holds a population with
// a ballast table, and reports the host bytes the empty table charged.
bool installEmptyTableAtBudget(uint64_t budget, uint64_t defaultBits, uint64_t* outResidentBytes)
{
    const ovspatialquery::detail::SourceInstanceConfig instanceConfig{
        8u, 0u, 1u, budget, nullptr, 0u,
    };
    ovspatialquery_instance_handle_t instance = OVSPATIALQUERY_INVALID_INSTANCE;
    if (ovspatialquery::detail::createInstance(instanceConfig, &instance) != OVSPATIALQUERY_API_SUCCESS)
    {
        return false;
    }
    const std::shared_ptr<SourceProbe> source = std::make_shared<SourceProbe>();
    ovspatialquery_population_handle_t populations[2]{OVSPATIALQUERY_INVALID_POPULATION,
                                                      OVSPATIALQUERY_INVALID_POPULATION};
    bool                               created = true;
    for (ovspatialquery_population_handle_t& population : populations)
    {
        ovspatialquery::detail::PopulationSourceBinding binding{
            OVSPATIALQUERY_SOURCE_OVSTAGE, source.get(), source.get(), source, acquireSnapshot,
        };
        created = created && ovspatialquery::detail::createEmptyPopulationForSource(
                                 instance, std::move(binding), 3u, 0u, population) == OVSPATIALQUERY_API_SUCCESS;
    }
    bool installed = created && installBallastTable(instance, populations[1]) == Outcome::success;
    if (installed)
    {
        const uint64_t before = hostUsage(instance);
        const Table    table{participationBits(defaultBits), nullptr, nullptr, 0u};
        installed = install(instance, populations[0], &table) == Outcome::success;
        if (installed && outResidentBytes != nullptr)
        {
            *outResidentBytes = hostUsage(instance) - before;
        }
    }
    bool cleanupSucceeded = true;
    for (const ovspatialquery_population_handle_t population : populations)
    {
        if (population != OVSPATIALQUERY_INVALID_POPULATION)
        {
            cleanupSucceeded =
                ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS &&
                cleanupSucceeded;
        }
    }
    cleanupSucceeded =
        ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS && cleanupSucceeded;
    return installed && cleanupSucceeded;
}

void testClearingAndPopulationDestructionReturnTheCharge()
{
    Fixture                                   fixture;
    const uint64_t                            baseline = hostUsage(fixture.instance);
    const uint64_t                            paths[]  = {11u};
    const ovspatialquery_participation_bits_t bits[]   = {participationBits(UINT64_C(0x2))};
    const Table                               table{participationBits(UINT64_C(0x1)), paths, bits, 1u};
    REQUIRE(install(fixture.instance, fixture.population, &table) == Outcome::success);
    REQUIRE(hostUsage(fixture.instance) > baseline);
    REQUIRE(install(fixture.instance, fixture.population, nullptr) == Outcome::success);
    REQUIRE(hostUsage(fixture.instance) == baseline);

    REQUIRE(install(fixture.instance, fixture.population, &table) == Outcome::success);
    REQUIRE(ovspatialquery_destroy_population(fixture.instance, fixture.population).status ==
            OVSPATIALQUERY_API_SUCCESS);
    fixture.population                                = OVSPATIALQUERY_INVALID_POPULATION;
    const ovspatialquery_population_handle_t untabled = fixture.createPopulation(OVSPATIALQUERY_SOURCE_OVSTAGE);
    REQUIRE(hostUsage(fixture.instance) == baseline);
    REQUIRE(ovspatialquery_destroy_population(fixture.instance, untabled).status == OVSPATIALQUERY_API_SUCCESS);
}

void testReplacementUsesConservativeFullCandidateReservation()
{
    uint64_t low  = 0u;
    uint64_t high = UINT64_C(1) << 20u;
    REQUIRE(installEmptyTableAtBudget(high, UINT64_C(0x1), nullptr));
    while (low < high)
    {
        const uint64_t middle = low + (high - low) / 2u;
        if (installEmptyTableAtBudget(middle, UINT64_C(0x1), nullptr))
        {
            high = middle;
        }
        else
        {
            low = middle + 1u;
        }
    }
    uint64_t residentBytes = 0u;
    REQUIRE(installEmptyTableAtBudget(UINT64_C(1) << 20u, UINT64_C(0x1), &residentBytes));
    REQUIRE(residentBytes != 0u);
    const uint64_t replacementBoundary = low + residentBytes - 1u;

    Fixture                                  fixture(replacementBoundary);
    const ovspatialquery_population_handle_t ballast = fixture.createPopulation(OVSPATIALQUERY_SOURCE_OVSTAGE);
    REQUIRE(installBallastTable(fixture.instance, ballast) == Outcome::success);
    const Table    original{participationBits(UINT64_C(0x1)), nullptr, nullptr, 0u};
    const Table    replacement{participationBits(UINT64_C(0x2)), nullptr, nullptr, 0u};
    const uint64_t withoutTable = hostUsage(fixture.instance);
    REQUIRE(install(fixture.instance, fixture.population, &original) == Outcome::success);
    REQUIRE(install(fixture.instance, fixture.population, &replacement) == Outcome::hostBudget);
    REQUIRE(hostUsage(fixture.instance) - withoutTable == residentBytes);

    // Clearing the ballast makes room for one publication, which captures the preserved table.
    REQUIRE(install(fixture.instance, ballast, nullptr) == Outcome::success);
    fixture.source->snapshots.emplace(1, SourceSnapshot{{object(11u, 0.0)}});
    fixture.publish(1);
    REQUIRE(fixture.bits(1) == ObjectBits({{11u, {UINT64_C(0x1), 0u}}}));

    REQUIRE(install(fixture.instance, fixture.population, nullptr) == Outcome::success);
    REQUIRE(install(fixture.instance, fixture.population, &replacement) == Outcome::success);
    REQUIRE(ovspatialquery_destroy_population(fixture.instance, ballast).status == OVSPATIALQUERY_API_SUCCESS);
}

void testPublicationFailurePreservesHeadHistoryAndTable()
{
    Fixture fixture;
    fixture.source->snapshots.emplace(1, SourceSnapshot{{object(11u, 0.0)}});
    fixture.source->snapshots.emplace(2, SourceSnapshot{{object(22u, 0.0)}});
    const Table table{participationBits(UINT64_C(0x1)), nullptr, nullptr, 0u};
    REQUIRE(install(fixture.instance, fixture.population, &table) == Outcome::success);
    fixture.publish(1);

    ovspatialquery_backend_runtime_t* runtime = ovspatialquery_test_oracle_get_default_runtime();
    REQUIRE(runtime != nullptr);
    ovspatialquery_test_oracle_fail_next_build(runtime, OVSPATIALQUERY_API_INTERNAL);
    ovspatialquery_op_id_t operation = fixture.submit(2);
    // wait_op always returns SUCCESS on a terminal state; a failed operation
    // is reported by out_wait.state == OP_FAILURE, not by the call status.
    REQUIRE(waitPublicOperation(fixture.instance, operation).state == OVSPATIALQUERY_OP_FAILURE);
    releasePublicOperation(fixture.instance, operation);

    REQUIRE(fixture.bits(1) == ObjectBits({{11u, {UINT64_C(0x1), 0u}}}));
    std::vector<uint64_t> absent;
    REQUIRE(fixture.tryQuery({}, 2, absent) == OVSPATIALQUERY_API_ORDINAL_NOT_RETAINED);

    ovspatialquery_op_id_t rejected = ovspatialquery_op_id_t{19u};
    REQUIRE(ovspatialquery::detail::submitPopulationSnapshotForSource(fixture.instance, OVSPATIALQUERY_SOURCE_OVSTAGE,
                                                                      fixture.population, nullptr, 3,
                                                                      rejected) == OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(rejected == OVSPATIALQUERY_INVALID_OP);
    REQUIRE(fixture.query(UINT64_C(0x1)) == std::vector<uint64_t>{11u});

    fixture.publish(2);
    REQUIRE(fixture.bits(2) == ObjectBits({{22u, {UINT64_C(0x1), 0u}}}));
}

void testSetterEnqueueAndQueryHaveOneLinearizationOrder()
{
    Fixture fixture;
    fixture.source->snapshots.emplace(1, SourceSnapshot{{object(11u, 0.0)}});
    fixture.source->snapshots.emplace(2, SourceSnapshot{{object(22u, 0.0)}});
    const Table first{participationBits(UINT64_C(0x1)), nullptr, nullptr, 0u};
    const Table second{participationBits(UINT64_C(0x2)), nullptr, nullptr, 0u};
    const Table third{participationBits(UINT64_C(0x4)), nullptr, nullptr, 0u};
    REQUIRE(install(fixture.instance, fixture.population, &first) == Outcome::success);
    fixture.publish(1);
    REQUIRE(install(fixture.instance, fixture.population, &second) == Outcome::success);

    ovspatialquery_backend_runtime_t* runtime = ovspatialquery_test_oracle_get_default_runtime();
    REQUIRE(runtime != nullptr);
    ovspatialquery_test_oracle_block_builds(runtime, 1u);
    ovspatialquery_op_id_t operation = fixture.submit(2);
    REQUIRE(ovspatialquery_test_oracle_wait_for_blocked_builds(runtime) == 1u);
    REQUIRE(install(fixture.instance, fixture.population, &third) == Outcome::busy);
    REQUIRE(fixture.query(UINT64_C(0x1)) == std::vector<uint64_t>{11u});
    REQUIRE(fixture.query(UINT64_C(0x2)).empty());

    ovspatialquery_test_oracle_release_builds(runtime);
    waitAndDestroy(fixture.instance, {OVSPATIALQUERY_API_SUCCESS, operation});
    REQUIRE(fixture.query(UINT64_C(0x1)).empty());
    REQUIRE(fixture.query(UINT64_C(0x2)) == std::vector<uint64_t>{22u});
    REQUIRE(install(fixture.instance, fixture.population, &third) == Outcome::success);
}

} // namespace

int main()
{
    testInvalidInstanceHasNoSideEffect();
    testInstalledTableAffectsOnlyFuturePublications();
    testRejectedTablesPreserveInstalledTable();
    testClearingAndPopulationDestructionReturnTheCharge();
    testReplacementUsesConservativeFullCandidateReservation();
    testPublicationFailurePreservesHeadHistoryAndTable();
    testSetterEnqueueAndQueryHaveOneLinearizationOrder();
    return 0;
}
