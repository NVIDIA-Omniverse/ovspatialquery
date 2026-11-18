/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Fixture objects are engine-private ovspatialquery_aabb_t boxes, but
 * population and pair-snapshot setup below goes through the public
 * instance/population/pair-snapshot API: every AABB fixture object is
 * reproduced by one registered box definition per distinct half-extent plus
 * an identity-rotation instance pose translated to the AABB center, and a
 * "complete AABB replacement" republish (what the removed private
 * population_set_async did unconditionally) is expressed as a public
 * population update whose changed rows are the complete new object set and
 * whose removed rows are every id no longer present.
 *
 * A handful of the original malformed-update injections directly corrupted a
 * raw AABB (reversed or nonfinite bounds). The public explicit-population API
 * has no raw-AABB submission path any more - bounds are always derived from a
 * registered sphere/box/capsule/convex/mesh definition plus an instance pose -
 * so those specific corruptions have no public equivalent and are not
 * reproduced here (see exerciseScheduledFailure below).
 */
#include "ovspatialquery_private_api.h"

#include "../../support/native/public_api_inputs.h"
#include "../../support/native/test_failure.h"
#include <ovspatialquery/ovspatialquery_explicit.h>
#include <ovspatialquery/ovspatialquery_pairs.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <set>
#include <utility>
#include <vector>

namespace
{

constexpr uint32_t kFramesPerSeed = 128u;

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

// World-space AABB test fixture, and the population/instance/pair-snapshot
// plumbing built on the public API below.
struct ovspatialquery_test_object_t
{
    ovspatialquery_object_id_t id;
    ovspatialquery_aabb_t      bounds;
};

// Maps every AABB fixture object to a registered box definition matching its
// half-extent: identity rotation, translation at the AABB center. `catalog`
// is the (definition value, half-extent) pairs a ProductFixture registered;
// a half-extent with no matching entry is a fixture bug, not a library
// rejection, so it aborts the test rather than returning a status.
void addAabbObjects(InstanceRows& rows, const std::vector<ovspatialquery_test_object_t>& objects,
                    const std::vector<std::pair<uint64_t, std::array<double, 3>>>& catalog)
{
    for (const ovspatialquery_test_object_t& object : objects)
    {
        const std::array<double, 3> halfExtent{
            0.5 * (object.bounds.maximum[0] - object.bounds.minimum[0]),
            0.5 * (object.bounds.maximum[1] - object.bounds.minimum[1]),
            0.5 * (object.bounds.maximum[2] - object.bounds.minimum[2]),
        };
        const auto definition =
            std::find_if(catalog.begin(), catalog.end(), [&](const auto& entry) { return entry.second == halfExtent; });
        if (definition == catalog.end())
        {
            fail("AABB fixture half-extent has no registered box definition");
        }
        rows.add(object.id.value, OVSPATIALQUERY_GEOMETRY_BOX, definition->first,
                 0.5 * (object.bounds.minimum[0] + object.bounds.maximum[0]),
                 0.5 * (object.bounds.minimum[1] + object.bounds.maximum[1]),
                 0.5 * (object.bounds.minimum[2] + object.bounds.maximum[2]));
    }
}

bool pairLess(const ovspatialquery_pair_t& left, const ovspatialquery_pair_t& right)
{
    return left.first.value < right.first.value ||
           (left.first.value == right.first.value && left.second.value < right.second.value);
}

bool pairEqual(const ovspatialquery_pair_t& left, const ovspatialquery_pair_t& right)
{
    return left.first.value == right.first.value && left.second.value == right.second.value;
}

bool pairVectorsEqual(const std::vector<ovspatialquery_pair_t>& left, const std::vector<ovspatialquery_pair_t>& right)
{
    if (left.size() != right.size())
    {
        return false;
    }
    for (size_t index = 0u; index < left.size(); ++index)
    {
        if (!pairEqual(left[index], right[index]))
        {
            return false;
        }
    }
    return true;
}

bool overlaps(const ovspatialquery_aabb_t& left, const ovspatialquery_aabb_t& right)
{
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        if (left.maximum[axis] < right.minimum[axis] || right.maximum[axis] < left.minimum[axis])
        {
            return false;
        }
    }
    return true;
}

std::vector<ovspatialquery_pair_t> bruteForceOracle(const std::vector<ovspatialquery_test_object_t>& populationA,
                                                    const std::vector<ovspatialquery_test_object_t>& populationB)
{
    std::vector<ovspatialquery_pair_t> result;
    for (const ovspatialquery_test_object_t& objectA : populationA)
    {
        for (const ovspatialquery_test_object_t& objectB : populationB)
        {
            if (overlaps(objectA.bounds, objectB.bounds))
            {
                result.push_back({objectA.id, objectB.id});
            }
        }
    }
    std::sort(result.begin(), result.end(), pairLess);
    return result;
}

class DeterministicRandom
{
  public:
    explicit DeterministicRandom(uint64_t seed) : mState(seed) { require(seed != 0u, "random seed must be non-zero"); }

    uint64_t next()
    {
        mState ^= mState >> 12u;
        mState ^= mState << 25u;
        mState ^= mState >> 27u;
        return mState * UINT64_C(2685821657736338717);
    }

    uint32_t bounded(uint32_t upperBound)
    {
        require(upperBound != 0u, "random upper bound must be non-zero");
        return static_cast<uint32_t>(next() % upperBound);
    }

  private:
    uint64_t mState;
};

// Owns one instance and its two populations, registers the box definitions
// every AABB fixture object needs (default: the one shape every grid object
// uses), and tracks each population's published ids so a "complete AABB
// replacement" publish can be expressed as a public update whose changed rows
// are the full new object set and whose removed rows are every id no longer
// present.
class ProductFixture
{
  public:
    explicit ProductFixture(std::vector<std::pair<uint64_t, std::array<double, 3>>> boxCatalog = {{1u,
                                                                                                   {0.5, 0.5, 0.5}}})
        : mBoxCatalog(std::move(boxCatalog))
    {
        const ovspatialquery_instance_desc_t instanceDesc{64u, 4u, UINT64_C(256) * 1024u * 1024u, nullptr, 0u};
        require(ovspatialquery_create_instance(&instanceDesc, &mInstance).status == OVSPATIALQUERY_API_SUCCESS,
                "instance creation failed");
        registerBoxCatalog();
        mFirstPopulation  = createExplicitPopulation(mInstance, 0u);
        mSecondPopulation = createExplicitPopulation(mInstance, 0u);
    }

    ~ProductFixture()
    {
        require(ovspatialquery_destroy_population(mInstance, mSecondPopulation).status == OVSPATIALQUERY_API_SUCCESS,
                "second population release failed");
        require(ovspatialquery_destroy_population(mInstance, mFirstPopulation).status == OVSPATIALQUERY_API_SUCCESS,
                "first population release failed");
        for (const auto& definition : mBoxCatalog)
        {
            require(
                ovspatialquery_destroy_geometry_definition(mInstance, {OVSPATIALQUERY_GEOMETRY_BOX, definition.first})
                        .status == OVSPATIALQUERY_API_SUCCESS,
                "box definition release failed");
        }
        require(ovspatialquery_destroy_instance(mInstance).status == OVSPATIALQUERY_API_SUCCESS,
                "instance release failed");
    }

    ProductFixture(const ProductFixture&)            = delete;
    ProductFixture& operator=(const ProductFixture&) = delete;

    ovspatialquery_instance_handle_t instance() const { return mInstance; }

    ovspatialquery_population_handle_t firstPopulation() const { return mFirstPopulation; }

    ovspatialquery_population_handle_t secondPopulation() const { return mSecondPopulation; }

    // Publishes `objects` as the complete head at `ordinal`: the population's
    // first complete set if it has no head yet, otherwise a public update
    // whose changed rows are the complete new set and whose removed rows are
    // every previously published id absent from `objects`.
    ovspatialquery_api_status_t publishComplete(ovspatialquery_population_handle_t               population,
                                                ovspatialquery_ordinal_t                         ordinal,
                                                const std::vector<ovspatialquery_test_object_t>& objects)
    {
        PopulationTrack& track = trackFor(population);
        InstanceRows     rows;
        addAabbObjects(rows, objects, mBoxCatalog);
        ovspatialquery_enqueue_result_t admission{};
        if (!track.hasHead)
        {
            const ovspatialquery_explicit_population_set_t set{ordinal, rows.table(), nullptr, 0u};
            admission = ovspatialquery_enqueue_explicit_population_set(mInstance, population, &set);
        }
        else
        {
            std::vector<uint64_t> removed = removedIds(track.ids, objects);
            InputArray            removedArray(removed.data(), static_cast<int64_t>(removed.size()), kDLUInt);
            const ovspatialquery_explicit_population_update_t update{
                track.ordinal,       ordinal, OVSPATIALQUERY_EXPLICIT_UPDATE_COMPLETE_ROWS, rows.table(),
                &removedArray.value, nullptr};
            admission = ovspatialquery_enqueue_explicit_population_update(mInstance, population, &update);
        }
        if (admission.status != OVSPATIALQUERY_API_SUCCESS)
        {
            return admission.status;
        }
        waitAndDestroy(mInstance, admission);
        track.hasHead = true;
        track.ordinal = ordinal;
        track.ids.clear();
        for (const ovspatialquery_test_object_t& object : objects)
        {
            track.ids.push_back(object.id);
        }
        return OVSPATIALQUERY_API_SUCCESS;
    }

    // Attempts a population update expected to be rejected at admission: no
    // wait, no state tracking, just the enqueue status.
    ovspatialquery_api_status_t tryEnqueueUpdate(ovspatialquery_population_handle_t               population,
                                                 ovspatialquery_ordinal_t                         baseOrdinal,
                                                 ovspatialquery_ordinal_t                         targetOrdinal,
                                                 const std::vector<ovspatialquery_test_object_t>& changed)
    {
        InstanceRows rows;
        addAabbObjects(rows, changed, mBoxCatalog);
        uint64_t                                          emptyRemoved = 0u;
        InputArray                                        removedArray(&emptyRemoved, 0, kDLUInt);
        const ovspatialquery_explicit_population_update_t update{
            baseOrdinal,  targetOrdinal,       OVSPATIALQUERY_EXPLICIT_UPDATE_COMPLETE_ROWS,
            rows.table(), &removedArray.value, nullptr};
        return ovspatialquery_enqueue_explicit_population_update(mInstance, population, &update).status;
    }

  private:
    struct PopulationTrack
    {
        bool                                    hasHead = false;
        ovspatialquery_ordinal_t                ordinal = 0;
        std::vector<ovspatialquery_object_id_t> ids;
    };

    void registerBoxCatalog()
    {
        std::vector<uint64_t> values;
        std::vector<double>   halfExtents;
        for (const auto& definition : mBoxCatalog)
        {
            values.push_back(definition.first);
            halfExtents.push_back(definition.second[0]);
            halfExtents.push_back(definition.second[1]);
            halfExtents.push_back(definition.second[2]);
        }
        InputArray valueArray(values.data(), static_cast<int64_t>(values.size()), kDLUInt);
        InputArray extentArray(halfExtents.data(), static_cast<int64_t>(values.size()), 3, kDLFloat);
        const ovspatialquery_box_definition_table_t boxes{&valueArray.value, &extentArray.value};
        waitAndDestroy(mInstance, ovspatialquery_enqueue_box_registration(mInstance, &boxes));
    }

    PopulationTrack& trackFor(ovspatialquery_population_handle_t population)
    {
        return population == mFirstPopulation ? mFirstTrack : mSecondTrack;
    }

    static std::vector<uint64_t> removedIds(const std::vector<ovspatialquery_object_id_t>&   previousIds,
                                            const std::vector<ovspatialquery_test_object_t>& current)
    {
        std::set<uint64_t> currentIds;
        for (const ovspatialquery_test_object_t& object : current)
        {
            currentIds.insert(object.id.value);
        }
        std::vector<uint64_t> removed;
        for (ovspatialquery_object_id_t id : previousIds)
        {
            if (currentIds.find(id.value) == currentIds.end())
            {
                removed.push_back(id.value);
            }
        }
        return removed;
    }

    std::vector<std::pair<uint64_t, std::array<double, 3>>> mBoxCatalog;
    ovspatialquery_instance_handle_t                        mInstance         = OVSPATIALQUERY_INVALID_INSTANCE;
    ovspatialquery_population_handle_t                      mFirstPopulation  = OVSPATIALQUERY_INVALID_POPULATION;
    ovspatialquery_population_handle_t                      mSecondPopulation = OVSPATIALQUERY_INVALID_POPULATION;
    PopulationTrack                                         mFirstTrack;
    PopulationTrack                                         mSecondTrack;
};

// The last complete pair snapshot, and how many pairs began and ended across a seed.
struct PairHistory
{
    std::vector<ovspatialquery_pair_t> current;
    size_t                             began = 0u;
    size_t                             ended = 0u;
};

void setGridBounds(ovspatialquery_test_object_t& object, int32_t slot, bool secondPopulation)
{
    const double offset   = secondPopulation ? 0.25 : 0.0;
    const double minimumX = static_cast<double>(slot) * 4.0 + offset;
    object.bounds         = {{minimumX, 0.0, 0.0}, {minimumX + 1.0, 1.0, 1.0}};
}

ovspatialquery_test_object_t makeGridObject(ovspatialquery_object_id_t id, int32_t slot, bool secondPopulation)
{
    ovspatialquery_test_object_t object{};
    object.id = id;
    setGridBounds(object, slot, secondPopulation);
    return object;
}

void requireUniqueIds(const std::vector<ovspatialquery_test_object_t>& objects)
{
    std::vector<ovspatialquery_object_id_t> ids;
    ids.reserve(objects.size());
    for (const ovspatialquery_test_object_t& object : objects)
    {
        ids.push_back(object.id);
    }
    std::sort(ids.begin(), ids.end(),
              [](ovspatialquery_object_id_t first, ovspatialquery_object_id_t second)
              { return first.value < second.value; });
    require(std::adjacent_find(ids.begin(), ids.end(),
                               [](ovspatialquery_object_id_t first, ovspatialquery_object_id_t second)
                               { return first.value == second.value; }) == ids.end(),
            "fixture generated duplicate IDs");
}

void updatePopulation(ProductFixture& fixture, ovspatialquery_population_handle_t population,
                      ovspatialquery_ordinal_t ordinal, const std::vector<ovspatialquery_test_object_t>& objects)
{
    requireUniqueIds(objects);
    require(fixture.publishComplete(population, ordinal, objects) == OVSPATIALQUERY_API_SUCCESS,
            "product population update failed");
}

// The zero-capacity probe below only reports whether pairs were truncated
// (out_result->truncated), not the exact required count the removed private
// ovspatialquery_pair_snapshot's BUFFER_TOO_SMALL status used to report; the
// materialized buffer is instead sized directly from the brute-force oracle.
void queryAndCompare(ProductFixture& fixture, const std::vector<ovspatialquery_test_object_t>& populationA,
                     const std::vector<ovspatialquery_test_object_t>& populationB, ovspatialquery_ordinal_t ordinalA,
                     ovspatialquery_ordinal_t ordinalB, PairHistory& history)
{
    const std::vector<ovspatialquery_pair_t> expected = bruteForceOracle(populationA, populationB);

    ovspatialquery_pair_snapshot_desc_t desc{};
    desc.first  = populationTarget(fixture.firstPopulation());
    desc.second = populationTarget(fixture.secondPopulation());

    ovspatialquery_pair_buffer_t          probeBuffer{nullptr, 0u};
    ovspatialquery_pair_snapshot_result_t probe{};
    require(ovspatialquery_query_pairs_sync(fixture.instance(), &desc, &probeBuffer, &probe).status ==
                OVSPATIALQUERY_API_SUCCESS,
            "pair count probe failed");
    require(probe.stored_pair_count == 0u && probe.truncated == (expected.empty() ? 0u : 1u),
            "pair count probe returned the wrong truncation flag");
    require(ordinalA == ordinalB && probe.first_source_ordinal == ordinalA && probe.second_source_ordinal == ordinalB,
            "complete snapshot returned the wrong published ordinal");

    std::vector<ovspatialquery_pair_t>    actual(expected.size());
    ovspatialquery_pair_buffer_t          buffer{actual.data(), actual.size()};
    ovspatialquery_pair_snapshot_result_t materialized{};
    require(ovspatialquery_query_pairs_sync(fixture.instance(), &desc, &buffer, &materialized).status ==
                OVSPATIALQUERY_API_SUCCESS,
            "materialized pair snapshot failed");
    require(materialized.stored_pair_count == expected.size() && materialized.truncated == 0u,
            "materialized pair count differs from double oracle");
    std::sort(actual.begin(), actual.end(), pairLess);
    for (size_t index = 0u; index < actual.size(); ++index)
    {
        require(pairEqual(actual[index], expected[index]), "materialized pair identity differs from double oracle");
        if (index != 0u)
        {
            require(pairLess(actual[index - 1u], actual[index]),
                    "materialized pairs are not canonical and duplicate-free");
        }
    }
    const auto countMissing =
        [](const std::vector<ovspatialquery_pair_t>& source, const std::vector<ovspatialquery_pair_t>& other)
    {
        return static_cast<size_t>(
            std::count_if(source.begin(), source.end(),
                          [&other](const ovspatialquery_pair_t& pair)
                          { return !std::binary_search(other.begin(), other.end(), pair, pairLess); }));
    };
    history.began += countMissing(actual, history.current);
    history.ended += countMissing(history.current, actual);
    history.current = std::move(actual);
}

void rotateInputOrder(std::vector<ovspatialquery_test_object_t>& objects, DeterministicRandom& random)
{
    if (objects.size() > 1u)
    {
        const size_t offset = static_cast<size_t>(random.bounded(static_cast<uint32_t>(objects.size())));
        std::rotate(objects.begin(), objects.begin() + offset, objects.end());
    }
}

struct MutationCounters
{
    uint64_t inserts   = 0u;
    uint64_t removes   = 0u;
    uint64_t refits    = 0u;
    uint64_t teleports = 0u;
};

void animatePopulation(std::vector<ovspatialquery_test_object_t>& objects, bool secondPopulation, uint32_t frame,
                       ovspatialquery_object_id_t& nextId, DeterministicRandom& random, MutationCounters& counters)
{
    if (objects.empty())
    {
        for (uint32_t index = 0u; index < 16u; ++index)
        {
            const int32_t slot = static_cast<int32_t>(random.bounded(24u));
            objects.push_back(makeGridObject(nextId, slot, secondPopulation));
            ++nextId.value;
            ++counters.inserts;
        }
        rotateInputOrder(objects, random);
        return;
    }

    if (frame % 23u == 0u)
    {
        if ((frame / 23u) % 2u == 0u || objects.size() <= 8u)
        {
            const int32_t slot = static_cast<int32_t>(random.bounded(24u));
            objects.push_back(makeGridObject(nextId, slot, secondPopulation));
            ++nextId.value;
            ++counters.inserts;
        }
        else
        {
            const size_t index = static_cast<size_t>(random.bounded(static_cast<uint32_t>(objects.size())));
            objects.erase(objects.begin() + static_cast<std::ptrdiff_t>(index));
            ++counters.removes;
        }
        rotateInputOrder(objects, random);
        return;
    }

    const size_t changeCount = std::min(objects.size(), static_cast<size_t>(1u + random.bounded(5u)));
    for (size_t change = 0u; change < changeCount; ++change)
    {
        const size_t index =
            (static_cast<size_t>(random.bounded(static_cast<uint32_t>(objects.size()))) + change) % objects.size();
        int32_t slot = static_cast<int32_t>(random.bounded(24u));
        if (frame % 19u == 0u && change == 0u)
        {
            slot = (frame % 38u == 0u ? 1 : -1) * static_cast<int32_t>(500u + frame);
            ++counters.teleports;
        }
        setGridBounds(objects[index], slot, secondPopulation);
        ++counters.refits;
    }
    rotateInputOrder(objects, random);
}

void requireRejectedUpdatePreservesSnapshot(
    ProductFixture& fixture, ovspatialquery_population_handle_t population, ovspatialquery_ordinal_t baseOrdinal,
    ovspatialquery_ordinal_t targetOrdinal, const std::vector<ovspatialquery_test_object_t>& changed,
    ovspatialquery_api_status_t expectedStatus, const std::vector<ovspatialquery_test_object_t>& populationA,
    const std::vector<ovspatialquery_test_object_t>& populationB, ovspatialquery_ordinal_t ordinalA,
    ovspatialquery_ordinal_t ordinalB, PairHistory& history)
{
    const std::vector<ovspatialquery_pair_t> previous = history.current;
    require(fixture.tryEnqueueUpdate(population, baseOrdinal, targetOrdinal, changed) == expectedStatus,
            "rejected update returned the wrong status");
    queryAndCompare(fixture, populationA, populationB, ordinalA, ordinalB, history);
    require(pairVectorsEqual(previous, history.current), "rejected update changed the complete pair snapshot");
}

// Frames 3, 7, 23, and 27 exercise malformed-update admission rejection with
// a representative defect each: a duplicate object id, a zero (invalid)
// object id, a stale (non-advancing) ordinal, and a null update descriptor.
// The original frames 11, 15, and 19 corrupted a raw AABB directly (reversed
// or nonfinite bounds); the public API derives bounds from a registered
// geometry definition plus an instance pose instead of accepting a raw AABB,
// so those specific corruptions have no public equivalent and are dropped
// (see the file header comment).
void exerciseScheduledFailure(ProductFixture& fixture, uint32_t frame,
                              std::vector<ovspatialquery_test_object_t>& populationA,
                              std::vector<ovspatialquery_test_object_t>& populationB, ovspatialquery_ordinal_t ordinalA,
                              ovspatialquery_ordinal_t ordinalB, PairHistory& history)
{
    // Copy-construct the fixtures below: GCC 13 diagnoses a null memmove
    // destination when inlining vector copy assignment into this empty vector.
    std::vector<ovspatialquery_test_object_t> candidate;
    ovspatialquery_population_handle_t        population       = fixture.firstPopulation();
    ovspatialquery_api_status_t               expectedStatus   = OVSPATIALQUERY_API_INVALID_ARGUMENT;
    ovspatialquery_ordinal_t                  baseOrdinal      = ordinalA;
    ovspatialquery_ordinal_t                  attemptedOrdinal = ordinalA + 7u;

    switch (frame)
    {
    case 3u:
        candidate = std::vector<ovspatialquery_test_object_t>(populationA);
        require(!candidate.empty(), "duplicate failure fixture is empty");
        candidate.push_back(candidate.front());
        break;
    case 7u:
        population       = fixture.secondPopulation();
        baseOrdinal      = ordinalB;
        attemptedOrdinal = ordinalB + 7u;
        candidate        = std::vector<ovspatialquery_test_object_t>(populationB);
        require(!candidate.empty(), "zero-ID failure fixture is empty");
        candidate.front().id = {0u};
        break;
    case 23u:
        population       = fixture.secondPopulation();
        baseOrdinal      = ordinalB;
        attemptedOrdinal = ordinalB;
        candidate        = std::vector<ovspatialquery_test_object_t>(populationB);
        expectedStatus   = OVSPATIALQUERY_API_STALE_ORDINAL;
        break;
    case 27u: {
        const std::vector<ovspatialquery_pair_t> previous = history.current;
        require(
            ovspatialquery_enqueue_explicit_population_update(fixture.instance(), fixture.firstPopulation(), nullptr)
                    .status == OVSPATIALQUERY_API_INVALID_ARGUMENT,
            "rejected null update returned the wrong status");
        queryAndCompare(fixture, populationA, populationB, ordinalA, ordinalB, history);
        require(pairVectorsEqual(previous, history.current), "rejected update changed the complete pair snapshot");
        return;
    }
    default: return;
    }

    requireRejectedUpdatePreservesSnapshot(fixture, population, baseOrdinal, attemptedOrdinal, candidate,
                                           expectedStatus, populationA, populationB, ordinalA, ordinalB, history);
}

void populateInitialGrid(std::vector<ovspatialquery_test_object_t>& objects, bool secondPopulation,
                         ovspatialquery_object_id_t& nextId, DeterministicRandom& random)
{
    for (uint32_t index = 0u; index < 24u; ++index)
    {
        const int32_t slot = static_cast<int32_t>((index + random.bounded(5u)) % 16u);
        objects.push_back(makeGridObject(nextId, slot, secondPopulation));
        ++nextId.value;
    }
    rotateInputOrder(objects, random);
}

void runAnimatedSeed(uint64_t seed, uint32_t seedIndex)
{
    ProductFixture                            fixture;
    DeterministicRandom                       random(seed);
    PairHistory                               history;
    MutationCounters                          counters;
    std::vector<ovspatialquery_test_object_t> populationA;
    std::vector<ovspatialquery_test_object_t> populationB;
    ovspatialquery_object_id_t nextIdA{UINT64_C(100000) + static_cast<uint64_t>(seedIndex) * UINT64_C(10000)};
    ovspatialquery_object_id_t nextIdB{UINT64_C(200000) + static_cast<uint64_t>(seedIndex) * UINT64_C(10000)};
    ovspatialquery_ordinal_t   ordinalA = static_cast<int64_t>(seedIndex) * INT64_C(20000) + 2;
    ovspatialquery_ordinal_t   ordinalB = ordinalA;

    updatePopulation(fixture, fixture.firstPopulation(), ordinalA, populationA);
    updatePopulation(fixture, fixture.secondPopulation(), ordinalB, populationB);
    queryAndCompare(fixture, populationA, populationB, ordinalA, ordinalB, history);

    populateInitialGrid(populationA, false, nextIdA, random);
    ordinalA += 3u;
    ordinalB = ordinalA;
    updatePopulation(fixture, fixture.firstPopulation(), ordinalA, populationA);
    updatePopulation(fixture, fixture.secondPopulation(), ordinalB, populationB);
    queryAndCompare(fixture, populationA, populationB, ordinalA, ordinalB, history);
    populateInitialGrid(populationB, true, nextIdB, random);
    ordinalA += 7u;
    ordinalB = ordinalA;
    updatePopulation(fixture, fixture.firstPopulation(), ordinalA, populationA);
    updatePopulation(fixture, fixture.secondPopulation(), ordinalB, populationB);
    queryAndCompare(fixture, populationA, populationB, ordinalA, ordinalB, history);

    for (uint32_t frame = 0u; frame < kFramesPerSeed; ++frame)
    {
        if (frame == 63u)
        {
            counters.removes += populationA.size();
            populationA.clear();
        }
        else
        {
            animatePopulation(populationA, false, frame, nextIdA, random, counters);
        }
        ordinalA += 1u + random.bounded(3u);
        ordinalB = ordinalA;
        updatePopulation(fixture, fixture.firstPopulation(), ordinalA, populationA);
        updatePopulation(fixture, fixture.secondPopulation(), ordinalB, populationB);
        queryAndCompare(fixture, populationA, populationB, ordinalA, ordinalB, history);

        if (frame == 79u)
        {
            counters.removes += populationB.size();
            populationB.clear();
        }
        else
        {
            animatePopulation(populationB, true, frame + 5u, nextIdB, random, counters);
        }
        ordinalA += 4u + random.bounded(5u);
        ordinalB = ordinalA;
        updatePopulation(fixture, fixture.firstPopulation(), ordinalA, populationA);
        updatePopulation(fixture, fixture.secondPopulation(), ordinalB, populationB);
        queryAndCompare(fixture, populationA, populationB, ordinalA, ordinalB, history);

        exerciseScheduledFailure(fixture, frame, populationA, populationB, ordinalA, ordinalB, history);
    }

    require(counters.inserts != 0u && counters.removes != 0u, "animated seed missed insert/remove coverage");
    require(counters.refits != 0u, "animated seed missed stable-ID refit coverage");
    require(counters.teleports != 0u, "animated seed missed teleport coverage");
    require(history.began != 0u && history.ended != 0u, "animated seed recorded no began/ended changes");
    std::printf("seed=%llu frames=%u began=%zu ended=%zu\n", static_cast<unsigned long long>(seed), kFramesPerSeed,
                history.began, history.ended);
}

void testLargeCoordinates()
{
    // Near the input domain's 2^60 bound, where a float ulp is 2^36.
    const double base     = std::ldexp(1.0, 59);
    const double floatUlp = std::ldexp(1.0, 36);
    // Every object below shares this half-extent (1.0 * floatUlp, 0.5, 0.5),
    // distinct from the grid fixture's default catalog, so this fixture
    // registers its own single box definition for it.
    ProductFixture                                  fixture({{1u, {floatUlp, 0.5, 0.5}}});
    const std::vector<ovspatialquery_test_object_t> populationA{
        {{9001u}, {{base + 0.25 * floatUlp, 0.0, 0.0}, {base + 2.25 * floatUlp, 1.0, 1.0}}},
        {{9002u}, {{-base - 3.25 * floatUlp, 0.0, 0.0}, {-base - 1.25 * floatUlp, 1.0, 1.0}}},
    };
    const std::vector<ovspatialquery_test_object_t> populationB{
        {{9101u}, {{base + 1.75 * floatUlp, 0.0, 0.0}, {base + 3.75 * floatUlp, 1.0, 1.0}}},
        {{9102u}, {{-base - 2.75 * floatUlp, 0.0, 0.0}, {-base - 0.75 * floatUlp, 1.0, 1.0}}},
    };
    updatePopulation(fixture, fixture.firstPopulation(), 207u, populationA);
    updatePopulation(fixture, fixture.secondPopulation(), 207u, populationB);
    PairHistory history;
    queryAndCompare(fixture, populationA, populationB, 207u, 207u, history);
}

} // namespace

int main()
{
    const uint64_t seeds[] = {
        UINT64_C(0x123456789abcdef1),
        UINT64_C(0x9e3779b97f4a7c15),
        UINT64_C(0xd1b54a32d192ed03),
    };
    for (uint32_t index = 0u; index < 3u; ++index)
    {
        runAnimatedSeed(seeds[index], index);
    }
    testLargeCoordinates();
    std::puts("installed-library real DLL oracle stress passed");
    return 0;
}
