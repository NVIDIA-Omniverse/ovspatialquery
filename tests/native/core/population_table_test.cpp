/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "oracle_backend.h"
#include "ovspatialquery_population_table.h"
#include "ovspatialquery_typed_geometry.h"
#include "test_failure.h"

#include <atomic>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

extern "C" void ovspatialquery_test_current_api_instance_bootstrap_budget(void);

namespace
{

std::vector<ovspatialquery::detail::PopulationObject> snapshot(const ovspatialquery::detail::PopulationObject* objects,
                                                               uint64_t objectCount)
{
    return objects == nullptr ? std::vector<ovspatialquery::detail::PopulationObject>{}
                              : std::vector<ovspatialquery::detail::PopulationObject>(objects, objects + objectCount);
}

struct IdentityLifetime
{
    explicit IdentityLifetime(std::shared_ptr<uint32_t> destructionCount) : destructions(std::move(destructionCount)) {}

    ~IdentityLifetime() { ++*destructions; }

    std::shared_ptr<uint32_t> destructions;
};

class GenerationResourceLedger
{
  public:
    GenerationResourceLedger() : mState(std::make_shared<State>()) {}

    std::shared_ptr<void> attach(uint64_t hostCharge, uint64_t deviceCharge)
    {
        std::lock_guard<std::mutex> lock(mState->mutex);
        mState->hostUsed += hostCharge;
        mState->deviceUsed += deviceCharge;
        const std::shared_ptr<State> state = mState;
        return std::shared_ptr<void>(new uint8_t(0u),
                                     [state, hostCharge, deviceCharge](void* value)
                                     {
                                         delete static_cast<uint8_t*>(value);
                                         std::lock_guard<std::mutex> releaseLock(state->mutex);
                                         state->hostUsed -= hostCharge;
                                         state->deviceUsed -= deviceCharge;
                                     });
    }

    std::pair<uint64_t, uint64_t> usage() const
    {
        std::lock_guard<std::mutex> lock(mState->mutex);
        return {mState->hostUsed, mState->deviceUsed};
    }

  private:
    struct State
    {
        std::mutex mutex;
        uint64_t   hostUsed   = 0u;
        uint64_t   deviceUsed = 0u;
    };

    std::shared_ptr<State> mState;
};

ovspatialquery::detail::GenerationLease buildGeneration(ovspatialquery::detail::PopulationTable&        table,
                                                        ovspatialquery_ordinal_t                        ordinal,
                                                        const ovspatialquery::detail::PopulationObject* objects,
                                                        uint64_t objectCount, std::shared_ptr<void> resourceLease = {})
{
    ovspatialquery::detail::GenerationLease generation;
    REQUIRE(table.buildGeneration(ordinal, snapshot(objects, objectCount), {}, generation, std::move(resourceLease)) ==
            OVSPATIALQUERY_API_SUCCESS);
    return generation;
}

void publish(ovspatialquery::detail::PopulationTable& table, ovspatialquery::detail::PopulationKey population,
             ovspatialquery_ordinal_t ordinal, const ovspatialquery::detail::PopulationObject* objects,
             uint64_t objectCount, std::shared_ptr<void> resourceLease = {})
{
    ovspatialquery::detail::GenerationLease base;
    REQUIRE(table.beginUpdate(population, ordinal, nullptr, base) == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery::detail::GenerationLease generation =
        buildGeneration(table, ordinal, objects, objectCount, std::move(resourceLease));
    REQUIRE(table.publishUpdate(population, generation) == OVSPATIALQUERY_API_SUCCESS);
    generation.reset();
    base.reset();
    table.finishUpdate(population);
}

void testRetainedOrdinalStore(ovspatialquery_backend_runtime_t* runtime)
{
    ovspatialquery::detail::PopulationTable table(runtime);
    ovspatialquery::detail::PopulationKey   population;
    REQUIRE(table.create(2u, population) == OVSPATIALQUERY_API_SUCCESS);

    ovspatialquery::detail::GenerationLease noHead;
    REQUIRE(table.pin(population, noHead) == OVSPATIALQUERY_API_NOT_READY);
    REQUIRE(!noHead.initialized());
    REQUIRE(table.pinExact(population, 0, noHead) == OVSPATIALQUERY_API_NOT_READY);

    const ovspatialquery::detail::PopulationObject at100[] = {
        {{1001u}, {{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}, {}, {}, {}, {}},
    };
    const ovspatialquery::detail::PopulationObject at104[] = {
        {{1041u}, {{4.0, 0.0, 0.0}, {5.0, 1.0, 1.0}}, {}, {}, {}, {}},
    };
    const ovspatialquery::detail::PopulationObject at110[] = {
        {{1101u}, {{10.0, 0.0, 0.0}, {11.0, 1.0, 1.0}}, {}, {}, {}, {}},
    };
    const ovspatialquery::detail::PopulationObject at120[] = {
        {{1201u}, {{20.0, 0.0, 0.0}, {21.0, 1.0, 1.0}}, {}, {}, {}, {}},
    };
    const ovspatialquery::detail::PopulationObject at130[] = {
        {{1301u}, {{30.0, 0.0, 0.0}, {31.0, 1.0, 1.0}}, {}, {}, {}, {}},
    };
    const ovspatialquery::detail::PopulationObject at140[] = {
        {{1401u}, {{40.0, 0.0, 0.0}, {41.0, 1.0, 1.0}}, {}, {}, {}, {}},
    };

    const std::shared_ptr<uint32_t> identityDestructions = std::make_shared<uint32_t>(0u);
    publish(table, population, 100, at100, 1u, std::make_shared<IdentityLifetime>(identityDestructions));
    publish(table, population, 104, at104, 1u);

    ovspatialquery::detail::GenerationLease slowSensor;
    ovspatialquery::detail::GenerationLease fastSensor;
    REQUIRE(table.pinExact(population, 100, slowSensor) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(table.pin(population, fastSensor) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(slowSensor.ordinal() == 100 && fastSensor.ordinal() == 104);

    publish(table, population, 110, at110, 1u);
    ovspatialquery::detail::GenerationLease current;
    REQUIRE(table.pin(population, current) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(current.ordinal() == 110 && current.objects()[0].id.value == 1101u);
    REQUIRE(slowSensor.ordinal() == 100 && slowSensor.objects()[0].id.value == 1001u);
    REQUIRE(fastSensor.ordinal() == 104 && fastSensor.objects()[0].id.value == 1041u);
    REQUIRE(*identityDestructions == 0u);

    publish(table, population, 120, at120, 1u);
    ovspatialquery::detail::GenerationLease evicted;
    REQUIRE(table.pinExact(population, 100, evicted) == OVSPATIALQUERY_API_ORDINAL_MISMATCH);
    REQUIRE(!evicted.initialized());
    REQUIRE(table.pinExact(population, 104, evicted) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(evicted.ordinal() == 104);
    evicted.reset();
    REQUIRE(*identityDestructions == 0u);
    slowSensor.reset();
    REQUIRE(*identityDestructions == 1u);

    publish(table, population, 130, at130, 1u);
    REQUIRE(fastSensor.ordinal() == 104 && fastSensor.objects()[0].id.value == 1041u);

    ovspatialquery::detail::GenerationLease operationVersion;
    REQUIRE(table.retainPersistentReferences({population}) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(table.pin(population, operationVersion) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(operationVersion.ordinal() == 130);
    publish(table, population, 140, at140, 1u);
    REQUIRE(operationVersion.ordinal() == 130 && operationVersion.objects()[0].id.value == 1301u);
    table.finishPersistentReferences({population});
    operationVersion.reset();

    ovspatialquery::detail::GenerationLease updateBase;
    REQUIRE(table.beginUpdate(population, 150, nullptr, updateBase) == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery::detail::GenerationLease wrongOrdinal = buildGeneration(table, 151, at140, 1u);
    REQUIRE(table.publishUpdate(population, wrongOrdinal) == OVSPATIALQUERY_API_ORDINAL_MISMATCH);
    wrongOrdinal.reset();
    updateBase.reset();
    table.finishUpdate(population);

    current.reset();
    REQUIRE(table.pin(population, current) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(current.ordinal() == 140);
    REQUIRE(table.pinExact(population, 120, evicted) == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery::detail::GenerationLease retained130;
    REQUIRE(table.pinExact(population, 130, retained130) == OVSPATIALQUERY_API_SUCCESS);

    current.reset();
    evicted.reset();
    retained130.reset();
    fastSensor.reset();
    std::shared_ptr<void> deferredDestruction;
    REQUIRE(table.release(population, deferredDestruction) == OVSPATIALQUERY_API_SUCCESS);
    deferredDestruction.reset();
    REQUIRE(table.populationCount() == 0u);
}

void testHandleReuse(ovspatialquery_backend_runtime_t* runtime)
{
    ovspatialquery::detail::PopulationTable        table(runtime);
    const ovspatialquery::detail::PopulationObject objects[] = {
        {{1u}, {{0.0, 0.0, 0.0}, {2.0, 2.0, 2.0}}, {}, {}, {}, {}},
    };
    ovspatialquery::detail::PopulationKey original;
    REQUIRE(table.create(2u, 7, snapshot(objects, 1u), original) == OVSPATIALQUERY_API_SUCCESS);
    std::shared_ptr<void> deferredDestruction;
    REQUIRE(table.release(original, deferredDestruction) == OVSPATIALQUERY_API_SUCCESS);
    deferredDestruction.reset();

    ovspatialquery::detail::PopulationKey reused;
    REQUIRE(table.create(2u, 8, snapshot(objects, 1u), reused) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(reused.slot == original.slot && reused.generation != original.generation);
    ovspatialquery::detail::GenerationLease stale;
    REQUIRE(table.pin(original, stale) == OVSPATIALQUERY_API_INVALID_ARGUMENT);
    REQUIRE(table.release(reused, deferredDestruction) == OVSPATIALQUERY_API_SUCCESS);
}

void testGenerationResourceLeaseLifetime(ovspatialquery_backend_runtime_t* runtime)
{
    ovspatialquery::detail::PopulationTable        table(runtime);
    const ovspatialquery::detail::PopulationObject object[] = {
        {{1u}, {{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}, {}, {}, {}, {}},
    };
    ovspatialquery::detail::PopulationKey population;
    REQUIRE(table.create(2u, population) == OVSPATIALQUERY_API_SUCCESS);
    GenerationResourceLedger resources;

    const auto publishWithResources = [&](ovspatialquery_ordinal_t ordinal)
    { publish(table, population, ordinal, object, 1u, resources.attach(10u, 20u)); };

    publishWithResources(1);
    ovspatialquery::detail::GenerationLease heldEvicted;
    REQUIRE(table.pin(population, heldEvicted) == OVSPATIALQUERY_API_SUCCESS);
    publishWithResources(2);
    publishWithResources(3);
    publishWithResources(4);
    REQUIRE(resources.usage() == std::make_pair(UINT64_C(40), UINT64_C(80)));
    ovspatialquery::detail::GenerationLease missingLease;
    REQUIRE(table.pinExact(population, 1, missingLease) == OVSPATIALQUERY_API_ORDINAL_MISMATCH);
    REQUIRE(heldEvicted.ordinal() == 1);

    ovspatialquery::detail::GenerationLease head;
    REQUIRE(table.pin(population, head) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(head.ordinal() == 4);
    ovspatialquery::detail::GenerationLease retained2;
    ovspatialquery::detail::GenerationLease retained3;
    REQUIRE(table.pinExact(population, 2, retained2) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(table.pinExact(population, 3, retained3) == OVSPATIALQUERY_API_SUCCESS);
    retained2.reset();
    retained3.reset();
    head.reset();

    heldEvicted.reset();
    REQUIRE(resources.usage() == std::make_pair(UINT64_C(30), UINT64_C(60)));
    publishWithResources(5);
    REQUIRE(resources.usage() == std::make_pair(UINT64_C(30), UINT64_C(60)));

    std::shared_ptr<void> deferredDestruction;
    REQUIRE(table.release(population, deferredDestruction) == OVSPATIALQUERY_API_SUCCESS);
    deferredDestruction.reset();
    REQUIRE(resources.usage() == std::make_pair(UINT64_C(0), UINT64_C(0)));
}

void testPinPublishAndDestroyLinearize(ovspatialquery_backend_runtime_t* runtime)
{
    ovspatialquery::detail::PopulationTable        table(runtime);
    const ovspatialquery::detail::PopulationObject object[] = {
        {{1u}, {{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}, {}, {}, {}, {}},
    };
    ovspatialquery::detail::PopulationKey population;
    REQUIRE(table.create(2u, 0, snapshot(object, 1u), population) == OVSPATIALQUERY_API_SUCCESS);

    ovspatialquery::detail::GenerationLease operationVersion;
    REQUIRE(table.retainPersistentReferences({population}) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(table.pin(population, operationVersion) == OVSPATIALQUERY_API_SUCCESS);
    std::atomic<bool>     running{true};
    std::atomic<bool>     failed{false};
    std::atomic<uint64_t> pinCount{0u};

    std::thread publisher(
        [&]()
        {
            for (ovspatialquery_ordinal_t ordinal = 1; ordinal <= 40; ++ordinal)
            {
                publish(table, population, ordinal, object, 1u);
            }
            running.store(false, std::memory_order_release);
        });
    const auto pinHeads = [&]()
    {
        do
        {
            ovspatialquery::detail::GenerationLease head;
            const ovspatialquery_api_status_t       status = table.pin(population, head);
            if (status != OVSPATIALQUERY_API_SUCCESS || !head.initialized() || head.ordinal() < 0 ||
                head.ordinal() > 40)
            {
                failed.store(true, std::memory_order_release);
                return;
            }
            pinCount.fetch_add(1u, std::memory_order_relaxed);
        } while (running.load(std::memory_order_acquire));
    };
    std::thread firstReader(pinHeads);
    std::thread secondReader(pinHeads);
    std::thread destroyer(
        [&]()
        {
            while (running.load(std::memory_order_acquire))
            {
                std::shared_ptr<void> deferredDestruction;
                if (table.release(population, deferredDestruction) != OVSPATIALQUERY_API_BUSY ||
                    deferredDestruction != nullptr)
                {
                    failed.store(true, std::memory_order_release);
                    return;
                }
            }
        });

    publisher.join();
    firstReader.join();
    secondReader.join();
    destroyer.join();
    REQUIRE(!failed.load(std::memory_order_acquire));
    REQUIRE(pinCount.load(std::memory_order_relaxed) != 0u);

    std::shared_ptr<void> deferredDestruction;
    REQUIRE(table.release(population, deferredDestruction) == OVSPATIALQUERY_API_BUSY &&
            deferredDestruction == nullptr);
    REQUIRE(operationVersion.ordinal() == 0);
    table.finishPersistentReferences({population});
    operationVersion.reset();
    REQUIRE(table.release(population, deferredDestruction) == OVSPATIALQUERY_API_SUCCESS);
}

void testNoHeadDestruction(ovspatialquery_backend_runtime_t* runtime)
{
    ovspatialquery::detail::PopulationTable table(runtime);
    ovspatialquery::detail::PopulationKey   population;
    REQUIRE(table.create(0u, population) == OVSPATIALQUERY_API_SUCCESS);
    std::shared_ptr<void> deferredDestruction;
    REQUIRE(table.release(population, deferredDestruction) == OVSPATIALQUERY_API_SUCCESS);
    deferredDestruction.reset();
    REQUIRE(table.populationCount() == 0u);
}

// A version's change record by comparison with its base (ADR-0019): moved
// objects, participation changes, view membership differences and objects
// appended after the base are listed by position; no base, a removal, a
// reordering, a different set of views or more than a quarter changed records
// everything.
void testDescribeChanges(ovspatialquery_backend_runtime_t* runtime)
{
    using ovspatialquery::detail::GenerationLease;
    using ovspatialquery::detail::GenerationViewMembership;
    using ovspatialquery::detail::ObjectTable;
    using ovspatialquery::detail::PopulationObject;
    using ovspatialquery::detail::PopulationTable;
    using ovspatialquery::detail::VersionChangeRecord;
    const auto object = [](uint64_t id, double x, uint64_t bits)
    {
        PopulationObject value{};
        value.id.value          = id;
        value.bounds            = {{x, 0.0, 0.0}, {x + 1.0, 1.0, 1.0}};
        value.participationBits = {{bits, 0u}};
        return value;
    };
    const auto ids = [](std::vector<uint64_t> values)
    { return std::make_shared<const std::vector<uint64_t>>(std::move(values)); };
    std::vector<PopulationObject> objects;
    for (uint64_t id = 1u; id <= 16u; ++id)
    {
        objects.push_back(object(id, 3.0 * static_cast<double>(id), 1u));
    }
    const std::vector<GenerationViewMembership> memberships{{{11u}, ids({2u, 4u, 6u})}, {{12u}, ids({1u, 2u})}};
    PopulationTable                             table(runtime);
    GenerationLease                             base;
    REQUIRE(table.buildGeneration(10, objects, memberships, base) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(base.changeRecord().everything);

    const auto describe = [&](const GenerationLease& from, const std::vector<PopulationObject>& current,
                              const std::vector<GenerationViewMembership>& currentMemberships)
    {
        VersionChangeRecord record;
        REQUIRE(PopulationTable::describeChanges(from, ObjectTable(current), currentMemberships, record) ==
                OVSPATIALQUERY_API_SUCCESS);
        return record;
    };
    const auto lists = [](const VersionChangeRecord& record, std::vector<uint32_t> positions)
    { return !record.everything && record.baseOrdinal == 10 && record.positions == positions; };

    // Unchanged, including a view membership republished with equal content.
    REQUIRE(lists(describe(base, objects, {{{12u}, ids({1u, 2u})}, {{11u}, ids({2u, 4u, 6u})}}), {}));

    // A move, a participation change and a view gaining a member.
    std::vector<PopulationObject> changed = objects;
    changed[1].bounds.maximum[0] += 0.5;
    changed[3].participationBits = {{2u, 0u}};
    REQUIRE(lists(describe(base, changed, {{{11u}, ids({2u, 4u, 6u, 8u})}, memberships[1]}), {1u, 3u, 7u}));

    // An object appended after the base's last one, joining a view.
    std::vector<PopulationObject> appended = objects;
    appended.push_back(object(17u, 60.0, 1u));
    REQUIRE(lists(describe(base, appended, {{{11u}, ids({2u, 4u, 6u, 17u})}, memberships[1]}), {16u}));

    // An object that gains stored exact geometry with unchanged bounds is
    // listed, so a version derived from its record counts it.
    std::vector<PopulationObject>                stored = objects;
    ovspatialquery::detail::SphereGeometrySource sphere{};
    sphere.center = {15.5, 0.5, 0.5};
    sphere.radius = 0.5;
    REQUIRE(ovspatialquery::detail::createSphereExactGeometry(sphere, stored[4].exactGeometry) ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(lists(describe(base, stored, memberships), {4u}));

    // A version built from its record answers like one built from scratch.
    const auto requireDerivedMatches = [&](const std::vector<PopulationObject>&         current,
                                           const std::vector<GenerationViewMembership>& currentMemberships)
    {
        GenerationLease derived;
        GenerationLease scratch;
        REQUIRE(table.buildGeneration(11, ObjectTable(current), currentMemberships, derived, {}, {}, false, {}, {},
                                      describe(base, current, currentMemberships), base) == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(table.buildGeneration(11, current, currentMemberships, scratch) == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(!derived.changeRecord().everything && scratch.changeRecord().everything);
        REQUIRE(derived.hasOnlyStoredExactGeometry() == scratch.hasOnlyStoredExactGeometry());
    };
    requireDerivedMatches(appended, memberships);
    requireDerivedMatches(stored, memberships);
    // From a base where one object lacks stored geometry, gaining it with
    // unchanged bounds makes the derived version answer with stored geometry only.
    std::vector<PopulationObject> allStored = objects;
    for (PopulationObject& value : allStored)
    {
        value.exactGeometry = stored[4].exactGeometry;
    }
    std::vector<PopulationObject> missingOne = allStored;
    missingOne[4].exactGeometry.reset();
    GenerationLease incomplete;
    REQUIRE(table.buildGeneration(12, missingOne, memberships, incomplete) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(!incomplete.hasOnlyStoredExactGeometry());
    GenerationLease completed;
    REQUIRE(table.buildGeneration(13, ObjectTable(allStored), memberships, completed, {}, {}, false, {}, {},
                                  describe(incomplete, allStored, memberships),
                                  incomplete) == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(!completed.changeRecord().everything && completed.hasOnlyStoredExactGeometry());

    // Everything: no base, a removal, a reordering, a different view set, and
    // more than a quarter of the objects changed.
    REQUIRE(describe(GenerationLease(), objects, memberships).everything);
    std::vector<PopulationObject> removed(objects.begin() + 1, objects.end());
    REQUIRE(describe(base, removed, memberships).everything);
    std::vector<PopulationObject> reordered = objects;
    std::swap(reordered[4], reordered[5]);
    REQUIRE(describe(base, reordered, memberships).everything);
    REQUIRE(describe(base, objects, {memberships[0]}).everything);
    std::vector<PopulationObject> busy = objects;
    for (size_t index = 0u; index <= VersionChangeRecord::positionLimit(busy.size()); ++index)
    {
        busy[index].bounds.minimum[1] -= 1.0;
    }
    REQUIRE(describe(base, busy, memberships).everything);
}

} // namespace

int main()
{
    ovspatialquery_backend_runtime_t* runtime = nullptr;
    REQUIRE(ovspatialquery_test_oracle_create_runtime(&runtime) == OVSPATIALQUERY_API_SUCCESS);
    testRetainedOrdinalStore(runtime);
    testHandleReuse(runtime);
    testGenerationResourceLeaseLifetime(runtime);
    testPinPublishAndDestroyLinearize(runtime);
    testNoHeadDestruction(runtime);
    testDescribeChanges(runtime);
    ovspatialquery_test_oracle_destroy_runtime(runtime);
    ovspatialquery_test_current_api_instance_bootstrap_budget();
    return 0;
}
