/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../support/native/public_api_inputs.h"
#include "../../support/native/test_failure.h"
#include <ovspatialquery/ovspatialquery_explicit.h>
#include <ovspatialquery/ovspatialquery_pairs.h>

#include "ovspatialquery_instance_operation_runtime.h"
#include "ovspatialquery_pair_materialization.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <set>
#include <utility>
#include <vector>

// Pair discovery through the public API: snapshot roles and canonical
// ordering, the world-AABB predicate with contact tolerance, truncation and its
// early stop, target validation and allocation failure with preserved outputs,
// and forward-only reporter advances with entered/exited deltas,
// STALE/BACKWARD ordinals, and capacity outcomes.

namespace
{

using PairSet = std::set<std::pair<uint64_t, uint64_t>>;

// Two sphere definitions cover every interval this test publishes: unit
// radius for width-2 intervals and half radius for width-1 intervals.
constexpr uint64_t kUnitSphere = 1u;
constexpr uint64_t kHalfSphere = 2u;

void registerSpheres(ovspatialquery_instance_handle_t instance)
{
    uint64_t                                       values[2] = {kUnitSphere, kHalfSphere};
    double                                         radii[2]  = {1.0, 0.5};
    InputArray                                     valueArray(values, 2, kDLUInt);
    InputArray                                     radiusArray(radii, 2, kDLFloat);
    const ovspatialquery_sphere_definition_table_t spheres{&valueArray.value, &radiusArray.value};
    waitAndDestroy(instance, ovspatialquery_enqueue_sphere_registration(instance, &spheres));
}

// One object occupying [minimumX, maximumX] on X; every object shares the same
// Y and Z extent so only X decides overlap.
struct Interval
{
    uint64_t id;
    double   minimumX;
    double   maximumX;
};

// One unit-radius sphere per width-2 interval and one half-radius sphere per
// width-1 interval, centered on the interval at Y = Z = 0.5.
void addIntervals(InstanceRows& rows, const std::vector<Interval>& intervals)
{
    for (const Interval& interval : intervals)
    {
        const double width = interval.maximumX - interval.minimumX;
        REQUIRE(width == 2.0 || width == 1.0);
        rows.add(interval.id, OVSPATIALQUERY_GEOMETRY_SPHERE, width == 2.0 ? kUnitSphere : kHalfSphere,
                 0.5 * (interval.minimumX + interval.maximumX), 0.5, 0.5);
    }
}

void publishSet(ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population,
                ovspatialquery_ordinal_t ordinal, const std::vector<Interval>& intervals)
{
    InstanceRows rows;
    addIntervals(rows, intervals);
    const ovspatialquery_explicit_population_set_t set{ordinal, rows.table(), nullptr, 0u};
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set));
    ovspatialquery_population_info_t info{};
    REQUIRE(ovspatialquery_get_population_info(instance, population, &info).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(info.ready != 0u && info.head_ordinal == ordinal && info.object_count == intervals.size());
}

void publishUpdate(ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population,
                   ovspatialquery_ordinal_t baseOrdinal, ovspatialquery_ordinal_t targetOrdinal,
                   const std::vector<Interval>& changed, std::vector<uint64_t> removed)
{
    InstanceRows rows;
    addIntervals(rows, changed);
    InputArray removedArray(removed.data(), static_cast<int64_t>(removed.size()), kDLUInt);
    const ovspatialquery_explicit_population_update_t update{
        baseOrdinal,  targetOrdinal,       OVSPATIALQUERY_EXPLICIT_UPDATE_COMPLETE_ROWS,
        rows.table(), &removedArray.value, nullptr};
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_update(instance, population, &update));
    ovspatialquery_population_info_t info{};
    REQUIRE(ovspatialquery_get_population_info(instance, population, &info).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(info.ready != 0u && info.head_ordinal == targetOrdinal);
}

ovspatialquery_query_target_t exactTarget(ovspatialquery_population_handle_t population,
                                          ovspatialquery_ordinal_t           ordinal)
{
    ovspatialquery_query_target_t target = populationTarget(population);
    target.ordinal.kind                  = OVSPATIALQUERY_ORDINAL_RETAINED_EXACT;
    target.ordinal.source_ordinal        = ordinal;
    return target;
}

ovspatialquery_pair_snapshot_desc_t snapshotDesc(ovspatialquery_query_target_t first,
                                                 ovspatialquery_query_target_t second, double contactTolerance = 0.0)
{
    ovspatialquery_pair_snapshot_desc_t desc{};
    desc.first             = first;
    desc.second            = second;
    desc.contact_tolerance = contactTolerance;
    return desc;
}

// Runs the snapshot three ways: a buffer large enough for everything, a
// zero-capacity probe, and a buffer one row short of the complete set. The
// last two must report truncation, the first must not, and every stored row
// must be a distinct pair drawn from the complete set.
PairSet snapshotPairs(ovspatialquery_instance_handle_t instance, const ovspatialquery_pair_snapshot_desc_t& desc,
                      ovspatialquery_ordinal_t expectedFirstOrdinal, ovspatialquery_ordinal_t expectedSecondOrdinal)
{
    std::vector<ovspatialquery_pair_t>    rows(16u);
    ovspatialquery_pair_buffer_t          buffer{rows.data(), rows.size()};
    ovspatialquery_pair_snapshot_result_t result{};
    REQUIRE(ovspatialquery_query_pairs_sync(instance, &desc, &buffer, &result).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.first_source_ordinal == expectedFirstOrdinal &&
            result.second_source_ordinal == expectedSecondOrdinal && result.truncated == 0u &&
            result.stored_pair_count <= rows.size());
    const uint64_t completeCount = result.stored_pair_count;
    PairSet        pairs;
    for (uint64_t index = 0u; index < completeCount; ++index)
    {
        REQUIRE(pairs.insert({rows[index].first.value, rows[index].second.value}).second);
    }

    ovspatialquery_pair_buffer_t          probe{nullptr, 0u};
    ovspatialquery_pair_snapshot_result_t probeResult{};
    REQUIRE(ovspatialquery_query_pairs_sync(instance, &desc, &probe, &probeResult).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(probeResult.stored_pair_count == 0u && probeResult.truncated == (completeCount > 0u ? 1u : 0u) &&
            probeResult.first_source_ordinal == expectedFirstOrdinal &&
            probeResult.second_source_ordinal == expectedSecondOrdinal);

    if (completeCount > 0u)
    {
        std::vector<ovspatialquery_pair_t>    shortRows(static_cast<size_t>(completeCount));
        ovspatialquery_pair_buffer_t          shortBuffer{shortRows.data(), completeCount - 1u};
        ovspatialquery_pair_snapshot_result_t shortResult{};
        REQUIRE(ovspatialquery_query_pairs_sync(instance, &desc, &shortBuffer, &shortResult).status ==
                OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(shortResult.stored_pair_count == completeCount - 1u && shortResult.truncated == 1u);
        for (uint64_t index = 0u; index < shortResult.stored_pair_count; ++index)
        {
            REQUIRE(pairs.count({shortRows[index].first.value, shortRows[index].second.value}) == 1u);
        }
    }
    return pairs;
}

// Rejected snapshots must leave the caller's buffer rows and result untouched.
void requireSnapshotRejected(ovspatialquery_instance_handle_t instance, const ovspatialquery_pair_snapshot_desc_t& desc,
                             ovspatialquery_api_status_t expected)
{
    ovspatialquery_pair_t                 row{{91u}, {92u}};
    ovspatialquery_pair_buffer_t          buffer{&row, 1u};
    ovspatialquery_pair_snapshot_result_t result{93, 94, 95u, 96u};
    const ovspatialquery_result_t         status = ovspatialquery_query_pairs_sync(instance, &desc, &buffer, &result);
    if (status.status != expected)
    {
        std::fprintf(stderr, "pair snapshot status %u, expected %u\n", static_cast<unsigned>(status.status),
                     static_cast<unsigned>(expected));
    }
    REQUIRE(status.status == expected);
    REQUIRE(row.first.value == 91u && row.second.value == 92u && result.first_source_ordinal == 93 &&
            result.second_source_ordinal == 94 && result.stored_pair_count == 95u && result.truncated == 96u);
}

PairSet tensorPairs(const DLTensor& tensor)
{
    REQUIRE(tensor.ndim == 2 && tensor.shape != nullptr && tensor.shape[1] == 2 && tensor.dtype.code == kDLUInt &&
            tensor.dtype.bits == 64u && tensor.device.device_type == kDLCPU);
    PairSet       pairs;
    const int64_t rowStride    = tensor.strides != nullptr ? tensor.strides[0] : 2;
    const int64_t columnStride = tensor.strides != nullptr ? tensor.strides[1] : 1;
    for (int64_t row = 0; row < tensor.shape[0]; ++row)
    {
        REQUIRE(tensor.data != nullptr);
        const uint64_t* base =
            static_cast<const uint64_t*>(tensor.data) + tensor.byte_offset / sizeof(uint64_t) + row * rowStride;
        REQUIRE(pairs.insert({base[0], base[columnStride]}).second);
    }
    return pairs;
}

struct AdvanceOutcome
{
    ovspatialquery_op_id_t                operation = OVSPATIALQUERY_INVALID_OP;
    ovspatialquery_pair_reporter_result_t result{};
    PairSet                               entered;
    PairSet                               exited;
};

ovspatialquery_pair_reporter_advance_t headAdvance()
{
    ovspatialquery_pair_reporter_advance_t advance{};
    advance.first_ordinal.kind  = OVSPATIALQUERY_ORDINAL_HEAD;
    advance.second_ordinal.kind = OVSPATIALQUERY_ORDINAL_HEAD;
    return advance;
}

// Enqueues one advance, waits for terminal success, and fetches the typed
// result. The caller destroys the operation once it has inspected the deltas.
AdvanceOutcome advance(ovspatialquery_instance_handle_t instance, ovspatialquery_pair_reporter_handle_t reporter,
                       const ovspatialquery_pair_reporter_advance_t& selection)
{
    ovspatialquery_pair_reporter_reservation_t reservation{};
    const ovspatialquery_enqueue_result_t      admission =
        ovspatialquery_enqueue_pair_reporter_advance(instance, reporter, &selection, &reservation);
    REQUIRE(admission.status == OVSPATIALQUERY_API_SUCCESS && admission.op_id != OVSPATIALQUERY_INVALID_OP);
    REQUIRE(reservation.result_device.device_type == kDLCPU && reservation.result_ready.value == 0u);
    ovspatialquery_op_wait_result_t wait{};
    REQUIRE(ovspatialquery_wait_op(instance, admission.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &wait).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(wait.state == OVSPATIALQUERY_OP_COMPLETED);
    AdvanceOutcome outcome;
    outcome.operation = admission.op_id;
    REQUIRE(ovspatialquery_fetch_pair_reporter(instance, admission.op_id, &outcome.result).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(outcome.result.result_slot == reservation.result_slot && outcome.result.result_ready.value == 0u);
    outcome.entered = tensorPairs(outcome.result.entered_pairs);
    outcome.exited  = tensorPairs(outcome.result.exited_pairs);
    REQUIRE(outcome.entered.size() + outcome.exited.size() <= reservation.delta_pair_capacity);
    return outcome;
}

void destroyOperation(ovspatialquery_instance_handle_t instance, ovspatialquery_op_id_t operation)
{
    REQUIRE(ovspatialquery_destroy_op(instance, &operation).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(operation == OVSPATIALQUERY_INVALID_OP);
}

void requireAdvanceRejected(ovspatialquery_instance_handle_t instance, ovspatialquery_pair_reporter_handle_t reporter,
                            const ovspatialquery_pair_reporter_advance_t& selection,
                            ovspatialquery_api_status_t                   expected)
{
    ovspatialquery_pair_reporter_reservation_t reservation{};
    reservation.result_slot         = 77u;
    reservation.delta_pair_capacity = 78u;
    const ovspatialquery_enqueue_result_t admission =
        ovspatialquery_enqueue_pair_reporter_advance(instance, reporter, &selection, &reservation);
    if (admission.status != expected)
    {
        std::fprintf(stderr, "pair reporter advance status %u, expected %u\n", static_cast<unsigned>(admission.status),
                     static_cast<unsigned>(expected));
    }
    REQUIRE(admission.status == expected && admission.op_id == OVSPATIALQUERY_INVALID_OP);
    REQUIRE(reservation.result_slot == 77u && reservation.delta_pair_capacity == 78u);
}

ovspatialquery_pair_reporter_handle_t createReporter(ovspatialquery_instance_handle_t   instance,
                                                     ovspatialquery_population_handle_t first,
                                                     ovspatialquery_population_handle_t second,
                                                     uint64_t activePairCapacity, uint64_t deltaPairCapacity)
{
    ovspatialquery_pair_reporter_desc_t desc{};
    desc.first                                     = populationRef(first);
    desc.second                                    = populationRef(second);
    desc.active_pair_capacity                      = activePairCapacity;
    desc.delta_pair_capacity                       = deltaPairCapacity;
    desc.result_slot_count                         = 2u;
    desc.contact_tolerance                         = 0.0;
    desc.result_device                             = DLDevice{kDLCPU, 0};
    ovspatialquery_pair_reporter_handle_t reporter = OVSPATIALQUERY_INVALID_PAIR_REPORTER;
    // A tolerance beyond the input domain's 2^60 is malformed, as for snapshots.
    desc.contact_tolerance = 0x1p61;
    REQUIRE(ovspatialquery_create_pair_reporter(instance, &desc, &reporter).status ==
                OVSPATIALQUERY_API_INVALID_ARGUMENT &&
            reporter == OVSPATIALQUERY_INVALID_PAIR_REPORTER);
    desc.contact_tolerance = 0.0;
    REQUIRE(ovspatialquery_create_pair_reporter(instance, &desc, &reporter).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(reporter != OVSPATIALQUERY_INVALID_PAIR_REPORTER);
    return reporter;
}

const PairSet kPairsAtSeven{{1u, 1u}, {1u, 2u}, {2u, 1u}, {2u, 2u}, {3u, 4u}};
const PairSet kPairsAtEight{{1u, 1u}, {1u, 2u}, {5u, 4u}};

} // namespace

int main()
{
    // Invalid or fabricated instance handles are rejected before anything else
    // is inspected and leave every caller output untouched.
    const ovspatialquery_pair_snapshot_desc_t nullDesc{};
    requireSnapshotRejected(OVSPATIALQUERY_INVALID_INSTANCE, nullDesc, OVSPATIALQUERY_API_INVALID_HANDLE);
    requireSnapshotRejected(ovspatialquery_instance_handle_t{UINT64_C(0xfedcba9876543210)}, nullDesc,
                            OVSPATIALQUERY_API_INVALID_HANDLE);

    // One operation slot: every operation below is destroyed before the next
    // one is admitted, except where the test deliberately holds the slot.
    const ovspatialquery_instance_desc_t instanceDesc{1u, 1u, UINT64_C(16) * 1024u * 1024u, nullptr, 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&instanceDesc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_instance_handle_t foreignInstance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&instanceDesc, &foreignInstance).status == OVSPATIALQUERY_API_SUCCESS);
    registerSpheres(instance);
    registerSpheres(foreignInstance);

    const ovspatialquery_population_handle_t first       = createExplicitPopulation(instance, 1u);
    const ovspatialquery_population_handle_t second      = createExplicitPopulation(instance, 1u);
    const ovspatialquery_population_handle_t unpublished = createExplicitPopulation(instance, 0u);
    const ovspatialquery_population_handle_t foreign     = createExplicitPopulation(foreignInstance, 0u);
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_BUSY);

    publishSet(instance, first, 7, {{1u, 0.0, 2.0}, {2u, 0.0, 2.0}, {3u, 5.0, 6.0}});
    publishSet(instance, second, 7, {{1u, 0.5, 1.5}, {2u, 0.0, 2.0}, {4u, 5.0, 6.0}});
    publishSet(foreignInstance, foreign, 7, {{1u, 0.0, 2.0}, {5u, 0.0, 2.0}});

    // Distinct populations keep their roles; equal numeric IDs across
    // populations are different objects and pair normally.
    REQUIRE(snapshotPairs(instance, snapshotDesc(populationTarget(first), populationTarget(second)), 7, 7) ==
            kPairsAtSeven);
    // A snapshot with capacity K stops at the K+1st matching pair, which proves
    // truncation, instead of visiting all five. Results cannot show where the
    // traversal stopped, so this test-build match counter is the only guard of
    // that bound.
    {
        const ovspatialquery_pair_snapshot_desc_t desc =
            snapshotDesc(populationTarget(first), populationTarget(second));
        std::array<ovspatialquery_pair_t, 2u> rows{};
        ovspatialquery_pair_buffer_t          buffer{rows.data(), rows.size()};
        ovspatialquery_pair_snapshot_result_t result{};
        ovspatialquery::detail::resetPairSnapshotObservedMatchCountForTest();
        REQUIRE(ovspatialquery_query_pairs_sync(instance, &desc, &buffer, &result).status ==
                    OVSPATIALQUERY_API_SUCCESS &&
                result.stored_pair_count == 2u && result.truncated == 1u &&
                ovspatialquery::detail::pairSnapshotObservedMatchCountForTest() == 3u);
    }
    REQUIRE(snapshotPairs(instance, snapshotDesc(populationTarget(second), populationTarget(first)), 7, 7) ==
            PairSet{{1u, 1u}, {1u, 2u}, {2u, 1u}, {2u, 2u}, {4u, 3u}});
    // An identical selection is unordered: no self-pairs, smaller ID first.
    REQUIRE(snapshotPairs(instance, snapshotDesc(populationTarget(first), populationTarget(first)), 7, 7) ==
            PairSet{{1u, 2u}});
    REQUIRE(snapshotPairs(instance, snapshotDesc(populationTarget(second), populationTarget(second)), 7, 7) ==
            PairSet{{1u, 2u}});
    REQUIRE(snapshotPairs(foreignInstance, snapshotDesc(populationTarget(foreign), populationTarget(foreign)), 7, 7) ==
            PairSet{{1u, 5u}});
    // The predicate is the inclusive world-AABB test with the caller's margin:
    // a margin of 4 bridges the gap between [0,2] and [5,6] in both roles.
    REQUIRE(snapshotPairs(instance, snapshotDesc(populationTarget(first), populationTarget(second), 4.0), 7, 7) ==
            PairSet{{1u, 1u}, {1u, 2u}, {1u, 4u}, {2u, 1u}, {2u, 2u}, {2u, 4u}, {3u, 1u}, {3u, 2u}, {3u, 4u}});

    // Target validation: malformed kind, negative tolerance or one beyond the
    // input domain, invalid, foreign, and unpublished handles, and an exact
    // ordinal that was never retained.
    ovspatialquery_query_target_t malformed = populationTarget(first);
    malformed.target.kind                   = static_cast<ovspatialquery_target_kind_t>(99);
    requireSnapshotRejected(instance, snapshotDesc(malformed, populationTarget(second)),
                            OVSPATIALQUERY_API_INVALID_ARGUMENT);
    requireSnapshotRejected(instance, snapshotDesc(populationTarget(first), populationTarget(second), -1.0),
                            OVSPATIALQUERY_API_INVALID_ARGUMENT);
    requireSnapshotRejected(instance, snapshotDesc(populationTarget(first), populationTarget(second), 0x1p61),
                            OVSPATIALQUERY_API_INVALID_ARGUMENT);
    requireSnapshotRejected(instance,
                            snapshotDesc(populationTarget(first), populationTarget(OVSPATIALQUERY_INVALID_POPULATION)),
                            OVSPATIALQUERY_API_INVALID_HANDLE);
    requireSnapshotRejected(instance, snapshotDesc(populationTarget(first), populationTarget(foreign)),
                            OVSPATIALQUERY_API_INVALID_HANDLE);
    requireSnapshotRejected(instance, snapshotDesc(populationTarget(first), populationTarget(unpublished)),
                            OVSPATIALQUERY_API_NOT_READY);
    requireSnapshotRejected(instance, snapshotDesc(populationTarget(first), exactTarget(first, 5)),
                            OVSPATIALQUERY_API_ORDINAL_NOT_RETAINED);

    // A reporter retains its targets for its lifetime.
    const ovspatialquery_pair_reporter_handle_t reporter = createReporter(instance, first, second, 8u, 8u);
    REQUIRE(ovspatialquery_destroy_population(instance, first).status == OVSPATIALQUERY_API_BUSY);

    // The first advance compares against an empty baseline: everything enters.
    AdvanceOutcome initial = advance(instance, reporter, headAdvance());
    REQUIRE(initial.result.outcome == OVSPATIALQUERY_PAIR_REPORTER_ADVANCED &&
            initial.result.first_source_ordinal == 7 && initial.result.second_source_ordinal == 7 &&
            initial.result.accepted_active_pair_count == kPairsAtSeven.size() && initial.entered == kPairsAtSeven &&
            initial.exited.empty());
    destroyOperation(instance, initial.operation);

    // Move object 2 away, add object 5, and move object 4 next to it.
    publishUpdate(instance, first, 7, 8, {{2u, 10.0, 12.0}, {5u, 20.0, 21.0}}, {});
    publishUpdate(instance, second, 7, 8, {{4u, 20.0, 21.0}}, {});
    REQUIRE(snapshotPairs(instance, snapshotDesc(populationTarget(first), populationTarget(second)), 8, 8) ==
            kPairsAtEight);
    REQUIRE(snapshotPairs(instance, snapshotDesc(exactTarget(first, 7), exactTarget(second, 7)), 7, 7) ==
            kPairsAtSeven);
    REQUIRE(snapshotPairs(instance, snapshotDesc(exactTarget(first, 7), populationTarget(second)), 7, 8) ==
            PairSet{{1u, 1u}, {1u, 2u}, {2u, 1u}, {2u, 2u}});
    // Two selections of one population must agree on the ordinal.
    requireSnapshotRejected(instance, snapshotDesc(exactTarget(first, 7), populationTarget(first)),
                            OVSPATIALQUERY_API_ORDINAL_MISMATCH);
    // No object of `first` overlaps another at ordinal 8: the complete set is
    // empty and its zero-capacity probe is not truncated.
    REQUIRE(snapshotPairs(instance, snapshotDesc(populationTarget(first), populationTarget(first)), 8, 8).empty());
    // A pair storage failure is OUT_OF_MEMORY with the caller's rows and result
    // untouched, and the retry succeeds. The zero-capacity probe allocates
    // nothing, so it succeeds with the failure armed and leaves it armed.
    {
        const ovspatialquery_pair_snapshot_desc_t desc =
            snapshotDesc(populationTarget(first), populationTarget(second));
        ovspatialquery::detail::failNextPairMaterializationAllocationForTest();
        ovspatialquery_pair_buffer_t          probe{nullptr, 0u};
        ovspatialquery_pair_snapshot_result_t probeResult{};
        REQUIRE(ovspatialquery_query_pairs_sync(instance, &desc, &probe, &probeResult).status ==
                    OVSPATIALQUERY_API_SUCCESS &&
                probeResult.stored_pair_count == 0u && probeResult.truncated == 1u);
        requireSnapshotRejected(instance, desc, OVSPATIALQUERY_API_OUT_OF_MEMORY);
        REQUIRE(snapshotPairs(instance, desc, 8, 8) == kPairsAtEight);
    }

    // The second advance reports only the set difference against the baseline.
    AdvanceOutcome moved = advance(instance, reporter, headAdvance());
    REQUIRE(moved.result.outcome == OVSPATIALQUERY_PAIR_REPORTER_ADVANCED && moved.result.first_source_ordinal == 8 &&
            moved.result.second_source_ordinal == 8 &&
            moved.result.accepted_active_pair_count == kPairsAtEight.size() && moved.entered == PairSet{{5u, 4u}} &&
            moved.exited == PairSet{{2u, 1u}, {2u, 2u}, {3u, 4u}});
    // Fetch is repeatable and non-consuming.
    ovspatialquery_pair_reporter_result_t again{};
    REQUIRE(ovspatialquery_fetch_pair_reporter(instance, moved.operation, &again).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(tensorPairs(again.entered_pairs) == moved.entered && tensorPairs(again.exited_pairs) == moved.exited);
    destroyOperation(instance, moved.operation);

    // No endpoint progressed: STALE_ORDINAL. An endpoint older than the
    // accepted ordinal: BACKWARD_ORDINAL. Neither issues an operation.
    requireAdvanceRejected(instance, reporter, headAdvance(), OVSPATIALQUERY_API_STALE_ORDINAL);
    ovspatialquery_pair_reporter_advance_t backward = headAdvance();
    backward.first_ordinal.kind                     = OVSPATIALQUERY_ORDINAL_RETAINED_EXACT;
    backward.first_ordinal.source_ordinal           = 7;
    requireAdvanceRejected(instance, reporter, backward, OVSPATIALQUERY_API_BACKWARD_ORDINAL);
    requireAdvanceRejected(instance, OVSPATIALQUERY_INVALID_PAIR_REPORTER, headAdvance(),
                           OVSPATIALQUERY_API_INVALID_HANDLE);
    // A reporter over a population with no published version cannot advance.
    const ovspatialquery_pair_reporter_handle_t notPublished = createReporter(instance, first, unpublished, 8u, 8u);
    requireAdvanceRejected(instance, notPublished, headAdvance(), OVSPATIALQUERY_API_NOT_READY);

    // Capacity outcomes are complete results that preserve the empty baseline,
    // so the same endpoints can be advanced again after the reporter grows.
    const ovspatialquery_pair_reporter_handle_t narrowActive = createReporter(instance, first, second, 2u, 8u);
    AdvanceOutcome                              overActive   = advance(instance, narrowActive, headAdvance());
    REQUIRE(overActive.result.outcome == OVSPATIALQUERY_PAIR_REPORTER_ACTIVE_CAPACITY &&
            overActive.result.accepted_active_pair_count == 0u && overActive.entered.empty() &&
            overActive.exited.empty());
    destroyOperation(instance, overActive.operation);
    const ovspatialquery_pair_reporter_handle_t narrowDelta = createReporter(instance, first, second, 8u, 2u);
    AdvanceOutcome                              overDelta   = advance(instance, narrowDelta, headAdvance());
    REQUIRE(overDelta.result.outcome == OVSPATIALQUERY_PAIR_REPORTER_DELTA_CAPACITY &&
            overDelta.result.accepted_active_pair_count == 0u && overDelta.entered.empty() && overDelta.exited.empty());
    destroyOperation(instance, overDelta.operation);
    REQUIRE(ovspatialquery_grow_pair_reporter_capacity(instance, narrowDelta, 8u, 8u).status ==
            OVSPATIALQUERY_API_SUCCESS);
    AdvanceOutcome grown = advance(instance, narrowDelta, headAdvance());
    REQUIRE(grown.result.outcome == OVSPATIALQUERY_PAIR_REPORTER_ADVANCED &&
            grown.result.accepted_active_pair_count == kPairsAtEight.size() && grown.entered == kPairsAtEight &&
            grown.exited.empty());
    destroyOperation(instance, grown.operation);

    // Remove object 4 so both narrow reporters see (5,4) leave; then, with
    // execution paused, a pending advance holds the instance's single
    // operation slot: the next admission is OPERATION_CAPACITY, fetch is
    // NOT_READY, and neither the pending operation nor its busy reporter can
    // be destroyed, nor the reporter grown.
    publishUpdate(instance, second, 8, 9, {}, {4u});
    // A backward endpoint is rejected even when the other one moved forward.
    requireAdvanceRejected(instance, reporter, backward, OVSPATIALQUERY_API_BACKWARD_ORDINAL);
    REQUIRE(ovspatialquery::detail::setInstanceOperationExecutionsPausedForTest(instance, true) ==
            OVSPATIALQUERY_API_SUCCESS);
    const ovspatialquery_pair_reporter_advance_t toHead = headAdvance();
    ovspatialquery_pair_reporter_reservation_t   pendingReservation{};
    const ovspatialquery_enqueue_result_t        pending =
        ovspatialquery_enqueue_pair_reporter_advance(instance, narrowActive, &toHead, &pendingReservation);
    REQUIRE(pending.status == OVSPATIALQUERY_API_SUCCESS);
    requireAdvanceRejected(instance, reporter, toHead, OVSPATIALQUERY_API_OPERATION_CAPACITY);
    ovspatialquery_pair_reporter_result_t notReady{};
    notReady.accepted_active_pair_count = 55u;
    REQUIRE(ovspatialquery_fetch_pair_reporter(instance, pending.op_id, &notReady).status ==
                OVSPATIALQUERY_API_NOT_READY &&
            notReady.accepted_active_pair_count == 55u);
    ovspatialquery_op_id_t pendingId = pending.op_id;
    REQUIRE(ovspatialquery_destroy_op(instance, &pendingId).status == OVSPATIALQUERY_API_BUSY &&
            pendingId == pending.op_id);
    REQUIRE(ovspatialquery_destroy_pair_reporter(instance, narrowActive).status == OVSPATIALQUERY_API_BUSY);
    REQUIRE(ovspatialquery_grow_pair_reporter_capacity(instance, narrowActive, 8u, 8u).status ==
            OVSPATIALQUERY_API_BUSY);
    REQUIRE(ovspatialquery::detail::setInstanceOperationExecutionsPausedForTest(instance, false) ==
            OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_op_wait_result_t pendingWait{};
    REQUIRE(ovspatialquery_wait_op(instance, pending.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &pendingWait).status ==
                OVSPATIALQUERY_API_SUCCESS &&
            pendingWait.state == OVSPATIALQUERY_OP_COMPLETED);
    ovspatialquery_pair_reporter_result_t settled{};
    REQUIRE(ovspatialquery_fetch_pair_reporter(instance, pending.op_id, &settled).status == OVSPATIALQUERY_API_SUCCESS);
    // At ordinals (8,9) object 4 is gone: two pairs fit the narrow reporter.
    REQUIRE(settled.outcome == OVSPATIALQUERY_PAIR_REPORTER_ADVANCED && settled.first_source_ordinal == 8 &&
            settled.second_source_ordinal == 9 && settled.accepted_active_pair_count == 2u &&
            tensorPairs(settled.entered_pairs) == PairSet{{1u, 1u}, {1u, 2u}} &&
            tensorPairs(settled.exited_pairs).empty());
    destroyOperation(instance, pending.op_id);

    // The grown reporter sees only the departure of (5,4). Its operation
    // reuses the single slot, which no longer answers to the destroyed ID.
    AdvanceOutcome             departed = advance(instance, narrowDelta, toHead);
    ovspatialquery_op_status_t destroyedStatus{};
    REQUIRE(ovspatialquery_query_op_status(instance, pending.op_id, &destroyedStatus).status ==
            OVSPATIALQUERY_API_INVALID_HANDLE);
    REQUIRE(departed.result.outcome == OVSPATIALQUERY_PAIR_REPORTER_ADVANCED &&
            departed.result.accepted_active_pair_count == 2u && departed.entered.empty() &&
            departed.exited == PairSet{{5u, 4u}});
    destroyOperation(instance, departed.operation);

    // Delta capacity after an accepted baseline keeps that baseline, however
    // often it repeats: moving both objects of `second` away and back again
    // then reports no change.
    const ovspatialquery_pair_reporter_handle_t smallDelta = createReporter(instance, first, second, 8u, 2u);
    AdvanceOutcome                              baseline   = advance(instance, smallDelta, toHead);
    REQUIRE(baseline.entered == PairSet{{1u, 1u}, {1u, 2u}});
    destroyOperation(instance, baseline.operation);
    publishUpdate(instance, second, 9, 10, {{1u, 10.0, 12.0}, {2u, 10.0, 12.0}}, {});
    for (int retry = 0; retry < 2; ++retry)
    {
        AdvanceOutcome overflow = advance(instance, smallDelta, toHead);
        REQUIRE(overflow.result.outcome == OVSPATIALQUERY_PAIR_REPORTER_DELTA_CAPACITY &&
                overflow.result.accepted_active_pair_count == 2u && overflow.entered.empty() &&
                overflow.exited.empty());
        destroyOperation(instance, overflow.operation);
    }
    publishUpdate(instance, second, 10, 11, {{1u, 0.5, 1.5}, {2u, 0.0, 2.0}}, {});
    AdvanceOutcome restored = advance(instance, smallDelta, toHead);
    REQUIRE(restored.result.outcome == OVSPATIALQUERY_PAIR_REPORTER_ADVANCED &&
            restored.result.accepted_active_pair_count == 2u && restored.entered.empty() && restored.exited.empty());
    destroyOperation(instance, restored.operation);

    // A scheduling failure is INTERNAL with no operation and rolls back the
    // admission: the retry takes the single slot under a fresh ID and reports
    // the departure of (5,4) against the untouched baseline.
    REQUIRE(ovspatialquery::detail::failNextInstanceOperationSchedulingForTest(instance) == OVSPATIALQUERY_API_SUCCESS);
    requireAdvanceRejected(instance, reporter, toHead, OVSPATIALQUERY_API_INTERNAL);
    AdvanceOutcome retried = advance(instance, reporter, toHead);
    REQUIRE(retried.operation != restored.operation && retried.entered.empty() && retried.exited == PairSet{{5u, 4u}});
    destroyOperation(instance, retried.operation);

    // Teardown follows the BUSY-on-destroy rules in order.
    for (ovspatialquery_pair_reporter_handle_t handle : {reporter, narrowActive, narrowDelta, notPublished, smallDelta})
    {
        REQUIRE(ovspatialquery_destroy_pair_reporter(instance, handle).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(ovspatialquery_destroy_pair_reporter(instance, handle).status == OVSPATIALQUERY_API_INVALID_HANDLE);
    }
    REQUIRE(ovspatialquery_destroy_population(instance, foreign).status == OVSPATIALQUERY_API_INVALID_HANDLE);
    for (ovspatialquery_population_handle_t population : {first, second, unpublished})
    {
        REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(ovspatialquery_destroy_population(instance, population).status == OVSPATIALQUERY_API_INVALID_HANDLE);
    }
    requireSnapshotRejected(instance, snapshotDesc(populationTarget(first), populationTarget(second)),
                            OVSPATIALQUERY_API_INVALID_HANDLE);
    REQUIRE(ovspatialquery_destroy_population(foreignInstance, foreign).status == OVSPATIALQUERY_API_SUCCESS);
    for (ovspatialquery_instance_handle_t handle : {instance, foreignInstance})
    {
        REQUIRE(ovspatialquery_destroy_instance(handle).status == OVSPATIALQUERY_API_BUSY);
        for (uint64_t definition : {kUnitSphere, kHalfSphere})
        {
            REQUIRE(ovspatialquery_destroy_geometry_definition(handle, {OVSPATIALQUERY_GEOMETRY_SPHERE, definition})
                        .status == OVSPATIALQUERY_API_SUCCESS);
        }
        REQUIRE(ovspatialquery_destroy_instance(handle).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(ovspatialquery_destroy_instance(handle).status == OVSPATIALQUERY_API_INVALID_HANDLE);
    }
    std::puts("pair snapshot and reporter API test passed");
    return 0;
}
