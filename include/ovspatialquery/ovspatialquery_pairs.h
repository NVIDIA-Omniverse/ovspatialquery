/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_PAIRS_H
#define OVSPATIALQUERY_PAIRS_H

#include <ovspatialquery/ovspatialquery.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @file
     * @brief AABB pair discovery: stateless snapshots and forward-only reporters.
     *
     * This header is the stateful half of ovspatialquery. Every function in it is
     * experimental and may change in any minor release, because pair discovery
     * is still shaped by its first callers. It reads populations published
     * through the core and shares its instance, targets, ordinals, object IDs,
     * status codes, budgets, and operation lifecycle.
     *
     * A pair snapshot is a synchronous, stateless set of overlapping pairs
     * between two independently selected versions. A pair reporter keeps one
     * accepted baseline of pairs and, on each explicit forward advance, reports
     * which pairs entered and which exited. Both test world AABBs with one
     * predicate on every axis,
     * `first_min[i] <= second_max[i] + tolerance && second_min[i] <= first_max[i] + tolerance`,
     * and neither claims exact contact.
     *
     * Typical reporter flow (status checks omitted):
     *
     *     ovspatialquery_create_pair_reporter(instance, &desc, &reporter);
     *     advance = ovspatialquery_enqueue_pair_reporter_advance(
     *         instance, reporter, &endpoints, &reservation);   // (20,100): every pair entered
     *     ovspatialquery_wait_op(instance, advance.op_id, timeout, &wait_result);
     *     ovspatialquery_fetch_pair_reporter(instance, advance.op_id, &report);
     *     ovspatialquery_destroy_op(instance, &advance.op_id);
     *     // later, after robots publishes 104: advance to (20,104) reports entered and exited
     *     ovspatialquery_destroy_pair_reporter(instance, reporter);
     *
     * Two targets in the same population must select one ordinal
     * (ORDINAL_MISMATCH otherwise). Distinct targets keep their roles, so two
     * overlapping views can produce (A,B) and (B,A) but never (A,A).
     */

    /** Reporter handle; valid only in its creating instance. Zero `raw_id` is invalid. */
    typedef struct ovspatialquery_pair_reporter_handle_t
    {
        uint64_t raw_id;
    } ovspatialquery_pair_reporter_handle_t;
#ifdef __cplusplus
#define OVSPATIALQUERY_INVALID_PAIR_REPORTER (ovspatialquery_pair_reporter_handle_t{0})
#else
#define OVSPATIALQUERY_INVALID_PAIR_REPORTER ((ovspatialquery_pair_reporter_handle_t){0})
#endif

    /** Returns whether `handle` is not OVSPATIALQUERY_INVALID_PAIR_REPORTER. */
    static inline bool ovspatialquery_pair_reporter_handle_is_valid(ovspatialquery_pair_reporter_handle_t handle)
    {
        return handle.raw_id != 0u;
    }

    /** Returns whether two pair-reporter handles have the same identity; both must come from the same instance. */
    static inline bool ovspatialquery_pair_reporter_handle_equal(ovspatialquery_pair_reporter_handle_t a,
                                                                 ovspatialquery_pair_reporter_handle_t b)
    {
        return a.raw_id == b.raw_id;
    }

    /* -------------------------------------------------------------------------- */
    /* Pair snapshots and reporting                                               */
    /* -------------------------------------------------------------------------- */

    typedef struct ovspatialquery_pair_t
    {
        /** Object from the first target. */
        ovspatialquery_object_id_t first;
        /** Object from the second target. */
        ovspatialquery_object_id_t second;
    } ovspatialquery_pair_t;

    /* `ovspatialquery_pair_t` is the CPU row used by synchronous pair snapshots. */

    typedef struct ovspatialquery_pair_buffer_t
    {
        /** Caller-owned rows written by a pair snapshot. */
        ovspatialquery_pair_t* pairs;
        /** Maximum rows that `pairs` can hold. */
        uint64_t capacity;
    } ovspatialquery_pair_buffer_t;

    typedef struct ovspatialquery_pair_snapshot_desc_t
    {
        /** First independently selected immutable version and endpoint role. */
        ovspatialquery_query_target_t first;
        /** Zero accepts all first-target objects; otherwise use bit intersection. */
        ovspatialquery_participation_bits_t first_query_bits;
        /** Second independently selected immutable version and endpoint role. */
        ovspatialquery_query_target_t second;
        /** Zero accepts all second-target objects; otherwise use bit intersection. */
        ovspatialquery_participation_bits_t second_query_bits;
        /** Nonnegative margin used by the defined world-AABB predicate. */
        double contact_tolerance;
    } ovspatialquery_pair_snapshot_desc_t;

    /**
     * A snapshot is one stateless AABB-pair test over selected immutable versions.
     *
     * Multi-rate populations require independent selections: a program can
     * compare static ground at ordinal 20 with moving robots at ordinal 104.
     * Numeric ordinal equality has no cross-population meaning. If both targets
     * have the same parent population, they must select the same ordinal;
     * otherwise the call returns ORDINAL_MISMATCH.
     *
     * An object passes one role's filter when that role's query bits are zero or
     * intersect the object's participation bits. For distinct targets, `first`
     * and `second` preserve their descriptor roles. Distinct overlapping views can
     * therefore produce both (A,B) and (B,A), but never (A,A). When both target,
     * ordinal, and query-bit selections are identical, pairs are unordered:
     * self-pairs are excluded and each {A,B} pair appears once with the smaller
     * object ID in `first`.
     *
     * Two admitted objects form a pair exactly when, for every axis i:
     *
     *     first_min[i] <= second_max[i] + contact_tolerance &&
     *     second_min[i] <= first_max[i] + contact_tolerance
     *
     * These are published world AABBs, so the predicate makes no exact-geometry
     * overlap claim. Acceleration layout and traversal policy are implementation
     * details.
     */

    typedef struct ovspatialquery_pair_snapshot_result_t
    {
        /** Exact ordinal read from the first target. */
        ovspatialquery_ordinal_t first_source_ordinal;
        /** Exact ordinal read from the second target. */
        ovspatialquery_ordinal_t second_source_ordinal;
        /** Valid rows written to the caller's pair buffer. */
        uint64_t stored_pair_count;
        /** Nonzero when additional matching pairs did not fit the buffer. */
        uint32_t truncated;
    } ovspatialquery_pair_snapshot_result_t;

    /** A reporter owns only its private accepted pair baseline and fixed results. */
    typedef struct ovspatialquery_pair_reporter_desc_t
    {
        /** Targets retained for reporter lifetime; ordinals are per advance. */
        ovspatialquery_target_ref_t first;
        /** Zero accepts all first-target objects; otherwise use bit intersection. */
        ovspatialquery_participation_bits_t first_query_bits;
        /** Second target retained for reporter lifetime. */
        ovspatialquery_target_ref_t second;
        /** Zero accepts all second-target objects; otherwise use bit intersection. */
        ovspatialquery_participation_bits_t second_query_bits;
        /** Maximum pairs in the private accepted baseline. */
        uint64_t active_pair_capacity;
        /**
         * Maximum combined entered and exited pairs in one result slot.
         * Each category has its own array, but their stored counts sum to no more
         * than this bound.
         */
        uint64_t delta_pair_capacity;
        /**
         * Nonzero number of immutable delta results that may coexist. A CUDA
         * reporter may also have this many advances pending at once.
         */
        uint32_t result_slot_count;
        /** Nonnegative margin used by the same AABB predicate as snapshots. */
        double contact_tolerance;
        /** kDLCPU or one CUDA device enabled by the instance. */
        DLDevice result_device;
    } ovspatialquery_pair_reporter_desc_t;

    /**
     * One advance of a reporter. A CPU reporter requires zero
     * `cuda_sync.has_stream` and `cuda_sync.wait_event` and endpoints that name
     * completed publications. A CUDA reporter queues the advance on the supplied
     * `cuda_sync.stream`, the legacy default stream 0 included, after a nonzero
     * `wait_event`, so a consumer the caller queues on that stream runs after
     * it; without a stream (`has_stream` 0) the advance runs on the reporter's
     * own stream, and the consumer then waits on `result_ready`. A
     * CUDA reporter may also select an admitted device-prepared publication on
     * its device with ADMITTED_EXACT: the advance is ordered after the
     * publication on the device, and if the publication fails the advance
     * completes with OPERATION_FAILED, writes the FAILED outcome to its device
     * `counts` and preserves the baseline.
     */
    typedef struct ovspatialquery_pair_reporter_advance_t
    {
        /** New first endpoint; it cannot precede the accepted first ordinal. */
        ovspatialquery_ordinal_selection_t first_ordinal;
        /** New second endpoint; it cannot precede the accepted second ordinal. */
        ovspatialquery_ordinal_selection_t second_ordinal;
        /** Zero for a CPU reporter; optional stream and wait event for CUDA. */
        ovspatialquery_cuda_sync_t cuda_sync;
    } ovspatialquery_pair_reporter_advance_t;

    /**
     * Outcome of a completed reporter advance operation. Active capacity is
     * checked first. Delta capacity is evaluated only when the complete active set
     * fits, so one operation has one unambiguous capacity outcome.
     */
    typedef enum ovspatialquery_pair_reporter_outcome_t
    {
        /** The complete new pair set and delta fit; the baseline advanced. */
        OVSPATIALQUERY_PAIR_REPORTER_ADVANCED = 0,
        /** The new pair set exceeded active_pair_capacity; baseline did not advance. */
        OVSPATIALQUERY_PAIR_REPORTER_ACTIVE_CAPACITY = 1,
        /** The complete delta exceeded delta_pair_capacity; baseline did not advance. */
        OVSPATIALQUERY_PAIR_REPORTER_DELTA_CAPACITY = 2,
        /**
         * The advance failed after admission: a selected admitted publication
         * failed, or the device work did. Written with zero counts to a CUDA
         * reporter's device `counts` so a consumer launched from the reservation
         * can tell a failure from an empty advance; typed fetch then returns
         * OPERATION_FAILED instead of a result. The baseline did not advance.
         */
        OVSPATIALQUERY_PAIR_REPORTER_FAILED = 3,
    } ovspatialquery_pair_reporter_outcome_t;

    /**
     * Immutable reservation facts returned by a successful reporter enqueue.
     *
     * The library writes this caller-owned value before return and never retains
     * or modifies it. The selected ordinals, typed outcome, accepted active count,
     * and entered/exited stored counts are completed facts returned on the host
     * only by fetch. A CUDA reporter's reservation also describes the device
     * storage the advance writes, shaped by the capacity like a batch
     * reservation: a consumer queued behind the advance on
     * `cuda_sync.stream`, or made to wait on `result_ready`, reads the same
     * bytes fetch later describes without a host round trip. A CPU reporter's
     * results stay fetch-only and its three descriptors are absent
     * (`shape == NULL`).
     */
    typedef struct ovspatialquery_pair_reporter_reservation_t
    {
        /** Reporter-owned result slot held exclusively by this operation. */
        uint32_t result_slot;
        /** CPU or CUDA result device fixed when the reporter was created. */
        DLDevice result_device;
        /** Maximum combined entered-plus-exited count in the completed result. */
        uint64_t delta_pair_capacity;
        /** Zero for CPU; library-owned completion event for CUDA. */
        ovspatialquery_cuda_event_t result_ready;
        /**
         * CUDA only: uint64[D,2] on the reporter's device, D the delta capacity.
         * Rows 0 to counts[1] hold the entered pairs once `result_ready` fires;
         * the rest are unspecified. Same storage as the fetched `entered_pairs`.
         */
        DLTensor entered_pairs;
        /** CUDA only: uint64[D,2]; rows 0 to counts[2] are the exited pairs. */
        DLTensor exited_pairs;
        /**
         * CUDA only: uint64[4] on the reporter's device, written before
         * `result_ready`: the outcome (`ovspatialquery_pair_reporter_outcome_t`),
         * the entered count, the exited count, and the accepted active pair count
         * after the advance. A capacity outcome or FAILED carries zero counts and
         * the unchanged accepted count. Shape and strides arrays are
         * operation-owned and valid until operation destruction.
         */
        DLTensor counts;
    } ovspatialquery_pair_reporter_reservation_t;

    /**
     * Completed reporter result returned by typed fetch after terminal success.
     * The library copies both tensor descriptor values into the caller-owned
     * result; their data storage and shape/strides arrays are operation-owned,
     * immutable after fetch, and valid until operation destruction.
     */
    typedef struct ovspatialquery_pair_reporter_result_t
    {
        /** Reporter-owned slot held exclusively by this operation. */
        uint32_t result_slot;
        /** Typed result. Capacity outcomes preserve the previous baseline. */
        ovspatialquery_pair_reporter_outcome_t outcome; /* ADVANCED, ACTIVE_CAPACITY,
                                                           or DELTA_CAPACITY */
        /** Evaluated ordinals; accepted only when outcome is ADVANCED. */
        ovspatialquery_ordinal_t first_source_ordinal;
        ovspatialquery_ordinal_t second_source_ordinal;
        /**
         * Pair count in the accepted baseline after this operation. On a capacity
         * outcome this is the unchanged previous-baseline count.
         */
        uint64_t accepted_active_pair_count;
        /** Zero for CPU; library-owned completion event for CUDA. */
        ovspatialquery_cuda_event_t result_ready;
        /**
         * Pairs present at the new endpoints but absent from the old baseline.
         * Device-neutral uint64[E,2]: columns 0 and 1 are first and second object.
         * A valid zero-length tensor is returned for either capacity outcome;
         * partial deltas are never exposed.
         */
        DLTensor entered_pairs;
        /**
         * Pairs present in the old baseline but absent from the new endpoints.
         * Device-neutral uint64[X,2] with the same column order. There is no
         * ordering relationship with `entered_pairs`. E + X never exceeds the
         * reporter's `delta_pair_capacity`. The tensor is operation-owned and uses
         * the reporter device. All DLPack metadata is authoritative.
         */
        DLTensor exited_pairs;
    } ovspatialquery_pair_reporter_result_t;

    /* -------------------------------------------------------------------------- */
    /* Pair snapshot and reporter functions                                       */
    /* -------------------------------------------------------------------------- */

    /**
     * Computes a stateless AABB-pair set between two selected immutable versions.
     * `inout_pairs->capacity` is the only output bound. Additional pairs set
     * `out_result->truncated`; no second descriptor capacity exists. Use this for
     * an arbitrary instantaneous comparison. It creates no accepted baseline.
     */
    OVSPATIALQUERY_EXPERIMENTAL OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_query_pairs_sync(
        ovspatialquery_instance_handle_t instance, const ovspatialquery_pair_snapshot_desc_t* desc,
        ovspatialquery_pair_buffer_t* inout_pairs, ovspatialquery_pair_snapshot_result_t* out_result);

    /**
     * Creates an empty-baseline, forward-only AABB-pair reporter.
     *
     * Typical sequence: create; advance to (100,100), which reports all current
     * pairs in `entered_pairs`; advance to (104,104), which reports only endpoint
     * set differences; fetch and consume; destroy each operation. Advancing to
     * (102,102) after 104 fails with BACKWARD_ORDINAL. Destroy and recreate the
     * reporter to start a new baseline after a discontinuity. The reporter owns
     * only the accepted pair set and selected ordinals. Ordinal selections made by
     * other queries or plans never affect a reporter; its baseline moves only
     * through its own advance call. BVH, SAP, rebuild, incremental-update, and
     * temporal-coherence choices are implementation details.
     *
     * The reporter uses the snapshot predicate, target-role, same-target,
     * self-pair, overlapping-view, and participation-bit rules. If its targets
     * share one parent population, each advance must select one common ordinal.
     */
    OVSPATIALQUERY_EXPERIMENTAL OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_create_pair_reporter(
        ovspatialquery_instance_handle_t instance, const ovspatialquery_pair_reporter_desc_t* desc,
        ovspatialquery_pair_reporter_handle_t* out_reporter);

    /**
     * Admits one forward advance. Both resolved ordinals must be nondecreasing and
     * at least one must increase. A jump compares only the accepted baseline with
     * the new endpoint pair set; it does not replay transient pairs. The first
     * successful advance compares against an empty set and reports every current
     * pair as entered.
     *
     * The first advance has no accepted endpoint ordinals to compare. For every
     * later advance, both selections are resolved atomically before admission. If
     * either resolved ordinal is older than its accepted ordinal, the call returns
     * BACKWARD_ORDINAL. Otherwise, if both equal their accepted ordinals, the call
     * returns STALE_ORDINAL. Otherwise the call can be admitted because at least
     * one endpoint advanced. Both rejections issue no operation ID, reserve no
     * result slot, and preserve the baseline and private accepted-advance count.
     *
     * A sensor loop can poll both endpoints with HEAD. When it polls faster
     * than publication, STALE_ORDINAL means it can retry the same HEAD request
     * after a later publication. An exact equal/equal request remains stale until
     * the caller selects a later ordinal. NOT_READY is reserved for a target that
     * has no selectable version; it does not report lack of progress.
     *
     * A completed capacity outcome is not a partial advance. ACTIVE_CAPACITY and
     * DELTA_CAPACITY return empty delta views and preserve the previous baseline.
     * Other failures also preserve the baseline.
     *
     * Successful admission writes only immutable reservation facts. The library
     * does not retain or modify `out_reservation` after return. Typed fetch returns
     * the selected ordinals, outcome, accepted active count, and exact entered and
     * exited counts. It is repeatable and non-consuming; it returns NOT_READY while
     * pending and OPERATION_FAILED after terminal failure, preserving its output.
     *
     * A CPU reporter has one advance pending at a time (BUSY otherwise). A CUDA
     * reporter's advance is enqueued on the device at admission (`cuda_sync`),
     * may select admitted device-prepared publications on its device, and
     * retires through `ovspatialquery_destroy_op_after_cuda` once its consumers
     * are done; see `ovspatialquery_pair_reporter_advance_t`. Up to
     * `result_slot_count` of its advances may be pending at once
     * (RESULT_CAPACITY beyond): while any is pending, BACKWARD_ORDINAL and
     * STALE_ORDINAL are judged against the newest admitted endpoints, the
     * advances compare on the device in admission order, and one admitted
     * behind a capacity outcome or a failure compares with the preserved
     * baseline, as the jump rule above allows. Outcomes are published in the
     * same order.
     */
    OVSPATIALQUERY_EXPERIMENTAL OVSPATIALQUERY_API ovspatialquery_enqueue_result_t
    ovspatialquery_enqueue_pair_reporter_advance(ovspatialquery_instance_handle_t              instance,
                                                 ovspatialquery_pair_reporter_handle_t         reporter,
                                                 const ovspatialquery_pair_reporter_advance_t* advance,
                                                 ovspatialquery_pair_reporter_reservation_t*   out_reservation);

    /** Fetches repeatable entered/exited views and the typed advance outcome. */
    OVSPATIALQUERY_EXPERIMENTAL OVSPATIALQUERY_API ovspatialquery_result_t
    ovspatialquery_fetch_pair_reporter(ovspatialquery_instance_handle_t instance, ovspatialquery_op_id_t op_id,
                                       ovspatialquery_pair_reporter_result_t* out_result);

    /** Destroys an idle reporter; BUSY means an advance operation is still live. */
    OVSPATIALQUERY_EXPERIMENTAL OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_destroy_pair_reporter(
        ovspatialquery_instance_handle_t instance, ovspatialquery_pair_reporter_handle_t reporter);

    /**
     * Grows one idle reporter's active and delta capacities atomically.
     *
     * Both requested capacities must be at least their current values and at least
     * one must grow. The call returns BUSY while any advance is live. It never
     * changes filters, result-slot count, device, accepted ordinals, or the
     * private baseline. All replacement storage is budgeted and prepared before
     * either capacity changes. HOST_BUDGET, DEVICE_BUDGET, OUT_OF_MEMORY, or any
     * other failure leaves both capacities and the baseline unchanged. Shrinking,
     * reset, rebase, and snapshot seeding are intentionally not operations here.
     */
    OVSPATIALQUERY_EXPERIMENTAL OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_grow_pair_reporter_capacity(
        ovspatialquery_instance_handle_t instance, ovspatialquery_pair_reporter_handle_t reporter,
        uint64_t active_pair_capacity, uint64_t delta_pair_capacity);

#ifdef __cplusplus
}

extern "C++"
{
    OVSPATIALQUERY_DETAIL_HANDLE_OPERATORS(ovspatialquery_pair_reporter_handle_t)
}
#endif

#endif /* OVSPATIALQUERY_PAIRS_H */
