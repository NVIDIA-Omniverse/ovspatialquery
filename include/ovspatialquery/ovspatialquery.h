/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_H
#define OVSPATIALQUERY_H

#include <dlpack/dlpack.h>
#include <ovspatialquery/ovspatialquery_export.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @file
     * @brief Backend-neutral spatial-query C interface.
     *
     * The ovspatialquery library answers raycast, sphere/box-sweep, and overlap
     * questions over geometry the caller publishes. This header is the scene-query
     * core. Sources are companion headers (`ovspatialquery_ovstage.h`, the
     * preferred one, and `ovspatialquery_explicit.h`), as is AABB pair discovery
     * (`ovspatialquery_pairs.h`); none of them is required by this header.
     *
     * Releases follow SemVer 2.0, and `ovspatialquery_get_version` reports the
     * library's version. A patch release changes no declaration. While the major
     * version is 0, a minor release may change any declaration or struct layout,
     * so a caller rebuilds against each minor release. From 1.0, changing the
     * layout of a public struct needs a major release; the interface grows by new
     * entry points instead (a `_v2` suffix when one replaces another), and the
     * replaced declaration carries `OVSPATIALQUERY_DEPRECATED` for at least one
     * minor release and is removed only in a major release. Experimental
     * declarations, marked `OVSPATIALQUERY_EXPERIMENTAL` or with a comment that
     * begins "Experimental:", are outside that rule: they may change in any minor
     * release. Device-prepared publications and everything they alone enable are
     * experimental, and so is all of `ovspatialquery_pairs.h`. A comment paragraph
     * that begins "Implementation note:" describes how this release behaves, to
     * help a caller size and diagnose its use; it is not a promise and may change
     * in any release.
     *
     * An instance owns resource budgets and every object created through it. A
     * population is one independently updated object set; each publication
     * creates an immutable version identified by an ordinal. The newest version
     * is the head, and the configured number of prior heads stay selectable. A
     * view is a stable subset of one population. A query targets a population or
     * view and selects its head or one retained ordinal, without copying the
     * population or its acceleration data.
     *
     * Typical flow (error handling omitted):
     *
     *     ovspatialquery_create_instance(&instance_desc, &instance);
     *     ovspatialquery_create_ovstage_population(instance, &source, &population);
     *     update = ovspatialquery_enqueue_ovstage_population_set(
     *         instance, population, ovstage_ordinal);
     *     ovspatialquery_wait_op(instance, update.op_id, timeout, &wait_result);
     *     ovspatialquery_raycast_closest_sync(instance, &ray, &hit);
     *     ovspatialquery_destroy_op(instance, &update.op_id);
     *
     * Rules that apply to every function unless its own comment says otherwise:
     * functions return status by value; zero handles are invalid; a rejected
     * call preserves its outputs; diagnostics are thread-local; and asynchronous
     * work must reach an observed terminal state (through a wait, a status poll,
     * or a terminal typed fetch) before its operation is destroyed. The caller
     * owns coordinates, units, frames, tolerances, and ordinals, and a query
     * performs no source I/O and no geometry preparation, except the one-time
     * host copy an ON_DEVICE_HOST_READABLE version makes for its first CPU
     * reader. Every resource is bounded by an explicit instance, population, or
     * plan capacity.
     *
     * Synchronous CPU queries return tagged records; CPU and CUDA batches return
     * DLPack result columns. This header embeds `DLTensor` by value, so the
     * DLPack C layout is part of the ovspatialquery ABI.
     */

    /* -------------------------------------------------------------------------- */
    /* Status and diagnostics                                                     */
    /* -------------------------------------------------------------------------- */

    /**
     * Status returned by the public interface.
     *
     * INVALID_ARGUMENT reports malformed values, including values outside the
     * input domain stated before `ovspatialquery_query_precision_t`.
     * INVALID_HANDLE reports a zero, unknown, wrong-kind, or wrong-instance
     * handle of the loaded copy of the library that is called; a handle from
     * another loaded copy is not detected (see the handle types below).
     * BUSY reports a valid object
     * that still has live work or dependents. Budget and capacity statuses report
     * hard caller-configured limits; the implementation does not exceed them or
     * silently fall back. A successful asynchronous admission later completes
     * with SUCCESS or OPERATION_FAILED. Values are never renumbered or reused,
     * so a status keeps its meaning across releases, including the two that
     * only `ovspatialquery_pairs.h` returns.
     */
    typedef enum ovspatialquery_api_status_t
    {
        OVSPATIALQUERY_API_SUCCESS          = 0,
        OVSPATIALQUERY_API_ERROR            = 1,
        OVSPATIALQUERY_API_TIMEOUT          = 2,
        OVSPATIALQUERY_API_INVALID_ARGUMENT = 3,
        OVSPATIALQUERY_API_INVALID_HANDLE   = 4,
        OVSPATIALQUERY_API_INVALID_STATE    = 5,
        OVSPATIALQUERY_API_NOT_SUPPORTED    = 6,
        OVSPATIALQUERY_API_NOT_READY        = 7,
        OVSPATIALQUERY_API_BUSY             = 8,
        /** A pair-reporter advance resolved to no progress (ovspatialquery_pairs.h only). */
        OVSPATIALQUERY_API_STALE_ORDINAL = 9,
        /** Exact-base publication mismatch or same-population pair version mismatch. */
        OVSPATIALQUERY_API_ORDINAL_MISMATCH = 10,
        /** The exact ordinal is neither the head nor an addressable retained version. */
        OVSPATIALQUERY_API_ORDINAL_NOT_RETAINED = 11,
        /** A pair-reporter advance selected an endpoint older than its baseline (pairs header only). */
        OVSPATIALQUERY_API_BACKWARD_ORDINAL = 12,
        /** The instance has no free asynchronous-operation entry. */
        OVSPATIALQUERY_API_OPERATION_CAPACITY = 13,
        /** A reusable plan or reporter has no free immutable result slot. */
        OVSPATIALQUERY_API_RESULT_CAPACITY = 14,
        /** The request would exceed the instance's hard host-byte budget. */
        OVSPATIALQUERY_API_HOST_BUDGET = 15,
        /** The request would exceed the selected device's hard byte budget. */
        OVSPATIALQUERY_API_DEVICE_BUDGET = 16,
        /** Accepted asynchronous work reached terminal failure. */
        OVSPATIALQUERY_API_OPERATION_FAILED = 17,
        /** Allocation failed without a more specific budget result. */
        OVSPATIALQUERY_API_OUT_OF_MEMORY = 18,
        /** An internal invariant failed; no partial public state was published. */
        OVSPATIALQUERY_API_INTERNAL = 19,
        /**
         * Detected CUDA device loss ended the affected admitted work with this
         * terminal failure.
         */
        OVSPATIALQUERY_API_DEVICE_LOST = 20,
    } ovspatialquery_api_status_t;

    /** Status-only result used by synchronous and lifecycle functions. */
    typedef struct ovspatialquery_result_t
    {
        ovspatialquery_api_status_t status;
    } ovspatialquery_result_t;

    /**
     * Common call rules.
     *
     * Unless a declaration says otherwise, failure preserves caller outputs and
     * public state. A create call first writes its invalid-handle value. A rejected
     * enqueue returns INVALID_OP and accepts no work. Bounded synchronous output
     * succeeds with its truncation flag set.
     *
     * Independent handles and immutable queries may be used concurrently. One
     * publication may be pending per population, except that device-prepared
     * publications may be pending up to the population's
     * `pending_publication_capacity`; one advance per CPU reporter and up to
     * `result_slot_count` per CUDA reporter. Plan launches may overlap while
     * result slots remain. Wait, poll, and typed fetch calls are repeatable with
     * distinct writable output structs. The caller serializes destruction with
     * all other use of that same handle or operation.
     */

    /**
     * Library-owned UTF-8 diagnostic. `length` excludes the null terminator.
     * The view is thread-local and ends at the next error-accessor call there.
     */
    typedef struct ovspatialquery_string_t
    {
        const char* ptr;
        size_t      length;
    } ovspatialquery_string_t;

    /* -------------------------------------------------------------------------- */
    /* Handles and common scalar types                                            */
    /* -------------------------------------------------------------------------- */

    /*
     * Handles and operation IDs are distinct structs holding one `uint64_t
     * raw_id`, as in the other Omniverse libraries, so passing one kind where
     * another is expected fails to compile. Use `raw_id` only at numeric
     * boundaries (hashing, logging, foreign bindings) and keep arguments and
     * storage typed: compare with each type's `_equal` and `_is_valid` functions
     * in C, or with `==` and `!=` in C++, where the types also order and hash
     * by `raw_id`. The types distinguish kinds, not instances or lifetimes:
     * a handle from another instance or a destroyed object is still a run-time
     * INVALID_HANDLE. Copying a handle does not retain its object. A zero
     * `raw_id` is invalid in every type. A process may load several copies of
     * the library, each working independently; passing a handle or operation
     * ID to a copy other than the one that returned it is undefined behavior
     * and is not detected.
     */

    /** Instance-scoped asynchronous operation ID. */
    typedef struct ovspatialquery_op_id_t
    {
        uint64_t raw_id;
    } ovspatialquery_op_id_t;
#ifdef __cplusplus
#define OVSPATIALQUERY_INVALID_OP (ovspatialquery_op_id_t{0})
#else
#define OVSPATIALQUERY_INVALID_OP ((ovspatialquery_op_id_t){0})
#endif

    /** Returns whether `handle` is not OVSPATIALQUERY_INVALID_OP. */
    static inline bool ovspatialquery_op_id_is_valid(ovspatialquery_op_id_t handle) { return handle.raw_id != 0u; }

    /** Returns whether two operation IDs have the same identity; both must come from the same instance. */
    static inline bool ovspatialquery_op_id_equal(ovspatialquery_op_id_t a, ovspatialquery_op_id_t b)
    {
        return a.raw_id == b.raw_id;
    }

    /** Result of asynchronous admission. `op_id` is valid only on SUCCESS. */
    typedef struct ovspatialquery_enqueue_result_t
    {
        ovspatialquery_api_status_t status;
        ovspatialquery_op_id_t      op_id;
    } ovspatialquery_enqueue_result_t;

    /** Library instance handle. */
    typedef struct ovspatialquery_instance_handle_t
    {
        uint64_t raw_id;
    } ovspatialquery_instance_handle_t;
#ifdef __cplusplus
#define OVSPATIALQUERY_INVALID_INSTANCE (ovspatialquery_instance_handle_t{0})
#else
#define OVSPATIALQUERY_INVALID_INSTANCE ((ovspatialquery_instance_handle_t){0})
#endif

    /** Returns whether `handle` is not OVSPATIALQUERY_INVALID_INSTANCE. */
    static inline bool ovspatialquery_instance_handle_is_valid(ovspatialquery_instance_handle_t handle)
    {
        return handle.raw_id != 0u;
    }

    /** Returns whether two instance handles have the same identity. */
    static inline bool ovspatialquery_instance_handle_equal(ovspatialquery_instance_handle_t a,
                                                            ovspatialquery_instance_handle_t b)
    {
        return a.raw_id == b.raw_id;
    }
    /** Population handle; valid only in its creating instance. */
    typedef struct ovspatialquery_population_handle_t
    {
        uint64_t raw_id;
    } ovspatialquery_population_handle_t;
#ifdef __cplusplus
#define OVSPATIALQUERY_INVALID_POPULATION (ovspatialquery_population_handle_t{0})
#else
#define OVSPATIALQUERY_INVALID_POPULATION ((ovspatialquery_population_handle_t){0})
#endif

    /** Returns whether `handle` is not OVSPATIALQUERY_INVALID_POPULATION. */
    static inline bool ovspatialquery_population_handle_is_valid(ovspatialquery_population_handle_t handle)
    {
        return handle.raw_id != 0u;
    }

    /** Returns whether two population handles have the same identity; both must come from the same instance. */
    static inline bool ovspatialquery_population_handle_equal(ovspatialquery_population_handle_t a,
                                                              ovspatialquery_population_handle_t b)
    {
        return a.raw_id == b.raw_id;
    }
    /** View handle; valid only in its creating instance. */
    typedef struct ovspatialquery_view_handle_t
    {
        uint64_t raw_id;
    } ovspatialquery_view_handle_t;
#ifdef __cplusplus
#define OVSPATIALQUERY_INVALID_VIEW (ovspatialquery_view_handle_t{0})
#else
#define OVSPATIALQUERY_INVALID_VIEW ((ovspatialquery_view_handle_t){0})
#endif

    /** Returns whether `handle` is not OVSPATIALQUERY_INVALID_VIEW. */
    static inline bool ovspatialquery_view_handle_is_valid(ovspatialquery_view_handle_t handle)
    {
        return handle.raw_id != 0u;
    }

    /** Returns whether two view handles have the same identity; both must come from the same instance. */
    static inline bool ovspatialquery_view_handle_equal(ovspatialquery_view_handle_t a, ovspatialquery_view_handle_t b)
    {
        return a.raw_id == b.raw_id;
    }
    /** Batch-plan handle; valid only in its creating instance. */
    typedef struct ovspatialquery_batch_plan_handle_t
    {
        uint64_t raw_id;
    } ovspatialquery_batch_plan_handle_t;
#ifdef __cplusplus
#define OVSPATIALQUERY_INVALID_BATCH_PLAN (ovspatialquery_batch_plan_handle_t{0})
#else
#define OVSPATIALQUERY_INVALID_BATCH_PLAN ((ovspatialquery_batch_plan_handle_t){0})
#endif

    /** Returns whether `handle` is not OVSPATIALQUERY_INVALID_BATCH_PLAN. */
    static inline bool ovspatialquery_batch_plan_handle_is_valid(ovspatialquery_batch_plan_handle_t handle)
    {
        return handle.raw_id != 0u;
    }

    /** Returns whether two batch-plan handles have the same identity; both must come from the same instance. */
    static inline bool ovspatialquery_batch_plan_handle_equal(ovspatialquery_batch_plan_handle_t a,
                                                              ovspatialquery_batch_plan_handle_t b)
    {
        return a.raw_id == b.raw_id;
    }

    /** Wait duration in nanoseconds. Zero polls; the sentinel waits indefinitely. */
    typedef uint64_t ovspatialquery_timeout_t;
    /**
     * Nonnegative population-scoped publication ordinal.
     *
     * An ovstage-backed publication uses the exact ovstage ordinal that it reads:
     * publishing ovstage snapshot 104 produces population ordinal 104. There is
     * no remapping or second counter. A population may skip source ordinals, and
     * several populations may publish the same numeric ordinal independently.
     * Other source adapters define their own population-scoped ordinal schedule.
     * A new population has no head. Its first complete set can publish any
     * nonnegative ordinal, including zero.
     */
    typedef int64_t ovspatialquery_ordinal_t;
#define OVSPATIALQUERY_TIMEOUT_INFINITE ((ovspatialquery_timeout_t)~UINT64_C(0))

    /**
     * Population terms used throughout this header.
     *
     * A population is one independently updated set of spatial objects. A
     * publication creates an immutable state identified by a population-scoped
     * ordinal. The head is the newest successfully published state; work still in
     * progress is not the head. A retained ordinal is an older state that remains
     * selectable by new queries. An admitted ordinal is a publication that has
     * been accepted and is still being prepared; CUDA plan launches may select it
     * when it is prepared on their device, and their work waits for it there.
     * Admission selects and keeps one immutable state alive without copying the
     * population or its acceleration data.
     *
     * For example, one query can select retained ordinal 102 while another
     * selects head ordinal 104 from the same population, and a CUDA overlap
     * batch can be enqueued against admitted ordinal 105 before the publication
     * of 105 has finished.
     */

    /* -------------------------------------------------------------------------- */
    /* CUDA synchronization                                                       */
    /* -------------------------------------------------------------------------- */

    /** Opaque CUDA event value. Zero means that no event is supplied or returned. */
    typedef struct ovspatialquery_cuda_event_t
    {
        uintptr_t value;
    } ovspatialquery_cuda_event_t;

    /**
     * CUDA producer ordering for one submission, laid out like the shared
     * `ovx_cuda_sync_t` so that adopting it later is a rename.
     *
     * `has_stream` separates "no stream" from stream 0, the legacy default
     * stream: 1 supplies `stream`, 0 supplies none and the library ignores
     * `stream`, and any other value returns INVALID_ARGUMENT. A supplied
     * `stream` is the unmodified native value on the selected CUDA device
     * (0 or CU_STREAM_LEGACY, CU_STREAM_PER_THREAD, or a stream handle); the
     * library issues its work on it inside the enqueue call, on the calling
     * thread. A supplied stream belongs to the selected device's primary
     * context, where the library runs its own work, so 0, CU_STREAM_LEGACY, and
     * CU_STREAM_PER_THREAD name that context's default streams. A stream that is
     * capturing into a CUDA graph returns NOT_SUPPORTED before anything is
     * queued, because a replayed graph would skip admission, ordinal selection,
     * and result slots. Each submission states whether it needs a stream and
     * what it queues there. The library waits on a nonzero `wait_event` before it reads
     * submitted data. The caller keeps both handles valid until the admitted
     * operation becomes terminal. CPU submissions supply neither.
     */
    typedef struct ovspatialquery_cuda_sync_t
    {
        uint32_t  has_stream; /* 0 = absent, 1 = supplied (including stream 0) */
        uintptr_t stream;     /* unmodified native CUDA stream value */
        uintptr_t wait_event; /* 0 = none; otherwise a recorded producer event */
    } ovspatialquery_cuda_sync_t;

    /**
     * Where the library validates and prepares a publication's version. The
     * explicit and ovstage adapters both take it, and each states what its
     * publications read on either side.
     *
     * ON_HOST: the library prepares objects, bounds, and the tree on the host,
     * so every reader works. ON_DEVICE: device kernels validate the
     * publication's poses and prepare the version on `dynamic_execution.device`,
     * and its tables stay there; a CUDA plan reads them directly and nothing
     * but a fixed-size status crosses to the host.
     *
     * A version does not move between the two: an ON_DEVICE publication on a
     * population whose head was prepared on the host returns NOT_SUPPORTED at
     * admission, and so does an ON_HOST publication on a device-prepared head.
     * Readers of a device-prepared version are CUDA overlap, raycast, and
     * sweep plans with BOUNDS precision in any hit mode over the whole
     * population, and CUDA pair reporters on that device over whole
     * populations; an ANY hit is the hit CLOSEST would report. Exact CUDA
     * plans and CPU readers need the version's objects, which only an
     * ON_DEVICE_HOST_READABLE publication supplies (NOT_SUPPORTED otherwise);
     * views and CPU reporters return NOT_SUPPORTED. Those plans and reporters
     * may select the version with ORDINAL_ADMITTED_EXACT before its publication
     * completes, and up to the population's `pending_publication_capacity`
     * device-prepared updates may be pending at once, each based on the one
     * admitted before it, so a producer publishes and queries every step
     * without a host wait. The first complete set must complete before an
     * update builds on it.
     *
     * ON_DEVICE_HOST_READABLE prepares the version on the device exactly as
     * ON_DEVICE does, and additionally lets CPU readers select it: the first
     * synchronous query, CPU plan launch, or pair snapshot that selects the
     * version downloads its poses and builds the host objects and tree, once,
     * charged to the host budget at that reader's admission (HOST_BUDGET when
     * it does not fit, and the reader is refused) and released with the
     * version. A pure GPU pipeline that never selects the version from the CPU
     * pays nothing. Plain ON_DEVICE keeps returning NOT_SUPPORTED to CPU
     * readers. Views and CPU pair reporters stay NOT_SUPPORTED for either.
     */
    typedef enum ovspatialquery_preparation_t
    {
        OVSPATIALQUERY_PREPARE_ON_HOST = 0,
        /** Experimental: device preparation may change in any minor release. */
        OVSPATIALQUERY_PREPARE_ON_DEVICE = 1,
        /** Experimental: as ON_DEVICE. */
        OVSPATIALQUERY_PREPARE_ON_DEVICE_HOST_READABLE = 2,
    } ovspatialquery_preparation_t;

    /**
     * Selects CPU or CUDA execution of a publication and where its version is
     * prepared. `device.device_type == kDLCPU` requires zero
     * `cuda_sync.has_stream` and `cuda_sync.wait_event` because no stream is
     * involved. For a CUDA device each adapter states what the library reads on
     * `cuda_sync.stream` and when it waits on `cuda_sync.wait_event`.
     */
    typedef struct ovspatialquery_dynamic_execution_t
    {
        DLDevice                     device;
        ovspatialquery_cuda_sync_t   cuda_sync;
        ovspatialquery_preparation_t preparation; /* ON_HOST, ON_DEVICE, or ON_DEVICE_HOST_READABLE */
    } ovspatialquery_dynamic_execution_t;

    /* -------------------------------------------------------------------------- */
    /* Stored object and geometry identity                                        */
    /* -------------------------------------------------------------------------- */

    /**
     * Identifies one source object within its population.
     *
     * The source adapter assigns `value` and defines what one object is: an
     * explicit object is a caller value, never reused during its population's
     * lifetime; an ovstage object is a prim path, and `value` is its
     * `ovx_primpath_t`. Query hits use it to answer "which object was hit?" An
     * ID names one object for its population's lifetime, including in results
     * from retained versions. Zero is invalid. The adapter also says when IDs
     * from two populations name the same object: ovstage populations whose
     * stages share a path dictionary report one ID for one prim path, and the
     * library never relates explicit IDs across populations.
     */
    typedef struct ovspatialquery_object_id_t
    {
        uint64_t value;
    } ovspatialquery_object_id_t;

    /**
     * 128 participation bits: an object's categories, or the categories a query
     * accepts. An object passes a query when the query's bits are all zero or
     * share at least one set bit with the object's, which is the test PhysX's
     * built-in scene-query filter applies to the 128 bits of `PxFilterData`.
     * `words[0]` holds bits 0-63 and `words[1]` bits 64-127, so on a
     * little-endian host the four 32-bit words of a `PxFilterData` copy over it
     * unchanged. A column of these is a uint64[N,2] DLPack array in the same
     * word order.
     */
    typedef struct ovspatialquery_participation_bits_t
    {
        uint64_t words[2];
    } ovspatialquery_participation_bits_t;

    /** Stored geometry family. Every implementation supports all five built-ins. */
    typedef enum ovspatialquery_geometry_type_t
    {
        /** Invalid geometry reference; never names a definition or hit. */
        OVSPATIALQUERY_GEOMETRY_INVALID = 0,
        /**
         * Reserved for a future user-geometry family; never reused. Every row,
         * reference, or definition call naming it returns NOT_SUPPORTED, and no
         * result carries it.
         */
        OVSPATIALQUERY_GEOMETRY_USER          = 1,
        OVSPATIALQUERY_GEOMETRY_SPHERE        = 2,
        OVSPATIALQUERY_GEOMETRY_BOX           = 3,
        OVSPATIALQUERY_GEOMETRY_CAPSULE       = 4,
        OVSPATIALQUERY_GEOMETRY_CONVEX        = 5,
        OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH = 6,
    } ovspatialquery_geometry_type_t;

    /**
     * Identifies one exact immutable geometry definition within an instance.
     *
     * The source adapter assigns `value`; zero is invalid. Callers compare the
     * complete structure for equality but do not interpret `value`. References
     * are instance-scoped and comparable across populations in that instance. Equal
     * references name the same adapter-recognized immutable definition. Unequal
     * references can still contain equal geometry bytes. References from separate
     * instances are not comparable. Values are not reused during instance lifetime.
     * The adapters split the value space so that they never contend for a value:
     * explicit registration takes caller values from 1 to 2^63 - 1, and the
     * ovstage adapter assigns values from 2^63 up. Result references are
     * therefore unambiguous across every population in the instance, and neither
     * adapter fails because of the order in which the two registered.
     *
     * Prepared/cooked reuse has no public hashes, cooking controls, or cache
     * interface, and sharing prepared artifacts across instances is not
     * guaranteed.
     *
     * Implementation note: the reuse key contains geometry family, stable source
     * content identity or digest, preparation format/options version, and
     * backend/device. An ovstage adapter uses source-provided
     * large-array/content hashes when available and may otherwise
     * hash acquired arrays privately. Avoiding that fallback re-hash is not
     * guaranteed until ovstage supplies stable content identity.
     *
     * Example: ovstage prims `/World/RobotA` and `/World/RobotB` can have distinct
     * object IDs but the same CONVEX geometry reference because both instance the
     * same collision prototype. Moving RobotB preserves the reference. Editing
     * the prototype makes the adapter return a different reference.
     *
     *     const ovspatialquery_geometry_ref_t* a =
     *         &robot_a_hit.data.exact.identity.geometry;
     *     const ovspatialquery_geometry_ref_t* b =
     *         &robot_b_hit.data.exact.identity.geometry;
     *     int same_geometry = a->type == b->type && a->value == b->value;
     */
    typedef struct ovspatialquery_geometry_ref_t
    {
        ovspatialquery_geometry_type_t type; /* INVALID, SPHERE, BOX, CAPSULE, CONVEX,
                                                TRIANGLE_MESH, or USER */
        uint64_t value;
    } ovspatialquery_geometry_ref_t;

    /* -------------------------------------------------------------------------- */
    /* Instance and resource budgets                                               */
    /* -------------------------------------------------------------------------- */

    /** Hard byte ceiling for one enabled CUDA device. Devices must be unique. */
    typedef struct ovspatialquery_device_budget_t
    {
        /** CUDA device charged by this entry. CPU memory uses host_byte_budget. */
        DLDevice device;
        /** Maximum instance-controlled resident bytes on `device`; must be nonzero. */
        uint64_t byte_budget;
    } ovspatialquery_device_budget_t;

    /**
     * Hard instance capacities and execution budgets.
     *
     * For example, a program can permit 32 live asynchronous operations, run at
     * most four CPU jobs at once, and cap instance-controlled host memory at
     * 1 GiB. There is no hidden global worker pool and no silent budget overflow.
     */
    typedef struct ovspatialquery_instance_desc_t
    {
        /** Nonzero maximum live operation IDs; a slot returns at op destruction. */
        uint32_t operation_capacity;
        /** Nonzero maximum asynchronous CPU jobs running at once; excess queues. */
        uint32_t max_cpu_concurrency;
        /** Nonzero hard ceiling for instance-controlled host-resident bytes. */
        uint64_t host_byte_budget;
        /** Optional array copied by create_instance; null exactly when count is zero. */
        const ovspatialquery_device_budget_t* device_budgets;
        /** Number of unique CUDA-device entries; zero disables CUDA execution. */
        uint32_t device_budget_count;
    } ovspatialquery_instance_desc_t;

    /* -------------------------------------------------------------------------- */
    /* Populations and immutable publication                                      */
    /* -------------------------------------------------------------------------- */

    /**
     * `retained_ordinal_capacity` is the maximum number of prior heads selectable
     * by new exact-ordinal work. The head is additional. Zero keeps no prior head
     * addressable. For example, capacity two after successful publications 100,
     * 104, and 110 guarantees that heads 104 and 100 remain selectable while 110
     * is the head. Ordinals do not need to be consecutive. Publication fails
     * atomically if the configured host/device budgets cannot preserve the
     * complete retention window. Older states already used by accepted work stay
     * alive until that work ends.
     *
     * Experimental: `pending_publication_capacity` is how many device-prepared publications
     * may be admitted and not yet terminal at once, so a GPU
     * producer can publish step N+1 while step N is still being prepared and
     * queried. Zero and one both keep the common rule of one pending publication.
     * Each pending publication is charged to the device budget at admission. A
     * host-prepared publication is admitted only while none is pending.
     */
    typedef struct ovspatialquery_population_desc_t
    {
        uint32_t retained_ordinal_capacity;
        uint32_t pending_publication_capacity;
    } ovspatialquery_population_desc_t;

    /** Current observable state of one population. */
    typedef struct ovspatialquery_population_info_t
    {
        /** Nonzero after the first complete set publishes successfully. */
        uint32_t ready;
        /** Newest published ordinal; valid only when `ready` is nonzero. */
        ovspatialquery_ordinal_t head_ordinal;
        /** Queryable spatial objects in the head; zero while not ready. */
        uint64_t object_count;
        /** Prior heads currently selectable by new exact-ordinal work. */
        uint32_t retained_ordinal_count;
        /** One if a set/update has been admitted and is not yet terminal. */
        uint32_t update_pending;
    } ovspatialquery_population_info_t;

    /** Repeatable result of one successful population set or update operation. */
    typedef struct ovspatialquery_population_update_result_t
    {
        /** Ordinal published as the new head by this operation. */
        ovspatialquery_ordinal_t published_ordinal;
        /** Queryable spatial objects in the new head. */
        uint64_t object_count;
        /**
         * Source objects omitted by the source adapter. Explicit populations
         * report zero and return two empty tensors below.
         */
        uint64_t excluded_object_count;
        /**
         * Complete exclusion list, sorted by object ID value with one row per
         * object: host uint64[E], E == excluded_object_count. The library owns
         * this tensor and its storage until the operation is destroyed.
         */
        DLTensor excluded_objects;
        /**
         * Host int32[E], parallel to excluded_objects; reason codes belong to
         * the source adapter. The library owns this tensor and its storage until
         * the operation is destroyed. Both tensors have shape [0] when E is zero.
         */
        DLTensor exclusion_reasons;
    } ovspatialquery_population_update_result_t;

    /* -------------------------------------------------------------------------- */
    /* Input arrays                                                               */
    /* -------------------------------------------------------------------------- */

    /** Per-column lifetime rule for the data referenced by an input DLTensor. */
    typedef enum ovspatialquery_input_ownership_t
    {
        /**
         * Caller keeps data alive and unchanged until the operation is terminal.
         * For example, a sensor keeps its ray arrays unchanged while one batch runs.
         */
        OVSPATIALQUERY_INPUT_BORROWED_UNTIL_TERMINAL = 0,
        /**
         * The library copies the logical contents before enqueue returns
         * successfully. On the CPU, the caller can reuse a temporary input array
         * immediately. CUDA reuse also follows the stream-ordering rule on
         * input_array_t.
         */
        OVSPATIALQUERY_INPUT_COPIED_ON_SUBMISSION = 1,
    } ovspatialquery_input_ownership_t;

    /**
     * One DLPack input column with an independent data-lifetime choice.
     *
     * The library copies tensor metadata, shape, and strides during admission;
     * `ownership` governs only `tensor.data`. The caller keeps a borrowed array
     * unchanged until terminal completion and may change a copied CPU array as
     * soon as admission succeeds. For copied CUDA data, the library queues its
     * copy on `cuda_sync.stream` before enqueue returns, so later writes that the
     * caller queues on that same stream are ordered after the copy. To overwrite
     * the source buffer from any other stream, the caller records its own event on
     * `cuda_sync.stream` after enqueue returns and makes that other stream wait on
     * the event first; successful admission alone does not make unordered
     * cross-stream reuse safe.
     *
     * A column with no rows has null `data` and zero `byte_offset`; the library
     * ignores its stride values, so a producer may report any (NumPy reports 0).
     *
     * Every input column, CPU or CUDA, follows one layout rule, so that one array
     * helper serves batches and publications. Strides count elements. A null
     * `strides` pointer means compact row-major, as DLPack before 1.2 allowed.
     * Any nonnegative stride is accepted: a slice such as a pose array's first
     * three columns, every k-th row, padded rows, column-major order, or a zero
     * stride that repeats one value; a dimension of extent one ignores its
     * stride. A negative stride, or a column whose last element's address
     * overflows, returns INVALID_ARGUMENT, and `ovspatialquery_get_last_error`
     * names the column and the rule. A CPU batch plan reads strided columns in
     * place. Elsewhere the library reads a compact column in place and copies any
     * other into compact storage of its own, whatever its ownership; the copy is
     * charged like a COPIED_ON_SUBMISSION copy and, on CUDA, runs on
     * `cuda_sync.stream` after its wait event. The caller still keeps a borrowed
     * column unchanged until the operation is terminal.
     */
    typedef struct ovspatialquery_input_array_t
    {
        DLTensor                         tensor;
        ovspatialquery_input_ownership_t ownership; /* BORROWED_UNTIL_TERMINAL or
                                                       COPIED_ON_SUBMISSION */
    } ovspatialquery_input_array_t;

    /* -------------------------------------------------------------------------- */
    /* Query targets and common filtering                                         */
    /* -------------------------------------------------------------------------- */

    typedef enum ovspatialquery_target_kind_t
    {
        /** Query all objects in one population. */
        OVSPATIALQUERY_TARGET_POPULATION = 0,
        /** Query one stable filter subset that shares its parent population data. */
        OVSPATIALQUERY_TARGET_VIEW = 1,
    } ovspatialquery_target_kind_t;

    /** Source-erased population or stable view. A view shares population data. */
    typedef struct ovspatialquery_target_ref_t
    {
        ovspatialquery_target_kind_t kind; /* POPULATION or VIEW */
        union
        {
            ovspatialquery_population_handle_t population;
            ovspatialquery_view_handle_t       view;
        } handle;
    } ovspatialquery_target_ref_t;

    /**
     * HEAD and RETAINED_EXACT name completed publications, so a reader never sees
     * work that may still fail. A GPU producer that publishes every step would
     * otherwise have to wait on the host for each publication before it can
     * enqueue the queries that read it. ADMITTED_EXACT lets a CUDA plan launch
     * or a CUDA pair reporter advance name a publication that has been admitted
     * and is still being prepared on its device: the work is ordered after the
     * publication on the device and the host waits for neither. If that
     * publication fails, the launch completes with OPERATION_FAILED and its rows
     * carry the FAILED status with zero hits; the advance completes with
     * OPERATION_FAILED and writes the FAILED outcome to its device counts. The
     * head and the retained window are unchanged by either outcome. Once the
     * library has observed the failure, the ordinal is neither retained nor
     * pending: a later ADMITTED_EXACT selection of it returns
     * ORDINAL_NOT_RETAINED at admission and an update based on it returns
     * ORDINAL_MISMATCH (NOT_READY when it was the first set), so the producer
     * publishes again on the head, or sets the population again when the
     * failed publication was its first set. Work enqueued while the failure is
     * in flight gets either outcome.
     */
    typedef enum ovspatialquery_ordinal_selection_kind_t
    {
        /** Select the head atomically at admission. */
        OVSPATIALQUERY_ORDINAL_HEAD = 0,
        /** Select exactly `source_ordinal`; never search latest-at-or-before. */
        OVSPATIALQUERY_ORDINAL_RETAINED_EXACT = 1,
        /**
         * Experimental: select exactly `source_ordinal`, which is a completed retained ordinal
         * or an admitted publication being prepared on the selecting work's CUDA
         * device. CUDA plan launches and CUDA pair reporter advances only;
         * synchronous queries, CPU plans, pair snapshots, and CPU reporters return
         * INVALID_ARGUMENT. A pending publication prepared on the host or on
         * another device, or one that has failed, is ORDINAL_NOT_RETAINED.
         */
        OVSPATIALQUERY_ORDINAL_ADMITTED_EXACT = 2,
    } ovspatialquery_ordinal_selection_kind_t;

    /** Chooses the head, one exact retained ordinal, or one admitted ordinal. */
    typedef struct ovspatialquery_ordinal_selection_t
    {
        ovspatialquery_ordinal_selection_kind_t kind; /* HEAD, RETAINED_EXACT, or ADMITTED_EXACT */
        /** Used by ORDINAL_RETAINED_EXACT and ORDINAL_ADMITTED_EXACT; ignored for ORDINAL_HEAD. */
        ovspatialquery_ordinal_t source_ordinal;
    } ovspatialquery_ordinal_selection_t;

    /**
     * Admission resolves this target once and pins that immutable version. Exact
     * selection returns API_ORDINAL_NOT_RETAINED when the ordinal is missing or
     * has been evicted, and also when a view has not yet become ready at the
     * requested ordinal. Admission never changes the population head or copies an
     * instance.
     */
    typedef struct ovspatialquery_query_target_t
    {
        ovspatialquery_target_ref_t        target;
        ovspatialquery_ordinal_selection_t ordinal;
    } ovspatialquery_query_target_t;

    /**
     * Exact queries evaluate stored surfaces rather than only their bounds.
     * The implementation chooses arithmetic precision and acceleration; input or
     * output dtypes do not prescribe intermediate precision. Local mesh vertices
     * use float32 while world placement retains double precision. Results near
     * edges, touching surfaces, or ill-conditioned geometry can differ between
     * implementations. No universal numerical error bound is promised.
     * Identity, filtering, clipping, and each query family's result semantics still
     * apply.
     *
     * Input domain. The CPU evaluators run in float inside each object's local
     * frame, so the library bounds what it accepts to keep every value it hands
     * them finite. Every coordinate, half extent, radius, distance, and
     * contact_tolerance a caller passes, every local vertex coordinate, half
     * extent, and radius of a stored definition, and every published object's
     * world AABB lies within +/-2^60 caller units; every uniform placement scale
     * lies within [2^-60, 2^60]. Outside it the input is malformed: a
     * synchronous call or plan creation returns INVALID_ARGUMENT, a batch row
     * reports BATCH_ROW_INVALID_ARGUMENT, and each source adapter states where
     * it refuses a definition or an object. The bound is in caller units, so a
     * stage in centimeters reaches it at a smaller physical size.
     */

    typedef enum ovspatialquery_query_precision_t
    {
        /** Return AABB candidates; candidate contact data is not object-surface data. */
        OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE = 0,
        /** Return exact results against stored geometry. */
        OVSPATIALQUERY_QUERY_EXACT = 1,
        /**
         * Permit binary32 or wider decisive surface evaluation for synchronous
         * CPU raycasts and CPU/CUDA raycast plans, in every hit mode. This can
         * change hit presence, distance, or the selected feature near numerical
         * boundaries. Results still carry HIT_EXACT_SURFACE, and source features
         * remain eligible; bounds-only evaluation is not permitted. Stronger
         * evaluation may be used, including for nonzero contact_tolerance and
         * geometry without a specialized evaluator. Sphere/box sweeps and overlaps return
         * API_NOT_SUPPORTED at admission; unknown precision values return
         * API_INVALID_ARGUMENT.
         */
        OVSPATIALQUERY_QUERY_EXACT_SINGLE = 2,
    } ovspatialquery_query_precision_t;

    /**
     * Options of one raycast or sweep, ORed into `query_flags` of a synchronous
     * query or of a batch launch. A bit this library does not know returns
     * NOT_SUPPORTED with no effect, so a caller that sets an option from a newer
     * header fails cleanly instead of being silently ignored.
     */
    typedef enum ovspatialquery_query_flag_t
    {
        /**
         * Every triangle of every mesh the query reaches counts from both sides,
         * as PhysX's eMESH_BOTH_SIDES does, so a caller can see a single-sided
         * mesh's back faces without republishing it (see the distance intervals
         * below). Other geometry, BOUNDS_CANDIDATE precision, and overlaps ignore it.
         */
        OVSPATIALQUERY_QUERY_MESH_BOTH_SIDES = 1,
    } ovspatialquery_query_flag_t;

    /** Fields shared by one instantaneous scene query. */
    typedef struct ovspatialquery_query_common_t
    {
        /** Population/view and immutable ordinal read by this query. */
        ovspatialquery_query_target_t target;
        /** Bounds candidates or exact geometry results. */
        ovspatialquery_query_precision_t precision; /* BOUNDS_CANDIDATE, EXACT, or EXACT_SINGLE */
        /** Zero accepts all; otherwise the object's bits must share a set bit with these. */
        ovspatialquery_participation_bits_t query_bits;
        /** Nonnegative caller-unit tolerance used by the family contact rule. */
        double contact_tolerance;
        /** Zero for none; unknown bits return NOT_SUPPORTED. */
        uint32_t query_flags; /* bits: MESH_BOTH_SIDES */
        /**
         * Room for two optional per-query inputs without a layout change. Null
         * means absent. A non-null entry returns NOT_SUPPORTED and the query has
         * no effect, so a caller that sets an input this library predates fails
         * cleanly.
         */
        const void* reserved[2];
    } ovspatialquery_query_common_t;

    /*
     * `contact_tolerance` is a geometric expansion in caller units: a ray tests a
     * tube of this radius; a sphere sweep and sphere overlap add it to the sphere
     * radius; an AABB expands each face by it; and an oriented box expands its
     * local half extents by it before its pose is applied. Pair operations in
     * `ovspatialquery_pairs.h` use their stated expanded-AABB predicate. Tolerance
     * never changes stored geometry.
     */

    /*
     * Ray and sweep distance intervals are closed: the first contact is the least
     * distance in [`min_distance`, `max_distance`] at which the ray point or moving
     * shape touches stored geometry. Stored spheres, boxes, capsules, and convex hulls
     * are solid, so clipping the start of a path behaves like moving its origin there:
     * a point or shape already touching or overlapping one at `min_distance` reports
     * that object at `min_distance`. The exact contact of such a ray or sphere is the
     * stored surface point nearest its center, with the outward normal there.
     * Triangle meshes are surfaces: a ray inside a closed mesh reports its next
     * crossing, and a sweep already touching a triangle it does not ignore reports
     * it at `min_distance`.
     *
     * A triangle's front normal n is (v1 - v0) x (v2 - v0). A single-sided triangle
     * stops a ray only through its front face. A sweep ignores a single-sided
     * triangle whose front normal points along the motion, so a shape resting on a
     * floor can leave it: a sphere sweep one with n . d > 0, a box sweep one with
     * n . d >= 0, where d is `unit_direction`, as PhysX's own sweeps decide. A box
     * moving parallel to a single-sided triangle therefore ignores it, and a sphere
     * does not. A sweep meets every other triangle as a two-sided surface: its first
     * contact is where the shape first touches the triangle's face, an edge, or a
     * vertex, from either side, including a contact at `min_distance` with the
     * shape's center behind the triangle. Double-sided triangles are never ignored.
     *
     * OVSPATIALQUERY_QUERY_MESH_BOTH_SIDES in a query's or a batch launch's
     * `query_flags` lets every triangle count from both sides. A ray also crosses
     * back faces. Its normal is the triangle's front normal on a single-sided mesh,
     * as PhysX reports it, so n . d > 0 marks a back-face hit; a double-sided mesh
     * still reports the normal facing the ray. A ray with a nonzero
     * contact_tolerance reports its contact normal there mirrored across the
     * triangle's plane to the front side, which on the face is the front normal.
     * A sweep ignores no triangle, so a shape resting on a floor and moving up
     * reports it at `min_distance`; its normal is unchanged.
     */

    /**
     * Normative minimum CPU support for mandatory stored geometry.
     *
     * Every cell below marked EXACT is required and cannot return NOT_SUPPORTED
     * because of the stored built-in geometry family. EXACT_SINGLE raycasts have
     * the same mandatory geometry coverage. The supported overlap query
     * shapes are sphere, oriented box, and AABB.
     *
     * Stored geometry   Raycast   Sphere/box sweep   Sphere/box/AABB overlap
     * Sphere            EXACT     EXACT          EXACT
     * Box               EXACT     EXACT          EXACT
     * Capsule           EXACT     EXACT          EXACT
     * Convex            EXACT     EXACT          EXACT
     * Triangle mesh     EXACT     EXACT          EXACT
     *
     * Pair snapshots and reporters (`ovspatialquery_pairs.h`) are bounds-only
     * operations that test published world AABBs and never claim exact contact.
     *
     * CUDA execution is optional at the implementation/device level. A CUDA plan
     * request can return NOT_SUPPORTED when CUDA or that query family is absent.
     * Once an exact CUDA plan is created, mandatory built-in geometry encountered
     * by that plan cannot cause NOT_SUPPORTED, and execution never falls back to
     * CPU. There are no per-built-in geometry capability bits.
     */

    /**
     * One question `ovspatialquery_query_capability` answers, so a caller picks
     * its CPU or CUDA path before creating anything instead of trying requests.
     * A key exists only for what this header lets an implementation, device, or
     * run time lack. A fixed rule a declaration states has no key (EXACT_SINGLE
     * outside raycasts, MESH_HITS_ALL with a nonzero contact_tolerance, views of
     * a device-prepared version, a CUDA stream that is capturing a graph, a
     * non-null reserved input); a release
     * that lifts one adds a key for it.
     * Built-in geometry is mandatory and has none.
     *
     * A `*_PLAN` key is that family's plans at BOUNDS_CANDIDATE precision in
     * every hit mode its plan descriptor accepts; its `*_PLAN_EXACT` key is the same family at EXACT
     * precision, and for raycasts EXACT_SINGLE too, which any exact evaluator
     * satisfies.
     */
    typedef enum ovspatialquery_capability_t
    {
        OVSPATIALQUERY_CAPABILITY_RAYCAST_PLAN            = 0,
        OVSPATIALQUERY_CAPABILITY_RAYCAST_PLAN_EXACT      = 1,
        OVSPATIALQUERY_CAPABILITY_SPHERE_SWEEP_PLAN       = 2,
        OVSPATIALQUERY_CAPABILITY_SPHERE_SWEEP_PLAN_EXACT = 3,
        OVSPATIALQUERY_CAPABILITY_BOX_SWEEP_PLAN          = 4,
        OVSPATIALQUERY_CAPABILITY_BOX_SWEEP_PLAN_EXACT    = 5,
        OVSPATIALQUERY_CAPABILITY_OVERLAP_PLAN            = 6,
        OVSPATIALQUERY_CAPABILITY_OVERLAP_PLAN_EXACT      = 7,
        /**
         * Publications prepared ON_DEVICE or ON_DEVICE_HOST_READABLE on the
         * device. An ovstage stage adds a requirement of its own, which
         * `ovspatialquery_query_ovstage_capability` checks.
         */
        OVSPATIALQUERY_CAPABILITY_PREPARE_ON_DEVICE = 8,
        /** Pair reporters with results on the device (ovspatialquery_pairs.h). */
        OVSPATIALQUERY_CAPABILITY_PAIR_REPORTER = 9,
    } ovspatialquery_capability_t;

    /**
     * Metadata returned by every synchronous `*_multiple_sync` query family.
     *
     * Raycast-multiple writes `ovspatialquery_raycast_hit_t` rows to its raycast
     * buffer, sphere- and box-sweep-multiple write `ovspatialquery_sphere_sweep_hit_t`
     * rows to a sweep buffer, and overlap-multiple writes
     * `ovspatialquery_overlap_hit_t` rows to its overlap buffer. This structure
     * reports only their shared outcome: the exact
     * ordinal read, how many rows were stored, and whether additional matches were
     * omitted because the caller's typed buffer was full.
     */
    typedef struct ovspatialquery_multiple_result_t
    {
        /** Exact population ordinal read by this query. */
        ovspatialquery_ordinal_t source_ordinal;
        /** Number of valid rows written to the query family's typed hit buffer. */
        uint64_t stored_hit_count;
        /** Nonzero if valid hits were omitted because the buffer was full. */
        uint32_t truncated;
    } ovspatialquery_multiple_result_t;

    /* -------------------------------------------------------------------------- */
    /* Synchronous raycasts                                                       */
    /* -------------------------------------------------------------------------- */

    /** States whether one returned row is AABB-candidate or exact-surface data. */
    typedef enum ovspatialquery_hit_truth_t
    {
        /** Contact fields describe the object's published world AABB only. */
        OVSPATIALQUERY_HIT_BOUNDS_CANDIDATE = 0,
        /** Contact fields describe the stored geometry surface. */
        OVSPATIALQUERY_HIT_EXACT_SURFACE = 1,
    } ovspatialquery_hit_truth_t;

    /** Object, immutable geometry, and material identity shared by all scene-query hits. */
    typedef struct ovspatialquery_hit_identity_t
    {
        ovspatialquery_object_id_t    object_id;
        ovspatialquery_geometry_ref_t geometry;
        /**
         * Material of the reported surface; zero means none. The population
         * assigns it, not the query, so a physics caller can look up friction
         * and restitution without a second query. An exact triangle-mesh
         * contact reports its source face's material when the mesh definition
         * assigns that face a nonzero one, and the object's material
         * otherwise; every other hit, bounds candidates and overlaps included,
         * reports the object's material. Explicit populations take the values
         * from the caller (`ovspatialquery_explicit.h`); ovstage populations
         * report the bound physics material's path value
         * (`ovspatialquery_ovstage.h`). A material belongs to the published
         * version, like the object's pose.
         */
        uint64_t material_id;
    } ovspatialquery_hit_identity_t;

    /** Exact stored-geometry feature kind. Candidate rows have no feature. */
    typedef enum ovspatialquery_geometry_feature_kind_t
    {
        /** The exact hit has no more geometry-specific data. */
        OVSPATIALQUERY_GEOMETRY_FEATURE_NONE = 0,
        /** The exact hit identifies one triangle through source-mesh topology. */
        OVSPATIALQUERY_GEOMETRY_FEATURE_TRIANGLE = 1,
    } ovspatialquery_geometry_feature_kind_t;

    /** Triangle identity of an exact raycast hit on a triangle mesh. */
    typedef struct ovspatialquery_triangle_feature_t
    {
        /**
         * Stable zero-based face index relative to the source mesh.
         *
         * Explicit triangle input uses the definition's `source_face_indices`,
         * or the submitted triangle ordinal without them. ovstage
         * input uses the source USD polygon ordinal in faceVertexCounts order.
         * The implementation maps any internal triangulation or reordering back
         * to this source numbering before return.
         */
        uint64_t source_face_index;
        /**
         * Three zero-based corner ordinals within `source_face_index`.
         *
         * For ovstage input, corner `c` resolves to the source point index at
         * `faceVertexIndices[face_offset + c]`, where `face_offset` is the sum of
         * preceding `faceVertexCounts`. For explicit triangle input these values
         * are the definition's `source_face_corner_indices`, or `{ 0, 1, 2 }` in
         * submitted triangle order without them. The array exposes source
         * topology, not an internal triangle ID or prepared vertex numbering.
         */
        uint32_t source_face_corner_indices[3];
        /**
         * Weights for `source_face_corner_indices[0..2]` in the same order.
         * The values sum to one. These are not authored texture coordinates.
         */
        float barycentrics[3];
    } ovspatialquery_triangle_feature_t;

    /**
     * Small tagged payload present only on exact stored-geometry contacts. The
     * payload is 48 bytes whatever `kind` holds, so a later feature kind of up
     * to that size keeps the size of every hit record.
     */
    typedef struct ovspatialquery_geometry_feature_t
    {
        ovspatialquery_geometry_feature_kind_t kind; /* NONE or TRIANGLE */
        union
        {
            /** Valid only when `kind` is GEOMETRY_FEATURE_TRIANGLE. */
            ovspatialquery_triangle_feature_t triangle;
            /** Fixes the payload size; unspecified contents. */
            uint64_t reserved[6];
        } data;
    } ovspatialquery_geometry_feature_t;

    /** One bounded ray interval inside the input domain. Direction is finite and unit length. */
    typedef struct ovspatialquery_ray_t
    {
        double origin[3];
        float  unit_direction[3];
        double min_distance;
        double max_distance;
    } ovspatialquery_ray_t;

    /**
     * How many rows one triangle mesh contributes to a MULTIPLE raycast.
     *
     * A ray can cross a concave or closed mesh several times. ALL, the default,
     * reports every crossing ("every surface along the ray"), as PhysX's
     * eMESH_MULTIPLE does. NEAREST keeps one row per object ("which objects does
     * the ray reach, and where first"), which is cheaper when that is the
     * question. Other geometry always contributes at most one row, and
     * BOUNDS_CANDIDATE precision one row per object. Closest and any raycasts
     * report one row whichever value they carry.
     */
    typedef enum ovspatialquery_mesh_hits_t
    {
        /**
         * One row per triangle the ray crosses within the interval, each with
         * its own source feature. A single-sided mesh reports only crossings
         * against its front faces unless the query sets MESH_BOTH_SIDES. A ray through a shared edge or vertex may
         * report each triangle sharing it. A MULTIPLE raycast with a nonzero
         * contact_tolerance returns NOT_SUPPORTED, because a tolerance tube has
         * no crossing count; it can ask for NEAREST.
         */
        OVSPATIALQUERY_MESH_HITS_ALL = 0,
        /** One row per object, its nearest contact. */
        OVSPATIALQUERY_MESH_HITS_NEAREST = 1,
    } ovspatialquery_mesh_hits_t;

    typedef struct ovspatialquery_raycast_desc_t
    {
        /** Target, ordinal, participation, precision, and tolerance. */
        ovspatialquery_query_common_t common;
        /** Ray geometry only. */
        ovspatialquery_ray_t ray;
        /** Read by the multiple raycast; closest and any accept either value. */
        ovspatialquery_mesh_hits_t mesh_hits; /* ALL or NEAREST */
    } ovspatialquery_raycast_desc_t;

    /** Broad-phase ray contact with an object's published world AABB. */
    typedef struct ovspatialquery_raycast_bounds_candidate_t
    {
        ovspatialquery_hit_identity_t identity;
        /** First point where the ray enters the published AABB. */
        double bounds_position[3];
        /** Outward normal of the AABB face at `bounds_position`. */
        float bounds_normal[3];
        /** Distance from the ray origin to `bounds_position`. */
        double distance;
    } ovspatialquery_raycast_bounds_candidate_t;

    /** Exact ray contact with the stored geometry surface. */
    typedef struct ovspatialquery_raycast_exact_hit_t
    {
        ovspatialquery_hit_identity_t identity;
        double                        position[3];
        float                         normal[3];
        double                        distance;
        /** NONE for non-mesh contact; TRIANGLE for triangle-mesh contact. */
        ovspatialquery_geometry_feature_t feature;
    } ovspatialquery_raycast_exact_hit_t;

    /**
     * Tagged raycast row. Inspect `truth` before selecting the union arm.
     * C cannot prevent reading the wrong arm; the distinct names and tag make the
     * validity rule explicit. A candidate arm never contains an exact feature.
     */
    typedef struct ovspatialquery_raycast_hit_t
    {
        /** Must match the descriptor/plan precision that produced this row. */
        ovspatialquery_hit_truth_t truth; /* BOUNDS_CANDIDATE or EXACT_SURFACE */
        union
        {
            ovspatialquery_raycast_bounds_candidate_t bounds_candidate;
            ovspatialquery_raycast_exact_hit_t        exact;
        } data;
    } ovspatialquery_raycast_hit_t;

    typedef struct ovspatialquery_raycast_hit_buffer_t
    {
        /** Caller-owned array written with at most `capacity` ray hits. */
        ovspatialquery_raycast_hit_t* hits;
        uint64_t                      capacity;
    } ovspatialquery_raycast_hit_buffer_t;

    /** Result shared by closest and any raycasts; the function defines selection. */
    typedef struct ovspatialquery_raycast_single_result_t
    {
        /** Exact population ordinal read by this query. */
        ovspatialquery_ordinal_t source_ordinal;
        /** Zero or one. `hit` is valid only when this is one. */
        uint32_t                     has_hit;
        ovspatialquery_raycast_hit_t hit;
    } ovspatialquery_raycast_single_result_t;

    /* -------------------------------------------------------------------------- */
    /* Synchronous sphere sweeps                                                  */
    /* -------------------------------------------------------------------------- */

    /** Geometry and motion of one bounded sphere sweep. */
    typedef struct ovspatialquery_sphere_sweep_t
    {
        double center[3];
        double radius;
        float  unit_direction[3];
        double min_distance;
        double max_distance;
    } ovspatialquery_sphere_sweep_t;

    typedef struct ovspatialquery_sphere_sweep_desc_t
    {
        /** Target, ordinal, participation, precision, and tolerance. */
        ovspatialquery_query_common_t common;
        /** Moving sphere geometry only. */
        ovspatialquery_sphere_sweep_t sweep;
    } ovspatialquery_sphere_sweep_desc_t;

    /** Broad-phase moving-sphere contact with an object's published world AABB. */
    typedef struct ovspatialquery_sphere_sweep_bounds_candidate_t
    {
        ovspatialquery_hit_identity_t identity;
        double                        query_position[3];
        double                        bounds_position[3];
        float                         bounds_normal[3];
        double                        distance;
    } ovspatialquery_sphere_sweep_bounds_candidate_t;

    /**
     * Exact moving-sphere contact with the stored geometry surface. A sweep that
     * starts clear of the object reports `object_position` one radius from
     * `query_position`; one that starts overlapping it reports the surface point
     * nearest the center. On a triangle the normal points from the contact toward
     * the sphere's center, which is the triangle's back side for a contact from
     * behind.
     */
    typedef struct ovspatialquery_sphere_sweep_exact_hit_t
    {
        ovspatialquery_hit_identity_t identity;
        /** Center of the moving query shape at the first reported contact. */
        double query_position[3];
        /** Contact point on the stored object surface. */
        double                            object_position[3];
        float                             normal[3];
        double                            distance;
        ovspatialquery_geometry_feature_t feature;
    } ovspatialquery_sphere_sweep_exact_hit_t;

    /** Tagged sweep row; candidate and exact contact data cannot be confused. */
    typedef struct ovspatialquery_sphere_sweep_hit_t
    {
        /** Must match the descriptor/plan precision that produced this row. */
        ovspatialquery_hit_truth_t truth; /* BOUNDS_CANDIDATE or EXACT_SURFACE */
        union
        {
            ovspatialquery_sphere_sweep_bounds_candidate_t bounds_candidate;
            ovspatialquery_sphere_sweep_exact_hit_t        exact;
        } data;
    } ovspatialquery_sphere_sweep_hit_t;

    /** Result shared by closest and any sweeps; the function defines selection. */
    typedef struct ovspatialquery_sphere_sweep_closest_result_t
    {
        ovspatialquery_ordinal_t source_ordinal;
        /** Zero or one. `hit` is valid only when this is one. */
        uint32_t                          has_hit;
        ovspatialquery_sphere_sweep_hit_t hit;
    } ovspatialquery_sphere_sweep_closest_result_t;

    /**
     * A box translated along a straight path; its orientation stays fixed.
     * Centers and half extents are inside the input domain, half extents positive; rotation is a
     * unit xyzw quaternion (squared-length tolerance 1e-5). Direction and the
     * closed distance interval follow sphere-sweep rules, except that a box ignores
     * a single-sided triangle it moves parallel to (see the distance intervals
     * above). Invalid geometry returns INVALID_ARGUMENT synchronously or marks the
     * batch row invalid.
     */
    typedef struct ovspatialquery_box_sweep_t
    {
        double center[3];
        double half_extents[3];
        float  rotation_xyzw[4];
        float  unit_direction[3];
        double min_distance;
        double max_distance;
    } ovspatialquery_box_sweep_t;

    typedef struct ovspatialquery_box_sweep_desc_t
    {
        ovspatialquery_query_common_t common;
        ovspatialquery_box_sweep_t    sweep;
    } ovspatialquery_box_sweep_desc_t;

    /** Sweep contacts share a layout; query_position is the moving shape's center. */
    typedef ovspatialquery_sphere_sweep_bounds_candidate_t ovspatialquery_box_sweep_bounds_candidate_t;
    typedef ovspatialquery_sphere_sweep_exact_hit_t        ovspatialquery_box_sweep_exact_hit_t;
    typedef ovspatialquery_sphere_sweep_hit_t              ovspatialquery_box_sweep_hit_t;
    typedef ovspatialquery_sphere_sweep_closest_result_t   ovspatialquery_box_sweep_closest_result_t;

    /**
     * Caller-owned rows of a multiple sweep. A multiple sweep reports one row per
     * object it reaches, at that object's first contact: a triangle mesh gives one
     * row, not one per triangle, as PhysX sweeps do. An object the shape already
     * touches at `min_distance` is reported there, and a triangle the sweep
     * ignores (see the distance intervals above) is ignored in every mode.
     */
    typedef struct ovspatialquery_sphere_sweep_hit_buffer_t
    {
        /** Caller-owned array written with at most `capacity` sweep hits. */
        ovspatialquery_sphere_sweep_hit_t* hits;
        uint64_t                           capacity;
    } ovspatialquery_sphere_sweep_hit_buffer_t;

    /** Box sweeps write the same rows. */
    typedef ovspatialquery_sphere_sweep_hit_buffer_t ovspatialquery_box_sweep_hit_buffer_t;

    /* -------------------------------------------------------------------------- */
    /* Synchronous overlaps                                                       */
    /* -------------------------------------------------------------------------- */

    /** Sphere overlap shape in caller coordinates inside the input domain. Radius is positive. */
    typedef struct ovspatialquery_overlap_sphere_t
    {
        double center[3];
        double radius;
    } ovspatialquery_overlap_sphere_t;

    /** Oriented-box overlap shape. The quaternion is normalized. */
    typedef struct ovspatialquery_overlap_box_t
    {
        double center[3];
        float  rotation_xyzw[4];
        double half_extents[3];
    } ovspatialquery_overlap_box_t;

    /** Axis-aligned overlap shape with componentwise ordered corners inside the input domain. */
    typedef struct ovspatialquery_overlap_aabb_t
    {
        double min_corner[3];
        double max_corner[3];
    } ovspatialquery_overlap_aabb_t;

    /** Overlap shapes used by this design. */
    typedef enum ovspatialquery_overlap_shape_kind_t
    {
        OVSPATIALQUERY_OVERLAP_SPHERE = 0,
        OVSPATIALQUERY_OVERLAP_BOX    = 1,
        OVSPATIALQUERY_OVERLAP_AABB   = 2,
    } ovspatialquery_overlap_shape_kind_t;

    /** Tagged overlap shape. It is distinct from stored object geometry. */
    typedef struct ovspatialquery_overlap_shape_t
    {
        ovspatialquery_overlap_shape_kind_t kind; /* SPHERE, BOX, or AABB */
        union
        {
            ovspatialquery_overlap_sphere_t sphere;
            ovspatialquery_overlap_box_t    box;
            ovspatialquery_overlap_aabb_t   aabb;
        } data;
    } ovspatialquery_overlap_shape_t;

    /** Tests one query shape at its supplied pose. */
    typedef struct ovspatialquery_overlap_desc_t
    {
        ovspatialquery_query_common_t  common;
        ovspatialquery_overlap_shape_t shape;
    } ovspatialquery_overlap_desc_t;

    /**
     * One overlap identity. `truth` reports whether the overlap test used only the
     * published AABB or exact stored geometry. Overlap has no unique contact point
     * and therefore no exact geometry feature.
     */
    typedef struct ovspatialquery_overlap_hit_t
    {
        /** Must match the descriptor/plan precision that produced this row. */
        ovspatialquery_hit_truth_t    truth; /* BOUNDS_CANDIDATE or EXACT_SURFACE */
        ovspatialquery_hit_identity_t identity;
    } ovspatialquery_overlap_hit_t;

    typedef struct ovspatialquery_overlap_hit_buffer_t
    {
        /** Caller-owned array written with at most `capacity` overlap hits. */
        ovspatialquery_overlap_hit_t* hits;
        uint64_t                      capacity;
    } ovspatialquery_overlap_hit_buffer_t;

    typedef struct ovspatialquery_overlap_any_result_t
    {
        /** Exact population ordinal read by this query. */
        ovspatialquery_ordinal_t source_ordinal;
        /** Zero or one. `hit` is valid only when this is one. */
        uint32_t                     has_hit;
        ovspatialquery_overlap_hit_t hit;
    } ovspatialquery_overlap_any_result_t;

    /* -------------------------------------------------------------------------- */
    /* Reusable batch queries                                                     */
    /* -------------------------------------------------------------------------- */

    typedef enum ovspatialquery_hit_mode_t
    {
        OVSPATIALQUERY_HIT_MODE_CLOSEST  = 0,
        OVSPATIALQUERY_HIT_MODE_ANY      = 1,
        OVSPATIALQUERY_HIT_MODE_MULTIPLE = 2,
    } ovspatialquery_hit_mode_t;

    /** Closed device-neutral status for one admitted batch row. */
    typedef enum ovspatialquery_batch_row_status_t
    {
        /** The row's span, truncation value, and stored hits are valid. */
        OVSPATIALQUERY_BATCH_ROW_SUCCESS = 0,
        /** This row's values are invalid; other rows remain independent. */
        OVSPATIALQUERY_BATCH_ROW_INVALID_ARGUMENT = 1,
        /** A data-dependent case cannot be evaluated by the admitted plan. */
        OVSPATIALQUERY_BATCH_ROW_NOT_SUPPORTED = 2,
        /** This row failed. A recoverable operation failure marks every row. */
        OVSPATIALQUERY_BATCH_ROW_FAILED = 3,
    } ovspatialquery_batch_row_status_t;

    /**
     * Maps submitted query rows to compact ranges in the flat hit columns.
     *
     * Q is the submitted query count. C is the payload capacity shown by an
     * enqueue reservation. H is the terminal number of stored hits. N is the
     * first dimension of a payload column: N is C in a reservation and H in a
     * successful fetched result.
     *
     * Example: three rays store zero, one, and two hits. Their offsets can be
     * `[0, 0, 1]`, their counts `[0, 1, 2]`, and H is 3. Ray 1 reads flat hit row
     * 0. Ray 2 reads rows 1 and 2. Ray 0 reads no row.
     */
    typedef struct ovspatialquery_batch_hit_ranges_t
    {
        /** uint32[Q]. One `ovspatialquery_batch_row_status_t` per query. */
        DLTensor query_status;
        /** uint64[1]. Device-readable H, the total valid flat payload rows. */
        DLTensor stored_hit_count;
        /** uint64[Q]. First row for a nonempty success; ignore for a success miss. */
        DLTensor hit_offsets;
        /** uint64[Q]. Number of stored hit rows for each query; zero is a miss. */
        DLTensor hit_counts;
        /** uint8[Q]. Nonzero when that query omitted one or more matching hits. */
        DLTensor truncated;
    } ovspatialquery_batch_hit_ranges_t;

    /** Optional semantic batch columns selected when a CPU or CUDA plan is created. */
    typedef enum ovspatialquery_batch_result_field_t
    {
        /** Candidate-bounds or exact-surface contact positions, float64[N,3]. */
        OVSPATIALQUERY_BATCH_RESULT_POSITION = 1,
        OVSPATIALQUERY_BATCH_RESULT_NORMAL   = 2,
        OVSPATIALQUERY_BATCH_RESULT_DISTANCE = 4,
        /** Requests the complete typed source-feature column bundle. */
        OVSPATIALQUERY_BATCH_RESULT_SOURCE_FEATURE = 8,
        /** Requests the complete geometry identity group for every stored hit. */
        OVSPATIALQUERY_BATCH_RESULT_GEOMETRY_IDENTITY = 16,
        /** Requests sphere-sweep query contact positions, float64[N,3]. */
        OVSPATIALQUERY_BATCH_RESULT_QUERY_POSITION = 32,
        /** Requests `material_ids`, uint64[N], valid for every family and precision. */
        OVSPATIALQUERY_BATCH_RESULT_MATERIAL = 64,
        /**
         * Not a column: a CUDA plan places all its result columns in mapped
         * host memory, so a host reader needs no copy (the batch column
         * contract below). A CPU plan accepts and ignores it, so one mask
         * serves both devices.
         */
        OVSPATIALQUERY_BATCH_RESULT_HOST_MAPPED = 128,
    } ovspatialquery_batch_result_field_t;

    /**
     * Pose-free overlap shape whose size every row of a reusable plan launch
     * uses unless the launch supplies that row's own radius or half extents.
     * Only BOX rows have a rotation column; sphere and AABB rows do not accept one.
     */
    typedef struct ovspatialquery_overlap_batch_shape_t
    {
        ovspatialquery_overlap_shape_kind_t kind; /* SPHERE, BOX, or AABB */
        union
        {
            struct
            {
                double radius;
            } sphere;
            struct
            {
                double half_extents[3];
            } box;
            struct
            {
                double half_extents[3];
            } aabb;
        } data;
    } ovspatialquery_overlap_batch_shape_t;

    /* Family-specific plan descriptors contain no fields for other families. */
    typedef struct ovspatialquery_raycast_plan_desc_t
    {
        /** Population or view retained by the plan; ordinal is chosen per launch. */
        ovspatialquery_target_ref_t target;
        /** Closest, any, or bounded multiple ray hits. */
        ovspatialquery_hit_mode_t        hit_mode;  /* CLOSEST, ANY, or MULTIPLE */
        ovspatialquery_query_precision_t precision; /* BOUNDS_CANDIDATE, EXACT, or EXACT_SINGLE */
        /** kDLCPU or one CUDA device enabled by the instance. */
        DLDevice device;
        /** Hard maximum rows accepted by one launch. */
        uint64_t maximum_query_count;
        /** Per-query limit used only by MULTIPLE mode; one otherwise. */
        uint64_t maximum_hits_per_query;
        /** Total hit rows reserved in each result slot. */
        uint64_t hit_capacity_per_slot;
        /** Number of launches whose immutable results may coexist. */
        uint32_t result_slot_count;
        /**
         * ORed `ovspatialquery_batch_result_field_t` values for either device.
         * SOURCE_FEATURE is valid only for an EXACT or EXACT_SINGLE plan, and QUERY_POSITION is
         * invalid for raycast. Object IDs are always present and are not requested
         * through this mask, so zero requests an identity-only result in which hit
         * ranges and object IDs remain present. Unknown bits make plan creation
         * return INVALID_ARGUMENT.
         */
        uint32_t result_fields; /* bits: POSITION | NORMAL | DISTANCE | SOURCE_FEATURE |
                                   GEOMETRY_IDENTITY | HOST_MAPPED */
        double contact_tolerance;
        /** Read by MULTIPLE plans; CLOSEST and ANY accept either value. */
        ovspatialquery_mesh_hits_t mesh_hits; /* ALL or NEAREST */
    } ovspatialquery_raycast_plan_desc_t;

    typedef struct ovspatialquery_sphere_sweep_plan_desc_t
    {
        /** Population or view retained by the plan; ordinal is chosen per launch. */
        ovspatialquery_target_ref_t      target;
        ovspatialquery_query_precision_t precision; /* BOUNDS_CANDIDATE or EXACT */
        /** Positive default radius, used when a launch omits its radii column. */
        double radius;
        /** kDLCPU or one CUDA device enabled by the instance. */
        DLDevice device;
        uint64_t maximum_query_count;
        uint32_t result_slot_count;
        /**
         * Same device-neutral optional-column mask as the raycast plan.
         * QUERY_POSITION is legal for sphere and box sweeps, SOURCE_FEATURE is
         * exact-only, and zero requests only hit ranges and object IDs.
         */
        uint32_t result_fields; /* bits: POSITION | NORMAL | DISTANCE | SOURCE_FEATURE |
                                   GEOMETRY_IDENTITY | QUERY_POSITION | HOST_MAPPED */
        /**
         * Total hit rows reserved in each result slot: exactly
         * `maximum_query_count` for CLOSEST and ANY, which store one hit per
         * row, and any total for MULTIPLE.
         */
        uint64_t hit_capacity_per_slot;
        double   contact_tolerance;
        /**
         * CLOSEST, the zero value, keeps each row's nearest hit; ANY keeps one hit
         * of any object reached; MULTIPLE keeps one hit per object reached, as the
         * multiple sweeps do. Other values return INVALID_ARGUMENT.
         */
        ovspatialquery_hit_mode_t hit_mode; /* CLOSEST, ANY, or MULTIPLE */
        /**
         * Positive per-query limit used only by MULTIPLE mode; CLOSEST and ANY
         * accept zero or one, so a zero-initialized descriptor stays valid. A truncated row
         * keeps its nearest hits by distance, then object ID.
         */
        uint64_t maximum_hits_per_query;
    } ovspatialquery_sphere_sweep_plan_desc_t;

    /** Box sweeps with per-row starting poses and a default shape size. */
    typedef struct ovspatialquery_box_sweep_plan_desc_t
    {
        ovspatialquery_target_ref_t      target;
        ovspatialquery_query_precision_t precision; /* BOUNDS_CANDIDATE or EXACT */
        /** Finite positive local half extents, used when a launch omits its half_extents column. */
        double   half_extents[3];
        DLDevice device;
        uint64_t maximum_query_count;
        uint32_t result_slot_count;
        /** Same optional columns as sphere sweeps. */
        uint32_t result_fields; /* bits: POSITION | NORMAL | DISTANCE | SOURCE_FEATURE |
                                   GEOMETRY_IDENTITY | QUERY_POSITION | HOST_MAPPED */
        /** As for sphere sweeps: `maximum_query_count` unless MULTIPLE. */
        uint64_t hit_capacity_per_slot;
        /** Nonnegative amount added to each local half extent. */
        double contact_tolerance;
        /** As for sphere sweeps. */
        ovspatialquery_hit_mode_t hit_mode; /* CLOSEST, ANY, or MULTIPLE */
        /** As for sphere sweeps: read only by MULTIPLE. */
        uint64_t maximum_hits_per_query;
    } ovspatialquery_box_sweep_plan_desc_t;

    typedef struct ovspatialquery_overlap_plan_desc_t
    {
        /** Population or view retained by the plan; ordinal is chosen per launch. */
        ovspatialquery_target_ref_t      target;
        ovspatialquery_hit_mode_t        hit_mode;  /* ANY or MULTIPLE; overlap has no closest hit */
        ovspatialquery_query_precision_t precision; /* BOUNDS_CANDIDATE, EXACT, or EXACT_SINGLE */
        /** Pose-free shape kind and default size for every row in each launch. */
        ovspatialquery_overlap_batch_shape_t shape;
        /** kDLCPU or one CUDA device enabled by the instance. */
        DLDevice device;
        uint64_t maximum_query_count;
        uint64_t maximum_hits_per_query;
        uint64_t hit_capacity_per_slot;
        uint32_t result_slot_count;
        /**
         * Zero, GEOMETRY_IDENTITY, MATERIAL, or both, each optionally with
         * HOST_MAPPED; hit ranges and object IDs are always present.
         */
        uint32_t result_fields; /* bits: GEOMETRY_IDENTITY | MATERIAL | HOST_MAPPED */
        double   contact_tolerance;
    } ovspatialquery_overlap_plan_desc_t;

    /**
     * Per-launch batch inputs. CPU plans require zero `cuda_sync.has_stream` and
     * `cuda_sync.wait_event`. CUDA plans require a stream (`has_stream` 1, any
     * stream value) and queue the launch on it after the wait event, so the
     * library reads no DLPack column before the caller's producer finishes. A
     * consumer that queues its reads of the result columns on the same stream
     * after enqueue returns needs no wait on `result_ready`: stream order
     * already places them after the launch.
     */
    typedef struct ovspatialquery_raycast_batch_t
    {
        ovspatialquery_ordinal_selection_t  ordinal;
        uint64_t                            query_count;
        const ovspatialquery_input_array_t* query_bits;      /* optional uint64[Q,2] */
        const ovspatialquery_input_array_t* origins;         /* float64[Q,3] */
        const ovspatialquery_input_array_t* unit_directions; /* float32[Q,3] */
        const ovspatialquery_input_array_t* min_distances;   /* optional float64[Q] */
        const ovspatialquery_input_array_t* max_distances;   /* float64[Q] */
        const ovspatialquery_input_array_t* maximum_hits;    /* multiple only, optional uint64[Q] */
        ovspatialquery_cuda_sync_t          cuda_sync;
        /**
         * Options for every row of this launch, as for a synchronous query's
         * `query_flags`; unknown bits make enqueue return NOT_SUPPORTED with no
         * effect. A caller that needs rows with different options enqueues them
         * as separate launches.
         */
        uint32_t query_flags; /* bits: MESH_BOTH_SIDES */
        /**
         * Room for two optional per-row input columns without a layout change.
         * Null means absent. A non-null entry makes enqueue return NOT_SUPPORTED
         * with no effect, so a caller that sets a column this library predates
         * fails cleanly.
         */
        const ovspatialquery_input_array_t* reserved[2];
    } ovspatialquery_raycast_batch_t;

    typedef struct ovspatialquery_sphere_sweep_batch_t
    {
        ovspatialquery_ordinal_selection_t  ordinal;
        uint64_t                            query_count;
        const ovspatialquery_input_array_t* query_bits;      /* optional uint64[Q,2] */
        const ovspatialquery_input_array_t* centers;         /* float64[Q,3] */
        const ovspatialquery_input_array_t* unit_directions; /* float32[Q,3] */
        const ovspatialquery_input_array_t* min_distances;   /* optional float64[Q] */
        const ovspatialquery_input_array_t* max_distances;   /* float64[Q] */
        /**
         * Optional float64[Q]. Each row replaces the plan radius before adding
         * contact_tolerance; a null column uses the plan radius. A positive value
         * inside the input domain is required per row, or that row reports
         * INVALID_ARGUMENT.
         * A malformed column descriptor makes enqueue return INVALID_ARGUMENT.
         */
        const ovspatialquery_input_array_t* radii;
        ovspatialquery_cuda_sync_t          cuda_sync;
        /** As for raycast batches. */
        uint32_t query_flags; /* bits: MESH_BOTH_SIDES */
        /** As for raycast batches. */
        const ovspatialquery_input_array_t* reserved[2];
    } ovspatialquery_sphere_sweep_batch_t;

    /** Fixed-orientation translations; input ownership matches sphere sweeps. */
    typedef struct ovspatialquery_box_sweep_batch_t
    {
        ovspatialquery_ordinal_selection_t  ordinal;
        uint64_t                            query_count;
        const ovspatialquery_input_array_t* query_bits;      /* optional uint64[Q,2] */
        const ovspatialquery_input_array_t* centers;         /* float64[Q,3] */
        const ovspatialquery_input_array_t* rotations_xyzw;  /* float32[Q,4], unit quaternions */
        const ovspatialquery_input_array_t* unit_directions; /* float32[Q,3] */
        const ovspatialquery_input_array_t* min_distances;   /* optional float64[Q] */
        const ovspatialquery_input_array_t* max_distances;   /* float64[Q] */
        ovspatialquery_cuda_sync_t          cuda_sync;
        /**
         * Optional float64[Q,3]. Each row's local half extents replace the plan's
         * before adding contact_tolerance; a null column uses the plan's. A
         * positive triple inside the input domain is required per row, or that
         * row reports INVALID_ARGUMENT. A malformed column descriptor makes enqueue return
         * INVALID_ARGUMENT.
         */
        const ovspatialquery_input_array_t* half_extents;
        /** As for raycast batches. */
        uint32_t query_flags; /* bits: MESH_BOTH_SIDES */
        /** As for raycast batches. */
        const ovspatialquery_input_array_t* reserved[2];
    } ovspatialquery_box_sweep_batch_t;

    typedef struct ovspatialquery_overlap_batch_t
    {
        ovspatialquery_ordinal_selection_t  ordinal;
        uint64_t                            query_count;
        const ovspatialquery_input_array_t* query_bits;   /* optional uint64[Q,2] */
        const ovspatialquery_input_array_t* translations; /* float64[Q,3] */
        /**
         * Required for BOX, float32[Q,4] or float64[Q,4] unit quaternions; null
         * for SPHERE or AABB. A float64 column orients each row's box in double
         * precision on the CPU and on the device.
         */
        const ovspatialquery_input_array_t* rotations_xyzw;
        /**
         * Optional float64[Q,3] for AABB and BOX plans. Each row's half extents
         * replace the plan shape's before adding contact_tolerance; a null
         * column uses the plan's. A positive triple inside the input domain is
         * required per row, and an AABB row's corners lie inside it too, or that
         * row reports INVALID_ARGUMENT. A non-null column with a
         * SPHERE plan, or a malformed column descriptor, makes enqueue return
         * INVALID_ARGUMENT.
         */
        const ovspatialquery_input_array_t* half_extents;
        const ovspatialquery_input_array_t* maximum_hits; /* multiple only, optional uint64[Q] */
        ovspatialquery_cuda_sync_t          cuda_sync;
        /**
         * Optional float64[Q] for SPHERE plans. Each row's radius replaces the
         * plan shape's before adding contact_tolerance; a null column uses the
         * plan's. A positive value inside the input domain is required per row,
         * or that row reports INVALID_ARGUMENT. A non-null column with a BOX or AABB plan,
         * or a malformed column descriptor, makes enqueue return INVALID_ARGUMENT.
         */
        const ovspatialquery_input_array_t* radii;
        /** As for raycast batches. */
        const ovspatialquery_input_array_t* reserved[2];
    } ovspatialquery_overlap_batch_t;

    /** Immutable admission metadata shared by reservations and completed results. */
    typedef struct ovspatialquery_batch_result_common_t
    {
        /** Plan-owned slot held exclusively by this operation. */
        uint32_t result_slot;
        /** Number of submitted query rows. */
        uint64_t query_count;
        /** Physical stored-hit capacity of this result slot. */
        uint64_t hit_capacity;
        /** Candidate or exact tag fixed by the admitted plan precision. */
        ovspatialquery_hit_truth_t truth; /* BOUNDS_CANDIDATE or EXACT_SURFACE */
        /** Exact immutable population ordinal selected by the launch. */
        ovspatialquery_ordinal_t source_ordinal;
        /** Exact CPU or CUDA device fixed by the admitted plan. */
        DLDevice execution_device;
        /** Zero for CPU; one library-owned completion event for all CUDA columns. */
        ovspatialquery_cuda_event_t result_ready;
    } ovspatialquery_batch_result_common_t;

    /** Geometry identity is one atomic optional column group. N is C or H below. */
    typedef struct ovspatialquery_geometry_identity_columns_t
    {
        DLTensor types;  /* int32[N], ovspatialquery_geometry_type_t values */
        DLTensor values; /* uint64[N] */
    } ovspatialquery_geometry_identity_columns_t;

    /** Exact source-feature columns are one atomic optional group. N is C or H. */
    typedef struct ovspatialquery_exact_feature_columns_t
    {
        DLTensor kinds;                 /* uint32[N], geometry_feature_kind values */
        DLTensor source_face_indices;   /* uint64[N] */
        DLTensor source_corner_indices; /* uint32[N,3] */
        DLTensor barycentrics;          /* float32[N,3] */
    } ovspatialquery_exact_feature_columns_t;

    /** Device-neutral raycast columns shared by reservations and completed results. */
    typedef struct ovspatialquery_raycast_batch_columns_t
    {
        ovspatialquery_batch_hit_ranges_t          hit_ranges;
        DLTensor                                   object_ids; /* required uint64[N] */
        ovspatialquery_geometry_identity_columns_t geometry;
        /** Optional uint64[N]: each hit's `ovspatialquery_hit_identity_t` material_id. */
        DLTensor material_ids;
        /** `common.truth` selects one arm. C cannot prevent wrong-arm access. */
        union
        {
            struct
            {
                DLTensor bounds_positions; /* optional float64[N,3] */
                DLTensor bounds_normals;   /* optional float32[N,3] */
                DLTensor distances;        /* optional float64[N] */
            } candidate;
            struct
            {
                DLTensor                               positions; /* optional float64[N,3] */
                DLTensor                               normals;   /* optional float32[N,3] */
                DLTensor                               distances; /* optional float64[N] */
                ovspatialquery_exact_feature_columns_t features;
            } exact;
        } contact;
        /**
         * Room for four more columns without a layout change. The library
         * returns them as all-zero absent descriptors.
         */
        DLTensor reserved[4];
    } ovspatialquery_raycast_batch_columns_t;

    /** Device-neutral sphere-sweep columns shared by reservations and results. */
    typedef struct ovspatialquery_sphere_sweep_batch_columns_t
    {
        ovspatialquery_batch_hit_ranges_t          hit_ranges;
        DLTensor                                   object_ids; /* required uint64[N] */
        ovspatialquery_geometry_identity_columns_t geometry;
        /** Optional uint64[N]: each hit's `ovspatialquery_hit_identity_t` material_id. */
        DLTensor material_ids;
        /** `common.truth` selects one arm. C cannot prevent wrong-arm access. */
        union
        {
            struct
            {
                DLTensor bounds_positions; /* optional float64[N,3] */
                DLTensor bounds_normals;   /* optional float32[N,3] */
                DLTensor query_positions;  /* optional float64[N,3] */
                DLTensor distances;        /* optional float64[N] */
            } candidate;
            struct
            {
                DLTensor                               positions;       /* optional float64[N,3] */
                DLTensor                               normals;         /* optional float32[N,3] */
                DLTensor                               query_positions; /* optional float64[N,3] */
                DLTensor                               distances;       /* optional float64[N] */
                ovspatialquery_exact_feature_columns_t features;
            } exact;
        } contact;
        /**
         * Room for four more columns without a layout change. The library
         * returns them as all-zero absent descriptors.
         */
        DLTensor reserved[4];
    } ovspatialquery_sphere_sweep_batch_columns_t;

    /** Overlap rows carry object identity and optional geometry identity. */
    typedef struct ovspatialquery_overlap_batch_columns_t
    {
        ovspatialquery_batch_hit_ranges_t          hit_ranges;
        DLTensor                                   object_ids; /* required uint64[N] */
        ovspatialquery_geometry_identity_columns_t geometry;
        /** Optional uint64[N]: each hit's `ovspatialquery_hit_identity_t` material_id. */
        DLTensor material_ids;
        /**
         * Room for four more columns without a layout change. The library
         * returns them as all-zero absent descriptors.
         */
        DLTensor reserved[4];
    } ovspatialquery_overlap_batch_columns_t;

    /**
     * Device-neutral batch column contract.
     *
     * Q is `common.query_count`, C is `common.hit_capacity`, and H is the terminal
     * `stored_hit_count`. Reservation payload tensors have capacity shapes based
     * on C. A successful typed fetch returns payload tensors with exact shapes
     * based on H. Hit-range tensors always use Q or scalar shape as documented.
     *
     * The library copies every returned DLTensor descriptor value into the
     * caller-owned output struct. Data storage and the shape and strides arrays
     * stay operation-owned. Reservation metadata arrays and fetched-result metadata
     * arrays are separate immutable snapshots that remain valid and unchanged
     * until the operation is destroyed.
     *
     * For each required or requested payload column, reservation and fetched
     * descriptors refer to the same backing storage. Their data, dtype, device,
     * strides, and byte_offset values are equal. Only shape[0] changes from C to H;
     * later dimensions do not change. A CUDA consumer can therefore enqueue from
     * reservation descriptors and wait on `common.result_ready` without a later
     * relocation invalidating its arguments.
     *
     * DLPack dtype, rank, shape, strides, device, and byte_offset are authoritative.
     * Every present tensor uses `common.execution_device`, except under
     * HOST_MAPPED below. CPU contents are unreadable while the operation is
     * pending, and all columns share readiness, failure, and lifetime.
     * Separate descriptors do not imply separate
     * allocations: published columns use fixed per-dimension element strides,
     * while backing allocation, internal packing, and any projection are
     * implementation details.
     *
     * An optional unrequested tensor is an all-zero absent descriptor; callers can
     * test `shape == NULL`. Every present result tensor has ndim >= 1, non-null
     * shape, and its documented dtype. A required or requested tensor with no
     * terminal hits is present with shape[0] == 0. Object IDs and hit-range tensors
     * are always present.
     *
     * Every present payload column except the triangle payload below is valid for
     * every stored hit. A provider cannot leave per-hit holes in such a column.
     *
     * Geometry `types` and `values` are present or absent together, as are the
     * exact feature columns. Within a present feature group, `kinds[i] == NONE`
     * leaves that row's source face, source corners, and barycentrics
     * unspecified, so a consumer must mask those columns by `kinds`;
     * `kinds[i] == TRIANGLE` makes its source face, three source corners, and
     * barycentrics valid. Candidate plans reject SOURCE_FEATURE. This semantic tag
     * replaces the former generic per-hit validity bitmap.
     *
     * `common.truth` selects the candidate or exact union arm. C cannot prevent a
     * caller from reading the wrong arm; the tag and distinct field names make the
     * validity rule explicit. Candidate contact columns describe only the
     * published AABB proxy, while exact contact columns describe the stored
     * surface. QUERY_POSITION is accepted only by sphere sweep and is the moving
     * sphere center at contact.
     *
     * On terminal success, successful nonempty ranges are pairwise disjoint and
     * densely cover [0,H), with no overlap or gap. Their order need not match query
     * order. Every offset + count is at most H, and H is at most C. A miss has
     * count zero. A non-success query row has zero offset, count, and truncation
     * and no payload. `truncated[i]` is nonzero exactly when a match was omitted by
     * that query's semantic limit or remaining capacity. A truncated raycast or
     * sweep keeps its nearest matches by distance, then object ID, so CPU and CUDA plans
     * store the same set, except which of one object's mesh crossings at exactly
     * the cut-off distance stay; order within a range is unspecified. Reserved storage in
     * [H,C) is unspecified and is outside every fetched payload shape.
     *
     * A recoverable CUDA operation-level failure publishes a safe empty device
     * result before signaling `result_ready`: stored_hit_count is zero; every query
     * status is BATCH_ROW_FAILED; and every offset, count, and truncation value is
     * zero. No payload row is readable. DEVICE_LOST makes no such storage or event
     * guarantee.
     *
     * A CUDA plan created with HOST_MAPPED in `result_fields` keeps every column
     * of every result slot in page-locked host memory mapped into the plan's
     * device, so a host reader reads in place instead of copying. Each present
     * tensor reports `{ kDLCUDAHost, 0 }`; its address is valid on the host and
     * on `common.execution_device`, which stays the plan's CUDA device. The
     * device writes every column, including the safe empty result above, before
     * `result_ready`, and the host may read once `result_ready` has completed:
     * a wait, status poll, or typed fetch that observes the terminal state
     * implies it. Launch, stream, wait-event, and ordinal rules are those of any
     * CUDA plan. Plan creation charges the mapped columns to the instance's
     * `host_byte_budget` (HOST_BUDGET when they do not fit) and the plan's
     * device working storage to the device budget.
     *
     * Implementation note: the plan computes on device storage as without
     * HOST_MAPPED, then one pass behind the launch copies each column into
     * mapped memory as one contiguous run, the payload columns sized from H on
     * the device, because scattered device stores to host memory cost several
     * times more than pageable copies.
     *
     * Exact plan admission rejects an unsupported device, family, or precision.
     * After admission, mandatory built-in geometry cannot produce NOT_SUPPORTED
     * and CUDA execution never falls back to CPU. A binding projects these tensors
     * directly and retains the operation until all consumers finish.
     */

    /**
     * Admission-time raycast descriptors; tensor contents are not yet readable.
     * This distinct type prevents a pending capacity-shaped reservation from being
     * passed to code that requires a terminal exact-shaped result.
     */
    typedef struct ovspatialquery_raycast_batch_reservation_t
    {
        ovspatialquery_batch_result_common_t   common;
        ovspatialquery_raycast_batch_columns_t columns;
    } ovspatialquery_raycast_batch_reservation_t;

    /**
     * Admission-time sphere-sweep descriptors; contents are not yet readable.
     * The distinct type represents pending capacity-shaped storage.
     */
    typedef struct ovspatialquery_sphere_sweep_batch_reservation_t
    {
        ovspatialquery_batch_result_common_t        common;
        ovspatialquery_sphere_sweep_batch_columns_t columns;
    } ovspatialquery_sphere_sweep_batch_reservation_t;

    /**
     * Admission-time overlap descriptors; tensor contents are not yet readable.
     * The distinct type represents pending capacity-shaped storage.
     */
    typedef struct ovspatialquery_overlap_batch_reservation_t
    {
        ovspatialquery_batch_result_common_t   common;
        ovspatialquery_overlap_batch_columns_t columns;
    } ovspatialquery_overlap_batch_reservation_t;

    /** Repeatable terminal raycast tensor descriptors returned by typed fetch. */
    typedef struct ovspatialquery_raycast_batch_result_t
    {
        ovspatialquery_batch_result_common_t   common;
        ovspatialquery_raycast_batch_columns_t columns;
    } ovspatialquery_raycast_batch_result_t;

    /** Repeatable terminal sphere-sweep descriptors returned by typed fetch. */
    typedef struct ovspatialquery_sphere_sweep_batch_result_t
    {
        ovspatialquery_batch_result_common_t        common;
        ovspatialquery_sphere_sweep_batch_columns_t columns;
    } ovspatialquery_sphere_sweep_batch_result_t;

    /** Box and sphere sweeps have identical device-neutral contact columns. */
    typedef ovspatialquery_sphere_sweep_batch_columns_t     ovspatialquery_box_sweep_batch_columns_t;
    typedef ovspatialquery_sphere_sweep_batch_reservation_t ovspatialquery_box_sweep_batch_reservation_t;
    typedef ovspatialquery_sphere_sweep_batch_result_t      ovspatialquery_box_sweep_batch_result_t;

    /** Repeatable terminal overlap tensor descriptors returned by typed fetch. */
    typedef struct ovspatialquery_overlap_batch_result_t
    {
        ovspatialquery_batch_result_common_t   common;
        ovspatialquery_overlap_batch_columns_t columns;
    } ovspatialquery_overlap_batch_result_t;

    /* -------------------------------------------------------------------------- */
    /* Asynchronous operation control                                             */
    /* -------------------------------------------------------------------------- */

    typedef enum ovspatialquery_op_state_t
    {
        /** Accepted work has not reached a terminal state. */
        OVSPATIALQUERY_OP_PENDING = 0,
        /** Work reached terminal success. */
        OVSPATIALQUERY_OP_COMPLETED = 1,
        /** Work reached terminal failure. */
        OVSPATIALQUERY_OP_FAILURE = 2,
    } ovspatialquery_op_state_t;

    /** Result of one wait or poll. `state` alone says pending, success, or failure. */
    typedef struct ovspatialquery_op_wait_result_t
    {
        ovspatialquery_op_state_t state; /* PENDING, COMPLETED, or FAILURE */
    } ovspatialquery_op_wait_result_t;

    /** Non-consuming operation status. Progress is in the closed range [0,1]. */
    typedef struct ovspatialquery_op_status_t
    {
        ovspatialquery_op_state_t state; /* PENDING, COMPLETED, or FAILURE */
        /**
         * Machine-readable terminal cause using `ovspatialquery_api_status_t`.
         * PENDING uses NOT_READY (7). COMPLETED uses SUCCESS (0). FAILURE uses
         * OPERATION_FAILED (17), OUT_OF_MEMORY (18), INTERNAL (19), or DEVICE_LOST
         * (20). HOST_BUDGET and DEVICE_BUDGET are admission failures: they issue
         * no operation ID and never occur here. TIMEOUT (2) describes only one
         * wait call and never occurs here.
         */
        ovspatialquery_api_status_t terminal_status;
        double                      progress;
    } ovspatialquery_op_status_t;

    /* -------------------------------------------------------------------------- */
    /* Library and diagnostic functions                                           */
    /* -------------------------------------------------------------------------- */

    /** Writes the implementation semantic version. Outputs are required. */
    OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_get_version(uint32_t* out_major, uint32_t* out_minor,
                                                                          uint32_t* out_patch);

    /**
     * Returns the calling thread's diagnostic for its last failed API call. After a
     * successful triangle-mesh registration or ovstage set or update that dropped
     * triangles without area, it names the first such meshes and triangles instead;
     * `ovspatialquery_get_dropped_triangle_count` and
     * `ovspatialquery_fetch_ovstage_dropped_triangles` report the counts.
     */
    OVSPATIALQUERY_API ovspatialquery_string_t ovspatialquery_get_last_error(void);

    /**
     * Returns the terminal diagnostic for one failed asynchronous operation.
     * `instance` must be the live instance that owns `op_id`, because operation IDs
     * are instance-scoped. The operation must still be live. A successful, pending,
     * invalid, or instance-mismatched operation has no operation-error text.
     */
    OVSPATIALQUERY_API ovspatialquery_string_t
    ovspatialquery_get_last_op_error(ovspatialquery_instance_handle_t instance, ovspatialquery_op_id_t op_id);

    /* -------------------------------------------------------------------------- */
    /* Instance and resource functions                                             */
    /* -------------------------------------------------------------------------- */

    /**
     * Creates an independent instance with the exact capacities in `desc`.
     * The library copies the descriptor arrays before return. On failure, no
     * instance is created. Calls on different instances may proceed independently.
     */
    OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_create_instance(
        const ovspatialquery_instance_desc_t* desc, ovspatialquery_instance_handle_t* out_instance);

    /**
     * Returns current instance-controlled bytes charged to one device budget.
     *
     * Use `{ kDLCPU, 0 }` for `host_byte_budget`, or one exact configured CUDA
     * device for its device budget. The result is one synchronized current total,
     * not a peak and not an allocation-category breakdown. This lets the caller
     * size a hard budget without exposing backend caches or acceleration
     * structures. An unconfigured device returns API_NOT_SUPPORTED
     * and preserves the output.
     */
    OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_get_instance_memory_usage(
        ovspatialquery_instance_handle_t instance, DLDevice device, uint64_t* out_resident_bytes);

    /**
     * Returns SUCCESS when `instance` can use `capability` on `device`, and
     * NOT_SUPPORTED, with a diagnostic naming the reason, when it cannot. It
     * admits no work, creates no object, and charges no budget, so a caller
     * asks before it has a population. The first question about a CUDA device
     * starts this library's CUDA backend on it, as the first CUDA plan would.
     * A key this library does not know returns NOT_SUPPORTED, so a program
     * built against a newer header still gets an answer.
     *
     * `{ kDLCPU, 0 }` has every plan and pair-reporter key, because CPU
     * execution of them is mandatory, and not PREPARE_ON_DEVICE. A CUDA device
     * needs this build's CUDA backend, a driver, that device, and an entry in
     * the instance's `device_budgets`. Answers do not change during the
     * instance's lifetime. SUCCESS reserves nothing: a request can still be
     * refused for its arguments, budgets, or selected version, or by a rule its
     * declaration states. Another device type, a CPU ID other than zero, or a
     * negative CUDA ID returns INVALID_ARGUMENT.
     */
    OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_query_capability(
        ovspatialquery_instance_handle_t instance, DLDevice device, ovspatialquery_capability_t capability);

    /**
     * Destroys an empty instance. Returns BUSY while any population, view,
     * geometry definition, plan, or operation remains live, or while an object
     * created through a companion header (a pair reporter) does.
     */
    OVSPATIALQUERY_API ovspatialquery_result_t
    ovspatialquery_destroy_instance(ovspatialquery_instance_handle_t instance);

    /* -------------------------------------------------------------------------- */
    /* Population, geometry, and view functions                                   */
    /* -------------------------------------------------------------------------- */

    /**
     * Reads current population state. A newly created population returns SUCCESS
     * with `ready == 0`; it has no valid head and queries return NOT_READY. The
     * first successful complete set changes `ready` to one atomically.
     */
    OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_get_population_info(
        ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population,
        ovspatialquery_population_info_t* out_info);

    /**
     * Destroys a population. Returns BUSY while a view, plan, reporter, admitted
     * operation, or other live object still retains the population or a version.
     */
    OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_destroy_population(
        ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population);

    /**
     * Fetches the repeatable result of one successful population publication.
     *
     * Use this after `wait_op` or `query_op_status` reports terminal success when
     * the caller needs the exact published ordinal, object count, or complete
     * exclusion list. Fetch does not consume the result or destroy the operation;
     * repeated fetches return the same operation-owned tensors without rebuilding them.
     * It returns NOT_READY while pending and OPERATION_FAILED after failed work.
     */
    OVSPATIALQUERY_API ovspatialquery_result_t
    ovspatialquery_fetch_population_update(ovspatialquery_instance_handle_t instance, ovspatialquery_op_id_t op_id,
                                           ovspatialquery_population_update_result_t* out_result);

    /**
     * Destroys a stable view. Returns BUSY while any operation, plan, reporter,
     * or other retained owner uses it.
     */
    OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_destroy_view(ovspatialquery_instance_handle_t instance,
                                                                           ovspatialquery_view_handle_t     view);

    /* -------------------------------------------------------------------------- */
    /* Synchronous query functions                                                */
    /* -------------------------------------------------------------------------- */

    /**
     * Runs one raycast on the caller thread and returns the closest hit, if any.
     * The selected immutable ordinal is reported even when there is no hit.
     */
    OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_raycast_closest_sync(
        ovspatialquery_instance_handle_t instance, const ovspatialquery_raycast_desc_t* desc,
        ovspatialquery_raycast_single_result_t* out_result);

    /**
     * Runs one raycast on the caller thread and returns any admitted hit, if any.
     * It makes no closest-hit or stable-choice promise.
     */
    OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_raycast_any_sync(
        ovspatialquery_instance_handle_t instance, const ovspatialquery_raycast_desc_t* desc,
        ovspatialquery_raycast_single_result_t* out_result);

    /**
     * Runs one raycast and writes up to `inout_hits->capacity` hit rows: one per
     * triangle crossing, or one per object reached with `desc->mesh_hits` NEAREST.
     * `out_result->stored_hit_count` says how many rows are valid. `truncated` is
     * nonzero when more hits existed, and the stored rows are then the nearest by
     * distance, then object ID (one object's mesh crossings at exactly the cut-off
     * distance are kept in either order). Returned row order is unspecified.
     */
    OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_raycast_multiple_sync(
        ovspatialquery_instance_handle_t instance, const ovspatialquery_raycast_desc_t* desc,
        ovspatialquery_raycast_hit_buffer_t* inout_hits, ovspatialquery_multiple_result_t* out_result);

    /**
     * Runs one moving-sphere query and returns its closest hit, if any. The legacy
     * verb order `sweep_sphere` means the same sphere-sweep family named by the
     * descriptors and batch-plan functions; it does not add another sweep shape.
     */
    OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_sweep_sphere_closest_sync(
        ovspatialquery_instance_handle_t instance, const ovspatialquery_sphere_sweep_desc_t* desc,
        ovspatialquery_sphere_sweep_closest_result_t* out_result);

    /**
     * Runs one moving-sphere query and returns a hit of any object it reaches, if
     * any: that object's first contact, as the closest sweep reports it. It makes
     * no closest-hit or stable-choice promise.
     */
    OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_sweep_sphere_any_sync(
        ovspatialquery_instance_handle_t instance, const ovspatialquery_sphere_sweep_desc_t* desc,
        ovspatialquery_sphere_sweep_closest_result_t* out_result);

    /**
     * Runs one moving-sphere query and writes up to `inout_hits->capacity` rows,
     * one per object reached. `out_result->stored_hit_count` says how many rows
     * are valid. `truncated` is nonzero when more objects were reached, and the
     * stored rows are then the nearest by distance, then object ID. Returned row
     * order is unspecified.
     */
    OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_sweep_sphere_multiple_sync(
        ovspatialquery_instance_handle_t instance, const ovspatialquery_sphere_sweep_desc_t* desc,
        ovspatialquery_sphere_sweep_hit_buffer_t* inout_hits, ovspatialquery_multiple_result_t* out_result);

    /** Runs one fixed-orientation box sweep and returns its closest hit, if any. */
    OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_sweep_box_closest_sync(
        ovspatialquery_instance_handle_t instance, const ovspatialquery_box_sweep_desc_t* desc,
        ovspatialquery_box_sweep_closest_result_t* out_result);

    /** Runs one box sweep and returns a hit of any object it reaches, as the sphere sweep does. */
    OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_sweep_box_any_sync(
        ovspatialquery_instance_handle_t instance, const ovspatialquery_box_sweep_desc_t* desc,
        ovspatialquery_box_sweep_closest_result_t* out_result);

    /** Runs one box sweep and writes one row per object reached, as the sphere sweep does. */
    OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_sweep_box_multiple_sync(
        ovspatialquery_instance_handle_t instance, const ovspatialquery_box_sweep_desc_t* desc,
        ovspatialquery_box_sweep_hit_buffer_t* inout_hits, ovspatialquery_multiple_result_t* out_result);

    /** Runs one overlap and returns any overlapping object, if one exists. */
    OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_overlap_any_sync(
        ovspatialquery_instance_handle_t instance, const ovspatialquery_overlap_desc_t* desc,
        ovspatialquery_overlap_any_result_t* out_result);

    /**
     * Runs one overlap and writes bounded object/geometry identities.
     * It uses the same `ovspatialquery_multiple_result_t` count/truncation contract
     * as multiple raycast, while `inout_hits` supplies overlap-specific rows.
     */
    OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_overlap_multiple_sync(
        ovspatialquery_instance_handle_t instance, const ovspatialquery_overlap_desc_t* desc,
        ovspatialquery_overlap_hit_buffer_t* inout_hits, ovspatialquery_multiple_result_t* out_result);

    /* -------------------------------------------------------------------------- */
    /* Reusable batch-query functions                                             */
    /* -------------------------------------------------------------------------- */

    /**
     * Batch workflow (error handling omitted):
     *
     *     create_*_plan(instance, &plan_desc, &plan);
     *     queued = enqueue_*_plan(instance, plan, &batch, &reservation);
     *     wait_op(instance, queued.op_id, timeout, &wait_result);
     *     fetch_*_batch(instance, queued.op_id, &result);
     *     // Read result.columns. CUDA consumers first wait on result_ready.
     *     destroy_op(instance, &queued.op_id);
     *     destroy_batch_plan(instance, plan);
     *
     * A plan fixes one family, target, device, capacities, and result slots. Each
     * launch selects an ordinal and supplies rows only for that family. Each
     * result uses the same semantic DLPack columns on CPU and CUDA. Tensor device
     * metadata identifies where the column lives.
     *
     * A successful enqueue writes one immutable reservation snapshot and never
     * retains or later modifies that caller-owned struct. While pending, callers
     * may inspect reservation metadata and storage descriptors but may not read
     * CPU contents. CUDA callers may enqueue downstream work from the reservation
     * tensors and event without a host wait. Typed fetch returns terminal tensor
     * shapes on either device.
     * Fetch is repeatable and non-consuming; it returns NOT_READY while pending.
     * It returns OPERATION_FAILED after terminal failure and preserves its output,
     * because a failed operation has no fetched payload result. After recoverable
     * CUDA failure, only the documented safe empty reservation hit ranges are
     * readable.
     */

    /** Creates a reusable raycast plan and reserves its fixed result slots. */
    OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_create_raycast_plan(
        ovspatialquery_instance_handle_t instance, const ovspatialquery_raycast_plan_desc_t* desc,
        ovspatialquery_batch_plan_handle_t* out_plan);

    /**
     * Admits one raycast launch and returns its immutable reservation facts.
     */
    OVSPATIALQUERY_API ovspatialquery_enqueue_result_t ovspatialquery_enqueue_raycast_plan(
        ovspatialquery_instance_handle_t instance, ovspatialquery_batch_plan_handle_t plan,
        const ovspatialquery_raycast_batch_t* batch, ovspatialquery_raycast_batch_reservation_t* out_reservation);

    /** Fetches the repeatable terminal result views of one raycast-plan launch. */
    OVSPATIALQUERY_API ovspatialquery_result_t
    ovspatialquery_fetch_raycast_batch(ovspatialquery_instance_handle_t instance, ovspatialquery_op_id_t op_id,
                                       ovspatialquery_raycast_batch_result_t* out_result);

    /** Creates a reusable sphere-sweep plan and reserves its fixed result slots. */
    OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_create_sphere_sweep_plan(
        ovspatialquery_instance_handle_t instance, const ovspatialquery_sphere_sweep_plan_desc_t* desc,
        ovspatialquery_batch_plan_handle_t* out_plan);

    /** Admits one sphere-sweep launch and returns its immutable reservation facts. */
    OVSPATIALQUERY_API ovspatialquery_enqueue_result_t ovspatialquery_enqueue_sphere_sweep_plan(
        ovspatialquery_instance_handle_t instance, ovspatialquery_batch_plan_handle_t plan,
        const ovspatialquery_sphere_sweep_batch_t*       batch,
        ovspatialquery_sphere_sweep_batch_reservation_t* out_reservation);

    /** Fetches the repeatable terminal result views of one sphere-sweep launch. */
    OVSPATIALQUERY_API ovspatialquery_result_t
    ovspatialquery_fetch_sphere_sweep_batch(ovspatialquery_instance_handle_t instance, ovspatialquery_op_id_t op_id,
                                            ovspatialquery_sphere_sweep_batch_result_t* out_result);

    /** Creates a reusable box-sweep plan and reserves its fixed result slots. */
    OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_create_box_sweep_plan(
        ovspatialquery_instance_handle_t instance, const ovspatialquery_box_sweep_plan_desc_t* desc,
        ovspatialquery_batch_plan_handle_t* out_plan);

    /** Admits a box-sweep launch with the same reservation lifecycle as sphere sweeps. */
    OVSPATIALQUERY_API ovspatialquery_enqueue_result_t ovspatialquery_enqueue_box_sweep_plan(
        ovspatialquery_instance_handle_t instance, ovspatialquery_batch_plan_handle_t plan,
        const ovspatialquery_box_sweep_batch_t* batch, ovspatialquery_box_sweep_batch_reservation_t* out_reservation);

    /** Fetches repeatable terminal box-sweep contact columns. */
    OVSPATIALQUERY_API ovspatialquery_result_t
    ovspatialquery_fetch_box_sweep_batch(ovspatialquery_instance_handle_t instance, ovspatialquery_op_id_t op_id,
                                         ovspatialquery_box_sweep_batch_result_t* out_result);

    /** Creates a reusable overlap plan and reserves its fixed result slots. */
    OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_create_overlap_plan(
        ovspatialquery_instance_handle_t instance, const ovspatialquery_overlap_plan_desc_t* desc,
        ovspatialquery_batch_plan_handle_t* out_plan);

    /** Admits one overlap launch and returns its immutable reservation facts. */
    OVSPATIALQUERY_API ovspatialquery_enqueue_result_t ovspatialquery_enqueue_overlap_plan(
        ovspatialquery_instance_handle_t instance, ovspatialquery_batch_plan_handle_t plan,
        const ovspatialquery_overlap_batch_t* batch, ovspatialquery_overlap_batch_reservation_t* out_reservation);

    /** Fetches the repeatable terminal result views of one overlap-plan launch. */
    OVSPATIALQUERY_API ovspatialquery_result_t
    ovspatialquery_fetch_overlap_batch(ovspatialquery_instance_handle_t instance, ovspatialquery_op_id_t op_id,
                                       ovspatialquery_overlap_batch_result_t* out_result);

    /** Destroys an idle plan; BUSY means an operation still owns a result slot. */
    OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_destroy_batch_plan(
        ovspatialquery_instance_handle_t instance, ovspatialquery_batch_plan_handle_t plan);

    /* -------------------------------------------------------------------------- */
    /* Asynchronous operation functions                                           */
    /* -------------------------------------------------------------------------- */

    /**
     * Waits until an operation is terminal or `timeout` expires. Zero polls;
     * TIMEOUT means this call observed pending work, not operation failure.
     * TIMEOUT_INFINITE waits until a terminal state. A timeout is not cancellation:
     * the operation and all input-lifetime obligations remain live. A caller CUDA
     * wait event must eventually resolve; a permanently unsignaled event violates
     * the caller contract. Detected device loss ends the affected work with a
     * terminal DEVICE_LOST failure.
     *
     * On either terminal state this function returns SUCCESS and writes the state;
     * query_op_status supplies the machine-readable terminal cause. Returning a
     * terminal state marks it observed, which is required before destruction.
     */
    OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_wait_op(ovspatialquery_instance_handle_t instance,
                                                                      ovspatialquery_op_id_t           op_id,
                                                                      ovspatialquery_timeout_t         timeout,
                                                                      ovspatialquery_op_wait_result_t* out_wait);

    /**
     * Polls operation state and approximate progress without consuming it.
     * Returning a terminal state marks that terminal state observed. A typed fetch
     * also observes a terminal state when it returns success or OPERATION_FAILED;
     * a pending fetch that returns NOT_READY does not.
     */
    OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_query_op_status(ovspatialquery_instance_handle_t instance,
                                                                              ovspatialquery_op_id_t           op_id,
                                                                              ovspatialquery_op_status_t* out_status);

    /**
     * Destroys a terminal-observed CPU operation or terminal-observed operation
     * with no returned CUDA views. Pending or terminal-unobserved work returns
     * BUSY. Success writes INVALID_OP and releases borrowed inputs/results. Use
     * `destroy_op_after_cuda` for every CUDA operation with returned device views,
     * whether it ended in success or a recoverable failure. A HOST_MAPPED plan's
     * launch may use either: this function when no device work the caller
     * queued still reads its columns, because a terminal state implies the
     * library's own device work is done.
     */
    OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_destroy_op(ovspatialquery_instance_handle_t instance,
                                                                         ovspatialquery_op_id_t*          inout_op_id);

    /**
     * Retires a terminal-observed CUDA operation after consumers.
     *
     * `inout_consumer_done` is one caller-owned event covering every consumer of
     * the result views. Success transfers and zeroes the event, invalidates the op
     * ID, and lets the implementation release result storage when device-safe.
     * A recoverable terminal batch failure still signals `result_ready` after it
     * writes the safe empty hit-range state documented above. Consumers can
     * observe that state and safely join `consumer_done` before storage retirement.
     * `OVSPATIALQUERY_API_DEVICE_LOST` is terminal, but does not promise usable
     * CUDA result storage, a ready event, or normal consumer-event retirement.
     * CPU operations, pending operations, and terminal states not yet observed
     * through wait, status, or typed fetch are invalid.
     */
    OVSPATIALQUERY_API ovspatialquery_result_t
    ovspatialquery_destroy_op_after_cuda(ovspatialquery_instance_handle_t instance, ovspatialquery_op_id_t* inout_op_id,
                                         ovspatialquery_cuda_event_t* inout_consumer_done);

#ifdef __cplusplus
}

#include <cstddef>
#include <cstdint>
#include <functional>
#include <type_traits>

/*
 * Same-kind equality, raw-ID ordering, and hashing for a handle struct, as the
 * shared Omniverse handle types define them. Ordering is by raw_id, not by
 * creation or content. Invoke at global scope after the struct definition.
 */
#define OVSPATIALQUERY_DETAIL_HANDLE_OPERATORS(Type)                                                                   \
    static_assert(sizeof(Type) == sizeof(uint64_t), #Type " must match uint64_t size");                                \
    static_assert(std::is_trivially_copyable<Type>::value, #Type " must be trivially copyable");                       \
    inline constexpr bool operator==(Type a, Type b)                                                                   \
    {                                                                                                                  \
        return a.raw_id == b.raw_id;                                                                                   \
    }                                                                                                                  \
    inline constexpr bool operator!=(Type a, Type b)                                                                   \
    {                                                                                                                  \
        return !(a == b);                                                                                              \
    }                                                                                                                  \
    inline constexpr bool operator<(Type a, Type b)                                                                    \
    {                                                                                                                  \
        return a.raw_id < b.raw_id;                                                                                    \
    }                                                                                                                  \
    inline constexpr bool operator>(Type a, Type b)                                                                    \
    {                                                                                                                  \
        return b < a;                                                                                                  \
    }                                                                                                                  \
    inline constexpr bool operator<=(Type a, Type b)                                                                   \
    {                                                                                                                  \
        return !(b < a);                                                                                               \
    }                                                                                                                  \
    inline constexpr bool operator>=(Type a, Type b)                                                                   \
    {                                                                                                                  \
        return !(a < b);                                                                                               \
    }                                                                                                                  \
    namespace std                                                                                                      \
    {                                                                                                                  \
    template <>                                                                                                        \
    struct hash<Type>                                                                                                  \
    {                                                                                                                  \
        size_t operator()(Type handle) const noexcept                                                                  \
        {                                                                                                              \
            return hash<uint64_t>{}(handle.raw_id);                                                                    \
        }                                                                                                              \
    };                                                                                                                 \
    }

extern "C++"
{
    OVSPATIALQUERY_DETAIL_HANDLE_OPERATORS(ovspatialquery_op_id_t)
    OVSPATIALQUERY_DETAIL_HANDLE_OPERATORS(ovspatialquery_instance_handle_t)
    OVSPATIALQUERY_DETAIL_HANDLE_OPERATORS(ovspatialquery_population_handle_t)
    OVSPATIALQUERY_DETAIL_HANDLE_OPERATORS(ovspatialquery_view_handle_t)
    OVSPATIALQUERY_DETAIL_HANDLE_OPERATORS(ovspatialquery_batch_plan_handle_t)
}
#endif

#endif /* OVSPATIALQUERY_H */
