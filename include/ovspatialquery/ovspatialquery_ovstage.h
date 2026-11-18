/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_OVSTAGE_H
#define OVSPATIALQUERY_OVSTAGE_H

#include <ovspatialquery/ovspatialquery.h>
#include <ovstage/ovstage_api/ovstage_api.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @file
     * @brief Preferred ovstage source adapter.
     *
     * The adapter reads sphere, box, capsule, convex, and triangle-mesh prims
     * from a sealed ovstage stage and publishes them into the same population
     * and query model every source uses; ovstage types do not appear in the core
     * header. One stage may feed several populations through different filters,
     * each publishing on its own clock. There is no custom (USER) geometry path
     * from ovstage, and the adapter calls no application callbacks.
     *
     * Typical lifetime (status checks omitted):
     *
     *     // `stage` is already sealed. Keep it and its path dictionary alive.
     *     ovspatialquery_ovstage_population_desc_t source = {
     *         .stage = stage,
     *         .filter = robot_filter,
     *         .common = { .retained_ordinal_capacity = 4 },
     *     };
     *     ovspatialquery_create_ovstage_population(instance, &source, &robots);
     *
     *     // Reads sealed ovstage ordinal 100 and publishes it with the same value.
     *     op = ovspatialquery_enqueue_ovstage_population_set(instance, robots, 100);
     *     ovspatialquery_wait_op(instance, op.op_id,
     *                            OVSPATIALQUERY_TIMEOUT_INFINITE, &wait_result);
     *     ovspatialquery_destroy_op(instance, &op.op_id);
     *
     *     // Queries now select head 100 or an exact retained ordinal.
     *     ovspatialquery_raycast_closest_sync(instance, &raycast, &hit);
     *
     *     // After sealing source ordinal 104, publish a complete new snapshot.
     *     op = ovspatialquery_enqueue_ovstage_population_update(
     *         instance, robots, 100, 104);
     *     // Wait, then destroy the operation.
     *
     *     // Destroy every other dependent plan/reporter/operation first.
     *     ovspatialquery_destroy_population(instance, robots);
     *
     * Each publication captures one sealed ordinal synchronously inside the
     * enqueue call and prepares that copy asynchronously, so the caller seals the
     * ordinal first and keeps filter-relevant ovstage writes quiescent until the
     * call returns; preparation never rereads the stage. An object is a prim
     * path: its object ID's `value` is that path's `ovx_primpath_t` in the
     * stage's path dictionary, which the caller keeps alive until population
     * destruction. A prim authored again at a path that left the population
     * therefore continues the same object, and a pair reporter advancing across
     * the gap reports it exited and re-entered. Populations whose stages return
     * the same `ovstage_get_path_dictionary` report the same ID for the same
     * prim, so a caller merging their hits compares IDs directly. Populations
     * over one stage always do; separate stages may not, because ovstage may
     * give each stage its own dictionary.
     *
     * A hit's material (`ovspatialquery_hit_identity_t`) is a path value in the
     * same dictionary: the first target of `material:binding:physics` on the
     * prim or its nearest ancestor that has one, else of `material:binding` the
     * same way, else zero, as omni.physx resolves a collider's physics material.
     * An exact contact on a mesh face reports the binding of a face GeomSubset
     * child of the `materialBind` family that lists the face, its own
     * `material:binding:physics`, or, when the mesh has no physics binding, its
     * own `material:binding`; overlapping subsets resolve in path order, the
     * later winning. The adapter reads the bindings at the sealed ordinal of each
     * host-prepared publication, so a changed binding appears in the next one.
     * Collection bindings, binding strength, and the bindings of a native
     * instance's prototype are not read. A device-prepared publication
     * (ON_DEVICE or ON_DEVICE_HOST_READABLE) reports zero.
     *
     * The types this header takes from ovstage (`ovstage_instance_t`,
     * `ovx_primpath_t`, and `ovstage_filter_t`, which both descriptors embed by
     * value) have the layout of the ovstage releases `stage` accepts. If ovstage
     * changes one of them, the descriptors here change in the ovspatialquery
     * release that adopts that ovstage release, under the core's evolution rule,
     * and no other ovspatialquery header changes. This header is compiled
     * against the ovstage 0.4 headers, where `ovx_primpath_t` and `ovx_token_t`
     * are one-member structs with the size and passing of 0.3's integers.
     */

    /** Per-prim content reasons reported by a successful publication. */
    typedef enum ovspatialquery_ovstage_exclusion_reason_t
    {
        /** No _worldExtent row, or the row contains the missing-bounds value. */
        OVSPATIALQUERY_OVSTAGE_EXCLUSION_NO_SPATIAL_SOURCE = 1,
        /** The selected prim's type has no supported geometry family. */
        OVSPATIALQUERY_OVSTAGE_EXCLUSION_UNSUPPORTED_TYPE = 2,
        /** A Sphere or Capsule has non-uniform scale or shear outside the tolerance below. */
        OVSPATIALQUERY_OVSTAGE_EXCLUSION_UNSUPPORTED_SHAPE = 3,
        /** Invalid topology, points, or transform, including a zero determinant. */
        OVSPATIALQUERY_OVSTAGE_EXCLUSION_MALFORMED_GEOMETRY = 4,
        /** Experimental: a device-prepared shape change awaits host rebuilding at the next publication. */
        OVSPATIALQUERY_OVSTAGE_EXCLUSION_RESHAPE_PENDING = 5,
    } ovspatialquery_ovstage_exclusion_reason_t;

    /**
     * Narrows an ovstage population or view by prim path, so a caller can select
     * everything under a root except some subtrees, which an ovstage filter cannot
     * express. Walking from a prim up through its ancestors, the nearest path
     * listed in either list decides: an include admits it, an exclude removes it,
     * and at equal depth the exclude wins. Paths compare by element, so
     * `/World/Env` does not cover `/World/Environment`; an include below an
     * exclude re-includes that subtree. An empty include list stands for `/`, so a
     * zero-initialized scope selects everything the filter selects. Entries are
     * nonzero path values from the population's path dictionary, `/` included;
     * each pointer is null exactly when its count is zero. Creation validates,
     * deduplicates, and deep-copies both lists, charged to the host budget, and
     * returns INVALID_ARGUMENT for an entry the dictionary cannot decompose. A
     * prim outside the scope is not selected, as if the filter had rejected it,
     * so it is never reported as an exclusion and its attributes are not read.
     * The scope is fixed for the life of its population or view.
     */
    typedef struct ovspatialquery_ovstage_scope_t
    {
        const ovx_primpath_t* include_paths; /* include_count entries */
        const ovx_primpath_t* exclude_paths; /* exclude_count entries */
        uint32_t              include_count;
        uint32_t              exclude_count;
    } ovspatialquery_ovstage_scope_t;

    typedef struct ovspatialquery_ovstage_population_desc_t
    {
        /**
         * Sealed source read by publication calls; it outlives the population. Its
         * runtime version must be a 0.3 or 0.4 ovstage release, any patch;
         * another version returns NOT_SUPPORTED at creation. A 0.4 release orders
         * ordinals as int64; pre-release 0.4 builds that kept 0.3's uint64
         * ordinals report the same version and are not supported. Convex hulls and
         * physics materials are read from the prims' `physics` and
         * `material:binding` properties: ovstage 0.3's PHYSICS domain populates
         * them, and on 0.4, which has no such domain, the caller populates them
         * through a selector; without them a mesh publishes as a triangle mesh
         * and reports no physics material. A convex hull's vertex limit is read
         * from its `physxConvexHullCollision` properties, which a 0.4 selector
         * populates the same way; without them the limit is 64.
         */
        ovstage_instance_t* stage;
        /**
         * Selects the stage prims in this population; deep-copied at creation. The
         * filter language is ovstage's own, an AND of attribute predicates, and this
         * adapter defines no selection vocabulary of its own except for the empty
         * filter. ovstage's empty filter matches every prim, including the pseudo-root,
         * Xforms, Scopes, and ovstage's own bookkeeping prims, which no population can
         * publish. The adapter's empty filter (`count` zero) therefore selects every
         * prim whose `usd-prim-type` is Sphere, Cube, Capsule, or Mesh, or the
         * matching instance-proxy type (CubeInstance and so on): ovstage writes one
         * row per proxy of a native instance, at its own path and world matrix. An
         * explicit filter selects proxies in ovstage's terms, by `usd-path` or an
         * instance type; `usd-prim-type IN [Cube]` names authored cubes only. Every
         * filter drops, uncounted, the prims that only define what instances draw:
         * ovstage's prototype copies (prims at or under a `PrototypeRootAPI` root),
         * and a PointInstancer's prototype prims and RENDERING's copies of them
         * (prims at or under one of its prototype targets, or carrying
         * `_instancerPath`). No point instance is published. Only RENDERING (or
         * ALL) writes proxy rows, so on a stage without any a matched
         * native-instancing copy makes set and update return NOT_SUPPORTED instead
         * of publishing the stage without its instances.
         * ovstage evaluates the filter at each publication, except that a filter of exactly
         * one `usd-path IN` predicate names its prims: the adapter interns those paths
         * once through the stage's path dictionary and each publication reads the
         * prims that exist at them, which selects the same prims as ovstage's
         * evaluation of that predicate without its per-path cost. A selected prim
         * of another type is excluded as UNSUPPORTED_TYPE; a supported prim without
         * `_worldExtent` state or with missing bounds is NO_SPATIAL_SOURCE.
         */
        ovstage_filter_t filter;
        /**
         * Prims the population may hold, applied together with `filter`. The
         * adapter keeps each selected prim's answer, charged to the host budget.
         *
         * Implementation note: it evaluates a prim's path once while the prim
         * stays selected, so a publication pays only for prims it has not seen.
         */
        ovspatialquery_ovstage_scope_t scope;
        /** Configures how many prior population heads remain selectable. */
        ovspatialquery_population_desc_t common;
    } ovspatialquery_ovstage_population_desc_t;

    typedef struct ovspatialquery_ovstage_view_desc_t
    {
        /** Stable subset filter, deep-copied at view creation and resolved like a population's, empty included. */
        ovstage_filter_t filter;
        /** Narrows the view's selection from its parent's objects, evaluated like a population's scope. */
        ovspatialquery_ovstage_scope_t scope;
    } ovspatialquery_ovstage_view_desc_t;

    /**
     * Provisional per-path categories captured by future ovstage publications.
     *
     * An object whose prim path is listed takes that entry's bits; every other
     * object takes `default_bits`, so a listed path can also drop a default bit.
     * Paths are nonzero values from the population's path dictionary and appear
     * at most once (INVALID_ARGUMENT otherwise); order is free. `paths` and `bits`
     * are null exactly when `count` is zero. A listed path that is absent at a
     * publication contributes nothing and still counts against the host budget.
     * No installed table means all 128 bits set for every object.
     */
    typedef struct ovspatialquery_ovstage_participation_table_t
    {
        ovspatialquery_participation_bits_t        default_bits;
        const ovx_primpath_t*                      paths; /* count entries */
        const ovspatialquery_participation_bits_t* bits;  /* count entries */
        uint32_t                                   count;
    } ovspatialquery_ovstage_participation_table_t;

    /**
     * Creates a NOT_READY population with no head and deep-copies the filter.
     * The caller keeps `stage` and its path dictionary alive until the population
     * is destroyed, because object IDs are values in that dictionary. The adapter
     * assigns object IDs from prim paths and instance-wide geometry references;
     * the caller does not create those values.
     */
    OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_create_ovstage_population(
        ovspatialquery_instance_handle_t instance, const ovspatialquery_ovstage_population_desc_t* desc,
        ovspatialquery_population_handle_t* out_population);

    /**
     * Creates a filter view. It becomes ready on the next successful parent
     * publication and then shares all parent versions and acceleration data.
     */
    OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_create_ovstage_view(
        ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population,
        const ovspatialquery_ovstage_view_desc_t* desc, ovspatialquery_view_handle_t* out_view);

    /**
     * Atomically replaces the participation table used by future publications.
     *
     * `population` must have been created by `create_ovstage_population`, because
     * the table is a feature of this adapter alone. The call validates and
     * deep-copies a non-null `table` before success, so the caller can release its
     * arrays after return. Passing null clears the table and restores
     * all bits set for every object in future publications.
     *
     * The function returns BUSY while a set/update operation is pending; otherwise
     * successful return is the linearization point, and concurrent setters and
     * publication enqueues on the same population are serialized. Failure returns
     * INVALID_ARGUMENT, INVALID_HANDLE, BUSY, HOST_BUDGET, OUT_OF_MEMORY, or
     * INTERNAL and preserves the installed table. Its storage is charged to the
     * instance host budget.
     *
     * The call does not change the current head, retained versions, admitted work,
     * or pair-reporter baselines. Each later successfully admitted ovstage set or
     * update captures one complete installed table, looks up each object's path,
     * and publishes the resulting bit column atomically with
     * geometry, poses, bounds, object/view membership, the new head, and
     * retention; a later table replacement cannot affect that work.
     *
     * Exact retained queries therefore keep the bits stored at their publication.
     * Pair reporters observe bit-caused enters/exits only after advancing to a
     * later published ordinal; same-ordinal results never change. Reporter capacity
     * failure still preserves its accepted baseline.
     *
     * A category-only change becomes visible only after a later sealed ovstage
     * ordinal is published. Immediate between-ordinal mutation is not supported
     * because it would require a second public version system. The implementation
     * may share unchanged geometry and acceleration data; all table storage and
     * publication materialization remain subject to hard budgets.
     */
    OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_set_ovstage_participation_map(
        ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population,
        const ovspatialquery_ovstage_participation_table_t* table);

    /**
     * Captures one sealed source snapshot before successful admission returns.
     * Capture runs synchronously inside this call on the calling thread: before
     * returning, it copies the filtered subset of geometry definitions, instance
     * bindings, paths, bounds, and one complete installed participation table out of
     * ovstage, which is why the caller keeps filter-relevant ovstage writes
     * quiescent until the call returns. Only preparation runs asynchronously, on
     * that copy. This first complete set may use any nonnegative ovstage ordinal,
     * including zero. `target_ordinal` must equal the ordinal read from ovstage
     * because the adapter does not translate or renumber it: capturing ovstage
     * ordinal 104 publishes population ordinal 104 and query results report 104.
     * A successful operation publishes atomically; failure leaves the population
     * NOT_READY. Per-prim content never fails a publication: the adapter omits the
     * prim and reports its object ID and reason in the complete sorted exclusion
     * list. Caller/configuration errors, unsupported runtimes, missing instance
     * proxy rows, budgets, and device loss still fail the operation or admission.
     *
     * An instance proxy takes its family from `_protoPrimType` and its definition
     * from `_protoPath`. Proxies of a prototype share geometry for transforms
     * differing by a rigid motion and positive uniform scale, separately for each
     * determinant sign. Other transforms share only a bitwise-equal shape metric.
     *
     * The adapter splits each prim's linear world block L into a placement, a
     * proper rotation and a positive uniform scale s, and a shape K that keeps the
     * surface t + L p exactly, so no world matrix is approximated; only a
     * Sphere's or Capsule's radius, below, may grow within the tolerance to
     * contain it.
     * Nonfinite/out-of-domain input, zero determinant, malformed topology, and
     * invalid hulls are MALFORMED_GEOMETRY. Whether K is uniform, mirrored,
     * axis-aligned, or sheared is decided within one small relative tolerance,
     * the same on host and device, with no absolute scale epsilon. Families use
     * the split as follows:
     *
     *   Family                 Uniform/mirror    Non-uniform axes    Shear
     *   Mesh or convexHull     apply K           apply K             apply K
     *   Cube                   BOX               BOX, scaled extents CONVEX corners
     *   Sphere or Capsule      exact             UNSUPPORTED_SHAPE   UNSUPPORTED_SHAPE
     *
     * Within tolerance a Sphere uses the largest singular value as its scale;
     * a Capsule keeps its segment exactly, including the direction of K times its
     * authored axis, and uses the largest singular value of K for its local radius.
     * The stored Capsule contains the authored surface; its outward deviation is
     * bounded by (sigma_max - sigma_min) times the authored radius. A
     * transform-only edit that moves a prim rigidly, within the same tolerance,
     * keeps its prepared shape.
     *
     * Implementation note: the split is the polar factorization L = R' P, with
     * P = sqrt(L^T L), s = cbrt(abs(det L)), R = R' and K = P/s for a positive
     * determinant, R = -R' and K = -P/s otherwise. The tolerance is tau = 2^-20:
     * K counts as +/-I when max|L^T L/s^2 - I| <= 2*tau, and axes count as
     * orthogonal when each off-diagonal entry of L^T L divided by its two axis
     * lengths is within 2*tau. A prepared shape is retained across a
     * transform-only edit when A = L K^-1/s has max|A^T A-I| <= 2*tau and
     * det(A) > 0; its pose rotation is the normalized quaternion of A.
     *
     * A Mesh with PhysicsMeshCollisionAPI and `physics:approximation =
     * "convexHull"` is CONVEX geometry: the backend registers its authored points
     * as the explicit convex table does under a `hull_vertex_limit`, so queries
     * answer for the convex a PhysX simulation of the stage collides with. The
     * limit is `physxConvexHullCollision:hullVertexLimit` when the stage carries
     * it and 64, PhysxSchema's default, otherwise; a negative value counts as
     * unauthored and the rest is clamped to 8 to 64, as that simulation reads it.
     * Fewer than four non-coplanar points, or points PhysX cooks no valid hull
     * for, are excluded as MALFORMED_GEOMETRY. Backends may produce slightly
     * different valid hulls, so CONVEX contacts are not bit-identical across
     * devices. Other approximation values remain unspecified.
     *
     * Triangle meshes use authored faces, ignoring `subdivisionScheme`. The
     * adapter skips `holeIndices` faces and reverses each triangle for authored
     * `leftHanded` orientation and again for a mirrored K, so culling and normals
     * follow the authored front face. Hits preserve authored face and corner
     * ordinals. A triangle without area, judged as for explicit triangle meshes on
     * the authored points with K applied, is dropped and the rest of the Mesh
     * published; each set or update judges the points it reads, so a triangle can
     * leave or return as they move, and lists the Meshes it read that dropped
     * triangles, with their counts, for `ovspatialquery_fetch_ovstage_dropped_triangles`;
     * the calling thread's diagnostic also names the first few. All-hole meshes, and meshes left with no triangle, are
     * MALFORMED_GEOMETRY. `doubleSided` is immutable mesh-definition state. A publication that reads or derives a mesh,
     * or reads a convex hull, charges the query structure and build scratch of that mesh or of the hull's triangle
     * surface at admission, as `ovspatialquery_enqueue_triangle_mesh_registration` describes, and returns HOST_BUDGET
     * when they do not fit.
     */
    OVSPATIALQUERY_API ovspatialquery_enqueue_result_t ovspatialquery_enqueue_ovstage_population_set(
        ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population,
        ovspatialquery_ordinal_t target_ordinal);

    /**
     * Reacquires one complete target snapshot with the same synchronous capture
     * as `enqueue_ovstage_population_set`. `base_ordinal` must equal the current
     * head and the target must strictly advance it; skipped source ordinals are
     * not replayed. The call captures the complete installed participation table for
     * `target_ordinal`. Publication fails atomically if budgets cannot preserve
     * the configured latest N prior heads. Per-prim content is excluded under the
     * set rules, so invalid content can advance the head with fewer objects; a
     * caller that requires a clean version can retain its previous exact ordinal.
     */
    OVSPATIALQUERY_API ovspatialquery_enqueue_result_t ovspatialquery_enqueue_ovstage_population_update(
        ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population,
        ovspatialquery_ordinal_t base_ordinal, ovspatialquery_ordinal_t target_ordinal);

    /**
     * `enqueue_ovstage_population_set` with a choice of where the version is
     * prepared. A null `execution`, or ON_HOST on a CPU device with zero
     * `cuda_sync.has_stream` and `cuda_sync.wait_event`, is that call exactly;
     * ON_HOST on a CUDA device is INVALID_ARGUMENT, because the host path reads
     * nothing on a stream.
     *
     * ON_DEVICE and ON_DEVICE_HOST_READABLE prepare the version on the CUDA
     * device `execution->device` and need a stream (`cuda_sync.has_stream` 1,
     * any stream value, the legacy default stream 0 included). They
     * read poses through ovstage's experimental placement extension, at the
     * extension version this library was built against; a stage without it
     * (a build serving another extension version) returns NOT_SUPPORTED before admission, and the library never
     * prepares such a publication on the host instead. A population with an ovstage view returns NOT_SUPPORTED too,
     * because no view can select a device version.
     *
     * Capture stays synchronous. Inside the call the adapter reads on the host,
     * at the sealed target ordinal, the prims the filter matches bucket by
     * bucket and their definition attributes, those of the prototype prims its
     * instance proxies name included, and compares them with the previous
     * device publication. Only when they differ, or a participation
     * table was installed since, does it reread identities and definitions as
     * the host path does and publish the new membership. It then enqueues one
     * device read of `omni:fabric:worldMatrix` (and of `worldMatrix` for prims
     * without one) on `cuda_sync.stream`, of the whole snapshot at the target
     * for an update as for a set, because ovstage keeps no change history for
     * some pose writers (`update_from_usd_time`); the host never reads a pose.
     * Device kernels validate the poses and prepare the version behind that read
     * and a nonzero `cuda_sync.wait_event`. An ovstage write of the world-matrix
     * column (`compute_hierarchy`) issued before that device work has read the
     * poses may wait for it.
     *
     * Implementation note: the library holds ovstage's read groups until its
     * kernels have consumed them.
     *
     * Both host- and device-prepared versions derive world bounds from each
     * prim's geometry and world matrix. `_worldExtent` decides only which prims
     * have a spatial source; a stale extent does not change the published box.
     * Device rows never fail publication for content.
     * Invalid matrices become MALFORMED_GEOMETRY; a Sphere or Capsule outside
     * its exact transform family becomes UNSUPPORTED_SHAPE and returns when
     * its transform is supported again. A BOX takes its current axis extents
     * every publication. Other shapes keep their prepared shape only across a
     * rigid change of their transform, as in a set. A changed mesh/hull shape,
     * or a Cube becoming sheared, is RESHAPE_PENDING for that version.
     *
     * A RESHAPE_PENDING prim is rebuilt for the first publication enqueued after the
     * one that excluded it completes, whether or not that operation was fetched or
     * destroyed, and returns unless another reason excludes it. It is therefore
     * absent from exactly one version when each publication follows the previous
     * one's completion, and from more when publications outrun completion. The
     * exclusion list is complete and sorted at completion; its tensors remain valid
     * until operation destruction.
     *
     * Implementation note: the kernel records row exclusions in budgeted pinned
     * host storage, and the next enqueue checks completion without blocking,
     * rereads only the pending prims, and rebuilds their definitions, keeping
     * every other slot's definition and device tables.
     */
    OVSPATIALQUERY_EXPERIMENTAL OVSPATIALQUERY_API ovspatialquery_enqueue_result_t
    ovspatialquery_enqueue_ovstage_population_set_with_execution(ovspatialquery_instance_handle_t   instance,
                                                                 ovspatialquery_population_handle_t population,
                                                                 ovspatialquery_ordinal_t           target_ordinal,
                                                                 const ovspatialquery_dynamic_execution_t* execution);

    /**
     * `enqueue_ovstage_population_update` with a choice of where the version is
     * prepared, under the rules of `enqueue_ovstage_population_set_with_execution`.
     * A device-prepared update may name the newest admitted device publication as
     * its base, so a caller publishes every step without waiting, and names the CUDA
     * device its base was prepared on (NOT_SUPPORTED otherwise). A poses-only update
     * compares each prim's pose and K with its base's on the device, bit for bit,
     * and records exactly the moved prims as its changed slots, so CUDA pair
     * reporters patch from it; a membership change makes them recompute. Its host
     * staging charge is a set's of the same population, so an instance whose host
     * budget covers the set and the updates it keeps pending admits it.
     */
    OVSPATIALQUERY_EXPERIMENTAL OVSPATIALQUERY_API ovspatialquery_enqueue_result_t
    ovspatialquery_enqueue_ovstage_population_update_with_execution(
        ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population,
        ovspatialquery_ordinal_t base_ordinal, ovspatialquery_ordinal_t target_ordinal,
        const ovspatialquery_dynamic_execution_t* execution);

    /**
     * Triangles without area that one successful publication dropped, per Mesh prim:
     * the machine-readable form of the report the set paragraph describes. A
     * publication lists every Mesh prim whose points or topology it read and
     * that dropped at least one triangle, so a set lists all of them and an update
     * those it read again; a Mesh excluded from the version is not listed.
     */
    typedef struct ovspatialquery_ovstage_dropped_triangles_t
    {
        /**
         * Host uint64[D] object IDs of the listed Mesh prims, sorted by value. The
         * library owns this tensor and its storage until the operation is destroyed.
         */
        DLTensor objects;
        /**
         * Host uint64[D], parallel to objects: how many of the prim's triangles, after
         * triangulation, were dropped. Both tensors have shape [0] when D is zero.
         */
        DLTensor dropped_triangle_counts;
    } ovspatialquery_ovstage_dropped_triangles_t;

    /**
     * Fetches the dropped-triangle list of one successful population publication,
     * under the rules of `ovspatialquery_fetch_population_update`: repeatable,
     * NOT_READY while pending, and OPERATION_FAILED after failed work. A
     * publication of an explicit population lists nothing.
     */
    OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_fetch_ovstage_dropped_triangles(
        ovspatialquery_instance_handle_t instance, ovspatialquery_op_id_t op_id,
        ovspatialquery_ovstage_dropped_triangles_t* out_result);

    /**
     * `ovspatialquery_query_capability` for populations of `stage`. A null
     * `stage` returns INVALID_ARGUMENT, and a stage `create_ovstage_population`
     * refuses returns NOT_SUPPORTED. PREPARE_ON_DEVICE also needs the stage's
     * runtime to provide the placement extension the device publication calls
     * read poses through; those calls refuse through the same check. Every
     * other key has the core answer.
     */
    OVSPATIALQUERY_API ovspatialquery_result_t
    ovspatialquery_query_ovstage_capability(ovspatialquery_instance_handle_t instance, ovstage_instance_t* stage,
                                            DLDevice device, ovspatialquery_capability_t capability);

#ifdef __cplusplus
}
#endif

#endif /* OVSPATIALQUERY_OVSTAGE_H */
