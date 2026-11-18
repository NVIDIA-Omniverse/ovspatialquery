/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_EXPLICIT_H
#define OVSPATIALQUERY_EXPLICIT_H

#include <ovspatialquery/ovspatialquery.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @file
     * @brief Explicit-source adapter: publish geometry and poses from caller arrays.
     *
     * Use this adapter when the scene is not in ovstage. The caller registers
     * immutable geometry definitions, then publishes populations whose instance
     * rows bind object IDs to those definitions with one rigid pose of positive
     * uniform scale each. The library derives world bounds from definition and
     * pose, and triangle sidedness belongs to the triangle definition, not the
     * instance. The resulting populations use the same query, version, and
     * result model as every other source.
     *
     * Typical lifetime (status checks and DLTensor setup omitted):
     *
     *     // Register convex 7, then publish two robots that both use it.
     *     op = ovspatialquery_enqueue_convex_registration(instance, &convexes);
     *     ovspatialquery_wait_op(instance, op.op_id, timeout, &wait_result);
     *     ovspatialquery_destroy_op(instance, &op.op_id);
     *
     *     ovspatialquery_create_explicit_population(instance, &population_desc,
     *                                                &robots);
     *     set.target_ordinal = 100;
     *     set.instances = two_robot_instances; // geometry { CONVEX, 7 }
     *     set.dynamic_execution = NULL;        // CPU capture
     *     op = ovspatialquery_enqueue_explicit_population_set(instance, robots, &set);
     *     // Wait, fetch if needed, then destroy the operation.
     *
     *     // Later: publish only the changed rows as ordinal 104.
     *     op = ovspatialquery_enqueue_explicit_population_update(
     *         instance, robots, &update_100_to_104);
     *
     *     ovspatialquery_destroy_population(instance, robots);
     *     ovspatialquery_destroy_geometry_definition(
     *         instance, (ovspatialquery_geometry_ref_t){
     *             OVSPATIALQUERY_GEOMETRY_CONVEX, 7 });
     *
     * Each input column follows its own ownership rule. The caller keeps borrowed
     * data alive and unchanged until the registration or publication operation is
     * terminal, and can overwrite copied-on-submission CPU data as soon as
     * admission succeeds. For copied-on-submission CUDA data the library queues
     * its copy on `cuda_sync.stream` before enqueue returns; the caller records
     * its own event on that stream after enqueue returns and makes any other
     * stream wait on that event before overwriting the source buffer.
     */

    /**
     * How an explicit publication uses `ovspatialquery_dynamic_execution_t`
     * (its `preparation` comment in `ovspatialquery.h` names the readers of
     * each side).
     *
     * For a CUDA device `cuda_sync` needs a stream (`has_stream` 1, any stream
     * value); `wait_event` may be zero, because the stream alone orders the
     * library's reads after the caller's writes queued on it before enqueue. The
     * library makes `cuda_sync.stream` wait on a nonzero `wait_event`, queues its
     * copies of copied columns there, and reads borrowed device arrays on its own
     * streams ordered after that work.
     * ON_HOST reads the rows on the CPU or copied down from a CUDA device.
     * ON_DEVICE requires CUDA rows (INVALID_ARGUMENT otherwise). Every exact
     * family prepares on the device (spheres, boxes, capsules, convex hulls,
     * triangle meshes), deriving the world bounds the host derives for the
     * same row; a USER row returns NOT_SUPPORTED at admission.
     *
     * A complete ON_DEVICE set reads its identity, binding, and
     * participation columns once at admission to define the initial membership
     * and the slot capacity; poses never cross. A device row that changes an
     * object's binding, or a POSES or POSES_AND_BITS row that names an unknown
     * or removed object, completes with OPERATION_FAILED and leaves the head
     * intact. Participation bits belong to the version: a COMPLETE_ROWS or
     * POSES_AND_BITS row sets its object's bits for that version and the ones
     * built on it, and a retained version keeps the bits it was published with.
     * Membership changes on the device path are COMPLETE_ROWS updates: a row
     * naming an unknown ID adds that object into the next free slot, up to the
     * set's `slot_capacity` (INVALID_ARGUMENT beyond it), and
     * `removed_object_values` removes objects. Such an update reads its identity
     * and removed columns back once at admission and is meant for event rate;
     * POSES updates never read anything back. A removed object's ID is retired
     * for the population's life: naming it again in complete rows or removals is
     * INVALID_ARGUMENT at admission, and a POSES row naming it fails the
     * publication. Retained versions keep the removed object.
     */

    /**
     * Immutable rigid instance bindings for this separate source path.
     *
     * `object_values` are nonzero and unique within the population. Each
     * `(geometry_types, geometry_values)` pair names one instance-wide registered
     * exact definition and must not collide with a definition from another source
     * adapter, because geometry references are instance-wide identity. Built-ins
     * have no public revision. A USER row returns NOT_SUPPORTED at admission,
     * because no definition of that reserved family can exist. Uniform scale
     * is positive and, like the translations and each object's world AABB,
     * inside the input domain of `ovspatialquery.h`. Float64 pose columns are
     * not narrowed before pose application. A `rotations_xyzw` row is valid
     * only when its four components are finite and its squared length is
     * within 1e-5 of one; the library normalizes it before use. Host- and
     * device-prepared versions apply the same rules, and a row outside them
     * fails the publication.
     * A registered convex instance retains its prepared local hull under this
     * pose. Subject to pose validity, finite inverse, and finite world-bound
     * checks, rounding that would make reconstructed world vertices coincide
     * or become nonplanar does not by itself invalidate the instance.
     */
    typedef struct ovspatialquery_explicit_instance_table_t
    {
        const ovspatialquery_input_array_t* object_values;      /* uint64[N] */
        const ovspatialquery_input_array_t* geometry_types;     /* int32[N], geometry_type */
        const ovspatialquery_input_array_t* geometry_values;    /* uint64[N] */
        const ovspatialquery_input_array_t* translations;       /* float32/64[N,3] */
        const ovspatialquery_input_array_t* rotations_xyzw;     /* float32/64[N,4] */
        const ovspatialquery_input_array_t* uniform_scales;     /* optional float32/64[N], default 1 */
        const ovspatialquery_input_array_t* participation_bits; /* optional uint64[N,2], default all */
        /**
         * Optional uint64[N]: the material each object's hits report, any value,
         * zero meaning none (`ovspatialquery_hit_identity_t`). Null gives every row
         * zero. It belongs to the version: a COMPLETE_ROWS row sets it, and POSES
         * and POSES_AND_BITS rows require null and keep each object's material.
         * Rows on a CUDA device have no material path yet, whatever their
         * preparation: a non-null column there returns NOT_SUPPORTED before
         * admission, and those objects report zero.
         */
        const ovspatialquery_input_array_t* material_ids;
    } ovspatialquery_explicit_instance_table_t;

    typedef struct ovspatialquery_explicit_population_set_t
    {
        /**
         * First ordinal to publish. The population has no head yet, so this may be
         * any nonnegative value, including zero. Later updates strictly advance.
         */
        ovspatialquery_ordinal_t                  target_ordinal;
        ovspatialquery_explicit_instance_table_t  instances;
        const ovspatialquery_dynamic_execution_t* dynamic_execution; /* null = CPU */
        /**
         * Experimental: device-prepared sets only. The number of distinct objects
         * the population may hold over its life, because the device tables are
         * sized once here and a slot is never reused after its object is removed.
         * Zero means exactly this set's rows, so no object can be added later. A
         * host-prepared set requires zero (INVALID_ARGUMENT).
         */
        uint64_t slot_capacity;
    } ovspatialquery_explicit_population_set_t;

    /**
     * What an update's changed rows carry.
     *
     * COMPLETE_ROWS: each row is a complete instance binding that adds or
     * replaces an object, as a set's rows do. POSES: each row moves an object
     * that already exists, which is what a physics engine emits every step, so
     * the binding columns stay null and the device path validates only the ID
     * and the pose. POSES_AND_BITS: POSES rows that also carry
     * `participation_bits`, for a producer whose objects change category at
     * pose rate (a package handed from one owner to another); unlisted objects
     * keep their bits. With POSES, `geometry_types`, `geometry_values`, and
     * `participation_bits` must be null; with POSES_AND_BITS the geometry
     * columns must be null and `participation_bits` must be present. With
     * both, `removed_object_values` must be empty (INVALID_ARGUMENT
     * otherwise), a null `uniform_scales` keeps each object's scale, and a row
     * naming an object the base does not have fails the publication.
     */
    typedef enum ovspatialquery_explicit_update_kind_t
    {
        OVSPATIALQUERY_EXPLICIT_UPDATE_COMPLETE_ROWS  = 0,
        OVSPATIALQUERY_EXPLICIT_UPDATE_POSES          = 1,
        OVSPATIALQUERY_EXPLICIT_UPDATE_POSES_AND_BITS = 2,
    } ovspatialquery_explicit_update_kind_t;

    /**
     * Exact-base update. Changed rows add or replace complete instance bindings,
     * or move existing objects when `kind` is POSES or POSES_AND_BITS. Removed
     * values are uint64[M]. The two identity sets are unique and disjoint.
     */
    typedef struct ovspatialquery_explicit_population_update_t
    {
        /**
         * Must equal the exact current head, or, for a device-prepared update
         * while device-prepared updates are pending, the newest admitted ordinal
         * (ORDINAL_MISMATCH otherwise). If the base publication fails, this one
         * completes with OPERATION_FAILED as well.
         */
        ovspatialquery_ordinal_t base_ordinal;
        /** Must be strictly greater than `base_ordinal`. */
        ovspatialquery_ordinal_t                  target_ordinal;
        ovspatialquery_explicit_update_kind_t     kind; /* COMPLETE_ROWS, POSES, or POSES_AND_BITS */
        ovspatialquery_explicit_instance_table_t  changed_instances;
        const ovspatialquery_input_array_t*       removed_object_values;
        const ovspatialquery_dynamic_execution_t* dynamic_execution;
    } ovspatialquery_explicit_population_update_t;

    typedef struct ovspatialquery_explicit_view_desc_t
    {
        /** CPU object_id[N], copied during creation. */
        const ovspatialquery_input_array_t* object_ids;
    } ovspatialquery_explicit_view_desc_t;

    /**
     * The caller assigns each exact built-in definition a `definition_value`
     * from 1 to 2^63 - 1, unique within its geometry type and instance; zero or
     * a value from 2^63 up, which the ovstage adapter assigns, returns
     * INVALID_ARGUMENT before admission (`ovspatialquery_geometry_ref_t`).
     * Instances bind `(geometry_type, definition_value)`. Changed geometry uses a
     * new value; there is no public revision lineage.
     */

    /**
     * Sphere local origin is its center. Radii are positive and inside the
     * input domain; otherwise registration returns INVALID_ARGUMENT before
     * admission.
     */
    typedef struct ovspatialquery_sphere_definition_table_t
    {
        const ovspatialquery_input_array_t* definition_values; /* nonzero uint64[G] */
        const ovspatialquery_input_array_t* radii;             /* float64[G] */
    } ovspatialquery_sphere_definition_table_t;

    /**
     * Box local origin is its center and axes are local XYZ. Half extents are
     * positive and inside the input domain; otherwise registration returns
     * INVALID_ARGUMENT before admission.
     */
    typedef struct ovspatialquery_box_definition_table_t
    {
        const ovspatialquery_input_array_t* definition_values; /* nonzero uint64[G] */
        const ovspatialquery_input_array_t* half_extents;      /* float64[G,3] */
    } ovspatialquery_box_definition_table_t;

    /**
     * Capsule segment lies on local Y with hemispheres at +/- half_segment.
     * Radii are positive, half segments nonnegative, and both inside the input
     * domain; otherwise registration returns INVALID_ARGUMENT before admission.
     */
    typedef struct ovspatialquery_capsule_definition_table_t
    {
        const ovspatialquery_input_array_t* definition_values; /* nonzero uint64[G] */
        const ovspatialquery_input_array_t* radii;             /* float64[G] */
        const ovspatialquery_input_array_t* half_segments;     /* float64[G] */
    } ovspatialquery_capsule_definition_table_t;

    /**
     * Packed convex source polyhedra use float32 local vertices. Other vertex
     * dtypes return NOT_SUPPORTED. The backend computes the convex hull of
     * each definition's vertices during private preparation, so the input
     * need not be convex, closed, or manifold. A hull has at most 255
     * vertices and 254 faces, where the backend counts neighboring nearly
     * coplanar triangles as one face, and no hull is reduced to fit unless
     * the call names a hull vertex limit. Vertices inside the hull do not
     * count, and any definition of at most 129 distinct vertices fits, since
     * such a hull has at most 2 * 129 - 4 = 254 faces.
     * `vertex_offsets/counts` select each definition's vertices. The five
     * face columns are optional and must be null or non-null together; when
     * present, `face_offsets/counts` select faces per definition and each
     * face selects a loop through `face_vertex_offsets/counts` into
     * mesh-relative vertex indices, kept as source topology but not required
     * to be convex. A registered hull's vertices are input vertices and no
     * input vertex lies farther outside one of its face planes than
     * `enclosure_tolerance` times the largest half extent of its definition's
     * vertex box, so queries never see another shape; a definition for which
     * the implementation cannot build such a hull, such as points that lie
     * nearly in one plane, is refused rather than replaced. Zero is the
     * default 2^-20, about eight float32 steps; a value from 2^-20 to 2^-16
     * lets a contact lie that much inside the vertices' own hull.
     *
     * A `hull_vertex_limit` from 8 to 64 registers instead the hull a PhysX
     * simulation collides with for a `convexHull` collider of these vertices
     * under that limit, so queries answer for the simulated shape: it has at
     * most that many vertices, they need not be input vertices, and input
     * vertices may lie outside it, so neither the enclosure tolerance nor the
     * vertex and face limits apply. Zero registers the vertices' own hull.
     * The library cooks that hull from the vertices in the definition's own
     * frame and units, and an instance's rotation, translation and uniform
     * scale then place it, as the simulation places a collider's hull by its
     * prim's transform. So the caller registers a stage's collider from its
     * local `points` in stage units and places it with the prim's world
     * rotation, its world translation times the stage's meters per unit, and
     * a uniform scale of the prim's scale times meters per unit, so the hull
     * lands in metres where the prim is; vertices with that placement or a unit
     * conversion already applied can give another hull. A prim under
     * nonuniform scale has no such placement.
     * Both members cover every definition in the call. Vertices outside the
     * input domain (nonfinite included), fewer than four non-coplanar
     * vertices in a definition, no such hull, a hull beyond either limit,
     * another tolerance or vertex limit, or loop indices outside the
     * definition return INVALID_ARGUMENT before admission. Data is CPU and
     * follows the ownership rule on each input array.
     *
     * Implementation note: without a hull vertex limit PhysX cooks the hull
     * about the points' center at unit size, merging nearly coplanar
     * triangles into one face. Past the vertex limit
     * it stops at the limit and leaves input vertices outside the hull, which
     * the library refuses as beyond that limit. For nearly coplanar points it
     * returns a thickened box or a hull that leaves points outside, which the
     * library refuses. A cooked polygon whose corners lie off one plane is
     * split into triangles of its corners; the face limit counts the polygons
     * before that split. Measured, and the same wherever the points sit as
     * long as float32 resolves the thickness there: an 8- to 127-sided prism
     * of radius 1 registers 1e-4 thick and is refused 3e-6 thick; 16 random
     * points over a 2 by 2 square register 1e-4 thick, 64 or 255 register
     * 1e-2 thick and, depending on the points, 1e-3 thick, and all are
     * refused 1e-5 thick, at the default tolerance. Under a hull vertex limit
     * PhysX cooks the vertices as given, each in its order at its own
     * coordinates, with its default plane tolerance and GPU-compatible data,
     * as a stage's PhysX simulation cooks a `convexHull` collider. Its
     * tolerances have absolute floors and it crops an oriented box, so that
     * hull depends on where the vertices sit, how they are turned and their
     * units as well as on their shape. It cooks with a tolerance scale of one
     * length unit, where the simulation uses one over meters per unit; that
     * scale changes the hull only of vertices whose box is thinner than 1e-6
     * of it, which the simulation thickens first. The library splits the
     * polygons PhysX merges, so queries answer for the hull of PhysX's cooked
     * vertices. At 64, beveled
     * cartons of 152 to 279 points keep 34 to 38 of them and leave points up
     * to 0.9 percent of the half extent outside; at 8 the hull is an oriented
     * box whose corners are not input vertices.
     */
    typedef struct ovspatialquery_convex_definition_table_t
    {
        const ovspatialquery_input_array_t* definition_values;   /* nonzero uint64[G] */
        const ovspatialquery_input_array_t* vertex_offsets;      /* uint64[G] */
        const ovspatialquery_input_array_t* vertex_counts;       /* uint64[G] */
        const ovspatialquery_input_array_t* face_offsets;        /* uint64[G] */
        const ovspatialquery_input_array_t* face_counts;         /* uint64[G] */
        const ovspatialquery_input_array_t* vertices;            /* float32[V,3] */
        const ovspatialquery_input_array_t* face_vertex_offsets; /* uint64[F] */
        const ovspatialquery_input_array_t* face_vertex_counts;  /* uint32[F] */
        const ovspatialquery_input_array_t* face_vertex_indices; /* uint32[I] */
        double                              enclosure_tolerance; /* 0 = 2^-20 */
        uint32_t                            hull_vertex_limit;   /* 0 = the points' own hull */
    } ovspatialquery_convex_definition_table_t;

    /**
     * Packed triangle meshes. The optional `double_sided` column is uint8[G];
     * null makes every mesh single-sided, which rays and sweeps treat as the
     * distance-interval comment in `ovspatialquery.h` states. Local vertices use float32; other vertex dtypes
     * return NOT_SUPPORTED, and vertices outside the input domain return
     * INVALID_ARGUMENT before admission. World poses retain double precision. The
     * implementation chooses query arithmetic.
     *
     * A triangle whose corners are collinear or repeated has no area, so no query
     * could hit it: registration drops it and keeps the mesh's other triangles,
     * which the per-triangle columns below still number in submitted order.
     * Its edge vectors' cross product, computed in double from the float32
     * vertices, is exactly zero; there is no tolerance, so any nonzero computed
     * area is kept at any scale. `ovspatialquery_get_dropped_triangle_count` returns
     * how many a definition lost, and the registration's diagnostic on the calling
     * thread (`ovspatialquery_get_last_error`) also names the first few. A mesh with
     * no triangle left, or a triangle that indexes a vertex outside its definition,
     * returns INVALID_ARGUMENT before admission.
     */
    typedef struct ovspatialquery_triangle_mesh_definition_table_t
    {
        const ovspatialquery_input_array_t* definition_values; /* nonzero uint64[G] */
        const ovspatialquery_input_array_t* vertex_offsets;    /* uint64[G] */
        const ovspatialquery_input_array_t* vertex_counts;     /* uint64[G] */
        const ovspatialquery_input_array_t* triangle_offsets;  /* uint64[G] */
        const ovspatialquery_input_array_t* triangle_counts;   /* uint64[G] */
        const ovspatialquery_input_array_t* vertices;          /* float32[V,3] */
        const ovspatialquery_input_array_t* triangle_indices;  /* uint32[T,3] */
        const ovspatialquery_input_array_t* double_sided;      /* optional uint8[G] */
        /**
         * Optional uint64[T], one per submitted triangle in `triangle_indices`
         * order: the material an exact contact on that triangle (its source face)
         * reports. Zero, or a null column, lets the triangle report its
         * instance's material, as a face outside every material subset of a USD
         * mesh takes the mesh's own binding. The table is part of the immutable
         * definition; different face materials need a new definition value.
         */
        const ovspatialquery_input_array_t* source_face_material_ids;
        /**
         * Optional uint64[T] and uint32[T,3], null or non-null together, one row
         * per submitted triangle in `triangle_indices` order: the source face (a
         * polygon the caller triangulated) and that triangle's three corner
         * ordinals within it, which an exact hit on the triangle reports in
         * `ovspatialquery_triangle_feature_t`, as an ovstage Mesh reports its USD
         * polygon and the polygon's corners. Null reports the submitted triangle
         * ordinal and corners { 0, 1, 2 }. The library reports the values as given
         * and does not check that a face's triangles tile it; a triangle naming
         * one corner twice returns INVALID_ARGUMENT before admission. When
         * `source_face_material_ids` holds a nonzero value, a material belongs to
         * a source face, so the triangles of one face carry one material and every
         * face value is below 2^32 - 1 (INVALID_ARGUMENT otherwise), and the
         * definition keeps one material per face value up to the largest, charged
         * to the host budget (HOST_BUDGET when it cannot fit). Dropped triangles
         * and their count still number submitted triangles.
         */
        const ovspatialquery_input_array_t* source_face_indices;        /* optional uint64[T] */
        const ovspatialquery_input_array_t* source_face_corner_indices; /* optional uint32[T,3] */
    } ovspatialquery_triangle_mesh_definition_table_t;

    /**
     * Creates a NOT_READY population with no head. Its first complete set may
     * publish any nonnegative ordinal, including zero. Unlike the general rule,
     * a non-null `out_population` is set to OVSPATIALQUERY_INVALID_POPULATION
     * before the other arguments are validated, so a rejected call leaves it
     * invalid.
     */
    OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_create_explicit_population(
        ovspatialquery_instance_handle_t instance, const ovspatialquery_population_desc_t* desc,
        ovspatialquery_population_handle_t* out_population);

    /**
     * Captures the first complete instance set and publishes its ordinal.
     * Every referenced geometry must already have completed registration. Success
     * is atomic; terminal failure leaves the population NOT_READY. Calling set on
     * a population that already has a head returns INVALID_STATE.
     */
    OVSPATIALQUERY_API ovspatialquery_enqueue_result_t ovspatialquery_enqueue_explicit_population_set(
        ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population,
        const ovspatialquery_explicit_population_set_t* set);

    /**
     * Captures changed/removed rows against the exact current head, or against
     * the newest admitted device-prepared update. The base must equal that
     * ordinal and the target must be strictly greater. Skipped ordinals are
     * allowed and are not replayed. A publication beyond the population's
     * pending capacity returns BUSY. Publication also fails atomically when
     * budgets cannot preserve the configured number of prior heads. Where the
     * version is prepared follows `dynamic_execution->preparation`; its comment
     * lists what each choice accepts.
     */
    OVSPATIALQUERY_API ovspatialquery_enqueue_result_t ovspatialquery_enqueue_explicit_population_update(
        ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population,
        const ovspatialquery_explicit_population_update_t* update);

    /**
     * Creates a copied-ID stable subset that shares all parent versions. Unlike
     * the general rule, a non-null `out_view` is set to OVSPATIALQUERY_INVALID_VIEW
     * before the other arguments are validated, so a rejected call leaves it
     * invalid.
     */
    OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_create_explicit_view(
        ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population,
        const ovspatialquery_explicit_view_desc_t* desc, ovspatialquery_view_handle_t* out_view);

    /** Captures and prepares an immutable batch of sphere definitions. */
    OVSPATIALQUERY_API ovspatialquery_enqueue_result_t ovspatialquery_enqueue_sphere_registration(
        ovspatialquery_instance_handle_t instance, const ovspatialquery_sphere_definition_table_t* definitions);

    /** Captures and prepares an immutable batch of box definitions. */
    OVSPATIALQUERY_API ovspatialquery_enqueue_result_t ovspatialquery_enqueue_box_registration(
        ovspatialquery_instance_handle_t instance, const ovspatialquery_box_definition_table_t* definitions);

    /** Captures and prepares an immutable batch of capsule definitions. */
    OVSPATIALQUERY_API ovspatialquery_enqueue_result_t ovspatialquery_enqueue_capsule_registration(
        ovspatialquery_instance_handle_t instance, const ovspatialquery_capsule_definition_table_t* definitions);

    /**
     * Captures and prepares immutable convex definitions. Two instances can bind
     * the same returned geometry identity without duplicating this data. Each
     * hull's triangle surface, which sphere and box sweeps on the hull read, is
     * charged as `ovspatialquery_enqueue_triangle_mesh_registration` charges a mesh.
     */
    OVSPATIALQUERY_API ovspatialquery_enqueue_result_t ovspatialquery_enqueue_convex_registration(
        ovspatialquery_instance_handle_t instance, const ovspatialquery_convex_definition_table_t* definitions);

    /**
     * Captures and prepares an immutable batch of triangle-mesh definitions. The
     * first exact reader of a mesh builds its query structure, so a mesh only ever
     * read for bounds costs no build time; the admission charges that structure and
     * the scratch of building it, so the build never refuses or fails a reader, and
     * returns HOST_BUDGET when they do not fit. Meshes not yet built share one
     * scratch charge sized for the largest of them, which shrinks as they are built
     * or destroyed.
     */
    OVSPATIALQUERY_API ovspatialquery_enqueue_result_t ovspatialquery_enqueue_triangle_mesh_registration(
        ovspatialquery_instance_handle_t instance, const ovspatialquery_triangle_mesh_definition_table_t* definitions);

    /**
     * Writes how many of a registered triangle-mesh definition's submitted triangles
     * registration dropped for having no area; zero when it dropped none. The value
     * must name an explicit triangle-mesh definition whose registration has
     * completed and that is not destroyed; otherwise INVALID_ARGUMENT, with the
     * output preserved. A null output is INVALID_ARGUMENT.
     */
    OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_get_dropped_triangle_count(
        ovspatialquery_instance_handle_t instance, uint64_t definition_value, uint64_t* out_count);

    /**
     * Destroys one explicit geometry definition. Returns BUSY
     * while a published version or admitted operation still references it.
     * ovstage-owned definitions remain owned by the ovstage adapter and cannot be
     * destroyed here. `geometry` must be a non-invalid instance-wide reference;
     * a USER reference returns NOT_SUPPORTED.
     */
    OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_destroy_geometry_definition(
        ovspatialquery_instance_handle_t instance, ovspatialquery_geometry_ref_t geometry);

#ifdef __cplusplus
}
#endif

#endif /* OVSPATIALQUERY_EXPLICIT_H */
