// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// C11: open a USD file into Ovstage, publish three gprims, run closest raycasts
// (exact, plus one bounds candidate for contrast), turn hit ids back into paths.
// Usage: raycast [scene.usda]   (relative paths resolve against the cwd)

#include <ovspatialquery/ovspatialquery.h>
#include <ovspatialquery/ovspatialquery_ovstage.h>
#include <ovstage/ovstage.h>
#include <ovx/path_dictionary/path_dictionary.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char* const kTargets[] = {"/World/Targets/Box", "/World/Targets/OffRay", "/World/Targets/MeshParent/Fin"};
#define TARGET_COUNT (sizeof(kTargets) / sizeof(kTargets[0]))

typedef struct Case
{
    const char* name;
    double      origin[3];
    double      direction[3]; /* any length; normalized below */
    double      max_distance;
    const char* want_path; /* NULL: expect a miss */
    double      want_distance;
    double      want_position[3];
} Case;

static int g_failures = 0;

static void die(const char* what)
{
    ovspatialquery_string_t detail = ovspatialquery_get_last_error();
    fprintf(stderr, "error: %s: %.*s\n", what, (int)detail.length, detail.ptr ? detail.ptr : "");
    exit(1);
}

static void check(int ok, const char* what)
{
    if (!ok)
    {
        fprintf(stderr, "check failed: %s\n", what);
        ++g_failures;
    }
}

/* Ovstage stage ops (write floor, hierarchy): wait, read the op error, then release. */
static void stage_op(ovstage_instance_t* stage, ovstage_enqueue_result_t op, const char* what)
{
    if (op.status != OVSTAGE_OK)
    {
        fprintf(stderr, "error: %s rejected (status %d)\n", what, (int)op.status);
        exit(1);
    }
    ovstage_op_wait_result_t waited;
    memset(&waited, 0, sizeof(waited));
    ovstage_api_status_t status = ovstage_wait_op(stage, op.op_index, OVSTAGE_TIMEOUT_INFINITE, &waited);
    if (status != OVSTAGE_OK || waited.error_op_id_count != 0u)
    {
        ovx_string_t detail = ovstage_get_last_op_error(stage, op.op_index);
        fprintf(stderr, "error: %s: %.*s\n", what, (int)detail.length, detail.ptr ? detail.ptr : "");
        exit(1);
    }
    ovstage_release_op(stage, op.op_index);
}

/* object_id.value is an ovx_primpath_t; the C dictionary has no path_to_string. */
static void path_string(path_dictionary_instance_t* dict, uint64_t id, char* out, size_t capacity)
{
    ovx_primpath_t   path = (ovx_primpath_t){id};
    ovx_token_t      buffer[64];
    ovx_token_t*     tokens    = NULL;
    size_t           count     = 0u;
    size_t           processed = 0u;
    ovx_api_result_t result =
        dict->vtable->get_tokens_from_paths(dict->context, &path, 1u, buffer, 64u, &tokens, &count, &processed);
    ovx_string_t names[64];
    if (result.status != OVX_API_SUCCESS || processed != 1u ||
        dict->vtable->get_strings_from_tokens(dict->context, tokens, count, names).status != OVX_API_SUCCESS)
    {
        fprintf(stderr, "error: cannot convert object id to a path\n");
        exit(1);
    }
    size_t used = 0u;
    out[0]      = '\0';
    if (count == 0u)
    {
        snprintf(out, capacity, "/"); /* zero tokens is the root */
    }
    for (size_t i = 0u; i < count && used < capacity; ++i)
    {
        used += (size_t)snprintf(out + used, capacity - used, "/%.*s", (int)names[i].length, names[i].ptr);
    }
}

static ovspatialquery_raycast_desc_t make_ray(ovspatialquery_population_handle_t population, const Case* c,
                                              ovspatialquery_query_precision_t precision)
{
    ovspatialquery_raycast_desc_t desc;
    memset(&desc, 0, sizeof(desc));
    desc.common.target.target.kind              = OVSPATIALQUERY_TARGET_POPULATION;
    desc.common.target.target.handle.population = population;
    desc.common.target.ordinal.kind             = OVSPATIALQUERY_ORDINAL_HEAD;
    desc.common.precision                       = precision;
    double length =
        sqrt(c->direction[0] * c->direction[0] + c->direction[1] * c->direction[1] + c->direction[2] * c->direction[2]);
    for (int axis = 0; axis < 3; ++axis)
    {
        desc.ray.origin[axis]         = c->origin[axis];
        desc.ray.unit_direction[axis] = (float)(c->direction[axis] / length); /* float[3], origin is double[3] */
    }
    desc.ray.min_distance = 0.0;
    desc.ray.max_distance = c->max_distance;
    return desc;
}

int main(int argc, char** argv)
{
    const char* scene = argc > 1 ? argv[1] : "scene.usda";

    /* --- Ovstage: open into ordinal 1, seal, world transforms into ordinal 2, seal. */
    ovstage_instance_desc_t stage_desc = {.name = "raycast-c"};
    ovstage_instance_t*     stage      = NULL;
    if (ovstage_create_instance(&stage_desc, &stage) != OVSTAGE_OK || !stage)
    {
        fprintf(stderr, "error: ovstage_create_instance failed\n");
        return 1;
    }
    #if OVSTAGE_VERSION_MINOR >= 4
    /* Ovstage 0.4 has no PHYSICS domain: populate RENDERING plus every prim's physics,
     * physxConvexHullCollision and material:binding properties, which ovspatialquery reads for
     * convex hulls, their vertex limits and materials. */
    static const ovx_string_t physicsNamespaces[] = {
        {"physics", 7u}, {"physxConvexHullCollision", 24u}, {"material:binding", 16u}};
    ovstage_population_selector_t physics;
    memset(&physics, 0, sizeof(physics));
    physics.prim_predicate.kind               = OVSTAGE_POPULATION_PRIM_PREDICATE_ALL;
    physics.property_predicate.kind           = OVSTAGE_POPULATION_PROPERTY_PREDICATE_IN_NAMESPACE;
    physics.property_predicate.values.strings = physicsNamespaces;
    physics.property_predicate.value_count    = 3u;
    ovstage_population_desc_t stageContent;
    memset(&stageContent, 0, sizeof(stageContent));
    stageContent.domains        = OVSTAGE_POPULATION_DOMAIN_RENDERING;
    stageContent.selectors      = &physics;
    stageContent.selector_count = 1u;
    ovstage_population_enqueue_result_t opened = ovstage_population_open_usd_from_file_with_desc(
        stage, (ovx_string_t){scene, strlen(scene)}, 1u, 0.0, &stageContent, 1u);
#else
    ovstage_population_enqueue_result_t opened = ovstage_population_open_usd_from_file(
        stage, (ovx_string_t){scene, strlen(scene)}, 1u, 0.0, OVSTAGE_POPULATION_DOMAIN_ALL);
#endif
    ovstage_population_op_wait_result_t populated;
    memset(&populated, 0, sizeof(populated));
    /* A missing file is accepted here and reported only by the population wait. */
    if (opened.status != OVSTAGE_OK ||
        ovstage_population_wait_op(stage, opened.op_index, OVSTAGE_TIMEOUT_INFINITE, &populated) != OVSTAGE_OK ||
        populated.error_op_id_count != 0u)
    {
        ovx_string_t detail = ovstage_population_get_last_op_error(opened.op_index);
        fprintf(stderr, "error: open %s: %.*s\n", scene, (int)detail.length, detail.ptr ? detail.ptr : "");
        return 1;
    }
    ovstage_write_floor_desc_t floor_desc = {.ordinal = 1u, .scope = OVSTAGE_SCOPE_ALL};
    stage_op(stage, ovstage_advance_write_floor(stage, &floor_desc), "advance write floor 1");
    stage_op(stage, ovstage_compute_hierarchy(stage, OVSTAGE_HIERARCHY_COMPUTATION_MODEL_DEFAULT_CPU, 1u, 2u),
             "compute hierarchy");
    floor_desc.ordinal = 2u;
    stage_op(stage, ovstage_advance_write_floor(stage, &floor_desc), "advance write floor 2");

    /* --- ovspatialquery instance: all three fields must be nonzero. */
    ovspatialquery_instance_desc_t instance_desc;
    memset(&instance_desc, 0, sizeof(instance_desc));
    instance_desc.operation_capacity          = 8u;
    instance_desc.max_cpu_concurrency         = 2u;
    instance_desc.host_byte_budget            = UINT64_C(256) << 20;
    ovspatialquery_instance_handle_t instance = OVSPATIALQUERY_INVALID_INSTANCE;
    if (ovspatialquery_create_instance(&instance_desc, &instance).status != OVSPATIALQUERY_API_SUCCESS)
    {
        die("ovspatialquery_create_instance");
    }

    /* --- Population: exact gprim paths (usd-path IN). A matched Xform would be left out as UNSUPPORTED_TYPE. */
    ovx_string_t values[TARGET_COUNT];
    for (size_t i = 0u; i < TARGET_COUNT; ++i)
    {
        values[i] = (ovx_string_t){kTargets[i], strlen(kTargets[i])};
    }
    ovstage_predicate_t predicate;
    memset(&predicate, 0, sizeof(predicate));
    predicate.attribute.string = (ovx_string_t){"usd-path", strlen("usd-path")};
    predicate.op               = OVSTAGE_FILTER_OP_IN;
    predicate.values           = values;
    predicate.value_count      = TARGET_COUNT;

    ovspatialquery_ovstage_population_desc_t population_desc;
    memset(&population_desc, 0, sizeof(population_desc));
    population_desc.stage                         = stage;
    population_desc.filter                        = (ovstage_filter_t){&predicate, 1u};
    ovspatialquery_population_handle_t population = OVSPATIALQUERY_INVALID_POPULATION;
    if (ovspatialquery_create_ovstage_population(instance, &population_desc, &population).status !=
        OVSPATIALQUERY_API_SUCCESS)
    {
        die("ovspatialquery_create_ovstage_population");
    }

    /* --- Publish exactly the last sealed ordinal (2); anything else is STALE_ORDINAL. */
    ovspatialquery_enqueue_result_t publish = ovspatialquery_enqueue_ovstage_population_set(instance, population, 2);
    if (publish.status != OVSPATIALQUERY_API_SUCCESS)
    {
        die("ovspatialquery_enqueue_ovstage_population_set");
    }
    ovspatialquery_op_wait_result_t waited;
    memset(&waited, 0, sizeof(waited));
    if (ovspatialquery_wait_op(instance, publish.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status !=
            OVSPATIALQUERY_API_SUCCESS ||
        waited.state != OVSPATIALQUERY_OP_COMPLETED)
    {
        ovspatialquery_string_t detail = ovspatialquery_get_last_op_error(instance, publish.op_id);
        fprintf(stderr, "error: publication: %.*s\n", (int)detail.length, detail.ptr ? detail.ptr : "");
        return 1;
    }
    ovspatialquery_population_update_result_t receipt;
    memset(&receipt, 0, sizeof(receipt));
    if (ovspatialquery_fetch_population_update(instance, publish.op_id, &receipt).status != OVSPATIALQUERY_API_SUCCESS)
    {
        die("ovspatialquery_fetch_population_update");
    }
    /* The op holds the population: destroy it now, or destroy_population returns BUSY. */
    if (ovspatialquery_destroy_op(instance, &publish.op_id).status != OVSPATIALQUERY_API_SUCCESS)
    {
        die("ovspatialquery_destroy_op");
    }
    check(receipt.object_count == TARGET_COUNT, "published object count");
    /* Stage content never fails a publication: a selected prim that cannot be published
       exactly (wrong type, a Sphere under non-uniform scale) is left out and listed in
       receipt.excluded_objects / exclusion_reasons instead. */
    check(receipt.excluded_object_count == 0u, "no prim left out");

    /* --- Closest exact raycasts. Expected values are worked out from scene.usda. */
    const Case cases[] = {
        {"axis", {-4, 0, 0}, {1, 0, 0}, 10.0, "/World/Targets/Box", 4.0, {0, 0, 0}},
        {"diagonal", {-2, -1.5, 0}, {1, 1, 0}, 10.0, "/World/Targets/Box", 2.0 * 1.4142135623730951, {0, 0.5, 0}},
        {"mesh", {0, 4.5, 0.8}, {1, 0, 0}, 10.0, "/World/Targets/MeshParent/Fin", 5.0, {5, 4.5, 0.8}},
        {"down", {3.2, 5, -3}, {0, -1, 0}, 10.0, "/World/Targets/OffRay", 4.25, {3.2, 0.75, -3}},
        {"too_short", {-4, 0, 0}, {1, 0, 0}, 3.5, NULL, 0.0, {0, 0, 0}},
        {"away", {-4, 0, 0}, {-1, 0, 0}, 10.0, NULL, 0.0, {0, 0, 0}},
    };
    const size_t                case_count = sizeof(cases) / sizeof(cases[0]);
    path_dictionary_instance_t* dict       = ovstage_get_path_dictionary(stage);

    printf("{\"object_count\": %llu, \"rays\": [", (unsigned long long)receipt.object_count);
    for (size_t i = 0u; i < case_count; ++i)
    {
        const Case*                            c    = &cases[i];
        ovspatialquery_raycast_desc_t          desc = make_ray(population, c, OVSPATIALQUERY_QUERY_EXACT);
        ovspatialquery_raycast_single_result_t result;
        memset(&result, 0, sizeof(result));
        if (ovspatialquery_raycast_closest_sync(instance, &desc, &result).status != OVSPATIALQUERY_API_SUCCESS)
        {
            die("ovspatialquery_raycast_closest_sync");
        }
        printf("%s\n  {\"name\": \"%s\", ", i ? "," : "", c->name);
        if (!result.has_hit)
        {
            printf("\"path\": null}");
            check(c->want_path == NULL, c->name);
            continue;
        }
        check(result.hit.truth == OVSPATIALQUERY_HIT_EXACT_SURFACE, "exact truth tag");
        const ovspatialquery_raycast_exact_hit_t* hit = &result.hit.data.exact; /* .exact arm for EXACT */
        char                                      path[512];
        path_string(dict, hit->identity.object_id.value, path, sizeof(path));
        printf(
            "\"path\": \"%s\", \"distance\": %.6f, \"position\": [%.6f, %.6f, %.6f], \"normal\": [%.3f, %.3f, %.3f]}",
            path, hit->distance, hit->position[0], hit->position[1], hit->position[2], hit->normal[0], hit->normal[1],
            hit->normal[2]);
        check(c->want_path != NULL && strcmp(path, c->want_path) == 0, c->name);
        check(fabs(hit->distance - c->want_distance) < 1e-4, c->name);
        for (int axis = 0; axis < 3; ++axis)
        {
            check(fabs(hit->position[axis] - c->want_position[axis]) < 1e-4, c->name);
        }
    }

    /* Bounds candidates (the default precision elsewhere) stop at the AABB: 4.0, not 5.0. */
    ovspatialquery_raycast_desc_t bounds = make_ray(population, &cases[2], OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE);
    ovspatialquery_raycast_single_result_t candidate;
    memset(&candidate, 0, sizeof(candidate));
    if (ovspatialquery_raycast_closest_sync(instance, &bounds, &candidate).status != OVSPATIALQUERY_API_SUCCESS)
    {
        die("bounds raycast");
    }
    check(candidate.has_hit && candidate.hit.truth == OVSPATIALQUERY_HIT_BOUNDS_CANDIDATE &&
              fabs(candidate.hit.data.bounds_candidate.distance - 4.0) < 1e-4,
          "bounds candidate distance");
    printf("\n], \"mesh_bounds_candidate_distance\": %.6f, \"failures\": %d}\n",
           candidate.has_hit ? candidate.hit.data.bounds_candidate.distance : -1.0, g_failures);

    /* --- Cleanup: ops (already destroyed) -> population -> instance -> stage. */
    if (ovspatialquery_destroy_population(instance, population).status != OVSPATIALQUERY_API_SUCCESS)
    {
        die("ovspatialquery_destroy_population");
    }
    if (ovspatialquery_destroy_instance(instance).status != OVSPATIALQUERY_API_SUCCESS)
    {
        die("ovspatialquery_destroy_instance");
    }
    ovstage_destroy_instance(stage);
    return g_failures == 0 ? 0 : 1;
}
