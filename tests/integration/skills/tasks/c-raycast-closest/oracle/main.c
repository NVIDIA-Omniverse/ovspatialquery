// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Reference solution: closest exact raycast against the obstacles of scene.usda.

#include <ovspatialquery/ovspatialquery.h>
#include <ovspatialquery/ovspatialquery_ovstage.h>
#include <ovstage/ovstage.h>
#include <ovx/path_dictionary/path_dictionary.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_RAYS 256

static const char* const kObstaclePaths[] = {
    "/World/Obstacles/Crate",
    "/World/Obstacles/Ball",
    "/World/Obstacles/Shelf/Board",
    "/World/Obstacles/Pillar",
};
#define OBSTACLE_COUNT (sizeof(kObstaclePaths) / sizeof(kObstaclePaths[0]))

typedef struct Ray
{
    double origin[3];
    double direction[3];
    double max_distance;
} Ray;

static void fail(const char* what)
{
    fprintf(stderr, "error: %s\n", what);
    exit(1);
}

static void failSq(const char* what)
{
    ovspatialquery_string_t detail = ovspatialquery_get_last_error();
    fprintf(stderr, "error: %s: %.*s\n", what, (int)detail.length, detail.ptr ? detail.ptr : "");
    exit(1);
}

static char* readFile(const char* path)
{
    FILE* file = fopen(path, "rb");
    if (!file)
    {
        fprintf(stderr, "error: cannot open %s\n", path);
        exit(1);
    }
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    char* text = malloc((size_t)size + 1u);
    if (!text || fread(text, 1u, (size_t)size, file) != (size_t)size)
    {
        fail("cannot read input file");
    }
    text[size] = '\0';
    fclose(file);
    return text;
}

/* Finds `"key"` inside [begin, end) and returns the first character after its colon. */
static const char* findKey(const char* begin, const char* end, const char* key)
{
    char quoted[64];
    snprintf(quoted, sizeof(quoted), "\"%s\"", key);
    size_t length = strlen(quoted);
    for (const char* cursor = begin; cursor + length <= end; ++cursor)
    {
        if (memcmp(cursor, quoted, length) == 0)
        {
            const char* colon = memchr(cursor + length, ':', (size_t)(end - cursor - (ptrdiff_t)length));
            if (!colon)
            {
                break;
            }
            return colon + 1;
        }
    }
    fprintf(stderr, "error: rays.json: missing key %s\n", key);
    exit(1);
}

static const char* parseVector(const char* cursor, double out[3])
{
    cursor = strchr(cursor, '[');
    if (!cursor)
    {
        fail("rays.json: expected [");
    }
    ++cursor;
    for (int axis = 0; axis < 3; ++axis)
    {
        while (*cursor == ' ' || *cursor == '\n' || *cursor == '\r' || *cursor == '\t' || *cursor == ',')
        {
            ++cursor;
        }
        char* next = NULL;
        out[axis]  = strtod(cursor, &next);
        if (next == cursor)
        {
            fail("rays.json: expected a number");
        }
        cursor = next;
    }
    return cursor;
}

static size_t parseRays(const char* text, Ray* rays)
{
    const char* cursor = strstr(text, "\"rays\"");
    if (!cursor)
    {
        fail("rays.json: missing \"rays\"");
    }
    cursor = strchr(cursor, '[');
    if (!cursor)
    {
        fail("rays.json: missing ray array");
    }
    size_t count = 0u;
    for (;;)
    {
        const char* open  = strchr(cursor, '{');
        const char* close = strchr(cursor, ']');
        /* The array ends at the first ']' not inside a ray object. */
        if (!open || (close && close < open))
        {
            break;
        }
        const char* end = strchr(open, '}');
        if (!end || count == MAX_RAYS)
        {
            fail("rays.json: malformed ray object");
        }
        Ray* ray = &rays[count++];
        parseVector(findKey(open, end, "origin"), ray->origin);
        parseVector(findKey(open, end, "direction"), ray->direction);
        ray->max_distance = strtod(findKey(open, end, "max_distance"), NULL);
        cursor            = end + 1;
    }
    return count;
}

static void waitStage(ovstage_instance_t* stage, ovstage_enqueue_result_t operation, const char* what)
{
    if (operation.status != OVSTAGE_OK)
    {
        fail(what);
    }
    ovstage_op_wait_result_t result;
    memset(&result, 0, sizeof(result));
    ovstage_api_status_t waited = ovstage_wait_op(stage, operation.op_index, OVSTAGE_TIMEOUT_INFINITE, &result);
    int                  failed = waited != OVSTAGE_OK || result.error_op_id_count != 0u;
    ovstage_release_op(stage, operation.op_index);
    if (failed)
    {
        fail(what);
    }
}

/* Converts a prim-path handle (a hit's object_id) back to "/A/B/C". */
static void pathToString(path_dictionary_instance_t* dictionary, ovx_primpath_t path, char* out, size_t capacity)
{
    ovx_token_t      tokens[64];
    ovx_token_t*     perPath   = NULL;
    size_t           numTokens = 0u;
    size_t           processed = 0u;
    ovx_api_result_t result    = dictionary->vtable->get_tokens_from_paths(dictionary->context, &path, 1u, tokens, 64u,
                                                                           &perPath, &numTokens, &processed);
    if (result.status != OVX_API_SUCCESS || processed != 1u)
    {
        fail("cannot decompose hit path");
    }
    ovx_string_t names[64];
    result = dictionary->vtable->get_strings_from_tokens(dictionary->context, perPath, numTokens, names);
    if (result.status != OVX_API_SUCCESS)
    {
        fail("cannot resolve hit path tokens");
    }
    size_t used = 0u;
    out[0]      = '\0';
    for (size_t index = 0u; index < numTokens; ++index)
    {
        used += (size_t)snprintf(out + used, capacity - used, "/%.*s", (int)names[index].length, names[index].ptr);
    }
}

int main(void)
{
    Ray    rays[MAX_RAYS];
    char*  raysText = readFile("rays.json");
    size_t rayCount = parseRays(raysText, rays);
    free(raysText);

    /* --- Ovstage: open scene.usda, seal ordinal 1, compute world transforms into ordinal 2. */
    ovstage_instance_desc_t stageDesc = {.name = "raycast"};
    ovstage_instance_t*     stage     = NULL;
    if (ovstage_create_instance(&stageDesc, &stage) != OVSTAGE_OK || !stage)
    {
        fail("ovstage_create_instance failed");
    }
    const char* scenePath = "scene.usda";
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
    stageContent.domains                       = OVSTAGE_POPULATION_DOMAIN_RENDERING;
    stageContent.selectors                     = &physics;
    stageContent.selector_count                = 1u;
    ovstage_population_enqueue_result_t opened = ovstage_population_open_usd_from_file_with_desc(
        stage, (ovx_string_t){scenePath, strlen(scenePath)}, 1u, 0.0, &stageContent, 1u);
#else
    ovstage_population_enqueue_result_t opened = ovstage_population_open_usd_from_file(
        stage, (ovx_string_t){scenePath, strlen(scenePath)}, 1u, 0.0, OVSTAGE_POPULATION_DOMAIN_ALL);
#endif
    if (opened.status != OVSTAGE_OK)
    {
        fail("ovstage_population_open_usd_from_file rejected");
    }
    ovstage_population_op_wait_result_t populated;
    memset(&populated, 0, sizeof(populated));
    if (ovstage_population_wait_op(stage, opened.op_index, OVSTAGE_TIMEOUT_INFINITE, &populated) != OVSTAGE_OK ||
        populated.error_op_id_count != 0u)
    {
        ovx_string_t detail = ovstage_population_get_last_op_error(opened.op_index);
        fprintf(stderr, "error: population failed: %.*s\n", (int)detail.length, detail.ptr ? detail.ptr : "");
        return 1;
    }
    ovstage_write_floor_desc_t floor = {.ordinal = 1u, .scope = OVSTAGE_SCOPE_ALL};
    waitStage(stage, ovstage_advance_write_floor(stage, &floor), "advance write floor to 1");
    waitStage(stage, ovstage_compute_hierarchy(stage, OVSTAGE_HIERARCHY_COMPUTATION_MODEL_DEFAULT_CPU, 1u, 2u),
              "compute hierarchy");
    floor.ordinal = 2u;
    waitStage(stage, ovstage_advance_write_floor(stage, &floor), "advance write floor to 2");

    /* --- ovspatialquery: one population of exactly the obstacle prims, published at ordinal 2. */
    ovspatialquery_instance_desc_t instanceDesc = {
        .operation_capacity  = 8u,
        .max_cpu_concurrency = 2u,
        .host_byte_budget    = UINT64_C(256) << 20,
    };
    ovspatialquery_instance_handle_t instance = OVSPATIALQUERY_INVALID_INSTANCE;
    if (ovspatialquery_create_instance(&instanceDesc, &instance).status != OVSPATIALQUERY_API_SUCCESS)
    {
        failSq("ovspatialquery_create_instance");
    }

    ovx_string_t values[OBSTACLE_COUNT];
    for (size_t index = 0u; index < OBSTACLE_COUNT; ++index)
    {
        values[index] = (ovx_string_t){kObstaclePaths[index], strlen(kObstaclePaths[index])};
    }
    ovstage_predicate_t predicate;
    memset(&predicate, 0, sizeof(predicate));
    predicate.attribute.string = (ovx_string_t){"usd-path", strlen("usd-path")};
    predicate.op               = OVSTAGE_FILTER_OP_IN;
    predicate.values           = values;
    predicate.value_count      = OBSTACLE_COUNT;

    ovspatialquery_ovstage_population_desc_t populationDesc;
    memset(&populationDesc, 0, sizeof(populationDesc));
    populationDesc.stage                            = stage;
    populationDesc.filter                           = (ovstage_filter_t){&predicate, 1u};
    populationDesc.common.retained_ordinal_capacity = 0u;
    ovspatialquery_population_handle_t population   = OVSPATIALQUERY_INVALID_POPULATION;
    if (ovspatialquery_create_ovstage_population(instance, &populationDesc, &population).status !=
        OVSPATIALQUERY_API_SUCCESS)
    {
        failSq("ovspatialquery_create_ovstage_population");
    }

    ovspatialquery_enqueue_result_t publish = ovspatialquery_enqueue_ovstage_population_set(instance, population, 2);
    if (publish.status != OVSPATIALQUERY_API_SUCCESS)
    {
        failSq("ovspatialquery_enqueue_ovstage_population_set");
    }
    ovspatialquery_op_wait_result_t waited = {OVSPATIALQUERY_OP_PENDING};
    if (ovspatialquery_wait_op(instance, publish.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status !=
            OVSPATIALQUERY_API_SUCCESS ||
        waited.state != OVSPATIALQUERY_OP_COMPLETED)
    {
        ovspatialquery_string_t detail = ovspatialquery_get_last_op_error(instance, publish.op_id);
        fprintf(stderr, "error: publication failed: %.*s\n", (int)detail.length, detail.ptr ? detail.ptr : "");
        return 1;
    }
    ovspatialquery_population_update_result_t receipt;
    memset(&receipt, 0, sizeof(receipt));
    if (ovspatialquery_fetch_population_update(instance, publish.op_id, &receipt).status !=
            OVSPATIALQUERY_API_SUCCESS ||
        receipt.object_count != OBSTACLE_COUNT)
    {
        failSq("population publication receipt");
    }
    ovspatialquery_destroy_op(instance, &publish.op_id);

    /* --- Closest exact raycast per input ray. */
    path_dictionary_instance_t* dictionary = ovstage_get_path_dictionary(stage);
    printf("{\"hits\": [");
    for (size_t index = 0u; index < rayCount; ++index)
    {
        const Ray*                    input = &rays[index];
        ovspatialquery_raycast_desc_t desc;
        memset(&desc, 0, sizeof(desc));
        desc.common.target.target.kind              = OVSPATIALQUERY_TARGET_POPULATION;
        desc.common.target.target.handle.population = population;
        desc.common.target.ordinal.kind             = OVSPATIALQUERY_ORDINAL_HEAD;
        desc.common.precision                       = OVSPATIALQUERY_QUERY_EXACT;
        double length = sqrt(input->direction[0] * input->direction[0] + input->direction[1] * input->direction[1] +
                             input->direction[2] * input->direction[2]);
        for (int axis = 0; axis < 3; ++axis)
        {
            desc.ray.origin[axis]         = input->origin[axis];
            desc.ray.unit_direction[axis] = (float)(input->direction[axis] / length);
        }
        desc.ray.min_distance = 0.0;
        desc.ray.max_distance = input->max_distance;

        ovspatialquery_raycast_single_result_t result;
        memset(&result, 0, sizeof(result));
        if (ovspatialquery_raycast_closest_sync(instance, &desc, &result).status != OVSPATIALQUERY_API_SUCCESS)
        {
            failSq("ovspatialquery_raycast_closest_sync");
        }
        printf("%s\n  ", index == 0u ? "" : ",");
        if (!result.has_hit)
        {
            printf("{\"path\": null, \"distance\": null, \"position\": null}");
            continue;
        }
        const ovspatialquery_raycast_exact_hit_t* hit = &result.hit.data.exact;
        char                                      path[1024];
        pathToString(dictionary, (ovx_primpath_t){hit->identity.object_id.value}, path, sizeof(path));
        printf("{\"path\": \"%s\", \"distance\": %.9g, \"position\": [%.9g, %.9g, %.9g]}", path, hit->distance,
               hit->position[0], hit->position[1], hit->position[2]);
    }
    printf("\n]}\n");

    if (ovspatialquery_destroy_population(instance, population).status != OVSPATIALQUERY_API_SUCCESS ||
        ovspatialquery_destroy_instance(instance).status != OVSPATIALQUERY_API_SUCCESS)
    {
        failSq("ovspatialquery cleanup");
    }
    ovstage_destroy_instance(stage);
    return 0;
}
