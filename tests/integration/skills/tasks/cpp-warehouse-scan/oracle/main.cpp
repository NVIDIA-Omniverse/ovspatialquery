// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Reference solution: a 128 x 128 forklift scanner over an instanced warehouse, through
// one reusable ovspatialquery batch raycast plan on the CPU.

#include <ovspatialquery/ovspatialquery.h>
#include <ovspatialquery/ovspatialquery_ovstage.h>
#include <ovstage/ovstage.h>
#include <ovx/path_dictionary/path_dictionary.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{

[[noreturn]] void fail(const std::string& what)
{
    std::fprintf(stderr, "error: %s\n", what.c_str());
    std::exit(1);
}

void requireSq(ovspatialquery_result_t result, const char* what)
{
    if (result.status != OVSPATIALQUERY_API_SUCCESS)
    {
        ovspatialquery_string_t detail = ovspatialquery_get_last_error();
        fail(std::string(what) + ": " + std::string(detail.ptr ? detail.ptr : "", detail.length));
    }
}

std::string readFile(const char* path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        fail(std::string("cannot open ") + path);
    }
    std::ostringstream text;
    text << file.rdbuf();
    return text.str();
}

/* Minimal lookup in the flat scanner.json: the value text after "key":. */
const char* jsonValue(const std::string& text, const char* key)
{
    const std::string quoted = std::string("\"") + key + "\"";
    size_t            at     = text.find(quoted);
    if (at == std::string::npos || (at = text.find(':', at)) == std::string::npos)
    {
        fail(std::string("scanner.json: missing ") + key);
    }
    return text.c_str() + at + 1;
}

double jsonNumber(const std::string& text, const char* key) { return std::strtod(jsonValue(text, key), nullptr); }

std::vector<double> jsonNumbers(const std::string& text, const char* key)
{
    const char* cursor = std::strchr(jsonValue(text, key), '[');
    if (!cursor)
    {
        fail(std::string("scanner.json: ") + key + " is not an array");
    }
    std::vector<double> values;
    ++cursor;
    for (;;)
    {
        while (*cursor == ' ' || *cursor == ',' || *cursor == '\n' || *cursor == '\r' || *cursor == '\t')
        {
            ++cursor;
        }
        if (*cursor == ']')
        {
            return values;
        }
        char* next = nullptr;
        values.push_back(std::strtod(cursor, &next));
        if (next == cursor)
        {
            fail(std::string("scanner.json: bad number in ") + key);
        }
        cursor = next;
    }
}

void waitStage(ovstage_instance_t* stage, ovstage_enqueue_result_t operation, const char* what)
{
    if (operation.status != OVSTAGE_OK)
    {
        fail(what);
    }
    ovstage_op_wait_result_t   result{};
    const ovstage_api_status_t waited = ovstage_wait_op(stage, operation.op_index, OVSTAGE_TIMEOUT_INFINITE, &result);
    const bool                 failed = waited != OVSTAGE_OK || result.error_op_id_count != 0u;
    ovstage_release_op(stage, operation.op_index);
    if (failed)
    {
        fail(what);
    }
}

/* Converts a prim-path handle (a hit's object_id) back to "/A/B/C". */
std::string pathToString(path_dictionary_instance_t* dictionary, ovx_primpath_t path)
{
    ovx_token_t      tokens[64];
    ovx_token_t*     perPath   = nullptr;
    size_t           numTokens = 0u;
    size_t           processed = 0u;
    ovx_api_result_t result    = dictionary->vtable->get_tokens_from_paths(dictionary->context, &path, 1u, tokens, 64u,
                                                                           &perPath, &numTokens, &processed);
    if (result.status != OVX_API_SUCCESS || processed != 1u)
    {
        fail("cannot decompose hit path");
    }
    std::vector<ovx_string_t> names(numTokens);
    result = dictionary->vtable->get_strings_from_tokens(dictionary->context, perPath, numTokens, names.data());
    if (result.status != OVX_API_SUCCESS)
    {
        fail("cannot resolve hit path tokens");
    }
    std::string out;
    for (const ovx_string_t& name : names)
    {
        out += '/';
        out.append(name.ptr, name.length);
    }
    return out;
}

ovspatialquery_input_array_t cpuArray(void* data, DLDataType dtype, int32_t rank, int64_t* shape)
{
    ovspatialquery_input_array_t array{};
    array.tensor.data    = data;
    array.tensor.device  = DLDevice{kDLCPU, 0};
    array.tensor.ndim    = rank;
    array.tensor.dtype   = dtype;
    array.tensor.shape   = shape;
    array.tensor.strides = nullptr; /* compact row-major */
    array.ownership      = OVSPATIALQUERY_INPUT_COPIED_ON_SUBMISSION;
    return array;
}

template <typename T>
const T* column(const DLTensor& tensor, const char* what)
{
    if (tensor.shape == nullptr || tensor.device.device_type != kDLCPU || tensor.dtype.bits != sizeof(T) * 8u ||
        (tensor.strides != nullptr && tensor.strides[0] != 1))
    {
        fail(std::string("unexpected result column ") + what);
    }
    return reinterpret_cast<const T*>(static_cast<const uint8_t*>(tensor.data) + tensor.byte_offset);
}

} // namespace

int main()
{
    const std::string scanner  = readFile("scanner.json");
    const auto        position = jsonNumbers(scanner, "position");
    const double      yaw      = jsonNumber(scanner, "yaw_deg") * M_PI / 180.0;
    const double      hfov     = jsonNumber(scanner, "horizontal_fov_deg") * M_PI / 180.0;
    const double      vfov     = jsonNumber(scanner, "vertical_fov_deg") * M_PI / 180.0;
    const auto        columns  = static_cast<uint64_t>(jsonNumber(scanner, "columns"));
    const auto        rows     = static_cast<uint64_t>(jsonNumber(scanner, "rows"));
    const double      maxRange = jsonNumber(scanner, "max_range_m");
    const auto        edges    = jsonNumbers(scanner, "histogram_edges_m");
    const auto        repeats  = static_cast<int>(jsonNumber(scanner, "repeats"));
    const auto        topCount = static_cast<size_t>(jsonNumber(scanner, "top_racks"));
    const uint64_t    rayCount = columns * rows;

    /* The racks are native instances; each rack box is an instance proxy at its own path
     * (/World/Racks/Rack_NNN/Upright_0, a CubeInstance). Only the RENDERING (or ALL)
     * population domain writes proxy rows. */
    const std::string usda = readFile("warehouse.usda");

    /* --- Ovstage: open, seal ordinal 1, compute world transforms into ordinal 2. */
    ovstage_instance_desc_t stageDesc{};
    stageDesc.name            = "warehouse-scan";
    ovstage_instance_t* stage = nullptr;
    if (ovstage_create_instance(&stageDesc, &stage) != OVSTAGE_OK || !stage)
    {
        fail("ovstage_create_instance failed");
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
    stageContent.domains                             = OVSTAGE_POPULATION_DOMAIN_RENDERING;
    stageContent.selectors                           = &physics;
    stageContent.selector_count                      = 1u;
    const ovstage_population_enqueue_result_t opened = ovstage_population_open_usd_from_string_with_desc(
        stage, ovx_string_t{usda.data(), usda.size()}, 1u, 0.0, &stageContent, 1u);
#else
    const ovstage_population_enqueue_result_t opened = ovstage_population_open_usd_from_string(
        stage, ovx_string_t{usda.data(), usda.size()}, 1u, 0.0, OVSTAGE_POPULATION_DOMAIN_ALL);
#endif
    ovstage_population_op_wait_result_t populated{};
    if (opened.status != OVSTAGE_OK ||
        ovstage_population_wait_op(stage, opened.op_index, OVSTAGE_TIMEOUT_INFINITE, &populated) != OVSTAGE_OK ||
        populated.error_op_id_count != 0u)
    {
        const ovx_string_t detail = ovstage_population_get_last_op_error(opened.op_index);
        fail("population failed: " + std::string(detail.ptr ? detail.ptr : "", detail.length));
    }
    ovstage_write_floor_desc_t floor{};
    floor.ordinal = 1u;
    floor.scope   = OVSTAGE_SCOPE_ALL;
    waitStage(stage, ovstage_advance_write_floor(stage, &floor), "advance write floor to 1");
    waitStage(stage, ovstage_compute_hierarchy(stage, OVSTAGE_HIERARCHY_COMPUTATION_MODEL_DEFAULT_CPU, 1u, 2u),
              "compute hierarchy");
    floor.ordinal = 2u;
    waitStage(stage, ovstage_advance_write_floor(stage, &floor), "advance write floor to 2");

    /* --- ovspatialquery population: the rack boxes under /World/Racks/ (the floor stays out).
     * An explicit filter keeps Ovstage's vocabulary: the proxies are CubeInstance, not Cube.
     * The prim-type predicate matters: a filter that also matches the rack roots
     * (XformInstance) makes publication fail with NOT_SUPPORTED. */
    ovspatialquery_instance_desc_t instanceDesc{};
    instanceDesc.operation_capacity           = 8u;
    instanceDesc.max_cpu_concurrency          = 4u;
    instanceDesc.host_byte_budget             = UINT64_C(512) << 20;
    ovspatialquery_instance_handle_t instance = OVSPATIALQUERY_INVALID_INSTANCE;
    requireSq(ovspatialquery_create_instance(&instanceDesc, &instance), "ovspatialquery_create_instance");

    const ovx_string_t  prefix[] = {{"/World/Racks/", std::strlen("/World/Racks/")}};
    const ovx_string_t  cube[]   = {{"CubeInstance", std::strlen("CubeInstance")}};
    ovstage_predicate_t predicates[2]{};
    predicates[0].attribute.string = ovx_string_t{"usd-path", std::strlen("usd-path")};
    predicates[0].op               = OVSTAGE_FILTER_OP_PREFIX;
    predicates[0].values           = prefix;
    predicates[0].value_count      = 1u;
    predicates[1].attribute.string = ovx_string_t{"usd-prim-type", std::strlen("usd-prim-type")};
    predicates[1].op               = OVSTAGE_FILTER_OP_IN;
    predicates[1].values           = cube;
    predicates[1].value_count      = 1u;

    ovspatialquery_ovstage_population_desc_t populationDesc{};
    populationDesc.stage                          = stage;
    populationDesc.filter                         = ovstage_filter_t{predicates, 2u};
    ovspatialquery_population_handle_t population = OVSPATIALQUERY_INVALID_POPULATION;
    requireSq(ovspatialquery_create_ovstage_population(instance, &populationDesc, &population),
              "ovspatialquery_create_ovstage_population");
    ovspatialquery_enqueue_result_t publish = ovspatialquery_enqueue_ovstage_population_set(instance, population, 2);
    if (publish.status != OVSPATIALQUERY_API_SUCCESS)
    {
        requireSq(ovspatialquery_result_t{publish.status}, "ovspatialquery_enqueue_ovstage_population_set");
    }
    ovspatialquery_op_wait_result_t waited{};
    requireSq(ovspatialquery_wait_op(instance, publish.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited),
              "wait for publication");
    ovspatialquery_population_update_result_t receipt{};
    if (waited.state != OVSPATIALQUERY_OP_COMPLETED ||
        ovspatialquery_fetch_population_update(instance, publish.op_id, &receipt).status != OVSPATIALQUERY_API_SUCCESS)
    {
        const ovspatialquery_string_t detail = ovspatialquery_get_last_op_error(instance, publish.op_id);
        fail("publication failed: " + std::string(detail.ptr ? detail.ptr : "", detail.length));
    }
    std::fprintf(stderr, "published %llu rack boxes\n", static_cast<unsigned long long>(receipt.object_count));
    ovspatialquery_destroy_op(instance, &publish.op_id);

    /* --- One reusable CPU plan for the whole 128 x 128 scan. */
    ovspatialquery_raycast_plan_desc_t planDesc{};
    planDesc.target.kind                    = OVSPATIALQUERY_TARGET_POPULATION;
    planDesc.target.handle.population       = population;
    planDesc.hit_mode                       = OVSPATIALQUERY_HIT_MODE_CLOSEST;
    planDesc.precision                      = OVSPATIALQUERY_QUERY_EXACT;
    planDesc.device                         = DLDevice{kDLCPU, 0};
    planDesc.maximum_query_count            = rayCount;
    planDesc.maximum_hits_per_query         = 1u;
    planDesc.hit_capacity_per_slot          = rayCount;
    planDesc.result_slot_count              = 1u;
    planDesc.result_fields                  = OVSPATIALQUERY_BATCH_RESULT_DISTANCE;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    requireSq(ovspatialquery_create_raycast_plan(instance, &planDesc, &plan), "ovspatialquery_create_raycast_plan");

    std::vector<double> origins(rayCount * 3u);
    std::vector<float>  directions(rayCount * 3u);
    std::vector<double> minimum(rayCount, 0.0);
    std::vector<double> maximum(rayCount, maxRange);
    for (uint64_t row = 0u; row < rows; ++row)
    {
        const double elevation = vfov * ((static_cast<double>(row) + 0.5) / static_cast<double>(rows) - 0.5);
        for (uint64_t col = 0u; col < columns; ++col)
        {
            const double azimuth = yaw + hfov * ((static_cast<double>(col) + 0.5) / static_cast<double>(columns) - 0.5);
            const uint64_t ray   = row * columns + col;
            for (int axis = 0; axis < 3; ++axis)
            {
                origins[ray * 3u + axis] = position[axis];
            }
            directions[ray * 3u + 0] = static_cast<float>(std::cos(elevation) * std::cos(azimuth));
            directions[ray * 3u + 1] = static_cast<float>(std::cos(elevation) * std::sin(azimuth));
            directions[ray * 3u + 2] = static_cast<float>(std::sin(elevation));
        }
    }
    int64_t                      shape2[]       = {static_cast<int64_t>(rayCount), 3};
    int64_t                      shape1[]       = {static_cast<int64_t>(rayCount)};
    ovspatialquery_input_array_t originArray    = cpuArray(origins.data(), DLDataType{kDLFloat, 64u, 1u}, 2, shape2);
    ovspatialquery_input_array_t directionArray = cpuArray(directions.data(), DLDataType{kDLFloat, 32u, 1u}, 2, shape2);
    ovspatialquery_input_array_t minimumArray   = cpuArray(minimum.data(), DLDataType{kDLFloat, 64u, 1u}, 1, shape1);
    ovspatialquery_input_array_t maximumArray   = cpuArray(maximum.data(), DLDataType{kDLFloat, 64u, 1u}, 1, shape1);

    path_dictionary_instance_t*       dictionary = ovstage_get_path_dictionary(stage);
    std::unordered_map<uint64_t, int> rackOfObject;
    std::vector<double>               times;
    uint64_t                          hitCount = 0u;
    std::vector<uint64_t>             histogram(edges.size() - 1u, 0u);
    std::unordered_map<int, uint64_t> rackHits;
    for (int repeat = 0; repeat < repeats; ++repeat)
    {
        ovspatialquery_raycast_batch_t launch{};
        launch.ordinal.kind    = OVSPATIALQUERY_ORDINAL_HEAD;
        launch.query_count     = rayCount;
        launch.origins         = &originArray;
        launch.unit_directions = &directionArray;
        launch.min_distances   = &minimumArray;
        launch.max_distances   = &maximumArray;

        const auto                                 start = std::chrono::steady_clock::now();
        ovspatialquery_raycast_batch_reservation_t reservation{};
        ovspatialquery_enqueue_result_t            operation =
            ovspatialquery_enqueue_raycast_plan(instance, plan, &launch, &reservation);
        if (operation.status != OVSPATIALQUERY_API_SUCCESS)
        {
            requireSq(ovspatialquery_result_t{operation.status}, "ovspatialquery_enqueue_raycast_plan");
        }
        ovspatialquery_op_wait_result_t launchWait{};
        requireSq(ovspatialquery_wait_op(instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &launchWait),
                  "wait for raycast batch");
        ovspatialquery_raycast_batch_result_t result{};
        requireSq(ovspatialquery_fetch_raycast_batch(instance, operation.op_id, &result),
                  "ovspatialquery_fetch_raycast_batch");
        times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());

        if (repeat == repeats - 1)
        {
            const DLTensor& stored  = result.columns.hit_ranges.stored_hit_count;
            const uint64_t  total   = column<uint64_t>(stored, "stored_hit_count")[0];
            const uint32_t* status  = column<uint32_t>(result.columns.hit_ranges.query_status, "query_status");
            const uint64_t* counts  = column<uint64_t>(result.columns.hit_ranges.hit_counts, "hit_counts");
            const uint64_t* offsets = column<uint64_t>(result.columns.hit_ranges.hit_offsets, "hit_offsets");
            const uint64_t* ids     = column<uint64_t>(result.columns.object_ids, "object_ids");
            const double*   dists   = column<double>(result.columns.contact.exact.distances, "distances");
            for (uint64_t ray = 0u; ray < rayCount; ++ray)
            {
                if (status[ray] != OVSPATIALQUERY_BATCH_ROW_SUCCESS)
                {
                    fail("ray row failed");
                }
                if (counts[ray] == 0u)
                {
                    continue;
                }
                const uint64_t flat = offsets[ray];
                if (flat >= total)
                {
                    fail("hit offset out of range");
                }
                ++hitCount;
                const double distance = dists[flat];
                size_t       bin =
                    static_cast<size_t>(std::upper_bound(edges.begin(), edges.end(), distance) - edges.begin());
                bin = std::min(std::max<size_t>(bin, 1u), histogram.size()) - 1u;
                ++histogram[bin];
                auto found = rackOfObject.find(ids[flat]);
                if (found == rackOfObject.end())
                {
                    const std::string path = pathToString(dictionary, ovx_primpath_t{ids[flat]});
                    const size_t      at   = path.find("/Rack_");
                    if (at == std::string::npos)
                    {
                        fail("hit outside a rack: " + path);
                    }
                    found = rackOfObject.emplace(ids[flat], std::atoi(path.c_str() + at + 6)).first;
                }
                ++rackHits[found->second];
            }
        }
        requireSq(ovspatialquery_destroy_op(instance, &operation.op_id), "destroy raycast batch");
    }

    std::vector<std::pair<int, uint64_t>> ranked(rackHits.begin(), rackHits.end());
    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b)
              { return a.second != b.second ? a.second > b.second : a.first < b.first; });
    ranked.resize(std::min(ranked.size(), topCount));
    double mean = 0.0;
    for (double t : times)
    {
        mean += t;
    }
    mean /= static_cast<double>(times.size());

    std::printf("{\"ray_count\": %llu, \"hit_count\": %llu, \"histogram\": [",
                static_cast<unsigned long long>(rayCount), static_cast<unsigned long long>(hitCount));
    for (size_t bin = 0u; bin < histogram.size(); ++bin)
    {
        std::printf("%s%llu", bin ? ", " : "", static_cast<unsigned long long>(histogram[bin]));
    }
    std::printf("], \"top_racks\": [");
    for (size_t index = 0u; index < ranked.size(); ++index)
    {
        std::printf("%s{\"rack\": %d, \"hits\": %llu}", index ? ", " : "", ranked[index].first,
                    static_cast<unsigned long long>(ranked[index].second));
    }
    std::printf("], \"query_times_ms\": [");
    for (size_t index = 0u; index < times.size(); ++index)
    {
        std::printf("%s%.6f", index ? ", " : "", times[index]);
    }
    std::printf("], \"mean_query_ms\": %.6f}\n", mean);

    requireSq(ovspatialquery_destroy_batch_plan(instance, plan), "destroy plan");
    requireSq(ovspatialquery_destroy_population(instance, population), "destroy population");
    requireSq(ovspatialquery_destroy_instance(instance), "destroy instance");
    ovstage_destroy_instance(stage);
    return 0;
}
