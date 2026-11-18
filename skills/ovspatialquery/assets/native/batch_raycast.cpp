// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// C++17: one reusable CPU raycast plan, launched several times with DLPack input
// columns; reads hit ranges, object ids and distances; times each launch.
// The scene is embedded USDA opened with ovstage_population_open_usd_from_string.

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
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{

// Wall x in [10,11], y,z in [-10,10]. Post x in [4,6], y,z in [-1,1]. Ball radius 1.5 at (6,5,-4).
// Glass is outside /World/Room/ and Pipe is a Cylinder: the filter must drop both.
constexpr char kScene[] = R"usda(#usda 1.0
(
    upAxis = "Y"
    metersPerUnit = 1
)
def Xform "World"
{
    def Xform "Room"
    {
        def Cube "Wall"
        {
            double size = 1
            float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
            double3 xformOp:translate = (10.5, 0, 0)
            float3 xformOp:scale = (1, 20, 20)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }
        def Cube "Post"
        {
            double size = 2
            float3[] extent = [(-1, -1, -1), (1, 1, 1)]
            double3 xformOp:translate = (5, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }
        def Sphere "Ball"
        {
            double radius = 1.5
            float3[] extent = [(-1.5, -1.5, -1.5), (1.5, 1.5, 1.5)]
            double3 xformOp:translate = (6, 5, -4)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }
        def Cylinder "Pipe"
        {
            double radius = 0.2
            double height = 4
            float3[] extent = [(-0.2, -0.2, -2), (0.2, 0.2, 2)]
            double3 xformOp:translate = (3, -8, 8)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }
    }
    def Cube "Glass"
    {
        double size = 1
        float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
        double3 xformOp:translate = (2, 6, 6)
        float3 xformOp:scale = (0.1, 4, 4)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }
}
)usda";

constexpr int      kSide     = 64; // kSide x kSide parallel rays along +x
constexpr uint64_t kRays     = uint64_t(kSide) * kSide;
constexpr double   kRange    = 20.0;
constexpr int      kLaunches = 5;

int g_failures = 0;

[[noreturn]] void fail(const std::string& what)
{
    std::fprintf(stderr, "error: %s\n", what.c_str());
    std::exit(1);
}

void check(bool ok, const std::string& what)
{
    if (!ok)
    {
        std::fprintf(stderr, "check failed: %s\n", what.c_str());
        ++g_failures;
    }
}

void require(ovspatialquery_api_status_t status, const char* what)
{
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        const ovspatialquery_string_t detail = ovspatialquery_get_last_error();
        fail(std::string(what) + " (status " + std::to_string(int(status)) +
             "): " + std::string(detail.ptr ? detail.ptr : "", detail.length));
    }
}

// Waits for an async op; on terminal failure prints its own diagnostic.
void waitOp(ovspatialquery_instance_handle_t instance, ovspatialquery_op_id_t op, const char* what)
{
    ovspatialquery_op_wait_result_t waited{};
    require(ovspatialquery_wait_op(instance, op, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited).status, what);
    if (waited.state != OVSPATIALQUERY_OP_COMPLETED)
    {
        const ovspatialquery_string_t detail = ovspatialquery_get_last_op_error(instance, op);
        fail(std::string(what) + ": " + std::string(detail.ptr ? detail.ptr : "", detail.length));
    }
}

void stageOp(ovstage_instance_t* stage, ovstage_enqueue_result_t op, const char* what)
{
    ovstage_op_wait_result_t waited{};
    if (op.status != OVSTAGE_OK ||
        ovstage_wait_op(stage, op.op_index, OVSTAGE_TIMEOUT_INFINITE, &waited) != OVSTAGE_OK ||
        waited.error_op_id_count != 0u)
    {
        const ovx_string_t detail = ovstage_get_last_op_error(stage, op.op_index);
        fail(std::string(what) + ": " + std::string(detail.ptr ? detail.ptr : "", detail.length));
    }
    ovstage_release_op(stage, op.op_index);
}

std::string pathString(path_dictionary_instance_t* dict, uint64_t id)
{
    const ovx_primpath_t path = ovx_primpath_t{id};
    ovx_token_t          buffer[64];
    ovx_token_t*         tokens    = nullptr;
    size_t               count     = 0u;
    size_t               processed = 0u;
    if (dict->vtable->get_tokens_from_paths(dict->context, &path, 1u, buffer, 64u, &tokens, &count, &processed)
                .status != OVX_API_SUCCESS ||
        processed != 1u)
    {
        fail("get_tokens_from_paths");
    }
    std::vector<ovx_string_t> names(count);
    if (dict->vtable->get_strings_from_tokens(dict->context, tokens, count, names.data()).status != OVX_API_SUCCESS)
    {
        fail("get_strings_from_tokens");
    }
    std::string out;
    for (const ovx_string_t& name : names)
    {
        out += '/';
        out.append(name.ptr, name.length);
    }
    return out.empty() ? "/" : out;
}

// Compact row-major CPU input column (strides = nullptr). COPIED: the buffer may be
// reused as soon as enqueue returns.
ovspatialquery_input_array_t cpuColumn(void* data, uint8_t floatBits, int32_t ndim, int64_t* shape)
{
    ovspatialquery_input_array_t array{};
    array.tensor.data    = data;
    array.tensor.device  = DLDevice{kDLCPU, 0};
    array.tensor.ndim    = ndim;
    array.tensor.dtype   = DLDataType{kDLFloat, floatBits, 1};
    array.tensor.shape   = shape;
    array.tensor.strides = nullptr;
    array.ownership      = OVSPATIALQUERY_INPUT_COPIED_ON_SUBMISSION;
    return array;
}

// Result columns are DLTensors: honour byte_offset; an unrequested column has shape == nullptr.
template <typename T>
const T* columnData(const DLTensor& tensor, const char* what)
{
    if (tensor.shape == nullptr || tensor.device.device_type != kDLCPU || tensor.dtype.bits != sizeof(T) * 8 ||
        (tensor.strides != nullptr && tensor.strides[tensor.ndim - 1] != 1))
    {
        fail(std::string("unexpected result column ") + what);
    }
    return reinterpret_cast<const T*>(static_cast<const uint8_t*>(tensor.data) + tensor.byte_offset);
}

// Hand-derived closest hit for a +x ray at (0, y, z).
const char* expectedHit(double y, double z, double* distance)
{
    const double d2 = (y - 5.0) * (y - 5.0) + (z + 4.0) * (z + 4.0);
    if (std::fabs(y) < 1.0 && std::fabs(z) < 1.0)
    {
        *distance = 4.0;
        return "/World/Room/Post";
    }
    if (d2 < 2.25)
    {
        *distance = 6.0 - std::sqrt(2.25 - d2);
        return "/World/Room/Ball";
    }
    if (std::fabs(y) < 10.0 && std::fabs(z) < 10.0)
    {
        *distance = 10.0;
        return "/World/Room/Wall";
    }
    return nullptr;
}

} // namespace

int main()
{
    // --- Ovstage: open from a string at ordinal 1, seal, hierarchy into 2, seal.
    ovstage_instance_desc_t stageDesc{};
    stageDesc.name            = "batch-raycast";
    ovstage_instance_t* stage = nullptr;
    if (ovstage_create_instance(&stageDesc, &stage) != OVSTAGE_OK || stage == nullptr)
    {
        fail("ovstage_create_instance");
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
    const ovstage_population_enqueue_result_t opened = ovstage_population_open_usd_from_string_with_desc(
        stage, ovx_string_t{kScene, std::strlen(kScene)}, 1u, 0.0, &stageContent, 1u);
#else
    const ovstage_population_enqueue_result_t opened = ovstage_population_open_usd_from_string(
        stage, ovx_string_t{kScene, std::strlen(kScene)}, 1u, 0.0, OVSTAGE_POPULATION_DOMAIN_ALL);
#endif
    ovstage_population_op_wait_result_t populated{};
    if (opened.status != OVSTAGE_OK ||
        ovstage_population_wait_op(stage, opened.op_index, OVSTAGE_TIMEOUT_INFINITE, &populated) != OVSTAGE_OK ||
        populated.error_op_id_count != 0u)
    {
        const ovx_string_t detail = ovstage_population_get_last_op_error(opened.op_index);
        fail("open usd: " + std::string(detail.ptr ? detail.ptr : "", detail.length));
    }
    ovstage_write_floor_desc_t floorDesc{};
    floorDesc.ordinal = 1u;
    floorDesc.scope   = OVSTAGE_SCOPE_ALL;
    stageOp(stage, ovstage_advance_write_floor(stage, &floorDesc), "write floor 1");
    stageOp(stage, ovstage_compute_hierarchy(stage, OVSTAGE_HIERARCHY_COMPUTATION_MODEL_DEFAULT_CPU, 1u, 2u),
            "compute hierarchy");
    floorDesc.ordinal = 2u;
    stageOp(stage, ovstage_advance_write_floor(stage, &floorDesc), "write floor 2");

    ovspatialquery_instance_desc_t instanceDesc{};
    instanceDesc.operation_capacity           = 8u;
    instanceDesc.max_cpu_concurrency          = 4u;
    instanceDesc.host_byte_budget             = uint64_t(256) << 20;
    ovspatialquery_instance_handle_t instance = OVSPATIALQUERY_INVALID_INSTANCE;
    require(ovspatialquery_create_instance(&instanceDesc, &instance).status, "create instance");

    // --- Filter: predicates are ANDed. PREFIX alone would also match the Cylinder, which the
    // publication would leave out as UNSUPPORTED_TYPE; the trailing slash keeps the Room Xform out.
    const ovx_string_t  prefix[] = {{"/World/Room/", std::strlen("/World/Room/")}};
    const ovx_string_t  types[]  = {{"Cube", 4u}, {"Sphere", 6u}};
    ovstage_predicate_t predicates[2]{};
    predicates[0].attribute.string = ovx_string_t{"usd-path", 8u};
    predicates[0].op               = OVSTAGE_FILTER_OP_PREFIX;
    predicates[0].values           = prefix;
    predicates[0].value_count      = 1u;
    predicates[1].attribute.string = ovx_string_t{"usd-prim-type", 13u};
    predicates[1].op               = OVSTAGE_FILTER_OP_IN;
    predicates[1].values           = types;
    predicates[1].value_count      = 2u;

    ovspatialquery_ovstage_population_desc_t populationDesc{};
    populationDesc.stage                          = stage;
    populationDesc.filter                         = ovstage_filter_t{predicates, 2u};
    ovspatialquery_population_handle_t population = OVSPATIALQUERY_INVALID_POPULATION;
    require(ovspatialquery_create_ovstage_population(instance, &populationDesc, &population).status,
            "create population");
    ovspatialquery_enqueue_result_t publish = ovspatialquery_enqueue_ovstage_population_set(instance, population, 2);
    require(publish.status, "enqueue population set");
    waitOp(instance, publish.op_id, "publication");
    ovspatialquery_population_update_result_t receipt{};
    require(ovspatialquery_fetch_population_update(instance, publish.op_id, &receipt).status, "fetch update");
    require(ovspatialquery_destroy_op(instance, &publish.op_id).status, "destroy publication op");
    check(receipt.object_count == 3u, "published 3 objects (Wall, Post, Ball)");
    // A prim the filter selects but cannot be published exactly is left out, not an error.
    check(receipt.excluded_object_count == 0u, "no prim left out");

    // --- Plan: CLOSEST + EXACT on the CPU. For CLOSEST/ANY, hit capacity must cover
    // every query or extra hits are dropped silently (truncated[i] != 0).
    ovspatialquery_raycast_plan_desc_t planDesc{};
    planDesc.target.kind              = OVSPATIALQUERY_TARGET_POPULATION;
    planDesc.target.handle.population = population;
    planDesc.hit_mode                 = OVSPATIALQUERY_HIT_MODE_CLOSEST;
    planDesc.precision                = OVSPATIALQUERY_QUERY_EXACT; // default 0 is BOUNDS_CANDIDATE
    planDesc.device                   = DLDevice{kDLCPU, 0};
    planDesc.maximum_query_count      = kRays;
    planDesc.maximum_hits_per_query   = 1u;
    planDesc.hit_capacity_per_slot    = kRays;
    planDesc.result_slot_count        = 1u; // so destroy each launch's op before the next launch
    planDesc.result_fields            = OVSPATIALQUERY_BATCH_RESULT_DISTANCE | OVSPATIALQUERY_BATCH_RESULT_POSITION;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    require(ovspatialquery_create_raycast_plan(instance, &planDesc, &plan).status, "create raycast plan");

    // --- Inputs: origins float64[N,3], unit_directions float32[N,3], distances float64[N].
    std::vector<double> origins(kRays * 3), minDistances(kRays, 0.0), maxDistances(kRays, kRange);
    std::vector<float>  directions(kRays * 3, 0.0f);
    for (int row = 0; row < kSide; ++row)
    {
        for (int col = 0; col < kSide; ++col)
        {
            const uint64_t ray      = uint64_t(row) * kSide + col;
            origins[ray * 3 + 0]    = 0.0;
            origins[ray * 3 + 1]    = -12.0 + 24.0 * (row + 0.5) / kSide;
            origins[ray * 3 + 2]    = -12.0 + 24.0 * (col + 0.5) / kSide;
            directions[ray * 3 + 0] = 1.0f;
        }
    }
    int64_t                            shapeN3[]       = {int64_t(kRays), 3};
    int64_t                            shapeN[]        = {int64_t(kRays)};
    const ovspatialquery_input_array_t originColumn    = cpuColumn(origins.data(), 64, 2, shapeN3);
    const ovspatialquery_input_array_t directionColumn = cpuColumn(directions.data(), 32, 2, shapeN3);
    const ovspatialquery_input_array_t minColumn       = cpuColumn(minDistances.data(), 64, 1, shapeN);
    const ovspatialquery_input_array_t maxColumn       = cpuColumn(maxDistances.data(), 64, 1, shapeN);

    path_dictionary_instance_t*               dict = ovstage_get_path_dictionary(stage);
    std::unordered_map<uint64_t, std::string> pathCache; // few distinct ids; convert each once
    std::map<std::string, uint64_t>           hitsByPath;
    std::vector<double>                       launchMs;
    uint64_t                                  misses   = 0;
    double                                    maxError = 0.0;

    for (int launchIndex = 0; launchIndex < kLaunches; ++launchIndex)
    {
        ovspatialquery_raycast_batch_t batch{};
        batch.ordinal.kind    = OVSPATIALQUERY_ORDINAL_HEAD;
        batch.query_count     = kRays;
        batch.origins         = &originColumn;
        batch.unit_directions = &directionColumn;
        batch.min_distances   = &minColumn;
        batch.max_distances   = &maxColumn;

        const auto                                 start = std::chrono::steady_clock::now();
        ovspatialquery_raycast_batch_reservation_t reservation{};
        ovspatialquery_enqueue_result_t op = ovspatialquery_enqueue_raycast_plan(instance, plan, &batch, &reservation);
        require(op.status, "enqueue raycast plan");
        waitOp(instance, op.op_id, "raycast launch");
        ovspatialquery_raycast_batch_result_t result{};
        require(ovspatialquery_fetch_raycast_batch(instance, op.op_id, &result).status, "fetch raycast batch");
        launchMs.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());

        if (launchIndex == kLaunches - 1)
        {
            const ovspatialquery_raycast_batch_columns_t& c = result.columns;
            const uint64_t  stored  = columnData<uint64_t>(c.hit_ranges.stored_hit_count, "stored_hit_count")[0];
            const uint32_t* status  = columnData<uint32_t>(c.hit_ranges.query_status, "query_status");
            const uint64_t* offsets = columnData<uint64_t>(c.hit_ranges.hit_offsets, "hit_offsets");
            const uint64_t* counts  = columnData<uint64_t>(c.hit_ranges.hit_counts, "hit_counts");
            const uint8_t*  cut     = columnData<uint8_t>(c.hit_ranges.truncated, "truncated");
            const uint64_t* ids     = columnData<uint64_t>(c.object_ids, "object_ids");
            // EXACT plan: read the .exact arm (common.truth says which arm is valid).
            check(result.common.truth == OVSPATIALQUERY_HIT_EXACT_SURFACE, "exact truth tag");
            const double* distances = columnData<double>(c.contact.exact.distances, "distances");
            const double* positions = columnData<double>(c.contact.exact.positions, "positions");
            for (uint64_t ray = 0; ray < kRays; ++ray)
            {
                check(status[ray] == OVSPATIALQUERY_BATCH_ROW_SUCCESS && cut[ray] == 0, "row status");
                double      wantDistance = 0.0;
                const char* wantPath     = expectedHit(origins[ray * 3 + 1], origins[ray * 3 + 2], &wantDistance);
                if (counts[ray] == 0)
                {
                    ++misses;
                    check(wantPath == nullptr, "unexpected miss");
                    continue;
                }
                const uint64_t flat = offsets[ray]; // flat hit row; not necessarily == ray
                check(flat < stored, "hit offset in range");
                auto found = pathCache.find(ids[flat]);
                if (found == pathCache.end())
                {
                    found = pathCache.emplace(ids[flat], pathString(dict, ids[flat])).first;
                }
                ++hitsByPath[found->second];
                check(wantPath != nullptr && found->second == wantPath, "hit path " + found->second);
                const double error = std::fabs(distances[flat] - wantDistance);
                maxError           = std::max(maxError, error);
                check(error < 1e-4, "hit distance");
                check(std::fabs(positions[flat * 3 + 0] - wantDistance) < 1e-4, "hit position x");
            }
        }
        // Results are readable only until the op is destroyed; destroy before the next launch.
        require(ovspatialquery_destroy_op(instance, &op.op_id).status, "destroy launch op");
    }

    // Literal totals for this 64 x 64 grid (6x6 rays on the post, 54x54 inside the wall).
    check(hitsByPath["/World/Room/Post"] == 36, "post hits");
    check(hitsByPath["/World/Room/Ball"] == 49, "ball hits");
    check(hitsByPath["/World/Room/Wall"] == 2831, "wall hits");
    check(misses == 1180, "misses");

    double mean = 0.0;
    for (double ms : launchMs)
    {
        mean += ms / launchMs.size();
    }
    std::printf("{\"object_count\": %llu, \"rays\": %llu, \"hits_by_path\": {",
                (unsigned long long)receipt.object_count, (unsigned long long)kRays);
    const char* sep = "";
    for (const auto& [path, hits] : hitsByPath)
    {
        std::printf("%s\"%s\": %llu", sep, path.c_str(), (unsigned long long)hits);
        sep = ", ";
    }
    std::printf("}, \"misses\": %llu, \"max_distance_error\": %.3g, \"launch_ms\": [", (unsigned long long)misses,
                maxError);
    for (size_t i = 0; i < launchMs.size(); ++i)
    {
        std::printf("%s%.3f", i ? ", " : "", launchMs[i]);
    }
    std::printf("], \"mean_launch_ms\": %.3f, \"failures\": %d}\n", mean, g_failures);

    // --- Teardown in reverse order of creation: plan, population, instance, stage.
    require(ovspatialquery_destroy_batch_plan(instance, plan).status, "destroy plan");
    require(ovspatialquery_destroy_population(instance, population).status, "destroy population");
    require(ovspatialquery_destroy_instance(instance).status, "destroy instance");
    ovstage_destroy_instance(stage);
    return g_failures == 0 ? 0 : 1;
}
