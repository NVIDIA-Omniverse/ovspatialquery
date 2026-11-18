/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// A filter of exactly one `usd-path IN` predicate selects the prims that exist at its paths
// at each publication, whichever way the adapter resolves it: through the interned path list
// it queries for that shape, or through Ovstage's evaluation, which a second predicate keeps.
// Both publish the same object and excluded counts when a path names no prim, when a path
// repeats, and after a prim appears at a listed path; a
// view's path filter follows the same rule. Where a CUDA device and the placement extension
// are available, the populations publish the same selections again prepared on the device.
//
// An empty filter selects the Sphere, Cube, Capsule and Mesh prims, not ovstage's every prim,
// and selected prims without a supported spatial source are excluded without removing
// the valid objects from that publication. The result identifies every excluded prim.
//
// No filter publishes a PointInstancer's prototype prims or RENDERING's copies of them, and
// none counts them as excluded.
#include <ovspatialquery/ovspatialquery_ovstage.h>

#include <ovstage/ovstage.h>
#include <ovx/path_dictionary/path_dictionary.h>

#include "../../support/native/ovstage_companion_fixtures.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <utility>
#include <vector>

namespace
{

constexpr char kAuthored[]    = "/World/AuthoredParent/Authored";
constexpr char kSynthesized[] = "/World/Synthesized";
constexpr char kAdded[]       = "/World/Added";
constexpr char kLater[]       = "/World/Later";

ovx_string_t text(const char* value, size_t size) { return ovx_string_t{value, size - 1u}; }

ovstage_predicate_t pathsIn(const ovx_string_t* values, size_t count)
{
    ovstage_predicate_t predicate{};
    predicate.attribute.string = text("usd-path", sizeof("usd-path"));
    predicate.op               = OVSTAGE_FILTER_OP_IN;
    predicate.values           = values;
    predicate.value_count      = count;
    return predicate;
}

ovstage_predicate_t hasExtent()
{
    ovstage_predicate_t predicate{};
    predicate.attribute.string = text("_worldExtent", sizeof("_worldExtent"));
    predicate.op               = OVSTAGE_FILTER_OP_HAS;
    return predicate;
}

// Compare the public result with authored paths, independent of dictionary insertion order.
bool finishesWithExclusions(
    ovspatialquery_instance_handle_t instance, path_dictionary_instance_t* dictionary,
    ovspatialquery_enqueue_result_t operation, ovspatialquery_ordinal_t ordinal, uint64_t objects,
    std::initializer_list<std::pair<const char*, ovspatialquery_ovstage_exclusion_reason_t>> expected)
{
    if (operation.status != OVSPATIALQUERY_API_SUCCESS)
        return false;
    ovspatialquery_op_wait_result_t           waited{};
    ovspatialquery_population_update_result_t result{};
    if (!succeeded(ovspatialquery_wait_op(instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited)) ||
        waited.state != OVSPATIALQUERY_OP_COMPLETED ||
        !succeeded(ovspatialquery_fetch_population_update(instance, operation.op_id, &result)) ||
        result.published_ordinal != ordinal || result.object_count != objects ||
        result.excluded_object_count != expected.size() || result.excluded_objects.data == nullptr ||
        result.exclusion_reasons.data == nullptr)
        return false;
    std::vector<std::pair<uint64_t, int32_t>> rows;
    for (const auto& entry : expected)
    {
        const ovx_string_t name{entry.first, std::strlen(entry.first)};
        ovx_primpath_t     path{};
        if (!dictionaryCallSucceeded(
                dictionary, dictionary->vtable->create_paths_from_strings(dictionary->context, &name, 1u, &path)))
            return false;
        rows.emplace_back(handleValue(path), entry.second);
    }
    std::sort(rows.begin(), rows.end());
    const auto* ids     = static_cast<const uint64_t*>(result.excluded_objects.data);
    const auto* reasons = static_cast<const int32_t*>(result.exclusion_reasons.data);
    for (size_t i = 0; i < rows.size(); ++i)
        if (ids[i] != rows[i].first || reasons[i] != rows[i].second)
            return false;
    return succeeded(ovspatialquery_destroy_op(instance, &operation.op_id));
}

ovspatialquery_ovstage_population_desc_t populationDesc(ovstage_instance_t* stage, const ovstage_filter_t& filter)
{
    ovspatialquery_ovstage_population_desc_t desc{};
    desc.stage  = stage;
    desc.filter = filter;
    desc.common = aabbPopulationDesc();
    return desc;
}

// A Cube of size 2 at x = -8, created through the write API with the attributes the typed
// acquisition reads: prim type, size, world matrix and world bounds.
bool authorLaterCube(ovstage_instance_t* stage, path_dictionary_instance_t* dictionary, ovstage_ordinal_t ordinal)
{
    const ovx_string_t cubeName = text("Cube", sizeof("Cube"));
    ovx_token_t        cube     = OVX_INVALID_TOKEN;
    if (!dictionaryCallSucceeded(
            dictionary, dictionary->vtable->create_tokens_from_strings(dictionary->context, &cubeName, 1u, &cube)))
    {
        return false;
    }
    double size = 2.0;
    double matrix[16]{1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -8.0, 0.0, 0.0, 1.0};
    double bounds[6]{-9.0, -1.0, -1.0, -7.0, 1.0, 1.0};
    return writeAttribute(stage, dictionary, kLater, sizeof(kLater) - 1u, "usd-prim-type", ordinal, &cube,
                          DLDataType{kDLUInt, 64u, 1u}) &&
           writeAttribute(stage, dictionary, kLater, sizeof(kLater) - 1u, "size", ordinal, &size,
                          DLDataType{kDLFloat, 64u, 1u}) &&
           writeAttribute(stage, dictionary, kLater, sizeof(kLater) - 1u, "omni:fabric:worldMatrix", ordinal, matrix,
                          DLDataType{kDLFloat, 64u, 16u}, 1u, OVSTAGE_SEMANTIC_MATRIX) &&
           writeBounds(stage, dictionary, kLater, sizeof(kLater) - 1u, ordinal, bounds);
}

// Objects a target answers for a bounds overlap that covers the whole fixture.
bool countsObjects(ovspatialquery_instance_handle_t instance, ovspatialquery_query_target_t target,
                   ovspatialquery_ordinal_t expectedOrdinal, uint64_t expected)
{
    ovspatialquery_overlap_desc_t overlap{};
    overlap.common.target    = target;
    overlap.common.precision = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    overlap.shape.kind       = OVSPATIALQUERY_OVERLAP_AABB;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        overlap.shape.data.aabb.min_corner[axis] = -100.0;
        overlap.shape.data.aabb.max_corner[axis] = 100.0;
    }
    std::array<ovspatialquery_overlap_hit_t, 8> hits{};
    ovspatialquery_overlap_hit_buffer_t         buffer{hits.data(), hits.size()};
    ovspatialquery_multiple_result_t            result{};
    return succeeded(ovspatialquery_overlap_multiple_sync(instance, &overlap, &buffer, &result)) &&
           result.source_ordinal == expectedOrdinal && result.truncated == 0u && result.stored_hit_count == expected;
}

// `execution` null publishes on the host; otherwise every publication uses it.
int testExactPathFilters(const ovspatialquery_dynamic_execution_t* execution)
{
    ovstage_instance_t*           stage = nullptr;
    const ovstage_instance_desc_t stageDesc{"ovspatialquery-path-filter-test"};
    CHECK(ovstage_create_instance(&stageDesc, &stage) == OVSTAGE_OK);
    path_dictionary_instance_t* dictionary = ovstage_get_path_dictionary(stage);
    CHECK(dictionary != nullptr);
    CHECK(populateBoundsFixture(stage, dictionary));

    const ovspatialquery_device_budget_t budget{{kDLCUDA, 0}, UINT64_C(64) << 20u};
    ovspatialquery_instance_desc_t       instanceDesc = commonInstanceDesc();
    if (execution != nullptr)
    {
        instanceDesc.device_budgets      = &budget;
        instanceDesc.device_budget_count = 1u;
    }
    ovspatialquery_instance_handle_t instance = OVSPATIALQUERY_INVALID_INSTANCE;
    CHECK(succeeded(ovspatialquery_create_instance(&instanceDesc, &instance)));
    const auto set = [&](ovspatialquery_population_handle_t population, ovspatialquery_ordinal_t ordinal)
    { return ovspatialquery_enqueue_ovstage_population_set_with_execution(instance, population, ordinal, execution); };
    const auto update = [&](ovspatialquery_population_handle_t population, ovspatialquery_ordinal_t base,
                            ovspatialquery_ordinal_t ordinal)
    {
        return ovspatialquery_enqueue_ovstage_population_update_with_execution(instance, population, base, ordinal,
                                                                               execution);
    };

    // Two prims with bounds, a path with no prim yet, and one prim named twice.
    const ovx_string_t listed[] = {text(kAuthored, sizeof(kAuthored)), text(kSynthesized, sizeof(kSynthesized)),
                                   text(kLater, sizeof(kLater)), text(kSynthesized, sizeof(kSynthesized))};
    // Compare the companion with the actual runtime's selector interpretation:
    // older runtimes prioritize nonzero tokens, newer runtimes supplied strings.
    const ovx_string_t pathAttribute = text("usd-path", sizeof("usd-path"));
    ovx_token_t        pathToken{};
    CHECK(dictionaryCallSucceeded(dictionary, dictionary->vtable->create_tokens_from_strings(
                                                  dictionary->context, &pathAttribute, 1u, &pathToken)));
    const decltype(ovstage_predicate_t::attribute) selectors[] = {
        {{0u}, pathAttribute}, {pathToken, {nullptr, 0u}}, {pathToken, text("usd-prim-type", sizeof("usd-prim-type"))},
        {pathToken, {"", 0u}}, {{0u}, {nullptr, 0u}},      {{UINT64_MAX}, {nullptr, 0u}},
    };
    for (const auto& selector : selectors)
    {
        ovstage_predicate_t predicate = pathsIn(listed, 4u);
        predicate.attribute           = selector;
        const ovstage_filter_t filter{&predicate, 1u};
        ovstage_query_handle_t query = OVSTAGE_INVALID_QUERY_HANDLE;
        const bool             queried =
            waitStageOperation(stage, stageVtable(stage)->query(stageContext(stage), &filter, nullptr, 0u, &query));
        auto                               desc       = populationDesc(stage, filter);
        ovspatialquery_population_handle_t population = OVSPATIALQUERY_INVALID_POPULATION;
        const auto created = ovspatialquery_create_ovstage_population(instance, &desc, &population);
        if (queried)
        {
            ovstage_query_result_t expected{};
            CHECK(stageVtable(stage)->fetch_query_result(stageContext(stage), query, OVSTAGE_TIMEOUT_INFINITE,
                                                         &expected) == OVSTAGE_OK);
            CHECK(succeeded(created));
            CHECK(finishPopulationUpdate(instance, set(population, 2), 2, expected.total_prim_count, 0u));
            CHECK(countsObjects(instance, populationTarget(population), 2, expected.total_prim_count));
            CHECK(stageVtable(stage)->release_query_result(stageContext(stage), &expected) == OVSTAGE_OK);
            CHECK(succeeded(ovspatialquery_destroy_population(instance, population)));
        }
        else
        {
            CHECK(created.status == OVSPATIALQUERY_API_INVALID_ARGUMENT);
        }
        if (query != OVSTAGE_INVALID_QUERY_HANDLE)
        {
            CHECK(waitStageOperation(stage, stageVtable(stage)->release_query(stageContext(stage), query)));
        }
    }
    const ovstage_predicate_t                exact           = pathsIn(listed, 4u);
    const ovstage_predicate_t                evaluated[]     = {hasExtent(), exact};
    ovspatialquery_ovstage_population_desc_t exactDesc       = populationDesc(stage, ovstage_filter_t{&exact, 1u});
    ovspatialquery_ovstage_population_desc_t evaluatedDesc   = populationDesc(stage, ovstage_filter_t{evaluated, 2u});
    ovspatialquery_population_handle_t       exactPopulation = OVSPATIALQUERY_INVALID_POPULATION;
    ovspatialquery_population_handle_t       evaluatedPopulation = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &exactDesc, &exactPopulation)));
    CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &evaluatedDesc, &evaluatedPopulation)));

    // A view over the exact population that names one present and one absent path. A
    // population with a view publishes only on the host.
    const ovx_string_t        viewPaths[]   = {text(kSynthesized, sizeof(kSynthesized)), text(kLater, sizeof(kLater))};
    const ovstage_predicate_t viewPredicate = pathsIn(viewPaths, 2u);
    ovspatialquery_ovstage_view_desc_t viewDesc{};
    viewDesc.filter                   = ovstage_filter_t{&viewPredicate, 1u};
    ovspatialquery_view_handle_t view = OVSPATIALQUERY_INVALID_VIEW;
    if (execution == nullptr)
        CHECK(succeeded(ovspatialquery_create_ovstage_view(instance, exactPopulation, &viewDesc, &view)));

    CHECK(finishPopulationUpdate(instance, set(exactPopulation, 2), 2, 2u, 0u));
    CHECK(finishPopulationUpdate(instance, set(evaluatedPopulation, 2), 2, 2u, 0u));
    CHECK(countsObjects(instance, populationTarget(exactPopulation), 2, 2u));
    CHECK(countsObjects(instance, populationTarget(evaluatedPopulation), 2, 2u));
    if (execution == nullptr)
        CHECK(countsObjects(instance, viewTarget(view), 2, 1u));

    // Both filter routes report the selected prim without bounds and retain the good prim.
    const ovx_string_t        withAbsent[]      = {text(kAuthored, sizeof(kAuthored)), text(kAdded, sizeof(kAdded))};
    const ovstage_predicate_t absentExact       = pathsIn(withAbsent, 2u);
    const ovstage_predicate_t absentEvaluated[] = {hasExtent(), absentExact};
    for (uint32_t route = 0u; route < 2u; ++route)
    {
        const ovstage_filter_t filter =
            route == 0u ? ovstage_filter_t{&absentExact, 1u} : ovstage_filter_t{absentEvaluated, 2u};
        ovspatialquery_ovstage_population_desc_t selected   = populationDesc(stage, filter);
        ovspatialquery_population_handle_t       population = OVSPATIALQUERY_INVALID_POPULATION;
        CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &selected, &population)));
        CHECK(finishesWithExclusions(instance, dictionary, set(population, 2), 2, 1u,
                                     {{kAdded, OVSPATIALQUERY_OVSTAGE_EXCLUSION_NO_SPATIAL_SOURCE}}));
        CHECK(countsObjects(instance, populationTarget(population), 2, 1u));
        CHECK(succeeded(ovspatialquery_destroy_population(instance, population)));
    }

    // A prim appears at the listed path after the populations exist: both routes admit it
    // at the next publication, and so does the view.
    CHECK(authorLaterCube(stage, dictionary, 3u));
    const ovstage_write_floor_desc_t floorThree{3u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
    CHECK(waitStageOperation(stage, ovstage_advance_write_floor(stage, &floorThree)));
    CHECK(finishPopulationUpdate(instance, update(exactPopulation, 2, 3), 3, 3u, 0u));
    CHECK(finishPopulationUpdate(instance, update(evaluatedPopulation, 2, 3), 3, 3u, 0u));
    CHECK(countsObjects(instance, populationTarget(exactPopulation), 3, 3u));
    CHECK(countsObjects(instance, populationTarget(evaluatedPopulation), 3, 3u));
    if (execution == nullptr)
    {
        CHECK(countsObjects(instance, viewTarget(view), 3, 2u));
        CHECK(succeeded(ovspatialquery_destroy_view(instance, view)));
    }
    CHECK(succeeded(ovspatialquery_destroy_population(instance, exactPopulation)));
    CHECK(succeeded(ovspatialquery_destroy_population(instance, evaluatedPopulation)));
    CHECK(succeeded(ovspatialquery_destroy_instance(instance)));
    CHECK(ovstage_destroy_instance(stage) == OVSTAGE_OK);
    return 0;
}

int testEmptyFilterAndUnpublishablePrims()
{
    ovstage_instance_t*           stage = nullptr;
    const ovstage_instance_desc_t stageDesc{"ovspatialquery-empty-filter-test"};
    CHECK(ovstage_create_instance(&stageDesc, &stage) == OVSTAGE_OK);
    path_dictionary_instance_t* dictionary = ovstage_get_path_dictionary(stage);
    CHECK(dictionary != nullptr);
    CHECK(populateBoundsFixture(stage, dictionary));

    const ovspatialquery_instance_desc_t instanceDesc = commonInstanceDesc();
    ovspatialquery_instance_handle_t     instance     = OVSPATIALQUERY_INVALID_INSTANCE;
    CHECK(succeeded(ovspatialquery_create_instance(&instanceDesc, &instance)));

    // The fixture's four Cubes, of which Added has no bounds; its two Xforms, the
    // pseudo-root and ovstage's bookkeeping prims are not selected.
    const ovstage_filter_t                   empty{nullptr, 0u};
    ovspatialquery_ovstage_population_desc_t selected   = populationDesc(stage, empty);
    ovspatialquery_population_handle_t       population = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &selected, &population)));
    CHECK(finishesWithExclusions(instance, dictionary,
                                 ovspatialquery_enqueue_ovstage_population_set(instance, population, 2), 2, 3u,
                                 {{kAdded, OVSPATIALQUERY_OVSTAGE_EXCLUSION_NO_SPATIAL_SOURCE}}));
    CHECK(countsObjects(instance, populationTarget(population), 2, 3u));
    CHECK(succeeded(ovspatialquery_destroy_population(instance, population)));

    // The prefix "/World" selects the World Xform itself, which carries _worldExtent but is
    // not a type the adapter publishes.
    const ovx_string_t  world                           = text("/World", sizeof("/World"));
    ovstage_predicate_t worldPrefix                     = pathsIn(&world, 1u);
    worldPrefix.op                                      = OVSTAGE_FILTER_OP_PREFIX;
    ovspatialquery_ovstage_population_desc_t underWorld = populationDesc(stage, ovstage_filter_t{&worldPrefix, 1u});
    CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &underWorld, &population)));
    CHECK(finishesWithExclusions(instance, dictionary,
                                 ovspatialquery_enqueue_ovstage_population_set(instance, population, 2), 2, 3u,
                                 {{"/World", OVSPATIALQUERY_OVSTAGE_EXCLUSION_UNSUPPORTED_TYPE},
                                  {"/World/AuthoredParent", OVSPATIALQUERY_OVSTAGE_EXCLUSION_UNSUPPORTED_TYPE},
                                  {kAdded, OVSPATIALQUERY_OVSTAGE_EXCLUSION_NO_SPATIAL_SOURCE}}));
    CHECK(countsObjects(instance, populationTarget(population), 2, 3u));
    CHECK(succeeded(ovspatialquery_destroy_population(instance, population)));

    ovspatialquery_ovstage_population_desc_t excluding = populationDesc(stage, empty);
    CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &excluding, &population)));
    ovspatialquery_ovstage_view_desc_t viewDesc{};
    viewDesc.filter                   = empty;
    ovspatialquery_view_handle_t view = OVSPATIALQUERY_INVALID_VIEW;
    CHECK(succeeded(ovspatialquery_create_ovstage_view(instance, population, &viewDesc, &view)));
    CHECK(finishPopulationUpdate(instance, ovspatialquery_enqueue_ovstage_population_set(instance, population, 2), 2,
                                 3u, 1u));
    CHECK(countsObjects(instance, populationTarget(population), 2, 3u));
    CHECK(countsObjects(instance, viewTarget(view), 2, 3u));
    CHECK(succeeded(ovspatialquery_destroy_view(instance, view)));
    CHECK(succeeded(ovspatialquery_destroy_population(instance, population)));

    // A prim created through the write API with a size and nothing else has no type and no
    // _worldExtent, so the empty filter leaves it out and a filter naming it reports an exclusion.
    constexpr char kBare[] = "/World/Bare";
    double         size    = 1.0;
    CHECK(
        writeAttribute(stage, dictionary, kBare, sizeof(kBare) - 1u, "size", 3u, &size, DLDataType{kDLFloat, 64u, 1u}));
    const ovstage_write_floor_desc_t floorThree{3u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
    CHECK(waitStageOperation(stage, ovstage_advance_write_floor(stage, &floorThree)));

    const ovx_string_t        bare         = text(kBare, sizeof(kBare));
    const ovstage_predicate_t barePath     = pathsIn(&bare, 1u);
    ovstage_predicate_t       hasSize      = hasExtent();
    hasSize.attribute.string               = text("size", sizeof("size"));
    const ovstage_predicate_t evaluated[]  = {hasSize, barePath};
    const ovx_string_t        xform        = text("Xform", sizeof("Xform"));
    ovstage_predicate_t       xforms       = pathsIn(&xform, 1u);
    xforms.attribute.string                = text("usd-prim-type", sizeof("usd-prim-type"));
    const ovstage_filter_t unpublishable[] = {
        ovstage_filter_t{&barePath, 1u},
        ovstage_filter_t{evaluated, 2u},
        ovstage_filter_t{&xforms, 1u},
    };
    for (const auto& filter : unpublishable)
    {
        ovspatialquery_ovstage_population_desc_t desc = populationDesc(stage, filter);
        CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &desc, &population)));
        const auto operation = ovspatialquery_enqueue_ovstage_population_set(instance, population, 3);
        if (filter.predicates == &xforms)
        {
            CHECK(
                finishesWithExclusions(instance, dictionary, operation, 3, 0u,
                                       {{"/World", OVSPATIALQUERY_OVSTAGE_EXCLUSION_UNSUPPORTED_TYPE},
                                        {"/World/AuthoredParent", OVSPATIALQUERY_OVSTAGE_EXCLUSION_UNSUPPORTED_TYPE}}));
        }
        else
        {
            CHECK(finishesWithExclusions(instance, dictionary, operation, 3, 0u,
                                         {{kBare, OVSPATIALQUERY_OVSTAGE_EXCLUSION_UNSUPPORTED_TYPE}}));
        }
        CHECK(countsObjects(instance, populationTarget(population), 3, 0u));
        CHECK(succeeded(ovspatialquery_destroy_population(instance, population)));
    }

    CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &excluding, &population)));
    CHECK(finishPopulationUpdate(instance, ovspatialquery_enqueue_ovstage_population_set(instance, population, 3), 3,
                                 3u, 1u));
    CHECK(succeeded(ovspatialquery_destroy_population(instance, population)));

    CHECK(succeeded(ovspatialquery_destroy_instance(instance)));
    CHECK(ovstage_destroy_instance(stage) == OVSTAGE_OK);
    return 0;
}

// A PointInstancer whose prototypes are a Cube and an Xform of two Spheres, beside a plain
// Cube. RENDERING copies each prototype prim under the instancer at the origin, with a
// _worldExtent; ALL also keeps the authored prototypes, and the authored Box is given a
// _worldExtent of its own so that only the prototype rule can leave it out.
constexpr char kInstancerScene[] = R"usda(#usda 1.0
def Xform "World" {
  def PointInstancer "PI" {
    rel prototypes = [</World/PI/Protos/Box>, </World/PI/Protos/Pair>]
    int[] protoIndices = [0, 1, 0]
    point3f[] positions = [(0,0,0), (5,0,0), (10,0,0)]
    def Scope "Protos" {
      def Cube "Box" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
        double size = 1
      }
      def Xform "Pair" {
        def Sphere "A" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
          double radius = 0.5
        }
        def Sphere "B" {
          double radius = 0.25
        }
      }
    }
  }
  def Cube "Plain" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
    double size = 1
    double3 xformOp:translate = (-5, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]
  }
}
)usda";

// True when a bounds overlap over the whole scene answers exactly one object, `expected`.
bool answersOnly(ovspatialquery_instance_handle_t instance, ovspatialquery_query_target_t target,
                 ovspatialquery_ordinal_t ordinal, ovx_primpath_t expected)
{
    ovspatialquery_overlap_desc_t overlap{};
    overlap.common.target    = target;
    overlap.common.precision = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    overlap.shape.kind       = OVSPATIALQUERY_OVERLAP_AABB;
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        overlap.shape.data.aabb.min_corner[axis] = -100.0;
        overlap.shape.data.aabb.max_corner[axis] = 100.0;
    }
    std::array<ovspatialquery_overlap_hit_t, 8> hits{};
    ovspatialquery_overlap_hit_buffer_t         buffer{hits.data(), hits.size()};
    ovspatialquery_multiple_result_t            result{};
    return succeeded(ovspatialquery_overlap_multiple_sync(instance, &overlap, &buffer, &result)) &&
           result.source_ordinal == ordinal && result.truncated == 0u && result.stored_hit_count == 1u &&
           hits[0].identity.object_id.value == handleValue(expected);
}

int testPointInstancerPrototypesAreNotObjects(bool withPhysics)
{
    ovstage_instance_t*           stage = nullptr;
    const ovstage_instance_desc_t stageDesc{"ovspatialquery-point-instancer-test"};
    CHECK(ovstage_create_instance(&stageDesc, &stage) == OVSTAGE_OK);
    path_dictionary_instance_t* dictionary = ovstage_get_path_dictionary(stage);
    CHECK(dictionary != nullptr);
    const ovstage_population_enqueue_result_t opened =
        withPhysics ? openUsdWithPhysics(stage, text(kInstancerScene, sizeof(kInstancerScene)), 1u, 0.0)
                    : ovstage_population_open_usd_from_string(stage, text(kInstancerScene, sizeof(kInstancerScene)), 1u,
                                                              0.0, OVSTAGE_POPULATION_DOMAIN_RENDERING);
    ovstage_population_op_wait_result_t opening{};
    CHECK(opened.status == OVSTAGE_OK &&
          ovstage_population_wait_op(stage, opened.op_index, OVSTAGE_TIMEOUT_INFINITE, &opening) == OVSTAGE_OK);
    const ovstage_write_floor_desc_t floorOne{1u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
    const ovstage_write_floor_desc_t floorTwo{2u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
    CHECK(waitStageOperation(stage, ovstage_advance_write_floor(stage, &floorOne)));
    CHECK(waitStageOperation(
        stage, ovstage_compute_hierarchy(stage, OVSTAGE_HIERARCHY_COMPUTATION_MODEL_DEFAULT_CPU, 1u, 2u)));
    constexpr char kBox[] = "/World/PI/Protos/Box";
    if (withPhysics)
    {
        double bounds[6]{-0.5, -0.5, -0.5, 0.5, 0.5, 0.5};
        CHECK(writeBounds(stage, dictionary, kBox, sizeof(kBox) - 1u, 2u, bounds));
    }
    CHECK(waitStageOperation(stage, ovstage_advance_write_floor(stage, &floorTwo)));

    constexpr char     kPlain[] = "/World/Plain";
    const ovx_string_t plainName{kPlain, sizeof(kPlain) - 1u};
    ovx_primpath_t     plain = OVX_INVALID_PRIMPATH;
    CHECK(dictionaryCallSucceeded(
        dictionary, dictionary->vtable->create_paths_from_strings(dictionary->context, &plainName, 1u, &plain)));
    const ovspatialquery_instance_desc_t instanceDesc = commonInstanceDesc();
    ovspatialquery_instance_handle_t     instance     = OVSPATIALQUERY_INVALID_INSTANCE;
    CHECK(succeeded(ovspatialquery_create_instance(&instanceDesc, &instance)));

    // The empty filter, an explicit type filter that matches the copy and the authored Box,
    // and a path list naming a copy, the authored Box where it exists, and Plain.
    const ovx_string_t  cube = text("Cube", sizeof("Cube"));
    ovstage_predicate_t cubes{};
    cubes.attribute.string = text("usd-prim-type", sizeof("usd-prim-type"));
    cubes.op               = OVSTAGE_FILTER_OP_IN;
    cubes.values           = &cube;
    cubes.value_count      = 1u;
    const ovx_string_t        named[]{text("/World/PI/proto0", sizeof("/World/PI/proto0")),
                                      text("/World/PI/proto1/A", sizeof("/World/PI/proto1/A")), plainName,
                                      text(kBox, sizeof(kBox))};
    const ovstage_predicate_t paths = pathsIn(named, withPhysics ? 4u : 3u);
    const ovstage_filter_t    filters[]{{nullptr, 0u}, {&cubes, 1u}, {&paths, 1u}};
    for (const ovstage_filter_t& filter : filters)
    {
        const ovspatialquery_ovstage_population_desc_t desc       = populationDesc(stage, filter);
        ovspatialquery_population_handle_t             population = OVSPATIALQUERY_INVALID_POPULATION;
        CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &desc, &population)));
        CHECK(finishPopulationUpdate(instance, ovspatialquery_enqueue_ovstage_population_set(instance, population, 2),
                                     2, 1u, 0u));
        CHECK(answersOnly(instance, populationTarget(population), 2, plain));
        CHECK(succeeded(ovspatialquery_destroy_population(instance, population)));
    }
    CHECK(succeeded(ovspatialquery_destroy_instance(instance)));
    CHECK(ovstage_destroy_instance(stage) == OVSTAGE_OK);
    return 0;
}

} // namespace

// Whether ovstage populations can prepare on CUDA device 0 here.
bool devicePreparationAvailable()
{
    ovstage_instance_t*           stage = nullptr;
    const ovstage_instance_desc_t stageDesc{"ovspatialquery-path-filter-capability"};
    if (ovstage_create_instance(&stageDesc, &stage) != OVSTAGE_OK)
        return false;
    const ovspatialquery_device_budget_t budget{{kDLCUDA, 0}, UINT64_C(64) << 20u};
    ovspatialquery_instance_desc_t       desc  = commonInstanceDesc();
    desc.device_budgets                        = &budget;
    desc.device_budget_count                   = 1u;
    ovspatialquery_instance_handle_t instance  = OVSPATIALQUERY_INVALID_INSTANCE;
    bool                             available = false;
    if (succeeded(ovspatialquery_create_instance(&desc, &instance)))
    {
        available = ovspatialquery_query_ovstage_capability(instance, stage, DLDevice{kDLCUDA, 0},
                                                            OVSPATIALQUERY_CAPABILITY_PREPARE_ON_DEVICE)
                        .status == OVSPATIALQUERY_API_SUCCESS;
        (void)ovspatialquery_destroy_instance(instance);
    }
    (void)ovstage_destroy_instance(stage);
    return available;
}

int main()
{
    CHECK(testExactPathFilters(nullptr) == 0);
    // The same selections prepared on the device, where this build, the runtime and a GPU allow it.
    if (devicePreparationAvailable())
    {
        const ovspatialquery_dynamic_execution_t onDevice{
            {kDLCUDA, 0}, {1u, 0u, 0u}, OVSPATIALQUERY_PREPARE_ON_DEVICE_HOST_READABLE};
        CHECK(testExactPathFilters(&onDevice) == 0);
    }
    CHECK(testEmptyFilterAndUnpublishablePrims() == 0);
    CHECK(testPointInstancerPrototypesAreNotObjects(false) == 0);
    CHECK(testPointInstancerPrototypesAreNotObjects(true) == 0);
    return 0;
}
