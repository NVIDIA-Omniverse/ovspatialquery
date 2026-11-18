/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// A population's or view's scope selects, among the prims its filter selects, those whose
// nearest listed ancestor-or-self is an include (ADR-0055): an exclude prunes a subtree, an
// include below it re-includes one, an exclude wins at equal depth, paths match by element,
// an empty include list stands for "/", and an include root may appear later. A scoped-out
// prim is not an exclusion, even when it has no spatial source, and its content never fails a
// publication; malformed lists create nothing. Each case names the published prims, not only their number.
#include <ovspatialquery/ovspatialquery_ovstage.h>

#include <ovstage/ovstage.h>
#include <ovx/path_dictionary/path_dictionary.h>

#include "../../support/native/ovstage_companion_fixtures.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <vector>

namespace
{

// Six cubes: three nested under /World/A, two under the sibling Xforms Env and
// Environment, whose names share a prefix, and one directly under /World.
constexpr char kScene[] = R"usda(#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
    metersPerUnit = 1
)
def Xform "World"
{
    def Xform "A"
    {
        def Cube "c1"
        {
            double size = 1
            double3 xformOp:translate = (0, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }
        def Xform "B"
        {
            def Cube "c2"
            {
                double size = 1
                double3 xformOp:translate = (2, 0, 0)
                uniform token[] xformOpOrder = ["xformOp:translate"]
            }
            def Xform "C"
            {
                def Cube "c3"
                {
                    double size = 1
                    double3 xformOp:translate = (4, 0, 0)
                    uniform token[] xformOpOrder = ["xformOp:translate"]
                }
            }
        }
    }
    def Xform "Env"
    {
        def Cube "c4"
        {
            double size = 1
            double3 xformOp:translate = (6, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }
    }
    def Xform "Environment"
    {
        def Cube "c5"
        {
            double size = 1
            double3 xformOp:translate = (8, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }
    }
    def Cube "c6"
    {
        double size = 1
        double3 xformOp:translate = (10, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
}
)usda";

constexpr char kC1[]    = "/World/A/c1";
constexpr char kC2[]    = "/World/A/B/c2";
constexpr char kC3[]    = "/World/A/B/C/c3";
constexpr char kC4[]    = "/World/Env/c4";
constexpr char kC5[]    = "/World/Environment/c5";
constexpr char kC6[]    = "/World/c6";
constexpr char kLater[] = "/World/Later";

struct Fixture
{
    ovstage_instance_t*              stage      = nullptr;
    path_dictionary_instance_t*      dictionary = nullptr;
    ovspatialquery_instance_handle_t instance   = OVSPATIALQUERY_INVALID_INSTANCE;
};

bool openScene(Fixture& f)
{
    const ovstage_instance_desc_t stageDesc{"ovspatialquery-path-scope-test"};
    if (ovstage_create_instance(&stageDesc, &f.stage) != OVSTAGE_OK)
        return false;
    f.dictionary = ovstage_get_path_dictionary(f.stage);
    const ovstage_population_enqueue_result_t opened =
        openUsdWithPhysics(f.stage, ovx_string_t{kScene, sizeof(kScene) - 1u}, 1u, 0.0);
    ovstage_population_op_wait_result_t waited{};
    if (f.dictionary == nullptr || opened.status != OVSTAGE_OK ||
        ovstage_population_wait_op(f.stage, opened.op_index, OVSTAGE_TIMEOUT_INFINITE, &waited) != OVSTAGE_OK)
        return false;
    const ovstage_write_floor_desc_t     floorOne{1u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
    const ovstage_write_floor_desc_t     floorTwo{2u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
    const ovspatialquery_instance_desc_t instanceDesc = commonInstanceDesc();
    return waitStageOperation(f.stage, ovstage_advance_write_floor(f.stage, &floorOne)) &&
           waitStageOperation(
               f.stage, ovstage_compute_hierarchy(f.stage, OVSTAGE_HIERARCHY_COMPUTATION_MODEL_DEFAULT_CPU, 1u, 2u)) &&
           waitStageOperation(f.stage, ovstage_advance_write_floor(f.stage, &floorTwo)) &&
           succeeded(ovspatialquery_create_instance(&instanceDesc, &f.instance));
}

ovx_primpath_t path(const Fixture& f, const char* text)
{
    const ovx_string_t name{text, std::strlen(text)};
    ovx_primpath_t     value = OVX_INVALID_PRIMPATH;
    (void)dictionaryCallSucceeded(
        f.dictionary, f.dictionary->vtable->create_paths_from_strings(f.dictionary->context, &name, 1u, &value));
    return value;
}

std::vector<ovx_primpath_t> paths(const Fixture& f, std::initializer_list<const char*> texts)
{
    std::vector<ovx_primpath_t> values;
    for (const char* text : texts)
        values.push_back(path(f, text));
    return values;
}

struct Lists
{
    std::vector<ovx_primpath_t> include, exclude;

    ovspatialquery_ovstage_scope_t scope() const
    {
        ovspatialquery_ovstage_scope_t value{};
        value.include_paths = include.empty() ? nullptr : include.data();
        value.include_count = static_cast<uint32_t>(include.size());
        value.exclude_paths = exclude.empty() ? nullptr : exclude.data();
        value.exclude_count = static_cast<uint32_t>(exclude.size());
        return value;
    }
};

ovspatialquery_ovstage_population_desc_t populationDesc(const Fixture& f, const ovspatialquery_ovstage_scope_t& scope)
{
    ovspatialquery_ovstage_population_desc_t desc{};
    desc.stage  = f.stage;
    desc.scope  = scope;
    desc.common = aabbPopulationDesc();
    return desc;
}

// The objects a target answers for a bounds overlap that covers the whole scene, as path values.
bool selects(const Fixture& f, ovspatialquery_query_target_t target, ovspatialquery_ordinal_t ordinal,
             std::initializer_list<const char*> expected)
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
    std::array<ovspatialquery_overlap_hit_t, 16> hits{};
    ovspatialquery_overlap_hit_buffer_t          buffer{hits.data(), hits.size()};
    ovspatialquery_multiple_result_t             result{};
    if (!succeeded(ovspatialquery_overlap_multiple_sync(f.instance, &overlap, &buffer, &result)) ||
        result.source_ordinal != ordinal || result.truncated != 0u)
        return false;
    std::vector<uint64_t> got;
    for (uint64_t i = 0u; i < result.stored_hit_count; ++i)
        got.push_back(hits[i].identity.object_id.value);
    std::vector<uint64_t> want;
    for (const char* text : expected)
        want.push_back(handleValue(path(f, text)));
    std::sort(got.begin(), got.end());
    std::sort(want.begin(), want.end());
    return got == want;
}

// Publishes ordinal 2 into a population with `lists` and checks which prims it holds.
int expectPopulation(Fixture& f, const Lists& lists, std::initializer_list<const char*> expected)
{
    const ovspatialquery_ovstage_population_desc_t desc       = populationDesc(f, lists.scope());
    ovspatialquery_population_handle_t             population = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(succeeded(ovspatialquery_create_ovstage_population(f.instance, &desc, &population)));
    CHECK(finishPopulationUpdate(f.instance, ovspatialquery_enqueue_ovstage_population_set(f.instance, population, 2),
                                 2, expected.size(), 0u));
    CHECK(selects(f, populationTarget(population), 2, expected));
    CHECK(succeeded(ovspatialquery_destroy_population(f.instance, population)));
    return 0;
}

int testMembershipRule()
{
    Fixture f;
    CHECK(openScene(f));
    // A zero scope selects everything the filter selects.
    CHECK(expectPopulation(f, Lists{}, {kC1, kC2, kC3, kC4, kC5, kC6}) == 0);
    // Excludes alone mean everything else; an exclude prunes its whole subtree.
    CHECK(expectPopulation(f, Lists{{}, paths(f, {"/World/A"})}, {kC4, kC5, kC6}) == 0);
    // An include below an exclude re-includes, to any depth.
    CHECK(expectPopulation(f, Lists{paths(f, {"/World/A", "/World/A/B/C"}), paths(f, {"/World/A/B"})}, {kC1, kC3}) ==
          0);
    // The same path in both lists is excluded.
    CHECK(expectPopulation(f, Lists{paths(f, {"/World/A"}), paths(f, {"/World/A"})}, {}) == 0);
    // Paths match by element: /World/Env does not cover /World/Environment.
    CHECK(expectPopulation(f, Lists{{}, paths(f, {"/World/Env"})}, {kC1, kC2, kC3, kC5, kC6}) == 0);
    // An exclude under no include changes nothing.
    CHECK(expectPopulation(f, Lists{paths(f, {"/World/A"}), paths(f, {"/World/Env"})}, {kC1, kC2, kC3}) == 0);
    // "/" is an entry in either list, and duplicates collapse.
    CHECK(expectPopulation(f, Lists{paths(f, {"/", "/"}), paths(f, {"/World/A/B"})}, {kC1, kC4, kC5, kC6}) == 0);
    CHECK(expectPopulation(f, Lists{paths(f, {"/World/Env"}), paths(f, {"/"})}, {kC4}) == 0);
    // A leaf prim is an entry like any other path.
    CHECK(expectPopulation(f, Lists{{}, paths(f, {kC6, kC2})}, {kC1, kC3, kC4, kC5}) == 0);
    CHECK(succeeded(ovspatialquery_destroy_instance(f.instance)));
    CHECK(ovstage_destroy_instance(f.stage) == OVSTAGE_OK);
    return 0;
}

// A view narrows its parent by its own scope; a scoped-out prim is not an exclusion even
// without a spatial source, and an include may name a prim that appears later.
int testViewsExclusionsAndLaterRoots()
{
    Fixture f;
    CHECK(openScene(f));
    // The population leaves out c6 and so never reports it, before or after it loses its bounds.
    const Lists                                    outer{{}, paths(f, {kC6})};
    const ovspatialquery_ovstage_population_desc_t desc       = populationDesc(f, outer.scope());
    ovspatialquery_population_handle_t             population = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(succeeded(ovspatialquery_create_ovstage_population(f.instance, &desc, &population)));
    const Lists                        inner{paths(f, {"/World/A", "/World/Later"}), paths(f, {"/World/A/B"})};
    ovspatialquery_ovstage_view_desc_t viewDesc{};
    viewDesc.scope                    = inner.scope();
    ovspatialquery_view_handle_t view = OVSPATIALQUERY_INVALID_VIEW;
    CHECK(succeeded(ovspatialquery_create_ovstage_view(f.instance, population, &viewDesc, &view)));
    CHECK(finishPopulationUpdate(f.instance, ovspatialquery_enqueue_ovstage_population_set(f.instance, population, 2),
                                 2, 5u, 0u));
    CHECK(selects(f, populationTarget(population), 2, {kC1, kC2, kC3, kC4, kC5}));
    CHECK(selects(f, viewTarget(view), 2, {kC1}));
    // Ovstage keeps no history, so ordinal 2 is published before 3 is written: c6 loses its
    // bounds at 3, and /World/Later appears then.
    double             absent[6]{absentBoundsValue(true),  absentBoundsValue(true),  absentBoundsValue(true),
                                 absentBoundsValue(false), absentBoundsValue(false), absentBoundsValue(false)};
    const ovx_string_t cubeName = {"Cube", 4u};
    ovx_token_t        cube     = OVX_INVALID_TOKEN;
    CHECK(dictionaryCallSucceeded(
        f.dictionary, f.dictionary->vtable->create_tokens_from_strings(f.dictionary->context, &cubeName, 1u, &cube)));
    double size = 1.0;
    double matrix[16]{1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 12.0, 0.0, 0.0, 1.0};
    double bounds[6]{11.5, -0.5, -0.5, 12.5, 0.5, 0.5};
    CHECK(writeBounds(f.stage, f.dictionary, kC6, sizeof(kC6) - 1u, 3u, absent));
    CHECK(writeAttribute(f.stage, f.dictionary, kLater, sizeof(kLater) - 1u, "usd-prim-type", 3u, &cube,
                         DLDataType{kDLUInt, 64u, 1u}));
    CHECK(writeAttribute(f.stage, f.dictionary, kLater, sizeof(kLater) - 1u, "size", 3u, &size,
                         DLDataType{kDLFloat, 64u, 1u}));
    CHECK(writeAttribute(f.stage, f.dictionary, kLater, sizeof(kLater) - 1u, "omni:fabric:worldMatrix", 3u, matrix,
                         DLDataType{kDLFloat, 64u, 16u}, 1u, OVSTAGE_SEMANTIC_MATRIX));
    CHECK(writeBounds(f.stage, f.dictionary, kLater, sizeof(kLater) - 1u, 3u, bounds));
    const ovstage_write_floor_desc_t floorThree{3u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
    CHECK(waitStageOperation(f.stage, ovstage_advance_write_floor(f.stage, &floorThree)));

    CHECK(finishPopulationUpdate(
        f.instance, ovspatialquery_enqueue_ovstage_population_update(f.instance, population, 2, 3), 3, 6u, 0u));
    CHECK(selects(f, populationTarget(population), 3, {kC1, kC2, kC3, kC4, kC5, kLater}));
    CHECK(selects(f, viewTarget(view), 3, {kC1, kLater}));
    CHECK(succeeded(ovspatialquery_destroy_view(f.instance, view)));
    CHECK(succeeded(ovspatialquery_destroy_population(f.instance, population)));

    // Without the scope, the same prim is reported as an exclusion.
    const ovspatialquery_ovstage_population_desc_t unscoped = populationDesc(f, Lists{}.scope());
    CHECK(succeeded(ovspatialquery_create_ovstage_population(f.instance, &unscoped, &population)));
    CHECK(finishPopulationUpdate(f.instance, ovspatialquery_enqueue_ovstage_population_set(f.instance, population, 3),
                                 3, 6u, 1u));
    CHECK(succeeded(ovspatialquery_destroy_population(f.instance, population)));
    CHECK(succeeded(ovspatialquery_destroy_instance(f.instance)));
    CHECK(ovstage_destroy_instance(f.stage) == OVSTAGE_OK);
    return 0;
}

// Content outside the scope never fails a publication, as if the filter had rejected it:
// here a Cube whose `size` is float32, a layout the adapter refuses, fails an unscoped
// publication and publishes nothing wrong when it is scoped out (on Ovstage 0.3 and later).
int testScopedOutContentNeverFails()
{
    Fixture f;
    CHECK(openScene(f));
    constexpr char     kOdd[]   = "/World/Odd/c8";
    const ovx_string_t cubeName = {"Cube", 4u};
    ovx_token_t        cube     = OVX_INVALID_TOKEN;
    CHECK(dictionaryCallSucceeded(
        f.dictionary, f.dictionary->vtable->create_tokens_from_strings(f.dictionary->context, &cubeName, 1u, &cube)));
    float  size = 1.0f;
    double matrix[16]{1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 14.0, 0.0, 0.0, 1.0};
    double bounds[6]{13.5, -0.5, -0.5, 14.5, 0.5, 0.5};
    CHECK(writeAttribute(f.stage, f.dictionary, kOdd, sizeof(kOdd) - 1u, "usd-prim-type", 3u, &cube,
                         DLDataType{kDLUInt, 64u, 1u}));
    CHECK(writeAttribute(f.stage, f.dictionary, kOdd, sizeof(kOdd) - 1u, "size", 3u, &size,
                         DLDataType{kDLFloat, 32u, 1u}));
    CHECK(writeAttribute(f.stage, f.dictionary, kOdd, sizeof(kOdd) - 1u, "omni:fabric:worldMatrix", 3u, matrix,
                         DLDataType{kDLFloat, 64u, 16u}, 1u, OVSTAGE_SEMANTIC_MATRIX));
    CHECK(writeBounds(f.stage, f.dictionary, kOdd, sizeof(kOdd) - 1u, 3u, bounds));
    const ovstage_write_floor_desc_t floorThree{3u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
    CHECK(waitStageOperation(f.stage, ovstage_advance_write_floor(f.stage, &floorThree)));

    const auto refuses = [&](ovspatialquery_population_handle_t population)
    {
        ovspatialquery_enqueue_result_t operation =
            ovspatialquery_enqueue_ovstage_population_set(f.instance, population, 3);
        if (operation.status != OVSPATIALQUERY_API_SUCCESS)
            return true;
        ovspatialquery_op_wait_result_t waited{};
        const bool                      completed =
            succeeded(ovspatialquery_wait_op(f.instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited)) &&
            waited.state == OVSPATIALQUERY_OP_COMPLETED;
        return succeeded(ovspatialquery_destroy_op(f.instance, &operation.op_id)) && !completed;
    };
    const ovspatialquery_ovstage_population_desc_t unscoped   = populationDesc(f, Lists{}.scope());
    ovspatialquery_population_handle_t             population = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(succeeded(ovspatialquery_create_ovstage_population(f.instance, &unscoped, &population)));
    CHECK(refuses(population));
    CHECK(succeeded(ovspatialquery_destroy_population(f.instance, population)));

    // An Ovstage 0.2 runtime still reads scoped-out attributes (the header's documented
    // limit), so there the scoped population is refused like the unscoped one.
    uint32_t major = 0u, minor = 0u, patch = 0u;
    stageVtable(f.stage)->get_version(stageContext(f.stage), &major, &minor, &patch);
    const Lists                                    outside{{}, paths(f, {"/World/Odd"})};
    const ovspatialquery_ovstage_population_desc_t scoped = populationDesc(f, outside.scope());
    CHECK(succeeded(ovspatialquery_create_ovstage_population(f.instance, &scoped, &population)));
    if (major == 0u && minor < 3u)
    {
        CHECK(refuses(population));
    }
    else
    {
        CHECK(finishPopulationUpdate(
            f.instance, ovspatialquery_enqueue_ovstage_population_set(f.instance, population, 3), 3, 6u, 0u));
        CHECK(selects(f, populationTarget(population), 3, {kC1, kC2, kC3, kC4, kC5, kC6}));
    }
    CHECK(succeeded(ovspatialquery_destroy_population(f.instance, population)));
    CHECK(succeeded(ovspatialquery_destroy_instance(f.instance)));
    CHECK(ovstage_destroy_instance(f.stage) == OVSTAGE_OK);
    return 0;
}

// Malformed lists return INVALID_ARGUMENT and create nothing.
int testMalformedScopes()
{
    Fixture f;
    CHECK(openScene(f));
    const ovx_primpath_t                                a    = path(f, "/World/A");
    const ovx_primpath_t                                zero = OVX_INVALID_PRIMPATH;
    const std::array<ovspatialquery_ovstage_scope_t, 4> malformed{{
        {&a, nullptr, 0u, 0u},      // a pointer with a zero count
        {nullptr, nullptr, 1u, 0u}, // a count without a pointer
        {nullptr, &a, 0u, 0u},
        {&zero, nullptr, 1u, 0u}, // a zero entry
    }};
    for (const ovspatialquery_ovstage_scope_t& scope : malformed)
    {
        const ovspatialquery_ovstage_population_desc_t desc       = populationDesc(f, scope);
        ovspatialquery_population_handle_t             population = OVSPATIALQUERY_INVALID_POPULATION;
        CHECK(ovspatialquery_create_ovstage_population(f.instance, &desc, &population).status ==
              OVSPATIALQUERY_API_INVALID_ARGUMENT);
        CHECK(population == OVSPATIALQUERY_INVALID_POPULATION);
    }
    const ovspatialquery_ovstage_population_desc_t desc       = populationDesc(f, Lists{}.scope());
    ovspatialquery_population_handle_t             population = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(succeeded(ovspatialquery_create_ovstage_population(f.instance, &desc, &population)));
    for (const ovspatialquery_ovstage_scope_t& scope : malformed)
    {
        ovspatialquery_ovstage_view_desc_t viewDesc{};
        viewDesc.scope                    = scope;
        ovspatialquery_view_handle_t view = OVSPATIALQUERY_INVALID_VIEW;
        CHECK(ovspatialquery_create_ovstage_view(f.instance, population, &viewDesc, &view).status ==
              OVSPATIALQUERY_API_INVALID_ARGUMENT);
        CHECK(view == OVSPATIALQUERY_INVALID_VIEW);
    }
    CHECK(succeeded(ovspatialquery_destroy_population(f.instance, population)));
    CHECK(succeeded(ovspatialquery_destroy_instance(f.instance)));
    CHECK(ovstage_destroy_instance(f.stage) == OVSTAGE_OK);
    return 0;
}

} // namespace

int main()
{
    CHECK(testMembershipRule() == 0);
    CHECK(testViewsExclusionsAndLaterRoots() == 0);
    CHECK(testScopedOutContentNeverFails() == 0);
    CHECK(testMalformedScopes() == 0);
    return 0;
}
