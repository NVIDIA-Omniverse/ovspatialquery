/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// An Ovstage population reports the path value of each prim's bound physics material on its
// hits (ADR-0066): its own `material:binding:physics`, an ancestor's, then the all-purpose
// `material:binding`, or zero. A mesh face in a `materialBind` face GeomSubset with its own
// binding reports the subset's material, a face in no such subset (another family included)
// the mesh's, and an index past the mesh's faces is ignored. A binding
// written at a later ordinal appears in the next publication, and the retained one keeps the
// material it was published with. Material is optional: a subset's indices written as
// `int64`, which the reads do not take, add no face material and do not fail the
// publication. Resolution reads only the subsets of the population's meshes, and charges what
// it copies to the host budget: a large subset of another prim costs nothing, and one of a
// published mesh that does not fit refuses the publication with HOST_BUDGET. An Ovstage 0.2
// runtime reads no relationship through a path-list query, so there every hit reports zero.
// A subset created through the write API after population is found from its path and
// reports its face material.
#include <ovspatialquery/ovspatialquery_ovstage.h>

#include <ovstage/ovstage.h>
#include <ovx/path_dictionary/path_dictionary.h>

#include "../../support/native/ovstage_companion_fixtures.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace
{

const char kMaterialStage[] = R"usda(#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
    metersPerUnit = 1
)
def Xform "World"
{
    def Material "Rubber" (prepend apiSchemas = ["PhysicsMaterialAPI"])
    {
    }
    def Material "Steel" (prepend apiSchemas = ["PhysicsMaterialAPI"])
    {
    }
    def Material "Ice" (prepend apiSchemas = ["PhysicsMaterialAPI"])
    {
    }
    def Material "Visual"
    {
    }
    def Xform "Group" (prepend apiSchemas = ["MaterialBindingAPI"])
    {
        rel material:binding:physics = </World/Steel>
        def Cube "Inherited"
        {
            double size = 1
        }
    }
    def Cube "Own" (prepend apiSchemas = ["MaterialBindingAPI"])
    {
        double size = 1
        rel material:binding = </World/Visual>
        rel material:binding:physics = </World/Rubber>
        double3 xformOp:translate = (5, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def Cube "VisualOnly" (prepend apiSchemas = ["MaterialBindingAPI"])
    {
        double size = 1
        rel material:binding = </World/Visual>
        double3 xformOp:translate = (10, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def Cube "Unbound"
    {
        double size = 1
        double3 xformOp:translate = (15, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def GeomSubset "Bulk" (prepend apiSchemas = ["MaterialBindingAPI"])
        {
            uniform token elementType = "face"
            uniform token familyName = "materialBind"
            int[] indices = [0]
            rel material:binding:physics = </World/Ice>
        }
    }
    def Mesh "Floor" (prepend apiSchemas = ["MaterialBindingAPI"])
    {
        uniform bool doubleSided = 1
        int[] faceVertexCounts = [4, 4, 4]
        int[] faceVertexIndices = [0, 4, 5, 1, 1, 5, 6, 2, 2, 6, 7, 3]
        point3f[] points = [(0, 0, 20), (1, 0, 20), (2, 0, 20), (3, 0, 20), (0, 0, 21), (1, 0, 21), (2, 0, 21), (3, 0, 21)]
        float3[] extent = [(0, 0, 20), (3, 0, 21)]
        rel material:binding:physics = </World/Rubber>
        def GeomSubset "IceFaces" (prepend apiSchemas = ["MaterialBindingAPI"])
        {
            uniform token elementType = "face"
            uniform token familyName = "materialBind"
            int[] indices = [1, 2147483647]
            rel material:binding:physics = </World/Ice>
        }
        def GeomSubset "WideFaces" (prepend apiSchemas = ["MaterialBindingAPI"])
        {
            uniform token elementType = "face"
            rel material:binding:physics = </World/Steel>
        }
        def GeomSubset "SteelFaces" (prepend apiSchemas = ["MaterialBindingAPI"])
        {
            uniform token elementType = "face"
            uniform token familyName = "materialBind"
            int[] indices = [2]
            rel material:binding:physics = </World/Steel>
        }
        def GeomSubset "Selection" (prepend apiSchemas = ["MaterialBindingAPI"])
        {
            uniform token elementType = "face"
            uniform token familyName = "selection"
            int[] indices = [0]
            rel material:binding:physics = </World/Ice>
        }
        def GeomSubset "Partition"
        {
            uniform token elementType = "face"
            uniform token familyName = "partition"
            int[] indices = [0]
        }
    }
}
)usda";

// The material stage, with `extraSubsets` more physics-bound materialBind subsets on a mesh
// "Other" that a population can leave out.
std::string materialStage(int extraSubsets)
{
    std::string usda(kMaterialStage, sizeof(kMaterialStage) - 1u);
    if (extraSubsets == 0)
        return usda;
    std::string other = "    def Mesh \"Other\" (prepend apiSchemas = [\"MaterialBindingAPI\"])\n    {\n"
                        "        int[] faceVertexCounts = [4]\n        int[] faceVertexIndices = [0, 1, 2, 3]\n"
                        "        point3f[] points = [(0, 0, 40), (1, 0, 40), (1, 0, 41), (0, 0, 41)]\n"
                        "        float3[] extent = [(0, 0, 40), (1, 0, 41)]\n";
    for (int subset = 0; subset < extraSubsets; ++subset)
    {
        other += "        def GeomSubset \"S" + std::to_string(subset) +
                 "\" (prepend apiSchemas = [\"MaterialBindingAPI\"])\n        {\n"
                 "            uniform token elementType = \"face\"\n"
                 "            uniform token familyName = \"materialBind\"\n            int[] indices = [0]\n"
                 "            rel material:binding:physics = </World/Steel>\n        }\n";
    }
    other += "    }\n";
    usda.insert(usda.rfind('}'), other);
    return usda;
}

bool populateMaterialStage(ovstage_instance_t* stage, int extraSubsets = 0)
{
    const std::string                         usda = materialStage(extraSubsets);
    const ovstage_population_enqueue_result_t population =
        openUsdWithPhysics(stage, ovx_string_t{usda.data(), usda.size()}, 1u, 0.0);
    ovstage_population_op_wait_result_t populationWait{};
    if (population.status != OVSTAGE_OK ||
        ovstage_population_wait_op(stage, population.op_index, OVSTAGE_TIMEOUT_INFINITE, &populationWait) != OVSTAGE_OK)
    {
        return false;
    }
    const ovstage_write_floor_desc_t floorOne{1u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
    return waitStageOperation(stage, ovstage_advance_write_floor(stage, &floorOne)) &&
           waitStageOperation(
               stage, ovstage_compute_hierarchy(stage, OVSTAGE_HIERARCHY_COMPUTATION_MODEL_DEFAULT_CPU, 1u, 2u));
}

// The material of the closest hit straight down onto (x, z), or UINT64_MAX when there is none.
uint64_t materialBelow(ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population,
                       ovspatialquery_query_precision_t precision, double x, double z,
                       ovspatialquery_ordinal_t ordinal = -1)
{
    ovspatialquery_raycast_desc_t ray{};
    ray.common.target    = populationTarget(population);
    ray.common.precision = precision;
    if (ordinal >= 0)
    {
        ray.common.target.ordinal.kind           = OVSPATIALQUERY_ORDINAL_RETAINED_EXACT;
        ray.common.target.ordinal.source_ordinal = ordinal;
    }
    ray.ray.origin[0]         = x;
    ray.ray.origin[1]         = 10.0;
    ray.ray.origin[2]         = z;
    ray.ray.unit_direction[1] = -1.0f;
    ray.ray.max_distance      = 20.0;
    ovspatialquery_raycast_single_result_t closest{};
    if (!succeeded(ovspatialquery_raycast_closest_sync(instance, &ray, &closest)) || closest.has_hit != 1u)
    {
        return UINT64_MAX;
    }
    return precision == OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE ? closest.hit.data.bounds_candidate.identity.material_id
                                                              : closest.hit.data.exact.identity.material_id;
}

int testOvstageMaterials()
{
    ovstage_instance_t*           stage = nullptr;
    const ovstage_instance_desc_t stageDesc{"ovspatialquery-material-test"};
    CHECK(ovstage_create_instance(&stageDesc, &stage) == OVSTAGE_OK);
    path_dictionary_instance_t* dictionary = ovstage_get_path_dictionary(stage);
    CHECK(dictionary != nullptr);
    CHECK(populateMaterialStage(stage));

    static const char  rubberPath[]    = "/World/Rubber";
    static const char  steelPath[]     = "/World/Steel";
    static const char  icePath[]       = "/World/Ice";
    static const char  visualPath[]    = "/World/Visual";
    const ovx_string_t materialPaths[] = {{rubberPath, sizeof(rubberPath) - 1u},
                                          {steelPath, sizeof(steelPath) - 1u},
                                          {icePath, sizeof(icePath) - 1u},
                                          {visualPath, sizeof(visualPath) - 1u}};
    ovx_primpath_t     materials[4]    = {};
    CHECK(dictionaryCallSucceeded(
        dictionary, dictionary->vtable->create_paths_from_strings(dictionary->context, materialPaths, 4u, materials)));
    const uint64_t rubber = handleValue(materials[0]), steel = handleValue(materials[1]);
    const uint64_t ice = handleValue(materials[2]), visual = handleValue(materials[3]);

    const ovspatialquery_instance_desc_t instanceDesc = commonInstanceDesc();
    ovspatialquery_instance_handle_t     instance     = OVSPATIALQUERY_INVALID_INSTANCE;
    CHECK(succeeded(ovspatialquery_create_instance(&instanceDesc, &instance)));
    ovspatialquery_ovstage_population_desc_t sourceDesc{};
    sourceDesc.stage                              = stage;
    sourceDesc.common                             = aabbPopulationDesc();
    ovspatialquery_population_handle_t population = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &sourceDesc, &population)));
    CHECK(finishPopulationUpdate(instance, ovspatialquery_enqueue_ovstage_population_set(instance, population, 1), 1,
                                 5u, 0u));

    for (const ovspatialquery_query_precision_t precision :
         {OVSPATIALQUERY_QUERY_EXACT, OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE})
    {
        CHECK(materialBelow(instance, population, precision, 0.0, 0.0) == steel);  // an ancestor's physics binding
        CHECK(materialBelow(instance, population, precision, 5.0, 0.0) == rubber); // physics before all-purpose
        CHECK(materialBelow(instance, population, precision, 10.0, 0.0) == visual);
        CHECK(materialBelow(instance, population, precision, 15.0, 0.0) == 0u);
    }
    // Faces 1 and 2 are in materialBind subsets; face 0, in a subset of another family only,
    // keeps the mesh's material, and so does a bounds candidate.
    CHECK(materialBelow(instance, population, OVSPATIALQUERY_QUERY_EXACT, 0.5, 20.5) == rubber);
    CHECK(materialBelow(instance, population, OVSPATIALQUERY_QUERY_EXACT, 1.5, 20.5) == ice);
    CHECK(materialBelow(instance, population, OVSPATIALQUERY_QUERY_EXACT, 2.5, 20.5) == steel);
    CHECK(materialBelow(instance, population, OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE, 1.5, 20.5) == rubber);

    // A physics binding written at ordinal 2 appears in publication 2; publication 1 keeps its own.
    // WideFaces' int64 indices are not read, so publication 2 still succeeds and face 0 keeps
    // the mesh's material. (Ovstage keeps an int32 column for a materialBind subset's indices,
    // so the int64 ones go to a face subset without a family, which also qualifies.)
    uint64_t iceTarget = handleValue(materials[2]);
    CHECK(writeAttribute(stage, dictionary, "/World/VisualOnly", sizeof("/World/VisualOnly") - 1u,
                         "material:binding:physics", 2u, &iceTarget, DLDataType{kDLUInt, 64u, 1u}, 1u,
                         OVSTAGE_SEMANTIC_RELATIONSHIP_PATH_ID, true));
    int64_t wideIndices[] = {0};
    CHECK(writeAttribute(stage, dictionary, "/World/Floor/WideFaces", sizeof("/World/Floor/WideFaces") - 1u, "indices",
                         2u, wideIndices, DLDataType{kDLInt, 64u, 1u}, 1u, OVSTAGE_SEMANTIC_NONE, true));
    const ovstage_write_floor_desc_t floorTwo{2u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
    CHECK(waitStageOperation(stage, ovstage_advance_write_floor(stage, &floorTwo)));
    CHECK(finishPopulationUpdate(instance, ovspatialquery_enqueue_ovstage_population_update(instance, population, 1, 2),
                                 2, 5u, 0u));
    CHECK(materialBelow(instance, population, OVSPATIALQUERY_QUERY_EXACT, 10.0, 0.0) == ice);
    CHECK(materialBelow(instance, population, OVSPATIALQUERY_QUERY_EXACT, 10.0, 0.0, 1) == visual);
    CHECK(materialBelow(instance, population, OVSPATIALQUERY_QUERY_EXACT, 0.5, 20.5) == rubber);

    CHECK(succeeded(ovspatialquery_destroy_population(instance, population)));

    CHECK(succeeded(ovspatialquery_destroy_instance(instance)));
    CHECK(ovstage_destroy_instance(stage) == OVSTAGE_OK);
    return 0;
}

// A face subset created through the write API at ordinal 2, under the published mesh, binds
// its face 0 to Steel: publication 2 reports Steel there, and the retained publication 1 the
// mesh's Rubber. Its parent comes from its path, so a populator-derived `usd-parent` that
// omits such a prim does not hide it.
int testLaterSubsetReportsItsMaterial()
{
    ovstage_instance_t*           stage = nullptr;
    const ovstage_instance_desc_t stageDesc{"ovspatialquery-material-later-subset-test"};
    CHECK(ovstage_create_instance(&stageDesc, &stage) == OVSTAGE_OK);
    path_dictionary_instance_t* dictionary = ovstage_get_path_dictionary(stage);
    CHECK(dictionary != nullptr);
    CHECK(populateMaterialStage(stage));

    const ovspatialquery_instance_desc_t instanceDesc = commonInstanceDesc();
    ovspatialquery_instance_handle_t     instance     = OVSPATIALQUERY_INVALID_INSTANCE;
    CHECK(succeeded(ovspatialquery_create_instance(&instanceDesc, &instance)));
    ovspatialquery_ovstage_population_desc_t sourceDesc{};
    sourceDesc.stage                              = stage;
    sourceDesc.common                             = aabbPopulationDesc();
    ovspatialquery_population_handle_t population = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &sourceDesc, &population)));
    CHECK(finishPopulationUpdate(instance, ovspatialquery_enqueue_ovstage_population_set(instance, population, 1), 1,
                                 5u, 0u));
    const uint64_t rubber = materialBelow(instance, population, OVSPATIALQUERY_QUERY_EXACT, 0.5, 20.5);
    const uint64_t steel  = materialBelow(instance, population, OVSPATIALQUERY_QUERY_EXACT, 2.5, 20.5);
    CHECK(rubber != steel);

    static const char  later[]    = "/World/Floor/Later";
    const ovx_string_t names[]    = {{"GeomSubset", sizeof("GeomSubset") - 1u},
                                     {"/World/Steel", sizeof("/World/Steel") - 1u}};
    ovx_token_t        subsetType = OVX_INVALID_TOKEN;
    ovx_primpath_t     steelPath  = OVX_INVALID_PRIMPATH;
    CHECK(dictionaryCallSucceeded(
        dictionary, dictionary->vtable->create_tokens_from_strings(dictionary->context, &names[0], 1u, &subsetType)));
    CHECK(dictionaryCallSucceeded(
        dictionary, dictionary->vtable->create_paths_from_strings(dictionary->context, &names[1], 1u, &steelPath)));
    uint64_t steelTarget = handleValue(steelPath);
    int32_t  indices[]   = {0};
    CHECK(writeAttribute(stage, dictionary, later, sizeof(later) - 1u, "usd-prim-type", 2u, &subsetType,
                         DLDataType{kDLUInt, 64u, 1u}));
    CHECK(writeAttribute(stage, dictionary, later, sizeof(later) - 1u, "indices", 2u, indices,
                         DLDataType{kDLInt, 32u, 1u}, 1u, OVSTAGE_SEMANTIC_NONE, true));
    CHECK(writeAttribute(stage, dictionary, later, sizeof(later) - 1u, "material:binding:physics", 2u, &steelTarget,
                         DLDataType{kDLUInt, 64u, 1u}, 1u, OVSTAGE_SEMANTIC_RELATIONSHIP_PATH_ID, true));
    const ovstage_write_floor_desc_t floorTwo{2u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
    CHECK(waitStageOperation(stage, ovstage_advance_write_floor(stage, &floorTwo)));
    CHECK(finishPopulationUpdate(instance, ovspatialquery_enqueue_ovstage_population_update(instance, population, 1, 2),
                                 2, 5u, 0u));
    CHECK(materialBelow(instance, population, OVSPATIALQUERY_QUERY_EXACT, 0.5, 20.5) == steel);
    CHECK(materialBelow(instance, population, OVSPATIALQUERY_QUERY_EXACT, 0.5, 20.5, 1) == rubber);

    CHECK(succeeded(ovspatialquery_destroy_population(instance, population)));
    CHECK(succeeded(ovspatialquery_destroy_instance(instance)));
    CHECK(ovstage_destroy_instance(stage) == OVSTAGE_OK);
    return 0;
}

// Writes `indices` to the subset at `path` at `ordinal`, then seals the ordinal.
bool writeBulkIndices(ovstage_instance_t* stage, path_dictionary_instance_t* dictionary, const char* path,
                      ovstage_ordinal_t ordinal, std::vector<int32_t>& indices)
{
    const ovstage_write_floor_desc_t floor{ordinal, OVSTAGE_SCOPE_ALL, nullptr, 0u};
    return writeAttribute(stage, dictionary, path, std::strlen(path), "indices", ordinal, indices.data(),
                          DLDataType{kDLInt, 32u, 1u}, indices.size(), OVSTAGE_SEMANTIC_NONE, true) &&
           waitStageOperation(stage, ovstage_advance_write_floor(stage, &floor));
}

int testMaterialScratchIsCharged()
{
    ovstage_instance_t*           stage = nullptr;
    const ovstage_instance_desc_t stageDesc{"ovspatialquery-material-budget-test"};
    CHECK(ovstage_create_instance(&stageDesc, &stage) == OVSTAGE_OK);
    path_dictionary_instance_t* dictionary = ovstage_get_path_dictionary(stage);
    CHECK(dictionary != nullptr);
    CHECK(populateMaterialStage(stage));

    ovspatialquery_instance_desc_t instanceDesc = commonInstanceDesc();
    instanceDesc.host_byte_budget               = UINT64_C(16) << 20u;
    ovspatialquery_instance_handle_t instance   = OVSPATIALQUERY_INVALID_INSTANCE;
    CHECK(succeeded(ovspatialquery_create_instance(&instanceDesc, &instance)));
    ovspatialquery_ovstage_population_desc_t sourceDesc{};
    sourceDesc.stage                              = stage;
    sourceDesc.common                             = aabbPopulationDesc();
    ovspatialquery_population_handle_t population = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &sourceDesc, &population)));
    CHECK(finishPopulationUpdate(instance, ovspatialquery_enqueue_ovstage_population_set(instance, population, 1), 1,
                                 5u, 0u));
    const uint64_t ice = materialBelow(instance, population, OVSPATIALQUERY_QUERY_EXACT, 1.5, 20.5);

    // 24 MiB of indices, more than the whole budget, on a subset of a Cube, which no mesh owns.
    std::vector<int32_t> bulk(size_t{6} << 20u, 0);
    CHECK(writeBulkIndices(stage, dictionary, "/World/Unbound/Bulk", 2u, bulk));
    CHECK(finishPopulationUpdate(instance, ovspatialquery_enqueue_ovstage_population_update(instance, population, 1, 2),
                                 2, 5u, 0u));
    CHECK(materialBelow(instance, population, OVSPATIALQUERY_QUERY_EXACT, 1.5, 20.5) == ice);

    // The same indices on a subset of the published mesh that binds no material cannot give
    // a face one, so they are not read (ADR-0066) and the update fits.
    CHECK(writeBulkIndices(stage, dictionary, "/World/Floor/Partition", 3u, bulk));
    CHECK(finishPopulationUpdate(instance, ovspatialquery_enqueue_ovstage_population_update(instance, population, 2, 3),
                                 3, 5u, 0u));
    CHECK(materialBelow(instance, population, OVSPATIALQUERY_QUERY_EXACT, 1.5, 20.5) == ice);

    // On a bound materialBind subset of the published mesh they are read, and do not fit:
    // the update is refused when it captures the stage, and leaves nothing charged.
    uint64_t usageBefore = 0u;
    CHECK(succeeded(ovspatialquery_get_instance_memory_usage(instance, DLDevice{kDLCPU, 0}, &usageBefore)));
    CHECK(writeBulkIndices(stage, dictionary, "/World/Floor/SteelFaces", 4u, bulk));
    const ovspatialquery_enqueue_result_t update =
        ovspatialquery_enqueue_ovstage_population_update(instance, population, 3, 4);
    CHECK(update.status == OVSPATIALQUERY_API_HOST_BUDGET);
    uint64_t usageAfter = 0u;
    CHECK(succeeded(ovspatialquery_get_instance_memory_usage(instance, DLDevice{kDLCPU, 0}, &usageAfter)));
    CHECK(usageAfter == usageBefore);
    CHECK(materialBelow(instance, population, OVSPATIALQUERY_QUERY_EXACT, 1.5, 20.5) == ice);

    CHECK(succeeded(ovspatialquery_destroy_population(instance, population)));
    CHECK(succeeded(ovspatialquery_destroy_instance(instance)));
    CHECK(ovstage_destroy_instance(stage) == OVSTAGE_OK);
    return 0;
}

} // namespace

// A population of the mesh alone has two object and ancestor paths, fewer than half the
// stage's physics-bound prims, so the adapter reads that binding for its own paths only
// (ADR-0066). The stage also holds 80 bound subsets of a mesh outside the population, more
// than the earlier bound on read subsets (twice the paths plus 64 per mesh) admitted; the
// mesh's own subsets still report their materials, because subsets are found by enumerating
// GeomSubset paths, not bindings.
int testScopedBindingReadMatches()
{
    ovstage_instance_t*           stage = nullptr;
    const ovstage_instance_desc_t stageDesc{"ovspatialquery-material-scoped-test"};
    CHECK(ovstage_create_instance(&stageDesc, &stage) == OVSTAGE_OK);
    path_dictionary_instance_t* dictionary = ovstage_get_path_dictionary(stage);
    CHECK(dictionary != nullptr);
    CHECK(populateMaterialStage(stage, 80));
    static const char  rubberPath[]    = "/World/Rubber";
    static const char  steelPath[]     = "/World/Steel";
    static const char  icePath[]       = "/World/Ice";
    const ovx_string_t materialPaths[] = {
        {rubberPath, sizeof(rubberPath) - 1u}, {steelPath, sizeof(steelPath) - 1u}, {icePath, sizeof(icePath) - 1u}};
    ovx_primpath_t materials[3] = {};
    CHECK(dictionaryCallSucceeded(
        dictionary, dictionary->vtable->create_paths_from_strings(dictionary->context, materialPaths, 3u, materials)));

    static const char   floorPath[]  = "/World/Floor";
    const ovx_string_t  floorPaths[] = {{floorPath, sizeof(floorPath) - 1u}};
    ovstage_predicate_t floorPredicate{};
    floorPredicate.attribute.string                   = ovx_string_t{"usd-path", sizeof("usd-path") - 1u};
    floorPredicate.op                                 = OVSTAGE_FILTER_OP_IN;
    floorPredicate.values                             = floorPaths;
    floorPredicate.value_count                        = 1u;
    const ovspatialquery_instance_desc_t instanceDesc = commonInstanceDesc();
    ovspatialquery_instance_handle_t     instance     = OVSPATIALQUERY_INVALID_INSTANCE;
    CHECK(succeeded(ovspatialquery_create_instance(&instanceDesc, &instance)));
    ovspatialquery_ovstage_population_desc_t sourceDesc{};
    sourceDesc.stage                              = stage;
    sourceDesc.filter                             = ovstage_filter_t{&floorPredicate, 1u};
    sourceDesc.common                             = aabbPopulationDesc();
    ovspatialquery_population_handle_t population = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &sourceDesc, &population)));
    CHECK(finishPopulationUpdate(instance, ovspatialquery_enqueue_ovstage_population_set(instance, population, 1), 1,
                                 1u, 0u));
    CHECK(materialBelow(instance, population, OVSPATIALQUERY_QUERY_EXACT, 0.5, 20.5) == handleValue(materials[0]));
    CHECK(materialBelow(instance, population, OVSPATIALQUERY_QUERY_EXACT, 1.5, 20.5) == handleValue(materials[2]));
    CHECK(materialBelow(instance, population, OVSPATIALQUERY_QUERY_EXACT, 2.5, 20.5) == handleValue(materials[1]));
    CHECK(succeeded(ovspatialquery_destroy_population(instance, population)));
    CHECK(succeeded(ovspatialquery_destroy_instance(instance)));
    CHECK(ovstage_destroy_instance(stage) == OVSTAGE_OK);
    return 0;
}

int main()
{
    CHECK(testOvstageMaterials() == 0);
    CHECK(testLaterSubsetReportsItsMaterial() == 0);
    CHECK(testMaterialScratchIsCharged() == 0);
    CHECK(testScopedBindingReadMatches() == 0);
    std::puts("ovstage material tests passed");
    return 0;
}
