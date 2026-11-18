/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <ovspatialquery/ovspatialquery_ovstage.h>

#include <ovstage/ovstage.h>
#include <ovx/path_dictionary/path_dictionary.h>

#include "../../support/native/ovstage_companion_fixtures.h"

#include <cstdint>
#include <string>
#include <type_traits>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif

// The public Ovstage entry points keep these exact C signatures.
static_assert(std::is_same<decltype(&ovspatialquery_create_ovstage_population),
                           ovspatialquery_result_t (*)(ovspatialquery_instance_handle_t,
                                                       const ovspatialquery_ovstage_population_desc_t*,
                                                       ovspatialquery_population_handle_t*)>::value,
              "Ovstage population signature");
static_assert(std::is_same<decltype(&ovspatialquery_create_ovstage_view),
                           ovspatialquery_result_t (*)(
                               ovspatialquery_instance_handle_t, ovspatialquery_population_handle_t,
                               const ovspatialquery_ovstage_view_desc_t*, ovspatialquery_view_handle_t*)>::value,
              "Ovstage view signature");
static_assert(
    std::is_same<decltype(&ovspatialquery_set_ovstage_participation_map),
                 ovspatialquery_result_t (*)(ovspatialquery_instance_handle_t, ovspatialquery_population_handle_t,
                                             const ovspatialquery_ovstage_participation_table_t*)>::value,
    "Ovstage participation table signature");
static_assert(std::is_same<decltype(&ovspatialquery_enqueue_ovstage_population_update),
                           ovspatialquery_enqueue_result_t (*)(
                               ovspatialquery_instance_handle_t, ovspatialquery_population_handle_t,
                               ovspatialquery_ordinal_t, ovspatialquery_ordinal_t)>::value,
              "Ovstage population update signature");

namespace
{

path_dictionary_instance_t* gDictionary = nullptr;

uint32_t gReportedMajor = 0u;
uint32_t gReportedMinor = 0u;

void reportedVersion(ovstage_data_plane_context_t*, uint32_t* major, uint32_t* minor, uint32_t* patch)
{
    *major = gReportedMajor;
    *minor = gReportedMinor;
    *patch = 7u;
}

path_dictionary_instance_t* fakeDictionary(ovstage_data_plane_context_t*) { return gDictionary; }

bool rejectedInstance(const ovspatialquery_instance_desc_t* desc, ovspatialquery_api_status_t expected)
{
    ovspatialquery_instance_handle_t output = ovspatialquery_instance_handle_t{UINT64_C(0xffffffffffffffff)};
    return ovspatialquery_create_instance(desc, &output).status == expected &&
           output == OVSPATIALQUERY_INVALID_INSTANCE;
}

bool rejectedOvstagePopulation(ovspatialquery_instance_handle_t                instance,
                               const ovspatialquery_ovstage_population_desc_t* desc,
                               ovspatialquery_api_status_t                     expected)
{
    ovspatialquery_population_handle_t output = ovspatialquery_population_handle_t{UINT64_C(0xffffffffffffffff)};
    return ovspatialquery_create_ovstage_population(instance, desc, &output).status == expected &&
           output == OVSPATIALQUERY_INVALID_POPULATION;
}

int testInstanceValidation(ovstage_instance_t* stage, path_dictionary_instance_t* dictionary)
{
    const ovspatialquery_instance_desc_t desc = commonInstanceDesc();

    ovspatialquery_instance_handle_t exactInstance = OVSPATIALQUERY_INVALID_INSTANCE;
    CHECK(succeeded(ovspatialquery_create_instance(&desc, &exactInstance)));
    uint64_t fixedCharge = 0u;
    CHECK(readHostBytes(exactInstance, fixedCharge));
    CHECK(fixedCharge != 0u);
    CHECK(succeeded(ovspatialquery_destroy_instance(exactInstance)));
    exactInstance                            = ovspatialquery_instance_handle_t{UINT64_C(0xabcdef)};
    ovspatialquery_instance_desc_t exactDesc = desc;
    exactDesc.host_byte_budget               = fixedCharge - 1u;
    CHECK(ovspatialquery_create_instance(&exactDesc, &exactInstance).status == OVSPATIALQUERY_API_HOST_BUDGET);
    CHECK(exactInstance == OVSPATIALQUERY_INVALID_INSTANCE);
    exactDesc.host_byte_budget = fixedCharge;
    CHECK(succeeded(ovspatialquery_create_instance(&exactDesc, &exactInstance)));
    uint64_t exactUsage = 0u;
    CHECK(readHostBytes(exactInstance, exactUsage));
    CHECK(exactUsage == fixedCharge);
    CHECK(succeeded(ovspatialquery_destroy_instance(exactInstance)));

    CHECK(rejectedInstance(nullptr, OVSPATIALQUERY_API_INVALID_ARGUMENT));
    CHECK(ovspatialquery_create_instance(&desc, nullptr).status == OVSPATIALQUERY_API_INVALID_ARGUMENT);

    ovspatialquery_instance_desc_t malformed = desc;
    malformed.operation_capacity             = 0u;
    CHECK(rejectedInstance(&malformed, OVSPATIALQUERY_API_INVALID_ARGUMENT));
    malformed                     = desc;
    malformed.max_cpu_concurrency = 0u;
    CHECK(rejectedInstance(&malformed, OVSPATIALQUERY_API_INVALID_ARGUMENT));
    malformed                  = desc;
    malformed.host_byte_budget = 0u;
    CHECK(rejectedInstance(&malformed, OVSPATIALQUERY_API_INVALID_ARGUMENT));
    malformed                     = desc;
    malformed.device_budget_count = 1u;
    CHECK(rejectedInstance(&malformed, OVSPATIALQUERY_API_INVALID_ARGUMENT));

    ovspatialquery_instance_handle_t instance = OVSPATIALQUERY_INVALID_INSTANCE;
    CHECK(succeeded(ovspatialquery_create_instance(&desc, &instance)));

    ovstage_predicate_t predicate{};
    predicate.attribute.string = ovx_string_t{"_worldExtent", sizeof("_worldExtent") - 1u};
    predicate.op               = OVSTAGE_FILTER_OP_HAS;
    ovspatialquery_ovstage_population_desc_t sourceDesc{};
    sourceDesc.stage  = stage;
    sourceDesc.filter = ovstage_filter_t{&predicate, 1u};
    sourceDesc.common = aabbPopulationDesc();

    CHECK(rejectedOvstagePopulation(instance, nullptr, OVSPATIALQUERY_API_INVALID_ARGUMENT));
    ovspatialquery_ovstage_population_desc_t malformedSource = sourceDesc;
    malformedSource.stage                                    = nullptr;
    CHECK(rejectedOvstagePopulation(instance, &malformedSource, OVSPATIALQUERY_API_INVALID_ARGUMENT));

    ovstage_data_plane_vtable_t vtableCopy = *stageVtable(stage);
    ovstage_instance_t          wrapped{};
    wrapped.data_plane    = {&vtableCopy, stageContext(stage)};
    malformedSource       = sourceDesc;
    malformedSource.stage = &wrapped;

    vtableCopy.query_extension                    = nullptr;
    ovspatialquery_population_handle_t population = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &malformedSource, &population)));
    CHECK(succeeded(ovspatialquery_destroy_population(instance, population)));

    vtableCopy             = *stageVtable(stage);
    vtableCopy.get_version = reportedVersion;
    for (const uint32_t qualifiedMinor : {3u, 4u})
    {
        gReportedMajor = 0u;
        gReportedMinor = qualifiedMinor;
        population     = OVSPATIALQUERY_INVALID_POPULATION;
        CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &malformedSource, &population)));
        CHECK(succeeded(ovspatialquery_destroy_population(instance, population)));
    }
    const uint32_t unqualified[][2] = {{0u, 1u}, {0u, 2u}, {0u, 5u}, {1u, 3u}};
    for (const auto& version : unqualified)
    {
        gReportedMajor = version[0];
        gReportedMinor = version[1];
        CHECK(rejectedOvstagePopulation(instance, &malformedSource, OVSPATIALQUERY_API_NOT_SUPPORTED));
        // The diagnostic names the version the stage reported.
        const ovspatialquery_string_t error = ovspatialquery_get_last_error();
        const std::string reported = "ovstage " + std::to_string(version[0]) + "." + std::to_string(version[1]) + ".7";
        CHECK(std::string(error.ptr, error.length).find(reported) != std::string::npos);
    }

    vtableCopy                           = *stageVtable(stage);
    vtableCopy.get_attribute_write_floor = nullptr;
    CHECK(rejectedOvstagePopulation(instance, &malformedSource, OVSPATIALQUERY_API_NOT_SUPPORTED));
    vtableCopy       = *stageVtable(stage);
    vtableCopy.query = nullptr;
    CHECK(rejectedOvstagePopulation(instance, &malformedSource, OVSPATIALQUERY_API_NOT_SUPPORTED));
    vtableCopy                 = *stageVtable(stage);
    vtableCopy.fetch_read_next = nullptr;
    CHECK(rejectedOvstagePopulation(instance, &malformedSource, OVSPATIALQUERY_API_NOT_SUPPORTED));
    vtableCopy                = *stageVtable(stage);
    vtableCopy.release_groups = nullptr;
    CHECK(rejectedOvstagePopulation(instance, &malformedSource, OVSPATIALQUERY_API_NOT_SUPPORTED));

    path_dictionary_vtable_t   dictionaryVtable = *dictionary->vtable;
    path_dictionary_instance_t wrappedDictionary{&dictionaryVtable, dictionary->context};
    gDictionary                            = &wrappedDictionary;
    vtableCopy                             = *stageVtable(stage);
    vtableCopy.get_path_dictionary         = fakeDictionary;
    dictionaryVtable.get_tokens_from_paths = nullptr;
    CHECK(rejectedOvstagePopulation(instance, &malformedSource, OVSPATIALQUERY_API_NOT_SUPPORTED));
    gDictionary = nullptr;
    CHECK(succeeded(ovspatialquery_destroy_instance(instance)));
    return 0;
}

} // namespace

int main()
{
#if defined(_WIN32)
    // Ovstage may launch a persistent Hub process. Keep CTest's output pipes
    // owned by this test so Hub cannot delay EOF after the test has exited.
    for (const DWORD stream : {STD_OUTPUT_HANDLE, STD_ERROR_HANDLE})
    {
        const HANDLE handle = GetStdHandle(stream);
        CHECK(handle != INVALID_HANDLE_VALUE);
        if (handle != nullptr)
        {
            CHECK(SetHandleInformation(handle, HANDLE_FLAG_INHERIT, 0) != 0);
        }
    }
#endif
    ovstage_instance_t*           stage = nullptr;
    const ovstage_instance_desc_t stageDesc{"ovspatialquery-companion-test"};
    CHECK(ovstage_create_instance(&stageDesc, &stage) == OVSTAGE_OK);
    CHECK(stage != nullptr);
    path_dictionary_instance_t* dictionary = ovstage_get_path_dictionary(stage);
    CHECK(dictionary != nullptr);
    CHECK(testInstanceValidation(stage, dictionary) == 0);
    CHECK(ovstage_destroy_instance(stage) == OVSTAGE_OK);
    return 0;
}
