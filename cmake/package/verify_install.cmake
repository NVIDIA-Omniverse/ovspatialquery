# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED OVSPATIALQUERY_INSTALL_ROOT)
    message(FATAL_ERROR "OVSPATIALQUERY_INSTALL_ROOT is required")
endif()
if(NOT IS_ABSOLUTE "${OVSPATIALQUERY_INSTALL_ROOT}")
    message(FATAL_ERROR "OVSPATIALQUERY_INSTALL_ROOT must be absolute: ${OVSPATIALQUERY_INSTALL_ROOT}")
endif()

if(NOT DEFINED OVSPATIALQUERY_EXPECT_OVSTAGE_COMPANION)
    set(OVSPATIALQUERY_EXPECT_OVSTAGE_COMPANION OFF)
endif()

if(WIN32)
    set(OVSPATIALQUERY_INSTALLED_DLL
        "${OVSPATIALQUERY_INSTALL_ROOT}/bin/ovspatialquery.dll")
    set(OVSPATIALQUERY_INSTALLED_LINK_ARTIFACT
        "${OVSPATIALQUERY_INSTALL_ROOT}/lib/ovspatialquery.lib")
else()
    set(OVSPATIALQUERY_INSTALLED_DLL
        "${OVSPATIALQUERY_INSTALL_ROOT}/lib/libovspatialquery.so")
    set(OVSPATIALQUERY_INSTALLED_LINK_ARTIFACT "${OVSPATIALQUERY_INSTALLED_DLL}")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/public_headers.cmake")
set(OVSPATIALQUERY_INSTALLED_INCLUDE_ROOT
    "${OVSPATIALQUERY_INSTALL_ROOT}/include/ovspatialquery")
set(OVSPATIALQUERY_REQUIRED_FILES
    "${OVSPATIALQUERY_INSTALLED_DLL}"
    "${OVSPATIALQUERY_INSTALLED_LINK_ARTIFACT}"
    "${OVSPATIALQUERY_INSTALL_ROOT}/LICENSE"
    "${OVSPATIALQUERY_INSTALL_ROOT}/NOTICE"
    "${OVSPATIALQUERY_INSTALL_ROOT}/include/dlpack/dlpack.h"
    "${OVSPATIALQUERY_INSTALL_ROOT}/licenses/dlpack-LICENSE.txt"
    "${OVSPATIALQUERY_INSTALL_ROOT}/licenses/physx-LICENSE.md"
    "${OVSPATIALQUERY_INSTALL_ROOT}/licenses/physx-PROVENANCE.json"
    "${OVSPATIALQUERY_INSTALL_ROOT}/lib/cmake/ovspatialquery/ovspatialqueryConfig.cmake"
    "${OVSPATIALQUERY_INSTALL_ROOT}/lib/cmake/ovspatialquery/ovspatialqueryConfigVersion.cmake"
)
foreach(OVSPATIALQUERY_PUBLIC_HEADER IN LISTS OVSPATIALQUERY_PUBLIC_HEADERS)
    list(APPEND OVSPATIALQUERY_REQUIRED_FILES
        "${OVSPATIALQUERY_INSTALLED_INCLUDE_ROOT}/${OVSPATIALQUERY_PUBLIC_HEADER}")
endforeach()
if(OVSPATIALQUERY_EXPECT_OVSTAGE_COMPANION)
    list(APPEND OVSPATIALQUERY_REQUIRED_FILES
        "${OVSPATIALQUERY_INSTALLED_INCLUDE_ROOT}/${OVSPATIALQUERY_OVSTAGE_PUBLIC_HEADER}")
endif()

foreach(OVSPATIALQUERY_REQUIRED_FILE IN LISTS OVSPATIALQUERY_REQUIRED_FILES)
    if(NOT EXISTS "${OVSPATIALQUERY_REQUIRED_FILE}")
        message(FATAL_ERROR "Required installed artifact is missing: ${OVSPATIALQUERY_REQUIRED_FILE}")
    endif()
endforeach()

# Compare installed legal evidence with its source authority, not duplicated hashes.
if(NOT OVSPATIALQUERY_PHYSX_SOURCE_PROVENANCE)
    set(OVSPATIALQUERY_PHYSX_SOURCE_PROVENANCE
        "${CMAKE_CURRENT_LIST_DIR}/../../deps/physx-license-provenance.json")
endif()
file(READ "${OVSPATIALQUERY_PHYSX_SOURCE_PROVENANCE}"
    OVSPATIALQUERY_PHYSX_PROVENANCE_CONTENT)
string(JSON OVSPATIALQUERY_EXPECTED_LICENSE_SHA256 GET
    "${OVSPATIALQUERY_PHYSX_PROVENANCE_CONTENT}" installed_license_sha256)
file(SHA256 "${OVSPATIALQUERY_INSTALL_ROOT}/licenses/physx-LICENSE.md"
    OVSPATIALQUERY_PHYSX_LICENSE_SHA256)
if(NOT OVSPATIALQUERY_PHYSX_LICENSE_SHA256 STREQUAL
        OVSPATIALQUERY_EXPECTED_LICENSE_SHA256)
    message(FATAL_ERROR "Installed PhysX license differs from the pinned package authority")
endif()
file(SHA256 "${OVSPATIALQUERY_PHYSX_SOURCE_PROVENANCE}"
    OVSPATIALQUERY_EXPECTED_PROVENANCE_SHA256)
file(SHA256 "${OVSPATIALQUERY_INSTALL_ROOT}/licenses/physx-PROVENANCE.json"
    OVSPATIALQUERY_INSTALLED_PROVENANCE_SHA256)
if(NOT OVSPATIALQUERY_INSTALLED_PROVENANCE_SHA256 STREQUAL
        OVSPATIALQUERY_EXPECTED_PROVENANCE_SHA256)
    message(FATAL_ERROR "Installed PhysX provenance differs from its source authority")
endif()

set(OVSPATIALQUERY_DLPACK_HEADER
    "${OVSPATIALQUERY_INSTALL_ROOT}/include/dlpack/dlpack.h")
set(OVSPATIALQUERY_DLPACK_LICENSE
    "${OVSPATIALQUERY_INSTALL_ROOT}/licenses/dlpack-LICENSE.txt")
file(READ "${OVSPATIALQUERY_DLPACK_HEADER}"
    OVSPATIALQUERY_DLPACK_HEADER_CONTENT)
file(READ "${OVSPATIALQUERY_DLPACK_LICENSE}"
    OVSPATIALQUERY_DLPACK_LICENSE_CONTENT)
string(REPLACE "\r\n" "\n" OVSPATIALQUERY_DLPACK_HEADER_CONTENT
    "${OVSPATIALQUERY_DLPACK_HEADER_CONTENT}")
string(REPLACE "\r\n" "\n" OVSPATIALQUERY_DLPACK_LICENSE_CONTENT
    "${OVSPATIALQUERY_DLPACK_LICENSE_CONTENT}")
string(SHA256 OVSPATIALQUERY_DLPACK_HEADER_SHA256
    "${OVSPATIALQUERY_DLPACK_HEADER_CONTENT}")
string(SHA256 OVSPATIALQUERY_DLPACK_LICENSE_SHA256
    "${OVSPATIALQUERY_DLPACK_LICENSE_CONTENT}")
if(NOT OVSPATIALQUERY_DLPACK_HEADER_SHA256 STREQUAL
        "41f9345a5409877618fff729f14474f133a5ee1d7917ad864b027f7f259fdd3f")
    message(FATAL_ERROR
        "Installed DLPack header differs from the pinned v1.3 ABI header: "
        "${OVSPATIALQUERY_DLPACK_HEADER_SHA256}")
endif()
if(NOT OVSPATIALQUERY_DLPACK_LICENSE_SHA256 STREQUAL
        "eabf0e000482c96db821ff0c176b983c29ad975f9c0ed099c6868d413ed1f05b")
    message(FATAL_ERROR
        "Installed DLPack license differs from the pinned v1.3 license: "
        "${OVSPATIALQUERY_DLPACK_LICENSE_SHA256}")
endif()

if(OVSPATIALQUERY_EXPECT_OVSTAGE_COMPANION)
    set(OVSPATIALQUERY_OVSTAGE_HEADER
        "${OVSPATIALQUERY_INSTALLED_INCLUDE_ROOT}/${OVSPATIALQUERY_OVSTAGE_PUBLIC_HEADER}")
    file(READ "${OVSPATIALQUERY_OVSTAGE_HEADER}"
        OVSPATIALQUERY_OVSTAGE_HEADER_CONTENT)
    foreach(OVSPATIALQUERY_OVSTAGE_IDENTIFIER IN ITEMS
            ovspatialquery_ovstage_population_desc_t
            ovspatialquery_ovstage_view_desc_t
            ovspatialquery_ovstage_participation_table_t
            ovspatialquery_ovstage_exclusion_reason_t
            ovspatialquery_create_ovstage_population
            ovspatialquery_create_ovstage_view
            ovspatialquery_set_ovstage_participation_map
            ovspatialquery_enqueue_ovstage_population_set
            ovspatialquery_enqueue_ovstage_population_update)
        string(FIND "${OVSPATIALQUERY_OVSTAGE_HEADER_CONTENT}"
            "${OVSPATIALQUERY_OVSTAGE_IDENTIFIER}"
            OVSPATIALQUERY_OVSTAGE_IDENTIFIER_OFFSET)
        if(OVSPATIALQUERY_OVSTAGE_IDENTIFIER_OFFSET EQUAL -1)
            message(FATAL_ERROR
                "Installed ovstage header is missing ${OVSPATIALQUERY_OVSTAGE_IDENTIFIER}")
        endif()
    endforeach()
endif()

# The include tree holds exactly the public headers, plus the companion header when expected,
# so a stale or unlisted header (one left in a reused install root) fails the install.
set(OVSPATIALQUERY_ALLOWED_HEADERS ${OVSPATIALQUERY_PUBLIC_HEADERS})
if(OVSPATIALQUERY_EXPECT_OVSTAGE_COMPANION)
    list(APPEND OVSPATIALQUERY_ALLOWED_HEADERS "${OVSPATIALQUERY_OVSTAGE_PUBLIC_HEADER}")
endif()
file(GLOB_RECURSE OVSPATIALQUERY_INSTALLED_HEADERS LIST_DIRECTORIES false
    RELATIVE "${OVSPATIALQUERY_INSTALLED_INCLUDE_ROOT}" "${OVSPATIALQUERY_INSTALLED_INCLUDE_ROOT}/*")
foreach(OVSPATIALQUERY_INSTALLED_HEADER IN LISTS OVSPATIALQUERY_INSTALLED_HEADERS)
    if(NOT OVSPATIALQUERY_INSTALLED_HEADER IN_LIST OVSPATIALQUERY_ALLOWED_HEADERS)
        message(FATAL_ERROR "The install ships a header outside the public list: ${OVSPATIALQUERY_INSTALLED_HEADER}")
    endif()
endforeach()

# The expected export set is derived directly from the OVSPATIALQUERY_API
# function declarations in the installed headers, so the binary and the
# header are compared to each other rather than to a hand-maintained copy.
function(ovspatialquery_read_installed_api_functions header_path output_variable)
    file(READ "${header_path}" _header_text)
    string(REGEX MATCHALL
        "OVSPATIALQUERY_API[ \t\r\n]+[A-Za-z_][A-Za-z0-9_]*[ \t\r\n]+ovspatialquery_[A-Za-z0-9_]+[ \t\r\n]*\\("
        _matches "${_header_text}")
    set(_names)
    foreach(_match IN LISTS _matches)
        string(REGEX REPLACE
            "^OVSPATIALQUERY_API[ \t\r\n]+[A-Za-z_][A-Za-z0-9_]*[ \t\r\n]+(ovspatialquery_[A-Za-z0-9_]+)[ \t\r\n]*\\($"
            "\\1" _name "${_match}")
        list(APPEND _names "${_name}")
    endforeach()
    set(${output_variable} "${_names}" PARENT_SCOPE)
endfunction()
set(OVSPATIALQUERY_REQUIRED_EXPORTS)
foreach(OVSPATIALQUERY_EXPORT_HEADER IN LISTS OVSPATIALQUERY_PUBLIC_HEADERS)
    ovspatialquery_read_installed_api_functions(
        "${OVSPATIALQUERY_INSTALLED_INCLUDE_ROOT}/${OVSPATIALQUERY_EXPORT_HEADER}"
        OVSPATIALQUERY_HEADER_FUNCTIONS)
    list(APPEND OVSPATIALQUERY_REQUIRED_EXPORTS ${OVSPATIALQUERY_HEADER_FUNCTIONS})
endforeach()
if(OVSPATIALQUERY_EXPECT_OVSTAGE_COMPANION)
    ovspatialquery_read_installed_api_functions(
        "${OVSPATIALQUERY_INSTALLED_INCLUDE_ROOT}/${OVSPATIALQUERY_OVSTAGE_PUBLIC_HEADER}"
        OVSPATIALQUERY_HEADER_FUNCTIONS)
    list(APPEND OVSPATIALQUERY_REQUIRED_EXPORTS ${OVSPATIALQUERY_HEADER_FUNCTIONS})
endif()
list(REMOVE_DUPLICATES OVSPATIALQUERY_REQUIRED_EXPORTS)
list(SORT OVSPATIALQUERY_REQUIRED_EXPORTS)

# Read the defined dynamic symbols and the runtime dependencies of the
# installed library with the platform's binary inspector: dumpbin (the MSVC
# linker) on Windows, nm plus readelf or objdump elsewhere. Both branches
# publish the same two variables.
if(WIN32)
    if(NOT DEFINED OVSPATIALQUERY_DUMPBIN OR NOT EXISTS "${OVSPATIALQUERY_DUMPBIN}")
        message(FATAL_ERROR "OVSPATIALQUERY_DUMPBIN must name the MSVC linker or dumpbin executable")
    endif()
    execute_process(
        COMMAND "${OVSPATIALQUERY_DUMPBIN}" /dump /exports "${OVSPATIALQUERY_INSTALLED_DLL}"
        RESULT_VARIABLE OVSPATIALQUERY_EXPORT_RESULT
        OUTPUT_VARIABLE OVSPATIALQUERY_EXPORT_OUTPUT
        ERROR_VARIABLE OVSPATIALQUERY_EXPORT_ERROR)
    if(NOT OVSPATIALQUERY_EXPORT_RESULT EQUAL 0)
        message(FATAL_ERROR
            "Installed library export probe failed (${OVSPATIALQUERY_EXPORT_RESULT}): ${OVSPATIALQUERY_EXPORT_ERROR}")
    endif()
    string(REGEX MATCHALL "ovspatialquery_[A-Za-z0-9_]+"
        OVSPATIALQUERY_FOUND_EXPORTS "${OVSPATIALQUERY_EXPORT_OUTPUT}")
    execute_process(
        COMMAND "${OVSPATIALQUERY_DUMPBIN}" /dump /dependents "${OVSPATIALQUERY_INSTALLED_DLL}"
        RESULT_VARIABLE OVSPATIALQUERY_DEPENDENTS_RESULT
        OUTPUT_VARIABLE OVSPATIALQUERY_DEPENDENTS_OUTPUT
        ERROR_VARIABLE OVSPATIALQUERY_DEPENDENTS_ERROR)
    set(OVSPATIALQUERY_FORBIDDEN_DEPENDENCY "ovstage.dll")
else()
    if(NOT DEFINED OVSPATIALQUERY_NM OR NOT EXISTS "${OVSPATIALQUERY_NM}")
        message(FATAL_ERROR "OVSPATIALQUERY_NM must name the nm executable")
    endif()
    execute_process(
        COMMAND "${OVSPATIALQUERY_NM}" -D --defined-only --format=posix
            "${OVSPATIALQUERY_INSTALLED_DLL}"
        RESULT_VARIABLE OVSPATIALQUERY_EXPORT_RESULT
        OUTPUT_VARIABLE OVSPATIALQUERY_EXPORT_OUTPUT
        ERROR_VARIABLE OVSPATIALQUERY_EXPORT_ERROR)
    if(NOT OVSPATIALQUERY_EXPORT_RESULT EQUAL 0)
        message(FATAL_ERROR
            "Installed library export probe failed (${OVSPATIALQUERY_EXPORT_RESULT}): ${OVSPATIALQUERY_EXPORT_ERROR}")
    endif()
    # posix format: one "name type value size" line per defined dynamic symbol.
    string(REGEX MATCHALL "ovspatialquery_[A-Za-z0-9_]+ [A-Za-z] "
        OVSPATIALQUERY_FOUND_EXPORTS "${OVSPATIALQUERY_EXPORT_OUTPUT}")
    list(TRANSFORM OVSPATIALQUERY_FOUND_EXPORTS REPLACE " [A-Za-z] $" "")
    if(DEFINED OVSPATIALQUERY_READELF AND EXISTS "${OVSPATIALQUERY_READELF}")
        set(OVSPATIALQUERY_DEPENDENCY_COMMAND "${OVSPATIALQUERY_READELF}" -d)
    elseif(DEFINED OVSPATIALQUERY_OBJDUMP AND EXISTS "${OVSPATIALQUERY_OBJDUMP}")
        set(OVSPATIALQUERY_DEPENDENCY_COMMAND "${OVSPATIALQUERY_OBJDUMP}" -p)
    else()
        message(FATAL_ERROR "OVSPATIALQUERY_READELF or OVSPATIALQUERY_OBJDUMP must name an ELF inspector")
    endif()
    execute_process(
        COMMAND ${OVSPATIALQUERY_DEPENDENCY_COMMAND} "${OVSPATIALQUERY_INSTALLED_DLL}"
        RESULT_VARIABLE OVSPATIALQUERY_DEPENDENTS_RESULT
        OUTPUT_VARIABLE OVSPATIALQUERY_DEPENDENTS_OUTPUT
        ERROR_VARIABLE OVSPATIALQUERY_DEPENDENTS_ERROR)
    string(TOLOWER "${OVSPATIALQUERY_DEPENDENTS_OUTPUT}" OVSPATIALQUERY_DEPENDENTS_LOWER)
    string(FIND "${OVSPATIALQUERY_DEPENDENTS_LOWER}" "needed" OVSPATIALQUERY_NEEDED_OFFSET)
    if(OVSPATIALQUERY_DEPENDENTS_RESULT EQUAL 0 AND OVSPATIALQUERY_NEEDED_OFFSET EQUAL -1)
        message(FATAL_ERROR
            "Installed library dependency probe did not report dynamic dependencies:\n${OVSPATIALQUERY_DEPENDENTS_OUTPUT}")
    endif()
    set(OVSPATIALQUERY_FORBIDDEN_DEPENDENCY "libovstage.so")

    # A dynamic relocation naming a public function means the library calls
    # itself through the global symbol scope. A process that loaded another
    # copy first, such as an application linked to the SDK that also imports
    # the wheel, would then run that copy's function on this copy's handles.
    if(DEFINED OVSPATIALQUERY_READELF AND EXISTS "${OVSPATIALQUERY_READELF}")
        set(OVSPATIALQUERY_RELOCATION_COMMAND "${OVSPATIALQUERY_READELF}" -rW)
    else()
        set(OVSPATIALQUERY_RELOCATION_COMMAND "${OVSPATIALQUERY_OBJDUMP}" -R)
    endif()
    execute_process(
        COMMAND ${OVSPATIALQUERY_RELOCATION_COMMAND} "${OVSPATIALQUERY_INSTALLED_DLL}"
        RESULT_VARIABLE OVSPATIALQUERY_RELOCATION_RESULT
        OUTPUT_VARIABLE OVSPATIALQUERY_RELOCATION_OUTPUT
        ERROR_VARIABLE OVSPATIALQUERY_RELOCATION_ERROR)
    if(NOT OVSPATIALQUERY_RELOCATION_RESULT EQUAL 0)
        message(FATAL_ERROR
            "Installed library relocation probe failed: ${OVSPATIALQUERY_RELOCATION_ERROR}")
    endif()
    string(REGEX MATCHALL "[ \t]ovspatialquery_[A-Za-z0-9_]+"
        OVSPATIALQUERY_SELF_RELOCATIONS "${OVSPATIALQUERY_RELOCATION_OUTPUT}")
    if(OVSPATIALQUERY_SELF_RELOCATIONS)
        message(FATAL_ERROR
            "Installed library binds calls to its own public functions at run time, so another "
            "loaded copy can interpose them:${OVSPATIALQUERY_SELF_RELOCATIONS}")
    endif()
endif()
if(NOT OVSPATIALQUERY_DEPENDENTS_RESULT EQUAL 0)
    message(FATAL_ERROR
        "Installed library dependency probe failed: ${OVSPATIALQUERY_DEPENDENTS_ERROR}")
endif()

list(REMOVE_DUPLICATES OVSPATIALQUERY_FOUND_EXPORTS)
list(SORT OVSPATIALQUERY_FOUND_EXPORTS)
foreach(OVSPATIALQUERY_REQUIRED_EXPORT IN LISTS OVSPATIALQUERY_REQUIRED_EXPORTS)
    if(NOT OVSPATIALQUERY_REQUIRED_EXPORT IN_LIST OVSPATIALQUERY_FOUND_EXPORTS)
        message(FATAL_ERROR
            "Installed library is missing C export ${OVSPATIALQUERY_REQUIRED_EXPORT}:\n${OVSPATIALQUERY_EXPORT_OUTPUT}")
    endif()
endforeach()
if(NOT OVSPATIALQUERY_FOUND_EXPORTS STREQUAL OVSPATIALQUERY_REQUIRED_EXPORTS)
    message(FATAL_ERROR
        "Installed library export set differs from the exact current API allowlist. "
        "Expected=${OVSPATIALQUERY_REQUIRED_EXPORTS}; "
        "found=${OVSPATIALQUERY_FOUND_EXPORTS}")
endif()
string(TOLOWER "${OVSPATIALQUERY_DEPENDENTS_OUTPUT}" OVSPATIALQUERY_DEPENDENTS_LOWER)
string(FIND "${OVSPATIALQUERY_DEPENDENTS_LOWER}" "${OVSPATIALQUERY_FORBIDDEN_DEPENDENCY}"
    OVSPATIALQUERY_OVSTAGE_IMPORT_OFFSET)
if(NOT OVSPATIALQUERY_OVSTAGE_IMPORT_OFFSET EQUAL -1)
    message(FATAL_ERROR
        "Core library has a forbidden ovstage runtime dependency:\n${OVSPATIALQUERY_DEPENDENTS_OUTPUT}")
endif()
foreach(OVSPATIALQUERY_FOUND_EXPORT IN LISTS OVSPATIALQUERY_FOUND_EXPORTS)
    string(TOLOWER "${OVSPATIALQUERY_FOUND_EXPORT}" OVSPATIALQUERY_FOUND_EXPORT_LOWER)
    if(OVSPATIALQUERY_FOUND_EXPORT_LOWER MATCHES "^ovspatialquery_physx_test_")
        message(FATAL_ERROR "Installed library leaks a source-private PhysX test hook: ${OVSPATIALQUERY_FOUND_EXPORT}")
    endif()
endforeach()

# The library carries its source commit as plain text for `strings`/`findstr`.
if(OVSPATIALQUERY_EXPECT_SOURCE_REVISION)
    file(STRINGS "${OVSPATIALQUERY_INSTALLED_DLL}" OVSPATIALQUERY_SOURCE_REVISION_STRINGS
        REGEX "ovspatialquery-source: ")
    string(REGEX MATCHALL "ovspatialquery-source: [^;]*" OVSPATIALQUERY_SOURCE_REVISION_FOUND
        "${OVSPATIALQUERY_SOURCE_REVISION_STRINGS}")
    if(NOT OVSPATIALQUERY_SOURCE_REVISION_FOUND STREQUAL
            "ovspatialquery-source: ${OVSPATIALQUERY_EXPECT_SOURCE_REVISION}")
        message(FATAL_ERROR
            "Installed library records '${OVSPATIALQUERY_SOURCE_REVISION_FOUND}', expected "
            "'ovspatialquery-source: ${OVSPATIALQUERY_EXPECT_SOURCE_REVISION}'")
    endif()
endif()

# The PTX lives next to the library that loads it.
get_filename_component(OVSPATIALQUERY_INSTALLED_LIBRARY_DIR
    "${OVSPATIALQUERY_INSTALLED_DLL}" DIRECTORY)
set(OVSPATIALQUERY_INSTALLED_PTX
    "${OVSPATIALQUERY_INSTALLED_LIBRARY_DIR}/ovspatialquery_cuda.ptx")
if(OVSPATIALQUERY_EXPECT_CUDA_BACKEND)
    if(NOT EXISTS "${OVSPATIALQUERY_INSTALLED_PTX}")
        message(FATAL_ERROR "Installed CUDA backend is missing ${OVSPATIALQUERY_INSTALLED_PTX}")
    endif()
elseif(EXISTS "${OVSPATIALQUERY_INSTALLED_PTX}")
    message(FATAL_ERROR "CUDA PTX was installed without an enabled CUDA backend")
endif()

message(STATUS "Verified installed ovspatialquery artifacts at ${OVSPATIALQUERY_INSTALL_ROOT}")
