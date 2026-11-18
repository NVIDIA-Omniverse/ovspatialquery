# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)

project(ovspatialquery LANGUAGES C CXX)

include(GNUInstallDirs)

# The SDK version (cmake/package/source_version.cmake). The cache entry holds only an
# explicit override; the selected version shadows it for the rest of the project.
include(cmake/package/source_version.cmake)
set(OVSPATIALQUERY_VERSION "" CACHE STRING
    "SDK version override: major.minor.patch, optionally followed by .devN or rcN; empty selects the default")
find_package(Git QUIET)
ovspatialquery_select_version("${CMAKE_CURRENT_SOURCE_DIR}"
    _OVSPATIALQUERY_SELECTED_VERSION _OVSPATIALQUERY_VERSION_ORIGIN)
set(OVSPATIALQUERY_VERSION "${_OVSPATIALQUERY_SELECTED_VERSION}")
set(OVSPATIALQUERY_SELECTED_VERSION "${OVSPATIALQUERY_VERSION}" CACHE INTERNAL
    "SDK version this configure selected")
if(_OVSPATIALQUERY_VERSION_ORIGIN STREQUAL "stamp")
    set(_OVSPATIALQUERY_VERSION_ORIGIN "${CMAKE_CURRENT_SOURCE_DIR}/VERSION.txt")
elseif(_OVSPATIALQUERY_VERSION_ORIGIN STREQUAL "source")
    message(STATUS "ovspatialquery: building development version ${OVSPATIALQUERY_VERSION}; "
        "set OVSPATIALQUERY_BUILD_VERSION=X.Y.Z[rcN] to override")
endif()
if(NOT OVSPATIALQUERY_VERSION MATCHES "^(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)((\\.dev|rc)(0|[1-9][0-9]*))?$")
    message(FATAL_ERROR "Invalid SDK version from ${_OVSPATIALQUERY_VERSION_ORIGIN}: ${OVSPATIALQUERY_VERSION}")
endif()
set(_OVSPATIALQUERY_VERSION_MAJOR "${CMAKE_MATCH_1}")
set(_OVSPATIALQUERY_VERSION_MINOR "${CMAKE_MATCH_2}")
set(_OVSPATIALQUERY_VERSION_PATCH "${CMAKE_MATCH_3}")
foreach(_component MAJOR MINOR PATCH)
    if(_OVSPATIALQUERY_VERSION_${_component} GREATER 4294967295)
        message(FATAL_ERROR "SDK version components must fit uint32_t")
    endif()
endforeach()
set_property(SOURCE source/ovspatialquery_instance.cpp APPEND PROPERTY COMPILE_DEFINITIONS
    "OVSPATIALQUERY_VERSION_MAJOR=${_OVSPATIALQUERY_VERSION_MAJOR}u"
    "OVSPATIALQUERY_VERSION_MINOR=${_OVSPATIALQUERY_VERSION_MINOR}u"
    "OVSPATIALQUERY_VERSION_PATCH=${_OVSPATIALQUERY_VERSION_PATCH}u")

# The commit compiled into the library as a plain string for `strings`/`findstr`.
# The facade passes it on every configure; plain CMake builds record "unknown".
set(OVSPATIALQUERY_SOURCE_REVISION "unknown" CACHE STRING
    "Source commit recorded in the library: 40 hex digits, optionally -dirty, or unknown")
string(REPEAT "[0-9a-f]" 40 _OVSPATIALQUERY_COMMIT_PATTERN)
if(NOT OVSPATIALQUERY_SOURCE_REVISION MATCHES "^(${_OVSPATIALQUERY_COMMIT_PATTERN}(-dirty)?|unknown)$")
    message(FATAL_ERROR "Invalid OVSPATIALQUERY_SOURCE_REVISION: ${OVSPATIALQUERY_SOURCE_REVISION}")
endif()
set_property(SOURCE source/ovspatialquery_source_revision.cpp APPEND PROPERTY COMPILE_DEFINITIONS
    "OVSPATIALQUERY_SOURCE_REVISION=\"${OVSPATIALQUERY_SOURCE_REVISION}\"")

option(OVSPATIALQUERY_BUILD_PRODUCT "Build the packaged public PhysX PxBVH backend" ON)
option(OVSPATIALQUERY_BUILD_TESTS "Build the deterministic test-oracle slice" ON)
option(OVSPATIALQUERY_BUILD_OVSTAGE_COMPANION
    "Build and install the optional ovstage source companion" OFF)
option(OVSPATIALQUERY_FETCH_DEPS "Fetch the pinned packaged PhysX SDK with Packman" ON)
set(OVSPATIALQUERY_CUDA_NVRTC_DIR "" CACHE PATH
    "Optional packaged CUDA NVRTC directory used to build the CUDA batch backend")
set(OVSPATIALQUERY_TARGET_DEPS "${CMAKE_CURRENT_SOURCE_DIR}/_build/target-deps" CACHE PATH
    "Packman target-deps root")
set(OVSPATIALQUERY_PYTHON_EXECUTABLE "" CACHE FILEPATH
    "Optional Python 3.12 interpreter for the ctypes tests")
set(OVSPATIALQUERY_OVSTAGE_SDK_DIR "" CACHE PATH
    "Installed ovstage package root containing include/, lib/, and bin/")
set(OVSPATIALQUERY_APP_ARTIFACT_DIR "${CMAKE_BINARY_DIR}/install" CACHE PATH
    "Deterministic app-integration artifact root")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Recorded by the bootstrap facade, which reconfigures from scratch when the selection changes.
set(OVSPATIALQUERY_COMPILER_SELECTION "" CACHE STRING "Bootstrap compiler selection: pinned or host")
if(MSVC)
    # MSVC 17.10 and newer construct std::mutex constexpr, which crashes when a host process
    # (Python, Kit) has already loaded an older msvcp140.dll. Keep the runtime-compatible form.
    add_compile_definitions(_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
    if(CMAKE_GENERATOR MATCHES "Visual Studio")
        # MSBuild builds projects in parallel but compiles one project's sources serially
        # without /MP; Ninja (the pinned compiler) already compiles per source in parallel.
        add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/MP>)
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    # Round every multiply and add as written. GCC fuses a*b+c by default where the
    # target has FMA (AArch64), and per inlined copy, so one expression could round
    # differently in two callers: an exact path then disagrees with the scan it must
    # match, and AArch64 results with x86-64's. MSVC's /fp:precise does not contract.
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-ffp-contract=off>)
endif()

set(OVSPATIALQUERY_HAS_CUDA_BACKEND OFF)
if(OVSPATIALQUERY_CUDA_NVRTC_DIR)
    # The PTX is compiled at build time with the packaged NVRTC and installed
    # next to the library that loads it (bin/ on Windows, lib/ on Linux).
    if(WIN32)
        set(OVSPATIALQUERY_NVRTC_LIBRARY
            "${OVSPATIALQUERY_CUDA_NVRTC_DIR}/nvrtc64_120_0.dll")
        set(OVSPATIALQUERY_NVRTC_BUILTINS
            "${OVSPATIALQUERY_CUDA_NVRTC_DIR}/nvrtc-builtins64_128.dll")
        set(OVSPATIALQUERY_CUDA_PTX_INSTALL_DIR ${CMAKE_INSTALL_BINDIR})
        set(OVSPATIALQUERY_PTX_COMPILER "$<TARGET_FILE:ovspatialquery_compile_cuda_ptx>")
    else()
        set(OVSPATIALQUERY_NVRTC_LIBRARY
            "${OVSPATIALQUERY_CUDA_NVRTC_DIR}/libnvrtc.so.12")
        file(GLOB OVSPATIALQUERY_NVRTC_BUILTINS
            "${OVSPATIALQUERY_CUDA_NVRTC_DIR}/libnvrtc-builtins.so.12.*")
        list(LENGTH OVSPATIALQUERY_NVRTC_BUILTINS _OVSPATIALQUERY_NVRTC_BUILTINS_COUNT)
        if(NOT _OVSPATIALQUERY_NVRTC_BUILTINS_COUNT EQUAL 1)
            set(OVSPATIALQUERY_NVRTC_BUILTINS "")
        endif()
        set(OVSPATIALQUERY_CUDA_PTX_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR})
        # NVRTC loads its builtins library by name from the same directory.
        set(OVSPATIALQUERY_PTX_COMPILER "${CMAKE_COMMAND}" -E env
            "LD_LIBRARY_PATH=${OVSPATIALQUERY_CUDA_NVRTC_DIR}"
            "$<TARGET_FILE:ovspatialquery_compile_cuda_ptx>")
    endif()
    if(NOT EXISTS "${OVSPATIALQUERY_NVRTC_LIBRARY}" OR
       NOT OVSPATIALQUERY_NVRTC_BUILTINS OR
       NOT EXISTS "${OVSPATIALQUERY_NVRTC_BUILTINS}")
        message(FATAL_ERROR
            "OVSPATIALQUERY_CUDA_NVRTC_DIR must contain the locked NVRTC compiler and builtins libraries: "
            "${OVSPATIALQUERY_CUDA_NVRTC_DIR}")
    endif()
    if(CMAKE_CROSSCOMPILING)
        set(OVSPATIALQUERY_HOST_PTX_COMPILER "" CACHE FILEPATH
            "Host executable for compiling PTX with the locked host NVRTC")
        if(NOT EXISTS "${OVSPATIALQUERY_HOST_PTX_COMPILER}")
            message(FATAL_ERROR "Cross-compiling CUDA requires OVSPATIALQUERY_HOST_PTX_COMPILER")
        endif()
        set(OVSPATIALQUERY_PTX_COMPILER "${CMAKE_COMMAND}" -E env
            "LD_LIBRARY_PATH=${OVSPATIALQUERY_CUDA_NVRTC_DIR}"
            "${OVSPATIALQUERY_HOST_PTX_COMPILER}")
        set(_OVSPATIALQUERY_PTX_COMPILER_DEPENDENCY "${OVSPATIALQUERY_HOST_PTX_COMPILER}")
    else()
        add_executable(ovspatialquery_compile_cuda_ptx tools/compile_cuda_ptx.cpp)
        target_include_directories(ovspatialquery_compile_cuda_ptx PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
        target_link_libraries(ovspatialquery_compile_cuda_ptx PRIVATE ${CMAKE_DL_LIBS})
        set(_OVSPATIALQUERY_PTX_COMPILER_DEPENDENCY ovspatialquery_compile_cuda_ptx)
    endif()
    # Compiles one .cu with the packaged NVRTC and names the target that
    # produces the PTX.  `source` is relative to the project root, `ptx` is the
    # absolute output path.
    function(ovspatialquery_add_cuda_ptx target source ptx)
        add_custom_command(
            OUTPUT "${ptx}"
            COMMAND ${OVSPATIALQUERY_PTX_COMPILER}
                "${OVSPATIALQUERY_NVRTC_LIBRARY}"
                "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
                "${ptx}"
            DEPENDS ${_OVSPATIALQUERY_PTX_COMPILER_DEPENDENCY}
                "${source}"
                ${ARGN}
                "${OVSPATIALQUERY_NVRTC_LIBRARY}"
                "${OVSPATIALQUERY_NVRTC_BUILTINS}"
            VERBATIM)
        add_custom_target(${target} DEPENDS "${ptx}")
    endfunction()
    set(OVSPATIALQUERY_CUDA_PTX
        "${CMAKE_CURRENT_BINARY_DIR}/ovspatialquery_cuda.ptx")
    ovspatialquery_add_cuda_ptx(ovspatialquery_cuda_ptx
        source/ovspatialquery_cuda_kernel.cu "${OVSPATIALQUERY_CUDA_PTX}"
        source/ovspatialquery_cuda_family_plan_types.h
        source/ovspatialquery_box_sweep_math.h
        source/ovspatialquery_cuda_pair_kernels.cuh
        source/ovspatialquery_cuda_pair_types.h
        source/ovspatialquery_cuda_device_boxes.cuh
        source/ovspatialquery_ovstage_transform.h)
    # The scheduling-delay gate the native tests launch is compiled separately so
    # it never reaches the shipped PTX; this one is copied beside test
    # executables and never installed.
    set(OVSPATIALQUERY_CUDA_TEST_PTX
        "${CMAKE_CURRENT_BINARY_DIR}/ovspatialquery_cuda_test.ptx")
    ovspatialquery_add_cuda_ptx(ovspatialquery_cuda_test_ptx
        source/ovspatialquery_cuda_test_kernel.cu "${OVSPATIALQUERY_CUDA_TEST_PTX}")
    # Every binary that loads the CUDA backend needs the PTX next to it.
    function(ovspatialquery_copy_cuda_ptx target)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${OVSPATIALQUERY_CUDA_PTX}"
                "$<TARGET_FILE_DIR:${target}>/ovspatialquery_cuda.ptx")
    endfunction()
    # Only tests that gate a stream with the delay kernel need the test PTX.
    function(ovspatialquery_copy_cuda_test_ptx target)
        add_dependencies(${target} ovspatialquery_cuda_test_ptx)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${OVSPATIALQUERY_CUDA_TEST_PTX}"
                "$<TARGET_FILE_DIR:${target}>/ovspatialquery_cuda_test.ptx")
    endfunction()
    set(OVSPATIALQUERY_HAS_CUDA_BACKEND ON)
endif()

set(_OVSPATIALQUERY_PYTHON_EXECUTABLE "")
if(OVSPATIALQUERY_BUILD_TESTS OR OVSPATIALQUERY_BUILD_PRODUCT)
    if(OVSPATIALQUERY_PYTHON_EXECUTABLE)
        execute_process(
            COMMAND "${OVSPATIALQUERY_PYTHON_EXECUTABLE}" --version
            RESULT_VARIABLE OVSPATIALQUERY_PYTHON_VERSION_RESULT
            OUTPUT_VARIABLE OVSPATIALQUERY_PYTHON_VERSION
            ERROR_VARIABLE OVSPATIALQUERY_PYTHON_VERSION_ERROR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_STRIP_TRAILING_WHITESPACE
        )
        if(NOT OVSPATIALQUERY_PYTHON_VERSION_RESULT EQUAL 0 OR
           NOT "${OVSPATIALQUERY_PYTHON_VERSION}${OVSPATIALQUERY_PYTHON_VERSION_ERROR}" MATCHES "Python 3\\.12\\.")
            message(FATAL_ERROR
                "OVSPATIALQUERY_PYTHON_EXECUTABLE must be Python 3.12: "
                "${OVSPATIALQUERY_PYTHON_VERSION}${OVSPATIALQUERY_PYTHON_VERSION_ERROR}")
        endif()
        set(_OVSPATIALQUERY_PYTHON_EXECUTABLE "${OVSPATIALQUERY_PYTHON_EXECUTABLE}")
    else()
        find_package(Python3 3.12 EXACT COMPONENTS Interpreter QUIET)
        set(_OVSPATIALQUERY_PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")
    endif()
endif()

if(OVSPATIALQUERY_BUILD_OVSTAGE_COMPANION)
    if(NOT EXISTS "${OVSPATIALQUERY_OVSTAGE_SDK_DIR}/lib/cmake/ovstage/ovstageConfig.cmake")
        message(FATAL_ERROR
            "The optional OVSTAGE-01 companion requires an explicit locked SDK; set "
            "OVSPATIALQUERY_OVSTAGE_SDK_DIR to its package root")
    endif()
    if(NOT (WIN32 OR (UNIX AND NOT APPLE)))
        message(FATAL_ERROR "The OVSTAGE-01 companion is not qualified on this platform")
    endif()
    # Follow the SDK directory when its pin changes; a cached ovstage_DIR would keep the old one.
    set(ovstage_DIR "${OVSPATIALQUERY_OVSTAGE_SDK_DIR}/lib/cmake/ovstage" CACHE PATH
        "The ovstage CMake package inside OVSPATIALQUERY_OVSTAGE_SDK_DIR" FORCE)
    find_package(ovstage CONFIG REQUIRED
        PATHS "${OVSPATIALQUERY_OVSTAGE_SDK_DIR}/lib/cmake/ovstage"
        NO_CMAKE_FIND_ROOT_PATH
        NO_DEFAULT_PATH)

    get_target_property(OVSPATIALQUERY_OVSTAGE_LIBRARY ovstage::ovstage IMPORTED_LOCATION)
    set_property(SOURCE
        source/ovspatialquery_ovstage_acquisition.cpp
        source/ovspatialquery_ovstage_device_publication.cpp
        source/ovspatialquery_ovstage_filter.cpp
        source/ovspatialquery_ovstage_mesh_geometry.cpp
        source/ovspatialquery_ovstage_scope.cpp
        source/ovspatialquery_ovstage_typed_acquisition.cpp
        APPEND PROPERTY
        INCLUDE_DIRECTORIES "${OVSPATIALQUERY_OVSTAGE_SDK_DIR}/include")
endif()

set(_OVSPATIALQUERY_RUNTIME_SOURCES
    source/ovspatialquery_instance.cpp
    source/ovspatialquery_instance_source.cpp
    source/ovspatialquery_internal_api.cpp
    source/ovspatialquery_operation_runtime.cpp
    source/ovspatialquery_explicit_registration_api.cpp
    source/ovspatialquery_current_api_query.cpp
    source/ovspatialquery_current_api_family.cpp
    source/ovspatialquery_current_api_pair.cpp
    source/ovspatialquery_current_api_operation.cpp
    source/ovspatialquery_instance_host_ledger.cpp
    source/ovspatialquery_physx_mesh.cpp
    source/ovspatialquery_geometry.cpp
    source/ovspatialquery_typed_geometry.cpp
    source/ovspatialquery_convex_geometry.cpp
    source/ovspatialquery_geometry_identity.cpp
    source/ovspatialquery_explicit_preparation.cpp
    source/ovspatialquery_explicit_registration_operation.cpp
    source/ovspatialquery_explicit_cuda_capture.cpp
    source/ovspatialquery_device_box_population.cpp
    source/ovspatialquery_cuda_backend.cpp
    source/ovspatialquery_cuda_driver.cpp
    source/ovspatialquery_cuda_population_version.cpp
    source/ovspatialquery_cuda_pair_reporter.cpp
    source/ovspatialquery_cuda_family_plan_catalogue.cpp
    source/ovspatialquery_cuda_dispatch.cpp
    source/ovspatialquery_cuda_family_plan.cpp
    source/ovspatialquery_input_layout.cpp
    source/ovspatialquery_prepared_array.cpp
    source/ovspatialquery_query_evaluation.cpp
    source/ovspatialquery_box_sweep_geometry.cpp
    source/ovspatialquery_family_batch_cpu.cpp
    source/ovspatialquery_family_batch_columns.cpp
    source/ovspatialquery_family_batch_plan_operation.cpp
    source/ovspatialquery_handle_directory.cpp
    source/ovspatialquery_input.cpp
    source/ovspatialquery_pair_materialization.cpp
    source/ovspatialquery_pair_reporter_operation.cpp
    source/ovspatialquery_pair_reporter_transition.cpp
    source/ovspatialquery_object_table.cpp
    source/ovspatialquery_participation.cpp
    source/ovspatialquery_population_table.cpp
    source/ovspatialquery_target_resolver.cpp)

set(_OVSPATIALQUERY_PRODUCT_RUNTIME_SOURCES
    ${_OVSPATIALQUERY_RUNTIME_SOURCES}
    source/ovspatialquery_physx_backend.cpp
    source/ovspatialquery_source_revision.cpp)
set(_OVSPATIALQUERY_PORTABLE_TEST_RUNTIME_SOURCES
    tests/support/native/oracle_backend.cpp
    tests/support/native/portable_convex_hull.cpp)
set(_OVSPATIALQUERY_PHYSX_TEST_RUNTIME_SOURCES
    source/ovspatialquery_physx_backend.cpp)

set(_OVSPATIALQUERY_OVSTAGE_RUNTIME_SOURCES
    source/ovspatialquery_ovstage_acquisition.cpp
    source/ovspatialquery_ovstage_device_publication.cpp
    source/ovspatialquery_ovstage_filter.cpp
    source/ovspatialquery_ovstage_mesh_geometry.cpp
    source/ovspatialquery_ovstage_scope.cpp
    source/ovspatialquery_ovstage_typed_acquisition.cpp)
if(OVSPATIALQUERY_BUILD_OVSTAGE_COMPANION)
    list(APPEND _OVSPATIALQUERY_PRODUCT_RUNTIME_SOURCES
        ${_OVSPATIALQUERY_OVSTAGE_RUNTIME_SOURCES})
    list(APPEND _OVSPATIALQUERY_PORTABLE_TEST_RUNTIME_SOURCES
        ${_OVSPATIALQUERY_OVSTAGE_RUNTIME_SOURCES})
endif()

# The installed library lives under bin/ on Windows and lib/ elsewhere;
# the binary inspector that reads its exports differs the same way.
if(WIN32)
    set(_OVSPATIALQUERY_INSTALLED_LIBRARY_DIR "${CMAKE_INSTALL_BINDIR}")
    set(_OVSPATIALQUERY_INSPECTOR_ARGS "-DOVSPATIALQUERY_DUMPBIN=${CMAKE_LINKER}")
else()
    set(_OVSPATIALQUERY_INSTALLED_LIBRARY_DIR "${CMAKE_INSTALL_LIBDIR}")
    set(_OVSPATIALQUERY_INSPECTOR_ARGS
        "-DOVSPATIALQUERY_NM=${CMAKE_NM}"
        "-DOVSPATIALQUERY_READELF=${CMAKE_READELF}"
        "-DOVSPATIALQUERY_OBJDUMP=${CMAKE_OBJDUMP}")
endif()
set(_OVSPATIALQUERY_INSTALLED_LIBRARY_PATH
    "${_OVSPATIALQUERY_INSTALLED_LIBRARY_DIR}/$<TARGET_FILE_NAME:ovspatialquery>")

if(OVSPATIALQUERY_BUILD_PRODUCT OR OVSPATIALQUERY_BUILD_TESTS)
    set(OVSPATIALQUERY_PHYSX_SDK_DIR "$ENV{OVSPATIALQUERY_PHYSX_SDK_DIR}" CACHE PATH "Explicit custom PhysX SDK root; empty selects the pinned package")
    set(OVSPATIALQUERY_CUSTOM_PHYSX OFF)
    if(OVSPATIALQUERY_PHYSX_SDK_DIR)
        set(OVSPATIALQUERY_CUSTOM_PHYSX ON)
    else()
        set(OVSPATIALQUERY_PHYSX_SDK_DIR "${OVSPATIALQUERY_TARGET_DEPS}/physx")
    endif()
    if(NOT EXISTS "${OVSPATIALQUERY_PHYSX_SDK_DIR}/include/PxPhysicsAPI.h" AND OVSPATIALQUERY_FETCH_DEPS AND NOT OVSPATIALQUERY_CUSTOM_PHYSX)
        message(STATUS "ovspatialquery: fetching packaged PhysX with Packman")
        execute_process(
            COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/package/fetch_physx_sdk.cmake"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            RESULT_VARIABLE OVSPATIALQUERY_FETCH_RESULT
        )
        if(NOT OVSPATIALQUERY_FETCH_RESULT EQUAL 0)
            message(FATAL_ERROR
                "ovspatialquery: packaged PhysX fetch failed (exit: ${OVSPATIALQUERY_FETCH_RESULT})")
        endif()
    endif()
    if(NOT EXISTS "${OVSPATIALQUERY_PHYSX_SDK_DIR}/include/PxPhysicsAPI.h")
        message(FATAL_ERROR
            "The product backend requires a PhysX SDK at ${OVSPATIALQUERY_PHYSX_SDK_DIR}. "
            "Run cmake/package/fetch_physx_sdk.cmake or enable OVSPATIALQUERY_FETCH_DEPS; no source-tree fallback is allowed.")
    endif()

    set(PHYSX_SDK_DIR "${OVSPATIALQUERY_PHYSX_SDK_DIR}")
    include(cmake/package/OvSpatialQueryPhysX.cmake)
    set(OVSPATIALQUERY_PHYSX_LIBRARY_ROOT
        "${PHYSX_SDK_DIR}/bin/${PHYSX_PLATFORM_BIN}")
    foreach(OVSPATIALQUERY_PHYSX_CONFIG release debug)
        if(NOT IS_DIRECTORY "${OVSPATIALQUERY_PHYSX_LIBRARY_ROOT}/${OVSPATIALQUERY_PHYSX_CONFIG}")
            message(FATAL_ERROR
                "Packaged PhysX library directory is missing: "
                "${OVSPATIALQUERY_PHYSX_LIBRARY_ROOT}/${OVSPATIALQUERY_PHYSX_CONFIG}")
        endif()
    endforeach()

endif()

if(OVSPATIALQUERY_BUILD_PRODUCT)
    add_library(ovspatialquery SHARED
        ${_OVSPATIALQUERY_PRODUCT_RUNTIME_SOURCES})
    include(cmake/package/PhysXProvenance.cmake)
    # PhysX is linked statically, so the library needs no runtime search path;
    # without this CMake would record the PhysX link directory as a build-tree
    # RUNPATH and rewrite it at install, and the installed file would never be
    # byte-identical to the built one that dev.sh verify compares it with.
    set_target_properties(ovspatialquery PROPERTIES SKIP_BUILD_RPATH ON)
    target_include_directories(ovspatialquery
        PUBLIC include
        PRIVATE source
    )
    target_include_directories(ovspatialquery SYSTEM PRIVATE
        "${PHYSX_SDK_DIR}/include")
    target_compile_definitions(ovspatialquery
        PRIVATE PX_PHYSX_STATIC_LIB OVSPATIALQUERY_BUILD_DLL
        INTERFACE OVSPATIALQUERY_USE_DLL
    )
    if(OVSPATIALQUERY_HAS_CUDA_BACKEND)
        target_compile_definitions(ovspatialquery
            PRIVATE OVSPATIALQUERY_HAS_CUDA_BACKEND)
        add_dependencies(ovspatialquery ovspatialquery_cuda_ptx)
        ovspatialquery_copy_cuda_ptx(ovspatialquery)
        target_link_libraries(ovspatialquery PRIVATE ${CMAKE_DL_LIBS})
        install(FILES "${OVSPATIALQUERY_CUDA_PTX}"
            DESTINATION ${OVSPATIALQUERY_CUDA_PTX_INSTALL_DIR})
    endif()
    set_target_properties(ovspatialquery PROPERTIES
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
    )
    if(UNIX AND NOT APPLE)
        # Hidden visibility alone still exports libstdc++ template
        # instantiations; the version script keeps only the public C prefix
        # global and --exclude-libs keeps the statically linked PhysX archives
        # from re-exporting theirs.  The installed-exports test compares the
        # result with the public headers.  -Bsymbolic-functions binds the
        # library's calls to its own exported functions inside the library, so
        # a second loaded copy cannot be routed to the first copy's functions
        # through the global symbol scope.
        set(OVSPATIALQUERY_ELF_VERSION_SCRIPT
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/package/ovspatialquery.exports.map")
        # Keep the hidden source marker even when unused sections are collected.
        target_link_options(ovspatialquery PRIVATE
            "LINKER:--build-id=sha1"
            "LINKER:--undefined=ovspatialquerySourceRevision"
            "-Wl,--exclude-libs,ALL"
            "-Wl,-Bsymbolic-functions"
            "-Wl,--version-script=${OVSPATIALQUERY_ELF_VERSION_SCRIPT}")
        set_property(TARGET ovspatialquery APPEND PROPERTY
            LINK_DEPENDS "${OVSPATIALQUERY_ELF_VERSION_SCRIPT}")
    endif()
    ovspatialquery_link_packaged_physx(ovspatialquery FULL)
    # The query evaluator's per-candidate functions switched between alignment modes with
    # unrelated edits to the same file (CPU bounds-candidate camera rays 113 against 150 ns
    # per ray from a 224-byte shift); cache-line alignment keeps them in the fast mode. The
    # mesh geometry's ray functions did the same when its box sweep changed (mesh grounding
    # rays 372 against 417 us per 1,024-ray batch).
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        set_source_files_properties(source/ovspatialquery_query_evaluation.cpp source/ovspatialquery_geometry.cpp
            PROPERTIES COMPILE_OPTIONS -falign-functions=64)
    endif()
    if(UNIX AND NOT APPLE)
        # Include our own source information as well as the packaged backend symbols.
        target_compile_options(ovspatialquery PRIVATE -g)
        # Keep downloadable symbols separate from the runtime used by every test/install.
        add_custom_command(TARGET ovspatialquery POST_BUILD
            BYPRODUCTS "${CMAKE_BINARY_DIR}/symbols/libovspatialquery.so.debug"
            COMMAND "${_OVSPATIALQUERY_PYTHON_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/bootstrap/debug_symbols.py"
                --objcopy "${CMAKE_OBJCOPY}"
                --library "$<TARGET_FILE:ovspatialquery>"
                --symbols "${CMAKE_BINARY_DIR}/symbols/$<TARGET_FILE_NAME:ovspatialquery>.debug"
            VERBATIM)
        set_property(TARGET ovspatialquery APPEND PROPERTY LINK_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/bootstrap/debug_symbols.py")
    endif()


    install(TARGETS ovspatialquery
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
    include(cmake/package/public_headers.cmake)
    set(_OVSPATIALQUERY_PUBLIC_HEADER_FILES ${OVSPATIALQUERY_PUBLIC_HEADERS})
    list(TRANSFORM _OVSPATIALQUERY_PUBLIC_HEADER_FILES PREPEND include/ovspatialquery/)
    install(FILES ${_OVSPATIALQUERY_PUBLIC_HEADER_FILES}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ovspatialquery
    )
    if(OVSPATIALQUERY_BUILD_OVSTAGE_COMPANION)
        install(FILES include/ovspatialquery/${OVSPATIALQUERY_OVSTAGE_PUBLIC_HEADER}
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ovspatialquery)
    endif()
    install(FILES include/dlpack/dlpack.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dlpack)

    # Relocatable package config: find_package(ovspatialquery CONFIG) from the SDK root.
    # Releases within one 0.y line are compatible, as the Python binding assumes.
    include(CMakePackageConfigHelpers)
    set(_OVSPATIALQUERY_PACKAGE_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/ovspatialquery)
    file(RELATIVE_PATH _OVSPATIALQUERY_CONFIG_TO_ROOT
        "${CMAKE_CURRENT_BINARY_DIR}/prefix/${_OVSPATIALQUERY_PACKAGE_DIR}"
        "${CMAKE_CURRENT_BINARY_DIR}/prefix")
    set(_OVSPATIALQUERY_LIBRARY_FILE
        "${CMAKE_SHARED_LIBRARY_PREFIX}ovspatialquery${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(_OVSPATIALQUERY_IMPORT_LIBRARY_FILE
        "${CMAKE_IMPORT_LIBRARY_PREFIX}ovspatialquery${CMAKE_IMPORT_LIBRARY_SUFFIX}")
    configure_file(cmake/package/ovspatialqueryConfig.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/package/ovspatialqueryConfig.cmake" @ONLY)
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/package/ovspatialqueryConfigVersion.cmake"
        VERSION "${_OVSPATIALQUERY_VERSION_MAJOR}.${_OVSPATIALQUERY_VERSION_MINOR}.${_OVSPATIALQUERY_VERSION_PATCH}"
        COMPATIBILITY SameMinorVersion)
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/package/ovspatialqueryConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/package/ovspatialqueryConfigVersion.cmake"
        DESTINATION ${_OVSPATIALQUERY_PACKAGE_DIR})
    if(NOT _OVSPATIALQUERY_PYTHON_EXECUTABLE)
        message(FATAL_ERROR "SDK documentation packaging requires Python 3.12")
    endif()
    install(CODE "
        execute_process(
            COMMAND \"${CMAKE_COMMAND}\" -E env \"OVSPATIALQUERY_BUILD_VERSION=${OVSPATIALQUERY_VERSION}\"
                \"${_OVSPATIALQUERY_PYTHON_EXECUTABLE}\"
                \"${CMAKE_CURRENT_SOURCE_DIR}/tools/bootstrap/consumer_content.py\"
                install \"\${CMAKE_INSTALL_PREFIX}\"
                --version \"${OVSPATIALQUERY_VERSION}\"
                --source-revision \"${OVSPATIALQUERY_SOURCE_REVISION}\"
            COMMAND_ERROR_IS_FATAL ANY)
    ")
    install(FILES LICENSE NOTICE DESTINATION .)
    install(FILES LICENSES/Apache-2.0.txt
        DESTINATION licenses RENAME dlpack-LICENSE.txt)
    install(FILES "${OVSPATIALQUERY_PHYSX_LICENSE_SOURCE}"
        DESTINATION licenses RENAME physx-LICENSE.md)
    install(FILES "${OVSPATIALQUERY_PHYSX_PROVENANCE_SOURCE}"
        DESTINATION licenses RENAME physx-PROVENANCE.json)
    if(MSVC)
        install(FILES "$<TARGET_PDB_FILE:ovspatialquery>"
            DESTINATION ${CMAKE_INSTALL_BINDIR}
            OPTIONAL
        )
    endif()

    if(OVSPATIALQUERY_APP_ARTIFACT_DIR)
        get_filename_component(_OVSPATIALQUERY_APP_ARTIFACT_DIR
            "${OVSPATIALQUERY_APP_ARTIFACT_DIR}" ABSOLUTE)
        get_filename_component(_OVSPATIALQUERY_BOOTSTRAP_INSTALL
            "${CMAKE_BINARY_DIR}/install" ABSOLUTE)
        if(NOT _OVSPATIALQUERY_APP_ARTIFACT_DIR STREQUAL _OVSPATIALQUERY_BOOTSTRAP_INSTALL)
            message(FATAL_ERROR "OVSPATIALQUERY_APP_ARTIFACT_DIR must be the build-local install directory")
        endif()
        set(_OVSPATIALQUERY_INSTALL_ALLOWED_PARENT "${CMAKE_BINARY_DIR}")
        set(_OVSPATIALQUERY_INSTALL_ALLOWED_LEAF "install")
        if(_OVSPATIALQUERY_PYTHON_EXECUTABLE)
            add_custom_target(ovspatialquery_verify_install
                COMMAND "${CMAKE_COMMAND}"
                    "-DOVSPATIALQUERY_BUILD_DIR=${CMAKE_BINARY_DIR}"
                    "-DOVSPATIALQUERY_BUILD_CONFIG=$<CONFIG>"
                    "-DOVSPATIALQUERY_VERSION=${OVSPATIALQUERY_VERSION}"
                    "-DOVSPATIALQUERY_INSTALL_ROOT=${OVSPATIALQUERY_APP_ARTIFACT_DIR}"
                    "-DOVSPATIALQUERY_ALLOWED_PARENT=${_OVSPATIALQUERY_INSTALL_ALLOWED_PARENT}"
                    "-DOVSPATIALQUERY_ALLOWED_LEAF=${_OVSPATIALQUERY_INSTALL_ALLOWED_LEAF}"
                    ${_OVSPATIALQUERY_INSPECTOR_ARGS}
                    "-DOVSPATIALQUERY_EXPECT_OVSTAGE_COMPANION=${OVSPATIALQUERY_BUILD_OVSTAGE_COMPANION}"
                    "-DOVSPATIALQUERY_EXPECT_CUDA_BACKEND=${OVSPATIALQUERY_HAS_CUDA_BACKEND}"
                    "-DOVSPATIALQUERY_EXPECT_SOURCE_REVISION=${OVSPATIALQUERY_SOURCE_REVISION}"
                    "-DOVSPATIALQUERY_PHYSX_SOURCE_PROVENANCE=${OVSPATIALQUERY_PHYSX_PROVENANCE_SOURCE}"
                    "-DOVSPATIALQUERY_PYTHON_EXECUTABLE=${_OVSPATIALQUERY_PYTHON_EXECUTABLE}"
                    "-DOVSPATIALQUERY_CTYPES_TEST=${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/installed/ctypes_api_test.py"
                    "-DOVSPATIALQUERY_LIBRARY_RELATIVE_PATH=${_OVSPATIALQUERY_INSTALLED_LIBRARY_PATH}"
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/package/install_verified_artifacts.cmake"
                DEPENDS ovspatialquery
                USES_TERMINAL
            )
            add_custom_target(ovspatialquery_install)
            add_dependencies(ovspatialquery_install ovspatialquery_verify_install)
        endif()
    endif()
endif()

# A native test that stalls with a live worker (a hang, or a failure path that
# waits on one) fails at this bound instead of stalling the lane. The slowest
# native test takes about 42 s (ASan Debug); CUDA tests about 21 s (Release).
set(OVSPATIALQUERY_NATIVE_TEST_TIMEOUT 300)

# One native test executable registered under its own name, with the timeout
# above. INCLUDES, DEFINITIONS and LINK are all PRIVATE; anything else (PhysX
# linkage, PTX copies, other test properties) follows the call.
function(ovspatialquery_add_native_test test_name)
    cmake_parse_arguments(_test "" "" "SOURCES;INCLUDES;DEFINITIONS;LINK" ${ARGN})
    add_executable(${test_name} ${_test_SOURCES})
    if(_test_INCLUDES)
        target_include_directories(${test_name} PRIVATE ${_test_INCLUDES})
    endif()
    if(_test_DEFINITIONS)
        target_compile_definitions(${test_name} PRIVATE ${_test_DEFINITIONS})
    endif()
    if(_test_LINK)
        target_link_libraries(${test_name} PRIVATE ${_test_LINK})
    endif()
    if(MSVC)
        target_compile_options(${test_name} PRIVATE /EHsc /W4 /WX /permissive- /utf-8)
    else()
        target_compile_options(${test_name} PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wfloat-conversion -Werror)
    endif()
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES TIMEOUT ${OVSPATIALQUERY_NATIVE_TEST_TIMEOUT})
endfunction()

if(OVSPATIALQUERY_BUILD_TESTS)
    add_executable(ovspatialquery_python_abi_witness
        tests/integration/installed/python_abi_witness.c)
    target_include_directories(ovspatialquery_python_abi_witness PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include")
    if(OVSPATIALQUERY_BUILD_OVSTAGE_COMPANION)
        target_compile_definitions(ovspatialquery_python_abi_witness PRIVATE
            OVSPATIALQUERY_ABI_WITH_OVSTAGE)
        target_link_libraries(ovspatialquery_python_abi_witness PRIVATE
            ovstage::ovstage)
    endif()
    if(OVSPATIALQUERY_BUILD_OVSTAGE_COMPANION)
        set(OVSPATIALQUERY_ABI_WITNESS_GOLDEN
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/tooling/cmake/abi_witness/current_api_with_ovstage.txt")
    else()
        set(OVSPATIALQUERY_ABI_WITNESS_GOLDEN
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/tooling/cmake/abi_witness/current_api.txt")
    endif()
    add_test(NAME ovspatialquery_python_abi_witness_test
        COMMAND "${CMAKE_COMMAND}"
            "-DOVSPATIALQUERY_ABI_WITNESS=$<TARGET_FILE:ovspatialquery_python_abi_witness>"
            "-DOVSPATIALQUERY_ABI_WITNESS_GOLDEN=${OVSPATIALQUERY_ABI_WITNESS_GOLDEN}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/tooling/cmake/verify_abi_witness.cmake")
    if(OVSPATIALQUERY_BUILD_OVSTAGE_COMPANION AND NOT WIN32)
        # GPU qualification runs this transferred witness from another checkout
        # path than its build-tree RUNPATH names; CTest is regenerated there.
        set_tests_properties(ovspatialquery_python_abi_witness_test PROPERTIES
            ENVIRONMENT_MODIFICATION
            "LD_LIBRARY_PATH=path_list_prepend:${OVSPATIALQUERY_OVSTAGE_SDK_DIR}/bin")
    endif()

    enable_testing()
    add_test(NAME ovspatialquery_source_version_test
        COMMAND "${CMAKE_COMMAND}"
            "-DOVSPATIALQUERY_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
            "-DOVSPATIALQUERY_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/source-version-test"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/tooling/cmake/verify_source_version.cmake")
    if(OVSPATIALQUERY_BUILD_PRODUCT)
        add_test(NAME ovspatialquery_core_install_without_ovstage_test
            COMMAND "${CMAKE_COMMAND}"
                "-DOVSPATIALQUERY_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
                "-DOVSPATIALQUERY_BUILD_ROOT=${CMAKE_CURRENT_BINARY_DIR}/core-only-install"
                "-DOVSPATIALQUERY_TARGET_DEPS=${OVSPATIALQUERY_TARGET_DEPS}"
                "-DOVSPATIALQUERY_PHYSX_SDK_DIR=$CACHE{OVSPATIALQUERY_PHYSX_SDK_DIR}"
                "-DOVSPATIALQUERY_BUILD_CONFIG=$<CONFIG>"
                "-DOVSPATIALQUERY_PREBUILT_CORE_INDEPENDENCE=${OVSPATIALQUERY_PREBUILT_CORE_INDEPENDENCE}"
                "-DOVSPATIALQUERY_VERSION=${OVSPATIALQUERY_VERSION}"
                "-DOVSPATIALQUERY_SOURCE_REVISION=${OVSPATIALQUERY_SOURCE_REVISION}"
                "-DOVSPATIALQUERY_CTEST_COMMAND=${CMAKE_CTEST_COMMAND}"
                "-DOVSPATIALQUERY_GENERATOR=${CMAKE_GENERATOR}"
                "-DOVSPATIALQUERY_C_COMPILER=${CMAKE_C_COMPILER}"
                "-DOVSPATIALQUERY_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
                "-DOVSPATIALQUERY_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
                "-DOVSPATIALQUERY_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
                ${_OVSPATIALQUERY_INSPECTOR_ARGS}
                "-DOVSPATIALQUERY_PYTHON_EXECUTABLE=${_OVSPATIALQUERY_PYTHON_EXECUTABLE}"
                "-DOVSPATIALQUERY_C_FLAGS=${CMAKE_C_FLAGS}"
                "-DOVSPATIALQUERY_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
                "-DOVSPATIALQUERY_C_FLAGS_CONFIG=$<$<CONFIG:Debug>:${CMAKE_C_FLAGS_DEBUG}>$<$<CONFIG:Release>:${CMAKE_C_FLAGS_RELEASE}>$<$<CONFIG:RelWithDebInfo>:${CMAKE_C_FLAGS_RELWITHDEBINFO}>$<$<CONFIG:MinSizeRel>:${CMAKE_C_FLAGS_MINSIZEREL}>"
                "-DOVSPATIALQUERY_CXX_FLAGS_CONFIG=$<$<CONFIG:Debug>:${CMAKE_CXX_FLAGS_DEBUG}>$<$<CONFIG:Release>:${CMAKE_CXX_FLAGS_RELEASE}>$<$<CONFIG:RelWithDebInfo>:${CMAKE_CXX_FLAGS_RELWITHDEBINFO}>$<$<CONFIG:MinSizeRel>:${CMAKE_CXX_FLAGS_MINSIZEREL}>"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/tooling/cmake/verify_core_independence.cmake")
        set_tests_properties(ovspatialquery_core_install_without_ovstage_test PROPERTIES
            RUN_SERIAL TRUE)
    endif()
    # Test runtimes bind the private backend seam (ovspatialquery_backend.h) at
    # link time: the core and Ovstage companion tests to the deterministic
    # oracle, tests/native/physx to PxBVH. Both backends define the same
    # symbols, so they stay in two archives, but the library sources they share
    # are compiled once, with every test hook, into one object library.
    function(ovspatialquery_configure_test_runtime target)
        target_include_directories(${target}
            PUBLIC include tests/support/native support/native
            PRIVATE source
        )
        target_include_directories(${target} SYSTEM PRIVATE "${PHYSX_SDK_DIR}/include")
        target_compile_definitions(${target} PRIVATE
            PX_PHYSX_STATIC_LIB
            OVSPATIALQUERY_ENABLE_TEST_HOOKS
            OVSPATIALQUERY_ENABLE_INSTANCE_OPERATION_RUNTIME_TEST_HOOK
            OVSPATIALQUERY_ENABLE_PAIR_MATERIALIZATION_TEST_HOOK
            OVSPATIALQUERY_ENABLE_PAIR_REPORTER_OPERATION_TEST_HOOK
            OVSPATIALQUERY_ENABLE_FAMILY_BATCH_PLAN_OPERATION_TEST_HOOK
            OVSPATIALQUERY_ENABLE_TARGET_RESOLVER_TEST_HOOK)
        if(OVSPATIALQUERY_HAS_CUDA_BACKEND)
            target_compile_definitions(${target} PRIVATE OVSPATIALQUERY_HAS_CUDA_BACKEND)
        endif()
        get_target_property(_type ${target} TYPE)
        if(_type STREQUAL "STATIC_LIBRARY")
            target_sources(${target} PRIVATE $<TARGET_OBJECTS:ovspatialquery_test_runtime_objects>)
            ovspatialquery_link_packaged_physx(${target} FULL)
            if(OVSPATIALQUERY_HAS_CUDA_BACKEND)
                target_link_libraries(${target} PUBLIC ${CMAKE_DL_LIBS})
                add_dependencies(${target} ovspatialquery_cuda_ptx)
            endif()
        endif()
    endfunction()
    add_library(ovspatialquery_test_runtime_objects OBJECT ${_OVSPATIALQUERY_RUNTIME_SOURCES})
    ovspatialquery_configure_test_runtime(ovspatialquery_test_runtime_objects)
    add_library(ovspatialquery_test_runtime STATIC
        ${_OVSPATIALQUERY_PORTABLE_TEST_RUNTIME_SOURCES})
    ovspatialquery_configure_test_runtime(ovspatialquery_test_runtime)

    ovspatialquery_add_native_test(ovspatialquery_candidate_pairs_test
        SOURCES tests/native/core/candidate_pairs_test.cpp
        INCLUDES source
        DEFINITIONS OVSPATIALQUERY_ENABLE_INSTANCE_OPERATION_RUNTIME_TEST_HOOK
            OVSPATIALQUERY_ENABLE_PAIR_MATERIALIZATION_TEST_HOOK
        LINK ovspatialquery_test_runtime)

    ovspatialquery_add_native_test(ovspatialquery_instance_host_ledger_test
        SOURCES tests/native/core/instance_host_ledger_test.cpp
            source/ovspatialquery_instance_host_ledger.cpp
        INCLUDES source)

    ovspatialquery_add_native_test(ovspatialquery_pair_materialization_test
        SOURCES tests/native/core/pair_materialization_test.cpp
        INCLUDES source
        DEFINITIONS OVSPATIALQUERY_ENABLE_PAIR_MATERIALIZATION_TEST_HOOK
        LINK ovspatialquery_test_runtime)

    ovspatialquery_add_native_test(ovspatialquery_pair_reporter_operation_test
        SOURCES tests/native/core/pair_reporter_operation_test.cpp
        INCLUDES source
        DEFINITIONS OVSPATIALQUERY_ENABLE_INSTANCE_OPERATION_RUNTIME_TEST_HOOK
            OVSPATIALQUERY_ENABLE_PAIR_MATERIALIZATION_TEST_HOOK
            OVSPATIALQUERY_ENABLE_PAIR_REPORTER_OPERATION_TEST_HOOK
        LINK ovspatialquery_test_runtime)

    ovspatialquery_add_native_test(ovspatialquery_object_table_test
        SOURCES tests/native/core/object_table_test.cpp
        INCLUDES source
        LINK ovspatialquery_test_runtime)

    ovspatialquery_add_native_test(ovspatialquery_pair_reporter_patch_test
        SOURCES tests/native/core/pair_reporter_patch_test.cpp
        INCLUDES source
        DEFINITIONS OVSPATIALQUERY_ENABLE_PAIR_REPORTER_OPERATION_TEST_HOOK
        LINK ovspatialquery_test_runtime)

    ovspatialquery_add_native_test(ovspatialquery_population_table_test
        SOURCES tests/native/core/population_table_test.cpp
            tests/native/core/publication_status_test.cpp
        INCLUDES source
        DEFINITIONS OVSPATIALQUERY_ENABLE_INSTANCE_OPERATION_RUNTIME_TEST_HOOK
        LINK ovspatialquery_test_runtime)

    ovspatialquery_add_native_test(ovspatialquery_target_resolver_test
        SOURCES tests/native/core/target_resolver_test.cpp
        INCLUDES source
        DEFINITIONS OVSPATIALQUERY_ENABLE_TARGET_RESOLVER_TEST_HOOK
        LINK ovspatialquery_test_runtime)

    ovspatialquery_add_native_test(ovspatialquery_instance_operation_runtime_test
        SOURCES tests/native/core/instance_operation_runtime_test.cpp
        INCLUDES source
        DEFINITIONS OVSPATIALQUERY_ENABLE_INSTANCE_OPERATION_RUNTIME_TEST_HOOK
            OVSPATIALQUERY_ENABLE_FAMILY_BATCH_PLAN_OPERATION_TEST_HOOK
            OVSPATIALQUERY_ENABLE_PAIR_REPORTER_OPERATION_TEST_HOOK
        LINK ovspatialquery_test_runtime)

    ovspatialquery_add_native_test(ovspatialquery_operation_acceptance_test
        SOURCES tests/native/core/operation_acceptance_test.cpp
        INCLUDES source
        DEFINITIONS OVSPATIALQUERY_ENABLE_INSTANCE_OPERATION_RUNTIME_TEST_HOOK
        LINK ovspatialquery_test_runtime)

    ovspatialquery_add_native_test(ovspatialquery_api_foundation_test
        SOURCES tests/native/core/api_foundation_test.cpp
        INCLUDES source
        LINK ovspatialquery_test_runtime)

    ovspatialquery_add_native_test(ovspatialquery_participation_capture_test
        SOURCES tests/native/core/participation_capture_test.cpp
        INCLUDES source
        LINK ovspatialquery_test_runtime)

    ovspatialquery_add_native_test(ovspatialquery_precise_geometry_test
        SOURCES tests/native/core/precise_geometry_test.cpp
        LINK ovspatialquery_test_runtime)

    ovspatialquery_add_native_test(ovspatialquery_typed_geometry_test
        SOURCES tests/native/core/typed_geometry_test.cpp
        INCLUDES source
        LINK ovspatialquery_test_runtime)

    ovspatialquery_add_native_test(ovspatialquery_box_sweep_geometry_test
        SOURCES tests/native/core/box_sweep_geometry_test.cpp
        INCLUDES source
        LINK ovspatialquery_test_runtime)

    ovspatialquery_add_native_test(ovspatialquery_sphere_sweep_contact_test
        SOURCES tests/native/core/sphere_sweep_contact_test.cpp
        INCLUDES source
        LINK ovspatialquery_test_runtime)

    ovspatialquery_add_native_test(ovspatialquery_geometry_identity_test
        SOURCES tests/native/core/geometry_identity_test.cpp
        INCLUDES source
        DEFINITIONS OVSPATIALQUERY_ENABLE_TEST_HOOKS
        LINK ovspatialquery_test_runtime)

    ovspatialquery_add_native_test(ovspatialquery_explicit_preparation_test
        SOURCES tests/native/core/explicit_preparation_test.cpp
        INCLUDES source
        DEFINITIONS OVSPATIALQUERY_ENABLE_TEST_HOOKS
        LINK ovspatialquery_test_runtime)

    ovspatialquery_add_native_test(ovspatialquery_reader_admission_budget_test
        SOURCES tests/native/core/reader_admission_budget_test.cpp
        LINK ovspatialquery_test_runtime)
    ovspatialquery_add_native_test(ovspatialquery_explicit_api_test
        SOURCES tests/native/core/explicit_api_test.cpp
        INCLUDES source
        DEFINITIONS OVSPATIALQUERY_ENABLE_INSTANCE_OPERATION_RUNTIME_TEST_HOOK
        LINK ovspatialquery_test_runtime)
    if(OVSPATIALQUERY_HAS_CUDA_BACKEND)
        target_compile_definitions(ovspatialquery_explicit_api_test PRIVATE
            OVSPATIALQUERY_TEST_HAS_CUDA_BACKEND=1)
        add_dependencies(ovspatialquery_explicit_api_test ovspatialquery_cuda_ptx)
        ovspatialquery_copy_cuda_ptx(ovspatialquery_explicit_api_test)
    endif()

    ovspatialquery_add_native_test(ovspatialquery_family_batch_semantics_test
        SOURCES tests/native/core/family_batch_semantics_test.cpp
        INCLUDES source
        LINK ovspatialquery_test_runtime)

    ovspatialquery_add_native_test(ovspatialquery_family_query_sync_test
        SOURCES tests/native/core/family_query_sync_test.cpp
        INCLUDES source
        LINK ovspatialquery_test_runtime)

    ovspatialquery_add_native_test(ovspatialquery_family_batch_plan_operation_test
        SOURCES tests/native/core/family_batch_plan_operation_test.cpp
        INCLUDES source
        DEFINITIONS OVSPATIALQUERY_ENABLE_INSTANCE_OPERATION_RUNTIME_TEST_HOOK
            OVSPATIALQUERY_ENABLE_FAMILY_BATCH_PLAN_OPERATION_TEST_HOOK
        LINK ovspatialquery_test_runtime)

    ovspatialquery_add_native_test(ovspatialquery_batch_api_test
        SOURCES tests/native/core/batch_api_test.cpp
        INCLUDES source
        LINK ovspatialquery_test_runtime)
    # The dense host-budget sweep can exceed five minutes with MSVC's Debug checks.
    # Keep every budget and result assertion; only this configuration gets more time.
    if(MSVC)
        set_tests_properties(ovspatialquery_batch_api_test PROPERTIES
            TIMEOUT "$<IF:$<CONFIG:Debug>,900,${OVSPATIALQUERY_NATIVE_TEST_TIMEOUT}>")
    endif()

    ovspatialquery_add_native_test(ovspatialquery_sweep_modes_test
        SOURCES tests/native/core/sweep_modes_test.cpp
        LINK ovspatialquery_test_runtime)

    ovspatialquery_add_native_test(ovspatialquery_operation_teardown_race_test
        SOURCES tests/native/core/operation_teardown_race_test.cpp
        INCLUDES source
        DEFINITIONS OVSPATIALQUERY_ENABLE_INSTANCE_OPERATION_RUNTIME_TEST_HOOK
        LINK ovspatialquery_test_runtime)

    if(OVSPATIALQUERY_HAS_CUDA_BACKEND)
        ovspatialquery_add_native_test(ovspatialquery_cuda_consumer_transfer_test
            SOURCES tests/native/cuda/consumer_transfer_test.cpp
            INCLUDES source
            LINK ovspatialquery_test_runtime)
        ovspatialquery_copy_cuda_ptx(ovspatialquery_cuda_consumer_transfer_test)
        ovspatialquery_copy_cuda_test_ptx(ovspatialquery_cuda_consumer_transfer_test)

        ovspatialquery_add_native_test(ovspatialquery_cuda_family_api_test
            SOURCES tests/native/cuda/family_api_test.cpp
            DEFINITIONS OVSPATIALQUERY_ENABLE_FAMILY_BATCH_PLAN_OPERATION_TEST_HOOK
                OVSPATIALQUERY_ENABLE_INSTANCE_OPERATION_RUNTIME_TEST_HOOK
            INCLUDES source
            LINK ovspatialquery_test_runtime)
        ovspatialquery_copy_cuda_ptx(ovspatialquery_cuda_family_api_test)
        ovspatialquery_copy_cuda_test_ptx(ovspatialquery_cuda_family_api_test)

        # The patch test's scenarios with CUDA result devices (ADR-0019 stage 3).
        ovspatialquery_add_native_test(ovspatialquery_cuda_pair_reporter_test
            SOURCES tests/native/core/pair_reporter_patch_test.cpp
            INCLUDES source
            DEFINITIONS OVSPATIALQUERY_ENABLE_PAIR_REPORTER_OPERATION_TEST_HOOK
                OVSPATIALQUERY_ENABLE_INSTANCE_OPERATION_RUNTIME_TEST_HOOK
                OVSPATIALQUERY_TEST_CUDA_RESULTS
            LINK ovspatialquery_test_runtime)
        ovspatialquery_copy_cuda_ptx(ovspatialquery_cuda_pair_reporter_test)
        ovspatialquery_copy_cuda_test_ptx(ovspatialquery_cuda_pair_reporter_test)

        ovspatialquery_add_native_test(ovspatialquery_cuda_family_plan_test
            SOURCES tests/native/cuda/family_plan_test.cpp
            INCLUDES source
            DEFINITIONS OVSPATIALQUERY_ENABLE_TEST_HOOKS
                OVSPATIALQUERY_ENABLE_INSTANCE_OPERATION_RUNTIME_TEST_HOOK
            LINK ovspatialquery_test_runtime)
        ovspatialquery_copy_cuda_ptx(ovspatialquery_cuda_family_plan_test)
        ovspatialquery_copy_cuda_test_ptx(ovspatialquery_cuda_family_plan_test)

        ovspatialquery_add_native_test(ovspatialquery_cuda_host_mapped_test
            SOURCES tests/native/cuda/host_mapped_test.cpp
            DEFINITIONS OVSPATIALQUERY_ENABLE_INSTANCE_OPERATION_RUNTIME_TEST_HOOK
            INCLUDES source
            LINK ovspatialquery_test_runtime)
        ovspatialquery_copy_cuda_ptx(ovspatialquery_cuda_host_mapped_test)
        ovspatialquery_copy_cuda_test_ptx(ovspatialquery_cuda_host_mapped_test)

        ovspatialquery_add_native_test(ovspatialquery_cuda_operation_teardown_race_test
            SOURCES tests/native/cuda/operation_teardown_race_test.cpp
            LINK ovspatialquery_test_runtime)
        ovspatialquery_copy_cuda_ptx(ovspatialquery_cuda_operation_teardown_race_test)

        # The sweep-modes scenarios again, through CUDA plans.
        ovspatialquery_add_native_test(ovspatialquery_cuda_sweep_modes_test
            SOURCES tests/native/core/sweep_modes_test.cpp
            DEFINITIONS OVSPATIALQUERY_TEST_CUDA_PLANS
            LINK ovspatialquery_test_runtime)
        ovspatialquery_copy_cuda_ptx(ovspatialquery_cuda_sweep_modes_test)

        set_tests_properties(
            ovspatialquery_cuda_consumer_transfer_test
            ovspatialquery_cuda_family_api_test
            ovspatialquery_cuda_family_plan_test
            ovspatialquery_cuda_operation_teardown_race_test
            ovspatialquery_cuda_pair_reporter_test
            ovspatialquery_cuda_sweep_modes_test
            PROPERTIES RUN_SERIAL TRUE)
    endif()

    ovspatialquery_add_native_test(ovspatialquery_view_api_test
        SOURCES tests/native/core/view_api_test.cpp
        LINK ovspatialquery_test_runtime)

    ovspatialquery_add_native_test(ovspatialquery_material_identity_test
        SOURCES tests/native/core/material_identity_test.cpp
        LINK ovspatialquery_test_runtime)

    if(OVSPATIALQUERY_BUILD_OVSTAGE_COMPANION)
        # An optional third argument names the runtime to link instead of the portable one.
        function(ovspatialquery_add_ovstage_companion_test test_name source_file)
            set(runtime ovspatialquery_test_runtime)
            if(ARGC GREATER 2)
                set(runtime ${ARGV2})
            endif()
            add_executable(${test_name} ${source_file})
            target_include_directories(${test_name} PRIVATE source)
            target_compile_definitions(${test_name} PRIVATE
                OVSPATIALQUERY_ENABLE_TEST_HOOKS)
            target_link_libraries(${test_name} PRIVATE
                ${runtime} ovstage::ovstage)
            if(WIN32)
                add_test(NAME ${test_name}
                    COMMAND "${CMAKE_COMMAND}" -E env
                        "PATH=${OVSPATIALQUERY_OVSTAGE_SDK_DIR}/bin/plugins$<SEMICOLON>${OVSPATIALQUERY_OVSTAGE_SDK_DIR}/bin$<SEMICOLON>$ENV{PATH}"
                        "$<TARGET_FILE:${test_name}>")
            else()
                add_test(NAME ${test_name}
                    COMMAND "${CMAKE_COMMAND}" -E env
                        "LD_LIBRARY_PATH=${OVSPATIALQUERY_OVSTAGE_SDK_DIR}/bin/plugins:${OVSPATIALQUERY_OVSTAGE_SDK_DIR}/bin:$ENV{LD_LIBRARY_PATH}"
                        "$<TARGET_FILE:${test_name}>")
            endif()
            set_tests_properties(${test_name} PROPERTIES TIMEOUT ${OVSPATIALQUERY_NATIVE_TEST_TIMEOUT})
        endfunction()

        ovspatialquery_add_ovstage_companion_test(
            ovspatialquery_ovstage_companion_instance_validation_test
            tests/integration/ovstage/companion_instance_validation_test.cpp)
        ovspatialquery_add_ovstage_companion_test(
            ovspatialquery_ovstage_companion_enqueue_lifecycle_test
            tests/integration/ovstage/companion_enqueue_lifecycle_test.cpp)
        ovspatialquery_add_ovstage_companion_test(
            ovspatialquery_ovstage_companion_aabb_source_test
            tests/integration/ovstage/companion_aabb_source_test.cpp)
        ovspatialquery_add_ovstage_companion_test(
            ovspatialquery_ovstage_companion_precise_source_test
            tests/integration/ovstage/companion_precise_source_test.cpp)
        ovspatialquery_add_ovstage_companion_test(
            ovspatialquery_ovstage_companion_shared_mesh_test
            tests/integration/ovstage/companion_shared_mesh_test.cpp)
        ovspatialquery_add_ovstage_companion_test(
            ovspatialquery_ovstage_companion_material_test
            tests/integration/ovstage/companion_material_test.cpp)
        ovspatialquery_add_ovstage_companion_test(
            ovspatialquery_ovstage_companion_path_filter_test
            tests/integration/ovstage/companion_path_filter_test.cpp)
        ovspatialquery_add_ovstage_companion_test(
            ovspatialquery_ovstage_companion_path_scope_test
            tests/integration/ovstage/companion_path_scope_test.cpp)
        ovspatialquery_add_ovstage_companion_test(
            ovspatialquery_ovstage_companion_device_publication_test
            tests/integration/ovstage/companion_device_publication_test.cpp)
        target_include_directories(ovspatialquery_ovstage_companion_device_publication_test PRIVATE support/native)
        if(OVSPATIALQUERY_HAS_CUDA_BACKEND)
            ovspatialquery_copy_cuda_ptx(ovspatialquery_ovstage_companion_device_publication_test)
        endif()
        set_tests_properties(ovspatialquery_ovstage_companion_device_publication_test PROPERTIES RUN_SERIAL TRUE)
        if(OVSPATIALQUERY_BUILD_PRODUCT)
            add_test(NAME ovspatialquery_ovstage_installed_consumers_test
                COMMAND "${CMAKE_COMMAND}"
                    "-DOVSPATIALQUERY_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
                    "-DOVSPATIALQUERY_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}"
                    "-DOVSPATIALQUERY_BUILD_CONFIG=$<CONFIG>"
                    "-DOVSPATIALQUERY_INSTALL_ROOT=${CMAKE_CURRENT_BINARY_DIR}/_test-install-ovstage"
                    "-DOVSPATIALQUERY_CONSUMER_BUILD=${CMAKE_CURRENT_BINARY_DIR}/ovstage-installed-consumers"
                    "-DOVSPATIALQUERY_OVSTAGE_SDK_DIR=${OVSPATIALQUERY_OVSTAGE_SDK_DIR}"
                    "-DOVSPATIALQUERY_GENERATOR=${CMAKE_GENERATOR}"
                    "-DOVSPATIALQUERY_C_COMPILER=${CMAKE_C_COMPILER}"
                    "-DOVSPATIALQUERY_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
                    "-DOVSPATIALQUERY_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
                    "-DOVSPATIALQUERY_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/tooling/cmake/verify_ovstage_installed_consumers.cmake")
            set_tests_properties(ovspatialquery_ovstage_installed_consumers_test PROPERTIES
                RUN_SERIAL TRUE)
        endif()
    endif()

    if(OVSPATIALQUERY_BUILD_PRODUCT)
        add_library(ovspatialquery_physx_test_runtime STATIC
            ${_OVSPATIALQUERY_PHYSX_TEST_RUNTIME_SOURCES})
        ovspatialquery_configure_test_runtime(ovspatialquery_physx_test_runtime)

        ovspatialquery_add_native_test(ovspatialquery_physx_backend_contract_test
            SOURCES tests/native/physx/backend_contract_test.cpp
            INCLUDES source
            DEFINITIONS PX_PHYSX_STATIC_LIB
            LINK ovspatialquery_physx_test_runtime)
        # The vertex-limit test cooks its oracle hull with PhysX directly (ADR-0078).
        target_include_directories(ovspatialquery_physx_backend_contract_test SYSTEM PRIVATE
            "${PHYSX_SDK_DIR}/include")
        if(OVSPATIALQUERY_BUILD_OVSTAGE_COMPANION)
            # The Ovstage adapter on the PhysX backend, whose convexHull Meshes the test compares
            # with PhysX cooking their points directly (ADR-0078).
            add_library(ovspatialquery_ovstage_physx_test_runtime STATIC
                ${_OVSPATIALQUERY_PHYSX_TEST_RUNTIME_SOURCES} ${_OVSPATIALQUERY_OVSTAGE_RUNTIME_SOURCES})
            ovspatialquery_configure_test_runtime(ovspatialquery_ovstage_physx_test_runtime)
            ovspatialquery_add_ovstage_companion_test(
                ovspatialquery_ovstage_companion_physx_convex_hull_test
                tests/integration/ovstage/companion_physx_convex_hull_test.cpp
                ovspatialquery_ovstage_physx_test_runtime)
            target_compile_definitions(ovspatialquery_ovstage_companion_physx_convex_hull_test PRIVATE
                PX_PHYSX_STATIC_LIB)
            target_include_directories(ovspatialquery_ovstage_companion_physx_convex_hull_test SYSTEM PRIVATE
                "${PHYSX_SDK_DIR}/include")
            ovspatialquery_link_packaged_physx(ovspatialquery_ovstage_companion_physx_convex_hull_test FULL)
        endif()
        ovspatialquery_add_native_test(ovspatialquery_physx_closest_query_test
            SOURCES tests/native/physx/closest_query_test.cpp
            LINK ovspatialquery_physx_test_runtime)

        ovspatialquery_add_native_test(ovspatialquery_physx_generation_concurrency_test
            SOURCES tests/native/physx/generation_concurrency_test.cpp
            DEFINITIONS PX_PHYSX_STATIC_LIB)
        target_include_directories(ovspatialquery_physx_generation_concurrency_test SYSTEM PRIVATE
            "${PHYSX_SDK_DIR}/include")
        ovspatialquery_link_packaged_physx(ovspatialquery_physx_generation_concurrency_test FULL)

        ovspatialquery_add_native_test(ovspatialquery_physx_query_api_contract_test
            SOURCES tests/native/physx/query_api_contract_test.cpp
            INCLUDES source
            DEFINITIONS OVSPATIALQUERY_ENABLE_TEST_HOOKS
            LINK ovspatialquery_physx_test_runtime)
        ovspatialquery_link_packaged_physx(ovspatialquery_physx_query_api_contract_test FULL)
        ovspatialquery_add_native_test(ovspatialquery_physx_overlap_modes_test
            SOURCES tests/native/physx/overlap_modes_test.cpp
            LINK ovspatialquery_physx_test_runtime)

        ovspatialquery_add_native_test(ovspatialquery_physx_dll_stress_test
            SOURCES tests/native/physx/dll_stress_test.cpp
            INCLUDES source
            LINK ovspatialquery_physx_test_runtime)

        foreach(_ovspatialquery_physx_test
                ovspatialquery_physx_backend_contract_test
                ovspatialquery_physx_closest_query_test
                ovspatialquery_physx_overlap_modes_test
                ovspatialquery_physx_dll_stress_test)
            ovspatialquery_link_packaged_physx(
                ${_ovspatialquery_physx_test} FULL)
        endforeach()

        if(_OVSPATIALQUERY_PYTHON_EXECUTABLE)
            set(_OVSPATIALQUERY_TEST_INSTALL_ROOT "${CMAKE_CURRENT_BINARY_DIR}/_test-install")
            add_test(NAME ovspatialquery_install_artifacts_test
                COMMAND "${CMAKE_COMMAND}"
                    "-DOVSPATIALQUERY_BUILD_DIR=${CMAKE_BINARY_DIR}"
                    "-DOVSPATIALQUERY_BUILD_CONFIG=$<CONFIG>"
                    "-DOVSPATIALQUERY_INSTALL_ROOT=${_OVSPATIALQUERY_TEST_INSTALL_ROOT}"
                    "-DOVSPATIALQUERY_ALLOWED_PARENT=${CMAKE_CURRENT_BINARY_DIR}"
                    "-DOVSPATIALQUERY_ALLOWED_LEAF=_test-install"
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/package/install_artifacts.cmake")
            set_tests_properties(ovspatialquery_install_artifacts_test PROPERTIES
                FIXTURES_SETUP ovspatialquery_installed_artifacts)

            add_test(NAME ovspatialquery_installed_exports_test
                COMMAND "${CMAKE_COMMAND}"
                    "-DOVSPATIALQUERY_INSTALL_ROOT=${_OVSPATIALQUERY_TEST_INSTALL_ROOT}"
                    ${_OVSPATIALQUERY_INSPECTOR_ARGS}
                    "-DOVSPATIALQUERY_EXPECT_OVSTAGE_COMPANION=${OVSPATIALQUERY_BUILD_OVSTAGE_COMPANION}"
                    "-DOVSPATIALQUERY_EXPECT_CUDA_BACKEND=${OVSPATIALQUERY_HAS_CUDA_BACKEND}"
                    "-DOVSPATIALQUERY_EXPECT_SOURCE_REVISION=${OVSPATIALQUERY_SOURCE_REVISION}"
                    "-DOVSPATIALQUERY_PHYSX_SOURCE_PROVENANCE=${OVSPATIALQUERY_PHYSX_PROVENANCE_SOURCE}"
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/package/verify_install.cmake")
            set_tests_properties(ovspatialquery_installed_exports_test PROPERTIES
                FIXTURES_REQUIRED ovspatialquery_installed_artifacts)

            add_executable(ovspatialquery_installed_python_abi_witness
                EXCLUDE_FROM_ALL
                tests/integration/installed/python_abi_witness.c)
            target_compile_features(ovspatialquery_installed_python_abi_witness
                PRIVATE c_std_11)
            target_include_directories(ovspatialquery_installed_python_abi_witness
                PRIVATE "${_OVSPATIALQUERY_TEST_INSTALL_ROOT}/include")
            if(OVSPATIALQUERY_BUILD_OVSTAGE_COMPANION)
                target_compile_definitions(
                    ovspatialquery_installed_python_abi_witness PRIVATE
                    OVSPATIALQUERY_ABI_WITH_OVSTAGE)
                target_link_libraries(
                    ovspatialquery_installed_python_abi_witness PRIVATE
                    ovstage::ovstage)
            endif()
            add_test(NAME ovspatialquery_installed_python_abi_test
                COMMAND "${CMAKE_COMMAND}"
                    "-DOVSPATIALQUERY_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}"
                    "-DOVSPATIALQUERY_BUILD_CONFIG=$<CONFIG>"
                    "-DOVSPATIALQUERY_ABI_TARGET=ovspatialquery_installed_python_abi_witness"
                    "-DOVSPATIALQUERY_ABI_WITNESS=$<TARGET_FILE:ovspatialquery_installed_python_abi_witness>"
                    "-DOVSPATIALQUERY_PYTHON_EXECUTABLE=${_OVSPATIALQUERY_PYTHON_EXECUTABLE}"
                    "-DOVSPATIALQUERY_ABI_COMPARATOR=${CMAKE_CURRENT_SOURCE_DIR}/python/tests/verify_abi_witness.py"
                    "-DOVSPATIALQUERY_EXPECT_OVSTAGE=${OVSPATIALQUERY_BUILD_OVSTAGE_COMPANION}"
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/tooling/cmake/verify_installed_python_abi.cmake")
            set_tests_properties(ovspatialquery_installed_python_abi_test PROPERTIES
                FIXTURES_REQUIRED ovspatialquery_installed_artifacts
                RUN_SERIAL TRUE)

            add_executable(ovspatialquery_installed_c_consumer EXCLUDE_FROM_ALL
                tests/integration/installed/native_consumer_c.c)
            target_compile_features(ovspatialquery_installed_c_consumer PRIVATE c_std_11)
            target_include_directories(ovspatialquery_installed_c_consumer PRIVATE
                "${_OVSPATIALQUERY_TEST_INSTALL_ROOT}/include")
            target_link_libraries(ovspatialquery_installed_c_consumer PRIVATE
                "${_OVSPATIALQUERY_TEST_INSTALL_ROOT}/lib/$<TARGET_LINKER_FILE_NAME:ovspatialquery>")
            add_executable(ovspatialquery_installed_cpp_consumer EXCLUDE_FROM_ALL
                tests/integration/installed/native_consumer_cpp.cpp)
            target_compile_features(ovspatialquery_installed_cpp_consumer PRIVATE cxx_std_17)
            target_include_directories(ovspatialquery_installed_cpp_consumer PRIVATE
                "${_OVSPATIALQUERY_TEST_INSTALL_ROOT}/include")
            target_link_libraries(ovspatialquery_installed_cpp_consumer PRIVATE
                "${_OVSPATIALQUERY_TEST_INSTALL_ROOT}/lib/$<TARGET_LINKER_FILE_NAME:ovspatialquery>")
            set(_OVSPATIALQUERY_INSTALLED_CONSUMER_ARGS
                "-DOVSPATIALQUERY_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}"
                "-DOVSPATIALQUERY_BUILD_CONFIG=$<CONFIG>"
                "-DOVSPATIALQUERY_INSTALL_ROOT=${_OVSPATIALQUERY_TEST_INSTALL_ROOT}"
                "-DOVSPATIALQUERY_C_EXECUTABLE=$<TARGET_FILE:ovspatialquery_installed_c_consumer>"
                "-DOVSPATIALQUERY_CPP_EXECUTABLE=$<TARGET_FILE:ovspatialquery_installed_cpp_consumer>")
            add_test(NAME ovspatialquery_installed_native_consumers_test
                COMMAND "${CMAKE_COMMAND}"
                    ${_OVSPATIALQUERY_INSTALLED_CONSUMER_ARGS}
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/tooling/cmake/verify_installed_native_consumers.cmake")
            set_tests_properties(ovspatialquery_installed_native_consumers_test PROPERTIES
                FIXTURES_REQUIRED ovspatialquery_installed_artifacts
                RUN_SERIAL TRUE)
        endif()
    endif()

    if(OVSPATIALQUERY_BUILD_PRODUCT AND _OVSPATIALQUERY_PYTHON_EXECUTABLE)
        add_test(NAME ovspatialquery_installed_ctypes_test
            COMMAND "${_OVSPATIALQUERY_PYTHON_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/installed/ctypes_api_test.py"
                --dll "${_OVSPATIALQUERY_TEST_INSTALL_ROOT}/${_OVSPATIALQUERY_INSTALLED_LIBRARY_PATH}")
        set_tests_properties(ovspatialquery_installed_ctypes_test PROPERTIES
            FIXTURES_REQUIRED ovspatialquery_installed_artifacts)
        add_test(NAME ovspatialquery_installed_explicit_dtypes_test
            COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python/src"
                "${_OVSPATIALQUERY_PYTHON_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/installed/explicit_dtypes_test.py"
                --dll "${_OVSPATIALQUERY_TEST_INSTALL_ROOT}/${_OVSPATIALQUERY_INSTALLED_LIBRARY_PATH}")
        set_tests_properties(ovspatialquery_installed_explicit_dtypes_test PROPERTIES
            FIXTURES_REQUIRED ovspatialquery_installed_artifacts)
        add_test(NAME ovspatialquery_installed_two_copies_test
            COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python/src"
                "${_OVSPATIALQUERY_PYTHON_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/installed/two_copies_test.py"
                --dll "${_OVSPATIALQUERY_TEST_INSTALL_ROOT}/${_OVSPATIALQUERY_INSTALLED_LIBRARY_PATH}")
        set_tests_properties(ovspatialquery_installed_two_copies_test PROPERTIES
            FIXTURES_REQUIRED ovspatialquery_installed_artifacts)
        set(_OVSPATIALQUERY_RESULT_DLPACK_ARGS)
        if(OVSPATIALQUERY_HAS_CUDA_BACKEND)
            set(_OVSPATIALQUERY_RESULT_DLPACK_ARGS --cuda)
        endif()
        add_test(NAME ovspatialquery_installed_result_dlpack_test
            COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python/src"
                "${_OVSPATIALQUERY_PYTHON_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/installed/result_dlpack_test.py"
                --dll "${_OVSPATIALQUERY_TEST_INSTALL_ROOT}/${_OVSPATIALQUERY_INSTALLED_LIBRARY_PATH}"
                ${_OVSPATIALQUERY_RESULT_DLPACK_ARGS})
        set_tests_properties(ovspatialquery_installed_result_dlpack_test PROPERTIES
            FIXTURES_REQUIRED ovspatialquery_installed_artifacts
            TIMEOUT 120)
        if(OVSPATIALQUERY_BUILD_OVSTAGE_COMPANION)
            # The consumer skills' programs, and the oracles of the benchmark that measures
            # those skills, must keep working against the installed library.
            # The benchmark (its kit layout and container) is Linux-only.
            set(_OVSPATIALQUERY_SKILL_BENCH_ARGS)
            if(NOT WIN32)
                set(_OVSPATIALQUERY_SKILL_BENCH_ARGS
                    --bench-tasks "${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/skills/tasks")
            endif()
            # A CUDA build is tested on a GPU host, where the GPU programs must run, not skip.
            if(OVSPATIALQUERY_HAS_CUDA_BACKEND)
                list(APPEND _OVSPATIALQUERY_SKILL_BENCH_ARGS --cuda)
            endif()
            add_test(NAME ovspatialquery_skill_assets_test
                COMMAND "${_OVSPATIALQUERY_PYTHON_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/skills/check_skill_assets.py"
                    --skills "${CMAKE_CURRENT_SOURCE_DIR}/skills"
                    --sdk "${_OVSPATIALQUERY_TEST_INSTALL_ROOT}"
                    --library "${_OVSPATIALQUERY_TEST_INSTALL_ROOT}/${_OVSPATIALQUERY_INSTALLED_LIBRARY_PATH}"
                    --ovstage-dir "${OVSPATIALQUERY_OVSTAGE_SDK_DIR}"
                    --python "${_OVSPATIALQUERY_PYTHON_EXECUTABLE}"
                    --cmake "${CMAKE_COMMAND}"
                    --generator "${CMAKE_GENERATOR}"
                    --c-compiler "${CMAKE_C_COMPILER}"
                    --cxx-compiler "${CMAKE_CXX_COMPILER}"
                    --make-program "${CMAKE_MAKE_PROGRAM}"
                    --build-dir "${CMAKE_CURRENT_BINARY_DIR}/skill-assets"
                    ${_OVSPATIALQUERY_SKILL_BENCH_ARGS})
            set_tests_properties(ovspatialquery_skill_assets_test PROPERTIES
                FIXTURES_REQUIRED ovspatialquery_installed_artifacts
                RUN_SERIAL TRUE
                TIMEOUT 900)
        endif()
        if(OVSPATIALQUERY_HAS_CUDA_BACKEND)
            add_test(NAME ovspatialquery_installed_device_publication_test
                COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python/src"
                    "${_OVSPATIALQUERY_PYTHON_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/installed/device_publication_probe.py"
                    --library "${_OVSPATIALQUERY_TEST_INSTALL_ROOT}/${_OVSPATIALQUERY_INSTALLED_LIBRARY_PATH}")
            set_tests_properties(ovspatialquery_installed_device_publication_test PROPERTIES
                FIXTURES_REQUIRED ovspatialquery_installed_artifacts
                RUN_SERIAL TRUE
                TIMEOUT 180)
        endif()
    elseif(OVSPATIALQUERY_BUILD_PRODUCT)
        message(STATUS "ovspatialquery: Python 3.12 not found; ctypes tests not registered")
    endif()
endif()
foreach(_target
        ovspatialquery_compile_cuda_ptx
        ovspatialquery
        ovspatialquery_test_runtime_objects
        ovspatialquery_test_runtime
        ovspatialquery_physx_test_runtime)
    if(TARGET ${_target})
        if(MSVC)
            target_compile_options(${_target} PRIVATE /EHsc /W4 /WX /permissive- /utf-8)
        else()
            target_compile_options(${_target} PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wfloat-conversion -Werror)
        endif()
    endif()
endforeach()
