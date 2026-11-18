# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)
project(ovspatialquery_examples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(OVSPATIALQUERY_INSTALL_DIR "" CACHE PATH
    "Exact successfully verified ovspatialquery native install")
set(OVSTAGE_DIR "" CACHE PATH "Exact locked Ovstage SDK/package root")
set(OVRTX_DIR "" CACHE PATH "Exact locked OvRTX SDK/package root")

if(UNIX AND NOT APPLE)
    find_package(X11 REQUIRED)
endif()

foreach(required_root IN ITEMS
    OVSPATIALQUERY_INSTALL_DIR
    OVSTAGE_DIR
    OVRTX_DIR)
    if(NOT IS_ABSOLUTE "${${required_root}}")
        message(FATAL_ERROR "${required_root} must be one explicit absolute SDK root")
    endif()
endforeach()

set(_core_header
    "${OVSPATIALQUERY_INSTALL_DIR}/include/ovspatialquery/ovspatialquery.h")
set(_companion_header
    "${OVSPATIALQUERY_INSTALL_DIR}/include/ovspatialquery/ovspatialquery_ovstage.h")
if(NOT EXISTS "${_core_header}" OR NOT EXISTS "${_companion_header}")
    message(FATAL_ERROR
        "OVSPATIALQUERY_INSTALL_DIR has no installed core/companion headers")
endif()
file(SHA256 "${_core_header}" _core_header_sha256)
file(SHA256 "${_companion_header}" _companion_header_sha256)

if(WIN32)
    set(_ovspatialquery_library
        "${OVSPATIALQUERY_INSTALL_DIR}/bin/ovspatialquery.dll")
    set(_ovspatialquery_implib
        "${OVSPATIALQUERY_INSTALL_DIR}/lib/ovspatialquery.lib")
else()
    set(_ovspatialquery_library
        "${OVSPATIALQUERY_INSTALL_DIR}/lib/libovspatialquery.so")
endif()
if(NOT EXISTS "${_ovspatialquery_library}")
    message(FATAL_ERROR
        "OVSPATIALQUERY_INSTALL_DIR has no installed platform library")
endif()
file(SHA256 "${_ovspatialquery_library}" _ovspatialquery_library_sha256)

add_library(ovspatialquery::installed SHARED IMPORTED)
set_target_properties(ovspatialquery::installed PROPERTIES
    IMPORTED_LOCATION "${_ovspatialquery_library}"
    INTERFACE_INCLUDE_DIRECTORIES "${OVSPATIALQUERY_INSTALL_DIR}/include")
if(WIN32)
    if(NOT EXISTS "${_ovspatialquery_implib}")
        message(FATAL_ERROR
            "OVSPATIALQUERY_INSTALL_DIR has no Windows import library")
    endif()
    set_target_properties(ovspatialquery::installed PROPERTIES
        IMPORTED_IMPLIB "${_ovspatialquery_implib}")
endif()

find_package(ovstage CONFIG REQUIRED
    PATHS "${OVSTAGE_DIR}/lib/cmake/ovstage"
    NO_DEFAULT_PATH)
find_package(ovrtx CONFIG REQUIRED
    PATHS "${OVRTX_DIR}/lib/cmake/ovrtx"
    NO_DEFAULT_PATH)

file(TO_CMAKE_PATH "${_ovspatialquery_library}" _library_identity_path)
file(TO_CMAKE_PATH "${_core_header}" _core_header_identity_path)
file(TO_CMAKE_PATH "${_companion_header}" _companion_header_identity_path)
file(TO_CMAKE_PATH "${OVSTAGE_DIR}" _ovstage_identity_path)
file(TO_CMAKE_PATH "${OVRTX_DIR}" _ovrtx_identity_path)
get_filename_component(_project_root "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
file(TO_CMAKE_PATH "${_project_root}" _project_root_path)

function(add_ovspatialquery_example name)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/native/${name}.cpp")
        message(FATAL_ERROR "Missing native lesson source: ${name}.cpp")
    endif()
    add_executable("${name}" "native/${name}.cpp")
    # Lessons 08 to 10 load the CUDA driver at run time through support/native/cuda_driver.h.
    target_include_directories("${name}" PRIVATE native "${_project_root}/support/native")
    target_link_libraries("${name}" PRIVATE
        ovspatialquery::installed
        ovstage::ovstage_static
        ovrtx::ovrtx_static
        ${CMAKE_DL_LIBS})
    if(UNIX AND NOT APPLE)
        target_link_libraries("${name}" PRIVATE X11::X11)
    endif()
    target_compile_definitions("${name}" PRIVATE
        OVSPATIALQUERY_LIBRARY_PATH="${_library_identity_path}"
        OVSPATIALQUERY_LIBRARY_SHA256="${_ovspatialquery_library_sha256}"
        OVSPATIALQUERY_CORE_HEADER_PATH="${_core_header_identity_path}"
        OVSPATIALQUERY_CORE_HEADER_SHA256="${_core_header_sha256}"
        OVSPATIALQUERY_COMPANION_HEADER_PATH="${_companion_header_identity_path}"
        OVSPATIALQUERY_COMPANION_HEADER_SHA256="${_companion_header_sha256}"
        OVSPATIALQUERY_PROJECT_ROOT="${_project_root_path}"
        OVSTAGE_PACKAGE_ROOT="${_ovstage_identity_path}"
        OVRTX_PACKAGE_ROOT="${_ovrtx_identity_path}")
    if(MSVC)
        target_compile_options("${name}" PRIVATE
            /EHsc /W4 /WX /permissive- /utf-8)
        target_link_libraries("${name}" PRIVATE delayimp)
        target_link_options("${name}" PRIVATE
            /DELAYLOAD:ovspatialquery.dll)
    else()
        target_compile_options("${name}" PRIVATE
            -Wall -Wextra -Wpedantic -Wshadow -Wfloat-conversion -Werror)
    endif()
endfunction()

foreach(example IN ITEMS
    01_static_candidate_raycast
    02_sphere_sweep
    03_shape_overlap
    04_filtered_view
    05_two_set_pair_snapshot
    06_precise_triangle
    07_time_sampled_update
    08_reusable_cpu_batch
    09_immutable_version_overlap
    10_pair_reporter_deltas)
    add_ovspatialquery_example("${example}")
endforeach()
