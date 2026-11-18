# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)
project(ovspatialquery_capability_showcase LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(OVSPATIALQUERY_INSTALL_DIR "" CACHE PATH "Exact verified ovspatialquery install")
set(OVSTAGE_DIR "" CACHE PATH "Exact locked Ovstage package root")
foreach(required_root IN ITEMS OVSPATIALQUERY_INSTALL_DIR OVSTAGE_DIR)
    if(NOT IS_ABSOLUTE "${${required_root}}")
        message(FATAL_ERROR "${required_root} must be one explicit absolute root")
    endif()
endforeach()

foreach(header IN ITEMS
    ovspatialquery.h
    ovspatialquery_explicit.h
    ovspatialquery_ovstage.h
    ovspatialquery_pairs.h)
    if(NOT EXISTS "${OVSPATIALQUERY_INSTALL_DIR}/include/ovspatialquery/${header}")
        message(FATAL_ERROR "verified install is missing ${header}")
    endif()
endforeach()

if(WIN32)
    set(_ovspatialquery_library "${OVSPATIALQUERY_INSTALL_DIR}/bin/ovspatialquery.dll")
    set(_ovspatialquery_implib "${OVSPATIALQUERY_INSTALL_DIR}/lib/ovspatialquery.lib")
else()
    set(_ovspatialquery_library "${OVSPATIALQUERY_INSTALL_DIR}/lib/libovspatialquery.so")
endif()
if(NOT EXISTS "${_ovspatialquery_library}")
    message(FATAL_ERROR "verified install has no platform library")
endif()

add_library(ovspatialquery::installed SHARED IMPORTED)
set_target_properties(ovspatialquery::installed PROPERTIES
    IMPORTED_LOCATION "${_ovspatialquery_library}"
    INTERFACE_INCLUDE_DIRECTORIES "${OVSPATIALQUERY_INSTALL_DIR}/include")
if(WIN32)
    if(NOT EXISTS "${_ovspatialquery_implib}")
        message(FATAL_ERROR "verified install has no Windows import library")
    endif()
    set_target_properties(ovspatialquery::installed PROPERTIES
        IMPORTED_IMPLIB "${_ovspatialquery_implib}")
endif()

find_package(ovstage CONFIG REQUIRED
    PATHS "${OVSTAGE_DIR}/lib/cmake/ovstage" NO_DEFAULT_PATH)
file(TO_CMAKE_PATH "${OVSTAGE_DIR}" _ovstage_package_root)
add_executable(11_capability_showcase "${CMAKE_CURRENT_SOURCE_DIR}/../native/11_capability_showcase.cpp")
target_include_directories(11_capability_showcase PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../support/native")
target_link_libraries(11_capability_showcase PRIVATE
    ovspatialquery::installed ovstage::ovstage_static ${CMAKE_DL_LIBS})
target_compile_definitions(11_capability_showcase PRIVATE
    OVSTAGE_PACKAGE_ROOT="${_ovstage_package_root}")
if(MSVC)
    target_compile_options(11_capability_showcase PRIVATE /EHsc /W4 /WX /permissive- /utf-8)
    target_link_libraries(11_capability_showcase PRIVATE delayimp)
    target_link_options(11_capability_showcase PRIVATE /DELAYLOAD:ovspatialquery.dll)
else()
    target_compile_options(11_capability_showcase PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wfloat-conversion -Werror)
endif()

include(CTest)
option(OVSPATIALQUERY_ENABLE_CUDA_CONTRACT
    "Register the strict real-CUDA lesson 11 contract" OFF)
if(BUILD_TESTING)
    add_test(NAME capability_showcase_cpu_headless_contract
        COMMAND 11_capability_showcase
            --json "${CMAKE_CURRENT_BINARY_DIR}/cpu-headless-contract.json")
    if(OVSPATIALQUERY_ENABLE_CUDA_CONTRACT)
        add_test(NAME capability_showcase_cuda_headless_contract
            COMMAND 11_capability_showcase --require-cuda
                --json "${CMAKE_CURRENT_BINARY_DIR}/cuda-headless-contract.json")
    endif()
    if(WIN32)
        set(_showcase_test_environment
            "PATH=path_list_prepend:${OVSPATIALQUERY_INSTALL_DIR}/bin;PATH=path_list_prepend:${OVSTAGE_DIR}/bin")
    else()
        set(_showcase_test_environment
            "LD_LIBRARY_PATH=path_list_prepend:${OVSPATIALQUERY_INSTALL_DIR}/lib;LD_LIBRARY_PATH=path_list_prepend:${OVSTAGE_DIR}/bin;LD_LIBRARY_PATH=path_list_prepend:${OVSTAGE_DIR}/bin/plugins")
    endif()
    set_tests_properties(capability_showcase_cpu_headless_contract PROPERTIES
        ENVIRONMENT_MODIFICATION "${_showcase_test_environment}")
    if(OVSPATIALQUERY_ENABLE_CUDA_CONTRACT)
        set_tests_properties(capability_showcase_cuda_headless_contract PROPERTIES
            ENVIRONMENT_MODIFICATION "${_showcase_test_environment}")
    endif()
endif()
