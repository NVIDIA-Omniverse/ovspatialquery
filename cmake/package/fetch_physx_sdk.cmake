# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)

get_filename_component(SCRIPT_DIR "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
get_filename_component(PROJECT_ROOT "${SCRIPT_DIR}/../.." ABSOLUTE)

if(NOT WIN32)
    if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
        set(OVSPATIALQUERY_HOST_MANYLINUX_PLATFORM "manylinux_2_35_aarch64")
    else()
        set(OVSPATIALQUERY_HOST_MANYLINUX_PLATFORM "manylinux_2_35_x86_64")
    endif()
endif()

if(NOT DEFINED PLATFORM)
    if(DEFINED ENV{PLATFORM} AND NOT "$ENV{PLATFORM}" STREQUAL "")
        set(PLATFORM "$ENV{PLATFORM}")
    elseif(WIN32)
        set(PLATFORM "windows-x86_64")
    else()
        set(PLATFORM "${OVSPATIALQUERY_HOST_MANYLINUX_PLATFORM}")
    endif()
endif()

if(WIN32 AND PLATFORM STREQUAL "x64")
    set(PLATFORM "windows-x86_64")
elseif(NOT WIN32 AND (PLATFORM STREQUAL "Linux" OR PLATFORM STREQUAL "linux"))
    set(PLATFORM "${OVSPATIALQUERY_HOST_MANYLINUX_PLATFORM}")
elseif(NOT WIN32 AND PLATFORM STREQUAL "linux-x86_64")
    set(PLATFORM "manylinux_2_35_x86_64")
elseif(NOT WIN32 AND PLATFORM STREQUAL "linux-aarch64")
    set(PLATFORM "manylinux_2_35_aarch64")
endif()

if(WIN32)
    set(PACKMAN_CMD "${PROJECT_ROOT}/tools/packman/packman.cmd")
else()
    set(PACKMAN_CMD "${PROJECT_ROOT}/tools/packman/packman")
endif()

if(NOT EXISTS "${PACKMAN_CMD}")
    message(FATAL_ERROR "Packman bootstrap was not found at ${PACKMAN_CMD}")
endif()

if(NOT DEFINED ENV{PM_PACKAGES_ROOT} OR "$ENV{PM_PACKAGES_ROOT}" STREQUAL "")
    if(WIN32)
        cmake_path(GET PROJECT_ROOT ROOT_PATH OVSPATIALQUERY_PACKMAN_DRIVE)
        set(ENV{PM_PACKAGES_ROOT} "${OVSPATIALQUERY_PACKMAN_DRIVE}packman-repo")
    elseif(DEFINED ENV{XDG_CACHE_HOME} AND NOT "$ENV{XDG_CACHE_HOME}" STREQUAL "")
        set(ENV{PM_PACKAGES_ROOT} "$ENV{XDG_CACHE_HOME}/packman")
    else()
        set(ENV{PM_PACKAGES_ROOT} "$ENV{HOME}/.cache/packman")
    endif()
endif()
file(MAKE_DIRECTORY "$ENV{PM_PACKAGES_ROOT}")

message(STATUS "Fetching ovspatialquery packaged PhysX dependency")
message(STATUS "Packman platform: ${PLATFORM}")
message(STATUS "Packman cache: $ENV{PM_PACKAGES_ROOT}")
message(STATUS "Packman config: ${PROJECT_ROOT}/tools/packman/config.packman.xml")
execute_process(
    COMMAND "${PACKMAN_CMD}" pull -p "${PLATFORM}"
        "${PROJECT_ROOT}/deps/target-deps.packman.xml"
    WORKING_DIRECTORY "${PROJECT_ROOT}"
    RESULT_VARIABLE FETCH_RESULT
)
if(NOT FETCH_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to fetch ovspatialquery dependencies (exit code: ${FETCH_RESULT})")
endif()

set(PHYSX_SDK_DIR "${PROJECT_ROOT}/_build/target-deps/physx")
if(NOT EXISTS "${PHYSX_SDK_DIR}/include/PxPhysicsAPI.h")
    message(FATAL_ERROR
        "Packman reported success but the packaged PhysX SDK was not staged at ${PHYSX_SDK_DIR}")
endif()

message(STATUS "Packaged PhysX SDK: ${PHYSX_SDK_DIR}")
