# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)

foreach(required_variable IN ITEMS
        OVSPATIALQUERY_BUILD_DIR
        OVSPATIALQUERY_BUILD_CONFIG
        OVSPATIALQUERY_ABI_TARGET
        OVSPATIALQUERY_ABI_WITNESS
        OVSPATIALQUERY_PYTHON_EXECUTABLE
        OVSPATIALQUERY_ABI_COMPARATOR
        OVSPATIALQUERY_EXPECT_OVSTAGE)
    if(NOT DEFINED ${required_variable})
        message(FATAL_ERROR "${required_variable} is required")
    endif()
endforeach()

execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${OVSPATIALQUERY_BUILD_DIR}"
        --config "${OVSPATIALQUERY_BUILD_CONFIG}"
        --target "${OVSPATIALQUERY_ABI_TARGET}"
    RESULT_VARIABLE build_result
    OUTPUT_VARIABLE build_output
    ERROR_VARIABLE build_error)
if(NOT build_result EQUAL 0)
    message(FATAL_ERROR
        "Installed compiler ABI witness build failed:\n${build_output}\n${build_error}")
endif()

set(comparator_arguments
    "${OVSPATIALQUERY_ABI_COMPARATOR}"
    --witness "${OVSPATIALQUERY_ABI_WITNESS}")
if(OVSPATIALQUERY_EXPECT_OVSTAGE)
    list(APPEND comparator_arguments --expect-ovstage)
endif()
execute_process(
    COMMAND "${OVSPATIALQUERY_PYTHON_EXECUTABLE}" ${comparator_arguments}
    RESULT_VARIABLE compare_result
    OUTPUT_VARIABLE compare_output
    ERROR_VARIABLE compare_error)
if(NOT compare_result EQUAL 0)
    message(FATAL_ERROR
        "Installed compiler/ctypes ABI comparison failed:\n${compare_output}\n${compare_error}")
endif()

message(STATUS "${compare_output}")
