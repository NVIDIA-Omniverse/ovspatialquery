# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)
project(ovspatialquery_core_independence LANGUAGES C CXX)

if(NOT DEFINED OVSPATIALQUERY_INSTALL_ROOT)
    message(FATAL_ERROR "OVSPATIALQUERY_INSTALL_ROOT is required")
endif()
# Consume the SDK the documented way: its installed package config, found only
# under the root being qualified (also when cross-compiling). No version is
# requested: a release may inject one that differs from the source tree's.
find_package(ovspatialquery CONFIG REQUIRED
    PATHS "${OVSPATIALQUERY_INSTALL_ROOT}"
    NO_DEFAULT_PATH
    NO_CMAKE_FIND_ROOT_PATH)
set(OVSPATIALQUERY_CORE_RUNTIME "$<TARGET_FILE:ovspatialquery::ovspatialquery>")

add_executable(ovspatialquery_core_only_c ../native_consumer_c.c)
target_compile_features(ovspatialquery_core_only_c PRIVATE c_std_11)
target_link_libraries(ovspatialquery_core_only_c PRIVATE
    ovspatialquery::ovspatialquery)

add_executable(ovspatialquery_core_only_cpp ../native_consumer_cpp.cpp)
target_compile_features(ovspatialquery_core_only_cpp PRIVATE cxx_std_17)
target_link_libraries(ovspatialquery_core_only_cpp PRIVATE
    ovspatialquery::ovspatialquery)

add_executable(ovspatialquery_core_only_compiler_ctypes_witness
    ../python_abi_witness.c)
target_compile_features(ovspatialquery_core_only_compiler_ctypes_witness
    PRIVATE c_std_11)
target_compile_definitions(ovspatialquery_core_only_compiler_ctypes_witness
    PRIVATE OVSPATIALQUERY_ABI_BATCH_COLUMNS_ONLY)
target_include_directories(ovspatialquery_core_only_compiler_ctypes_witness
    PRIVATE
    $<TARGET_PROPERTY:ovspatialquery::ovspatialquery,INTERFACE_INCLUDE_DIRECTORIES>)

find_package(Python3 3.12 EXACT REQUIRED COMPONENTS Interpreter)
add_custom_target(ovspatialquery_core_only_compiler_ctypes_check ALL
    COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_LIST_DIR}/../ctypes_api_test.py"
        --dll "${OVSPATIALQUERY_CORE_RUNTIME}"
        --abi-witness
        "$<TARGET_FILE:ovspatialquery_core_only_compiler_ctypes_witness>"
    DEPENDS ovspatialquery_core_only_compiler_ctypes_witness
    VERBATIM)

foreach(OVSPATIALQUERY_CONSUMER_TARGET IN ITEMS
        ovspatialquery_core_only_c
        ovspatialquery_core_only_cpp
        ovspatialquery_core_only_compiler_ctypes_witness)
    if(MSVC)
        target_compile_options(${OVSPATIALQUERY_CONSUMER_TARGET} PRIVATE /W4 /WX)
    else()
        target_compile_options(${OVSPATIALQUERY_CONSUMER_TARGET} PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
    endif()
endforeach()

# Transferred-artifact qualification runs through the pinned CTest on both platforms.
enable_testing()
add_test(NAME installed_c COMMAND ovspatialquery_core_only_c)
add_test(NAME installed_cpp COMMAND ovspatialquery_core_only_cpp)
add_test(NAME installed_ctypes
    COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_LIST_DIR}/../ctypes_api_test.py"
        --dll "${OVSPATIALQUERY_CORE_RUNTIME}"
        --abi-witness
        "$<TARGET_FILE:ovspatialquery_core_only_compiler_ctypes_witness>")
if(WIN32)
    set_tests_properties(installed_c installed_cpp PROPERTIES
        ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:ovspatialquery::ovspatialquery>")
else()
    set_tests_properties(installed_c installed_cpp PROPERTIES
        ENVIRONMENT_MODIFICATION
        "LD_LIBRARY_PATH=path_list_prepend:$<TARGET_FILE_DIR:ovspatialquery::ovspatialquery>")
endif()
