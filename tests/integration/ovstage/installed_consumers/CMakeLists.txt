# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)
project(ovspatialquery_ovstage_installed_consumers LANGUAGES C CXX)

foreach(OVSPATIALQUERY_REQUIRED_VARIABLE
        OVSPATIALQUERY_INSTALL_ROOT
        OVSPATIALQUERY_OVSTAGE_SDK_DIR)
    if(NOT DEFINED ${OVSPATIALQUERY_REQUIRED_VARIABLE})
        message(FATAL_ERROR "${OVSPATIALQUERY_REQUIRED_VARIABLE} is required")
    endif()
endforeach()

find_package(ovspatialquery CONFIG REQUIRED COMPONENTS ovstage
    PATHS "${OVSPATIALQUERY_INSTALL_ROOT}"
    NO_DEFAULT_PATH
    NO_CMAKE_FIND_ROOT_PATH)
# A cached ovstage_DIR outranks PATHS: after a pin change it would keep the previous SDK.
unset(ovstage_DIR CACHE)
find_package(ovstage CONFIG REQUIRED
    PATHS "${OVSPATIALQUERY_OVSTAGE_SDK_DIR}/lib/cmake/ovstage"
        NO_CMAKE_FIND_ROOT_PATH
    NO_DEFAULT_PATH)

add_executable(ovspatialquery_ovstage_installed_c
    ../installed_consumer_c.c)
target_compile_features(ovspatialquery_ovstage_installed_c PRIVATE c_std_11)
target_link_libraries(ovspatialquery_ovstage_installed_c PRIVATE
    ovspatialquery::ovspatialquery ovstage::ovstage)
