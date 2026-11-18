# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)
project(raycast_closest LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(OVSQ_SDK "$ENV{OVSQ_SDK}" CACHE PATH "Installed ovspatialquery SDK root")
set(OVSTAGE_DIR "$ENV{OVSTAGE_DIR}" CACHE PATH "Installed ovstage package root")

# Ovstage ships a CMake package; its shared loader lives in <root>/bin.
find_package(ovstage CONFIG REQUIRED PATHS "${OVSTAGE_DIR}/lib/cmake/ovstage" NO_DEFAULT_PATH)

# The SDK installs lib/cmake/ovspatialquery; its root is the package prefix.
find_package(ovspatialquery CONFIG REQUIRED PATHS "${OVSQ_SDK}" NO_DEFAULT_PATH)

add_executable(solution main.c)
target_link_libraries(solution PRIVATE ovspatialquery::ovspatialquery ovstage::ovstage
    $<$<NOT:$<PLATFORM_ID:Windows>>:m>)
# Imported shared libraries put their directories on the build-tree RPATH; state it
# explicitly too so the executable keeps finding both libraries.
set_target_properties(solution PROPERTIES
    BUILD_RPATH "${OVSQ_SDK}/lib;${OVSTAGE_DIR}/bin")
