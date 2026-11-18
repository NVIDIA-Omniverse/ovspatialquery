# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)
project(gpu_producer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OVSQ_SDK "$ENV{OVSQ_SDK}" CACHE PATH "Installed ovspatialquery SDK root")

# The SDK installs lib/cmake/ovspatialquery; its root is the package prefix.
find_package(ovspatialquery CONFIG REQUIRED PATHS "${OVSQ_SDK}" NO_DEFAULT_PATH)

# The CUDA driver is loaded at run time (dlopen), so no CUDA toolkit is needed to build.
add_executable(solution main.cpp)
target_link_libraries(solution PRIVATE ovspatialquery::ovspatialquery ${CMAKE_DL_LIBS})
set_target_properties(solution PROPERTIES BUILD_RPATH "${OVSQ_SDK}/lib")
