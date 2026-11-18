# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Ubuntu's hash-locked GCC 11 cross packages provide this target layout.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
find_program(CMAKE_C_COMPILER aarch64-linux-gnu-gcc-11 REQUIRED)
find_program(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++-11 REQUIRED)
find_program(CMAKE_AR aarch64-linux-gnu-ar REQUIRED)
find_program(CMAKE_RANLIB aarch64-linux-gnu-ranlib REQUIRED)
find_program(CMAKE_NM aarch64-linux-gnu-nm REQUIRED)
find_program(CMAKE_READELF aarch64-linux-gnu-readelf REQUIRED)
find_program(CMAKE_OBJDUMP aarch64-linux-gnu-objdump REQUIRED)
set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
