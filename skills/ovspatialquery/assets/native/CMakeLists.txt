# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# cmake -S . -B build -G Ninja -DOVSQ_SDK=<sdk root> -DOVSTAGE_DIR=<ovstage package dir>
# cmake --build build && ctest --test-dir build --output-on-failure
# Without -D, the OVSQ_SDK and OVSTAGE_DIR environment variables are used.
# OVSTAGE_DIR is the ovstage package root, e.g. <venv>/lib/python3.12/site-packages/ovstage.
cmake_minimum_required(VERSION 3.22)
project(ovsq_native_assets LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OVSQ_SDK "$ENV{OVSQ_SDK}" CACHE PATH "ovspatialquery SDK root (include/, lib/, lib/cmake/ovspatialquery)")
set(OVSTAGE_DIR "$ENV{OVSTAGE_DIR}" CACHE PATH "ovstage package root (bin/, include/, lib/cmake/ovstage)")
# Both roots hold lib/cmake/<name>/<name>Config.cmake, so a prefix path finds both packages.
list(APPEND CMAKE_PREFIX_PATH "${OVSQ_SDK}" "${OVSTAGE_DIR}")

# ovspatialquery::ovspatialquery is the shared library with its include dir (which also
# holds dlpack/dlpack.h). COMPONENTS ovstage fails early on an SDK without
# ovspatialquery_ovstage.h; the package never finds Ovstage for you.
find_package(ovspatialquery CONFIG REQUIRED COMPONENTS ovstage)

# ovstage::ovstage is the shared loader (bin/libovstage-dynamic.so), which finds the
# runtime beside itself. Do not use ovstage::ovstage_static unless the program first calls
# ovstage_initialize with ovstage_config_entry_binary_package_root_path("<OVSTAGE_DIR>/bin");
# otherwise it looks for libovstage.so next to the executable and ovstage_create_instance fails.
find_package(ovstage CONFIG REQUIRED)

# ovspatialquery_ovstage.h includes Ovstage headers, so every Ovstage consumer also links
# ovstage::ovstage (it provides that include dir). libm: sqrt in C needs -lm on Unix;
# MSVC has no separate math library.
add_executable(raycast raycast.c)
target_link_libraries(raycast PRIVATE ovspatialquery::ovspatialquery ovstage::ovstage
    $<$<NOT:$<PLATFORM_ID:Windows>>:m>)

add_executable(batch_raycast batch_raycast.cpp)
target_link_libraries(batch_raycast PRIVATE ovspatialquery::ovspatialquery ovstage::ovstage
    $<$<NOT:$<PLATFORM_ID:Windows>>:m>)

# A GPU producer loop. It needs only the explicit adapter and the CUDA driver, which it loads
# at run time (dlopen / LoadLibrary), so no CUDA toolkit is needed to build it.
add_executable(gpu_producer gpu_producer.cpp)
target_link_libraries(gpu_producer PRIVATE ovspatialquery::ovspatialquery ${CMAKE_DL_LIBS})

# Both imported targets put their directories (SDK lib/, Ovstage bin/) in the build
# RPATH, so the tests run from the build tree on Linux. Installed or copied binaries need
# INSTALL_RPATH or LD_LIBRARY_PATH. Windows has no RPATH: the tests below put the SDK's
# DLL directory and Ovstage's bin/ and bin/plugins/ on PATH instead.

enable_testing()
# Absolute scene path: a relative one resolves against the test's working directory.
add_test(NAME raycast COMMAND raycast "${CMAKE_CURRENT_SOURCE_DIR}/scene.usda")
add_test(NAME batch_raycast COMMAND batch_raycast)
add_test(NAME gpu_producer COMMAND gpu_producer)
# Exit code 77: no CUDA device, or a library built without CUDA.
set_tests_properties(gpu_producer PROPERTIES SKIP_RETURN_CODE 77)
if(WIN32)
    set_tests_properties(raycast batch_raycast gpu_producer PROPERTIES ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:ovspatialquery::ovspatialquery>;PATH=path_list_prepend:${OVSTAGE_DIR}/bin;PATH=path_list_prepend:${OVSTAGE_DIR}/bin/plugins")
endif()
