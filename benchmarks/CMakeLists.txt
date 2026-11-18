# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)
project(ovspatialquery_scene_benchmarks LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(OVSPATIALQUERY_INSTALL_DIR "" CACHE PATH "Verified production ovspatialquery install")
set(PHYSX_SDK_DIR "" CACHE PATH "Published pinned PhysX SDK used by that install")
set(OVSPATIALQUERY_OVSTAGE_SDK_DIR "" CACHE PATH "Pinned ovstage package root the install's companion was built against")
foreach(root IN ITEMS OVSPATIALQUERY_INSTALL_DIR PHYSX_SDK_DIR OVSPATIALQUERY_OVSTAGE_SDK_DIR)
    if(NOT IS_ABSOLUTE "${${root}}")
        message(FATAL_ERROR "${root} must be an explicit absolute package/install root")
    endif()
endforeach()
if(NOT EXISTS "${PHYSX_SDK_DIR}/include/PxPhysicsAPI.h")
    message(FATAL_ERROR "PHYSX_SDK_DIR does not contain the packaged headers")
endif()
if(NOT EXISTS "${OVSPATIALQUERY_OVSTAGE_SDK_DIR}/lib/cmake/ovstage/ovstageConfig.cmake")
    message(FATAL_ERROR "OVSPATIALQUERY_OVSTAGE_SDK_DIR does not contain the pinned ovstage package")
endif()
# A cached ovstage_DIR outranks PATHS: after a pin change it would keep the previous SDK.
unset(ovstage_DIR CACHE)
find_package(ovstage CONFIG REQUIRED
    PATHS "${OVSPATIALQUERY_OVSTAGE_SDK_DIR}/lib/cmake/ovstage"
    NO_DEFAULT_PATH)
add_library(ovspatialquery_installed SHARED IMPORTED)
set_target_properties(ovspatialquery_installed PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${OVSPATIALQUERY_INSTALL_DIR}/include")
if(WIN32)
    set(_library "${OVSPATIALQUERY_INSTALL_DIR}/bin/ovspatialquery.dll")
    set(_implib "${OVSPATIALQUERY_INSTALL_DIR}/lib/ovspatialquery.lib")
    if(NOT EXISTS "${_implib}")
        message(FATAL_ERROR "Verified install is missing the Windows import library")
    endif()
    set_target_properties(ovspatialquery_installed PROPERTIES IMPORTED_IMPLIB "${_implib}")
else()
    set(_library "${OVSPATIALQUERY_INSTALL_DIR}/lib/libovspatialquery.so")
endif()
if(NOT EXISTS "${_library}")
    message(FATAL_ERROR "Verified install is missing the production library")
endif()
set_target_properties(ovspatialquery_installed PROPERTIES IMPORTED_LOCATION "${_library}")
include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/package/OvSpatialQueryPhysX.cmake")
add_executable(scene_query_benchmark
    native/main.cpp
    native/common.cpp
    native/measurement.cpp
    native/fixture.cpp
    native/installed_adapter.cpp
    native/physx_adapter.cpp
    native/workloads/actor.cpp
    native/workloads/camera.cpp
    native/workloads/convex.cpp
    native/workloads/deform.cpp
    native/workloads/isaac.cpp
    native/workloads/mesh.cpp
    native/workloads/ovstage.cpp
    native/workloads/pairs.cpp
    native/workloads/sensor.cpp
    native/workloads/statics.cpp
    native/workloads/warehouse.cpp)
target_include_directories(scene_query_benchmark SYSTEM PRIVATE "${PHYSX_SDK_DIR}/include")
target_include_directories(scene_query_benchmark PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../support/native")
target_compile_definitions(scene_query_benchmark PRIVATE PX_PHYSX_STATIC_LIB
    BENCH_PHYSX_CONFIG="$<IF:$<CONFIG:Debug>,debug,release>")
target_link_libraries(scene_query_benchmark PRIVATE ovspatialquery_installed ovstage::ovstage)
ovspatialquery_link_packaged_physx(scene_query_benchmark)
# The ovstage loader and its runtime closure live in the package bin/; the executable
# links the loader, so point the ELF search path there (Windows relies on PATH).
if(NOT WIN32)
    set_target_properties(scene_query_benchmark PROPERTIES
        BUILD_RPATH "${OVSTAGE_BINARY_DIR};${OVSTAGE_BINARY_DIR}/plugins")
endif()
if(WIN32)
    target_compile_definitions(scene_query_benchmark PRIVATE NOMINMAX)
    set(_ovstage_runtime_variable PATH)
else()
    set(_ovstage_runtime_variable LD_LIBRARY_PATH)
endif()
if(MSVC)
    target_compile_options(scene_query_benchmark PRIVATE /W4 /WX /permissive- /EHsc)
    add_custom_command(TARGET scene_query_benchmark POST_BUILD COMMAND ${CMAKE_COMMAND} -E
        copy_if_different "${_library}" "$<TARGET_FILE_DIR:scene_query_benchmark>")
    if(EXISTS "${OVSPATIALQUERY_INSTALL_DIR}/bin/ovspatialquery_cuda.ptx")
        add_custom_command(TARGET scene_query_benchmark POST_BUILD COMMAND ${CMAKE_COMMAND} -E
            copy_if_different "${OVSPATIALQUERY_INSTALL_DIR}/bin/ovspatialquery_cuda.ptx"
            "$<TARGET_FILE_DIR:scene_query_benchmark>")
    endif()
else()
    target_compile_options(scene_query_benchmark PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wfloat-conversion -Werror)
endif()
enable_testing()
add_test(NAME isaac_query_parity COMMAND scene_query_benchmark --backend parity
    --workload rays --objects 100 --queries 64 --frames 2)
add_test(NAME actor_grounding_parity COMMAND scene_query_benchmark --backend parity
    --workload grounding --objects 100 --queries 64 --update-percent 10)
add_test(NAME actor_support_parity COMMAND scene_query_benchmark --backend parity
    --workload support --objects 100 --queries 1)
add_test(NAME actor_avoidance_parity COMMAND scene_query_benchmark --backend parity
    --workload avoidance-mixed --objects 100 --queries 16 --update-percent 10
    --actor-hit-capacity 1)
# One actor step as it runs today: avoidance sweeps start in the ground, so none confirms a dodge.
add_test(NAME actor_frame_parity COMMAND scene_query_benchmark --backend parity
    --workload actor-frame --objects 13 --queries 16 --update-percent 100 --actor-hit-capacity 1)
# The same step with avoidance rows restricted to dynamic bodies by participation bits.
add_test(NAME actor_frame_bits_parity COMMAND scene_query_benchmark --backend parity
    --workload actor-frame-bits --objects 13 --queries 16 --update-percent 100 --actor-hit-capacity 1)
add_test(NAME mesh_query_semantics COMMAND scene_query_benchmark --backend parity --workload mesh-parity)
add_test(NAME mesh_grounding_parity COMMAND scene_query_benchmark --backend parity
    --workload grounding-mesh --objects 16 --queries 1024 --update-percent 10)
add_test(NAME mesh_sweep_parity COMMAND scene_query_benchmark --backend parity
    --workload mesh-sweeps --objects 16 --queries 64 --unique-meshes 1 --update-percent 10)
# Box faces land on the terrain across several triangles at once; CPU and CUDA may report any of them.
add_test(NAME mesh_box_sweep_parity COMMAND scene_query_benchmark --backend parity
    --workload mesh-box-sweeps --objects 16 --queries 64 --update-percent 10)
# One 256 x 256 camera image of the 69,136-triangle blob; rays near its silhouette graze faces,
# where the CPU path's float contacts and CUDA's double ones are farthest apart.
add_test(NAME camera_ray_parity COMMAND scene_query_benchmark --backend parity
    --workload camera-mesh --objects 1 --queries 65536)
add_test(NAME convex_ray_parity COMMAND scene_query_benchmark --backend parity
    --workload convex-rays --objects 100 --queries 256 --update-percent 10 --subset even)
add_test(NAME convex_sweep_parity COMMAND scene_query_benchmark --backend parity
    --workload convex-sweeps --objects 100 --queries 64 --update-percent 10)
add_test(NAME convex_overlap_parity COMMAND scene_query_benchmark --backend parity
    --workload convex-overlaps --objects 100 --queries 64 --update-percent 10 --subset even)

add_test(NAME sensor_point_cloud_parity COMMAND scene_query_benchmark --backend parity
    --workload sensor-point-cloud --objects 5 --queries 16384)
add_test(NAME pair_reporter_parity COMMAND scene_query_benchmark --backend parity
    --workload pairs --objects 100 --queries 1 --update-percent 10)
add_test(NAME pair_tolerance_parity COMMAND scene_query_benchmark --backend parity
    --workload pairs --objects 100 --queries 1 --update-percent 10 --pair-tolerance-milli 500)
# Engine-style cells: one population against itself at the engine's 0.1 m margin, and two
# partially overlapping views of it; both lanes also check ABP against the oracle.
add_test(NAME pair_self_parity COMMAND scene_query_benchmark --backend parity
    --workload pairs-self --objects 256 --queries 64 --update-percent 10 --pair-tolerance-milli 100)
add_test(NAME pair_views_parity COMMAND scene_query_benchmark --backend parity
    --workload pairs-views --objects 256 --queries 16 --update-percent 10)
add_test(NAME ovstage_publication_parity COMMAND scene_query_benchmark
    --backend parity --workload ovstage --objects 100 --queries 64
    --update-percent 10)
# Deforming sheets: four cells cover every sheet size; mixed covers waves and an index rewrite.
add_test(NAME deform_mixed_parity COMMAND scene_query_benchmark --backend parity
    --workload deform-mixed --objects 4 --queries 128 --update-percent 50)
add_test(NAME deform_crumple_parity COMMAND scene_query_benchmark --backend parity
    --workload deform-crumple --objects 4 --queries 128 --update-percent 100)
add_test(NAME deform_rigid_parity COMMAND scene_query_benchmark --backend parity
    --workload deform-rigid --objects 4 --queries 64 --update-percent 50)

set(_warehouse_meshes "${CMAKE_CURRENT_BINARY_DIR}/../warehouse-racks.bin")
if(EXISTS "${_warehouse_meshes}")
    add_test(NAME warehouse_ray_parity COMMAND scene_query_benchmark --backend parity
        --workload warehouse-rays --objects 52 --queries 64 --warehouse-meshes "${_warehouse_meshes}")
    # The 1,024-row sphere lattice holds query 840, whose closer vertex contact PhysX's sphere
    # sweep misses; an installed path must find it.
    add_test(NAME warehouse_sweep_parity COMMAND scene_query_benchmark --backend parity
        --workload warehouse-sweeps --objects 52 --queries 1024 --warehouse-meshes "${_warehouse_meshes}")
    # The 4,096-row box lattice holds PhysX box points off the box and face, back faces and early
    # distances, which the 64-row suite case does not.
    add_test(NAME warehouse_box_sweep_parity COMMAND scene_query_benchmark --backend parity
        --workload warehouse-box-sweeps --objects 52 --queries 4096 --warehouse-meshes "${_warehouse_meshes}")
    add_test(NAME warehouse_overlap_parity COMMAND scene_query_benchmark --backend parity
        --workload warehouse-overlaps --objects 52 --queries 12 --warehouse-meshes "${_warehouse_meshes}")
else()
    message(STATUS "warehouse parity tests skipped: ${_warehouse_meshes} missing; run benchmarks/warehouse_meshes.py")
endif()
# Every workload links the same Ovstage loader, including standalone CTest runs
# outside run.py. Preserve the caller's search path and prepend the pinned runtime.
get_property(_benchmark_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${_benchmark_tests} PROPERTIES ENVIRONMENT_MODIFICATION
    "${_ovstage_runtime_variable}=path_list_prepend:${OVSTAGE_BINARY_DIR};${_ovstage_runtime_variable}=path_list_prepend:${OVSTAGE_BINARY_DIR}/plugins")
