# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Packaged PhysX platform layout and static-library linking for ovspatialquery.
#
# The default is the pinned public package; an explicit custom SDK uses the same
# installed layout. Sibling modules are never discovered implicitly. Debug builds
# link the package's debug archives and every other configuration its release
# archives.

include_guard(GLOBAL)

if(WIN32)
    set(PHYSX_PLATFORM_BIN "win.x86_64.vc142.md")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(PHYSX_PLATFORM_BIN "linux.aarch64")
else()
    set(PHYSX_PLATFORM_BIN "linux.x86_64")
endif()

function(ovspatialquery_link_packaged_physx target_name)
    set(link_mode "FULL")
    if(ARGC GREATER 1)
        set(link_mode "${ARGV1}")
    endif()

    set(physx_library_dir "${PHYSX_SDK_DIR}/bin/${PHYSX_PLATFORM_BIN}")
    # PhysX 5.12 and later fold cooking into PhysXCommon; earlier SDKs have a separate archive.
    set(physx_cooking)
    if(EXISTS "${physx_library_dir}/release/libPhysXCooking_static_64.a" OR
       EXISTS "${physx_library_dir}/release/PhysXCooking_static_64.lib")
        set(physx_cooking PhysXCooking_static_64)
    endif()
    set(physx_debug_cooking)
    if(EXISTS "${physx_library_dir}/debug/libPhysXCooking_static_64.a" OR
       EXISTS "${physx_library_dir}/debug/PhysXCooking_static_64.lib")
        set(physx_debug_cooking PhysXCooking_static_64)
    endif()
    if(NOT "${physx_cooking}" STREQUAL "${physx_debug_cooking}")
        message(FATAL_ERROR "PhysX SDK must have matching release/debug cooking archive layouts")
    endif()
    get_target_property(target_type ${target_name} TYPE)
    set(directory_scope PRIVATE)
    if(target_type STREQUAL "STATIC_LIBRARY")
        set(directory_scope PUBLIC)
    endif()
    # The pinned package's debug archives reach Winsock through the PVD socket
    # transport; its release archives compile PVD out. A custom SDK's release
    # archives may keep PVD, so they keep Ws2_32.
    set(physx_windows_system_libraries $<$<OR:$<CONFIG:Debug>,$<BOOL:${OVSPATIALQUERY_CUSTOM_PHYSX}>>:Ws2_32>)
    target_link_directories(${target_name} ${directory_scope}
        "${physx_library_dir}/$<IF:$<CONFIG:Debug>,debug,release>")

    if(link_mode STREQUAL "MINIMAL")
        if(WIN32)
            target_link_libraries(${target_name} PRIVATE
                PhysX_static_64
                PVDRuntime_static_64
                PhysXCommon_static_64
                PhysXFoundation_static_64
                ${physx_windows_system_libraries})
        else()
            target_link_libraries(${target_name} PRIVATE
                -Wl,--start-group
                PhysX_static_64
                PVDRuntime_static_64
                PhysXCommon_static_64
                PhysXFoundation_static_64
                -Wl,--end-group
                dl pthread)
        endif()
    else()
        if(WIN32)
            target_link_libraries(${target_name} PRIVATE
                PhysXVehicle_static_64
                PhysXExtensions_static_64
                PhysXCharacterKinematic_static_64
                PhysX_static_64
                PVDRuntime_static_64
                PhysXPvdSDK_static_64
                ${physx_cooking}
                PhysXCommon_static_64
                PhysXFoundation_static_64
                ${physx_windows_system_libraries})
        else()
            target_link_libraries(${target_name} PRIVATE
                -Wl,--start-group
                PhysXExtensions_static_64
                PhysX_static_64
                PVDRuntime_static_64
                PhysXPvdSDK_static_64
                PhysXVehicle_static_64
                PhysXCharacterKinematic_static_64
                ${physx_cooking}
                PhysXCommon_static_64
                PhysXFoundation_static_64
                -Wl,--end-group
                dl pthread)
        endif()
    endif()
endfunction()
