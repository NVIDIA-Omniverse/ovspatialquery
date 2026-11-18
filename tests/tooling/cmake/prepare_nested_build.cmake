# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Nested qualification builds survive the top-level --fresh configure. Remove
# their outputs and cache when the generator or Linux compiler changes, so they
# follow the parent build rather than retaining the previous toolchain.
function(ovspatialquery_prepare_nested_build build_dir generator)
    if(EXISTS "${build_dir}/CMakeCache.txt")
        file(STRINGS "${build_dir}/CMakeCache.txt" cached_generator
            REGEX "^CMAKE_GENERATOR:INTERNAL=")
        string(REPLACE "CMAKE_GENERATOR:INTERNAL=" "" cached_generator
            "${cached_generator}")
        set(reset FALSE)
        if(NOT cached_generator STREQUAL generator)
            set(reset TRUE)
        endif()
        if(NOT WIN32)
            foreach(language IN ITEMS C CXX)
                if(DEFINED OVSPATIALQUERY_${language}_COMPILER)
                    file(STRINGS "${build_dir}/CMakeCache.txt" cached_compiler
                        REGEX "^CMAKE_${language}_COMPILER:[^=]*=")
                    string(REGEX REPLACE "^[^=]*=" "" cached_compiler "${cached_compiler}")
                    if(NOT cached_compiler STREQUAL OVSPATIALQUERY_${language}_COMPILER)
                        set(reset TRUE)
                    endif()
                endif()
            endforeach()
        endif()
        if(reset)
            message(STATUS "Resetting nested build ${build_dir}: generator or compiler changed")
            file(REMOVE_RECURSE "${build_dir}")
        endif()
    endif()
endfunction()

# Generator arguments that make a nested configure follow the parent build.
# Forwarding preserves the C++ driver name: clang++ and clang invoke the same
# binary with different linking semantics. Windows gets its selected toolchain
# environment instead.
function(ovspatialquery_nested_generator_args out_var forward_compilers)
    set(arguments -G "${OVSPATIALQUERY_GENERATOR}")
    if(forward_compilers AND NOT WIN32)
        foreach(language IN ITEMS C CXX)
            if(DEFINED OVSPATIALQUERY_${language}_COMPILER)
                list(APPEND arguments
                    "-DCMAKE_${language}_COMPILER=${OVSPATIALQUERY_${language}_COMPILER}")
            endif()
        endforeach()
    endif()
    if(NOT "${OVSPATIALQUERY_GENERATOR_PLATFORM}" STREQUAL "")
        list(APPEND arguments -A "${OVSPATIALQUERY_GENERATOR_PLATFORM}")
    endif()
    if(NOT "${OVSPATIALQUERY_GENERATOR_TOOLSET}" STREQUAL "")
        list(APPEND arguments -T "${OVSPATIALQUERY_GENERATOR_TOOLSET}")
    endif()
    set(${out_var} "${arguments}" PARENT_SCOPE)
endfunction()
