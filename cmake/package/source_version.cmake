# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The SDK version a build reports, in the order tools/bootstrap/contract.py
# version_selection() uses:
#   1. an explicit OVSPATIALQUERY_VERSION (-D or cache), then OVSPATIALQUERY_BUILD_VERSION;
#   2. VERSION.txt, the stamp an exported candidate or final release carries;
#   3. a vX.Y.Z or vX.Y.ZrcN tag at the HEAD of a clean checkout (the final when both);
#   4. the development version in python/pyproject.toml.
# Sets <version_var> to the version and <origin_var> to "override", "stamp", "tag" or
# "source". GIT_EXECUTABLE selects Git; without it, or without <source_dir>/.git, no tag
# applies. The caller validates the version.
function(ovspatialquery_select_version source_dir version_var origin_var)
    if(NOT "${OVSPATIALQUERY_VERSION}" STREQUAL "")
        set(${version_var} "${OVSPATIALQUERY_VERSION}" PARENT_SCOPE)
        set(${origin_var} override PARENT_SCOPE)
        return()
    endif()
    if(NOT "$ENV{OVSPATIALQUERY_BUILD_VERSION}" STREQUAL "")
        set(${version_var} "$ENV{OVSPATIALQUERY_BUILD_VERSION}" PARENT_SCOPE)
        set(${origin_var} override PARENT_SCOPE)
        return()
    endif()
    ovspatialquery_source_version("${source_dir}" _version _origin)
    set(${version_var} "${_version}" PARENT_SCOPE)
    set(${origin_var} "${_origin}" PARENT_SCOPE)
endfunction()

# Steps 2 to 4 of ovspatialquery_select_version().
function(ovspatialquery_source_version source_dir version_var origin_var)
    if(EXISTS "${source_dir}/VERSION.txt")
        file(READ "${source_dir}/VERSION.txt" _version)
        string(STRIP "${_version}" _version)
        set(${version_var} "${_version}" PARENT_SCOPE)
        set(${origin_var} stamp PARENT_SCOPE)
        return()
    endif()
    # An extracted archive has no tags of its own; never adopt a containing checkout's.
    if(GIT_EXECUTABLE AND EXISTS "${source_dir}/.git")
        execute_process(COMMAND "${GIT_EXECUTABLE}" status --porcelain --untracked-files=normal
            WORKING_DIRECTORY "${source_dir}"
            RESULT_VARIABLE _status_result OUTPUT_VARIABLE _status ERROR_QUIET)
        execute_process(COMMAND "${GIT_EXECUTABLE}" tag --points-at HEAD
            WORKING_DIRECTORY "${source_dir}"
            RESULT_VARIABLE _tags_result OUTPUT_VARIABLE _tags ERROR_QUIET)
        if(_status_result EQUAL 0 AND _status STREQUAL "" AND _tags_result EQUAL 0)
            string(REPLACE "\n" ";" _tags "${_tags}")
            set(_selected "")
            set(_selected_order "")
            foreach(_tag IN LISTS _tags)
                if(_tag MATCHES "^v((0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)(rc(0|[1-9][0-9]*))?)$")
                    # A final release may be tagged at its last candidate's commit; it sorts after rcN.
                    if(CMAKE_MATCH_5)
                        set(_order "${CMAKE_MATCH_2}.${CMAKE_MATCH_3}.${CMAKE_MATCH_4}.0.${CMAKE_MATCH_6}")
                    else()
                        set(_order "${CMAKE_MATCH_2}.${CMAKE_MATCH_3}.${CMAKE_MATCH_4}.1.0")
                    endif()
                    if(NOT _selected_order OR _order VERSION_GREATER _selected_order)
                        set(_selected "${CMAKE_MATCH_1}")
                        set(_selected_order "${_order}")
                    endif()
                endif()
            endforeach()
            if(_selected)
                set(${version_var} "${_selected}" PARENT_SCOPE)
                set(${origin_var} tag PARENT_SCOPE)
                return()
            endif()
        endif()
    endif()
    file(STRINGS "${source_dir}/python/pyproject.toml" _line REGEX "^version = \"[^\"]+\"$")
    string(REGEX REPLACE "^version = \"([^\"]+)\"$" "\\1" _version "${_line}")
    set(${version_var} "${_version}" PARENT_SCOPE)
    set(${origin_var} source PARENT_SCOPE)
endfunction()
