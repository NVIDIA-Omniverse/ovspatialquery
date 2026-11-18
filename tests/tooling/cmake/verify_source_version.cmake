# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# A plain CMake build reports an explicit OVSPATIALQUERY_VERSION, else OVSPATIALQUERY_BUILD_VERSION,
# else the VERSION.txt stamp, else a release tag at a clean HEAD, else the pyproject development
# version (cmake/package/source_version.cmake). Git is optional: without it, the tag steps are skipped.
cmake_minimum_required(VERSION 3.24)
foreach(_required OVSPATIALQUERY_SOURCE_DIR OVSPATIALQUERY_WORK_DIR)
    if(NOT ${_required})
        message(FATAL_ERROR "${_required} is required")
    endif()
endforeach()
include("${OVSPATIALQUERY_SOURCE_DIR}/cmake/package/source_version.cmake")
find_program(GIT_EXECUTABLE git)
# The build's own environment may inject a version; each case below sets its own.
unset(ENV{OVSPATIALQUERY_BUILD_VERSION})

set(_repo "${OVSPATIALQUERY_WORK_DIR}/source-version")
file(REMOVE_RECURSE "${_repo}")
file(MAKE_DIRECTORY "${_repo}/python")
file(WRITE "${_repo}/python/pyproject.toml" "[project]\nversion = \"0.0.3.dev0\"\n")
file(WRITE "${_repo}/.gitignore" "_*/\n")

function(git)
    execute_process(COMMAND "${GIT_EXECUTABLE}" -c user.name=test -c user.email=test@example.invalid
        -c commit.gpgsign=false -c tag.gpgsign=false ${ARGN}
        WORKING_DIRECTORY "${_repo}" COMMAND_ERROR_IS_FATAL ANY OUTPUT_QUIET)
endfunction()

function(expect directory version origin)
    ovspatialquery_select_version("${directory}" _version _origin)
    if(NOT _version STREQUAL version OR NOT _origin STREQUAL origin)
        message(FATAL_ERROR "${directory}: expected ${version} (${origin}), got ${_version} (${_origin})")
    endif()
endfunction()

if(GIT_EXECUTABLE)
    git(init -q)
    git(add -A)
    git(commit -q -m source)
    expect("${_repo}" 0.0.3.dev0 source)

    # Candidates order numerically, and the final release sorts after them at the same commit.
    git(tag v0.0.3rc2)
    git(tag v0.0.3rc10)
    git(tag v0.0.3-unrelated)
    expect("${_repo}" 0.0.3rc10 tag)
    git(tag v0.0.3)
    expect("${_repo}" 0.0.3 tag)

    # Uncommitted or untracked changes keep the development version.
    file(WRITE "${_repo}/untracked.txt" "new\n")
    expect("${_repo}" 0.0.3.dev0 source)
    file(REMOVE "${_repo}/untracked.txt")
    expect("${_repo}" 0.0.3 tag)
else()
    message(STATUS "Git is unavailable; the release tag cases are skipped")
endif()

# A source extracted (ignored) inside the clean tagged checkout never adopts its tag.
file(MAKE_DIRECTORY "${_repo}/_extracted/python")
file(COPY "${_repo}/python/pyproject.toml" DESTINATION "${_repo}/_extracted/python")
expect("${_repo}/_extracted" 0.0.3.dev0 source)

# The exported stamp wins over a tag.
file(WRITE "${_repo}/_extracted/VERSION.txt" "0.0.3rc4\n")
expect("${_repo}/_extracted" 0.0.3rc4 stamp)
if(GIT_EXECUTABLE)
    file(COPY "${_repo}/_extracted/VERSION.txt" DESTINATION "${_repo}")
    git(add VERSION.txt)
    git(commit -q -m stamp)
    git(tag v0.0.4)
    expect("${_repo}" 0.0.3rc4 stamp)
endif()

# The environment wins over the stamp, and an explicit cache value over the environment, also
# when it equals the version a previous configure selected.
set(ENV{OVSPATIALQUERY_BUILD_VERSION} 0.0.5rc1)
expect("${_repo}/_extracted" 0.0.5rc1 override)
set(OVSPATIALQUERY_VERSION 0.0.3rc4)
expect("${_repo}/_extracted" 0.0.3rc4 override)
unset(OVSPATIALQUERY_VERSION)
unset(ENV{OVSPATIALQUERY_BUILD_VERSION})

file(REMOVE_RECURSE "${_repo}")
