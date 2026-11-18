# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)

# To regenerate a golden file after an intentional ABI change:
#   1. Build the ovspatialquery_python_abi_witness target for the desired
#      configuration (link ovstage::ovstage and define
#      OVSPATIALQUERY_ABI_WITH_OVSTAGE when regenerating the with-ovstage
#      golden).
#   2. Run the built witness binary and capture its stdout.
#   3. Normalize CRLF -> LF, strip leading/trailing whitespace, and append a
#      single trailing newline.
#   4. Save the result over the golden file named by this test's
#      OVSPATIALQUERY_ABI_WITNESS_GOLDEN argument (see CMakeLists.txt for the
#      current path) and review the diff in source control before committing.

if(NOT DEFINED OVSPATIALQUERY_ABI_WITNESS)
    message(FATAL_ERROR "OVSPATIALQUERY_ABI_WITNESS is required")
endif()
if(NOT DEFINED OVSPATIALQUERY_ABI_WITNESS_GOLDEN)
    message(FATAL_ERROR "OVSPATIALQUERY_ABI_WITNESS_GOLDEN is required")
endif()

execute_process(
    COMMAND "${OVSPATIALQUERY_ABI_WITNESS}"
    RESULT_VARIABLE _ovspatialquery_result
    OUTPUT_VARIABLE _ovspatialquery_output
    ERROR_VARIABLE _ovspatialquery_error)
if(NOT _ovspatialquery_result EQUAL 0)
    message(FATAL_ERROR
        "current API ABI witness failed: ${_ovspatialquery_error}")
endif()

string(REPLACE "\r\n" "\n" _ovspatialquery_output
    "${_ovspatialquery_output}")
string(REPLACE "\r" "\n" _ovspatialquery_output
    "${_ovspatialquery_output}")
string(STRIP "${_ovspatialquery_output}" _ovspatialquery_output)
string(APPEND _ovspatialquery_output "\n")

if(NOT EXISTS "${OVSPATIALQUERY_ABI_WITNESS_GOLDEN}")
    message(FATAL_ERROR
        "Missing ABI witness golden file: ${OVSPATIALQUERY_ABI_WITNESS_GOLDEN}\n"
        "Generate it by running the witness binary "
        "(${OVSPATIALQUERY_ABI_WITNESS}), normalizing CRLF to LF, stripping "
        "surrounding whitespace, appending a trailing newline, and saving the "
        "result at that path.")
endif()

set(_ovspatialquery_observed "${OVSPATIALQUERY_ABI_WITNESS}.observed.txt")
file(WRITE "${_ovspatialquery_observed}" "${_ovspatialquery_output}")

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E compare_files --ignore-eol
        "${_ovspatialquery_observed}" "${OVSPATIALQUERY_ABI_WITNESS_GOLDEN}"
    RESULT_VARIABLE _ovspatialquery_compare_result)

if(NOT _ovspatialquery_compare_result EQUAL 0)
    set(_ovspatialquery_diff_text "")
    find_program(_ovspatialquery_git NAMES git)
    if(_ovspatialquery_git)
        execute_process(
            COMMAND "${_ovspatialquery_git}" diff --no-index --
                "${OVSPATIALQUERY_ABI_WITNESS_GOLDEN}" "${_ovspatialquery_observed}"
            OUTPUT_VARIABLE _ovspatialquery_diff_text
            ERROR_VARIABLE _ovspatialquery_diff_error
            RESULT_VARIABLE _ovspatialquery_diff_result)
    endif()
    message(FATAL_ERROR
        "current API ABI witness changed from the golden file.\n"
        "Golden:   ${OVSPATIALQUERY_ABI_WITNESS_GOLDEN}\n"
        "Observed: ${_ovspatialquery_observed}\n"
        "Diff:\n${_ovspatialquery_diff_text}")
endif()

message(STATUS
    "Verified exact current API compiler ABI witness against "
    "${OVSPATIALQUERY_ABI_WITNESS_GOLDEN}")
