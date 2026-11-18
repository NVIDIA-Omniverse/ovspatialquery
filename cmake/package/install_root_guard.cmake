# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared by install_artifacts.cmake and install_verified_artifacts.cmake.
# Refuses to replace anything but the deterministic install leaf directly
# under the allowed parent, then publishes the normalized paths the callers
# pass on.

foreach(OVSPATIALQUERY_REQUIRED_VARIABLE
        OVSPATIALQUERY_INSTALL_ROOT
        OVSPATIALQUERY_ALLOWED_PARENT
        OVSPATIALQUERY_ALLOWED_LEAF)
    if(NOT DEFINED ${OVSPATIALQUERY_REQUIRED_VARIABLE})
        message(FATAL_ERROR "${OVSPATIALQUERY_REQUIRED_VARIABLE} is required")
    endif()
endforeach()

get_filename_component(OVSPATIALQUERY_NORMALIZED_INSTALL_ROOT
    "${OVSPATIALQUERY_INSTALL_ROOT}" ABSOLUTE)
get_filename_component(OVSPATIALQUERY_NORMALIZED_ALLOWED_PARENT
    "${OVSPATIALQUERY_ALLOWED_PARENT}" ABSOLUTE)
get_filename_component(OVSPATIALQUERY_INSTALL_LEAF
    "${OVSPATIALQUERY_NORMALIZED_INSTALL_ROOT}" NAME)
file(RELATIVE_PATH
    OVSPATIALQUERY_INSTALL_RELATIVE
    "${OVSPATIALQUERY_NORMALIZED_ALLOWED_PARENT}"
    "${OVSPATIALQUERY_NORMALIZED_INSTALL_ROOT}")
if(NOT OVSPATIALQUERY_INSTALL_LEAF STREQUAL OVSPATIALQUERY_ALLOWED_LEAF OR
   IS_ABSOLUTE "${OVSPATIALQUERY_INSTALL_RELATIVE}" OR
   OVSPATIALQUERY_INSTALL_RELATIVE MATCHES "^\\.\\.($|/)")
    message(FATAL_ERROR
        "Refusing to replace install root outside the allowed deterministic layout: "
        "${OVSPATIALQUERY_NORMALIZED_INSTALL_ROOT}")
endif()
