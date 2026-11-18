# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The installed public headers, relative to include/ovspatialquery/. The
# install rule in CMakeLists.txt, verify_install.cmake and the tooling's
# contract.public_headers() (installed-artifact and ARM64 header comparisons)
# consume this one list.
set(OVSPATIALQUERY_PUBLIC_HEADERS
    ovspatialquery_export.h
    ovspatialquery.h
    ovspatialquery_explicit.h
    ovspatialquery_pairs.h)
# Installed only with the Ovstage companion.
set(OVSPATIALQUERY_OVSTAGE_PUBLIC_HEADER ovspatialquery_ovstage.h)
