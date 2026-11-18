# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import platform
import struct
import sys


print(
    json.dumps(
        {
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "bits": struct.calcsize("P") * 8,
            "machine": platform.machine(),
        }
    )
)
