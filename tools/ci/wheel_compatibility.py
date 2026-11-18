# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consume an existing wheel in an isolated environment of the selected interpreter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

QUALIFICATION_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(
    os.environ.get(
        "OVSQ_SOURCE_ROOT", os.environ.get("OVSQ_PUBLIC_SOURCE_ROOT", QUALIFICATION_ROOT)
    )
)
sys.path.insert(0, str(SOURCE_ROOT / "tools/bootstrap"))
import main as bootstrap  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-dir", type=Path, required=True)
    parser.add_argument("--gpu", choices=("required", "unavailable"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    wheels = sorted(options.wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected exactly one prepared wheel, found {len(wheels)}")
    wheel = wheels[0].resolve()
    options.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ovspatialquery-compatibility-") as temporary:
        environment = Path(temporary) / "venv"
        subprocess.run([sys.executable, "-I", "-m", "venv", str(environment)], check=True)
        python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        subprocess.run(
            [
                str(python),
                "-I",
                "-m",
                "pip",
                "--isolated",
                "install",
                "--no-index",
                "--no-deps",
                str(wheel),
            ],
            check=True,
        )
        details = bootstrap._isolated_wheel_probe(
            python,
            options.output.parent,
            require_cuda=options.gpu == "required",
            require_unavailable_cuda=options.gpu == "unavailable",
            wheel=wheel,
        )
        for key in ("module_path", "library_path"):
            if not Path(details[key]).resolve().is_relative_to(environment.resolve()):
                raise RuntimeError(
                    f"Wheel probe resolved outside its fresh environment: {details[key]}"
                )
        report = {
            "python": platform.python_version(),
            "platform": sys.platform,
            "wheel": wheel.name,
            "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            "library_sha256": hashlib.sha256(
                Path(details["library_path"]).read_bytes()
            ).hexdigest(),
            "runtime": details,
        }
    options.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
