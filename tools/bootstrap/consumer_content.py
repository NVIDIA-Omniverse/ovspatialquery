# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select the public guide, consumer skills and runnable lesson sources for distribution."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import contract


SDK_CONTENT = Path("share/ovspatialquery")


def collect(
    root: Path,
    language: str = "sdk",
    *,
    version: str | None = None,
    source_revision: str | None = None,
) -> dict[str, bytes]:
    guide = (
        root / "docs/html"
        if (root / "docs/html/index.html").is_file()
        else root / "_build/docs/html"
    )
    if not (guide / "index.html").is_file() or not (guide / "licenses").is_dir():
        raise RuntimeError("Public HTML guide is missing; run ./dev.sh docs before packaging")
    sources: dict[str, Path] = {}

    def tree(source: Path, destination: str, patterns: tuple[str, ...] = ("**/*",)) -> None:
        for pattern in patterns:
            for path in source.glob(pattern):
                relative = path.relative_to(source)
                if (
                    path.is_file()
                    and path.suffix.lower() not in (".mp4", ".webm", ".mov", ".avi")
                    and not any(
                        part.startswith(".") or part == "__pycache__" for part in relative.parts
                    )
                ):
                    sources[f"{destination}/{relative.as_posix()}"] = path

    tree(guide, "docs")
    tree(root / "docs/site/snippets", "docs/snippets", ("*.py", "*.c", "*.out"))
    tree(root / "skills/ovspatialquery", "skills/ovspatialquery")
    tree(
        root / "examples/native",
        "examples/native",
        # The wheel's one native lesson, 11, and the lesson header it includes.
        (("11_*.cpp", "example_ovstage.h") if language == "wheel" else ("[0-9][0-9]_*.cpp", "*.h")),
    )
    tree(
        root / "examples/11_capability_showcase",
        "examples/11_capability_showcase",
        ("CMakeLists.txt",),
    )
    tree(root / "support/native", "support/native", ("*.h",))
    tree(root / "examples", "examples", ("requirements-*.txt",))
    if language == "sdk" and (root / "examples/CMakeLists.txt").is_file():
        sources["examples/CMakeLists.txt"] = root / "examples/CMakeLists.txt"
    if language == "wheel":
        tree(
            root / "examples/python",
            "examples/python",
            ("[0-9][0-9]_*.py", "_runtime.py", "_evidence.py"),
        )
    payloads = {name: path.read_bytes() for name, path in sorted(sources.items())}
    # The skill ships byte-identical to its source, so its NVSkills signature (skill.oms.sig)
    # verifies strictly in the package; the build identity goes in README.txt instead.
    identity = (
        f"skills/ovspatialquery: version {version or contract.artifact_version()}, "
        f"source revision {source_revision or contract.native_source_revision()}.\n"
    ).encode()
    if not (root / "examples/CMakeLists.txt").is_file():
        payloads["README.txt"] = (
            "Open docs/index.html for the guide and docs/snippets for runnable C/Python programs.\n"
            "skills/ovspatialquery is the installable consumer skill.\n"
            "Public headers are distributed in the SDK; see the guide for program dependencies.\n"
        ).encode() + identity
        return payloads
    payloads["README.txt"] = (
        "Open docs/index.html for the public guide; docs/snippets contains its complete programs.\n"
        "Its Python programs import Warp (warp-lang); the guide's Install page lists the packages.\n"
        "skills/ovspatialquery is the installable consumer agent skill, including executable assets.\n"
        "The numbered visual lessons require separately installed published Ovstage and OvRTX packages,\n"
        "Warp and a supported NVIDIA GPU; no renderer or datasets are bundled.\n"
        "Lesson 11's Python presenter also builds a native producer: pass --sdk-root <absolute-sdk>\n"
        "and --library <absolute-sdk-library>; CMake/CTest on PATH and a C++17 compiler are required.\n"
        "SDK public headers are distributed only in the SDK. See the guide for package requirements.\n"
    ).encode()
    requirements = (
        "python -m pip install --require-hashes --no-deps -r examples/requirements-<platform>.txt\n"
        "Replace <platform> with manylinux_2_35_x86_64 for Linux x86-64 or windows-x86_64\n"
        "for Windows x86-64. These optional packages are not bundled.\n"
        if any(name.startswith("examples/requirements-") for name in sources)
        else ""
    )
    if language == "wheel":
        # Python puts the working directory first on sys.path, so a command run in the installed
        # package directory imports the package's types.py in place of the standard library's.
        payloads["README.txt"] += (
            "Use a separate CPython 3.12 lesson environment and install this wheel into it first:\n"
            "python -m pip install --no-deps <path-to-ovspatialquery-wheel>\n"
            "Copy examples/ and support/ to a writable directory and run the following commands\n"
            "there. Do not run Python from this package directory: its types.py hides the standard\n"
            "library's types module.\n"
            f"{requirements}"
            "python examples/python/01_static_candidate_raycast.py --headless --evidence <output>.\n"
        ).encode() + identity
        return payloads
    if requirements:
        payloads["README.txt"] += (
            f"Use a separate CPython 3.12 lesson environment. From this directory, run\n{requirements}"
        ).encode()
    payloads["README.txt"] += (
        "Native: cmake -S examples -B <build> -DOVSPATIALQUERY_INSTALL_DIR=<absolute-sdk> "
        "-DOVSTAGE_DIR=<absolute-ovstage> -DOVRTX_DIR=<absolute-ovrtx>, then cmake --build <build>.\n"
    ).encode() + identity
    return payloads


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("install", "verify"))
    parser.add_argument("destination", type=Path)
    parser.add_argument("--version")
    parser.add_argument("--source-revision")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.command == "install":
        subprocess.run([sys.executable, str(root / "tools/docs/main.py")], check=True)
    for name, payload in collect(
        root, version=args.version, source_revision=args.source_revision
    ).items():
        target = args.destination / SDK_CONTENT / name
        if args.command == "verify":
            if not target.is_file() or target.read_bytes() != payload:
                raise RuntimeError(f"Installed consumer content differs from source: {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)


if __name__ == "__main__":
    main()
