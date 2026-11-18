#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Format and lint repository-owned native and Python sources; check the release version."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import release_version


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = Path(__file__).with_name("requirements.txt")
TOOL_ENV = PROJECT_ROOT / "_build" / "venv" / "format"
TOOL_MARKER = TOOL_ENV / ".requirements-sha256"

NATIVE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".cu",
    ".cuh",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
}
PYTHON_EXTENSIONS = {".py"}
OWNED_ROOTS = {
    "apps",
    "benchmarks",
    "examples",
    "include",
    "python",
    "source",
    "support",
    "tests",
    "tools",
}
EXCLUDED_PREFIXES = (
    "include/dlpack/",
    "tools/packman/",
)


def _run(command: list[str | Path], *, display: str | None = None) -> None:
    rendered = [str(part) for part in command]
    shown = display or subprocess.list2cmdline(rendered)
    print("[quality] + " + shown, file=sys.stderr, flush=True)
    subprocess.run(rendered, cwd=PROJECT_ROOT, check=True)


def _tool_path(name: str) -> Path:
    scripts = TOOL_ENV / ("Scripts" if os.name == "nt" else "bin")
    suffix = ".exe" if os.name == "nt" else ""
    return scripts / f"{name}{suffix}"


def _ensure_tools() -> tuple[Path, Path]:
    expected = hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()
    clang_format = _tool_path("clang-format")
    ruff = _tool_path("ruff")
    current = TOOL_MARKER.read_text(encoding="ascii").strip() if TOOL_MARKER.is_file() else ""
    if current == expected and clang_format.is_file() and ruff.is_file():
        return clang_format, ruff

    build_root = (PROJECT_ROOT / "_build").resolve()
    if not TOOL_ENV.resolve().is_relative_to(build_root):
        raise RuntimeError(f"tool environment must stay beneath {build_root}")
    if TOOL_ENV.exists():
        shutil.rmtree(TOOL_ENV)

    _run([sys.executable, "-m", "venv", TOOL_ENV])
    venv_python = _tool_path("python")
    _run(
        [
            venv_python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--only-binary=:all:",
            "--require-hashes",
            "-r",
            REQUIREMENTS,
        ]
    )
    TOOL_MARKER.write_text(expected + "\n", encoding="ascii")
    return clang_format, ruff


def _tracked_sources() -> tuple[list[Path], list[Path]]:
    if (PROJECT_ROOT / ".git").exists():
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
        )
        sources = [raw.decode("utf-8") for raw in result.stdout.split(b"\0") if raw]
    else:
        sources = sorted(
            path.relative_to(PROJECT_ROOT).as_posix()
            for directory in (*OWNED_ROOTS, "docs/site")
            for path in (PROJECT_ROOT / directory).rglob("*")
            if path.is_file()
            and not {"_build", ".venv", "__pycache__"}.intersection(
                path.relative_to(PROJECT_ROOT).parts
            )
        )
    native: list[Path] = []
    python: list[Path] = []
    for relative in sources:
        path = Path(relative)
        owned = path.parts[0] in OWNED_ROOTS or relative.startswith("docs/site/")
        if not owned or relative.startswith(EXCLUDED_PREFIXES):
            continue
        if path.suffix in NATIVE_EXTENSIONS:
            native.append(path)
        elif path.suffix in PYTHON_EXTENSIONS:
            python.append(path)
    return native, python


def _format(
    clang_format: Path,
    ruff: Path,
    native: list[Path],
    python: list[Path],
) -> None:
    _run(
        [clang_format, "-i", *native],
        display=f"{clang_format} -i <{len(native)} native/CUDA files>",
    )
    _run([ruff, "format", *python], display=f"{ruff} format <{len(python)} Python files>")


def _lint(ruff: Path, python: list[Path]) -> None:
    _run([ruff, "check", *python], display=f"{ruff} check <{len(python)} Python files>")


def _check(
    clang_format: Path,
    ruff: Path,
    native: list[Path],
    python: list[Path],
) -> None:
    _run(
        [clang_format, "--dry-run", "--Werror", "--ferror-limit=20", *native],
        display=f"{clang_format} --dry-run --Werror <{len(native)} native/CUDA files>",
    )
    _run(
        [ruff, "format", "--check", *python],
        display=f"{ruff} format --check <{len(python)} Python files>",
    )
    _lint(ruff, python)


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("format", "check", "lint"))
    options = parser.parse_args(arguments)

    clang_format, ruff = _ensure_tools()
    native, python = _tracked_sources()
    print(f"[quality] scope: {len(native)} native/CUDA files, {len(python)} Python files")

    try:
        if options.command == "format":
            _format(clang_format, ruff, native, python)
        elif options.command == "lint":
            _lint(ruff, python)
        else:
            _check(clang_format, ruff, native, python)
            if not release_version.check(PROJECT_ROOT):
                return 1
    except subprocess.CalledProcessError as exc:
        return exc.returncode or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
