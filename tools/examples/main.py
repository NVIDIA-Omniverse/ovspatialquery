# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve and run one visual lesson from the verified source checkout."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys


def _absolute_without_resolving(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return Path.cwd() / expanded


LESSONS = {
    "01": "01_static_candidate_raycast",
    "02": "02_sphere_sweep",
    "03": "03_shape_overlap",
    "04": "04_filtered_view",
    "05": "05_two_set_pair_snapshot",
    "06": "06_precise_triangle",
    "07": "07_time_sampled_update",
    "08": "08_reusable_cpu_batch",
    "09": "09_immutable_version_overlap",
    "10": "10_pair_reporter_deltas",
}
CAPABILITY_LESSON = "11"


@dataclass(frozen=True)
class Request:
    lesson: str
    language: str
    headless: bool
    frames: int
    python: Path | None
    native_dir: Path | None
    library: Path | None
    sdk_root: Path | None
    python_source: Path | None
    evidence: Path | None
    require_cuda: bool


def parse_arguments(arguments: list[str] | None = None) -> Request:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lesson", choices=(*LESSONS, CAPABILITY_LESSON))
    parser.add_argument(
        "--language",
        choices=("python", "native", "both"),
        default="python",
        help="Lesson implementation to run (default: python)",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--native-dir", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--sdk-root", type=Path)
    parser.add_argument("--python-source", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Lesson 11 only: fail unless the CUDA lane runs on a real device",
    )
    parsed = parser.parse_args(arguments)
    if parsed.frames <= 0:
        parser.error("--frames must be positive")
    if parsed.lesson == CAPABILITY_LESSON:
        if parsed.language != "python":
            parser.error("lesson 11 uses one C ABI producer and Python presenter")
    elif parsed.require_cuda:
        parser.error("--require-cuda applies to lesson 11 only")
    return Request(**vars(parsed))


def _train() -> str:
    """stable, or next for the opt-in in-development train with its own build state."""
    return os.environ.get("OVSPATIALQUERY_TRAIN") or "stable"


def _train_suffix() -> str:
    return "" if _train() == "stable" else f"-{_train()}"


def _default_install(project_root: Path) -> Path:
    return project_root / "_build" / "bootstrap" / f"native-{_train()}" / "install"


def _default_python(project_root: Path, platform_name: str) -> Path:
    suffix = Path("Scripts") / "python.exe" if platform_name == "win32" else Path("bin") / "python"
    return project_root / "_build" / "venv" / f"ovspatialqueryapp{_train_suffix()}" / suffix


def _default_library(project_root: Path, platform_name: str) -> Path:
    install = _default_install(project_root)
    if platform_name == "win32":
        return install / "bin" / "ovspatialquery.dll"
    return install / "lib" / "libovspatialquery.so"


def build_command(
    request: Request,
    *,
    project_root: Path,
    python_executable: Path,
    evidence: Path,
    platform_name: str,
) -> tuple[list[str], dict[str, str]]:
    project_root = project_root.resolve()
    source_python = _absolute_without_resolving(
        request.python
        if request.python is not None
        else _default_python(project_root, platform_name)
    )
    native_dir = (
        request.native_dir.resolve()
        if request.native_dir is not None
        else (
            project_root
            / "_build"
            / "examples"
            / f"native-vs{_train_suffix()}"
            / ("Release" if platform_name == "win32" else "")
        ).resolve()
    )
    library = (
        request.library.resolve()
        if request.library is not None
        else _default_library(project_root, platform_name).resolve()
    )
    sdk_root = (
        request.sdk_root.resolve()
        if request.sdk_root is not None
        else _default_install(project_root).resolve()
    )
    python_source = (
        request.python_source.resolve()
        if request.python_source is not None
        else (project_root / "python" / "src").resolve()
    )
    if request.lesson == CAPABILITY_LESSON:
        command = [
            str(_absolute_without_resolving(python_executable)),
            str((project_root / "examples" / "python" / "11_capability_showcase.py").resolve()),
            "--sdk-root",
            str(sdk_root),
            "--library",
            str(library),
            "--native-dir",
            str(project_root / "_build" / "examples" / f"11-capability-showcase{_train_suffix()}"),
            "--evidence",
            str(evidence.resolve()),
            "--frames",
            str(request.frames),
            "--expect-python-source",
            str(python_source),
        ]
        if request.headless:
            command.append("--headless")
        if request.require_cuda:
            command.append("--require-cuda")
        environment = os.environ.copy()
        for name in ("PYTHONHOME", "PYTHONPATH", "OVSPATIALQUERY_LIBRARY"):
            environment.pop(name, None)
        return command, environment
    command = [
        str(_absolute_without_resolving(python_executable)),
        str((project_root / "examples" / "run_all.py").resolve()),
        "--mode",
        request.language,
        "--python",
        str(source_python),
        "--native-dir",
        str(native_dir),
        "--library",
        str(library),
        "--sdk-root",
        str(sdk_root),
        "--expect-python-source",
        str(python_source),
        "--evidence",
        str(evidence.resolve()),
        "--frames",
        str(request.frames),
        "--include",
        LESSONS[request.lesson],
    ]
    if not request.headless:
        command.append("--interactive")
    environment = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "OVSPATIALQUERY_LIBRARY"):
        environment.pop(name, None)
    return command, environment


def main(arguments: list[str] | None = None) -> int:
    request = parse_arguments(arguments)
    project_root = Path(__file__).resolve().parents[2]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence = (
        request.evidence
        if request.evidence is not None
        else project_root
        / "_build"
        / "evidence"
        / "examples"
        / "manual"
        / f"{stamp}-{request.lesson}-{request.language}"
    )
    command, environment = build_command(
        request,
        project_root=project_root,
        python_executable=Path(sys.executable),
        evidence=evidence,
        platform_name=sys.platform,
    )
    print(f"[example] evidence: {evidence.resolve()}")
    return subprocess.run(
        command,
        cwd=project_root,
        env=environment,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
