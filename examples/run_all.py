# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run visual curriculum lessons in fresh processes and index their evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


def _absolute_without_resolving(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return Path.cwd() / expanded


EXAMPLES_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLES_ROOT / "python"))

from _evidence import compare_semantics, validate_evidence  # noqa: E402

RUNNABLE = (
    "01_static_candidate_raycast",
    "02_sphere_sweep",
    "03_shape_overlap",
    "04_filtered_view",
    "05_two_set_pair_snapshot",
    "06_precise_triangle",
    "07_time_sampled_update",
    "08_reusable_cpu_batch",
    "09_immutable_version_overlap",
    "10_pair_reporter_deltas",
)


def _arguments(arguments: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("python", "native", "both"), required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--native-dir", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--sdk-root", type=Path)
    parser.add_argument("--expect-python-source", type=Path)
    parser.add_argument("--reject-python-source", type=Path)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--include", action="append", choices=RUNNABLE)
    parsed = parser.parse_args(arguments)
    if parsed.frames <= 0:
        parser.error("--frames must be positive")
    if parsed.mode in ("native", "both") and parsed.native_dir is None:
        parser.error("--native-dir is required for native execution")
    if parsed.mode in ("native", "both") and parsed.library is None:
        parser.error("--library is required for native execution")
    if parsed.mode in ("python", "both") and parsed.sdk_root is None:
        parser.error("--sdk-root is required for header identity")
    if parsed.expect_python_source and parsed.reject_python_source:
        parser.error("source provenance expectations are mutually exclusive")
    return parsed


def _run(command: list[str], environment: dict[str, str]) -> None:
    completed = subprocess.run(
        command,
        cwd=EXAMPLES_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode:
        raise RuntimeError(
            f"fresh-process example failed ({completed.returncode}): "
            + subprocess.list2cmdline(command)
        )


def _python_command(
    options: argparse.Namespace,
    lesson: str,
) -> list[str]:
    command = [
        str(_absolute_without_resolving(options.python)),
        str(EXAMPLES_ROOT / "python" / f"{lesson}.py"),
        "--frames",
        str(options.frames),
        "--evidence",
        str(options.evidence.resolve()),
        "--sdk-root",
        str(options.sdk_root.resolve()),
    ]
    if not options.interactive:
        command.append("--headless")
    if options.library is not None:
        command.extend(("--library", str(options.library.resolve())))
    if options.expect_python_source is not None:
        command.extend(("--expect-python-source", str(options.expect_python_source.resolve())))
    return command


def _native_executable(directory: Path, lesson: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    executable = directory.resolve() / f"{lesson}{suffix}"
    if not executable.is_file():
        raise FileNotFoundError(f"native lesson is not built: {executable}")
    return executable


def _native_command(
    options: argparse.Namespace,
    lesson: str,
) -> list[str]:
    command = [
        str(_native_executable(options.native_dir, lesson)),
        "--frames",
        str(options.frames),
        "--evidence",
        str(options.evidence.resolve()),
    ]
    if not options.interactive:
        command.append("--headless")
    return command


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(arguments: list[str] | None = None) -> dict[str, Any]:
    options = _arguments(arguments)
    selected = tuple(options.include or RUNNABLE)
    languages = (
        ("python", "cpp")
        if options.mode == "both"
        else ("python",)
        if options.mode == "python"
        else ("cpp",)
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    native_environment = environment.copy()
    if options.mode in ("native", "both"):
        library = options.library.resolve()
        if not library.is_file():
            raise FileNotFoundError(f"verified native library is missing: {library}")
        loader_path = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        native_environment[loader_path] = os.pathsep.join(
            part
            for part in (
                str(options.library.parent.resolve()),
                native_environment.get(loader_path, ""),
            )
            if part
        )
    evidence_root = options.evidence.resolve()
    evidence_root.mkdir(parents=True, exist_ok=True)
    index: dict[str, Any] = {"schema_version": 2, "lessons": []}

    for lesson in selected:
        source = EXAMPLES_ROOT / "python" / f"{lesson}.py"
        if options.mode in ("python", "both") and not source.is_file():
            raise FileNotFoundError(f"Python lesson is not implemented: {source}")
        if options.mode in ("python", "both"):
            _run(_python_command(options, lesson), environment)
        if options.mode in ("native", "both"):
            _run(_native_command(options, lesson), native_environment)

        payloads = {
            language: validate_evidence(evidence_root / lesson / language / "semantic.json")
            for language in languages
        }
        if options.reject_python_source is not None and "python" in payloads:
            module = Path(payloads["python"]["native"]["ovspatialquery"]["python_module"]).resolve()
            rejected = options.reject_python_source.resolve()
            if module.is_relative_to(rejected):
                raise RuntimeError(
                    f"wheel mode imported repository source {module} under {rejected}"
                )
        parity = None
        if options.mode == "both":
            compare_semantics(payloads["python"], payloads["cpp"])
            parity = True
        index["lessons"].append(
            {
                "lesson": lesson,
                "semantic_parity": parity,
                "artifacts": {
                    language: {
                        "semantic_json": str(
                            (evidence_root / lesson / language / "semantic.json").resolve()
                        ),
                        "semantic_sha256": _sha256(
                            evidence_root / lesson / language / "semantic.json"
                        ),
                    }
                    for language, payload in payloads.items()
                },
            }
        )

    index_path = evidence_root / "index.json"
    index_path.write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"index": str(index_path)}))
    return index


if __name__ == "__main__":
    try:
        run()
    except BaseException as failure:
        print(f"run_all: {type(failure).__name__}: {failure}", file=sys.stderr)
        raise
