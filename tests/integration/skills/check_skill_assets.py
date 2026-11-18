# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check every consumer skill under skills/ against the installed library.

A skill is only useful while its code still works, so this runs as a CTest:

- SKILL.md frontmatter uses only Agent Skills standard keys, `name` matches the
  directory, and the description fits the 1024-character limit every harness enforces.
- Every relative link in a skill's Markdown resolves to a file in the skill.
- Every `assets/python/*.py` runs to success with the installed library
  (`OVSPATIALQUERY_LIBRARY`) and the given interpreter (which needs ovstage and numpy).
- Every `assets/native/CMakeLists.txt` configures against the installed SDK root and the
  Ovstage package, builds, and passes its own CTest tests.
- With `--bench-tasks`, every skill-benchmark oracle still passes its own verifier, so the
  benchmark keeps measuring the current API.

A program or verifier that needs a GPU exits 77 without a CUDA device or a CUDA-enabled
library, which counts as skipped; with `--cuda` (a CUDA build, tested on a GPU host) a
skip is a failure.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

STANDARD_KEYS = {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}
LINK = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")


def frontmatter(text: str) -> dict[str, str]:
    match = re.match(r"---\n(.*?)\n---\n", text, re.S)
    if match is None:
        raise ValueError("SKILL.md has no YAML frontmatter")
    keys: dict[str, str] = {}
    current = None
    for line in match.group(1).splitlines():
        top = re.match(r"([A-Za-z-]+):\s*(.*)", line)
        if top and not line.startswith(" "):
            current = top.group(1)
            keys[current] = top.group(2).strip()
        elif current is not None:
            keys[current] = (keys[current] + " " + line.strip()).strip()
    return keys


def check_structure(skill: Path) -> list[str]:
    errors = []
    keys = frontmatter((skill / "SKILL.md").read_text(encoding="utf-8"))
    if keys.get("name") != skill.name:
        errors.append(f"{skill.name}: frontmatter name {keys.get('name')!r} != directory")
    if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", skill.name) or len(skill.name) > 64:
        errors.append(f"{skill.name}: name must be lowercase-hyphenated, at most 64 chars")
    description = re.sub(r"^>-?\s*", "", keys.get("description", ""))
    if not description or len(description) > 1024:
        errors.append(f"{skill.name}: description missing or over 1024 chars ({len(description)})")
    for key in set(keys) - STANDARD_KEYS:
        errors.append(f"{skill.name}: non-standard frontmatter key {key!r}")
    for document in skill.rglob("*.md"):
        for target in LINK.findall(document.read_text(encoding="utf-8")):
            if "://" in target:
                continue
            resolved = (document.parent / target).resolve()
            if not resolved.exists():
                errors.append(f"{document.relative_to(skill.parent)}: broken link {target}")
            elif skill.resolve() not in resolved.parents and resolved != skill.resolve():
                errors.append(
                    f"{document.relative_to(skill.parent)}: link leaves the skill: {target}"
                )
    return errors


SKIPPED = 77  # a GPU program or verifier without a CUDA device or CUDA build


def run(
    command: list[str],
    env: dict[str, str],
    cwd: Path,
    label: str,
    skip: bool = False,
    forbid: str = "",
) -> str | None:
    """None on success, else the failure. With `skip`, exit code SKIPPED is a skip; a
    successful run whose output contains `forbid` fails."""
    done = subprocess.run(command, env=env, cwd=cwd, capture_output=True, text=True, timeout=600)
    if skip and done.returncode == SKIPPED:
        print(f"{label}: skipped (no CUDA device or CUDA build)", flush=True)
        return None
    if done.returncode != 0 or (forbid and forbid in done.stdout):
        return f"{label} failed ({done.returncode}):\n{done.stdout[-3000:]}\n{done.stderr[-3000:]}"
    return None


def check_python(skill: Path, python: str, library: Path, env: dict[str, str]) -> list[str]:
    errors = []
    env = dict(env, OVSPATIALQUERY_LIBRARY=str(library))
    for asset in sorted((skill / "assets" / "python").glob("*.py")):
        with tempfile.TemporaryDirectory() as scratch:
            print(f"python {asset.relative_to(skill.parent)}", flush=True)
            error = run([python, str(asset)], env, Path(scratch), str(asset.name))
            if error:
                errors.append(f"{skill.name}: {error}")
    return errors


def check_native(skill: Path, args, env: dict[str, str]) -> list[str]:
    source = skill / "assets" / "native"
    if not (source / "CMakeLists.txt").is_file():
        return []
    build = (args.prebuilt_native or args.build_dir) / skill.name
    build.mkdir(parents=True, exist_ok=True)
    # --fresh: the build tree outlives an SDK or Ovstage pin change, and a cached
    # ovstage_DIR or ovspatialquery_DIR would keep the previous package's headers.
    configure = [
        args.cmake,
        "--fresh",
        "-S",
        str(source),
        "-B",
        str(build),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DOVSQ_SDK={args.sdk}",
        f"-DOVSTAGE_DIR={args.ovstage_dir}",
    ]
    if args.generator:
        configure += ["-G", args.generator]
    for option, value in (
        ("CMAKE_C_COMPILER", args.c_compiler),
        ("CMAKE_CXX_COMPILER", args.cxx_compiler),
        ("CMAKE_MAKE_PROGRAM", args.make_program),
    ):
        if value:
            configure.append(f"-D{option}={value}")
    ctest = str(Path(args.cmake).with_name("ctest" + Path(args.cmake).suffix))
    print(f"native {source.relative_to(skill.parent)}", flush=True)
    steps = (
        (configure, "configure"),
        ([args.cmake, "--build", str(build), "--config", "Release"], "build"),
        (
            [
                ctest,
                "--test-dir",
                str(build),
                "-C",
                "Release",
                "--output-on-failure",
                "--no-tests=error",
            ],
            "ctest",
        ),
    )
    for command, label in steps[2:] if args.prebuilt_native else steps:
        # CTest passes a test that exits with its SKIP_RETURN_CODE and prints it as
        # skipped; on a CUDA build that would hide a GPU program that never ran.
        forbid = "***Skipped" if args.cuda and label == "ctest" else ""
        error = run(command, env, build.parent, f"{skill.name} native {label}", forbid=forbid)
        if error:
            return [error]
    return []


def check_oracles(tasks: Path, args, env: dict[str, str]) -> list[str]:
    """Run each task's oracle through its verifier against a kit made of links.

    The verifiers expect the benchmark kit's POSIX layout (sdk/, venv/bin/python,
    venv/lib/python3.12/site-packages/ovstage); it is assembled here from the given
    interpreter and Ovstage package, whether or not that interpreter is a venv.
    """
    if os.name == "nt":
        return ["--bench-tasks: the skill benchmark and its kit layout are Linux-only"]
    errors = []
    with tempfile.TemporaryDirectory() as scratch:
        kit = Path(scratch) / "kit"
        site = kit / "venv/lib/python3.12/site-packages"
        site.mkdir(parents=True)
        (kit / "venv/bin").mkdir()
        (kit / "sdk").symlink_to(args.sdk)
        (site / "ovstage").symlink_to(args.ovstage_dir)
        python = kit / "venv/bin/python"
        python.write_text(f'#!/bin/sh\nexec "{args.python}" "$@"\n')
        python.chmod(0o755)
        tool_dirs = {str(Path(args.cmake).parent)}
        if args.make_program:
            tool_dirs.add(str(Path(args.make_program).parent))
        oracle_env = dict(
            env,
            OVSPATIALQUERY_LIBRARY=str(args.library),
            PATH=os.pathsep.join([*sorted(tool_dirs), env.get("PATH", "")]),
        )
        for task in sorted(p for p in tasks.iterdir() if (p / "verify.py").is_file()):
            sandbox = Path(scratch) / task.name
            shutil.copytree(task / "files", sandbox, dirs_exist_ok=True)
            shutil.copytree(task / "oracle", sandbox, dirs_exist_ok=True)
            print(f"oracle {task.name}", flush=True)
            command = [args.python, "-I", str(task / "verify.py"), "--sandbox", str(sandbox)]
            if args.prebuilt_native and (task / "oracle/CMakeLists.txt").is_file():
                command += [
                    "--prebuilt-executable",
                    str(args.prebuilt_native / "oracles" / task.name / "solution"),
                ]
            error = run(
                [*command, "--kit", str(kit)],
                oracle_env,
                sandbox,
                f"oracle {task.name}",
                skip=not args.cuda,
            )
            if error:
                errors.append(error)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skills", type=Path, required=True)
    parser.add_argument("--sdk", type=Path, required=True, help="installed SDK root")
    parser.add_argument("--library", type=Path, required=True, help="installed shared library")
    parser.add_argument("--ovstage-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--generator", default="")
    parser.add_argument("--c-compiler", default="")
    parser.add_argument("--cxx-compiler", default="")
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument(
        "--prebuilt-native", type=Path, help="Cross-built native examples and oracle executables"
    )
    parser.add_argument("--make-program", default="", help="Ninja/make used by --generator")
    parser.add_argument(
        "--bench-tasks", type=Path, help="tests/integration/skills/tasks to check oracles"
    )
    parser.add_argument(
        "--cuda", action="store_true", help="a CUDA build on a GPU host: GPU programs must run"
    )
    args = parser.parse_args()
    if args.prebuilt_native:
        args.prebuilt_native = args.prebuilt_native.resolve()
    for name in ("skills", "sdk", "library", "ovstage_dir", "build_dir"):
        setattr(args, name, getattr(args, name).resolve())

    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    skills = sorted(p for p in args.skills.iterdir() if (p / "SKILL.md").is_file())
    if not skills:
        print(f"no skills under {args.skills}", file=sys.stderr)
        return 1
    errors: list[str] = []
    for skill in skills:
        errors += check_structure(skill)
        errors += check_python(skill, args.python, args.library, env)
        errors += check_native(skill, args, env)
    if args.bench_tasks:
        errors += check_oracles(args.bench_tasks.resolve(), args, env)
    for error in errors:
        print(error, file=sys.stderr)
    print(f"{len(skills)} skill(s), {len(errors)} problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
