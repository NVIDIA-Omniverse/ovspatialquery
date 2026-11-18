#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the SDK guide, or the whole internal hub, and optionally serve it locally."""

from __future__ import annotations

import argparse
import functools
import hashlib
import http.server
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GUIDE_SOURCE = PROJECT_ROOT / "docs" / "site"
GUIDE_OUTPUT = PROJECT_ROOT / "_build" / "docs" / "html"
HUB_BUILD = PROJECT_ROOT / "docs" / "presentations" / "site" / "build.py"
HUB_OUTPUT = PROJECT_ROOT / "_build" / "presentation" / "site"
SNIPPETS = GUIDE_SOURCE / "snippets"
NATIVE_INSTALL = PROJECT_ROOT / "_build" / "bootstrap" / "native-stable" / "install"
BOOTSTRAP_PYTHON = (
    PROJECT_ROOT
    / "_build"
    / "venv"
    / "ovspatialqueryapp"
    / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
)


def _run(command: list[str | Path]) -> None:
    rendered = [str(part) for part in command]
    print("[docs] + " + subprocess.list2cmdline(rendered), file=sys.stderr, flush=True)
    subprocess.run(rendered, cwd=PROJECT_ROOT, check=True)


def _ensure_env(requirements: Path, environment: Path) -> Path:
    """Create or refresh the docs venv whenever the lock changes."""
    expected = hashlib.sha256(requirements.read_bytes()).hexdigest()
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    marker = environment / ".requirements-sha256"
    current = marker.read_text(encoding="ascii").strip() if marker.is_file() else ""
    if current == expected and python.is_file():
        return python
    if sys.version_info[:2] != (3, 12):
        raise SystemExit(f"docs needs CPython 3.12 for its lock; this is {sys.version.split()[0]}")
    if environment.exists():
        shutil.rmtree(environment)
    _run([sys.executable, "-m", "venv", environment])
    _run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--only-binary=:all:",
            "--require-hashes",
            "-r",
            requirements,
        ]
    )
    marker.write_text(expected + "\n", encoding="ascii")
    return python


def _snippet_python() -> Path:
    wheel_root = PROJECT_ROOT / "_build/python-wheel"
    runtime = wheel_root / "runtime.json"
    pattern = "*.whl"
    if runtime.is_file():
        version = json.loads(runtime.read_text(encoding="utf-8"))["package_version"]
        pattern = f"ovspatialquery-{version}-*.whl"
    wheels = list((wheel_root / "dist").glob(pattern))
    if len(wheels) != 1:
        raise SystemExit(
            "Run dev.sh python-wheel with the desired scope before checking snippets, "
            "or pass --python with an installed wheel, Warp, NumPy and Ovstage."
        )
    python = _ensure_env(
        Path(__file__).with_name("snippet-requirements.txt"),
        PROJECT_ROOT / "_build/venv/docs-snippets",
    )
    _run([python, "-m", "pip", "install", "--no-deps", "--force-reinstall", wheels[0]])
    return python


def _serve(root: Path, port: int) -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as server:
        print(f"[docs] serving {root} at http://127.0.0.1:{port}/ (Ctrl+C stops)", flush=True)
        print(
            f"[docs] remote host? forward it with: ssh -L {port}:127.0.0.1:{port} <host>",
            flush=True,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


SNIPPET_TIMEOUT = 300
# Runs one program and, if it is still running near the timeout, prints every thread's
# Python stack and exits, so a hang names where it is stuck.
WATCHDOG = (
    "import faulthandler, runpy, sys; "
    "faulthandler.dump_traceback_later({seconds}, exit=True); "
    "sys.argv = [sys.argv[1]]; runpy.run_path(sys.argv[0], run_name='__main__')"
)
# Programs named cuda_*.py need a CUDA device and a library with the CUDA backend. This probe
# creates a CUDA plan the way they do and exits nonzero when either is missing.
CUDA_PROBE = """
from ovspatialquery import Device, DeviceBudget, HitMode, InstanceConfig, Library, RaycastPlanConfig
budget = DeviceBudget(Device.cuda(0), 1 << 24)
with Library.open() as library, library.instance(InstanceConfig(device_budgets=(budget,))) as instance:
    population = instance.explicit().create_population()
    instance.create_raycast_plan(
        RaycastPlanConfig(population.target.reference, HitMode.CLOSEST, 1, 1, 1, device=Device.cuda(0))
    )
"""


def _run_program(argv: list[str], **options) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            argv, capture_output=True, text=True, timeout=SNIPPET_TIMEOUT, **options
        )
    except subprocess.TimeoutExpired as expired:

        def text(value) -> str:
            return value.decode(errors="replace") if isinstance(value, bytes) else value or ""

        return subprocess.CompletedProcess(
            argv,
            -1,
            text(expired.stdout),
            text(expired.stderr) + f"\n[docs] timed out after {SNIPPET_TIMEOUT} s\n",
        )


def _compare(name: str, run: subprocess.CompletedProcess, expected_file: Path) -> bool:
    expected = expected_file.read_text(encoding="utf-8") if expected_file.is_file() else None
    if run.returncode == 0 and (expected is None or run.stdout == expected):
        return True
    print(f"[docs] FAILED {name} (exit {run.returncode})", file=sys.stderr)
    print(run.stdout + run.stderr, file=sys.stderr)
    if expected is not None and run.stdout != expected:
        print(f"[docs] expected:\n{expected}", file=sys.stderr)
    return False


def _check_c_snippets(sdk: Path) -> int:
    """Compile every C program against the SDK, run it and compare its output."""
    compiler = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
    if os.name == "nt" or compiler is None:
        print("[docs] C snippets skipped: no cc on this host", flush=True)
        return 0
    failures = 0
    for program in sorted(SNIPPETS.glob("*.c")):
        print(f"[docs] snippet {program.name}", flush=True)
        binary = PROJECT_ROOT / "_build" / "docs" / "snippets" / program.stem
        binary.parent.mkdir(parents=True, exist_ok=True)
        build = subprocess.run(
            [
                compiler,
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-Werror",
                f"-I{sdk / 'include'}",
                str(program),
            ]
            + [
                f"-L{sdk / 'lib'}",
                "-lovspatialquery",
                f"-Wl,-rpath,{sdk / 'lib'}",
                "-o",
                str(binary),
            ],
            capture_output=True,
            text=True,
        )
        if build.returncode != 0:
            failures += 1
            print(f"[docs] FAILED to compile {program.name}\n{build.stderr}", file=sys.stderr)
            continue
        run = _run_program([str(binary)])
        failures += not _compare(program.name, run, Path(f"{program}.out"))
    return failures


def _check_snippets(python: Path, sdk: Path, require_cuda: bool) -> int:
    """Run every guide program and compare its output with the .out file the page shows."""
    if not python.is_file():
        raise SystemExit(
            f"--check-snippets needs a Python with ovspatialquery installed; {python} is missing"
        )
    environment = dict(os.environ)
    if python == BOOTSTRAP_PYTHON and "OVSPATIALQUERY_LIBRARY" not in environment:
        # The bootstrap venv installs the package from source; point it at the native install.
        library = "bin/ovspatialquery.dll" if os.name == "nt" else "lib/libovspatialquery.so"
        environment["OVSPATIALQUERY_LIBRARY"] = str(NATIVE_INSTALL / library)
    failures = 0
    cuda: bool | None = None
    for program in sorted(SNIPPETS.glob("*.py")):
        expected_file = program.with_suffix(".out")
        if program.name.startswith("cuda_"):
            if cuda is None:
                probe = _run_program([str(python), "-I", "-c", CUDA_PROBE], env=environment)
                cuda = probe.returncode == 0
                if not cuda and require_cuda:
                    print(probe.stdout + probe.stderr, file=sys.stderr)
            if not cuda:
                if require_cuda:
                    print(f"[docs] FAILED {program.name}: no CUDA plan available", file=sys.stderr)
                    failures += 1
                else:
                    print(f"[docs] snippet {program.name} skipped: no CUDA plan available")
                continue
        print(f"[docs] snippet {program.name}", flush=True)
        watchdog = WATCHDOG.format(seconds=SNIPPET_TIMEOUT - 30)
        run = _run_program(
            [str(python), "-I", "-c", watchdog, str(program)],
            cwd=PROJECT_ROOT,
            env=environment,
        )
        failures += not _compare(program.name, run, expected_file)
    failures += _check_c_snippets(sdk)
    print(f"[docs] snippets: {failures} failed", flush=True)
    return 1 if failures else 0


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dev.sh docs", description=__doc__)
    parser.add_argument(
        "--hub",
        action="store_true",
        help="build the internal hub (guide, deck, slide screenshots) instead of the guide alone; further options go to its build.py",
    )
    parser.add_argument(
        "--serve", action="store_true", help="serve the result on 127.0.0.1 after building"
    )
    parser.add_argument("--port", type=int, default=8000, help="port for --serve (default: 8000)")
    parser.add_argument(
        "--check-snippets",
        action="store_true",
        help="run the guide's programs against an installed ovspatialquery and compare their output",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="with --check-snippets, fail instead of skipping cuda_*.py programs without CUDA",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=None,
        help="prepared interpreter for --check-snippets (default: isolated pinned snippet environment)",
    )
    parser.add_argument(
        "--sdk",
        type=Path,
        default=NATIVE_INSTALL,
        help="SDK root the C snippets compile against (default: the bootstrap native install)",
    )
    options, hub_arguments = parser.parse_known_args(arguments)
    if hub_arguments and not options.hub:
        parser.error(f"unrecognized arguments: {' '.join(hub_arguments)}")

    if options.check_snippets:
        return _check_snippets(
            options.python or _snippet_python(), options.sdk, options.require_cuda
        )
    if (PROJECT_ROOT / "tools/docs/build.py").is_file():
        import build

        if options.hub:
            _run([build.python(), HUB_BUILD, *hub_arguments])
            output = HUB_OUTPUT
        else:
            build.generate()
            output = GUIDE_OUTPUT
    else:
        if options.hub:
            parser.error("The internal documentation hub is not part of the public distribution")
        output = PROJECT_ROOT / "docs/html"
        if not (output / "index.html").is_file():
            raise SystemExit("Prebuilt documentation is missing from this source distribution")
    print(f"[docs] guide {output / 'index.html'}", flush=True)
    if options.serve:
        _serve(output, options.port)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as failure:
        sys.exit(failure.returncode)
