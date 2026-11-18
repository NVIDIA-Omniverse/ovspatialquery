# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check or bump the in-tree library version in python/pyproject.toml.

Releases inject their exact version at build time, so the source tree carries the
next development version (``X.Y.Z.dev0`` after a lower final ``vX.Y.Z`` tag) and a source
build never claims a released version. python/pyproject.toml is the only
hand-edited copy; the App's uv.lock records it for the editable sibling source,
and bump-version rewrites that entry with it.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = Path("python/pyproject.toml")
APP_ROOT = Path("apps/ovspatialqueryapp")
UV_LOCK = APP_ROOT / "uv.lock"

# The builder's accepted forms; see contract.artifact_version().
VERSION = re.compile(
    r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:(\.dev|rc)(0|[1-9][0-9]*))?"
)
# The same line CMakeLists.txt reads as the default SDK version.
VERSION_LINE = re.compile(r'^version = "([^"]+)"\r?$', re.MULTILINE)
LOCKED_EDITABLE = re.compile(
    r'^\[\[package\]\]\r?\nname = "ovspatialquery"\r?\nversion = "([^"]+)"\r?\n'
    r"source = \{ editable = ",
    re.MULTILINE,
)


def version_key(version: str) -> tuple[int, int, int, int, int] | None:
    """Order ``major.minor.patch[.devN|rcN]`` as PEP 440 does; None for any other form."""
    match = VERSION.fullmatch(version)
    if match is None:
        return None
    major, minor, patch, stage, number = match.groups()
    # Development builds precede release candidates, which precede the final release.
    return (
        int(major),
        int(minor),
        int(patch),
        {".dev": 0, "rc": 1, None: 2}[stage],
        int(number or 0),
    )


def source_version(root: Path) -> str:
    line = VERSION_LINE.search((root / PYPROJECT).read_text(encoding="utf-8"))
    return line.group(1) if line else ""


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def latest_release(root: Path) -> tuple[tuple[int, int, int, int, int] | None, str]:
    """Return the newest reachable final ``vX.Y.Z`` tag's key and name, or None and why it is unknown."""
    if not (root / ".git").exists():
        return None, "no Git history is available"
    try:
        shallow = _git(root, "rev-parse", "--is-shallow-repository") == "true"
        tags = _git(root, "tag", "--merged", "HEAD", "--list", "v*").split()
    except (OSError, subprocess.CalledProcessError):
        return None, "no Git history is available"
    if shallow:
        return None, "shallow clone (fetch full history and tags to check)"
    released = [
        (key, tag) for tag in tags if (key := version_key(tag[1:])) is not None and key[3] == 2
    ]
    if not released:
        return None, "no final vX.Y.Z release tag is reachable from HEAD"
    return max(released)


def check(root: Path) -> bool:
    """Print the outcome; False when the tree claims a released version or disagrees with uv.lock."""
    version = source_version(root)
    current = version_key(version)
    if current is None:
        print(
            f"[quality] version: {PYPROJECT} has unsupported version {version!r}", file=sys.stderr
        )
        return False
    ok = True
    if (root / APP_ROOT).exists():
        locked = LOCKED_EDITABLE.search((root / UV_LOCK).read_text(encoding="utf-8"))
        if locked is None or locked.group(1) != version:
            print(
                f"[quality] version: {UV_LOCK} records ovspatialquery "
                f"{locked.group(1) if locked else 'nowhere'}, but {PYPROJECT} says {version}. "
                f"Change the version with ./dev.sh bump-version, or set that entry to {version}; "
                "a full uv lock would drop the hand-locked ovRTX wheels.",
                file=sys.stderr,
            )
            ok = False
    latest_key, latest = latest_release(root)
    if latest_key is None:
        print(f"[quality] version: release-tag check skipped; {latest}")
    elif current <= latest_key:
        print(
            f"[quality] version: {version} in {PYPROJECT} is not greater than release tag "
            f"{latest} reachable from HEAD. After tagging, run ./dev.sh bump-version with the "
            "approved next development version.",
            file=sys.stderr,
        )
        ok = False
    else:
        print(f"[quality] version: {version} follows release {latest}")
    return ok


def next_version(current: str, part: str) -> str:
    """The next ``.dev0`` version: ``part`` is major, minor, patch or an explicit X.Y.Z."""
    if part not in ("major", "minor", "patch"):
        if VERSION.fullmatch(part) is None or version_key(part)[3] != 2:
            raise ValueError(f"Expected X.Y.Z, --patch, --minor or --major, not {part!r}")
        return f"{part}.dev0"
    key = version_key(current)
    if key is None:
        raise ValueError(f"{PYPROJECT} has unsupported version {current!r}")
    major, minor, patch = key[:3]
    if part == "major":
        return f"{major + 1}.0.0.dev0"
    if part == "minor":
        return f"{major}.{minor + 1}.0.dev0"
    return f"{major}.{minor}.{patch + 1}.dev0"


def _uv_lock_check(root: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        print(f"[bump-version] uv is not on PATH; {UV_LOCK} was updated but not checked with uv")
        return
    subprocess.run([uv, "lock", "--check"], cwd=root / APP_ROOT, check=True)


def bump(root: Path, part: str, lock_check=_uv_lock_check) -> str:
    """Write the next development version to pyproject and uv.lock; restore both on failure.

    Only the editable entry of uv.lock changes. A full ``uv lock`` would drop the
    hand-locked NVIDIA-index ovRTX wheels that bootstrap installs, so uv only checks it.
    """
    current = source_version(root)
    target = next_version(current, part)
    if version_key(target) <= version_key(current):
        raise ValueError(f"{target} is not greater than the current version {current}")
    latest_key, latest = latest_release(root)
    if latest_key is None:
        print(f"[bump-version] release-tag check skipped; {latest}")
    elif version_key(target) <= latest_key:
        raise ValueError(f"{target} is not greater than release tag {latest} reachable from HEAD")
    version_files = [(root / PYPROJECT, VERSION_LINE)]
    if (root / APP_ROOT).exists():
        version_files.append((root / UV_LOCK, LOCKED_EDITABLE))
    original = {path: path.read_bytes() for path, _ in version_files}
    edits = {}
    for path, pattern in version_files:
        text = original[path].decode("utf-8")
        match = pattern.search(text)
        if match is None:
            raise RuntimeError(f"{path} has no ovspatialquery version to replace")
        edits[path] = text[: match.start(1)] + target + text[match.end(1) :]
    try:
        for path, text in edits.items():
            path.write_bytes(text.encode("utf-8"))
        if (root / APP_ROOT).exists():
            lock_check(root)
    except BaseException:
        for path, payload in original.items():
            path.write_bytes(payload)
        raise
    return target


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bump-version",
        description="Set python/pyproject.toml and the App uv.lock to the next .dev0 version.",
    )
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("version", nargs="?", help="explicit next release X.Y.Z")
    for part in ("patch", "minor", "major"):
        choice.add_argument(
            f"--{part}", dest="part", action="store_const", const=part, help=f"next {part}"
        )
    options = parser.parse_args(arguments)
    try:
        target = bump(PROJECT_ROOT, options.version or options.part)
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as failure:
        print(f"[bump-version] {failure}", file=sys.stderr)
        return 1
    print(f"[bump-version] source version is now {target}.")
    print("[bump-version] Rerun bootstrap so the editable install records the new version.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
