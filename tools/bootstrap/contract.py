# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Locked dependency and native-build plan for the ovspatialquery bootstrap.

The locks are the checked-in files: ``apps/ovspatialqueryapp/uv.lock`` for
Python wheels, ``deps/target-deps.packman.xml`` for PhysX,
``deps/toolchain.packman.xml`` for CMake and the Linux clang, and
``deps/physx-content-lock.json`` and ``deps/compiler-content-lock.json`` for the hashes
of the fetched PhysX and clang files.  This module reads them; it does
not keep a second copy of anything they say.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tomllib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = PROJECT_ROOT / "python"
PYTHON_PROJECT = PYTHON_ROOT / "pyproject.toml"
APP_ROOT = PROJECT_ROOT / "apps" / "ovspatialqueryapp"
APP_PROJECT = APP_ROOT / "pyproject.toml"
APP_LOCK = APP_ROOT / "uv.lock"
PHYSX_MANIFEST = PROJECT_ROOT / "deps" / "target-deps.packman.xml"
TOOLCHAIN_MANIFEST = PROJECT_ROOT / "deps" / "toolchain.packman.xml"
PHYSX_CONTENT_LOCK = PROJECT_ROOT / "deps" / "physx-content-lock.json"
COMPILER_CONTENT_LOCK = PROJECT_ROOT / "deps" / "compiler-content-lock.json"
OVSTAGE_SDK_LOCK = PROJECT_ROOT / "deps" / "ovstage-sdk-lock.json"
OVSTAGE_SDK_ROOT = PROJECT_ROOT / "_build" / "ovstage-sdk"

EXPECTED_PYTHON = (3, 12)
DEFAULT_TARGET_PLATFORM = "windows-x86_64"
LINUX_X86_64_PLATFORM = "manylinux_2_35_x86_64"
# A native Linux aarch64 host builds with the host compiler; no compiler is pinned for it.
LINUX_AARCH64_PLATFORM = "manylinux_2_35_aarch64"
SUPPORTED_PLATFORMS = (DEFAULT_TARGET_PLATFORM, LINUX_X86_64_PLATFORM, LINUX_AARCH64_PLATFORM)


@dataclass(frozen=True)
class PlatformLayout:
    """The per-target-platform filenames and paths bootstrap tooling picks between."""

    venv_python: tuple[str, ...]
    cmake_executable: str
    ctest_executable: str
    library_dir: str
    library_file: str
    import_library: str | None
    wheel_platform_tag: str
    entrypoint_script: str
    packman_executable: str


PLATFORM_LAYOUTS: dict[str, PlatformLayout] = {
    DEFAULT_TARGET_PLATFORM: PlatformLayout(
        venv_python=("Scripts", "python.exe"),
        cmake_executable="cmake.exe",
        ctest_executable="ctest.exe",
        library_dir="bin",
        library_file="ovspatialquery.dll",
        import_library="ovspatialquery.lib",
        wheel_platform_tag="py3-none-win_amd64",
        entrypoint_script=".\\dev.ps1",
        packman_executable="packman.cmd",
    ),
    LINUX_X86_64_PLATFORM: PlatformLayout(
        venv_python=("bin", "python"),
        cmake_executable="cmake",
        ctest_executable="ctest",
        library_dir="lib",
        library_file="libovspatialquery.so",
        import_library=None,
        wheel_platform_tag="py3-none-manylinux_2_35_x86_64",
        entrypoint_script="./dev.sh",
        packman_executable="packman",
    ),
    LINUX_AARCH64_PLATFORM: PlatformLayout(
        venv_python=("bin", "python"),
        cmake_executable="cmake",
        ctest_executable="ctest",
        library_dir="lib",
        library_file="libovspatialquery.so",
        import_library=None,
        wheel_platform_tag="py3-none-manylinux_2_35_aarch64",
        entrypoint_script="./dev.sh",
        packman_executable="packman",
    ),
}


def platform_layout(target_platform: str) -> PlatformLayout:
    """Return the literal layout for ``target_platform``, or raise for an unsupported one."""

    try:
        return PLATFORM_LAYOUTS[target_platform]
    except KeyError:
        raise BootstrapError(
            f"Unsupported target platform {target_platform!r}; expected one of {SUPPORTED_PLATFORMS}",
            code="platform_unsupported",
            action="Select windows-x86_64, manylinux_2_35_x86_64 or manylinux_2_35_aarch64.",
        ) from None


LOCAL_EDITABLE_RUNTIME_PACKAGES = ("ovspatialquery",)
CMAKE_GENERATOR = "Visual Studio 17 2022"
CMAKE_ARCHITECTURE = "x64"
CMAKE_TOOLSET = "v143,host=x64"
LINUX_CMAKE_GENERATOR = "Ninja"

# Build tools installed into the venv before the runtime wheels.  pip installs
# them with --require-hashes, so these are the only pins that live here.
PIP_WHEEL = {
    "name": "pip",
    "version": "25.3",
    "sha256": "9655943313a94722b7774661c21049070f6bbb0a1516bf02f7c8d5d9201514cd",
    "url": (
        "https://files.pythonhosted.org/packages/44/3c/"
        "d717024885424591d5376220b5e836c2d5293ce2011523c9de23ff7bf068/"
        "pip-25.3-py3-none-any.whl"
    ),
}
SETUPTOOLS_WHEEL = {
    "name": "setuptools",
    "version": "83.0.0",
    "sha256": "29b23c360f22f414dc7336bb39178cc7bcbf6021ed2733cde173f09dba19abb3",
    "url": (
        "https://files.pythonhosted.org/packages/5d/40/"
        "e1e72872c6354b306daef1703549e8e83b4d43cfea356311bf722a043752/"
        "setuptools-83.0.0-py3-none-any.whl"
    ),
}
WHEEL_STUB_WHEEL = {
    "name": "wheel-stub",
    "version": "0.5.0",
    "sha256": "debb507277eb5793dcfa8e92f1f41ba4b93f7db05035725b5780cb6bcb444f3f",
    "url": (
        "https://files.pythonhosted.org/packages/06/bc/"
        "a04db4c1ca8991c38d9cc4004adaa38309d5e236568c77965a0ea61601a0/"
        "wheel_stub-0.5.0-py3-none-any.whl"
    ),
}
# ovrtx is published on PyPI as a wheel-stub sdist; the stub downloads the
# platform wheel from this index while pip installs it.
WHEEL_STUB_PACKAGES = ("ovrtx",)
WHEEL_STUB_INDEX_URL = "https://pypi.nvidia.com/"
VISUAL_EXAMPLE_RUNTIME_PACKAGES = frozenset({"ovrtx"})
# The numbered lessons also need Warp, which the Viewer does not import. Its pin and hashes
# are the guide programs'; pip installs it from the index, so the line names no wheel URL.
LESSON_ONLY_PACKAGES = ("warp-lang",)
LESSON_REQUIREMENTS = PROJECT_ROOT / "tools" / "docs" / "snippet-requirements.txt"
VIEWER_ONLY_RUNTIME_PACKAGES = frozenset(
    {
        "ovrtx",
        "usd-core",
        "pillow",
        "ovui",
        "ovui-data-adapters-common",
        "ovui-data-adapters-services",
        "ovui-widgets-common",
        "ovui-widgets-content",
        "ovui-widgets-stage",
        "ovui-widgets-viewport",
    }
)


class BootstrapError(RuntimeError):
    """A lock or plan input could not be read or is inconsistent."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "lock_invalid",
        action: str = "Restore or intentionally update the checked-in locks.",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.action = action


@dataclass(frozen=True)
class Blocker:
    code: str
    message: str
    action: str


def python_projection_version(project_file: Path = PYTHON_PROJECT) -> str:
    """The project.version of a pyproject file, the Python projection's by default."""
    with project_file.open("rb") as stream:
        project = tomllib.load(stream)
    version = str(project.get("project", {}).get("version", ""))
    if not version:
        raise BootstrapError(f"{project_file} has no project.version")
    return version


# The export writes the injected version of a candidate or final release here; the
# internal tree never carries it (docs/development/releases.md#source-version-stamp).
VERSION_STAMP = "VERSION.txt"
RELEASE_TAG = re.compile(
    r"v((?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)(?:rc(?:0|[1-9][0-9]*))?)"
)


def checked_version(version: str, origin: str) -> str:
    match = re.fullmatch(
        r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:(\.dev|rc)(0|[1-9][0-9]*))?", version
    )
    if match is None or any(int(part) > 4294967295 for part in match.groups()[:3]):
        raise BootstrapError(
            f"{origin} must be major.minor.patch, major.minor.patch.devN or major.minor.patchrcN "
            f"with uint32 components, not {version!r}"
        )
    return version


def source_version_stamp(root: Path = PROJECT_ROOT) -> str | None:
    """The validated version an exported source's VERSION.txt records, if it has one."""
    stamp = root / VERSION_STAMP
    if not stamp.is_file():
        return None
    return checked_version(stamp.read_bytes().decode("ascii", "replace").strip(), str(stamp))


def release_tag_version(root: Path = PROJECT_ROOT) -> str | None:
    """X.Y.Z[rcN] of a vX.Y.Z[rcN] tag at the HEAD of a clean checkout, the final if both."""
    # An extracted archive has no tags of its own; never adopt a containing checkout's.
    if not (root / ".git").exists():
        return None
    identity = source_identity(root)
    if identity is None or identity[1]:
        return None
    try:
        tags = subprocess.run(
            ["git", "tag", "--points-at", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        return None

    def order(version: str) -> tuple[int, ...]:
        # A final release may be tagged at its last candidate's commit; it sorts after rcN.
        core, _, candidate = version.partition("rc")
        return (*map(int, core.split(".")), not candidate, int(candidate or 0))

    versions = [match[1] for tag in tags if (match := RELEASE_TAG.fullmatch(tag))]
    return max(versions, key=order, default=None)


def version_selection(root: Path = PROJECT_ROOT) -> tuple[str, str]:
    """The artifact version and where it came from, in precedence order.

    ``override`` is OVSPATIALQUERY_BUILD_VERSION, ``stamp`` the exported VERSION.txt,
    ``tag`` a release tag at a clean HEAD and ``source`` the pyproject development version.
    """
    if version := os.environ.get("OVSPATIALQUERY_BUILD_VERSION"):
        return checked_version(version, "OVSPATIALQUERY_BUILD_VERSION"), "override"
    if version := source_version_stamp(root):
        return version, "stamp"
    if version := release_tag_version(root):
        return checked_version(version, f"Release tag v{version}"), "tag"
    project = root / "python/pyproject.toml"
    return checked_version(python_projection_version(project), str(project)), "source"


def artifact_version() -> str:
    return version_selection()[0]


def source_identity(root: Path = PROJECT_ROOT) -> tuple[str, bool] | None:
    """Return checkout HEAD/status or the commit stamped by git archive.

    Tracked modifications and untracked files that are not ignored both count;
    generated output under ``_build/`` is ignored. Archive identity describes the
    original snapshot; it cannot detect edits after extraction. Archive checksums
    establish the input qualified by release CI.
    """

    if not (root / ".git").exists():
        marker = root / ".git_archival.txt"
        revision = marker.read_text(encoding="ascii").strip() if marker.is_file() else ""
        # Never adopt a containing checkout's identity for an extracted archive.
        return (revision, False) if re.fullmatch(r"[0-9a-f]{40}", revision) else None

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()

    try:
        return git("rev-parse", "HEAD"), bool(
            git("status", "--porcelain", "--untracked-files=normal")
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def wheel_version(revision: str, dirty: bool) -> str:
    """artifact_version() with a PEP 440 local label naming the commit, ``+g<12 hex>[.dirty]``.

    An injected, stamped or tagged version stays exact. Only the wheel carries the label;
    the native getter, CMake and the editable install keep the unlabelled version.
    """

    version, origin = version_selection()
    if origin == "override":
        return release_wheel_version(version, revision)
    if origin != "source":
        return version
    return f"{version}+g{revision[:12]}" + (".dirty" if dirty else "")


def release_wheel_version(version: str, revision: str, branch: str | None = None) -> str:
    """Development bundles label wheels with their selected branch and source commit."""
    if branch is None:
        branch = os.environ.get("OVSPATIALQUERY_BUILD_BRANCH", "")
    if branch and ".dev" in version:
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise BootstrapError("Development bundle wheel requires a full source SHA")
        segments = re.findall(r"[a-z0-9]+", branch.lower())
        if not segments:
            raise BootstrapError("Development bundle branch has no alphanumeric label")
        # PEP 440 compares numeric local segments as integers; write them canonically.
        label = ".".join(str(int(s)) if s.isdigit() else s for s in segments)
        short_sha = revision[:12]
        for char in revision[12:]:
            if not short_sha.isdigit():
                break
            short_sha += char
        if short_sha.isdigit():
            raise BootstrapError("Development bundle source SHA has no hexadecimal letters")
        return f"{version}+{label}.{short_sha}"
    return version


def native_source_revision() -> str:
    """The commit compiled into the library: HEAD, -dirty with local changes, or unknown."""

    identity = source_identity()
    if identity is None:
        return "unknown"
    revision, dirty = identity
    return revision + ("-dirty" if dirty else "")


# OVSPATIALQUERY_TRAIN=next swaps the locked ovstage and ovrtx wheels for the
# in-development builds deps/next-train.json names, in separate venv and native
# trees, so the stable environment is never modified.
TRAIN_ENVIRONMENT = "OVSPATIALQUERY_TRAIN"
NEXT_TRAIN_RESOLUTION = PROJECT_ROOT / "_build" / "next-train" / "resolution.json"


def train() -> str:
    selected = os.environ.get(TRAIN_ENVIRONMENT) or "stable"
    if selected not in {"stable", "next"}:
        raise BootstrapError(
            f"Unsupported {TRAIN_ENVIRONMENT}={selected!r}",
            code="train_unsupported",
            action=f"Set {TRAIN_ENVIRONMENT} to stable (the default) or next.",
        )
    if selected != "stable" and not APP_PROJECT.is_file():
        raise BootstrapError(
            "This source distribution supports only the stable dependency train.",
            code="train_unsupported",
            action=f"Unset {TRAIN_ENVIRONMENT} or set it to stable.",
        )
    return selected


def next_train_resolution() -> dict[str, Any]:
    try:
        return json.loads(NEXT_TRAIN_RESOLUTION.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise BootstrapError(
            f"The next train has not been resolved: {NEXT_TRAIN_RESOLUTION} is missing",
            code="next_train_unresolved",
            action=f"Run {TRAIN_ENVIRONMENT}=next ./dev.sh bootstrap first.",
        ) from None


def _default_venv() -> Path:
    suffix = "-next" if train() == "next" else ""
    return PROJECT_ROOT / "_build" / "venv" / f"ovspatialqueryapp{suffix}"


def _default_native_build() -> Path:
    return PROJECT_ROOT / "_build" / "bootstrap" / f"native-{train()}"


def public_headers() -> tuple[str, ...]:
    """The installed public headers, read from the one CMake list that installs them."""

    text = (PROJECT_ROOT / "cmake" / "package" / "public_headers.cmake").read_text(encoding="utf-8")
    match = re.search(r"set\(OVSPATIALQUERY_PUBLIC_HEADERS\s+([^)]*)\)", text)
    if match is None:
        raise ValueError(
            "cmake/package/public_headers.cmake does not define OVSPATIALQUERY_PUBLIC_HEADERS"
        )
    return tuple(match.group(1).split())


OVSTAGE_PUBLIC_HEADER = "ovspatialquery_ovstage.h"


def _default_target_deps() -> Path:
    return PROJECT_ROOT / "_build" / "target-deps"


def _default_packman_cache() -> Path:
    if os.name == "nt":
        return Path(PROJECT_ROOT.anchor) / "packman-repo"
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_root / "packman"


def _selected_packman_cache() -> Path:
    override = os.environ.get("PM_PACKAGES_ROOT")
    return Path(override).resolve() if override else _default_packman_cache().resolve()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _wheel_matches_platform(filename: str, target_platform: str) -> bool:
    lower = filename.lower()
    if lower.endswith("-py3-none-any.whl"):
        return True
    if target_platform == DEFAULT_TARGET_PLATFORM:
        return lower.endswith("win_amd64.whl") and ("-cp312-" in lower or "-py3-none-" in lower)
    if target_platform == LINUX_X86_64_PLATFORM:
        return (
            "manylinux" in lower
            and lower.endswith("x86_64.whl")
            and ("-cp312-" in lower or "-py3-none-" in lower)
        )
    if target_platform == LINUX_AARCH64_PLATFORM:
        return (
            "manylinux" in lower
            and lower.endswith("aarch64.whl")
            and ("-cp312-" in lower or "-py3-none-" in lower)
        )
    return False


_MANYLINUX_TAG = re.compile(r"manylinux(?:_(\d+)_(\d+)|(1|2010|2014))_")
_MANYLINUX_LEGACY_GLIBC = {"1": (2, 5), "2010": (2, 12), "2014": (2, 17)}


def _manylinux_floor(filename: str) -> tuple[int, int]:
    """Return the highest glibc floor advertised by a wheel's manylinux tags.

    A single wheel may carry several compatible tags (for example
    ``manylinux2014_x86_64.manylinux_2_17_x86_64``). Installers select the most
    specific compatible wheel, so ranking by the highest advertised floor
    reproduces what pip and uv actually install.
    """

    floors = [
        _MANYLINUX_LEGACY_GLIBC[legacy] if legacy else (int(major), int(minor))
        for major, minor, legacy in _MANYLINUX_TAG.findall(filename.lower())
    ]
    return max(floors) if floors else (0, 0)


def _select_platform_wheel(
    name: str, candidates: list[dict[str, Any]], target_platform: str
) -> dict[str, Any]:
    """Choose the single wheel an installer would use for the target platform."""

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise BootstrapError(
            f"Expected one CPython 3.12 {target_platform} or universal wheel for {name}, found 0"
        )
    ranked = sorted(
        candidates,
        key=lambda wheel: _manylinux_floor(Path(str(wheel.get("url", ""))).name),
        reverse=True,
    )
    best = _manylinux_floor(Path(str(ranked[0].get("url", ""))).name)
    tied = [
        wheel for wheel in ranked if _manylinux_floor(Path(str(wheel.get("url", ""))).name) == best
    ]
    if best == (0, 0) or len(tied) != 1:
        raise BootstrapError(
            f"Locked wheels for {name} are ambiguous on {target_platform}: "
            + ", ".join(sorted(Path(str(wheel.get("url", ""))).name for wheel in candidates))
        )
    return tied[0]


def _sha256_digest(entry: dict[str, Any], what: str) -> str:
    digest = str(entry.get("hash", ""))
    if not digest.startswith("sha256:") or len(digest) != len("sha256:") + 64:
        raise BootstrapError(f"Locked {what} has no valid SHA-256 digest")
    return digest.removeprefix("sha256:")


def _physx_sdk_root(target_deps: Path) -> Path:
    override = os.environ.get("OVSPATIALQUERY_PHYSX_SDK_DIR")
    return Path(override).expanduser().resolve() if override else target_deps / "physx"


def _staged_physx_issue(
    target_deps: Path, target_platform: str = DEFAULT_TARGET_PLATFORM
) -> str | None:
    header = _physx_sdk_root(target_deps) / "include" / "PxPhysicsAPI.h"
    if not header.is_file():
        return f"Selected PhysX SDK header was not found at {header}"
    if os.environ.get("OVSPATIALQUERY_PHYSX_SDK_DIR"):
        return None  # CMake validates the selected custom SDK's layout and license.
    return _physx_content_issue(target_deps / "physx", target_platform)


def _content_lock_issue(
    lock_path: Path, what: str, package_root: Path, target_platform: str
) -> str | None:
    """Compare a fetched package's files against the SHA-256 digests in a content lock."""

    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"{what} content lock is unreadable: {exc}"
    platform_lock = lock.get("platforms", {}).get(target_platform)
    if not isinstance(platform_lock, dict) or not isinstance(platform_lock.get("files"), dict):
        return f"{what} content lock has no file entries for {target_platform}"
    for relative, expected_digest in platform_lock["files"].items():
        path = package_root / str(relative)
        if not path.is_file():
            return f"{what} content lock lists a missing file {path}"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_digest:
            return f"{what} content mismatch for {relative}: expected {expected_digest}, found {digest}"
    return None


def _physx_content_issue(package_root: Path, target_platform: str) -> str | None:
    """Compare the fetched PhysX files against deps/physx-content-lock.json."""

    return _content_lock_issue(PHYSX_CONTENT_LOCK, "PhysX", package_root, target_platform)


def _project_dependency_versions() -> dict[str, str]:
    """Return the exact ``name == version`` pins from the app pyproject.

    Local editable packages are unpinned: their sibling source owns the version.
    """

    with APP_PROJECT.open("rb") as stream:
        project = tomllib.load(stream)
    versions: dict[str, str] = {}
    for requirement in project.get("project", {}).get("dependencies", []):
        name, pin, version = str(requirement).partition("==")
        name = name.strip().lower()
        if name in LOCAL_EDITABLE_RUNTIME_PACKAGES:
            continue
        if not pin:
            raise BootstrapError(f"Runtime dependency must use an exact == pin: {requirement}")
        versions[name] = version.strip()
    return versions


def _locked_runtime_wheels(
    target_platform: str = DEFAULT_TARGET_PLATFORM,
    *,
    include_viewer: bool = True,
) -> list[dict[str, Any]]:
    """Select, from uv.lock, the one wheel per runtime dependency for the platform."""

    platform_layout(target_platform)
    library_lock = PROJECT_ROOT / "deps" / "library-runtime-lock.json"
    if not include_viewer and not APP_PROJECT.is_file():
        train()
        try:
            with library_lock.open(encoding="utf-8") as stream:
                return json.load(stream)[target_platform]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise BootstrapError(f"Invalid library dependency lock: {exc}") from exc
    project_versions = _project_dependency_versions()
    with APP_LOCK.open("rb") as stream:
        lock = tomllib.load(stream)
    packages = {str(item["name"]): item for item in lock.get("package", [])}
    wheels: list[dict[str, Any]] = []
    for name, pinned_version in project_versions.items():
        if not include_viewer and name in (
            VIEWER_ONLY_RUNTIME_PACKAGES | VISUAL_EXAMPLE_RUNTIME_PACKAGES
        ):
            continue
        package = packages.get(name)
        if package is None:
            raise BootstrapError(f"{APP_LOCK} has no package entry for {name}")
        if str(package["version"]) != pinned_version:
            raise BootstrapError(
                f"pyproject/uv.lock version mismatch for {name}: "
                f"{pinned_version} != {package['version']}"
            )
        candidates = [
            wheel
            for wheel in package.get("wheels", [])
            if _wheel_matches_platform(Path(str(wheel.get("url", ""))).name, target_platform)
        ]
        wheel = _select_platform_wheel(name, candidates, target_platform)
        url = str(wheel["url"])
        entry: dict[str, Any] = {
            "name": name,
            "version": str(package["version"]),
            "filename": Path(url).name,
            "sha256": _sha256_digest(wheel, f"wheel for {name}"),
            "url": url,
        }
        if name in WHEEL_STUB_PACKAGES:
            sdist = package.get("sdist")
            if not isinstance(sdist, dict):
                raise BootstrapError(f"{APP_LOCK} has no sdist entry for wheel-stub package {name}")
            entry["sdist"] = {
                "filename": Path(str(sdist["url"])).name,
                "sha256": _sha256_digest(sdist, f"sdist for {name}"),
                "url": str(sdist["url"]),
            }
        wheels.append(entry)
    if train() == "next":
        # Already selected for its platform and hash-pinned by next_train.resolve.
        resolved = next_train_resolution()
        if resolved["target_platform"] != target_platform:
            raise BootstrapError(
                f"The next train was resolved for {resolved['target_platform']}, "
                f"not {target_platform}",
                code="next_train_unresolved",
                action=f"Rerun bootstrap with {TRAIN_ENVIRONMENT}=next on this platform.",
            )
        wheels = [resolved["wheels"].get(entry["name"], entry) for entry in wheels]
    return wheels


def lesson_requirements(path: Path = LESSON_REQUIREMENTS) -> dict[str, tuple[str, str]]:
    """Return each lesson-only package's version and hash-pinned requirement line."""

    pins: dict[str, tuple[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        name, pin, rest = line.partition("==")
        if pin and name.strip() in LESSON_ONLY_PACKAGES:
            version = rest.split(maxsplit=1)[0]
            if "--hash=sha256:" not in rest:
                raise BootstrapError(f"{path.name} pins {name} without a hash")
            pins[name.strip()] = (version, line.strip())
    missing = set(LESSON_ONLY_PACKAGES) - pins.keys()
    if missing:
        raise BootstrapError(f"{path.name} does not pin {sorted(missing)}")
    return pins


def _build_tools(
    target_platform: str = DEFAULT_TARGET_PLATFORM,
    *,
    include_viewer: bool = True,
    include_cuda: bool = False,
) -> list[dict[str, Any]]:
    tools = []
    for pinned in (PIP_WHEEL, SETUPTOOLS_WHEEL, WHEEL_STUB_WHEEL):
        if not include_viewer and pinned["name"] == "wheel-stub":
            continue
        tool = dict(pinned)
        tool["filename"] = Path(str(tool["url"])).name
        tools.append(tool)
    if include_cuda and not include_viewer:
        with (PROJECT_ROOT / "deps" / "cuda-nvrtc-lock.json").open() as stream:
            tools.append(json.load(stream)[target_platform])
    return tools


def _packaged_physx_version(target_platform: str = DEFAULT_TARGET_PLATFORM) -> str:
    root = ET.parse(PHYSX_MANIFEST).getroot()
    versions = [
        str(package.attrib["version"])
        for package in root.findall("./dependency/package")
        if target_platform in package.attrib.get("platforms", "").split()
    ]
    if len(versions) != 1:
        raise BootstrapError(
            f"Expected one {target_platform} packaged PhysX coordinate in {PHYSX_MANIFEST}, found {len(versions)}"
        )
    return versions[0]


def _packaged_tool(dependency: str, target_platform: str) -> tuple[str, str]:
    root = ET.parse(TOOLCHAIN_MANIFEST).getroot()
    packages = [
        (str(package.attrib["name"]), str(package.attrib["version"]))
        for package in root.findall(f"./dependency[@name='{dependency}']/package")
        if target_platform in package.attrib.get("platforms", "").split()
    ]
    if len(packages) != 1:
        raise BootstrapError(
            f"Expected one {target_platform} {dependency} coordinate in {TOOLCHAIN_MANIFEST}, found {len(packages)}"
        )
    return packages[0]


def _cmake_executable(target_deps: Path, target_platform: str) -> Path:
    filename = platform_layout(target_platform).cmake_executable
    return target_deps / "cmake" / "bin" / filename


def _staged_cmake_issue(target_deps: Path, target_platform: str) -> str | None:
    executable = _cmake_executable(target_deps, target_platform)
    if not executable.is_file():
        return f"Pinned Packman CMake was not found at {executable}"
    return None


def compiler_selection() -> str:
    """`pinned` (the default) builds with the Packman compiler in deps/toolchain.packman.xml, so
    builds and benchmarks reproduce; `host` builds with the compiler the host provides (CMake's
    choice, or CC/CXX on Linux)."""
    selection = os.environ.get("OVSPATIALQUERY_COMPILER", "pinned")
    if selection not in {"pinned", "host"}:
        raise BootstrapError(
            f"Unsupported compiler selection: {selection}",
            code="native_compiler_unsupported",
            action="Select pinned (the default) or host through OVSPATIALQUERY_COMPILER.",
        )
    return selection


def _require_compiler(target_platform: str) -> None:
    """No compiler is pinned for Linux aarch64; its builds must select the host compiler."""
    if target_platform == LINUX_AARCH64_PLATFORM and compiler_selection() == "pinned":
        raise BootstrapError(
            f"No pinned compiler is locked for {target_platform}",
            code="native_compiler_unavailable",
            action="Set OVSPATIALQUERY_COMPILER=host to build with the host C/C++ compiler.",
        )


def _pinned_clang(target_deps: Path) -> tuple[Path, Path]:
    return target_deps / "clang" / "bin" / "clang", target_deps / "clang" / "bin" / "clang++"


def _windows_generator_arguments(target_deps: Path, configuration: str) -> list[str]:
    """Generator and compiler arguments for a Windows configure.

    The pinned MSVC builds through Ninja with `cl` found through compiler_environment(); the host
    compiler keeps the Visual Studio generator with the installed toolset."""
    if compiler_selection() == "host":
        return ["-G", CMAKE_GENERATOR, "-A", CMAKE_ARCHITECTURE, "-T", CMAKE_TOOLSET]
    return [
        "-G",
        LINUX_CMAKE_GENERATOR,
        f"-DCMAKE_MAKE_PROGRAM={target_deps / 'ninja' / 'ninja.exe'}",
        f"-DCMAKE_BUILD_TYPE={configuration}",
        "-DCMAKE_C_COMPILER=cl",
        "-DCMAKE_CXX_COMPILER=cl",
    ]


def compiler_arguments(target_deps: Path, target_platform: str) -> list[str]:
    """CMake arguments that select the compiler for a native configure on this platform."""
    if target_platform == DEFAULT_TARGET_PLATFORM:
        return _windows_generator_arguments(target_deps, "Release")
    _require_compiler(target_platform)
    if compiler_selection() == "host":
        return []
    c_compiler, cxx_compiler = _pinned_clang(target_deps)
    return [f"-DCMAKE_C_COMPILER={c_compiler}", f"-DCMAKE_CXX_COMPILER={cxx_compiler}"]


def _pinned_msvc_layout(target_deps: Path) -> dict[str, Path] | str:
    """The staged MSVC tools and Windows SDK directories, or why they are unusable."""
    tools = sorted((target_deps / "msvc" / "VC" / "Tools" / "MSVC").glob("*"))
    if not tools:
        return f"Pinned Packman MSVC tools were not found under {target_deps / 'msvc'}"
    # The pinned winsdk package is flat: Include/ucrt, Lib/um/x64 and bin/x64, no version folder.
    sdk = target_deps / "winsdk"
    include, library, binaries = sdk / "Include", sdk / "Lib", sdk / "bin" / "x64"
    layout = {
        "tools": tools[-1],
        "sdk_include": include,
        "sdk_lib": library,
        "sdk_bin": binaries,
        "ninja": target_deps / "ninja" / "ninja.exe",
    }
    for required in (
        layout["tools"] / "bin" / "Hostx64" / "x64" / "cl.exe",
        include / "ucrt",
        library / "um" / "x64",
        binaries / "rc.exe",
        layout["ninja"],
    ):
        if not required.exists():
            return f"Pinned Windows toolchain is incomplete: {required} is missing"
    return layout


def compiler_environment(target_deps: Path, target_platform: str) -> dict[str, str]:
    """INCLUDE, LIB and PATH for the pinned MSVC; empty where the compiler needs no environment."""
    if target_platform != DEFAULT_TARGET_PLATFORM or compiler_selection() == "host":
        return {}
    layout = _pinned_msvc_layout(target_deps)
    if isinstance(layout, str):
        raise BootstrapError(
            layout,
            code="msvc_missing",
            action="Run dev.ps1 bootstrap to stage the pinned Packman MSVC, Windows SDK and Ninja, "
            "or select an installed Visual Studio with OVSPATIALQUERY_COMPILER=host.",
        )
    tools, include, library = layout["tools"], layout["sdk_include"], layout["sdk_lib"]
    return {
        "INCLUDE": os.pathsep.join(
            str(path)
            for path in (tools / "include", include / "ucrt", include / "um", include / "shared")
        ),
        "LIB": os.pathsep.join(
            str(path)
            for path in (tools / "lib" / "x64", library / "ucrt" / "x64", library / "um" / "x64")
        ),
        "PATH": os.pathsep.join(
            [
                str(tools / "bin" / "Hostx64" / "x64"),
                str(layout["sdk_bin"]),
                str(layout["ninja"].parent),
                os.environ.get("PATH", ""),
            ]
        ),
    }


def apply_compiler_environment(target_deps: Path, target_platform: str) -> None:
    """Export the pinned compiler's environment to this process, so every configure, build, test
    and nested consumer build it starts uses the same compiler."""
    environment = compiler_environment(target_deps, target_platform)
    if environment and os.environ.get("OVSPATIALQUERY_PINNED_MSVC") != str(target_deps):
        os.environ.update(environment)
        os.environ["OVSPATIALQUERY_PINNED_MSVC"] = str(target_deps)


def _staged_compiler_issue(target_deps: Path, target_platform: str) -> str | None:
    """Why the pinned compiler for this platform is not staged and intact, if it is not."""
    _require_compiler(target_platform)
    if compiler_selection() == "host":
        return None
    if target_platform == DEFAULT_TARGET_PLATFORM:
        layout = _pinned_msvc_layout(target_deps)
        return layout if isinstance(layout, str) else None
    locked = (target_deps / "clang" / "bin" / "clang-14").resolve()
    # The C and C++ drivers CMake selects both link to the one locked binary.
    for executable in _pinned_clang(target_deps):
        if not executable.is_file():
            return f"Pinned Packman clang was not found at {executable}"
        if executable.resolve() != locked:
            return f"Pinned Packman {executable.name} does not resolve to the locked bin/clang-14"
    return _content_lock_issue(
        COMPILER_CONTENT_LOCK, "Pinned clang", target_deps / "clang", target_platform
    )


def _target_metadata(target_platform: str) -> tuple[dict[str, Any], dict[str, str]]:
    package_name, cmake_version = _packaged_tool("cmake", target_platform)
    if target_platform == DEFAULT_TARGET_PLATFORM:
        return (
            {
                "operating_system": "Windows",
                "architecture": "AMD64",
                "python": "CPython 3.12 (64-bit)",
            },
            {
                "cmake": f"Packman {package_name}@{cmake_version}",
                "architecture": CMAKE_ARCHITECTURE,
                **(
                    {
                        "compiler": "installed MSVC (OVSPATIALQUERY_COMPILER=host)",
                        "generator": CMAKE_GENERATOR,
                        "toolset": CMAKE_TOOLSET,
                        "windows_sdk": "required by the selected MSVC toolchain",
                    }
                    if compiler_selection() == "host"
                    else {
                        "compiler": "Packman {}@{}".format(
                            *_packaged_tool("msvc", target_platform)
                        ),
                        "generator": LINUX_CMAKE_GENERATOR,
                        "windows_sdk": "Packman {}@{}".format(
                            *_packaged_tool("winsdk", target_platform)
                        ),
                    }
                ),
            },
        )
    _require_compiler(target_platform)
    architecture = "aarch64" if target_platform == LINUX_AARCH64_PLATFORM else "x86_64"
    return (
        {
            "operating_system": "Linux",
            "architecture": architecture,
            "python": "CPython 3.12 (64-bit)",
            "display_server": "required by the interactive viewer",
        },
        {
            "cmake": f"Packman {package_name}@{cmake_version}",
            "architecture": architecture,
            "compiler": (
                "host C/C++ compiler (OVSPATIALQUERY_COMPILER=host; CMake's choice or CC/CXX)"
                if compiler_selection() == "host"
                else "Packman {}@{} with the host GCC C++ runtime and binutils".format(
                    *_packaged_tool("clang", target_platform)
                )
            ),
            "generator": LINUX_CMAKE_GENERATOR,
            "ninja": "required on PATH",
        },
    )


def native_lifecycle(
    native_build: Path,
    target_deps: Path,
    target_platform: str,
    *,
    cmake: str | Path = "cmake",
    python: str | Path = "python",
    include_viewer: bool = True,
    include_cuda: bool = True,
) -> dict[str, list[str]]:
    configuration = os.environ.get("OVSPATIALQUERY_NATIVE_CONFIG", "Release")
    if configuration not in {"Release", "Debug"}:
        raise BootstrapError(
            f"Unsupported native configuration: {configuration}",
            code="native_configuration_unsupported",
            action="Select Release or Debug through OVSPATIALQUERY_NATIVE_CONFIG.",
        )
    cmake_command = str(cmake)
    ctest_command = str(
        Path(cmake_command).with_name(platform_layout(target_platform).ctest_executable)
    )
    configure = [
        cmake_command,
        "-S",
        str(PROJECT_ROOT),
        "-B",
        str(native_build),
        f"-DOVSPATIALQUERY_VERSION={artifact_version()}",
        f"-DOVSPATIALQUERY_SOURCE_REVISION={native_source_revision()}",
        "-DOVSPATIALQUERY_PREBUILT_CORE_INDEPENDENCE=OFF",
        "-DCMAKE_SUPPRESS_REGENERATION=" + os.environ.get("CMAKE_SUPPRESS_REGENERATION", "OFF"),
        f"-DOVSPATIALQUERY_COMPILER_SELECTION={compiler_selection()}",
    ]
    # Bare --parallel overrides CMake's environment limit with the tool default.
    parallel = [] if "CMAKE_BUILD_PARALLEL_LEVEL" in os.environ else ["--parallel"]
    # Do not resolve: a venv interpreter is a symlink to its base interpreter,
    # and following it would point discovery at the base prefix.
    venv = Path(python).parents[1]
    configure.extend(
        [
            "-DOVSPATIALQUERY_BUILD_OVSTAGE_COMPANION=ON",
            f"-DOVSPATIALQUERY_OVSTAGE_SDK_DIR={ovstage_sdk_dir(venv, target_platform)}",
        ]
    )
    cuda_nvrtc = _cuda_nvrtc_configuration(
        venv, target_platform, include_viewer=include_viewer, include_cuda=include_cuda
    )
    configure.append(
        "-DOVSPATIALQUERY_CUDA_NVRTC_DIR=" + (str(cuda_nvrtc) if cuda_nvrtc is not None else "")
    )
    windows = target_platform == DEFAULT_TARGET_PLATFORM
    # Visual Studio is a multi-configuration generator; Ninja takes CMAKE_BUILD_TYPE.
    build_configuration = ["--config", configuration] if windows else []
    configure.extend(
        [
            *(
                _windows_generator_arguments(target_deps, configuration)
                if windows
                else [
                    "-G",
                    LINUX_CMAKE_GENERATOR,
                    f"-DCMAKE_BUILD_TYPE={configuration}",
                    *compiler_arguments(target_deps, target_platform),
                ]
            ),
            "-DOVSPATIALQUERY_FETCH_DEPS=OFF",
            f"-DOVSPATIALQUERY_TARGET_DEPS={target_deps}",
            "-DOVSPATIALQUERY_PHYSX_SDK_DIR="
            + (
                str(_physx_sdk_root(target_deps))
                if os.environ.get("OVSPATIALQUERY_PHYSX_SDK_DIR")
                else ""
            ),
            f"-DOVSPATIALQUERY_PYTHON_EXECUTABLE={python}",
            f"-DOVSPATIALQUERY_APP_ARTIFACT_DIR={native_build / 'install'}",
        ]
    )
    return {
        "configure": configure,
        "build": [cmake_command, "--build", str(native_build), *build_configuration, *parallel],
        "test": [
            ctest_command,
            "--test-dir",
            str(native_build),
            *(["-C", configuration] if windows else []),
            "--interactive-debug-mode",
            "0",
            "--output-on-failure",
        ],
        "install": [
            cmake_command,
            "--build",
            str(native_build),
            *build_configuration,
            "--target",
            "ovspatialquery_verify_install",
            *parallel,
        ],
    }


def ovstage_sdk_entry(target_platform: str) -> dict[str, Any]:
    """The pinned Ovstage SDK wheel for ``target_platform``, with the lock's version."""
    lock = json.loads(OVSTAGE_SDK_LOCK.read_text(encoding="utf-8"))
    try:
        entry = lock["platforms"][target_platform]
    except KeyError:
        raise BootstrapError(
            f"{OVSTAGE_SDK_LOCK.name} pins no Ovstage SDK for {target_platform}",
            code="ovstage_sdk_unpinned",
            action=f"Add a {target_platform} wheel to {OVSTAGE_SDK_LOCK.name}.",
        ) from None
    return entry


def ovstage_sdk_dir(venv: Path, target_platform: str) -> Path:
    """The Ovstage package the companion builds against: the pinned SDK (ADR-0069), or on
    the next train the venv's newer nightly. bootstrap stages the pinned one."""
    if train() == "next":
        return _site_packages(venv, target_platform) / "ovstage"
    return pinned_ovstage_sdk_dir(target_platform)


def pinned_ovstage_sdk_dir(target_platform: str) -> Path:
    return (
        OVSTAGE_SDK_ROOT
        / ovstage_sdk_entry(target_platform)["version"]
        / target_platform
        / "ovstage"
    )


def _site_packages(venv: Path, target_platform: str) -> Path:
    """Return the venv site-packages layout for the target, not for the host.

    Plans are statically routed per platform, so this must not consult os.name.
    """

    if target_platform == DEFAULT_TARGET_PLATFORM:
        return venv / "Lib" / "site-packages"
    return venv / "lib" / f"python3.{EXPECTED_PYTHON[1]}" / "site-packages"


def _cuda_nvrtc_configuration(
    venv: Path,
    target_platform: str,
    *,
    include_viewer: bool,
    include_cuda: bool,
) -> Path | None:
    """Select the published build-time compiler without shipping it in the library."""
    if not include_cuda:
        return None
    site = _site_packages(venv, target_platform)
    if include_viewer:
        return site / "ovrtx" / "bin" / "plugins" / "rtx"
    directory = "bin" if target_platform == DEFAULT_TARGET_PLATFORM else "lib"
    return site / "nvidia" / "cuda_nvrtc" / directory


def build_plan(
    venv: Path,
    native_build: Path,
    target_deps: Path,
    target_platform: str = DEFAULT_TARGET_PLATFORM,
    *,
    include_viewer: bool = True,
    include_cuda: bool = False,
) -> dict[str, Any]:
    platform_layout(target_platform)
    host, toolchain = _target_metadata(target_platform)
    cmake = _cmake_executable(target_deps, target_platform)
    return {
        "schema_version": 1,
        "target_platform": target_platform,
        "host": host,
        "python": {
            "application": str(APP_ROOT),
            "lock": str(APP_LOCK),
            "venv": str(venv.resolve()),
            "runtime_wheels": _locked_runtime_wheels(
                target_platform, include_viewer=include_viewer
            ),
            "build_tools": _build_tools(
                target_platform, include_viewer=include_viewer, include_cuda=include_cuda
            ),
            "install_viewer": include_viewer,
            "wheel_stub_index": WHEEL_STUB_INDEX_URL,
        },
        "native": {
            "packman_config": str(PROJECT_ROOT / "tools" / "packman" / "config.packman.xml"),
            "physx_manifest": str(PHYSX_MANIFEST.resolve()),
            "physx_package": "custom"
            if os.environ.get("OVSPATIALQUERY_PHYSX_SDK_DIR")
            else _packaged_physx_version(target_platform),
            "physx_sdk": str(_physx_sdk_root(target_deps)),
            "physx_content_lock": str(PHYSX_CONTENT_LOCK.resolve()),
            "toolchain_manifest": str(TOOLCHAIN_MANIFEST.resolve()),
            "target_deps": str(target_deps.resolve()),
            "build": str(native_build.resolve()),
            "install": str((native_build / "install").resolve()),
            "toolchain": toolchain,
            "lifecycle": native_lifecycle(
                native_build,
                target_deps,
                target_platform,
                cmake=cmake,
                python=_target_venv_python(venv, target_platform),
                include_viewer=include_viewer,
                include_cuda=include_cuda,
            ),
            "packman_cache": {
                "default": str(_default_packman_cache().resolve()),
                "environment_override": "PM_PACKAGES_ROOT",
            },
        },
    }


def _target_venv_python(venv: Path, target_platform: str) -> Path:
    """Return the venv interpreter layout for the selected target."""

    return venv.joinpath(*platform_layout(target_platform).venv_python)


def _venv_python(venv: Path) -> Path:
    """Return the venv interpreter layout used for execution on this host."""

    return _target_venv_python(venv, host_platform())


def host_platform() -> str:
    """The target platform of a native build on this host."""

    if os.name == "nt":
        return DEFAULT_TARGET_PLATFORM
    return LINUX_AARCH64_PLATFORM if os.uname().machine == "aarch64" else LINUX_X86_64_PLATFORM


def verify_environment(
    venv: Path,
    native_build: Path,
    target_deps: Path,
    target_platform: str = DEFAULT_TARGET_PLATFORM,
) -> list[Blocker]:
    """Check that the venv, staged dependencies, and native build exist."""

    entrypoint = platform_layout(target_platform).entrypoint_script
    blockers: list[Blocker] = []
    venv_python = _venv_python(venv)

    if not venv_python.is_file():
        blockers.append(
            Blocker(
                "venv_missing",
                f"Worktree-local Python environment was not found at {venv}",
                f"Run {entrypoint} bootstrap to create the worktree-local Python environment.",
            )
        )
    physx_issue = _staged_physx_issue(target_deps, target_platform)
    if physx_issue is not None:
        header = _physx_sdk_root(target_deps) / "include" / "PxPhysicsAPI.h"
        blockers.append(
            Blocker(
                "physx_missing" if not header.is_file() else "physx_content_invalid",
                physx_issue,
                f"Run {entrypoint} bootstrap to fetch the pinned packaged PhysX SDK with Packman.",
            )
        )
    cmake_issue = _staged_cmake_issue(target_deps, target_platform)
    if cmake_issue is not None:
        blockers.append(
            Blocker(
                "cmake_missing",
                cmake_issue,
                f"Run {entrypoint} bootstrap to stage the pinned Packman CMake tool.",
            )
        )
    compiler_issue = _staged_compiler_issue(target_deps, target_platform)
    if compiler_issue is not None:
        blockers.append(
            Blocker(
                "compiler_missing",
                compiler_issue,
                f"Run {entrypoint} bootstrap to stage the pinned Packman compiler, or select the "
                "host compiler with OVSPATIALQUERY_COMPILER=host.",
            )
        )
    if not (native_build / "CMakeCache.txt").is_file():
        blockers.append(
            Blocker(
                "native_build_missing",
                f"Native CMake build was not found at {native_build}",
                f"Run {entrypoint} bootstrap to configure and build the native ovspatialquery targets.",
            )
        )
    return blockers
