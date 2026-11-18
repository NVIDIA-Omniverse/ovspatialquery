# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap, build, test, and run ovspatialquery from the checked-in locks.

Commands: plan, bootstrap, configure, build, test, app, verify, python-wheel,
clean-bootstrap.  ``dev.ps1`` and ``dev.sh`` select the interpreter and call
this script directly.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import contract
import consumer_content
import python_wheel


def _fail(code: str, message: str, action: str) -> contract.BootstrapError:
    return contract.BootstrapError(message, code=code, action=action)


# Lines of a failed captured command's output kept in its error message.
_FAILURE_OUTPUT_LINES = 200


def _run(
    command: Sequence[str | Path],
    *,
    code: str,
    action: str,
    capture: bool = False,
    display: str | None = None,
    environment: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    rendered = [str(item) for item in command]
    shown = display or subprocess.list2cmdline(rendered)
    print(f"[bootstrap] + {shown}", file=sys.stderr, flush=True)
    try:
        result = subprocess.run(
            rendered,
            cwd=contract.PROJECT_ROOT if cwd is None else cwd,
            check=False,
            capture_output=capture,
            text=True,
            # Windows: os.environ already folds duplicate-case keys (Path/PATH) to one;
            # a child inheriting the raw block (env=None) would see both.
            env=dict(os.environ) if environment is None else environment,
        )
    except OSError as exc:
        raise _fail(code, f"Could not execute {rendered[0]}: {exc}", action) from exc
    if result.returncode != 0:
        detail = ""
        if capture:
            # Keep both streams: CTest names the failed tests on stdout and reports only
            # "Errors while running CTest" on stderr.
            lines = "\n".join(
                part.strip() for part in (result.stdout, result.stderr) if part and part.strip()
            )
            if lines:
                detail = "\n" + "\n".join(lines.splitlines()[-_FAILURE_OUTPUT_LINES:])
        raise _fail(
            code,
            f"Command failed with exit code {result.returncode}: {shown}{detail}",
            action,
        )
    return result


def _entrypoint(target_platform: str) -> str:
    return contract.platform_layout(target_platform).entrypoint_script


def _validate_host(target_platform: str) -> None:
    # The public source export omits the app's uv.lock, and its library lock's aarch64
    # entry holds only the cross-builder's Ovstage, so it cannot populate an aarch64 venv.
    if target_platform == contract.LINUX_AARCH64_PLATFORM and not contract.APP_PROJECT.is_file():
        raise _fail(
            "platform_unsupported",
            "This source export does not support native Linux aarch64 builds",
            "Build on a Linux x86_64 host.",
        )
    problems: list[str] = []
    if target_platform == contract.DEFAULT_TARGET_PLATFORM:
        if os.name != "nt":
            problems.append(f"host is {platform.system()}, not Windows")
        if platform.machine().upper() not in {"AMD64", "X86_64"}:
            problems.append(f"architecture is {platform.machine()}, not AMD64")
        rerun = "Pass -Python <path-to-64-bit-CPython-3.12> or set OVSPATIAL_PYTHON, then rerun .\\dev.ps1 bootstrap."
    else:
        architecture = "aarch64" if target_platform == contract.LINUX_AARCH64_PLATFORM else "x86_64"
        if platform.system() != "Linux":
            problems.append(f"host is {platform.system()}, not Linux")
        if platform.machine() != architecture:
            problems.append(f"architecture is {platform.machine()}, not {architecture}")
        rerun = f"Run ./dev.sh on a Linux {architecture} host with a 64-bit CPython 3.12."
    if (
        platform.python_implementation() != "CPython"
        or sys.version_info[:2] != contract.EXPECTED_PYTHON
    ):
        problems.append(
            f"interpreter is {platform.python_implementation()} {platform.python_version()}, not CPython 3.12"
        )
    if sys.maxsize <= 2**32:
        problems.append("interpreter is not 64-bit")
    if problems:
        raise _fail("python_incompatible", "; ".join(problems), rerun)


def _validate_toolchain(target_platform: str, target_deps: Path) -> Path:
    cmake_path = contract._cmake_executable(target_deps, target_platform)
    action = (
        f"Run {_entrypoint(target_platform)} bootstrap to stage the pinned Packman CMake tool. "
        + (
            (
                "The pinned Packman MSVC, Windows SDK and Ninja are staged with it; "
                "OVSPATIALQUERY_COMPILER=host builds with Visual Studio 2022 Build Tools instead."
            )
            if target_platform == contract.DEFAULT_TARGET_PLATFORM
            else "The pinned Packman clang uses the host GCC's C++ runtime and binutils; GCC/G++ "
            "and Ninja remain host prerequisites."
        )
    )
    if not cmake_path.is_file():
        raise _fail("cmake_missing", f"Pinned Packman CMake was not found at {cmake_path}", action)
    if target_platform != contract.DEFAULT_TARGET_PLATFORM:
        # The pinned clang uses the host GCC's C++ runtime; a host build lets CMake find its compiler.
        required = (
            ("gcc", "g++", "ninja") if contract.compiler_selection() == "pinned" else ("ninja",)
        )
        missing_tools = [tool for tool in required if shutil.which(tool) is None]
        if missing_tools:
            raise _fail(
                "native_toolchain_missing",
                "Required Linux build tools were not found on PATH: " + ", ".join(missing_tools),
                "Install GCC/G++ and Ninja; CMake and clang remain Packman-pinned.",
            )
    compiler_issue = contract._staged_compiler_issue(target_deps, target_platform)
    if compiler_issue is not None:
        raise _fail("compiler_missing", compiler_issue, action)
    # Every configure, build, test and nested consumer build this process starts inherits it.
    contract.apply_compiler_environment(target_deps, target_platform)
    return cmake_path.resolve()


def _ensure_cmake(target_deps: Path, target_platform: str) -> None:
    action = (
        "Check access to the Packman CMake and compiler coordinates in deps/toolchain.packman.xml."
    )
    if (
        target_platform == contract.DEFAULT_TARGET_PLATFORM
        and contract.compiler_selection() == "pinned"
    ):
        action += (
            " The pinned MSVC and Windows SDK require NVIDIA network/VPN access when uncached."
            " With Visual Studio 2022 installed, set OVSPATIALQUERY_COMPILER=host and rerun bootstrap."
        )
    packman_name = contract.platform_layout(target_platform).packman_executable
    # A host-compiler build does not need the pinned compiler.
    exclusions = ["-e", "compiler"] if contract.compiler_selection() == "host" else []
    _run(
        [
            contract.PROJECT_ROOT / "tools" / "packman" / packman_name,
            "pull",
            "-p",
            target_platform,
            *exclusions,
            contract.TOOLCHAIN_MANIFEST,
        ],
        code="cmake_fetch_failed",
        action=action,
        environment={
            "PM_PACKAGES_ROOT": str(contract._default_packman_cache().resolve()),
            **os.environ,
        },
    )
    issue = contract._staged_cmake_issue(
        target_deps, target_platform
    ) or contract._staged_compiler_issue(target_deps, target_platform)
    if issue is not None:
        raise _fail(
            "cmake_missing",
            issue,
            "Rerun the Packman tool pull after repairing _build/target-deps.",
        )


def _ensure_physx(cmake: Path, target_deps: Path, target_platform: str) -> None:
    if os.environ.get("OVSPATIALQUERY_PHYSX_SDK_DIR"):
        issue = contract._staged_physx_issue(target_deps, target_platform)
        if issue:
            raise _fail(
                "custom_physx_invalid",
                issue,
                "Set OVSPATIALQUERY_PHYSX_SDK_DIR to a built SDK root containing include and bin.",
            )
        return
    _run(
        [
            cmake,
            f"-DPLATFORM={target_platform}",
            "-P",
            contract.PROJECT_ROOT / "cmake" / "package" / "fetch_physx_sdk.cmake",
        ],
        code="physx_fetch_failed",
        action=(
            "Check Packman/network access to the package pinned in deps/target-deps.packman.xml; "
            "do not fall back to a sibling PhysX source checkout."
        ),
    )
    issue = contract._staged_physx_issue(target_deps, target_platform)
    if issue is not None:
        raise _fail(
            "physx_content_invalid",
            issue,
            "Repair the stale _build/target-deps staging and rerun the pinned Packman fetch.",
        )


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _requirement_line(entry: dict[str, Any]) -> str:
    # wheel-stub packages are installed from their PyPI sdist; the stub then
    # downloads the locked platform wheel from WHEEL_STUB_INDEX_URL.
    artifact = entry.get("sdist", entry)
    return f"{entry['name']} @ {artifact['url']} --hash=sha256:{artifact['sha256']}"


def _write_requirements(path: Path, entries: Sequence[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(map(_requirement_line, entries)) + "\n", encoding="utf-8")
    return path


# Import name used to prove each locked distribution is importable.
_IMPORT_PROBES = {
    "numpy": "numpy",
    "usd-core": "pxr.Usd",
    "ovrtx": "ovrtx",
    "ovspatialquery": "ovspatialquery",
    "ovspatialqueryapp": "ovspatialqueryapp",
    "ovstage": "ovstage",
    "ovui": "omni.ui",
    "ovui-data-adapters-common": "ovui_data_adapters",
    "ovui-data-adapters-services": "ovui_data_adapters.services",
    "ovui-widgets-common": "ovui_widgets",
    "ovui-widgets-content": "ovui_widgets.content",
    "ovui-widgets-stage": "ovui_widgets.stage",
    "ovui-widgets-viewport": "ovui_widgets.viewport",
    "pillow": "PIL",
    "warp-lang": "warp",
}

# Distributions a library scope leaves out of (and removes from) the venv.
_LIBRARY_SCOPE_OMITS = tuple(
    sorted(
        {
            *contract.VIEWER_ONLY_RUNTIME_PACKAGES,
            *contract.VISUAL_EXAMPLE_RUNTIME_PACKAGES,
            *contract.LESSON_ONLY_PACKAGES,
            "ovspatialqueryapp",
            "wheel-stub",
        }
    )
)


def _linux_nvidia_hardware_exposure() -> bool | None:
    try:
        if any(Path("/dev").glob("nvidia[0-9]*")):
            return True
        gpu_root = Path("/proc/driver/nvidia/gpus")
        if gpu_root.is_dir() and any(gpu_root.iterdir()):
            return True
        pci_root = Path("/sys/bus/pci/devices")
        if not pci_root.is_dir():
            return None
        for device in pci_root.iterdir():
            vendor = (device / "vendor").read_text(encoding="ascii").strip()
            device_class = (device / "class").read_text(encoding="ascii").strip()
            if vendor == "0x10de" and device_class.startswith("0x03"):
                return True
        return False
    except OSError:
        return None


def _gpu_runtime_status(target_platform: str) -> dict[str, Any]:
    if target_platform == contract.DEFAULT_TARGET_PLATFORM:
        return {
            "state": "required",
            "detail": "Windows keeps the full Viewer verification lane",
            "hardware_exposed": None,
        }

    hardware_exposed = _linux_nvidia_hardware_exposure()
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
    except OSError as exc:
        return {
            "state": "absent" if hardware_exposed is False else "error",
            "detail": (
                "no NVIDIA hardware or loadable libcuda runtime was found"
                if hardware_exposed is False
                else (
                    "NVIDIA hardware exposure is present or unknown but libcuda could not load: "
                    f"{exc}"
                )
            ),
            "hardware_exposed": hardware_exposed,
        }

    try:
        cuda.cuInit.argtypes = [ctypes.c_uint]
        cuda.cuInit.restype = ctypes.c_int
        cuda.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        cuda.cuDeviceGetCount.restype = ctypes.c_int
        init_status = int(cuda.cuInit(0))
        count = ctypes.c_int()
        count_status = int(cuda.cuDeviceGetCount(ctypes.byref(count))) if init_status == 0 else None
    except (AttributeError, TypeError, ValueError) as exc:
        return {
            "state": "error",
            "detail": f"libcuda capability probe failed: {exc}",
            "hardware_exposed": hardware_exposed,
        }

    available = init_status == 0 and count_status == 0 and count.value > 0
    absent = hardware_exposed is False and (
        init_status == 100 or (init_status == 0 and count_status == 0 and count.value == 0)
    )
    return {
        "state": "available" if available else "absent" if absent else "error",
        "detail": (
            f"CUDA reports {count.value} device(s)"
            if available
            else "libcuda reports no device and no NVIDIA hardware is exposed"
            if absent
            else f"CUDA probe failed: cuInit={init_status}, cuDeviceGetCount={count_status}"
        ),
        "hardware_exposed": hardware_exposed,
    }


def _verification_scope(target_platform: str, requested_scope: str = "auto") -> dict[str, Any]:
    if requested_scope in {"library", "library-cuda"}:
        gpu = {
            "state": "not-probed",
            "detail": "explicit library scope does not require a GPU runtime",
            "hardware_exposed": None,
        }
        library_only = True
    elif requested_scope == "auto":
        gpu = _gpu_runtime_status(target_platform)
        library_only = gpu["state"] == "absent" or not contract.APP_ROOT.exists()
    else:
        raise ValueError(f"Unsupported verification scope: {requested_scope}")

    omitted_packages = _LIBRARY_SCOPE_OMITS if library_only else []
    scope = "library-only" if library_only else "full-viewer"
    print(
        f"[bootstrap] verification scope: {scope} (requested: {requested_scope}); "
        f"GPU runtime {gpu['state']}: {gpu['detail']}",
        file=sys.stderr,
        flush=True,
    )
    if omitted_packages:
        print(
            "[bootstrap] Viewer Python distributions omitted in library scope: "
            + ", ".join(omitted_packages),
            file=sys.stderr,
            flush=True,
        )
    return {
        "name": scope,
        "requested": requested_scope,
        "gpu_runtime": gpu,
        "include_viewer": not library_only,
        "include_cuda": requested_scope == "library-cuda"
        or (requested_scope == "auto" and gpu["state"] != "absent"),
        "omitted_python_packages": omitted_packages,
        "exclusion_reason": (
            "Library scope was explicitly requested; Viewer, ovRTX, CUDA, and PTX are omitted"
            if requested_scope == "library"
            else "Library scope was explicitly requested; Viewer and ovRTX are omitted"
            if requested_scope == "library-cuda"
            else "Viewer sources are absent; only library distributions are installed"
            if not contract.APP_ROOT.exists()
            else "GPU runtime is absent; Viewer distributions are not needed by the CPU library"
            if library_only
            else None
        ),
    }


def _probe_python(
    python: Path,
    target_platform: str = contract.DEFAULT_TARGET_PLATFORM,
    *,
    include_viewer: bool = True,
    include_cuda: bool = False,
    failure_action: str | None = None,
) -> dict[str, Any]:
    """Check locked versions and import everything selected for this scope."""

    expected = {
        wheel["name"]: wheel["version"]
        for wheel in (
            *contract._build_tools(
                target_platform, include_viewer=include_viewer, include_cuda=include_cuda
            ),
            *contract._locked_runtime_wheels(target_platform, include_viewer=include_viewer),
        )
    }
    expected["ovspatialquery"] = contract.python_projection_version()
    if include_viewer:
        expected["ovspatialqueryapp"] = contract.python_projection_version(contract.APP_PROJECT)
        for name, (version, _) in contract.lesson_requirements().items():
            expected[name] = version
    imports = {name: module for name, module in _IMPORT_PROBES.items() if name in expected}
    forbidden = [] if include_viewer else _LIBRARY_SCOPE_OMITS
    probe = """
import importlib
import importlib.metadata
import json
import platform
import struct
import sys

expected = json.loads(sys.argv[1])
imports = json.loads(sys.argv[2])
forbidden = json.loads(sys.argv[3])
versions = {}
missing = []
for name in expected:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        missing.append(name)
unexpected = {}
for name in forbidden:
    try:
        unexpected[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        pass
modules = {
    name: importlib.import_module(module)
    for name, module in imports.items()
    if name not in missing
}
print(json.dumps({
    "executable": sys.executable,
    "implementation": platform.python_implementation(),
    "version": platform.python_version(),
    "bits": struct.calcsize("P") * 8,
    "machine": platform.machine(),
    "versions": versions,
    "missing": missing,
    "unexpected_versions": unexpected,
    "module_paths": {
        name: getattr(module, "__file__", None)
        for name, module in modules.items()
    },
}, sort_keys=True))
"""
    result = _run(
        [
            python,
            "-c",
            probe,
            json.dumps(expected),
            json.dumps(imports),
            json.dumps(forbidden),
        ],
        code="python_environment_invalid",
        action=failure_action
        or f"Rerun {_entrypoint(target_platform)} bootstrap to restore the locked environment.",
        capture=True,
        display=f"{python} -c <bootstrap verification probe>",
    )
    details = json.loads(result.stdout)
    details["imports"] = {
        "verified": sorted(imports),
    }
    if (
        details["implementation"] != "CPython"
        or tuple(int(part) for part in details["version"].split(".")[:2])
        != contract.EXPECTED_PYTHON
        or details["bits"] != 64
    ):
        raise _fail(
            "python_incompatible",
            f"Venv uses {details['implementation']} {details['version']} ({details['bits']}-bit)",
            f"Choose a new venv path and rerun {_entrypoint(target_platform)} bootstrap with 64-bit CPython 3.12.",
        )
    if details["missing"]:
        library_names = {
            "ovspatialquery",
            *(
                wheel["name"]
                for wheel in (
                    *contract._build_tools(target_platform, include_viewer=False),
                    *contract._locked_runtime_wheels(target_platform, include_viewer=False),
                )
            ),
        }
        if library_names.isdisjoint(details["missing"]):
            # Only packages a narrower --scope omits are absent: the venv was
            # bootstrapped with a different scope than this command selected.
            selected = (
                "full Viewer" if include_viewer else "library-cuda" if include_cuda else "library"
            )
            entrypoint = _entrypoint(target_platform)
            example = "-Scope Library" if entrypoint.endswith(".ps1") else "--scope library"
            raise _fail(
                "python_scope_mismatch",
                f"The venv lacks {details['missing']} required by the selected {selected} scope; "
                "it was bootstrapped with a narrower scope",
                f"Rerun this command with the scope bootstrap used (for example {example}), "
                f"or rerun {entrypoint} bootstrap with the {selected} scope.",
            )
        raise _fail(
            "python_version_drift",
            f"Locked packages are not installed: {details['missing']}",
            f"Rerun {_entrypoint(target_platform)} bootstrap to restore the hash-pinned dependency train.",
        )
    if details["versions"] != expected:
        raise _fail(
            "python_version_drift",
            f"Installed versions differ from the lock: {details['versions']}",
            f"Rerun {_entrypoint(target_platform)} bootstrap to restore the hash-pinned dependency train.",
        )
    if details["unexpected_versions"]:
        raise _fail(
            "python_scope_mismatch",
            f"Library scope still contains Viewer packages: {details['unexpected_versions']}",
            f"Rerun {_entrypoint(target_platform)} bootstrap with library scope before continuing.",
        )
    expected_projection_source = (
        contract.PYTHON_ROOT / "src" / "ovspatialquery" / "__init__.py"
    ).resolve()
    observed_projection_source = details.get("module_paths", {}).get("ovspatialquery")
    if (
        not observed_projection_source
        or Path(observed_projection_source).resolve() != expected_projection_source
    ):
        raise _fail(
            "python_projection_source_mismatch",
            (
                "ovspatialquery imported from "
                f"{observed_projection_source!r}, expected editable source "
                f"{expected_projection_source}"
            ),
            (
                f"Rerun {_entrypoint(target_platform)} bootstrap; the source-tree app, tests and "
                "examples must consume python/src rather than an installed wheel."
            ),
        )
    return details


def _read_cmake_cache(native_build: Path) -> dict[str, str]:
    cache: dict[str, str] = {}
    for line in (
        (native_build / "CMakeCache.txt").read_text(encoding="utf-8", errors="replace").splitlines()
    ):
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        key_and_type, value = line.split("=", 1)
        cache[key_and_type.split(":", 1)[0]] = value
    return cache


def _require_native_scope(
    cache: dict[str, str],
    *,
    target_platform: str,
    include_cuda: bool,
) -> None:
    configured_cuda = bool(cache.get("OVSPATIALQUERY_CUDA_NVRTC_DIR", ""))
    if configured_cuda == include_cuda:
        return
    expected = "CUDA-enabled" if include_cuda else "CPU-only"
    observed = "CUDA-enabled" if configured_cuda else "CPU-only"
    raise _fail(
        "native_scope_mismatch",
        f"Native tree is configured for {observed}, expected {expected}",
        f"Rerun {_entrypoint(target_platform)} bootstrap with the same scope before continuing.",
    )


def _install_issue(native_build: Path, expect_companion: bool, target_platform: str) -> str | None:
    """Compare every installed artifact with the build output or source it came from."""

    cache = _read_cmake_cache(native_build)
    selected_sdk = os.environ.get("OVSPATIALQUERY_PHYSX_SDK_DIR", "")
    configured_sdk = cache.get("OVSPATIALQUERY_PHYSX_SDK_DIR", "")
    if bool(selected_sdk) != bool(configured_sdk) or (
        selected_sdk and Path(selected_sdk).expanduser().resolve() != Path(configured_sdk).resolve()
    ):
        return "Selected PhysX SDK differs from the configured build; rerun build."
    install = native_build / "install"
    include = contract.PROJECT_ROOT / "include"
    layout = contract.platform_layout(target_platform)
    # Visual Studio builds into a folder per configuration; Ninja (Linux, and Windows with the
    # pinned MSVC) builds into the tree itself.
    if _read_cmake_cache(native_build).get("CMAKE_GENERATOR", "").startswith("Visual Studio"):
        built = native_build / os.environ.get("OVSPATIALQUERY_NATIVE_CONFIG", "Release")
    else:
        built = native_build
    comparisons = [
        (built / layout.library_file, install / layout.library_dir / layout.library_file),
    ]
    if layout.import_library is not None:
        comparisons.append((built / layout.import_library, install / "lib" / layout.import_library))
    ptx = (
        built / "ovspatialquery_cuda.ptx",
        install / layout.library_dir / "ovspatialquery_cuda.ptx",
    )
    # A full-to-library reconfiguration can leave an old build PTX. The
    # verified install is replaced and is authoritative for the selected scope.
    if ptx[1].is_file():
        comparisons.append(ptx)
    headers = list(contract.public_headers())
    if expect_companion:
        headers.append(contract.OVSTAGE_PUBLIC_HEADER)
    comparisons.extend(
        (include / "ovspatialquery" / header, install / "include" / "ovspatialquery" / header)
        for header in headers
    )
    comparisons.append(
        (include / "dlpack" / "dlpack.h", install / "include" / "dlpack" / "dlpack.h")
    )
    legal_sources = {
        relative: contract.PROJECT_ROOT / source
        for relative, source in python_wheel.LEGAL_FILES.items()
    }
    cache = _read_cmake_cache(native_build)
    for relative, key in (
        ("licenses/physx-LICENSE.md", "OVSPATIALQUERY_PHYSX_LICENSE_SOURCE"),
        ("licenses/physx-PROVENANCE.json", "OVSPATIALQUERY_PHYSX_PROVENANCE_SOURCE"),
    ):
        if cache.get(key):
            legal_sources[relative] = Path(cache[key])
    comparisons.extend((source, install / relative) for relative, source in legal_sources.items())
    for source, installed in comparisons:
        if not source.is_file():
            return f"Native install source is missing: {source}"
        if not installed.is_file():
            return f"Native install artifact is missing: {installed}"
        source_digest = _sha256(source)
        installed_digest = _sha256(installed)
        if installed_digest != source_digest:
            return (
                f"Native install artifact differs from what it was built from: {installed}; "
                f"expected sha256 {source_digest}, found {installed_digest}"
            )
    if cache.get("OVSPATIALQUERY_PHYSX_SDK_DIR"):
        provenance = json.loads(legal_sources["licenses/physx-PROVENANCE.json"].read_text())
        sdk = Path(cache["OVSPATIALQUERY_PHYSX_SDK_DIR"])
        for relative, expected in provenance["files"].items():
            source = sdk / relative
            if not source.is_file() or _sha256(source) != expected:
                return f"Custom PhysX SDK changed since configure: {source}; rerun build."
    cache = _read_cmake_cache(native_build)
    for name, payload in consumer_content.collect(
        contract.PROJECT_ROOT,
        version=cache["OVSPATIALQUERY_VERSION"],
        source_revision=cache["OVSPATIALQUERY_SOURCE_REVISION"],
    ).items():
        installed = install / consumer_content.SDK_CONTENT / name
        if not installed.is_file() or installed.read_bytes() != payload:
            return f"Installed consumer content differs from source: {installed}"
    return None


def _require_install(
    native_build: Path, cache: dict[str, str], target_platform: str, include_cuda: bool
) -> None:
    """Fail unless the tree has the selected scope and its install matches the build."""

    _require_native_scope(cache, include_cuda=include_cuda, target_platform=target_platform)
    install_issue = _install_issue(
        native_build,
        cache.get("OVSPATIALQUERY_BUILD_OVSTAGE_COMPANION", "OFF") == "ON",
        target_platform,
    )
    if install_issue is not None:
        raise _fail(
            "native_install_invalid",
            install_issue,
            f"Run {_entrypoint(target_platform)} build to regenerate and verify the native install.",
        )


def verify(
    venv: Path,
    native_build: Path,
    target_deps: Path,
    target_platform: str = contract.DEFAULT_TARGET_PLATFORM,
    requested_scope: str = "auto",
    *,
    run_native_tests: bool = True,
) -> dict[str, Any]:
    """Check the venv contents, the installed artifacts, and run the native tests."""

    verification_scope = _verification_scope(target_platform, requested_scope)
    blockers = contract.verify_environment(venv, native_build, target_deps, target_platform)
    python_details: dict[str, Any] | None = None
    native_details: dict[str, Any] | None = None

    venv_python = contract._venv_python(venv)
    if venv_python.is_file():
        try:
            python_details = _probe_python(
                venv_python,
                target_platform,
                include_viewer=verification_scope["include_viewer"],
                include_cuda=verification_scope["include_cuda"],
            )
            python_details["pip_check"] = _pip_check(venv_python)
        except contract.BootstrapError as exc:
            blockers.append(contract.Blocker(exc.code, str(exc), exc.action))

    cmake = contract._cmake_executable(target_deps, target_platform)
    if (native_build / "CMakeCache.txt").is_file() and cmake.is_file():
        cache = _read_cmake_cache(native_build)
        native_details = {
            "physx_package": "custom"
            if cache.get("OVSPATIALQUERY_PHYSX_SDK_DIR")
            else contract._packaged_physx_version(target_platform),
            "ovstage_companion": {
                "enabled": cache.get("OVSPATIALQUERY_BUILD_OVSTAGE_COMPANION", "OFF"),
                "sdk": cache.get("OVSPATIALQUERY_OVSTAGE_SDK_DIR", ""),
            },
            "cuda": {"nvrtc": cache.get("OVSPATIALQUERY_CUDA_NVRTC_DIR", "")},
        }
        try:
            _require_install(
                native_build, cache, target_platform, verification_scope["include_cuda"]
            )
            lifecycle = contract.native_lifecycle(
                native_build,
                target_deps,
                target_platform,
                cmake=cmake,
                python=sys.executable,
                include_viewer=verification_scope["include_viewer"],
                include_cuda=verification_scope["include_cuda"],
            )
            if run_native_tests:
                _run(
                    lifecycle["test"],
                    code="native_tests_failed",
                    action="Fix the failing native or installed-package test without weakening its assertion.",
                    capture=True,
                )
            native_details["tests"] = "passed" if run_native_tests else "not-run"
        except contract.BootstrapError as exc:
            blockers.append(contract.Blocker(exc.code, str(exc), exc.action))

    return {
        "schema_version": 1,
        "ok": not blockers,
        "target_platform": target_platform,
        "verification_scope": verification_scope,
        "paths": {
            "venv": str(venv.resolve()),
            "native_build": str(native_build.resolve()),
            "target_deps": str(target_deps.resolve()),
        },
        "python": python_details,
        "native": native_details,
        "blockers": [asdict(blocker) for blocker in blockers],
    }


def _pip_check(venv_python: Path) -> str:
    """Run pip check; on the next train, allow only the App's pins on the swapped wheels."""
    if contract.train() != "next":
        _run(
            [venv_python, "-m", "pip", "check"],
            code="python_dependency_conflict",
            action="Restore the checked-in lock/pyproject pairing; do not add ad-hoc dependency overrides.",
            capture=True,
        )
        return "passed"
    action = (
        "A swapped next-train wheel needs a dependency the lock does not provide; "
        "report it to the package owners."
    )
    try:
        completed = subprocess.run(
            [str(venv_python), "-m", "pip", "check"], capture_output=True, text=True, check=False
        )
    except OSError as exc:
        raise _fail(
            "python_dependency_conflict", f"Could not run pip check in {venv_python}: {exc}", action
        ) from exc
    swapped = contract.next_train_resolution()["wheels"]
    expected = tuple(
        f"ovspatialqueryapp {contract.python_projection_version(contract.APP_PROJECT)} has "
        f"requirement {name}=="
        for name in swapped
    )
    conflicts = [
        line
        for line in completed.stdout.splitlines()
        if line.strip()
        and not line.startswith(expected)
        and line != "No broken requirements found."
    ]
    if conflicts or (completed.returncode != 0 and not completed.stdout.strip()):
        raise _fail(
            "python_dependency_conflict",
            "pip check on the next train: "
            + "; ".join(conflicts or [completed.stderr.strip() or f"exit {completed.returncode}"]),
            action,
        )
    return "passed except the App's pins on " + ", ".join(sorted(swapped))


_LOCKED_INSTALL = ("--force-reinstall", "--require-hashes", "--only-binary=:all:")


def _pip_install(python: Path, *arguments: str | Path) -> list[str | Path]:
    common = ("-m", "pip", "install", "--disable-pip-version-check", "--no-index", "--no-deps")
    return [python, *common, *arguments]


def _create_application_environment(
    venv: Path,
    native_build: Path,
    target_platform: str,
    *,
    include_viewer: bool,
    include_cuda: bool = False,
) -> Path:
    """Create the worktree-local venv and install its selected locked packages."""

    venv_python = contract._venv_python(venv)
    if not venv_python.is_file():
        _run(
            [sys.executable, "-m", "venv", venv],
            code="venv_create_failed",
            action=(
                "Install a complete 64-bit CPython 3.12 (including venv/ensurepip), "
                f"then rerun {_entrypoint(target_platform)} bootstrap."
            ),
        )
    # The next train keeps its generated requirements beside its resolution.
    generated = (
        native_build.parent
        if contract.train() == "stable"
        else contract.NEXT_TRAIN_RESOLUTION.parent
    )
    build_requirements = _write_requirements(
        generated / "locked-python-build-requirements.txt",
        contract._build_tools(
            target_platform, include_viewer=include_viewer, include_cuda=include_cuda
        ),
    )
    runtime_wheels = contract._locked_runtime_wheels(target_platform, include_viewer=include_viewer)
    requirements = _write_requirements(generated / "locked-python-requirements.txt", runtime_wheels)
    _run(
        _pip_install(venv_python, *_LOCKED_INSTALL, "-r", build_requirements),
        code="python_build_tool_install_failed",
        action="Restore access to the hash-pinned pip, setuptools, and wheel-stub wheels.",
    )
    if not include_viewer:
        _run(
            [venv_python, "-m", "pip", "uninstall", "--yes", *_LIBRARY_SCOPE_OMITS],
            code="viewer_package_removal_failed",
            action="Repair the worktree venv and rerun library-only bootstrap.",
        )
    _run(
        _pip_install(
            venv_python,
            *_LOCKED_INSTALL,
            *(f"--no-binary={wheel['name']}" for wheel in runtime_wheels if "sdist" in wheel),
            "--no-build-isolation",
            "--check-build-dependencies",
            "-r",
            requirements,
        ),
        code="python_artifact_install_failed",
        action="Check access to the URLs shown by the plan; the wheels are hash-pinned in uv.lock.",
        environment={**os.environ, "WHEEL_STUB_PIP_INDEX_URL": contract.WHEEL_STUB_INDEX_URL},
    )
    editable = ("--no-cache-dir", "--no-build-isolation", "--editable")
    _run(
        _pip_install(venv_python, *editable, contract.PYTHON_ROOT),
        code="python_projection_install_failed",
        action="Fix the ovspatialquery Python source package; the bootstrap installs it editable.",
    )
    if include_viewer:
        _run(
            _pip_install(venv_python, *editable, contract.APP_ROOT),
            code="app_install_failed",
            action="Fix the app packaging failure; do not replace the install with PYTHONPATH.",
        )
        # The lessons' Warp line pins a version and hashes, not a URL, so pip reads the index.
        lessons = generated / "locked-lesson-requirements.txt"
        lessons.write_text(
            "".join(f"{line}\n" for _, line in contract.lesson_requirements().values()),
            encoding="utf-8",
        )
        _run(
            [
                venv_python,
                *("-m", "pip", "install", "--disable-pip-version-check", "--no-deps"),
                *_LOCKED_INSTALL,
                "-r",
                lessons,
            ],
            code="python_artifact_install_failed",
            action=f"Check access to the Python package index for the pins in {lessons.name}.",
        )
    return venv_python


def _require_existing(venv: Path, native_build: Path, command: str) -> Path:
    if not native_build.is_dir():
        raise _fail(
            "native_build_missing",
            f"Native build directory does not exist: {native_build}",
            f"Run the bootstrap command before {command}.",
        )
    venv_python = contract._venv_python(venv)
    if not venv_python.is_file():
        raise _fail(
            "venv_missing",
            f"Application environment does not exist: {venv}",
            f"Run the bootstrap command before {command}.",
        )
    return venv_python


def _run_native_lifecycle(lifecycle: dict[str, list[str]], phases: Sequence[str]) -> None:
    actions = {
        "configure": (
            "native_configure_failed",
            "Install the documented CMake and C++ toolchain, then rerun.",
        ),
        "build": (
            "native_build_failed",
            "Fix the native build failure; do not weaken the product backend.",
        ),
        "test": (
            "native_tests_failed",
            "Fix the failing native test without weakening its assertion.",
        ),
        "install": (
            "native_install_failed",
            "Fix the installed library/export/ctypes verification failure.",
        ),
    }
    for phase in phases:
        code, action = actions[phase]
        command = lifecycle[phase]
        if phase == "configure":
            command = _fresh_if_compiler_changed(command)
            _announce_development_version()
        _run(command, code=code, action=action)


def _announce_development_version() -> None:
    """Say once per configure when nothing selected a release version."""
    version, origin = contract.version_selection()
    if origin != "source":
        return
    identity = contract.source_identity()
    if identity is not None:
        version = contract.wheel_version(*identity)
    print(
        f"[bootstrap] building development version {version}; "
        "set OVSPATIALQUERY_BUILD_VERSION=X.Y.Z[rcN] to override",
        file=sys.stderr,
        flush=True,
    )


def _fresh_if_compiler_changed(configure: list[str]) -> list[str]:
    """Reconfigure from scratch when a tree was configured for the other compiler selection.

    CMake otherwise keeps a cached compiler (a host build would reuse the pinned clang) or
    refuses a different generator (Windows Ninja against Visual Studio)."""
    native_build = Path(configure[configure.index("-B") + 1])
    if not (native_build / "CMakeCache.txt").is_file():
        return configure
    recorded = _read_cmake_cache(native_build).get("OVSPATIALQUERY_COMPILER_SELECTION")
    if recorded == contract.compiler_selection():
        return configure
    return [configure[0], "--fresh", *configure[1:]]


def configure(
    venv: Path,
    native_build: Path,
    target_deps: Path,
    target_platform: str,
    requested_scope: str = "auto",
    *,
    prebuilt_core_independence: bool = False,
) -> dict[str, Any]:
    contract.platform_layout(target_platform)
    _validate_host(target_platform)
    _ensure_cmake(target_deps, target_platform)
    cmake = _validate_toolchain(target_platform, target_deps)
    verification_scope = _verification_scope(target_platform, requested_scope)
    if contract.train() == "next":
        import next_train

        next_train.write(target_platform)
    else:
        import ovstage_sdk

        ovstage_sdk.stage(target_platform)
    venv_python = _create_application_environment(
        venv,
        native_build,
        target_platform,
        include_viewer=verification_scope["include_viewer"],
        include_cuda=verification_scope["include_cuda"],
    )
    _ensure_physx(cmake, target_deps, target_platform)
    lifecycle = contract.native_lifecycle(
        native_build,
        target_deps,
        target_platform,
        cmake=cmake,
        python=venv_python,
        include_viewer=verification_scope["include_viewer"],
        include_cuda=verification_scope["include_cuda"],
    )
    if prebuilt_core_independence:
        lifecycle["configure"].append("-DOVSPATIALQUERY_PREBUILT_CORE_INDEPENDENCE=ON")
    _run_native_lifecycle(lifecycle, ("configure",))
    return {"schema_version": 1, "ok": True, "command": "configure", "lifecycle": lifecycle}


def bootstrap(
    venv: Path,
    native_build: Path,
    target_deps: Path,
    target_platform: str,
    requested_scope: str = "auto",
    *,
    run_native_tests: bool = True,
) -> dict[str, Any]:
    configured = configure(venv, native_build, target_deps, target_platform, requested_scope)
    _run_native_lifecycle(configured["lifecycle"], ("build", "install"))
    report = verify(
        venv,
        native_build,
        target_deps,
        target_platform,
        requested_scope,
        run_native_tests=run_native_tests,
    )
    if not report["ok"]:
        raise _fail(**report["blockers"][0])
    return report


def build_existing(
    venv: Path,
    native_build: Path,
    target_deps: Path,
    target_platform: str,
    requested_scope: str = "auto",
) -> dict[str, Any]:
    _validate_host(target_platform)
    cmake = _validate_toolchain(target_platform, target_deps)
    venv_python = _require_existing(venv, native_build, "build")
    verification_scope = _verification_scope(target_platform, requested_scope)
    if contract.train() == "stable":
        import ovstage_sdk

        ovstage_sdk.stage(target_platform)
    _probe_python(
        venv_python,
        target_platform,
        include_viewer=verification_scope["include_viewer"],
        include_cuda=verification_scope["include_cuda"],
    )
    lifecycle = contract.native_lifecycle(
        native_build,
        target_deps,
        target_platform,
        cmake=cmake,
        python=venv_python,
        include_viewer=verification_scope["include_viewer"],
        include_cuda=verification_scope["include_cuda"],
    )
    _run_native_lifecycle(lifecycle, ("configure", "build", "install"))
    _require_install(
        native_build,
        _read_cmake_cache(native_build),
        target_platform,
        verification_scope["include_cuda"],
    )
    return {
        "schema_version": 1,
        "ok": True,
        "command": "build",
        "verification_scope": verification_scope,
    }


def test_existing(
    venv: Path,
    native_build: Path,
    target_deps: Path,
    target_platform: str,
    requested_scope: str = "auto",
) -> dict[str, Any]:
    _validate_host(target_platform)
    cmake = _validate_toolchain(target_platform, target_deps)
    venv_python = _require_existing(venv, native_build, "test")
    verification_scope = _verification_scope(target_platform, requested_scope)
    _probe_python(
        venv_python,
        target_platform,
        include_viewer=verification_scope["include_viewer"],
        include_cuda=verification_scope["include_cuda"],
    )
    cache = _read_cmake_cache(native_build) if (native_build / "CMakeCache.txt").is_file() else {}
    _require_native_scope(
        cache,
        include_cuda=verification_scope["include_cuda"],
        target_platform=target_platform,
    )
    lifecycle = contract.native_lifecycle(
        native_build,
        target_deps,
        target_platform,
        cmake=cmake,
        python=sys.executable,
        include_viewer=verification_scope["include_viewer"],
        include_cuda=verification_scope["include_cuda"],
    )
    _run_native_lifecycle(lifecycle, ("test",))
    app_suites = [(contract.APP_ROOT / "tests", contract.APP_ROOT, "test_*.py")]
    example_root = contract.PROJECT_ROOT / "tests" / "examples"
    example_suites = [(example_root, example_root, "test_*.py")]
    excluded_test_suites: list[str] = []
    test_environment = os.environ.copy()
    layout = contract.platform_layout(target_platform)
    test_environment["OVSPATIALQUERY_LIBRARY"] = str(
        native_build / "install" / layout.library_dir / layout.library_file
    )
    app_test_environment = test_environment.copy()
    if verification_scope["name"] == "library-only":
        app_suites = [
            (
                contract.APP_ROOT / "tests" / "unit" / "results",
                contract.APP_ROOT,
                "test_*.py",
            ),
            (contract.APP_ROOT / "tests" / "acceptance", contract.APP_ROOT, "test_*.py"),
        ]
        example_suites = [
            (example_root, example_root, path.name)
            for path in sorted(example_root.glob("test_*.py"))
            if path.name
            not in (
                "test_capability_showcase.py",
                "test_rendered_transform_motion.py",
                "test_evidence_contract.py",
            )
        ]
        excluded_test_suites = [
            "apps/ovspatialqueryapp/tests/unit/runtime",
            "apps/ovspatialqueryapp/tests/unit/ui",
            "apps/ovspatialqueryapp/tests/integration",
            "tests/examples/test_evidence_contract.py",
            "tests/examples/test_capability_showcase.py",
            "tests/examples/test_rendered_transform_motion.py",
        ]
        print(
            "[bootstrap] Viewer test suites excluded in library scope: "
            + ", ".join(excluded_test_suites),
            file=sys.stderr,
            flush=True,
        )
        app_test_environment["PYTHONPATH"] = os.pathsep.join(
            part
            for part in (
                str(contract.APP_ROOT / "src"),
                app_test_environment.get("PYTHONPATH"),
            )
            if part
        )
    test_suites = [
        *app_suites,
        (
            contract.PROJECT_ROOT / "tests" / "tooling" / "bootstrap",
            contract.PROJECT_ROOT / "tests" / "tooling" / "bootstrap",
            "test_*.py",
        ),
        *example_suites,
        (contract.PYTHON_ROOT / "tests", contract.PYTHON_ROOT, "test_*.py"),
    ]
    if not contract.APP_ROOT.exists():
        test_suites = [suite for suite in test_suites if suite not in app_suites]
    for test_root, top_level, pattern in test_suites:
        _run(
            [
                venv_python,
                "-m",
                "unittest",
                "discover",
                "-s",
                test_root,
                "-t",
                top_level,
                "-p",
                pattern,
            ],
            code="python_tests_failed",
            action=f"Fix the failing Python tests under {test_root}.",
            environment=(
                app_test_environment
                if contract._is_within(test_root, contract.APP_ROOT)
                else test_environment
            ),
        )
    return {
        "schema_version": 1,
        "ok": True,
        "command": "test",
        "verification_scope": {
            **verification_scope,
            "excluded_test_suites": excluded_test_suites,
            "test_suites": [
                str(
                    (root / pattern if pattern != "test_*.py" else root).relative_to(
                        contract.PROJECT_ROOT
                    )
                )
                for root, _, pattern in test_suites
            ],
        },
    }


def _isolated_wheel_probe(
    python: Path,
    wheel_root: Path,
    *,
    require_cuda: bool = False,
    require_unavailable_cuda: bool = False,
    wheel: Path | None = None,
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("OVSPATIALQUERY_LIBRARY", None)
    environment["PYTHONNOUSERSITE"] = "1"
    with tempfile.TemporaryDirectory(prefix="ovspatialquery-wheel-") as isolated_cwd:
        result = _run(
            [
                python,
                "-I",
                contract.PROJECT_ROOT / "tools/bootstrap/wheel_smoke.py",
                *(["--require-cuda"] if require_cuda else []),
                *(["--require-unavailable-cuda"] if require_unavailable_cuda else []),
            ],
            code="python_wheel_runtime_failed",
            action=(
                "Fix the bundled loader or installed queries; do not use "
                "OVSPATIALQUERY_LIBRARY or a repository PYTHONPATH."
            ),
            capture=True,
            environment=environment,
            cwd=Path(isolated_cwd),
        )
    details = json.loads(result.stdout)
    module = Path(details["module_path"]).resolve()
    library = Path(details["library_path"]).resolve()
    source_roots = (
        (contract.PYTHON_ROOT / "src").resolve(),
        (contract.APP_ROOT / "src").resolve(),
    )
    if any(
        contract._is_within(path, source_root)
        for path in (module, library)
        for source_root in source_roots
    ):
        raise _fail(
            "python_wheel_source_leak",
            (
                "Fresh wheel verification resolved repository content: "
                f"module={module}, library={library}"
            ),
            "Remove source-path leakage and rerun python-wheel.",
        )
    expected_native_root = module.parent / "_native"
    if (
        module.parent.name != "ovspatialquery"
        or not contract._is_within(library, expected_native_root)
        or library.name
        not in tuple(layout.library_file for layout in contract.PLATFORM_LAYOUTS.values())
    ):
        raise _fail(
            "python_wheel_runtime_invalid",
            f"Fresh wheel runtime probe returned invalid state: {details}",
            "Fix the wheel loader/runtime contract and rerun python-wheel.",
        )
    provenance = json.loads((expected_native_root / "provenance.json").read_text())
    revision = provenance["source_revision"] + ("-dirty" if provenance["source_dirty"] else "")
    if wheel is None:
        payloads = consumer_content.collect(
            contract.PROJECT_ROOT,
            "wheel",
            version=details["package_version"],
            source_revision=revision,
        )
    else:
        # Consumer runners have the qualified wheel, not the producer's HTML build.
        with zipfile.ZipFile(wheel) as archive:
            payloads = {
                name.removeprefix("ovspatialquery/"): archive.read(name)
                for name in archive.namelist()
                if name.startswith("ovspatialquery/") and not name.endswith("/")
            }
        for required in ("README.txt", "docs/index.html", "skills/ovspatialquery/SKILL.md"):
            if required not in payloads:
                raise _fail(
                    "python_wheel_content_invalid",
                    f"Wheel consumer content is missing: {required}",
                    "Rebuild the wheel with its public consumer content.",
                )
    for name, payload in payloads.items():
        installed = module.parent / name
        if not installed.is_file() or installed.read_bytes() != payload:
            raise _fail(
                "python_wheel_content_invalid",
                f"Installed wheel content differs: {installed}",
                "Rebuild and reinstall the wheel.",
            )
    details["consumer_content"] = "passed"
    (wheel_root / "runtime.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    return details


def build_python_wheel(
    venv: Path,
    native_build: Path,
    target_deps: Path,
    target_platform: str,
    requested_scope: str = "auto",
    *,
    run_native_tests: bool = True,
    require_unavailable_cuda: bool = False,
) -> dict[str, Any]:
    """Build a platform wheel from the verified install and import it in a fresh venv."""

    verification = verify(
        venv,
        native_build,
        target_deps,
        target_platform,
        requested_scope,
        run_native_tests=run_native_tests,
    )
    if not verification["ok"]:
        raise _fail(**verification["blockers"][0])
    identity = contract.source_identity()
    if identity is None:
        raise _fail(
            "python_wheel_source_identity_failed",
            "Source revision is unavailable",
            "Build from a Git checkout with a resolvable HEAD or a git archive with its expanded .git_archival.txt.",
        )
    revision, dirty = identity
    wheel_root = (contract.PROJECT_ROOT / "_build" / "python-wheel").resolve()
    try:
        artifact = python_wheel.build(
            source_root=contract.PYTHON_ROOT,
            install_root=native_build / "install",
            destination=wheel_root / "dist",
            target_platform=target_platform,
            source_revision=revision,
            source_dirty=dirty,
            version=contract.wheel_version(revision, dirty),
        )
    except python_wheel.WheelBuildError as failure:
        raise _fail(
            "python_wheel_build_failed",
            str(failure),
            "Restore the verified native install and Python sources; setuptools must not invoke CMake.",
        ) from failure

    verification_venv = wheel_root / "verify-venv"
    if verification_venv.exists():
        shutil.rmtree(verification_venv)
    _run(
        [sys.executable, "-m", "venv", verification_venv],
        code="python_wheel_venv_failed",
        action="Repair CPython 3.12 venv support and rerun python-wheel.",
    )
    verification_python = contract._venv_python(verification_venv)
    _run(
        _pip_install(verification_python, "--no-cache-dir", artifact.path),
        code="python_wheel_install_failed",
        action="Fix the platform wheel metadata/content and rerun python-wheel.",
        environment={
            key: value
            for key, value in os.environ.items()
            if key not in ("PYTHONPATH", "OVSPATIALQUERY_LIBRARY")
        },
        cwd=wheel_root,
    )
    isolated = _isolated_wheel_probe(
        verification_python,
        wheel_root,
        require_cuda=verification["verification_scope"]["include_cuda"]
        and not require_unavailable_cuda,
        require_unavailable_cuda=require_unavailable_cuda,
    )
    return {
        "schema_version": 1,
        "ok": True,
        "command": "python-wheel",
        "verification_scope": verification["verification_scope"],
        "wheel": {
            "path": str(artifact.path),
            "sha256": artifact.sha256,
            "size": artifact.size,
            "tag": artifact.tag,
            "manifest": artifact.manifest,
        },
        "isolated_runtime": isolated,
    }


def clean_bootstrap(venv: Path, native_build: Path) -> dict[str, Any]:
    build_root = contract.PROJECT_ROOT / "_build"
    removed: list[str] = []
    for path in (venv, native_build):
        if path.resolve() == build_root.resolve() or not contract._is_within(path, build_root):
            raise _fail(
                "output_path_unsupported",
                f"clean-bootstrap only removes directories beneath {build_root}: {path}",
                "Choose venv and native-build paths under the worktree _build directory.",
            )
        if path.exists():
            shutil.rmtree(path)
            removed.append(str(path))
    return {
        "schema_version": 1,
        "ok": True,
        "command": "clean-bootstrap",
        "removed": removed,
        "preserved": [
            str(contract._selected_packman_cache()),
            str(contract._default_target_deps()),
        ],
    }


def run_app(venv: Path, native_build: Path, target_platform: str) -> int:
    venv_python = _require_existing(venv, native_build, "app")
    verification_scope = _verification_scope(target_platform)
    if not verification_scope["include_viewer"]:
        if not contract.APP_ROOT.exists():
            raise _fail(
                "viewer_runtime_unavailable",
                "Viewer sources are absent from this source distribution.",
                "Use the library commands supported by this distribution.",
            )
        raise _fail(
            "viewer_runtime_unavailable",
            "No NVIDIA GPU runtime is available; library-only bootstrap omits Viewer packages.",
            "Run the Viewer on a host with a compatible NVIDIA GPU and driver/runtime.",
        )
    _probe_python(
        venv_python,
        target_platform,
        failure_action=(
            "Repair the reported Viewer dependency, GPU driver/runtime, or display support; "
            "library-only bootstrap does not qualify the Viewer."
        ),
    )
    return _run(
        [venv_python, "-m", "ovspatialqueryapp"],
        code="app_failed",
        action="Fix the reported viewer startup or runtime failure.",
    ).returncode


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "plan",
        "bootstrap",
        "configure",
        "build",
        "test",
        "app",
        "verify",
        "python-wheel",
        "clean-bootstrap",
    ):
        child = subparsers.add_parser(command)
        if command == "python-wheel":
            child.add_argument(
                "--require-unavailable-cuda",
                action="store_true",
                help="CI CPU builder: consume the CUDA-capable wheel without a usable GPU; real CUDA qualification remains a separate gate",
            )
        if command in {"bootstrap", "python-wheel"}:
            child.add_argument(
                "--skip-native-tests",
                action="store_true",
                help="CI: run native tests once with the separate test command; retain install checks and fresh-wheel queries",
            )
        child.add_argument("--venv", type=Path, default=contract._default_venv())
        child.add_argument("--native-build", type=Path, default=contract._default_native_build())
        child.add_argument("--target-deps", type=Path, default=contract._default_target_deps())
        child.add_argument(
            "--platform",
            default=contract.DEFAULT_TARGET_PLATFORM,
            choices=contract.SUPPORTED_PLATFORMS,
        )
        if command in {"plan", "bootstrap", "configure", "build", "test", "verify", "python-wheel"}:
            child.add_argument(
                "--scope",
                default="auto",
                choices=("auto", "library", "library-cuda"),
                help="auto selects the host scope; library is CPU-only; library-cuda adds pinned NVRTC and PTX without Viewer",
            )
    return parser


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(arguments: Sequence[str] | None = None) -> int:
    try:
        options = _parser().parse_args(arguments)
        venv = options.venv.resolve()
        native_build = options.native_build.resolve()
        target_deps = options.target_deps.resolve()
        if options.command == "plan":
            verification_scope = _verification_scope(options.platform, options.scope)
            plan = contract.build_plan(
                venv,
                native_build,
                target_deps,
                options.platform,
                include_viewer=verification_scope["include_viewer"],
                include_cuda=verification_scope["include_cuda"],
            )
            plan["verification_scope"] = verification_scope
            _emit(plan)
            return 0
        scoped_arguments = (
            venv,
            native_build,
            target_deps,
            options.platform,
            getattr(options, "scope", "auto"),
        )
        if options.command == "verify":
            report = verify(*scoped_arguments)
            _emit(report)
            return 0 if report["ok"] else 2
        if options.command == "bootstrap":
            _emit(bootstrap(*scoped_arguments, run_native_tests=not options.skip_native_tests))
            return 0
        if options.command == "configure":
            _emit(configure(*scoped_arguments))
            return 0
        if options.command == "build":
            _emit(build_existing(*scoped_arguments))
            return 0
        if options.command == "test":
            _emit(test_existing(*scoped_arguments))
            return 0
        if options.command == "python-wheel":
            _emit(
                build_python_wheel(
                    *scoped_arguments,
                    run_native_tests=not options.skip_native_tests,
                    require_unavailable_cuda=options.require_unavailable_cuda,
                )
            )
            return 0
        if options.command == "app":
            return run_app(venv, native_build, options.platform)
        if options.command == "clean-bootstrap":
            _emit(clean_bootstrap(venv, native_build))
            return 0
    except contract.BootstrapError as exc:
        blocker = contract.Blocker(exc.code, str(exc), exc.action)
        _emit({"schema_version": 1, "ok": False, "blockers": [asdict(blocker)]})
        return 2
    raise AssertionError(f"Unhandled command: {options.command}")


if __name__ == "__main__":
    raise SystemExit(main())
