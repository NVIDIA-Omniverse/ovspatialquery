# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualify a CI runner's host, pinned dependencies and native build tools."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import contract
import main as bootstrap


ROOT = contract.PROJECT_ROOT
CUDA_FAILURE = "ovspatialquery CUDA runner preflight failed"


class CudaUnavailableError(RuntimeError):
    """The required GPU runner cannot initialize a usable CUDA device."""


def cuda_status() -> dict[str, object]:
    name = "nvcuda.dll" if os.name == "nt" else "libcuda.so.1"
    try:
        driver = (ctypes.WinDLL if os.name == "nt" else ctypes.CDLL)(name)
    except OSError:
        return {"driver_loaded": False, "devices": 0}
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuInit.restype = ctypes.c_int
    driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    driver.cuDeviceGetCount.restype = ctypes.c_int
    init = driver.cuInit(0)
    count = ctypes.c_int()
    status = driver.cuDeviceGetCount(ctypes.byref(count)) if init == 0 else None
    result = {
        "driver_loaded": True,
        "cuInit": init,
        "cuDeviceGetCount": status,
        "devices": count.value,
    }
    # A device can enumerate yet fail on first use (a broken MIG slice returns
    # CUDA_ERROR_UNKNOWN), so create and release each primary context, as the tests do.
    context = 0
    if status == 0:
        driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        driver.cuDeviceGet.restype = ctypes.c_int
        driver.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        driver.cuDevicePrimaryCtxRetain.restype = ctypes.c_int
        driver.cuDevicePrimaryCtxRelease.argtypes = [ctypes.c_int]
        driver.cuDevicePrimaryCtxRelease.restype = ctypes.c_int
        for ordinal in range(count.value):
            device = ctypes.c_int()
            context = driver.cuDeviceGet(ctypes.byref(device), ordinal)
            if context == 0:
                handle = ctypes.c_void_p()
                context = driver.cuDevicePrimaryCtxRetain(ctypes.byref(handle), device)
                if context == 0:
                    context = driver.cuDevicePrimaryCtxRelease(device)
            if context != 0:
                result["context_device"] = ordinal
                break
        result["primary_context"] = context
    driver.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    driver.cuDriverGetVersion.restype = ctypes.c_int
    version = ctypes.c_int()
    if driver.cuDriverGetVersion(ctypes.byref(version)) == 0:
        result["driver_api_version"] = version.value
    failure = init or status or context
    if failure:
        for function, key in (("cuGetErrorName", "error_name"), ("cuGetErrorString", "error")):
            lookup = getattr(driver, function)
            lookup.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
            lookup.restype = ctypes.c_int
            value = ctypes.c_char_p()
            if lookup(failure, ctypes.byref(value)) == 0 and value.value:
                result[key] = value.value.decode("utf-8", errors="replace")
    return result


def cuda_usable(cuda: dict[str, object]) -> bool:
    """Whether cuda_status() found a device that initializes and creates a context."""
    return (
        cuda.get("cuInit") == 0
        and cuda.get("cuDeviceGetCount") == 0
        and cuda["devices"] > 0
        and cuda.get("primary_context") == 0
    )


def check_host(target: str, gpu: str) -> dict[str, object]:
    bootstrap._validate_host(target)
    cuda = cuda_status()
    print(json.dumps(cuda), flush=True)
    if gpu == "required" and not cuda_usable(cuda):
        raise CudaUnavailableError(
            f"GPU lane requires a working CUDA driver and a real device: {cuda}"
        )
    if gpu == "absent" and cuda["driver_loaded"]:
        raise RuntimeError(f"Driver-free lane must not load an NVIDIA GPU driver: {cuda}")
    if (
        gpu == "unavailable"
        and cuda["driver_loaded"]
        and not (
            cuda.get("cuInit") == 100  # CUDA_ERROR_NO_DEVICE
            or (
                cuda.get("cuInit") == 0
                and cuda.get("cuDeviceGetCount") == 0
                and cuda["devices"] == 0
            )
        )
    ):
        raise RuntimeError(f"CPU lane requires no usable CUDA device: {cuda}")
    return cuda


def host_diagnostics() -> dict[str, object]:
    """Capture bounded host evidence without changing a readiness verdict."""
    result = {
        "runner": os.environ.get("RUNNER_NAME"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
    }
    command = [
        "nvidia-smi",
        "--query-gpu=uuid,name,driver_version,pci.bus_id",
        "--format=csv,noheader",
    ]
    try:
        probe = subprocess.run(command, capture_output=True, text=True, timeout=10)
        result["nvidia_smi"] = {
            "returncode": probe.returncode,
            "stdout": probe.stdout[-16000:],
            "stderr": probe.stderr[-16000:],
        }
    except (OSError, subprocess.SubprocessError) as exc:
        result["nvidia_smi"] = {"unavailable": str(exc)}
    if os.name != "nt":
        for name, path in (
            ("uptime", "/proc/uptime"),
            ("kernel_driver", "/proc/driver/nvidia/version"),
        ):
            try:
                result[name] = Path(path).read_text()[:16000]
            except OSError as exc:
                result[name] = {"unavailable": str(exc)}
        devices = []
        for path in sorted(Path("/dev").glob("nvidia*")):
            try:
                info = path.stat()
                devices.append(
                    {
                        "path": str(path),
                        "mode": oct(info.st_mode),
                        "uid": info.st_uid,
                        "gid": info.st_gid,
                    }
                )
            except OSError as exc:
                devices.append({"path": str(path), "unavailable": str(exc)})
        result["device_nodes"] = devices
    return result


def check_compiler(cmake: Path, target: str, output: Path, deps: Path) -> None:
    source = output / "compiler-source"
    build = output / "compiler-build"
    source.mkdir(parents=True, exist_ok=True)
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(readiness LANGUAGES C CXX)\n"
        "set(CMAKE_C_STANDARD 11)\n"
        "set(CMAKE_C_STANDARD_REQUIRED ON)\n"
        "set(CMAKE_CXX_STANDARD 17)\n"
        "set(CMAKE_CXX_STANDARD_REQUIRED ON)\n"
        "add_executable(readiness main.c main.cpp)\n",
        encoding="utf-8",
    )
    (source / "main.c").write_text("int c_probe(void) { return 41; }\n", encoding="utf-8")
    (source / "main.cpp").write_text(
        '#include <string_view>\nextern "C" int c_probe(void);\n'
        'int main() { return c_probe() == 41 && std::string_view("ready").size() == 5 ? 0 : 1; }\n',
        encoding="utf-8",
    )
    command = [str(cmake), "-S", str(source), "-B", str(build)]
    if target == contract.DEFAULT_TARGET_PLATFORM:
        command += contract.compiler_arguments(deps, target)
        # The host Visual Studio generator is multi-config; the pinned MSVC builds through Ninja.
        multi_config = contract.compiler_selection() == "host"
        executable = build / ("Release" if multi_config else "") / "readiness.exe"
    else:
        command += [
            "-G",
            contract.LINUX_CMAKE_GENERATOR,
            "-DCMAKE_BUILD_TYPE=Release",
            *contract.compiler_arguments(deps, target),
        ]
        executable = build / "readiness"
    subprocess.run(command, check=True)
    subprocess.run([cmake, "--build", build, "--config", "Release", "--parallel", "2"], check=True)
    subprocess.run([executable], check=True)


def check_wheels(target: str, output: Path) -> None:
    build = output / "python" / "native"
    requirements = (
        bootstrap._write_requirements(
            build.parent / "locked-python-build-requirements.txt",
            contract._build_tools(target, include_viewer=False, include_cuda=False),
        ),
        bootstrap._write_requirements(
            build.parent / "locked-python-requirements.txt",
            contract._locked_runtime_wheels(target, include_viewer=False),
        ),
    )
    for path in requirements:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "download",
                "--disable-pip-version-check",
                "--only-binary=:all:",
                "--require-hashes",
                "--no-deps",
                "--no-cache-dir",
                "--dest",
                output / "wheelhouse",
                "-r",
                path,
            ],
            check=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", choices=("required", "unavailable", "absent"), required=True)
    parser.add_argument("--host-only", action="store_true")
    options = parser.parse_args()
    target = contract.DEFAULT_TARGET_PLATFORM if os.name == "nt" else contract.LINUX_X86_64_PLATFORM
    output = ROOT / "_build" / "ci-readiness"
    output.mkdir(parents=True, exist_ok=True)
    if options.gpu == "required":
        os.environ.setdefault("CUDA_LOG_FILE", str(output / "cuda-driver.log"))
    identity = contract.source_identity(ROOT)
    if identity is None:
        raise RuntimeError("Git could not report the source revision and status")
    report = {
        "source_revision": identity[0],
        "source_dirty": identity[1],
        "os": platform.platform(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "target_platform": target,
        "expected_gpu": options.gpu,
        "host_only": options.host_only,
        "checks": {},
    }

    def check(name, operation):
        try:
            result = operation()
        except (
            RuntimeError,
            OSError,
            ValueError,
            AttributeError,
            subprocess.SubprocessError,
            contract.BootstrapError,
        ) as exc:
            report["checks"][name] = {"status": "failed", "detail": str(exc)}
            if isinstance(exc, CudaUnavailableError):
                report["checks"][name]["failure_category"] = "cuda-driver"
                print(f"::error::{CUDA_FAILURE}: {exc}", flush=True)
            print(f"{name}: failed: {exc}", file=sys.stderr, flush=True)
            return False
        report["checks"][name] = {"status": "passed", "detail": result}
        print(f"{name}: passed", flush=True)
        return True

    check("host", lambda: check_host(target, options.gpu))
    if options.gpu == "required":
        report["host_diagnostics"] = host_diagnostics()
    if not options.host_only:
        deps = ROOT / "_build" / "target-deps"
        if check("cmake_fetch", lambda: bootstrap._ensure_cmake(deps, target)):
            cmake = contract._cmake_executable(deps, target)
            check("physx_fetch_and_hashes", lambda: bootstrap._ensure_physx(cmake, deps, target))
            if check("toolchain", lambda: str(bootstrap._validate_toolchain(target, deps))):
                check("compiler_and_sdk", lambda: check_compiler(cmake, target, output, deps))
        check("locked_wheel_downloads", lambda: check_wheels(target, output))
    path = output / "readiness.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write("## Runner readiness\n\n")
            stream.write(
                f"Attempt {os.environ.get('GITHUB_RUN_ATTEMPT', 'local')} on "
                f"`{os.environ.get('RUNNER_NAME', platform.node())}`. "
                "Host and driver diagnostics are retained with this attempt.\n\n"
            )
            stream.write("| Check | Result |\n| --- | --- |\n")
            for name, result in report["checks"].items():
                stream.write(f"| {name} | {result['status']} |\n")
    return int(any(result["status"] == "failed" for result in report["checks"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
