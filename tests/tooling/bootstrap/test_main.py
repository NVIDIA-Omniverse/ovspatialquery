# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BOOTSTRAP_ROOT = PROJECT_ROOT / "tools" / "bootstrap"
sys.path.insert(0, str(BOOTSTRAP_ROOT))

import contract  # noqa: E402
import main as bootstrap_main  # noqa: E402


# The checked-in stable plan; OVSPATIALQUERY_TRAIN=next swaps two wheels and paths.
@mock.patch.dict(os.environ, {"OVSPATIALQUERY_TRAIN": "stable"})
class VerificationScopeTests(unittest.TestCase):
    def test_scoped_commands_accept_explicit_library_selection(self) -> None:
        for command in ("plan", "bootstrap", "build", "test", "verify", "python-wheel"):
            with self.subTest(command=command):
                options = bootstrap_main._parser().parse_args([command, "--scope", "library"])
                self.assertEqual(options.scope, "library")

    @mock.patch.dict(os.environ, {"OVSPATIALQUERY_COMPILER": "host"})
    def test_cuda_library_plan_uses_pinned_nvrtc_without_viewer(self) -> None:
        for platform in contract.SUPPORTED_PLATFORMS:
            with self.subTest(platform=platform):
                scope = bootstrap_main._verification_scope(platform, "library-cuda")
                self.assertFalse(scope["include_viewer"])
                self.assertTrue(scope["include_cuda"])
                plan = contract.build_plan(
                    Path("venv"),
                    Path("native"),
                    Path("deps"),
                    platform,
                    include_viewer=False,
                    include_cuda=True,
                )
                build_names = {x["name"] for x in plan["python"]["build_tools"]}
                self.assertIn("nvidia-cuda-nvrtc-cu12", build_names)
                cuda_argument = next(
                    x
                    for x in plan["native"]["lifecycle"]["configure"]
                    if x.startswith("-DOVSPATIALQUERY_CUDA_NVRTC_DIR=")
                )
                self.assertIn("cuda_nvrtc", cuda_argument)
                with self.assertRaises(contract.BootstrapError):
                    bootstrap_main._require_native_scope(
                        {"OVSPATIALQUERY_CUDA_NVRTC_DIR": ""},
                        include_cuda=True,
                        target_platform=platform,
                    )

    def test_verify_rejects_different_custom_sdk_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "CMakeCache.txt").write_text(
                "OVSPATIALQUERY_PHYSX_SDK_DIR:PATH=" + str(root / "sdk-a") + "\n"
            )
            for selection in (str(root / "sdk-b"), ""):
                with mock.patch.dict(os.environ, {"OVSPATIALQUERY_PHYSX_SDK_DIR": selection}):
                    self.assertIn(
                        "Selected PhysX SDK differs",
                        bootstrap_main._install_issue(root, False, contract.LINUX_X86_64_PLATFORM),
                    )
            (root / "CMakeCache.txt").write_text(
                "OVSPATIALQUERY_PHYSX_SDK_DIR:PATH=" + str(Path.home() / "sdk-a") + "\n"
            )
            with mock.patch.dict(os.environ, {"OVSPATIALQUERY_PHYSX_SDK_DIR": "~/sdk-a"}):
                self.assertNotIn(
                    "Selected PhysX SDK differs",
                    bootstrap_main._install_issue(root, False, contract.LINUX_X86_64_PLATFORM),
                )

    def test_custom_physx_selection_and_reset_reach_cmake(self) -> None:
        for selected in ("custom-sdk", ""):
            with (
                self.subTest(selected=selected),
                mock.patch.dict(os.environ, {"OVSPATIALQUERY_PHYSX_SDK_DIR": selected}),
            ):
                plan = contract.build_plan(
                    Path("venv"),
                    Path("native"),
                    Path("deps"),
                    contract.LINUX_X86_64_PLATFORM,
                    include_viewer=False,
                )
                expected = str(Path(selected).resolve()) if selected else ""
                self.assertIn(
                    "-DOVSPATIALQUERY_PHYSX_SDK_DIR=" + expected,
                    plan["native"]["lifecycle"]["configure"],
                )
                if selected:
                    with self.assertRaises(contract.BootstrapError) as caught:
                        bootstrap_main._ensure_physx(
                            Path("not-executed-cmake"), Path("deps"), contract.LINUX_X86_64_PLATFORM
                        )
                    self.assertEqual(caught.exception.code, "custom_physx_invalid")

    def test_auto_keeps_windows_cuda_and_selects_available_sources(self) -> None:
        scope = bootstrap_main._verification_scope(contract.DEFAULT_TARGET_PLATFORM, "auto")

        self.assertEqual(scope["include_viewer"], contract.APP_ROOT.exists())
        self.assertTrue(scope["include_cuda"])
        self.assertEqual(scope["gpu_runtime"]["state"], "required")

    # Linux aarch64 has no pinned compiler.
    @mock.patch.dict(os.environ, {"OVSPATIALQUERY_COMPILER": "host"})
    def test_native_configure_records_the_source_revision(self) -> None:
        commit = "0123456789abcdef0123456789abcdef01234567"
        for identity, expected in (
            ((commit, False), commit),
            ((commit, True), f"{commit}-dirty"),
            (None, "unknown"),
        ):
            with (
                self.subTest(identity=identity),
                mock.patch.object(contract, "source_identity", return_value=identity),
            ):
                for platform in contract.SUPPORTED_PLATFORMS:
                    configure = contract.native_lifecycle(
                        Path("native"), Path("deps"), platform, python=Path("venv/bin/python")
                    )["configure"]
                    self.assertIn(f"-DOVSPATIALQUERY_SOURCE_REVISION={expected}", configure)

    def test_linux_configure_selects_the_pinned_clang_or_the_host_compiler(self) -> None:
        def compilers(
            environment: dict[str, str], platform: str = contract.LINUX_X86_64_PLATFORM
        ) -> list[str]:
            with mock.patch.dict(os.environ, environment):
                if "OVSPATIALQUERY_COMPILER" not in environment:
                    # The default must not depend on the caller's environment.
                    os.environ.pop("OVSPATIALQUERY_COMPILER", None)
                configure = contract.native_lifecycle(
                    Path("native"),
                    Path("deps"),
                    platform,
                    python=Path("venv/bin/python"),
                )["configure"]
            return [argument for argument in configure if "_COMPILER=" in argument]

        clang = Path("deps") / "clang" / "bin"
        self.assertEqual(
            compilers({}),
            [f"-DCMAKE_C_COMPILER={clang / 'clang'}", f"-DCMAKE_CXX_COMPILER={clang / 'clang++'}"],
        )
        # A host build leaves the choice to CMake, which honors CC and CXX.
        self.assertEqual(compilers({"OVSPATIALQUERY_COMPILER": "host"}), [])
        with self.assertRaises(contract.BootstrapError):
            compilers({"OVSPATIALQUERY_COMPILER": "gcc"})
        # Linux aarch64 pins no compiler: the default fails until the host compiler is named.
        aarch64 = contract.LINUX_AARCH64_PLATFORM
        with self.assertRaises(contract.BootstrapError) as unavailable:
            compilers({}, aarch64)
        self.assertEqual(unavailable.exception.code, "native_compiler_unavailable")
        self.assertEqual(compilers({"OVSPATIALQUERY_COMPILER": "host"}, aarch64), [])

    def test_failed_captured_command_reports_stdout_and_stderr(self) -> None:
        # CTest names the failed tests on stdout and writes only a summary to stderr. The
        # texts are joined at run time so the echoed command line cannot contain them.
        script = "import sys; print('FAILED' + ':cuda_test'); sys.exit('Errors while' + ' running CTest')"
        with self.assertRaises(contract.BootstrapError) as failed:
            bootstrap_main._run(
                [sys.executable, "-c", script],
                code="native_tests_failed",
                action="fix",
                capture=True,
            )
        self.assertIn("FAILED:cuda_test", str(failed.exception))
        self.assertIn("Errors while running CTest", str(failed.exception))

    @unittest.skipUnless(
        contract.APP_PROJECT.is_file(), "Viewer metadata is absent from this source export"
    )
    def test_app_lock_has_one_wheel_per_pin_on_every_platform(self) -> None:
        # uv.lock is hand-edited (a full `uv lock` drops the NVIDIA-index ovRTX wheels), and
        # only x86-64 CI runs this, so a missing aarch64 wheel would otherwise surface on ARM.
        # The selection raises unless every pin has one wheel at its version for the platform.
        for platform in contract.SUPPORTED_PLATFORMS:
            with (
                self.subTest(platform=platform),
                mock.patch.dict(os.environ, {contract.TRAIN_ENVIRONMENT: "stable"}),
            ):
                contract._locked_runtime_wheels(platform)

    def test_public_source_refuses_native_aarch64_builds(self) -> None:
        # Without the app's lock, the public library lock has no complete aarch64 entry.
        with mock.patch.object(contract, "APP_PROJECT", Path("absent") / "pyproject.toml"):
            with self.assertRaises(contract.BootstrapError) as refused:
                bootstrap_main._validate_host(contract.LINUX_AARCH64_PLATFORM)
        self.assertEqual(refused.exception.code, "platform_unsupported")

    def test_windows_configure_selects_the_pinned_msvc_or_visual_studio(self) -> None:
        def generator(selection: str) -> list[str]:
            with mock.patch.dict(
                os.environ,
                {"OVSPATIALQUERY_COMPILER": selection, "OVSPATIALQUERY_NATIVE_CONFIG": "Release"},
            ):
                configure = contract.native_lifecycle(
                    Path("native"),
                    Path("deps"),
                    contract.DEFAULT_TARGET_PLATFORM,
                    python=Path("venv/Scripts/python.exe"),
                )["configure"]
            start = configure.index("-G")
            return configure[start : start + 6]

        self.assertEqual(
            generator("pinned"),
            [
                "-G",
                "Ninja",
                f"-DCMAKE_MAKE_PROGRAM={Path('deps') / 'ninja' / 'ninja.exe'}",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_C_COMPILER=cl",
                "-DCMAKE_CXX_COMPILER=cl",
            ],
        )
        self.assertEqual(
            generator("host"),
            ["-G", contract.CMAKE_GENERATOR, "-A", "x64", "-T", contract.CMAKE_TOOLSET],
        )

    def test_pinned_msvc_environment_comes_from_the_staged_packages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            deps = Path(temporary)
            tools = deps / "msvc" / "VC" / "Tools" / "MSVC" / "14.44.35207"
            sdk = deps / "winsdk"
            for directory in (
                tools / "bin" / "Hostx64" / "x64",
                sdk / "Include" / "ucrt",
                sdk / "Lib" / "um" / "x64",
                sdk / "bin" / "x64",
                deps / "ninja",
            ):
                directory.mkdir(parents=True)
            for path in (
                tools / "bin" / "Hostx64" / "x64" / "cl.exe",
                sdk / "bin" / "x64" / "rc.exe",
                deps / "ninja" / "ninja.exe",
            ):
                path.touch()
            with mock.patch.dict(os.environ, {"OVSPATIALQUERY_COMPILER": "pinned", "PATH": "host"}):
                self.assertIsNone(
                    contract._staged_compiler_issue(deps, contract.DEFAULT_TARGET_PLATFORM)
                )
                environment = contract.compiler_environment(deps, contract.DEFAULT_TARGET_PLATFORM)
                self.assertEqual(
                    environment["PATH"].split(os.pathsep),
                    [
                        str(tools / "bin" / "Hostx64" / "x64"),
                        str(sdk / "bin" / "x64"),
                        str(deps / "ninja"),
                        "host",
                    ],
                )
                self.assertIn(
                    str(sdk / "Include" / "ucrt"),
                    environment["INCLUDE"].split(os.pathsep),
                )
                self.assertIn(
                    str(sdk / "Lib" / "um" / "x64"),
                    environment["LIB"].split(os.pathsep),
                )
                (deps / "ninja" / "ninja.exe").unlink()
                self.assertIn(
                    "ninja.exe is missing",
                    contract._staged_compiler_issue(deps, contract.DEFAULT_TARGET_PLATFORM) or "",
                )

    def test_pinned_clang_must_match_its_content_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            deps = Path(temporary) / "deps"
            binaries = deps / "clang" / "bin"
            binaries.mkdir(parents=True)
            (binaries / "clang-14").write_bytes(b"locked compiler")
            lock = Path(temporary) / "lock.json"
            digest = "2e1e1d71d4f3d0ad1d6fa2a6c3ad7c3e4d7e0a29d1c5ab9d6a5ab8bb7e5e5c9e"
            lock.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "platforms": {
                            contract.LINUX_X86_64_PLATFORM: {"files": {"bin/clang-14": digest}}
                        },
                    }
                )
            )
            with (
                mock.patch.object(
                    contract, "_pinned_clang", return_value=(binaries / "clang-14",) * 2
                ),
                mock.patch.object(contract, "COMPILER_CONTENT_LOCK", lock),
                mock.patch.dict(os.environ, {"OVSPATIALQUERY_COMPILER": "pinned"}),
            ):
                issue = contract._staged_compiler_issue(deps, contract.LINUX_X86_64_PLATFORM)
                self.assertIn("content mismatch for bin/clang-14", issue or "")
                lock.write_text(
                    lock.read_text().replace(
                        digest, contract.hashlib.sha256(b"locked compiler").hexdigest()
                    )
                )
                self.assertIsNone(
                    contract._staged_compiler_issue(deps, contract.LINUX_X86_64_PLATFORM)
                )

    def test_pinned_clang_drivers_resolve_to_the_locked_binary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            deps = Path(temporary) / "deps"
            binaries = deps / "clang" / "bin"
            binaries.mkdir(parents=True)
            (binaries / "clang-14").write_bytes(b"locked compiler")
            try:
                (binaries / "clang").symlink_to("clang-14")
                (binaries / "clang++").symlink_to("clang")
            except OSError as error:
                self.skipTest(f"Compiler-alias test requires symlink support: {error}")
            with mock.patch.dict(os.environ, {"OVSPATIALQUERY_COMPILER": "pinned"}):
                (binaries / "other").write_bytes(b"another compiler")
                (binaries / "clang").unlink()
                (binaries / "clang").symlink_to("other")
                self.assertIn(
                    "clang does not resolve",
                    contract._staged_compiler_issue(deps, contract.LINUX_X86_64_PLATFORM) or "",
                )
                (binaries / "clang").unlink()
                (binaries / "clang").symlink_to("clang-14")
                (binaries / "clang++").unlink()
                (binaries / "clang++").symlink_to("other")
                self.assertIn(
                    "clang++ does not resolve",
                    contract._staged_compiler_issue(deps, contract.LINUX_X86_64_PLATFORM) or "",
                )

    def test_a_changed_compiler_selection_reconfigures_from_scratch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            configure = ["cmake", "-S", "source", "-B", temporary, "-DX=1"]
            self.assertEqual(bootstrap_main._fresh_if_compiler_changed(configure), configure)
            cache = Path(temporary) / "CMakeCache.txt"
            cache.write_text("OVSPATIALQUERY_COMPILER_SELECTION:STRING=pinned\n")
            with mock.patch.dict(os.environ, {"OVSPATIALQUERY_COMPILER": "pinned"}):
                self.assertEqual(bootstrap_main._fresh_if_compiler_changed(configure), configure)
            with mock.patch.dict(os.environ, {"OVSPATIALQUERY_COMPILER": "host"}):
                self.assertEqual(
                    bootstrap_main._fresh_if_compiler_changed(configure),
                    ["cmake", "--fresh", *configure[1:]],
                )

    def test_library_scope_rejects_a_stale_cuda_configuration(self) -> None:
        with self.assertRaises(contract.BootstrapError) as raised:
            bootstrap_main._require_native_scope(
                {"OVSPATIALQUERY_CUDA_NVRTC_DIR": "/stale/ovrtx"},
                include_cuda=False,
                target_platform=contract.DEFAULT_TARGET_PLATFORM,
            )

        self.assertEqual(raised.exception.code, "native_scope_mismatch")

    @unittest.skipUnless(
        contract.APP_PROJECT.is_file(), "Viewer metadata is absent from this source export"
    )
    def test_python_probe_names_a_narrower_bootstrap_scope(self) -> None:
        def probe(missing: list[str]) -> contract.BootstrapError:
            report = {
                "implementation": "CPython",
                "version": "3.12.13",
                "bits": 64,
                "versions": {},
                "missing": missing,
                "unexpected_versions": {},
                "module_paths": {},
            }
            with (
                mock.patch.object(
                    bootstrap_main,
                    "_run",
                    return_value=mock.Mock(stdout=json.dumps(report)),
                ),
                self.assertRaises(contract.BootstrapError) as raised,
            ):
                bootstrap_main._probe_python(Path("python"), "manylinux_2_35_x86_64")
            return raised.exception

        scope = probe(["ovspatialqueryapp", "ovrtx", "wheel-stub"])
        self.assertEqual(scope.code, "python_scope_mismatch")
        self.assertIn("full Viewer scope", str(scope))
        self.assertIn("--scope library", scope.action)
        # A missing package every scope needs is a broken venv, not a scope choice.
        broken = probe(["numpy", "ovrtx"])
        self.assertEqual(broken.code, "python_version_drift")

    @mock.patch.dict(os.environ, {"OVSPATIALQUERY_COMPILER": "host"})
    def test_library_plan_omits_viewer_and_clears_cuda_cache(self) -> None:
        test_root = PROJECT_ROOT / "_build" / "tests"
        test_root.mkdir(parents=True, exist_ok=True)
        with (
            tempfile.TemporaryDirectory(dir=test_root) as temporary,
            mock.patch.object(
                bootstrap_main,
                "_gpu_runtime_status",
                side_effect=AssertionError("library scope must not probe a GPU"),
            ),
        ):
            root = Path(temporary)
            stdout = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
                result = bootstrap_main.main(
                    [
                        "plan",
                        "--platform",
                        contract.DEFAULT_TARGET_PLATFORM,
                        "--scope",
                        "library",
                        "--venv",
                        str(root / "venv"),
                        "--native-build",
                        str(root / "native"),
                        "--target-deps",
                        str(root / "deps"),
                    ]
                )

        self.assertEqual(result, 0)
        plan = json.loads(stdout.getvalue())
        self.assertFalse(plan["python"]["install_viewer"])
        runtime_names = {package["name"] for package in plan["python"]["runtime_wheels"]}
        self.assertTrue(contract.VIEWER_ONLY_RUNTIME_PACKAGES.isdisjoint(runtime_names))
        self.assertNotIn("pillow", runtime_names)
        build_names = {package["name"] for package in plan["python"]["build_tools"]}
        self.assertNotIn("wheel-stub", build_names)
        self.assertIn(
            "-DOVSPATIALQUERY_CUDA_NVRTC_DIR=",
            plan["native"]["lifecycle"]["configure"],
        )
        self.assertEqual(plan["verification_scope"]["requested"], "library")


if __name__ == "__main__":
    unittest.main()
