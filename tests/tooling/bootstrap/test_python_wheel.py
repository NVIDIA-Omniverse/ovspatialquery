# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import re
import sys
import tempfile
import tomllib
import unittest
from unittest import mock
from pathlib import Path
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BOOTSTRAP_ROOT = PROJECT_ROOT / "tools" / "bootstrap"
sys.path.insert(0, str(BOOTSTRAP_ROOT))

import python_wheel  # noqa: E402
import consumer_content  # noqa: E402

VERSION = python_wheel._project(PROJECT_ROOT / "python")["version"]


class PythonWheelTests(unittest.TestCase):
    def _temporary_directory(self) -> tempfile.TemporaryDirectory[str]:
        test_root = PROJECT_ROOT / "_build" / "tests"
        test_root.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(dir=test_root)

    def _fake_install(
        self,
        root: Path,
        *,
        linux: bool = False,
        cuda: bool = True,
        version: str = VERSION,
        revision: str = "revision",
    ) -> Path:
        install = root / "install"
        native = (
            ("lib/libovspatialquery.so", b"native-so")
            if linux
            else ("bin/ovspatialquery.dll", b"native-dll")
        )
        files = [
            native,
            ("LICENSE", b"project-license"),
            ("NOTICE", b"redistribution-notice"),
            ("include/ovspatialquery/ovspatialquery.h", b"core-header"),
            (
                "include/ovspatialquery/ovspatialquery_ovstage.h",
                b"companion-header",
            ),
            ("include/dlpack/dlpack.h", b"dlpack-header"),
            ("licenses/dlpack-LICENSE.txt", b"dlpack-license"),
            ("licenses/physx-LICENSE.md", b"physx-license"),
            (
                "licenses/physx-PROVENANCE.json",
                b'{"component":"NVIDIA PhysX SDK","package":"test-sdk","license_expression":"Apache-2.0"}',
            ),
        ]
        if cuda:
            ptx_dir = "lib" if linux else "bin"
            files.append((f"{ptx_dir}/ovspatialquery_cuda.ptx", b"cuda-ptx"))
        for relative, payload in files:
            destination = install / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        for name, payload in consumer_content.collect(
            PROJECT_ROOT, version=version, source_revision=revision
        ).items():
            target = install / consumer_content.SDK_CONTENT / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        return install

    def test_windows_wheel_is_platform_tagged_and_self_describing(self) -> None:
        with self._temporary_directory() as temporary:
            root = Path(temporary)
            artifact = python_wheel.build(
                source_root=PROJECT_ROOT / "python",
                install_root=self._fake_install(root, revision="0123456789abcdef-dirty"),
                destination=root / "dist",
                target_platform="windows-x86_64",
                source_revision="0123456789abcdef",
                source_dirty=True,
            )

            self.assertEqual(
                artifact.path.name,
                f"ovspatialquery-{VERSION}-py3-none-win_amd64.whl",
            )
            names, manifest = self._inspect(artifact.path)
            self.assertEqual(manifest, artifact.manifest)
            with zipfile.ZipFile(artifact.path) as archive:
                self.assertIn("ovspatialquery/_native/bin/ovspatialquery.dll", names)
                self.assertIn("ovspatialquery/_native/bin/ovspatialquery_cuda.ptx", names)
                self.assertNotIn("ovspatialquery/_native/lib/libovspatialquery.so", names)
                self.assertIn("ovspatialquery/_native/licenses/physx-LICENSE.md", names)
                self.assertIn("ovspatialquery/_native/licenses/physx-PROVENANCE.json", names)
                self.assertIn(
                    f"ovspatialquery-{VERSION}.dist-info/licenses/physx-LICENSE.md", names
                )
                self.assertEqual(manifest["tag"], "py3-none-win_amd64")
                self.assertEqual(manifest["source_revision"], "0123456789abcdef")
                self.assertIs(manifest["source_dirty"], True)
                self.assertNotIn(str(root), json.dumps(manifest))
                wheel_metadata = archive.read(f"ovspatialquery-{VERSION}.dist-info/WHEEL").decode()
                self.assertIn("Root-Is-Purelib: false", wheel_metadata)
                self.assertIn("Tag: py3-none-win_amd64", wheel_metadata)
                metadata = archive.read(f"ovspatialquery-{VERSION}.dist-info/METADATA").decode()
                self.assertIn(f"\nVersion: {VERSION}\n", metadata)
                pyproject = (PROJECT_ROOT / "python/pyproject.toml").read_text()
                project = tomllib.loads(pyproject)["project"]
                self.assertIn(f"\nRequires-Python: {project['requires-python']}\n", metadata)
                # PyPI renders the project page from these: the summary, the README as the
                # Markdown body, and the sidebar links.
                header, _, body = metadata.partition("\n\n")
                self.assertIn(f"\nSummary: {project['description']}\n", header)
                self.assertIn("\nDescription-Content-Type: text/markdown", header)
                self.assertEqual(body, (PROJECT_ROOT / "python" / project["readme"]).read_text())
                for label, url in project["urls"].items():
                    self.assertTrue(url.startswith("https://"), url)
                    self.assertIn(f"\nProject-URL: {label}, {url}\n", header)
                for classifier in project["classifiers"]:
                    self.assertIn(f"\nClassifier: {classifier}\n", header)
                # The wheel has no runtime dependencies, and the source project agrees.
                self.assertNotIn("dependencies", project)
                self.assertNotIn("Requires-Dist:", metadata)
                self.assertIn(
                    f"\nLicense-Expression: {python_wheel.LICENSE_EXPRESSION}\n", metadata
                )
                for name in ("LICENSE", "NOTICE", "dlpack-LICENSE.txt", "physx-LICENSE.md"):
                    self.assertIn(f"License-File: {name}\n", metadata)
                    self.assertIn(f"ovspatialquery-{VERSION}.dist-info/licenses/{name}", names)

    def test_linux_wheel_uses_manylinux_tag_and_omits_absent_ptx(self) -> None:
        with self._temporary_directory() as temporary:
            root = Path(temporary)
            artifact = python_wheel.build(
                source_root=PROJECT_ROOT / "python",
                install_root=self._fake_install(
                    root,
                    linux=True,
                    cuda=False,
                ),
                destination=root / "dist",
                target_platform="manylinux_2_35_x86_64",
                source_revision="revision",
                source_dirty=False,
            )
            self.assertEqual(
                artifact.path.name,
                f"ovspatialquery-{VERSION}-py3-none-manylinux_2_35_x86_64.whl",
            )
            names, manifest = self._inspect(artifact.path)
            self.assertIn("ovspatialquery/_native/lib/libovspatialquery.so", names)
            self.assertNotIn("ovspatialquery/_native/bin/ovspatialquery_cuda.ptx", names)
            self.assertFalse(manifest["capabilities"]["cuda_ptx_bundled"])

    def test_arm_wheel_records_target_architecture_and_bundles_cuda_payload(self) -> None:
        with self._temporary_directory() as temporary:
            root = Path(temporary)
            artifact = python_wheel.build(
                source_root=PROJECT_ROOT / "python",
                install_root=self._fake_install(root, linux=True, revision="arm-source"),
                destination=root / "dist",
                target_platform="manylinux_2_35_aarch64",
                source_revision="arm-source",
                source_dirty=False,
            )
            names, manifest = self._inspect(artifact.path)
            self.assertTrue(artifact.path.name.endswith("-manylinux_2_35_aarch64.whl"))
            self.assertEqual(manifest["target_platform"], "manylinux_2_35_aarch64")
            self.assertIn("ovspatialquery/_native/lib/ovspatialquery_cuda.ptx", names)
            self.assertTrue(manifest["capabilities"]["cuda_ptx_bundled"])

    def test_missing_verified_native_input_fails_closed(self) -> None:
        with self._temporary_directory() as temporary:
            root = Path(temporary)
            install = self._fake_install(root)
            (install / "bin" / "ovspatialquery.dll").unlink()
            with self.assertRaisesRegex(
                python_wheel.WheelBuildError,
                "native artifact",
            ):
                python_wheel.build(
                    source_root=PROJECT_ROOT / "python",
                    install_root=install,
                    destination=root / "dist",
                    target_platform="windows-x86_64",
                    source_revision="revision",
                    source_dirty=False,
                )

    def test_explicit_version_identifies_skill_without_build_environment(self) -> None:
        version = "0.1.0rc1"
        with (
            self._temporary_directory() as temporary,
            mock.patch.dict("os.environ", {}, clear=True),
        ):
            root = Path(temporary)
            artifact = python_wheel.build(
                source_root=PROJECT_ROOT / "python",
                install_root=self._fake_install(root, version=version),
                destination=root / "dist",
                target_platform="windows-x86_64",
                source_revision="revision",
                source_dirty=False,
                version=version,
            )
            with zipfile.ZipFile(artifact.path) as archive:
                readme = archive.read("ovspatialquery/README.txt").decode()
                skill = archive.read("ovspatialquery/skills/ovspatialquery/SKILL.md")
            self.assertIn(
                f"skills/ovspatialquery: version {version}, source revision revision.", readme
            )
            self.assertEqual(skill, (PROJECT_ROOT / "skills/ovspatialquery/SKILL.md").read_bytes())

    def test_shipped_native_lessons_find_every_local_header(self) -> None:
        # Lesson 11 builds from the installed wheel with ../native and ../../support/native on
        # its include path, so each quoted include must ship in one of those directories.
        for language in ("wheel", "sdk"):
            content = consumer_content.collect(PROJECT_ROOT, language)
            lessons = [name for name in content if name.startswith("examples/native/")]
            self.assertTrue(any(name.endswith(".cpp") for name in lessons), language)
            for name in lessons:
                for header in re.findall(r'^#include "([^"]+)"', content[name].decode(), re.M):
                    self.assertTrue(
                        f"examples/native/{header}" in content
                        or f"support/native/{header}" in content,
                        f"{language}: {name} includes {header}, which does not ship",
                    )

    def _inspect(self, path: Path) -> tuple[set[str], dict[str, object]]:
        """Check RECORD and the provenance identities; return the names and the manifest."""
        with zipfile.ZipFile(path) as archive:
            self._verify_record(archive)
            manifest = json.loads(archive.read("ovspatialquery/_native/provenance.json"))
            revision = manifest["source_revision"] + ("-dirty" if manifest["source_dirty"] else "")
            for name, payload in consumer_content.collect(
                PROJECT_ROOT, "wheel", version=manifest["version"], source_revision=revision
            ).items():
                self.assertEqual(archive.read(f"ovspatialquery/{name}"), payload)
            for identity in (*manifest["source_files"], *manifest["native_artifacts"]):
                payload = archive.read(identity["wheel_path"])
                self.assertEqual(hashlib.sha256(payload).hexdigest(), identity["sha256"])
                self.assertEqual(len(payload), identity["size"])
            return set(archive.namelist()), manifest

    def _verify_record(self, archive: zipfile.ZipFile) -> None:
        record_name = f"ovspatialquery-{VERSION}.dist-info/RECORD"
        rows = csv.reader(io.StringIO(archive.read(record_name).decode("utf-8")))
        observed = {row[0]: row[1:] for row in rows}
        self.assertEqual(set(observed), set(archive.namelist()))
        self.assertEqual(observed[record_name], ["", ""])
        for name, (digest, size) in observed.items():
            if name == record_name:
                continue
            payload = archive.read(name)
            expected = (
                base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
                .rstrip(b"=")
                .decode("ascii")
            )
            self.assertEqual(digest, f"sha256={expected}")
            self.assertEqual(int(size), len(payload))


if __name__ == "__main__":
    unittest.main()
