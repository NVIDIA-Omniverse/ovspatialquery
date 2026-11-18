# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Development wheels name their commit; injected release versions stay exact."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/bootstrap"))
import contract  # noqa: E402
import wheel_smoke  # noqa: E402
from pip._vendor.packaging.version import Version  # noqa: E402

COMMIT = "2147f83e38f4c0ffee0123456789abcdef012345"


class WheelVersionTests(unittest.TestCase):
    def wheel_version(
        self, dirty: bool, injected: str = "", branch: str = "", stamp=None, tag=None
    ) -> str:
        # CI passes an empty OVSPATIALQUERY_BUILD_VERSION outside release builds. The
        # checkout's own stamp or tag must not leak in: an exported source carries a stamp.
        with (
            mock.patch.dict(
                os.environ,
                {
                    "OVSPATIALQUERY_BUILD_VERSION": injected,
                    "OVSPATIALQUERY_BUILD_BRANCH": branch,
                },
            ),
            mock.patch.object(contract, "source_version_stamp", return_value=stamp),
            mock.patch.object(contract, "release_tag_version", return_value=tag),
            mock.patch.object(contract, "python_projection_version", return_value="0.0.3.dev0"),
        ):
            return contract.wheel_version(COMMIT, dirty)

    def test_development_wheel_names_its_commit(self) -> None:
        self.assertEqual(self.wheel_version(False), "0.0.3.dev0+g2147f83e38f4")
        self.assertEqual(self.wheel_version(True), "0.0.3.dev0+g2147f83e38f4.dirty")

    def test_injected_release_version_stays_exact(self) -> None:
        self.assertEqual(self.wheel_version(True, injected="0.0.3"), "0.0.3")
        self.assertEqual(self.wheel_version(False, injected="0.0.3.dev1"), "0.0.3.dev1")
        self.assertEqual(self.wheel_version(True, injected="0.1.0rc1"), "0.1.0rc1")

    def test_stamped_or_tagged_release_version_stays_exact(self) -> None:
        self.assertEqual(self.wheel_version(False, stamp="0.1.0rc10"), "0.1.0rc10")
        self.assertEqual(self.wheel_version(False, tag="0.1.0"), "0.1.0")
        # The override still labels a branch's development bundle over a stamp.
        self.assertEqual(
            self.wheel_version(False, injected="0.1.0.dev2", branch="main", stamp="0.1.0rc10"),
            "0.1.0.dev2+main.2147f83e38f4",
        )

    def test_branch_development_wheel_names_branch_and_source(self) -> None:
        self.assertEqual(
            self.wheel_version(False, injected="0.1.0.dev237", branch="experimental/Queries"),
            "0.1.0.dev237+experimental.queries.2147f83e38f4",
        )
        self.assertEqual(
            self.wheel_version(False, injected="0.1.0.dev237", branch="main"),
            "0.1.0.dev237+main.2147f83e38f4",
        )

    def test_branch_label_survives_version_normalization(self) -> None:
        # setuptools names the wheel, and pip resolves a pinned version, by the normalized
        # form; Artifactory folders and the published() lookup use the computed one.
        for branch, revision, expected in (
            ("release/0.1", COMMIT, "0.1.0.dev237+release.0.1.2147f83e38f4"),
            ("release/1.10.2", COMMIT, "0.1.0.dev237+release.1.10.2.2147f83e38f4"),
            ("feature/001", COMMIT, "0.1.0.dev237+feature.1.2147f83e38f4"),
            (
                "main",
                "000000000000a123456789abcdef0123456789ab",
                "0.1.0.dev237+main.000000000000a",
            ),
        ):
            with self.subTest(branch=branch):
                version = contract.release_wheel_version("0.1.0.dev237", revision, branch)
                self.assertEqual(version, expected)
                self.assertEqual(str(Version(version)), version)

    def test_isolated_consumer_ignores_the_label_when_comparing_the_native_getter(self) -> None:
        for package_version in (
            "0.0.3",
            "0.0.3.dev0",
            "0.0.3rc1",
            "0.0.3rc1+g2147f83e38f4.dirty",
            "0.0.3.dev0+g2147f83e38f4.dirty",
            "0.0.3+g2147f83e38f4",
        ):
            wheel_smoke.require_matching_versions(package_version, (0, 0, 3))
        for package_version, native_version in (
            ("0.0.2.dev0+g2147f83e38f4", (0, 0, 3)),
            ("0.0.3.dev0", (0, 0, 2)),
            ("0.0.3rc1", (0, 1, 0)),
        ):
            with self.subTest(package=package_version, native=native_version):
                with self.assertRaisesRegex(RuntimeError, "native and wheel versions disagree"):
                    wheel_smoke.require_matching_versions(package_version, native_version)


if __name__ == "__main__":
    unittest.main()
