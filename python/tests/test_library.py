# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from ovspatialquery.library import Library, LibraryLoadError


class _Function:
    def __init__(self, result=None) -> None:
        self.argtypes = None
        self.restype = None
        self._result = result

    def __call__(self, *args):
        if self._result is not None:
            return self._result(*args)
        return None


class _NativeLibrary:
    def __init__(self, *, companion: bool = True, explicit: bool = True) -> None:
        from ovspatialquery import raw

        for name in raw.CORE_EXPORT_SPECS:
            setattr(self, name, _Function())
        if companion:
            for name in raw.OVSTAGE_EXPORT_SPECS:
                setattr(self, name, _Function())
        if explicit:
            for name in raw.EXPLICIT_EXPORT_SPECS:
                setattr(self, name, _Function())

        def version(major, minor, patch):
            major._obj.value = 0
            minor._obj.value = 1
            patch._obj.value = 1
            return raw.Result(raw.ApiStatus.SUCCESS)

        self.ovspatialquery_get_version = _Function(version)


class LibraryResolutionTests(unittest.TestCase):
    def test_explicit_path_precedes_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            explicit = root / "explicit.dll"
            environment = root / "environment.dll"
            explicit.touch()
            environment.touch()
            with (
                mock.patch.dict(os.environ, {"OVSPATIALQUERY_LIBRARY": str(environment)}),
                mock.patch(
                    "ovspatialquery.library.ctypes.CDLL",
                    return_value=_NativeLibrary(),
                ) as load,
            ):
                library = Library.open(explicit)
                self.addCleanup(library.close)
            load.assert_called_once_with(str(explicit.resolve()))
            self.assertEqual(library.path, explicit.resolve())

    def test_environment_precedes_bundled_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            native = Path(directory) / "environment.dll"
            native.touch()
            with (
                mock.patch.dict(os.environ, {"OVSPATIALQUERY_LIBRARY": str(native)}),
                mock.patch(
                    "ovspatialquery.library.ctypes.CDLL",
                    return_value=_NativeLibrary(),
                ) as load,
            ):
                library = Library.open()
                self.addCleanup(library.close)
            load.assert_called_once_with(str(native.resolve()))

    def test_missing_explicit_path_fails_without_ambient_search(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch("ovspatialquery.library.ctypes.CDLL") as load,
        ):
            missing = Path(directory) / "missing.dll"
            with self.assertRaisesRegex(LibraryLoadError, "does not exist"):
                Library.open(missing)
        load.assert_not_called()

    def test_optional_export_groups_are_lazy_and_fail_closed(self) -> None:
        for missing, present, require, message in (
            ("companion", "has_ovstage", "require_ovstage", "Ovstage companion"),
            ("explicit", "has_explicit", "require_explicit", "explicit ingestion"),
        ):
            with self.subTest(group=missing), tempfile.TemporaryDirectory() as directory:
                native = Path(directory) / "core.dll"
                native.touch()
                with mock.patch(
                    "ovspatialquery.library.ctypes.CDLL",
                    return_value=_NativeLibrary(**{missing: False}),
                ):
                    library = Library.open(native)
                    self.addCleanup(library.close)
                    self.assertFalse(getattr(library, present))
                    with self.assertRaisesRegex(LibraryLoadError, message):
                        getattr(library, require)()
                    with (
                        mock.patch.object(Library, "close", autospec=True) as close,
                        self.assertRaisesRegex(LibraryLoadError, message),
                    ):
                        Library.open(native, **{require: True})
                    close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
