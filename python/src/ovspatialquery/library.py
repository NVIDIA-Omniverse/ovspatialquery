# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic native-library loading and exact export qualification."""

from __future__ import annotations
import ctypes
import os
from pathlib import Path
import sys
from types import TracebackType
from typing import Any, Final
from . import raw

LIBRARY_ENV: Final = "OVSPATIALQUERY_LIBRARY"
SUPPORTED_VERSION_FAMILY: Final = (0, 1)


class LibraryLoadError(RuntimeError):
    """The selected native artifact cannot satisfy the public interface."""


def _bundled_artifact() -> Path:
    package_root = Path(__file__).resolve().parent
    if sys.platform == "win32":
        return package_root / "_native" / "bin" / "ovspatialquery.dll"
    if sys.platform.startswith("linux"):
        return package_root / "_native" / "lib" / "libovspatialquery.so"
    raise LibraryLoadError(f"ovspatialquery is not qualified on {sys.platform}")


def _resolve_path(path: Path | str | None) -> Path:
    if path is not None:
        selected = Path(path).expanduser()
        authority = "explicit"
    else:
        environment = os.environ.get(LIBRARY_ENV)
        if environment:
            selected = Path(environment).expanduser()
            authority = LIBRARY_ENV
        else:
            selected = _bundled_artifact()
            authority = "bundled wheel artifact"
    resolved = selected.resolve()
    if not resolved.is_file():
        raise LibraryLoadError(f"{authority} ovspatialquery library does not exist: {resolved}")
    return resolved


class Library:
    """One qualified native artifact and its retained platform loader state."""

    def __init__(
        self,
        path: Path,
        native: Any,
        exports: dict[str, Any],
        version: tuple[int, int, int],
        dll_directory: Any | None,
    ) -> None:
        self.path = path
        self.native = native
        self.exports = exports
        self.version = version
        self._dll_directory = dll_directory
        self._explicit_exports: dict[str, Any] | None = None
        self._ovstage_exports: dict[str, Any] | None = None
        self._closed = False

    @classmethod
    def open(
        cls,
        path: Path | str | None = None,
        *,
        require_explicit: bool = False,
        require_ovstage: bool = False,
    ) -> "Library":
        resolved = _resolve_path(path)
        dll_directory = None
        try:
            if sys.platform == "win32":
                dll_directory = os.add_dll_directory(str(resolved.parent))
            native = ctypes.CDLL(str(resolved))
            exports = raw.bind_exports(native, raw.CORE_EXPORT_SPECS)
        except (AttributeError, OSError) as failure:
            if dll_directory is not None:
                dll_directory.close()
            raise LibraryLoadError(
                f"Failed to load the complete core interface from {resolved}: {type(failure).__name__}: {failure}"
            ) from failure
        major = ctypes.c_uint32()
        minor = ctypes.c_uint32()
        patch = ctypes.c_uint32()
        result = exports["ovspatialquery_get_version"](
            ctypes.byref(major), ctypes.byref(minor), ctypes.byref(patch)
        )
        if int(result.status) != raw.ApiStatus.SUCCESS:
            if dll_directory is not None:
                dll_directory.close()
            raise LibraryLoadError(
                f"ovspatialquery_get_version failed for {resolved}: status={int(result.status)}"
            )
        version = (int(major.value), int(minor.value), int(patch.value))
        if version[:2] != SUPPORTED_VERSION_FAMILY:
            if dll_directory is not None:
                dll_directory.close()
            raise LibraryLoadError(
                f"Unsupported ovspatialquery interface version {version}; "
                f"expected {SUPPORTED_VERSION_FAMILY[0]}.{SUPPORTED_VERSION_FAMILY[1]}.x"
            )
        opened = cls(resolved, native, exports, version, dll_directory)
        if require_explicit or require_ovstage:
            try:
                if require_explicit:
                    opened.require_explicit()
                if require_ovstage:
                    opened.require_ovstage()
            except BaseException:
                opened.close()
                raise
        return opened

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def has_ovstage(self) -> bool:
        return all((hasattr(self.native, name) for name in raw.OVSTAGE_EXPORT_SPECS))

    @property
    def has_explicit(self) -> bool:
        return all((hasattr(self.native, name) for name in raw.EXPLICIT_EXPORT_SPECS))

    def require_explicit(self) -> dict[str, Any]:
        if self._closed:
            raise LibraryLoadError("The ovspatialquery library is closed")
        if self._explicit_exports is not None:
            return self._explicit_exports
        try:
            self._explicit_exports = raw.bind_exports(self.native, raw.EXPLICIT_EXPORT_SPECS)
        except AttributeError as failure:
            raise LibraryLoadError(
                f"explicit ingestion exports are unavailable in {self.path}: {failure}"
            ) from failure
        return self._explicit_exports

    def require_ovstage(self) -> dict[str, Any]:
        if self._closed:
            raise LibraryLoadError("The ovspatialquery library is closed")
        if self._ovstage_exports is not None:
            return self._ovstage_exports
        try:
            self._ovstage_exports = raw.bind_exports(self.native, raw.OVSTAGE_EXPORT_SPECS)
        except AttributeError as failure:
            raise LibraryLoadError(
                f"Ovstage companion exports are unavailable in {self.path}: {failure}"
            ) from failure
        return self._ovstage_exports

    def instance(self, config: Any = None) -> Any:
        from .instance import Instance
        from .types import InstanceConfig

        return Instance.create(self, InstanceConfig() if config is None else config)

    def close(self) -> None:
        if self._closed:
            return
        self._explicit_exports = None
        self._ovstage_exports = None
        self.exports = {}
        self.native = None
        if self._dll_directory is not None:
            self._dll_directory.close()
            self._dll_directory = None
        self._closed = True

    def __enter__(self) -> "Library":
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


__all__ = ["LIBRARY_ENV", "Library", "LibraryLoadError"]
