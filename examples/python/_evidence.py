# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared evidence mechanics for the Python examples.

This module deliberately knows nothing about spatial descriptors or queries.
Each example keeps those calls in its own single source file; this module only
parses the shared command line, presents frames and records query results.
It imports no third-party package, so the lesson runner can use it anywhere.
PNG capture and image review live in the private tools/example_capture directory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import ctypes
import ctypes.util
import hashlib
import json
import os
from pathlib import Path
import sys
from time import sleep
from typing import Any, Callable


WIDTH = 1280
HEIGHT = 720
X11_KEY_PRESS = 2
X11_DESTROY_NOTIFY = 17
X11_CLIENT_MESSAGE = 33
X11_ESCAPE_KEYSYM = 0xFF1B


@dataclass(frozen=True)
class SourceCheckoutDefaults:
    evidence: Path
    library: Path | None
    sdk_root: Path | None
    python_source: Path | None


def source_checkout_defaults() -> SourceCheckoutDefaults:
    project = Path(__file__).resolve().parents[2]
    source = (project / "python" / "src").resolve()
    try:
        import ovspatialquery

        module = Path(ovspatialquery.__file__).resolve()
        source_mode = module.is_relative_to(source)
    except (ImportError, TypeError):
        source_mode = False
    install = (project / "_build" / "bootstrap" / "native-stable" / "install").resolve()
    library_name = (
        install / "bin" / "ovspatialquery.dll"
        if sys.platform == "win32"
        else install / "lib" / "libovspatialquery.so"
    )
    return SourceCheckoutDefaults(
        evidence=(project / "_build" / "evidence" / "examples" / "manual-direct").resolve(),
        library=library_name if source_mode else None,
        sdk_root=install if source_mode else None,
        python_source=source if source_mode else None,
    )


def example_argument_parser(description: str | None) -> argparse.ArgumentParser:
    """Return the parser every lesson shares; add lesson-specific arguments before parsing."""

    defaults = source_checkout_defaults()
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--evidence", type=Path, default=defaults.evidence)
    parser.add_argument("--library", type=Path, default=defaults.library)
    parser.add_argument("--sdk-root", type=Path, default=defaults.sdk_root)
    parser.add_argument(
        "--expect-python-source",
        type=Path,
        default=defaults.python_source,
    )
    return parser


def parse_example_arguments(
    parser: argparse.ArgumentParser, arguments: list[str] | None
) -> argparse.Namespace:
    parsed = parser.parse_args(arguments)
    if parsed.frames <= 0:
        parser.error("--frames must be positive")
    return parsed


def require_python_source(options: argparse.Namespace) -> None:
    """Fail unless ovspatialquery was imported from --expect-python-source (lessons 01-10)."""

    if options.expect_python_source is None:
        return
    import ovspatialquery

    module = Path(ovspatialquery.__file__).resolve()
    expected = options.expect_python_source.resolve()
    if not module.is_relative_to(expected):
        raise RuntimeError(f"ovspatialquery imported from {module}, not {expected}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class EvidencePaths:
    root: Path
    json: Path

    @classmethod
    def create(cls, root: Path, example: str, language: str) -> "EvidencePaths":
        resolved = root.expanduser().resolve() / example / language
        resolved.mkdir(parents=True, exist_ok=True)
        return cls(resolved, resolved / "semantic.json")


def _require_interactive_viewport_available() -> None:
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        raise RuntimeError(
            "interactive Linux viewport requires an X11 DISPLAY; "
            "start from a graphical session or use --headless"
        )


def _x11_event_requests_close(
    *,
    event_type: int,
    client_atom: int,
    wm_delete_atom: int,
    keysym: int,
) -> bool:
    if event_type == X11_CLIENT_MESSAGE:
        return client_atom == wm_delete_atom
    if event_type == X11_DESTROY_NOTIFY:
        return True
    return event_type == X11_KEY_PRESS and keysym == X11_ESCAPE_KEYSYM


def present_interactive(title: str, pixels: Any) -> None:
    """Open the shared presentation-only live orbit for the latest scene."""

    _require_interactive_viewport_available()
    from _runtime import present_orbit

    present_orbit(title, pixels)


def _present_window_x11(
    title: str,
    pixels: Any,
    next_frame: Callable[[], Any],
) -> None:
    _require_interactive_viewport_available()
    library_name = ctypes.util.find_library("X11") or "libX11.so.6"
    x11 = ctypes.CDLL(library_name)

    class XImageFuncs(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_void_p)
            for name in (
                "create_image",
                "destroy_image",
                "get_pixel",
                "put_pixel",
                "sub_image",
                "add_pixel",
            )
        ]

    class XImage(ctypes.Structure):
        _fields_ = [
            ("width", ctypes.c_int),
            ("height", ctypes.c_int),
            ("xoffset", ctypes.c_int),
            ("format", ctypes.c_int),
            ("data", ctypes.c_void_p),
            ("byte_order", ctypes.c_int),
            ("bitmap_unit", ctypes.c_int),
            ("bitmap_bit_order", ctypes.c_int),
            ("bitmap_pad", ctypes.c_int),
            ("depth", ctypes.c_int),
            ("bytes_per_line", ctypes.c_int),
            ("bits_per_pixel", ctypes.c_int),
            ("red_mask", ctypes.c_ulong),
            ("green_mask", ctypes.c_ulong),
            ("blue_mask", ctypes.c_ulong),
            ("obdata", ctypes.c_void_p),
            ("f", XImageFuncs),
        ]

    class XKeyEvent(ctypes.Structure):
        _fields_ = [
            ("type", ctypes.c_int),
            ("serial", ctypes.c_ulong),
            ("send_event", ctypes.c_int),
            ("display", ctypes.c_void_p),
            ("window", ctypes.c_ulong),
            ("root", ctypes.c_ulong),
            ("subwindow", ctypes.c_ulong),
            ("time", ctypes.c_ulong),
            ("x", ctypes.c_int),
            ("y", ctypes.c_int),
            ("x_root", ctypes.c_int),
            ("y_root", ctypes.c_int),
            ("state", ctypes.c_uint),
            ("keycode", ctypes.c_uint),
            ("same_screen", ctypes.c_int),
        ]

    class XClientMessageData(ctypes.Union):
        _fields_ = [
            ("bytes", ctypes.c_char * 20),
            ("shorts", ctypes.c_short * 10),
            ("longs", ctypes.c_long * 5),
        ]

    class XClientMessageEvent(ctypes.Structure):
        _fields_ = [
            ("type", ctypes.c_int),
            ("serial", ctypes.c_ulong),
            ("send_event", ctypes.c_int),
            ("display", ctypes.c_void_p),
            ("window", ctypes.c_ulong),
            ("message_type", ctypes.c_ulong),
            ("format", ctypes.c_int),
            ("data", XClientMessageData),
        ]

    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XDefaultScreen.argtypes = [ctypes.c_void_p]
    x11.XDefaultScreen.restype = ctypes.c_int
    x11.XRootWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
    x11.XRootWindow.restype = ctypes.c_ulong
    x11.XBlackPixel.argtypes = [ctypes.c_void_p, ctypes.c_int]
    x11.XBlackPixel.restype = ctypes.c_ulong
    x11.XCreateSimpleWindow.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    x11.XCreateSimpleWindow.restype = ctypes.c_ulong
    x11.XStoreName.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_char_p]
    x11.XSelectInput.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_long]
    x11.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    x11.XInternAtom.restype = ctypes.c_ulong
    x11.XSetWMProtocols.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.c_int,
    ]
    x11.XMapWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    x11.XDefaultGC.argtypes = [ctypes.c_void_p, ctypes.c_int]
    x11.XDefaultGC.restype = ctypes.c_void_p
    x11.XInitImage.argtypes = [ctypes.POINTER(XImage)]
    x11.XInitImage.restype = ctypes.c_int
    x11.XPutImage.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_void_p,
        ctypes.POINTER(XImage),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.c_uint,
    ]
    x11.XPending.argtypes = [ctypes.c_void_p]
    x11.XPending.restype = ctypes.c_int
    x11.XNextEvent.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    x11.XLookupKeysym.argtypes = [ctypes.POINTER(XKeyEvent), ctypes.c_int]
    x11.XLookupKeysym.restype = ctypes.c_ulong
    x11.XFlush.argtypes = [ctypes.c_void_p]
    x11.XDestroyWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]

    display = x11.XOpenDisplay(None)
    if not display:
        raise RuntimeError("could not open X11 DISPLAY for interactive Linux viewport")
    window = ctypes.c_ulong(0)
    window_owned = False
    try:
        screen = x11.XDefaultScreen(display)
        root = x11.XRootWindow(display, screen)
        black = x11.XBlackPixel(display, screen)
        window = x11.XCreateSimpleWindow(display, root, 0, 0, WIDTH, HEIGHT, 1, black, black)
        if not window:
            raise RuntimeError("could not create interactive Linux viewport")
        window_owned = True
        x11.XStoreName(display, window, title.encode("utf-8", errors="replace"))
        x11.XSelectInput(display, window, (1 << 0) | (1 << 15) | (1 << 17))
        wm_delete = ctypes.c_ulong(x11.XInternAtom(display, b"WM_DELETE_WINDOW", 0))
        x11.XSetWMProtocols(display, window, ctypes.byref(wm_delete), 1)
        x11.XMapWindow(display, window)
        gc = x11.XDefaultGC(display, screen)

        def put(frame: Any) -> None:
            image = XImage(
                WIDTH,
                HEIGHT,
                0,
                2,
                ctypes.c_void_p(frame.ptr),
                0,
                32,
                0,
                32,
                24,
                WIDTH * 4,
                32,
                0x00FF0000,
                0x0000FF00,
                0x000000FF,
                None,
                XImageFuncs(),
            )
            if not x11.XInitImage(ctypes.byref(image)):
                raise RuntimeError("could not initialize Linux viewport image")
            x11.XPutImage(display, window, gc, ctypes.byref(image), 0, 0, 0, 0, WIDTH, HEIGHT)
            x11.XFlush(display)

        put(pixels)
        event = ctypes.create_string_buffer(192)
        running = True
        while running:
            while x11.XPending(display):
                x11.XNextEvent(display, ctypes.byref(event))
                event_type = ctypes.c_int.from_buffer(event).value
                client_atom = 0
                keysym = 0
                if event_type == X11_CLIENT_MESSAGE:
                    client = ctypes.cast(
                        ctypes.byref(event),
                        ctypes.POINTER(XClientMessageEvent),
                    ).contents
                    client_atom = int(client.data.longs[0])
                elif event_type == X11_KEY_PRESS:
                    keysym = int(
                        x11.XLookupKeysym(
                            ctypes.cast(
                                ctypes.byref(event),
                                ctypes.POINTER(XKeyEvent),
                            ),
                            0,
                        )
                    )
                if _x11_event_requests_close(
                    event_type=event_type,
                    client_atom=client_atom,
                    wm_delete_atom=int(wm_delete.value),
                    keysym=keysym,
                ):
                    if event_type == X11_DESTROY_NOTIFY:
                        window_owned = False
                    running = False
                    break
            if not running:
                break
            put(next_frame())
            sleep(1.0 / 60.0)
    finally:
        if window and window_owned:
            x11.XDestroyWindow(display, window)
        x11.XCloseDisplay(display)


def present_window(
    title: str,
    pixels: Any,
    next_frame: Callable[[], Any],
) -> None:
    """Present changing frames in one bare native viewport. Each frame is host memory holding
    HEIGHT x WIDTH BGRA bytes whose address is its `ptr`, such as a Warp CPU array."""

    if sys.platform.startswith("linux"):
        _present_window_x11(title, pixels, next_frame)
        return
    if sys.platform != "win32":
        raise RuntimeError(
            "interactive viewport is supported on Windows and Linux/X11; "
            "use --headless on this platform"
        )

    from ctypes import wintypes

    bgra = pixels
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class BitmapInfoHeader(ctypes.Structure):
        _fields_ = [
            ("biSize", wintypes.DWORD),
            ("biWidth", wintypes.LONG),
            ("biHeight", wintypes.LONG),
            ("biPlanes", wintypes.WORD),
            ("biBitCount", wintypes.WORD),
            ("biCompression", wintypes.DWORD),
            ("biSizeImage", wintypes.DWORD),
            ("biXPelsPerMeter", wintypes.LONG),
            ("biYPelsPerMeter", wintypes.LONG),
            ("biClrUsed", wintypes.DWORD),
            ("biClrImportant", wintypes.DWORD),
        ]

    class BitmapInfo(ctypes.Structure):
        _fields_ = [("bmiHeader", BitmapInfoHeader), ("bmiColors", wintypes.DWORD)]

    class WindowClass(ctypes.Structure):
        _fields_ = [
            ("style", wintypes.UINT),
            ("lpfnWndProc", ctypes.c_void_p),
            ("cbClsExtra", ctypes.c_int),
            ("cbWndExtra", ctypes.c_int),
            ("hInstance", wintypes.HINSTANCE),
            ("hIcon", wintypes.HICON),
            ("hCursor", wintypes.HANDLE),
            ("hbrBackground", wintypes.HBRUSH),
            ("lpszMenuName", wintypes.LPCWSTR),
            ("lpszClassName", wintypes.LPCWSTR),
        ]

    class PaintStruct(ctypes.Structure):
        _fields_ = [
            ("hdc", wintypes.HDC),
            ("fErase", wintypes.BOOL),
            ("rcPaint", wintypes.RECT),
            ("fRestore", wintypes.BOOL),
            ("fIncUpdate", wintypes.BOOL),
            ("rgbReserved", wintypes.BYTE * 32),
        ]

    callback_type = ctypes.WINFUNCTYPE(
        ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
    )
    kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    kernel32.GetModuleHandleW.restype = wintypes.HINSTANCE
    user32.RegisterClassW.argtypes = [ctypes.POINTER(WindowClass)]
    user32.RegisterClassW.restype = wintypes.ATOM
    user32.LoadCursorW.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR]
    user32.LoadCursorW.restype = wintypes.HANDLE
    user32.AdjustWindowRect.argtypes = [
        ctypes.POINTER(wintypes.RECT),
        wintypes.DWORD,
        wintypes.BOOL,
    ]
    user32.CreateWindowExW.argtypes = [
        wintypes.DWORD,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.HWND,
        wintypes.HMENU,
        wintypes.HINSTANCE,
        ctypes.c_void_p,
    ]
    user32.CreateWindowExW.restype = wintypes.HWND
    user32.BeginPaint.argtypes = [wintypes.HWND, ctypes.POINTER(PaintStruct)]
    user32.BeginPaint.restype = wintypes.HDC
    user32.EndPaint.argtypes = [wintypes.HWND, ctypes.POINTER(PaintStruct)]
    user32.DestroyWindow.argtypes = [wintypes.HWND]
    user32.PostQuitMessage.argtypes = [ctypes.c_int]
    user32.DefWindowProcW.argtypes = [
        wintypes.HWND,
        wintypes.UINT,
        wintypes.WPARAM,
        wintypes.LPARAM,
    ]
    user32.DefWindowProcW.restype = ctypes.c_ssize_t
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.UpdateWindow.argtypes = [wintypes.HWND]
    user32.GetMessageW.argtypes = [
        ctypes.POINTER(wintypes.MSG),
        wintypes.HWND,
        wintypes.UINT,
        wintypes.UINT,
    ]
    user32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
    user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
    user32.DispatchMessageW.restype = ctypes.c_ssize_t
    user32.PeekMessageW.argtypes = [
        ctypes.POINTER(wintypes.MSG),
        wintypes.HWND,
        wintypes.UINT,
        wintypes.UINT,
        wintypes.UINT,
    ]
    user32.PeekMessageW.restype = wintypes.BOOL
    user32.InvalidateRect.argtypes = [
        wintypes.HWND,
        ctypes.POINTER(wintypes.RECT),
        wintypes.BOOL,
    ]
    gdi32.SetDIBitsToDevice.argtypes = [
        wintypes.HDC,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.UINT,
        wintypes.UINT,
        ctypes.c_void_p,
        ctypes.POINTER(BitmapInfo),
        wintypes.UINT,
    ]
    bitmap = BitmapInfo(
        BitmapInfoHeader(ctypes.sizeof(BitmapInfoHeader), WIDTH, -HEIGHT, 1, 32, 0, 0, 0, 0, 0, 0),
        0,
    )

    def window_proc(hwnd: int, message: int, wparam: int, lparam: int) -> int:
        if message == 0x000F:
            paint = PaintStruct()
            device = user32.BeginPaint(hwnd, ctypes.byref(paint))
            gdi32.SetDIBitsToDevice(
                device,
                0,
                0,
                WIDTH,
                HEIGHT,
                0,
                0,
                0,
                HEIGHT,
                ctypes.c_void_p(bgra.ptr),
                ctypes.byref(bitmap),
                0,
            )
            user32.EndPaint(hwnd, ctypes.byref(paint))
            return 0
        if message == 0x0100 and wparam == 0x1B:
            user32.DestroyWindow(hwnd)
            return 0
        if message == 0x0002:
            user32.PostQuitMessage(0)
            return 0
        return int(user32.DefWindowProcW(hwnd, message, wparam, lparam))

    callback = callback_type(window_proc)
    instance = kernel32.GetModuleHandleW(None)
    class_name = "OvSpatialQueryExampleViewport"
    window_class = WindowClass(
        0,
        ctypes.cast(callback, ctypes.c_void_p),
        0,
        0,
        instance,
        None,
        user32.LoadCursorW(None, ctypes.cast(ctypes.c_void_p(32512), wintypes.LPCWSTR)),
        None,
        None,
        class_name,
    )
    atom = user32.RegisterClassW(ctypes.byref(window_class))
    if not atom and ctypes.get_last_error() != 1410:
        raise ctypes.WinError(ctypes.get_last_error())
    style = 0x00CF0000
    rectangle = wintypes.RECT(0, 0, WIDTH, HEIGHT)
    user32.AdjustWindowRect(ctypes.byref(rectangle), style, False)
    window = user32.CreateWindowExW(
        0,
        class_name,
        title,
        style,
        0x80000000,
        0x80000000,
        rectangle.right - rectangle.left,
        rectangle.bottom - rectangle.top,
        None,
        None,
        instance,
        None,
    )
    if not window:
        raise ctypes.WinError(ctypes.get_last_error())
    user32.ShowWindow(window, 5)
    user32.UpdateWindow(window)
    message = wintypes.MSG()
    running = True
    while running:
        while user32.PeekMessageW(ctypes.byref(message), None, 0, 0, 0x0001):
            if message.message == 0x0012:
                running = False
                break
            user32.TranslateMessage(ctypes.byref(message))
            user32.DispatchMessageW(ctypes.byref(message))
        if not running:
            break
        bgra = next_frame()
        user32.InvalidateRect(window, None, False)


def write_semantic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def validate_evidence(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "example",
        "language",
        "execution",
        "inputs",
        "publication",
        "result",
        "native",
        "evidence",
        "cleanup",
        "errors",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"semantic evidence is missing: {sorted(missing)}")
    if payload["schema_version"] not in (1, 2):
        raise ValueError("unsupported semantic evidence schema version")
    if payload["execution"]["resolution"] != [WIDTH, HEIGHT]:
        raise ValueError("semantic evidence has the wrong fixed resolution")
    if payload["errors"]:
        raise ValueError(f"example reported errors: {payload['errors']}")
    if not payload["cleanup"].get("clean"):
        raise ValueError(f"example did not close cleanly: {payload['cleanup']}")
    image = payload["evidence"]
    if payload["schema_version"] == 1:
        if not image.get("png") or not image.get("sha256"):
            raise ValueError("schema version 1 requires PNG path and digest")
        png = Path(image["png"]).resolve()
        if not png.is_file() or sha256_file(png) != image["sha256"]:
            raise ValueError("PNG path or digest does not match semantic evidence")
    if int(image["nonblack_pixels"]) <= 0 or int(image["unique_rgb_count"]) < 2:
        raise ValueError("Rendered frame evidence is blank")
    if payload["schema_version"] == 2 and (image["width"] != WIDTH or image["height"] != HEIGHT):
        raise ValueError("semantic evidence has the wrong frame dimensions")
    return payload


def _semantic_projection(payload: dict[str, Any]) -> dict[str, Any]:
    if payload["schema_version"] not in (1, 2):
        raise ValueError("unsupported semantic evidence schema version")
    projection = {
        key: payload[key]
        for key in (
            "example",
            "execution",
            "inputs",
            "publication",
            "result",
            "cleanup",
            "errors",
        )
    }
    return _without_instance_local_ids(projection)


def _without_instance_local_ids(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_instance_local_ids(item)
            for key, item in value.items()
            if not key.endswith(("object_id", "object_ids"))
        }
    if isinstance(value, list):
        return [_without_instance_local_ids(item) for item in value]
    return value


def compare_semantics(first: dict[str, Any], second: dict[str, Any]) -> None:
    left = _semantic_projection(first)
    right = _semantic_projection(second)
    if left != right:
        raise ValueError(
            "semantic mismatch between example twins:\n"
            + json.dumps({"first": left, "second": right}, indent=2, sort_keys=True)
        )


__all__ = [
    "EvidencePaths",
    "HEIGHT",
    "WIDTH",
    "compare_semantics",
    "example_argument_parser",
    "parse_example_arguments",
    "present_interactive",
    "require_python_source",
    "sha256_file",
    "validate_evidence",
    "write_semantic_json",
]
