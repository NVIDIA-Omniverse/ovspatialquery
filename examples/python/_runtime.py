# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ovstage/OvRTX presentation runtime shared by the Python lessons.

This module intentionally owns no ovspatialquery instance, descriptor,
publication, query, result, or assertion. Rendered frames are Warp arrays on
the CUDA device when Warp finds one, and on Warp's CPU device otherwise.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from time import monotonic
from typing import Any

import ovrtx
import ovstage
import warp as wp

from _evidence import HEIGHT, WIDTH, sha256_file


wp.config.log_level = wp.LOG_WARNING  # skip Warp's start-up banner
wp.init()
FRAME_DEVICE = "cuda:0" if wp.get_cuda_device_count() else "cpu"
RENDER_PRODUCT = "/Render/CameraProduct"
RENDER_VAR = "/Render/CameraProduct/LdrColor"
CAMERA_PATH = "/Render/Camera"
_LAST_SCENE_USDA: str | None = None


def physics_desc(ovstage):
    """What ovspatialquery reads from a stage on Ovstage 0.3 and 0.4: 0.3's ALL domain, or
    0.4's RENDERING, which has no PHYSICS domain, plus every prim's physics,
    physxConvexHullCollision and material:binding properties for convex hulls, their vertex
    limits and physics materials."""
    population = ovstage.population
    domain = ovstage.PopulationDomain
    physics = population.Selector(
        prim_predicate=population.PrimPredicate.all(),
        property_predicate=population.PropertyPredicate.in_namespace(
            "physics", "physxConvexHullCollision", "material:binding"
        ),
    )
    return population.Desc(domains=getattr(domain, "ALL", domain.RENDERING), selectors=[physics])


def prim_path(value: Any) -> Any:
    """An object ID's value as the path dictionary takes it: an int on Ovstage 0.3, a
    PrimPath on 0.4."""
    handle = getattr(ovstage, "PrimPath", None)
    value = int(getattr(value, "raw_id", value))
    return value if handle is None else handle(value)


class Runtime:
    def __init__(self, name: str, scene_usda: str, *, selection: bool = False) -> None:
        global _LAST_SCENE_USDA
        self._name = str(name)
        self._scene_usda = str(scene_usda)
        _LAST_SCENE_USDA = self._scene_usda
        self.stage: Any | None = None
        self.renderer: Any | None = None
        self._selection_enabled = selection
        self._closed = False
        ovrtx.register_schema_paths()
        try:
            self.stage = ovstage.Stage(self._name)
            self.renderer = ovrtx.Renderer(
                ovrtx.RendererConfig(
                    sync_mode=True,
                    active_cuda_gpus="0",
                    selection_outline_enabled=selection,
                    selection_outline_width=2,
                    selection_fill_mode=(
                        ovrtx.SelectionFillMode.GROUP_FILL_COLOR
                        if selection
                        else ovrtx.SelectionFillMode.EDGE_ONLY
                    ),
                )
            )
            self.renderer.attach_ovstage(self.stage)
        except BaseException:
            self.close()
            raise

    def open_and_compute(
        self,
        open_ordinal: int = 1,
        hierarchy_ordinal: int = 2,
    ) -> None:
        if self._closed or self.stage is None:
            raise RuntimeError("presentation runtime is closed")
        ovstage.population.open_usd_from_string_with_desc(
            self.stage, self._scene_usda, open_ordinal, 0.0, physics_desc(ovstage)
        )
        self.stage.advance_write_floor(ordinal=open_ordinal).wait()
        self.stage.compute_hierarchy(
            open_ordinal,
            hierarchy_ordinal,
            ovstage.HierarchyComputationModel.DEFAULT_CPU,
        )
        self.stage.advance_write_floor(ordinal=hierarchy_ordinal).wait()
        if self._selection_enabled:
            # The string-path selection API resolves against OvRTX's path
            # dictionary. Populate it before callers assign groups so the
            # first visible frame includes those assignments.
            products = self.renderer.step(
                render_products={RENDER_PRODUCT},
                delta_time=0.0,
                ordinal=hierarchy_ordinal,
            )
            del products

    @staticmethod
    def _copy_color(products: Any) -> wp.array:
        product = products[RENDER_PRODUCT]
        color = product.frames[0].render_vars[RENDER_VAR]
        device = ovrtx.Device.CPU if FRAME_DEVICE == "cpu" else ovrtx.Device.CUDA
        with color.map(device=device) as mapping:
            # The mapping is valid only inside this block, so keep a copy of the frame.
            return wp.clone(wp.from_dlpack(mapping))

    def render(self, frames: int, ordinal: int) -> wp.array:
        if frames <= 0:
            raise ValueError("frames must be positive")
        if self._closed or self.renderer is None:
            raise RuntimeError("presentation runtime is closed")
        pixels: wp.array | None = None
        for _ in range(frames):
            products = self.renderer.step(
                render_products={RENDER_PRODUCT},
                delta_time=1.0 / 60.0,
                ordinal=ordinal,
            )
            pixels = self._copy_color(products)
            del products
        if pixels is None:
            raise RuntimeError("renderer produced no frame")
        return pixels

    def path_to_string(self, token: int) -> str:
        if self._closed or self.stage is None:
            raise RuntimeError("presentation runtime is closed")
        return str(ovstage.PathDictionary(self.stage).path_to_string(prim_path(token)))

    def native_identity(self, library: Any, sdk_root: Path | None) -> dict[str, Any]:
        if self.renderer is None:
            raise RuntimeError("presentation runtime is closed")
        import ovspatialquery

        python_module = Path(ovspatialquery.__file__).resolve()
        spatial: dict[str, Any] = {
            "python_module": str(python_module),
            "runtime": list(library.version),
            "library": str(library.path.resolve()),
            "library_sha256": sha256_file(library.path),
        }
        if sdk_root is not None:
            include = sdk_root.resolve() / "include" / "ovspatialquery"
            for label, name in (
                ("core_header", "ovspatialquery.h"),
                ("companion_header", "ovspatialquery_ovstage.h"),
            ):
                path = include / name
                if not path.is_file():
                    raise FileNotFoundError(f"installed header is required: {path}")
                spatial[label] = str(path)
                spatial[f"{label}_sha256"] = sha256_file(path)
        else:
            provenance = python_module.parent / "_native" / "provenance.json"
            if not provenance.is_file():
                raise FileNotFoundError("wheel mode requires bundled _native/provenance.json")
            manifest = json.loads(provenance.read_text(encoding="utf-8"))
            installed = {item["install_path"]: item for item in manifest["installed_contract"]}
            spatial["wheel_provenance"] = str(provenance)
            spatial["core_header_sha256"] = installed["include/ovspatialquery/ovspatialquery.h"][
                "sha256"
            ]
            spatial["companion_header_sha256"] = installed[
                "include/ovspatialquery/ovspatialquery_ovstage.h"
            ]["sha256"]
        return {
            "ovspatialquery": spatial,
            "ovstage": {
                "package": ovstage.__version__,
                "runtime": list(ovstage.library_version()),
            },
            "ovrtx": {
                "package": ovrtx.__version__,
                "runtime": list(self.renderer.version),
            },
        }

    def close(self) -> list[str]:
        if self._closed:
            return []
        errors: list[str] = []
        stage = self.stage
        renderer = self.renderer
        for label, action in (
            (
                "detach renderer",
                lambda: (
                    renderer.detach_ovstage()
                    if renderer is not None and stage is not None
                    else None
                ),
            ),
            ("destroy renderer", lambda: renderer.destroy() if renderer else None),
            ("destroy stage", lambda: stage.destroy() if stage else None),
        ):
            try:
                action()
            except BaseException as failure:
                errors.append(f"{label}: {type(failure).__name__}: {failure}")
        self.renderer = None
        self.stage = None
        self._closed = True
        return errors

    def __enter__(self) -> "Runtime":
        return self

    def __exit__(self, *_exception: object) -> None:
        self.close()


@wp.kernel
def _pack_rgb(
    pixels: wp.array3d(dtype=wp.uint8),
    keys: wp.array(dtype=wp.int32),
    nonblack: wp.array(dtype=wp.int32),
):
    row, column = wp.tid()
    key = (
        (wp.int32(pixels[row, column, 0]) << 16)
        | (wp.int32(pixels[row, column, 1]) << 8)
        | wp.int32(pixels[row, column, 2])
    )
    keys[row * pixels.shape[1] + column] = key
    if key != 0:
        wp.atomic_add(nonblack, 0, 1)


@wp.kernel
def _count_distinct(sorted_keys: wp.array(dtype=wp.int32), distinct: wp.array(dtype=wp.int32)):
    index = wp.tid()
    if index == 0 or sorted_keys[index] != sorted_keys[index - 1]:
        wp.atomic_add(distinct, 0, 1)


@wp.kernel
def _bgra(pixels: wp.array3d(dtype=wp.uint8), out: wp.array3d(dtype=wp.uint8)):
    row, column = wp.tid()
    out[row, column, 0] = pixels[row, column, 2]
    out[row, column, 1] = pixels[row, column, 1]
    out[row, column, 2] = pixels[row, column, 0]
    out[row, column, 3] = wp.uint8(255)


def _require_frame(pixels: wp.array) -> None:
    if pixels.ndim != 3 or pixels.shape[0:2] != (HEIGHT, WIDTH) or pixels.dtype != wp.uint8:
        raise ValueError(
            f"LdrColor must be uint8 {WIDTH}x{HEIGHT}; observed {pixels.dtype} {pixels.shape}"
        )
    if pixels.shape[2] not in (3, 4):
        raise ValueError(f"LdrColor must have three or four channels; observed {pixels.shape[2]}")


def frame_summary(pixels: wp.array) -> dict[str, Any]:
    """Counts the frame's nonblack pixels and distinct colors with Warp, on its device."""

    _require_frame(pixels)
    count = HEIGHT * WIDTH
    with wp.ScopedDevice(pixels.device):
        # radix_sort_pairs sorts in place and needs room for twice the keys it sorts.
        keys = wp.empty(2 * count, dtype=wp.int32)
        values = wp.zeros(2 * count, dtype=wp.int32)
        nonblack = wp.zeros(1, dtype=wp.int32)
        distinct = wp.zeros(1, dtype=wp.int32)
        wp.launch(_pack_rgb, dim=(HEIGHT, WIDTH), inputs=[pixels, keys, nonblack])
        wp.utils.radix_sort_pairs(keys, values, count)
        wp.launch(_count_distinct, dim=count, inputs=[keys, distinct])
    nonblack_pixels = nonblack.list()[0]
    unique_rgb = distinct.list()[0]
    if nonblack_pixels == 0 or unique_rgb < 2:
        raise ValueError("Rendered frame is blank")
    return {
        "width": WIDTH,
        "height": HEIGHT,
        "nonblack_pixels": nonblack_pixels,
        "unique_rgb_count": unique_rgb,
    }


def _host_bgra(pixels: wp.array) -> wp.array:
    """Returns the frame as BGRA bytes on Warp's CPU device, the layout the viewports draw."""

    _require_frame(pixels)
    out = wp.empty((HEIGHT, WIDTH, 4), dtype=wp.uint8, device=pixels.device)
    wp.launch(_bgra, dim=(HEIGHT, WIDTH), inputs=[pixels, out], device=pixels.device)
    return out.to("cpu")


def _look_at(
    position: tuple[float, float, float],
    target: tuple[float, float, float],
) -> list[float]:
    def unit(vector: tuple[float, float, float]) -> tuple[float, float, float]:
        length = math.hypot(*vector)
        return (vector[0] / length, vector[1] / length, vector[2] / length)

    def cross(
        left: tuple[float, float, float], right: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        return (
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        )

    forward = unit(tuple(t - p for t, p in zip(target, position)))
    right = unit(cross(forward, (0.0, 1.0, 0.0)))
    upward = cross(right, forward)
    # A row-major USD matrix: the camera's axes, then its position.
    return [
        *right, 0.0,
        *upward, 0.0,
        *(-axis for axis in forward), 0.0,
        *position, 1.0,
    ]  # fmt: skip


def present_orbit(title: str, canonical_pixels: wp.array) -> None:
    """Orbit a presentation-only stage; no spatial object is reachable here."""

    if _LAST_SCENE_USDA is None:
        raise RuntimeError("no example scene is available for interactive orbit")
    from _evidence import present_window

    runtime = Runtime(f"{title}.orbit", _LAST_SCENE_USDA)
    paths: Any | None = None
    path_list: Any | None = None
    try:
        runtime.open_and_compute()
        if runtime.stage is None:
            raise RuntimeError("presentation runtime has no stage")
        paths = ovstage.PathDictionary(runtime.stage)
        path_list = paths.create_path_list_from_strings([CAMERA_PATH])
        with runtime.stage.query_from_path_list(path_list) as query:
            attribute = paths.intern_token("omni:xform")
            with runtime.stage.read_attributes(
                query, [attribute], ovstage.OrdinalRange.latest(2)
            ) as reader:
                group = reader.fetch_next()
                initial = wp.clone(wp.from_dlpack(group.dlpack(0)), device="cpu").list()
                runtime.stage.release_group(group)
            eye = tuple(initial[12:15])
            forward = tuple(-value for value in initial[8:11])
            target = (0.0, 0.0, 0.0)
            if abs(forward[1]) > 1e-5:
                distance = -eye[1] / forward[1]
                if 0.0 < distance < 1000.0:
                    target = tuple(e + distance * f for e, f in zip(eye, forward))
            offset = tuple(e - t for e, t in zip(eye, target))
            radius = max(math.hypot(offset[0], offset[2]), 1.0)
            height = offset[1]
            initial_angle = math.atan2(offset[2], offset[0])
            started = monotonic()
            ordinal = 2
            matrix_type = ovstage.DLDataType(
                code=ovstage.DLDataTypeCode.kDLFloat, bits=64, lanes=16
            )

            def next_frame() -> wp.array:
                nonlocal ordinal
                angle = initial_angle + (monotonic() - started) * 0.12
                position = (
                    target[0] + radius * math.cos(angle),
                    target[1] + height,
                    target[2] + radius * math.sin(angle),
                )
                transform = wp.array([_look_at(position, target)], dtype=wp.float64, device="cpu")
                tensor = ovstage.make_dltensor(transform, dtype=matrix_type, shape=[1], ndim=1)
                ordinal += 1
                runtime.stage.write_attribute(
                    query,
                    attribute,
                    ordinal,
                    tensor,
                    is_array=False,
                    semantic=ovstage.AttributeSemantic.MATRIX,
                ).wait()
                # OvRTX consumes the global stage ordinal. This isolated
                # presentation stage has only the camera write at this
                # ordinal, so advancing ALL exposes no spatial publication.
                runtime.stage.advance_write_floor(ordinal, ovstage.Scope.ALL).wait()
                return _host_bgra(runtime.render(1, ordinal))

            present_window(title, _host_bgra(canonical_pixels), next_frame)
    finally:
        if paths is not None and path_list is not None:
            paths.destroy_path_list(path_list)
        errors = runtime.close()
        if errors:
            raise RuntimeError("; ".join(errors))


__all__ = [
    "FRAME_DEVICE",
    "RENDER_PRODUCT",
    "RENDER_VAR",
    "Runtime",
    "frame_summary",
    "present_orbit",
]
