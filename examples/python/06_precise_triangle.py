# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ovstage triangle population plus one exact raycast with source topology."""

from __future__ import annotations

import sys
from typing import Any

import ovstage

from ovspatialquery import Library, QueryPrecision, RaycastQuery

from _evidence import (
    EvidencePaths,
    HEIGHT,
    WIDTH,
    example_argument_parser,
    parse_example_arguments,
    present_interactive,
    require_python_source,
    validate_evidence,
    write_semantic_json,
)
from _runtime import Runtime, frame_summary

EXAMPLE = "06_precise_triangle"
MESH_PATH = "/World/MeshParent/PreciseMesh"
RAY_ORIGIN = (0.0, 0.0, 0.0)
RAY_DIRECTION = (1.0, 0.0, 0.0)
TIME_DOMAIN = 0xE08

SCENE_USDA = f"""#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
    metersPerUnit = 1
)
def Material "Green"
{{
    token outputs:surface.connect = </Green/P.outputs:surface>
    def Shader "P"
    {{
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:diffuseColor = (0.04, 0.8, 0.23)
        float inputs:roughness = 0.28
        token outputs:surface
    }}
}}
def Material "Orange"
{{
    token outputs:surface.connect = </Orange/P.outputs:surface>
    def Shader "P"
    {{
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:diffuseColor = (1, 0.42, 0.01)
        color3f inputs:emissiveColor = (0.25, 0.05, 0)
        token outputs:surface
    }}
}}
# PRESENTATION:USDA:BEGIN
def Xform "World"
{{
# PRESENTATION:USDA:END
    # PRESENTATION:USDA:BEGIN
    def Xform "MeshParent"
    {{
        double3 xformOp:translate = (3, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Mesh "PreciseMesh" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        )
        {{
            uniform bool doubleSided = 0
            uniform token subdivisionScheme = "none"
            int[] faceVertexCounts = [4, 3]
            int[] faceVertexIndices = [0, 1, 2, 3, 4, 5, 6]
            point3f[] points = [(2, -1, -1), (2, -1, 1), (2, 1, 1), (2, 1, -1), (1, -1, -1), (1, 0, 1), (1, 1, -1)]
            float3[] extent = [(1, -1, -1), (2, 1, 1)]
            rel material:binding = </Green>
        }}
    }}
    # PRESENTATION:USDA:END
    def Cube "Ray" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        double size = 2
        matrix4d xformOp:transform = ((2, 0, 0, 0), (0, 0.035, 0, 0), (0, 0, 0.035, 0), (2, 0, 0, 1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
        rel material:binding = </Orange>
    }}
    def Sphere "Contact" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        double radius = 0.18
        double3 xformOp:translate = (4, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        rel material:binding = </Orange>
    }}
# PRESENTATION:USDA:BEGIN
}}
# PRESENTATION:USDA:END
def Xform "Render"
{{
    def Camera "Camera"
    {{
        float focalLength = 20
        double2 clippingRange = (0.01, 1000)
        matrix4d xformOp:transform = ((0.92, 0, -0.39, 0), (-0.1, 0.97, -0.23, 0), (0.38, 0.25, 0.89, 0), (10, 5, 15, 1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }}
    def RenderProduct "CameraProduct"
    {{
        int2 resolution = ({WIDTH}, {HEIGHT})
        rel camera = </Render/Camera>
        rel orderedVars = [<LdrColor>]
        def RenderVar "LdrColor"
        {{
            string sourceName = "LdrColor"
        }}
    }}
    def SphereLight "Key"
    {{
        float inputs:intensity = 28000
        float inputs:radius = 1.5
        double3 xformOp:translate = (-3, 7, 7)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}
    def DomeLight "Environment"
    {{
        color3f inputs:color = (0.12, 0.2, 0.36)
        float inputs:intensity = 260
    }}
}}
"""


def run(arguments: list[str] | None = None) -> tuple[dict[str, Any], Any]:
    options = parse_example_arguments(example_argument_parser(__doc__), arguments)
    require_python_source(options)
    paths = EvidencePaths.create(options.evidence, EXAMPLE, "python")

    runtime = Runtime(f"ovspatialquery.{EXAMPLE}.python", SCENE_USDA)
    cleanup_errors: list[str] = []
    pixels: Any | None = None
    publication: Any | None = None
    result: Any | None = None
    prim_path = ""
    native: dict[str, Any] = {}
    try:
        runtime.open_and_compute()
        public_filter = ovstage.Filter(
            [ovstage.Predicate("usd-path", ovstage.FilterOp.IN, (MESH_PATH,))]
        )
        with (
            Library.open(options.library, require_ovstage=True) as library,
            library.instance() as instance,
        ):
            adapter = instance.ovstage(ovstage, runtime.stage)
            # PRESENTATION:API:BEGIN
            population = adapter.create_population(public_filter)
            # PRESENTATION:API:END
            with population.enqueue_set(2) as operation:
                operation.wait()
                publication = operation.fetch_population_update()

            # PRESENTATION:API:BEGIN
            result = population.target.raycast_closest(
                RaycastQuery(
                    origin=RAY_ORIGIN,
                    unit_direction=RAY_DIRECTION,
                    max_distance=20.0,
                    precision=QueryPrecision.EXACT,
                )
            )
            # PRESENTATION:API:END
            if result.hit is None:
                raise AssertionError("precise ray must hit the triangle mesh")
            prim_path = runtime.path_to_string(result.hit.object_id)
            if (
                prim_path != MESH_PATH
                or result.source_ordinal != 2
                or result.hit.source_face_index != 1
                or abs(result.hit.distance - 4.0) > 1.0e-9
                or result.hit.barycentrics is None
            ):
                raise AssertionError(f"unexpected precise hit: {prim_path}, {result}")
            pixels = runtime.render(options.frames, 2)
            native = runtime.native_identity(library, options.sdk_root)
    finally:
        cleanup_errors.extend(runtime.close())

    if pixels is None or publication is None or result is None or result.hit is None:
        raise AssertionError("example produced incomplete evidence")
    image = frame_summary(pixels)
    payload = {
        "schema_version": 2,
        "example": EXAMPLE,
        "language": "python",
        "execution": {
            "headless": bool(options.headless),
            "frames": options.frames,
            "resolution": [WIDTH, HEIGHT],
        },
        "inputs": {
            "stage": "inline-usda",
            "target_paths": [MESH_PATH],
            "ray": {
                "origin": list(RAY_ORIGIN),
                "unit_direction": list(RAY_DIRECTION),
                "max_distance": 20.0,
                "precision": "exact",
                "mode": "closest",
            },
        },
        "publication": {
            "source_ordinal": publication.published_ordinal,
            "object_count": publication.object_count,
            "excluded": publication.excluded_object_count,
        },
        "result": {
            "source_ordinal": result.source_ordinal,
            "has_hit": True,
            "object_id": result.hit.object_id,
            "prim_path": prim_path,
            "distance": result.hit.distance,
            "position": list(result.hit.position),
            "normal": list(result.hit.normal),
            "truth": result.hit.truth.name.lower(),
            "source_face_index": result.hit.source_face_index,
            "source_face_corner_indices": list(result.hit.source_face_corner_indices or ()),
            "barycentrics": list(result.hit.barycentrics or ()),
        },
        "native": native,
        "evidence": image,
        "cleanup": {"clean": not cleanup_errors, "errors": cleanup_errors},
        "errors": [],
    }
    write_semantic_json(paths.json, payload)
    validate_evidence(paths.json)
    if not options.headless:
        present_interactive("ovspatialquery: precise triangle", pixels)
    return payload, pixels


if __name__ == "__main__":
    try:
        run()
    except BaseException as failure:
        print(f"{EXAMPLE}: {type(failure).__name__}: {failure}", file=sys.stderr)
        raise
