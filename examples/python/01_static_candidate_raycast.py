# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static Ovstage population plus one synchronous AABB-candidate raycast."""

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
from _runtime import prim_path as dictionary_path

EXAMPLE = "01_static_candidate_raycast"
TARGET_PATHS = ("/World/Targets/Hit", "/World/Targets/OffRay")
HIT_PATH = TARGET_PATHS[0]
RAY_ORIGIN = (-4.0, 0.0, 0.0)
RAY_DIRECTION = (1.0, 0.0, 0.0)
RAY_DISTANCE = 10.0

SCENE_USDA = f"""#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
    metersPerUnit = 1
)

def Scope "Looks"
{{
    def Material "Target"
    {{
        token outputs:surface.connect = </Looks/Target/Preview.outputs:surface>
        def Shader "Preview"
        {{
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (0.08, 0.32, 0.95)
            float inputs:metallic = 0.15
            float inputs:roughness = 0.28
            token outputs:surface
        }}
    }}
    def Material "Hit"
    {{
        token outputs:surface.connect = </Looks/Hit/Preview.outputs:surface>
        def Shader "Preview"
        {{
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (0.05, 0.9, 0.28)
            float inputs:roughness = 0.24
            token outputs:surface
        }}
    }}
    def Material "Ray"
    {{
        token outputs:surface.connect = </Looks/Ray/Preview.outputs:surface>
        def Shader "Preview"
        {{
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (1, 0.58, 0.02)
            color3f inputs:emissiveColor = (0.35, 0.1, 0)
            float inputs:roughness = 0.35
            token outputs:surface
        }}
    }}
    def Material "Ground"
    {{
        token outputs:surface.connect = </Looks/Ground/Preview.outputs:surface>
        def Shader "Preview"
        {{
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (0.045, 0.06, 0.1)
            float inputs:roughness = 0.6
            token outputs:surface
        }}
    }}
}}

def Xform "World"
{{
    # PRESENTATION:USDA:BEGIN
    def Xform "Targets"
    {{
        def Cube "Hit" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        )
        {{
            double size = 2
            float3[] extent = [(-1, -1, -1), (1, 1, 1)]
            double3 xformOp:translate = (1, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
            rel material:binding = </Looks/Hit>
        }}
        def Cube "OffRay" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        )
        {{
            double size = 1.5
            float3[] extent = [(-0.75, -0.75, -0.75), (0.75, 0.75, 0.75)]
            double3 xformOp:translate = (3.2, 0, -3)
            uniform token[] xformOpOrder = ["xformOp:translate"]
            rel material:binding = </Looks/Target>
        }}
    }}
    # PRESENTATION:USDA:END
    def Xform "QueryVisual"
    {{
        def Cube "Ray" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        )
        {{
            double size = 2
            float3[] extent = [(-1, -1, -1), (1, 1, 1)]
            matrix4d xformOp:transform = ((2, 0, 0, 0), (0, 0.035, 0, 0), (0, 0, 0.035, 0), (-2, 0, 0, 1))
            uniform token[] xformOpOrder = ["xformOp:transform"]
            rel material:binding = </Looks/Ray>
        }}
        def Sphere "Origin" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        )
        {{
            double radius = 0.16
            double3 xformOp:translate = (-4, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
            rel material:binding = </Looks/Ray>
        }}
        def Sphere "CandidateContact" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        )
        {{
            double radius = 0.22
            double3 xformOp:translate = (0, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
            rel material:binding = </Looks/Hit>
        }}
    }}
    def Cube "Ground" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        double size = 2
        float3[] extent = [(-1, -1, -1), (1, 1, 1)]
        matrix4d xformOp:transform = ((6, 0, 0, 0), (0, 0.08, 0, 0), (0, 0, 4, 0), (0, -1.35, -0.8, 1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
        rel material:binding = </Looks/Ground>
    }}
}}

def Xform "Render"
{{
    def Camera "Camera"
    {{
        uniform token projection = "perspective"
        float focalLength = 18.45401
        float horizontalAperture = 27.183644
        float verticalAperture = 15.2908
        double2 clippingRange = (0.01, 1000)
        matrix4d xformOp:transform = ((0.88235294, 0, -0.47058824, 0), (-0.1448566, 0.95144452, -0.27160613, 0), (0.4477386, 0.30782029, 0.83950987, 0), (8, 5.5, 16, 1))
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
        color3f inputs:color = (1, 0.82, 0.66)
        float inputs:intensity = 32000
        float inputs:radius = 1.5
        double3 xformOp:translate = (-5, 7, 8)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}
    def RectLight "Fill"
    {{
        color3f inputs:color = (0.48, 0.68, 1)
        float inputs:intensity = 1100
        float inputs:width = 8
        float inputs:height = 8
        double3 xformOp:translate = (6, 4, 6)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}
    def DomeLight "Environment"
    {{
        color3f inputs:color = (0.16, 0.22, 0.38)
        float inputs:intensity = 220
    }}
}}
"""


def run(arguments: list[str] | None = None) -> tuple[dict[str, Any], Any]:
    options = parse_example_arguments(example_argument_parser(__doc__), arguments)
    require_python_source(options)
    paths = EvidencePaths.create(options.evidence, EXAMPLE, "python")
    cleanup_errors: list[str] = []
    pixels: Any | None = None
    publication: Any | None = None
    result: Any | None = None
    prim_path = ""

    runtime = Runtime(f"ovspatialquery.{EXAMPLE}.python", SCENE_USDA)
    try:
        runtime.open_and_compute()
        stage = runtime.stage
        if stage is None:
            raise RuntimeError("presentation runtime returned no stage")

        public_filter = ovstage.Filter(
            [ovstage.Predicate("usd-path", ovstage.FilterOp.IN, TARGET_PATHS)]
        )

        with (
            Library.open(options.library, require_ovstage=True) as library,
            library.instance() as instance,
        ):
            population = instance.ovstage(ovstage, stage).create_population(public_filter)
            with population.enqueue_set(2) as operation:
                operation.wait()
                publication = operation.fetch_population_update()

            # PRESENTATION:API:BEGIN
            query = RaycastQuery(
                origin=RAY_ORIGIN,
                unit_direction=RAY_DIRECTION,
                min_distance=0.0,
                max_distance=RAY_DISTANCE,
                precision=QueryPrecision.BOUNDS_CANDIDATE,
            )
            result = population.target.raycast_closest(query)
            # PRESENTATION:API:END
            if result.hit is None:
                raise AssertionError("candidate ray must hit the first target AABB")
            if population.target.raycast_any(query).hit is None:
                raise AssertionError("an any-hit ray on the same path must hit")
            dictionary = ovstage.PathDictionary(stage)
            prim_path = str(dictionary.path_to_string(dictionary_path(result.hit.object_id)))
            if prim_path != HIT_PATH or result.source_ordinal != 2:
                raise AssertionError(
                    f"unexpected hit path/ordinal: {prim_path!r}, {result.source_ordinal}"
                )

            pixels = runtime.render(options.frames, ordinal=2)
            native = runtime.native_identity(library, options.sdk_root)
    finally:
        cleanup_errors.extend(runtime.close())

    if pixels is None or publication is None or result is None or result.hit is None:
        raise AssertionError("example did not produce complete query and render evidence")
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
            "target_paths": list(TARGET_PATHS),
            "ray": {
                "origin": list(RAY_ORIGIN),
                "unit_direction": list(RAY_DIRECTION),
                "min_distance": 0.0,
                "max_distance": RAY_DISTANCE,
                "precision": "aabb_candidate",
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
            "geometry_type": result.hit.geometry_type.name.lower(),
        },
        "native": native,
        "evidence": image,
        "cleanup": {"clean": not cleanup_errors, "errors": cleanup_errors},
        "errors": [],
    }
    write_semantic_json(paths.json, payload)
    validate_evidence(paths.json)
    if not options.headless:
        present_interactive("ovspatialquery: static candidate raycast", pixels)
    return payload, pixels


if __name__ == "__main__":
    try:
        run()
    except BaseException as failure:
        print(f"{EXAMPLE}: {type(failure).__name__}: {failure}", file=sys.stderr)
        raise
