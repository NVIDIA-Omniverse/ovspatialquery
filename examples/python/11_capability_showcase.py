# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run lesson 11's installed capability showcase and render its evidence.

The native executable is the source of query truth.  This file never imports
or calls OvSpatialQuery: it consumes the completed JSON only and renders a
presentation-only Ovstage/OvRTX mirror for explanation.  CUDA is an explicit
strict mode; otherwise its station is visibly unavailable and the CPU/Ovstage
showcase remains complete.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import textwrap
from typing import Any

import ovrtx
import ovstage

from _evidence import (
    EvidencePaths,
    HEIGHT,
    WIDTH,
    example_argument_parser,
    parse_example_arguments,
    present_interactive,
    sha256_file,
    validate_evidence,
    write_semantic_json,
)
from _runtime import Runtime, frame_summary


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = "11_capability_showcase"


@dataclass(frozen=True)
class Station:
    prim_name: str
    panel: str
    lane: str
    translation: tuple[float, float, float]
    legend_x: int
    legend_label: str
    legend_color: tuple[int, int, int, int]


STATIONS = (
    Station(
        "Ovstage",
        "ovstage",
        "ovstage",
        (-4.2, 2.2, 0.0),
        20,
        "OVSTAGE 0x40 HIT / 0x80 BLOCKED",
        (26, 140, 242, 255),
    ),
    Station(
        "Explicit",
        "explicit",
        "explicit_cpu",
        (4.2, 2.2, 0.0),
        340,
        "EXPLICIT: 3 PUBLISHED + CONVEX",
        (26, 217, 102, 255),
    ),
    Station(
        "Cuda",
        "cuda",
        "cuda",
        (-4.2, -2.2, 0.0),
        650,
        "CUDA: RAY / SWEEP / OVERLAP",
        (242, 140, 20, 255),
    ),
)


def _display_point(point: list[float | int]) -> tuple[float, float, float]:
    return (float(point[0]) * 0.5 + float(point[2]) * 0.5, float(point[1]), -3.0)


def _station_contents(station: Station, payload: dict[str, Any]) -> str:
    panel = station.panel
    colors = {
        "ovstage": "(0.10, 0.55, 0.95)",
        "explicit": "(0.10, 0.85, 0.40)",
        "cuda": "(0.95, 0.55, 0.08)",
    }
    color = colors[panel]
    if panel == "cuda" and payload["lanes"]["cuda"] == "NOT_AVAILABLE":
        return """def Xform "Unavailable"
    {
        def Cube "Slash"
        {
            double size = 1
            double xformOp:rotateZ = 45
            double3 xformOp:scale = (1.0, 0.08, 0.08)
            uniform token[] xformOpOrder = ["xformOp:rotateZ", "xformOp:scale"]
            color3f[] primvars:displayColor = [(0.48, 0.52, 0.60)]
            uniform token primvars:displayColor:interpolation = "constant"
        }
        def Cube "Backslash"
        {
            double size = 1
            double xformOp:rotateZ = -45
            double3 xformOp:scale = (1.0, 0.08, 0.08)
            uniform token[] xformOpOrder = ["xformOp:rotateZ", "xformOp:scale"]
            color3f[] primvars:displayColor = [(0.48, 0.52, 0.60)]
            uniform token primvars:displayColor:interpolation = "constant"
        }
    }"""
    visibility = "inherited"
    candidate_visibility = "invisible"
    pair_visibility = "invisible"
    rejected_visibility = "invisible"
    sweep_visibility = "invisible"
    overlap_visibility = "invisible"
    source_definition = ""
    source_center = (0.0, 0.0, -3.0)
    source_radius = 0.5
    pair_first = (-0.5, -1.7, -2.8)
    pair_second = (0.5, -1.7, -2.8)
    ray: dict[str, Any]
    if panel == "ovstage":
        evidence = payload["ovstage"]
        ray = evidence["accepted"]
        source_center = _display_point(evidence["source"]["translation"])
        source_radius = float(evidence["source"]["radius"]) * 0.5
        candidate_visibility = "inherited"
        rejected_visibility = "inherited"
    elif panel == "explicit":
        evidence = payload["explicit"]
        convex = payload["convex_preparation"]
        ray = evidence["candidate"]
        objects = {item["object_id"]: item for item in evidence["objects"]}
        sphere = objects[ray["object_id"]]
        source_center = _display_point(sphere["translation"])
        source_radius = float(sphere["radius"]) * 0.5
        paired = objects[evidence["pair"][1]]
        paired_center = _display_point(paired["translation"])
        paired_radius = float(paired["radius"]) * 0.5
        pair_first = (source_center[0], -1.7, -2.8)
        pair_second = (paired_center[0], -1.7, -2.8)
        authored_points = "\n".join(
            f"""        def Sphere "Point{index}"
        {{
            double radius = 0.07
            double3 xformOp:translate = {tuple(float(value) * 0.38 for value in point)}
            uniform token[] xformOpOrder = ["xformOp:translate"]
            color3f[] primvars:displayColor = [(1.0, 0.52, 0.08)]
            uniform token primvars:displayColor:interpolation = "constant"
        }}"""
            for index, point in enumerate(convex["authored"]["points"])
        )
        prepared_contact = tuple(float(value) * 0.38 for value in convex["exact"]["position"])
        source_definition = f"""def Sphere "Source"
    {{
        double radius = {source_radius}
        double3 xformOp:translate = {source_center}
        uniform token[] xformOpOrder = ["xformOp:translate"]
        color3f[] primvars:displayColor = [{color}]
        uniform token primvars:displayColor:interpolation = "constant"
    }}
    def Sphere "PairedSphere"
    {{
        double radius = {paired_radius}
        double3 xformOp:translate = {paired_center}
        uniform token[] xformOpOrder = ["xformOp:translate"]
        color3f[] primvars:displayColor = [{color}]
        uniform token primvars:displayColor:interpolation = "constant"
    }}
    def Xform "AuthoredPoints"
    {{
{authored_points}
    }}
    def Cube "PreparedEnvelope"
    {{
        double size = 2
        double3 xformOp:scale = (0.38, 0.38, 0.38)
        uniform token[] xformOpOrder = ["xformOp:scale"]
        color3f[] primvars:displayColor = [(0.15, 0.65, 1.0)]
        float[] primvars:displayOpacity = [0.16]
        uniform token primvars:displayColor:interpolation = "constant"
        uniform token primvars:displayOpacity:interpolation = "constant"
    }}
    def Sphere "PreparedEnvelopeHit"
    {{
        double radius = 0.11
        double3 xformOp:translate = {prepared_contact}
        uniform token[] xformOpOrder = ["xformOp:translate"]
        color3f[] primvars:displayColor = [(1.0, 0.16, 0.12)]
        uniform token primvars:displayColor:interpolation = "constant"
    }}"""
        candidate_visibility = "inherited"
        pair_visibility = "inherited"
    elif panel == "cuda":
        evidence = payload["cuda"]
        ray = evidence["families"]["raycast"]
    else:
        raise RuntimeError(f"unknown showcase station: {panel}")

    if not source_definition:
        source_definition = f"""def Sphere "Source"
    {{
        double radius = {source_radius}
        double3 xformOp:translate = {source_center}
        uniform token[] xformOpOrder = ["xformOp:translate"]
        color3f[] primvars:displayColor = [{color}]
        uniform token primvars:displayColor:interpolation = "constant"
    }}"""

    ray_origin = _display_point(ray["origin"])
    contact_hit = _display_point(ray["position"])
    query_translation = tuple((ray_origin[axis] + contact_hit[axis]) * 0.5 for axis in range(3))
    query_scale = (
        max(abs(contact_hit[0] - ray_origin[0]), 0.05),
        max(abs(contact_hit[1] - ray_origin[1]), 0.05),
        0.05,
    )
    hit_translation = contact_hit
    normal = ray["normal"]
    displayed_normal = (
        float(normal[0]) * 0.5 + float(normal[2]) * 0.5,
        float(normal[1]),
    )
    normal_translation = (
        contact_hit[0] + displayed_normal[0] * 0.35,
        contact_hit[1] + displayed_normal[1] * 0.35,
        contact_hit[2] + 0.1,
    )
    normal_scale = (
        max(abs(displayed_normal[0]) * 0.7, 0.04),
        max(abs(displayed_normal[1]) * 0.7, 0.04),
        0.04,
    )
    pair_translation = tuple((pair_first[axis] + pair_second[axis]) * 0.5 for axis in range(3))
    pair_scale = (
        max(abs(pair_second[0] - pair_first[0]), 0.04),
        0.04,
        0.04,
    )
    sweep_translation = (0.0, 0.0, -3.0)
    if panel == "cuda":
        sweep = evidence["families"]["sphere_sweep"]
        triangle = ((4.0, -1.0, -1.0), (4.0, 1.0, -1.0), (4.0, 0.0, 1.0))
        displayed_triangle = tuple(_display_point(list(point)) for point in triangle)
        barycentrics = ray["feature"]["barycentrics"]
        feature_hit = tuple(
            sum(barycentrics[index] * point[axis] for index, point in enumerate(displayed_triangle))
            for axis in range(3)
        )
        contact_hit = _display_point(ray["position"])
        if any(abs(feature_hit[index] - contact_hit[index]) > 1.0e-6 for index in range(3)):
            raise RuntimeError("CUDA contact and triangle feature evidence disagree")
        source_definition = f"""def Mesh "Source"
    {{
        point3f[] points = [{displayed_triangle[0]}, {displayed_triangle[1]}, {displayed_triangle[2]}]
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = {ray["feature"]["source_corners"]}
        color3f[] primvars:displayColor = [{color}]
        uniform token primvars:displayColor:interpolation = "constant"
    }}"""
        sweep_translation = _display_point(sweep["query_position"])
        sweep_visibility = "inherited"
        overlap_visibility = "inherited"
    return f'''{source_definition}
    def Xform "CandidateBounds"
    {{
        token visibility = "{candidate_visibility}"
        def Cube "Top"
        {{
            double size = 1
            double3 xformOp:translate = ({source_center[0]}, {source_center[1] + source_radius}, -2.9)
            double3 xformOp:scale = ({source_radius * 2.0}, 0.035, 0.035)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            color3f[] primvars:displayColor = [(0.55, 0.58, 0.64)]
            uniform token primvars:displayColor:interpolation = "constant"
        }}
        def Cube "Bottom"
        {{
            double size = 1
            double3 xformOp:translate = ({source_center[0]}, {source_center[1] - source_radius}, -2.9)
            double3 xformOp:scale = ({source_radius * 2.0}, 0.035, 0.035)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            color3f[] primvars:displayColor = [(0.55, 0.58, 0.64)]
            uniform token primvars:displayColor:interpolation = "constant"
        }}
        def Cube "Left"
        {{
            double size = 1
            double3 xformOp:translate = ({source_center[0] - source_radius}, {source_center[1]}, -2.9)
            double3 xformOp:scale = (0.035, {source_radius * 2.0}, 0.035)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            color3f[] primvars:displayColor = [(0.55, 0.58, 0.64)]
            uniform token primvars:displayColor:interpolation = "constant"
        }}
        def Cube "Right"
        {{
            double size = 1
            double3 xformOp:translate = ({source_center[0] + source_radius}, {source_center[1]}, -2.9)
            double3 xformOp:scale = (0.035, {source_radius * 2.0}, 0.035)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
            color3f[] primvars:displayColor = [(0.55, 0.58, 0.64)]
            uniform token primvars:displayColor:interpolation = "constant"
        }}
    }}
    def Cube "QueryRay"
    {{
        token visibility = "{visibility}"
        double size = 1
        double3 xformOp:translate = {query_translation}
        double3 xformOp:scale = {query_scale}
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        color3f[] primvars:displayColor = [(0.95, 0.95, 0.25)]
        uniform token primvars:displayColor:interpolation = "constant"
    }}
    def Sphere "Hit"
    {{
        token visibility = "{visibility}"
        double radius = 0.16
        double3 xformOp:translate = {hit_translation}
        uniform token[] xformOpOrder = ["xformOp:translate"]
        color3f[] primvars:displayColor = [(1, 0.2, 0.15)]
        uniform token primvars:displayColor:interpolation = "constant"
    }}
    def Cube "HitNormal"
    {{
        token visibility = "{visibility}"
        double size = 1
        double3 xformOp:translate = {normal_translation}
        double3 xformOp:scale = {normal_scale}
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        color3f[] primvars:displayColor = [(1, 0.2, 0.15)]
        uniform token primvars:displayColor:interpolation = "constant"
    }}
    def Cube "PairLink"
    {{
        token visibility = "{pair_visibility}"
        double size = 1
        double3 xformOp:translate = {pair_translation}
        double3 xformOp:scale = {pair_scale}
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        color3f[] primvars:displayColor = [(0.9, 0.6, 0.15)]
        uniform token primvars:displayColor:interpolation = "constant"
    }}
    def Sphere "PairFirst"
    {{
        token visibility = "{pair_visibility}"
        double radius = 0.16
        double3 xformOp:translate = {pair_first}
        uniform token[] xformOpOrder = ["xformOp:translate"]
        color3f[] primvars:displayColor = [(0.9, 0.6, 0.15)]
        uniform token primvars:displayColor:interpolation = "constant"
    }}
    def Sphere "PairSecond"
    {{
        token visibility = "{pair_visibility}"
        double radius = 0.16
        double3 xformOp:translate = {pair_second}
        uniform token[] xformOpOrder = ["xformOp:translate"]
        color3f[] primvars:displayColor = [(0.9, 0.6, 0.15)]
        uniform token primvars:displayColor:interpolation = "constant"
    }}
    def Sphere "SweepCenter"
    {{
        token visibility = "{sweep_visibility}"
        double radius = 0.22
        double3 xformOp:translate = {sweep_translation}
        uniform token[] xformOpOrder = ["xformOp:translate"]
        color3f[] primvars:displayColor = [(0.2, 0.8, 1.0)]
        uniform token primvars:displayColor:interpolation = "constant"
    }}
    def Cube "RejectedRay"
    {{
        token visibility = "{rejected_visibility}"
        double size = 1
        double3 xformOp:translate = ({query_translation[0]}, -1.2, -2.9)
        double3 xformOp:scale = {query_scale}
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        color3f[] primvars:displayColor = [(0.42, 0.46, 0.52)]
        uniform token primvars:displayColor:interpolation = "constant"
    }}
    def Xform "RejectedByBits"
    {{
        token visibility = "{rejected_visibility}"
        double3 xformOp:translate = ({contact_hit[0]}, -1.2, -2.8)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Cube "Slash"
        {{
            double size = 1
            double xformOp:rotateZ = 45
            double3 xformOp:scale = (0.5, 0.055, 0.055)
            uniform token[] xformOpOrder = ["xformOp:rotateZ", "xformOp:scale"]
            color3f[] primvars:displayColor = [(0.92, 0.14, 0.14)]
            uniform token primvars:displayColor:interpolation = "constant"
        }}
        def Cube "Backslash"
        {{
            double size = 1
            double xformOp:rotateZ = -45
            double3 xformOp:scale = (0.5, 0.055, 0.055)
            uniform token[] xformOpOrder = ["xformOp:rotateZ", "xformOp:scale"]
            color3f[] primvars:displayColor = [(0.92, 0.14, 0.14)]
            uniform token primvars:displayColor:interpolation = "constant"
        }}
    }}
    def Cube "SweepPath"
    {{
        token visibility = "{sweep_visibility}"
        double size = 1
        double3 xformOp:translate = (0.875, -0.45, -2.9)
        double3 xformOp:scale = (1.75, 0.045, 0.045)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        color3f[] primvars:displayColor = [(0.2, 0.8, 1.0)]
        uniform token primvars:displayColor:interpolation = "constant"
    }}
    def Sphere "OverlapVolume"
    {{
        token visibility = "{overlap_visibility}"
        double radius = 0.5
        double3 xformOp:translate = (2, 0, -3)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        color3f[] primvars:displayColor = [(1.0, 0.18, 0.72)]
        float[] primvars:displayOpacity = [0.42]
        uniform token primvars:displayColor:interpolation = "constant"
        uniform token primvars:displayOpacity:interpolation = "constant"
    }}
'''


def _overview_scene(payload: dict[str, Any]) -> str:
    groups: list[str] = []
    for station in STATIONS:
        contents = textwrap.indent(
            _station_contents(station, payload),
            "        ",
        )
        groups.append(
            f'''    def Xform "{station.prim_name}"
    {{
        double3 xformOp:translate = {station.translation}
        double3 xformOp:scale = (0.72, 0.72, 0.72)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        def Cube "Panel"
        {{
            double size = 2
            matrix4d xformOp:transform = ((3.4,0,0,0),(0,2.3,0,0),(0,0,0.05,0),(0,0,-5.2,1))
            uniform token[] xformOpOrder = ["xformOp:transform"]
            color3f[] primvars:displayColor = [(0.035, 0.055, 0.09)]
            uniform token primvars:displayColor:interpolation = "constant"
        }}
{contents}
    }}'''
        )
    return f"""#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
    metersPerUnit = 1
)

def Xform "World"
{{
{chr(10).join(groups)}
}}

def Xform "Render"
{{
    def Camera "Camera"
    {{
        uniform token projection = "perspective"
        float focalLength = 22
        float horizontalAperture = 27.183644
        float verticalAperture = 15.2908
        double2 clippingRange = (0.01, 1000)
        matrix4d xformOp:transform = ((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,12,1))
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
        float inputs:intensity = 26000
        float inputs:radius = 1.5
        double3 xformOp:translate = (-5, 8, 10)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}
    def RectLight "Fill"
    {{
        color3f inputs:color = (0.48, 0.68, 1)
        float inputs:intensity = 900
        float inputs:width = 10
        float inputs:height = 10
        double3 xformOp:translate = (7, 5, 8)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}
    def DomeLight "Environment"
    {{
        color3f inputs:color = (0.12, 0.17, 0.28)
        float inputs:intensity = 180
    }}
}}
"""


def _native_path(native_dir: Path) -> Path:
    name = "11_capability_showcase.exe" if os.name == "nt" else "11_capability_showcase"
    release = native_dir / "Release" / name
    return release if release.is_file() else native_dir / name


def _verified_library(sdk_root: Path, platform_name: str) -> Path:
    return sdk_root / (
        "bin/ovspatialquery.dll" if platform_name == "win32" else "lib/libovspatialquery.so"
    )


def _require_verified_library(sdk_root: Path, library: Path, platform_name: str) -> Path:
    expected = _verified_library(sdk_root.resolve(), platform_name)
    if library.resolve() != expected:
        raise ValueError("--library must be the platform library under --sdk-root")
    return expected


def _run(command: list[str], *, environment: dict[str, str] | None = None) -> None:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        text=True,
        capture_output=True,
        env=environment,
        timeout=240,
    )
    if completed.returncode != 0:
        # MSBuild reports compiler errors on stdout, so a failure shows both streams.
        output = "\n".join(
            part for part in (completed.stdout.strip(), completed.stderr.strip()) if part
        )
        raise RuntimeError(f"command failed ({completed.returncode}): {output}")
    if completed.stdout:
        print(completed.stdout, end="")


def _configured_tool(sdk_root: Path, variable: str) -> Path:
    cache = sdk_root.resolve().parent / "CMakeCache.txt"
    if not cache.is_file():
        tool = shutil.which({"CMAKE_COMMAND": "cmake", "CMAKE_CTEST_COMMAND": "ctest"}[variable])
        if tool is None:
            raise FileNotFoundError(f"{variable} is unavailable; install CMake and put it on PATH")
        return Path(tool).resolve()
    prefix = f"{variable}:INTERNAL="
    matches = [
        line.removeprefix(prefix)
        for line in cache.read_text(encoding="utf-8").splitlines()
        if line.startswith(prefix)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"configured build does not identify {variable}")
    tool = Path(matches[0])
    if not tool.is_absolute() or not tool.is_file():
        raise FileNotFoundError(f"configured {variable} is unavailable")
    return tool.resolve()


def _configure_and_build(options: argparse.Namespace) -> tuple[Path, Path]:
    import ovstage

    native_dir = options.native_dir.resolve()
    native_dir.mkdir(parents=True, exist_ok=True)
    ovstage_root = Path(ovstage.__file__).resolve().parent
    cmake = _configured_tool(options.sdk_root, "CMAKE_COMMAND")
    _run(
        [
            str(cmake),
            "-S",
            str(PROJECT_ROOT / "examples" / "11_capability_showcase"),
            "-B",
            str(native_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DOVSPATIALQUERY_INSTALL_DIR={options.sdk_root.resolve()}",
            f"-DOVSTAGE_DIR={ovstage_root}",
            "-DOVSPATIALQUERY_ENABLE_CUDA_CONTRACT=" + ("ON" if options.require_cuda else "OFF"),
        ]
    )
    _run([str(cmake), "--build", str(native_dir), "--config", "Release"])
    executable = _native_path(native_dir)
    if not executable.is_file():
        raise FileNotFoundError(f"installed showcase executable is missing: {executable}")
    return executable, native_dir


def _qualified_ctest(sdk_root: Path) -> Path:
    return _configured_tool(sdk_root, "CMAKE_CTEST_COMMAND")


def _validate(payload: Any, *, require_cuda: bool = False) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError("native showcase emitted an invalid evidence schema")
    if payload.get("status") != "PASS" or payload.get("cleanup") is not True:
        raise RuntimeError("native showcase did not prove clean PASS lifecycle")
    lanes = payload.get("lanes")
    if not isinstance(lanes, dict) or any(
        lanes.get(name) != "PASS" for name in ("core", "ovstage", "participation", "explicit_cpu")
    ):
        raise RuntimeError("native showcase did not prove required CPU/Ovstage lanes")
    cuda_state = lanes.get("cuda")
    if cuda_state not in ("PASS", "NOT_AVAILABLE"):
        raise RuntimeError("native showcase emitted an invalid CUDA state")
    if require_cuda and cuda_state != "PASS":
        raise RuntimeError("this contract requires real CUDA evidence")
    expected_scenarios = {
        "ovstage": ("Ovstage participation accept/reject", "BOUNDS_CANDIDATE"),
        "explicit": (
            "Explicit publication, candidate, and pair",
            "BOUNDS_CANDIDATE",
        ),
        "cuda": (
            ("CUDA exact family columns", "EXACT_SURFACE")
            if cuda_state == "PASS"
            else ("CUDA unavailable", "NOT_AVAILABLE")
        ),
    }
    scenarios = payload.get("scenarios")
    observed_scenarios = (
        {
            scenario.get("panel"): (scenario.get("label"), scenario.get("truth"))
            for scenario in scenarios
            if isinstance(scenario, dict)
        }
        if isinstance(scenarios, list)
        else {}
    )
    if observed_scenarios != expected_scenarios:
        raise RuntimeError("native showcase emitted invalid visualization scenarios")
    ovstage_evidence = payload.get("ovstage")
    ovstage_publication = (
        ovstage_evidence.get("publication") if isinstance(ovstage_evidence, dict) else None
    )
    if not isinstance(ovstage_evidence, dict) or any(
        (
            not isinstance(ovstage_publication, dict),
            ovstage_publication.get("object_count") != 1
            if isinstance(ovstage_publication, dict)
            else True,
            ovstage_evidence.get("accepted_query_bits") != 0x40,
            ovstage_evidence.get("rejected_query_bits") != 0x80,
            ovstage_evidence.get("source") != {"translation": [2, 0, 0], "radius": 1},
        )
    ):
        raise RuntimeError("native showcase emitted invalid Ovstage participation evidence")
    accepted = ovstage_evidence.get("accepted")
    if not isinstance(accepted, dict) or any(
        (
            accepted.get("truth") != "BOUNDS_CANDIDATE",
            accepted.get("source_ordinal") != ovstage_publication.get("source_ordinal"),
            accepted.get("object_id", 0) <= 0,
            accepted.get("geometry_type") != "SPHERE",
            accepted.get("origin") != [-3, 0, 0],
            accepted.get("position") != [1, 0, 0],
            accepted.get("normal") != [-1, 0, 0],
            accepted.get("distance") != 4,
        )
    ):
        raise RuntimeError("native showcase emitted invalid Ovstage accepted hit")
    explicit_evidence = payload.get("explicit")
    explicit_objects = [
        {"object_id": 101, "geometry_type": "SPHERE", "translation": [4, 0, 0], "radius": 2},
        {"object_id": 202, "geometry_type": "SPHERE", "translation": [5, 0, 0], "radius": 1},
        {"object_id": 404, "geometry_type": "CONVEX", "translation": [0, 0, 0]},
    ]
    if not isinstance(explicit_evidence, dict) or any(
        (
            explicit_evidence.get("publication") != {"source_ordinal": 1, "object_count": 3},
            explicit_evidence.get("objects") != explicit_objects,
            explicit_evidence.get("pair") != [101, 202],
        )
    ):
        raise RuntimeError("native showcase emitted invalid explicit publication evidence")
    candidate = explicit_evidence.get("candidate")
    if not isinstance(candidate, dict) or any(
        (
            candidate.get("truth") != "BOUNDS_CANDIDATE",
            candidate.get("source_ordinal") != 1,
            candidate.get("object_id") != 101,
            candidate.get("geometry_type") != "SPHERE",
            candidate.get("geometry_value") != 7,
            candidate.get("origin") != [-2, 0, 0],
            candidate.get("position") != [2, 0, 0],
            candidate.get("normal") != [-1, 0, 0],
            candidate.get("distance") != 4,
        )
    ):
        raise RuntimeError("native showcase emitted invalid explicit candidate hit")
    convex = payload.get("convex_preparation")
    authored = convex.get("authored") if isinstance(convex, dict) else None
    convex_exact = convex.get("exact") if isinstance(convex, dict) else None
    expected_points = [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
        [0, 0, 0],
    ]
    if (
        not isinstance(authored, dict)
        or authored.get("points") != expected_points
        or authored.get("dent_position") != [0, 0, 0]
        or convex.get("publication") != {"source_ordinal": 1, "object_count": 3}
        or not isinstance(convex_exact, dict)
        or any(
            (
                convex_exact.get("truth") != "EXACT_SURFACE",
                convex_exact.get("source_ordinal") != 1,
                convex_exact.get("object_id") != 404,
                convex_exact.get("geometry_type") != "CONVEX",
                convex_exact.get("geometry_value") != 91,
                convex_exact.get("origin") != [0, 3, 0],
                convex_exact.get("position") != [0, 1, 0],
                convex_exact.get("normal") != [0, 1, 0],
                convex_exact.get("distance") != 2,
                convex_exact.get("feature") != "NONE",
            )
        )
    ):
        raise RuntimeError("native showcase did not prove the prepared convex envelope")
    cuda = payload.get("cuda")
    if cuda_state == "NOT_AVAILABLE":
        if (
            not isinstance(cuda, dict)
            or set(cuda) != {"state", "reason"}
            or any(
                (
                    cuda.get("state") != "NOT_AVAILABLE",
                    not isinstance(cuda.get("reason"), str),
                    not cuda.get("reason"),
                )
            )
        ):
            raise RuntimeError("native showcase omitted the CUDA unavailability reason")
        return payload
    if (
        not isinstance(cuda, dict)
        or cuda.get("device") != "kDLCUDA:0"
        or cuda.get("source_ordinal") != 9
        or cuda.get("state") != "PASS"
    ):
        raise RuntimeError("native showcase emitted invalid CUDA device/source truth")
    families = cuda.get("families")
    if not isinstance(families, dict) or set(families) != {"raycast", "sphere_sweep", "overlap"}:
        raise RuntimeError("native showcase omitted a CUDA query family")
    for name, family in families.items():
        if not isinstance(family, dict) or any(
            (
                family.get("truth") != "EXACT_SURFACE",
                family.get("row_status") != "SUCCESS",
                family.get("stored_hit_count") != 1,
                family.get("hit_offset") != 0,
                family.get("hit_count") != 1,
                family.get("truncated") != 0,
                family.get("object_id") != 303,
                family.get("geometry_type") != "TRIANGLE_MESH",
                family.get("geometry_value") != 88,
            )
        ):
            raise RuntimeError(f"native showcase emitted invalid CUDA {name} columns")
    for name in ("raycast", "sphere_sweep"):
        expected = {
            "origin": [0, 0, 0],
            "position": [4, 0, 0],
            "normal": [-1, 0, 0],
            "distance": 4 if name == "raycast" else 3.5,
        }
        if any(
            family_value != families[name].get(field) for field, family_value in expected.items()
        ):
            raise RuntimeError(f"native showcase emitted invalid CUDA {name} contact")
        if name == "sphere_sweep" and families[name].get("query_position") != [3.5, 0, 0]:
            raise RuntimeError("native showcase emitted invalid CUDA sweep query position")
        feature = families[name].get("feature")
        if not isinstance(feature, dict) or any(
            (
                feature.get("kind") != "TRIANGLE",
                feature.get("source_face") != 0,
                feature.get("source_corners") != [0, 1, 2],
                feature.get("barycentrics") != [0.25, 0.25, 0.5],
            )
        ):
            raise RuntimeError(f"native showcase emitted invalid CUDA {name} feature")
    return payload


def _arguments(arguments: list[str] | None) -> argparse.Namespace:
    parser = example_argument_parser(__doc__)
    parser.add_argument(
        "--native-dir",
        type=Path,
        default=PROJECT_ROOT / "_build" / "examples" / "11-capability-showcase",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Run the strict real-CUDA contract instead of the portable CPU contract",
    )
    options = parse_example_arguments(parser, arguments)
    if options.sdk_root is None or options.library is None:
        parser.error("lesson 11 requires one verified source-checkout install")
    return options


def _native_identity(
    runtime: Runtime,
    library: Path,
    sdk_root: Path,
    native_json: Path,
) -> dict[str, Any]:
    if runtime.renderer is None:
        raise RuntimeError("presentation runtime returned no renderer")
    include = sdk_root.resolve() / "include" / "ovspatialquery"
    spatial: dict[str, Any] = {
        "library": str(library.resolve()),
        "library_sha256": sha256_file(library),
    }
    for label, name in (
        ("core_header", "ovspatialquery.h"),
        ("explicit_header", "ovspatialquery_explicit.h"),
        ("ovstage_header", "ovspatialquery_ovstage.h"),
        ("pairs_header", "ovspatialquery_pairs.h"),
    ):
        path = include / name
        if not path.is_file():
            raise FileNotFoundError(f"installed header is required: {path}")
        spatial[label] = str(path)
        spatial[f"{label}_sha256"] = sha256_file(path)
    return {
        "ovspatialquery": spatial,
        "producer": {
            "kind": "installed-public-c-abi",
            "json": str(native_json.resolve()),
            "json_sha256": sha256_file(native_json),
        },
        "ovstage": {
            "package": ovstage.__version__,
            "runtime": list(ovstage.library_version()),
        },
        "ovrtx": {
            "package": ovrtx.__version__,
            "runtime": list(runtime.renderer.version),
        },
    }


def run(arguments: list[str] | None = None) -> tuple[dict[str, Any], Any]:
    options = _arguments(arguments)
    library = _require_verified_library(options.sdk_root, options.library, sys.platform)
    _, native_dir = _configure_and_build(options)
    paths = EvidencePaths.create(options.evidence, EXAMPLE, "python")
    native_json = paths.root / "native.json"
    environment = os.environ.copy()
    loader = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
    environment[loader] = os.pathsep.join((str(library.parent), environment.get(loader, "")))
    contract_stem = "cuda" if options.require_cuda else "cpu"
    contract_name = f"capability_showcase_{contract_stem}_headless_contract"
    contract_json = native_dir / f"{contract_stem}-headless-contract.json"
    contract_json.unlink(missing_ok=True)
    _run(
        [
            str(_qualified_ctest(options.sdk_root)),
            "--test-dir",
            str(native_dir),
            "-C",
            "Release",
            "-R",
            f"^{contract_name}$",
            "--interactive-debug-mode",
            "0",
            "--output-on-failure",
        ],
        environment=environment,
    )
    if not contract_json.is_file():
        raise RuntimeError(
            f"{contract_stem.upper()} headless contract did not write native evidence"
        )
    native_json.write_bytes(contract_json.read_bytes())
    payload = _validate(
        json.loads(native_json.read_text(encoding="utf-8")),
        require_cuda=options.require_cuda,
    )
    print(
        "[showcase] installed lanes: "
        + ", ".join(f"{name}={status}" for name, status in payload["lanes"].items())
    )
    if payload["cuda"]["state"] == "PASS":
        for name, family in payload["cuda"]["families"].items():
            print(
                f"[showcase] CUDA {name}: {family['truth']}, "
                f"object={family['object_id']}, hits={family['stored_hit_count']}"
            )
    else:
        print(f"[showcase] CUDA NOT_AVAILABLE: {payload['cuda']['reason']}")
    cuda_legend = (
        "orange=CUDA mesh, yellow=ray, cyan=sweep, magenta=overlap"
        if payload["cuda"]["state"] == "PASS"
        else "grey=CUDA unavailable"
    )
    print(f"[showcase] visual legend: blue=Ovstage bit hit/reject, green=explicit, {cuda_legend}")
    cleanup_errors: list[str] = []
    runtime: Runtime | None = None
    pixels: Any | None = None
    native: dict[str, Any] | None = None
    try:
        runtime = Runtime(
            "ovspatialquery.11_capability_showcase.python",
            _overview_scene(payload),
        )
        runtime.open_and_compute()
        pixels = runtime.render(options.frames, 2)
        native = _native_identity(runtime, library, options.sdk_root, native_json)
    finally:
        if runtime is not None:
            cleanup_errors.extend(runtime.close())
    if pixels is None or native is None:
        raise AssertionError("showcase did not produce complete render evidence")
    image = frame_summary(pixels)
    semantic = {
        "schema_version": 2,
        "example": EXAMPLE,
        "language": "python",
        "execution": {
            "headless": bool(options.headless),
            "frames": options.frames,
            "resolution": [WIDTH, HEIGHT],
            "interactive": not options.headless,
        },
        "inputs": {
            "native_truth": "installed public C ABI",
            "presentation": "one combined Ovstage/OvRTX scene",
            "device": payload["cuda"].get("device"),
            "scenarios": payload["scenarios"],
        },
        "publication": {
            "ovstage": payload["ovstage"]["publication"],
            "explicit": payload["explicit"]["publication"],
            "cuda_source_ordinal": payload["cuda"].get("source_ordinal"),
            "participation": {
                "accepted_query_bits": payload["ovstage"]["accepted_query_bits"],
                "rejected_query_bits": payload["ovstage"]["rejected_query_bits"],
            },
            "pair": payload["explicit"]["pair"],
            "convex": payload["convex_preparation"]["publication"],
        },
        "result": {
            "lanes": payload["lanes"],
            "ovstage": payload["ovstage"],
            "explicit": payload["explicit"],
            "convex_preparation": payload["convex_preparation"],
            "cuda": payload["cuda"],
        },
        "native": native,
        "evidence": image,
        "cleanup": {
            "clean": payload["cleanup"] is True and not cleanup_errors,
            "errors": cleanup_errors,
        },
        "errors": [],
    }
    write_semantic_json(paths.json, semantic)
    validate_evidence(paths.json)
    print(json.dumps({"semantic": str(paths.json)}))
    if not options.headless:
        print("[showcase] opening live OvRTX overview; Escape or close exits")
        present_interactive("ovspatialquery: installed capability showcase", pixels)
    return semantic, pixels


if __name__ == "__main__":
    try:
        run()
    except BaseException as failure:
        print(f"{EXAMPLE}: {type(failure).__name__}: {failure}", file=sys.stderr)
        raise
