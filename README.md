# NVIDIA ovspatialquery

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%E2%80%933.13-blue)](#system-requirements)
[![Linux | Windows](https://img.shields.io/badge/platform-linux%20%7C%20windows-lightgrey)](#system-requirements)

**ovspatialquery** is a C and Python library for spatial queries: raycasts, sphere and box
sweeps, overlaps, and bounding-box pairs over geometry that a simulation, an animation system,
or a tool publishes. It runs on the CPU with no GPU required, and on CUDA devices when you ask
for it. It performs no dynamics simulation.

**ovspatialquery** is suited for developers building simulation, animation, sensor, and robotics
applications in which several engines, sensors, and tools need to ask where things are. They all
query the same published geometry, so none of them keeps its own scene-query structure in step
with the scene.

**ovspatialquery** integrates with the NVIDIA Omniverse [**ovstage**](https://github.com/NVIDIA-Omniverse/ovstage)
library: it publishes the prims of an ovstage stage, with their shapes and world transforms, at
the ordinals ovstage seals. Geometry that does not live in a stage can be published from arrays
your program supplies.

> [!NOTE]
> ovspatialquery is currently **pre-release** software. Releases follow SemVer: while the
> version is 0.x, a minor release may change any declaration or struct layout, so rebuild
> against each minor release; a patch release changes no declaration. Experimental
> declarations, including all of pair discovery (`ovspatialquery_pairs.h`) and
> device-prepared publications, are marked `OVSPATIALQUERY_EXPERIMENTAL` or "Experimental:"
> in the headers and may change in any minor release, also after 1.0.
> A public roadmap will be shared as ovspatialquery approaches general availability.

## Start Here

Choose a task:

* [Install and run a first query in Python](#getting-started-in-python)
* [Install the C SDK and build a first query in C/C++](#getting-started-in-cc)
* [Run an example lesson](examples/README.md)
* [Read the API documentation](#documentation)
* [Use the agent skill with a coding agent](#agent-skills)
* [Get help](#support)

## Overview

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/query-families-dark.svg">
  <img alt="A robot in a warehouse aisle queries one published scene three ways: a fan of rays, a swept sphere, and an overlap box." src="docs/images/query-families-light.svg" width="760">
</picture>

### High-Level Feature Set

* Raycasts that return the closest hit, any hit, or every hit.
* Sphere sweeps and fixed-orientation box sweeps that return the first contact, any contact, or
  the first contact with every object.
* Overlaps of a sphere, an axis-aligned box, or an oriented box that return any or every object.
* Bounding-box pair discovery between two populations or views: snapshots of one instant, and
  reporters of the contacts that started and ended since the last call.
* Geometry from an ovstage stage (`Cube`, `Sphere`, `Capsule`, `Mesh`, and convex hulls), or
  from spheres, boxes, capsules, convex hulls, and triangle meshes you register yourself.
* Exact surface hits that report the authored mesh face, or cheaper bounding-box candidates.
  A hit can also report the material the population assigns to the surface it reports.
* Batch plans that answer many queries per launch on the CPU or a CUDA device, with results as
  DLPack columns.
* A C11 interface and a Python package that exchanges arrays with NVIDIA Warp, PyTorch, CuPy,
  and any other DLPack library.

### How It Differs From an Engine's Scene Queries

A physics engine answers scene queries about its own bodies, between its own steps. ovspatialquery
is a query layer that any number of publishers and readers share. These are the differences that
matter most:

| What You Do | How ovspatialquery Handles It |
| --- | --- |
| Publish geometry | Each **population** publishes immutable **versions** on its own clock, labeled with the source's ordinal. A scene that changes rarely and bodies that move every frame are two populations. |
| Choose what a query reads | Every query names its version: the newest one, or an exact older one that is still retained. A delayed sensor reads the world as it was, without a copy. |
| Query at any time | Readers never wait for a simulation step, and a query never sees a half-published version. |
| Run many queries at once | A **batch plan** runs many rays, sweeps, or overlaps per launch. On a CUDA device, the rows run in parallel and a GPU kernel can read the result columns without a copy. |
| Interpret a hit | Every hit is tagged as a bounding-box candidate or an exact surface contact. An exact hit on a mesh returns the authored face. |
| Track contacts | A pair snapshot reports the bounding-box contacts between two targets at one instant. A pair reporter reports the contacts that started and ended since its last advance. |
| Stay within limits | You set memory and operation budgets. Work that exceeds one fails with a status, and CUDA work never falls back to the CPU. |

## Requirements

### Packaging and Dependencies

ovspatialquery is distributed as a Python wheel for each platform, on
[PyPI](https://pypi.org/project/ovspatialquery/), and a C SDK (`.zip` file) for each platform, on
the [Releases page](https://github.com/NVIDIA-Omniverse/ovspatialquery/releases) of this
repository. Both contain the same native library. The PhysX geometry libraries it uses
internally are statically linked, so no separate PhysX install is needed.

ovstage is optional: install it only to publish geometry from a stage. CUDA work uses the
installed NVIDIA driver, and neither the CUDA runtime nor the CUDA Toolkit is needed. The driver
compiles the bundled `ovspatialquery_cuda.ptx` the first time the library uses a device.

### System Requirements

Each release provides these packages:

| Platform | Python Wheel Tag | C SDK |
| --- | --- | --- |
| Linux x86-64 (glibc 2.35 or newer) | `manylinux_2_35_x86_64` | yes |
| Linux AArch64 (glibc 2.35 or newer) | `manylinux_2_35_aarch64` | yes |
| Windows x86-64 | `win_amd64` | yes |

The Python package supports these versions:

| Requirement | Supported Versions |
| --- | --- |
| Python | 3.11, 3.12, 3.13 |
| ovstage (optional, to publish from a stage) | 0.3, 0.4 |
| ovstage for the examples | 0.4 |
| NVIDIA Warp (`warp-lang`, used by the examples) | 1.18 |

Queries run on the CPU without a GPU. CUDA work has these minimums:

| Requirement | Minimum |
| --- | --- |
| GPU compute capability | 7.5 (Turing) |
| CUDA driver version | 12.8 (R570) |
| NVIDIA driver on Linux | 570.26 |
| NVIDIA driver on Windows | 570.65 |

## Getting Started

### Getting Started in Python

Install ovspatialquery from PyPI, with Warp and ovstage 0.4, which the examples use:

```bash
python -m pip install ovspatialquery warp-lang "ovstage==0.4.*"
```

The following program loads a Universal Scene Description (USD) scene with one prim into an
ovstage stage, publishes it, and casts one ray.
[`docs/site/snippets/ovstage_query.py`](docs/site/snippets/ovstage_query.py) is the commented
version:

```python
import ovstage
from ovspatialquery import Library, RaycastQuery

SCENE = """#usda 1.0
def Xform "World" {
    def Cube "Crate" {
        double size = 1
        double3 xformOp:translate = (4, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
}
"""

# ovstage seals each edit at an ordinal: 1 holds the scene, 2 its world transforms.
stage = ovstage.Stage("readme")
ovstage.population.open_usd_from_string(stage, SCENE, 1, 0.0, ovstage.PopulationDomain.RENDERING)
stage.advance_write_floor(ordinal=1).wait()
stage.compute_hierarchy(1, 2, ovstage.HierarchyComputationModel.DEFAULT_CPU)
stage.advance_write_floor(ordinal=2).wait()
world = ovstage.Filter([ovstage.Predicate("usd-path", ovstage.FilterOp.PREFIX, ["/World/"])])

with Library.open(require_ovstage=True) as library, library.instance() as instance:
    population = instance.ovstage(ovstage, stage).create_population(world)
    with population.enqueue_set(2) as publication:  # publish sealed ordinal 2
        publication.wait()

    ray = population.target.raycast_closest(
        RaycastQuery(origin=(0, 0, 0), unit_direction=(1, 0, 0), max_distance=20)
    )
    path = ovstage.PathDictionary(stage).path_to_string(ovstage.PrimPath(ray.hit.object_id))
    print(path, ray.hit.distance, ray.source_ordinal)  # /World/Crate 3.5 2

stage.destroy()
```

With ovstage 0.3, which has no `PrimPath`, pass the ID itself: `path_to_string(ray.hit.object_id)`.

### Getting Started in C/C++

Unpack the SDK, point `CMAKE_PREFIX_PATH` at its root and link the imported target:

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_app LANGUAGES C)

find_package(ovspatialquery CONFIG REQUIRED)
add_executable(my_app first_query.c)
target_link_libraries(my_app PRIVATE ovspatialquery::ovspatialquery)
```

[`docs/site/snippets/first_query.c`](docs/site/snippets/first_query.c) publishes one box from
arrays and casts a ray at it. To verify the build, run `my_app`, which prints
`ray hit 10 at distance 3.500, ordinal 1`. On Windows, first add the SDK's `bin` directory to
`PATH`, because Windows looks for `ovspatialquery.dll` next to the executable and on `PATH`, not
in the SDK.

To use the ovstage header, add `COMPONENTS ovstage` to `find_package`, then call
`find_package(ovstage CONFIG REQUIRED)` and link `ovstage::ovstage` next to ovspatialquery's
target. ovstage's C headers and CMake package are in the ovstage Python package directory
(`site-packages/ovstage`), so add that directory to `CMAKE_PREFIX_PATH` too. On Windows, if you
call ovstage functions, also add ovstage's `bin` directory to `PATH`.

## Documentation

The following resources describe the library:

| Resource | Where |
| --- | --- |
| User guide: install, first query, concepts with animated figures, examples, and API reference | Run `./dev.sh docs --serve` and open `http://127.0.0.1:8000/`. Source releases include the built guide under `docs/html/`. |
| C API, the contract | [`include/ovspatialquery/`](include/ovspatialquery/) |
| Python package | [`python/`](python/README.md) |
| Where each guide topic starts, including Troubleshooting | [`docs/README.md`](docs/README.md) |

### Building From Source

Use a Linux x86-64 or Windows x86-64 host with network access to the pinned package sources.
Bootstrap downloads the published Python, PhysX, CMake, and other dependencies. It does not use
sibling source repositories. Run these commands from the repository root:

```bash
./dev.sh bootstrap --scope library
./dev.sh test --scope library
./dev.sh python-wheel --scope library
```

The SDK installs under `_build/bootstrap/native-stable/install`, and the wheel and its isolated
verification live under `_build/python-wheel`.

A source build reports the first version found in this order:

| Source | Reported Version |
| --- | --- |
| `OVSPATIALQUERY_BUILD_VERSION=X.Y.Z[rcN]`, or `-DOVSPATIALQUERY_VERSION=...` for a plain CMake build | That version |
| `VERSION.txt` at the source root, which a released source archive or snapshot carries | The release it came from, such as `X.Y.ZrcN` |
| A Git checkout without local changes whose `HEAD` is tagged `vX.Y.Z` or `vX.Y.ZrcN` | That tag's version; the final release when both tags apply |
| Anything else | The development version in `python/pyproject.toml`; the wheel adds the commit, as in `X.Y.Z.devN+g<commit>` |

A development build prints `building development version ...; set OVSPATIALQUERY_BUILD_VERSION=X.Y.Z[rcN] to override`
when it configures. The native version getter reports the numeric `X.Y.Z` part in every case.
The editable package that bootstrap installs in `_build/venv` keeps the development version from
`python/pyproject.toml` in its package metadata; the SDK and the wheel carry the reported version.

On Linux, install GCC and G++, Ninja, curl, and the OpenGL library `libGL.so.1`. The pinned clang
uses the host GCC's C++ runtime, and ovstage loads libGL in the tests. On Ubuntu, this command
installs them:

```bash
sudo apt-get install build-essential ninja-build curl ca-certificates libgl1
```

On Windows, install Visual Studio 2022 with its C++ tools and Windows SDK, set
`OVSPATIALQUERY_COMPILER=host`, and run `dev.cmd` with `-Scope Library` instead of
`--scope library`. To include the CUDA backend, select `library-cuda` (`Library-Cuda` on Windows).

#### Use Your Own PhysX SDK

To build against an already-built PhysX SDK, set `OVSPATIALQUERY_PHYSX_SDK_DIR` before you run
the same commands. On Linux:

```bash
export OVSPATIALQUERY_PHYSX_SDK_DIR=<physx-sdk-root>
./dev.sh bootstrap --scope library
./dev.sh python-wheel --scope library
```

On Windows Command Prompt, run `set OVSPATIALQUERY_PHYSX_SDK_DIR=<physx-sdk-root>` and then the
`dev.cmd` commands. The SDK root must meet these conditions:

- It contains `include/PxPhysicsAPI.h`.
- It contains compatible static libraries under `bin/<platform>/release` and
  `bin/<platform>/debug`.
- It contains the original PhysX Apache license notice.

The variable selects an SDK; it does not compile PhysX. The SDK and wheel record the selected
SDK's fingerprints, and an invalid root fails without downloading a replacement. Keep the
variable set for build, test, verify, and benchmark commands, and use separate native build
directories to compare SDK versions. To return to the pinned public SDK, unset the variable and
run bootstrap again.

#### Check the Example Programs

After you build the wheel, run the programs under `docs/site/snippets/` against it:

```bash
./dev.sh docs --check-snippets
```

The command installs the hash-pinned Warp, NumPy, and ovstage packages into
`_build/venv/docs-snippets`, not into the library environment. Programs named `cuda_*.py` are
skipped on a host without CUDA. Add `--require-cuda` on a GPU host to make them mandatory. To use
an environment that already has the wheel and those packages, pass `--python <interpreter>`.

### Repository Layout

The repository has the following top-level areas:

| Path | Contents |
| --- | --- |
| `include/ovspatialquery/` | Public C headers: the core, the ovstage and explicit sources, and pair discovery |
| `source/` | Native implementation and its PhysX and CUDA backends |
| `python/` | The `ovspatialquery` Python package and its tests |
| `skills/` | The agent skill for coding agents and its checked programs |
| `examples/` | Rendered lessons and their native and Python helpers |
| `tests/` | Native, installed-package, and tooling tests |
| `benchmarks/` | Installed-library benchmark workloads |
| `cmake/`, `deps/`, `tools/` | Build, packaging, dependency pins, and the `dev.*` command driver |
| `third_party/`, `LICENSES/` | Bundled third-party notices and license texts |

## Examples

Every program under [`docs/site/snippets/`](docs/site/snippets/) runs as shown, in Python or C,
and has its expected output next to it. [`examples/`](examples/README.md) has 11 rendered
lessons, each in Python and C++. The lessons also need the ovRTX renderer; Python lessons
use Warp arrays and validate frames with Warp. Refer to
[Set Up the Lessons](examples/README.md#set-up-the-lessons) for optional dependencies.

## Agent Skills

[`skills/ovspatialquery/`](skills/ovspatialquery/SKILL.md) is an agent skill that teaches Claude
Code, Codex, and other coding agents to write ovspatialquery programs. Copy it to where your
harness looks for skills, such as `~/.claude/skills/` or `~/.codex/skills/`, and start from the
programs under `docs/site/snippets/` to validate an installation.
[`skills/README.md`](skills/README.md) lists the skills and where each harness looks for them.

## Releases

The [Releases page](https://github.com/NVIDIA-Omniverse/ovspatialquery/releases) provides the C
SDK and the source archive of each release, and [PyPI](https://pypi.org/project/ovspatialquery/)
provides the Python wheels. [CHANGELOG.md](CHANGELOG.md) lists
what each release changes. [System Requirements](#system-requirements) lists the supported
platforms, Python versions, ovstage versions, and CUDA minimums.

## Support

Report documentation issues, installation problems, and runtime issues through the
[NVIDIA Omniverse developer forum](https://forums.developer.nvidia.com/c/omniverse/300).

## Contributing

At this time this project is not open to external contributions. Refer to
[CONTRIBUTING.md](CONTRIBUTING.md).

## Authors and Acknowledgment

NVIDIA Corporation

## License and Security

ovspatialquery is licensed under [Apache-2.0](LICENSE). [NOTICE](NOTICE) lists bundled components
and separately obtained dependencies. Bundled third-party components keep their own terms:

| Component | License | Terms |
| --- | --- | --- |
| PhysX (`third_party/physx`) | Apache-2.0, with upstream copyright notices | [PhysX license](third_party/physx/LICENSE) |
| DLPack (`include/dlpack/dlpack.h`) and Packman (`tools/packman`) | Apache-2.0 | [Apache-2.0 text](LICENSES/Apache-2.0.txt); original notices remain in the source files |
| Built guide assets (`docs/html` in source releases) | BSD-2-Clause, BSD-3-Clause, MIT, OFL-1.1, CC-BY-4.0, and the NVIDIA License Agreement for the NVIDIA theme | Original notices under `docs/html/licenses/`, inventoried in `docs/html/licenses/components.json` |

This project downloads and installs additional third-party open source software during its
build. Review the license terms of those projects before use.

To report a security issue, refer to [SECURITY.md](SECURITY.md). Do not report security
vulnerabilities through GitHub.

---

*Copyright (c) 2026 NVIDIA Corporation.*
