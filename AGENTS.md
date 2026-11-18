# AGENTS.md: ovspatialquery

This file tells coding agents how this repository is organized and how to build, test, and
check it. To write a program that uses ovspatialquery, read the agent skill in
[`skills/ovspatialquery/SKILL.md`](skills/ovspatialquery/SKILL.md) instead.

## What the Library Owns

ovspatialquery is a C11 library with a Python package for spatial queries: raycasts, sphere and
box sweeps, overlaps, and bounding-box pairs over published geometry. It owns these parts:

- Populations of geometry and the immutable versions they publish.
- Scene queries, batch plans on the CPU or a CUDA device, and pair discovery.
- The public C headers under `include/ovspatialquery/`, which are the contract, and the
  `ovspatialquery` Python package that wraps them.

It performs no dynamics simulation, owns no scene graph, and renders nothing.

## Relationship to ovstage

ovstage is optional. When a program uses it, the responsibilities split as follows:

| Responsibility | Owner |
| --- | --- |
| Loading USD, editing prims, and computing world transforms | ovstage |
| Sealing each edit at an ordinal | ovstage |
| Publishing the prims of a stage, with their shapes and world transforms, at a sealed ordinal | ovspatialquery |
| Answering queries against a published version | ovspatialquery |

Geometry that does not live in a stage is published from arrays through
`include/ovspatialquery/ovspatialquery_explicit.h`. The supported ovstage versions are listed in
the README's System Requirements section.

## Start Here

Read these files in order:

1. [README.md](README.md): what the library does, requirements, and a first query in Python
   and C.
2. [examples/README.md](examples/README.md): the rendered lessons, each in Python and C++.
3. [skills/README.md](skills/README.md): the agent skill catalog and where to install it.

Source releases include the built user guide under `docs/html/`; [docs/README.md](docs/README.md)
lists where each topic starts, including Troubleshooting.

## Repository Layout

The repository has the following top-level areas:

| Path | Contents |
| --- | --- |
| `include/ovspatialquery/` | Public C headers: the core, the ovstage and explicit sources, and pair discovery |
| `source/` | Native implementation and its PhysX and CUDA backends |
| `python/` | The `ovspatialquery` Python package and its tests |
| `skills/` | The agent skill for coding agents and its checked programs |
| `examples/` | Rendered lessons and their native and Python helpers |
| `docs/site/snippets/` | Complete programs from the guide, each with its expected output |
| `tests/` | Native, installed-package, and tooling tests |
| `benchmarks/` | Installed-library benchmark workloads |
| `cmake/`, `deps/`, `tools/` | Build, packaging, dependency pins, and the `dev.*` command driver |

## Build and Test

Run these commands from the repository root on Linux x86-64:

```bash
./dev.sh bootstrap --scope library
./dev.sh test --scope library
./dev.sh python-wheel --scope library
./dev.sh docs --check-snippets
```

The commands download the pinned dependencies, build and install the SDK under
`_build/bootstrap/native-stable/install`, run the native and Python tests, build the wheel
under `_build/python-wheel`, and run the guide's programs against that wheel. On Windows, run
`dev.cmd` with `-Scope Library` instead of `--scope library`. The README's Building From Source
section lists the Windows prerequisites and the CUDA scope.

## Validate at Runtime

To confirm that an installed wheel loads, print the native library's version and path:

```bash
python -c "from ovspatialquery import Library; lib = Library.open(); print(lib.version, lib.path)"
```

A library that cannot be loaded raises `LibraryLoadError` with the path it tried. For an end-to-end
check, run `docs/site/snippets/first_query.py` and compare its output with
`docs/site/snippets/first_query.out`.

## Constraints

Keep these rules when you change or use the library:

- The headers under `include/ovspatialquery/` are the contract. A declaration marked
  `OVSPATIALQUERY_EXPERIMENTAL` or "Experimental:" can change in any minor release.
- While the version is 0.x, a minor release can change any declaration or struct layout, so
  rebuild against each minor release.
- CUDA work never falls back to the CPU. If CUDA is not available, creating a CUDA plan
  fails with a status.
- Every query names the version it reads, and a published version never changes.
- No public header exposes a PhysX type. Only `ovspatialquery_ovstage.h` includes the ovstage
  API; the core, explicit, and pair headers do not depend on ovstage.
