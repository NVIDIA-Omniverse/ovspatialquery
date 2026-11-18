# Changelog

All notable changes to ovspatialquery are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the version is 0.x, a minor
release can change any declaration or struct layout.

## [Unreleased]

## [0.1.1] - 2026-11-18

### Added

- The first public release of ovspatialquery, a C11 library with a Python package for spatial
  queries over published geometry.
- Raycasts that return the closest hit, any hit, or multiple hits up to a capacity the caller
  sets; sphere and fixed-orientation box sweeps; and sphere, axis-aligned box, and oriented box
  overlaps.
- Populations that publish immutable versions from an ovstage stage (ovstage 0.3 and 0.4) or
  from arrays that the caller supplies, and queries that name the version they read.
- Batch plans that answer many queries per launch on the CPU or a CUDA device, with results as
  DLPack columns.
- A C SDK and a Python wheel for Linux x86-64, Linux AArch64, and Windows x86-64, with Python
  3.11, 3.12, and 3.13.
- An agent skill for coding agents, shipped in the SDK and the wheel.
- An Authenticode-signed `ovspatialquery.dll` in the Windows SDK and wheel.
- Python wheels on PyPI (`pip install ovspatialquery`), whose project page shows the package
  README and links to the guide and the repository.
- Experimental: bounding-box pair discovery (`ovspatialquery_pairs.h`) and device-prepared
  publications. Experimental declarations are marked `OVSPATIALQUERY_EXPERIMENTAL` or
  "Experimental:" in the headers and can change in any minor release.
