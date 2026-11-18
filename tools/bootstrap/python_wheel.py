# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a deterministic platform wheel from a verified install."""

from __future__ import annotations

import base64
import csv
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import tomllib
from typing import Any, Final
import zipfile

import contract
import consumer_content


PACKAGE: Final = "ovspatialquery"
DIST_INFO_TEMPLATE: Final = "ovspatialquery-{version}.dist-info"
_ZIP_TIMESTAMP: Final = (1980, 1, 1, 0, 0, 0)
LEGAL_FILES: Final = {
    "LICENSE": "LICENSE",
    "NOTICE": "NOTICE",
    "licenses/dlpack-LICENSE.txt": "LICENSES/Apache-2.0.txt",
    "licenses/physx-LICENSE.md": "third_party/physx/LICENSE",
    "licenses/physx-PROVENANCE.json": "deps/physx-license-provenance.json",
}


class WheelBuildError(RuntimeError):
    """The source project or verified native install cannot form a wheel."""


@dataclass(frozen=True)
class WheelArtifact:
    path: Path
    sha256: str
    size: int
    tag: str
    manifest: dict[str, object]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _record_digest(payload: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
    return f"sha256={encoded.rstrip(b'=').decode('ascii')}"


def _project(source_root: Path) -> dict[str, Any]:
    """The pyproject [project] fields the wheel METADATA repeats."""

    project = source_root / "pyproject.toml"
    try:
        with project.open("rb") as stream:
            table = tomllib.load(stream)["project"]
        fields: dict[str, Any] = {
            key: str(table[key])
            for key in ("version", "description", "requires-python", "license", "readme")
        }
        fields["authors"] = [str(author["name"]) for author in table["authors"]]
        fields["keywords"] = [str(keyword) for keyword in table["keywords"]]
        fields["classifiers"] = [str(classifier) for classifier in table["classifiers"]]
        fields["urls"] = {str(label): str(url) for label, url in table["urls"].items()}
    except (KeyError, OSError, TypeError, tomllib.TOMLDecodeError) as failure:
        raise WheelBuildError(
            f"Python project metadata is invalid: {project}: {failure}"
        ) from failure
    if not all(fields.values()) or any("\n" in str(value) for value in fields.values()):
        raise WheelBuildError(f"Python project has an empty or multi-line field: {project}")
    return fields


LICENSE_EXPRESSION: Final = (
    _project(contract.PYTHON_ROOT)["license"]
    + " AND BSD-2-Clause AND BSD-3-Clause AND MIT AND CC-BY-4.0 AND OFL-1.1"
    + " AND LicenseRef-NVIDIA-License-Agreement"
)


def _read_required(path: Path, label: str) -> bytes:
    try:
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.read_bytes()
    except OSError as failure:
        raise WheelBuildError(f"Required {label} is missing or unreadable: {path}") from failure


def _source_payloads(source_root: Path) -> dict[str, bytes]:
    package_root = source_root / "src" / PACKAGE
    sources = {
        f"{PACKAGE}/{path.name}": path.read_bytes()
        for path in sorted(package_root.glob("*.py"))
        if path.is_file()
    }
    if f"{PACKAGE}/__init__.py" not in sources:
        raise WheelBuildError(f"Python package sources are incomplete: {package_root}")
    return sources


def _wheel_layout(target_platform: str) -> contract.PlatformLayout:
    try:
        return contract.platform_layout(target_platform)
    except contract.BootstrapError:
        raise WheelBuildError(f"Python wheel is not qualified for {target_platform}") from None


def _native_payloads(
    install_root: Path,
    layout: contract.PlatformLayout,
) -> tuple[dict[str, bytes], list[dict[str, object]], bool]:
    relative = Path(layout.library_dir) / layout.library_file
    wheel_path = f"{PACKAGE}/_native/{layout.library_dir}/{layout.library_file}"

    native = _read_required(
        install_root / relative,
        "verified native artifact",
    )
    payloads = {wheel_path: native}
    identities: list[dict[str, object]] = [
        {
            "install_path": relative.as_posix(),
            "wheel_path": wheel_path,
            "sha256": _sha256(native),
            "size": len(native),
        }
    ]
    # The PTX is installed, and bundled, next to the library that loads it.
    ptx_relative = relative.with_name("ovspatialquery_cuda.ptx")
    ptx_path = install_root / ptx_relative
    has_cuda = ptx_path.is_file()
    if has_cuda:
        ptx = _read_required(ptx_path, "installed CUDA PTX")
        ptx_wheel_path = wheel_path.rsplit("/", 1)[0] + "/ovspatialquery_cuda.ptx"
        payloads[ptx_wheel_path] = ptx
        identities.append(
            {
                "install_path": ptx_relative.as_posix(),
                "wheel_path": ptx_wheel_path,
                "sha256": _sha256(ptx),
                "size": len(ptx),
            }
        )
    return payloads, identities, has_cuda


def _installed_contract(
    install_root: Path,
) -> tuple[list[dict[str, object]], bool]:
    relative_paths = (
        Path("include/ovspatialquery/ovspatialquery.h"),
        Path("include/dlpack/dlpack.h"),
        *(Path(relative) for relative in LEGAL_FILES),
    )
    companion = Path("include/ovspatialquery/ovspatialquery_ovstage.h")
    if (install_root / companion).is_file():
        relative_paths = (*relative_paths, companion)
    identities = []
    for relative in relative_paths:
        payload = _read_required(
            install_root / relative,
            "verified install contract",
        )
        identities.append(
            {
                "install_path": relative.as_posix(),
                "sha256": _sha256(payload),
                "size": len(payload),
            }
        )
    return identities, (install_root / companion).is_file()


def _metadata(
    version: str,
    project: dict[str, Any],
    readme: str,
    license_expression: str,
    documentation: dict[str, bytes],
) -> bytes:
    # The README is the body, after the header block; PyPI renders it as the project page.
    return (
        "Metadata-Version: 2.4\n"
        f"Name: {PACKAGE}\n"
        f"Version: {version}\n"
        f"Summary: {project['description']}\n"
        f"Keywords: {','.join(project['keywords'])}\n"
        + "".join(f"Author: {author}\n" for author in project["authors"])
        + "".join(f"Classifier: {classifier}\n" for classifier in project["classifiers"])
        + "".join(f"Project-URL: {label}, {url}\n" for label, url in project["urls"].items())
        + f"Requires-Python: {project['requires-python']}\n"
        f"License-Expression: {license_expression}\n"
        "License-File: LICENSE\n"
        "License-File: NOTICE\n"
        "License-File: dlpack-LICENSE.txt\n"
        "License-File: physx-LICENSE.md\n"
        + "".join(f"License-File: documentation/{name}\n" for name in sorted(documentation))
        + "Description-Content-Type: text/markdown\n"
        "\n" + readme
    ).encode("utf-8")


def _wheel_metadata(tag: str) -> bytes:
    return (
        "Wheel-Version: 1.0\n"
        "Generator: ovspatialquery-python-wheel\n"
        "Root-Is-Purelib: false\n"
        f"Tag: {tag}\n"
        "\n"
    ).encode("utf-8")


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, _ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    return info


def _record(payloads: dict[str, bytes], record_name: str) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    for name in sorted(payloads):
        payload = payloads[name]
        writer.writerow((name, _record_digest(payload), len(payload)))
    writer.writerow((record_name, "", ""))
    return stream.getvalue().encode("utf-8")


def build(
    *,
    source_root: Path,
    install_root: Path,
    destination: Path,
    target_platform: str,
    source_revision: str,
    source_dirty: bool,
    version: str | None = None,
) -> WheelArtifact:
    """Build one deterministic wheel from already-verified inputs."""

    source_root = source_root.resolve()
    install_root = install_root.resolve()
    destination = destination.resolve()
    layout = _wheel_layout(target_platform)
    tag = layout.wheel_platform_tag
    project = _project(source_root)
    version = version or project["version"]
    sources = _source_payloads(source_root)
    revision = source_revision + ("-dirty" if source_dirty else "")
    content = consumer_content.collect(
        source_root.parent, "wheel", version=version, source_revision=revision
    )
    for name, payload in consumer_content.collect(
        source_root.parent,
        version=version.split("+", 1)[0],
        source_revision=revision,
    ).items():
        installed = install_root / consumer_content.SDK_CONTENT / name
        if _read_required(installed, "installed consumer content") != payload:
            raise WheelBuildError(f"Installed consumer content differs from source: {installed}")
    native, native_identities, has_cuda = _native_payloads(
        install_root,
        layout,
    )
    installed_contract, has_ovstage = _installed_contract(install_root)
    legal = {
        relative: _read_required(install_root / relative, "installed legal file")
        for relative in LEGAL_FILES
    }
    documentation = {
        name.removeprefix("docs/licenses/"): payload
        for name, payload in content.items()
        if name.startswith("docs/licenses/") and not name.endswith((".json", "/SOURCES.md"))
    }
    own_license, notice = legal["LICENSE"], legal["NOTICE"]
    license_payload = legal["licenses/dlpack-LICENSE.txt"]
    physx_license_payload = legal["licenses/physx-LICENSE.md"]
    physx_provenance = json.loads(legal["licenses/physx-PROVENANCE.json"])
    source_identities = [
        {
            "wheel_path": name,
            "sha256": _sha256(payload),
            "size": len(payload),
        }
        for name, payload in sorted(sources.items())
    ]
    manifest: dict[str, object] = {
        "schema_version": 1,
        "package": PACKAGE,
        "version": version,
        "target_platform": target_platform,
        "tag": tag,
        "source_revision": source_revision,
        "source_dirty": source_dirty,
        "source_files": source_identities,
        "native_artifacts": native_identities,
        "installed_contract": installed_contract,
        "capabilities": {
            "ovstage_companion_bundled": has_ovstage,
            "cuda_ptx_bundled": has_cuda,
        },
        "documentation_licenses": [
            {
                "bundled_file": f"{PACKAGE}/docs/licenses/{name}",
                "sha256": _sha256(payload),
                "size": len(payload),
            }
            for name, payload in sorted(documentation.items())
        ],
        "licenses": [
            {
                "expression": project["license"],
                "component": "ovspatialquery",
                "bundled_file": f"{PACKAGE}/_native/LICENSE",
                "notice_file": f"{PACKAGE}/_native/NOTICE",
                "sha256": _sha256(own_license),
                "size": len(own_license),
            },
            {
                "expression": "Apache-2.0",
                "bundled_file": (f"{PACKAGE}/_native/licenses/dlpack-LICENSE.txt"),
                "sha256": _sha256(license_payload),
                "size": len(license_payload),
            },
            {
                "expression": physx_provenance["license_expression"],
                "component": physx_provenance["component"],
                "version": physx_provenance["package"],
                "bundled_file": (f"{PACKAGE}/_native/licenses/physx-LICENSE.md"),
                "provenance_file": (f"{PACKAGE}/_native/licenses/physx-PROVENANCE.json"),
                "sha256": _sha256(physx_license_payload),
                "size": len(physx_license_payload),
            },
        ],
    }
    dist_info = DIST_INFO_TEMPLATE.format(version=version)
    payloads = {
        **sources,
        **{f"{PACKAGE}/{name}": payload for name, payload in content.items()},
        **native,
        **{f"{PACKAGE}/_native/{relative}": payload for relative, payload in legal.items()},
        f"{PACKAGE}/_native/provenance.json": (
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
        f"{dist_info}/METADATA": _metadata(
            version,
            project,
            # LF line endings, whatever the checkout used, so METADATA is the same everywhere.
            _read_required(source_root / project["readme"], "Python package README")
            .decode("utf-8")
            .replace("\r\n", "\n"),
            LICENSE_EXPRESSION,
            documentation,
        ),
        f"{dist_info}/WHEEL": _wheel_metadata(tag),
        f"{dist_info}/top_level.txt": f"{PACKAGE}\n".encode("utf-8"),
        f"{dist_info}/licenses/dlpack-LICENSE.txt": license_payload,
        f"{dist_info}/licenses/physx-LICENSE.md": physx_license_payload,
        f"{dist_info}/licenses/LICENSE": own_license,
        f"{dist_info}/licenses/NOTICE": notice,
        **{
            f"{dist_info}/licenses/documentation/{name}": payload
            for name, payload in documentation.items()
        },
    }
    record_name = f"{dist_info}/RECORD"
    record = _record(payloads, record_name)
    destination.mkdir(parents=True, exist_ok=True)
    filename = f"{PACKAGE}-{version}-{tag}.whl"
    output = destination / filename
    temporary = destination / f".{filename}.tmp"
    try:
        with zipfile.ZipFile(temporary, "w") as archive:
            for name in sorted(payloads):
                archive.writestr(_zip_info(name), payloads[name])
            archive.writestr(_zip_info(record_name), record)
        temporary.replace(output)
    except OSError as failure:
        raise WheelBuildError(f"Could not write platform wheel {output}: {failure}") from failure
    payload = output.read_bytes()
    return WheelArtifact(
        path=output,
        sha256=_sha256(payload),
        size=len(payload),
        tag=tag,
        manifest=manifest,
    )


__all__ = [
    "WheelArtifact",
    "WheelBuildError",
    "build",
]
