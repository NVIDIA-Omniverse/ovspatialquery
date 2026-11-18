# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage the pinned Ovstage SDK the companion is built against (ADR-0069).

deps/ovstage-sdk-lock.json names one Ovstage wheel per platform, by URL and SHA-256.
This downloads it into _build/ovstage-sdk, verifies the digest and extracts its
`ovstage/` package, the CMake package root the companion and its native tests use.
The application venv keeps the Ovstage of the application lock.
"""

from __future__ import annotations

import hashlib
import http.client
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

import contract

# Artifactory otherwise redirects file downloads to a CloudFront CDN whose WAF
# rejects hosts outside its allowlist with 403; this asks it to serve them itself.
DIRECT_DOWNLOAD = {"X-JFrog-Download-Redirect-To": "None"}


def _download(url: str, destination: Path, attempts: int = 4) -> None:
    request = urllib.request.Request(url, headers=DIRECT_DOWNLOAD)
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                with destination.open("wb") as stream:
                    shutil.copyfileobj(response, stream, 1 << 20)
            return
        # A connection dropped mid-body raises IncompleteRead, which is not an OSError.
        except (OSError, http.client.IncompleteRead) as failure:
            if attempt == attempts:
                raise contract.BootstrapError(
                    f"Cannot download the pinned Ovstage SDK {url}: {failure}",
                    code="ovstage_sdk_unavailable",
                    action=(
                        "Check that this host reaches the URL in the lock; otherwise copy the wheel "
                        f"named in {contract.OVSTAGE_SDK_LOCK.name} into {destination.parent} "
                        "from a host that can, and rerun."
                    ),
                ) from failure
            time.sleep(10 * attempt)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stage(target_platform: str) -> Path:
    """Return the extracted pinned SDK, downloading and extracting it when missing."""
    entry = contract.ovstage_sdk_entry(target_platform)
    package = contract.pinned_ovstage_sdk_dir(target_platform)
    marker = package.parent / ".sha256"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == entry["sha256"]:
        return package
    wheel = contract.OVSTAGE_SDK_ROOT / entry["filename"]
    wheel.parent.mkdir(parents=True, exist_ok=True)
    if not wheel.is_file() or _sha256(wheel) != entry["sha256"]:
        partial = wheel.with_name(wheel.name + ".partial")
        _download(entry["url"], partial)
        actual = _sha256(partial)
        if actual != entry["sha256"]:
            partial.unlink()
            raise contract.BootstrapError(
                f"{entry['url']}: sha256 {actual}, expected {entry['sha256']}",
                code="ovstage_sdk_unavailable",
                action="Retry; if it persists, the mirror changed the wheel after it was pinned.",
            )
        partial.replace(wheel)
    shutil.rmtree(package.parent, ignore_errors=True)
    package.parent.mkdir(parents=True)
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            if not member.filename.startswith("ovstage/"):
                continue
            extracted = Path(archive.extract(member, package.parent))
            mode = member.external_attr >> 16
            if mode and not member.is_dir():
                extracted.chmod(mode & 0o777)
    marker.write_text(entry["sha256"] + "\n", encoding="utf-8")
    print(f"[bootstrap] ovstage SDK {entry['version']} staged at {package}", flush=True)
    return package
