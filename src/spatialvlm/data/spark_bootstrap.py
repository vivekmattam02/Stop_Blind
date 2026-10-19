"""Authorized, source-locked R2R-CE and Matterport3D preparation for DGX Spark."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile, is_zipfile


class BootstrapError(RuntimeError):
    """Raised when data preparation cannot continue without weakening a gate."""


@dataclass(frozen=True)
class ArchiveSpec:
    """Immutable metadata for an authorized source archive."""

    name: str
    url: str
    expected_bytes: int


MP3D_HABITAT_ARCHIVE = ArchiveSpec(
    name="mp3d_habitat.zip",
    url="https://kaldir.vc.cit.tum.de/matterport/v1/tasks/mp3d_habitat.zip",
    expected_bytes=16_085_306_031,
)
R2R_CE_V1_3_ARCHIVE = ArchiveSpec(
    name="R2R_VLNCE_v1-3_preprocessed.zip",
    url=(
        "https://drive.usercontent.google.com/download"
        "?id=1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr&export=download&confirm=t"
    ),
    expected_bytes=261_610_900,
)

MP3D_SCAN_IDS = frozenset(
    """
17DRP5sb8fy 1LXtFkjw3qL 1pXnuDYAj8r 29hnd4uzFmX 2azQ1b91cZZ 2n8kARJN3HM
2t7WUuJeko7 5LpN3gDmAk7 5q7pvUzZiYa 5ZKStnWn8Zo 759xd9YjKW5 7y3sRwLe3Va
8194nk5LbLH 82sE5b5pLXE 8WUmhLawc2A aayBHfsNo7d ac26ZMwG7aT ARNzJeq3xxb
B6ByNegPMKs b8cTxDM8gDG cV4RVeZvu5T D7G3Y4RVNrH D7N2EKCX4Sj dhjEzFoUFzH
E9uDoFAP3SH e9zR4mvMWw7 EDJbREhghzL EU6Fwq7SyZv fzynW3qQPVF GdvgFV5R1Z5
gTV8FGcVJC9 gxdoqLR6rwA gYvKGZ5eRqb gZ6f7yhEvPG HxpKQynjfin i5noydFURQK
JeFG25nYj2p JF19kD82Mey jh4fc5c5qoQ JmbYfDe2QKZ jtcxE69GiFV kEZ7cmS4wCh
mJXqzFtmKg4 oLBMNvg9in8 p5wJjkQkbXX pa4otMbVnkk pLe4wQe7qrG Pm6F8kyY3z2
pRbA3pwrgk9 PuKPg4mmafe PX4nDJXEHrG q9vSo1VnCiC qoiz87JEwZ2 QUCTc6BB5sX
r1Q1Z4BcV1o r47D5H71a5s rPc6DW4iMge RPmz2sHmrrY rqfALeAoiTq s8pcmisQ38h
S9hNv5qa7GM sKLMLpTHeUy SN83YJsR3w2 sT4fr6TAbpF TbHJrupSAjP ULsKaCPVFJR
uNb9QFRL6hY ur6pFq6Qu1A UwV83HsGsw3 Uxmj2M2itWa V2XKFyX4ASd VFuaQ6m2Qom
VLzqgDo317F Vt2qJdWjCF2 VVfe2KiqLaN Vvot9Ly1tCj vyrNrziPKCB VzqfbhrpDEA
wc2JMjhGNzB WYY7iVyf5p8 X7HyMhZNoso x8F5xyUWy9e XcA2TqTSSAj YFuZgdQ5vWj
YmJkqBEsHnH yqstnuAEVhm YVUC4YcDtcY Z6MFQCViBuw ZMojNkEp431 zsNo4HB9uLZ
""".split()
)
R2R_CE_VALIDATION_SPLITS = ("train", "val_seen", "val_unseen")


def require_mp3d_terms_acceptance(accepted: bool) -> None:
    """Fail before side effects unless the authorized user explicitly accepts the terms."""
    if not accepted:
        raise BootstrapError(
            "Refusing to download Matterport3D without explicit --accept-mp3d-terms."
        )


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_free_space(path: Path, *, minimum_bytes: int = 80 * 1024**3) -> int:
    """Require enough room for both source archives, staging, and extracted scenes."""
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free_bytes = shutil.disk_usage(probe).free
    if free_bytes < minimum_bytes:
        raise BootstrapError(
            f"Only {free_bytes / 1024**3:.1f} GiB free at {probe}; "
            f"at least {minimum_bytes / 1024**3:.1f} GiB is required."
        )
    return free_bytes


def download_archive(spec: ArchiveSpec, destination: Path) -> Path:
    """Download exactly one locked archive with retries and HTTP-range resumption."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        observed = destination.stat().st_size
        if observed != spec.expected_bytes:
            raise BootstrapError(
                f"Existing {destination} has {observed} bytes; expected {spec.expected_bytes}. "
                "It was left untouched."
            )
        print(f"[download] verified existing {destination}", flush=True)
        return destination

    partial = destination.with_name(destination.name + ".part")
    if partial.exists() and partial.stat().st_size > spec.expected_bytes:
        raise BootstrapError(
            f"Partial file {partial} is larger than the locked source size; it was left untouched."
        )
    if partial.exists() and partial.stat().st_size == spec.expected_bytes:
        os.replace(partial, destination)
        return destination

    print(f"[download] {spec.name} ({spec.expected_bytes / 1024**3:.2f} GiB)", flush=True)
    last_error: Exception | None = None
    for attempt in range(1, 9):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "GridNav-data-bootstrap/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = Request(spec.url, headers=headers)
        try:
            with urlopen(request, timeout=60) as response:  # noqa: S310 - locked HTTPS URLs
                status = getattr(response, "status", response.getcode())
                if offset and status != 206:
                    raise BootstrapError(
                        f"Server did not honor byte-range resume for {spec.name}; "
                        "the partial file was left untouched."
                    )
                mode = "ab" if offset else "wb"
                with partial.open(mode) as target:
                    while chunk := response.read(8 * 1024 * 1024):
                        target.write(chunk)
            observed = partial.stat().st_size
            if observed == spec.expected_bytes:
                break
            if observed > spec.expected_bytes:
                raise BootstrapError(
                    f"Downloaded {spec.name} exceeded its locked size; "
                    "the partial file was left untouched."
                )
            last_error = BootstrapError(
                f"Connection ended at {observed} of {spec.expected_bytes} bytes."
            )
        except BootstrapError:
            raise
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
        if attempt < 8:
            print(f"[download] retry {attempt}/8 after: {last_error}", flush=True)
            time.sleep(10)
    else:
        raise BootstrapError(f"Download failed after 8 attempts: {last_error}") from last_error

    observed = partial.stat().st_size
    if observed != spec.expected_bytes:
        raise BootstrapError(
            f"Downloaded {spec.name} has {observed} bytes; expected {spec.expected_bytes}. "
            "The .part file was preserved for inspection or resumption."
        )
    os.replace(partial, destination)
    return destination


def _validated_member_destination(root: Path, member: str) -> Path:
    member_path = PurePosixPath(member)
    if member_path.is_absolute() or ".." in member_path.parts:
        raise BootstrapError(f"Unsafe ZIP member path: {member!r}")
    destination = root.joinpath(*member_path.parts)
    resolved_root = root.resolve()
    resolved_destination = destination.resolve()
    if resolved_destination != resolved_root and resolved_root not in resolved_destination.parents:
        raise BootstrapError(f"ZIP member escapes extraction root: {member!r}")
    return destination


def extract_zip_resumable(archive: Path, staging_root: Path) -> None:
    """Extract through per-file temporary paths so interruption is safely resumable."""
    if not is_zipfile(archive):
        raise BootstrapError(f"Not a readable ZIP archive: {archive}")
    staging_root.mkdir(parents=True, exist_ok=True)
    try:
        with ZipFile(archive) as handle:
            for member in handle.infolist():
                destination = _validated_member_destination(staging_root, member.filename)
                unix_mode = member.external_attr >> 16
                if stat.S_ISLNK(unix_mode):
                    raise BootstrapError(f"Refusing symbolic link in ZIP: {member.filename!r}")
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    if destination.stat().st_size != member.file_size:
                        raise BootstrapError(
                            f"Existing staged file has the wrong size: {destination}. "
                            "It was left untouched."
                        )
                    continue
                partial = destination.with_name(destination.name + ".part")
                with handle.open(member) as source, partial.open("wb") as target:
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                if partial.stat().st_size != member.file_size:
                    raise BootstrapError(f"Extracted size mismatch for {member.filename!r}")
                os.replace(partial, destination)
    except BadZipFile as exc:
        raise BootstrapError(f"Corrupt ZIP archive {archive}: {exc}") from exc


def validate_mp3d_root(root: Path) -> dict[str, Any]:
    """Require all 90 canonical MP3D Habitat GLBs at `<id>/<id>.glb`."""
    present = {
        scan_id
        for scan_id in MP3D_SCAN_IDS
        if (root / scan_id / f"{scan_id}.glb").is_file()
        and (root / scan_id / f"{scan_id}.glb").stat().st_size > 0
    }
    missing = sorted(MP3D_SCAN_IDS - present)
    if missing:
        raise BootstrapError(
            f"MP3D root {root} is missing {len(missing)} canonical GLBs: {missing[:8]}"
        )
    return {"scene_count": len(present), "scene_ids": sorted(present)}


def locate_mp3d_root(extraction_root: Path) -> Path:
    """Find the unique extracted directory containing the canonical scene layout."""
    candidates = [extraction_root, *extraction_root.rglob("mp3d")]
    valid: list[Path] = []
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        try:
            validate_mp3d_root(candidate)
        except BootstrapError:
            continue
        valid.append(candidate)
    unique = list(dict.fromkeys(path.resolve() for path in valid))
    if len(unique) != 1:
        raise BootstrapError(
            f"Expected one canonical MP3D root under {extraction_root}, found {len(unique)}."
        )
    return unique[0]


def validate_r2r_ce_root(root: Path) -> dict[str, Any]:
    """Require episode and ground-truth archives for the evaluable v1-3 splits."""
    required: list[Path] = []
    for split in R2R_CE_VALIDATION_SPLITS:
        required.extend((root / split / f"{split}.json.gz", root / split / f"{split}_gt.json.gz"))
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise BootstrapError(f"R2R-CE root {root} is missing required files: {missing}")
    return {"validated_splits": list(R2R_CE_VALIDATION_SPLITS)}


def locate_r2r_ce_root(extraction_root: Path) -> Path:
    """Find the unique extracted directory containing official split subdirectories."""
    candidates = {path.parent.parent for path in extraction_root.rglob("train/train.json.gz")}
    valid: list[Path] = []
    for candidate in candidates:
        try:
            validate_r2r_ce_root(candidate)
        except BootstrapError:
            continue
        valid.append(candidate.resolve())
    if len(valid) != 1:
        raise BootstrapError(
            f"Expected one R2R-CE v1-3 root under {extraction_root}, found {len(valid)}."
        )
    return valid[0]


def _promote_directory(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise BootstrapError(f"Refusing to overwrite existing directory: {destination}")
    source.rename(destination)


def run_release_checks(
    *,
    repo_root: Path,
    r2r_root: Path,
    artifact_root: Path,
    image: str,
    inside_container: bool = False,
) -> list[Path]:
    """Run the repository's strict source-lock checker inside the tested Spark image."""
    artifact_root.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for split in R2R_CE_VALIDATION_SPLITS:
        output_name = f"{split}_source_lock.json"
        if inside_container:
            command = [
                sys.executable,
                str(repo_root / "scripts" / "check_r2r_ce_release.py"),
                "--episodes",
                str(r2r_root / split / f"{split}.json.gz"),
                "--ground-truth",
                str(r2r_root / split / f"{split}_gt.json.gz"),
                "--split",
                split,
                "--output",
                str(artifact_root / output_name),
            ]
        else:
            command = [
                "docker",
                "run",
                "--rm",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--mount",
                f"type=bind,src={repo_root},dst=/workspace/GridNav,readonly",
                "--mount",
                f"type=bind,src={r2r_root},dst=/datasets/r2r,readonly",
                "--mount",
                f"type=bind,src={artifact_root},dst=/artifacts",
                "--env",
                "PYTHONDONTWRITEBYTECODE=1",
                "--workdir",
                "/workspace/GridNav",
                image,
                "python",
                "scripts/check_r2r_ce_release.py",
                "--episodes",
                f"/datasets/r2r/{split}/{split}.json.gz",
                "--ground-truth",
                f"/datasets/r2r/{split}/{split}_gt.json.gz",
                "--split",
                split,
                "--output",
                f"/artifacts/{output_name}",
            ]
        print(f"[validate] R2R-CE {split}", flush=True)
        subprocess.run(command, check=True)
        outputs.append(artifact_root / output_name)
    return outputs


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON artifact without exposing a partially written final path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def prepare_spark_data(
    *,
    repo_root: Path,
    data_root: Path,
    artifact_root: Path,
    image: str,
    inside_container: bool = False,
) -> Path:
    """Download, extract, validate, and source-lock both required datasets."""
    free_before = ensure_free_space(data_root)
    downloads = data_root / "downloads"
    staging = data_root / ".staging"
    mp3d_root = data_root / "scene_datasets" / "mp3d"
    r2r_root = data_root / "episodes" / "R2R_VLNCE_v1-3_preprocessed"

    mp3d_archive = download_archive(MP3D_HABITAT_ARCHIVE, downloads / MP3D_HABITAT_ARCHIVE.name)
    r2r_archive = download_archive(R2R_CE_V1_3_ARCHIVE, downloads / R2R_CE_V1_3_ARCHIVE.name)
    archive_hashes = {
        MP3D_HABITAT_ARCHIVE.name: sha256_file(mp3d_archive),
        R2R_CE_V1_3_ARCHIVE.name: sha256_file(r2r_archive),
    }

    if mp3d_root.exists():
        mp3d_summary = validate_mp3d_root(mp3d_root)
    else:
        mp3d_stage = staging / f"mp3d-{archive_hashes[MP3D_HABITAT_ARCHIVE.name][:16]}"
        print(f"[extract] {mp3d_archive} -> {mp3d_stage}", flush=True)
        extract_zip_resumable(mp3d_archive, mp3d_stage)
        extracted_mp3d = locate_mp3d_root(mp3d_stage)
        _promote_directory(extracted_mp3d, mp3d_root)
        mp3d_summary = validate_mp3d_root(mp3d_root)

    if r2r_root.exists():
        r2r_summary = validate_r2r_ce_root(r2r_root)
    else:
        r2r_stage = staging / f"r2r-{archive_hashes[R2R_CE_V1_3_ARCHIVE.name][:16]}"
        print(f"[extract] {r2r_archive} -> {r2r_stage}", flush=True)
        extract_zip_resumable(r2r_archive, r2r_stage)
        extracted_r2r = locate_r2r_ce_root(r2r_stage)
        _promote_directory(extracted_r2r, r2r_root)
        r2r_summary = validate_r2r_ce_root(r2r_root)

    reports = run_release_checks(
        repo_root=repo_root,
        r2r_root=r2r_root,
        artifact_root=artifact_root,
        image=image,
        inside_container=inside_container,
    )
    report_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in reports]
    if any(report.get("status") != "passed" for report in report_payloads):
        raise BootstrapError("At least one R2R-CE source-lock report did not pass.")

    manifest_path = artifact_root / "data_bootstrap_manifest.json"
    write_json_atomic(
        manifest_path,
        {
            "schema_version": 1,
            "status": "passed",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "terms": {
                "matterport3d_acceptance_required": True,
                "accepted_by_invocation_flag": True,
            },
            "archives": {
                MP3D_HABITAT_ARCHIVE.name: {
                    "url": MP3D_HABITAT_ARCHIVE.url,
                    "bytes": MP3D_HABITAT_ARCHIVE.expected_bytes,
                    "sha256": archive_hashes[MP3D_HABITAT_ARCHIVE.name],
                    "path": str(mp3d_archive.resolve()),
                },
                R2R_CE_V1_3_ARCHIVE.name: {
                    "url": R2R_CE_V1_3_ARCHIVE.url,
                    "bytes": R2R_CE_V1_3_ARCHIVE.expected_bytes,
                    "sha256": archive_hashes[R2R_CE_V1_3_ARCHIVE.name],
                    "path": str(r2r_archive.resolve()),
                },
            },
            "datasets": {
                "mp3d": {"path": str(mp3d_root.resolve()), **mp3d_summary},
                "r2r_ce_v1_3": {"path": str(r2r_root.resolve()), **r2r_summary},
            },
            "source_lock_reports": [str(path.resolve()) for path in reports],
            "docker_image": image,
            "free_bytes_before": free_before,
            "free_bytes_after": shutil.disk_usage(data_root).free,
        },
    )
    return manifest_path
