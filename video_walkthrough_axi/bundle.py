"""Build, verify, cache and publish versioned evidence bundles."""

import bisect
import fcntl
import hashlib
import json
import os
import tempfile
from importlib.metadata import version
from pathlib import Path

import av

from . import __version__
from .detect import SETTINGS, select_frames
from .evidence import backend_config, screen_text, transcribe
from .media import EvidenceError, device_signature, digest, pick_frame, probe, run, samples, verify_png

SCHEMA_VERSION = 1


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def local_artifact(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise EvidenceError("Bundle artifact path escapes the bundle.")
    return path


def read_manifest(root: Path) -> dict:
    manifest = read_json(root / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise EvidenceError(f"Expected evidence manifest schema version {SCHEMA_VERSION}.")
    return manifest


def ensure_unchanged(path: Path, signature: tuple) -> None:
    if device_signature(path) != signature:
        raise EvidenceError("The source changed during processing. Retry with a stable file.")


def activate(root: Path, manifest: dict) -> None:
    with tempfile.TemporaryDirectory(prefix=".publish-", dir=root) as tmp:
        staged = Path(tmp) / "manifest.json"
        write_json(staged, manifest)
        staged.replace(root / "manifest.json")


def evidence_files(manifest: dict):
    yield from manifest["artifacts"].values()
    for frame in manifest["frames"]:
        yield {"path": frame["path"], "sha256": frame["sha256"]}


def cached_manifest(root: Path, evidence: Path, cache_key: str) -> dict | None:
    for folder in sorted(evidence.glob(f"{cache_key}-*")):
        try:
            if folder.is_symlink():
                continue
            manifest = read_manifest(folder)
            if manifest["cache_key"] != cache_key:
                continue
            if all(
                digest(local_artifact(root, item["path"])) == item["sha256"]
                for item in evidence_files(manifest)
            ):
                return manifest
        except (OSError, ValueError, KeyError, TypeError, EvidenceError):
            continue
    return None


def event_time(event: dict) -> float:
    return event["change_start"] if event["kind"] == "frame" else event["time"]


def merge(frames: list[dict], transcript: dict) -> list[dict]:
    times = [f["change_start"] for f in frames]
    events = [{"kind": "frame", **frame} for frame in frames]
    for segment in transcript["segments"]:
        index = bisect.bisect_right(times, segment["start"]) - 1
        events.append(
            {
                "kind": "speech",
                "time": segment["start"],
                "speech": segment,
                "preceding_frame": frames[index]["path"] if index >= 0 else None,
            }
        )
    return sorted(events, key=event_time)


def build(staging: Path, name: str, source: Path, manifest: dict, model: Path | None) -> None:
    prefix = f"evidence/{name}"
    frames_dir = staging / "frames"
    frames_dir.mkdir()
    media = manifest["media"]
    settings = manifest["settings"]
    frames = []
    paths = []
    for index, selection in enumerate(
        select_frames(samples(source, media)),
        start=1,
    ):
        label = f"{index:06}.png"
        record = selection.record()
        record["path"] = f"{prefix}/frames/{label}"
        path = frames_dir / label
        selection.sample.save(path)
        verify_png(path, (record["width"], record["height"]))
        record["sha256"] = digest(path)
        frames.append(record)
        paths.append(path)
    if not frames:
        raise EvidenceError("The source contains no decodable video frames.")

    transcript = transcribe(source, media, model, staging / "audio.tmp.wav")
    ocr = screen_text(paths, enabled=settings["ocr"] is not None)
    timeline = merge(
        [dict(frame, ocr=text) for frame, text in zip(frames, ocr, strict=True)], transcript
    )
    manifest["flags"] = list(transcript["flags"])
    if settings["ocr"] is None:
        manifest["flags"].append(
            "ocr-disabled" if settings["ocr_request"] == "off" else "ocr-unavailable"
        )
    manifest["frames"] = frames
    manifest["counts"] = {
        "frames": len(frames),
        "speech_segments": len(transcript["segments"]),
        "events": len(timeline),
        "ocr_frames": sum(item is not None for item in ocr),
    }
    manifest["artifacts"] = {}
    for key, value in (("transcript", transcript), ("timeline", timeline)):
        path = staging / f"{key}.json"
        write_json(path, value)
        # Reopen generated files before publishing a manifest that points to them.
        if read_json(path) != value:
            raise EvidenceError(f"{key.capitalize()} validation failed.")
        manifest["artifacts"][key] = {"path": f"{prefix}/{path.name}", "sha256": digest(path)}


def prepare(source: Path, root: Path, model: Path | None, ocr: str) -> dict:
    if not source.is_file():
        raise EvidenceError("Provide a regular, local recording file.")
    signature = device_signature(source)
    sha = digest(source)
    media = probe(source)
    settings, evidence_backends, signatures = backend_config(media, model, ocr)
    config = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": __version__,
        "settings": {**settings, "detector": SETTINGS},
        "backends": {
            **evidence_backends,
            "ffmpeg": run(["ffmpeg", "-version"]).splitlines()[0],
            "ffprobe": run(["ffprobe", "-version"]).splitlines()[0],
            "pyav": av.__version__,
            "libav": {k: ".".join(map(str, v)) for k, v in av.library_versions.items()},
            "pillow": version("Pillow"),
            "numpy": version("numpy"),
        },
    }
    cache_key = hashlib.sha256(json.dumps([sha, config], sort_keys=True).encode()).hexdigest()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    evidence = local_artifact(root, "evidence")
    evidence.mkdir(mode=0o700, exist_ok=True)
    with local_artifact(root, ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / "manifest.json").exists():
            read_manifest(root)
        manifest = cached_manifest(root, evidence, cache_key)
        cache_hit = manifest is not None
        source_info = {"path": str(source), "sha256": sha, "bytes": signature[2]}
        if cache_hit:
            manifest["source"] = source_info
        else:
            manifest = {
                **config,
                "cache_key": cache_key,
                "source": source_info,
                "media": media,
            }
            with tempfile.TemporaryDirectory(prefix=".stage-", dir=evidence) as tmp:
                staging = Path(tmp)
                name = f"{cache_key}-{staging.name.removeprefix('.stage-')}"
                build(staging, name, source, manifest, model)
                if backend_config(media, model, ocr) != (settings, evidence_backends, signatures):
                    raise EvidenceError("Speech or OCR backend inputs changed during processing.")
                ensure_unchanged(source, signature)
                write_json(staging / "manifest.json", manifest)
                staging.rename(evidence / name)
        ensure_unchanged(source, signature)
        activate(root, manifest)
    return {
        "bundle": str(root),
        "manifest": "manifest.json",
        "cache_hit": cache_hit,
        **manifest["counts"],
        "flags": manifest["flags"],
    }


def extract_frame(root: Path, at: float, crop: tuple | None) -> dict:
    manifest = read_manifest(root)
    source = Path(manifest["source"]["path"])
    if not source.is_absolute() or not source.is_file():
        raise EvidenceError("Source is unavailable. Rerun prepare with its current path.")
    signature = device_signature(source)
    if digest(source) != manifest["source"]["sha256"]:
        raise EvidenceError("Source SHA-256 differs from the recording used to build this bundle.")
    sample = pick_frame(source, manifest["media"], at)
    ensure_unchanged(source, signature)
    output = local_artifact(root, "extracts")
    output.mkdir(mode=0o700, exist_ok=True)
    fd, filename = tempfile.mkstemp(dir=output, prefix="frame-", suffix=".png")
    os.close(fd)
    path = Path(filename)
    size = crop[2:] if crop is not None else (sample.frame.width, sample.frame.height)
    try:
        sample.save(path, crop)
        verify_png(path, size)
    except BaseException:
        path.unlink()
        raise
    return {
        "path": str(path),
        "requested_time": at,
        "time": sample.time,
        "pts": sample.frame.pts,
        "time_base": str(sample.frame.time_base),
        "selection": "exact" if abs(sample.time - at) <= 1e-9 else "nearest-frame",
        "width": size[0],
        "height": size[1],
        "crop": list(crop) if crop else None,
    }


def load_timeline(root: Path, manifest: dict) -> list[dict]:
    artifact = manifest["artifacts"]["timeline"]
    data = local_artifact(root, artifact["path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != artifact["sha256"]:
        raise EvidenceError(f"Bundle timeline SHA-256 differs: {artifact['path']}")
    return json.loads(data)
