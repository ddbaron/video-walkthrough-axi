import json
import os
import re
import shutil
import subprocess
import sys

import pytest
import toon_format
from demo_recording import AMOUNT_CROP, FPS, SIZE, STATES, make_recording
from PIL import Image

from video_walkthrough_axi.media import digest


def cli(*args, code=0):
    result = subprocess.run(
        [sys.executable, "-m", "video_walkthrough_axi.cli", *map(str, args)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == code, result.stdout + result.stderr
    return result.stdout


def toon(*args, code=0):
    return toon_format.decode(cli(*args, code=code))


def manifest_of(bundle):
    return json.loads((bundle / "manifest.json").read_text())


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    work = tmp_path_factory.mktemp("demo")
    source = work / "synthetic demo.mp4"
    bundle = work / "bundle"
    make_recording(source)
    summary = toon("prepare", source, "--out", bundle, "--ocr", "off")
    assert summary["cache_hit"] is False
    return source, bundle


def test_help_and_usage():
    assert "walkthrough" in cli("--help")
    for command in ("prepare", "inspect", "frame", "walkthrough"):
        assert "usage:" in cli(command, "--help")
    assert "0.1.0" in cli("--version")
    assert "help" in toon("inspect", "bundle", "--from", "-1", "--to", "4", code=2)
    assert "help" in toon("inspect", "bundle", "--from", "4", "--to", "3", code=1)
    assert "help" in toon("frame", "bundle", "--at", "0", "--crop", "0,0,1,0", code=2)


def test_native_coverage_and_manifest(demo):
    source, bundle = demo
    manifest = manifest_of(bundle)
    frames = manifest["frames"]
    assert manifest["schema_version"] == 1
    assert manifest["source"]["sha256"] == digest(source)
    assert manifest["media"]["streams"][0]["offset"] == 0
    assert "no-audio-stream" in manifest["flags"]
    assert len(frames) < 15
    for name, (start, end) in STATES.items():
        matches = [f for f in frames if start / FPS - 1e-8 <= f["time"] < end / FPS - 1e-8]
        assert matches, name
        if name in {"dropdown", "confirmation", "single-frame"}:
            assert any(f["reason"] == "short-return" for f in matches), name
    assert [f["time"] for f in frames] == sorted({f["time"] for f in frames})
    for frame in frames:
        path = bundle / frame["path"]
        assert digest(path) == frame["sha256"]
        with Image.open(path) as png:
            assert png.size == SIZE
            assert png.format == "PNG"


def test_inspect_cursor_and_markdown(demo):
    _, bundle = demo
    page = toon("inspect", bundle, "--from", 1, "--to", "00:10", "--limit", 2)
    assert page["preceding_screen"]["time"] == 0
    assert page["returned"] == 2
    events = page["events"]
    invalid = toon(
        "inspect", bundle, "--from", 0, "--to", 10, "--cursor", page["next_cursor"], code=1
    )
    assert "Invalid cursor" in invalid["error"]
    while page["next_cursor"]:
        page = toon(
            "inspect",
            bundle,
            "--from",
            1,
            "--to",
            10,
            "--limit",
            2,
            "--cursor",
            page["next_cursor"],
        )
        events += page["events"]
    assert [e["time"] for e in events] == sorted({e["time"] for e in events})
    assert all(e["speech"] is None and e["ocr"] is None for e in events)
    md = cli("walkthrough", bundle)
    links = re.findall(r"\[Screenshot at .*?\]\((.*?)\)", md)
    assert links == [f["path"] for f in manifest_of(bundle)["frames"]]
    assert all((bundle / path).is_file() for path in links)
    assert "**Speech**" in md and "**On-screen text**" in md


def test_cache_checks_artifacts_and_source_identity(demo, tmp_path):
    original, _ = demo
    source = tmp_path / "source.mp4"
    shutil.copyfile(original, source)
    bundle = tmp_path / "cache"
    assert not toon("prepare", source, "--out", bundle, "--ocr", "off")["cache_hit"]
    first = manifest_of(bundle)
    assert toon("prepare", source, "--out", bundle, "--ocr", "off")["cache_hit"]
    (bundle / first["frames"][0]["path"]).write_bytes(b"invalid")
    assert not toon("prepare", source, "--out", bundle, "--ocr", "off")["cache_hit"]
    assert manifest_of(bundle)["frames"][0]["path"] != first["frames"][0]["path"]
    with source.open("ab") as file:
        file.write(b"changed")
    assert "SHA-256 differs" in toon("frame", bundle, "--at", 0, code=1)["error"]


def test_frame_crops_and_nearest_timestamp(demo):
    _, bundle = demo
    exact = toon("frame", bundle, "--at", 8)
    assert exact["selection"] == "exact"
    assert Image.open(exact["path"]).size == SIZE
    earlier = toon("frame", bundle, "--at", 8.01, "--crop", ",".join(map(str, AMOUNT_CROP)))
    assert earlier["selection"] == "nearest-frame"
    assert earlier["time"] == 8
    with Image.open(earlier["path"]) as cropped, Image.open(exact["path"]) as full:
        x, y, w, h = AMOUNT_CROP
        assert cropped.tobytes() == full.crop((x, y, x + w, y + h)).tobytes()
    later = toon("frame", bundle, "--at", 8.03)
    assert later["time"] == pytest.approx(8 + 1 / FPS)
    after = toon("frame", bundle, "--at", 120)
    assert after["time"] == pytest.approx(12 - 1 / FPS)
    assert "exceeds" in toon("frame", bundle, "--at", 0, "--crop", "639,0,2,5", code=1)["error"]


@pytest.mark.skipif(not shutil.which("tesseract"), reason="Tesseract is optional")
def test_ocr_amount_and_warning(demo, tmp_path):
    source, _ = demo
    bundle = tmp_path / "ocr"
    assert not toon("prepare", source, "--out", bundle, "--ocr", "off")["cache_hit"]
    without_ocr = manifest_of(bundle)
    summary = toon("prepare", source, "--out", bundle, "--ocr", "tesseract")
    assert not summary["cache_hit"]
    assert summary["ocr_frames"] == summary["frames"]
    page = toon("inspect", bundle, "--from", 0, "--to", 12, "--limit", 100)
    text = " ".join(event["ocr"] for event in page["events"])
    assert all(s in text for s in ("$24.50", "$18.50", "Selection saved", "approval required"))
    assert all(e["speech"] is None for e in page["events"])
    assert toon("prepare", source, "--out", bundle, "--ocr", "off")["cache_hit"]
    assert manifest_of(bundle)["frames"] == without_ocr["frames"]


@pytest.mark.skipif(
    not os.environ.get("VIDEO_WALKTHROUGH_TEST_MODEL") or not shutil.which("say"),
    reason="requires a local MLX model and macOS say",
)
def test_real_mlx_narration(tmp_path):
    video = tmp_path / "narrated.mp4"
    bundle = tmp_path / "narrated"
    make_recording(video, narration=True)
    summary = toon(
        "prepare",
        video,
        "--out",
        bundle,
        "--model",
        os.environ["VIDEO_WALKTHROUGH_TEST_MODEL"],
        "--language",
        "en",
        "--ocr",
        "off",
    )
    assert summary["speech_segments"] > 0
    manifest = manifest_of(bundle)
    path = manifest["artifacts"]["transcript"]["path"]
    segments = json.loads((bundle / path).read_text())["segments"]
    assert any(segment["words"] for segment in segments)
    assert any("approval" in segment["text"].lower() for segment in segments)
    assert any("quiet-audio" in segment["flags"] for segment in segments)
    assert all(
        segment["start"] <= w["start"] <= w["end"] <= segment["end"]
        for segment in segments
        for w in segment["words"]
    )
    page = toon("inspect", bundle, "--from", 0, "--to", 12, "--limit", 100)
    assert any(e["speech"] and e["ocr"] is None for e in page["events"])
    assert "approval" in cli("walkthrough", bundle).lower()


def test_audio_stream_offset(demo, tmp_path):
    import numpy as np

    from video_walkthrough_axi.evidence import audio_pcm
    from video_walkthrough_axi.media import probe

    video, _ = demo
    offset = tmp_path / "offset.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video),
            "-itsoffset",
            "0.4",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=500:duration=0.5",
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "copy",
            "-c:a",
            "pcm_s16le",
            str(offset),
        ],
        check=True,
    )
    media = probe(offset)
    assert media["streams"][1]["offset"] == pytest.approx(0.4)
    pcm = audio_pcm(offset, media, tmp_path / "pcm.wav")
    assert np.flatnonzero(np.abs(pcm) > 100)[0] / 16000 == pytest.approx(0.4, abs=0.002)
