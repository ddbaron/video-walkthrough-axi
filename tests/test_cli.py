import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from fractions import Fraction

import av
import pytest
import toon_format
from demo_recording import (
    AMOUNT_CROP,
    FPS,
    QUIET_APPROVAL_END,
    QUIET_APPROVAL_START,
    SIZE,
    STATES,
    make_recording,
    screen,
)
from PIL import Image, ImageChops, ImageStat

from video_walkthrough_axi import bundle as bundles
from video_walkthrough_axi import evidence
from video_walkthrough_axi.detect import select_frames
from video_walkthrough_axi.media import EvidenceError, Sample


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


def amount_box():
    x, y, width, height = AMOUNT_CROP
    return x, y, x + width, y + height


def assert_price_region(image, state):
    other = "baseline" if state == "result" else "result"
    actual = image.convert("RGB")
    expected = screen(state).crop(amount_box())
    alternate = screen(other).crop(amount_box())
    error = sum(ImageStat.Stat(ImageChops.difference(actual, expected)).mean)
    alternate_error = sum(ImageStat.Stat(ImageChops.difference(actual, alternate)).mean)
    assert error < alternate_error


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
    assert manifest["source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest["media"]["streams"][0]["offset"] == 0
    assert "no-audio-stream" in manifest["flags"]
    assert len(frames) < 15
    for name, (start, end) in STATES.items():
        matches = [f for f in frames if start / FPS - 1e-8 <= f["time"] < end / FPS - 1e-8]
        assert matches, name
        if name in {"dropdown", "confirmation", "single-frame"}:
            assert any(f["reason"] == "short-return" for f in matches), name
        if name == "approval-indicator":
            assert matches[0]["reason"] == "short-transition"
    assert [f["time"] for f in frames] == sorted({f["time"] for f in frames})
    for frame in frames:
        path = bundle / frame["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == frame["sha256"]
        with Image.open(path) as png:
            assert png.size == SIZE
            assert png.format == "PNG"
            if frame["reason"] == "opening":
                assert_price_region(png.crop(amount_box()), "baseline")
            elif frame["time"] >= STATES["result"][0] / FPS:
                assert_price_region(png.crop(amount_box()), "result")


@pytest.mark.parametrize(
    ("ending", "expected"),
    [
        ([], [(0.1, "end-of-video")]),
        ([(0.2, "approval-indicator")], [(0.1, "short-at-end")]),
        (
            [(0.2, "approval-indicator"), (0.3, "smaller")],
            [(0.1, "short-transition"), (0.3, "resolution-change")],
        ),
    ],
)
def test_localized_screen_at_end_or_resolution_change(ending, expected):
    samples = []
    for time, state in [(0.0, "baseline"), (0.1, "approval-indicator"), *ending]:
        image = screen(state)
        if state == "smaller":
            image = image.resize((320, 180))
        samples.append(Sample(time, av.VideoFrame.from_image(image)))
    selections = list(select_frames(samples))
    assert [(item.sample.time, item.reason) for item in selections[1:]] == expected


@pytest.mark.parametrize(
    ("ending", "reason"),
    [
        ("baseline", "short-return"),
        ("two-indicators", "short-transition"),
        ("smaller", "short-transition"),
        (None, "end-of-video"),
    ],
)
def test_four_frame_localized_dwell_at_60_fps(ending, reason):
    opening = av.VideoFrame.from_image(screen("baseline"))
    badge = av.VideoFrame.from_image(screen("approval-indicator"))
    samples = [Sample(0, opening), *(Sample(i / 60, badge) for i in range(1, 5))]
    if ending is not None:
        image = screen(ending)
        if ending == "smaller":
            image = image.resize((320, 180))
        successor = av.VideoFrame.from_image(image)
        samples += [Sample(5 / 60, successor), Sample(40 / 60, successor)]

    _, first, *following = select_frames(samples)
    assert first.sample is samples[1]
    assert first.reason == reason
    if ending is not None:
        assert first.flags == ("brief-state",)
        assert first.last_observed == 4 / 60
    if ending == "two-indicators":
        assert len(following) == 1
        assert following[0].reason == "localized"
        assert following[0].onset == 5 / 60
        assert following[0].sample is samples[-1]


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
    frames = manifest_of(bundle)["frames"]
    assert [e["path"] for e in events] == [
        f["path"] for f in frames if 1 <= f["change_start"] <= 10
    ]
    assert all(e["speech"] is None and e["ocr"] is None for e in events)
    md = cli("walkthrough", bundle)
    links = re.findall(r"\[Screenshot at .*?\]\((.*?)\)", md)
    assert links == [f["path"] for f in frames]
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


def test_readers_reject_replaced_evidence(demo, tmp_path):
    _, original = demo
    bundle = tmp_path / "bundle"
    shutil.copytree(original, bundle)
    manifest = manifest_of(bundle)
    frames = manifest["frames"]
    (bundle / frames[0]["path"]).write_bytes((bundle / frames[1]["path"]).read_bytes())
    assert "screenshot SHA-256 differs" in toon("walkthrough", bundle, code=1)["error"]
    for begin in (0, 0.1):
        result = toon("inspect", bundle, "--from", begin, "--to", 0.2, code=1)
        assert "screenshot SHA-256 differs" in result["error"]

    timeline = bundle / manifest["artifacts"]["timeline"]["path"]
    timeline.write_text("[]\n")
    assert "timeline SHA-256 differs" in toon("walkthrough", bundle, code=1)["error"]
    result = toon("inspect", bundle, "--from", 0, "--to", 10, code=1)
    assert "timeline SHA-256 differs" in result["error"]


def test_narration_uses_screen_onset(demo, tmp_path, monkeypatch):
    source, _ = demo
    bundle = tmp_path / "narrated"
    segments = [
        {
            "start": 4.4,
            "end": 4.7,
            "text": "Continue carefully",
            "words": [{"start": 4.4, "text": "Continue"}, {"start": 4.6, "text": " carefully"}],
            "flags": [],
        },
        {"start": 4.65, "end": 4.8, "text": "Approval needed", "words": [], "flags": []},
    ]
    monkeypatch.setattr(
        bundles, "transcribe", lambda *_: {"language": "en", "segments": segments, "flags": []}
    )
    bundles.prepare(source, bundle, None, "off")

    page = toon("inspect", bundle, "--from", 4.55, "--to", 4.8)
    warning = page["preceding_screen"]
    assert warning["change_start"] == STATES["warning"][0] / FPS
    assert warning["time"] > 4.8
    after_onset = toon("inspect", bundle, "--from", 4.55, "--to", 5.2)
    assert after_onset["preceding_screen"] == warning
    assert all(event["kind"] == "speech" for event in after_onset["events"])
    indicator = next(
        frame
        for frame in manifest_of(bundle)["frames"]
        if frame["time"] == STATES["approval-indicator"][0] / FPS
    )
    assert page["events"][0]["preceding_frame"] == indicator["path"]
    assert page["events"][1]["preceding_frame"] == warning["path"]
    steps = cli("walkthrough", bundle).split("\n## ")
    warning_index = next(i for i, step in enumerate(steps) if warning["path"] in step)
    assert "- Continue\n" in steps[warning_index - 1]
    assert "- carefully\n" in steps[warning_index]
    assert "- Approval needed\n" in steps[warning_index]

    args = ("inspect", bundle, "--from", 4.49, "--to", 4.8, "--limit", 1)
    before_onset = toon(*args)
    assert before_onset["preceding_screen"]["path"] == indicator["path"]
    next_page = toon(*args, "--cursor", before_onset["next_cursor"])
    assert next_page["events"] == [warning]
    last_cursor = next_page["next_cursor"]
    assert toon(*args, "--cursor", last_cursor)["events"] == [page["events"][1]]
    (bundle / warning["path"]).write_bytes((bundle / indicator["path"]).read_bytes())
    result = toon(*args, "--cursor", last_cursor, code=1)
    assert "screenshot SHA-256 differs" in result["error"]


def test_extended_motion_uses_capture_time(demo, tmp_path, monkeypatch):
    source, _ = demo
    bundle = tmp_path / "scroll"
    opening = av.VideoFrame.from_image(screen("baseline"))
    motion = [
        av.VideoFrame.from_image(Image.new("RGB", SIZE, color)) for color in ("black", "#808080")
    ]
    samples = [Sample(0, opening), *(Sample(i / FPS, motion[i % 2]) for i in range(FPS, 97))]
    for sample in samples:
        sample.frame.pts = round(sample.time * FPS)
        sample.frame.time_base = Fraction(1, FPS)
    segments = [
        {"start": 2, "end": 2.5, "text": "Scrolling", "words": [], "flags": []},
        {"start": 4, "end": 4.5, "text": "Row visible", "words": [], "flags": []},
    ]
    monkeypatch.setattr(bundles, "samples", lambda *_: iter(samples))
    monkeypatch.setattr(
        bundles, "transcribe", lambda *_: {"language": "en", "segments": segments, "flags": []}
    )
    bundles.prepare(source, bundle, None, "off")

    page = toon("inspect", bundle, "--from", 1, "--to", 4)
    before, captured, after = page["events"]
    assert page["preceding_screen"]["time"] == 0
    assert before["preceding_frame"] == page["preceding_screen"]["path"]
    assert captured["time"] == captured["change_start"] == 4
    assert captured["reason"] == "extended-motion"
    assert "unsettled-screen" in captured["flags"]
    assert after["preceding_frame"] == captured["path"]
    later = toon("inspect", bundle, "--from", 4.1, "--to", 5)
    assert later["preceding_screen"] == captured
    _, opening_step, motion_step = cli("walkthrough", bundle).split("\n## ")
    assert "- Scrolling\n" in opening_step
    assert "Screen at 00:04.000" in motion_step
    assert "- Row visible\n" in motion_step


@pytest.mark.parametrize("rollback", [False, True])
def test_prepare_rejects_ocr_data_change(demo, tmp_path, monkeypatch, rollback):
    source, _ = demo
    bundle = tmp_path / "bundle"
    initial = toon("prepare", source, "--out", bundle, "--ocr", "off")
    assert not initial["cache_hit"]
    manifest = manifest_of(bundle)
    traineddata = tmp_path / "eng.traineddata"
    traineddata.write_bytes(b"original")
    monkeypatch.setattr(evidence.shutil, "which", lambda _: "tesseract")
    monkeypatch.setattr(
        evidence,
        "run",
        lambda command: f'List of available languages in "{tmp_path}" (1):\neng'
        if command[-1] == "--list-langs"
        else "tesseract test",
    )

    def change_language_data(paths, *, enabled):
        assert enabled
        saved = traineddata.with_suffix(".original")
        traineddata.rename(saved)
        traineddata.write_bytes(b"replacement")
        if rollback:
            saved.replace(traineddata)
        return [{"text": "", "words": [], "flags": []} for _ in paths]

    monkeypatch.setattr(bundles, "screen_text", change_language_data)
    with pytest.raises(EvidenceError, match="backend inputs changed"):
        bundles.prepare(source, bundle, None, "auto")
    assert manifest_of(bundle) == manifest
    assert len(list((bundle / "evidence").iterdir())) == 1


def test_frame_crops_and_nearest_timestamp(demo):
    _, bundle = demo
    exact = toon("frame", bundle, "--at", 8)
    assert exact["selection"] == "exact"
    assert exact["time"] == 8
    with Image.open(exact["path"]) as image:
        assert image.size == SIZE
        exact_price = image.crop(amount_box())
        assert_price_region(exact_price, "result")
    earlier = toon("frame", bundle, "--at", 8.01, "--crop", ",".join(map(str, AMOUNT_CROP)))
    assert earlier["selection"] == "nearest-frame"
    assert earlier["time"] == 8
    with Image.open(earlier["path"]) as cropped:
        assert cropped.size == AMOUNT_CROP[2:]
        assert_price_region(cropped, "result")
        assert cropped.tobytes() == exact_price.tobytes()
    later = toon("frame", bundle, "--at", 8.03)
    assert later["time"] == pytest.approx(8 + 1 / FPS)
    with Image.open(later["path"]) as image:
        assert image.size == SIZE
        assert_price_region(image.crop(amount_box()), "result")
    after = toon("frame", bundle, "--at", 120)
    assert after["time"] == pytest.approx(12 - 1 / FPS)
    assert "exceeds" in toon("frame", bundle, "--at", 0, "--crop", "639,0,2,5", code=1)["error"]


def test_ocr_amount_and_warning(demo, tmp_path):
    tesseract = shutil.which("tesseract")
    if tesseract is None:
        pytest.skip("requires Tesseract with English language data")
    languages = subprocess.run([tesseract, "--list-langs"], capture_output=True, text=True)
    if "eng" not in languages.stdout.splitlines()[1:]:
        pytest.skip("requires Tesseract English language data")
    source, _ = demo
    bundle = tmp_path / "ocr"
    assert not toon("prepare", source, "--out", bundle, "--ocr", "off")["cache_hit"]
    without_ocr = manifest_of(bundle)
    summary = toon("prepare", source, "--out", bundle, "--ocr", "auto")
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
        "--ocr",
        "off",
    )
    assert summary["speech_segments"] > 0
    manifest = manifest_of(bundle)
    path = manifest["artifacts"]["transcript"]["path"]
    segments = json.loads((bundle / path).read_text())["segments"]
    assert any(segment["words"] for segment in segments)
    approval_starts = [
        [
            word["start"]
            for word in segment["words"]
            if QUIET_APPROVAL_START <= word["start"] < QUIET_APPROVAL_END
        ]
        for segment in segments
        if "approval" in segment["text"].lower() and "quiet-audio" in segment["flags"]
    ]
    assert any(starts == sorted(starts) and len(set(starts)) > 1 for starts in approval_starts)
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
