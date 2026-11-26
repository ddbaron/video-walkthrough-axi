import hashlib
import json
import os
import re
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import av
import pytest
from cli_helpers import cli, frames_of, manifest_of, toon
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
from video_walkthrough_axi.media import Sample


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


def test_help_and_usage():
    assert "walkthrough" in cli("--help")
    for command in ("prepare", "inspect", "overview", "contact-sheet", "frame", "walkthrough"):
        assert "usage:" in cli(command, "--help")
    assert "0.1.0" in cli("--version")
    assert "help" in toon("inspect", "bundle", "--from", "-1", "--to", "4", code=2)
    assert "help" in toon("inspect", "bundle", "--from", "4", "--to", "3", code=1)
    assert "help" in toon("frame", "bundle", "--at", "0", "--crop", "0,0,1,0", code=2)


def test_native_coverage_and_manifest(demo):
    source, bundle = demo
    manifest = manifest_of(bundle)
    frames = frames_of(bundle)
    assert manifest["schema_version"] == 2
    assert "frames" not in manifest
    assert "\n  " in (bundle / "manifest.json").read_text()
    assert manifest["source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest["media"]["streams"][0]["offset"] == 0
    assert "no-audio-stream" in manifest["flags"]
    assert len(frames) < 15
    for name, (start, end) in STATES.items():
        matches = [f for f in frames if start / FPS - 1e-8 <= f["time"] < end / FPS - 1e-8]
        assert matches, name
        assert matches[0]["change_start"] == pytest.approx(start / FPS)
        if name == "single-frame":
            assert "single-frame" in matches[0]["flags"]
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
    frames = frames_of(bundle)
    assert [e["path"] for e in events] == [
        f["path"] for f in frames if 1 <= f["change_start"] <= 10
    ]
    assert all(e["speech"] is None and "ocr" not in e for e in events)
    md = cli("walkthrough", bundle)
    links = re.findall(r"\[Screenshot at .*?\]\((.*?)\)", md)
    assert links == [f["path"] for f in frames]
    assert all((bundle / path).is_file() for path in links)
    assert "**Speech**" in md and "**On-screen text**" not in md


def test_cache_checks_artifacts_and_source_identity(demo, tmp_path):
    original, _ = demo
    source = tmp_path / "source.mp4"
    shutil.copyfile(original, source)
    bundle = tmp_path / "cache"
    assert not toon("prepare", source, "--out", bundle, "--ocr", "off")["cache_hit"]
    first = frames_of(bundle)[0]
    assert toon("prepare", source, "--out", bundle, "--ocr", "off")["cache_hit"]
    (bundle / first["path"]).write_bytes(b"invalid")
    assert not toon("prepare", source, "--out", bundle, "--ocr", "off")["cache_hit"]
    assert frames_of(bundle)[0]["path"] != first["path"]
    with source.open("ab") as file:
        file.write(b"changed")
    assert "SHA-256 differs" in toon("frame", bundle, "--at", 0, code=1)["error"]


def test_overview_and_contact_sheet_for_legacy_bundle(demo, tmp_path):
    _, original = demo
    bundle = tmp_path / "legacy"
    shutil.copytree(original, bundle)
    manifest = manifest_of(bundle)
    frames = frames_of(bundle)
    manifest["schema_version"] = 1
    manifest["frames"] = frames
    (bundle / "manifest.json").write_text(json.dumps(manifest))

    summary = toon("overview", bundle)
    assert summary["media"] == manifest["media"]
    assert summary["speech"] == []
    assert [frame["path"] for frame in summary["frames"]] == [frame["path"] for frame in frames]
    assert all("ocr" not in frame for frame in summary["frames"])

    all_sheets = toon("contact-sheet", bundle)
    assert all_sheets["frames"] == len(frames)
    assert all_sheets == toon("contact-sheet", bundle)
    selected = toon("contact-sheet", bundle, "--from", 0, "--to", 0)
    assert selected["frames"] == 1
    assert selected["sheets"][0]["first"] == selected["sheets"][0]["last"] == 1
    path = selected["sheets"][0]["path"]
    assert path.startswith("extracts/")
    with Image.open(bundle / path) as sheet:
        assert sheet.format == "PNG"
        assert len(sheet.getcolors(sheet.width * sheet.height)) > 2
    assert toon("contact-sheet", bundle, "--from", 99) == {"frames": 0, "sheets": []}

    timeline = manifest["artifacts"]["timeline"]
    data = json.dumps([dict(frames[0], change_start=i, time=i) for i in range(23)]).encode()
    (bundle / timeline["path"]).write_bytes(data)
    timeline["sha256"] = hashlib.sha256(data).hexdigest()
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    paginated = toon("contact-sheet", bundle)
    assert paginated["frames"] == 23
    assert [(page["first"], page["last"]) for page in paginated["sheets"]] == [(1, 20), (21, 23)]
    for page in paginated["sheets"]:
        assert (bundle / page["path"]).is_file()


def test_inspect_filters_and_silence_summary(demo, tmp_path, monkeypatch):
    source, _ = demo
    bundle = tmp_path / "speech"
    segments = [
        (1.0, 1.2, "Spurious phrase", ["quiet-audio", "low-asr-confidence"]),
        (1.3, 1.6, "Spurious repeat", ["quiet-audio", "low-word-confidence"]),
        (2.0, 2.3, "Quiet but clear", ["quiet-audio"]),
        (2.4, 2.6, "Uncertain but audible", ["low-asr-confidence"]),
    ]
    speech = [
        {"start": a, "end": b, "text": text, "words": [], "flags": flags}
        for a, b, text, flags in segments
    ]
    monkeypatch.setattr(
        bundles, "transcribe", lambda *_: {"language": "en", "segments": speech, "flags": []}
    )
    bundles.prepare(source, bundle, None)
    words = toon("overview", bundle)["speech"]
    assert words[0]["start"] == 1
    assert words[0]["end"] == 1.6
    assert words[0]["omitted_segments"] == 2
    assert "likely-silence-hallucination" in words[0]["flags"]
    assert "text" not in words[0]
    assert [entry["text"] for entry in words[1:]] == [segment[2] for segment in segments[2:]]
    args = ("inspect", bundle, "--from", 0, "--to", 12)
    speech_page = toon(*args, "--kind", "speech", "--min-delta", 1)
    assert [event["speech"] for event in speech_page["events"]] == [s[2] for s in segments]
    frame_args = (*args, "--kind", "frame", "--min-delta", 0.002)
    frame_page = toon(*frame_args, "--limit", 1, "--ocr", "off")
    frame = frame_page["events"][0]
    assert frame["kind"] == "frame"
    assert frame["global_delta"] >= 0.002
    cursor = frame_page["next_cursor"]
    assert cursor
    assert toon(*frame_args, "--cursor", cursor)["returned"]
    assert "Invalid cursor" in toon(*frame_args, "--ocr", "on", "--cursor", cursor, code=1)["error"]
    assert "Invalid cursor" in toon(*args, "--kind", "frame", "--cursor", cursor, code=1)["error"]


@pytest.mark.parametrize("command", ["prepare", "frame"])
def test_source_directory_rollback_keeps_verified_recording(demo, tmp_path, monkeypatch, command):
    original, expected_bundle = demo
    active = tmp_path / "active"
    alternate = tmp_path / "alternate"
    saved = tmp_path / "saved"
    active.mkdir()
    alternate.mkdir()
    source = active / "input.mp4"
    shutil.copyfile(original, source)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(original),
            "-vf",
            "negate",
            str(alternate / source.name),
        ],
        check=True,
    )
    bundle = tmp_path / "bundle"
    expected = expected_bundle / frames_of(expected_bundle)[0]["path"]
    operation = "build" if command == "prepare" else "pick_frame"
    consume = getattr(bundles, operation)

    def swap_while_reading(*args):
        recording = args[2] if command == "prepare" else args[0]
        assert not recording.resolve().is_relative_to(bundle.resolve())
        active.rename(saved)
        alternate.rename(active)
        try:
            return consume(*args)
        finally:
            active.rename(alternate)
            saved.rename(active)

    if command == "frame":
        bundles.prepare(source, bundle, None, "off")
    monkeypatch.setattr(bundles, operation, swap_while_reading)
    if command == "prepare":
        bundles.prepare(source, bundle, None, "off")
        manifest = manifest_of(bundle)
        assert manifest["source"]["sha256"] == hashlib.sha256(original.read_bytes()).hexdigest()
        actual = bundle / frames_of(bundle)[0]["path"]
    else:
        actual = bundle / bundles.extract_frame(bundle, 0, None)["path"]
    with Image.open(actual) as screenshot, Image.open(expected) as reference:
        assert screenshot.tobytes() == reference.tobytes()


def test_readers_reject_replaced_evidence(demo, tmp_path):
    _, original = demo
    bundle = tmp_path / "bundle"
    shutil.copytree(original, bundle)
    manifest = manifest_of(bundle)
    frames = frames_of(bundle)
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
        for frame in frames_of(bundle)
        if frame["change_start"] == STATES["approval-indicator"][0] / FPS
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


@pytest.mark.parametrize(("fps", "repeat"), [(10, 2), (FPS, 1)])
def test_extended_motion_uses_capture_time(demo, tmp_path, monkeypatch, fps, repeat):
    source, _ = demo
    bundle = tmp_path / "scroll"
    opening = av.VideoFrame.from_image(screen("baseline"))
    samples = [Sample(0, opening)]
    for tick in range(3 * fps):
        shade = (tick // repeat % 6) * 40
        image = Image.new("RGB", SIZE, (shade, shade, shade))
        samples.append(Sample(1 + tick / fps, av.VideoFrame.from_image(image)))
    for sample in samples:
        sample.frame.pts = round(sample.time * fps)
        sample.frame.time_base = Fraction(1, fps)
    capture_time = samples[-1].time
    segments = [
        {"start": 3.6, "end": 3.7, "text": "Scrolling", "words": [], "flags": []},
        {"start": capture_time, "end": 4, "text": "Row visible", "words": [], "flags": []},
    ]
    monkeypatch.setattr(bundles, "samples", lambda *_: iter(samples))
    monkeypatch.setattr(
        bundles, "transcribe", lambda *_: {"language": "en", "segments": segments, "flags": []}
    )
    bundles.prepare(source, bundle, None, "off")

    page = toon("inspect", bundle, "--from", 3, "--to", 4)
    first, second, before, captured, after = page["events"]
    assert page["preceding_screen"]["time"] < 3
    assert first["kind"] == second["kind"] == "frame"
    assert before["preceding_frame"] == second["path"]
    assert captured["time"] == pytest.approx(capture_time)
    assert captured["change_start"] == 1
    assert captured["reason"] == "extended-motion"
    assert "unsettled-screen" in captured["flags"]
    assert after["preceding_frame"] == captured["path"]
    timeline = json.loads((bundle / page["timeline"]).read_text())
    for second in range(1, 4):
        assert (
            sum(
                event["kind"] == "frame" and second <= event["time"] < second + 1
                for event in timeline
            )
            == 3
        )
    assert [(event["kind"], event["time"]) for event in timeline[-3:]] == [
        ("speech", 3.6),
        ("frame", capture_time),
        ("speech", capture_time),
    ]
    later = toon("inspect", bundle, "--from", 4, "--to", 5)
    assert later["preceding_screen"] == captured
    *_, preceding_step, motion_step = cli("walkthrough", bundle).split("\n## ")
    assert "- Scrolling\n" in preceding_step
    assert "- Row visible\n" in motion_step


def test_frame_crops_and_nearest_timestamp(demo):
    _, bundle = demo
    exact = toon("frame", bundle, "--at", 8)
    assert exact["selection"] == "exact"
    assert exact["time"] == 8
    assert exact["path"].startswith("extracts/frame-8.")
    assert exact["path"] == toon("frame", bundle, "--at", 8.01)["path"]
    with Image.open(bundle / exact["path"]) as image:
        assert image.size == SIZE
        exact_price = image.crop(amount_box())
        assert_price_region(exact_price, "result")
    earlier = toon("frame", bundle, "--at", 8.01, "--crop", ",".join(map(str, AMOUNT_CROP)))
    assert earlier["selection"] == "nearest-frame"
    assert earlier["time"] == 8
    assert earlier["path"] != exact["path"]
    with Image.open(bundle / earlier["path"]) as cropped:
        assert cropped.size == AMOUNT_CROP[2:]
        assert_price_region(cropped, "result")
        assert cropped.tobytes() == exact_price.tobytes()
    later = toon("frame", bundle, "--at", 8.03)
    assert later["time"] == pytest.approx(8 + 1 / FPS)
    with Image.open(bundle / later["path"]) as image:
        assert image.size == SIZE
        assert_price_region(image.crop(amount_box()), "result")
    after = toon("frame", bundle, "--at", 120)
    assert after["time"] == pytest.approx(12 - 1 / FPS)
    assert "exceeds" in toon("frame", bundle, "--at", 0, "--crop", "639,0,2,5", code=1)["error"]


def tesseract_english_data():
    tesseract = shutil.which("tesseract")
    if tesseract is None:
        return None
    listing = subprocess.run(
        [tesseract, "--list-langs"], capture_output=True, text=True, check=False
    )
    lines = listing.stdout.splitlines()
    if listing.returncode or not lines or '"' not in lines[0]:
        return None
    traineddata = Path(lines[0].split('"')[1]) / "eng.traineddata"
    return traineddata if traineddata.is_file() else None


def test_ocr_amount_and_warning(demo, tmp_path):
    if tesseract_english_data() is None:
        pytest.skip("requires Tesseract with English language data")
    source, _ = demo
    bundle = tmp_path / "ocr"
    assert not toon("prepare", source, "--out", bundle, "--ocr", "off")["cache_hit"]
    without_ocr = frames_of(bundle)
    summary = toon("prepare", source, "--out", bundle, "--ocr", "auto")
    assert not summary["cache_hit"]
    assert summary["ocr_frames"] == summary["frames"]
    page = toon("inspect", bundle, "--from", 0, "--to", 12, "--limit", 100)
    assert all("ocr" not in event for event in page["events"])
    assert "On-screen text" not in cli("walkthrough", bundle)
    page = toon("inspect", bundle, "--from", 0, "--to", 12, "--limit", 100, "--ocr", "on")
    assert "On-screen text" in cli("walkthrough", bundle, "--ocr", "on")
    text = " ".join(event["ocr"] for event in page["events"])
    assert all(s in text for s in ("$24.50", "$18.50", "Selection saved", "approval required"))
    assert all(e["speech"] is None for e in page["events"])
    assert toon("prepare", source, "--out", bundle, "--ocr", "off")["cache_hit"]
    assert frames_of(bundle) == without_ocr


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
    assert any(e["speech"] and "ocr" not in e for e in page["events"])
    assert "approval" in cli("walkthrough", bundle).lower()
