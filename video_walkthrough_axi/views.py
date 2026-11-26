"""Read evidence summaries, timeline windows and visual previews."""

import bisect
import hashlib
import re
import tempfile
from pathlib import Path
from urllib.parse import quote

from PIL import Image, ImageDraw

from .bundle import event_time, load_timeline, local_artifact, read_manifest, screen_time
from .media import EvidenceError, digest

TEXT_LIMIT = 500


def preview(text: str) -> tuple[str, bool]:
    return text[:TEXT_LIMIT], len(text) > TEXT_LIMIT


def frame_index(event: dict) -> dict:
    fields = ("time", "change_start", "path", "reason", "global_delta", "tile_delta", "flags")
    return {field: event[field] for field in fields}


def frame_preview(event: dict, include_ocr: bool) -> dict:
    result = {"kind": "frame", **frame_index(event), "speech": None}
    if include_ocr:
        ocr = event.get("ocr")
        text, truncated = preview(ocr["text"]) if ocr is not None else (None, False)
        result.update(ocr=text, ocr_truncated=truncated)
        if ocr is not None:
            result["flags"] = event["flags"] + ocr["flags"]
    return result


def event_preview(event: dict, include_ocr: bool) -> dict:
    if event["kind"] == "frame":
        return frame_preview(event, include_ocr)
    segment = event["speech"]
    text, truncated = preview(segment["text"])
    result = {
        "kind": "speech",
        "time": event["time"],
        "end": segment["end"],
        "preceding_frame": event["preceding_frame"],
        "speech": text,
        "speech_truncated": truncated,
        "flags": segment["flags"],
    }
    if include_ocr:
        result["ocr"] = None
    return result


def verify_screenshot(root: Path, frame: dict) -> None:
    path = frame["path"]
    if digest(local_artifact(root, path)) != frame["sha256"]:
        raise EvidenceError(f"Bundle screenshot SHA-256 differs: {path}")


def _cursor_offset(cursor: str | None, identity: str, window_size: int) -> int:
    if cursor is None:
        return 0
    if not re.fullmatch(rf"{identity}:[0-9]{{1,10}}", cursor):
        raise EvidenceError("Invalid cursor for this bundle, time window and filters.")
    offset = int(cursor.split(":")[1])
    if not 0 < offset < window_size:
        raise EvidenceError("Cursor offset is outside this time window.")
    return offset


def inspect_window(
    root: Path,
    begin: float,
    end: float,
    limit: int,
    cursor: str | None,
    *,
    kind: str | None = None,
    min_delta: float = 0.0,
    include_ocr: bool = False,
) -> dict:
    manifest = read_manifest(root)
    timeline = load_timeline(root, manifest)
    context = None
    window = []
    for event in timeline:
        time = event_time(event)
        if time > end:
            break
        if event["kind"] == "frame":
            if time < begin:
                context = event
            elif kind != "speech" and event["global_delta"] >= min_delta:
                window.append(event)
        elif kind != "frame" and event["speech"]["end"] > begin:
            window.append(event)

    identity = hashlib.sha256(
        (
            f"{manifest['artifacts']['timeline']['sha256']}:{begin}:{end}:"
            f"{kind}:{min_delta}:{include_ocr}"
        ).encode()
    ).hexdigest()[:20]
    offset = _cursor_offset(cursor, identity, len(window))
    events = window[offset : offset + limit]
    following = offset + len(events)
    frames = {event["path"]: event for event in timeline if event["kind"] == "frame"}
    paths = {e["path"] if e["kind"] == "frame" else e["preceding_frame"] for e in events}
    if context is not None:
        paths.add(context["path"])
    paths.discard(None)
    for path in sorted(paths):
        verify_screenshot(root, frames[path])
    return {
        "events": [event_preview(event, include_ocr) for event in events],
        "preceding_screen": frame_preview(context, include_ocr) if context else None,
        "from": begin,
        "to": end,
        "returned": len(events),
        "next_cursor": f"{identity}:{following}" if following < len(window) else None,
        "timeline": manifest["artifacts"]["timeline"]["path"],
        "flags": [flag for flag in manifest["flags"] if include_ocr or not flag.startswith("ocr-")],
    }


def overview(root: Path) -> dict:
    manifest = read_manifest(root)
    timeline = load_timeline(root, manifest)
    frames = []
    speech = []
    for event in timeline:
        if event["kind"] == "frame":
            frames.append({"index": len(frames) + 1, **frame_index(event)})
            continue
        segment = event["speech"]
        flags = set(segment["flags"])
        if "quiet-audio" in flags and flags & {"low-asr-confidence", "low-word-confidence"}:
            if speech and "omitted_segments" in speech[-1]:
                entry = speech[-1]
                entry["end"] = max(entry["end"], segment["end"])
                entry["omitted_segments"] += 1
                entry["flags"] = sorted(set(entry["flags"]) | flags)
            else:
                speech.append(
                    {
                        "start": segment["start"],
                        "end": segment["end"],
                        "omitted_segments": 1,
                        "flags": sorted(flags | {"likely-silence-hallucination"}),
                    }
                )
        else:
            text, truncated = preview(segment["text"])
            speech.append(
                {
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": text,
                    "truncated": truncated,
                    "flags": segment["flags"],
                }
            )
    return {
        "media": manifest["media"],
        "flags": manifest["flags"],
        "counts": {key: manifest["counts"][key] for key in ("frames", "speech_segments", "events")},
        "transcript": manifest["artifacts"]["transcript"]["path"],
        "speech": speech,
        "frames": frames,
    }


def contact_sheet(root: Path, begin: float | None, end: float | None) -> dict:
    manifest = read_manifest(root)
    timeline = load_timeline(root, manifest)
    frames = [event for event in timeline if event["kind"] == "frame"]
    indexed = [
        (index, frame)
        for index, frame in enumerate(frames, 1)
        if (begin is None or screen_time(frame) >= begin)
        and (end is None or screen_time(frame) <= end)
    ]
    if not indexed:
        return {"frames": 0, "sheets": []}

    extracts = local_artifact(root, "extracts")
    extracts.mkdir(mode=0o700, exist_ok=True)
    generation = manifest["artifacts"]["timeline"]["sha256"][:16]
    columns, per_page = 4, 20
    width, height = 256, 170
    sheets = []
    with tempfile.TemporaryDirectory(prefix=".contact-sheet-", dir=extracts) as temporary:
        for offset in range(0, len(indexed), per_page):
            page = indexed[offset : offset + per_page]
            first, last = page[0][0], page[-1][0]
            filename = f"contact-sheet-{generation}-{first:06d}-{last:06d}.png"
            rows = (len(page) + columns - 1) // columns
            sheet = Image.new("RGB", (columns * width, rows * height), "#20252c")
            draw = ImageDraw.Draw(sheet)
            for cell, (index, frame) in enumerate(page):
                verify_screenshot(root, frame)
                with Image.open(local_artifact(root, frame["path"])) as thumbnail:
                    thumbnail.thumbnail((width - 12, height - 29))
                    x, y = cell % columns * width, cell // columns * height
                    sheet.paste(thumbnail, (x + (width - thumbnail.width) // 2, y + 5))
                draw.text(
                    (x + 8, y + height - 20),
                    f"{index:06d}  {timestamp(frame['time'])}",
                    fill="white",
                )
            path = Path(temporary) / filename
            path.touch(mode=0o600)
            sheet.save(path)
            sheets.append({"path": f"extracts/{filename}", "first": first, "last": last})
        for sheet in sheets:
            filename = Path(sheet["path"]).name
            (Path(temporary) / filename).replace(extracts / filename)
    return {"frames": len(indexed), "sheets": sheets}


def timestamp(time: float) -> str:
    millis = round(abs(time) * 1000)
    seconds, ms = divmod(millis, 1000)
    minutes, sec = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)
    clock = f"{hours}:{minute:02d}:{sec:02d}" if hours else f"{minute:02d}:{sec:02d}"
    return f"{'-' if time < 0 else ''}{clock}.{ms:03d}"


def escape(text: str) -> str:
    return re.sub(r"([\\`*_{}\[\]<>#!|&])", r"\\\1", text)


def speech_by_frame(frames: list[dict], timeline: list[dict]) -> list[list[tuple[str, list]]]:
    times = [screen_time(frame) for frame in frames]
    # Slot zero contains speech before the first decoded screenshot, if any.
    slots = [[] for _ in range(len(times) + 1)]
    for event in timeline:
        if event["kind"] != "speech":
            continue
        speech = event["speech"]
        if not speech["words"]:
            slots[bisect.bisect_right(times, speech["start"])].append(
                (speech["text"], speech["flags"])
            )
            continue
        parts: dict[int, list[str]] = {}
        for word in speech["words"]:
            index = bisect.bisect_right(times, word["start"])
            parts.setdefault(index, []).append(word["text"])
        for index, words in parts.items():
            slots[index].append((" ".join("".join(words).split()), speech["flags"]))
    return slots


def speech_lines(parts: list[tuple[str, list]]) -> list[str]:
    if not parts:
        return ["- No speech evidence in this interval."]
    lines = []
    for text, flags in parts:
        review = f" (review: {', '.join(flags)})" if flags else ""
        lines.append(f"- {escape(text)}{review}")
    return lines


def walkthrough(root: Path, *, include_ocr: bool = False) -> str:
    manifest = read_manifest(root)
    timeline = load_timeline(root, manifest)
    frames = [e for e in timeline if e["kind"] == "frame"]
    speech = speech_by_frame(frames, timeline)
    lines = [
        "# Draft walkthrough",
        "",
        "Automated evidence in recording order. Verify actions, wording and amounts "
        "against the cited screenshots and recording.",
        "",
    ]
    if speech[0]:
        lines += [
            "## Before the first screenshot",
            "",
            "**Speech**",
            "",
            *speech_lines(speech[0]),
            "",
        ]
    for index, (frame, parts) in enumerate(zip(frames, speech[1:], strict=True), 1):
        time = timestamp(frame["time"])
        path = frame["path"]
        verify_screenshot(root, frame)
        flags = f" (review: {', '.join(frame['flags'])})" if frame["flags"] else ""
        lines += [
            f"## {index}. Screen at {time}",
            "",
            f"[Screenshot at {time}]({quote(path)}) · {frame['reason']}{flags}",
            "",
            "**Speech**",
            "",
            *speech_lines(parts),
            "",
        ]
        if not include_ocr:
            continue
        lines += ["**On-screen text**", ""]
        ocr = frame.get("ocr")
        if ocr is None:
            lines.append("OCR unavailable for this frame.")
        elif not ocr["text"]:
            lines.append("OCR found no text.")
        else:
            lines.append(f"> {escape(ocr['text'])}")
            if ocr["flags"]:
                lines += ["", f"Review OCR ({', '.join(ocr['flags'])})."]
        lines.append("")
    return "\n".join(lines)
