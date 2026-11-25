"""Read bounded timeline windows and render a screenshot-led Markdown draft."""

import bisect
import hashlib
import re
from pathlib import Path
from urllib.parse import quote

from .bundle import load_timeline, local_artifact, read_manifest
from .media import EvidenceError, digest

TEXT_LIMIT = 500


def preview(text: str) -> tuple[str, bool]:
    return text[:TEXT_LIMIT], len(text) > TEXT_LIMIT


def frame_preview(event: dict) -> dict:
    ocr = event["ocr"]
    text, truncated = preview(ocr["text"]) if ocr is not None else (None, False)
    return {
        "kind": "frame",
        "time": event["time"],
        "change_start": event["change_start"],
        "path": event["path"],
        "reason": event["reason"],
        "speech": None,
        "ocr": text,
        "ocr_truncated": truncated,
        "flags": event["flags"] + (ocr["flags"] if ocr is not None else []),
    }


def event_preview(event: dict) -> dict:
    if event["kind"] == "frame":
        return frame_preview(event)
    segment = event["speech"]
    text, truncated = preview(segment["text"])
    return {
        "kind": "speech",
        "time": event["time"],
        "end": segment["end"],
        "preceding_frame": event["preceding_frame"],
        "speech": text,
        "speech_truncated": truncated,
        "ocr": None,
        "flags": segment["flags"],
    }


def inspect_window(root: Path, begin: float, end: float, limit: int, cursor: str | None) -> dict:
    manifest = read_manifest(root)
    timeline = load_timeline(root, manifest)
    context = None
    window = []
    for event in timeline:
        time = event["time"]
        if event["kind"] == "frame":
            if event["change_start"] < begin:
                context = event
            if begin <= time <= end:
                window.append(event)
            if time > end:
                break
        elif time <= end and event["speech"]["end"] > begin:
            window.append(event)

    identity = hashlib.sha256(
        f"{manifest['artifacts']['timeline']['sha256']}:{begin}:{end}".encode()
    ).hexdigest()[:20]
    offset = 0
    if cursor is not None:
        if not re.fullmatch(rf"{identity}:[0-9]{{1,10}}", cursor):
            raise EvidenceError("Invalid cursor for this bundle and time window.")
        offset = int(cursor.split(":")[1])
        if not 0 < offset < len(window):
            raise EvidenceError("Cursor offset is outside this time window.")

    events = window[offset : offset + limit]
    following = offset + len(events)
    return {
        "events": [event_preview(event) for event in events],
        "preceding_screen": frame_preview(context) if context else None,
        "from": begin,
        "to": end,
        "returned": len(events),
        "next_cursor": f"{identity}:{following}" if following < len(window) else None,
        "timeline": manifest["artifacts"]["timeline"]["path"],
        "flags": manifest["flags"],
    }


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
    times = [frame["change_start"] for frame in frames]
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


def walkthrough(root: Path) -> str:
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
        if digest(local_artifact(root, path)) != frame["sha256"]:
            raise EvidenceError(f"Bundle screenshot SHA-256 differs: {path}")
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
            "**On-screen text**",
            "",
        ]
        ocr = frame["ocr"]
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
