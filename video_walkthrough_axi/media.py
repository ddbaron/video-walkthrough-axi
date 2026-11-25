"""Read local media and keep decoded frame timestamps in the source time base."""

import hashlib
import json
import math
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import av
from PIL import Image


class EvidenceError(Exception):
    """A user-facing processing error."""


def device_signature(path: Path) -> tuple:
    info = path.stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(command: list[str], *, env: dict[str, str] | None = None) -> str:
    try:
        return subprocess.run(command, capture_output=True, text=True, check=True, env=env).stdout
    except FileNotFoundError as error:
        raise EvidenceError(f"Install {command[0]} and make it available on PATH.") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip().splitlines()
        message = detail[-1] if detail else f"exit {error.returncode}"
        raise EvidenceError(f"{command[0]} failed: {message}") from error


def probe(source: Path) -> dict:
    result = json.loads(
        run(
            [
                "ffprobe",
                "-v",
                "error",
                "-protocol_whitelist",
                "file",
                "-show_format",
                "-show_streams",
                "-of",
                "json",
                str(source),
            ]
        )
    )
    formats = set(result["format"]["format_name"].split(","))
    if formats.intersection({"concat", "dash", "hls"}):
        raise EvidenceError("Provide a standalone media file instead of a playlist.")
    raw = result["streams"]
    video = next(
        (
            s
            for s in raw
            if s["codec_type"] == "video" and not s.get("disposition", {}).get("attached_pic")
        ),
        None,
    )
    audio = next((s for s in raw if s["codec_type"] == "audio"), None)
    if video is None:
        raise EvidenceError("The source has no video stream.")

    def time_value(record, field):
        value = record.get(field)
        return float(value) if value not in (None, "N/A") else None

    streams = []
    start = time_value(result["format"], "start_time")
    if start is None:
        known = [t for s in raw if (t := time_value(s, "start_time")) is not None]
        start = min(known, default=0.0)
    for s in (video, audio):
        if s is None:
            continue
        beginning = time_value(s, "start_time")
        stream = {
            "index": s["index"],
            "type": s["codec_type"],
            "codec": s.get("codec_name"),
            "time_base": s["time_base"],
            "start": beginning,
            "offset": round(beginning - start, 9) if beginning is not None else None,
            "duration": time_value(s, "duration"),
        }
        if s is video:
            stream.update(width=s["width"], height=s["height"])
        streams.append(stream)
    return {
        "time_origin": start,
        "streams": streams,
        "duration": time_value(result["format"], "duration"),
    }


@dataclass(frozen=True)
class Sample:
    time: float
    frame: av.VideoFrame

    def save(self, path: Path, crop: tuple | None = None) -> None:
        image = self.frame.to_image()
        if crop is not None:
            x, y, width, height = crop
            if x + width > image.width or y + height > image.height:
                raise EvidenceError(
                    f"Crop exceeds the decoded frame ({image.width}×{image.height})."
                )
            image = image.crop((x, y, x + width, y + height))
        image.save(path, format="PNG", compress_level=3)


@contextmanager
def open_video(source: Path, index: int):
    with av.open(str(source), options={"protocol_whitelist": "file"}) as container:
        stream = container.streams[index]
        stream.thread_type = "AUTO"
        yield container, stream


def decoded(container, stream, origin: float) -> Iterator[Sample]:
    previous = -math.inf
    for frame in container.decode(stream):
        if frame.pts is None or frame.time_base is None:
            raise EvidenceError("A decoded video frame has no presentation timestamp.")
        time = round(float(frame.pts * frame.time_base) - origin, 9)
        if time < previous:
            raise EvidenceError("Video presentation timestamps move backwards.")
        previous = time
        yield Sample(time, frame)


def samples(source: Path, media: dict) -> Iterator[Sample]:
    with open_video(source, media["streams"][0]["index"]) as (container, stream):
        yield from decoded(container, stream, media["time_origin"])


def pick_frame(source: Path, media: dict, at: float) -> Sample:
    """Use source PTS to select the closest frame, favoring the later one on ties."""
    index = media["streams"][0]["index"]
    origin = media["time_origin"]
    with open_video(source, index) as (container, stream):
        absolute = at + origin
        if stream.time_base and stream.duration is not None and stream.start_time is not None:
            end = float((stream.start_time + stream.duration) * stream.time_base)
            absolute = min(absolute, end)
        if stream.start_time is not None and stream.time_base:
            try:
                target = math.floor(absolute / float(stream.time_base))
                if target > stream.start_time:
                    container.seek(target, stream=stream)
                    sample, bracketed = nearest(container, stream, origin, at)
                    if sample is not None and bracketed:
                        return sample
            except (av.FFmpegError, OverflowError):
                pass
    with open_video(source, index) as (container, stream):
        sample, _ = nearest(container, stream, origin, at)
    if sample is None:
        raise EvidenceError("The source contains no decodable video frames.")
    return sample


def nearest(container, stream, origin: float, at: float) -> tuple[Sample | None, bool]:
    last = None
    for sample in decoded(container, stream, origin):
        if sample.time >= at - 1e-9:
            if last is not None and at - last.time < sample.time - at - 1e-9:
                return last, True
            return sample, last is not None or abs(sample.time - at) <= 1e-9
        last = sample
    return last, last is not None


def verify_png(path: Path, expected: tuple[int, int]) -> None:
    with Image.open(path) as image:
        if image.format != "PNG" or image.size != expected:
            raise EvidenceError("Frame validation failed: unexpected PNG format or dimensions.")
        image.verify()
