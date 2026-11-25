"""Collect local speech and optional screen text with uncertainty cues."""

import csv
import math
import os
import shutil
import sys
import wave
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from .media import EvidenceError, device_signature, digest, run

ASR_SETTINGS = {
    "sample_rate": 16000,
    "word_timestamps": True,
    "condition_on_previous_text": False,
    "quiet_dbfs": -42.0,
    "quiet_word_min_seconds": 0.1,
    "low_logprob": -0.8,
    "no_speech_probability": 0.6,
    "low_word_probability": 0.35,
}
OCR_SETTINGS = {"page_segmentation": 11, "language": "eng"}


def fingerprint(path: Path, signatures: dict) -> str:
    resolved = path.resolve()
    entries = {path.parent, path, resolved.parent, resolved}
    pending = [path.absolute()]
    visited = set()
    while pending:
        current = pending.pop()
        for entry in (current, *current.parents):
            if entry in visited:
                continue
            visited.add(entry)
            if entry.is_symlink():
                entries.update((entry, entry.parent))
                pending.append(entry.parent / entry.readlink())
    for entry in entries:
        signatures[entry] = device_signature(entry, follow_symlinks=False)
    return digest(resolved)


def backend_config(
    media: dict,
    model: Path | None,
    ocr: str,
) -> tuple[dict, dict, dict]:
    settings = {"asr": None, "ocr": None}
    backends = {"asr": None, "ocr": None}
    signatures = {}
    if len(media["streams"]) > 1:
        if model is None or not (model / "config.json").is_file():
            raise EvidenceError("Audio requires --model DIR.")
        weights = next(
            (model / f for f in ("weights.safetensors", "weights.npz") if (model / f).is_file()),
            None,
        )
        if weights is None:
            raise EvidenceError("The model directory needs weights.safetensors or weights.npz.")
        try:
            mlx_whisper = version("mlx-whisper")
            mlx = version("mlx")
        except PackageNotFoundError as error:
            raise EvidenceError("Install video-walkthrough-axi[mlx] on Apple Silicon.") from error
        settings["asr"] = ASR_SETTINGS
        backends["asr"] = {
            "name": "mlx-whisper",
            "version": mlx_whisper,
            "mlx_version": mlx,
            "model": {
                "config_sha256": fingerprint(model / "config.json", signatures),
                "weights_file": weights.name,
                "weights_sha256": fingerprint(weights, signatures),
            },
        }
    if ocr == "auto" and shutil.which("tesseract"):
        traineddata = tesseract_language_file()
        if traineddata.is_file():
            settings["ocr"] = OCR_SETTINGS
            backends["ocr"] = {
                "name": "tesseract",
                "version": run(["tesseract", "--version"]).splitlines()[0],
                "language_sha256": fingerprint(traineddata, signatures),
            }
    settings["ocr_request"] = ocr
    return settings, backends, signatures


def tesseract_language_file() -> Path:
    output = run(["tesseract", "--list-langs"]).splitlines()
    if not output or '"' not in output[0]:
        raise EvidenceError("Tesseract did not report its language data directory.")
    directory = Path(output[0].split('"')[1])
    language = OCR_SETTINGS["language"]
    return directory / f"{language}.traineddata"


def audio_pcm(source: Path, media: dict, wav: Path) -> np.ndarray:
    rate = ASR_SETTINGS["sample_rate"]
    audio = media["streams"][1]
    origin = media["time_origin"]
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-protocol_whitelist",
            "file",
            "-copyts",
            "-i",
            str(source),
            "-map",
            f"0:{audio['index']}",
            "-ac",
            "1",
            "-af",
            f"asetpts=PTS-({origin})/TB,aresample={rate}:async=1:first_pts=0",
            "-c:a",
            "pcm_s16le",
            str(wav),
        ]
    )
    with wave.open(str(wav), "rb") as stream:
        return np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")


def normalized_speech(raw: dict, pcm: np.ndarray) -> list[dict]:
    rate = ASR_SETTINGS["sample_rate"]
    duration = len(pcm) / rate

    def bound(value):
        return round(max(0.0, min(duration, float(value))), 6)

    def signal_dbfs(start, end):
        signal = pcm[round(start * rate) : round(end * rate)].astype(np.float32) / 32768
        power = float(np.mean(signal**2)) if signal.size else 0.0
        return round(10 * math.log10(max(power, 1e-12)), 2)

    segments = []
    for segment in raw["segments"]:
        start, end = bound(segment["start"]), bound(segment["end"])
        text = " ".join(segment["text"].split())
        if end <= start or not text:
            continue
        words = []
        for word in segment.get("words", ()):
            word_start = max(start, bound(word["start"]))
            word_end = min(end, bound(word["end"]))
            if word_end >= word_start and word["word"].strip():
                words.append(
                    {
                        "start": word_start,
                        "end": word_end,
                        "text": word["word"],
                        "probability": round(float(word["probability"]), 6),
                        "rms_dbfs": signal_dbfs(word_start, word_end),
                    }
                )

        words.sort(key=lambda w: w["start"])
        dbfs = signal_dbfs(start, end)
        quiet = ASR_SETTINGS["quiet_dbfs"]
        quiet_words = [
            w["end"] - w["start"] >= ASR_SETTINGS["quiet_word_min_seconds"]
            and w["rms_dbfs"] < quiet
            for w in words
        ]
        logprob = float(segment.get("avg_logprob", 0))
        no_speech = float(segment.get("no_speech_prob", 0))
        flags = []
        if dbfs < quiet or any(a and b for a, b in zip(quiet_words, quiet_words[1:], strict=False)):
            flags.append("quiet-audio")
        if logprob < ASR_SETTINGS["low_logprob"]:
            flags.append("low-asr-confidence")
        if no_speech >= ASR_SETTINGS["no_speech_probability"]:
            flags.append("possible-non-speech")
        if any(w["probability"] < ASR_SETTINGS["low_word_probability"] for w in words):
            flags.append("low-word-confidence")
        if not words:
            flags.append("missing-word-timestamps")
        segments.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "words": words,
                "flags": flags,
                "avg_logprob": logprob,
                "no_speech_probability": no_speech,
                "rms_dbfs": dbfs,
            }
        )
    return sorted(segments, key=lambda s: (s["start"], s["end"]))


def transcribe(
    source: Path,
    media: dict,
    model: Path | None,
    wav: Path,
) -> dict:
    if len(media["streams"]) == 1:
        return {"language": None, "flags": ["no-audio-stream"], "segments": []}
    pcm = audio_pcm(source, media, wav)
    wav.unlink()
    if not np.any(pcm):
        return {"language": None, "flags": ["silent-audio"], "segments": []}

    # An absolute directory path keeps model loading local. Set the Hub's
    # offline mode before importing MLX Whisper as a second line of defense.
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        import mlx_whisper

        with redirect_stdout(sys.stderr):
            raw = mlx_whisper.transcribe(
                pcm.astype(np.float32) / 32768,
                path_or_hf_repo=str(model),
                task="transcribe",
                word_timestamps=ASR_SETTINGS["word_timestamps"],
                condition_on_previous_text=ASR_SETTINGS["condition_on_previous_text"],
                verbose=None,
            )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise EvidenceError(f"Local MLX Whisper failed: {error}") from error
    segments = normalized_speech(raw, pcm)
    return {
        "language": raw.get("language"),
        "flags": [] if segments else ["no-speech-detected"],
        "segments": segments,
    }


def recognize(path: Path) -> dict:
    tsv = run(
        [
            "tesseract",
            str(path),
            "stdout",
            "-l",
            OCR_SETTINGS["language"],
            "--psm",
            str(OCR_SETTINGS["page_segmentation"]),
            "tsv",
        ],
        env={**os.environ, "OMP_THREAD_LIMIT": "1"},
    )
    words = [
        {
            "text": row["text"].strip(),
            "confidence": float(row["conf"]),
            "box": [int(row[k]) for k in ("left", "top", "width", "height")],
        }
        for row in csv.DictReader(tsv.splitlines(), delimiter="\t")
        if row["level"] == "5" and row["text"].strip()
    ]
    return {
        "text": " ".join(w["text"] for w in words),
        "words": words,
        "flags": ["low-ocr-confidence"] if any(w["confidence"] < 50 for w in words) else [],
    }


def screen_text(paths: list[Path], enabled: bool) -> list[dict | None]:
    if not enabled:
        return [None] * len(paths)
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(recognize, paths))
