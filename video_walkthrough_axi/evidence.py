"""Collect local speech and optional screen text with uncertainty cues."""

import csv
import math
import os
import shutil
import sys
import wave
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from dataclasses import dataclass
from functools import partial
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


@dataclass(frozen=True)
class BackendInput:
    path: Path
    identity: tuple
    sha256: str


def fingerprint(path: Path, inputs: dict[str, BackendInput]) -> str:
    resolved = path.resolve(strict=True)
    identity = device_signature(resolved)[:4]
    sha256 = digest(resolved)
    inputs[path.name] = BackendInput(resolved, identity, sha256)
    return sha256


def snapshot_inputs(directory: Path, inputs: dict[str, BackendInput]) -> None:
    for name, item in inputs.items():
        snapshot = directory / name
        shutil.copyfile(item.path, snapshot)
        if digest(snapshot) != item.sha256:
            raise EvidenceError("Speech or OCR backend inputs changed during processing.")


def backend_config(
    media: dict,
    model: Path | None,
    ocr: str,
) -> tuple[dict, dict, dict[str, BackendInput]]:
    settings = {"asr": None, "ocr": None}
    backends = {"asr": None, "ocr": None}
    inputs = {}
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
                "config_sha256": fingerprint(model / "config.json", inputs),
                "weights_file": weights.name,
                "weights_sha256": fingerprint(weights, inputs),
            },
        }
    if ocr == "auto" and shutil.which("tesseract"):
        traineddata = tesseract_language_file()
        if traineddata is not None and traineddata.is_file():
            settings["ocr"] = OCR_SETTINGS
            backends["ocr"] = {
                "name": "tesseract",
                "version": run(["tesseract", "--version"]).splitlines()[0],
                "language_sha256": fingerprint(traineddata, inputs),
            }
    settings["ocr_request"] = ocr
    return settings, backends, inputs


def tesseract_language_file() -> Path | None:
    try:
        output = run(["tesseract", "--list-langs"]).splitlines()
    except EvidenceError:
        return None
    if not output or '"' not in output[0]:
        return None
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
            f"asetpts=PTS-({origin})/TB,aresample={rate}:async=1:"
            "min_comp=0:min_hard_comp=0:first_pts=0",
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
        if dbfs < quiet or any(quiet_words):
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
    model: Path,
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


def recognize(path: Path, data_dir: Path) -> dict:
    tsv = run(
        [
            "tesseract",
            str(path),
            "stdout",
            "--tessdata-dir",
            str(data_dir),
            "-l",
            OCR_SETTINGS["language"],
            "--psm",
            str(OCR_SETTINGS["page_segmentation"]),
            "-c",
            "tessedit_create_tsv=1",
        ],
        env={**os.environ, "OMP_THREAD_LIMIT": "1"},
    )
    words = [
        {
            "text": row["text"].strip(),
            "confidence": float(row["conf"]),
            "box": [int(row[k]) for k in ("left", "top", "width", "height")],
        }
        for row in csv.DictReader(tsv.splitlines(), delimiter="\t", quoting=csv.QUOTE_NONE)
        if row["level"] == "5" and row["text"].strip()
    ]
    return {
        "text": " ".join(w["text"] for w in words),
        "words": words,
        "flags": ["low-ocr-confidence"] if any(w["confidence"] < 50 for w in words) else [],
    }


def screen_text(paths: list[Path], data_dir: Path | None) -> list[dict | None]:
    if data_dir is None:
        return [None] * len(paths)
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(partial(recognize, data_dir=data_dir), paths))
