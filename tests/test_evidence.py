import subprocess

import numpy as np
import pytest
from cli_helpers import manifest_of, toon

from video_walkthrough_axi import bundle as bundles
from video_walkthrough_axi import evidence
from video_walkthrough_axi.evidence import audio_pcm
from video_walkthrough_axi.media import EvidenceError, probe


def test_one_quiet_word_flags_loud_segment():
    rate = evidence.ASR_SETTINGS["sample_rate"]
    duration = 2.0
    pcm = np.full(int(duration * rate), int(0.05 * 32767), dtype="<i2")
    quiet_start, quiet_end = 0.5, 0.7
    pcm[int(quiet_start * rate) : int(quiet_end * rate)] = int(0.003 * 32767)
    raw = {
        "segments": [
            {
                "start": 0.0,
                "end": duration,
                "text": "the total is thirty two",
                "avg_logprob": -0.1,
                "no_speech_prob": 0.0,
                "words": [
                    {"start": 0.1, "end": 0.3, "word": " the", "probability": 0.9},
                    {"start": quiet_start, "end": quiet_end, "word": " total", "probability": 0.9},
                    {"start": 0.9, "end": 1.1, "word": " thirty", "probability": 0.9},
                ],
            }
        ]
    }
    segment = evidence.normalized_speech(raw, pcm)[0]
    assert segment["rms_dbfs"] >= evidence.ASR_SETTINGS["quiet_dbfs"]
    assert segment["words"][1]["rms_dbfs"] < evidence.ASR_SETTINGS["quiet_dbfs"]
    assert "quiet-audio" in segment["flags"]


@pytest.mark.parametrize(
    "change", ["replacement", "rollback", "intermediate-link", "ordinary-ancestor", "unrelated"]
)
def test_prepare_binds_ocr_data(demo, tmp_path, monkeypatch, change):
    source, _ = demo
    bundle = tmp_path / "bundle"
    initial = toon("prepare", source, "--out", bundle, "--ocr", "off")
    assert not initial["cache_hit"]
    manifest = manifest_of(bundle)
    home = tmp_path / "home"
    home.mkdir()
    traineddata = home / "eng.traineddata"
    if change == "intermediate-link":
        active = home / "active"
        active.mkdir()
        original = tmp_path / "original"
        alternate = tmp_path / "alternate"
        for directory, data in ((original, b"original"), (alternate, b"replacement")):
            directory.mkdir()
            (directory / traineddata.name).write_bytes(data)
        link = active / "current"
        link.symlink_to(original, target_is_directory=True)
        traineddata.symlink_to(link / traineddata.name)
    elif change == "ordinary-ancestor":
        active = home / "active"
        traineddata = active / "share" / "tessdata" / traineddata.name
        traineddata.parent.mkdir(parents=True)
        traineddata.write_bytes(b"original")
    else:
        traineddata.write_bytes(b"original")
    monkeypatch.setattr(evidence.shutil, "which", lambda _: "tesseract")
    monkeypatch.setattr(
        evidence,
        "run",
        lambda command: (
            f'List of available languages in "{traineddata.parent}" (1):\neng'
            if command[-1] == "--list-langs"
            else "tesseract test"
        ),
    )

    def change_language_data(paths, data_dir):
        data = data_dir / traineddata.name
        if change == "intermediate-link":
            saved = link.with_name("saved")
            link.rename(saved)
            link.symlink_to(alternate, target_is_directory=True)
            assert traineddata.read_bytes() == b"replacement"
            observed = data.read_text()
            saved.replace(link)
        elif change == "ordinary-ancestor":
            saved = active.with_name("saved")
            active.rename(saved)
            traineddata.parent.mkdir(parents=True)
            traineddata.write_bytes(b"replacement")
            observed = data.read_text()
            active.rename(active.with_name("alternate"))
            saved.rename(active)
        elif change == "unrelated":
            (home / "unrelated.txt").write_text("unrelated")
            observed = data.read_text()
        else:
            saved = traineddata.with_suffix(".original")
            traineddata.rename(saved)
            traineddata.write_bytes(b"replacement")
            observed = data.read_text()
            if change == "rollback":
                saved.replace(traineddata)
        return [{"text": observed, "words": [], "flags": []} for _ in paths]

    monkeypatch.setattr(bundles, "screen_text", change_language_data)
    if change == "replacement":
        with pytest.raises(EvidenceError, match="backend inputs changed"):
            bundles.prepare(source, bundle, None, "auto")
        assert manifest_of(bundle) == manifest
        assert len(list((bundle / "evidence").iterdir())) == 1
    else:
        assert not bundles.prepare(source, bundle, None, "auto")["cache_hit"]
        page = toon("inspect", bundle, "--from", 0, "--to", 12)
        assert page["events"]
        assert {event["ocr"] for event in page["events"]} == {"original"}
        assert bundles.prepare(source, bundle, None, "auto")["cache_hit"]


@pytest.mark.parametrize(
    "listing",
    [
        EvidenceError("tesseract failed"),
        "List of available languages (2):\neng\ndeu\n",
    ],
)
def test_unusable_tesseract_leaves_ocr_unavailable(monkeypatch, listing):
    def list_languages(command):
        if isinstance(listing, Exception):
            raise listing
        return listing

    monkeypatch.setattr(evidence.shutil, "which", lambda _: "tesseract")
    monkeypatch.setattr(evidence, "run", list_languages)
    settings, backends, inputs = evidence.backend_config({"streams": [{}]}, None, "auto")
    assert settings == {"asr": None, "ocr": None, "ocr_request": "auto"}
    assert backends == {"asr": None, "ocr": None}
    assert inputs == {}


def test_tesseract_language_file_reads_quoted_directory(tmp_path, monkeypatch):
    data = tmp_path / "tessdata"
    data.mkdir()
    (data / "eng.traineddata").write_bytes(b"data")
    monkeypatch.setattr(
        evidence,
        "run",
        lambda command: f'List of available languages in "{data}" (1):\neng',
    )
    assert evidence.tesseract_language_file() == data / "eng.traineddata"


def test_tesseract_literal_quote_preserves_word_boxes(tmp_path, monkeypatch):
    tsv = (
        "level\tleft\ttop\twidth\theight\tconf\ttext\n"
        '5\t1\t2\t30\t5\t97\t"Supervisor\n'
        '5\t33\t2\t27\t5\t92\tapproval"\n'
    )
    monkeypatch.setattr(evidence, "run", lambda command, *, env: tsv)
    result = evidence.recognize(tmp_path / "screen.png", tmp_path)
    assert result["text"] == '"Supervisor approval"'
    assert result["words"] == [
        {"text": '"Supervisor', "confidence": 97, "box": [1, 2, 30, 5]},
        {"text": 'approval"', "confidence": 92, "box": [33, 2, 27, 5]},
    ]


def test_audio_stream_offset(demo, tmp_path):
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


def test_audio_preserves_midstream_gap(demo, tmp_path):
    video, _ = demo
    gapped = tmp_path / "gapped.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video),
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=500:sample_rate=16000:duration=0.4",
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-af",
            "asetnsamples=n=320:p=0,asetpts=PTS+gte(N\\,3200)*0.08/TB",
            "-c:v",
            "copy",
            "-c:a",
            "pcm_s16le",
            str(gapped),
        ],
        check=True,
    )
    pcm = audio_pcm(gapped, probe(gapped), tmp_path / "pcm.wav")
    assert len(pcm) == 7680
    assert not np.any(pcm[3200:4480])
    assert np.any(pcm[2880:3200])
    assert np.any(pcm[4480:4800])
