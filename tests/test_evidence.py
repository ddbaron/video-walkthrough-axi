import subprocess

import numpy as np
import pytest
from cli_helpers import manifest_of, toon

from video_walkthrough_axi import bundle as bundles
from video_walkthrough_axi import evidence
from video_walkthrough_axi.evidence import audio_pcm
from video_walkthrough_axi.media import EvidenceError, probe


@pytest.mark.parametrize(
    "change", ["replacement", "rollback", "intermediate-link", "ordinary-ancestor"]
)
def test_prepare_rejects_ocr_data_change(demo, tmp_path, monkeypatch, change):
    source, _ = demo
    bundle = tmp_path / "bundle"
    initial = toon("prepare", source, "--out", bundle, "--ocr", "off")
    assert not initial["cache_hit"]
    manifest = manifest_of(bundle)
    traineddata = tmp_path / "eng.traineddata"
    if change == "intermediate-link":
        active = tmp_path / "active"
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
        active = tmp_path / "active"
        traineddata = active / "share" / "tessdata" / traineddata.name
        traineddata.parent.mkdir(parents=True)
        traineddata.write_bytes(b"original")
    else:
        traineddata.write_bytes(b"original")
    monkeypatch.setattr(evidence.shutil, "which", lambda _: "tesseract")
    monkeypatch.setattr(
        evidence,
        "run",
        lambda command: f'List of available languages in "{traineddata.parent}" (1):\neng'
        if command[-1] == "--list-langs"
        else "tesseract test",
    )

    def change_language_data(paths, *, enabled):
        assert enabled
        if change == "intermediate-link":
            saved = link.with_name("saved")
            link.rename(saved)
            link.symlink_to(alternate, target_is_directory=True)
            assert traineddata.read_bytes() == b"replacement"
            saved.replace(link)
        elif change == "ordinary-ancestor":
            saved = active.with_name("saved")
            active.rename(saved)
            traineddata.parent.mkdir(parents=True)
            traineddata.write_bytes(b"replacement")
            active.rename(active.with_name("alternate"))
            saved.rename(active)
        else:
            saved = traineddata.with_suffix(".original")
            traineddata.rename(saved)
            traineddata.write_bytes(b"replacement")
            if change == "rollback":
                saved.replace(traineddata)
        return [{"text": "", "words": [], "flags": []} for _ in paths]

    monkeypatch.setattr(bundles, "screen_text", change_language_data)
    with pytest.raises(EvidenceError, match="backend inputs changed"):
        bundles.prepare(source, bundle, None, "auto")
    assert manifest_of(bundle) == manifest
    assert len(list((bundle / "evidence").iterdir())) == 1


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
