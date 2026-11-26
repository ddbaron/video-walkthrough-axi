import pytest
from cli_helpers import toon
from demo_recording import make_recording


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    work = tmp_path_factory.mktemp("demo")
    source = work / "synthetic demo.mp4"
    bundle = work / "bundle"
    make_recording(source)
    summary = toon("prepare", source, "--out", bundle)
    assert summary["cache_hit"] is False
    assert summary["ocr_frames"] == 0
    return source, bundle
