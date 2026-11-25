import json
import subprocess
import sys

import toon_format


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
