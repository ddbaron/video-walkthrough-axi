# video-walkthrough-axi

Turn a software recording into a timestamped transcript, native screenshots, a merged timeline, and a draft UX walkthrough. The [agent skill](skills/video-walkthrough/SKILL.md) guides any harness through reviewing that evidence and writing a finished workflow.

## Install

Use Python 3.11 or later on macOS or Linux. Install FFmpeg and FFprobe and put them on `PATH`. For recordings with audio, use Apple Silicon and install the `mlx` extra. Tesseract with English language data is optional.

```sh
git clone https://github.com/ddbaron/video-walkthrough-axi.git
cd video-walkthrough-axi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[mlx]'
```

Use `python -m pip install .` to prepare video-only recordings or read existing bundles without MLX. On macOS, Homebrew installs the media tools with `brew install ffmpeg tesseract`.

For audio, download a local MLX Whisper model once. The following example uses a quantized large-v3-turbo model.

```sh
hf download mlx-community/whisper-large-v3-turbo-q4 config.json weights.npz \
  --local-dir "$HOME/.local/share/video-walkthrough-axi/whisper"
export VIDEO_WALKTHROUGH_MODEL="$HOME/.local/share/video-walkthrough-axi/whisper"
```

`hf` is installed with the `mlx` extra. An existing MLX Whisper directory with a `config.json` and `weights.safetensors` or `weights.npz` also works. Pass its local path with `--model DIR` or the environment variable above.

## Commands

```sh
video-walkthrough-axi prepare recording.mp4 --out ./bundle
video-walkthrough-axi inspect ./bundle --from 00:10 --to 00:25 --limit 10
video-walkthrough-axi frame ./bundle --at 00:15.250 --crop 400,160,240,90
video-walkthrough-axi walkthrough ./bundle > ./bundle/walkthrough.md
```

- **`prepare FILE --out DIR`** scans decoded frames at native resolution, combining a tile and dwell detector with a pass for short screen states. It builds a private evidence bundle, adds word-timed speech when an audio stream is present, and publishes the manifest after validating the artifacts. `--language en` selects English speech when appropriate. `--ocr auto` uses Tesseract if available. Use `--ocr tesseract` to require it or `--ocr off` to skip it. An existing bundle can reuse evidence keyed by source SHA-256, settings, and backend and model versions.
- **`inspect BUNDLE --from TIME --to TIME`** returns events in recording order with the preceding screenshot, where available. It includes separate speech and OCR previews. Follow `next_cursor` with `--cursor` and the same window. The default limit is 20 events, with a maximum of 100 per page. Previews are capped at 500 characters per evidence field. Open the recorded timeline path for full text and OCR word boxes.
- **`frame BUNDLE --at TIME`** checks the source SHA-256 and saves a native PNG under `BUNDLE/extracts/`. The result includes the actual presentation timestamp, source PTS and time base, and whether the chosen frame was exact or nearest. For `--crop X,Y,W,H`, use source pixel coordinates from the upper left before any player rotation or scaling.
- **`walkthrough BUNDLE`** prints ordered Markdown steps with screenshot links, speech, and on-screen text. Save the Markdown in the bundle directory so its relative image links resolve.

Times accept seconds, `MM:SS`, or `HH:MM:SS`, including fractional seconds. Commands accept `--help`. Evidence summaries and window results are TOON. The draft is Markdown.

## Bundle and privacy

Once dependencies and the model are installed, all four commands run offline without credentials. The tool loads the specified model from disk and opens the recording with local file protocols. OCR runs locally. The original video stays at its source path.

`manifest.json` identifies schema version 1, the source path and SHA-256, stream offsets, backend versions, model fingerprints, settings, and selection reasons. Referenced generations under `evidence/` contain the native PNGs, `transcript.json`, and `timeline.json`. The transcript stores words and their timestamps. The timeline keeps speech and frame events distinct and includes OCR words, confidence scores, and pixel boxes when available. Bundle times are relative to the recording origin, with that origin and the selected streams documented in the manifest.

Bundles, terminal output, crops and derived drafts may contain private material. Keep them with the recording and use a private destination when redirecting CLI output. Prior generations remain in the bundle so earlier screenshot links and cache entries remain usable.

Detection depends on visible pixel changes, so a brief cursor or scroll intermediate may be selected and subtle changes may be missed. A `limited-dwell` flag marks especially brief evidence. Speech and OCR flags mark items to review, including quiet audio and low-confidence words.

## Acceptance checks

With FFmpeg on `PATH`, install the development tools with `python -m pip install -e '.[dev]'` and run `./acceptance.sh`. The tests generate a fictional software demo with a dropdown, a brief confirmation, a warning, and dollar amounts. OCR checks run when Tesseract is installed. To include the macOS `say` narration and quiet speech check, install the `mlx` extra and set `VIDEO_WALKTHROUGH_TEST_MODEL` to a local model directory before running the checks.

Generate the narrated fixture separately with `python tests/demo_recording.py /tmp/workshop-demo.mp4 --narration`. The generator can also produce a silent fixture when invoked without `--narration`.
