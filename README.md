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
```

`hf` is installed with the `mlx` extra. An existing MLX Whisper directory with a `config.json` and `weights.safetensors` or `weights.npz` also works. Pass its local path with `--model DIR`.

## Commands

```sh
bundle="./bundle"
video-walkthrough-axi prepare ./recording.mp4 --out "$bundle" \
  --model "$HOME/.local/share/video-walkthrough-axi/whisper"
video-walkthrough-axi overview "$bundle"
video-walkthrough-axi inspect "$bundle" --from 00:10 --to 00:25 --kind frame --min-delta 0.005
video-walkthrough-axi contact-sheet "$bundle" --from 00:10 --to 00:25
video-walkthrough-axi frame "$bundle" --at 00:15.250 --crop 400,160,240,90
video-walkthrough-axi walkthrough "$bundle" > "$bundle/walkthrough.md"
```

- **`prepare FILE --out DIR`** scans decoded frames at native resolution and merges near-identical screens using a color difference threshold. It builds a private evidence bundle, detects the speech language automatically when an audio stream is present, and publishes the manifest after validating the artifacts. OCR defaults to off. Pass `--ocr auto` to run Tesseract when available. An existing bundle can reuse evidence keyed by source SHA-256, settings, and backend and model versions.
- **`overview BUNDLE`** returns recording metadata, flags, speech previews, and the full frame index. Consecutive speech segments flagged for both quiet audio and low confidence appear as a single entry with a time range and a count. The linked transcript retains the original speech evidence.
- **`inspect BUNDLE --from TIME --to TIME`** returns events in recording order with the preceding screenshot, where available. Use `--kind frame` or `--kind speech` to filter events and `--min-delta FRACTION` to filter frames by the fraction of screen pixels that changed. Frame events enter the window at `change_start` and report the actual PNG timestamp as `time`. For extended-motion frames, `change_start` records when the motion began, and `time` determines window placement, screen context, and speech attribution. Follow `next_cursor` with `--cursor` and the same window and filters. The default limit is 20 events, with a maximum of 100 per page. Speech previews are capped at 500 characters. Pass `--ocr on` to include previews of stored OCR, and open the timeline for full text and word boxes.
- **`contact-sheet BUNDLE`** saves PNG grids of up to 20 thumbnails each under `BUNDLE/extracts/`. Labels show each screenshot’s position in the overview frame index and its actual timestamp. Use `--from TIME` and `--to TIME` to narrow the selection. Open the native screenshots to verify text and values.
- **`frame BUNDLE --at TIME`** checks the source SHA-256 and saves a native PNG under `BUNDLE/extracts/`. The result gives a bundle-relative path named for the actual timestamp, with distinct names for different crops. It includes the actual presentation timestamp, source PTS and time base, and whether the chosen frame was exact or nearest. For `--crop X,Y,W,H`, use source pixel coordinates from the upper left before any player rotation or scaling.
- **`walkthrough BUNDLE`** prints ordered Markdown steps with screenshot links and speech. Pass `--ocr on` to include stored on-screen text. Save the Markdown in the bundle directory so its relative image links resolve.

Times accept seconds, `MM:SS`, or `HH:MM:SS`, including fractional seconds. Commands accept `--help`. Evidence summaries and window results are TOON. The draft is Markdown.

## Bundle and privacy

Once dependencies and the model are installed, all commands run offline without credentials. The tool loads the specified model from disk and opens the recording with local file protocols. OCR runs locally. The original video stays at its source path.

Prepare and frame use a private directory under the system temporary directory, outside the evidence bundle, to read a copy of the recording that matches the verified hash. Allow enough space there for the copy while either command runs.

`manifest.json` is a formatted schema version 2 summary with the source path and SHA-256, stream offsets, backend versions, model fingerprints, settings, counts, and artifact references. Referenced generations under `evidence/` contain the native PNGs, `transcript.json`, and `timeline.json`. The transcript stores words and their timestamps. The timeline contains frame metadata and selection reasons alongside speech events, and includes OCR words, confidence scores, and pixel boxes when available. The CLI also reads existing schema version 1 bundles. Bundle times are relative to the recording origin, with that origin and the selected streams documented in the manifest.

Bundles, terminal output, crops and derived drafts may contain private material. Keep them with the recording and use a private destination when redirecting CLI output. Prior generations remain in the bundle so earlier screenshot links and cache entries remain usable.

Distinct screens can be captured even when visible for one frame. The `single-frame` flag identifies those screenshots. The detector skips small, localized cursor-sized changes and compares retained candidates against the last emitted screenshot. It condenses short fades and bursts of large visual changes to the settled endpoint, with priority for held screens and brief excursions that return to the previous state. Sustained motion remains capped at three screenshots per recording second. When sustained motion is thinned, the latest retained motion screenshot carries the `extended-motion` reason and an `unsettled-screen` flag. Its change start follows the observed motion across seconds and resets after a held screen. Speech and OCR flags mark items to review, including quiet audio and low-confidence words.

## Checks

Run these commands from the repository root. Put FFmpeg and FFprobe on `PATH` before running pytest. The [GitHub Actions workflow](.github/workflows/checks.yml) lists the CI runners and the pinned Ruff version.

```sh
python -m pip install -e '.[dev]'
python -m ruff check .
python -m pytest
python scripts/check_file_size.py
```

See the [Ruff configuration](pyproject.toml) for the complexity limit and the [size checker](scripts/check_file_size.py) for how it counts code lines in tracked Python files that are present in the working tree.

The tests generate a fictional software demo with a dropdown, a brief confirmation, a warning, and dollar amounts. OCR checks run when Tesseract and English language data are installed. To include the macOS `say` narration and quiet speech check, install the `mlx` extra and set `VIDEO_WALKTHROUGH_TEST_MODEL` to a local model directory before running the checks.
