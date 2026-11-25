"""Generate a small fictional settings workflow, with optional macOS narration."""

import argparse
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SIZE = (640, 360)
FPS = 24
SECONDS = 12
STATES = {
    "dropdown": (36, 42),
    "confirmation": (72, 75),
    "approval-indicator": (105, 108),
    "warning": (108, 144),
    "single-frame": (157, 158),
    "result": (192, FPS * SECONDS),
}
AMOUNT_CROP = (390, 105, 192, 56)
QUIET_APPROVAL_START = 4.6
QUIET_APPROVAL_END = 7.0


def screen(state: str) -> Image.Image:
    image = Image.new("RGB", SIZE, "#edf0f5")
    draw = ImageDraw.Draw(image)
    regular = ImageFont.load_default(size=17)
    bold = ImageFont.load_default(size=23)

    def words(x, y, text, color="#18263b"):
        draw.text((x, y), text, font=regular, fill=color)

    draw.rectangle((0, 0, 640, 54), fill="#203449")
    draw.text((24, 13), "Orbit Workshop", fill="#ffffff", font=bold)
    draw.rounded_rectangle((24, 81, 616, 322), 12, fill="#ffffff")
    draw.text((48, 105), "Reserve a workbench", fill="#18263b", font=bold)
    draw.rounded_rectangle((390, 105, 582, 161), 6, fill="#e6ecff")
    price = "$18.50" if state == "result" else "$24.50"
    draw.text((409, 121), f"Total {price}", fill="#18263b", font=bold)
    words(48, 156, "Bench type")
    draw.rounded_rectangle((48, 184, 260, 220), 5, outline="#3f5373", width=2)
    words(62, 191, "Standard" if state != "result" else "Compact")
    draw.polygon(((230, 198), (244, 198), (237, 205)), fill="#3f5373")
    draw.rounded_rectangle((452, 251, 582, 287), 5, fill="#235cca")
    words(473, 259, "Reserve", "#ffffff")
    if state == "dropdown":
        draw.rounded_rectangle((48, 222, 260, 298), 6, fill="#203449")
        words(62, 231, "Standard", "#ffffff")
        words(62, 263, "Compact", "#ffffff")
    elif state == "confirmation":
        draw.rounded_rectangle((325, 181, 582, 229), 6, fill="#d3f2db")
        draw.text((345, 191), "Selection saved", font=bold, fill="#102c20")
    elif state in {"approval-indicator", "two-indicators"}:
        draw.rectangle((492, 184, 500, 191), fill="#203449")
        if state == "two-indicators":
            draw.rectangle((532, 184, 540, 191), fill="#203449")
    elif state == "warning":
        draw.rounded_rectangle((44, 240, 420, 308), 6, fill="#ffe4a2")
        draw.text((61, 247), "Warning", font=bold, fill="#271301")
        draw.text((61, 275), "Supervisor approval required", font=bold, fill="#271301")
    elif state == "single-frame":
        draw.rounded_rectangle((482, 184, 582, 221), 6, fill="#203449")
        words(490, 193, "Price held", "#ffffff")
    return image


def make_recording(output: Path, narration: bool = False) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        silent = work / "silent.mp4" if narration else output
        frames = {state: screen(state).tobytes() for state in ("baseline", *STATES)}
        with subprocess.Popen(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{SIZE[0]}x{SIZE[1]}",
                "-r",
                str(FPS),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                "14",
                "-pix_fmt",
                "yuv420p",
                "-g",
                str(FPS),
                str(silent),
            ],
            stdin=subprocess.PIPE,
        ) as ffmpeg:
            for index in range(FPS * SECONDS):
                state = next(
                    (name for name, (a, b) in STATES.items() if a <= index < b), "baseline"
                )
                ffmpeg.stdin.write(frames[state])
            ffmpeg.stdin.close()
            if ffmpeg.wait():
                raise RuntimeError("FFmpeg could not encode the synthetic recording.")

        if narration:
            normal, quiet, final = (work / f"{name}.aiff" for name in ("normal", "quiet", "final"))
            for path, text in (
                (normal, "Choose a workbench. The total is twenty four dollars and fifty cents."),
                (quiet, "Supervisor approval is required."),
                (final, "The compact bench is eighteen dollars and fifty cents."),
            ):
                subprocess.run(["say", "-r", "175", "-o", str(path), text], check=True)
            soundtrack = work / "narration.m4a"
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-nostdin",
                    "-y",
                    "-i",
                    str(normal),
                    "-i",
                    str(quiet),
                    "-i",
                    str(final),
                    "-filter_complex",
                    f"[1:a]volume=-29dB,adelay={round(QUIET_APPROVAL_START * 1000)}[q];"
                    "[2:a]adelay=8050[f];"
                    f"[0:a][q][f]amix=inputs=3:normalize=0,apad=whole_dur={SECONDS}[a]",
                    "-map",
                    "[a]",
                    "-c:a",
                    "aac",
                    str(soundtrack),
                ],
                check=True,
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-nostdin",
                    "-y",
                    "-i",
                    str(silent),
                    "-i",
                    str(soundtrack),
                    "-map",
                    "0:v",
                    "-map",
                    "1:a",
                    "-c",
                    "copy",
                    str(output),
                ],
                check=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--narration", action="store_true", help="use the macOS say command")
    args = parser.parse_args()
    make_recording(args.output.resolve(), args.narration)
