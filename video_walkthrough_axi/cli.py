"""Command-line interface for creating and reading evidence bundles."""

import argparse
import math
import os
import sys
from pathlib import Path

import av
import toon_format

from . import __version__
from .bundle import extract_frame, prepare
from .media import EvidenceError
from .views import inspect_window, walkthrough


def output(values: dict) -> None:
    print(toon_format.encode(values))


class EvidenceParser(argparse.ArgumentParser):
    def error(self, message):
        output({"error": message, "help": f"Run {self.prog} --help."})
        self.exit(2)


def parse_time(value: str) -> float:
    try:
        if ":" not in value:
            result = float(value)
        else:
            parts = value.split(":")
            if len(parts) not in (2, 3):
                raise ValueError
            leading = [int(p) for p in parts[:-1] if p.isascii() and p.isdigit()]
            seconds = float(parts[-1])
            if len(leading) != len(parts) - 1 or not 0 <= seconds < 60:
                raise ValueError
            if len(leading) == 2 and leading[1] >= 60:
                raise ValueError
            result = sum(x * 60**i for i, x in enumerate(reversed([*leading, seconds])))
        if result < 0 or not math.isfinite(result):
            raise ValueError
        return result or 0.0
    except (ValueError, OverflowError) as error:
        raise argparse.ArgumentTypeError("use non-negative seconds, MM:SS, or HH:MM:SS") from error


def parse_crop(value: str) -> tuple[int, int, int, int]:
    try:
        x, y, width, height = (int(v) for v in value.split(","))
        if min(x, y) < 0 or min(width, height) < 1:
            raise ValueError
        return x, y, width, height
    except ValueError as error:
        raise argparse.ArgumentTypeError("use X,Y,W,H with positive width and height") from error


def parse_limit(value: str) -> int:
    try:
        result = int(value)
        if not 1 <= result <= 100:
            raise ValueError
        return result
    except ValueError as error:
        raise argparse.ArgumentTypeError("use a result limit between 1 and 100") from error


def parser() -> EvidenceParser:
    main = EvidenceParser(
        prog="video-walkthrough-axi",
        description="Turn local recordings into walkthrough evidence.",
        allow_abbrev=False,
    )
    main.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = main.add_subparsers(dest="command", required=True)

    setup = commands.add_parser(
        "prepare",
        help="build or reuse an offline evidence bundle",
        allow_abbrev=False,
    )
    setup.add_argument("file", type=Path, metavar="FILE")
    setup.add_argument("--out", type=Path, required=True, metavar="DIR", help="bundle directory")
    setup.add_argument(
        "--model",
        type=Path,
        metavar="DIR",
        help="local MLX Whisper model; defaults to VIDEO_WALKTHROUGH_MODEL",
    )
    setup.add_argument("--language", metavar="CODE", help="speech language code; auto-detected")
    setup.add_argument(
        "--ocr",
        choices=("auto", "tesseract", "off"),
        default="auto",
        help="auto uses Tesseract when installed (default: auto)",
    )

    inspect = commands.add_parser(
        "inspect",
        help="page through a time window in a bundle",
        allow_abbrev=False,
    )
    inspect.add_argument("bundle", type=Path, metavar="BUNDLE")
    inspect.add_argument("--from", dest="begin", required=True, type=parse_time, metavar="TIME")
    inspect.add_argument("--to", dest="end", required=True, type=parse_time, metavar="TIME")
    inspect.add_argument("--limit", type=parse_limit, default=20, help="events per page (1–100)")
    inspect.add_argument("--cursor", help="continue using the previous next_cursor")

    frame = commands.add_parser(
        "frame",
        help="extract a native screenshot from the original source",
        allow_abbrev=False,
    )
    frame.add_argument("bundle", type=Path, metavar="BUNDLE")
    frame.add_argument("--at", required=True, type=parse_time, metavar="TIME")
    frame.add_argument("--crop", type=parse_crop, metavar="X,Y,W,H", help="source pixel rectangle")

    draft = commands.add_parser(
        "walkthrough",
        help="print a Markdown walkthrough draft",
        allow_abbrev=False,
    )
    draft.add_argument("bundle", type=Path, metavar="BUNDLE")
    return main


def main() -> int:
    cli = parser()
    args = cli.parse_args()
    mask = os.umask(0o077)
    try:
        match args.command:
            case "prepare":
                model = args.model
                if model is None and (model_dir := os.environ.get("VIDEO_WALKTHROUGH_MODEL")):
                    model = Path(model_dir)
                output(
                    prepare(
                        args.file.resolve(),
                        args.out.resolve(),
                        model.resolve() if model else None,
                        args.language,
                        args.ocr,
                    )
                )
            case "inspect":
                if args.end < args.begin:
                    raise EvidenceError("--to must be equal to or later than --from.")
                output(
                    inspect_window(
                        args.bundle.resolve(),
                        args.begin,
                        args.end,
                        args.limit,
                        args.cursor,
                    )
                )
            case "frame":
                output(extract_frame(args.bundle.resolve(), args.at, args.crop))
            case "walkthrough":
                print(walkthrough(args.bundle.resolve()), end="")
    except (EvidenceError, OSError, ValueError, KeyError, av.FFmpegError) as error:
        if isinstance(error, BrokenPipeError):
            return 1
        output({"error": str(error), "help": f"Run {cli.prog} {args.command} --help."})
        return 1
    finally:
        os.umask(mask)
    return 0


if __name__ == "__main__":
    sys.exit(main())
