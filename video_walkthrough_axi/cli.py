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
from .views import contact_sheet, inspect_window, overview, walkthrough


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


def parse_delta(value: str) -> float:
    try:
        result = float(value)
        if 0 <= result <= 1:
            return result or 0.0
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("use a changed-pixel fraction between 0 and 1")


def add_view_ocr(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--ocr",
        choices=("on", "off"),
        nargs="?",
        const="on",
        default="off",
        help="include stored OCR when on (default: off)",
    )


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
        help="local MLX Whisper model",
    )
    setup.add_argument(
        "--ocr",
        choices=("auto", "off"),
        default="off",
        help="auto uses Tesseract when installed (default: off)",
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
    inspect.add_argument("--cursor", help="continue using the same window and filters")
    inspect.add_argument("--kind", choices=("frame", "speech"), help="return only this event kind")
    inspect.add_argument(
        "--min-delta",
        type=parse_delta,
        metavar="FRACTION",
        default=0.0,
        help="minimum fraction of changed screen pixels for frame events (0–1)",
    )
    add_view_ocr(inspect)

    summary = commands.add_parser(
        "overview",
        help="summarize recording metadata, speech and frames",
        allow_abbrev=False,
    )
    summary.description = "Summarize the recording and group likely silence hallucinations."
    summary.add_argument("bundle", type=Path, metavar="BUNDLE")

    sheets = commands.add_parser(
        "contact-sheet",
        help="save labeled thumbnail grids under bundle extracts",
        allow_abbrev=False,
    )
    sheets.add_argument("bundle", type=Path, metavar="BUNDLE")
    sheets.add_argument(
        "--from", dest="begin", type=parse_time, metavar="TIME", help="first screen time to include"
    )
    sheets.add_argument(
        "--to", dest="end", type=parse_time, metavar="TIME", help="last screen time to include"
    )

    frame = commands.add_parser(
        "frame",
        help="extract a native screenshot from the original source",
        allow_abbrev=False,
    )
    frame.description = "Save a timestamped PNG with a path relative to the bundle."
    frame.add_argument("bundle", type=Path, metavar="BUNDLE")
    frame.add_argument("--at", required=True, type=parse_time, metavar="TIME")
    frame.add_argument("--crop", type=parse_crop, metavar="X,Y,W,H", help="source pixel rectangle")

    draft = commands.add_parser(
        "walkthrough",
        help="print a Markdown walkthrough draft",
        allow_abbrev=False,
    )
    draft.add_argument("bundle", type=Path, metavar="BUNDLE")
    add_view_ocr(draft)
    return main


def main() -> int:
    cli = parser()
    args = cli.parse_args()
    mask = os.umask(0o077)
    try:
        if (
            args.command in {"inspect", "contact-sheet"}
            and args.begin is not None
            and args.end is not None
            and args.end < args.begin
        ):
            raise EvidenceError("--to must be equal to or later than --from.")
        match args.command:
            case "prepare":
                output(
                    prepare(
                        args.file.resolve(),
                        args.out.resolve(),
                        args.model.resolve() if args.model is not None else None,
                        args.ocr,
                    )
                )
            case "inspect":
                output(
                    inspect_window(
                        args.bundle.resolve(),
                        args.begin,
                        args.end,
                        args.limit,
                        args.cursor,
                        kind=args.kind,
                        min_delta=args.min_delta,
                        include_ocr=args.ocr == "on",
                    )
                )
            case "overview":
                output(overview(args.bundle.resolve()))
            case "contact-sheet":
                output(contact_sheet(args.bundle.resolve(), args.begin, args.end))
            case "frame":
                output(extract_frame(args.bundle.resolve(), args.at, args.crop))
            case "walkthrough":
                print(walkthrough(args.bundle.resolve(), include_ocr=args.ocr == "on"), end="")
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
