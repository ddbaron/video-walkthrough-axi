"""Select distinct screens from native pixels."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from math import floor

import numpy as np

from .media import Sample

SCREENSHOTS_PER_SECOND = 3
SETTINGS = {
    "sampling": "all-decoded-frames",
    "grid_columns": 16,
    "grid_rows": 12,
    "changed_rgb_channel": 15,
    "changed_global_fraction": 0.007,
    "changed_tile_fraction": 0.032,
    "screenshots_per_second": SCREENSHOTS_PER_SECOND,
}


class TileGrid:
    def __init__(self, shape: tuple[int, int]):
        height, width = shape
        ys = np.linspace(0, height, min(SETTINGS["grid_rows"], height) + 1, dtype=int)
        xs = np.linspace(0, width, min(SETTINGS["grid_columns"], width) + 1, dtype=int)
        self.rows = ys[:-1]
        self.columns = xs[:-1]
        self.areas = np.outer(np.diff(ys), np.diff(xs))

    def difference(self, current: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
        if current.shape != reference.shape:
            return 1.0, 1.0
        delta = np.subtract(current, reference, dtype=np.int16)
        np.abs(delta, out=delta)
        changed = np.any(delta >= SETTINGS["changed_rgb_channel"], axis=2)
        total = int(np.count_nonzero(changed))
        fraction = total / changed.size
        columns = np.add.reduceat(changed, self.columns, axis=1, dtype=np.int32)
        tiles = np.add.reduceat(columns, self.rows, axis=0, dtype=np.int32)
        return fraction, float((tiles / self.areas).max())


def distinct(score: tuple[float, float]) -> bool:
    return (
        score[0] >= SETTINGS["changed_global_fraction"]
        or score[1] >= SETTINGS["changed_tile_fraction"]
    )


@dataclass
class Selection:
    sample: Sample
    reason: str
    onset: float
    score: tuple[float, float]
    flags: tuple[str, ...] = ()
    last_observed: float | None = None

    def record(self) -> dict:
        frame = self.sample.frame
        return {
            "time": self.sample.time,
            "pts": frame.pts,
            "time_base": str(frame.time_base),
            "width": frame.width,
            "height": frame.height,
            "change_start": self.onset,
            "last_observed": self.last_observed,
            "reason": self.reason,
            "global_delta": round(self.score[0], 6),
            "tile_delta": round(self.score[1], 6),
            "flags": list(self.flags),
        }


def screen_changes(samples: Iterable[Sample]) -> Iterator[Selection]:
    iterator = iter(samples)
    first = next(iterator, None)
    if first is None:
        return
    reference = first.frame.to_ndarray(format="rgb24")
    grid = TileGrid(reference.shape[:2])
    yield Selection(first, "opening", first.time, (0, 0))

    previous = None
    start = last = first
    score = (0.0, 0.0)
    reason = "opening"
    for sample in iterator:
        pixels = sample.frame.to_ndarray(format="rgb24")
        difference = grid.difference(pixels, reference)
        if not distinct(difference):
            last = sample
            continue

        if start is not first:
            yield screen_selection(start, last, reason, score)
        if pixels.shape != reference.shape:
            reason = "resolution-change"
            grid = TileGrid(pixels.shape[:2])
        elif previous is not None and not distinct(grid.difference(pixels, previous)):
            reason = "reverted"
        else:
            reason = "screen-change"
        previous, reference = reference, pixels
        start = last = sample
        score = difference

    if start is not first:
        yield screen_selection(start, last, reason, score)


def screen_selection(
    start: Sample, last: Sample, reason: str, score: tuple[float, float]
) -> Selection:
    capture = start if reason in {"reverted", "resolution-change"} else last
    flags = ("single-frame",) if start is last else ()
    return Selection(capture, reason, start.time, score, flags, last.time)


def select_frames(samples: Iterable[Sample]) -> Iterator[Selection]:
    second = None
    selected: list[Selection] = []
    for screen in screen_changes(samples):
        capture_second = floor(screen.sample.time)
        if capture_second != second:
            yield from selected
            selected = []
            second = capture_second
        if len(selected) < SCREENSHOTS_PER_SECOND:
            selected.append(screen)
        else:
            first_change = selected[1] if selected[0].reason == "opening" else selected[0]
            selected[-1] = replace(
                screen,
                reason="extended-motion",
                onset=first_change.onset,
                flags=(*screen.flags, "unsettled-screen"),
            )
    yield from selected
