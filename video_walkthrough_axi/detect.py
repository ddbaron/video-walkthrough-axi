"""Select distinct screens from native pixels."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from itertools import groupby
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
    "comparison": "adjacent-frames",
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
    excursion: bool = False
    between_changes: bool = False

    def record(self) -> dict:
        frame = self.sample.frame
        return {
            "time": self.sample.time,
            "pts": frame.pts,
            "time_base": str(frame.time_base),
            "width": frame.width,
            "height": frame.height,
            "change_start": self.onset,
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
    pixels = first.frame.to_ndarray(format="rgb24")
    grid = TileGrid(pixels.shape[:2])
    yield Selection(first, "opening", first.time, (0, 0))

    previous_run = None
    start = last = first
    score = (0.0, 0.0)
    reason = "opening"
    for sample in iterator:
        incoming = sample.frame.to_ndarray(format="rgb24")
        difference = grid.difference(incoming, pixels)
        if not distinct(difference):
            last, pixels = sample, incoming
            continue

        if start is not first:
            excursion = (
                start is last
                and previous_run.shape == incoming.shape == pixels.shape
                and not distinct(grid.difference(incoming, previous_run))
            )
            yield screen_selection(start, last, reason, score, excursion, has_successor=True)
        if incoming.shape != pixels.shape:
            reason = "resolution-change"
            grid = TileGrid(incoming.shape[:2])
        elif previous_run is not None and not distinct(grid.difference(incoming, previous_run)):
            reason = "reverted"
        else:
            reason = "screen-change"
        previous_run, pixels = pixels, incoming
        start = last = sample
        score = difference

    if start is not first:
        yield screen_selection(start, last, reason, score, False, has_successor=False)


def screen_selection(
    start: Sample,
    last: Sample,
    reason: str,
    score: tuple[float, float],
    excursion: bool,
    *,
    has_successor: bool,
) -> Selection:
    capture = start if reason in {"reverted", "resolution-change"} else last
    single = start is last
    flags = ("single-frame",) if single else ()
    return Selection(capture, reason, start.time, score, flags, excursion, has_successor and single)


def select_frames(samples: Iterable[Sample]) -> Iterator[Selection]:
    previous_second = None
    motion_start = None
    by_second = groupby(screen_changes(samples), key=lambda screen: floor(screen.sample.time))
    for second, bucket in by_second:
        screens = list(bucket)
        changes = [screen for screen in screens if screen.reason != "opening"]
        uninterrupted = len(changes) > SCREENSHOTS_PER_SECOND and all(
            "single-frame" in screen.flags for screen in changes
        )
        protected = []
        motion = []
        for index, screen in enumerate(screens):
            if (
                screen.reason in {"opening", "resolution-change"}
                or screen.excursion
                or (screen.between_changes and not uninterrupted)
            ):
                protected.append((index, screen))
            else:
                motion.append((index, screen))
        remaining = SCREENSHOTS_PER_SECOND - len(protected)
        if len(motion) > max(remaining, 0):
            if motion_start is None or previous_second != second - 1:
                motion_start = changes[0].onset
            if remaining > 0:
                protected.extend(motion[: remaining - 1])
                index, latest = motion[-1]
                protected.append(
                    (
                        index,
                        replace(
                            latest,
                            reason="extended-motion",
                            onset=motion_start,
                            flags=(*latest.flags, "unsettled-screen"),
                        ),
                    )
                )
        else:
            protected.extend(motion)
            motion_start = None
        for _, screen in sorted(protected):
            yield screen
        previous_second = second
