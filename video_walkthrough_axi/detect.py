"""Select distinct screens from native pixels."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from itertools import groupby
from math import floor

import numpy as np

from .media import Sample

SCREENSHOTS_PER_SECOND = 3
MOTION_PAUSE_SECONDS = 0.5
BURST_GAP_SECONDS = 0.1
SETTINGS = {
    "sampling": "all-decoded-frames",
    "grid_columns": 16,
    "grid_rows": 12,
    "changed_rgb_channel": 15,
    "changed_global_fraction": 0.007,
    "changed_tile_fraction": 0.032,
    "cursor_fraction": 0.001,
    "cursor_min_pixels": 100,
    "cursor_max_extent": 0.06,
    "comparison": "adjacent-screen-and-last-emitted",
    "selection_priority": "held-before-transient",
    "screenshots_per_second": SCREENSHOTS_PER_SECOND,
    "motion_pause_seconds": MOTION_PAUSE_SECONDS,
    "burst_gap_seconds": BURST_GAP_SECONDS,
    "scroll_global_fraction": 0.15,
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
        if total >= SETTINGS["cursor_min_pixels"] and fraction < SETTINGS["cursor_fraction"]:
            ys = np.flatnonzero(changed.any(axis=1))
            xs = np.flatnonzero(changed.any(axis=0))
            height, width = changed.shape
            extent = max((ys[-1] - ys[0] + 1) / height, (xs[-1] - xs[0] + 1) / width)
            if extent <= SETTINGS["cursor_max_extent"]:
                return fraction, 0.0
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
    motion_start: float | None = None
    held: bool = False

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
    anchor = pixels
    start = last = first
    score = (0.0, 0.0)
    reason = "opening"
    motion_start = first.time
    starts_after_change = False
    for sample in iterator:
        incoming = sample.frame.to_ndarray(format="rgb24")
        difference = grid.difference(incoming, pixels)
        adjacent_change = distinct(difference)
        if not adjacent_change and anchor is not pixels:
            difference = grid.difference(incoming, anchor)
        if not adjacent_change and not distinct(difference):
            last, pixels = sample, incoming
            continue

        held = sample.time >= start.time + MOTION_PAUSE_SECONDS
        excursion = False
        if start is not first:
            excursion = (
                not held
                and starts_after_change
                and adjacent_change
                and previous_run.shape == incoming.shape == pixels.shape
                and not distinct(grid.difference(incoming, previous_run))
            )
            yield screen_selection(
                start,
                last,
                reason,
                score,
                excursion,
                motion_start,
                between_changes=starts_after_change and adjacent_change,
                held=held,
            )
        if start is first or excursion or held:
            motion_start = sample.time
        if incoming.shape != pixels.shape:
            reason = "resolution-change"
            grid = TileGrid(incoming.shape[:2])
            motion_start = sample.time
        elif previous_run is not None and not distinct(grid.difference(incoming, previous_run)):
            reason = "reverted"
        else:
            reason = "screen-change"
        previous_run, pixels = pixels, incoming
        anchor = incoming
        start = last = sample
        starts_after_change = adjacent_change
        score = difference

    if start is not first:
        yield screen_selection(
            start,
            last,
            reason,
            score,
            False,
            motion_start,
            held=last.time >= start.time + MOTION_PAUSE_SECONDS,
        )


def screen_selection(
    start: Sample,
    last: Sample,
    reason: str,
    score: tuple[float, float],
    excursion: bool,
    motion_start: float,
    *,
    between_changes: bool = False,
    held: bool,
) -> Selection:
    capture = start if reason in {"reverted", "resolution-change"} else last
    single = start is last
    flags = ("single-frame",) if single else ()
    return Selection(
        capture,
        reason,
        start.time,
        score,
        flags,
        excursion,
        between_changes and single,
        motion_start,
        held,
    )


def fade_step(base: Sample, middle: Sample, end: Sample) -> bool:
    """Recognize an intermediate blend by its changed channels' pixel values."""
    before, during, after = (
        sample.frame.to_ndarray(format="rgb24")[::3, ::3].astype(np.int16)
        for sample in (base, middle, end)
    )
    step = during - before
    target = after - before
    changed = np.abs(step) >= SETTINGS["changed_rgb_channel"]
    if not changed.any():
        return False
    # A discrete state that already contains final pixels is not a fade step.
    between = (step * np.sign(target) > 0) & (np.abs(step) + 4 < np.abs(target))
    return np.count_nonzero(changed & between) / np.count_nonzero(changed) >= 0.9


def collapse_bursts(screens: Iterable[Selection]) -> Iterator[Selection]:
    pending: list[Selection] = []
    previous = None
    base = None
    sustained = False
    for screen in screens:
        barrier = screen.reason in {"opening", "resolution-change"}
        continuous = (
            previous is not None
            and not previous.held
            and previous.reason not in {"opening", "resolution-change"}
            and screen.onset - previous.onset <= BURST_GAP_SECONDS + 1e-9
        )
        if pending and (barrier or not continuous):
            yield from settled_burst(pending, base)
            pending.clear()
        if not continuous:
            sustained = False
        if barrier or sustained:
            yield screen
        else:
            if not pending:
                base = previous.sample if previous is not None else None
            pending.append(screen)
            if screen.onset - pending[0].onset >= MOTION_PAUSE_SECONDS - 1e-9:
                # Keep the existing rate limit for sustained motion.
                yield from pending
                pending.clear()
                sustained = True
        previous = screen
    yield from settled_burst(pending, base)


def settled_burst(pending: list[Selection], base: Sample | None) -> Iterator[Selection]:
    if len(pending) < 2 or base is None:
        yield from pending
        return
    if short_scroll(pending, base):
        yield replace(pending[-1], reason="settled-transition", between_changes=False)
        return
    skipped = False
    for index, screen in enumerate(pending):
        if (
            not screen.excursion
            and index + 1 < len(pending)
            and fade_step(base, screen.sample, pending[index + 1].sample)
        ):
            skipped = True
            continue
        if skipped:
            screen = replace(screen, reason="settled-transition", between_changes=False)
        yield screen
        base = screen.sample
        skipped = False


def short_scroll(pending: list[Selection], base: Sample) -> bool:
    if len(pending) < 3 or any(
        screen.excursion or screen.score[0] < SETTINGS["scroll_global_fraction"]
        for screen in pending
    ):
        return False
    anchor = base.frame.to_ndarray(format="rgb24")
    grid = TileGrid(anchor.shape[:2])
    # A quick return to the preceding screen can be a temporary popup.
    return all(
        distinct(grid.difference(screen.sample.frame.to_ndarray(format="rgb24"), anchor))
        for screen in pending
    )


def rate_limited(screens: Iterable[Selection]) -> Iterator[Selection]:
    motion_run = None
    motion_start = None
    by_second = groupby(screens, key=lambda screen: floor(screen.sample.time))
    for _, bucket in by_second:
        screens = list(bucket)
        changes = [screen for screen in screens if screen.reason != "opening"]
        uninterrupted = len(changes) > SCREENSHOTS_PER_SECOND and all(
            "single-frame" in screen.flags for screen in changes
        )
        protected = []
        motion = []
        for index, screen in enumerate(screens):
            if (
                screen.reason in {"opening", "resolution-change", "settled-transition"}
                or screen.held
                or screen.excursion
                or (screen.between_changes and not uninterrupted)
            ):
                protected.append((index, screen))
            else:
                motion.append((index, screen))
                if screen.motion_start != motion_run:
                    motion_run = screen.motion_start
                    motion_start = screen.onset
        protected.sort(
            key=lambda item: (item[1].reason != "opening", not item[1].held, not item[1].excursion)
        )
        del protected[SCREENSHOTS_PER_SECOND:]
        remaining = SCREENSHOTS_PER_SECOND - len(protected)
        if len(motion) > remaining:
            if remaining > 0:
                protected.extend(motion[: remaining - 1])
                index, latest = motion[-1]
                protected.append(
                    (
                        index,
                        replace(
                            latest,
                            reason="extended-motion",
                            onset=latest.onset if motion_start is None else motion_start,
                            flags=(*latest.flags, "unsettled-screen"),
                        ),
                    )
                )
        else:
            protected.extend(motion)
        for _, screen in sorted(protected):
            yield screen


def select_frames(samples: Iterable[Sample]) -> Iterator[Selection]:
    screens = rate_limited(collapse_bursts(screen_changes(samples)))
    first = next(screens, None)
    if first is None:
        return
    yield first
    emitted = first.sample.frame.to_ndarray(format="rgb24")
    grid = TileGrid(emitted.shape[:2])
    for screen in screens:
        pixels = screen.sample.frame.to_ndarray(format="rgb24")
        score = grid.difference(pixels, emitted)
        if distinct(score) or (screen.excursion and not np.array_equal(pixels, emitted)):
            if pixels.shape != emitted.shape:
                grid = TileGrid(pixels.shape[:2])
            emitted = pixels
            yield replace(screen, score=score)
