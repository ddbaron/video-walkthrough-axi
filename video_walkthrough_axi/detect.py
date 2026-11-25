"""Select stable screens and short-lived states from native pixels."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from .media import Sample

SETTINGS = {
    "sampling": "all-decoded-frames",
    "grid_columns": 16,
    "grid_rows": 12,
    "changed_rgb_channel": 15,
    "settled_global": 0.007,
    "settled_tile": 0.085,
    "local_global_floor": 0.00025,
    "local_tile": 0.032,
    "stability_global": 0.0025,
    "impulse_global": 0.0035,
    "impulse_tile": 0.12,
    "impulse_motion_ceiling": 0.035,
    "short_dwell": 0.065,
    "settled_dwell": 0.55,
    "motion_timeout": 3.0,
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
        if fraction < SETTINGS["local_global_floor"]:
            return fraction, 0.0
        columns = np.add.reduceat(changed, self.columns, axis=1, dtype=np.int32)
        tiles = np.add.reduceat(columns, self.rows, axis=0, dtype=np.int32)
        return fraction, float((tiles / self.areas).max())


def strong(score: tuple[float, float]) -> bool:
    return score[0] >= SETTINGS["settled_global"] or score[1] >= SETTINGS["settled_tile"]


def meaningful(score: tuple[float, float]) -> bool:
    return strong(score) or (
        score[0] >= SETTINGS["local_global_floor"] and score[1] >= SETTINGS["local_tile"]
    )


def similar(score: tuple[float, float]) -> bool:
    return score[0] < SETTINGS["stability_global"] and not meaningful(score)


@dataclass
class Candidate:
    sample: Sample
    pixels: np.ndarray
    score: tuple[float, float]
    last: Sample


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


def ending_brief(
    candidate: Candidate | None, reason: str, until: float | None = None
) -> Selection | None:
    if candidate is None:
        return None
    observed_end = candidate.last.end if until is None else until
    global_delta, tile_delta = candidate.score
    localized = global_delta <= SETTINGS["impulse_motion_ceiling"]
    confirmed = observed_end - candidate.sample.time >= SETTINGS["short_dwell"] - 1e-8 and (
        localized or candidate.last.time > candidate.sample.time
    )
    impulse = localized and (
        global_delta >= SETTINGS["impulse_global"] or tile_delta >= SETTINGS["impulse_tile"]
    )
    if not (confirmed or impulse):
        if until is None:
            return Selection(
                candidate.sample,
                "end-of-video",
                candidate.sample.time,
                candidate.score,
                ("unconfirmed-screen",),
            )
        return None
    return Selection(
        candidate.sample,
        reason,
        candidate.sample.time,
        candidate.score,
        ("brief-state",) if confirmed else ("brief-state", "limited-dwell"),
        candidate.last.time,
    )


def select_frames(samples: Iterable[Sample]) -> Iterator[Selection]:
    iterator = iter(samples)
    first = next(iterator, None)
    if first is None:
        return
    reference = first.frame.to_ndarray(format="rgb24")
    grid = TileGrid(reference.shape[:2])
    yield Selection(first, "opening", first.time, (0, 0))

    candidate = None
    motion_since = None
    for sample in iterator:
        pixels = sample.frame.to_ndarray(format="rgb24")
        if pixels.shape != reference.shape:
            if selection := ending_brief(candidate, "short-transition", sample.time):
                yield selection
            yield Selection(sample, "resolution-change", sample.time, (1, 1))
            reference, grid = pixels, TileGrid(pixels.shape[:2])
            candidate, motion_since = None, None
            continue

        score = grid.difference(pixels, reference)
        if not meaningful(score):
            if candidate is not None and (
                selection := ending_brief(candidate, "short-return", sample.time)
            ):
                yield selection
                score = grid.difference(pixels, candidate.pixels)
                yield Selection(sample, "reverted", sample.time, score)
                reference = pixels
            candidate, motion_since = None, None
            continue

        if motion_since is None:
            motion_since = sample.time
        if candidate is None:
            candidate = Candidate(sample, pixels, score, sample)
        elif similar(next_score := grid.difference(pixels, candidate.pixels)):
            candidate.last = sample
        else:
            if selection := ending_brief(candidate, "short-transition", sample.time):
                yield selection
                reference = candidate.pixels
                score = next_score
                motion_since = sample.time
                if not meaningful(score):
                    candidate, motion_since = None, None
                    continue
            candidate = Candidate(sample, pixels, score, sample)

        if sample.time - candidate.sample.time >= SETTINGS["settled_dwell"] - 1e-8:
            reason = "settled" if strong(candidate.score) else "localized"
            yield Selection(
                sample,
                reason,
                candidate.sample.time,
                score,
                last_observed=sample.time,
            )
        elif sample.time - motion_since >= SETTINGS["motion_timeout"]:
            yield Selection(
                sample,
                "extended-motion",
                motion_since,
                score,
                ("unsettled-screen",),
            )
        else:
            continue
        reference = pixels
        candidate, motion_since = None, None

    if selection := ending_brief(candidate, "short-at-end"):
        yield selection
