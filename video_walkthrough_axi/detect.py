"""Select stable screens and short-lived states from native pixels."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from .media import Sample

SETTINGS = {
    "sampling": "all-decoded-frames",
    "grid_columns": 16,
    "grid_rows": 12,
    "changed_luma": 15,
    "settled_global": 0.007,
    "settled_tile": 0.085,
    "local_global_floor": 0.00025,
    "local_tile": 0.032,
    "stability_global": 0.0025,
    "stability_tile": 0.105,
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
        changed = delta >= SETTINGS["changed_luma"]
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
    return score[0] < SETTINGS["stability_global"] and score[1] < SETTINGS["stability_tile"]


def abrupt(score: tuple[float, float]) -> bool:
    return score[0] >= SETTINGS["impulse_global"] or score[1] >= SETTINGS["impulse_tile"]


def localized_impulse(score: tuple[float, float]) -> bool:
    global_delta, tile_delta = score
    return (
        SETTINGS["impulse_global"] <= global_delta <= SETTINGS["impulse_motion_ceiling"]
        and tile_delta >= SETTINGS["impulse_tile"]
    )


@dataclass
class Candidate:
    sample: Sample
    gray: np.ndarray
    score: tuple[float, float]
    last: float

    @property
    def confirmed(self) -> bool:
        return self.last - self.sample.time >= SETTINGS["short_dwell"] - 1e-8


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


def brief(candidate: Candidate, reason: str) -> Selection:
    flags = ("brief-state",)
    if not candidate.confirmed:
        flags += ("limited-dwell",)
    return Selection(
        candidate.sample,
        reason,
        candidate.sample.time,
        candidate.score,
        flags,
        candidate.last,
    )


def select_frames(samples: Iterable[Sample]) -> Iterator[Selection]:
    iterator = iter(samples)
    first = next(iterator, None)
    if first is None:
        return
    reference = first.frame.to_ndarray(format="gray")
    grid = TileGrid(reference.shape)
    yield Selection(first, "opening", first.time, (0, 0))

    candidate = None
    motion_since = None
    for sample in iterator:
        gray = sample.frame.to_ndarray(format="gray")
        if gray.shape != reference.shape:
            if candidate is not None and candidate.confirmed:
                yield brief(candidate, "short-transition")
            yield Selection(sample, "resolution-change", sample.time, (1, 1))
            reference, grid = gray, TileGrid(gray.shape)
            candidate, motion_since = None, None
            continue

        score = grid.difference(gray, reference)
        if not meaningful(score):
            if candidate is not None and (candidate.confirmed or abrupt(candidate.score)):
                yield brief(candidate, "short-return")
                score = grid.difference(gray, candidate.gray)
                yield Selection(sample, "reverted", sample.time, score)
                reference = gray
            candidate, motion_since = None, None
            continue

        if motion_since is None:
            motion_since = sample.time
        if candidate is None:
            candidate = Candidate(sample, gray, score, sample.time)
        elif similar(next_score := grid.difference(gray, candidate.gray)):
            candidate.last = sample.time
        else:
            if candidate.confirmed or (
                localized_impulse(candidate.score) and localized_impulse(next_score)
            ):
                yield brief(candidate, "short-transition")
                reference = candidate.gray
                score = next_score
                motion_since = sample.time
                if not meaningful(score):
                    candidate, motion_since = None, None
                    continue
            candidate = Candidate(sample, gray, score, sample.time)

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
        reference = gray
        candidate, motion_since = None, None

    if candidate is not None and candidate.confirmed:
        yield brief(candidate, "short-at-end")
    elif candidate is not None:
        yield Selection(
            candidate.sample,
            "end-of-video",
            candidate.sample.time,
            candidate.score,
            ("unconfirmed-screen",),
        )
