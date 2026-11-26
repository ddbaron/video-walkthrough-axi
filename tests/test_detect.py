import av
import pytest
from demo_recording import FPS, screen
from PIL import Image, ImageDraw

from video_walkthrough_axi.detect import select_frames
from video_walkthrough_axi.media import Sample


def screen_sample(time, state):
    image = screen(state)
    if state == "smaller":
        image = image.resize((320, 180))
    return Sample(time, av.VideoFrame.from_image(image))


@pytest.mark.parametrize(
    ("ending", "reason"),
    [
        ("baseline", "reverted"),
        ("full-confirmation", "screen-change"),
        ("smaller", "resolution-change"),
        (None, None),
    ],
)
@pytest.mark.parametrize(("state", "fps"), [("single-frame", FPS), ("brief-warning", 10)])
def test_single_frame_change_survives_each_exit(state, fps, ending, reason):
    samples = [screen_sample(0, "baseline"), screen_sample(1 / fps, state)]
    if ending is not None:
        samples.append(screen_sample(2 / fps, ending))

    opening, changed, *following = select_frames(samples)
    assert opening.sample is samples[0]
    assert changed.sample is samples[1]
    assert changed.onset == 1 / fps
    assert changed.flags == ("single-frame",)
    assert [item.reason for item in following] == ([reason] if reason else [])
    if following:
        assert following[0].sample is samples[2]


@pytest.mark.parametrize("ending", ["baseline", "two-indicators", "smaller", None])
@pytest.mark.parametrize(("fps", "frames"), [(10, 2), (60, 4)])
def test_consecutive_localized_screens_merge_only_matching_frames(ending, fps, frames):
    samples = [screen_sample(0, "baseline")]
    samples.extend(screen_sample(i / fps, "approval-indicator") for i in range(1, frames + 1))
    if ending is not None:
        successor = screen_sample((frames + 1) / fps, ending)
        samples.append(successor)
        if ending == "two-indicators":
            samples.append(screen_sample((frames + 2) / fps, ending))

    _, changed, *following = select_frames(samples)
    assert changed.onset == 1 / fps
    assert changed.sample is samples[frames]
    assert changed.flags == ()
    if ending is None:
        assert following == []
    else:
        assert len(following) == 1
        assert following[0].onset == successor.time
        assert following[0].sample is samples[-1]


def test_color_only_warning_and_return():
    samples = [
        screen_sample(0, "ready-status"),
        screen_sample(0.1, "warning-status"),
        screen_sample(0.7, "warning-status"),
        screen_sample(0.8, "ready-status"),
        screen_sample(1.4, "ready-status"),
    ]
    assert [(item.sample.time, item.onset) for item in select_frames(samples)] == [
        (0, 0),
        (0.7, 0.1),
        (0.8, 0.8),
    ]


def test_adjacent_color_excursion_survives_merging():
    samples = []
    for index, shade in enumerate((100, 110, 94, 110)):
        image = screen("baseline")
        ImageDraw.Draw(image).rectangle((320, 180, 399, 239), fill=(shade, shade, shade))
        samples.append(Sample(index / 10, av.VideoFrame.from_image(image)))

    assert [screen.sample.time for screen in select_frames(samples)] == [0, 0.2, 0.3]


def test_warning_survives_full_screenshot_bucket():
    states = ("baseline", "dropdown", "brief-warning", "baseline")
    samples = [screen_sample(index / 10, state) for index, state in enumerate(states)]
    assert [screen.sample.time for screen in select_frames(samples)] == [0, 0.1, 0.2]


def test_held_warning_survives_transient_transitions():
    states = ("baseline", "dropdown", "confirmation", "brief-warning", "brief-warning")
    times = (0, 0.05, 0.1, 0.15, 0.95)
    samples = [screen_sample(time, state) for time, state in zip(times, states, strict=True)]
    selected = list(select_frames(samples))
    assert any(item.sample is samples[4] for item in selected)
    assert len(selected) <= 3


def test_alternating_screens_stay_within_screenshot_cap():
    samples = [
        screen_sample(index / 10, "dropdown" if index % 2 else "baseline") for index in range(21)
    ]
    counts = {}
    for selection in select_frames(samples):
        second = int(selection.sample.time)
        counts[second] = counts.get(second, 0) + 1
    assert max(counts.values()) <= 3


@pytest.mark.parametrize("returns", [False, True])
def test_gradual_color_change_reaches_held_screen(returns):
    shades = [100, 110, 120, 130, 130]
    if returns:
        shades.append(100)
    samples = []
    for index, shade in enumerate(shades):
        image = screen("baseline")
        ImageDraw.Draw(image).rectangle((320, 180, 399, 239), fill=(shade, shade, shade))
        samples.append(Sample(index / 10, av.VideoFrame.from_image(image)))

    assert any(selection.sample is samples[4] for selection in select_frames(samples))


def test_motion_onset_resets_after_held_screen():
    opening = Image.new("RGB", (64, 48), "white")
    samples = [Sample(0, av.VideoFrame.from_image(opening))]
    for tick, shade in enumerate(
        (0, 0, 40, 40, 80, 80, 120, 160, 160, 160, 160, 160, 160, 200, 200, 0, 0, 40, 40, 80),
        start=10,
    ):
        image = Image.new("RGB", (64, 48), (shade,) * 3)
        samples.append(Sample(tick / 10, av.VideoFrame.from_image(image)))
    selected = list(select_frames(samples))
    held = next(change for change in selected if change.reason == "settled-transition")
    assert held.onset == 1.7
    assert held.sample.time == 2.2
    motion = selected[-1]
    assert motion.reason == "extended-motion"
    assert motion.onset == 2.3
    assert motion.sample.time == 2.9


def test_variable_rate_hold_resets_motion_onset():
    times = (0, 1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3)
    shades = (255, 0, 40, 80, 120, 160, 180, 200, 220)
    samples = [
        Sample(time, av.VideoFrame.from_image(Image.new("RGB", (64, 48), (shade,) * 3)))
        for time, shade in zip(times, shades, strict=True)
    ]
    selections = list(select_frames(samples))
    assert [change.sample.time for change in selections] == [0, 1.3, 2.3]
    assert [change.reason for change in selections[1:]] == ["settled-transition"] * 2


def test_cursor_sized_change_and_single_frame_popup():
    baseline = screen("baseline")
    cursor = baseline.copy()
    ImageDraw.Draw(cursor).rectangle((10, 69, 23, 82), fill="#061526")
    images = [baseline, cursor, baseline, screen("single-frame"), baseline]
    samples = [
        Sample(index / 24, av.VideoFrame.from_image(image)) for index, image in enumerate(images)
    ]
    opening, popup, reverted = select_frames(samples)
    assert opening.sample is samples[0]
    assert popup.sample is samples[3]
    assert popup.flags == ("single-frame",)
    assert reverted.sample is samples[4]


@pytest.mark.parametrize("steps", (2, 5))
def test_fade_across_second_boundary_emits_settled_endpoint(steps):
    before, after = screen("baseline"), screen("brief-warning")
    opening = Sample(0, av.VideoFrame.from_image(before))
    samples = [opening]
    for index in range(1, steps + 1):
        blend = Image.blend(before, after, index / steps)
        samples.append(Sample(0.96 + 0.03 * (index - 1), av.VideoFrame.from_image(blend)))
    samples.append(Sample(1.2, av.VideoFrame.from_image(after)))

    first, settled = select_frames(samples)
    assert first.sample is opening
    assert settled.sample is samples[-1]
    assert settled.reason == "settled-transition"
    assert settled.onset == samples[-2].time


@pytest.mark.parametrize("start", (0.0, 0.93))
def test_single_frame_popup_survives_fading_in_and_out(start):
    before, popup = screen("baseline"), screen("brief-warning")
    half = Image.blend(before, popup, 0.5)
    samples = [
        Sample(start + index / 24, av.VideoFrame.from_image(image))
        for index, image in enumerate((before, half, popup, half, before))
    ]
    assert [change.sample for change in select_frames(samples)] == [samples[i] for i in (0, 2, 4)]
