import os
from unittest.mock import MagicMock
from app.extraction.video import extract_audio, extract_keyframes, _select_keyframe_timestamps

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures", "sample.mp4")


def _scene(seconds: float):
    start = MagicMock()
    start.get_seconds.return_value = seconds
    return (start, None)


def test_extract_audio_produces_wav(tmp_path):
    wav_path = extract_audio(FIXTURE, str(tmp_path))
    assert os.path.exists(wav_path)
    assert wav_path.endswith(".wav")


def test_extract_keyframes_produces_timestamped_images(tmp_path):
    frames = extract_keyframes(FIXTURE, str(tmp_path))
    assert len(frames) >= 1
    for f in frames:
        assert os.path.exists(f.image_path)
        assert isinstance(f.timestamp, float)


def test_select_keyframe_timestamps_collapses_scene_changes_closer_than_the_minimum_interval():
    # Regression guard: a choppy video previously generated one OCR+vision
    # call per scene change even when changes landed a second apart.
    scenes = [_scene(0.0), _scene(1.0), _scene(2.0), _scene(6.0), _scene(6.5)]
    timestamps = _select_keyframe_timestamps(scenes, duration=10.0)
    assert timestamps == [0.0, 6.0]


def test_select_keyframe_timestamps_caps_total_count_for_pathological_scene_counts():
    scenes = [_scene(float(i) * 5) for i in range(500)]  # 500 scenes, well-spaced
    timestamps = _select_keyframe_timestamps(scenes, duration=2500.0)
    assert len(timestamps) <= 120


def test_select_keyframe_timestamps_falls_back_to_start_when_no_scenes_detected():
    assert _select_keyframe_timestamps([], duration=30.0) == [0.0]
