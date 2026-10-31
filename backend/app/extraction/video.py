import os
import subprocess
from collections import namedtuple
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

Keyframe = namedtuple("Keyframe", ["timestamp", "image_path"])

# Every keyframe costs one OCR pass plus one vision-caption API call. A long
# video with lots of quick cuts can trigger a scene "change" every second or
# two, most of them near-duplicates of their neighbor -- collapsing anything
# closer than this keeps genuinely distinct scenes while cutting redundant
# API calls. MAX_KEYFRAMES is a hard backstop so a pathological video (every
# frame flagged as a scene change) can't produce unbounded calls regardless.
MIN_KEYFRAME_INTERVAL_SECONDS = 3.0
MAX_KEYFRAMES = 120


def extract_audio(video_path: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    wav_path = os.path.join(out_dir, "audio.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "16000", wav_path],
        check=True, capture_output=True,
    )
    return wav_path


def _select_keyframe_timestamps(scenes: list, duration: float) -> list[float]:
    if not scenes:
        return [0.0]

    timestamps = []
    last_ts = None
    for start, _end in scenes:
        ts = start.get_seconds()
        if last_ts is None or ts - last_ts >= MIN_KEYFRAME_INTERVAL_SECONDS:
            timestamps.append(ts)
            last_ts = ts

    if len(timestamps) > MAX_KEYFRAMES:
        # Even spacing across the already-filtered list preserves coverage
        # across the whole timeline instead of just truncating to the start.
        step = len(timestamps) / MAX_KEYFRAMES
        timestamps = [timestamps[int(i * step)] for i in range(MAX_KEYFRAMES)]

    return timestamps


def extract_keyframes(video_path: str, out_dir: str) -> list[Keyframe]:
    os.makedirs(out_dir, exist_ok=True)
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=27.0))
    scene_manager.detect_scenes(video)
    scenes = scene_manager.get_scene_list()
    timestamps = _select_keyframe_timestamps(scenes, video.duration.get_seconds())

    frames = []
    for i, ts in enumerate(timestamps):
        image_path = os.path.join(out_dir, f"frame_{i:04d}.jpg")
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(ts), "-i", video_path, "-frames:v", "1", image_path],
            check=True, capture_output=True,
        )
        frames.append(Keyframe(timestamp=ts, image_path=image_path))
    return frames
