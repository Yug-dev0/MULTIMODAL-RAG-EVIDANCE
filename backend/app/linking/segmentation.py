from dataclasses import dataclass
import uuid
from app.linking.scoring_utils import cosine_similarity

PAUSE_NORM_SECONDS = 2.0
BOUNDARY_THRESHOLD = 0.5


@dataclass
class SemanticWindow:
    start: float
    end: float
    evidence_unit_ids: list[uuid.UUID]


def _boundary_strength(prev, curr, scene_timestamps: list[float]) -> float:
    similarity_drop = max(0.0, 1.0 - cosine_similarity(prev.embedding, curr.embedding))
    speaker_change = 1.0 if prev.metadata_json.get("speaker") != curr.metadata_json.get("speaker") else 0.0
    gap = max(0.0, curr.timestamp_start - prev.timestamp_end)
    pause_norm = min(1.0, gap / PAUSE_NORM_SECONDS)
    scene_change = 1.0 if any(prev.timestamp_end <= t <= curr.timestamp_start for t in scene_timestamps) else 0.0
    return 0.40 * similarity_drop + 0.25 * speaker_change + 0.20 * scene_change + 0.15 * pause_norm


def segment_into_windows(speech_units: list, scene_timestamps: list[float] | None = None) -> list[SemanticWindow]:
    scene_timestamps = scene_timestamps or []
    ordered = sorted(speech_units, key=lambda u: u.timestamp_start)
    if not ordered:
        return []
    windows: list[SemanticWindow] = []
    current_ids = [ordered[0].id]
    current_start = ordered[0].timestamp_start
    current_end = ordered[0].timestamp_end
    for prev, curr in zip(ordered, ordered[1:]):
        if _boundary_strength(prev, curr, scene_timestamps) >= BOUNDARY_THRESHOLD:
            windows.append(SemanticWindow(current_start, current_end, current_ids))
            current_ids = [curr.id]
            current_start = curr.timestamp_start
        else:
            current_ids.append(curr.id)
        current_end = curr.timestamp_end
    windows.append(SemanticWindow(current_start, current_end, current_ids))
    return windows
