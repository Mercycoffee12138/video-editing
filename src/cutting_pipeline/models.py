from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TrimmedVideoRecord:
    source_path: str
    trimmed_path: str
    original_duration: float
    trim_start: float
    trim_end: float
    trimmed_duration: float


@dataclass(frozen=True)
class FightSegmentRecord:
    source_path: str
    trimmed_path: str
    video_duration: float
    start: float
    end: float
    peak_time: float
    mean_motion: float
    peak_motion: float
    score: float


@dataclass(frozen=True)
class MusicHighlightRecord:
    time: float
    score: float
    energy: float
    accent: float


@dataclass(frozen=True)
class MusicTrackRecord:
    music_path: str
    duration: float
    highlights: list[MusicHighlightRecord]


@dataclass(frozen=True)
class MatchedClipRecord:
    order: int
    source_path: str
    trimmed_path: str
    clip_start: float
    clip_end: float
    duration: float
    segment_score: float


@dataclass(frozen=True)
class MatchPlanRecord:
    music_path: str
    audio_excerpt_start: float
    audio_excerpt_end: float
    output_duration: float
    selected_highlights: list[MusicHighlightRecord]
    timeline_durations: list[float]
    clips: list[MatchedClipRecord]
    plan_score: float


def to_dict(instance: Any) -> Any:
    return asdict(instance)
