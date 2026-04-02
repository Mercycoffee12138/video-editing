from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TrimConfig:
    head_trim_seconds: float = 3.0
    tail_trim_seconds: float = 3.0
    minimum_remaining_seconds: float = 5.0


@dataclass(frozen=True)
class MotionConfig:
    analysis_fps: int = 6
    analysis_width: int = 192
    analysis_height: int = 108
    smoothing_seconds: float = 1.5
    threshold_quantile: float = 0.86
    threshold_floor: float = 0.045
    min_segment_seconds: float = 1.2
    merge_gap_seconds: float = 0.7


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 22050
    frame_length: int = 2048
    hop_length: int = 1024
    min_peak_distance_seconds: float = 4.0
    top_highlights: int = 12
    peak_threshold_quantile: float = 0.9


@dataclass(frozen=True)
class MatchConfig:
    selected_music_filename: str | None = "011.MP3"
    use_full_track_duration: bool = True
    highlight_cluster_window_seconds: float = 60.0
    max_highlights_per_track: int = 10
    intro_padding_seconds: float = 4.0
    outro_padding_seconds: float = 6.0
    min_clip_seconds: float = 0.8
    max_clip_seconds: float = 4.8
    source_reuse_penalty: float = 0.25


@dataclass(frozen=True)
class RenderConfig:
    output_width: int = 1920
    output_height: int = 1080
    output_fps: int = 24
    video_crf: int = 18
    video_preset: str = "veryfast"
    audio_bitrate: str = "192k"


@dataclass(frozen=True)
class StageWindow:
    stage_name: str
    label: str
    start_percent: float
    end_percent: float


@dataclass(frozen=True)
class PipelinePaths:
    project_root: Path
    source_root: Path
    video_source_dir: Path
    music_source_dir: Path
    build_dir: Path
    stage_01_trim_dir: Path
    stage_05_clip_dir: Path
    stage_05_temp_dir: Path


@dataclass(frozen=True)
class PipelineConfig:
    paths: PipelinePaths
    trim: TrimConfig = field(default_factory=TrimConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    match: MatchConfig = field(default_factory=MatchConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    stage_windows: tuple[StageWindow, ...] = (
        StageWindow("stage_01_trim_videos", "Trim Videos", 0.0, 20.0),
        StageWindow("stage_02_detect_fight_segments", "Detect Fight Segments", 20.0, 45.0),
        StageWindow("stage_03_detect_music_highlights", "Detect Music Highlights", 45.0, 65.0),
        StageWindow("stage_04_match_segments", "Match Segments", 65.0, 80.0),
        StageWindow("stage_05_render_final_video", "Render Final Video", 80.0, 100.0),
    )


def build_default_config(project_root: Path) -> PipelineConfig:
    source_root = project_root / "source"
    build_dir = project_root / "build"

    paths = PipelinePaths(
        project_root=project_root,
        source_root=source_root,
        video_source_dir=source_root / "video",
        music_source_dir=source_root / "music",
        build_dir=build_dir,
        stage_01_trim_dir=build_dir / "stage_01_trimmed_videos",
        stage_05_clip_dir=build_dir / "stage_05_render_clips",
        stage_05_temp_dir=build_dir / "stage_05_temp",
    )
    return PipelineConfig(paths=paths)
