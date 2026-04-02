from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .ffmpeg_tools import iter_gray_frames
from .json_io import write_json
from .models import FightSegmentRecord
from .progress import StageReporter


def _moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1 or values.size == 0:
        return values
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    return np.convolve(values, kernel, mode="same")


def _merge_segments(
    segments: list[tuple[int, int]],
    scores: np.ndarray,
    fps: int,
    merge_gap_seconds: float,
) -> list[tuple[int, int]]:
    if not segments:
        return []

    merged: list[tuple[int, int]] = [segments[0]]
    for start_idx, end_idx in segments[1:]:
        previous_start, previous_end = merged[-1]
        gap_seconds = (start_idx - previous_end - 1) / fps
        if gap_seconds <= merge_gap_seconds:
            merged[-1] = (previous_start, end_idx)
        else:
            merged.append((start_idx, end_idx))
    return merged


def _relative(path: Path, project_root: Path) -> str:
    return str(path.relative_to(project_root))


def _detect_segments(trimmed_path: Path, duration: float, config: PipelineConfig) -> tuple[list[FightSegmentRecord], dict]:
    motion_values: list[float] = []
    previous_frame: np.ndarray | None = None

    for frame in iter_gray_frames(
        trimmed_path,
        fps=config.motion.analysis_fps,
        width=config.motion.analysis_width,
        height=config.motion.analysis_height,
    ):
        current_frame = frame.astype(np.int16)
        if previous_frame is not None:
            delta = np.abs(current_frame - previous_frame)
            motion_values.append(float(delta.mean() / 255.0))
        previous_frame = current_frame

    if not motion_values:
        return [], {"threshold": 0.0, "frame_diffs": 0}

    raw_scores = np.asarray(motion_values, dtype=np.float32)
    smoothed_scores = _moving_average(
        raw_scores,
        max(1, int(round(config.motion.smoothing_seconds * config.motion.analysis_fps))),
    )
    threshold = max(
        float(np.quantile(smoothed_scores, config.motion.threshold_quantile)),
        float(smoothed_scores.mean() + (smoothed_scores.std() * 0.75)),
        config.motion.threshold_floor,
    )

    active_mask = smoothed_scores >= threshold
    segments: list[tuple[int, int]] = []
    active_start: int | None = None
    for index, is_active in enumerate(active_mask):
        if is_active and active_start is None:
            active_start = index
        elif not is_active and active_start is not None:
            segments.append((active_start, index - 1))
            active_start = None
    if active_start is not None:
        segments.append((active_start, len(active_mask) - 1))

    merged_segments = _merge_segments(
        segments,
        smoothed_scores,
        fps=config.motion.analysis_fps,
        merge_gap_seconds=config.motion.merge_gap_seconds,
    )

    fight_segments: list[FightSegmentRecord] = []
    for start_idx, end_idx in merged_segments:
        start_time = start_idx / config.motion.analysis_fps
        end_time = (end_idx + 1) / config.motion.analysis_fps
        segment_duration = end_time - start_time
        if segment_duration < config.motion.min_segment_seconds:
            continue

        segment_scores = smoothed_scores[start_idx : end_idx + 1]
        peak_offset = int(segment_scores.argmax())
        peak_time = (start_idx + peak_offset) / config.motion.analysis_fps
        mean_motion = float(segment_scores.mean())
        peak_motion = float(segment_scores.max())
        score = float(mean_motion * segment_duration * (1.0 + peak_motion))

        fight_segments.append(
            FightSegmentRecord(
                source_path="",
                trimmed_path="",
                video_duration=round(duration, 3),
                start=round(start_time, 3),
                end=round(min(end_time, duration), 3),
                peak_time=round(min(peak_time, duration), 3),
                mean_motion=round(mean_motion, 6),
                peak_motion=round(peak_motion, 6),
                score=round(score, 6),
            )
        )

    return fight_segments, {
        "threshold": round(threshold, 6),
        "frame_diffs": len(raw_scores),
        "average_motion": round(float(raw_scores.mean()), 6),
    }


def run(config: PipelineConfig, reporter: StageReporter, trim_manifest: dict) -> dict:
    trimmed_videos = trim_manifest["trimmed_videos"]
    reporter.start(f"Analyzing motion in {len(trimmed_videos)} trimmed videos.")

    videos_payload: list[dict] = []
    all_segments: list[FightSegmentRecord] = []

    for index, record in enumerate(trimmed_videos, start=1):
        progress = (index - 1) / max(len(trimmed_videos), 1)
        reporter.update(progress, f"Detecting fight segments in {Path(record['trimmed_path']).name} ({index}/{len(trimmed_videos)}).")

        trimmed_path = config.paths.project_root / record["trimmed_path"]
        segments, stats = _detect_segments(trimmed_path, record["trimmed_duration"], config)

        hydrated_segments: list[FightSegmentRecord] = []
        for segment in segments:
            hydrated = FightSegmentRecord(
                source_path=record["source_path"],
                trimmed_path=record["trimmed_path"],
                video_duration=segment.video_duration,
                start=segment.start,
                end=segment.end,
                peak_time=segment.peak_time,
                mean_motion=segment.mean_motion,
                peak_motion=segment.peak_motion,
                score=segment.score,
            )
            hydrated_segments.append(hydrated)
            all_segments.append(hydrated)

        videos_payload.append(
            {
                "source_path": record["source_path"],
                "trimmed_path": record["trimmed_path"],
                "trimmed_duration": record["trimmed_duration"],
                "analysis_stats": stats,
                "segments": [asdict(segment) for segment in hydrated_segments],
            }
        )

    sorted_segments = sorted(all_segments, key=lambda item: item.score, reverse=True)
    payload = {
        "stage": "stage_02_detect_fight_segments",
        "videos": videos_payload,
        "top_segments": [asdict(segment) for segment in sorted_segments[:100]],
        "segment_count": len(sorted_segments),
    }
    output_path = config.paths.build_dir / "stage_02_fight_segments.json"
    write_json(output_path, payload)

    reporter.complete(f"Saved {len(sorted_segments)} fight segments.")
    return payload
