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


def _frame_quality_metrics(frame: np.ndarray) -> tuple[float, float, float]:
    frame_float = frame.astype(np.float32) / 255.0
    grad_x = np.abs(np.diff(frame_float, axis=1))
    grad_y = np.abs(np.diff(frame_float, axis=0))
    sharpness = float((grad_x.mean() + grad_y.mean()) * 0.5)
    contrast = float(frame_float.std())
    brightness = float(frame_float.mean())
    return sharpness, contrast, brightness


def _normalize_metric(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lower = float(np.percentile(values, 10))
    upper = float(np.percentile(values, 90))
    spread = upper - lower
    if spread <= 1e-8:
        return np.full_like(values, 0.5, dtype=np.float32)
    normalized = (values - lower) / spread
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def _exposure_balance(brightness: np.ndarray) -> np.ndarray:
    if brightness.size == 0:
        return brightness
    return np.clip(1.0 - (np.abs(brightness - 0.5) / 0.5), 0.0, 1.0).astype(np.float32)


def _calm_segment_score(
    duration_ratio: float,
    calmness: float,
    stability: float,
    sharpness: float,
    contrast: float,
    exposure: float,
) -> float:
    return float(
        (duration_ratio * 0.18)
        + (calmness * 0.22)
        + (stability * 0.22)
        + (sharpness * 0.2)
        + (contrast * 0.1)
        + (exposure * 0.08)
    )


def _build_segment_record(
    start_idx: int,
    end_idx: int,
    duration: float,
    fps: int,
    scores: np.ndarray,
    score_value: float,
) -> FightSegmentRecord:
    start_time = start_idx / fps
    end_time = (end_idx + 1) / fps
    segment_scores = scores[start_idx : end_idx + 1]
    peak_offset = int(segment_scores.argmax())
    peak_time = (start_idx + peak_offset) / fps
    mean_motion = float(segment_scores.mean())
    peak_motion = float(segment_scores.max())
    return FightSegmentRecord(
        source_path="",
        trimmed_path="",
        video_duration=round(duration, 3),
        start=round(start_time, 3),
        end=round(min(end_time, duration), 3),
        peak_time=round(min(peak_time, duration), 3),
        mean_motion=round(mean_motion, 6),
        peak_motion=round(peak_motion, 6),
        score=round(score_value, 6),
    )


def _detect_segments(
    trimmed_path: Path,
    duration: float,
    config: PipelineConfig,
) -> tuple[list[FightSegmentRecord], list[FightSegmentRecord], dict]:
    motion_values: list[float] = []
    sharpness_values: list[float] = []
    contrast_values: list[float] = []
    brightness_values: list[float] = []
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
            sharpness, contrast, brightness = _frame_quality_metrics(frame)
            sharpness_values.append(sharpness)
            contrast_values.append(contrast)
            brightness_values.append(brightness)
        previous_frame = current_frame

    if not motion_values:
        return [], [], {"threshold": 0.0, "calm_threshold": 0.0, "frame_diffs": 0}

    raw_scores = np.asarray(motion_values, dtype=np.float32)
    raw_sharpness = np.asarray(sharpness_values, dtype=np.float32)
    raw_contrast = np.asarray(contrast_values, dtype=np.float32)
    raw_brightness = np.asarray(brightness_values, dtype=np.float32)
    smoothed_scores = _moving_average(
        raw_scores,
        max(1, int(round(config.motion.smoothing_seconds * config.motion.analysis_fps))),
    )
    normalized_sharpness = _normalize_metric(raw_sharpness)
    normalized_contrast = _normalize_metric(raw_contrast)
    exposure_scores = _exposure_balance(raw_brightness)
    threshold = max(
        float(np.quantile(smoothed_scores, config.motion.threshold_quantile)),
        float(smoothed_scores.mean() + (smoothed_scores.std() * 0.75)),
        config.motion.threshold_floor,
    )
    calm_threshold = min(
        float(np.quantile(smoothed_scores, config.motion.calm_threshold_quantile)),
        config.motion.calm_threshold_ceiling,
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
        mean_motion = float(segment_scores.mean())
        peak_motion = float(segment_scores.max())
        score = float(mean_motion * segment_duration * (1.0 + peak_motion))

        fight_segments.append(
            _build_segment_record(
                start_idx,
                end_idx,
                duration,
                config.motion.analysis_fps,
                smoothed_scores,
                score,
            )
        )

    calm_mask = smoothed_scores <= calm_threshold
    calm_segments_raw: list[tuple[int, int]] = []
    calm_start: int | None = None
    for index, is_calm in enumerate(calm_mask):
        if is_calm and calm_start is None:
            calm_start = index
        elif not is_calm and calm_start is not None:
            calm_segments_raw.append((calm_start, index - 1))
            calm_start = None
    if calm_start is not None:
        calm_segments_raw.append((calm_start, len(calm_mask) - 1))

    merged_calm_segments = _merge_segments(
        calm_segments_raw,
        smoothed_scores,
        fps=config.motion.analysis_fps,
        merge_gap_seconds=config.motion.calm_merge_gap_seconds,
    )

    calm_segments: list[FightSegmentRecord] = []
    for start_idx, end_idx in merged_calm_segments:
        start_time = start_idx / config.motion.analysis_fps
        end_time = (end_idx + 1) / config.motion.analysis_fps
        segment_duration = end_time - start_time
        if segment_duration < config.motion.calm_min_segment_seconds:
            continue

        if segment_duration > config.motion.calm_max_segment_seconds:
            max_frames = int(round(config.motion.calm_max_segment_seconds * config.motion.analysis_fps))
            end_idx = min(end_idx, start_idx + max_frames - 1)
            end_time = (end_idx + 1) / config.motion.analysis_fps
            segment_duration = end_time - start_time

        segment_scores = smoothed_scores[start_idx : end_idx + 1]
        mean_motion = float(segment_scores.mean())
        calmness = max(0.0, calm_threshold - mean_motion) + 1e-6
        stability = max(0.0, 1.0 - (mean_motion / max(calm_threshold, 1e-6)))
        sharpness_score = float(normalized_sharpness[start_idx : end_idx + 1].mean())
        contrast_score = float(normalized_contrast[start_idx : end_idx + 1].mean())
        exposure_score = float(exposure_scores[start_idx : end_idx + 1].mean())
        duration_ratio = min(segment_duration / config.motion.calm_max_segment_seconds, 1.0)
        calmness_score = min(calmness / max(calm_threshold, 1e-6), 1.0)
        score = _calm_segment_score(
            duration_ratio=duration_ratio,
            calmness=calmness_score,
            stability=stability,
            sharpness=sharpness_score,
            contrast=contrast_score,
            exposure=exposure_score,
        )
        calm_segments.append(
            _build_segment_record(
                start_idx,
                end_idx,
                duration,
                config.motion.analysis_fps,
                smoothed_scores,
                score,
            )
        )

    return fight_segments, calm_segments, {
        "threshold": round(threshold, 6),
        "calm_threshold": round(calm_threshold, 6),
        "frame_diffs": len(raw_scores),
        "average_motion": round(float(raw_scores.mean()), 6),
    }


def run(config: PipelineConfig, reporter: StageReporter, trim_manifest: dict) -> dict:
    trimmed_videos = trim_manifest["trimmed_videos"]
    reporter.start(f"Analyzing motion in {len(trimmed_videos)} trimmed videos.")

    videos_payload: list[dict] = []
    all_segments: list[FightSegmentRecord] = []
    all_calm_segments: list[FightSegmentRecord] = []

    for index, record in enumerate(trimmed_videos, start=1):
        progress = (index - 1) / max(len(trimmed_videos), 1)
        reporter.update(progress, f"Detecting fight segments in {Path(record['trimmed_path']).name} ({index}/{len(trimmed_videos)}).")

        trimmed_path = config.paths.project_root / record["trimmed_path"]
        segments, calm_segments, stats = _detect_segments(trimmed_path, record["trimmed_duration"], config)

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

        hydrated_calm_segments: list[FightSegmentRecord] = []
        for segment in calm_segments:
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
            hydrated_calm_segments.append(hydrated)
            all_calm_segments.append(hydrated)

        videos_payload.append(
            {
                "source_path": record["source_path"],
                "trimmed_path": record["trimmed_path"],
                "trimmed_duration": record["trimmed_duration"],
                "analysis_stats": stats,
                "segments": [asdict(segment) for segment in hydrated_segments],
                "calm_segments": [asdict(segment) for segment in hydrated_calm_segments],
            }
        )

    sorted_segments = sorted(all_segments, key=lambda item: item.score, reverse=True)
    sorted_calm_segments = sorted(all_calm_segments, key=lambda item: item.score, reverse=True)
    payload = {
        "stage": "stage_02_detect_fight_segments",
        "videos": videos_payload,
        "top_segments": [asdict(segment) for segment in sorted_segments[:100]],
        "calm_segments": [asdict(segment) for segment in sorted_calm_segments[:150]],
        "segment_count": len(sorted_segments),
        "calm_segment_count": len(sorted_calm_segments),
    }
    output_path = config.paths.build_dir / "stage_02_fight_segments.json"
    write_json(output_path, payload)

    reporter.complete(f"Saved {len(sorted_segments)} fight segments.")
    return payload
