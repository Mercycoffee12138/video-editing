from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .ffmpeg_tools import decode_audio_mono, get_media_duration
from .json_io import write_json
from .models import MusicHighlightRecord, MusicTrackRecord
from .progress import StageReporter


def _moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1 or values.size == 0:
        return values
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    return np.convolve(values, kernel, mode="same")


def _frame_metric(samples: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if samples.size < frame_length:
        padded = np.pad(samples, (0, frame_length - samples.size))
        return np.asarray([float(np.sqrt(np.mean(np.square(padded)) + 1e-8))], dtype=np.float32)

    frame_values: list[float] = []
    for start in range(0, samples.size - frame_length + 1, hop_length):
        frame = samples[start : start + frame_length]
        frame_values.append(float(np.sqrt(np.mean(np.square(frame)) + 1e-8)))
    return np.asarray(frame_values, dtype=np.float32)


def _normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    center = float(np.median(values))
    spread = float(np.percentile(values, 75) - np.percentile(values, 25))
    if spread < 1e-8:
        spread = float(values.std()) + 1e-8
    return (values - center) / spread


def _pick_peaks(
    scores: np.ndarray,
    min_distance_frames: int,
    limit: int,
    threshold: float,
) -> list[int]:
    if scores.size < 3:
        return []

    candidates = [
        index
        for index in range(1, len(scores) - 1)
        if scores[index] >= scores[index - 1]
        and scores[index] > scores[index + 1]
        and scores[index] >= threshold
    ]

    kept: list[int] = []
    for index in sorted(candidates, key=lambda item: scores[item], reverse=True):
        if all(abs(index - existing) >= min_distance_frames for existing in kept):
            kept.append(index)
        if len(kept) >= limit:
            break

    return sorted(kept)


def _relative(path: Path, project_root: Path) -> str:
    return str(path.relative_to(project_root))


def _analyze_track(path: Path, config: PipelineConfig) -> MusicTrackRecord:
    samples = decode_audio_mono(path, sample_rate=config.audio.sample_rate)
    accent_samples = np.diff(samples, prepend=samples[0])

    energy = _frame_metric(samples, config.audio.frame_length, config.audio.hop_length)
    accent = _frame_metric(accent_samples, config.audio.frame_length, config.audio.hop_length)

    energy_focus = np.maximum(
        energy - _moving_average(energy, max(3, int(round(config.audio.sample_rate * 3.0 / config.audio.hop_length)))),
        0.0,
    )
    accent_focus = np.maximum(
        accent - _moving_average(accent, max(3, int(round(config.audio.sample_rate * 1.0 / config.audio.hop_length)))),
        0.0,
    )

    combined = (_normalize(energy_focus) * 0.4) + (_normalize(accent_focus) * 0.6)
    combined = _moving_average(combined.astype(np.float32), 5)

    threshold = max(
        float(np.quantile(combined, config.audio.peak_threshold_quantile)),
        float(combined.mean() + combined.std() * 0.7),
    )
    min_distance_frames = max(
        1,
        int(round(config.audio.min_peak_distance_seconds * config.audio.sample_rate / config.audio.hop_length)),
    )
    peak_indices = _pick_peaks(
        combined,
        min_distance_frames=min_distance_frames,
        limit=config.audio.top_highlights,
        threshold=threshold,
    )

    highlights: list[MusicHighlightRecord] = []
    seconds_per_frame = config.audio.hop_length / config.audio.sample_rate
    for peak_index in peak_indices:
        highlights.append(
            MusicHighlightRecord(
                time=round(peak_index * seconds_per_frame, 3),
                score=round(float(combined[peak_index]), 6),
                energy=round(float(energy[peak_index]), 6),
                accent=round(float(accent[peak_index]), 6),
            )
        )

    return MusicTrackRecord(
        music_path=_relative(path, config.paths.project_root),
        duration=round(get_media_duration(path), 3),
        highlights=highlights,
    )


def run(config: PipelineConfig, reporter: StageReporter) -> dict:
    music_files = sorted(config.paths.music_source_dir.glob("*"))
    reporter.start(f"Analyzing {len(music_files)} music tracks.")

    tracks: list[MusicTrackRecord] = []
    for index, music_path in enumerate(music_files, start=1):
        progress = (index - 1) / max(len(music_files), 1)
        reporter.update(progress, f"Detecting highlights in {music_path.name} ({index}/{len(music_files)}).")
        tracks.append(_analyze_track(music_path, config))

    payload = {
        "stage": "stage_03_detect_music_highlights",
        "tracks": [asdict(track) for track in tracks],
    }
    output_path = config.paths.build_dir / "stage_03_music_highlights.json"
    write_json(output_path, payload)

    reporter.complete(f"Saved highlight metadata for {len(tracks)} tracks.")
    return payload
