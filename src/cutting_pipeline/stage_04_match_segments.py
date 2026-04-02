from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict

from .config import MatchConfig, PipelineConfig
from .json_io import write_json
from .models import MatchPlanRecord, MatchedClipRecord, MusicHighlightRecord
from .progress import StageReporter


def select_highlight_cluster(
    highlights: list[dict],
    config: MatchConfig,
) -> list[dict]:
    if not highlights:
        return []

    best_selection: list[dict] = []
    best_score = float("-inf")

    for start_index in range(len(highlights)):
        window_start = highlights[start_index]["time"]
        candidates = [
            highlight
            for highlight in highlights[start_index:]
            if highlight["time"] - window_start <= config.highlight_cluster_window_seconds
        ]
        if not candidates:
            continue

        ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)
        selected = sorted(
            ranked[: config.max_highlights_per_track],
            key=lambda item: item["time"],
        )
        span = max(selected[-1]["time"] - selected[0]["time"], 1.0)
        score = sum(item["score"] for item in selected) + (0.35 * len(selected)) - (0.01 * span)
        if score > best_score:
            best_score = score
            best_selection = selected

    return best_selection


def build_timeline_durations(
    audio_start: float,
    audio_end: float,
    selected_highlights: list[dict],
    config: MatchConfig,
) -> list[float]:
    return [chunk["duration"] for chunk in build_timeline_chunks(audio_start, audio_end, selected_highlights, config)]


def _normalized_highlight_scores(selected_highlights: list[dict]) -> list[float]:
    if not selected_highlights:
        return []

    scores = [float(highlight["score"]) for highlight in selected_highlights]
    minimum = min(scores)
    maximum = max(scores)
    spread = maximum - minimum
    if spread <= 1e-8:
        return [1.0 for _ in scores]
    return [(score - minimum) / spread for score in scores]


def _target_intensity(
    center_time: float,
    audio_start: float,
    audio_end: float,
    selected_highlights: list[dict],
) -> float:
    total_duration = max(audio_end - audio_start, 1e-6)
    progress = (center_time - audio_start) / total_duration

    # Keep the first stretch more restrained, then raise overall energy later in the song.
    base_intensity = 0.2 + (0.35 * progress)

    peak_window_seconds = min(8.0, max(total_duration * 0.12, 3.0))
    peak_influence = 0.0
    normalized_scores = _normalized_highlight_scores(selected_highlights)
    for highlight, score_weight in zip(selected_highlights, normalized_scores):
        distance = abs(center_time - float(highlight["time"]))
        closeness = max(0.0, 1.0 - (distance / peak_window_seconds))
        peak_influence = max(peak_influence, (0.45 + (0.55 * score_weight)) * closeness)

    intensity = base_intensity + (0.55 * peak_influence)
    return max(0.0, min(1.0, intensity))


def build_timeline_chunks(
    audio_start: float,
    audio_end: float,
    selected_highlights: list[dict],
    config: MatchConfig,
) -> list[dict]:
    cut_points = [audio_start] + [highlight["time"] for highlight in selected_highlights] + [audio_end]
    raw_gaps = [end - start for start, end in zip(cut_points, cut_points[1:])]

    timeline: list[dict] = []
    buffer = 0.0
    buffer_start = audio_start
    for gap in raw_gaps:
        buffer += gap
        if buffer < config.min_clip_seconds:
            continue
        chunk_count = max(1, math.ceil(buffer / config.max_clip_seconds))
        chunk_duration = buffer / chunk_count
        for chunk_index in range(chunk_count):
            chunk_start = buffer_start + (chunk_duration * chunk_index)
            chunk_end = chunk_start + chunk_duration
            center_time = (chunk_start + chunk_end) / 2.0
            timeline.append(
                {
                    "start": round(chunk_start, 3),
                    "end": round(chunk_end, 3),
                    "duration": round(chunk_duration, 3),
                    "target_intensity": round(
                        _target_intensity(center_time, audio_start, audio_end, selected_highlights),
                        4,
                    ),
                }
            )
        buffer_start += buffer
        buffer = 0.0

    if buffer > 0:
        if timeline:
            timeline[-1]["end"] = round(timeline[-1]["end"] + buffer, 3)
            timeline[-1]["duration"] = round(timeline[-1]["duration"] + buffer, 3)
        else:
            chunk_end = buffer_start + buffer
            timeline.append(
                {
                    "start": round(buffer_start, 3),
                    "end": round(chunk_end, 3),
                    "duration": round(buffer, 3),
                    "target_intensity": round(
                        _target_intensity((buffer_start + chunk_end) / 2.0, audio_start, audio_end, selected_highlights),
                        4,
                    ),
                }
            )

    return timeline


def assign_clips(
    fight_segments: list[dict],
    timeline_chunks: list[dict],
    config: MatchConfig,
) -> list[MatchedClipRecord]:
    ranked_segments = sorted(fight_segments, key=lambda item: item["score"], reverse=True)
    if not ranked_segments:
        raise ValueError("No fight segments are available for clip assignment.")

    segment_scores = [float(segment["score"]) for segment in ranked_segments]
    min_score = min(segment_scores)
    max_score = max(segment_scores)
    spread = max(max_score - min_score, 1e-8)

    remaining_segments = list(ranked_segments)
    reuse_count: defaultdict[str, int] = defaultdict(int)
    clips: list[MatchedClipRecord] = []

    for order, chunk in enumerate(timeline_chunks, start=1):
        if not remaining_segments:
            remaining_segments = list(ranked_segments)

        duration = float(chunk["duration"])
        target_intensity = float(chunk["target_intensity"])
        best_index = 0
        best_score = float("-inf")
        for index, segment in enumerate(remaining_segments):
            normalized_segment_score = (float(segment["score"]) - min_score) / spread
            intensity_match = 1.0 - abs(normalized_segment_score - target_intensity)
            reuse_penalty = 1.0 + (reuse_count[segment["source_path"]] * config.source_reuse_penalty)
            weighted_score = (
                (intensity_match * 0.7)
                + (normalized_segment_score * 0.3)
            ) / reuse_penalty
            if weighted_score > best_score:
                best_score = weighted_score
                best_index = index

        selected_segment = remaining_segments.pop(best_index)
        reuse_count[selected_segment["source_path"]] += 1

        max_start = max(selected_segment["video_duration"] - duration, 0.0)
        centered_start = selected_segment["peak_time"] - (duration / 2.0)
        clip_start = max(0.0, min(centered_start, max_start))
        clip_end = clip_start + duration

        clips.append(
            MatchedClipRecord(
                order=order,
                source_path=selected_segment["source_path"],
                trimmed_path=selected_segment["trimmed_path"],
                clip_start=round(clip_start, 3),
                clip_end=round(clip_end, 3),
                duration=round(duration, 3),
                segment_score=round(selected_segment["score"], 6),
            )
        )

    return clips


def build_plan_for_track(track: dict, fight_segments: list[dict], config: MatchConfig) -> MatchPlanRecord | None:
    selected_highlights = select_highlight_cluster(track["highlights"], config)
    if not selected_highlights:
        return None

    if config.use_full_track_duration:
        audio_start = 0.0
        audio_end = track["duration"]
    else:
        audio_start = max(0.0, selected_highlights[0]["time"] - config.intro_padding_seconds)
        audio_end = min(track["duration"], selected_highlights[-1]["time"] + config.outro_padding_seconds)

    timeline_chunks = build_timeline_chunks(audio_start, audio_end, selected_highlights, config)
    timeline_durations = [chunk["duration"] for chunk in timeline_chunks]
    if not timeline_durations:
        return None

    clips = assign_clips(fight_segments, timeline_chunks, config)
    highlight_records = [MusicHighlightRecord(**highlight) for highlight in selected_highlights]

    average_segment_score = sum(clip.segment_score for clip in clips) / max(len(clips), 1)
    average_highlight_score = sum(highlight.score for highlight in highlight_records) / max(len(highlight_records), 1)
    output_duration = round(sum(timeline_durations), 3)
    plan_score = round((average_segment_score * 100.0) + average_highlight_score + (len(clips) * 0.15), 6)

    return MatchPlanRecord(
        music_path=track["music_path"],
        audio_excerpt_start=round(audio_start, 3),
        audio_excerpt_end=round(audio_end, 3),
        output_duration=output_duration,
        selected_highlights=highlight_records,
        timeline_durations=timeline_durations,
        clips=clips,
        plan_score=plan_score,
    )


def run(
    config: PipelineConfig,
    reporter: StageReporter,
    fight_segments_payload: dict,
    music_highlights_payload: dict,
) -> dict:
    tracks = music_highlights_payload["tracks"]
    fight_segments = fight_segments_payload["top_segments"]

    requested_music = config.match.selected_music_filename
    if requested_music:
        requested_path = f"source/music/{requested_music}"
        filtered_tracks = [track for track in tracks if track["music_path"] == requested_path]
        if not filtered_tracks:
            available = ", ".join(track["music_path"] for track in tracks)
            raise ValueError(
                f"Configured music file '{requested_music}' was not found. Available tracks: {available}"
            )
        tracks = filtered_tracks

    reporter.start(f"Building match plans for {len(tracks)} tracks.")

    plans: list[MatchPlanRecord] = []
    for index, track in enumerate(tracks, start=1):
        progress = (index - 1) / max(len(tracks), 1)
        reporter.update(progress, f"Matching clips to {track['music_path']} ({index}/{len(tracks)}).")
        plan = build_plan_for_track(track, fight_segments, config.match)
        if plan is not None:
            plans.append(plan)

    if not plans:
        raise ValueError("No match plans could be created.")

    plans.sort(key=lambda item: item.plan_score, reverse=True)
    selected_plan = plans[0]

    payload = {
        "stage": "stage_04_match_segments",
        "selected_music_path": selected_plan.music_path,
        "plans": [asdict(plan) for plan in plans],
    }
    output_path = config.paths.build_dir / "stage_04_match_plan.json"
    write_json(output_path, payload)

    if requested_music:
        reporter.complete(f"Using configured music track {selected_plan.music_path}.")
    else:
        reporter.complete(f"Selected {selected_plan.music_path} as the best match plan.")
    return payload
