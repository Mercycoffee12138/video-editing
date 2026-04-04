from __future__ import annotations

from .config import PipelineConfig
from .json_io import write_json
from .progress import StageReporter
from .stage_02_review_fight_segments import (
    _export_collision_event_previews,
    _extract_refined_collision_candidates,
)


def _enrich_segment_with_collision_events(config: PipelineConfig, segment: dict) -> dict:
    payload = dict(segment)
    collision_events = _extract_refined_collision_candidates(config, payload)
    review = dict(payload.get("review") or {})
    review["collision_events"] = collision_events
    review["key_event_times"] = []
    payload["review"] = review
    payload["key_event_times"] = []
    return payload


def _select_diversified_key_event_times(
    segments: list[dict],
    config: PipelineConfig,
) -> list[dict]:
    candidate_lists: list[list[dict]] = []
    for segment in segments:
        ranked_candidates = sorted(
            list((segment.get("review") or {}).get("collision_events") or []),
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True,
        )
        candidate_lists.append(ranked_candidates[: config.fight_ai.max_key_events_per_segment])

    selected_times_by_segment: list[list[float]] = [[] for _ in segments]
    pass_index = 0
    while pass_index < config.fight_ai.max_key_events_per_segment:
        any_selected = False
        for segment_index, candidates in enumerate(candidate_lists):
            if pass_index >= len(candidates):
                continue

            candidate = candidates[pass_index]
            candidate_time = round(float(candidate["time"]), 3)
            candidate_score = float(candidate.get("score", 0.0))
            existing_times = selected_times_by_segment[segment_index]
            if candidate_time in existing_times:
                continue

            if pass_index > 0 and candidates:
                best_score = float(candidates[0].get("score", 0.0))
                minimum_repeat_score = best_score * config.fight_ai.collision_repeat_score_ratio
                if candidate_score < minimum_repeat_score:
                    continue

            existing_times.append(candidate_time)
            any_selected = True

        if not any_selected:
            break
        pass_index += 1

    enriched_segments: list[dict] = []
    for segment, selected_times in zip(segments, selected_times_by_segment):
        payload = dict(segment)
        review = dict(payload.get("review") or {})
        review["key_event_times"] = sorted(selected_times)
        payload["review"] = review
        payload["key_event_times"] = sorted(selected_times)
        enriched_segments.append(payload)
    return enriched_segments


def run(
    config: PipelineConfig,
    reporter: StageReporter,
    reviewed_fight_segments_payload: dict,
) -> dict:
    top_segments = list(reviewed_fight_segments_payload.get("top_segments") or [])
    reporter.start(f"Extracting collision events from {len(top_segments)} refined fight segments.")

    enriched_segments: list[dict] = []
    for index, segment in enumerate(top_segments, start=1):
        progress = (index - 1) / max(len(top_segments), 1)
        reporter.update(progress, f"Extracting collision events for segment {index}/{len(top_segments)}.")
        enriched_segments.append(_enrich_segment_with_collision_events(config, segment))

    enriched_segments = _select_diversified_key_event_times(enriched_segments, config)

    enriched_segments.sort(
        key=lambda item: (
            float(item.get("fight_probability", 0.0)),
            float(item.get("score", 0.0)),
        ),
        reverse=True,
    )

    collision_event_preview = _export_collision_event_previews(config, reporter, enriched_segments)
    payload = dict(reviewed_fight_segments_payload)
    payload["stage"] = "stage_02_extract_collision_events"
    payload["top_segments"] = enriched_segments
    payload["collision_event_preview"] = collision_event_preview

    output_path = config.paths.build_dir / "stage_02_collision_events.json"
    write_json(output_path, payload)
    reporter.complete(f"Extracted collision events for {len(enriched_segments)} refined fight segments.")
    return payload
