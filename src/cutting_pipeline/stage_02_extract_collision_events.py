from __future__ import annotations

from .config import PipelineConfig
from .json_io import write_json
from .progress import StageReporter
from .stage_02_review_fight_segments import (
    _export_collision_event_previews,
    _extract_refined_collision_candidates,
    _resolve_collision_event_times,
)


def _enrich_segment_with_collision_events(config: PipelineConfig, segment: dict) -> dict:
    payload = dict(segment)
    collision_events = _extract_refined_collision_candidates(config, payload)
    key_event_times = _resolve_collision_event_times(collision_events, config)

    review = dict(payload.get("review") or {})
    review["collision_events"] = collision_events
    review["key_event_times"] = key_event_times
    payload["review"] = review
    payload["key_event_times"] = key_event_times
    return payload


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
