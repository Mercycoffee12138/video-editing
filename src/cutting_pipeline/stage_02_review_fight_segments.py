from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .ffmpeg_tools import export_video_frame
from .json_io import write_json
from .progress import StageReporter
from .qwen_vision import analyze_images, load_config_from_env


def _review_prompt() -> str:
    return (
        "你在审核短视频动作片段是否属于打斗、对抗、冲突或明显攻击防守场景。"
        "我会提供同一个片段的多张抽帧。"
        "请只返回 JSON，不要输出 markdown 代码块，不要补充解释。"
        'JSON 格式必须是: {"contains_fight": true, "confidence": 0.0, '
        '"summary": "...", "ocr_text": "..."}。'
        "contains_fight 表示画面是否明显包含人物打斗、互殴、攻击、防守、摔打、武器对抗或强烈肢体冲突。"
        "confidence 取 0 到 1。"
        "summary 用一句中文描述画面。"
        "ocr_text 提取画面中能看清的主要文字，没有则返回空字符串。"
    )


def _parse_review_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = [line for line in cleaned.splitlines() if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Review model did not return JSON: {text}")

    parsed = json.loads(cleaned[start : end + 1])
    return {
        "contains_fight": bool(parsed.get("contains_fight")),
        "confidence": round(float(parsed.get("confidence", 0.0)), 4),
        "summary": str(parsed.get("summary", "")).strip(),
        "ocr_text": str(parsed.get("ocr_text", "")).strip(),
    }


def _segment_frame_times(segment: dict) -> list[float]:
    start = float(segment["start"])
    end = float(segment["end"])
    peak = float(segment["peak_time"])
    duration = max(end - start, 0.01)
    early = start + min(duration * 0.2, 0.5)
    late = end - min(duration * 0.2, 0.5)
    return [round(early, 3), round(peak, 3), round(max(start, late), 3)]


def _extract_review_frames(
    config: PipelineConfig,
    segment: dict,
    segment_index: int,
) -> list[Path]:
    trimmed_path = config.paths.project_root / segment["trimmed_path"]
    segment_dir = config.paths.stage_02_review_frames_dir / f"segment_{segment_index:03d}"
    segment_dir.mkdir(parents=True, exist_ok=True)

    frame_paths: list[Path] = []
    for frame_index, timestamp in enumerate(_segment_frame_times(segment), start=1):
        frame_path = segment_dir / f"frame_{frame_index:02d}.jpg"
        export_video_frame(trimmed_path, timestamp, frame_path)
        frame_paths.append(frame_path)
    return frame_paths


def _build_reviewed_segment(
    project_root: Path,
    segment: dict,
    review: dict[str, Any],
    frame_paths: list[Path],
    accepted: bool,
) -> dict[str, Any]:
    payload = dict(segment)
    payload["review"] = {
        "accepted": accepted,
        "accepted_by_relaxation": False,
        "contains_fight": review["contains_fight"],
        "confidence": review["confidence"],
        "summary": review["summary"],
        "ocr_text": review["ocr_text"],
        "frame_paths": [str(path.relative_to(project_root)) for path in frame_paths],
    }
    if accepted:
        payload["score"] = round(float(segment["score"]) * (1.0 + (review["confidence"] * 0.35)), 6)
    else:
        payload["score"] = round(float(segment["score"]) * 0.25, 6)
    return payload


def _review_rank_key(segment: dict[str, Any]) -> tuple[float, float, float]:
    review = segment["review"]
    contains_fight_score = 1.0 if review["contains_fight"] else 0.0
    return (
        contains_fight_score,
        float(review["confidence"]),
        float(segment["score"]),
    )


def _apply_relaxed_acceptance(reviewed_segments: list[dict[str, Any]], config: PipelineConfig) -> list[dict[str, Any]]:
    accepted_segments = [segment for segment in reviewed_segments if segment["review"]["accepted"]]
    minimum_needed = min(config.review.min_accepted_segments, len(reviewed_segments))
    if len(accepted_segments) >= minimum_needed:
        return accepted_segments

    promoted_segments = sorted(
        [
            segment
            for segment in reviewed_segments
            if not segment["review"]["accepted"]
            and segment["review"]["contains_fight"]
            and float(segment["review"]["confidence"]) >= config.review.relaxed_min_confidence
        ],
        key=_review_rank_key,
        reverse=True,
    )

    remaining_needed = minimum_needed - len(accepted_segments)
    if len(promoted_segments) < remaining_needed:
        extra_candidates = sorted(
            [
                segment
                for segment in reviewed_segments
                if not segment["review"]["accepted"] and segment not in promoted_segments
            ],
            key=_review_rank_key,
            reverse=True,
        )
        promoted_segments.extend(extra_candidates[: remaining_needed - len(promoted_segments)])

    for segment in promoted_segments[:remaining_needed]:
        segment["review"]["accepted"] = True
        segment["review"]["accepted_by_relaxation"] = True
        segment["score"] = round(float(segment["score"]) * 2.2, 6)
        accepted_segments.append(segment)

    return accepted_segments


def run(config: PipelineConfig, reporter: StageReporter, fight_segments_payload: dict) -> dict:
    candidates = list(fight_segments_payload.get("top_segments") or [])[: config.review.max_candidate_segments]
    reporter.start(f"Reviewing {len(candidates)} candidate fight segments with Qwen VL.")

    vision_config = load_config_from_env() if config.review.enabled else None
    if not vision_config:
        payload = {
            "stage": "stage_02_review_fight_segments",
            "review_enabled": False,
            "candidate_count": len(candidates),
            "reviewed_count": 0,
            "accepted_count": len(candidates),
            "top_segments": candidates,
            "reviewed_segments": [],
        }
        output_path = config.paths.build_dir / "stage_02_reviewed_fight_segments.json"
        write_json(output_path, payload)
        reporter.complete("Vision review skipped because no API key was configured.")
        return payload

    reviewed_segments: list[dict[str, Any]] = []
    for index, segment in enumerate(candidates, start=1):
        progress = (index - 1) / max(len(candidates), 1)
        reporter.update(progress, f"Reviewing segment {index}/{len(candidates)}.")

        frame_paths = _extract_review_frames(config, segment, index)
        response = analyze_images(frame_paths, _review_prompt(), vision_config)
        review = _parse_review_json(response["text"])
        accepted = review["contains_fight"] and review["confidence"] >= config.review.min_confidence
        reviewed_segments.append(
            _build_reviewed_segment(
                config.paths.project_root,
                segment,
                review,
                frame_paths,
                accepted,
            )
        )

    accepted_segments = _apply_relaxed_acceptance(reviewed_segments, config)
    top_segments = accepted_segments or candidates
    top_segments = sorted(top_segments, key=lambda item: float(item["score"]), reverse=True)

    payload = {
        "stage": "stage_02_review_fight_segments",
        "review_enabled": True,
        "candidate_count": len(candidates),
        "reviewed_count": len(reviewed_segments),
        "accepted_count": len(accepted_segments),
        "relaxed_acceptance_applied": any(
            segment["review"]["accepted_by_relaxation"] for segment in reviewed_segments
        ),
        "top_segments": top_segments,
        "reviewed_segments": reviewed_segments,
    }
    output_path = config.paths.build_dir / "stage_02_reviewed_fight_segments.json"
    write_json(output_path, payload)

    reporter.complete(f"Accepted {len(accepted_segments)} of {len(candidates)} candidate segments.")
    return payload
