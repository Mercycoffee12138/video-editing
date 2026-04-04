import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cutting_pipeline.config import PipelineConfig, PipelinePaths, ReviewConfig
from cutting_pipeline.stage_02_review_fight_segments import (  # noqa: E402
    _apply_relaxed_acceptance,
    _collision_preview_window,
    _content_blocked_review,
    _merge_collision_candidates,
    _resolve_collision_event_times,
    _parse_review_json,
    _segment_frame_times,
)


class ReviewFightSegmentsTests(unittest.TestCase):
    def _config(self, review: ReviewConfig) -> PipelineConfig:
        project_root = PROJECT_ROOT
        return PipelineConfig(
            paths=PipelinePaths(
                project_root=project_root,
                source_root=project_root / "source",
                video_source_dir=project_root / "source" / "video",
                music_source_dir=project_root / "source" / "music",
                build_dir=project_root / "build",
                stage_01_trim_dir=project_root / "build" / "stage_01_trimmed_videos",
                stage_02_review_frames_dir=project_root / "build" / "stage_02_review_frames",
                stage_05_clip_dir=project_root / "build" / "stage_05_render_clips",
                stage_05_temp_dir=project_root / "build" / "stage_05_temp",
            ),
            review=review,
        )

    def test_parse_review_json_handles_plain_json(self) -> None:
        parsed = _parse_review_json(
            json.dumps(
                {
                    "contains_fight": True,
                    "confidence": 0.91,
                    "summary": "两个人在近距离搏斗。",
                    "ocr_text": "ROUND 1",
                },
                ensure_ascii=False,
            )
        )

        self.assertTrue(parsed["contains_fight"])
        self.assertEqual(parsed["confidence"], 0.91)
        self.assertEqual(parsed["summary"], "两个人在近距离搏斗。")
        self.assertEqual(parsed["ocr_text"], "ROUND 1")

    def test_parse_review_json_handles_code_fence(self) -> None:
        parsed = _parse_review_json(
            """```json
            {"contains_fight": false, "confidence": 0.2, "summary": "采访镜头", "ocr_text": ""}
            ```"""
        )

        self.assertFalse(parsed["contains_fight"])
        self.assertEqual(parsed["confidence"], 0.2)

    def test_parse_review_json_reads_refined_fields(self) -> None:
        parsed = _parse_review_json(
            json.dumps(
                {
                    "contains_fight": True,
                    "confidence": 0.78,
                    "refined_start_ratio": 0.15,
                    "refined_end_ratio": 0.88,
                    "peak_candidate_index": 2,
                    "key_event_candidate_indices": [2, 4, 4],
                    "summary": "连续对打后出现一次明显碰撞。",
                    "ocr_text": "",
                },
                ensure_ascii=False,
            )
        )

        self.assertEqual(parsed["refined_start_ratio"], 0.15)
        self.assertEqual(parsed["refined_end_ratio"], 0.88)
        self.assertEqual(parsed["peak_candidate_index"], 2)
        self.assertEqual(parsed["key_event_candidate_indices"], [2, 4])

    def test_segment_frame_times_cover_early_peak_and_late(self) -> None:
        times = _segment_frame_times(
            {
                "start": 10.0,
                "peak_time": 11.2,
                "end": 13.0,
            }
        )

        self.assertEqual(times[1], 11.2)
        self.assertGreaterEqual(times[0], 10.0)
        self.assertLessEqual(times[2], 13.0)
        self.assertEqual(len(times), 3)

    def test_content_blocked_review_keeps_segment_as_fight_with_audio_candidates(self) -> None:
        config = self._config(ReviewConfig())
        review = _content_blocked_review(
            segment={
                "confidence": 0.63,
            },
            candidate_events=[
                {"candidate_index": 2, "score": 1.1},
                {"candidate_index": 1, "score": 0.8},
            ],
            config=config,
            error_message="inspection blocked",
        )

        self.assertTrue(review["contains_fight"])
        self.assertTrue(review["content_blocked"])
        self.assertGreaterEqual(review["confidence"], config.review.min_confidence)
        self.assertEqual(review["peak_candidate_index"], 2)
        self.assertEqual(review["key_event_candidate_indices"], [2, 1])

    def test_resolve_collision_event_times_prefers_strongest_refined_events(self) -> None:
        config = self._config(ReviewConfig())
        times = _resolve_collision_event_times(
            [
                {"time": 11.2, "score": 2.0},
                {"time": 10.8, "score": 3.1},
                {"time": 11.2, "score": 1.1},
            ],
            config,
        )

        self.assertEqual(times, [10.8, 11.2])

    def test_merge_collision_candidates_combines_audio_and_visual_scores(self) -> None:
        merged = _merge_collision_candidates(
            audio_candidates=[
                {
                    "candidate_index": 1,
                    "time": 10.0,
                    "score": 1.2,
                    "audio_energy": 0.6,
                    "audio_accent": 0.7,
                    "candidate_source": "audio",
                }
            ],
            visual_candidates=[
                {
                    "candidate_index": 1,
                    "time": 10.05,
                    "score": 0.8,
                    "visual_motion": 0.9,
                    "visual_flash": 0.4,
                    "candidate_source": "visual",
                }
            ],
            merge_window_seconds=0.1,
            limit=4,
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["candidate_source"], "audio_visual")
        self.assertGreater(merged[0]["score"], 0.8)

    def test_collision_preview_window_respects_segment_bounds(self) -> None:
        config = self._config(ReviewConfig())
        clip_start, clip_end = _collision_preview_window(
            {
                "start": 10.0,
                "end": 10.42,
            },
            event_time=10.04,
            config=config,
        )

        self.assertGreaterEqual(clip_start, 10.0)
        self.assertLessEqual(clip_end, 10.42)
        self.assertGreaterEqual(round(clip_end - clip_start, 3), 0.4)

    def test_apply_relaxed_acceptance_promotes_near_misses_when_too_few_segments_pass(self) -> None:
        config = self._config(
            ReviewConfig(
                min_confidence=0.55,
                min_accepted_segments=3,
                relaxed_min_confidence=0.35,
            )
        )
        reviewed_segments = [
            {
                "score": 10.0,
                "review": {
                    "accepted": True,
                    "accepted_by_relaxation": False,
                    "contains_fight": True,
                    "confidence": 0.8,
                },
            },
            {
                "score": 8.0,
                "review": {
                    "accepted": False,
                    "accepted_by_relaxation": False,
                    "contains_fight": True,
                    "confidence": 0.5,
                },
            },
            {
                "score": 7.0,
                "review": {
                    "accepted": False,
                    "accepted_by_relaxation": False,
                    "contains_fight": True,
                    "confidence": 0.42,
                },
            },
            {
                "score": 9.0,
                "review": {
                    "accepted": False,
                    "accepted_by_relaxation": False,
                    "contains_fight": False,
                    "confidence": 0.9,
                },
            },
        ]

        accepted = _apply_relaxed_acceptance(reviewed_segments, config)

        self.assertEqual(len(accepted), 3)
        self.assertTrue(reviewed_segments[1]["review"]["accepted"])
        self.assertTrue(reviewed_segments[1]["review"]["accepted_by_relaxation"])
        self.assertTrue(reviewed_segments[2]["review"]["accepted"])
        self.assertFalse(reviewed_segments[3]["review"]["accepted"])


if __name__ == "__main__":
    unittest.main()
