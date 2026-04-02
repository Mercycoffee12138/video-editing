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
