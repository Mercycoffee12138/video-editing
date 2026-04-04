import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cutting_pipeline.config import build_default_config  # noqa: E402
from cutting_pipeline.stage_02_extract_collision_events import _enrich_segment_with_collision_events  # noqa: E402


class ExtractCollisionEventsTests(unittest.TestCase):
    def test_enrich_segment_with_collision_events_updates_review_and_key_events(self) -> None:
        config = build_default_config(PROJECT_ROOT)
        segment = {
            "source_path": "source/video/001.mp4",
            "trimmed_path": "build/stage_01_trimmed_videos/001_stage_01_trimmed.mp4",
            "start": 10.0,
            "end": 14.0,
            "peak_time": 12.0,
            "review": {"accepted": True},
        }

        with patch(
            "cutting_pipeline.stage_02_extract_collision_events._extract_refined_collision_candidates",
            return_value=[
                {"time": 11.2, "score": 2.0},
                {"time": 12.6, "score": 3.0},
            ],
        ):
            enriched = _enrich_segment_with_collision_events(config, segment)

        self.assertEqual(enriched["key_event_times"], [11.2, 12.6])
        self.assertEqual(enriched["review"]["key_event_times"], [11.2, 12.6])
        self.assertEqual(len(enriched["review"]["collision_events"]), 2)


if __name__ == "__main__":
    unittest.main()
