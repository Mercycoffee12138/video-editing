import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cutting_pipeline.config import build_default_config  # noqa: E402
from cutting_pipeline.stage_02_extract_collision_events import (  # noqa: E402
    _enrich_segment_with_collision_events,
    _select_diversified_key_event_times,
)


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

        self.assertEqual(enriched["key_event_times"], [])
        self.assertEqual(enriched["review"]["key_event_times"], [])
        self.assertEqual(len(enriched["review"]["collision_events"]), 2)

    def test_select_diversified_key_event_times_delays_repeats_until_other_segments_get_first_pick(self) -> None:
        config = build_default_config(PROJECT_ROOT)
        segments = [
            {
                "review": {
                    "collision_events": [
                        {"time": 10.1, "score": 4.0},
                        {"time": 10.4, "score": 3.6},
                        {"time": 10.9, "score": 2.0},
                    ]
                }
            },
            {
                "review": {
                    "collision_events": [
                        {"time": 20.2, "score": 3.5},
                    ]
                }
            },
        ]

        diversified = _select_diversified_key_event_times(segments, config)

        self.assertEqual(diversified[0]["key_event_times"], [10.1, 10.4])
        self.assertEqual(diversified[1]["key_event_times"], [20.2])

    def test_select_diversified_key_event_times_filters_weak_repeats(self) -> None:
        config = build_default_config(PROJECT_ROOT)
        segments = [
            {
                "review": {
                    "collision_events": [
                        {"time": 10.1, "score": 4.0},
                        {"time": 10.4, "score": 2.5},
                    ]
                }
            }
        ]

        diversified = _select_diversified_key_event_times(segments, config)

        self.assertEqual(diversified[0]["key_event_times"], [10.1])


if __name__ == "__main__":
    unittest.main()
