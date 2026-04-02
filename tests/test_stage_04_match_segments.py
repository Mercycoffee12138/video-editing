import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cutting_pipeline.config import MatchConfig
from cutting_pipeline.stage_04_match_segments import (
    build_timeline_durations,
    select_highlight_cluster,
)


class MatchStageTests(unittest.TestCase):
    def test_select_highlight_cluster_prefers_dense_window(self) -> None:
        config = MatchConfig(
            highlight_cluster_window_seconds=20.0,
            max_highlights_per_track=4,
        )
        highlights = [
            {"time": 5.0, "score": 0.5, "energy": 0.1, "accent": 0.1},
            {"time": 9.0, "score": 0.6, "energy": 0.1, "accent": 0.1},
            {"time": 14.0, "score": 0.7, "energy": 0.1, "accent": 0.1},
            {"time": 18.0, "score": 0.8, "energy": 0.1, "accent": 0.1},
            {"time": 80.0, "score": 3.0, "energy": 0.1, "accent": 0.1},
        ]

        selected = select_highlight_cluster(highlights, config)

        self.assertEqual([item["time"] for item in selected], [5.0, 9.0, 14.0, 18.0])

    def test_build_timeline_durations_preserves_total_duration(self) -> None:
        config = MatchConfig(min_clip_seconds=0.8, max_clip_seconds=4.0)
        selected_highlights = [
            {"time": 10.0, "score": 1.0, "energy": 0.1, "accent": 0.1},
            {"time": 14.0, "score": 1.0, "energy": 0.1, "accent": 0.1},
            {"time": 22.0, "score": 1.0, "energy": 0.1, "accent": 0.1},
        ]

        durations = build_timeline_durations(6.0, 28.0, selected_highlights, config)

        self.assertAlmostEqual(sum(durations), 22.0, places=2)
        self.assertTrue(all(duration > 0 for duration in durations))


if __name__ == "__main__":
    unittest.main()
