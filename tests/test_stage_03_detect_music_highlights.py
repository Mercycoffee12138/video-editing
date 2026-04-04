import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cutting_pipeline.stage_03_detect_music_highlights import (  # noqa: E402
    _merge_highlight_indices,
    _records_from_indices,
    _pick_peaks,
)


class MusicHighlightsTests(unittest.TestCase):
    def test_pick_peaks_respects_threshold_and_spacing(self) -> None:
        scores = np.asarray([0.0, 0.6, 0.2, 0.9, 0.1, 0.7, 0.1], dtype=np.float32)

        peaks = _pick_peaks(scores, min_distance_frames=2, limit=4, threshold=0.5)

        self.assertEqual(peaks, [1, 3, 5])

    def test_merge_highlight_indices_keeps_dense_beats_without_duplicates(self) -> None:
        merged = _merge_highlight_indices(
            primary_indices=[10, 30],
            secondary_indices=[9, 11, 30, 31],
            scores=np.asarray(
                [0.0] * 9 + [0.7, 1.0, 0.6] + [0.0] * 18 + [0.9, 0.8] + [0.0] * 9,
                dtype=np.float32,
            ),
            min_distance_frames=2,
        )

        self.assertIn(10, merged)
        self.assertIn(30, merged)
        self.assertNotIn(9, merged)
        self.assertNotIn(11, merged)
        self.assertNotIn(31, merged)

    def test_records_from_indices_builds_timed_records(self) -> None:
        records = _records_from_indices(
            indices=[2, 4],
            seconds_per_frame=0.5,
            scores=np.asarray([0.0, 0.0, 1.2, 0.0, 0.9], dtype=np.float32),
            energy=np.asarray([0.0, 0.0, 0.3, 0.0, 0.4], dtype=np.float32),
            accent=np.asarray([0.0, 0.0, 0.5, 0.0, 0.6], dtype=np.float32),
        )

        self.assertEqual([record.time for record in records], [1.0, 2.0])
        self.assertEqual([record.score for record in records], [1.2, 0.9])


if __name__ == "__main__":
    unittest.main()
