import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cutting_pipeline.stage_02_detect_fight_segments import (  # noqa: E402
    _calm_segment_score,
    _exposure_balance,
    _frame_quality_metrics,
    _parse_coarse_review_json,
)


class DetectFightSegmentsTests(unittest.TestCase):
    def test_frame_quality_metrics_prefers_sharper_frame(self) -> None:
        blurry = np.full((8, 8), 128, dtype=np.uint8)
        detailed = np.zeros((8, 8), dtype=np.uint8)
        detailed[:, ::2] = 255

        blurry_sharpness, _, _ = _frame_quality_metrics(blurry)
        detailed_sharpness, _, _ = _frame_quality_metrics(detailed)

        self.assertGreater(detailed_sharpness, blurry_sharpness)

    def test_exposure_balance_penalizes_extreme_brightness(self) -> None:
        scores = _exposure_balance(np.asarray([0.05, 0.5, 0.95], dtype=np.float32))

        self.assertLess(scores[0], scores[1])
        self.assertLess(scores[2], scores[1])

    def test_calm_segment_score_rewards_clear_stable_frames(self) -> None:
        strong = _calm_segment_score(
            duration_ratio=0.9,
            calmness=0.9,
            stability=0.95,
            sharpness=0.85,
            contrast=0.7,
            exposure=0.9,
        )
        weak = _calm_segment_score(
            duration_ratio=0.9,
            calmness=0.9,
            stability=0.95,
            sharpness=0.1,
            contrast=0.1,
            exposure=0.2,
        )

        self.assertGreater(strong, weak)

    def test_parse_coarse_review_json_clamps_ratios(self) -> None:
        parsed = _parse_coarse_review_json(
            '{"contains_fight": true, "confidence": 0.81, '
            '"active_start_ratio": -0.2, "active_end_ratio": 1.6, "summary": "两人近身互殴"}'
        )

        self.assertTrue(parsed["contains_fight"])
        self.assertEqual(parsed["confidence"], 0.81)
        self.assertEqual(parsed["active_start_ratio"], 0.0)
        self.assertEqual(parsed["active_end_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
