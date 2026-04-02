import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cutting_pipeline.config import MatchConfig
from cutting_pipeline.stage_04_match_segments import (
    assign_clips,
    build_timeline_chunks,
    build_timeline_durations,
    enrich_selected_highlights,
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

    def test_assign_clips_reuses_segments_when_timeline_is_longer_than_candidates(self) -> None:
        config = MatchConfig(source_reuse_penalty=0.25)
        fight_segments = [
            {
                "source_path": "source/video/001.mp4",
                "trimmed_path": "build/stage_01_trimmed_videos/001.mp4",
                "video_duration": 100.0,
                "peak_time": 10.0,
                "score": 2.0,
            },
            {
                "source_path": "source/video/002.mp4",
                "trimmed_path": "build/stage_01_trimmed_videos/002.mp4",
                "video_duration": 100.0,
                "peak_time": 20.0,
                "score": 1.5,
            },
        ]

        clips = assign_clips(
            fight_segments,
            [],
            [
                {"duration": 3.0, "target_intensity": 0.2},
                {"duration": 3.0, "target_intensity": 0.4},
                {"duration": 3.0, "target_intensity": 0.8},
                {"duration": 3.0, "target_intensity": 0.9},
            ],
            config,
        )

        self.assertEqual(len(clips), 4)
        self.assertEqual({clip.source_path for clip in clips[:2]}, {"source/video/001.mp4", "source/video/002.mp4"})
        self.assertGreater(clips[2].segment_score, clips[0].segment_score)

    def test_assign_clips_prefers_calm_pool_for_low_intensity_chunks(self) -> None:
        config = MatchConfig(source_reuse_penalty=0.25)
        fight_segments = [
            {
                "source_path": "source/video/fight.mp4",
                "trimmed_path": "build/stage_01_trimmed_videos/fight.mp4",
                "video_duration": 100.0,
                "peak_time": 30.0,
                "score": 2.5,
            },
        ]
        calm_segments = [
            {
                "source_path": "source/video/calm.mp4",
                "trimmed_path": "build/stage_01_trimmed_videos/calm.mp4",
                "video_duration": 100.0,
                "peak_time": 15.0,
                "score": 1.8,
            },
        ]

        clips = assign_clips(
            fight_segments,
            calm_segments,
            [
                {"duration": 4.0, "target_intensity": 0.2},
                {"duration": 4.0, "target_intensity": 0.9},
            ],
            config,
        )

        self.assertEqual(clips[0].source_path, "source/video/calm.mp4")
        self.assertEqual(clips[1].source_path, "source/video/fight.mp4")

    def test_assign_clips_prefers_fight_pool_once_intensity_crosses_threshold(self) -> None:
        config = MatchConfig(source_reuse_penalty=0.25)
        fight_segments = [
            {
                "source_path": "source/video/fight.mp4",
                "trimmed_path": "build/stage_01_trimmed_videos/fight.mp4",
                "video_duration": 100.0,
                "peak_time": 30.0,
                "score": 2.5,
            },
        ]
        calm_segments = [
            {
                "source_path": "source/video/calm.mp4",
                "trimmed_path": "build/stage_01_trimmed_videos/calm.mp4",
                "video_duration": 100.0,
                "peak_time": 15.0,
                "score": 1.8,
            },
        ]

        clips = assign_clips(
            fight_segments,
            calm_segments,
            [
                {"duration": 2.6, "target_intensity": 0.53},
            ],
            config,
        )

        self.assertEqual(clips[0].source_path, "source/video/fight.mp4")

    def test_build_timeline_chunks_raises_intensity_near_highlights(self) -> None:
        config = MatchConfig(min_clip_seconds=0.8, max_clip_seconds=4.0)
        selected_highlights = [
            {"time": 10.0, "score": 0.8, "energy": 0.1, "accent": 0.1},
            {"time": 22.0, "score": 2.0, "energy": 0.1, "accent": 0.1},
        ]

        chunks = build_timeline_chunks(0.0, 28.0, selected_highlights, config)

        self.assertTrue(chunks)
        early_intensity = chunks[0]["target_intensity"]
        late_peak_intensity = max(chunk["target_intensity"] for chunk in chunks if chunk["start"] >= 18.0)
        self.assertGreater(late_peak_intensity, early_intensity)

    def test_build_timeline_chunks_shortens_cuts_near_peaks(self) -> None:
        config = MatchConfig(min_clip_seconds=0.8, max_clip_seconds=4.8)
        selected_highlights = [
            {"time": 12.0, "score": 0.7, "energy": 0.1, "accent": 0.1},
            {"time": 24.0, "score": 2.2, "energy": 0.1, "accent": 0.1},
        ]

        chunks = build_timeline_chunks(0.0, 30.0, selected_highlights, config)

        early_durations = [chunk["duration"] for chunk in chunks if chunk["end"] <= 10.0]
        peak_durations = [chunk["duration"] for chunk in chunks if 20.0 <= chunk["start"] <= 26.0]
        self.assertTrue(early_durations)
        self.assertTrue(peak_durations)
        self.assertLess(min(peak_durations), max(early_durations))

    def test_enrich_selected_highlights_keeps_strong_tail_peaks_for_full_track(self) -> None:
        config = MatchConfig(use_full_track_duration=True)
        track = {
            "duration": 100.0,
            "highlights": [
                {"time": 20.0, "score": 1.0, "energy": 0.1, "accent": 0.1},
                {"time": 40.0, "score": 1.1, "energy": 0.1, "accent": 0.1},
                {"time": 84.0, "score": 1.4, "energy": 0.1, "accent": 0.1},
                {"time": 97.0, "score": 2.1, "energy": 0.1, "accent": 0.1},
            ],
        }
        selected = [
            {"time": 20.0, "score": 1.0, "energy": 0.1, "accent": 0.1},
            {"time": 40.0, "score": 1.1, "energy": 0.1, "accent": 0.1},
        ]

        enriched = enrich_selected_highlights(track, selected, config)

        self.assertIn(97.0, [item["time"] for item in enriched])

    def test_build_timeline_chunks_never_exceeds_max_clip_duration(self) -> None:
        config = MatchConfig(min_clip_seconds=0.8, max_clip_seconds=4.8)
        selected_highlights = [
            {"time": 33.065, "score": 1.658, "energy": 0.1, "accent": 0.1},
            {"time": 38.638, "score": 1.794, "energy": 0.1, "accent": 0.1},
            {"time": 46.022, "score": 1.602, "energy": 0.1, "accent": 0.1},
            {"time": 62.183, "score": 2.307, "energy": 0.1, "accent": 0.1},
            {"time": 66.27, "score": 2.198, "energy": 0.1, "accent": 0.1},
            {"time": 81.038, "score": 1.92, "energy": 0.1, "accent": 0.1},
            {"time": 86.703, "score": 1.626, "energy": 0.1, "accent": 0.1},
            {"time": 92.183, "score": 1.65, "energy": 0.1, "accent": 0.1},
            {"time": 106.951, "score": 2.526, "energy": 0.1, "accent": 0.1},
        ]

        chunks = build_timeline_chunks(0.0, 107.52, selected_highlights, config)

        self.assertTrue(chunks)
        self.assertTrue(all(chunk["duration"] <= 4.8 for chunk in chunks))

    def test_build_timeline_chunks_accelerates_sustained_late_high_intensity_sections(self) -> None:
        config = MatchConfig(min_clip_seconds=0.8, max_clip_seconds=4.8)
        selected_highlights = [
            {"time": 62.183, "score": 2.307, "energy": 0.1, "accent": 0.1},
            {"time": 66.27, "score": 2.198, "energy": 0.1, "accent": 0.1},
            {"time": 81.038, "score": 1.92, "energy": 0.1, "accent": 0.1},
            {"time": 86.703, "score": 1.626, "energy": 0.1, "accent": 0.1},
            {"time": 92.183, "score": 1.65, "energy": 0.1, "accent": 0.1},
            {"time": 106.951, "score": 2.526, "energy": 0.1, "accent": 0.1},
        ]

        chunks = build_timeline_chunks(0.0, 107.52, selected_highlights, config)

        target_window = [
            chunk["duration"]
            for chunk in chunks
            if chunk["end"] >= 72.0 and chunk["start"] <= 85.0
        ]
        self.assertTrue(target_window)
        self.assertLess(max(target_window), 4.0)


if __name__ == "__main__":
    unittest.main()
