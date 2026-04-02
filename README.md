# Stage-Based Video Editing Pipeline

This project builds a maintainable editing pipeline around the assets in `source/video` and `source/music`.

## What it does

1. `stage_01_trim_videos`
   Trims the beginning and ending of each source video.
2. `stage_02_detect_fight_segments`
   Detects high-motion segments from the trimmed videos and saves them to JSON.
3. `stage_03_detect_music_highlights`
   Detects highlight points from every music track and saves them to JSON.
4. `stage_04_match_segments`
   Matches the fight segments against music highlights and selects the best plan.
5. `stage_05_render_final_video`
   Renders the final edited video.

Each stage prints the overall pipeline progress as a percentage while it runs.

## Project structure

```text
stage_00_run_pipeline.py
src/cutting_pipeline/
  config.py
  ffmpeg_tools.py
  json_io.py
  models.py
  pipeline.py
  progress.py
  stage_01_trim_videos.py
  stage_02_detect_fight_segments.py
  stage_02_review_fight_segments.py
  stage_03_detect_music_highlights.py
  stage_04_match_segments.py
  stage_05_render_final_video.py
tests/
  test_stage_02_review_fight_segments.py
  test_stage_04_match_segments.py
build/
  stage_01_trim_manifest.json
  stage_02_fight_segments.json
  stage_02_reviewed_fight_segments.json
  stage_03_music_highlights.json
  stage_04_match_plan.json
  stage_05_final_video.mp4
```

## Run

```bash
python3 stage_00_run_pipeline.py
```

If you want to force the pipeline to use one specific music file, set
`MatchConfig.selected_music_filename` in [config.py](/Users/wz/Desktop/cutting/src/cutting_pipeline/config.py)
to a file name from `source/music`, for example `008.MP3`.

If you want to keep highlight-based cutting but use the full selected song,
leave `MatchConfig.use_full_track_duration = True`.

If you want to continue from an existing stage output, run with `--start-stage`.

```bash
python3 stage_00_run_pipeline.py --start-stage stage_03_detect_music_highlights
```

This will reuse the earlier JSON artifacts in `build/` and continue from stage 03.

## Vision Review

Stage `stage_02_review_fight_segments` uses Qwen VL through the Zhizengzeng
Alibaba proxy to review the top detected motion segments before music matching.
It extracts three frames from each candidate clip and asks the model whether
the scene really looks like a fight or confrontation.

Set your API key before running the pipeline:

```bash
export ZZZ_API_KEY="your_zhizengzeng_key"
```

If no API key is set, the review stage is skipped automatically and the pipeline
falls back to the motion-only fight segments.

## Main outputs

- `build/stage_01_trimmed_videos/`
- `build/stage_01_trim_manifest.json`
- `build/stage_02_fight_segments.json`
- `build/stage_02_reviewed_fight_segments.json`
- `build/stage_03_music_highlights.json`
- `build/stage_04_match_plan.json`
- `build/stage_05_final_video.mp4`

## Notes

- Stage 01 uses fast GOP-aligned trimming with `ffmpeg -c copy` to keep preprocessing efficient.
- Stage 02 uses low-resolution grayscale frame differences to estimate motion intensity.
- Stage 02 review uses Qwen VL to reject high-motion clips that do not visually look like fights.
- Stage 03 uses PCM energy changes to detect music highlight points without extra Python audio packages.
- Stage 05 re-encodes only the short selected clips for the final output.
