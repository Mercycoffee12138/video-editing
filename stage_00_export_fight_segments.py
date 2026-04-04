#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cutting_pipeline.ffmpeg_tools import export_video_clip, run_command
from cutting_pipeline.json_io import read_json, write_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export detected fight segments for manual review.")
    parser.add_argument(
        "--artifact",
        help=(
            "Optional stage artifact path. Defaults to build/stage_02_reviewed_fight_segments.json "
            "and falls back to build/stage_02_fight_segments.json."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional export limit. Use 0 to export all segments.",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Optional output directory relative to project root. "
            "Defaults to build/stage_02_fight_segment_exports."
        ),
    )
    return parser.parse_args()


def _default_artifact_path(project_root: Path) -> Path:
    reviewed = project_root / "build" / "stage_02_reviewed_fight_segments.json"
    if reviewed.exists():
        return reviewed
    return project_root / "build" / "stage_02_fight_segments.json"


def _load_segments(artifact_path: Path) -> list[dict]:
    payload = read_json(artifact_path)
    segments = list(payload.get("top_segments") or [])
    if not segments:
        raise ValueError(f"No top_segments were found in {artifact_path}.")
    return segments


def _segment_clip_name(index: int, segment: dict) -> str:
    source_name = Path(segment["trimmed_path"]).stem
    start = float(segment["start"])
    end = float(segment["end"])
    return f"{index:03d}_{source_name}_{start:.2f}-{end:.2f}.mp4"


def _concat_exports(exported_paths: list[Path], output_path: Path) -> None:
    concat_list_path = output_path.parent / "fight_segments_concat_list.txt"
    concat_lines = [f"file '{path.resolve()}'" for path in exported_paths]
    concat_list_path.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")
    run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list_path),
            "-c",
            "copy",
            str(output_path),
        ]
    )


def main() -> None:
    args = _parse_args()
    artifact_path = Path(args.artifact) if args.artifact else _default_artifact_path(PROJECT_ROOT)
    if not artifact_path.is_absolute():
        artifact_path = (PROJECT_ROOT / artifact_path).resolve()
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact was not found: {artifact_path}")

    segments = _load_segments(artifact_path)
    if args.limit > 0:
        segments = segments[: args.limit]

    export_dir = (
        (PROJECT_ROOT / args.output_dir).resolve()
        if args.output_dir
        else PROJECT_ROOT / "build" / "stage_02_fight_segment_exports"
    )
    export_dir.mkdir(parents=True, exist_ok=True)

    exported_paths: list[Path] = []
    manifest_segments: list[dict] = []
    for index, segment in enumerate(segments, start=1):
        input_path = PROJECT_ROOT / segment["trimmed_path"]
        output_path = export_dir / _segment_clip_name(index, segment)
        export_video_clip(
            input_path=input_path,
            output_path=output_path,
            start_time=float(segment["start"]),
            duration=float(segment["end"]) - float(segment["start"]),
        )
        exported_paths.append(output_path)

        segment_payload = dict(segment)
        segment_payload["export_path"] = str(output_path.relative_to(PROJECT_ROOT))
        manifest_segments.append(segment_payload)
        print(f"[{index}/{len(segments)}] exported {output_path.relative_to(PROJECT_ROOT)}")

    preview_path = export_dir / "fight_segments_preview.mp4"
    if exported_paths:
        _concat_exports(exported_paths, preview_path)

    manifest_path = export_dir / "fight_segments_manifest.json"
    write_json(
        manifest_path,
        {
            "artifact_path": str(artifact_path.relative_to(PROJECT_ROOT)),
            "segment_count": len(manifest_segments),
            "preview_path": str(preview_path.relative_to(PROJECT_ROOT)) if exported_paths else None,
            "segments": manifest_segments,
        },
    )

    print(f"Exported {len(manifest_segments)} fight segments to {export_dir.relative_to(PROJECT_ROOT)}")
    if exported_paths:
        print(f"Preview video: {preview_path.relative_to(PROJECT_ROOT)}")
    print(f"Manifest: {manifest_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
