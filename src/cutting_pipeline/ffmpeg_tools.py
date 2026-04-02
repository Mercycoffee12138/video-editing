from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Generator

import numpy as np


def run_command(command: list[str]) -> None:
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{message}")


def run_ffprobe_json(command: list[str]) -> dict:
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout)


def get_media_duration(path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return float(result.stdout.strip())


def iter_gray_frames(
    path: Path,
    fps: int,
    width: int,
    height: int,
) -> Generator[np.ndarray, None, None]:
    frame_size = width * height
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vf",
        f"fps={fps},scale={width}:{height}:flags=lanczos,format=gray",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=frame_size * 8,
    )
    assert process.stdout is not None
    try:
        while True:
            chunk = process.stdout.read(frame_size)
            if len(chunk) < frame_size:
                break
            yield np.frombuffer(chunk, dtype=np.uint8).reshape(height, width)
    finally:
        stderr = b""
        if process.stderr is not None:
            stderr = process.stderr.read()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(stderr.decode("utf-8", errors="replace").strip())


def decode_audio_mono(path: Path, sample_rate: int) -> np.ndarray:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "-",
    ]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode != 0:
        raise RuntimeError(process.stderr.decode("utf-8", errors="replace").strip())
    samples = np.frombuffer(process.stdout, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return np.zeros(1, dtype=np.float32)
    return samples / 32768.0


def export_video_frame(path: Path, timestamp: float, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(timestamp, 0.0):.3f}",
        "-i",
        str(path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        "-y",
        str(output_path),
    ]
    run_command(command)
