"""FFmpeg detection and subprocess wrapper."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List, Optional


def ffmpeg_available() -> bool:
    """Return True if ``ffmpeg`` is found on PATH."""
    return shutil.which("ffmpeg") is not None


def ffprobe_duration(video_path: Path) -> Optional[float]:
    """Return duration in seconds via ffprobe, or None on failure."""
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            stderr=subprocess.DEVNULL,
        )
        return float(out.decode().strip())
    except Exception:
        return None


def run_encode(
    input_path: Path,
    output_path: Path,
    codec: str = "libx264",
    crf: int = 23,
    extra_args: Optional[List[str]] = None,
) -> subprocess.CompletedProcess:
    """Run an ffmpeg encode and return the CompletedProcess."""
    cmd: List[str] = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-c:v", codec,
        "-crf", str(crf),
        "-preset", "fast",
        "-an",  # drop audio for speed
    ]
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(str(output_path))
    return subprocess.run(cmd, capture_output=True, text=True, check=True)
