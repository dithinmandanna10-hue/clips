"""ffmpeg/ffprobe helpers: locate the binary, probe media, run with progress."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
from functools import lru_cache

_lock = threading.Lock()
_audio_cache: dict[str, bool] = {}


def ffmpeg_exe() -> str:
    sys_ff = shutil.which("ffmpeg")
    if sys_ff:
        return sys_ff
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _run(cmd: list[str], timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout
    )


@lru_cache(maxsize=1)
def probe(path: str) -> dict:
    """Return {width, height, fps, duration, has_audio} using cv2 + ffmpeg."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    duration = n / fps if n > 0 else 0.0
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration,
        "has_audio": has_audio(path),
    }


def has_audio(path: str) -> bool:
    with _lock:
        if path in _audio_cache:
            return _audio_cache[path]
    ff = ffmpeg_exe()
    try:
        proc = _run([ff, "-hide_banner", "-i", path], timeout=30)
        txt = proc.stderr.decode("utf-8", "replace")
        ok = "Audio:" in txt
    except Exception:
        ok = False
    with _lock:
        _audio_cache[path] = ok
    return ok


def extract_frame(path: str, t: float, out_jpg: str) -> str:
    """Extract a single JPEG frame at time t (seconds)."""
    ff = ffmpeg_exe()
    os.makedirs(os.path.dirname(out_jpg) or ".", exist_ok=True)
    proc = _run(
        [
            ff,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{max(0.0, t):.3f}",
            "-i",
            path,
            "-frames:v",
            "1",
            "-q:v",
            "3",
            out_jpg,
        ],
        timeout=60,
    )
    if proc.returncode != 0 or not os.path.exists(out_jpg):
        raise RuntimeError(
            "frame extract failed: " + proc.stderr.decode("utf-8", "replace")[-400:]
        )
    return out_jpg


_TIME_RE = re.compile(r"out_time=(\d+):(\d{2}):(\d{2}\.\d+)")


def run_ffmpeg_progress(
    cmd: list[str],
    expected_seconds: float,
    on_progress=None,
    timeout: int = 3600,
) -> None:
    """Run an ffmpeg command, parsing `-progress pipe:1` on stdout.

    `cmd` must already include ['-progress', 'pipe:1', '-nostats'] and end with
    the output path. on_progress(frac 0..1) is called periodically.
    """
    env = dict(os.environ)
    env["AV_LOG_FORCE_NOCOLOR"] = "1"
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        env=env,
        text=True,
        errors="replace",
    )
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            m = _TIME_RE.search(line)
            if m and on_progress and expected_seconds > 0:
                secs = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
                frac = min(0.99, max(0.0, secs / expected_seconds))
                on_progress(frac)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise
    if proc.returncode != 0:
        err = ""
        if proc.stderr is not None:
            # read remaining stderr (bounded)
            try:
                err = proc.stderr.read()[-2000:]
            except Exception:
                err = ""
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {err}")


def duration_of(path: str) -> float:
    return probe(path)["duration"]


def media_json(path: str) -> dict:
    """Unused generic helper kept for future ffprobe-based probing."""
    ff = ffmpeg_exe()
    proc = _run([ff, "-hide_banner", "-i", path], timeout=30)
    return {"stderr": proc.stderr.decode("utf-8", "replace")}
