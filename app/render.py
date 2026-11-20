"""Render a long-form source segment into a 9:16 short.

Layout (stream mode): content cover-cropped to fill 1080x1920, the original
facecam spot blurred out if it survives the crop, the extracted facecam
repositioned as a small bordered window top-right, captions burned in from the
transcript, audio loudness-normalized.
"""
from __future__ import annotations

import os
import subprocess
from typing import Callable

from .captions import build_ass
from .facecam import Rect
from .ffmpeg_utils import extract_frame, ffmpeg_exe, probe, run_ffmpeg_progress
from .transcript import Segment

OUT_W, OUT_H = 1080, 1920
FONTS_DIR = "/usr/share/fonts/truetype/dejavu"


def _esc_filter_path(path: str) -> str:
    """Escape a filename for use inside an ffmpeg filter argument."""
    out = path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    out = out.replace(",", "\\,").replace("[", "\\[").replace("]", "\\]")
    return out


def _cover_transform(src_w: int, src_h: int) -> tuple[float, int, int]:
    """Scale factor and crop offsets for cover-fitting src into OUT_W x OUT_H."""
    s = max(OUT_W / src_w, OUT_H / src_h)
    cx = (src_w * s - OUT_W) / 2.0
    cy = (src_h * s - OUT_H) / 2.0
    return s, cx, cy


def render_clip(
    src: str,
    start: float,
    end: float,
    out_path: str,
    segments: list[Segment] | None = None,
    facecam: Rect | None = None,
    cam_width_frac: float = 0.30,
    cam_margin: int = 24,
    captions: bool = True,
    on_progress: Callable[[float], None] | None = None,
) -> dict:
    """Render source [start, end] to a vertical short at out_path.

    Returns {"path", "poster"}.
    """
    info = probe(src)
    src_w, src_h = info["width"], info["height"]
    duration = end - start
    if duration < 0.5:
        raise ValueError("clip too short (need >= 0.5s)")
    start = max(0.0, start)
    end = min(end, info["duration"] or end)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # --- where does the original facecam land after cover-crop? -------------
    blur_region: tuple[int, int, int, int] | None = None
    s, cx_off, cy_off = _cover_transform(src_w, src_h)
    if facecam is not None:
        mx = facecam.x * s - cx_off
        my = facecam.y * s - cy_off
        mw = facecam.w * s
        mh = facecam.h * s
        # intersect with output canvas
        ix0, iy0 = max(0.0, mx), max(0.0, my)
        ix1, iy1 = min(float(OUT_W), mx + mw), min(float(OUT_H), my + mh)
        if ix1 - ix0 >= 8 and iy1 - iy0 >= 8:
            # only blur if a meaningful part survives the crop
            vis = (ix1 - ix0) * (iy1 - iy0) / max(1.0, mw * mh)
            if vis >= 0.15:
                bx, by = int(ix0), int(iy0)
                bw = int(ix1 - ix0)
                bh = int(iy1 - iy0)
                # boxblur needs even-ish sizes; clamp into frame
                bw = min(bw, OUT_W - bx)
                bh = min(bh, OUT_H - by)
                if bw >= 8 and bh >= 8:
                    blur_region = (bx, by, bw, bh)

    # --- filter graph --------------------------------------------------------
    fc: list[str] = []
    fc.append(
        f"[0:v]split=2[srcmain][srccam]"
        if facecam is not None
        else "[0:v]null[srcmain]"
    )
    fc.append(
        f"[srcmain]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
        f"crop={OUT_W}:{OUT_H},setsar=1[covered]"
    )

    if blur_region is not None:
        bx, by, bw, bh = blur_region
        fc.append("[covered]split=2[c0][c1]")
        fc.append(f"[c0]crop={bw}:{bh}:{bx}:{by},boxblur=22:3[blurred]")
        fc.append(f"[c1][blurred]overlay={bx}:{by}[bg]")
        bg_label = "[bg]"
    else:
        fc.append("[covered]null[bg]")
        bg_label = "[bg]"

    vlabel = bg_label
    if facecam is not None:
        cw = max(64, int(OUT_W * cam_width_frac))
        fc.append(
            f"[srccam]crop={facecam.w}:{facecam.h}:{facecam.x}:{facecam.y},"
            f"scale={cw}:-2,setsar=1,"
            f"drawbox=x=0:y=0:w=iw:h=ih:color=white@0.92:t=6[cam]"
        )
        ox = OUT_W - cw - cam_margin
        oy = cam_margin
        fc.append(f"{bg_label}[cam]overlay={ox}:{oy}[withcam]")
        vlabel = "[withcam]"

    # captions
    if captions and segments:
        ass_path = os.path.splitext(out_path)[0] + ".ass"
        build_ass(segments, start, end, ass_path, width=OUT_W, height=OUT_H)
        fc.append(
            f"{vlabel}ass={_esc_filter_path(ass_path)}"
            f":fontsdir={_esc_filter_filter_path(FONTS_DIR)}[vout]"
        )
        vlabel = "[vout]"
    else:
        fc.append(f"{vlabel}null[vout]")
        vlabel = "[vout]"

    has_audio = info.get("has_audio", False)
    if has_audio:
        fc.append(
            "[0:a]loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000[aout]"
        )

    cmd = [
        ffmpeg_exe(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        src,
        "-filter_complex",
        ";".join(fc),
        "-map",
        "[vout]",
    ]
    if has_audio:
        cmd += ["-map", "[aout]", "-c:a", "aac", "-b:a", "160k"]
    else:
        cmd += ["-an"]
    cmd += [
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-progress",
        "pipe:1",
        "-nostats",
        out_path,
    ]

    run_ffmpeg_progress(cmd, expected_seconds=duration, on_progress=on_progress)

    # poster frame at the middle of the clip
    poster = os.path.splitext(out_path)[0] + ".jpg"
    try:
        extract_frame(out_path, min(1.0, duration / 2.0), poster)
    except Exception:
        poster = ""

    return {"path": out_path, "poster": poster}


def _esc_filter_filter_path(path: str) -> str:
    return _esc_filter_path(path)
