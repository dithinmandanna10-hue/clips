"""yt-dlp wrapper: fetch metadata + subtitles, download the source video."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .transcript import Segment, parse_any


@dataclass
class VideoMeta:
    id: str
    url: str
    title: str
    duration: float
    thumbnail: str = ""
    uploader: str = ""
    segments: list[Segment] = field(default_factory=list)
    transcript_source: str = ""  # "manual" | "auto" | ""


def _ydl(opts: dict):
    import yt_dlp

    return yt_dlp.YoutubeDL(opts)


def fetch_meta(url: str, workdir: str, lang: str = "en") -> VideoMeta:
    """Fetch metadata + best-effort subtitles without downloading video."""
    os.makedirs(workdir, exist_ok=True)
    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": [f"{lang}", f"{lang}.*", f"{lang}-*"],
        "subtitlesformat": "vtt/json3/srt/best",
        "outtmpl": os.path.join(workdir, "%(id)s.%(ext)s"),
    }
    with _ydl(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    vid = info.get("id") or "unknown"
    meta = VideoMeta(
        id=vid,
        url=url,
        title=info.get("title") or vid,
        duration=float(info.get("duration") or 0),
        thumbnail=info.get("thumbnail") or "",
        uploader=info.get("uploader") or info.get("channel") or "",
    )

    # locate downloaded subtitle files
    subs = info.get("requested_subtitles") or {}
    sub_path = ""
    source = ""
    for key in (f"{lang}", f"{lang}.orig", f"{lang}-orig"):
        if key in subs and subs[key].get("filepath"):
            sub_path = subs[key]["filepath"]
            source = "manual" if not info.get("automatic_captions") else ""
            break
    if not sub_path:
        for key, val in subs.items():
            if val.get("filepath"):
                sub_path = val["filepath"]
                break
    if sub_path and os.path.exists(sub_path):
        try:
            text = Path(sub_path).read_text(encoding="utf-8", errors="replace")
            meta.segments = parse_any(text, os.path.basename(sub_path))
            # distinguish auto vs manual: yt-dlp marks automatic ones in info
            auto_langs = set((info.get("automatic_captions") or {}).keys())
            man_langs = set((info.get("subtitles") or {}).keys())
            chosen = os.path.basename(sub_path).rsplit(".", 2)[0]
            if chosen in man_langs and chosen not in auto_langs:
                meta.transcript_source = "manual"
            else:
                meta.transcript_source = "auto"
        except Exception:
            meta.segments = []
    return meta


def download_video(
    url: str, workdir: str, on_progress=None, max_height: int = 1080
) -> str:
    """Download the video (mp4 preferred) into workdir. Returns file path."""
    os.makedirs(workdir, exist_ok=True)
    hooks = []

    def _hook(d):
        if on_progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            done = d.get("downloaded_bytes") or 0
            if total:
                on_progress(min(0.99, done / total))
        elif on_progress and d.get("status") == "finished":
            on_progress(1.0)

    hooks.append(_hook)
    opts = {
        "quiet": True,
        "no_warnings": True,
        "format": (
            f"bv*[height<={max_height}][ext=mp4]+ba[ext=m4a]/"
            f"b[height<={max_height}][ext=mp4]/"
            f"bv*[height<={max_height}]+ba/b"
        ),
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(workdir, "%(id)s.%(ext)s"),
        "progress_hooks": hooks,
        "noplaylist": True,
    }
    with _ydl(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    vid = info.get("id") or "unknown"
    path = os.path.join(workdir, f"{vid}.mp4")
    if not os.path.exists(path):
        # fall back: newest file for this id
        cands = sorted(
            (os.path.join(workdir, f) for f in os.listdir(workdir)
             if f.startswith(vid)),
            key=os.path.getmtime,
        )
        if not cands:
            raise RuntimeError("download finished but file not found")
        path = cands[-1]
    return path


_URL_ID_RE = re.compile(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{6,})")


def looks_like_youtube(url: str) -> bool:
    return bool(_URL_ID_RE.search(url)) or "youtube.com" in url
