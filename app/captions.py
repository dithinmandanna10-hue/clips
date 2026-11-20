"""Build ASS subtitle files for burned-in shorts captions."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .transcript import Segment, Word, from_segments

ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
ScaledBorderAndShadow: yes
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Short,DejaVu Sans,{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{outline},0,2,70,70,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _fmt_time(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _escape(text: str) -> str:
    text = text.replace("{", "(").replace("}", ")")
    text = text.replace("\\", "/")
    return text


@dataclass
class CaptionChunk:
    start: float
    end: float
    text: str


def chunk_words(
    words: list[tuple[str, float, float]],
    max_words: int = 4,
    max_dur: float = 1.7,
    gap: float = 0.5,
) -> list[CaptionChunk]:
    """Group (text, start, end) tuples into short on-screen phrases."""
    chunks: list[CaptionChunk] = []
    cur: list[tuple[str, float, float]] = []

    def flush():
        if not cur:
            return
        text = " ".join(w[0] for w in cur).strip()
        text = re.sub(r"\s+", " ", text)
        if text:
            chunks.append(CaptionChunk(cur[0][1], cur[-1][2], text))
        cur.clear()

    for i, (text, ws, we) in enumerate(words):
        if cur:
            prev_end = cur[-1][2]
            dur = we - cur[0][1]
            too_many = len(cur) >= max_words
            too_long = dur >= max_dur
            too_gap = ws - prev_end > gap
            if too_many or too_long or too_gap:
                flush()
        cur.append((text, ws, we))
        # hard break after sentence-ending punctuation
        if re.search(r"[.!?…][\"')\]]?$", text) and len(cur) >= 1:
            flush()
    flush()
    return chunks


def _words_from_segments(
    segments: list[Segment], start: float, end: float
) -> list[tuple[str, float, float]]:
    clipped = from_segments(segments, start, end)
    out: list[tuple[str, float, float]] = []
    for seg in clipped:
        if seg.words:
            for w in seg.words:
                if w.start >= start - 0.01 and w.end <= end + 0.01:
                    out.append((w.text, max(0.0, w.start - start), min(end - start, w.end - start)))
        else:
            # even distribution across the segment
            toks = seg.text.split()
            if not toks:
                continue
            s = max(seg.start, start) - start
            e = min(seg.end, end) - start
            step = (e - s) / len(toks)
            for i, tok in enumerate(toks):
                out.append((tok, s + i * step, s + (i + 1) * step))
    return out


def build_ass(
    segments: list[Segment],
    start: float,
    end: float,
    out_path: str,
    width: int = 1080,
    height: int = 1920,
    font_size: int = 74,
    outline: int = 5,
    margin_v: int = 560,
) -> str:
    """Write an ASS file with phrase-chunked captions for [start, end]."""
    words = _words_from_segments(segments, start, end)
    chunks = chunk_words(words) if words else []

    lines = [
        ASS_HEADER.format(
            w=width, h=height, size=font_size, outline=outline, margin_v=margin_v
        )
    ]
    for ch in chunks:
        if ch.end - ch.start < 0.05:
            continue
        lines.append(
            f"Dialogue: 0,{_fmt_time(ch.start)},{_fmt_time(ch.end)},Short,,0,0,0,,"
            f"{_escape(ch.text)}\n"
        )
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return out_path
