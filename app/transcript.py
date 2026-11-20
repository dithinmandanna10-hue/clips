"""Parse subtitles (YouTube VTT / SRT / json3) into timed segments with words."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


_TS_RE = re.compile(r"(?:(\d{1,2}):)?(\d{1,2}):(\d{1,2})[.,](\d{1,3})")
_INLINE_TS_RE = re.compile(r"<((?:\d{1,2}:)?\d{1,2}:\d{1,2}[.,]\d{1,3})>")
_TAG_RE = re.compile(r"<[^>]+>")


def _ts_to_sec(h: str | None, m: str, s: str, ms: str) -> float:
    ms = (ms + "000")[:3]
    hours = int(h) if h else 0
    return hours * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _cue_words(text: str, cue_start: float, cue_end: float) -> list[Word]:
    """Split cue text into words with per-word times.

    Handles YouTube's inline timestamps `<00:00:01.234>`: each timestamp applies
    to the text that follows it. Without inline timings, words are distributed
    evenly across the cue.
    """
    parts = _INLINE_TS_RE.split(text)
    # parts: [pre, ts1, chunk1, ts2, chunk2, ...] (pre may be '')
    timed: list[tuple[float | None, str]] = []
    if len(parts) == 1:
        timed.append((None, parts[0]))
    else:
        if parts[0].strip():
            timed.append((None, parts[0]))
        for i in range(1, len(parts) - 1, 2):
            ts_raw = parts[i]
            m = _TS_RE.match(ts_raw)
            t = _ts_to_sec(*m.groups()) if m else None
            timed.append((t, parts[i + 1]))

    words: list[tuple[str, float]] = []  # (text, nominal_start)
    running = cue_start
    for t, chunk in timed:
        if t is not None:
            running = max(cue_start, min(t, cue_end))
        clean = _TAG_RE.sub(" ", chunk)
        for w in clean.split():
            w = w.strip()
            if w:
                words.append((w, running))
                running += 0.0  # nominal start shared within chunk; end filled below

    if not words:
        return []
    # Assign ends: next word's nominal start (or spread evenly if all equal).
    starts = [ws for _, ws in words]
    if len(set(starts)) <= 1:  # no usable inline timing -> even distribution
        n = len(words)
        step = (cue_end - cue_start) / n
        return [
            Word(wt, cue_start + i * step, cue_start + (i + 1) * step)
            for i, (wt, _) in enumerate(words)
        ]
    out: list[Word] = []
    for i, (wt, ws) in enumerate(words):
        we = words[i + 1][1] if i + 1 < len(words) else cue_end
        we = max(ws + 0.05, min(we, cue_end))
        if i == 0:
            ws = min(ws, cue_start)
        out.append(Word(wt, max(cue_start, ws), we))
    return out


def parse_vtt(data: str) -> list[Segment]:
    data = data.lstrip("\ufeff")
    segments: list[Segment] = []
    blocks = re.split(r"\n\s*\n", data)
    for block in blocks:
        lines = [ln.rstrip() for ln in block.splitlines() if ln.strip() != ""]
        if not lines:
            continue
        header = lines[0]
        if header.startswith(("NOTE", "STYLE", "REGION", "WEBVTT")):
            # timing may still be on a later line of this block
            timing_idx = next(
                (i for i, ln in enumerate(lines) if "-->" in ln), None
            )
            if timing_idx is None:
                continue
        else:
            timing_idx = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
            if timing_idx is None:
                continue
        timing = lines[timing_idx]
        left, right = timing.split("-->")
        m1 = _TS_RE.match(left.strip())
        m2 = _TS_RE.match(right.strip().split()[0])
        if not (m1 and m2):
            continue
        start = _ts_to_sec(*m1.groups())
        end = _ts_to_sec(*m2.groups())
        text = " ".join(lines[timing_idx + 1 :]).strip()
        text = _TAG_RE.sub(
            lambda m: m.group(0)
            if _INLINE_TS_RE.match("<" + m.group(0)[1:])
            else " ",
            text,
        )
        text = re.sub(r"\s+", " ", text).strip()
        # remove inline timestamps themselves from displayed text
        text = _INLINE_TS_RE.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        words = _cue_words(" ".join(lines[timing_idx + 1 :]), start, end)
        segments.append(Segment(start=start, end=end, text=text, words=words))
    segments.sort(key=lambda s: s.start)
    return _finalize(segments)


def parse_srt(data: str) -> list[Segment]:
    data = data.lstrip("\ufeff")
    segments: list[Segment] = []
    for block in re.split(r"\n\s*\n", data.strip()):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        idx = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if idx is None:
            continue
        left, right = lines[idx].split("-->")
        m1 = _TS_RE.match(left.strip())
        m2 = _TS_RE.match(right.strip().split()[0])
        if not (m1 and m2):
            continue
        start = _ts_to_sec(*m1.groups())
        end = _ts_to_sec(*m2.groups())
        text = " ".join(lines[idx + 1 :]).strip()
        text = _TAG_RE.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        words = _cue_words(text, start, end)
        segments.append(Segment(start=start, end=end, text=text, words=words))
    segments.sort(key=lambda s: s.start)
    return _finalize(segments)


def parse_json3(data: str) -> list[Segment]:
    obj = json.loads(data)
    segments: list[Segment] = []
    for ev in obj.get("events", []):
        if "tStartMs" not in ev:
            continue
        segs = ev.get("segs") or []
        raw = "".join(s.get("utf8", "") for s in segs)
        text = re.sub(r"\s+", " ", raw).strip()
        if not text or text == "\n":
            continue
        start = ev["tStartMs"] / 1000.0
        dur = (ev.get("dDurationMs") or 0) / 1000.0
        end = start + dur if dur > 0 else start + 1.0
        # word-level: segs may carry absolute tStartMs
        words: list[Word] = []
        seg_times = [
            (s["tStartMs"] / 1000.0 if "tStartMs" in s else None) for s in segs
        ]
        if any(t is not None for t in seg_times) and len(segs) > 1:
            cur_t = start
            prev_t = start
            for i, s in enumerate(segs):
                wtext = s.get("utf8", "")
                if not wtext.strip():
                    continue
                if seg_times[i] is not None:
                    cur_t = seg_times[i]
                wstart = cur_t
                wend = seg_times[i + 1] if i + 1 < len(segs) and seg_times[i + 1] is not None else None
                if wend is None:
                    # next text-bearing seg start or cue end
                    wend = next(
                        (
                            seg_times[j]
                            for j in range(i + 1, len(segs))
                            if seg_times[j] is not None and segs[j].get("utf8", "").strip()
                        ),
                        None,
                    )
                if wend is None:
                    wend = end
                words.append(Word(wtext.strip(), max(0.0, wstart), max(wstart + 0.02, min(wend, end))))
                prev_t = cur_t
        if not words:
            words = _cue_words(text, start, end)
        segments.append(Segment(start=start, end=end, text=text, words=words))
    segments.sort(key=lambda s: s.start)
    return _finalize(segments)


def parse_any(data: str, filename: str = "") -> list[Segment]:
    """Sniff format from filename/content and parse."""
    name = filename.lower()
    stripped = data.lstrip("\ufeff \n\r\t")
    if name.endswith(".json3") or stripped.startswith("{"):
        try:
            return parse_json3(data)
        except Exception:
            pass
    if stripped.startswith("WEBVTT") or name.endswith(".vtt"):
        return parse_vtt(data)
    if name.endswith(".srt") or "-->" in data[:2000]:
        return parse_srt(data)
    return parse_vtt(data)


_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def _finalize(segments: list[Segment]) -> list[Segment]:
    """Normalize segments for windowing: keep cues separate, but split
    overlong cues at sentence boundaries so highlight windows have natural
    start/end points."""
    out: list[Segment] = []
    for seg in segments:
        if seg.end - seg.start <= 7.0 or not _SENT_SPLIT.search(seg.text):
            out.append(seg)
            continue
        # split proportionally to sentence character length
        sentences = [s for s in _SENT_SPLIT.split(seg.text) if s.strip()]
        if len(sentences) <= 1:
            out.append(seg)
            continue
        total_chars = sum(len(s) for s in sentences) or 1
        t0 = seg.start
        span = seg.end - seg.start
        cursor = 0
        # assign word timings to sentences when available
        for sent in sentences:
            frac0 = cursor / total_chars
            cursor += len(sent)
            frac1 = cursor / total_chars
            s_start = t0 + span * frac0
            s_end = t0 + span * frac1
            words = [
                w for w in seg.words
                if w.start >= s_start - 1e-6 and w.end <= s_end + 1e-6
            ]
            out.append(Segment(start=s_start, end=s_end, text=sent.strip(), words=words))
    out.sort(key=lambda s: s.start)
    return out


def from_segments(segments: list[Segment], start: float, end: float) -> list[Segment]:
    """Clip a segment list to [start, end], clamping word times."""
    out: list[Segment] = []
    for seg in segments:
        if seg.end <= start or seg.start >= end:
            continue
        s = max(seg.start, start)
        e = min(seg.end, end)
        words = [
            Word(w.text, max(w.start, start), min(w.end, end))
            for w in seg.words
            if w.end > start and w.start < end
        ]
        out.append(Segment(start=s, end=e, text=seg.text, words=words))
    return out
