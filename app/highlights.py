"""Auto-highlight detection from a timed transcript.

Heuristic scorer: slides candidate windows over the transcript, scores them on
hook strength, speech density, emotional energy, topic-shift boundaries and
punchline proximity, penalizes intro/outro/sponsor patter, then picks the top
non-overlapping windows snapped to sentence boundaries.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .transcript import Segment

# --- lexicons ---------------------------------------------------------------

HOOK_START = re.compile(
    r"^(why|how|what|who|when|where|nobody|everyone|the (best|worst|biggest|craziest|only))"
    r"\b|(the thing (is|about))|(here'?s (the|what|why|how))|(imagine)|("
    r"did you (know|know that))|(most people)|("
    r"you (probably|might|wouldn'?t) (never|not) (know|think|heard))",
    re.I,
)
QUESTION = re.compile(r"\?\s*$|\b(why|how come|what if|ever wonder|have you ever)\b", re.I)
PUNCHLINE = re.compile(
    r"\b(lol|lmao|lmfao|haha+|no way|are you (kidding|serious)|"
    r"that'?s (insane|crazy|wild)|unbelievable|wait (what|till)|"
    r"plot twist|turns out|i can'?t believe)\b",
    re.I,
)
ENERGY = re.compile(
    r"\b(crazy|insane|wild|unbelievable|amazing|terrible|horrible|huge|massive|"
    r"destroyed|destroying|destroy|dominated|clutched|choke|throw|epic|disaster|"
    r"secret|mistake|truth|lie|scam|exposed|ruined|perfect|never|always|"
    r"first time|final|instantly|immediately)\b",
    re.I,
)
EMOJI_EXCLAIM = re.compile(r"[!]{2,}|\b(wow|omg|bro|dude|insane)\b", re.I)
INTRO_PENALTY = re.compile(
    r"\b(welcome back|subscribe|hit the bell|like (this|and)|comment below|"
    r"today'?s video|in this video|before we (start|begin)|"
    r"sponsor|sponsored|patreon|link in (the )?description|promo code|use code|"
    r"thanks for (watching|listening)|intro|outro|shoutout to)\b",
    re.I,
)
NUMBERS = re.compile(r"\b\d+(\.\d+)?(%|\s?(million|billion|thousand|k|m))?\b")


@dataclass
class ClipCandidate:
    start: float
    end: float
    score: float
    text: str
    reasons: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


def _words(seg: Segment) -> list[str]:
    return re.findall(r"[a-z0-9']+", seg.text.lower())


def _windows(
    segments: list[Segment], target: float, min_len: float, max_len: float
) -> list[tuple[float, float, int, int]]:
    """Yield (start, end, i0, i1) candidate windows starting at each segment."""
    out = []
    n = len(segments)
    for i0 in range(n):
        start = segments[i0].start
        # extend until we hit target length; stop hard at max_len
        i1 = i0
        while i1 + 1 < n and segments[i1 + 1].end - start <= target:
            i1 += 1
        end = segments[i1].end
        if end - start < min_len:
            # try to grow further (but never past max_len)
            while i1 + 1 < n and segments[i1 + 1].end - start <= max_len:
                i1 += 1
                end = segments[i1].end
                if end - start >= min_len:
                    break
        if end - start < min_len:
            continue
        if end - start > max_len:
            continue
        # snap end back to a sentence-ish boundary if we overshot target a lot
        out.append((start, min(end, start + max_len), i0, i1))
    return out


def _bag(words_a: list[str], words_b: list[str]) -> float:
    """Cheap cosine over term sets for topic-shift measurement."""
    if not words_a or not words_b:
        return 0.0
    sa, sb = set(words_a), set(words_b)
    inter = len(sa & sb)
    if not inter:
        return 0.0
    return inter / math.sqrt(len(sa) * len(sb))


def find_highlights(
    segments: list[Segment],
    duration: float,
    clip_len: float = 30.0,
    min_len: float = 12.0,
    count: int = 6,
    max_len: float = 30.0,
) -> list[ClipCandidate]:
    """Return up to `count` non-overlapping highlight candidates.

    clip_len is the target window length; windows never exceed max_len.
    """
    if not segments:
        return []
    max_len = max(max_len, clip_len)
    min_len = min(min_len, clip_len)

    cand_windows = _windows(segments, clip_len, min_len, max_len)
    if not cand_windows:
        # very short video: one window covering everything
        start, end = segments[0].start, segments[-1].end
        if end - start > max_len:
            end = start + max_len
        return [
            ClipCandidate(
                start=start,
                end=end,
                score=0.5,
                text=_snippet(segments, start, end),
                reasons=["short video"],
            )
        ]

    scored: list[tuple[float, list[str], float, float, int, int]] = []
    for start, end, i0, i1 in cand_windows:
        segs = segments[i0 : i1 + 1]
        text = " ".join(s.text for s in segs)
        reasons: list[str] = []
        score = 0.0

        # speech density: ideal ~2.4-3.2 words/sec for punchy clips
        wc = sum(len(_words(s)) for s in segs)
        dens = wc / max(0.5, end - start)
        if 2.0 <= dens <= 4.0:
            score += 1.5
            reasons.append(f"dense speech ({dens:.1f} wps)")
        elif dens >= 1.2:
            score += 0.6

        # hook at the opening
        opening = " ".join(s.text for s in segs[:2])
        if HOOK_START.search(opening):
            score += 2.5
            reasons.append("strong hook opening")
        elif QUESTION.search(opening):
            score += 1.4
            reasons.append("opens with a question")

        # energy / emotion anywhere in the window
        energy_hits = len(ENERGY.findall(text))
        if energy_hits:
            score += min(2.0, 0.5 * energy_hits)
            reasons.append(f"high-energy words ×{energy_hits}")
        excl = len(EMOJI_EXCLAIM.findall(text))
        if excl:
            score += min(1.0, 0.25 * excl)

        # laughter / punchline
        if PUNCHLINE.search(text):
            score += 1.6
            reasons.append("punchline / laughter")

        # numbers add specificity
        nums = len(NUMBERS.findall(text))
        if nums:
            score += min(0.9, 0.3 * nums)

        # topic-shift boundary right before the window start
        if i0 > 0:
            prev_words = _words(segments[i0 - 1])
            cur_words = _words(segments[i0])
            sim = _bag(prev_words, cur_words)
            if sim < 0.12:
                score += 1.2
                reasons.append("starts at a topic shift")

        # prefer windows that end on a complete thought (terminal punctuation)
        if segs[-1].text.rstrip().endswith((".", "!", "?")):
            score += 0.6
            reasons.append("clean ending")

        # penalty: intro/outro/sponsor patter
        intro_hits = len(INTRO_PENALTY.findall(text))
        if intro_hits:
            score -= min(3.0, 1.5 * intro_hits)
            reasons.append("intro/outro/sponsor pattern")

        # small penalty for very beginning / very end of video (usually not the meat)
        if start < 15:
            score -= 0.8
        if duration and end > duration - 10:
            score -= 0.5

        # tiny bonus for closer to target length
        score += 0.4 * (1.0 - abs((end - start) - clip_len) / max(clip_len, 1))

        scored.append((score, reasons, start, end, i0, i1))

    scored.sort(key=lambda t: -t[0])

    # greedy non-overlap selection with a minimum gap
    picked: list[tuple[float, list[str], float, float, int, int]] = []
    gap = 5.0
    for cand in scored:
        if cand[0] <= 0:
            break
        _, _, s, e, _, _ = cand
        if any(not (e + gap <= p[2] or s >= p[3] + gap) for p in picked):
            continue
        picked.append(cand)
        if len(picked) >= count:
            break

    picked.sort(key=lambda t: t[2])
    return [
        ClipCandidate(
            start=s,
            end=e,
            score=round(max(0.0, min(1.0, sc / 8.0)), 3),
            text=_snippet(segments, s, e),
            reasons=rs,
        )
        for sc, rs, s, e, _, _ in picked
    ]


def _snippet(segments: list[Segment], start: float, end: float, limit: int = 180) -> str:
    text = " ".join(s.text for s in segments if s.end > start and s.start < end)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"
