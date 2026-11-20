"""End-to-end tests that don't need network: synthetic video + local files."""
from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.captions import build_ass, chunk_words
from app.facecam import Rect, detect_facecam
from app.ffmpeg_utils import ffmpeg_exe, probe
from app.highlights import find_highlights
from app.render import render_clip
from app.transcript import parse_json3, parse_srt, parse_vtt

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


# ---------------------------------------------------------------- transcript
SAMPLE_VTT = """WEBVTT

00:00:00.000 --> 00:00:03.500
Welcome back everyone, today we are going to <00:00:01.200>talk about <00:00:02.300>something crazy.

00:00:03.500 --> 00:00:07.000
So the thing is, nobody talks about this trick.

00:00:07.000 --> 00:00:11.000
Why does it always break on the second attempt? That's insane!

00:00:11.000 --> 00:00:15.500
First you open the config file, then you change the timeout value.
"""


def test_parse_vtt_basic():
    segs = parse_vtt(SAMPLE_VTT)
    assert len(segs) >= 3
    assert segs[0].start == 0.0
    assert "Welcome back" in segs[0].text
    # inline timestamps become word timings
    assert segs[0].words, "expected word timings from inline timestamps"
    assert segs[0].words[0].start == 0.0


def test_parse_srt():
    srt = """1
00:00:01,000 --> 00:00:04,000
Hello there, this is a test.

2
00:00:04,500 --> 00:00:08,000
Second line follows.
"""
    segs = parse_srt(srt)
    assert len(segs) == 2
    assert segs[0].start == 1.0
    assert segs[1].start == 4.5


def test_parse_json3():
    data = """{
      "events": [
        {"tStartMs": 0, "dDurationMs": 2500, "segs": [{"utf8": "hey "}, {"utf8": "there"}]},
        {"tStartMs": 2500, "dDurationMs": 3000, "segs": [{"utf8": "second cue"}]}
      ]
    }"""
    segs = parse_json3(data)
    assert len(segs) == 2
    assert segs[1].start == 2.5
    assert segs[1].text == "second cue"


# ---------------------------------------------------------------- highlights
def _mk_segments():
    from app.transcript import Segment

    texts = [
        (0.0, 4.0, "Welcome back to the channel, don't forget to subscribe and like."),
        (4.0, 9.0, "Today's video is sponsored by our friends over at Example Co."),
        (10.0, 14.0, "So the thing is, nobody talks about this one trick."),
        (14.0, 18.0, "Why does it always break on the second attempt? That's insane!"),
        (18.0, 23.0, "The craziest part is that 90% of people get it wrong every time."),
        (25.0, 30.0, "Meanwhile the weather in Paris was lovely last week."),
        (30.0, 35.0, "Here's the secret ingredient: you have to double the amount."),
        (35.0, 40.0, "No way, that actually worked? That's insane, haha!"),
        (42.0, 47.0, "Thanks for watching, hit the bell and see you next time."),
    ]
    return [Segment(s, e, t) for s, e, t in texts]


def test_find_highlights_picks_hooky_windows():
    segs = _mk_segments()
    clips = find_highlights(segs, duration=48.0, clip_len=30.0, count=3, max_len=30.0)
    assert clips, "expected at least one clip"
    for c in clips:
        assert c.end - c.start <= 30.5
        assert c.end - c.start >= 10.0
        assert 0 <= c.start < c.end <= 48.0
    # should not pick the pure intro/outro windows as top clips
    starts = [c.start for c in clips]
    assert not any(s < 1.0 for s in starts), "intro should be penalized"
    # top clip should overlap the hooky middle (10-40s)
    best = max(clips, key=lambda c: c.score)
    assert best.start < 40 and best.end > 10


def test_highlights_non_overlapping():
    segs = _mk_segments()
    clips = find_highlights(segs, duration=48.0, clip_len=20.0, count=5, max_len=25.0)
    clips_sorted = sorted(clips, key=lambda c: c.start)
    for a, b in zip(clips_sorted, clips_sorted[1:]):
        assert b.start >= a.end - 4.0, "clips should not substantially overlap"


def test_find_highlights_empty():
    assert find_highlights([], duration=100.0) == []


# ---------------------------------------------------------------- captions
def test_chunk_words_limits():
    words = [(f"w{i}", i * 0.3, i * 0.3 + 0.28) for i in range(20)]
    chunks = chunk_words(words, max_words=4, max_dur=1.7)
    assert chunks
    for ch in chunks:
        n = len(ch.text.split())
        assert 1 <= n <= 6
        assert ch.end > ch.start


def test_build_ass_writes_file(tmp_path):
    segs = _mk_segments()
    out = str(tmp_path / "c.ass")
    build_ass(segs, 10.0, 40.0, out)
    content = open(out).read()
    assert "[Events]" in content
    assert "Dialogue:" in content
    assert "nobody talks" in content


# ---------------------------------------------------------------- render
@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    """60s 1280x720 video with tone audio + a static face image bottom-left."""
    path = str(tmp_path_factory.mktemp("vid") / "src.mp4")
    face = os.path.join(FIXTURES, "face1.png")
    ff = ffmpeg_exe()
    cmd = [
        ff, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=60",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=60",
        "-i", face,
        "-filter_complex",
        # face pasted bottom-left as a fake facecam (scaled up so Haar sees it)
        "[2:v]scale=300:-1[fc];"
        "[0:v][fc]overlay=40:H-h-40[v]",
        "-map", "[v]", "-map", "1:a",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        path,
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=300)
    return path


def test_probe(synthetic_video):
    info = probe(synthetic_video)
    assert info["width"] == 1280
    assert info["height"] == 720
    assert 58 < info["duration"] < 62
    assert info["has_audio"]


def test_facecam_detection(synthetic_video):
    res = detect_facecam(synthetic_video, samples=7)
    assert res.rect is not None, f"expected facecam, got: {res.note}"
    r = res.rect
    # face1.png is 280x484 with the face at (123,31,57,57); it's scaled to
    # width 300 and overlaid at x=40, y=H-h-40. Expected face center in the
    # 1280x720 frame:
    import cv2
    face = cv2.imread(os.path.join(FIXTURES, "face1.png"))
    fh, fw = face.shape[:2]
    scale = 300 / fw
    disp_h = int(fh * scale)
    top = 720 - disp_h - 40
    cx = 40 + (123 + 57 / 2) * scale
    cy = top + (31 + 57 / 2) * scale
    # detected window must contain the face center
    assert r.x <= cx <= r.x + r.w, f"face cx={cx:.0f} not in {r}"
    assert r.y <= cy <= r.y + r.h, f"face cy={cy:.0f} not in {r}"
    # and it should be in the left half / lower-ish region of the frame
    assert r.x < 640, f"rect too far right: {r}"
    assert r.w > 40 and r.h > 40


def test_render_clip(tmp_path, synthetic_video):
    # pick the region where the face is: bottom-left
    segs = []
    out = str(tmp_path / "short.mp4")
    facecam = Rect(x=40, y=720 - 400, w=300, h=400)

    def prog(f):
        assert 0 <= f <= 1

    result = render_clip(
        synthetic_video, 10.0, 20.0, out,
        segments=segs, facecam=facecam, captions=False,
        on_progress=prog,
    )
    assert os.path.exists(result["path"])
    info = probe(result["path"])
    assert info["width"] == 1080
    assert info["height"] == 1920
    assert 9.5 < info["duration"] < 10.5
    assert info["has_audio"]
    if result["poster"]:
        assert os.path.exists(result["poster"])


def test_render_with_captions(tmp_path, synthetic_video):
    segs = _mk_segments()
    out = str(tmp_path / "short_cap.mp4")
    result = render_clip(
        synthetic_video, 10.0, 25.0, out,
        segments=segs, facecam=None, captions=True,
    )
    info = probe(result["path"])
    assert info["width"] == 1080 and info["height"] == 1920
    # ass sidecar should exist next to output
    assert os.path.exists(out[:-4] + ".ass")


def test_render_too_short_raises(tmp_path, synthetic_video):
    with pytest.raises(ValueError):
        render_clip(synthetic_video, 10.0, 10.2, str(tmp_path / "x.mp4"))
