# ✂️ clips

Turn long-form YouTube videos into vertical shorts — locally, in your browser.

Paste a URL (or upload a video), the app reads the transcript, auto-picks
highlight-worthy ~30s windows, detects the streamer's facecam, and renders
1080×1920 shorts with the content filling the frame, the creator in a small
top-right window, and captions burned in.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.server:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000

## How it works

**1. Analyze** — `yt-dlp` fetches metadata plus the best available subtitles
(manual captions preferred, auto-captions as fallback). Uploaded files can
pair with an optional captions file (`.vtt`/`.srt`/`.json3`) to get the same
transcript-driven highlights; without one, uploads get fixed-interval
suggestions instead.

> **Blocked from YouTube?** Some sandboxes/office networks block YouTube at
> the TLS layer — the app detects this and tells you. Workaround: on a
> machine with YouTube access run
> `yt-dlp -f "bv*[height<=1080]+ba/b" -o video.mp4 --write-subs --write-auto-subs --sub-langs en.* <url>`
> then upload `video.mp4` (+ the downloaded `.vtt`) here. Everything —
> highlight detection, facecam placement, captions, rendering — works offline.

**2. Highlight detection** (`app/highlights.py`) — a heuristic scorer slides
~30s windows over the transcript and scores them on:

- strong hook openings ("why…", "nobody…", "the thing is…")
- speech density (2–4 words/sec is punchy; dead air is not)
- high-energy words, punchlines and laughter markers
- topic-shift boundaries (a window that starts where the subject changes)
- clean sentence endings
- penalties for intro/outro/sponsor patter ("subscribe", "today's video", …)

Top non-overlapping windows are returned with per-clip reason chips. Adjust
start/end times by hand, add manual clips, or re-detect with a different
clip length (15/30/45/60s).

**3. Facecam detection** (`app/facecam.py`) — samples frames across the
video, runs Haar face detection on each, and clusters detections by position.
The facecam is the face that stays in the *same place* in most frames; faces
inside the content move around. The stable cluster's median box is expanded
to a head-and-shoulders window.

**4. Render** (`app/render.py`) — one ffmpeg pass per clip:

- content cover-cropped to fill 1080×1920
- the original facecam spot blurred out if it survives the crop
- the extracted facecam repositioned as a bordered window top-right
- captions burned in from the transcript (word-timed phrase chunks, ASS/libass)
- audio loudness-normalized to ≈ −16 LUFS, H.264 + AAC, faststart

## Layout

```
app/
  server.py      FastAPI routes + job runner
  downloader.py  yt-dlp metadata/subtitles/download
  transcript.py  VTT / SRT / json3 parsing with word timings
  highlights.py  heuristic highlight scoring
  facecam.py     stable-face clustering for cam detection
  captions.py    ASS caption generation (phrase chunks)
  render.py      ffmpeg filter graph for the 9:16 output
  ffmpeg_utils.py ffmpeg location, probing, progress parsing
static/          single-page UI (no build step)
tests/           offline end-to-end tests (synthetic video, no network)
data/            per-session downloads + transcripts   (gitignored)
outputs/         rendered shorts                       (gitignored)
```

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

The suite builds a synthetic 720p video with a pasted-in "facecam", then
exercises transcript parsing, highlight scoring, facecam detection, caption
generation and a full 1080×1920 render — no network required.

## Notes

- **YouTube access**: analyzing URLs needs normal internet access to
  youtube.com. If your network blocks it (e.g. a locked-down sandbox), use
  the file-upload path — everything else works offline.
- **Caption language**: defaults to `en`; pass a different code by editing
  `subtitleslangs` in `app/downloader.py`.
- **Rendered files** land in `outputs/<session>/` and are served at
  `/outputs/…`; source downloads live in `data/<session>/`.
