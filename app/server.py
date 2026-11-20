"""FastAPI server: analyze YouTube URLs or uploaded files, render shorts."""
from __future__ import annotations

import os
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .downloader import download_video, fetch_meta, looks_like_youtube
from .facecam import Rect, detect_facecam
from .ffmpeg_utils import probe
from .highlights import find_highlights
from .render import render_clip

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
STATIC = ROOT / "static"
SESSIONS: dict[str, dict] = {}
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()

app = FastAPI(title="clips")
OUTPUTS.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)


# ---------------------------------------------------------------- models
class AnalyzeReq(BaseModel):
    url: str


class ClipIn(BaseModel):
    start: float
    end: float
    label: str = ""


class RenderReq(BaseModel):
    session_id: str
    clips: list[ClipIn]
    facecam: dict | None = None  # {x,y,w,h} or None => auto/no cam
    facecam_mode: str = "auto"  # auto | custom | none
    captions: bool = True
    cam_width_frac: float = 0.30


class JobStatus(BaseModel):
    id: str
    kind: str
    state: str  # queued | running | done | error
    progress: float
    message: str
    result: dict | None = None


# ---------------------------------------------------------------- helpers
def _set_job(job_id: str, **kw):
    with JOBS_LOCK:
        job = JOBS[job_id]
        job.update(kw)


def _new_job(kind: str) -> str:
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "kind": kind,
            "state": "queued",
            "progress": 0.0,
            "message": "",
            "result": None,
        }
    return job_id


def _run_job(job_id: str, fn):
    def runner():
        _set_job(job_id, state="running", message="starting…")
        try:
            result = fn(lambda p, m=None: _set_job(
                job_id, progress=max(0.0, min(1.0, p)),
                **({"message": m} if m else {})
            ))
            _set_job(job_id, state="done", progress=1.0, result=result,
                     message="done")
        except Exception as e:
            traceback.print_exc()
            _set_job(job_id, state="error", message=str(e))

    threading.Thread(target=runner, daemon=True).start()
    return job_id


def _session_dir(sid: str) -> Path:
    d = DATA / sid
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- routes
@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.post("/api/analyze")
def api_analyze(req: AnalyzeReq):
    url = req.url.strip()
    if not url:
        raise HTTPException(400, "empty URL")
    if not looks_like_youtube(url):
        raise HTTPException(400, "that doesn't look like a YouTube URL")

    sid = uuid.uuid4().hex[:12]
    try:
        meta = fetch_meta(url, str(_session_dir(sid)))
    except Exception as e:
        shutil.rmtree(DATA / sid, ignore_errors=True)
        raise HTTPException(400, f"could not fetch video: {e}")

    duration = meta.duration
    clips = find_highlights(meta.segments, duration, clip_len=30.0, count=6)
    SESSIONS[sid] = {
        "id": sid,
        "url": url,
        "video_id": meta.id,
        "title": meta.title,
        "duration": duration,
        "thumbnail": meta.thumbnail,
        "uploader": meta.uploader,
        "segments": meta.segments,
        "transcript_source": meta.transcript_source,
        "source_path": None,
        "facecam": None,
        "suggested": [
            {
                "start": round(c.start, 2),
                "end": round(c.end, 2),
                "score": c.score,
                "text": c.text,
                "reasons": c.reasons,
            }
            for c in clips
        ],
    }
    return {
        "session_id": sid,
        "title": meta.title,
        "duration": duration,
        "thumbnail": meta.thumbnail,
        "uploader": meta.uploader,
        "transcript_source": meta.transcript_source,
        "segment_count": len(meta.segments),
        "suggested": SESSIONS[sid]["suggested"],
    }


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    sid = uuid.uuid4().hex[:12]
    sdir = _session_dir(sid)
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    if suffix.lower() not in {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi"}:
        raise HTTPException(400, f"unsupported file type: {suffix}")
    dest = sdir / f"source{suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        info = probe(str(dest))
    except Exception as e:
        shutil.rmtree(sdir, ignore_errors=True)
        raise HTTPException(400, f"cannot read video: {e}")

    duration = info["duration"]
    # fixed-interval suggestions when there's no transcript
    step = 30.0
    suggested = []
    t = 0.0
    while t + 10.0 <= duration and len(suggested) < 8:
        end = min(t + step, duration)
        if end - t >= 10.0:
            suggested.append({
                "start": round(t, 2), "end": round(end, 2), "score": 0.0,
                "text": "", "reasons": ["fixed interval (no transcript)"],
            })
        t += step
    if not suggested and duration >= 5:
        suggested = [{
            "start": 0.0, "end": round(min(duration, 30.0), 2), "score": 0.0,
            "text": "", "reasons": ["whole video"],
        }]

    SESSIONS[sid] = {
        "id": sid,
        "url": "",
        "video_id": sid,
        "title": file.filename or "uploaded video",
        "duration": duration,
        "thumbnail": "",
        "uploader": "",
        "segments": [],
        "transcript_source": "",
        "source_path": str(dest),
        "facecam": None,
        "suggested": suggested,
    }
    return {
        "session_id": sid,
        "title": SESSIONS[sid]["title"],
        "duration": duration,
        "thumbnail": "",
        "uploader": "",
        "transcript_source": "",
        "segment_count": 0,
        "suggested": suggested,
    }


def _ensure_source(sid: str, progress) -> str:
    sess = SESSIONS.get(sid)
    if not sess:
        raise RuntimeError("unknown session")
    if sess.get("source_path") and os.path.exists(sess["source_path"]):
        progress(1.0, "source ready")
        return sess["source_path"]

    def dl_p(p):
        progress(p * 0.9, f"downloading video… {p * 100:.0f}%")

    path = download_video(sess["url"], str(_session_dir(sid)), on_progress=dl_p)
    sess["source_path"] = path
    progress(1.0, "source ready")
    return path


@app.post("/api/prepare")
def api_prepare(req: dict):
    """Download source (if needed) + auto-detect facecam. Returns job id."""
    sid = req.get("session_id", "")
    if sid not in SESSIONS:
        raise HTTPException(404, "unknown session")
    job_id = _new_job("prepare")

    def work(progress):
        path = _ensure_source(sid, progress)
        progress(0.92, "detecting facecam…")
        result = detect_facecam(path)
        sess = SESSIONS[sid]
        if result.rect:
            sess["facecam"] = result.rect.as_dict()
        progress(1.0, "ready")
        return {
            "facecam": sess.get("facecam"),
            "confidence": result.confidence,
            "note": result.note,
            "duration": sess["duration"],
            "title": sess["title"],
        }

    return {"job_id": _run_job(job_id, work)}


@app.post("/api/detect_facecam")
def api_detect_facecam(req: dict):
    sid = req.get("session_id", "")
    sess = SESSIONS.get(sid)
    if not sess:
        raise HTTPException(404, "unknown session")
    path = sess.get("source_path")
    if not path or not os.path.exists(path):
        raise HTTPException(400, "source video not downloaded yet")
    result = detect_facecam(path)
    if result.rect:
        sess["facecam"] = result.rect.as_dict()
    return {
        "facecam": sess.get("facecam"),
        "confidence": result.confidence,
        "note": result.note,
    }


class FacecamReq(BaseModel):
    session_id: str
    rect: dict | None


@app.post("/api/set_facecam")
def api_set_facecam(req: FacecamReq):
    sess = SESSIONS.get(req.session_id)
    if not sess:
        raise HTTPException(404, "unknown session")
    sess["facecam"] = req.rect
    return {"ok": True, "facecam": req.rect}


@app.post("/api/render")
def api_render(req: RenderReq):
    sess = SESSIONS.get(req.session_id)
    if not sess:
        raise HTTPException(404, "unknown session")
    if not req.clips:
        raise HTTPException(400, "no clips selected")
    for c in req.clips:
        if c.end - c.start < 1.0:
            raise HTTPException(
                400, f"clip {c.label or c.start} is shorter than 1s"
            )
        if c.start < 0 or (sess["duration"] and c.end > sess["duration"] + 1):
            raise HTTPException(400, f"clip times out of range: {c.start}-{c.end}")

    clips = [c.model_dump() for c in req.clips]
    facecam_mode = req.facecam_mode
    facecam_rect = req.facecam
    captions = req.captions
    cam_frac = req.cam_width_frac
    job_id = _new_job("render")

    def work(progress):
        path = _ensure_source(
            sess["id"], lambda p, m: progress(p * 0.15, m or "preparing…")
        )

        # resolve facecam
        rect: Rect | None = None
        if facecam_mode == "auto":
            if not sess.get("facecam"):
                progress(0.16, "detecting facecam…")
                res = detect_facecam(path)
                if res.rect:
                    sess["facecam"] = res.rect.as_dict()
            if sess.get("facecam"):
                r = sess["facecam"]
                rect = Rect(int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]))
        elif facecam_mode == "custom" and facecam_rect:
            rect = Rect(
                int(facecam_rect["x"]), int(facecam_rect["y"]),
                int(facecam_rect["w"]), int(facecam_rect["h"]),
            )

        segments = sess.get("segments") or []
        outdir = OUTPUTS / sess["id"]
        outdir.mkdir(parents=True, exist_ok=True)

        results = []
        n = len(clips)
        for i, c in enumerate(clips):
            base = 0.15 + 0.85 * (i / n)
            span = 0.85 / n
            label = c["label"] or f"clip{i + 1}"
            safe = "".join(
                ch if ch.isalnum() or ch in "-_" else "_" for ch in label
            )[:40]
            out = str(outdir / f"{i + 1:02d}_{safe}.mp4")

            def vp(f, i=i, base=base, span=span, label=label):
                progress(base + span * f, f"rendering {label}… {f * 100:.0f}%")

            render_clip(
                path,
                c["start"],
                c["end"],
                out,
                segments=segments if captions else None,
                facecam=rect,
                cam_width_frac=cam_frac,
                captions=captions and bool(segments),
                on_progress=vp,
            )
            meta = probe(out)
            results.append({
                "label": label,
                "start": c["start"],
                "end": c["end"],
                "file": f"{sess['id']}/{os.path.basename(out)}",
                "poster": (
                    f"{sess['id']}/{os.path.basename(out)[:-4]}.jpg"
                    if os.path.exists(out[:-4] + ".jpg") else ""
                ),
                "duration": round(meta["duration"], 2),
                "width": meta["width"],
                "height": meta["height"],
            })
        return {"clips": results, "facecam": rect.as_dict() if rect else None}

    return {"job_id": _run_job(job_id, work)}


@app.get("/api/job/{job_id}", response_model=JobStatus)
def api_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "unknown job")
        return dict(job)


# Serve rendered outputs (mp4 + posters)
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS)), name="outputs")


# Serve UI assets (index.html itself is served by the / route above)
@app.get("/style.css")
def style_css():
    return FileResponse(STATIC / "style.css", media_type="text/css")


@app.get("/app.js")
def app_js():
    return FileResponse(STATIC / "app.js", media_type="application/javascript")


@app.get("/api/thumbnail/{sid}")
def api_thumbnail(sid: str):
    """Proxy the YouTube thumbnail (sandbox/user may block i.ytimg.com)."""
    sess = SESSIONS.get(sid)
    if not sess or not sess.get("thumbnail"):
        raise HTTPException(404, "no thumbnail")
    import urllib.request

    cache = _session_dir(sid) / "thumb.jpg"
    if not cache.exists():
        try:
            req = urllib.request.Request(
                sess["thumbnail"], headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=15) as r:
                cache.write_bytes(r.read())
        except Exception as e:
            raise HTTPException(502, f"thumbnail fetch failed: {e}")
    return FileResponse(cache, media_type="image/jpeg")


class SuggestReq(BaseModel):
    session_id: str
    clip_len: float = 30.0
    count: int = 6


@app.post("/api/resuggest")
def api_resuggest(req: SuggestReq):
    sess = SESSIONS.get(req.session_id)
    if not sess:
        raise HTTPException(404, "unknown session")
    if not sess.get("segments"):
        raise HTTPException(400, "no transcript for this video")
    clips = find_highlights(
        sess["segments"], sess["duration"],
        clip_len=req.clip_len, count=req.count,
    )
    return {
        "suggested": [
            {
                "start": round(c.start, 2),
                "end": round(c.end, 2),
                "score": c.score,
                "text": c.text,
                "reasons": c.reasons,
            }
            for c in clips
        ]
    }
