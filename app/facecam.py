"""Locate a streamer's facecam region in a source video.

Strategy: sample frames across the video, run Haar face detection on each,
then cluster detections by position. A facecam face sits at (nearly) the same
place in every frame, while faces inside the content (game cutscenes, etc.)
move around. The most temporally-stable cluster wins; its bounding box is
expanded to a webcam-style window (head + some margin) and returned.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import cv2
import numpy as np

from .ffmpeg_utils import extract_frame, probe


@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int

    def as_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


@dataclass
class FacecamResult:
    rect: Rect | None
    confidence: float
    samples: int
    note: str = ""


def _detect_faces(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    base = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    alt = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_alt2.xml")
    faces: list[tuple[int, int, int, int]] = []
    for path in (base, alt):
        casc = cv2.CascadeClassifier(path)
        if casc.empty():
            continue
        found = casc.detectMultiScale(gray, 1.1, 5, minSize=(36, 36))
        for (x, y, w, h) in found:
            faces.append((int(x), int(y), int(w), int(h)))
    # de-dup overlapping boxes (both cascades firing)
    uniq: list[tuple[int, int, int, int]] = []
    for f in sorted(faces, key=lambda b: -b[2] * b[3]):
        if not any(_iou(f, u) > 0.45 for u in uniq):
            uniq.append(f)
    return uniq


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def detect_facecam(
    path: str,
    samples: int = 9,
    stability_tol: float = 0.045,
    min_presence: float = 0.5,
) -> FacecamResult:
    """Detect the static facecam rectangle in `path`.

    stability_tol: max normalized center drift for the same face across frames.
    min_presence: fraction of sampled frames a cluster must appear in.
    """
    info = probe(path)
    W, H, dur = info["width"], info["height"], info["duration"]
    if dur <= 0:
        return FacecamResult(None, 0.0, 0, "cannot read duration")

    # evenly spaced sample times, avoiding the very first/last second
    times = [
        1.0 + (dur - 2.0) * i / max(1, samples - 1) for i in range(samples)
    ]
    times = [max(0.0, min(t, max(0.0, dur - 0.1))) for t in times]

    tmpdir = tempfile.mkdtemp(prefix="facecam_")
    detections: list[list[tuple[int, int, int, int]]] = []
    used = 0
    for i, t in enumerate(times):
        jpg = os.path.join(tmpdir, f"s{i}.jpg")
        try:
            extract_frame(path, t, jpg)
        except Exception:
            detections.append([])
            continue
        img = cv2.imread(jpg)
        if img is None:
            detections.append([])
            continue
        # downscale large frames for speed, detection scales back up
        scale = 1.0
        if img.shape[1] > 960:
            scale = img.shape[1] / 960.0
            img = cv2.resize(img, (960, int(img.shape[0] / scale)))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = [(int(x * scale), int(y * scale), int(w * scale), int(h * scale))
                 for (x, y, w, h) in _detect_faces(gray)]
        detections.append(faces)
        used += 1

    # clean up temp frames
    for f in os.listdir(tmpdir):
        try:
            os.remove(os.path.join(tmpdir, f))
        except OSError:
            pass
    try:
        os.rmdir(tmpdir)
    except OSError:
        pass

    if used == 0:
        return FacecamResult(None, 0.0, 0, "no frames sampled")

    # cluster face centers across frames (simple greedy nearest-center cluster)
    centers: list[tuple[float, float, tuple]] = []  # (cx, cy, box)
    for faces in detections:
        for b in faces:
            x, y, w, h = b
            centers.append((x + w / 2.0, y + h / 2.0, b))

    if not centers:
        return FacecamResult(None, 0.0, used, "no faces detected")

    tol_px = stability_tol * max(W, H)
    clusters: list[dict] = []  # {boxes: [...], frames: set()}
    for fi, faces in enumerate(detections):
        for b in faces:
            cx, cy = b[0] + b[2] / 2.0, b[1] + b[3] / 2.0
            best, best_d = None, tol_px
            for cl in clusters:
                d = ((cx - cl["cx"]) ** 2 + (cy - cl["cy"]) ** 2) ** 0.5
                if d < best_d:
                    best, best_d = cl, d
            if best is None:
                clusters.append(
                    {"cx": cx, "cy": cy, "boxes": [b], "frames": {fi}}
                )
            else:
                best["boxes"].append(b)
                best["frames"].add(fi)
                n = len(best["boxes"])
                best["cx"] = (best["cx"] * (n - 1) + cx) / n
                best["cy"] = (best["cy"] * (n - 1) + cy) / n

    if not clusters:
        return FacecamResult(None, 0.0, used, "no stable face")

    # rank: presence in frames first, then box consistency (low variance)
    best_cl = None
    best_key = (-1.0, -1.0)
    for cl in clusters:
        presence = len(cl["frames"]) / used
        if presence < min_presence:
            continue
        boxes = np.array(cl["boxes"], dtype=float)
        spread = boxes[:, 2:].mean() and float(
            np.linalg.norm(boxes[:, 2:].std(axis=0))
        )
        # key: presence, then inverse spread (stable size => likely facecam)
        key = (presence, -spread / max(1.0, float(boxes[:, 2].mean())))
        if key > best_key:
            best_key, best_cl = key, cl

    if best_cl is None:
        return FacecamResult(
            None, 0.0, used, "no face stable enough to be a facecam"
        )

    boxes = np.array(best_cl["boxes"], dtype=float)
    # use median box for robustness
    mx, my, mw, mh = np.median(boxes, axis=0)
    presence = len(best_cl["frames"]) / used

    # expand to a webcam-style window: head + margin, keep aspect around 4:3..1:1
    pad_x = mw * 0.45
    pad_top = mh * 0.55
    pad_bot = mh * 0.9
    x = int(max(0, mx - pad_x))
    y = int(max(0, my - pad_top))
    w = int(min(W - x, mw + 2 * pad_x))
    h = int(min(H - y, mh + pad_top + pad_bot))
    # facecams are rarely more than ~40% of the frame width
    if w > 0.45 * W:
        scale = 0.45 * W / w
        nw, nh = int(w * scale), int(h * scale)
        x = int(min(max(0, mx - nw / 2), W - nw))
        y = int(max(0, my - nh * 0.35))
        w, h = nw, min(nh, H - y)

    rect = Rect(x=max(0, x), y=max(0, y), w=max(8, w), h=max(8, h))
    conf = round(float(presence * min(1.0, len(best_cl["boxes"]) / used)), 3)
    return FacecamResult(
        rect=rect,
        confidence=conf,
        samples=used,
        note=f"face present in {presence:.0%} of samples",
    )
