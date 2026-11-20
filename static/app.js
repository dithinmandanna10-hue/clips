/* clips frontend — analyze → pick clips → render */
"use strict";

const $ = (id) => document.getElementById(id);

let session = null;        // {session_id, duration, suggested, ...}
let clips = [];            // [{start, end, label, text, reasons, score, on}]
let preparing = false;

// ---------------------------------------------------------------- helpers
function fmt(t) {
  t = Math.max(0, Number(t) || 0);
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function parseTime(v) {
  v = String(v).trim();
  if (v.includes(":")) {
    const parts = v.split(":").map(Number);
    if (parts.some(isNaN)) return NaN;
    return parts.reduce((a, b) => a * 60 + b, 0);
  }
  return Number(v);
}

function show(el, on = true) { el.hidden = !on; }

function setError(id, msg) {
  const el = $(id);
  if (!msg) { show(el, false); return; }
  el.textContent = msg;
  show(el, true);
}

function setProgress(id, frac, msg) {
  const box = $(id);
  show(box, true);
  box.querySelector(".fill").style.width = `${Math.round(frac * 100)}%`;
  if (msg) box.querySelector(".msg").textContent = msg;
}

async function pollJob(jobId, onTick) {
  for (;;) {
    const r = await fetch(`/api/job/${jobId}`);
    if (!r.ok) throw new Error(`job status ${r.status}`);
    const j = await r.json();
    onTick(j);
    if (j.state === "done") return j.result;
    if (j.state === "error") throw new Error(j.message || "job failed");
    await new Promise((res) => setTimeout(res, 600));
  }
}

// ---------------------------------------------------------------- analyze
async function analyze() {
  const url = $("url-input").value.trim();
  setError("source-error", "");
  if (!url) { setError("source-error", "paste a YouTube URL first"); return; }

  const btn = $("analyze-btn");
  btn.disabled = true;
  setProgress("analyze-progress", 0.3, "fetching metadata & transcript…");
  try {
    const r = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || `analyze failed (${r.status})`);
    enterSession(data);
  } catch (e) {
    setError("source-error", e.message);
  } finally {
    btn.disabled = false;
    show($("analyze-progress"), false);
  }
}

async function uploadFile(file) {
  setError("source-error", "");
  setProgress("analyze-progress", 0.3, "uploading…");
  try {
    const fd = new FormData();
    fd.append("file", file);
    const r = await fetch("/api/upload", { method: "POST", body: fd });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || `upload failed (${r.status})`);
    enterSession(data);
  } catch (e) {
    setError("source-error", e.message);
  } finally {
    show($("analyze-progress"), false);
  }
}

function enterSession(data) {
  session = data;
  clips = (data.suggested || []).map((s, i) => ({
    start: s.start,
    end: s.end,
    label: `highlight ${i + 1}`,
    text: s.text || "",
    reasons: s.reasons || [],
    score: s.score || 0,
    on: true,
  }));

  $("video-title").textContent = data.title || "untitled";
  const dur = fmt(data.duration);
  const tsLabel = data.transcript_source === "manual" ? "manual captions"
    : data.transcript_source === "auto" ? "auto captions"
    : "no transcript — fixed intervals";
  $("video-sub").textContent =
    `${dur} · ${data.uploader ? data.uploader + " · " : ""}${tsLabel}` +
    (data.segment_count ? ` · ${data.segment_count} cues` : "");

  const thumb = $("thumb");
  if (data.thumbnail) {
    thumb.src = `/api/thumbnail/${data.session_id}`;
    thumb.hidden = false;
    thumb.onerror = () => { thumb.hidden = true; };
  } else {
    thumb.hidden = true;
  }

  renderClips();
  show($("clips-card"), true);
  show($("render-card"), false);
  $("clips-card").scrollIntoView({ behavior: "smooth", block: "start" });

  // kick off source download + facecam detection right away
  startPrepare();
}

// ---------------------------------------------------------------- prepare
async function startPrepare() {
  if (!session || preparing) return;
  preparing = true;
  try {
    const r = await fetch("/api/prepare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: session.session_id }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || "prepare failed");
    const res = await pollJob(data.job_id, (j) =>
      setProgress("prepare-progress", j.progress, j.message)
    );
    if (res && res.facecam) {
      session.facecam = res.facecam;
    }
    show($("prepare-progress"), false);
  } catch (e) {
    // non-fatal: render will retry preparation
    setProgress("prepare-progress", 1, `note: ${e.message}`);
    setTimeout(() => show($("prepare-progress"), false), 6000);
  } finally {
    preparing = false;
  }
}

// ---------------------------------------------------------------- clip list
function renderClips() {
  const list = $("clips-list");
  list.innerHTML = "";
  clips.forEach((c, i) => {
    const div = document.createElement("div");
    div.className = "clip" + (c.on ? "" : " off");
    const dur = Math.max(0, c.end - c.start);

    const chips = (c.reasons || []).map((r) =>
      `<span class="chip">${escapeHtml(r)}</span>`).join("");
    const scoreChip = c.score > 0
      ? `<span class="chip score">score ${(c.score * 100).toFixed(0)}%</span>` : "";

    div.innerHTML = `
      <input type="checkbox" ${c.on ? "checked" : ""} title="include" />
      <div class="body">
        <div class="times">
          <input type="text" class="t-start" value="${fmt(c.start)}" title="start" />
          <span class="dash">→</span>
          <input type="text" class="t-end" value="${fmt(c.end)}" title="end" />
          <span class="dur">${dur.toFixed(0)}s</span>
        </div>
        <div class="snippet">${escapeHtml(c.text || "(no transcript — adjust times manually)")}</div>
        <div class="reasons">${scoreChip}${chips}</div>
      </div>
      <div class="side">
        <button class="ghost danger" title="remove">✕</button>
      </div>`;

    div.querySelector('input[type="checkbox"]').addEventListener("change", (e) => {
      c.on = e.target.checked;
      div.classList.toggle("off", !c.on);
    });
    const sInput = div.querySelector(".t-start");
    const eInput = div.querySelector(".t-end");
    const commit = () => {
      const s = parseTime(sInput.value);
      const e = parseTime(eInput.value);
      if (!isNaN(s) && !isNaN(e) && e > s) {
        c.start = s; c.end = e;
        div.querySelector(".dur").textContent = `${(e - s).toFixed(0)}s`;
      } else {
        sInput.value = fmt(c.start);
        eInput.value = fmt(c.end);
      }
    };
    sInput.addEventListener("change", commit);
    eInput.addEventListener("change", commit);
    div.querySelector(".danger").addEventListener("click", () => {
      clips.splice(i, 1);
      renderClips();
    });
    list.appendChild(div);
  });

  if (!clips.length) {
    list.innerHTML = `<div class="hint">no clips yet — hit “re-detect” or “manual clip”</div>`;
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (ch) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}

async function resuggest() {
  if (!session) return;
  const btn = $("resuggest-btn");
  btn.disabled = true;
  try {
    const r = await fetch("/api/resuggest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: session.session_id,
        clip_len: Number($("clip-len").value),
        count: Number($("clip-count").value),
      }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || "re-detect failed");
    clips = data.suggested.map((s, i) => ({
      start: s.start, end: s.end, label: `highlight ${i + 1}`,
      text: s.text, reasons: s.reasons, score: s.score, on: true,
    }));
    renderClips();
  } catch (e) {
    setError("render-error", e.message);
  } finally {
    btn.disabled = false;
  }
}

function addManual() {
  const dur = session ? session.duration : 0;
  const last = clips.length ? Math.min(clips[clips.length - 1].end + 2, dur) : 0;
  const start = Math.max(0, Math.min(last, Math.max(0, dur - 15)));
  const end = Math.min(start + 30, dur || start + 30);
  clips.push({
    start, end, label: `manual ${clips.length + 1}`,
    text: "", reasons: ["manual"], score: 0, on: true,
  });
  renderClips();
}

// ---------------------------------------------------------------- render
async function renderShorts() {
  setError("render-error", "");
  const active = clips.filter((c) => c.on && c.end - c.start >= 1);
  if (!active.length) {
    setError("render-error", "select at least one clip of ≥ 1 second");
    return;
  }
  if (!session) return;

  const btn = $("render-btn");
  btn.disabled = true;
  show($("render-card"), true);
  show($("done-row"), false);
  $("render-results").innerHTML = "";
  setProgress("render-progress", 0.02, "preparing…");
  $("render-card").scrollIntoView({ behavior: "smooth", block: "start" });

  try {
    const facecamMode =
      !$("facecam-toggle").checked ? "none"
      : session.facecam ? (session.facecam.custom ? "custom" : "auto")
      : "auto";

    const r = await fetch("/api/render", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: session.session_id,
        clips: active.map((c) => ({ start: c.start, end: c.end, label: c.label })),
        facecam_mode: facecamMode,
        facecam: session.facecam || null,
        captions: $("captions-toggle").checked,
        cam_width_frac: Number($("cam-size").value) / 100,
      }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || `render failed (${r.status})`);

    const result = await pollJob(data.job_id, (j) =>
      setProgress("render-progress", j.progress, j.message)
    );
    showResults(result.clips || []);
    show($("done-row"), true);
    setProgress("render-progress", 1, `done — ${result.clips.length} short(s) ready`);
  } catch (e) {
    setError("render-error", e.message);
    setProgress("render-progress", 0, "failed");
  } finally {
    btn.disabled = false;
  }
}

function showResults(list) {
  const box = $("render-results");
  box.innerHTML = "";
  list.forEach((c) => {
    const div = document.createElement("div");
    div.className = "result";
    const poster = c.poster ? `poster="/outputs/${c.poster}"` : "";
    div.innerHTML = `
      <video controls preload="metadata" ${poster} src="/outputs/${c.file}"></video>
      <div class="cap">
        <div class="name">${escapeHtml(c.label)}</div>
        <div class="sub">${fmt(c.start)} → ${fmt(c.end)} · ${c.duration}s · ${c.width}×${c.height}</div>
        <a href="/outputs/${c.file}" download>⬇ download</a>
      </div>`;
    box.appendChild(div);
  });
}

function downloadAll() {
  document.querySelectorAll("#render-results a[download]").forEach((a) => {
    const link = document.createElement("a");
    link.href = a.href;
    link.download = "";
    document.body.appendChild(link);
    link.click();
    link.remove();
  });
}

function restart() {
  session = null;
  clips = [];
  show($("clips-card"), false);
  show($("render-card"), false);
  $("url-input").value = "";
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ---------------------------------------------------------------- wire up
$("analyze-btn").addEventListener("click", analyze);
$("url-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") analyze();
});
$("file-input").addEventListener("change", (e) => {
  const f = e.target.files && e.target.files[0];
  if (f) uploadFile(f);
  e.target.value = "";
});
$("resuggest-btn").addEventListener("click", resuggest);
$("add-clip-btn").addEventListener("click", addManual);
$("render-btn").addEventListener("click", renderShorts);
$("download-all").addEventListener("click", downloadAll);
$("restart").addEventListener("click", restart);
$("cam-size").addEventListener("input", (e) => {
  $("cam-size-val").textContent = `${e.target.value}%`;
});
