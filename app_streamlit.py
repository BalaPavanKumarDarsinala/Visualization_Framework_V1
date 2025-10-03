# app_streamlit.py
# -------------------------------------------------------------------
# Streamlit Cloud–ready app:
# - Lists clips that have all three files:
#     data/mosi_videos/<id>.mp4
#     data/transcripts/<id>.csv   (id,start,end,word)
#     data/waveforms/<id>.json    (list[float] in [-1,1])
# - Embeds video + transcript + waveform in one HTML component (no flaky JS bridges)
#
# Run locally:   streamlit run app_streamlit.py
# Deploy:        push this and a small sample in data/ to GitHub, deploy on Streamlit Cloud

from __future__ import annotations
import base64
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as st_html

# ---------------------------- Paths ---------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
VID_DIR = DATA / "mosi_videos"
TRN_DIR = DATA / "transcripts"
WVF_DIR = DATA / "waveforms"

for p in (VID_DIR, TRN_DIR, WVF_DIR):
    p.mkdir(parents=True, exist_ok=True)

# --------------------------- Helpers --------------------------------
def list_ids_with_all() -> list[str]:
    vids = {p.stem for p in VID_DIR.glob("*.mp4")}
    trns = {p.stem for p in TRN_DIR.glob("*.csv")}
    wvfs = {p.stem for p in WVF_DIR.glob("*.json")}
    return sorted(list(vids & trns & wvfs))

def read_transcript_records(vid_id: str):
    df = pd.read_csv(TRN_DIR / f"{vid_id}.csv")
    # keep only needed cols; be forgiving with extra columns
    df = df[["start", "end", "word"]].copy()
    df["start"] = df["start"].astype(float)
    df["end"] = df["end"].astype(float)
    df["word"] = df["word"].astype(str)
    df = df.sort_values("start").reset_index(drop=True)
    return df.to_dict("records")

def read_waveform_samples(vid_id: str):
    jsn = WVF_DIR / f"{vid_id}.json"
    return json.loads(jsn.read_text(encoding="utf-8"))

def b64_video(vid_id: str) -> str:
    mp4_path = VID_DIR / f"{vid_id}.mp4"
    data = mp4_path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:video/mp4;base64,{b64}"

# --------------------------- App UI ---------------------------------
st.set_page_config(page_title="MOSI Sync Viewer", layout="wide")
st.title("MOSI Sync Viewer — Video · Transcript · Waveform")

ids = list_ids_with_all()
if not ids:
    st.info(
        "Add at least one clip with all files:\n\n"
        "• data/mosi_videos/<id>.mp4\n"
        "• data/transcripts/<id>.csv  (columns: id,start,end,word)\n"
        "• data/waveforms/<id>.json   (list[float] in [-1,1])"
    )
    st.stop()

vid = st.selectbox("Choose a clip", ids, index=0)

# Prepare data for the embedded component (kept inside one iframe)
video_src = b64_video(vid)                 # robust for Cloud/demo; fine for small samples
words = read_transcript_records(vid)       # [{start,end,word}, ...]
samples = read_waveform_samples(vid)       # [float, ...] in [-1,1]

# You can tweak canvas width/height here (also adjust the <video> width)
CANVAS_W, CANVAS_H = 640, 110

component_html = f"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  :root {{ color-scheme: dark; }}
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background:#0e1117; color:#e7e7e7; margin:0; }}
  .wrap {{ display:flex; gap:24px; padding:10px 10px 16px 10px; }}
  video {{ width: 640px; max-width: 60vw; border-radius: 8px; display:block; }}
  .right {{ flex:1; min-width: 260px; }}
  .panel {{ margin: 6px 0 10px 0; display:flex; align-items:center; gap:10px; }}
  .transcript {{ line-height: 1.9; max-height: 420px; overflow:auto; padding-right: 8px; border-left: 1px solid #222; padding-left: 16px; }}
  .word {{ padding:2px 4px; margin-right:2px; cursor:pointer; border-radius:4px; }}
  .word.active {{ background:#ffeb3b; color:#000; }}
  canvas {{ display:block; margin-top:12px; border:1px solid #444; background:#111; border-radius:4px; }}
  select, button, label {{ font-size: 14px; }}
</style>
</head>
<body>
<div class="wrap">
  <div>
    <video id="v" controls src="{video_src}"></video>
    <div class="panel">
      <label>Rate:
        <select id="rate">
          <option>0.5</option><option>0.75</option><option selected>1.0</option>
          <option>1.25</option><option>1.5</option><option>2.0</option>
        </select>
      </label>
      <label><input type="checkbox" id="autos" checked> Auto-scroll transcript</label>
      <label>Offset (ms):
        <input id="off" type="range" min="-500" max="500" value="0" step="10">
        <span id="offv">0</span>
      </label>
    </div>
    <canvas id="wave" width="{CANVAS_W}" height="{CANVAS_H}"></canvas>
  </div>
  <div class="right">
    <div class="transcript" id="tx"></div>
  </div>
</div>

<script>
const words   = {json.dumps(words)};
const samples = {json.dumps(samples)};

const v   = document.getElementById("v");
const box = document.getElementById("tx");
const cvs = document.getElementById("wave");
const ctx = cvs.getContext("2d");

const rateSel = document.getElementById("rate");
rateSel.onchange = () => v.playbackRate = parseFloat(rateSel.value);
v.playbackRate = parseFloat(rateSel.value);

const off = document.getElementById("off");
const offv = document.getElementById("offv");
off.addEventListener("input", ()=> offv.textContent = off.value);
offv.textContent = off.value;

// --- transcript render
function renderTranscript() {{
  const frag = document.createDocumentFragment();
  for (let i=0;i<words.length;i++) {{
    const w = words[i];
    const span = document.createElement("span");
    span.className = "word";
    span.dataset.i = i;
    span.dataset.s = w.start;
    span.dataset.e = w.end;
    span.textContent = w.word + " ";
    span.onclick = () => v.currentTime = w.start;
    frag.appendChild(span);
  }}
  box.innerHTML = "";
  box.appendChild(frag);
}}
renderTranscript();
function spans() {{ return box.children; }}

// --- waveform draw
function drawWaveform(time) {{
  ctx.clearRect(0,0,cvs.width,cvs.height);
  const mid = cvs.height * 0.5;
  const L = samples.length || 1;

  ctx.beginPath();
  ctx.moveTo(0, mid);
  for (let i=0;i<L;i++) {{
    const x = i * (cvs.width / L);
    const y = mid - samples[i] * (cvs.height * 0.45);
    ctx.lineTo(x, y);
  }}
  ctx.strokeStyle = "#0f0";
  ctx.lineWidth = 1;
  ctx.stroke();

  if (v.duration && Number.isFinite(v.duration)) {{
    const x = (time / v.duration) * cvs.width;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, cvs.height);
    ctx.strokeStyle = "red";
    ctx.lineWidth = 1;
    ctx.stroke();
  }}
}}

// click-to-seek on waveform
cvs.addEventListener("click", (e) => {{
  if (!v.duration || !Number.isFinite(v.duration)) return;
  const r = cvs.getBoundingClientRect();
  const x = e.clientX - r.left;
  const rel = Math.min(Math.max(x / cvs.width, 0), 1);
  v.currentTime = rel * v.duration;
}});

// keyboard shortcuts: space/play, arrows seek, +/- rate
document.addEventListener("keydown", (e) => {{
  if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
  if (e.code === "Space") {{ e.preventDefault(); v.paused ? v.play() : v.pause(); }}
  if (e.key === "ArrowRight") v.currentTime = Math.min(v.currentTime + (e.shiftKey?5:1), v.duration||v.currentTime);
  if (e.key === "ArrowLeft")  v.currentTime = Math.max(v.currentTime - (e.shiftKey?5:1), 0);
  if (e.key === "=" || e.key === "+") v.playbackRate = Math.min((v.playbackRate||1)+0.25, 3);
  if (e.key === "-")           v.playbackRate = Math.max((v.playbackRate||1)-0.25, 0.25);
}});

// --- sync loop
let active = -1;
const autos = document.getElementById("autos");

function syncLoop() {{
  const t = Math.max(0, v.currentTime + (parseInt(off.value,10)/1000));
  let i = active;

  if (i < 0 || i >= words.length || t < words[i].start || t >= words[i].end) {{
    while (i + 1 < words.length && t >= words[i + 1].start) i++;
    while (i > 0 && t < words[i].start) i--;
  }}

  const ok = i >= 0 && i < words.length && t >= words[i].start && t < words[i].end;
  if (ok && i !== active) {{
    if (active >= 0) spans()[active].classList.remove("active");
    spans()[i].classList.add("active");
    active = i;

    if (autos && autos.checked) {{
      const el = spans()[i];
      const pr = box.getBoundingClientRect();
      const er = el.getBoundingClientRect();
      if (er.top < pr.top || er.bottom > pr.bottom) {{
        el.scrollIntoView({{ block:"center", behavior:"smooth" }});
      }}
    }}
  }}

  drawWaveform(t);
  requestAnimationFrame(syncLoop);
}}

v.addEventListener("play", () => requestAnimationFrame(syncLoop));
v.addEventListener("seeking", () => {{ active = -1; }});
</script>
</body>
</html>
"""

st_html(component_html, height=600, scrolling=True)
