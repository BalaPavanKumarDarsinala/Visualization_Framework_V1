# app_streamlit.py
# -------------------------------------------------------------------
# MOSI Sync Viewer (Enhanced)
# - Video + Transcript highlight + Waveform (click-to-seek)
# - Sentiment bar + Dynamic sentiment badge
# - OpenFace AUs (synced) + Emotion band (rule-based from AUs)
# - Text↔Face Agreement bar
# - Auto-scroll transcript toggle
# - Metrics
# - Pattern timeline (Smile/Frown/Gaze/Gesture/Prosody; click-to-seek)
# - Right-aligned pattern labels
# - Top co-occurring behaviours + optional segments list (PAGINATED)
# - Clip insights panel (bullets)
# - Feedback loop (Google Sheets + CSV fallback)
#
# Expects:
#   data/mosi_videos/<id>.mp4
#   data/transcripts/<id>.csv    (start,end,word)
#   data/waveforms/<id>.json     (list[float] in [-1,1])
#   data/sentiment/<id>.csv      (text,start,end,label,score,polarity) [optional]
#   data/openface/<id>.json      (time[], aus{AUxx_r:[]})               [optional]
#   data/openface_raw/<id>.json  (time[], pose_Rx, pose_Ry, pose_Rz)    [optional]
#   data/patterns/<id>.json      (optional; will be augmented)
#   data/patterns_cooc/<id>.json ({combos}, {segments})                 [optional]
#
# Run: streamlit run app_streamlit.py

from __future__ import annotations
import base64, json, csv
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as st_html

# --- Google Sheets deps ---
import gspread
from google.oauth2.service_account import Credentials

# ---------------- Paths ----------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
VID_DIR = DATA / "mosi_videos"
TRN_DIR = DATA / "transcripts"
WVF_DIR = DATA / "waveforms"
SEN_DIR = DATA / "sentiment"
OF_DIR  = DATA / "openface"
OF_RAW  = DATA / "openface_raw"   # head-pose JSONs
FB_DIR  = DATA / "feedback"
PAT_DIR = DATA / "patterns"
COC_DIR = DATA / "patterns_cooc"

for p in (VID_DIR, TRN_DIR, WVF_DIR, SEN_DIR, OF_DIR, OF_RAW, FB_DIR, PAT_DIR, COC_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ---------------- Google Sheets helpers ----------------
GS_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
DEFAULT_TAB = "feedback"

@st.cache_resource(show_spinner=False)
def _gs_client():
    # Build gspread client from Streamlit secrets
    creds = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]),
        scopes=GS_SCOPES
    )
    return gspread.authorize(creds)

@st.cache_resource(show_spinner=False)
def _gs_worksheet():
    """Open the worksheet; create it (with headers) if missing."""
    gc = _gs_client()
    sh = gc.open_by_key(st.secrets["GSHEET_ID"])
    tab = st.secrets.get("GSHEET_TAB", DEFAULT_TAB)
    headers = [
        "timestamp_utc",
        "clip_id",
        "seg_index",
        "seg_start",
        "seg_end",
        "model_label",
        "model_score",
        "polarity",
        "user_rating",
        "note",
    ]
    try:
        ws = sh.worksheet(tab)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=tab, rows="1000", cols=str(len(headers) + 2))
        ws.append_row(headers, value_input_option="RAW")
        return ws

    # Ensure header row exists/matches
    first = ws.row_values(1)
    if not first:
        ws.append_row(headers, value_input_option="RAW")
    elif first != headers:
        ws.update("1:1", [headers])
    return ws

def append_feedback_to_sheet(row_dict: dict):
    """Append one feedback row to Google Sheets."""
    ws = _gs_worksheet()
    values = [
        datetime.now(timezone.utc).isoformat(),
        row_dict.get("clip_id"),
        int(row_dict.get("seg_index")),
        float(row_dict.get("seg_start")),
        float(row_dict.get("seg_end")),
        row_dict.get("model_label"),
        float(row_dict.get("model_score")),
        float(row_dict.get("polarity")),
        row_dict.get("user_rating"),
        row_dict.get("note") or "",
    ]
    ws.append_row(values, value_input_option="USER_ENTERED")

# ---------------- Helpers ----------------
def list_ids_with_min() -> list[str]:
    vids = {p.stem for p in VID_DIR.glob("*.mp4")}
    trns = {p.stem for p in TRN_DIR.glob("*.csv")}
    wvfs = {p.stem for p in WVF_DIR.glob("*.json")}
    return sorted(list(vids & trns & wvfs))

def read_transcript_records(vid_id: str):
    df = pd.read_csv(TRN_DIR / f"{vid_id}.csv")
    df = df[["start", "end", "word"]].copy()
    df["start"] = df["start"].astype(float)
    df["end"]   = df["end"].astype(float)
    df["word"]  = df["word"].astype(str)
    return df.sort_values("start").to_dict("records")

def read_waveform_samples(vid_id: str):
    return json.loads((WVF_DIR / f"{vid_id}.json").read_text(encoding="utf-8"))

def read_sentences(vid_id: str):
    fp = SEN_DIR / f"{vid_id}.csv"
    if not fp.exists(): return []
    df = pd.read_csv(fp)
    for c in ("text","start","end","label","score","polarity"):
        if c not in df.columns: df[c] = None
    df["start"]    = pd.to_numeric(df["start"], errors="coerce").fillna(0.0)
    df["end"]      = pd.to_numeric(df["end"],   errors="coerce").fillna(df["start"])
    df["score"]    = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
    df["polarity"] = pd.to_numeric(df["polarity"], errors="coerce").fillna(0.0)
    def _norm(x: str | None):
        if not isinstance(x, str): return None
        x = x.strip().upper()
        if x in ("POS","POSITIVE"): return "POSITIVE"
        if x in ("NEG","NEGATIVE"): return "NEGATIVE"
        return None
    df["label"] = df["label"].map(_norm)
    return df[["text","start","end","label","score","polarity"]].to_dict("records")

def read_openface_bundle(vid_id: str):
    fp = OF_DIR / f"{vid_id}.json"
    if not fp.exists(): return {}
    return json.loads(fp.read_text(encoding="utf-8"))

def read_openface_raw(vid_id: str):
    fp = OF_RAW / f"{vid_id}.json"
    if not fp.exists(): return {}
    return json.loads(fp.read_text(encoding="utf-8"))

def read_patterns(vid_id: str):
    fp = PAT_DIR / f"{vid_id}.json"
    if not fp.exists(): return {}
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return {}

def read_cooccurrence(vid_id: str):
    fp = COC_DIR / f"{vid_id}.json"
    if not fp.exists(): return [], []
    try:
        d = json.loads(fp.read_text(encoding="utf-8"))
        combos = d.get("combos", {})
        segs   = d.get("segments", [])
        combo_list = sorted(combos.items(), key=lambda kv: (-kv[1], kv[0]))
        return combo_list, segs
    except Exception:
        return [], []

def b64_video(vid_id: str) -> str:
    b = (VID_DIR / f"{vid_id}.mp4").read_bytes()
    return f"data:video/mp4;base64,{base64.b64encode(b).decode('utf-8')}"

# -------- Metrics (basic) ----------
def compute_basic_metrics(sents: list[dict], of_bundle: dict):
    pos_time = 0.0
    total_time = 0.0
    flips = 0
    last_label = None
    for s in sents:
        dur = max(0.0, float(s["end"]) - float(s["start"]))
        total_time += dur
        lab = s.get("label")
        pol = float(s.get("polarity") or 0.0)
        if not lab:
            lab = "POSITIVE" if pol >= 0 else "NEGATIVE"
        if lab == "POSITIVE":
            pos_time += dur
        if last_label is not None and lab != last_label:
            flips += 1
        last_label = lab

    pos_pct = (pos_time/total_time*100.0) if total_time > 0 else 0.0

    mean_au12 = None
    mean_au04 = None
    aus = (of_bundle.get("aus") or {})
    if "AU12_r" in aus and len(aus["AU12_r"]) > 0:
        mean_au12 = float(pd.Series(aus["AU12_r"]).mean())
    if "AU04_r" in aus and len(aus["AU04_r"]) > 0:
        mean_au04 = float(pd.Series(aus["AU04_r"]).mean())

    return {
        "pos_pct": pos_pct,
        "flips": flips,
        "mean_au12": mean_au12,
        "mean_au04": mean_au04,
        "total_time": total_time
    }

# -------- Insights helpers ----------
def _estimate_duration(words, sents, of_bundle, of_raw=None):
    dur = 0.0
    try:
        if of_bundle and of_bundle.get("time"):
            dur = max(dur, float(of_bundle["time"][-1]))
    except Exception:
        pass
    try:
        if of_raw and of_raw.get("time"):
            dur = max(dur, float(of_raw["time"][-1]))
    except Exception:
        pass
    try:
        if sents:
            dur = max(dur, max(float(s.get("end") or 0.0) for s in sents))
    except Exception:
        pass
    try:
        if words:
            dur = max(dur, max(float(w.get("end") or 0.0) for w in words))
    except Exception:
        pass
    return dur or None

def _sentiment_summary(sents):
    pos_t, tot_t, flips, last = 0.0, 0.0, 0, None
    for s in sents or []:
        st = float(s.get("start") or 0.0)
        en = float(s.get("end")   or st)
        dur = max(0.0, en - st)
        if dur <= 0:
            continue
        lab = s.get("label")
        pol = float(s.get("polarity") or 0.0)
        if not lab:
            lab = "POSITIVE" if pol >= 0 else "NEGATIVE"
        if lab == "POSITIVE":
            pos_t += dur
        tot_t += dur
        if last is not None and lab != last:
            flips += 1
        last = lab
    pos_pct = (pos_t / tot_t * 100.0) if tot_t > 0 else 0.0
    return pos_pct, flips

def _hotspot_windows(segs_list, duration, bins=24, topk=2):
    """Find dense co-occurrence windows by binning segment midpoints."""
    if not segs_list or not duration:
        return []
    mids = []
    for s in segs_list:
        st = float(s.get("start") or 0.0)
        en = float(s.get("end")   or st)
        mids.append(max(0.0, min(duration, 0.5*(st+en))))
    if not mids:
        return []
    counts = [0]*bins
    for m in mids:
        idx = min(bins-1, int((m/duration)*bins))
        counts[idx] += 1
    ranked = sorted(range(bins), key=lambda i: counts[i], reverse=True)[:topk]
    res = []
    for i in ranked:
        t0 = ( i    / bins)*duration
        t1 = ((i+1) / bins)*duration
        res.append((t0, t1, counts[i]))
    res.sort(key=lambda x: x[0])
    return res

def compute_clip_insights(words, sents, of_bundle, cooc_list, segs_list, of_raw=None):
    insights = []
    if cooc_list:
        top3 = cooc_list[:3]
        tops = [f"**{p}** ({c})" for p, c in top3]
        insights.append("Most frequent patterns: " + ", ".join(tops) + ".")
    duration = _estimate_duration(words, sents, of_bundle, of_raw)
    hotspots = _hotspot_windows(segs_list, duration, bins=24, topk=2)
    if hotspots:
        nice = [f"{t0:.0f}–{t1:.0f}s ({cnt} segs)" for (t0, t1, cnt) in hotspots]
        insights.append("Hotspots for co-occurring cues: " + ", ".join(nice) + ".")
    pos_pct, flips = _sentiment_summary(sents)
    insights.append(f"Text sentiment is **{pos_pct:.0f}% positive** with **{flips} flips**.")
    return insights

# -------- Pattern derivation (Gesture + Prosody) ----------
def compute_gesture_bursts(of_raw: dict, z_thresh: float = 1.5):
    """Rapid head motion from pose_Rx/Ry/Rz velocity (z-score > threshold)."""
    if not of_raw or "time" not in of_raw: return [], None
    try:
        T  = np.asarray(of_raw["time"], dtype=float)
        rx = np.asarray(of_raw.get("pose_Rx", []), dtype=float)
        ry = np.asarray(of_raw.get("pose_Ry", []), dtype=float)
        rz = np.asarray(of_raw.get("pose_Rz", []), dtype=float)
        if len(T) < 3 or len(rx) != len(T) or len(ry) != len(T) or len(rz) != len(T):
            return [], T
        dt  = np.diff(T)
        vel = np.sqrt(np.diff(rx)**2 + np.diff(ry)**2 + np.diff(rz)**2) / np.maximum(dt, 1e-6)
        vmu, vsd = float(vel.mean()), float(vel.std() + 1e-6)
        z = (vel - vmu) / vsd
        idx = np.where(z > z_thresh)[0].tolist()  # indices refer to interval [i,i+1]
        return idx, T
    except Exception:
        return [], None

def compute_prosody_change_times(samples: list[float], window: int = 500, dE_thresh: float = 0.25, duration_hint: float | None = None):
    """Energy change points (approx). Returns times in seconds if duration_hint provided, else fractional positions."""
    if not samples or len(samples) < window*2:
        return []
    s = np.asarray(samples, dtype=float)
    # Short-time energy (RMS)
    energy = np.sqrt(np.convolve(s**2, np.ones(window)/window, mode="valid"))
    energy = (energy - energy.min()) / (energy.ptp() + 1e-9)
    dE = np.abs(np.diff(energy))
    peaks = np.where(dE > dE_thresh)[0]  # indices in energy/dE domain
    if duration_hint and duration_hint > 0:
        # Map index -> time by linear proportion of the clip duration
        times = (peaks / max(1, len(dE))) * duration_hint
        return times.tolist()
    else:
        # return fractional positions (0..1)
        return (peaks / max(1, len(dE))).tolist()

# ---------------- App ----------------
st.set_page_config(page_title="MOSI Sync Viewer", layout="wide")
st.title("MOSI Sync Viewer")

ids = list_ids_with_min()
if not ids:
    st.info("Put files under data/ as described, then reload.")
    st.stop()

c1, c2, c3 = st.columns([2,1,1])
with c1: vid = st.selectbox("Choose a clip", ids, index=0)
with c2: show_sent = st.toggle("Show sentiment bar", value=True)
with c3: show_aus  = st.toggle("Show facial AUs (OpenFace)", value=True)

# ---- load ----
video_src = b64_video(vid)
words     = read_transcript_records(vid)
samples   = read_waveform_samples(vid)
sents     = read_sentences(vid) if show_sent else []
of_bundle = read_openface_bundle(vid) if show_aus else {}
of_raw    = read_openface_raw(vid)
patterns  = read_patterns(vid)
cooc_list, segs_list = read_cooccurrence(vid)

# ---- derive pattern timeline base (seconds array) ----
duration_hint = _estimate_duration(words, sents, of_bundle, of_raw)
pat_time = None
if of_bundle.get("time"):
    pat_time = np.asarray(of_bundle["time"], dtype=float)
elif of_raw.get("time"):
    pat_time = np.asarray(of_raw["time"], dtype=float)
elif duration_hint:
    pat_time = np.linspace(0.0, duration_hint, num=1001)
else:
    pat_time = np.linspace(0.0, 1.0, num=1001)  # last resort (normalized)

# Ensure patterns.time exists for the JS mapping
if "time" not in patterns or not patterns.get("time"):
    patterns["time"] = pat_time.tolist()

# ---- compute Gesture & Prosody and align to pattern timeline indices ----
gesture_idx_raw, gT = compute_gesture_bursts(of_raw)
gesture_idx = []
if gesture_idx_raw:
    if gT is not None:
        gTimes = [float(gT[i]) for i in gesture_idx_raw]
        pT = np.asarray(patterns["time"], dtype=float)
        gesture_idx = [int(np.clip(np.searchsorted(pT, t, side="left"), 0, len(pT)-1)) for t in gTimes]

prosody_times = compute_prosody_change_times(samples, window=500, dE_thresh=0.25, duration_hint=duration_hint)
prosody_idx = []
if prosody_times:
    pT = np.asarray(patterns["time"], dtype=float)
    prosody_idx = [int(np.clip(np.searchsorted(pT, float(t), side="left"), 0, len(pT)-1)) for t in prosody_times]

if gesture_idx:
    patterns["gesture_bursts"] = gesture_idx
if prosody_idx:
    patterns["prosody_changes"] = prosody_idx

# ---- AUs to display (only if present) ----
DEFAULT_AUS = ["AU06_r","AU12_r","AU04_r"]
present_aus = sorted(list((of_bundle.get("aus") or {}).keys()))
au_to_plot  = [au for au in DEFAULT_AUS if au in present_aus][:3]
if not au_to_plot and present_aus:
    r_first = [k for k in present_aus if k.endswith("_r")]
    au_to_plot = r_first[:3] if r_first else present_aus[:3]

# --- Metrics (basic)
metrics = compute_basic_metrics(sents, of_bundle) if (sents or of_bundle) else None
if metrics:
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Positive time %", f"{metrics['pos_pct']:.1f}%")
    mcol2.metric("Sentiment flips", f"{metrics['flips']}")
    mcol3.metric("Mean AU12_r (smile)", "-" if metrics['mean_au12'] is None else f"{metrics['mean_au12']:.2f}")
    mcol4.metric("Mean AU04_r (frown)", "-" if metrics['mean_au04'] is None else f"{metrics['mean_au04']:.2f}")

# --- Top patterns (summary box)
if cooc_list:
    st.markdown("#### Top co-occurring behaviours")
    df = pd.DataFrame(cooc_list, columns=["Pattern", "Count"])
    st.dataframe(df, use_container_width=True, hide_index=True)

    bullets = compute_clip_insights(words, sents, of_bundle, cooc_list, segs_list, of_raw)
    if bullets:
        st.markdown("#### Clip insights")
        for b in bullets:
            st.markdown(f"- {b}")

# --- Optional: segments list (PAGINATED, shows ALL)
show_segs = st.toggle("Show pattern segments (2s windows)", value=False) if segs_list else False
if show_segs and segs_list:
    st.markdown("#### Pattern segments")
    page_size = st.selectbox(
        "Rows per page",
        [20, 30, 50, 100, 200, 500],
        index=0,
        key=f"seg_ps_{vid}",
        help="How many segments to show per page"
    )
    total = len(segs_list)
    total_pages = max(1, (total + page_size - 1) // page_size)
    pg_key = f"seg_pg_{vid}"
    if pg_key not in st.session_state:
        st.session_state[pg_key] = 1

    cA, cB, cC, cD, cE = st.columns([1,1,3,1,1])
    with cA:
        if st.button("⏮ First", disabled=st.session_state[pg_key] <= 1, key=f"first_{vid}"):
            st.session_state[pg_key] = 1
    with cB:
        if st.button("◀ Prev", disabled=st.session_state[pg_key] <= 1, key=f"prev_{vid}"):
            st.session_state[pg_key] -= 1
    with cD:
        if st.button("Next ▶", disabled=st.session_state[pg_key] >= total_pages, key=f"next_{vid}"):
            st.session_state[pg_key] += 1
    with cE:
        if st.button("Last ⏭", disabled=st.session_state[pg_key] >= total_pages, key=f"last_{vid}"):
            st.session_state[pg_key] = total_pages
    with cC:
        st.markdown(
            f"Page **{st.session_state[pg_key]}** / **{total_pages}** — "
            f"showing **{page_size}** per page — **{total}** total segments"
        )

    start = (st.session_state[pg_key] - 1) * page_size
    end = min(start + page_size, total)

    for j in range(start, end):
        seg = segs_list[j]
        labels = ", ".join(seg.get("labels", []))
        st.write(f"**{j+1}.** {seg['start']:.2f}s → {seg['end']:.2f}s — {labels}")

    export_df = pd.DataFrame(
        [{"index": i+1,
          "start": s.get("start"),
          "end": s.get("end"),
          "labels": ", ".join(s.get("labels", []))}
         for i, s in enumerate(segs_list)]
    )
    st.download_button(
        "Download all segments (CSV)",
        export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{vid}_pattern_segments.csv",
        mime="text/csv",
        help="Exports ALL segments for this video (not just current page)"
    )

# canvases
CANVAS_W, CANVAS_H = 640, 110
SENT_H  = 12
AGREE_H = 10
EMO_H   = 10
AU_H    = 120
PAT_H   = 84  # slightly taller to fit 5 lanes nicely

# ----------------- Component (HTML+JS) -----------------
component_html = f"""
<!doctype html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<style>
  :root {{ color-scheme: dark; }}
  body {{ font-family: system-ui,-apple-system,Segoe UI,Roboto,sans-serif; background:#0e1117; color:#e7e7e7; margin:0; }}
  .wrap {{ display:flex; gap:24px; padding:10px; }}
  video {{ width:640px; max-width:60vw; border-radius:8px; display:block; }}
  .right {{ flex:1; min-width:260px; }}
  .panel {{ display:flex; gap:16px; align-items:center; margin:8px 0 6px; font-size:14px; flex-wrap:wrap; }}
  .lbl {{ font-size:12px; color:#bfbfbf; margin:6px 0 2px 2px; }}
  .transcript {{ line-height:1.9; max-height:420px; overflow:auto; padding-right:8px; border-left:1px solid #222; padding-left:16px; }}
  .word {{ padding:2px 4px; margin-right:2px; cursor:pointer; border-radius:4px; }}
  .word.active {{ background:#ffeb3b; color:#000; }}
  canvas {{ display:block; margin-top:6px; border:1px solid #444; background:#111; border-radius:4px; }}
  #wave {{ cursor: pointer; }}
  #au   {{ cursor: pointer; }}

  /* Sentiment & Emotion badges */
  #sentLabel, #emoLabel {{
    margin-top:10px;
    font-size:16px;
    font-weight:700;
    display:inline-block;
    padding:4px 10px;
    border-radius:999px;
    line-height:1.25;
    letter-spacing:.2px;
    background:#1d1f26;
    color:#e7e7e7;
  }}
  #emoLabel{{ margin-left:10px; }}
  #sentLabel.pos {{ background:#15803d; color:#fff; }}
  #sentLabel.neg {{ background:#b91c1c; color:#fff; }}
  #emoLabel.joy    {{ background:#d4f542; color:#1a1a1a; }}
  #emoLabel.anger  {{ background:#ef4444; color:#fff; }}
  #emoLabel.sad    {{ background:#60a5fa; color:#0a1a2b; }}
  #emoLabel.neutral{{ background:#6b7280; color:#fff; }}

  /* Pattern labels in a right-side column */
  .patrow {{ display:flex; align-items:center; gap:10px; }}
  .patlabels {{
    width:80px;
    display:flex; flex-direction:column;
    justify-content:space-between;
    height:{PAT_H}px;
    font-size:12px; color:#bfbfbf;
    line-height:1; user-select:none; text-align:left;
  }}
  .patlabels div {{ display:flex; align-items:center; height:{PAT_H//5}px; }}
</style></head>
<body>
<div class="wrap">
  <div>
    <video id="v" controls src="{video_src}"></video>

    <div class="panel">
      <label><input type="checkbox" id="autos" checked> Auto-scroll transcript</label>
    </div>

    <div class="lbl">Waveform</div>
    <canvas id="wave" width="{CANVAS_W}" height="{CANVAS_H}"></canvas>

    {"<div class='lbl'>Sentiment (text)</div>" if len(sents)>0 else ""}
    {"<canvas id='sentbar'  width='"+str(CANVAS_W)+"' height='"+str(SENT_H)+"'></canvas>" if len(sents)>0 else ""}

    {"<div class='lbl'>Text ↔ Face Agreement</div>" if (len(sents)>0 and len(of_bundle)>0) else ""}
    {"<canvas id='agreebar' width='"+str(CANVAS_W)+"' height='"+str(AGREE_H)+"'></canvas>" if (len(sents)>0 and len(of_bundle)>0) else ""}

    {"<div class='lbl'>Emotion (AUs)</div>" if len(of_bundle)>0 else ""}
    {"<canvas id='emoband'  width='"+str(CANVAS_W)+"' height='"+str(EMO_H)+"'></canvas>" if len(of_bundle)>0 else ""}

    {"<div class='lbl'>Patterns (click ticks to seek)</div>" if len(patterns)>0 else ""}
    {(
      "<div class='patrow'>"
      + "<canvas id='pat' width='"+str(CANVAS_W)+"' height='"+str(PAT_H)+"' style='margin-top:2px;'></canvas>"
      + "<div class='patlabels'><div>Smile</div><div>Frown</div><div>Gaze</div><div>Gesture</div><div>Prosody</div></div>"
      + "</div>"
      ) if len(patterns)>0 else ""}

    <div id="sentLabel"></div>
    <div id="emoLabel"></div>

    {"<div class='lbl'>Action Units</div>" if len(of_bundle)>0 else ""}
    {"<canvas id='au' width='"+str(CANVAS_W)+"' height='"+str(AU_H)+"'></canvas>" if len(of_bundle)>0 else ""}
  </div>

  <div class="right">
    <div class="transcript" id="tx"></div>
  </div>
</div>

<script>
const words    = {json.dumps(read_transcript_records(vid))};
const samples  = {json.dumps(samples)};
const sents    = {json.dumps(sents)};
const ofBundle = {json.dumps(of_bundle)};
const auToPlot = {json.dumps(au_to_plot)};
const patterns = {json.dumps(patterns)};

const v = document.getElementById("v");
const box = document.getElementById("tx");
const autos = document.getElementById("autos");
const cvs = document.getElementById("wave");
const ctx = cvs.getContext("2d");
const sentCanvas  = document.getElementById("sentbar");
const sentCtx     = sentCanvas ? sentCanvas.getContext("2d") : null;
const agreeCanvas = document.getElementById("agreebar");
const agreeCtx    = agreeCanvas ? agreeCanvas.getContext("2d") : null;
const emoCanvas   = document.getElementById("emoband");
const emoCtx      = emoCanvas ? emoCanvas.getContext("2d") : null;
const patCanvas   = document.getElementById("pat");
const patCtx      = patCanvas ? patCanvas.getContext("2d") : null;
const sentLabel = document.getElementById("sentLabel");
const emoLabel  = document.getElementById("emoLabel");
const aucvs = document.getElementById("au");
const auctx = aucvs ? aucvs.getContext("2d") : null;

// ---------- Transcript ----------
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

    if (sents && sents.length) {{
      const s = sents.find(ss => w.start >= ss.start && w.end <= ss.end);
      if (s) {{
        const sc = (typeof s.score === "number" ? s.score.toFixed(2) : s.score);
        const po = (typeof s.polarity === "number" ? s.polarity.toFixed(2) : s.polarity);
        span.title = `Sentiment: ${{s.label}} | Score: ${{sc}} | Polarity: ${{po}}`;
      }}
    }}
    frag.appendChild(span);
  }}
  box.innerHTML = "";
  box.appendChild(frag);
}}
renderTranscript();
function spans() {{ return box.children; }}

// ---------- Waveform (click-to-seek) ----------
function drawWaveform(time) {{
  ctx.clearRect(0,0,cvs.width,cvs.height);
  const mid = cvs.height * 0.5;
  const L = samples.length || 1;
  ctx.beginPath(); ctx.moveTo(0, mid);
  for (let i=0;i<L;i++) {{
    const x = i * (cvs.width / L);
    const y = mid - samples[i] * (cvs.height * 0.45);
    ctx.lineTo(x, y);
  }}
  ctx.strokeStyle = "#0f0"; ctx.lineWidth = 1; ctx.stroke();
  if (v.duration) {{
    const x = (time / v.duration) * cvs.width;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, cvs.height);
    ctx.strokeStyle = "red"; ctx.lineWidth = 1; ctx.stroke();
  }}
}}
if (cvs) {{
  cvs.addEventListener("click", (e) => {{
    if (!v.duration) return;
    const rect = cvs.getBoundingClientRect();
    const x = e.clientX - rect.left;
    v.currentTime = Math.min(Math.max(x / cvs.width, 0), 1) * v.duration;
  }});
}}

// ---------- Sentiment bar ----------
function drawSentiment() {{
  if (!sentCtx || !v.duration) return;
  sentCtx.clearRect(0,0,sentCanvas.width,sentCanvas.height);
  for (const s of sents) {{
    const x0 = (s.start / v.duration) * sentCanvas.width;
    const x1 = (s.end   / v.duration) * sentCanvas.width;
    const pol = Math.max(-1, Math.min(1, Number(s.polarity||0)));
    const g = Math.round(200 * Math.max(0,  pol));
    const r = Math.round(200 * Math.max(0, -pol));
    sentCtx.fillStyle = `rgb(${{r}},${{g}},0)`;
    sentCtx.fillRect(x0, 0, Math.max(1, x1-x0), sentCanvas.height);
  }}
}}

// ---------- Emotion mapping (rule-based from AUs) ----------
function computeEmotionAt(i) {{
  if (!ofBundle || !ofBundle.aus) return "NEUTRAL";
  const aus = ofBundle.aus;
  const get = (k) => (aus[k] && typeof aus[k][i] === "number") ? aus[k][i] : 0;

  const AU01 = get("AU01_r"), AU02 = get("AU02_r"), AU04 = get("AU04_r"),
        AU05 = get("AU05_r"), AU06 = get("AU06_r"), AU07 = get("AU07_r"),
        AU12 = get("AU12_r"), AU15 = get("AU15_r");

  if (AU12 > 1.0 && AU06 > 0.5) return "JOY";
  if ((AU04 > 1.2 && AU05 > 0.5) || AU07 > 1.0) return "ANGER";
  if ((AU01 + AU04) > 1.5 && AU15 > 0.5) return "SAD";
  return "NEUTRAL";
}}
function drawEmotionBand() {{
  if (!emoCtx || !ofBundle.time || !v.duration) return;
  const W = emoCanvas.width, H = emoCanvas.height, T = ofBundle.time;
  emoCtx.clearRect(0,0,W,H);
  for (let i=0;i<T.length-1;i++) {{
    const x0 = (T[i]   / v.duration) * W;
    const x1 = (T[i+1] / v.duration) * W;
    const emo = computeEmotionAt(i);
    let color = "#666";
    if (emo === "JOY") color = "#d4f542";
    else if (emo === "ANGER") color = "#f44";
    else if (emo === "SAD") color = "#6aa0ff";
    emoCtx.fillStyle = color;
    emoCtx.fillRect(x0, 0, Math.max(1, x1-x0), H);
  }}
}}

// ---------- Agreement bar (text vs face valence) ----------
function facialValenceAt(i) {{
  const aus = ofBundle.aus || {{}};
  const get = (k) => (aus[k] && typeof aus[k][i] === "number") ? aus[k][i] : 0;
  return get("AU12_r") - 0.5*(get("AU04_r") + get("AU15_r")); // >0 => positive face
}}
function sentimentAtTime(t) {{
  if (!sents || !sents.length) return null;
  for (const s of sents) if (t >= s.start && t <= s.end) return s;
  return null;
}}
function drawAgreement() {{
  if (!agreeCtx || !ofBundle.time || !v.duration) return;
  const W = agreeCanvas.width, H = agreeCanvas.height, T = ofBundle.time;
  agreeCtx.clearRect(0,0,W,H);
  for (let i=0;i<T.length-1;i++) {{
    const mid = (T[i] + T[i+1]) * 0.5;
    const seg = sentimentAtTime(mid);
    const x0 = (T[i]   / v.duration) * W;
    const x1 = (T[i+1] / v.duration) * W;
    let color = "#444";
    if (seg) {{
      const pol = Number(seg.polarity || 0);
      const fval = facialValenceAt(i);
      const agree = (pol >= 0 && fval >= 0) || (pol < 0 && fval < 0);
      color = agree ? "#19d17c" : "#ff8a65";
    }}
    agreeCtx.fillStyle = color;
    agreeCtx.fillRect(x0, 0, Math.max(1, x1-x0), H);
  }}
}}

// ---------- Pattern ticks (Smile/Frown/Gaze/Gesture/Prosody) ----------
function drawPatterns() {{
  if (!patCtx || !patterns || !patterns.time || !v.duration) return;
  const W = patCanvas.width, H = patCanvas.height;
  patCtx.clearRect(0,0,W,H);

  const ROWS = 5;
  const laneY = (rowIdx) => Math.round(((rowIdx + 0.5) / ROWS) * H);

  const lanes = [
    {{key:"AU12_r_peaks",   color:"#7dd3fc", y: laneY(0)}},  // Smile
    {{key:"AU04_r_peaks",   color:"#f472b6", y: laneY(1)}},  // Frown
    {{key:"gaze_shifts",    color:"#facc15", y: laneY(2)}},  // Gaze
    {{key:"gesture_bursts", color:"#34d399", y: laneY(3)}},  // Gesture
    {{key:"prosody_changes",color:"#a78bfa", y: laneY(4)}},  // Prosody
  ];

  function xOfIdx(i){{ return (patterns.time[Math.min(i, patterns.time.length-1)] / v.duration) * W; }}

  patCanvas._tickTimes = [];

  lanes.forEach(l => {{
    const arr = patterns[l.key];
    if (!arr) return;
    patCtx.strokeStyle = l.color;
    patCtx.lineWidth = 2;
    arr.forEach(i => {{
      const x = xOfIdx(i);
      patCtx.beginPath(); patCtx.moveTo(x, l.y-6); patCtx.lineTo(x, l.y+6); patCtx.stroke();
      patCanvas._tickTimes.push(x / W * v.duration);
    }});
  }});

  const x = (v.currentTime / v.duration) * W;
  patCtx.beginPath(); patCtx.moveTo(x, 0); patCtx.lineTo(x, H);
  patCtx.strokeStyle = "red"; patCtx.lineWidth = 1; patCtx.stroke();
}}

if (patCanvas) {{
  patCanvas.addEventListener("click", (e) => {{
    if (!v.duration || !patCanvas._tickTimes || !patCanvas._tickTimes.length) return;
    const rect = patCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const t = Math.min(Math.max(x / patCanvas.width, 0), 1) * v.duration;
    // snap to nearest tick (≤1s) if available; else jump to position
    let best = t, bestd = 1e9;
    for (const tt of patCanvas._tickTimes) {{
      const d = Math.abs(t - tt);
      if (d < bestd) {{ best = tt; bestd = d; }}
    }}
    v.currentTime = (bestd < 1.0) ? best : t;
  }});
}}

// ---------- AUs (click-to-seek) ----------
const AU_COLORS = ["#4da6ff", "#ffd24d", "#ff66cc", "#66ff99", "#ff8c66", "#b366ff"];
function drawAUs(time) {{
  if (!auctx || !ofBundle.aus || !ofBundle.time || !v.duration) return;
  const W = aucvs.width, H = aucvs.height;
  auctx.clearRect(0,0,W,H);
  const xFromSec = s => (s / v.duration) * W;

  const T = ofBundle.time;
  auToPlot.forEach((auName, k) => {{
    const arr = (ofBundle.aus || {{}})[auName];
    if (!arr) return;
    const maxV = 5.0;
    auctx.beginPath();
    for (let i=0;i<arr.length;i++) {{
      const x = xFromSec(T[i]);
      const y = H - (Math.max(0, Math.min(arr[i], maxV)) / maxV) * (H - 14) - 7;
      if (i===0) auctx.moveTo(x,y); else auctx.lineTo(x,y);
    }}
    auctx.strokeStyle = AU_COLORS[k % AU_COLORS.length];
    auctx.lineWidth = 1.2; auctx.stroke();

    auctx.fillStyle = AU_COLORS[k % AU_COLORS.length];
    auctx.fillRect(6 + k*90, 6, 10, 10);
    auctx.fillStyle = "#ddd";
    auctx.font = "12px system-ui, sans-serif";
    auctx.fillText(auName, 20 + k*90, 15);
  }});

  const x = xFromSec(time);
  auctx.beginPath(); auctx.moveTo(x, 0); auctx.lineTo(x, H);
  auctx.strokeStyle = "red"; auctx.lineWidth = 1; auctx.stroke();
}}
if (aucvs) {{
  aucvs.addEventListener("click", (e) => {{
    if (!v.duration) return;
    const rect = aucvs.getBoundingClientRect();
    const x = e.clientX - rect.left;
    v.currentTime = Math.min(Math.max(x / aucvs.width, 0), 1) * v.duration;
  }});
}}

// ---------- Emotion label helper ----------
function emotionAtTime(tSec) {{
  if (!ofBundle || !ofBundle.time || !ofBundle.time.length) return "NEUTRAL";
  const T = ofBundle.time;
  let i = 0;
  while (i+1 < T.length && Math.abs(T[i+1] - tSec) < Math.abs(T[i] - tSec)) i++;
  return computeEmotionAt(i);
}}

// ---------- Scrub-safe highlighting ----------
let active = -1;
function clearActive() {{
  if (!box) return;
  const cs = spans();
  for (let k = 0; k < cs.length; k++) cs[k].classList.remove("active");
  active = -1;
}}

// ---------- Sync ----------
function syncLoop() {{
  const t = Math.max(0, v.currentTime);

  // transcript pointer
  let i = active;
  if (i<0 || i>=words.length || t<words[i].start || t>=words[i].end) {{
    while (i+1<words.length && t>=words[i+1].start) i++;
    while (i>0 && t<words[i].start) i--;
  }}
  const ok = i>=0 && i<words.length && t>=words[i].start && t<words[i].end;
  if (ok && i!==active) {{
    if (active>=0) spans()[active].classList.remove("active");
    spans()[i].classList.add("active");
    active = i;
    if (autos && autos.checked) {{
      const el = spans()[i], pr = box.getBoundingClientRect(), er = el.getBoundingClientRect();
      if (er.top < pr.top || er.bottom > pr.bottom) el.scrollIntoView({{block:"center", behavior:"smooth"}});
    }}
  }}

  drawWaveform(t);
  if (sentCtx)  drawSentiment();
  if (agreeCtx) drawAgreement();
  if (emoCtx)   drawEmotionBand();
  if (patCtx)   drawPatterns();
  if (auctx)    drawAUs(t);

  // dynamic badges
  if (sents && sents.length && v.duration) {{
    const seg = sents.find(ss => t >= ss.start && t <= ss.end);
    if (seg) {{
      const label = seg.label ? seg.label.toUpperCase() : (seg.polarity >= 0 ? "POSITIVE" : "NEGATIVE");
      const nice  = label.charAt(0) + label.slice(1).toLowerCase();
      sentLabel.textContent = "Sentiment: " + nice;
      sentLabel.className = (label === "POSITIVE") ? "pos" : "neg";
    }} else {{
      sentLabel.textContent = ""; sentLabel.className = "";
    }}
  }}

  if (ofBundle && ofBundle.time && ofBundle.time.length) {{
    const emo = emotionAtTime(t);
    let cls = "neutral", pretty = "Neutral";
    if (emo === "JOY")    {{ cls = "joy";   pretty = "Joy"; }}
    if (emo === "ANGER")  {{ cls = "anger"; pretty = "Anger"; }}
    if (emo === "SAD")    {{ cls = "sad";   pretty = "Sad"; }}
    emoLabel.textContent = "Emotion: " + pretty;
    emoLabel.className = cls;
  }} else {{
    emoLabel.textContent = "";
    emoLabel.className = "";
  }}

  requestAnimationFrame(syncLoop);
}}
v.addEventListener("play", () => requestAnimationFrame(syncLoop));
v.addEventListener("pause", clearActive);
v.addEventListener("seeking", clearActive);
v.addEventListener("seeked", () => {{
  const t = Math.max(0, v.currentTime);
  drawWaveform(t);
  if (sentCtx)  drawSentiment();
  if (agreeCtx) drawAgreement();
  if (emoCtx)   drawEmotionBand();
  if (patCtx)   drawPatterns();
  if (auctx)    drawAUs(t);
}});

// init
drawWaveform(0);
if (sentCtx)  drawSentiment();
if (agreeCtx) drawAgreement();
if (emoCtx)   drawEmotionBand();
if (patCtx)   drawPatterns();
if (auctx)    drawAUs(0);
</script>
</body></html>
"""

st_html(
    component_html,
    height= 960 if (len(of_bundle)>0 or len(patterns)>0) else (800 if len(sents)>0 else 740),
    scrolling=True
)

# ---- Feedback form (BOTTOM) ----
st.markdown("### Feedback")
if sents:
    seg_labels = [
        f"[{i:02d}] {max(0.0,float(s['start'])):.2f}-{max(0.0,float(s['end'])):.2f}s  |  "
        f"{str(s.get('label') or ('POS' if (s.get('polarity',0)>=0) else 'NEG'))}  |  "
        f"{(s.get('text') or '').strip()[:80]}"
        for i,s in enumerate(sents)
    ]
    with st.form("feedback_form", clear_on_submit=True):
        seg_idx = st.selectbox("Segment to rate", list(range(len(sents))), format_func=lambda i: seg_labels[i])
        rating  = st.radio("Is the sentiment label correct?", ["✔ Correct","✖ Incorrect"], horizontal=True, index=0)
        note    = st.text_area("Notes (optional)")
        submitted = st.form_submit_button("Save feedback")
        if submitted:
            row = sents[seg_idx]
            newrow = {
                "clip_id": vid,
                "seg_index": seg_idx,
                "seg_start": float(row["start"]),
                "seg_end":   float(row["end"]),
                "model_label": row.get("label") or ("POSITIVE" if (float(row.get("polarity") or 0.0) >= 0) else "NEGATIVE"),
                "model_score": float(row.get("score") or 0.0),
                "polarity": float(row.get("polarity") or 0.0),
                "user_rating": "correct" if rating.startswith("✔") else "incorrect",
                "note": note,
            }

            # Try Google Sheets first
            saved_to_sheets = False
            try:
                append_feedback_to_sheet(newrow)
                saved_to_sheets = True
                st.success("Saved feedback to Google Sheets ✅")
            except Exception as e:
                st.warning("Could not write to Google Sheets; saving locally as fallback.")
                st.caption(f"(Sheets error: {e})")

            # Local CSV fallback (and optional backup even if Sheets worked)
            try:
                fb_path = FB_DIR / f"{vid}.csv"
                write_header = not fb_path.exists()
                with fb_path.open("a", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(newrow.keys()))
                    if write_header: w.writeheader()
                    w.writerow(newrow)
                if not saved_to_sheets:
                    st.success(f"Saved feedback locally → {fb_path.name} ✅")
            except Exception as e2:
                st.error(f"Local CSV save failed: {e2}")
