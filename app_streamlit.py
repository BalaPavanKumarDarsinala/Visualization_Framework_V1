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
# - RIGHT column: Transcript + mini-graphs (Sentiment Donut, Signal coverage KPI, Agreement 2x2)
# - Top co-occurring behaviours + optional segments list (PAGINATED)
# - Clip insights panel (bullets)
# - Feedback loop (Google Sheets + CSV fallback)
#
# Data layout:
#   data/mosi_videos/<id>.mp4
#   data/mosi_videos/<id>_overlay.mp4      (optional, created by merge step)
#   data/transcripts/<id>.csv              (start,end,word)
#   data/waveforms/<id>.json               (list[float] in [-1,1])
#   data/sentiment/<id>.csv                (text,start,end,label,score,polarity) [optional]
#   data/openface/<id>.json                (time[], aus{AUxx_r:[]})               [optional]
#   data/openface_raw/<id>.json            (time[], pose_Rx, pose_Ry, pose_Rz)    [optional]
#   data/patterns/<id>.json                (optional; will be augmented)
#   data/patterns_cooc/<id>.json           ({combos}, {segments})                 [optional]
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

def b64_from_path(path: Path) -> str:
    """Base64-encode a *full* video path (overlay/non-overlay)."""
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    b = path.read_bytes()
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
        if dur <= 0: continue
        lab = s.get("label")
        pol = float(s.get("polarity") or 0.0)
        if not lab:
            lab = "POSITIVE" if pol >= 0 else "NEGATIVE"
        if lab == "POSITIVE": pos_t += dur
        tot_t += dur
        if last is not None and lab != last: flips += 1
        last = lab
    pos_pct = (pos_t / tot_t * 100.0) if tot_t > 0 else 0.0
    return pos_pct, flips

def _hotspot_windows(segs_list, duration, bins=24, topk=2):
    if not segs_list or not duration: return []
    mids = []
    for s in segs_list:
        st = float(s.get("start") or 0.0)
        en = float(s.get("end")   or st)
        mids.append(max(0.0, min(duration, 0.5*(st+en))))
    if not mids: return []
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
        idx = np.where(z > z_thresh)[0].tolist()
        return idx, T
    except Exception:
        return [], None

def compute_prosody_change_times(samples: list[float], window: int = 500, dE_thresh: float = 0.25, duration_hint: float | None = None):
    """Energy change points (approx). Returns seconds if duration_hint provided."""
    if not samples or len(samples) < window*2:
        return []
    s = np.asarray(samples, dtype=float)
    energy = np.sqrt(np.convolve(s**2, np.ones(window)/window, mode="valid"))
    energy = (energy - energy.min()) / (energy.ptp() + 1e-9)
    dE = np.abs(np.diff(energy))
    peaks = np.where(dE > dE_thresh)[0]
    if duration_hint and duration_hint > 0:
        times = (peaks / max(1, len(dE))) * duration_hint
        return times.tolist()
    else:
        return (peaks / max(1, len(dE))).tolist()

# ---------------- App ----------------
st.set_page_config(page_title="MOSI Sync Viewer", layout="wide", initial_sidebar_state="collapsed")
st.title("MOSI Sync Viewer")

# ---- Dataset info banner (top of page) ----
st.markdown(
    """
<div style="border:1px solid #333;border-radius:10px;padding:10px 14px;
            background:#111;margin-bottom:10px;">
  <details open>
    <summary style="cursor:pointer;font-weight:600">📘 Dataset info</summary>
    <div style="font-size:14px;color:#cfcfcf;margin-top:8px;line-height:1.5">
      <div><b>Dataset:</b> CMU-MOSI (public multimodal sentiment dataset)</div>
      <div><b>Source:</b> YouTube monologue clips for research and teaching use</div>
      <div><b>Selection logic:</b> Frontal-face visibility, clear smile/frown/gaze, 
      and continuous speech ensuring synchronized audio–video–text alignment</div>
    </div>
  </details>
</div>
""",
    unsafe_allow_html=True
)


# --- Controls (hamburger-style expander instead of sidebar) ---
with st.expander("☰ Controls", expanded=False):
    mode = st.radio(
        "Select video type:",
        ("with", "without"),
        index=0,
        format_func=lambda m: "🎥 With overlays (tracking)" if m == "with"
                             else "📹 Without overlays (original video)"
    )
    show_sent = st.toggle("Show sentiment bar", value=True)
    show_aus  = st.toggle("Show facial AUs (OpenFace)", value=True)

# Build lists for each mode (only clips with transcript+waveform)
trns = {p.stem for p in TRN_DIR.glob("*.csv")}
wvfs = {p.stem for p in WVF_DIR.glob("*.json")}

overlay_ids_all = [p.stem.replace("_overlay", "") for p in VID_DIR.glob("*_overlay.mp4")]
normal_ids_all  = [p.stem for p in VID_DIR.glob("*.mp4") if not p.stem.endswith("_overlay")]

overlay_ids = sorted([i for i in overlay_ids_all if i in trns and i in wvfs])
normal_ids  = sorted([i for i in normal_ids_all  if i in trns and i in wvfs])

ids = overlay_ids if mode == "with" else normal_ids
if not ids:
    st.info("No matching videos found — generate overlays or add normal videos with matching transcript & waveform.")
    st.stop()

# UI selector (main row)
c1, c2 = st.columns([3,1])
with c1:
    vid = st.selectbox("Choose a clip", ids, index=0)
    

# ---- select video path per mode & load ----
video_path = VID_DIR / (f"{vid}_overlay.mp4" if mode == "with" else f"{vid}.mp4")
try:
    video_src = b64_from_path(video_path)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

words     = read_transcript_records(vid)
samples   = read_waveform_samples(vid)
sents     = read_sentences(vid) if show_sent else []
of_bundle = read_openface_bundle(vid) if show_aus else {}
of_raw    = read_openface_raw(vid)
patterns  = read_patterns(vid)
cooc_list, segs_list = read_cooccurrence(vid)

# ---- derive pattern timeline base (seconds array) ----
duration_hint = _estimate_duration(words, sents, of_bundle, of_raw)

# ---- Signal coverage numbers (video/text/face/both) ----
vid_dur = float(duration_hint or 0.0)

text_seconds = 0.0
for s in sents or []:
    stt = float(s.get("start", 0.0)); enn = float(s.get("end", stt))
    text_seconds += max(0.0, min(enn, vid_dur) - max(stt, 0.0))

face_seconds = 0.0
Tcov = (of_bundle or {}).get("time") or []
if len(Tcov) > 1:
    for i in range(len(Tcov)-1):
        t0, t1 = max(0.0, Tcov[i]), max(0.0, Tcov[i+1])
        if t1 <= 0 or t0 >= vid_dur: 
            continue
        face_seconds += max(0.0, min(t1, vid_dur) - max(t0, 0.0))

both_seconds = 0.0
if len(Tcov) > 1 and sents:
    for i in range(len(Tcov)-1):
        t0, t1 = Tcov[i], Tcov[i+1]
        if t1 <= 0 or t0 >= vid_dur:
            continue
        mid = 0.5*(t0+t1)
        seg = None
        for s in sents:
            if mid >= float(s.get("start",0.0)) and mid <= float(s.get("end",0.0)):
                seg = s; break
        if seg:
            both_seconds += max(0.0, min(t1, vid_dur) - max(t0, 0.0))

COVER_DICT = {"video": vid_dur, "text": text_seconds, "face": face_seconds, "both": both_seconds}
COVER_JSON = json.dumps(COVER_DICT)

# ---- pattern timeline base ----
pat_time = None
if of_bundle.get("time"):
    pat_time = np.asarray(of_bundle["time"], dtype=float)
elif of_raw.get("time"):
    pat_time = np.asarray(of_raw["time"], dtype=float)
elif duration_hint:
    pat_time = np.linspace(0.0, duration_hint, num=1001)
else:
    pat_time = np.linspace(0.0, 1.0, num=1001)

if "time" not in patterns or not patterns.get("time"):
    patterns["time"] = pat_time.tolist()

# ---- compute Gesture & Prosody and align ----
gesture_idx_raw, gT = compute_gesture_bursts(of_raw)
gesture_idx = []
if gesture_idx_raw and gT is not None:
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
has_sent = bool(sents)

def _au_band(val: float | None) -> str:
    if val is None: return "—"
    if val < 0.10:  return "Very low"
    if val < 0.30:  return "Low"
    if val < 0.60:  return "Moderate"
    return "High"

if metrics:
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric(
        "Positive time %",
        "—" if not has_sent else f"{metrics['pos_pct']:.1f}%",
        help="Share of the video duration predicted as positive sentiment."
    )
    mcol2.metric(
        "Sentiment flips",
        "—" if not has_sent else f"{metrics['flips']}",
        help="Number of times the sentiment switched polarity."
    )
    smile_mean = metrics.get("mean_au12")
    frown_mean = metrics.get("mean_au04")
    mcol3.metric("Smiling intensity",  _au_band(smile_mean),
                 ("—" if smile_mean is None else f"{smile_mean:.2f}"),
                 help="Average activation of AU12 (smile).")
    mcol4.metric("Frowning intensity", _au_band(frown_mean),
                 ("—" if frown_mean is None else f"{frown_mean:.2f}"),
                 help="Average activation of AU04 (frown).")

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

# --- Optional: segments list (PAGINATED)
show_segs = st.toggle("Show pattern segments (2s windows)", value=False) if segs_list else False
if show_segs and segs_list:
    st.markdown("#### Pattern segments")
    page_size = st.selectbox("Rows per page", [20, 30, 50, 100, 200, 500], index=0, key=f"seg_ps_{vid}")
    total = len(segs_list)
    total_pages = max(1, (total + page_size - 1) // page_size)
    pg_key = f"seg_pg_{vid}"
    if pg_key not in st.session_state: st.session_state[pg_key] = 1

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
        st.markdown(f"Page **{st.session_state[pg_key]}** / **{total_pages}** — showing **{page_size}** per page — **{total}** total segments")

    start = (st.session_state[pg_key] - 1) * page_size
    end = min(start + page_size, total)
    for j in range(start, end):
        seg = segs_list[j]
        labels = ", ".join(seg.get("labels", []))
        st.write(f"**{j+1}.** {seg['start']:.2f}s → {seg['end']:.2f}s — {labels}")

    export_df = pd.DataFrame(
        [{"index": i+1, "start": s.get("start"), "end": s.get("end"), "labels": ", ".join(s.get("labels", []))}
         for i, s in enumerate(segs_list)]
    )
    st.download_button(
        "Download all segments (CSV)",
        export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{vid}_pattern_segments.csv",
        mime="text/csv",
    )

# canvases
CANVAS_W, CANVAS_H = 640, 110
SENT_H  = 12
AGREE_H = 10
EMO_H   = 10
AU_H    = 120
PAT_H   = 84  # 5 lanes

# ----------------- Component (HTML+JS) -----------------
WORDS_JSON    = json.dumps(words)
SAMPLES_JSON  = json.dumps(samples)
SENTS_JSON    = json.dumps(sents)
OFBUNDLE_JSON = json.dumps(of_bundle)
AUPLOT_JSON   = json.dumps(au_to_plot)
PAT_JSON      = json.dumps(patterns)

HTML_TEMPLATE = """
<!doctype html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<style>
  :root { color-scheme: dark; }
  body { font-family: system-ui,-apple-system,Segoe UI,Roboto,sans-serif; background:#0e1117; color:#e7e7e7; margin:0; }
  .wrap { display:flex; gap:24px; padding:10px; }
  video { width:640px; max-width:60vw; border-radius:8px; display:block; }
  .right { flex:1; min-width:320px; }
  .panel { display:flex; gap:16px; align-items:center; margin:8px 0 6px; font-size:14px; flex-wrap:wrap; }
  .lbl { font-size:12px; color:#bfbfbf; margin:6px 0 2px 2px; }
  .transcript { line-height:1.9; max-height:420px; overflow:auto; padding-right:8px; border-left:1px solid #222; padding-left:16px; }
  .word { padding:2px 4px; margin-right:2px; cursor:pointer; border-radius:4px; }
  .word.active { background:#ffeb3b; color:#000; }
  canvas { display:block; margin-top:6px; border:1px solid #444; background:#111; border-radius:4px; }
  #wave { cursor: pointer; }
  #au   { cursor: pointer; }

  #sentLabel, #emoLabel {
    margin-top:10px; font-size:16px; font-weight:700; display:inline-block;
    padding:4px 10px; border-radius:999px; line-height:1.25; letter-spacing:.2px;
    background:#1d1f26; color:#e7e7e7;
  }
  #emoLabel{ margin-left:10px; }
  #sentLabel.pos { background:#15803d; color:#fff; }
  #sentLabel.neg { background:#b91c1c; color:#fff; }
  #emoLabel.joy    { background:#d4f542; color:#1a1a1a; }
  #emoLabel.anger  { background:#ef4444; color:#fff; }
  #emoLabel.sad    { background:#60a5fa; color:#0a1a2b; }
  #emoLabel.neutral{ background:#6b7280; color:#fff; }

  .patrow { display:flex; align-items:center; gap:10px; }
  .patlabels {
    width:80px; display:flex; flex-direction:column; justify-content:space-between;
    height:__PAT_H__px; font-size:12px; color:#bfbfbf; line-height:1; user-select:none; text-align:left;
  }
  .patlabels div { display:flex; align-items:center; height:__PAT_H_DIV__px; }

  /* Right column mini-graphs */
  .mini { margin-top:14px; }
  .mini h4 { margin:8px 0 4px; font-size:14px; color:#cfcfcf; font-weight:600; }
  .row2 { display:flex; gap:10px; align-items:center; }
  .cellnote { font-size:12px; color:#bfbfbf; margin-top:4px; }

  .kpi.small { font-size:14px; font-weight:700; padding:6px 10px; border-radius:8px; display:inline-block; background:#1d1f26; }

  .helpicon { font-weight:800; margin-left:6px; color:#9aa0a6; cursor:help; }
</style></head>
<body>
<div class="wrap">
  <div>
    <video id="v" controls src="__VIDEO_SRC__"></video>

    <div class="panel">
      <label><input type="checkbox" id="autos" checked> Auto-scroll transcript</label>
    </div>

    <div class="lbl">Waveform</div>
    <canvas id="wave" width="__CANVAS_W__" height="__CANVAS_H__"></canvas>

    __SENT_BLOCK__
    __AGREE_BLOCK__
    __EMO_BLOCK__
    __PAT_BLOCK__

    <div id="sentLabel"></div>
    <div id="emoLabel"></div>

    __AU_BLOCK__
  </div>

  <div class="right">
    <div class="transcript" id="tx"></div>

    <!-- RIGHT: Mini-graphs -->
    <div class="mini">
      <h4>Sentiment time %</h4>
      <div class="row2">
        <canvas id="donut" width="140" height="140" style="border:none;background:transparent;"></canvas>
        <div class="cellnote" id="donutTxt"></div>
      </div>
    </div>

    <div class="mini">
      <h4>Signal coverage</h4>
      <div class="kpi small" id="covKPI"></div>
    </div>

    <div class="mini">
      <h4>Text ↔ Face agreement (time-weighted)</h4>
      <canvas id="mat" width="220" height="160" style="border:1px solid #444;"></canvas>
      <div class="cellnote">Rows = Text (Pos/Neg), Columns = Face (Pos/Neg)</div>
    </div>

  </div>
</div>

<script>
const words    = __WORDS_JSON__;
const samples  = __SAMPLES_JSON__;
const sents    = __SENTS_JSON__;
const ofBundle = __OFBUNDLE_JSON__;
const auToPlot = __AUPLOT_JSON__;
const patterns = __PAT_JSON__;

// coverage numbers passed from Python (video/text/face/both, in seconds)
const COVER    = __COVER_JSON__;

// DOM
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

// Right-column canvases
const donutCanvas = document.getElementById("donut");
const donutCtx    = donutCanvas ? donutCanvas.getContext("2d") : null;
const donutTxt    = document.getElementById("donutTxt");
const matCanvas   = document.getElementById("mat");
const matCtx      = matCanvas ? matCanvas.getContext("2d") : null;

// ---------- Transcript ----------
function renderTranscript() {
  const frag = document.createDocumentFragment();
  for (let i=0;i<words.length;i++) {
    const w = words[i];
    const span = document.createElement("span");
    span.className = "word";
    span.dataset.i = i;
    span.dataset.s = w.start;
    span.dataset.e = w.end;
    span.textContent = w.word + " ";
    span.onclick = () => v.currentTime = w.start;

    if (sents && sents.length) {
      const s = sents.find(ss => w.start >= ss.start && w.end <= ss.end);
      if (s) {
        const sc = (typeof s.score === "number" ? s.score.toFixed(2) : s.score);
        const po = (typeof s.polarity === "number" ? s.polarity.toFixed(2) : s.polarity);
        span.title = `Sentiment: ${s.label} | Score: ${sc} | Polarity: ${po}`;
      }
    }
    frag.appendChild(span);
  }
  box.innerHTML = "";
  box.appendChild(frag);
}
renderTranscript();
function spans() { return box.children; }

// ---------- Waveform (click-to-seek) ----------
function drawWaveform(time) {
  ctx.clearRect(0,0,cvs.width,cvs.height);
  const mid = cvs.height * 0.5;
  const L = samples.length || 1;
  ctx.beginPath(); ctx.moveTo(0, mid);
  for (let i=0;i<L;i++) {
    const x = i * (cvs.width / L);
    const y = mid - samples[i] * (cvs.height * 0.45);
    ctx.lineTo(x, y);
  }
  ctx.strokeStyle = "#0f0"; ctx.lineWidth = 1; ctx.stroke();
  if (v.duration) {
    const x = (time / v.duration) * cvs.width;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, cvs.height);
    ctx.strokeStyle = "red"; ctx.lineWidth = 1; ctx.stroke();
  }
}
if (cvs) {
  cvs.addEventListener("click", (e) => {
    if (!v.duration) return;
    const rect = cvs.getBoundingClientRect();
    const x = e.clientX - rect.left;
    v.currentTime = Math.min(Math.max(x / cvs.width, 0), 1) * v.duration;
  });
}

// ---------- Sentiment bar ----------
function drawSentiment() {
  if (!sentCtx || !v.duration) return;
  sentCtx.clearRect(0,0,sentCanvas.width,sentCanvas.height);
  for (const s of sents) {
    const x0 = (s.start / v.duration) * sentCanvas.width;
    const x1 = (s.end   / v.duration) * sentCanvas.width;
    const pol = Math.max(-1, Math.min(1, Number(s.polarity||0)));
    const g = Math.round(200 * Math.max(0,  pol));
    const r = Math.round(200 * Math.max(0, -pol));
    sentCtx.fillStyle = `rgb(${r},${g},0)`;
    sentCtx.fillRect(x0, 0, Math.max(1, x1-x0), sentCanvas.height);
  }
}

// ---------- Emotion mapping (rule-based from AUs) ----------
function computeEmotionAt(i) {
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
}
function drawEmotionBand() {
  if (!emoCtx || !ofBundle.time || !v.duration) return;
  const W = emoCanvas.width, H = emoCanvas.height, T = ofBundle.time;
  emoCtx.clearRect(0,0,W,H);
  for (let i=0;i<T.length-1;i++) {
    const x0 = (T[i]   / v.duration) * W;
    const x1 = (T[i+1] / v.duration) * W;
    const emo = computeEmotionAt(i);
    let color = "#666";
    if (emo === "JOY") color = "#d4f542";
    else if (emo === "ANGER") color = "#f44";
    else if (emo === "SAD") color = "#6aa0ff";
    emoCtx.fillStyle = color;
    emoCtx.fillRect(x0, 0, Math.max(1, x1-x0), H);
  }
}

// ---------- Agreement bar ----------
function facialValenceAt(i) {
  const aus = ofBundle.aus || {};
  const get = (k) => (aus[k] && typeof aus[k][i] === "number") ? aus[k][i] : 0;
  return get("AU12_r") - 0.5*(get("AU04_r") + get("AU15_r")); // >0 => positive face
}
function sentimentAtTime(t) {
  if (!sents || !sents.length) return null;
  for (const s of sents) if (t >= s.start && t <= s.end) return s;
  return null;
}
function drawAgreement() {
  if (!agreeCtx || !ofBundle.time || !v.duration) return;
  const W = agreeCanvas.width, H = agreeCanvas.height, T = ofBundle.time;
  agreeCtx.clearRect(0,0,W,H);
  for (let i=0;i<T.length-1;i++) {
    const mid = (T[i] + T[i+1]) * 0.5;
    const seg = sentimentAtTime(mid);
    const x0 = (T[i]   / v.duration) * W;
    const x1 = (T[i+1] / v.duration) * W;
    let color = "#444";
    if (seg) {
      const pol = Number(seg.polarity || 0);
      const fval = facialValenceAt(i);
      const agree = (pol >= 0 && fval >= 0) || (pol < 0 && fval < 0);
      color = agree ? "#19d17c" : "#ff8a65";
    }
    agreeCtx.fillStyle = color;
    agreeCtx.fillRect(x0, 0, Math.max(1, x1-x0), H);
  }
}

// ---------- Pattern ticks ----------
function drawPatterns() {
  if (!patCtx || !patterns || !patterns.time || !v.duration) return;
  const W = patCanvas.width, H = patCanvas.height;
  patCtx.clearRect(0,0,W,H);

  const ROWS = 5;
  const laneY = (rowIdx) => Math.round(((rowIdx + 0.5) / ROWS) * H);

  const lanes = [
    {key:"AU12_r_peaks",   color:"#7dd3fc", y: laneY(0)},
    {key:"AU04_r_peaks",   color:"#f472b6", y: laneY(1)},
    {key:"gaze_shifts",    color:"#facc15", y: laneY(2)},
    {key:"gesture_bursts", color:"#34d399", y: laneY(3)},
    {key:"prosody_changes",color:"#a78bfa", y: laneY(4)},
  ];

  function xOfIdx(i){ return (patterns.time[Math.min(i, patterns.time.length-1)] / v.duration) * W; }

  patCanvas._tickTimes = [];

  lanes.forEach(l => {
    const arr = patterns[l.key];
    if (!arr) return;
    patCtx.strokeStyle = l.color;
    patCtx.lineWidth = 2;
    arr.forEach(i => {
      const x = xOfIdx(i);
      patCtx.beginPath(); patCtx.moveTo(x, l.y-6); patCtx.lineTo(x, l.y+6); patCtx.stroke();
      patCanvas._tickTimes.push(x / W * v.duration);
    });
  });

  const x = (v.currentTime / v.duration) * W;
  patCtx.beginPath(); patCtx.moveTo(x, 0); patCtx.lineTo(x, H);
  patCtx.strokeStyle = "red"; patCtx.lineWidth = 1; patCtx.stroke();
}

if (patCanvas) {
  patCanvas.addEventListener("click", (e) => {
    if (!v.duration || !patCanvas._tickTimes || !patCanvas._tickTimes.length) return;
    const rect = patCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const t = Math.min(Math.max(x / patCanvas.width, 0), 1) * v.duration;
    // snap to nearest tick (≤1s)
    let best = t, bestd = 1e9;
    for (const tt of patCanvas._tickTimes) {
      const d = Math.abs(t - tt);
      if (d < bestd) { best = tt; bestd = d; }
    }
    v.currentTime = (bestd < 1.0) ? best : t;
  });
}

// ---------- AUs (click-to-seek) ----------
const AU_COLORS = ["#4da6ff", "#ffd24d", "#ff66cc", "#66ff99", "#ff8c66", "#b366ff"];
function drawAUs(time) {
  if (!auctx || !ofBundle.aus || !ofBundle.time || !v.duration) return;
  const W = aucvs.width, H = aucvs.height;
  auctx.clearRect(0,0,W,H);
  const xFromSec = s => (s / v.duration) * W;

  const T = ofBundle.time;
  auToPlot.forEach((auName, k) => {
    const arr = (ofBundle.aus || {})[auName];
    if (!arr) return;
    const maxV = 5.0;
    auctx.beginPath();
    for (let i=0;i<arr.length;i++) {
      const x = xFromSec(T[i]);
      const y = H - (Math.max(0, Math.min(arr[i], maxV)) / maxV) * (H - 14) - 7;
      if (i===0) auctx.moveTo(x,y); else auctx.lineTo(x,y);
    }
    auctx.strokeStyle = AU_COLORS[k % AU_COLORS.length];
    auctx.lineWidth = 1.2; auctx.stroke();

    auctx.fillStyle = AU_COLORS[k % AU_COLORS.length];
    auctx.fillRect(6 + k*90, 6, 10, 10);
    auctx.fillStyle = "#ddd";
    auctx.font = "12px system-ui, sans-serif";
    auctx.fillText(auName, 20 + k*90, 15);
  });

  const x = xFromSec(time);
  auctx.beginPath(); auctx.moveTo(x, 0); auctx.lineTo(x, H);
  auctx.strokeStyle = "red"; auctx.lineWidth = 1; auctx.stroke();
}
if (aucvs) {
  aucvs.addEventListener("click", (e) => {
    if (!v.duration) return;
    const rect = aucvs.getBoundingClientRect();
    const x = e.clientX - rect.left;
    v.currentTime = Math.min(Math.max(x / aucvs.width, 0), 1) * v.duration;
  });
}

// ---------- Right-column: mini-graphs data ----------
function totalSentimentTime() {
  if (!sents || !sents.length) return {pos:0, neg:0, tot:0};
  let pos=0, neg=0, tot=0;
  for (const s of sents) {
    const st = +s.start || 0, en = Math.max(st, +s.end || st);
    const d = Math.max(0, en - st);
    tot += d;
    const lab = (s.label && s.label.toUpperCase()) || ((+s.polarity||0)>=0 ? "POSITIVE":"NEGATIVE");
    if (lab === "POSITIVE") pos += d; else neg += d;
  }
  return {pos, neg, tot};
}
function faceValenceAtTime(t) {
  if (!ofBundle || !ofBundle.time || !ofBundle.aus) return 0;
  const T = ofBundle.time;
  // nearest index
  let i=0;
  while (i+1<T.length && Math.abs(T[i+1]-t) < Math.abs(T[i]-t)) i++;
  const aus = ofBundle.aus;
  const get=(k)=> (aus[k]&&typeof aus[k][i]==="number")? aus[k][i]:0;
  const val = get("AU12_r") - 0.5*(get("AU04_r")+get("AU15_r"));
  return val; // >0 positive, <0 negative
}
function agreementMatrix() {
  // time-weighted confusion between text sign vs face sign
  const bins = [[0,0],[0,0]];
  if (!sents || !sents.length || !ofBundle || !ofBundle.time) return bins;
  const T = ofBundle.time;
  for (let i=0;i<T.length-1;i++){
    const mid = (T[i]+T[i+1])*0.5;
    const seg = sentimentAtTime(mid);
    if (!seg) continue;
    const textPos = ((seg.label && seg.label.toUpperCase()==="POSITIVE") || (+seg.polarity||0)>=0);
    const facePos = faceValenceAtTime(mid) >= 0;
    const dur = Math.max(0, T[i+1]-T[i]);
    if (textPos && facePos) bins[0][0]+=dur;
    else if (textPos && !facePos) bins[0][1]+=dur;
    else if (!textPos && facePos) bins[1][0]+=dur;
    else bins[1][1]+=dur;
  }
  return bins;
}

// ---------- Right-column: rendering ----------
function drawDonut() {
  if (!donutCtx) return;
  donutCtx.clearRect(0,0,donutCanvas.width,donutCanvas.height);
  const {pos,neg,tot} = totalSentimentTime();
  const pct = tot>0 ? (pos/tot) : 0;
  const cx = donutCanvas.width/2, cy = donutCanvas.height/2, r=56, t=18;
  const start = -Math.PI/2;
  const endPos = start + 2*Math.PI*pct;

  // ring background
  donutCtx.beginPath();
  donutCtx.arc(cx,cy,r,0,2*Math.PI);
  donutCtx.strokeStyle = "#333"; donutCtx.lineWidth = t; donutCtx.stroke();

  // positive arc
  donutCtx.beginPath();
  donutCtx.arc(cx,cy,r,start,endPos);
  donutCtx.strokeStyle = "#19d17c"; donutCtx.lineWidth = t; donutCtx.stroke();

  // negative arc
  donutCtx.beginPath();
  donutCtx.arc(cx,cy,r,endPos,start+2*Math.PI);
  donutCtx.strokeStyle = "#ff6b6b"; donutCtx.lineWidth = t; donutCtx.stroke();

  donutCtx.fillStyle="#e7e7e7"; donutCtx.font="bold 16px system-ui, sans-serif";
  donutCtx.textAlign="center"; donutCtx.textBaseline="middle";
  donutCtx.fillText(`${(pct*100).toFixed(0)}%`, cx, cy);

  if (donutTxt) {
    donutTxt.innerHTML = `
      <div><b>Positive time:</b> ${(pct*100).toFixed(1)}%</div>
      <div><b>Negative time:</b> ${((1-pct)*100).toFixed(1)}%</div>
    `;
  }
}

// Coverage KPI only
function setCoverageKPI() {
  const el = document.getElementById("covKPI");
  if (!el || !COVER) return;
  const v = Number(COVER.video||0), tx = Number(COVER.text||0), fc = Number(COVER.face||0), bo = Number(COVER.both||0);
  const pct = (x,den) => den>0 ? (100*x/den).toFixed(1) : "0.0";
  el.textContent = `Text: ${tx.toFixed(1)}s (${pct(tx,v)}%) | Face: ${fc.toFixed(1)}s (${pct(fc,v)}%) | Both: ${bo.toFixed(1)}s (${pct(bo,v)}%) of ${v.toFixed(1)}s`;
}

function drawMatrix() {
  if (!matCtx) return;
  const bins = agreementMatrix();
  matCtx.clearRect(0,0,matCanvas.width,matCanvas.height);
  const W = matCanvas.width, H = matCanvas.height;
  const pad = 30, cellW = (W - pad*1.5)/2, cellH = (H - pad*1.6)/2;
  const baseX = pad*0.7, baseY = pad*0.6;

  const max = Math.max(1, bins[0][0], bins[0][1], bins[1][0], bins[1][1]);

  function cell(x,y,val,goodColor){
    const ratio = val/max;
    const col = goodColor ? `rgba(25,209,124,${0.25+0.65*ratio})`
                          : `rgba(255,138,101,${0.25+0.65*ratio})`;
    matCtx.fillStyle = col;
    matCtx.fillRect(x,y,cellW,cellH);
    matCtx.strokeStyle="#666"; matCtx.strokeRect(x,y,cellW,cellH);
    matCtx.fillStyle="#e7e7e7"; matCtx.font="12px system-ui, sans-serif";
    matCtx.textAlign="center"; matCtx.textBaseline="middle";
    matCtx.fillText(val.toFixed(1)+"s", x+cellW/2, y+cellH/2);
  }

  // labels
  matCtx.fillStyle="#bfbfbf"; matCtx.font="12px system-ui, sans-serif";
  matCtx.fillText("Face +", baseX + cellW*0.5, baseY-10);
  matCtx.fillText("Face −", baseX + cellW*1.5 + pad*0.3, baseY-10);
  matCtx.save();
  matCtx.translate(10, baseY + cellH*0.5);
  matCtx.rotate(-Math.PI/2);
  matCtx.fillText("Text +", 0, 0);
  matCtx.restore();
  matCtx.save();
  matCtx.translate(10, baseY + cellH*1.5 + pad*0.3);
  matCtx.rotate(-Math.PI/2);
  matCtx.fillText("Text −", 0, 0);
  matCtx.restore();

  // cells (TP/TN diagonals are "good")
  cell(baseX,                    baseY,                    bins[0][0], true ); // text+, face+
  cell(baseX+cellW+pad*0.3,     baseY,                    bins[0][1], false); // text+, face-
  cell(baseX,                    baseY+cellH+pad*0.3,     bins[1][0], false); // text-, face+
  cell(baseX+cellW+pad*0.3,     baseY+cellH+pad*0.3,     bins[1][1], true ); // text-, face-
}

// ---------- Emotion label helper ----------
function emotionAtTime(tSec) {
  if (!ofBundle || !ofBundle.time || !ofBundle.time.length) return "NEUTRAL";
  const T = ofBundle.time;
  let i = 0;
  while (i+1 < T.length && Math.abs(T[i+1] - tSec) < Math.abs(T[i] - tSec)) i++;
  return computeEmotionAt(i);
}

// ---------- Scrub-safe highlighting ----------
let active = -1;
function clearActive() {
  if (!box) return;
  const cs = spans();
  for (let k = 0; k < cs.length; k++) cs[k].classList.remove("active");
  active = -1;
}

// ---------- Sync ----------
function syncLoop() {
  const t = Math.max(0, v.currentTime);

  // transcript pointer
  let i = active;
  if (i<0 || i>=words.length || t<words[i].start || t>=words[i].end) {
    while (i+1<words.length && t>=words[i+1].start) i++;
    while (i>0 && t<words[i].start) i--;
  }
  const ok = i>=0 && i<words.length && t>=words[i].start && t<words[i].end;
  if (ok && i!==active) {
    if (active>=0) spans()[active].classList.remove("active");
    spans()[i].classList.add("active");
    active = i;
    if (autos && autos.checked) {
      const el = spans()[i], pr = box.getBoundingClientRect(), er = el.getBoundingClientRect();
      if (er.top < pr.top || er.bottom > pr.bottom) el.scrollIntoView({block:"center", behavior:"smooth"});
    }
  }

  drawWaveform(t);
  if (sentCtx)  drawSentiment();
  if (agreeCtx) drawAgreement();
  if (emoCtx)   drawEmotionBand();
  if (patCtx)   drawPatterns();
  if (auctx)    drawAUs(t);

  // dynamic badges
  if (sents && sents.length && v.duration) {
    const seg = sents.find(ss => t >= ss.start && t <= ss.end);
    if (seg) {
      const label = seg.label ? seg.label.toUpperCase() : (seg.polarity >= 0 ? "POSITIVE" : "NEGATIVE");
      const nice  = label.charAt(0) + label.slice(1).toLowerCase();
      sentLabel.textContent = "Sentiment: " + nice;
      sentLabel.className = (label === "POSITIVE") ? "pos" : "neg";
    } else {
      sentLabel.textContent = ""; sentLabel.className = "";
    }
  }

  if (ofBundle && ofBundle.time && ofBundle.time.length) {
    const emo = emotionAtTime(t);
    let cls = "neutral", pretty = "Neutral";
    if (emo === "JOY")    { cls = "joy";   pretty = "Joy"; }
    if (emo === "ANGER")  { cls = "anger"; pretty = "Anger"; }
    if (emo === "SAD")    { cls = "sad";   pretty = "Sad"; }
    emoLabel.textContent = "Emotion: " + pretty;
    emoLabel.className = cls;
  } else {
    emoLabel.textContent = "";
    emoLabel.className = "";
  }

  // Right mini-graphs refresh (cheap)
  drawDonut();
  setCoverageKPI();
  drawMatrix();

  requestAnimationFrame(syncLoop);
}
v.addEventListener("play", () => requestAnimationFrame(syncLoop));
v.addEventListener("pause", clearActive);
v.addEventListener("seeking", clearActive);
v.addEventListener("seeked", () => {
  const t = Math.max(0, v.currentTime);
  drawWaveform(t);
  if (sentCtx)  drawSentiment();
  if (agreeCtx) drawAgreement();
  if (emoCtx)   drawEmotionBand();
  if (patCtx)   drawPatterns();
  if (auctx)    drawAUs(t);
  drawDonut();
  setCoverageKPI();
  drawMatrix();
});

// init
drawWaveform(0);
if (sentCtx)  drawSentiment();
if (agreeCtx) drawAgreement();
if (emoCtx)   drawEmotionBand();
if (patCtx)   drawPatterns();
if (auctx)    drawAUs(0);
drawDonut();
setCoverageKPI();
drawMatrix();
</script>
</body></html>
"""

SENT_BLOCK = (
    f"<div class='lbl'>Sentiment (text)</div>"
    f"<canvas id='sentbar' width='{CANVAS_W}' height='{SENT_H}'></canvas>"
    if len(sents) > 0 else ""
)
AGREE_BLOCK = (
    f"<div class='lbl'>Text ↔ Face Agreement</div>"
    f"<canvas id='agreebar' width='{CANVAS_W}' height='{AGREE_H}'></canvas>"
    if (len(sents) > 0 and len(of_bundle) > 0) else ""
)
EMO_BLOCK = (
    f"<div class='lbl'>Emotion (AUs)</div>"
    f"<canvas id='emoband' width='{CANVAS_W}' height='{EMO_H}'></canvas>"
    if len(of_bundle) > 0 else ""
)
PAT_BLOCK = (
    "<div class='lbl'>Patterns (click ticks to seek)</div>"
    "<div class='patrow'>"
    f"<canvas id='pat' width='{CANVAS_W}' height='{PAT_H}' style='margin-top:2px;'></canvas>"
    "<div class='patlabels'><div>Smile</div><div>Frown</div><div>Gaze</div><div>Gesture</div><div>Prosody</div></div>"
    "</div>"
    if len(patterns) > 0 else ""
)
# Action Units label + tooltip
AU_BLOCK = (
    "<div class='lbl'>Action Units "
    "<span class='helpicon' title='AU06_r: Cheek Raiser (Duchenne smile)\nAU12_r: Lip Corner Puller (smile)\nAU04_r: Brow Lowerer (frown)'> ( ? )</span></div>"
    f"<canvas id='au' width='{CANVAS_W}' height='{AU_H}'></canvas>"
    if len(of_bundle) > 0 else ""
)

component_html = (
    HTML_TEMPLATE
    .replace("__VIDEO_SRC__", video_src)
    .replace("__CANVAS_W__", str(CANVAS_W))
    .replace("__CANVAS_H__", str(CANVAS_H))
    .replace("__SENT_BLOCK__", SENT_BLOCK)
    .replace("__AGREE_BLOCK__", AGREE_BLOCK)
    .replace("__EMO_BLOCK__", EMO_BLOCK)
    .replace("__PAT_BLOCK__", PAT_BLOCK)
    .replace("__AU_BLOCK__", AU_BLOCK)
    .replace("__PAT_H__", str(PAT_H))
    .replace("__PAT_H_DIV__", str(PAT_H//5))
    .replace("__WORDS_JSON__", WORDS_JSON)
    .replace("__SAMPLES_JSON__", SAMPLES_JSON)
    .replace("__SENTS_JSON__", SENTS_JSON)
    .replace("__OFBUNDLE_JSON__", OFBUNDLE_JSON)
    .replace("__AUPLOT_JSON__", AUPLOT_JSON)
    .replace("__PAT_JSON__", PAT_JSON)
    .replace("__COVER_JSON__", json.dumps(COVER_DICT))
)

st_html(
    component_html,
    height= 980 if (len(of_bundle)>0 or len(patterns)>0) else (860 if len(sents)>0 else 780),
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

            # Local CSV fallback
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
