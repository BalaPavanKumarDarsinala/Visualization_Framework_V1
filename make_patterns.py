# make_patterns.py
from pathlib import Path
import json, math
import numpy as np
import pandas as pd

DATA = Path("data")
OF_DIR = DATA / "openface"
WVF_DIR = DATA / "waveforms"
TRN_DIR = DATA / "transcripts"
PAT_DIR = DATA / "patterns"
COC_DIR = DATA / "patterns_cooc"
for p in (PAT_DIR, COC_DIR): p.mkdir(parents=True, exist_ok=True)

def load_openface(vid):
    p = OF_DIR / f"{vid}.json"
    if not p.exists(): return {}
    return json.loads(p.read_text(encoding="utf-8"))

def load_waveform(vid):
    p = WVF_DIR / f"{vid}.json"
    if not p.exists(): return []
    return json.loads(p.read_text(encoding="utf-8"))

def load_transcript(vid):
    p = TRN_DIR / f"{vid}.csv"
    if not p.exists(): return pd.DataFrame(columns=["start","end","word"])
    df = pd.read_csv(p)
    for c in ["start","end"]: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df.sort_values("start")

def moving_diff(x, w=3):
    x = np.asarray(x, float)
    if len(x) < 2: return np.zeros_like(x)
    dx = np.diff(x, prepend=x[:1])
    if w>1:
        k = np.ones(w)/w
        dx = np.convolve(dx, k, mode="same")
    return dx

def peak_indices(x, thresh, min_gap=5):
    """Simple local peaks above thresh with a minimum index gap."""
    x = np.asarray(x, float)
    peaks = []
    last = -999999
    for i in range(1, len(x)-1):
        if x[i] > thresh and x[i] >= x[i-1] and x[i] >= x[i+1]:
            if i - last >= min_gap:
                peaks.append(i)
                last = i
    return peaks

def contiguous_regions(mask):
    """Returns [(start_idx, end_idx), ...] for True runs in a boolean mask."""
    spans = []
    start = None
    for i, m in enumerate(mask):
        if m and start is None: start = i
        elif not m and start is not None:
            spans.append((start, i-1)); start=None
    if start is not None: spans.append((start, len(mask)-1))
    return spans

def detect_pauses(ts, words, min_gap=0.6):
    # Build a boolean “speech” mask using transcript words
    # For each timestamp, are we inside any word interval?
    t = np.asarray(ts, float)
    speech = np.zeros_like(t, dtype=bool)
    for _,row in words.iterrows():
        speech |= (t >= row["start"]) & (t <= row["end"])
    silent = ~speech
    spans = contiguous_regions(silent)
    # Keep only spans with duration >= min_gap
    out = []
    for s,e in spans:
        dt = t[e]-t[s] if e> s else 0.0
        if dt >= min_gap:
            out.append((s,e))
    return out

def norm01(x):
    x = np.asarray(x, float)
    if len(x)==0: return x
    lo, hi = np.nanpercentile(x, 1), np.nanpercentile(x, 99)
    if hi<=lo: return np.zeros_like(x)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0, 1)

def detect_patterns(vid):
    of = load_openface(vid)
    if not of or "time" not in of: return None
    T = np.asarray(of["time"], float)
    n = len(T)
    aus  = of.get("aus", {})
    gaze = of.get("gaze", {})
    pose = of.get("pose", {})

    AU12 = np.asarray(aus.get("AU12_r", [0]*n), float)
    AU04 = np.asarray(aus.get("AU04_r", [0]*n), float)

    # Smile / Frown peaks
    smile_peaks = peak_indices(AU12, thresh=np.nanmean(AU12)+np.nanstd(AU12)*0.8, min_gap=5)
    frown_peaks = peak_indices(AU04, thresh=np.nanmean(AU04)+np.nanstd(AU04)*0.8, min_gap=5)

    # Gaze shifts (speed of gaze angles)
    gx = np.asarray(gaze.get("gaze_angle_x", [0]*n), float)
    gy = np.asarray(gaze.get("gaze_angle_y", [0]*n), float)
    gspd = np.hypot(moving_diff(gx, w=3), moving_diff(gy, w=3))
    gz_thresh = np.nanmean(gspd) + np.nanstd(gspd)*1.0
    gaze_shifts = peak_indices(gspd, thresh=gz_thresh, min_gap=5)

    # Gesture bursts from head pose velocity
    rx = np.asarray(pose.get("pose_Rx", [0]*n), float)
    ry = np.asarray(pose.get("pose_Ry", [0]*n), float)
    rz = np.asarray(pose.get("pose_Rz", [0]*n), float)
    hspd = np.hypot(moving_diff(rx,3), moving_diff(ry,3))
    hspd = np.hypot(hspd, moving_diff(rz,3))
    hb_thresh = np.nanmean(hspd) + np.nanstd(hspd)*1.2
    gesture_bursts = peak_indices(hspd, thresh=hb_thresh, min_gap=5)

    # Prosody changes from waveform loudness (RMS), aligned by index
    wav = load_waveform(vid)
    if len(wav)>0:
        w = np.asarray(wav, float)
        # downsample/upsample RMS to length n using simple indexing
        # assume waveform len >> n; map frames
        idx = np.linspace(0, len(w)-1, num=n).astype(int)
        rms = np.sqrt(w[idx]**2)
        rms_s = pd.Series(rms).rolling(5, min_periods=1, center=True).mean().values
        rmsd = np.abs(moving_diff(rms_s, w=3))
        pr_thresh = np.nanmean(rmsd) + np.nanstd(rmsd)*1.2
        prosody_changes = peak_indices(rmsd, thresh=pr_thresh, min_gap=5)
    else:
        prosody_changes = []

    # Pauses from transcript
    words = load_transcript(vid)
    pauses = detect_pauses(T, words, min_gap=0.6)

    # Save lanes for the UI
    pat = {
        "time": T.tolist(),
        "AU12_r_peaks": smile_peaks,
        "AU04_r_peaks": frown_peaks,
        "gaze_shifts": gaze_shifts,
        "gesture_bursts": gesture_bursts,
        "prosody_changes": prosody_changes,
        "pauses": pauses
    }
    (PAT_DIR / f"{vid}.json").write_text(json.dumps(pat), encoding="utf-8")

    # Co-occurrence mining (2.0s window): count combos and extract segments
    window = 2.0
    lanes = {
        "smile": set(smile_peaks),
        "frown": set(frown_peaks),
        "gaze":  set(gaze_shifts),
        "gesture": set(gesture_bursts),
        "prosody": set(prosody_changes)
    }

    # build time lookup per index
    combos = {}
    segments = []  # list of {start,end,labels}
    i = 0
    while i < n:
        t0 = T[i]; t1 = t0 + window
        # all ticks whose T is in [t0,t1]
        inwin = {name: [k for k in ks if t0 <= T[k] <= t1] for name,ks in lanes.items()}
        active = sorted([name for name, arr in inwin.items() if len(arr)>0])
        if len(active) >= 2:
            key = "+".join(active)
            combos[key] = combos.get(key, 0) + 1
            segments.append({"start": float(t0), "end": float(min(t1, T[-1])), "labels": active})
            # jump ahead ~window to avoid near-duplicates
            # find first index with time >= t1
            while i < n and T[i] < t1: i += 1
        else:
            i += 1

    cooc = {"combos": combos, "segments": segments}
    (COC_DIR / f"{vid}.json").write_text(json.dumps(cooc), encoding="utf-8")
    print("patterns/cooc done:", vid)

def ids():
    vids = {p.stem for p in OF_DIR.glob("*.json")}
    return sorted(list(vids))

if __name__ == "__main__":
    for vid in ids():
        detect_patterns(vid)
