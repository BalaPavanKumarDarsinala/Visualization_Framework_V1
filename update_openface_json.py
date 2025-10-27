# update_openface_json.py
from pathlib import Path
import pandas as pd, json

RAW = Path("data/openface_raw")
OUT = Path("data/openface")
OUT.mkdir(parents=True, exist_ok=True)

KEEP_AU_R = [f"AU{n:02d}_r" for n in [1,2,4,5,6,7,9,10,12,14,15,17,20,23,25,26,45]]
KEEP = ["timestamp","gaze_angle_x","gaze_angle_y","pose_Rx","pose_Ry","pose_Rz"] + KEEP_AU_R

for csv_path in RAW.glob("*.csv"):
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df.columns = df.columns.str.strip()
    have = [c for c in KEEP if c in df.columns]

    # keep only confident rows if available
    if " success" in df.columns: df = df[df[" success"].astype(str).str.strip()=="1"]
    if "success" in df.columns:  df = df[df["success"]==1]

    # timestamps
    tcol = "timestamp" if "timestamp" in df.columns else " timestamp"
    if tcol not in df.columns:
        # fallback to frame index at 30 fps
        df["timestamp"] = df.index / 30.0
        tcol = "timestamp"

    out = {"time": df[tcol].astype(float).tolist(), "aus": {}, "gaze": {}, "pose": {}}

    # AUs
    for au in KEEP_AU_R:
        if au in df.columns:
            out["aus"][au] = df[au].astype(float).tolist()

    # gaze
    for g in ["gaze_angle_x","gaze_angle_y"]:
        if g in df.columns:
            out["gaze"][g] = df[g].astype(float).tolist()

    # head pose (in radians)
    for p in ["pose_Rx","pose_Ry","pose_Rz"]:
        if p in df.columns:
            out["pose"][p] = df[p].astype(float).tolist()

    out["duration"] = float(df[tcol].max()) if len(df) else 0.0
    vid = csv_path.stem
    (OUT / f"{vid}.json").write_text(json.dumps(out), encoding="utf-8")
    print("wrote", OUT / f"{vid}.json")
