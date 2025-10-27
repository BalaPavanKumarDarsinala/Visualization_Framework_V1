# make_openface_json.py
from pathlib import Path
import pandas as pd
import numpy as np
import json
import re

ROOT = Path(__file__).resolve().parent
IN   = ROOT / "data" / "openface_raw"
OUT  = ROOT / "data" / "openface"
OUT.mkdir(parents=True, exist_ok=True)

AU_PAT = re.compile(r"^AU\d{2}_(r|c)$", re.IGNORECASE)

def read_csv_smart(path: Path) -> pd.DataFrame:
    # Let pandas infer delimiter (sep=None) and strip weird spaces from headers.
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = df.columns.str.strip()           # <<< important
    return df

def time_vector(df: pd.DataFrame) -> np.ndarray:
    # Prefer timestamp (strip handled already)
    if "timestamp" in df.columns:
        t = pd.to_numeric(df["timestamp"], errors="coerce")
        t = t.fillna(method="ffill").fillna(0.0).to_numpy()
        if np.isfinite(t).sum() >= 2:
            return t
    # Fallback: frame/fps
    if "frame" in df.columns:
        fps = 30.0
        if "fps" in df.columns:
            try: fps = float(df["fps"].iloc[0])
            except Exception: pass
        fr = pd.to_numeric(df["frame"], errors="coerce").fillna(method="ffill").fillna(0.0).to_numpy()
        return fr / fps
    # Last resort
    return np.arange(len(df), dtype=float) / 30.0

def detect_aus(df: pd.DataFrame) -> dict:
    # Look for both *_r and *_c, ignore case, after .str.strip()
    mapping = {}
    for c in df.columns:
        name = str(c).strip()
        if AU_PAT.match(name):
            # normalize case to AUxx_r / AUxx_c
            head, tail = name[:-2], name[-2:].lower()
            norm = head + tail
            mapping[norm] = c
    return mapping  # normalized -> original

def process(csv_path: Path, max_points=1500):
    df = read_csv_smart(csv_path)

    # --- NEW: compute success_rate BEFORE filtering ---
    success_rate = None
    try:
        if "success" in df.columns:
            total_frames = int(len(df))
            kept_frames = int((df["success"] == 1).sum())
            success_rate = (kept_frames / total_frames * 100.0) if total_frames > 0 else 0.0
    except Exception:
        success_rate = None
    # -----------------------------------------------

    # Keep successful frames if available
    if "success" in df.columns:
        ok = df["success"] == 1
        if ok.any():
            df = df[ok].copy()

    t = time_vector(df)
    if len(t) == 0:
        return {
            "time": [],
            "duration": 0.0,
            "aus": {},
            "meta": {"success_rate": success_rate},  # <<< NEW
        }

    au_map = detect_aus(df)           # normalized -> original
    aus_dict = {}
    for norm, orig in au_map.items():
        vals = pd.to_numeric(df[orig], errors="coerce").fillna(0.0).to_numpy()
        aus_dict[norm] = vals

    # Downsample for light rendering
    if len(t) > max_points:
        idx = np.linspace(0, len(t)-1, max_points).astype(int)
    else:
        idx = np.arange(len(t))

    bundle = {
        "time": np.round(t[idx], 4).tolist(),
        "duration": float(t[idx][-1]) if len(idx) else 0.0,
        "aus": {k: np.round(v[idx], 3).tolist() for k, v in aus_dict.items()},
        "meta": {"success_rate": success_rate},   # <<< NEW
    }
    return bundle

def main():
    csvs = list(IN.glob("*.csv"))
    if not csvs:
        print(f"[warn] No CSVs found in {IN}")
        return
    for csv in csvs:
        try:
            out = OUT / f"{csv.stem}.json"
            bundle = process(csv)
            out.write_text(json.dumps(bundle), encoding="utf-8")
            print(f"[OK] {csv.name} -> {out}  (AUs: {len(bundle['aus'])}, points: {len(bundle['time'])}, success%: {bundle.get('meta',{}).get('success_rate')})")
        except Exception as e:
            print(f"[ERR] {csv.name}: {e}")

if __name__ == "__main__":
    main()
