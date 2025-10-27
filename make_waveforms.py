# make_waveforms.py
# ---------------------------------------------------------
# For every .mp4 in data/mosi_videos:
#  - ensure mono 16k WAV exists in data/audio
#  - compute a peak-preserving downsampled waveform
#  - write to data/waveforms/<id>.json  (list[float] in [-1,1])
#
# Usage:
#   python make_waveforms.py
#
# Requires: soundfile, numpy, ffmpeg on PATH

from pathlib import Path
import json
import math
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parent
VID_DIR = ROOT / "data" / "mosi_videos"
AUD_DIR = ROOT / "data" / "audio"
WVF_DIR = ROOT / "data" / "waveforms"

AUD_DIR.mkdir(parents=True, exist_ok=True)
WVF_DIR.mkdir(parents=True, exist_ok=True)

NUM_POINTS = 640         # canvas width; change to 1000 for more detail
TARGET_SR  = 16000       # match the app’s expectation

def ensure_wav(mp4_path: Path, wav_path: Path, sr: int = TARGET_SR):
    if wav_path.exists():
        return
    cmd = ["ffmpeg", "-y", "-i", str(mp4_path), "-ac", "1", "-ar", str(sr), str(wav_path)]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def read_wav(path: Path):
    import soundfile as sf
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y[:, 0]
    return y, sr

def downsample_peak_preserving(y: np.ndarray, n: int) -> np.ndarray:
    """Peak-preserving bucket reducer: for each bucket, take max abs sample (with sign)."""
    if len(y) == 0:
        return np.zeros(n, dtype=np.float32)
    if len(y) <= n:
        # pad/trim to n
        out = np.zeros(n, dtype=np.float32)
        out[:len(y)] = y
        return out
    # split into n buckets by index range
    edges = np.linspace(0, len(y), num=n+1, dtype=int)
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        seg = y[edges[i]:edges[i+1]]
        if seg.size == 0:
            continue
        # pick the element with max absolute value to keep sharp peaks visible
        j = int(np.argmax(np.abs(seg)))
        out[i] = float(seg[j])
    # normalize to [-1,1] if needed
    m = float(np.max(np.abs(out))) if np.any(out) else 1.0
    if m > 0:
        out /= m
    return out

def main():
    mp4s = sorted(VID_DIR.glob("*.mp4"))
    if not mp4s:
        print(f"No videos found in {VID_DIR}")
        return

    for mp4 in mp4s:
        vid = mp4.stem
        wav = AUD_DIR / f"{vid}.wav"
        jsn = WVF_DIR / f"{vid}.json"

        print(f"[{vid}] extracting WAV (if needed) …")
        ensure_wav(mp4, wav, TARGET_SR)

        print(f"[{vid}] loading audio …")
        y, sr = read_wav(wav)
        if sr != TARGET_SR:
            print(f"  Warning: WAV sr={sr}, expected {TARGET_SR}")

        print(f"[{vid}] computing waveform ({NUM_POINTS} pts) …")
        samples = downsample_peak_preserving(y, NUM_POINTS).tolist()

        print(f"[{vid}] writing {jsn.relative_to(ROOT)}")
        with open(jsn, "w", encoding="utf-8") as f:
            json.dump(samples, f)

    print("Done. Waveform JSONs in data/waveforms/")

if __name__ == "__main__":
    main()
