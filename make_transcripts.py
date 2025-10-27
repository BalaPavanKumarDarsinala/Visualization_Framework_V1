# make_transcripts.py
# Generate word-level transcripts from raw MP4s in data/mosi_videos/
# Outputs: CSV per video in data/transcripts/<video_id>.csv

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import csv
import json
from faster_whisper import WhisperModel


ROOT = Path(__file__).resolve().parent
VID_DIR = ROOT / "data" / "mosi_videos"
OUT_DIR = ROOT / "data" / "transcripts"
RAW_DIR = ROOT / "data" / "transcripts_raw_json"

OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Choose model
MODEL_NAME = "small.en"   # try "medium.en" for better accuracy (slower)
DEVICE = "cpu"            # use "cuda" if you have NVIDIA GPU
COMPUTE_TYPE = "int8"     # "float16" on cuda; "int8" is efficient on cpu

def main():
    model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)

    for mp4 in sorted(VID_DIR.glob("*.mp4")):
        vid = mp4.stem
        out_csv = OUT_DIR / f"{vid}.csv"
        out_json = RAW_DIR / f"{vid}.json"

        print(f"[{vid}] transcribing...")
        segments, info = model.transcribe(
            str(mp4),
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
        )

        words = []
        raw_dump = {"segments": []}

        for seg in segments:
            raw_seg = {"start": seg.start, "end": seg.end, "text": seg.text, "words": []}
            if seg.words:
                for w in seg.words:
                    if w.start and w.end:
                        words.append({"start": w.start, "end": w.end, "word": w.word.strip()})
                        raw_seg["words"].append({"start": w.start, "end": w.end, "word": w.word})
            raw_dump["segments"].append(raw_seg)

        # save CSV
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "start", "end", "word"])
            for w in words:
                writer.writerow([vid, f"{w['start']:.3f}", f"{w['end']:.3f}", w["word"]])

        with open(out_json, "w", encoding="utf-8") as jf:
            json.dump(raw_dump, jf, ensure_ascii=False, indent=2)

        print(f"  saved {out_csv}")

if __name__ == "__main__":
    main()
