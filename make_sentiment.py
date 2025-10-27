# make_sentiment.py  — robust version
# ---------------------------------------------------------
# - Groups word-level transcripts into sentences (by punctuation OR time gaps)
# - Handles NaNs / non-string tokens safely
# - Runs HF sentiment per sentence (batched) and saves:
#       data/sentiment/<id>.csv  with columns:
#       text,start,end,label,score,polarity   (polarity in [-1, 1])
#
# Usage:
#   python make_sentiment.py
#
# Pre-req:
#   pip install -U transformers torch sentencepiece

from pathlib import Path
import pandas as pd
import numpy as np
from transformers import pipeline

ROOT = Path(__file__).resolve().parent
TRN_DIR = ROOT / "data" / "transcripts"
SEN_DIR = ROOT / "data" / "sentiment"
SEN_DIR.mkdir(parents=True, exist_ok=True)

# Load a small, fast model (binary POS/NEG)
clf = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def _norm_word(x):
    """Return a clean string or '' for NaNs/invalid."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    # Filter out placeholders that aren't spoken words if you like:
    # e.g., [Music], [Laughter], etc. (optional)
    return s

def words_to_sentences(df, gap_ms=600):
    """
    Convert word-level rows into sentences by:
      - punctuation at end of a word (., !, ?)
      - OR long pause between words (> gap_ms)
    Returns DataFrame[text,start,end]
    """
    # Ensure expected columns and dtypes
    need = {"start", "end", "word"}
    if not need.issubset(df.columns):
        raise ValueError(f"Transcript must contain columns {need}")
    df = df[["start", "end", "word"]].copy()
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"]   = pd.to_numeric(df["end"], errors="coerce")
    df["word"]  = df["word"].apply(_norm_word)

    # Drop rows with missing times or empty words
    df = df.dropna(subset=["start", "end"]).reset_index(drop=True)
    df = df[df["word"] != ""].reset_index(drop=True)

    rows = []
    cur_words = []
    s0 = None
    e_last = None

    # Helper: does token end with sentence punctuation?
    def ends_sentence(tok: str) -> bool:
        return tok.endswith((".", "!", "?"))

    for _, r in df.iterrows():
        w = r["word"]
        s = float(r["start"])
        e = float(r["end"])

        if s0 is None:
            s0 = s

        # Append current token
        cur_words.append(w)

        # Sentence break if long pause from previous token
        if e_last is not None and (s - e_last) * 1000.0 > gap_ms:
            prev_words = cur_words[:-1]
            if prev_words:  # flush previous sentence
                rows.append((" ".join(prev_words).strip(), s0, e_last))
            # Start a new sentence from current word
            cur_words = [w]
            s0 = s

        # End-of-sentence by punctuation
        if ends_sentence(w):
            rows.append((" ".join(cur_words).strip(), s0, e))
            cur_words, s0 = [], None

        e_last = e

    # Flush leftovers
    if cur_words:
        rows.append((" ".join(cur_words).strip(), s0 if s0 is not None else 0.0, e_last if e_last is not None else 0.0))

    out = pd.DataFrame(rows, columns=["text", "start", "end"])
    # Drop super-short/empty sentences
    out["text"] = out["text"].astype(str).str.strip()
    out = out[out["text"] != ""].reset_index(drop=True)
    return out

def score_sentences(sents_df: pd.DataFrame, batch_size=32) -> pd.DataFrame:
    if sents_df.empty:
        return sents_df.assign(label=[], score=[], polarity=[])
    texts = sents_df["text"].tolist()
    preds = clf(texts, truncation=True, batch_size=batch_size)
    out = sents_df.copy()
    out["label"] = [p["label"] for p in preds]          # POSITIVE / NEGATIVE
    out["score"] = [float(p["score"]) for p in preds]    # confidence 0..1
    out["polarity"] = np.where(out["label"].str.startswith("POS"), 1, -1) * out["score"]
    return out

def process_one(csv_path: Path):
    vid = csv_path.stem
    try:
        df = pd.read_csv(csv_path)
        sents = words_to_sentences(df)
        if sents.empty:
            print(f"[{vid}] no sentences extracted (skipping).")
            return
        scored = score_sentences(sents)
        out = SEN_DIR / f"{vid}.csv"
        scored.to_csv(out, index=False)
        print(f"[{vid}] -> {out}")
    except Exception as e:
        print(f"[{vid}] ERROR: {e}")

def main():
    any_found = False
    for csv in sorted(TRN_DIR.glob("*.csv")):
        any_found = True
        process_one(csv)
    if not any_found:
        print(f"No transcript CSVs found in {TRN_DIR}")

if __name__ == "__main__":
    main()
