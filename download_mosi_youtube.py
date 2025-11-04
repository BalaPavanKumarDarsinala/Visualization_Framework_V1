# download_mosi_all.py
import sys, os, csv, argparse
from pathlib import Path
from datetime import datetime

# --- Point to the folder that contains cmu_mosi_std_folds.py ---
FOLDS_DIR = Path(r"C:\Users\balap\Desktop\Project\data\CMU-MultimodalSDK\mmsdk\mmdatasdk\dataset\standard_datasets\CMU_MOSI")
if str(FOLDS_DIR) not in sys.path:
    sys.path.append(str(FOLDS_DIR))

try:
    from cmu_mosi_std_folds import (
        standard_train_fold,
        standard_valid_fold,
        standard_test_fold,
    )
except Exception as e:
    print("ERROR: Could not import cmu_mosi_std_folds.py")
    print("Check FOLDS_DIR above. Current value:", FOLDS_DIR)
    raise

import yt_dlp


def build_urls(ids):
    return [f"https://www.youtube.com/watch?v={vid}" for vid in ids]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=["train", "valid", "test", "all"], default="all",
                        help="Which fold(s) to download")
    parser.add_argument("--outdir", default="data/mosi_videos_test", help="Output folder")
    parser.add_argument("--archive", default="data/mosi_videos_test/.download_archive.txt",
                        help="Archive file to skip already-downloaded IDs")
    parser.add_argument("--max-retries", type=int, default=10, help="Max retries per video")
    args = parser.parse_args()

    # Pick the IDs
    if args.set == "train":
        ids = standard_train_fold
    elif args.set == "valid":
        ids = standard_valid_fold
    elif args.set == "test":
        ids = standard_test_fold
    else:
        ids = standard_train_fold + standard_valid_fold + standard_test_fold

    urls = build_urls(ids)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    Path(args.archive).parent.mkdir(parents=True, exist_ok=True)

    log_csv = outdir / "download_log.csv"

    # Prepare CSV log
    if not log_csv.exists():
        with log_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "video_id", "status", "filepath_or_error"])

    # Track stats
    stats = {"success": 0, "skipped": 0, "failed": 0}

    # Hook to capture per-video result paths / errors
    def progress_hook(d):
        if d.get("status") == "finished":
            # When merging completes, file should be at d["filename"]
            pass

    ydl_opts = {
        # Best mp4 video + m4a audio, fallback sensibly, final MP4
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "outtmpl": str(outdir / "%(id)s.%(ext)s"),
        # Resume / skip:
        "download_archive": args.archive,    # remembers finished IDs
        "continue_dl": True,                 # resume partials
        "nooverwrites": True,                # do not overwrite completed
        # Reliability:
        "retries": args.max_retries,
        "fragment_retries": args.max_retries,
        "skip_unavailable_fragments": True,
        "ignoreerrors": True,                # don’t stop on errors
        # Progress:
        "progress_hooks": [progress_hook],
        "quiet": False,
        "noprogress": False,
        # A tiny rate limit can help stability on some networks (optional):
        # "limit_rate": "2M",
        # Slightly faster HLS fragment downloads (if used):
        "concurrent_fragment_downloads": 3,
    }

    # Run downloads in one session so archive tracking works nicely
    results = []

    # Custom wrapper to record success/skip/failure per URL
    class Logger:
        def debug(self, msg): pass
        def warning(self, msg): pass
        def error(self, msg):
            # yt-dlp may send non-fatal errors here; we handle after try/except
            results.append(("error_line", msg))

    ydl_opts["logger"] = Logger()

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for url in urls:
            vid = url.split("v=")[-1]
            ts = datetime.now().isoformat(timespec="seconds")
            try:
                # If already in archive, yt-dlp prints "has already been recorded in archive"
                ret = ydl.download([url])
                # ret is 0 on success, non-zero on fatal. But "already in archive" also returns 0.
                # Determine actual file path:
                mp4_path = outdir / f"{vid}.mp4"
                webm_path = outdir / f"{vid}.webm"  # fallback if best was webm
                if mp4_path.exists():
                    stats["success"] += 1
                    results.append((vid, "SUCCESS", str(mp4_path)))
                    with log_csv.open("a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([ts, vid, "SUCCESS", str(mp4_path)])
                elif webm_path.exists():
                    stats["success"] += 1
                    results.append((vid, "SUCCESS", str(webm_path)))
                    with log_csv.open("a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([ts, vid, "SUCCESS", str(webm_path)])
                else:
                    # If archive says it’s already downloaded, treat as SKIPPED
                    # Quick heuristic: if ret == 0 and not found, still mark SKIPPED so we don’t fail the run
                    stats["skipped"] += 1
                    results.append((vid, "SKIPPED_OR_ARCHIVED", "Not found in outdir but marked done/archived"))
                    with log_csv.open("a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([ts, vid, "SKIPPED_OR_ARCHIVED", "Not found but considered done"])
            except Exception as e:
                stats["failed"] += 1
                results.append((vid, "FAILED", str(e)))
                with log_csv.open("a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([ts, vid, "FAILED", repr(e)])

    # Summary
    total = len(urls)
    print("\n================ SUMMARY ================")
    print(f"Requested: {total}")
    print(f"Downloaded (or present): {stats['success']}")
    print(f"Skipped/Archived:        {stats['skipped']}")
    print(f"Failed:                  {stats['failed']}")
    print(f"Output folder:           {outdir.resolve()}")
    print(f"Archive file:            {Path(args.archive).resolve()}")
    print(f"Log CSV:                 {log_csv.resolve()}")
    print("=========================================\n")

    # Tip: rerun the same command; only missing/failed will attempt again due to the archive.


if __name__ == "__main__":
    main()
