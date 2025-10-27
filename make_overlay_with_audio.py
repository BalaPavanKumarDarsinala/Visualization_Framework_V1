import subprocess
from pathlib import Path

# Define base paths
BASE = Path(__file__).resolve().parent
avi_dir = BASE / "data" / "openface_raw"
mp4_dir = BASE / "data" / "mosi_videos"

# Iterate over all AVI files from OpenFace
for avi in avi_dir.glob("*.avi"):
    vid_id = avi.stem  # e.g., 7JsX8y1ysxY
    mp4 = mp4_dir / f"{vid_id}.mp4"
    out = mp4_dir / f"{vid_id}_overlay.mp4"

    if not mp4.exists():
        print(f"❌ Missing audio source for {vid_id}")
        continue

    if out.exists():
        print(f"✅ Already exists: {out.name}")
        continue

    cmd = [
        "ffmpeg",
        "-i", str(avi),
        "-i", str(mp4),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(out)
    ]

    print(f"🔄 Merging: {vid_id} ...")
    subprocess.run(cmd, shell=True)
    print(f"✅ Done: {out.name}\n")

print("🎬 All overlays processed.")
