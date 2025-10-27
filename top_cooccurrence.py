import json, numpy as np
from pathlib import Path

PAT = Path("data/patterns")
OUT = Path("data/patterns_cooc"); OUT.mkdir(parents=True, exist_ok=True)

WINDOW = 1.0  # seconds

def time_of(idx, tarr):
    idx = np.clip(idx, 0, len(tarr)-1)
    return tarr[idx]

for jf in PAT.glob("*.json"):
    d = json.loads(jf.read_text())
    T = np.array(d["time"])
    # define named event streams as time arrays
    streams = {
        "smile":   [time_of(i, T) for i in d.get("AU12_r_peaks", [])],
        "frown":   [time_of(i, T) for i in d.get("AU04_r_peaks", [])],
        "cheek":   [time_of(i, T) for i in d.get("AU06_r_peaks", [])],
        "gaze":    [time_of(i, T) for i in d.get("gaze_shifts", [])],
        # "prosody": [ ... from audio if available ... ],
    }

    # scan each event in each stream; look for matches in others within ±WINDOW
    combos = {}
    all_names = list(streams.keys())
    for a in all_names:
        for ta in streams[a]:
            hit = [a]
            for b in all_names:
                if b==a: continue
                if any(abs(tb-ta)<=WINDOW for tb in streams[b]):
                    hit.append(b)
            if len(hit)>=2:
                key = "+".join(sorted(set(hit)))
                combos[key] = combos.get(key, 0) + 1

    OUT.joinpath(jf.name).write_text(json.dumps({"combos": combos}, indent=2))
    print("cooc ->", jf.stem, combos)
