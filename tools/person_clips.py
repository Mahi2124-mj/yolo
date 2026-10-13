"""Clip agent (10 Oct 2026, the user: "whenever a person appears, take their clip and
snapshot"). The monitor writes one snapshot + a persons.csv row per person
(poketenashi/person_log.py); this light process (no models, no OpenCV) cuts each
person's time span, padded, out of the raw camera recording that tools.light_record
writes (ffmpeg stream copy, no re-encoding) into
  <persons>/<YYYYMMDD>/<cam>/<HHMMSS>_<cam>_ID<tid>.mp4
and lists it in <persons>/<YYYYMMDD>/clips.csv. A row is cut once the segment that
holds it is closed (a newer segment exists). Checks once a minute.

    python -m tools.person_clips --until 09:00 --raw D:/poketenashi-data/persons_raw
"""
import argparse
import csv
import datetime as dt
import glob
import os
import re
import subprocess
import time

PAD = 2.0


def segments(raw, cam):
    out = []
    for f in glob.glob(os.path.join(raw, "*", "raw", f"{cam}_*.mp4")):
        m = re.search(r"_(\d{8}_\d{6})\.mp4$", f)
        if m:
            out.append((dt.datetime.strptime(m.group(1), "%Y%m%d_%H%M%S"), f))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", default="09:00")
    ap.add_argument("--persons", default="D:/poketenashi-data/persons")
    ap.add_argument("--raw", default="D:/poketenashi-data/persons_raw")
    a = ap.parse_args()
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    now = dt.datetime.now()
    h, m = map(int, a.until.split(":"))
    end = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if end <= now:
        end += dt.timedelta(days=1)
    done, loaded = set(), set()
    while dt.datetime.now() < end + dt.timedelta(minutes=15):
        try:
            for day_dir in sorted(glob.glob(os.path.join(a.persons, "2*"))):
                pc = os.path.join(day_dir, "persons.csv")
                if not os.path.exists(pc):
                    continue
                cl = os.path.join(day_dir, "clips.csv")
                if os.path.exists(cl) and day_dir not in loaded:
                    with open(cl, encoding="utf-8") as f:
                        done |= {r[0] for r in csv.reader(f) if r}
                loaded.add(day_dir)
                with open(pc, encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                segs = {}
                for r in rows:
                    snap = r["snapshot"]
                    if snap in done:
                        continue
                    cam = r["camera_id"]
                    if cam not in segs:
                        segs[cam] = segments(a.raw, cam)
                    t0 = dt.datetime.strptime(r["start"], "%Y-%m-%d %H:%M:%S") - dt.timedelta(seconds=PAD)
                    t1 = dt.datetime.strptime(r["end"], "%Y-%m-%d %H:%M:%S") + dt.timedelta(seconds=PAD)
                    ss = segs[cam]
                    k = max((i for i, (st, _) in enumerate(ss) if st <= t0), default=None)
                    if k is None:
                        if ss and ss[0][0] > t1:          # before the recording started: never
                            done.add(snap)
                        continue
                    if k == len(ss) - 1:                   # segment still being written
                        continue
                    st, f = ss[k]
                    outp = os.path.join(day_dir, os.path.splitext(snap)[0] + ".mp4")
                    os.makedirs(os.path.dirname(outp), exist_ok=True)
                    cmd = [ff, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{(t0 - st).total_seconds():.1f}",
                           "-i", f, "-t", f"{(t1 - t0).total_seconds():.1f}", "-c", "copy", outp]
                    ok = subprocess.run(cmd, capture_output=True).returncode == 0 and os.path.exists(outp)
                    with open(cl, "a", newline="", encoding="utf-8") as fo:
                        csv.writer(fo).writerow([snap, os.path.relpath(outp, day_dir) if ok else "", r["start"], r["end"], r["alerts"]])
                    done.add(snap)
        except Exception as e:                      # never die on one bad pass
            print(f"{dt.datetime.now():%H:%M:%S} {type(e).__name__}: {e}", flush=True)
        if dt.datetime.now() >= end + dt.timedelta(minutes=15):
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
