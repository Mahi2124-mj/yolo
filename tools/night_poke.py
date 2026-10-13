"""Overnight PO / KE review folder: every new PO and KE alert (all cameras) gets its
clip and poster copied into <out>/PO or <out>/KE with a readable name, plus a row
in <out>/<rule>/review.csv with an empty OK_NG column for the user. Checks once a
minute until the end time; a clip still being encoded is picked up later. Light:
no camera / model imports (the PC's memory is tight).

    python -m tools.night_poke --until 09:00 --out "PO KE clips 7-8 Oct"
"""
import argparse
import csv
import datetime as dt
import json
import os
import shutil
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RULES = ("PO", "KE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", default="09:00")
    ap.add_argument("--out", default="PO KE clips")
    ap.add_argument("--log", default=os.path.join(ROOT, "violations.jsonl"))
    ap.add_argument("--clips", default="D:/poketenashi-data/clips")
    ap.add_argument("--rules", default=",".join(RULES), help="comma list, e.g. PO,KE,NA,SHI,RUN,FALL")
    ap.add_argument("--cams", default="", help="only these camera ids, e.g. cam1 (default: all)")
    a = ap.parse_args()
    rules = tuple(x.strip() for x in a.rules.split(",") if x.strip())
    cams = {x.strip() for x in a.cams.split(",") if x.strip()}
    out = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    now = dt.datetime.now()
    h, m = map(int, a.until.split(":"))
    end = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if end <= now:
        end += dt.timedelta(days=1)
    since = now.strftime("%Y-%m-%d %H:%M:%S")
    for r in rules:
        os.makedirs(os.path.join(out, r), exist_ok=True)
        p = os.path.join(out, r, "review.csv")
        if not os.path.exists(p):
            with open(p, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["clip", "time", "camera", "track", "person", "OK_NG (fill in)", "note"])
    seen = set()
    pending = {}                                   # source name -> destination path
    log = open(os.path.join(out, "copy.log"), "a", encoding="utf-8")
    log.write(f"{now:%Y-%m-%d %H:%M:%S} watching {",".join(rules)} alerts ({",".join(sorted(cams)) or "all cameras"}) from {since} until {end:%Y-%m-%d %H:%M}\n")
    log.flush()
    while True:
        try:
            for l in open(a.log, encoding="utf-8"):
                try:
                    r = json.loads(l)
                except ValueError:
                    continue
                if r.get("rule") not in rules or r.get("time", "") < since:
                    continue
                if cams and (r.get("camera") or "cam1") not in cams:
                    continue
                k = (r.get("time"), r.get("camera"), r.get("track_id"), r.get("rule"))
                if k in seen:
                    continue
                seen.add(k)
                base = f"{r['time'][5:].replace('-', '').replace(':', '').replace(' ', '_')}_{r.get('camera_name') or r.get('camera')}_ID{r.get('track_id')}"
                base = base.replace(" ", "")
                for src, ext in ((r.get("clip"), ".mp4"), (r.get("poster"), ".jpg")):
                    if src:
                        pending[src] = os.path.join(out, r["rule"], base + ext)
                with open(os.path.join(out, r["rule"], "review.csv"), "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([base + ".mp4", r.get("time"), r.get("camera_name") or r.get("camera"),
                                            r.get("track_id"), r.get("person", ""), "", ""])
            for src, dst in list(pending.items()):
                p = os.path.join(a.clips, src)
                if os.path.exists(p) and time.time() - os.path.getmtime(p) > 20:
                    shutil.copy2(p, dst)
                    del pending[src]
        except Exception as e:                     # never die overnight on one bad pass
            log.write(f"{dt.datetime.now():%H:%M:%S} {type(e).__name__}: {e}\n")
            log.flush()
        if dt.datetime.now() >= end and not pending or dt.datetime.now() >= end + dt.timedelta(minutes=10):
            break
        time.sleep(60)
    log.write(f"{dt.datetime.now():%H:%M:%S} finished\n")


if __name__ == "__main__":
    main()
