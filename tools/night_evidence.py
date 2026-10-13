"""Keep every alert of the night: copy each new alert's clip and poster (and its
log row) into the night folder, so the 2-day retention cannot delete them
before they are reviewed and labelled. Runs until the end time, checking once
a minute; a clip that is still being encoded is picked up on a later pass.

    python -m tools.night_evidence --until 09:00 --out D:/poketenashi-data/night
"""
import argparse
import datetime as dt
import json
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", default="09:00")
    ap.add_argument("--out", default="D:/poketenashi-data/night")
    a = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))
    log_file = cfg["alerts"].get("log_file", "violations.jsonl")
    log_file = log_file if os.path.isabs(log_file) else os.path.join(ROOT, log_file)
    clip_dir = cfg.get("clips", {}).get("dir", "clips")
    now = dt.datetime.now()
    h, m = map(int, a.until.split(":"))
    end = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if end <= now:
        end += dt.timedelta(days=1)
    since = now.strftime("%Y-%m-%d %H:%M:%S")
    out = os.path.join(a.out, now.strftime("%Y%m%d"), "alerts")
    os.makedirs(out, exist_ok=True)
    rows_path = os.path.join(out, "alerts.jsonl")
    seen = set()
    if os.path.exists(rows_path):
        seen = {json.dumps(json.loads(l), sort_keys=True) for l in open(rows_path, encoding="utf-8") if l.strip()}
    pending = set()
    while True:
        try:
            for l in open(log_file, encoding="utf-8"):
                try:
                    r = json.loads(l)
                except ValueError:
                    continue
                if r.get("time", "") < since:
                    continue
                k = json.dumps(r, sort_keys=True)
                if k not in seen:
                    seen.add(k)
                    with open(rows_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(r) + "\n")
                    pending.update(x for x in (r.get("clip"), r.get("poster"), r.get("snapshot")) if x)
            for name in list(pending):
                src = os.path.join(clip_dir, name)
                if os.path.exists(src) and time.time() - os.path.getmtime(src) > 20:
                    shutil.copy2(src, os.path.join(out, name))
                    pending.discard(name)
        except Exception as e:                 # never die overnight on one bad pass
            with open(os.path.join(out, "copy.log"), "a", encoding="utf-8") as f:
                f.write(f"{dt.datetime.now():%H:%M:%S} {type(e).__name__}: {e}\n")
        if dt.datetime.now() >= end and not pending:
            break
        if dt.datetime.now() >= end + dt.timedelta(minutes=10):
            break
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
