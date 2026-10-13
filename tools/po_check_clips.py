"""PO check - clips + page (13 Oct 2026; see poketenashi/po_watch.py).

For every person in PO_check/<day>/po_check.csv it copies that person's clean clip (no
overlay) from D:/poketenashi-data/persons/<day>/clips.csv - same camera + track id, cut by
tools.person_clips from the continuous recording - next to the snapshot, and writes
PO_check/<day>/index.html: people WITHOUT a PO alert first (possible misses), then the
ones the model alerted on. No models, file copies only. Run by tools.person_agent.

    python -m tools.po_check_clips [--once]
"""
import csv
import datetime as dt
import glob
import html
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECK = os.path.join(ROOT, "PO_check")
PERSONS = "D:/poketenashi-data/persons"


def T(s):
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def one_day(d):
    day = os.path.basename(d)
    pc = os.path.join(d, "po_check.csv")
    if not os.path.exists(pc):
        return
    rows = list(csv.DictReader(open(pc, encoding="utf-8")))
    clips = {}                                   # (camera_id, track) -> [(start, end, clip)]
    cc = os.path.join(PERSONS, day, "clips.csv")
    pp = os.path.join(PERSONS, day, "persons.csv")
    if os.path.exists(pp):
        for r in csv.DictReader(open(pp, encoding="utf-8")):
            if r.get("clip"):
                clips.setdefault((r["camera_id"], r["track"]), []).append((r["start"], r["end"], r["clip"]))
    if os.path.exists(cc):
        for r in csv.reader(open(cc, encoding="utf-8")):
            if len(r) >= 4 and r[1]:
                b = os.path.basename(r[0])                     # <HHMMSS>_<cam>_ID<tid>.jpg
                parts = b[:-4].split("_")
                if len(parts) >= 3 and parts[2].startswith("ID"):
                    clips.setdefault((parts[1], parts[2][2:]), []).append((r[2], r[3], r[1]))
    for r in rows:
        r["clip"] = ""
        stem = r["snapshot"][:-4]
        have = os.path.join(d, stem + ".mp4")
        if os.path.exists(have):
            r["clip"] = stem + ".mp4"
            continue
        t0 = T(r["start"])
        for s, e, c in clips.get((r["camera_id"], r["track"]), []):
            if abs((T(s) - t0).total_seconds()) <= 120:
                src = os.path.join(PERSONS, day, c)
                if os.path.exists(src):
                    shutil.copy2(src, have)
                    r["clip"] = stem + ".mp4"
                break
    rows.sort(key=lambda r: (int(r["po_alert"]), r["start"]))
    miss = [r for r in rows if r["po_alert"] == "0"]
    h = [f"<!doctype html><meta charset=utf-8><title>PO check {day}</title>",
         "<style>body{font-family:sans-serif;margin:16px}.c{display:inline-block;width:250px;margin:6px;vertical-align:top;"
         "border:1px solid #ccc;padding:4px;font-size:13px}.c img{max-width:240px;max-height:330px}.m{border-color:#d33}</style>",
         f"<h2>PO check {day} - hand hidden at the hip</h2>",
         f"<p><b>{len(miss)}</b> people with NO PO alert (possible misses - look at these) &middot; "
         f"{len(rows) - len(miss)} with a PO alert. Click a picture for the clip.</p>"]
    for title, part, cls in (("No PO alert - possible miss", miss, "c m"),
                             ("PO alert given", [r for r in rows if r["po_alert"] == "1"], "c")):
        h.append(f"<h3>{title} ({len(part)})</h3>")
        for r in part:
            img = f"<img src='{html.escape(r['snapshot'])}'>"
            link = f"<a href='{html.escape(r['clip'])}'>{img}</a>" if r["clip"] else img
            h.append(f"<div class='{cls}'>{link}<br>{r['start'][11:]} {html.escape(r['camera'])} ID{r['track']}<br>"
                     f"hand hidden {r['hidden']}x of {r['checks']} looks{'' if r['clip'] else ' (clip not ready)'}</div>")
    open(os.path.join(d, "index.html"), "w", encoding="utf-8").write("\n".join(h))


def main():
    while True:
        for d in sorted(glob.glob(os.path.join(CHECK, "2*")))[-2:]:
            try:
                one_day(d)
            except Exception as e:
                print(f"{dt.datetime.now():%H:%M:%S} {os.path.basename(d)}: {type(e).__name__}: {e}", flush=True)
        if "--once" in sys.argv:
            break
        time.sleep(600)


if __name__ == "__main__":
    main()
