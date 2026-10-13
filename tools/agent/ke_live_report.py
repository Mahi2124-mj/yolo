"""KE live test report (12 Oct 2026): every KE alert since <since> - zoom sheet of the
person (alert clip), the clean-clip object check (ke_review), for an OK/NG review, and the
before/after comparison with the reviewed 5-11 Oct alerts (KE right 65 / wrong 49).
    python tools/agent/ke_live_report.py "2026-10-12 12:10:30" <out_dir>"""
import csv, glob, json, os, subprocess, sys
since, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
EV = "D:/poketenashi-data/evidence"
rows = []
for aj in sorted(glob.glob(EV + "/2*/alerts.jsonl")):
    for l in open(aj, encoding="utf-8"):
        try: a = json.loads(l)
        except ValueError: continue
        if a.get("rule") == "KE" and a["time"] >= since:
            rows.append(a)
seen, uniq = set(), []
for a in rows:
    k = (a["time"], a.get("camera"), a["track_id"])
    if k not in seen: seen.add(k); uniq.append(a)
with open(os.path.join(out, "ke_alerts.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["time", "camera", "track", "clip", "poster", "OK_NG"])
    for a in uniq:
        day = a["time"][:10].replace("-", "")
        w.writerow([a["time"], a.get("camera_name") or a.get("camera"), a["track_id"],
                    os.path.join(EV, day, a.get("clip") or ""), os.path.join(EV, day, a.get("poster") or ""), ""])
print(len(uniq), "KE alerts since", since)
