"""KE shadow report (13 Oct 2026, the user: "option 2 - show me the result tomorrow; any case
that looks NG, unclear or excluded: its images and clips, all in one file, sorted, so I can
teach you clearly").

Reads D:/poketenashi-data/ke_shadow/<day>/shadow.csv (live phone check vs the shadow checker,
poketenashi/ke.py) between FROM and TO, joins each row to
  - the live KE alert (evidence/<day>/alerts.jsonl: clip + poster), and
  - the person's clean clip (persons/<day>/persons.csv - no overlay),
and sorts every person into
  both_fire / live_only / checker_only / neither
(the folder name says what each method did; OK / NG is the user's call). Claude's suggestion
can be added in suggest.csv (time,track,suggestion,reason) next to the output.

    python tools/agent/ke_shadow_report.py <out_dir> "2026-10-13 10:35:00" "2026-10-14 07:00:00"
Output: <out_dir>/<group>/ (person crop, alert poster+clip, clean clip), report.csv, REPORT.html
"""
import base64
import csv
import datetime as dt
import glob
import html
import json
import os
import shutil
import sys

D = "D:/poketenashi-data"
GROUPS = [("checker_only", "Only the new checker would alert (live said no)"),
          ("live_only", "Only the live KE alerted (new checker would ignore)"),
          ("both_fire", "Both alert"),
          ("neither", "Both ignore (stopped)")]


def T(s):
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def main(out, t0, t1):
    t0, t1 = T(t0), T(t1)
    days = sorted({(t0 + dt.timedelta(days=i)).strftime("%Y%m%d") for i in range((t1 - t0).days + 1)})
    people = {}
    for day in days:
        f = os.path.join(D, "ke_shadow", day, "shadow.csv")
        if not os.path.exists(f):
            continue
        for r in csv.DictReader(open(f, encoding="utf-8")):
            t = T(r["time"])
            if not t0 <= t <= t1:
                continue
            k = (day, r["track"])
            a = people.setdefault(k, {"day": day, "track": r["track"], "time": r["time"], "live": 0, "chk": 0,
                                      "p": 0.0, "why": r["live_why"], "imgs": []})
            a["live"] |= int(r["live_fires"]); a["chk"] |= int(r["checker_fires"])
            a["p"] = max(a["p"], float(r["checker_p"])); a["imgs"].append(os.path.join(D, "ke_shadow", day, r["image"]))
            if int(r["live_fires"]):
                a["why"] = r["live_why"]
    alerts, plog = {}, {}
    for day in days:
        aj = os.path.join(D, "evidence", day, "alerts.jsonl")
        for l in open(aj, encoding="utf-8") if os.path.exists(aj) else []:
            try:
                x = json.loads(l)
            except ValueError:
                continue
            if x.get("rule") == "KE":
                alerts.setdefault((day, str(x["track_id"])), []).append(x)
        pc = os.path.join(D, "persons", day, "persons.csv")
        for r in csv.DictReader(open(pc, encoding="utf-8")) if os.path.exists(pc) else []:
            plog.setdefault((day, r["track"]), []).append(r)
    sug = {}
    sp = os.path.join(out, "suggest.csv")
    if os.path.exists(sp):
        for r in csv.DictReader(open(sp, encoding="utf-8-sig")):
            sug[(r["time"], r["track"])] = (r["suggestion"], r.get("reason", ""))
    rows = []
    for (day, tid), a in sorted(people.items(), key=lambda kv: kv[1]["time"]):
        g = {(1, 1): "both_fire", (1, 0): "live_only", (0, 1): "checker_only", (0, 0): "neither"}[(a["live"], a["chk"])]
        gd = os.path.join(out, g); os.makedirs(gd, exist_ok=True)
        stem = f"{a['time'][11:19].replace(':', '')}_ID{tid}"
        files = []
        for im in a["imgs"][-2:]:
            if os.path.exists(im):
                dst = os.path.join(gd, f"{stem}_crop_{os.path.basename(im).split('_', 2)[-1]}")
                shutil.copy2(im, dst); files.append(dst)
        t = T(a["time"]); cam, poster = "", ""
        for x in alerts.get((day, tid), []):
            if abs((T(x["time"]) - t).total_seconds()) <= 120:
                cam = x.get("camera_name", "")
                for kk in ("poster", "clip"):
                    src = os.path.join(D, "evidence", day, x.get(kk) or "")
                    if x.get(kk) and os.path.exists(src):
                        dst = os.path.join(gd, f"{stem}_alert_{x[kk]}"); shutil.copy2(src, dst); files.append(dst)
                        if kk == "poster":
                            poster = dst
                break
        for r in plog.get((day, tid), []):
            if T(r["start"]) - dt.timedelta(seconds=5) <= t <= T(r["end"]) + dt.timedelta(seconds=5):
                cam = cam or r["camera"]
                for kk in ("snapshot", "clip"):
                    src = os.path.join(D, "persons", day, r[kk]) if r.get(kk) else ""
                    if src and os.path.exists(src):
                        dst = os.path.join(gd, f"{stem}_person_{os.path.basename(src)}"); shutil.copy2(src, dst); files.append(dst)
                break
        s, why = sug.get((a["time"], tid), ("", ""))
        rows.append({"group": g, "time": a["time"], "camera": cam, "track": tid, "live_fires": a["live"],
                     "checker_fires": a["chk"], "checker_p": round(a["p"], 2), "live_why": a["why"],
                     "claude_suggestion": s, "reason": why, "files": " | ".join(os.path.basename(f) for f in files),
                     "_img": files[0] if files else "", "_poster": poster})
    with open(os.path.join(out, "report.csv"), "w", newline="", encoding="utf-8-sig") as f:
        keys = [k for k in rows[0] if not k.startswith("_")] if rows else ["group"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    write_html(out, rows, t0, t1)
    for g, _ in GROUPS:
        print(g, sum(r["group"] == g for r in rows))


def thumb(p):
    try:
        import cv2
        im = cv2.imread(p)
        h = 260; im = cv2.resize(im, (max(1, int(im.shape[1] * h / im.shape[0])), h))
        return "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 80])[1]).decode()
    except Exception:
        return ""


def write_html(out, rows, t0, t1):
    h = [f"<!doctype html><meta charset=utf-8><title>KE shadow report</title><style>body{{font-family:sans-serif;margin:16px}}"
         "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px;vertical-align:top;font-size:13px}"
         "img{height:200px;margin-right:4px}</style>",
         f"<h2>KE: live check vs new checker (shadow) - {t0} to {t1}</h2>",
         "<p>" + " &middot; ".join(f"{d}: <b>{sum(r['group'] == g for r in rows)}</b>" for g, d in GROUPS) + "</p>"]
    for g, d in GROUPS:
        rs = [r for r in rows if r["group"] == g]
        h.append(f"<h3>{html.escape(d)} ({len(rs)}) - folder {g}</h3><table><tr><th>time</th><th>camera / ID</th>"
                 "<th>live / checker</th><th>Claude</th><th>pictures</th><th>files</th></tr>")
        for r in rs:
            ims = "".join(f"<img src='{thumb(p)}'>" for p in (r["_img"], r["_poster"]) if p)
            h.append(f"<tr><td>{r['time']}</td><td>{html.escape(r['camera'])} ID{r['track']}</td>"
                     f"<td>live {'YES' if r['live_fires'] else 'no'} ({html.escape(r['live_why'])})<br>checker "
                     f"{'YES' if r['checker_fires'] else 'no'} (P {r['checker_p']})</td>"
                     f"<td>{html.escape(r['claude_suggestion'])}<br>{html.escape(r['reason'])}</td><td>{ims}</td>"
                     f"<td>{html.escape(r['files']).replace(' | ', '<br>')}</td></tr>")
        h.append("</table>")
    open(os.path.join(out, "REPORT.html"), "w", encoding="utf-8").write("\n".join(h))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
