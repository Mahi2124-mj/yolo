"""Complete KE report (12 Oct 2026, the user): every KE alert in the evidence archive
with its picture, time, camera, ID, day / night, the label (user / review agent), the
object the KE review agent found in the hand (since 10 Oct 15:17, clean clips), the
clean clip and the alert clip. Read-only.

    python tools/agent/ke_report.py <out_dir>
"""
import csv
import glob
import json
import os
import sys
from collections import Counter

import cv2

ROOT = r"C:\Users\DX-ADMIN\Pictures\yolo"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
EV = "D:/poketenashi-data/evidence"


def main(out):
    from model_compare import build_labels, label_for, _secs
    os.makedirs(os.path.join(out, "img"), exist_ok=True)
    L = build_labels()
    rev = {}
    for p in glob.glob("D:/poketenashi-data/ke_review/2*/review.csv"):
        day = os.path.basename(os.path.dirname(p))
        for r in csv.DictReader(open(p, encoding="utf-8")):
            rev[(day, int(r["track"]), r["time"][11:19].replace(":", ""))] = (r, os.path.dirname(p))
    rows, seen = [], set()
    for day in sorted(os.path.basename(d) for d in glob.glob(os.path.join(EV, "2*"))):
        aj = os.path.join(EV, day, "alerts.jsonl")
        if not os.path.exists(aj):
            continue
        for line in open(aj, encoding="utf-8"):
            try:
                a = json.loads(line)
            except ValueError:
                continue
            if a.get("rule") != "KE":
                continue
            key = (a["time"], a.get("camera"), a["track_id"])
            if key in seen:
                continue
            seen.add(key)
            hh, tid = a["time"][11:19].replace(":", ""), int(a["track_id"])
            lab, src = label_for(L, "KE", day, hh, tid)
            obj, review_img, clean = "", "", ""
            for (d, t, h), (r, rdir) in rev.items():
                if d == day and t == tid and 0 <= _secs(hh) - _secs(h) <= 120:
                    obj = r.get("objects_in_hand") or "none"
                    if r.get("review_image"):
                        review_img = os.path.join(rdir, r["review_image"])
                    if r.get("clip"):
                        clean = os.path.join("D:/poketenashi-data/persons", day, r["clip"])
            poster = os.path.join(EV, day, a.get("poster") or "")
            th = ""
            src_img = review_img if review_img and os.path.exists(review_img) else poster
            if os.path.exists(src_img):
                im = cv2.imread(src_img)
                if im is not None:
                    th = os.path.join("img", f"{day}_{hh}_ID{tid}.jpg")
                    w = 220 if src_img == review_img else 420
                    cv2.imwrite(os.path.join(out, th), cv2.resize(im, (w, int(w * im.shape[0] / im.shape[1]))),
                                [cv2.IMWRITE_JPEG_QUALITY, 80])
            hour = int(hh[:2])
            rows.append({"time": a["time"], "camera": a.get("camera_name") or a.get("camera"), "track": tid,
                         "night": hour >= 19 or hour < 7, "label": lab, "label_src": src, "object": obj,
                         "thumb": th, "clip": os.path.join(EV, day, a["clip"]), "clean_clip": clean})
    lab = [r for r in rows if r["label"]]
    c = Counter((r["label"], r["label_src"]) for r in rows)
    lines = ["# KE report - every KE alert", "",
             f"KE alerts: {len(rows)} (day {sum(not r['night'] for r in rows)}, night {sum(r['night'] for r in rows)}). "
             f"Labelled: {len(lab)} - user OK {c[('OK','user')]}, user NG {c[('NG','user')]}, "
             f"agent OK {c[('OK','agent')]}, agent NG {c[('NG','agent')]}. Not labelled: {len(rows)-len(lab)}.", "",
             "Per day:", ""]
    for d, n in sorted(Counter(r["time"][:10] for r in rows).items()):
        lines.append(f"- {d}: {n}")
    ob = [r for r in rows if r["object"]]
    lines += ["", f"Checked by the object model (clean clip, since 10 Oct 15:17): {len(ob)}", ""]
    for r in ob:
        lines.append(f"- {r['time']} {r['camera']} ID{r['track']}: in hand = {r['object']}; label {r['label'] or '?'}")
    html = ["<!doctype html><meta charset=utf-8><title>KE report</title>",
            "<style>body{font-family:sans-serif;margin:16px}table{border-collapse:collapse}td,th{border:1px solid #ccc;"
            "padding:3px 6px;font-size:13px;vertical-align:top}.OK{background:#e3f5e3}.NG{background:#fbe3e3}</style>",
            "<h1>KE report - every KE alert</h1>", "<pre>" + "\n".join(lines) + "</pre>",
            "<table><tr><th>picture<th>time<th>camera<th>ID<th>day/night<th>label<th>object in hand<th>clips"]
    for r in sorted(rows, key=lambda r: r["time"], reverse=True):
        clips = f"<a href='file:///{r['clip']}'>alert clip</a>"
        if r["clean_clip"]:
            clips += f"<br><a href='file:///{r['clean_clip']}'>clean clip</a>"
        html.append(f"<tr class='{r['label']}'><td><img src='{r['thumb']}'><td>{r['time']}<td>{r['camera']}<td>{r['track']}"
                    f"<td>{'night' if r['night'] else 'day'}<td>{r['label'] or '?'} {('('+r['label_src']+')') if r['label_src'] else ''}"
                    f"<td>{r['object'] or '-'}<td>{clips}")
    html.append("</table>")
    with open(os.path.join(out, "ke_alerts.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ["OK_NG (fill in)"])
        w.writeheader()
        for r in rows:
            w.writerow(dict(r, **{"OK_NG (fill in)": ""}))
    open(os.path.join(out, "summary.md"), "w", encoding="utf-8").write("\n".join(lines))
    open(os.path.join(out, "report.html"), "w", encoding="utf-8").write("\n".join(html))
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
