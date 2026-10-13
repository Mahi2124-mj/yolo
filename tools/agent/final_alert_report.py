"""Final report of every PO / KE alert in the evidence archive (12 Oct 2026, the user:
"run PO and KE on every evidence clip, report which is right and which NG and on what
basis you judged it"). Joins:
  - the user's own labels (OK/NG folders, csvs)                      -> basis "user label"
  - the visual review of the zoom sheets (verdict_batch_*.csv, then recheck_verdict_*.csv
    for the ones first marked UNSURE)                                 -> basis "seen: <reason>"
  - the model comparison (model_compare results.jsonl)                -> each model's decision
Output in PO_review_v4/alert_review_20261012: final.csv, final_summary.md, final_report.html
"""
import csv
import glob
import json
import os
import sys
from collections import Counter

ROOT = r"C:\Users\DX-ADMIN\Pictures\yolo"
D = os.path.join(ROOT, "PO_review_v4", "alert_review_20261012")
MC = os.path.join(ROOT, "PO_review_v4", "model_compare_20261012", "results.jsonl")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PO_M = ["v1", "v4", "v6", "v7", "v8", "dual"]
KE_M = ["zoom_y11m", "zoom_y8s", "verify", "verify_hand"]


def key_of_sheet(p):
    b = os.path.basename(p)[:-4]                         # 20261010_082425_cam1_ID8441
    day, hh, cam, tid = b.split("_")
    return os.path.basename(os.path.dirname(p)), day, hh, cam, int(tid[2:])


def main():
    from model_compare import build_labels, label_for
    L = build_labels()
    seen = {}
    for pat in ("verdict_batch_*.csv", "recheck_verdict_*.csv"):          # recheck overrides
        for f in sorted(glob.glob(os.path.join(D, pat))):
            for r in csv.DictReader(open(f, encoding="utf-8")):
                v = (r.get("verdict") or "").strip()
                k = key_of_sheet(r["sheet"].strip())
                if v and (pat.startswith("verdict") or v != "UNSURE" or k not in seen):
                    seen[k] = (v, (r.get("reason") or "").strip())
    mc = {}
    for l in open(MC, encoding="utf-8"):
        try:
            r = json.loads(l)
        except ValueError:
            continue
        mc[(r["rule"], r["time"][:10].replace("-", ""), r["time"][11:19].replace(":", ""), int(r["track"]))] = r
    rows = []
    for r in csv.DictReader(open(os.path.join(D, "v2", "index.csv"), encoding="utf-8")):
        if r["rule"] == "rule":
            continue
        day, hh, tid = r["time"][:10].replace("-", ""), r["time"][11:19].replace(":", ""), int(r["track"])
        cam = os.path.basename(r["sheet"]).split("_")[2]
        lab, src = label_for(L, r["rule"], day, hh, tid)
        if src == "user":
            verdict, basis = lab, "user label"
        else:
            v = seen.get((r["rule"], day, hh, cam, tid))
            verdict, basis = (v[0], "seen: " + v[1]) if v else ("UNSURE", "not reviewed")
        m = mc.get((r["rule"], day, hh, tid), {})
        hour = int(hh[:2])
        row = {"rule": r["rule"], "time": r["time"], "camera": r["camera"], "track": tid,
               "night": "night" if (hour >= 19 or hour < 7) else "day", "verdict": verdict, "basis": basis,
               "sheet": os.path.join("v2", r["sheet"]), "clip": r["clip"], "pictures": m.get("pictures", 0)}
        for k in (PO_M if r["rule"] == "PO" else KE_M):
            row[k] = ("fires" if m.get(f"{k}_fire") else "no") if m else "-"
            row[k + "_score"] = m.get(f"{k}_max", "")
        rows.append(row)

    lines = ["# Every PO / KE alert - right or wrong, and why", ""]
    for rule, ms in (("PO", PO_M), ("KE", KE_M)):
        R = [r for r in rows if r["rule"] == rule]
        c = Counter((r["verdict"], "user" if r["basis"] == "user label" else "seen") for r in R)
        lines += [f"## {rule}: {len(R)} alerts", "",
                  f"- OK (right): {c[('OK','user')]+c[('OK','seen')]}  (user label {c[('OK','user')]}, seen {c[('OK','seen')]})",
                  f"- NG (wrong): {c[('NG','user')]+c[('NG','seen')]}  (user label {c[('NG','user')]}, seen {c[('NG','seen')]})",
                  f"- UNSURE (picture not clear enough): {c[('UNSURE','seen')]}", ""]
        for scope in ("all", "day", "night"):
            S = [r for r in R if r["verdict"] in ("OK", "NG") and r[ms[0]] != "-" and int(r["pictures"] or 0) > 0
                 and (scope == "all" or r["night"] == scope)]
            ok = [r for r in S if r["verdict"] == "OK"]; ng = [r for r in S if r["verdict"] == "NG"]
            lines += [f"### {rule} models - {scope} (judged with pictures: OK {len(ok)}, NG {len(ng)})", "",
                      "| model | OK caught | OK missed | NG stopped | NG still fired |", "|---|---|---|---|---|"]
            for m in ms:
                cc = sum(r[m] == "fires" for r in ok); ss = sum(r[m] == "no" for r in ng)
                lines.append(f"| {m} | {cc}/{len(ok)} | {len(ok)-cc} | {ss}/{len(ng)} | {len(ng)-ss} |")
            lines.append("")
        reasons = Counter()
        for r in R:
            if r["verdict"] == "NG" and r["basis"].startswith("seen: "):
                reasons[r["basis"][6:].split(",")[0].strip().strip('"').lower()[:60]] += 1
        lines += [f"### {rule} - most common reasons for NG (seen)", ""] + [f"- {n} x {t}" for t, n in reasons.most_common(12)] + [""]

    with open(os.path.join(D, "final.csv"), "w", newline="", encoding="utf-8") as f:
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    open(os.path.join(D, "final_summary.md"), "w", encoding="utf-8").write("\n".join(lines))
    html = ["<!doctype html><meta charset=utf-8><title>Alert review</title>",
            "<style>body{font-family:sans-serif;margin:16px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:3px 6px;"
            "font-size:13px;vertical-align:top}.OK{background:#e3f5e3}.NG{background:#fbe3e3}.UNSURE{background:#f4f4f4}"
            "img{max-width:520px}.f{color:#c00;font-weight:bold}.n{color:#080}</style>",
            "<h1>Every PO / KE alert - right or wrong, and why</h1><pre>" + "\n".join(lines) + "</pre>"]
    for rule, ms in (("PO", PO_M), ("KE", KE_M)):
        html.append(f"<h2>{rule} - every alert</h2><table><tr><th>person (from the alert clip)<th>time<th>camera<th>ID<th>day/night"
                    "<th>verdict<th>basis" + "".join(f"<th>{m}" for m in ms) + "<th>clip")
        for r in sorted((r for r in rows if r["rule"] == rule), key=lambda r: r["time"], reverse=True):
            cells = "".join(f"<td class={'f' if r[m]=='fires' else 'n'}>{r[m]} {r.get(m+'_score','')}" for m in ms)
            html.append(f"<tr class={r['verdict']}><td><img src='{r['sheet']}'><td>{r['time']}<td>{r['camera']}<td>{r['track']}"
                        f"<td>{r['night']}<td><b>{r['verdict']}</b><td>{r['basis']}{cells}"
                        f"<td><a href='file:///{r['clip']}'>{os.path.basename(r['clip'])}</a>")
        html.append("</table>")
    open(os.path.join(D, "final_report.html"), "w", encoding="utf-8").write("\n".join(html))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
