"""KE phone checker - fit and test (13 Oct 2026, the user: "train the model on the KE OK/NG in
report_20261013/sorted; these cases must not be missed now - alert on OK, ignore NG").

Data: ke_phone_data.py (every "cell phone" candidate >= 0.15 at a wrist of the alert person,
labelled with the alert's OK/NG). Decision per alert, as it would run live:
    fire = any candidate with phone conf >= DET_MIN, size <= 0.45 torso and checker P >= thr
Compared with the live rule today (phone conf >= 0.40, size <= 0.45 torso, no checker).

  ke_phone_fit.py <data_dir> <out.pt>
    1. 5-fold cross-validation split BY ALERT (every alert scored by a model that never saw it)
    2. picks thr / DET_MIN: all sorted OK caught and all sorted NG stopped in CV if possible,
       else the most of them, then the fewest errors overall
    3. trains on everything and saves <out.pt> + <out>.json (thr, DET_MIN, per-alert results)
"""
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ke_phone_train as T  # noqa: E402

MAX_LONG = 0.45


def is_sorted(a):
    return a.startswith(("blk_", "1791"))         # report_20261013/sorted (evidence names / stopped KE)


def decide(rows, p, thr, dmin):
    A = {}
    for r, pp in zip(rows, p):
        a = A.setdefault(r["alert"], {"label": r["label"], "live": False, "new": False, "best": 0.0})
        if r["long_torso"] > MAX_LONG:
            continue
        if r["conf"] >= 0.40:
            a["live"] = True
        if r["conf"] >= dmin:
            a["best"] = max(a["best"], float(pp))
            if pp >= thr:
                a["new"] = True
    return A


def score(A, key, only_sorted=False):
    v = [(k, a) for k, a in A.items() if not only_sorted or is_sorted(k)]
    ok = [a for _, a in v if a["label"] == "OK"]; ng = [a for _, a in v if a["label"] == "NG"]
    return sum(a[key] for a in ok), len(ok), sum(a[key] for a in ng), len(ng)


def main(d, out):
    torch.manual_seed(0)
    rows, ims = T.load(d)
    y = np.array([r["label"] == "OK" for r in rows], np.float32)
    alerts = sorted({r["alert"] for r in rows}); random.seed(1); random.shuffle(alerts)
    fold = {a: i % 5 for i, a in enumerate(alerts)}
    P = np.zeros(len(rows))
    for f in range(5):
        te = [i for i, r in enumerate(rows) if fold[r["alert"]] == f]
        tr = [i for i, r in enumerate(rows) if fold[r["alert"]] != f]
        P[te] = T.predict(T.train(ims, y, tr), ims, te)
        print("fold", f + 1, "done", flush=True)
    best = None
    for dmin in (0.15, 0.20, 0.25, 0.30):
        for thr in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
            A = decide(rows, P, thr, dmin)
            so, son, sn, snn = score(A, "new", True)
            ao, aon, an, ann = score(A, "new")
            key = ((son - so) + sn, (aon - ao) + an, -thr)          # sorted errors, then all errors
            if best is None or key < best[0]:
                best = (key, thr, dmin)
    _, thr, dmin = best
    A = decide(rows, P, thr, dmin)
    res = {"thr": thr, "det_min": dmin, "alerts": len(A), "candidates": len(rows)}
    for nm, key in (("live_today", "live"), ("new_checker_cv", "new")):
        res[nm] = {"all": dict(zip(("ok_caught", "ok", "ng_fired", "ng"), score(A, key))),
                   "sorted": dict(zip(("ok_caught", "ok", "ng_fired", "ng"), score(A, key, True)))}
    res["sorted_cases_cv"] = {k: {"label": a["label"], "live_today": a["live"], "new": a["new"], "best_p": round(a["best"], 2)}
                              for k, a in sorted(A.items()) if is_sorted(k)}
    print(json.dumps({k: v for k, v in res.items() if k != "sorted_cases_cv"}, indent=1), flush=True)
    net = T.train(ims, y, list(range(len(rows))))
    torch.save(net.state_dict(), out)
    Pf = T.predict(net, ims, list(range(len(rows))))
    Af = decide(rows, Pf, thr, dmin)
    res["sorted_cases_final_model"] = {k: {"label": a["label"], "new": a["new"], "best_p": round(a["best"], 2)}
                                       for k, a in sorted(Af.items()) if is_sorted(k)}
    res["final_model_all"] = dict(zip(("ok_caught", "ok", "ng_fired", "ng"), score(Af, "new")))
    json.dump(res, open(out[:-3] + ".json", "w"), indent=1)
    for k, v in res["sorted_cases_cv"].items():
        print(k[:45], v, "| final:", res["sorted_cases_final_model"][k], flush=True)
    print("final model, all alerts:", res["final_model_all"], "saved", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
