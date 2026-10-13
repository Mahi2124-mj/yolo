"""v7 + elbow-out (akimbo) veto - offline test (13 Oct 2026, the user: "test the elbow
(akimbo) idea offline"; v7 calls a hand ON the hip a pocket).

Per picture of PO_review_v4/OK_PO_train (OK PO) and NG_PO_train (NG PO): pose, every wrist at
the hip -> v7 P(pocket) on its arm crop, and
  elbow_out = (|elbow_x - mid_shoulders_x| - |shoulder_x - mid_shoulders_x|) / torso
  (> 0 = the elbow sticks out sideways past its own shoulder, as on a hand on the hip)
An arm counts as pocket when P >= 0.50 AND elbow_out <= E; the picture fires if any arm does.
Prints OK caught / NG fired for E = off, 0.30 ... 0.00, all pictures and the user's flagged
cases (_v7_missed_these.txt / _v7_false_PO_these.txt), and writes <out>/arms.csv.

    python tools/agent/po_akimbo_test.py <out_dir>
"""
import csv
import glob
import os
import sys

import cv2
import numpy as np

ROOT = r"C:\Users\DX-ADMIN\Pictures\yolo"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lowprio  # noqa: E402,F401

THR = 0.50
SETS = [("OK", "PO_review_v4/OK_PO_train"), ("NG", "PO_review_v4/NG_PO_train")]


def main(out):
    import torch
    torch.set_num_threads(2)
    from ultralytics import YOLO
    from poketenashi import po_model
    pose = YOLO(os.path.join(ROOT, "yolo11n-pose.pt"))
    v7 = po_model.PocketModel(os.path.join(ROOT, "models", "po_arm_v7.pt"))
    flag = set(open(os.path.join(ROOT, "PO_review_v4/OK_PO_train/_v7_missed_these.txt")).read().split()) | \
        set(open(os.path.join(ROOT, "PO_review_v4/NG_PO_train/_v7_false_PO_these.txt")).read().split())
    os.makedirs(out, exist_ok=True)
    pics = []                                    # (label, file, flagged, [(P, elbow_out)])
    for lab, d in SETS:
        for f in sorted(glob.glob(os.path.join(ROOT, d, "*.jpg"))):
            img = cv2.imread(f)
            if img is None:
                continue
            r = pose.predict(img, imgsz=320 if max(img.shape[:2]) <= 700 else 960, conf=0.3, verbose=False, device="cpu")[0]
            arms = []
            if r.keypoints is not None and len(r.boxes):
                b = r.boxes.xyxy.numpy()
                kp = r.keypoints.data.numpy()[int(np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))]
                if min(kp[[5, 6, 11, 12], 2]) >= 0.3:
                    torso = float(np.linalg.norm(kp[[5, 6], :2].mean(0) - kp[[11, 12], :2].mean(0)))
                    cx = kp[[5, 6], 0].mean()
                    for s, e, wi, hi in ((5, 7, 9, 11), (6, 8, 10, 12)):
                        W, H = kp[wi, :2], kp[hi, :2]
                        if torso < 25 or kp[wi, 2] < .3 or abs(W[1] - H[1]) > .5 * torso or abs(W[0] - H[0]) > .6 * torso:
                            continue
                        c = po_model.arm_crop(img, W, torso)
                        if c is None:
                            continue
                        eo = (abs(kp[e, 0] - cx) - abs(kp[s, 0] - cx)) / torso if kp[e, 2] >= .3 else None
                        arms.append((float(v7.score([c])[0]), eo))
            pics.append((lab, os.path.basename(f), os.path.basename(f) in flag, arms))
            print(lab, os.path.basename(f), [(round(p, 2), None if e is None else round(e, 2)) for p, e in arms], flush=True)
    with open(os.path.join(out, "arms.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["label", "file", "flagged", "v7_p", "elbow_out"])
        for lab, fn, fl, arms in pics:
            for p, e in arms:
                w.writerow([lab, fn, int(fl), round(p, 3), "" if e is None else round(e, 3)])
    report(pics)


def fires(arms, E):
    return any(p >= THR and (E is None or e is None or e <= E) for p, e in arms)


def report(pics):
    judged = [x for x in pics if x[3]]
    print(f"\npictures with an arm at the hip: {len(judged)} of {len(pics)}")
    for E in (None, 0.30, 0.20, 0.15, 0.10, 0.05, 0.00):
        row = []
        for name, sel in (("all", lambda x: True), ("flagged", lambda x: x[2])):
            ok = [x for x in judged if x[0] == "OK" and sel(x)]; ng = [x for x in judged if x[0] == "NG" and sel(x)]
            row.append(f"{name}: OK caught {sum(fires(x[3], E) for x in ok)}/{len(ok)}, NG fired {sum(fires(x[3], E) for x in ng)}/{len(ng)}")
        print(f"elbow_out max {'off' if E is None else E}: " + " | ".join(row))


if __name__ == "__main__":
    main(sys.argv[1])
