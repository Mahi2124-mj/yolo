"""v7 on the user's OK / NG PO folders (13 Oct 2026, the user: "run the v7 model on OK/NG and
give me snapshots: PO OK / NG caught right, not caught, unclear").

Per picture (or 6 frames of a clip), as the live arm model sees it: pose -> every wrist at the
hip (|dy| <= 0.5 torso, |dx| <= 0.6 torso) -> po_model.arm_crop -> v7 P(pocket); the picture's
score = the highest arm. Fires at THR (live: 0.50). No masters, no other guard.
Sorted copies (each with the arms ringed and their scores written on) in <out>/:
  OK_pakda (OK, v7 fires)  OK_miss (OK, v7 silent)  NG_sahi_roka (NG, silent)
  NG_galat_PO (NG, fires)  UNCLEAR (no person / no arm at the hip - v7 cannot judge)
plus results.csv and index.html.

    python tools/agent/po_v7_folder_check.py <out_dir>
"""
import base64
import csv
import glob
import html
import os
import shutil
import sys

import cv2
import numpy as np

ROOT = r"C:\Users\DX-ADMIN\Pictures\yolo"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lowprio  # noqa: E402,F401

THR = 0.50
SETS = [("OK", "PO_review_v4/OK_PO_real_crop"), ("NG", "PO_review_v4/NG_PO_false")]
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".mp4")


def pictures(f):
    if f.lower().endswith(".mp4"):
        cap = cv2.VideoCapture(f); tot = int(cap.get(7) or 0)
        for fi in np.linspace(0, max(0, tot - 1), 6).astype(int):
            cap.set(1, int(fi)); ok, fr = cap.read()
            if ok:
                yield fr
        cap.release()
    else:
        im = cv2.imread(f)
        if im is not None:
            yield im


def judge(f, pose, model):
    """-> (best P or None, annotated picture)"""
    from poketenashi import po_model
    best, shown = None, None
    for img in pictures(f):
        r = pose.predict(img, imgsz=320 if max(img.shape[:2]) <= 700 else 960, conf=0.3, verbose=False, device="cpu")[0]
        if r.keypoints is None or len(r.boxes) == 0:
            continue
        b = r.boxes.xyxy.numpy()
        kp = r.keypoints.data.numpy()[int(np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))]
        if min(kp[[5, 6, 11, 12], 2]) < 0.3:
            continue
        torso = float(np.linalg.norm(kp[[5, 6], :2].mean(0) - kp[[11, 12], :2].mean(0)))
        if torso < 25:
            continue
        arms = []
        for wi, hi in ((9, 11), (10, 12)):
            W, H = kp[wi, :2], kp[hi, :2]
            if kp[wi, 2] < .3 or abs(W[1] - H[1]) > .5 * torso or abs(W[0] - H[0]) > .6 * torso:
                continue
            c = po_model.arm_crop(img, W, torso)
            if c is not None:
                arms.append((W, float(model.score([c])[0])))
        if not arms:
            continue
        p = max(a[1] for a in arms)
        if best is None or p > best:
            best, shown = p, img.copy()
            for W, s in arms:
                col = (0, 0, 255) if s >= THR else (0, 200, 0)
                cv2.circle(shown, (int(W[0]), int(W[1])), max(10, int(0.25 * torso)), col, 2)
                cv2.putText(shown, f"{s:.2f}", (int(W[0]) + 6, int(W[1]) - 6), 0, 0.5, col, 2)
    if shown is None:
        shown = next(pictures(f), None)
    return best, shown


def main(out):
    import torch
    torch.set_num_threads(2)
    from ultralytics import YOLO
    from poketenashi import po_model
    pose = YOLO(os.path.join(ROOT, "yolo11n-pose.pt"))
    model = po_model.PocketModel(os.path.join(ROOT, "models", "po_arm_v7.pt"))
    groups = ["OK_pakda", "OK_miss", "NG_sahi_roka", "NG_galat_PO", "UNCLEAR"]
    for g in groups:
        os.makedirs(os.path.join(out, g), exist_ok=True)
    rows = []
    for lab, d in SETS:
        for f in sorted(glob.glob(os.path.join(ROOT, d, "*"))):
            if not f.lower().endswith(EXTS):
                continue
            p, img = judge(f, pose, model)
            if p is None:
                g = "UNCLEAR"
            elif lab == "OK":
                g = "OK_pakda" if p >= THR else "OK_miss"
            else:
                g = "NG_galat_PO" if p >= THR else "NG_sahi_roka"
            name = os.path.splitext(os.path.basename(f))[0] + ".jpg"
            if img is not None:
                cv2.imwrite(os.path.join(out, g, f"{lab}_{name}"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            rows.append({"label": lab, "file": os.path.basename(f), "v7_p": "" if p is None else round(p, 2), "group": g,
                         "snapshot": f"{g}/{lab}_{name}"})
            print(lab, g, os.path.basename(f), rows[-1]["v7_p"], flush=True)
    with open(os.path.join(out, "results.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    cnt = {g: sum(r["group"] == g for r in rows) for g in groups}
    h = ["<!doctype html><meta charset=utf-8><title>v7 on OK / NG PO</title><style>body{font-family:sans-serif;margin:16px}"
         ".c{display:inline-block;width:190px;margin:4px;vertical-align:top;font-size:12px;border:1px solid #ccc;padding:3px}"
         ".c img{max-width:184px;max-height:260px}</style>",
         f"<h2>v7 (threshold {THR}) on the OK / NG PO folders</h2><p>Circle = a hand at the hip, number = v7 score "
         "(red = counts as pocket). " + " &middot; ".join(f"{g}: <b>{n}</b>" for g, n in cnt.items()) + "</p>"]
    for g in ["OK_miss", "NG_galat_PO", "UNCLEAR", "OK_pakda", "NG_sahi_roka"]:
        rs = [r for r in rows if r["group"] == g]
        h.append(f"<h3>{g} ({len(rs)})</h3>")
        for r in rs:
            h.append(f"<div class=c><a href='{html.escape(r['snapshot'])}'><img loading=lazy src='{html.escape(r['snapshot'])}'></a>"
                     f"<br>{html.escape(r['file'][:40])}<br>v7 {r['v7_p'] if r['v7_p'] != '' else '-'}</div>")
    open(os.path.join(out, "index.html"), "w", encoding="utf-8").write("\n".join(h))
    print(cnt)


if __name__ == "__main__":
    main(sys.argv[1])
