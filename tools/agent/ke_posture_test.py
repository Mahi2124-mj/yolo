"""KE by posture - offline test (13 Oct 2026, the user: "test offline" the idea that a phone
too small to detect (ID498, 13:16) can be caught by the 'looking at a phone' posture).

Posture per frame, pose model only (yolo11n-pose on the 1920-wide frame):
  head_down = (nose_y - ears_y) / torso      (ears = mean of the visible ears; > 0 = face down)
  a wrist in FRONT of the chest: between shoulder_y + 0.1 torso and hip_y + 0.1 torso,
       x within the shoulders +- 0.35 torso, elbow bent (angle < 130 deg)
A clip scores the longest unbroken run (seconds) of frames with BOTH, for the person:
  - alert clips (PO_review_v4/OK KE, NG KE, report_20261013/sorted/KE_*): the red-boxed person
  - extra misses: the given clips (target = the red box, or the largest person)
  - everyday people: person clips of one day without a KE alert (every person in the clip,
    tracked by box overlap) -> how often ordinary walkers would trigger it
Output: <out>/results.csv and a summary for run thresholds 1 / 2 / 3 s and head_down 0 / 0.1.

    python tools/agent/ke_posture_test.py <out_dir> [day=20261013] [n_everyday=150]
"""
import csv
import glob
import os
import random
import sys

import cv2
import numpy as np

ROOT = r"C:\Users\DX-ADMIN\Pictures\yolo"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lowprio  # noqa: E402,F401
from ke_phone_data import red_box, iou  # noqa: E402

EXTRA = {"D:/poketenashi-data/clips/1791877616_ID498_PO.mp4": "OK",           # 13 Oct 13:16, phone too small
         "D:/poketenashi-data/persons/20261013/cam1/094813_cam1_ID384.mp4": "OK"}  # 09:48, stopped by the phone check


def angle(a, b, c):
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / n, -1, 1)))) if n > 0 else 180.0


def posture(kp, hd_min):
    if min(kp[[5, 6, 11, 12], 2]) < 0.3:
        return None
    sh, hp = kp[[5, 6], :2].mean(0), kp[[11, 12], :2].mean(0)
    t = float(np.linalg.norm(sh - hp))
    if t < 25:
        return None
    ears = [kp[i, :2] for i in (3, 4) if kp[i, 2] >= 0.3]
    if kp[0, 2] < 0.3 or not ears:
        return None
    hd = (kp[0, 1] - np.mean([e[1] for e in ears])) / t
    xl, xr = min(kp[5, 0], kp[6, 0]) - 0.35 * t, max(kp[5, 0], kp[6, 0]) + 0.35 * t
    front = False
    for s, e, w in ((5, 7, 9), (6, 8, 10)):
        if min(kp[[s, e, w], 2]) < 0.3:
            continue
        W = kp[w, :2]
        if sh[1] + 0.1 * t <= W[1] <= hp[1] + 0.1 * t and xl <= W[0] <= xr and angle(kp[s, :2], kp[e, :2], W) < 130:
            front = True
    return {"hd": hd, "front": front}


def run_len(flags, dt):
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best * dt


def clip_frames(path, step_fps=5.0):
    cap = cv2.VideoCapture(path)
    fps = cap.get(5) or 10.0
    step = max(1, int(round(fps / step_fps)))
    i = -1
    while True:
        if not cap.grab():
            break
        i += 1
        if i % step:
            continue
        ok, f = cap.retrieve()
        if ok:
            yield f, step / fps
    cap.release()


def score_target(path, pose, hd_list):
    """the red-boxed person (alert clip) or the largest person; -> {hd: run seconds}"""
    seq, dt = [], 0.2
    for f, dt in clip_frames(path):
        rb = red_box(f)
        s = 1920.0 / f.shape[1]
        F = cv2.resize(f, None, fx=s, fy=s) if s != 1.0 else f
        rb = [v * s for v in rb] if rb else None
        r = pose.predict(F, imgsz=480, conf=0.3, verbose=False, device="cpu")[0]
        if r.keypoints is None or len(r.boxes) == 0:
            seq.append(None); continue
        bs, ks = r.boxes.xyxy.numpy(), r.keypoints.data.numpy()
        if rb is not None:
            j = int(np.argmax([iou(b, rb) for b in bs]))
            if iou(bs[j], rb) < 0.3:
                seq.append(None); continue
        else:
            j = int(np.argmax((bs[:, 2] - bs[:, 0]) * (bs[:, 3] - bs[:, 1])))
        seq.append(posture(ks[j], 0))
    return {h: run_len([bool(p and p["front"] and p["hd"] > h) for p in seq], dt) for h in hd_list}, len(seq)


def score_everyone(path, pose, hd_list):
    tracks, dt = [], 0.2                                   # [last box, [postures]]
    for f, dt in clip_frames(path):
        r = pose.predict(f, imgsz=480, conf=0.3, verbose=False, device="cpu")[0]
        if r.keypoints is None or len(r.boxes) == 0:
            continue
        for b, k in zip(r.boxes.xyxy.numpy(), r.keypoints.data.numpy()):
            best = max(tracks, key=lambda tr: iou(tr[0], b), default=None)
            if best is None or iou(best[0], b) < 0.3:
                best = [b, []]; tracks.append(best)
            best[0] = b; best[1].append(posture(k, 0))
    return {h: max([run_len([bool(p and p["front"] and p["hd"] > h) for p in tr[1]], dt) for tr in tracks], default=0.0)
            for h in hd_list}, len(tracks)


def main(out, day="20261013", n_every=150):
    from ultralytics import YOLO
    pose = YOLO(os.path.join(ROOT, "yolo11n-pose.pt"))
    os.makedirs(out, exist_ok=True)
    HD = [0.0, 0.1]
    items = []
    for lab, pat in (("OK", "PO_review_v4/OK KE/*.mp4"), ("NG", "PO_review_v4/NG KE/*.mp4"),
                     ("OK", "report_20261013/sorted/KE_OK/*.mp4"), ("NG", "report_20261013/sorted/KE_NG/*.mp4")):
        items += [("alert", lab, f) for f in glob.glob(os.path.join(ROOT, pat))]
    seen, uniq = set(), []
    for it in items:
        b = os.path.basename(it[2])
        if b not in seen:
            seen.add(b); uniq.append(it)
    uniq += [("miss", lab, p) for p, lab in EXTRA.items() if os.path.exists(p)]
    pc = os.path.join("D:/poketenashi-data/persons", day, "persons.csv")
    cc = os.path.join("D:/poketenashi-data/persons", day, "clips.csv")
    clips = {r[0]: r[1] for r in csv.reader(open(cc, encoding="utf-8")) if len(r) > 1 and r[1]} if os.path.exists(cc) else {}
    every = [r for r in csv.DictReader(open(pc, encoding="utf-8")) if "KE" not in (r.get("alerts") or "") and clips.get(r["snapshot"])]
    random.seed(0)
    for r in random.sample(every, min(n_every, len(every))):
        uniq.append(("everyday", "NG?", os.path.join("D:/poketenashi-data/persons", day, clips[r["snapshot"]])))
    rp = os.path.join(out, "results.csv")
    done = {r["file"] for r in csv.DictReader(open(rp, encoding="utf-8"))} if os.path.exists(rp) else set()
    f = open(rp, "a", newline="", encoding="utf-8"); w = csv.writer(f)
    if not done:
        w.writerow(["kind", "label", "file", "frames_or_tracks"] + [f"run_hd{h}" for h in HD])
    for kind, lab, p in uniq:
        if p in done:
            continue
        try:
            sc, n = (score_everyone if kind == "everyday" else score_target)(p, pose, HD)
        except Exception as e:
            print("skip", p, e, flush=True); continue
        w.writerow([kind, lab, p, n] + [round(sc[h], 1) for h in HD]); f.flush()
        print(kind, lab, os.path.basename(p), {h: round(v, 1) for h, v in sc.items()}, flush=True)
    f.close()
    summary(rp)


def summary(rp):
    rows = list(csv.DictReader(open(rp, encoding="utf-8")))
    for h in ("0.0", "0.1"):
        for need in (1.0, 2.0, 3.0):
            g = lambda kind, lab: [r for r in rows if r["kind"] == kind and r["label"] == lab]
            fire = lambda rs: sum(float(r[f"run_hd{h}"]) >= need for r in rs)
            ok, ng, miss, ev = g("alert", "OK"), g("alert", "NG"), g("miss", "OK"), g("everyday", "NG?")
            print(f"head_down>{h} run>={need}s: OK alerts {fire(ok)}/{len(ok)} | NG alerts {fire(ng)}/{len(ng)} | "
                  f"missed KE {fire(miss)}/{len(miss)} | everyday clips firing {fire(ev)}/{len(ev)}")


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        summary(sys.argv[2])
    else:
        main(sys.argv[1], *(sys.argv[2:3] or []), *([int(sys.argv[3])] if len(sys.argv) > 3 else []))
