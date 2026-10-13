"""Model comparison over EVERY PO and KE alert in the evidence archive (12 Oct 2026,
the user: "check all PO and KE of this project on every model, by their labels; a
summary of which model stops the most NG alerts and which one misses OK ones - each
with ID, snapshot, clip - so I can fix one model; include the night alerts, the
lighting is different at night").

Read-only: nothing live changes. Below-normal priority, 2 threads.

Alerts: D:/poketenashi-data/evidence/<day>/alerts.jsonl (+ clip / poster there).
Labels (user first, then the review agent):
  PO  PO_review_v4/OK_PO_real, NG_PO_false (file names: time + ID), PO_review_v4/
      PO_review_v4.csv, PO_review/{OK_PO_real,NG_PO_false}, PO_review_v4/night_10oct/
      night_alerts.csv, the review agent's D:/poketenashi-data/agent/<day>/review.csv
  KE  PO_review_v4/OK KE, NG KE, ke_review/<day>/review.csv, agent review.csv
Person: the clean candidate snapshots (po_candidates/<day>/alert) when there are, else
frames of the alert clip with the person in the red box (overlay drawn - fine for
scoring, never for training).
PO models (per hip arm, best over the sampled pictures): v1 (mnv3), v4, v6, v7, v8,
dual = min(v8, v1). A model "fires" on an alert when >= 2 pictures have an arm at or
above its threshold (the live rule needs 2 pocket frames).
KE methods (per clip frame of the person): zoom (yolo11m on the raised-hand zoom, the
KE path before 10 Oct: >= 0.55 once or >= 0.40 twice), verify (the person-crop phone
check live since 10 Oct 15:33, >= 0.40), verify+hand (verify AND the phone hand not
stretched out / above the head - the user's idea, 10 Oct), yolov8s zoom.
Night = 19:00-07:00, plus the picture's colour saturation (IR night picture < 25).

Output: <out>/results.csv, <out>/summary.md, <out>/report.html (thumbnails in
<out>/img, links to the clips).

    python tools/agent/model_compare.py <out_dir>
"""
import csv
import datetime as dt
import glob
import json
import os
import re
import shutil
import sys
import time

import cv2
import numpy as np

ROOT = r"C:\Users\DX-ADMIN\Pictures\yolo"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lowprio  # noqa: E402,F401

EV = "D:/poketenashi-data/evidence"
CAND = "D:/poketenashi-data/po_candidates"
PO_MODELS = [("v1", "models/po_arm_mnv3.pt", 0.80), ("v4", "models/po_arm_v4.pt", 0.80),
             ("v6", "models/po_arm_v6.pt", 0.80), ("v7", "models/po_arm_v7.pt", 0.50),
             ("v8", "models/po_arm_v8.pt", 0.50)]
DUAL_THR = 0.35
KE_METHODS = ["zoom_y11m", "zoom_y8s", "verify", "verify_hand"]


# ---------------------------------------------------------------- labels
def _parse_name(name):
    """-> (date 'YYYYMMDD' or None, 'HHMMSS', track id) from a label file name."""
    m = re.search(r"(20\d\d)-(\d\d)-(\d\d)_(\d{6})_\w*?ID(\d+)", name)
    if m:
        return m.group(1) + m.group(2) + m.group(3), m.group(4), int(m.group(5))
    m = re.search(r"(?:^|_)(\d{4})_(\d{6})_[A-Za-z-]+_ID(\d+)", name)
    if m:
        return "2026" + m.group(1), m.group(2), int(m.group(3))
    m = re.search(r"(?:^|_)(\d{6})_(?:cam\d_)?ID(\d+)", name)
    if m:
        return None, m.group(1), int(m.group(2))
    m = re.search(r"^(\d{10})_(?:cam\d_)?ID(\d+)_", name)               # alert clip name (unix time)
    if m:
        t = time.localtime(int(m.group(1)))
        return time.strftime("%Y%m%d", t), time.strftime("%H%M%S", t), int(m.group(2))
    return None


def build_labels():
    L = []          # (rule, date|None, hhmmss, tid, label OK/NG, source)

    def add_dir(rule, d, lab, src):
        for f in glob.glob(os.path.join(ROOT, d, "**", "*"), recursive=True):
            p = _parse_name(os.path.basename(f))
            if p:
                L.append((rule, p[0], p[1], p[2], lab, src))
    add_dir("PO", "PO_review_v4/OK_PO_real", "OK", "user")
    add_dir("PO", "PO_review_v4/NG_PO_false", "NG", "user")
    add_dir("PO", "PO_review/OK_PO_real", "OK", "user")
    add_dir("PO", "PO_review/NG_PO_false", "NG", "user")
    add_dir("KE", "PO_review_v4/OK KE", "OK", "user")
    add_dir("KE", "PO_review_v4/NG KE", "NG", "user")
    p = os.path.join(ROOT, "PO_review_v4", "PO_review_v4.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p, encoding="utf-8")):
            q = _parse_name(r.get("clip", ""))
            if q and r.get("verdict") in ("OK", "NG"):
                L.append(("PO", q[0], q[1], q[2], r["verdict"], "user"))
    p = os.path.join(ROOT, "PO_review_v4", "night_10oct", "night_alerts.csv")
    if os.path.exists(p):
        for r in csv.reader(open(p, encoding="utf-8")):
            if len(r) > 6 and r[6] in ("OK", "NG"):
                L.append(("PO", r[1][:10].replace("-", ""), r[1][11:19].replace(":", ""), int(r[3]), r[6], "user"))
    for p in glob.glob("D:/poketenashi-data/ke_review/*/review.csv"):
        for r in csv.DictReader(open(p, encoding="utf-8")):
            v = (r.get("OK_NG (fill in)") or "").strip()
            if v in ("OK", "NG"):
                L.append(("KE", r["time"][:10].replace("-", ""), r["time"][11:19].replace(":", ""), int(r["track"]), v, "user"))
    for p in glob.glob("D:/poketenashi-data/agent/2*/review.csv"):
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r.get("verdict") in ("OK", "NG") and r.get("rule") in ("PO", "KE"):
                L.append((r["rule"], r["time"][:10].replace("-", ""), r["time"][11:19].replace(":", ""),
                          int(r["track"]), r["verdict"], "agent"))
    return L


def _secs(hhmmss):
    return int(hhmmss[:2]) * 3600 + int(hhmmss[2:4]) * 60 + int(hhmmss[4:6])


def label_for(L, rule, day, hhmmss, tid):
    best = None
    for r, d, t, i, lab, src in L:
        if r != rule or i != tid or (d is not None and d != day):
            continue
        dsec = abs(_secs(t) - _secs(hhmmss))
        if dsec > 180:
            continue
        rank = (0 if src == "user" else 1, dsec)
        if best is None or rank < best[0]:
            best = (rank, lab, src)
    return (best[1], best[2]) if best else ("", "")


# ---------------------------------------------------------------- the person
def red_box(img):
    """The alert's red box (overlay) in a frame -> [x1, y1, x2, y2] or None."""
    m = cv2.inRange(img, (0, 0, 171), (69, 69, 255))
    n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
    cand = [st[i] for i in range(1, n) if st[i][2] > 15 and st[i][3] > 30 and st[i][4] < 0.25 * st[i][2] * st[i][3]]   # outline, not a red machine
    if not cand:
        return None
    x, y, w, h, _ = max(cand, key=lambda s: s[2] * s[3])
    return [x, y, x + w, y + h]


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def clip_people(clip, pose, n=40):
    """[(frame, bbox, keypoints)] of the red-boxed person in up to n frames of the clip."""
    cap = cv2.VideoCapture(clip)
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idx = set(np.linspace(0, max(0, tot - 1), n).astype(int).tolist()) if tot else set()
    out, i = [], -1
    while True:
        ok = cap.grab()
        if not ok:
            break
        i += 1
        if i not in idx:
            continue
        ok, f = cap.retrieve()
        if not ok:
            continue
        if f.shape[1] < 1920:                              # archived clips are 1280 wide; live runs at 1920
            sc = 1920.0 / f.shape[1]
            f = cv2.resize(f, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
        rb = red_box(f)
        if rb is None:
            continue
        r = pose.predict(f, imgsz=640, conf=0.3, verbose=False, device="cpu")[0]
        if r.keypoints is None or len(r.boxes) == 0:
            continue
        bs, ks = r.boxes.xyxy.numpy(), r.keypoints.data.numpy()
        j = int(np.argmax([_iou(b, rb) for b in bs]))
        if _iou(bs[j], rb) >= 0.3:
            out.append((f, bs[j], ks[j]))
    cap.release()
    return out


def person_clip_people(day, cam, tid, hh, pose, n=40):
    """The clean raw clip of this person (persons/<day>/<cam>/HHMMSS_<cam>_ID<tid>.mp4, cut from
    the raw recording - no overlay, full resolution). The person is found by matching
    their snapshot in the first frames, then followed by box overlap."""
    best = None
    for f in glob.glob(f"D:/poketenashi-data/persons/{day}/{cam}/*_{cam}_ID{tid}.mp4"):
        t0 = os.path.basename(f)[:6]
        d = _secs(hh) - _secs(t0)
        if -5 <= d <= 120 and (best is None or d < best[0]):
            best = (d, f)
    if best is None:
        return []
    clip = best[1]
    snap = cv2.imread(clip[:-4] + ".jpg")
    cap = cv2.VideoCapture(clip)
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idx = set(np.linspace(0, max(0, tot - 1), n).astype(int).tolist()) if tot else set()
    out, i, prev = [], -1, None
    while True:
        ok = cap.grab()
        if not ok:
            break
        i += 1
        if i not in idx:
            continue
        ok, f = cap.retrieve()
        if not ok:
            continue
        r = pose.predict(f, imgsz=640, conf=0.3, verbose=False, device="cpu")[0]
        if r.keypoints is None or len(r.boxes) == 0:
            continue
        bs, ks = r.boxes.xyxy.numpy(), r.keypoints.data.numpy()
        if prev is None:
            if snap is None or snap.shape[0] > f.shape[0] or snap.shape[1] > f.shape[1]:
                continue
            res = cv2.matchTemplate(f, snap, cv2.TM_CCOEFF_NORMED)
            _, mv, _, ml = cv2.minMaxLoc(res)
            if mv < 0.5:
                continue
            ref = [ml[0], ml[1], ml[0] + snap.shape[1], ml[1] + snap.shape[0]]
        else:
            ref = prev
        j = int(np.argmax([_iou(b, ref) for b in bs]))
        if _iou(bs[j], ref) >= 0.25:
            prev = list(bs[j])
            out.append((f, bs[j], ks[j]))
    cap.release()
    return out


def snap_people(files, pose):
    out = []
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        r = pose.predict(img, imgsz=320 if max(img.shape[:2]) <= 700 else 960, conf=0.3, verbose=False, device="cpu")[0]
        if r.keypoints is None or len(r.boxes) == 0:
            continue
        bs, ks = r.boxes.xyxy.numpy(), r.keypoints.data.numpy()
        j = int(np.argmax((bs[:, 2] - bs[:, 0]) * (bs[:, 3] - bs[:, 1])))
        out.append((img, bs[j], ks[j]))
    return out


def saturation(img):
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].mean())


# ---------------------------------------------------------------- PO
def po_scores(people, models):
    from poketenashi import po_model
    per = {k: [] for k, _, _ in PO_MODELS}
    per["dual"] = []
    for img, b, kp in people:
        if min(kp[[5, 6, 11, 12], 2]) < 0.3:
            continue
        torso = float(np.linalg.norm(kp[[5, 6], :2].mean(0) - kp[[11, 12], :2].mean(0)))
        if torso < 20:
            continue
        crops = []
        for wi, hi in ((9, 11), (10, 12)):
            W, H = kp[wi, :2], kp[hi, :2]
            if kp[wi, 2] < 0.3 or abs(W[1] - H[1]) > 0.5 * torso or abs(W[0] - H[0]) > 0.6 * torso:
                continue
            c = po_model.arm_crop(img, W, torso)
            if c is not None:
                crops.append(c)
        if not crops:
            for k in per:
                per[k].append(0.0)
            continue
        sc = {k: m.score(crops) for k, m in models.items()}
        for k in models:
            per[k].append(max(sc[k]))
        per["dual"].append(max(min(a, b) for a, b in zip(sc["v8"], sc["v1"])))
    return per


# ---------------------------------------------------------------- KE
def _zoom_phone(det, img, kp, imgsz=256):
    from poketenashi.ke import raised_wrists, zoom_box
    wr, torso = raised_wrists(kp)
    if not wr:
        return 0.0
    h, w = img.shape[:2]
    box = zoom_box(wr, torso, w, h)
    if box is None:
        return 0.0
    x1, y1, x2, y2 = box
    crop = img[y1:y2, x1:x2]
    s = imgsz / float(max(crop.shape[:2]))
    crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    r = det.predict(crop, classes=[67], imgsz=imgsz, conf=0.2, verbose=False, device="cpu")[0]
    best = 0.0
    for (a, b_, c, d), cf in zip(r.boxes.xyxy.cpu().numpy().tolist(), r.boxes.conf.cpu().numpy().tolist()):
        if max(c - a, d - b_) / s > 0.5 * torso:          # zoom_max_long_torso
            continue
        pc = np.array([x1 + (a + c) / 2 / s, y1 + (b_ + d) / 2 / s])
        if min(float(np.linalg.norm(pc - q)) for q in wr) <= 0.6 * torso:
            best = max(best, float(cf))
    return best


def _hand_in_use(kp, wi, torso):
    W = kp[wi, :2]
    head = [kp[i, 1] for i in range(5) if kp[i, 2] >= 0.3]
    sh = [kp[i, 1] for i in (5, 6) if kp[i, 2] >= 0.3]
    top = min(head) if head else (min(sh) - 0.6 * torso if sh else None)
    if top is not None and W[1] < top - 0.15 * torso:
        return False
    s_i, e_i = (5, 7) if wi == 9 else (6, 8)
    if kp[s_i, 2] >= 0.3 and kp[e_i, 2] >= 0.3:
        S, E = kp[s_i, :2], kp[e_i, :2]
        v1, v2 = S - E, W - E
        ang = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / max(1e-6, np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)))
        if np.linalg.norm(W - S) >= 1.05 * torso and ang >= 155:
            return False
    return True


def _verify(det, img, b, kp, min_conf=0.40):
    """-> (phone conf at a wrist if it is the strongest hand object, wrist index) or (0, None)."""
    x1, y1, x2, y2 = b
    H, W = img.shape[:2]
    pw, ph = 0.4 * (x2 - x1), 0.1 * (y2 - y1)
    cx, cy = max(0, int(x1 - pw)), max(0, int(y1 - ph))
    crop = img[cy:min(H, int(y2 + ph)), cx:min(W, int(x2 + pw))]
    if crop.size == 0:
        return 0.0, None
    torso = float(np.linalg.norm(kp[[5, 6], :2].mean(0) - kp[[11, 12], :2].mean(0))) or 0.3 * (y2 - y1)
    wr = [(i, kp[i, :2] - [cx, cy]) for i in (9, 10) if kp[i, 2] >= 0.3]
    if not wr:
        return 0.0, None
    r = det.predict(crop, classes=[63, 65, 67, 73], imgsz=640, conf=0.2, verbose=False, device="cpu")[0]
    best = None
    for bb, c, s in zip(r.boxes.xyxy.numpy(), r.boxes.cls.numpy(), r.boxes.conf.numpy()):
        pc = np.array([(bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2])
        dist, wi = min((float(np.linalg.norm(pc - w)), i) for i, w in wr)
        if dist <= 0.6 * torso and (best is None or s > best[1]):
            best = (int(c), float(s), wi)
    if best is None or best[0] != 67:
        return 0.0, None
    return best[1], best[2]


def ke_scores(people, det11, det8):
    z11, z8, ver, verh = [], [], [], []
    for img, b, kp in people:
        if min(kp[[5, 6, 11, 12], 2]) < 0.3:
            continue
        torso = float(np.linalg.norm(kp[[5, 6], :2].mean(0) - kp[[11, 12], :2].mean(0)))
        z11.append(_zoom_phone(det11, img, kp)); z8.append(_zoom_phone(det8, img, kp))
        cf, wi = _verify(det11, img, b, kp)
        ver.append(cf)
        verh.append(cf if (wi is not None and _hand_in_use(kp, wi, torso)) else 0.0)
    return {"zoom_y11m": z11, "zoom_y8s": z8, "verify": ver, "verify_hand": verh}


def ke_fires(k, v):
    v = np.array(v) if len(v) else np.zeros(0)
    if k.startswith("zoom"):
        return bool((v >= 0.55).any() or (v >= 0.40).sum() >= 2)
    return bool((v >= 0.40).any())


# ---------------------------------------------------------------- main
def main(out):
    import torch
    torch.set_num_threads(2)
    from ultralytics import YOLO
    from poketenashi import po_model
    os.makedirs(os.path.join(out, "img"), exist_ok=True)
    pose = YOLO(os.path.join(ROOT, "yolo11n-pose.pt"))
    models = {k: po_model.PocketModel(os.path.join(ROOT, p)) for k, p, _ in PO_MODELS}
    det11, det8 = YOLO(os.path.join(ROOT, "yolo11m.pt")), YOLO(os.path.join(ROOT, "yolov8s.pt"))
    L = build_labels()
    rows = []
    part = os.path.join(out, "results.jsonl")           # one line per alert, kept across restarts
    done = {}
    if os.path.exists(part):
        for l in open(part, encoding="utf-8"):
            try:
                r = json.loads(l); done[(r["time"], r["camera"], r["track"], r["rule"])] = r
            except ValueError:
                pass
    days = sorted(os.path.basename(d) for d in glob.glob(os.path.join(EV, "2*")))
    seen = set()
    for day in days:
        aj = os.path.join(EV, day, "alerts.jsonl")
        if not os.path.exists(aj):
            continue
        for line in open(aj, encoding="utf-8"):
            try:
                a = json.loads(line)
            except ValueError:
                continue
            rule = a.get("rule")
            if rule not in ("PO", "KE") or not a.get("clip"):
                continue
            key = (a["time"], a.get("camera"), a["track_id"], rule)
            if key in seen:
                continue
            seen.add(key)
            dk = (a["time"], a.get("camera_name") or a.get("camera") or "cam1", int(a["track_id"]), rule)
            if dk in done:
                rows.append(done[dk])
                continue
            import gc
            gc.collect()
            clip = os.path.join(EV, day, a["clip"])
            poster = os.path.join(EV, day, a.get("poster") or "")
            hh = a["time"][11:19].replace(":", "")
            tid = int(a["track_id"])
            cam = a.get("camera") or "cam1"
            lab, src = label_for(L, rule, day, hh, tid)
            people, from_ = [], "clip"
            if rule == "PO":
                snaps = []
                for f in glob.glob(os.path.join(CAND, day, "alert", f"*_{cam}_ID{tid}_P*_*.jpg")):
                    t0 = os.path.basename(f)[:6]
                    if 0 <= _secs(hh) - _secs(t0) <= 120:
                        snaps.append(f)
                if snaps:
                    people, from_ = snap_people(sorted(snaps), pose), "snapshot"
            if not people:
                people = person_clip_people(day, cam, tid, hh, pose)
                from_ = "clean clip" if people else from_
            if not people and os.path.exists(clip):
                people = clip_people(clip, pose)
                from_ = "alert clip"
            hour = int(hh[:2])
            sat = float(np.mean([saturation(p[0]) for p in people])) if people else -1
            night = hour >= 19 or hour < 7
            r = {"rule": rule, "day": day, "time": a["time"], "camera": a.get("camera_name") or cam,
                 "track": tid, "label": lab, "label_src": src, "night": int(night), "ir": int(0 <= sat < 25),
                 "saturation": round(sat, 1), "pictures": len(people), "from": from_,
                 "clip": clip, "poster": poster}
            if rule == "PO":
                sc = po_scores(people, models)
                for k, _, thr in PO_MODELS:
                    v = np.array(sc[k]) if sc[k] else np.zeros(0)
                    r[f"{k}_max"] = round(float(v.max()), 2) if len(v) else 0.0
                    r[f"{k}_fire"] = int((v >= thr).sum() >= (2 if len(v) >= 2 else 1)) if len(v) else 0
                v = np.array(sc["dual"]) if sc["dual"] else np.zeros(0)
                r["dual_max"] = round(float(v.max()), 2) if len(v) else 0.0
                r["dual_fire"] = int((v >= DUAL_THR).sum() >= (2 if len(v) >= 2 else 1)) if len(v) else 0
            else:
                sc = ke_scores(people, det11, det8)
                for k in KE_METHODS:
                    r[f"{k}_max"] = round(max(sc[k]), 2) if sc[k] else 0.0
                    r[f"{k}_fire"] = int(ke_fires(k, sc[k]))
            th = os.path.join(out, "img", os.path.splitext(os.path.basename(poster))[0] + ".jpg")
            if os.path.exists(poster) and not os.path.exists(th):
                im = cv2.imread(poster)
                if im is not None:
                    cv2.imwrite(th, cv2.resize(im, (480, int(480 * im.shape[0] / im.shape[1]))), [cv2.IMWRITE_JPEG_QUALITY, 80])
            r["thumb"] = os.path.relpath(th, out) if os.path.exists(th) else ""
            rows.append(r)
            with open(part, "a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")
            print(f"{rule} {a['time']} ID{tid} {lab or '-'} pics {len(people)}", flush=True)
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(out, "results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    report(rows, out)


def report(rows, out):
    po_m = [k for k, _, _ in PO_MODELS] + ["dual"]
    nopic = sum(1 for r in rows if int(r["pictures"]) == 0)
    lines = ["# Model comparison - every PO and KE alert", "",
             f"Alerts with no clip on disk / person not found (not counted): {nopic}", "",
             f"Made {dt.datetime.now():%Y-%m-%d %H:%M}. Alerts: PO {sum(r['rule']=='PO' for r in rows)}, "
             f"KE {sum(r['rule']=='KE' for r in rows)}. Labelled: PO {sum(r['rule']=='PO' and bool(r['label']) for r in rows)}, "
             f"KE {sum(r['rule']=='KE' and bool(r['label']) for r in rows)} (user + review agent).", ""]
    html = ["<!doctype html><meta charset=utf-8><title>Model comparison</title>",
            "<style>body{font-family:sans-serif;margin:16px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:3px 6px;font-size:13px}"
            ".ok{background:#e3f5e3}.ng{background:#fbe3e3}img{width:220px}.y{color:#0a0;font-weight:bold}.n{color:#c00}</style>",
            "<h1>Model comparison - every PO and KE alert</h1>"]
    for rule, ms in (("PO", po_m), ("KE", KE_METHODS)):
        R = [r for r in rows if r["rule"] == rule]
        for scope, sel in (("all", lambda r: True), ("day", lambda r: not r["night"]), ("night", lambda r: r["night"])):
            S = [r for r in R if sel(r) and r["label"] and int(r["pictures"]) > 0]   # no clip / no person found: not counted
            ok = [r for r in S if r["label"] == "OK"]; ng = [r for r in S if r["label"] == "NG"]
            lines += [f"## {rule} - {scope} (labelled: OK {len(ok)}, NG {len(ng)})", "",
                      "| model | OK caught | OK missed | NG stopped | NG still fired |", "|---|---|---|---|---|"]
            html += [f"<h2>{rule} - {scope} (labelled OK {len(ok)}, NG {len(ng)})</h2><table><tr><th>model<th>OK caught<th>OK missed<th>NG stopped<th>NG still fired"]
            for m in ms:
                c = sum(r[f"{m}_fire"] for r in ok); s = sum(1 - r[f"{m}_fire"] for r in ng)
                lines.append(f"| {m} | {c}/{len(ok)} | {len(ok)-c} | {s}/{len(ng)} | {len(ng)-s} |")
                html.append(f"<tr><td>{m}<td>{c}/{len(ok)}<td>{len(ok)-c}<td>{s}/{len(ng)}<td>{len(ng)-s}")
            lines.append(""); html.append("</table>")
        html.append(f"<h2>{rule} - every alert</h2><table><tr><th>picture<th>time<th>camera<th>ID<th>label<th>night / IR<th>"
                    + "".join(f"<th>{m}" for m in ms) + "<th>clip")
        for r in sorted(R, key=lambda r: r["time"]):
            cls = "ok" if r["label"] == "OK" else "ng" if r["label"] == "NG" else ""
            cells = "".join(f"<td class={'y' if r[f'{m}_fire'] else 'n'}>{'fires' if r[f'{m}_fire'] else 'no'} {r[f'{m}_max']}" for m in ms)
            html.append(f"<tr class={cls}><td><img src='{r['thumb']}'><td>{r['time']}<td>{r['camera']}<td>{r['track']}"
                        f"<td>{r['label'] or '?'} {('('+r['label_src']+')') if r['label_src']=='agent' else ''}"
                        f"<td>{'night' if r['night'] else 'day'}{' IR' if r['ir'] else ''}{cells}"
                        f"<td><a href='file:///{r['clip']}'>{os.path.basename(r['clip'])}</a>")
        html.append("</table>")
    open(os.path.join(out, "summary.md"), "w", encoding="utf-8").write("\n".join(lines))
    open(os.path.join(out, "report.html"), "w", encoding="utf-8").write("\n".join(html))
    print("\n".join(lines))


if __name__ == "__main__":
    if sys.argv[1] == "report":
        rows = []
        for r in csv.DictReader(open(os.path.join(sys.argv[2], "results.csv"), encoding="utf-8")):
            for k, v in r.items():
                if k.endswith("_fire") or k in ("night", "ir", "track", "pictures"):
                    r[k] = int(float(v)) if v != "" else 0
            rows.append(r)
        report(rows, sys.argv[2])
    else:
        main(sys.argv[1])
