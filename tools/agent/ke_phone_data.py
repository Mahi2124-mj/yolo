"""Training data for the KE phone checker (12 Oct 2026, the user: "you have NG samples now -
train the model; use object detection well; no KE for a cap or any object other than a
phone").

Every labelled KE alert clip (OK: PO_review_v4/OK KE + sorted/KE_OK, NG: PO_review_v4/NG KE):
30 frames, the alert overlay (pure red / green / blue / cyan lines and text) painted out,
scaled to 1920, the red-boxed person found (before painting), and yolo11m run on the
person crop. Every "cell phone" candidate (conf >= 0.15) within 0.6 torso of a wrist is
cut (box x2.2, 112 px) and labelled with the alert's label (OK -> phone, NG -> not phone).
Output: <out>/img/*.jpg + <out>/index.jsonl (alert key for a group split).

    python tools/agent/ke_phone_data.py <out_dir>
"""
import glob
import json
import os
import re
import sys

import cv2
import numpy as np

ROOT = r"C:\Users\DX-ADMIN\Pictures\yolo"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lowprio  # noqa: E402,F401


def overlay_mask(img):
    """pure saturated overlay colours: red / green / blue / cyan boxes, lines, text"""
    b, g, r = [img[..., i].astype(np.int16) for i in range(3)]
    red = (r > 170) & (g < 70) & (b < 70)
    grn = (g > 170) & (r < 70) & (b < 90)
    blu = (b > 170) & (r < 70) & (g < 90)
    cya = (g > 170) & (b > 170) & (r < 80)
    m = (red | grn | blu | cya).astype(np.uint8) * 255
    return cv2.dilate(m, np.ones((3, 3), np.uint8))


def red_box(img):
    m = cv2.inRange(img, (0, 0, 171), (69, 69, 255)); n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
    c = [st[i] for i in range(1, n) if st[i][2] > 12 and st[i][3] > 25 and st[i][4] < 0.25 * st[i][2] * st[i][3]]
    if not c:
        return None
    x, y, w, h, _ = max(c, key=lambda s: s[2] * s[3]); return [x, y, x + w, y + h]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0


def alert_key(name):
    m = re.search(r"(\d{4,8})_(\d{6})_[A-Za-z-]+_ID(\d+)", name)
    if not m:
        return None
    d = m.group(1)[-4:]
    return f"{d}_{m.group(2)}_{m.group(3)}"


SORTED = "report_20261013/sorted"      # 13 Oct (user): "train on the KE OK/NG in sorted - these cases must not be missed"


def clips():
    out = {}
    for lab, pat in (("OK", SORTED + "/KE_OK/*.mp4"), ("NG", SORTED + "/KE_NG/*.mp4"),     # newest labels first
                     ("OK", "PO_review_v4/OK KE/*.mp4"), ("OK", "PO_review_v4/alert_review_20261012/sorted/KE_OK/*.mp4"),
                     ("NG", "PO_review_v4/NG KE/*.mp4")):
        for f in glob.glob(os.path.join(ROOT, pat)):
            b = os.path.basename(f)
            k = alert_key(b) or (b[:-4] if re.match(r"\d{10}_(cam\d+_)?ID\d+_KE", b) else None)   # evidence name: <epoch>_ID<n>_KE
            if k and k not in out:
                out[k] = (lab, f)
    # KE stopped by the phone check (person crop, no overlay), sorted by the user into
    # KE_OK (a real KE the check stopped) / KE_NG (rightly stopped)
    for lab, sub in (("OK", "KE_OK"), ("NG", "KE_NG")):
        for f in glob.glob(os.path.join(ROOT, SORTED, sub, "blocked_*.jpg")):
            out["blk_" + os.path.basename(f)[:-4]] = (lab, f)
    return out


def frames(f, n=30):
    """30 frames of a clip, or the one picture of a stopped KE"""
    if f.lower().endswith(".jpg"):
        im = cv2.imread(f)
        if im is not None:
            yield im
        return
    cap = cv2.VideoCapture(f); tot = int(cap.get(7) or 0)
    for fi in np.linspace(0, max(0, tot - 1), n).astype(int):
        cap.set(1, int(fi)); ok, fr = cap.read()
        if ok:
            yield fr
    cap.release()


def free_gb():
    import ctypes

    class M(ctypes.Structure):
        _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in
                    ("tp", "ap", "tpf", "apf", "tv", "av", "aev")]
    m = M(); m.l = ctypes.sizeof(M); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.apf / 2 ** 30                           # free commit (page file included)


def main(out):
    import torch
    torch.set_num_threads(2)
    from ultralytics import YOLO
    det, pose = YOLO(os.path.join(ROOT, "yolo11m.pt")), YOLO(os.path.join(ROOT, "yolo11n-pose.pt"))
    os.makedirs(os.path.join(out, "img"), exist_ok=True)
    ip = os.path.join(out, "index.jsonl")
    done, n = set(), 0                           # resume after a crash (the PC runs out of memory)
    if os.path.exists(ip):
        for l in open(ip, encoding="utf-8"):
            try:
                r = json.loads(l); done.add(r["alert"]); n = max(n, int(r["id"][1:]) + 1)
            except (ValueError, KeyError):
                pass
    dp = os.path.join(out, "done_alerts.txt")
    if os.path.exists(dp):
        done |= set(open(dp, encoding="utf-8").read().split())
    idx = open(ip, "a", encoding="utf-8")
    C = clips()
    print(len(C), "labelled KE clips", sum(v[0] == "OK" for v in C.values()), "OK", flush=True)
    for k, (lab, f) in sorted(C.items()):
        if k in done:
            continue
        import gc
        import time
        gc.collect()
        while free_gb() < 1.5:                       # never squeeze the live monitor
            print("waiting: free commit %.2f GB" % free_gb(), flush=True); time.sleep(120)
        got = 0
        for fr in frames(f):
            rb = red_box(fr)
            m = overlay_mask(fr)
            if m.any():
                fr = cv2.inpaint(fr, m, 3, cv2.INPAINT_TELEA)
            s = 1920.0 / fr.shape[1] if fr.shape[1] < 1920 else 1.0
            if s != 1.0:
                fr = cv2.resize(fr, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
                rb = [v * s for v in rb] if rb else None
            r = pose.predict(fr, imgsz=640, conf=0.3, verbose=False, device="cpu")[0]
            if r.keypoints is None or len(r.boxes) == 0:
                continue
            ppl = list(zip(r.boxes.xyxy.numpy(), r.keypoints.data.numpy()))
            if rb is not None:                         # alert clip: only the red-boxed person
                j = int(np.argmax([iou(b, rb) for b, _ in ppl]))
                ppl = [ppl[j]] if iou(ppl[j][0], rb) >= 0.3 else []
            for b, kp in ppl:
                x1, y1, x2, y2 = b; H, W = fr.shape[:2]
                if y2 - y1 < 100:
                    continue
                pw, ph = 0.4 * (x2 - x1), 0.1 * (y2 - y1); cx, cy = max(0, int(x1 - pw)), max(0, int(y1 - ph))
                crop = fr[cy:min(H, int(y2 + ph)), cx:min(W, int(x2 + pw))]
                t = float(np.linalg.norm(kp[[5, 6], :2].mean(0) - kp[[11, 12], :2].mean(0))) if min(kp[[5, 6, 11, 12], 2]) >= .3 else .3 * (y2 - y1)
                wr = [kp[i, :2] - [cx, cy] for i in (9, 10) if kp[i, 2] >= .3]
                if not wr or crop.size == 0:
                    continue
                d = det.predict(crop, classes=[67], imgsz=640, conf=0.15, verbose=False, device="cpu")[0]
                for bb, cf in zip(d.boxes.xyxy.numpy(), d.boxes.conf.numpy()):
                    pc = np.array([(bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2])
                    if min(np.linalg.norm(pc - w) for w in wr) > 0.6 * t:
                        continue
                    bw, bh = bb[2] - bb[0], bb[3] - bb[1]; side = 1.1 * max(bw, bh, 0.15 * t)
                    a1, b1 = int(max(0, pc[0] - side)), int(max(0, pc[1] - side))
                    a2, b2 = int(min(crop.shape[1], pc[0] + side)), int(min(crop.shape[0], pc[1] + side))
                    c = crop[b1:b2, a1:a2]
                    if c.size == 0:
                        continue
                    name = f"k{n}"
                    cv2.imwrite(os.path.join(out, "img", name + ".jpg"), cv2.resize(c, (112, 112), interpolation=cv2.INTER_AREA))
                    idx.write(json.dumps({"id": name, "alert": k, "label": lab, "conf": round(float(cf), 3),
                                          "long_torso": round(float(max(bw, bh) / t), 3), "file": os.path.basename(f)}) + "\n")
                    n += 1; got += 1
        idx.flush()
        open(dp, "a", encoding="utf-8").write(k + "\n")
        print(lab, k, "candidates", got, flush=True)
    print("total", n)


if __name__ == "__main__":
    main(sys.argv[1])
