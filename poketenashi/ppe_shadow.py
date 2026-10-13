"""PPE shadow mode (8 Oct 2026, the user): cap and safety shoes are judged on
every camera but raise NO alert - each verdict is logged with its crop so the
user can mark OK / NG and the models can be retrained before they go live.

  cap   : models/ppe_cap_v1.pt (MobileNetV3-Small on the head, any cap colour);
          5-fold CV per person 8 Oct: P>=0.5 caught 36/71 no-cap, flagged
          28/156 cap wearers - too weak for alerts, hence shadow.
  shoes : black = OK (any shade); the share of dark, low-saturation pixels just
          below the ankles. Measured 8 Oct: misaligned crops and sunlight make
          this unreliable - shadow only.

Per track, a look every look_sec (head and feet need a person >= min_h px
tall); after `looks` looks the track's mean is written once to
<dir>/<YYYYMMDD>/shadow.jsonl with its crops in <dir>/<YYYYMMDD>/{head,feet}/.
One model for all cameras; at most max_per_frame looks per frame, so a crowd
cannot slow the camera down."""
import json
import os
import threading
import time
from collections import defaultdict

import cv2
import numpy as np

_MODEL = {}
_LOCK = threading.Lock()


def _cap_model(path):
    with _LOCK:
        if path not in _MODEL:
            import torch
            import torch.nn as nn
            import torchvision.models as tvm
            net = tvm.mobilenet_v3_small(weights=None)
            net.classifier[3] = nn.Linear(net.classifier[3].in_features, 1)
            net.load_state_dict(torch.load(path, map_location="cpu"))
            _MODEL[path] = net.eval()
        return _MODEL[path]


MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def head_box(p, H, W):
    k, bb = p.keypoints, p.bbox
    h = float(bb[3] - bb[1])
    if k[5, 2] < 0.5 or k[6, 2] < 0.5:
        return None
    sx = sorted([float(k[5, 0]), float(k[6, 0])])
    sw = max(sx[1] - sx[0], 0.12 * h)
    cx = (sx[0] + sx[1]) / 2
    b = [int(cx - 0.7 * sw), int(bb[1] - 0.05 * h), int(cx + 0.7 * sw), int((k[5, 1] + k[6, 1]) / 2)]
    b = [max(0, b[0]), max(0, b[1]), min(W, b[2]), min(H, b[3])]
    return b if b[2] - b[0] >= 12 and b[3] - b[1] >= 12 else None


def feet_box(p, H, W):
    k, bb = p.keypoints, p.bbox
    h = float(bb[3] - bb[1])
    if k[15, 2] < 0.4 or k[16, 2] < 0.4:
        return None
    ax = sorted([float(k[15, 0]), float(k[16, 0])])
    ay = max(float(k[15, 1]), float(k[16, 1]))
    b = [int(ax[0] - 0.08 * h), int(ay - 0.06 * h), int(ax[1] + 0.08 * h), int(min(bb[3] + 0.04 * h, ay + 0.1 * h))]
    b = [max(0, b[0]), max(0, b[1]), min(W, b[2]), min(H, b[3])]
    return b if b[2] - b[0] >= 10 and b[3] - b[1] >= 8 else None


def dark_frac(crop):
    h, w = crop.shape[:2]
    r = crop[int(h * .4):int(h * .9), int(w * .1):int(w * .9)]
    if r.size == 0:
        return None
    hsv = cv2.cvtColor(r, cv2.COLOR_BGR2HSV)
    return float(((hsv[..., 2] < 85) & (hsv[..., 1] < 120)).mean())


class PPEShadow:
    def __init__(self, cfg, root, cam_id, cam_name):
        c = cfg or {}
        self.cam, self.name = cam_id, cam_name
        self.dir = c.get("dir", "D:/poketenashi-data/ppe/shadow")
        self.look_sec = float(c.get("look_sec", 1.0))
        self.looks = int(c.get("looks", 4))
        self.min_h = float(c.get("min_h", 160))
        self.per_frame = int(c.get("max_per_frame", 2))
        self.cap_th = float(c.get("cap_threshold", 0.5))
        self.shoe_th = float(c.get("shoe_dark_min", 0.12))
        self.net = _cap_model(os.path.join(root, c.get("cap_model", "models/ppe_cap_v1.pt")))
        self.st = defaultdict(lambda: {"t": 0.0, "cap": [], "shoe": [], "head": None, "feet": None, "done": False})

    def _p_nocap(self, crop):
        import torch
        im = cv2.cvtColor(cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(((im - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32))
        with torch.inference_mode():
            return float(torch.sigmoid(self.net(x)).item())

    def observe(self, frame, persons):
        if frame is None:
            return
        now = time.time()
        H, W = frame.shape[:2]
        n = 0
        for p in persons:
            if n >= self.per_frame:
                break
            if not p.tracked or float(p.bbox[3] - p.bbox[1]) < self.min_h:
                continue
            s = self.st[p.track_id]
            if s["done"] or now - s["t"] < self.look_sec:
                continue
            s["t"] = now
            n += 1
            hb, fb = head_box(p, H, W), feet_box(p, H, W)
            if hb:
                c = frame[hb[1]:hb[3], hb[0]:hb[2]]
                s["cap"].append(self._p_nocap(c))
                s["head"] = c.copy()
            if fb:
                c = frame[fb[1]:fb[3], fb[0]:fb[2]]
                d = dark_frac(c)
                if d is not None:
                    s["shoe"].append(d)
                    s["feet"] = c.copy()
            if len(s["cap"]) >= self.looks or len(s["shoe"]) >= self.looks:
                self._write(p.track_id, s)
        if len(self.st) > 400:
            for k in list(self.st)[:200]:
                if self.st[k]["done"] or now - self.st[k]["t"] > 60:
                    del self.st[k]

    def _write(self, tid, s):
        s["done"] = True
        day = time.strftime("%Y%m%d")
        d = os.path.join(self.dir, day)
        aid = f"{time.strftime('%H%M%S')}_{self.cam}_ID{tid}"
        rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "camera": self.cam, "camera_name": self.name, "track_id": tid, "id": aid}
        if s["cap"]:
            p = float(np.mean(s["cap"]))
            rec.update(cap_p_nocap=round(p, 3), cap_verdict="NO CAP" if p >= self.cap_th else "cap", cap_looks=len(s["cap"]))
        if s["shoe"]:
            q = float(np.median(s["shoe"]))
            rec.update(shoe_dark=round(q, 3), shoe_verdict="NOT BLACK" if q < self.shoe_th else "black", shoe_looks=len(s["shoe"]))
        try:
            for kind in ("head", "feet"):
                if s[kind] is not None:
                    os.makedirs(os.path.join(d, kind), exist_ok=True)
                    cv2.imwrite(os.path.join(d, kind, aid + ".jpg"), s[kind], [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(os.path.join(d, "shadow.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as e:                       # never disturb the camera loop
            print(f"[ppe] shadow log failed: {e}", flush=True)
        s["head"] = s["feet"] = None
