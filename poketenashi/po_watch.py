"""PO check (13 Oct 2026, the user: "record continuously and take a snapshot + clip of
anyone whose hand you cannot see - the model does its own work - so I can see that no PO
was missed").

Shadow only: it never raises or stops an alert. Per tracked person, at most once every
check_sec, each wrist that sits at the hip (wrist-hip distance < near_torso x torso, not
raised above the hip) is looked at with MediaPipe on a zoomed crop - the same check the
fingertip rule uses. No hand found there = "hand hidden at the hip" (a pocket, or just
not visible). When the track ends, a person with >= min_hidden such looks is written to
  <dir>/<YYYYMMDD>/<HHMMSS>_<cam>_ID<tid>.jpg  (clearest hidden-hand moment, wrists ringed)
  <dir>/<YYYYMMDD>/po_check.csv  (start, end, camera, track, hidden, checks, po_alert, snapshot)
tools/po_check_clips.py adds each person's clean clip (persons/, same camera + track) and
an index.html, the people WITHOUT a PO alert first (the possible misses)."""
import csv
import os
import time

import cv2
import numpy as np


class POWatch:
    def __init__(self, cfg, cam_id, cam_name, hand):
        c = cfg or {}
        self.cam, self.name, self.hand = cam_id, cam_name, hand
        self.dir = c.get("dir", "C:/Users/DX-ADMIN/Pictures/yolo/PO_check")
        self.every = float(c.get("check_sec", 1.0))
        self.near = float(c.get("near_torso", 0.6))
        self.min_hidden = int(c.get("min_hidden", 3))
        self.min_torso = float(c.get("min_torso_px", 50))
        self.gap = float(c.get("track_end_sec", 3.0))
        self.kc = float(c.get("min_keypoint_conf", 0.4))
        self.tr = {}

    def observe(self, frame, persons, po_ids):
        now = time.time()
        H, W = frame.shape[:2]
        for p in persons:
            if not getattr(p, "tracked", True):
                continue
            t = self.tr.setdefault(p.track_id, {"start": now, "last": 0.0, "hidden": 0, "checks": 0,
                                                "po": False, "img": None, "best": -1.0})
            t["seen"] = now
            if p.track_id in po_ids:
                t["po"] = True
            if now - t["last"] < self.every:
                continue
            t["last"] = now
            k = np.asarray(p.keypoints, dtype=float)
            if min(k[[5, 6, 11, 12], 2]) < self.kc:
                continue
            torso = float(np.linalg.norm(k[[5, 6], :2].mean(0) - k[[11, 12], :2].mean(0)))
            if torso < self.min_torso:
                continue
            hidden_here, wrists = False, []
            for wi, hi in ((9, 11), (10, 12)):
                if k[wi, 2] < self.kc:
                    continue
                Wp, Hp = k[wi, :2], k[hi, :2]
                if np.linalg.norm(Wp - Hp) > self.near * torso or Wp[1] < Hp[1] - 0.15 * torso:
                    continue                                  # not at the hip
                r = int(max(0.75 * torso, 12)); yo = int(0.2 * r)
                hands = self.hand.detect_in_region(frame, (int(Wp[0]) - r, int(Wp[1]) - r + yo,
                                                           int(Wp[0]) + r, int(Wp[1]) + r + yo), upscale_to=448)
                seen = any(h["n_tips"] >= 1 and np.linalg.norm(h["wrist"] - Wp) / torso < 0.65 for h in hands)
                t["checks"] += 1
                if not seen:
                    hidden_here = True
                    wrists.append(Wp)
            if hidden_here:
                t["hidden"] += 1
                if torso > t["best"]:                         # keep the closest (largest) look
                    x1, y1, x2, y2 = [float(v) for v in p.bbox]
                    pw, ph = 0.3 * (x2 - x1), 0.08 * (y2 - y1)
                    img = frame[max(0, int(y1 - ph)):min(H, int(y2 + ph)), max(0, int(x1 - pw)):min(W, int(x2 + pw))].copy()
                    for w in wrists:
                        cv2.circle(img, (int(w[0] - max(0, int(x1 - pw))), int(w[1] - max(0, int(y1 - ph)))),
                                   max(8, int(0.15 * torso)), (0, 255, 255), 2)
                    t["img"], t["best"] = img, torso
        for tid in [tid for tid, t in self.tr.items() if now - t.get("seen", now) > self.gap]:
            self._close(tid, self.tr.pop(tid))

    def _close(self, tid, t):
        if t["hidden"] < self.min_hidden or t["img"] is None:
            return
        day = time.strftime("%Y%m%d", time.localtime(t["start"]))
        d = os.path.join(self.dir, day)
        os.makedirs(d, exist_ok=True)
        name = f"{time.strftime('%H%M%S', time.localtime(t['start']))}_{self.cam}_ID{tid}.jpg"
        cv2.imwrite(os.path.join(d, name), t["img"], [cv2.IMWRITE_JPEG_QUALITY, 88])
        p = os.path.join(d, "po_check.csv")
        new = not os.path.exists(p)
        with open(p, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["start", "end", "camera_id", "camera", "track", "hidden", "checks", "po_alert", "snapshot"])
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t["start"])),
                        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t["seen"])),
                        self.cam, self.name, tid, t["hidden"], t["checks"], int(t["po"]), name])
