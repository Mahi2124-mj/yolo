"""Trucks near people - for a camera with "truck_exempt" in monitor_cameras.json
(8 Oct 2026, the user, Material Gate: "whoever gets in or out of a truck or is
around it, and cannot be identified as company or outside - leave them").

A small general detector (yolo11n, COCO truck / bus / car) looks every
every_sec seconds; between looks the last boxes are kept. A person counts as
"at a truck" when their box touches a truck box widened by margin (of the
truck's width), and stays exempt for hold_sec after (getting in and out takes
a few steps). Only people WITHOUT a face ID code are exempted; the caller
decides that."""
import time

import numpy as np

CLASSES = {7: "truck", 5: "bus", 6: "train", 2: "car"}   # a truck cut off at the frame edge reads as "train"


class TruckNear:
    def __init__(self, cfg, root):
        c = cfg if isinstance(cfg, dict) else {}
        self.every = float(c.get("every_sec", 1.0))
        self.margin = float(c.get("margin", 0.25))
        self.hold = float(c.get("hold_sec", 8.0))
        self.min_conf = float(c.get("conf", 0.35))
        self.classes = [int(k) for k in c.get("classes", [7, 5, 6])]
        self.max_margin = float(c.get("max_margin_px", 150))          # at 1920 wide
        self.ignore = [list(map(float, b)) for b in c.get("ignore_boxes", [])]   # e.g. the guard booth
        from ultralytics import YOLO
        import os
        self.model = YOLO(os.path.join(root, c.get("model", "yolo11n.pt")))
        self.boxes, self.t = [], 0.0
        self.near_until = {}

    def update(self, frame, persons):
        """-> set of track ids at / just away from a truck now."""
        now = time.time()
        if now - self.t >= self.every:
            self.t = now
            r = self.model.predict(frame, imgsz=640, conf=self.min_conf, classes=self.classes, verbose=False)[0]
            sc = frame.shape[1] / 1920.0
            ign = [[v * sc for v in b] for b in self.ignore]

            def inside(b, g):
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                return g[0] <= cx <= g[2] and g[1] <= cy <= g[3]
            self.boxes = [b for b in r.boxes.xyxy.cpu().numpy().tolist() if not any(inside(b, g) for g in ign)]
            self._sc = sc
        out = set()
        for p in persons:
            x1, y1, x2, y2 = [float(v) for v in p.bbox]
            for b in self.boxes:
                m = min(self.margin * (b[2] - b[0]), self.max_margin * getattr(self, "_sc", 1.0))
                if x1 < b[2] + m and x2 > b[0] - m and y1 < b[3] + m * 0.5 and y2 > b[1]:
                    self.near_until[p.track_id] = now + self.hold
                    break
            if self.near_until.get(p.track_id, 0) > now:
                out.add(p.track_id)
        if len(self.near_until) > 500:
            for k in [k for k, t in self.near_until.items() if t < now][:250]:
                del self.near_until[k]
        return out
