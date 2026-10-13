"""Show the PC's time instead of the camera's own clock (8 Oct 2026, the user:
"every camera's time must be real time, like the PC's"). The cameras' clocks
drift (Dx ~22 min behind, Help-desk ~47 s ahead) and can only be set by
logging in to each camera, so instead the monitor covers the camera's
timestamp in the picture with the PC time - on the dashboard, the TV and the
evidence clips. Alert times were always PC time.

The camera's timestamp is FOUND, per camera, with no setup (a new camera too):
during the first learn_sec seconds a 1-per-second sample of the top and bottom
bands is kept; pixels that are bright text AND change at almost every second
are the seconds digits; the row of bright text around them, scanned left and
right, is the whole date-time string. That box is painted over and the PC
time written into it. Not found -> the PC time is written top-left instead.
"""
import time

import cv2
import numpy as np


class ClockCover:
    def __init__(self, cfg=None):
        c = cfg or {}
        self.learn_sec = float(c.get("learn_sec", 7.0))
        self.band = float(c.get("band", 0.16))
        self.samples, self.t_last, self.t0 = [], 0.0, None
        self.box = None          # (x1, y1, x2, y2) in frame pixels
        self.done = False
        self.size = None
        # a box set for this camera (monitor_cameras.json "clock_box", in pixels
        # of a 1920-wide picture) wins over the search
        self.fixed = c.get("box")

    def _learn(self, frame):
        now = time.time()
        if self.t0 is None:
            self.t0 = now
        if now - self.t_last < 0.95:
            return
        self.t_last = now
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.samples.append(g)
        if now - self.t0 < self.learn_sec or len(self.samples) < 6:
            return
        self.done = True
        self.box = self._find(self.samples)
        self.samples = []
        H, W = frame.shape[:2]
        where = "not found - PC time written top-left" if self.box is None else f"at {self.box}"
        print(f"[clock] camera timestamp {where} ({W}x{H})", flush=True)

    def _find(self, gs):
        H, W = gs[0].shape
        bh = int(H * self.band)
        best = None
        for y0 in (0, H - bh):
            st = np.stack([g[y0:y0 + bh] for g in gs]).astype(np.int16)
            bright = (st > 170).mean(axis=0) > 0.3                     # light text (OSD is white)
            ch = (np.abs(np.diff(st, axis=0)) > 60).mean(axis=0) >= 0.6  # changes nearly every second
            m = (ch & bright).astype(np.uint8) * 255
            m = cv2.dilate(m, np.ones((5, 9), np.uint8))
            n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
            for i in range(1, n):
                x, y, w, h, a = stats[i]
                if h < 8 or h > bh * 0.8 or w > W * 0.15 or a < 40:
                    continue
                if best is None or a > best[0]:
                    best = (a, x, y + y0, w, h, y0)
        if best is None:
            return None
        _, x, y, w, h, y0 = best
        # the whole string: bright text on that row, left and right of the digits
        row = np.stack([g[y:y + h] for g in gs]).max(axis=0) > 170
        cols = row.mean(axis=0) > 0.04
        gap_max = int(h * 1.0)
        x1, gap = x, 0
        while x1 > 0 and gap < gap_max:
            x1 -= 1
            gap = 0 if cols[x1] else gap + 1
        x1 += gap
        x2, gap = x + w, 0
        while x2 < W - 1 and gap < gap_max:
            x2 += 1
            gap = 0 if cols[x2] else gap + 1
        x2 -= gap
        if x2 - x1 > 16 * h:                          # longer than a date-time string: keep the digits' side
            x1 = max(x1, x + w - 14 * h)
        pad = int(h * 0.35)
        return (max(0, x1 - pad), max(0, y - pad), min(W, x2 + pad), min(H, y + h + pad))

    def apply(self, frame, raw=None):
        """Paint the PC time onto `frame` (in place) and return it."""
        if self.size != frame.shape[:2]:              # first frame, or the picture size changed
            self.size = frame.shape[:2]
            self.samples, self.t0, self.done, self.box = [], None, False, None
            if self.fixed:
                k = frame.shape[1] / 1920.0
                self.box = tuple(int(v * k) for v in self.fixed)
                self.done = True
        if not self.done:
            self._learn(raw if raw is not None else frame)
            return frame
        txt = time.strftime("%Y-%m-%d %H:%M:%S")
        H, W = frame.shape[:2]
        if self.box is not None:
            x1, y1, x2, y2 = self.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
            hh = y2 - y1
            sc = max(0.4, (hh * 0.62) / 22.0)
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, sc, 2)
            if tw > x2 - x1:
                sc *= (x2 - x1) / float(tw) * 0.97
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, sc, 2)
            cv2.putText(frame, txt, (x1 + max(2, (x2 - x1 - tw) // 2), y1 + (hh + th) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, sc, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            sc = max(0.6, W / 1920.0 * 1.1)
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, sc, 2)
            cv2.rectangle(frame, (8, H - th - 60), (20 + tw, H - 44), (0, 0, 0), -1)
            cv2.putText(frame, txt, (14, H - 50), cv2.FONT_HERSHEY_SIMPLEX, sc, (255, 255, 255), 2, cv2.LINE_AA)
        return frame
