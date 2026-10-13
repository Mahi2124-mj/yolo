"""Every person the camera sees (10 Oct 2026, the user: "an agent behind the model -
whenever any person appears, take their clip and snapshot").

Per track the clearest look (the tallest box, sampled at most every sample_sec) is
kept; when the track ends (no sighting for track_end_sec, or after max_track_sec)
one clean snapshot is written to
  <dir>/<YYYYMMDD>/<cam>/<HHMMSS>_<cam>_ID<tid>.jpg
and a row to <dir>/<YYYYMMDD>/persons.csv: start, end, camera, track, rules the
person broke (alerts), snapshot. The clip of that time span is cut afterwards from
the raw recording by tools/person_clips.py (a separate light process: no video
encoding in the monitor). Image writes only - light."""
import csv
import os
import time

import cv2


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


class PersonLog:
    def __init__(self, cfg, cam_id, cam_name):
        c = cfg or {}
        self.cam, self.name = cam_id, cam_name
        self.dir = c.get("dir", "D:/poketenashi-data/persons")
        self.gap = float(c.get("track_end_sec", 3.0))
        self.max_sec = float(c.get("max_track_sec", 60.0))
        self.sample = float(c.get("sample_sec", 0.5))
        self.min_h = int(c.get("min_height_px", 60))
        # fixed things the pose model takes for a person (10 Oct: the PPE cartoon
        # standee at the Help-desk) - a box overlapping one of these is skipped
        self.ignore = [list(map(float, b)) for b in (c.get("ignore_boxes") or {}).get(cam_id, [])]
        self.tr = {}

    def observe(self, frame, persons, alerts):
        """alerts: {track_id: set(rule codes)} of this frame."""
        now = time.time()
        H, W = frame.shape[:2]
        for p in persons:
            if not getattr(p, "tracked", True):
                continue
            tid = p.track_id
            if self.ignore and any(_iou(p.bbox, b) >= 0.5 for b in self.ignore):
                continue
            t = self.tr.setdefault(tid, {"start": now, "best_h": 0.0, "crop": None, "rules": set(), "t_s": 0.0})
            t["last"] = now
            t["rules"] |= alerts.get(tid, set())
            x1, y1, x2, y2 = [float(v) for v in p.bbox]
            h = y2 - y1
            if h >= self.min_h and h > t["best_h"] and now - t["t_s"] >= self.sample:
                pw, ph = 0.25 * (x2 - x1), 0.08 * h
                crop = frame[max(0, int(y1 - ph)):min(H, int(y2 + ph)), max(0, int(x1 - pw)):min(W, int(x2 + pw))]
                if crop.size:
                    t["crop"], t["best_h"], t["t_s"] = crop.copy(), h, now
        for tid in list(self.tr):
            t = self.tr[tid]
            if now - t["last"] > self.gap or now - t["start"] > self.max_sec:
                self._write(tid, t)
                del self.tr[tid]

    def _write(self, tid, t):
        if t["crop"] is None:
            return
        day = time.strftime("%Y%m%d", time.localtime(t["start"]))
        d = os.path.join(self.dir, day, self.cam)
        try:
            os.makedirs(d, exist_ok=True)
            name = f"{time.strftime('%H%M%S', time.localtime(t['start']))}_{self.cam}_ID{tid}.jpg"
            cv2.imwrite(os.path.join(d, name), t["crop"], [cv2.IMWRITE_JPEG_QUALITY, 88])
            path = os.path.join(self.dir, day, "persons.csv")
            new = not os.path.exists(path)
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["start", "end", "camera_id", "camera", "track", "alerts", "snapshot", "clip"])
                w.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t["start"])),
                            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t["last"])),
                            self.cam, self.name, tid, "+".join(sorted(t["rules"])),
                            os.path.join(self.cam, name), ""])
        except Exception as e:
            print(f"[persons] write failed: {e}", flush=True)
