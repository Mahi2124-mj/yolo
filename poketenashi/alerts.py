"""Violation handling: one alert per violation episode, log to JSONL, save snapshots."""
import os
import json
import threading
import time

import cv2
import numpy as np

from . import ruleinfo

# Held while a row is appended to the log. retention.py rewrites the same file
# from its own thread; without a shared lock a row appended between its read and
# its replace simply vanished.
LOG_LOCK = threading.Lock()

# One label per rule, owned by poketenashi/ruleinfo.py: the log row, the
# dashboard and the TV page must not be able to describe the same rule
# differently.
RULE_NAMES = ruleinfo.LABELS


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


class AlertManager:
    def __init__(self, cfg, clip_recorder=None, camera=None):
        a = cfg["alerts"]
        # (id, name) of the camera this manager logs for: every row says which
        # camera saw it, and evidence files of cameras other than cam1 carry the
        # camera id in their name (track ids repeat across cameras).
        self.cam_id, self.cam_name = camera or (None, None)
        # An episode ends once its violation has not been seen for this long.
        self.cooldown = a.get("cooldown_sec", 8.0)
        # A violation still going on is alerted again this often (0 = never).
        self.repeat_sec = float(a.get("repeat_sec", 60.0))
        # A "new" track this close to a running episode of the same rule is the
        # same person whose track id changed, not a second person.
        self.merge_iou = float(a.get("merge_iou", 0.4))
        self.log_file = a.get("log_file", "violations.jsonl")
        self.save_snap = a.get("save_snapshots", True)
        self.snap_dir = a.get("snapshot_dir", "snapshots")
        # Optional ClipRecorder. Naming lives here rather than in the caller so a
        # row's snapshot and clip always share one base name.
        self.clips = clip_recorder
        self._episodes = []   # {"rule", "keys", "bbox", "seen", "alerted"}
        if self.save_snap:
            os.makedirs(self.snap_dir, exist_ok=True)

    def handle(self, violations, frame, present=None):
        """Log the violations that START an episode and return them.

        `present`: the track ids in this frame. An episode whose own person is
        still among them cannot be taken over by a neighbour's overlapping box -
        that is a second person breaking the rule, with an alert of their own.

        One alert per incident, not one per 8 seconds of it. The old per-track
        cooldown re-alerted a continuing violation every 8 s, and re-alerted it
        immediately whenever the tracker changed the person's id (every few
        seconds at 5 fps): one group of three men looking at a phone produced
        four KE alerts in a minute. Now an alert opens an episode, the episode
        follows the person by track id or, when the id changes, by overlap, and
        it only alerts again after `repeat_sec` if the violation never stopped.

        Each returned dict is enriched with `time`, `desc` and the `snapshot` /
        `clip` filenames, so a caller (the web layer) can push a live alert
        without re-reading the log it just wrote.
        """
        now = time.time()
        fresh = []
        present = set(present) if present is not None else {v["track_id"] for v in violations}
        for v in violations:
            key = (v["track_id"], v["rule"])
            ep = self._episode_of(v, key, present)
            if ep is None:
                self._episodes.append({"rule": v["rule"], "keys": {key},
                                       "bbox": np.asarray(v["bbox"], float),
                                       "seen": now, "alerted": now})
                fresh.append(v)
                continue
            ep["keys"].add(key)
            ep["bbox"], ep["seen"] = np.asarray(v["bbox"], float), now
            if self.repeat_sec and now - ep["alerted"] >= self.repeat_sec:
                ep["alerted"] = now
                fresh.append(v)
        self._episodes = [e for e in self._episodes if now - e["seen"] <= self.cooldown]
        for v in fresh:
            self._log(v, now, frame)
        return fresh

    def _episode_of(self, v, key, present=()):
        best, best_iou = None, self.merge_iou
        for e in self._episodes:
            if e["rule"] != v["rule"]:
                continue
            if key in e["keys"]:
                return e
            # Overlap only stands in for a CHANGED track id: if the episode's
            # own person is still in the picture, this is somebody else - two
            # testers side by side with their phones got one KE alert between
            # them (28 Sep live test).
            if any(tid in present for tid, _ in e["keys"]):
                continue
            ov = _iou(e["bbox"], v["bbox"])
            if ov >= best_iou:
                best, best_iou = e, ov
        return best

    def _log(self, v, now, frame):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
        desc = RULE_NAMES.get(v["rule"], v["rule"])
        where = f"  [{self.cam_name}]" if self.cam_id and self.cam_id != "cam1" else ""
        print(f"[ALERT] {ts}  ID{v['track_id']:>3}  {v['rule']:<3} - {desc}{where}")

        # The evidence filenames go INTO the row. A reader (the web UI) would
        # otherwise have to re-derive them from the timestamp, which only works
        # while the naming scheme never changes and breaks silently when it does.
        prefix = f"{self.cam_id}_" if self.cam_id and self.cam_id != "cam1" else ""
        base = f"{int(now)}_{prefix}ID{v['track_id']}_{v['rule']}"
        snap = None
        if self.save_snap and frame is not None:
            try:
                cv2.imwrite(os.path.join(self.snap_dir, base + ".jpg"), frame)
                snap = base + ".jpg"
            except Exception as e:
                print("[alerts] snapshot failed:", e)

        # The clip is named now but only lands once its post-roll has been
        # recorded and encoded, a few seconds from here. That gap is normal: a
        # reader has to tolerate the file 404ing for a moment.
        # Keyed by (person, rule) so the recorder can keep the clip running for
        # as long as that same bypass is still on screen.
        clip = (self.clips.start(base + ".mp4", (v["track_id"], v["rule"]))
                if self.clips else None)
        # This alert's own thumbnail - the frame it fired on - even when the clip
        # it points at was opened by an earlier alert.
        poster = self.clips.request_poster(base + ".jpg") if clip else None

        rec = {"time": ts, "track_id": v["track_id"], "rule": v["rule"], "desc": desc}
        if self.cam_id:
            rec["camera"], rec["camera_name"] = self.cam_id, self.cam_name
        if v.get("person"):                 # employee code from face ID, when known
            rec["person"] = v["person"]
        if snap:
            rec["snapshot"] = snap
        if clip:
            rec["clip"] = clip
        if poster:
            rec["poster"] = poster
        if self.log_file:
            try:
                with LOG_LOCK, open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception as e:
                print("[alerts] log write failed:", e)

        v.update(time=ts, desc=desc, snapshot=snap, clip=clip, poster=poster,
                 camera=self.cam_id, camera_name=self.cam_name)
