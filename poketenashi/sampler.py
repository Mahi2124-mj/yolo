"""Collect training samples from the live camera, for PO and KE.

The rules are only as good as what they were tuned on, so the monitor keeps
small crops of the moments that matter - people near the camera with a hand at
the hip (PO) or raised (KE) - WHETHER OR NOT a rule fired, together with what
the rules said. Looked at, they show both the false alerts and the misses;
labelled, they train the pocket / phone classifiers.

Bounded on purpose: a few samples per person (spread over their visit), a daily
cap per kind, small JPEGs. Layout:
    <dir>/<po|ke>/<YYYYMMDD>/<epoch_ms>_ID<track>_<fired>_crop.jpg
                                                 ..._person.jpg
    <dir>/<po|ke>/index.jsonl     one line per sample (what the rules said)
"""
import json
import os
import time
from collections import defaultdict

import cv2
import numpy as np

from .ke import phone_call_pose, phone_use_pose, raised_wrists, zoom_box

MC = 0.4


class Sampler:
    def __init__(self, cfg):
        c = (cfg or {}).get("samples") or {}
        self.enabled = bool(c.get("enabled", False))
        self.dir = c.get("dir", "D:/poketenashi-data/samples")
        self.min_torso = float(c.get("min_torso", 90))
        self.every = float(c.get("every_sec", 1.5))
        self.per_track = int(c.get("per_track", 4))
        self.cap = int(c.get("daily_cap", 4000))
        self._last = {}                       # (kind, tid) -> time of the last sample
        self._count = defaultdict(int)        # (kind, tid) -> samples taken
        self._day = None
        self._day_count = defaultdict(int)    # kind -> samples today
        self.saved = 0
        self._errors, self._err_t = 0, 0.0
        if self.enabled:
            for kind in ("po", "ke"):
                os.makedirs(os.path.join(self.dir, kind), exist_ok=True)
            print(f"[samples] collecting PO/KE training samples into {self.dir} "
                  f"(people with torso >= {self.min_torso:.0f}px, {self.per_track} per person, "
                  f"{self.cap}/day per kind)", flush=True)

    # -- one frame ------------------------------------------------------------
    def observe(self, frame, persons, violations, rules):
        if not self.enabled or frame is None:
            return
        try:
            self._observe(frame, persons, violations, rules)
        except Exception as e:                       # never take the monitor down -
            self._errors += 1                        # and never flood the log either
            now = time.time()
            if now - self._err_t >= 60.0:
                print(f"[samples] skipped frames ({self._errors} so far): "
                      f"{type(e).__name__}: {e}", flush=True)
                self._err_t = now

    def _observe(self, frame, persons, violations, rules):
        now = time.time()
        day = time.strftime("%Y%m%d", time.localtime(now))
        if day != self._day:
            self._day, self._day_count = day, defaultdict(int)
        fired = defaultdict(set)
        for v in violations:
            fired[v["track_id"]].add(v["rule"])
        live = set()
        for p in persons:
            if not getattr(p, "tracked", True) or not getattr(p, "in_zone", True):
                continue
            tid = p.track_id
            live.add(tid)
            torso = p.torso_len(MC)
            if torso < self.min_torso:
                continue
            k = p.keypoints
            # PO: a wrist near the hip, at hip height
            H = [p.kp(s, MC) for s in ("l_hip", "r_hip")]
            Wr = [p.kp(s, MC) for s in ("l_wrist", "r_wrist")]
            if all(h is not None for h in H):
                pel = (H[0] + H[1]) / 2.0
                near = [w for w, h in zip(Wr, H) if w is not None
                        and abs(w[1] - h[1]) <= 0.5 * torso and abs(w[0] - h[0]) <= 0.6 * torso]
                arms = self._arms(p, rules, torso)
                # "pocket-like": a still wrist parked at the hip line - the frames
                # worth a label most (a pocket, or a look-alike to learn from)
                pocketish = any(a["still"] and abs(a["hang"]) <= 0.25 and 0.1 <= a["dist"] <= 0.5
                                for a in arms)
                if near and self._due("po", tid, now, pocketish or "PO" in fired[tid]):
                    xs = [h[0] for h in H] + [w[0] for w in near]
                    box = (min(xs) - 0.45 * torso, pel[1] - 0.45 * torso,
                           max(xs) + 0.45 * torso, pel[1] + 0.75 * torso)
                    self._save("po", frame, p, box, "PO" in fired[tid], now, {
                        "torso": round(torso), "speed": round(rules.speed_ps(tid) / torso, 2),
                        "pocketish": pocketish, "arms": arms, "rules": sorted(fired[tid])})
            # KE: a raised hand - the same zoom KE looks at
            wr, t = raised_wrists(k)
            if wr and t:
                use, call = bool(phone_use_pose(k)), bool(phone_call_pose(k))
                if self._due("ke", tid, now, use or call or "KE" in fired[tid]):
                    zb = zoom_box(wr, t, frame.shape[1], frame.shape[0])
                    if zb is not None:
                        hits = rules.ke._hits.get(tid) if rules.ke is not None else None
                        self._save("ke", frame, p, zb, "KE" in fired[tid], now, {
                            "torso": round(torso), "hits": len(hits or []),
                            "best": round(max((c for _, c in hits), default=0.0), 2) if hits else 0.0,
                            "use_pose": use, "call_pose": call, "rules": sorted(fired[tid])})
        for key in [key for key in self._last if key[1] not in live]:
            self._last.pop(key, None)
            self._count.pop(key, None)

    def _due(self, kind, tid, now, priority=False):
        """Rate limit per person. A `priority` moment (the phone-use / call pose,
        a pocket-like parked wrist, or a rule firing) may take up to per_track
        MORE samples: those are rare, and they are what the labels need."""
        limit = self.per_track * (2 if priority else 1)
        if self._day_count[kind] >= self.cap or self._count[(kind, tid)] >= limit:
            return False
        return now - self._last.get((kind, tid), 0.0) >= self.every

    @staticmethod
    def _arms(p, rules, torso):
        out = []
        for side in ("l", "r"):
            S, E = p.kp(f"{side}_shoulder", MC), p.kp(f"{side}_elbow", MC)
            W, H = p.kp(f"{side}_wrist", MC), p.kp(f"{side}_hip", MC)
            if W is None or H is None:
                continue
            ang = None
            if S is not None and E is not None:
                a, c = S - E, W - E
                cos = float(a @ c) / (float(np.linalg.norm(a) * np.linalg.norm(c)) + 1e-6)
                ang = round(float(np.degrees(np.arccos(max(-1.0, min(1.0, cos))))))
            still, mv = rules.wrist_still(p.track_id, side)
            out.append({"side": side, "hang": round(float((W[1] - H[1]) / torso), 2),
                        "dist": round(float(np.linalg.norm(W - H) / torso), 2), "angle": ang,
                        "mv": None if mv is None else round(float(mv), 3),
                        "still": bool(still and mv is not None)})
        return out

    def _save(self, kind, frame, p, box, fired, now, meta):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 < 16 or y2 - y1 < 16:
            return
        bx1, by1, bx2, by2 = [int(v) for v in p.bbox]
        pad = int(0.06 * (by2 - by1))
        person = frame[max(0, by1 - pad):min(h, by2 + pad), max(0, bx1 - pad):min(w, bx2 + pad)]
        if person.size == 0:
            return
        s = 256.0 / max(1, person.shape[0])
        person = cv2.resize(person, (max(1, int(person.shape[1] * s)), 256))
        day_dir = os.path.join(self.dir, kind, self._day)
        os.makedirs(day_dir, exist_ok=True)
        sid = f"{int(now * 1000)}_ID{p.track_id}_{'fired' if fired else 'quiet'}"
        cv2.imwrite(os.path.join(day_dir, sid + "_crop.jpg"), frame[y1:y2, x1:x2],
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(os.path.join(day_dir, sid + "_person.jpg"), person, [cv2.IMWRITE_JPEG_QUALITY, 85])
        rec = {"id": sid, "day": self._day, "time": time.strftime("%Y-%m-%d %H:%M:%S",
                                                                  time.localtime(now)),
               "track": int(p.track_id), "fired": bool(fired), "box": [x1, y1, x2, y2], **meta}
        with open(os.path.join(self.dir, kind, "index.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        self._last[(kind, p.track_id)] = now
        self._count[(kind, p.track_id)] += 1
        self._day_count[kind] += 1
        self.saved += 1
