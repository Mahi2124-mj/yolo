"""PO candidates for review (8 Oct 2026, the user: "a PO may be missed - save it
instead; every case near the pocket, till 7 tomorrow").

Every person whose hand the PO rule looked at at the hip is a candidate. Per
track the best look (highest pocket score P) is kept; when the track ends (or
after max_track_sec) up to `shots` snapshots are written:
  <dir>/<YYYYMMDD>/{alert, likely, maybe, near}/<HHMMSS>_<cam>_ID<tid>_P<p>.jpg
    alert  = the rule raised PO (also in the evidence)
    likely = P >= likely_p, no alert  -> look here first for missed PO
    maybe  = maybe_p <= P < likely_p
    near   = hand at the hip, model says out
plus <dir>/<YYYYMMDD>/candidates.csv (time, camera, track, P, folder, OK_NG).
Each snapshot = the person, padded, from the clean camera picture. Light:
image writes only. Stops by itself at until (HH:MM)."""
import csv
import datetime as dt
import os
import shutil
import time

import cv2


class POCandidates:
    def __init__(self, cfg, cam_id, cam_name):
        c = cfg or {}
        self.cam, self.name = cam_id, cam_name
        self.dir = c.get("dir", "D:/poketenashi-data/po_candidates")
        self.likely, self.maybe = float(c.get("likely_p", 0.5)), float(c.get("maybe_p", 0.2))
        self.shots = int(c.get("shots", 3))
        self.gap = float(c.get("track_end_sec", 3.0))
        self.max_sec = float(c.get("max_track_sec", 60.0))
        now = dt.datetime.now()
        if not c.get("until"):                       # 9 Oct: no end - the teacher runs for good
            self.end = float("inf")
        else:
            h, m = map(int, str(c["until"]).split(":"))
            end = now.replace(hour=h, minute=m, second=0, microsecond=0)
            self.end = (end if end > now else end + dt.timedelta(days=1)).timestamp()
        self.tr = {}
        # 9 Oct 2026 (user): "a missed / likely PO: confirm it against the masters
        # first - an exact match teaches the camera as OK PO, no match as NG PO".
        # A likely track (no alert) is judged by its best OK-minus-NG margin
        # (rules.py, the masters in PO_review_v4): >= ok_margin -> its snapshots go
        # to OK_PO_real/_agent_auto, <= ng_margin -> NG_PO_false/_agent_auto, in
        # between -> _agent_review (for the user, not a master). The masters
        # reload themselves when those folders change (rules.PO.ref_reload_sec).
        t = c.get("teach") or {}
        self.teach = bool(t.get("enabled", False))
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        absd = lambda d: d if os.path.isabs(d) else os.path.join(root, d)
        self.teach_ok = absd(t.get("ok_dir", "PO_review_v4/OK_PO_real/_agent_auto"))
        self.teach_ng = absd(t.get("ng_dir", "PO_review_v4/NG_PO_false/_agent_auto"))
        self.teach_rev = absd(t.get("review_dir", "PO_review_v4/_agent_review"))
        self.ok_margin = float(t.get("ok_margin", 0.15))
        self.ng_margin = float(t.get("ng_margin", 0.0))
        self.teach_max = int(t.get("max_files", 600))

    def observe(self, frame, cand, po_alert_tids):
        now = time.time()
        if now > self.end:
            return
        H, W = frame.shape[:2]
        for tid, c in cand.items():
            t = self.tr.setdefault(tid, {"start": now, "best": -1.0, "shots": [], "alert": False})
            t["last"] = now
            t["alert"] = t["alert"] or tid in po_alert_tids
            x1, y1, x2, y2 = c["bbox"]
            pw, ph = 0.25 * (x2 - x1), 0.08 * (y2 - y1)
            crop = frame[max(0, int(y1 - ph)):min(H, int(y2 + ph)), max(0, int(x1 - pw)):min(W, int(x2 + pw))]
            if crop.size == 0:
                continue
            t["shots"].append((c["P"], crop.copy()))
            t["shots"] = sorted(t["shots"], key=lambda s: -s[0])[: self.shots]
            t["best"] = max(t["best"], c["P"])
            for k, v in (c.get("cnt") or {}).items():
                cnt = t.setdefault("cnt", {})
                if k in ("acc", "refsim"):
                    cnt[k] = max(cnt.get(k, -1.0 if k == "refsim" else 0), v)
                else:
                    cnt[k] = cnt.get(k, 0) + v
        for tid in list(self.tr):
            t = self.tr[tid]
            if now - t["last"] > self.gap or now - t["start"] > self.max_sec:
                self._write(tid, t)
                del self.tr[tid]

    def _write(self, tid, t):
        if not t["shots"]:
            return
        kind = "alert" if t["alert"] else "likely" if t["best"] >= self.likely else "maybe" if t["best"] >= self.maybe else "near"
        day = time.strftime("%Y%m%d", time.localtime(t["start"]))
        d = os.path.join(self.dir, day, kind)
        try:
            os.makedirs(d, exist_ok=True)
            base = f"{time.strftime('%H%M%S', time.localtime(t['start']))}_{self.cam}_ID{tid}_P{int(round(t['best'] * 100)):02d}"
            for k, (p, crop) in enumerate(t["shots"]):
                cv2.imwrite(os.path.join(d, f"{base}_{k + 1}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            taught = self._teach(kind, t, d, base) if self.teach else ""
            path = os.path.join(self.dir, day, "candidates.csv")
            new = not os.path.exists(path)
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["time", "camera", "track", "best_P", "folder", "files", "OK_NG (fill in)", "frames POCKET/UNSURE/OUT/UNKNOWN/REPOSE", "max_acc", "max_ref_sim", "taught"])
                w.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t["start"])), self.name, tid,
                            round(t["best"], 2), kind, base + "_*.jpg", "",
                            "/".join(str((t.get("cnt") or {}).get(k, 0)) for k in ("POCKET", "UNSURE", "VISIBLE_OUT", "UNKNOWN", "REPOSE")),
                            (t.get("cnt") or {}).get("acc", 0), (t.get("cnt") or {}).get("refsim", ""), taught])
        except Exception as e:
            print(f"[po-cand] write failed: {e}", flush=True)

    def _teach(self, kind, t, d, base):
        """Copy a likely track's snapshots into the OK / NG masters (or review)."""
        cnt = t.get("cnt") or {}
        if kind != "likely" or not cnt.get("refn"):
            return ""                      # no alert-free likely look the masters judged
        m = float(cnt.get("refsim", -1.0))
        dst, tag = ((self.teach_ok, "OK") if m >= self.ok_margin else
                    (self.teach_ng, "NG") if m <= self.ng_margin else (self.teach_rev, "REVIEW"))
        try:
            os.makedirs(dst, exist_ok=True)
            if tag != "REVIEW" and len(os.listdir(dst)) >= self.teach_max:
                return f"{tag} (folder full)"
            for k in range(len(t["shots"])):
                src = os.path.join(d, f"{base}_{k + 1}.jpg")
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(dst, f"{self.name}_{base}_{k + 1}_m{m:+.2f}.jpg"))
            print(f"[po-teach] {self.name} ID{base.split('_ID')[-1]}: margin {m:+.3f} -> {tag}", flush=True)
        except Exception as e:
            print(f"[po-teach] failed: {e}", flush=True)
            return "error"
        return tag
