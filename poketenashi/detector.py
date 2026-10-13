"""YOLO11 wrapper: person detection + pose keypoints + tracking, plus a second
model pass for objects we care about (phones for the KE rule)."""

import numpy as np
from ultralytics import YOLO

# COCO-17 keypoint indices
KP = {
    "nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4,
    "l_shoulder": 5, "r_shoulder": 6, "l_elbow": 7, "r_elbow": 8,
    "l_wrist": 9, "r_wrist": 10, "l_hip": 11, "r_hip": 12,
    "l_knee": 13, "r_knee": 14, "l_ankle": 15, "r_ankle": 16,
}

PERSON_CLASS = 0
PHONE_CLASS = 67


class Person:
    """One tracked person in a single frame."""

    def __init__(self, track_id, bbox, conf, keypoints, tracked=True):
        self.track_id = int(track_id)
        # False = the tracker has not confirmed this detection (it carries a
        # throwaway negative id that changes every frame).
        self.tracked = bool(tracked)
        # False = standing outside the rules' zone (set by RuleEngine), so the
        # overlay can show that this person is seen but not judged.
        self.in_zone = True
        self.bbox = np.asarray(bbox, dtype=float)
        self.conf = float(conf)
        self.keypoints = np.asarray(keypoints, dtype=float)

        # Remove phantom (0,0) keypoints.
        self.keypoints[
            (self.keypoints[:, 0] == 0) &
            (self.keypoints[:, 1] == 0), 2
        ] = 0.0

    def kp(self, name, min_conf=0.3):
        x, y, c = self.keypoints[KP[name]]
        if c < min_conf:
            return None
        return np.array([x, y], dtype=float)

    def center(self):
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])

    def feet(self):
        pts = [p for p in (self.kp("l_ankle"), self.kp("r_ankle")) if p is not None]
        if pts:
            return np.mean(pts, axis=0)

        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2.0, y2])

    def torso_len(self, min_conf=0.3):
        vals = []
        for a, b in (("l_shoulder", "l_hip"), ("r_shoulder", "r_hip")):
            pa, pb = self.kp(a, min_conf), self.kp(b, min_conf)
            if pa is not None and pb is not None:
                vals.append(np.linalg.norm(pa - pb))

        if vals:
            return float(np.mean(vals))

        x1, y1, x2, y2 = self.bbox
        return (y2 - y1) * 0.5

    # ==========================================================
    # PO helper functions
    # ==========================================================

    def shoulder_width(self, min_conf=0.3):
        ls = self.kp("l_shoulder", min_conf)
        rs = self.kp("r_shoulder", min_conf)

        if ls is None or rs is None:
            return self.torso_len(min_conf)

        return float(np.linalg.norm(ls - rs))

    def pelvis_center(self, min_conf=0.3):
        lh = self.kp("l_hip", min_conf)
        rh = self.kp("r_hip", min_conf)

        if lh is None and rh is None:
            x1, y1, x2, y2 = self.bbox
            return np.array([(x1 + x2) / 2.0,
                             y1 + (y2 - y1) * 0.60])

        if lh is None:
            return rh

        if rh is None:
            return lh

        return (lh + rh) / 2.0

    def body_scale(self, min_conf=0.3):
        return max(
            self.torso_len(min_conf),
            self.shoulder_width(min_conf)
        )

    def elbow_angle(self, side="left", min_conf=0.3):
        if side.startswith("l"):
            s = self.kp("l_shoulder", min_conf)
            e = self.kp("l_elbow", min_conf)
            w = self.kp("l_wrist", min_conf)
        else:
            s = self.kp("r_shoulder", min_conf)
            e = self.kp("r_elbow", min_conf)
            w = self.kp("r_wrist", min_conf)

        if s is None or e is None or w is None:
            return None

        v1 = s - e
        v2 = w - e

        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)

        if n1 < 1 or n2 < 1:
            return None

        cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        return float(np.degrees(np.arccos(cosang)))

    def wrist_below_elbow(self, side="left", min_conf=0.3):
        if side.startswith("l"):
            wrist = self.kp("l_wrist", min_conf)
            elbow = self.kp("l_elbow", min_conf)
        else:
            wrist = self.kp("r_wrist", min_conf)
            elbow = self.kp("r_elbow", min_conf)

        if wrist is None or elbow is None:
            return False

        return wrist[1] > elbow[1]


class Detector:
    def __init__(self, cfg):
        m = cfg["models"]
        self.device = m.get("device") or None
        self.conf = m.get("conf", 0.35)
        self.low_memory = bool(m.get("low_memory", False))
        self.imgsz = m.get("low_mem_imgsz", 480) if self.low_memory else m.get("imgsz", 640)
        self.tracker = m.get("tracker", "bytetrack.yaml")
        self._untracked_id = 0

        if self.low_memory:
            try:
                import cv2 as _cv2
                import torch as _torch
                _cv2.setNumThreads(2)
                _torch.set_num_threads(int(m.get("torch_threads", 4)))
            except Exception:
                pass

        self.pose = YOLO(m["pose"])
        self.detect = None if self.low_memory else YOLO(m["detect"])

        if self.low_memory:
            print(f"[detector] LOW-MEMORY mode: pose model only, imgsz={self.imgsz}. "
                  "The general object model is off; KE still runs its own phone detector (rules.KE.phone_model).")

    def _next_untracked(self):
        self._untracked_id -= 1
        return self._untracked_id

    def __call__(self, frame):
        return self._persons(frame), self._phones(frame)

    def _persons(self, frame):
        persons = []

        r = self.pose.track(
            frame,
            persist=True,
            classes=[PERSON_CLASS],
            conf=self.conf,
            imgsz=self.imgsz,
            tracker=self.tracker,
            device=self.device,
            verbose=False,
        )[0]

        if r.boxes is None or r.keypoints is None or len(r.boxes) == 0:
            return persons

        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        kpts = r.keypoints.data.cpu().numpy()

        tracked = r.boxes.id is not None
        if tracked:
            ids = r.boxes.id.cpu().numpy().astype(int)
        else:
            ids = np.array([self._next_untracked() for _ in range(len(boxes))])

        for i in range(len(boxes)):
            persons.append(Person(ids[i], boxes[i], confs[i], kpts[i], tracked=tracked))

        return persons

    def _phones(self, frame):
        phones = []

        if self.detect is None:
            return phones

        d = self.detect.predict(
            frame,
            classes=[PHONE_CLASS],
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]

        if d.boxes is None or len(d.boxes) == 0:
            return phones

        for b, c in zip(d.boxes.xyxy.cpu().numpy(), d.boxes.conf.cpu().numpy()):
            phones.append({
                "bbox": np.asarray(b, dtype=float),
                "conf": float(c),
            })

        return phones