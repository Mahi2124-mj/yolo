"""Fingertip / hand-visibility check using MediaPipe HandLandmarker (Tasks API).

Idea: YOLO COCO pose has no finger keypoints, so it can't tell a hand that is
INSIDE a pocket from one just hanging at the hip. But a hand inside a pocket
hides its fingers. So we crop the region around the wrist and ask MediaPipe
whether a hand (with fingertips) is visible there:

    fingertips visible  -> hand is OUT of the pocket  -> NOT a PO violation
    no hand detected    -> inconclusive (could be in a pocket, or just missed)

We therefore use it only as a *veto*: a clearly visible hand cancels a PO that
the geometry flagged. We never create a PO from "no hand seen" alone, because a
low-res stream can simply miss a real hand.
"""
import os

import numpy as np
import cv2

_DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "hand_landmarker.task")

# MediaPipe hand-landmark indices for the five fingertips.
FINGERTIPS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky

try:
    import mediapipe as mp
    from mediapipe.tasks import python as _mp_python
    from mediapipe.tasks.python import vision as _mp_vision
    _MP_OK = True
except Exception as _e:  # pragma: no cover
    _MP_OK = False
    _IMPORT_ERR = _e


class HandChecker:
    def __init__(self, model_path=None, num_hands=1, min_conf=0.3, upscale_to=256):
        self.ok = False
        self.upscale_to = upscale_to
        model_path = model_path or _DEFAULT_MODEL

        if not _MP_OK:
            print("[hands] mediapipe unavailable -> fingertip check disabled:", _IMPORT_ERR)
            return
        if not os.path.exists(model_path):
            print(f"[hands] model file not found ({model_path}) -> fingertip check disabled.")
            return
        try:
            opts = _mp_vision.HandLandmarkerOptions(
                base_options=_mp_python.BaseOptions(model_asset_path=model_path),
                num_hands=num_hands,
                min_hand_detection_confidence=min_conf,
                min_hand_presence_confidence=min_conf,
            )
            self.landmarker = _mp_vision.HandLandmarker.create_from_options(opts)
            self.ok = True
            print("[hands] MediaPipe HandLandmarker ready (fingertip veto enabled).")
        except Exception as e:
            print("[hands] failed to init HandLandmarker -> disabled:", e)

    def fingertips_at(self, frame, point, region_px):
        """Count fingertip landmarks visible in a crop centred on `point`.

        Returns:
            >=0  number of fingertips found inside the crop (0 = no hand detected)
            -1   checker unavailable / crop invalid (caller should treat as unknown)
        """
        if not self.ok or point is None or frame is None:
            return -1
        h, w = frame.shape[:2]
        r = int(max(region_px, 8))
        x, y = int(point[0]), int(point[1])
        x1, y1 = max(0, x - r), max(0, y - r)
        x2, y2 = min(w, x + r), min(h, y + r)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return -1

        ch, cw = crop.shape[:2]
        if max(ch, cw) < self.upscale_to:  # help detection on tiny hands
            s = self.upscale_to / float(max(ch, cw))
            crop = cv2.resize(crop, (int(cw * s), int(ch * s)), interpolation=cv2.INTER_CUBIC)

        rgb = np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        try:
            res = self.landmarker.detect(mp_img)
        except Exception:
            return -1
        if not res.hand_landmarks:
            return 0

        lm = res.hand_landmarks[0]
        return sum(1 for i in FINGERTIPS if 0.0 <= lm[i].x <= 1.0 and 0.0 <= lm[i].y <= 1.0)

    def detect_at_point(self, frame, point, region_px, upscale_to=320):
        """Detect hands in a small square crop centred on a known wrist keypoint.

        Zooming into the wrist (instead of scanning the whole lower body) makes the
        hand large in the input image, which is what MediaPipe needs to find small
        far-away hands on a ceiling/bullet camera.
        """
        if point is None:
            return []
        r = int(max(region_px, 12))
        x, y = int(point[0]), int(point[1])
        return self.detect_in_region(frame, (x - r, y - r, x + r, y + r),
                                     upscale_to=upscale_to)

    def detect_in_region(self, frame, box, upscale_to=384):
        """Detect visible hands inside a sub-region of the frame.

        The region is cropped and upscaled so MediaPipe actually has a chance to
        find small hands. Returns a list of hands in FRAME pixel coords:
            {"wrist", "tips", "n_tips", "landmarks" (21 points)}
        A hand here means the fingers are VISIBLE (i.e. NOT hidden in a pocket).
        """
        if not self.ok or frame is None:
            return []
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return []

        crop = frame[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        if max(ch, cw) < upscale_to:  # zoom in so tiny hands become detectable
            s = upscale_to / float(max(ch, cw))
            crop = cv2.resize(crop, (int(cw * s), int(ch * s)), interpolation=cv2.INTER_CUBIC)

        rgb = np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        try:
            res = self.landmarker.detect(mp_img)
        except Exception:
            return []
        if not res.hand_landmarks:
            return []

        rw, rh = (x2 - x1), (y2 - y1)   # normalized coords map straight onto the region
        out = []
        for lm in res.hand_landmarks:
            pts = [np.array([x1 + lm[i].x * rw, y1 + lm[i].y * rh]) for i in range(21)]
            tips = [pts[i] for i in FINGERTIPS
                    if -0.1 <= lm[i].x <= 1.1 and -0.1 <= lm[i].y <= 1.1]
            out.append({
                "wrist": pts[0],
                "tips": tips,
                "n_tips": len(tips),
                "landmarks": pts,    # full 21-point finger skeleton (frame coords)
            })
        return out
