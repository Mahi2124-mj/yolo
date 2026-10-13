"""Shared wrist-region crop used by BOTH the master-capture tool and the live
matcher, so the reference samples and the live queries are framed identically.
"""
import numpy as np


def wrist_crop(frame, person, side, scale=0.6, min_conf=0.3):
    """Square crop around one wrist. Returns (crop_bgr, (x1,y1,x2,y2)) or (None, None)."""
    W = person.kp(f"{side}_wrist", min_conf)
    if W is None:
        return None, None
    torso = person.torso_len(min_conf)
    r = int(max(scale * torso, 12))
    h, w = frame.shape[:2]
    x, y = int(W[0]), int(W[1])
    x1, y1 = max(0, x - r), max(0, y - r)
    x2, y2 = min(w, x + r), min(h, y + r)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None, None
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)
