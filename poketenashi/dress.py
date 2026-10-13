"""Company dress check - for ONE camera only (8 Oct 2026, the user: "dress code
only on the Material Gate camera, nowhere else"; everywhere else clothing colour
plays no part in any rule).

At the Material Gate the guards, drivers and visitors wear their own clothes;
the rules apply to the company's people only. Company shirts (the user's
examples, 8 Oct): royal-blue / navy / black polo, light-blue shirt, white, grey.
So a shirt counts as company when its colour is BLUE or NEUTRAL (white / grey /
black); a clearly coloured shirt (red, yellow, orange, green, pink, brown ...)
is not. Per track the verdict is a vote over several looks, so one odd frame
(shadow, sun) does not flip it.

Measured on the user's 7 examples (torso median, OpenCV Lab):
  blue polo   a 126-130  b 88-92      navy/black  L 9   a 128 b 120
  light blue  a 119-127  b 103-110    grey/white  a 125-129 b 124-126
"""
from collections import defaultdict, deque

import cv2
import numpy as np


def torso_lab(frame, p, mc=0.4):
    """Median Lab of the shirt: between the shoulders, from just below them to
    above the hips (the arms are cut away). None when not visible."""
    k = p.keypoints
    sh, hp = k[5:7], k[11:13]
    if min(sh[:, 2].min(), hp[:, 2].min()) < mc:
        return None
    x1, x2 = sorted([float(sh[0, 0]), float(sh[1, 0])])
    y1, y2 = float(sh[:, 1].mean()), float(hp[:, 1].mean())
    w, h = x2 - x1, y2 - y1
    if w < 10 or h < 10:
        return None
    x1, x2 = int(x1 + 0.2 * w), int(x2 - 0.2 * w)
    y1, y2 = int(y1 + 0.15 * h), int(y2 - 0.25 * h)
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    lab = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
    return np.median(lab, axis=0)


def is_company(lab, c):
    da, db = float(lab[1]) - 128.0, float(lab[2]) - 128.0
    if (da * da + db * db) ** 0.5 <= float(c.get("neutral_chroma", 12)):
        return True                                       # white / grey / black
    return db <= -float(c.get("blue_min_b", 8)) and abs(da) <= float(c.get("blue_max_a", 14))   # blue


class DressCode:
    def __init__(self, cfg):
        self.c = cfg or {}
        self.need = int(self.c.get("looks", 3))
        self.votes = defaultdict(lambda: deque(maxlen=int(self.c.get("window", 15))))
        self.last = {}

    def judged(self, frame, persons):
        """The people the rules may judge on this camera: company dress only.
        A person not yet seen enough is not judged until their dress is known."""
        out = []
        for p in persons:
            if not p.tracked:
                continue
            lab = torso_lab(frame, p)
            v = self.votes[p.track_id]
            if lab is not None:
                v.append(is_company(lab, self.c))
            if len(v) >= self.need:
                self.last[p.track_id] = sum(v) * 2 >= len(v)       # majority (ties: company)
            if self.last.get(p.track_id):
                out.append(p)
        if len(self.votes) > 500:
            live = {p.track_id for p in persons}
            for k in [k for k in self.votes if k not in live][:250]:
                self.votes.pop(k, None)
                self.last.pop(k, None)
        return out
