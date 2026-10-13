"""KE rule - "no phone use while walking / in the gangway".

A person is flagged only when a phone is actually SEEN in their hand.

How: for everyone whose hand is raised - in front of the belly or chest, or at
the ear - a crop around the raised hand(s) is cut from the full-resolution
frame, zoomed, and the phone detector runs on that crop. From this ceiling
camera a phone is only ~15-40 px, so on the whole frame the detector needed
imgsz 1280 to find it at all: that pass cost ~490 ms, 70% of every frame's CPU,
and was the main reason the dashboard reacted late. In the zoomed crop the
phone is at a size the model knows, for a fraction of the cost.

KE fires when the zoom has found a phone in at least `zoom_min_hits` frames of
the last `zoom_window_sec`, the person is walking or in the gangway, and
upright. A phone held in front of the body cannot be seen from behind, so a
back view only counts when the hand is at the ear (a call shows from behind).

Removed on 2026-09-26, and why:
  * the pose-only path - hands together at the chest => KE, no phone needed.
    Of the 33 KE alerts of that day checked by eye, ~25 were paper, trolleys,
    bags, gloves or clasped hands.
  * the full-frame 1280 pass - see above.
Measured on those alert clips: a phone plainly in the hand scored up to 0.68 in
the zoom (yolo11m); paper, trolleys and hands at most 0.49, and never in two
frames. A phone hidden in the fingers and seen from above is not found by a
COCO model at all - KE misses those rather than guess.
"""
import time
from collections import defaultdict, deque

import cv2
import numpy as np

# COCO-17 indices (same layout the pose model outputs).
NOSE, L_EAR, R_EAR = 0, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
KPT_CONF_MIN = 0.3

# ---- geometry (in torso lengths) --------------------------------------------
KE_EAR_NEAR = 0.50       # wrist within this*torso of a head point = at ear level
KE_SIDE_MIN = 0.10       # wrist this*torso to the SIDE of the nose (rejects face-touch)
KE_EAR_MAX_DROP = 0.35   # wrist no lower than this*torso below shoulders (rejects bend)
KE_WRIST_NEAR = 0.60     # phone within this*torso of a raised wrist = in that hand
KE_UPRIGHT_MIN = 0.15    # head must sit this*torso ABOVE the shoulder line
KE_RAISED_MIN = 0.05     # a wrist this*torso above the hip line is "raised"
# ---- timing -----------------------------------------------------------------
KE_LATCH_SEC = 3.0       # stay red this long after the phone was last seen ...
KE_HANDS_DOWN_SEC = 1.5  # ... or, while the hand stays raised (still using the
KE_MAX_GAP_SEC = 10.0    #     phone), up to this long after the last sighting
KE_BOX_HOLD_SEC = 0.6    # a found phone stays on the overlay this long
# ---- zoom defaults (config rules.KE.* overrides) ----------------------------
ZOOM_IMGSZ = 256         # crop's long side; 256 scored real phones best (vs 320/416)
ZOOM_CONF = 0.40
ZOOM_MIN_HITS = 2
ZOOM_WINDOW_SEC = 3.0
ZOOM_EVERY = 1
ZOOM_MAX_LONG = 0.5      # a "phone" box longer than this * torso is paper / a handle
# ---- ear call ---------------------------------------------------------------
# A phone at the ear is mostly hidden by the hand and the head: the zoom scored
# real calls 0.25-0.42 on the live call-pose samples (28 Sep). So while the hand
# is AT THE EAR, a phone box near the head counts from EAR_CONF - and a call
# needs the hand to stay there (a cap, a scratch, a glance at a watch are over
# in a second) plus either EAR_MIN_HITS such boxes or a phone seen in the hand
# a moment before (the pre-26-Sep ear path's "phone memory").
EAR_CONF = 0.25
EAR_MIN_HITS = 2
EAR_WINDOW_SEC = 4.0
EAR_HEAD_NEAR = 0.60     # phone box centre within this*torso of the nose / an ear
CALL_SEC = 1.5           # the hand has been at the ear in >= CALL_RATIO of this long
CALL_RATIO = 0.7
PHONE_MEMORY_SEC = 8.0
PHONE_CLASS = 67


_MODELS = {}


def _shared_model(path):
    """One phone model for every camera: KE only calls predict() (no tracking
    state), and all cameras are judged from the one detection loop - so N
    cameras do not hold N copies of yolo11m in memory."""
    m = _MODELS.get(path)
    if m is None:
        from ultralytics import YOLO
        m = _MODELS[path] = YOLO(path)
    return m


# 12 Oct (user: 'a cup, a laptop... it must know the basic things'): every everyday hand-held
# COCO object - umbrella, handbag, bottle, wine glass, cup, knife, laptop, mouse, remote,
# keyboard, cell phone, book (paper / notebook / file), scissors
HAND_OBJECTS = [25, 26, 39, 40, 41, 43, 63, 64, 65, 66, 67, 73, 76]
VERIFY_WHY = {"why": ""}                      # why the last verify_phone_in_hand() said no (logged)
PHONE_MAX_LONG_TORSO = 0.45                  # a 'phone' longer than this is a tablet / laptop / file               # laptop, remote, cell phone, book (paper shows as book)


def verify_phone_in_hand(det, frame, p, min_conf):
    """True when the object model finds a cell phone (>= min_conf) at one of p's
    wrists, in p's own padded crop, AND it is the strongest hand-held object there."""
    k = p.keypoints
    try:
        x1, y1, x2, y2 = [float(v) for v in p.bbox]
        H, W = frame.shape[:2]
        pw, ph = 0.4 * (x2 - x1), 0.1 * (y2 - y1)
        cx1, cy1 = max(0, int(x1 - pw)), max(0, int(y1 - ph))
        crop = frame[cy1:min(H, int(y2 + ph)), cx1:min(W, int(x2 + pw))]
        if crop.size == 0:
            return False
        sh = [k[i][:2] for i in (5, 6) if float(k[i][2]) >= KPT_CONF_MIN]
        hp = [k[i][:2] for i in (11, 12) if float(k[i][2]) >= KPT_CONF_MIN]
        torso = (float(np.linalg.norm(np.mean(sh, axis=0) - np.mean(hp, axis=0)))
                 if sh and hp else 0.3 * (y2 - y1))
        wr = [np.array([float(k[i][0]) - cx1, float(k[i][1]) - cy1]) for i in (9, 10)
              if float(k[i][2]) >= KPT_CONF_MIN]
        if not wr:
            return False
        r = det.predict(crop, classes=HAND_OBJECTS, imgsz=640, conf=0.2, verbose=False)[0]
        held = []
        for (a, b, c, d), cls, conf in zip(r.boxes.xyxy.cpu().numpy().tolist(),
                                           r.boxes.cls.cpu().numpy().tolist(),
                                           r.boxes.conf.cpu().numpy().tolist()):
            pc = np.array([(a + c) / 2.0, (b + d) / 2.0])
            if min(float(np.linalg.norm(pc - w)) for w in wr) <= 0.6 * torso:
                held.append((int(cls), float(conf), (a, b, c, d)))
        _shadow_score(crop, held, wr, torso)
        if not held:
            VERIFY_WHY["why"] = "nothing in the hand"
            return False
        best = max(held, key=lambda h: h[1])
        names = getattr(det, "names", {}) or {}
        if best[0] != 67 or best[1] < min_conf:
            VERIFY_WHY["why"] = f"in hand: {names.get(best[0], best[0])} {best[1]:.2f}"
            return False                       # the clearest thing in the hand is not a phone
        a, b, c, d = best[2]
        if max(c - a, d - b) > PHONE_MAX_LONG_TORSO * torso:
            VERIFY_WHY["why"] = f"phone {best[1]:.2f} too big ({max(c - a, d - b) / torso:.2f} torso)"
            return False                       # too big for a phone: a tablet / laptop / file
        for cls, conf, box in held:            # the same thing also seen as a cup / laptop / book...
            if cls != 67 and conf >= 0.25 and _box_iou(box, best[2]) >= 0.3:
                VERIFY_WHY["why"] = f"phone {best[1]:.2f} is also a {names.get(cls, cls)} {conf:.2f}"
                return False
        VERIFY_WHY["why"] = f"phone {best[1]:.2f}"
        return True
    except Exception:
        return True                            # never let the check itself hide a KE


# 13 Oct 2026 (user: "option 2 - run it quietly, show me the result tomorrow"): the KE
# phone CHECKER (tools/agent/ke_phone_fit.py, trained on the user's OK/NG KE) runs in
# SHADOW only - it never changes an alert. Per verify it scores every "cell phone" box
# at a wrist (crop as in ke_phone_data.py) and says whether it WOULD fire.
SHADOW = {"cfg": None, "net": None, "fire": False, "p": 0.0, "n": 0}


def _shadow_score(crop, held, wr, torso):
    SHADOW.update(fire=False, p=0.0, n=0)
    c = SHADOW["cfg"]
    if not c or not c.get("path"):
        return
    try:
        if SHADOW["net"] is None:
            import torch
            import torchvision.models as tvm
            net = tvm.mobilenet_v3_small(weights=None)
            net.classifier[3] = torch.nn.Linear(net.classifier[3].in_features, 1)
            import os
            path = c["path"] if os.path.isabs(c["path"]) else os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), c["path"])
            net.load_state_dict(torch.load(path, map_location="cpu"))
            SHADOW["net"] = net.eval()
        import torch
        ims = []
        for cls, conf, (a, b, cc, d) in held:
            if cls != 67 or conf < float(c.get("det_min", 0.15)) or max(cc - a, d - b) > PHONE_MAX_LONG_TORSO * torso:
                continue
            pc = np.array([(a + cc) / 2.0, (b + d) / 2.0])
            side = 1.1 * max(cc - a, d - b, 0.15 * torso)
            x1, y1 = int(max(0, pc[0] - side)), int(max(0, pc[1] - side))
            x2, y2 = int(min(crop.shape[1], pc[0] + side)), int(min(crop.shape[0], pc[1] + side))
            im = crop[y1:y2, x1:x2]
            if im.size:
                im = cv2.cvtColor(cv2.resize(im, (112, 112), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                ims.append(im.astype(np.float32) / 255.0)
        SHADOW["n"] = len(ims)
        if not ims:
            return
        mean, std = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
        a = (np.stack(ims) - mean) / std
        with torch.inference_mode():
            t = torch.from_numpy(a.transpose(0, 3, 1, 2).copy())
            p = (torch.sigmoid(SHADOW["net"](t)) + torch.sigmoid(SHADOW["net"](torch.flip(t, dims=[3])))) / 2
        SHADOW["p"] = float(p.max())
        SHADOW["fire"] = SHADOW["p"] >= float(c.get("thr", 0.5))
    except Exception as e:                     # the shadow must never touch the live KE
        SHADOW["p"] = -1.0
        VERIFY_WHY["shadow_err"] = repr(e)[:80]


def _box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def _valid(kp):
    return float(kp[2]) >= KPT_CONF_MIN and (float(kp[0]) != 0.0 or float(kp[1]) != 0.0)


def _torso(k):
    pts = []
    for a, b in ((L_SHOULDER, R_SHOULDER), (L_HIP, R_HIP)):
        ka, kb = k[a], k[b]
        if _valid(ka) and _valid(kb):
            pts.append((ka[:2] + kb[:2]) / 2.0)
        elif _valid(ka):
            pts.append(ka[:2])
        elif _valid(kb):
            pts.append(kb[:2])
        else:
            return None
    if len(pts) < 2:
        return None
    d = float(np.linalg.norm(pts[0] - pts[1]))
    return d if d > 1.0 else None


def phone_call_pose(k):
    """Hand raised to the ear, to the side of the face, above the shoulder line."""
    torso = _torso(k)
    if torso is None:
        return False
    heads = [k[i] for i in (L_EAR, R_EAR, NOSE) if _valid(k[i])]
    if not heads:
        return False
    nose = k[NOSE]
    shoulders = [k[i] for i in (L_SHOULDER, R_SHOULDER) if _valid(k[i])]
    sh_y = float(np.mean([float(s[1]) for s in shoulders])) if shoulders else None
    for wi in (L_WRIST, R_WRIST):
        w = k[wi]
        if float(w[2]) < 0.20 or (w[0] == 0.0 and w[1] == 0.0):
            continue
        at_head = any(float(np.hypot(float(w[0]) - float(h[0]),
                                     float(w[1]) - float(h[1]))) <= KE_EAR_NEAR * torso
                      for h in heads)
        if not at_head:
            continue
        if _valid(nose) and abs(float(w[0]) - float(nose[0])) < KE_SIDE_MIN * torso:
            continue
        if sh_y is not None and float(w[1]) > sh_y + KE_EAR_MAX_DROP * torso:
            continue
        return True
    return False


def both_hands_at_head(k, near=0.9):
    """BOTH wrists up at the head (above the shoulder line, within `near`
    torso of the nose / ears / eyes): fixing a cap, a helmet or the hair,
    wiping the face, stretching. A call holds the phone with one hand. On the
    night of 6-7 Oct three KE alerts were people adjusting their caps with both
    hands - the zoom found a "phone" in the cap's brim."""
    torso = _torso(k)
    if torso is None:
        return False
    heads = [k[i] for i in (NOSE, 1, 2, L_EAR, R_EAR) if _valid(k[i])]
    shoulders = [k[i] for i in (L_SHOULDER, R_SHOULDER) if _valid(k[i])]
    if not heads or not shoulders:
        return False
    sh_y = float(np.mean([float(s[1]) for s in shoulders]))
    for wi in (L_WRIST, R_WRIST):
        w = k[wi]
        if not _valid(w) or float(w[1]) > sh_y:
            return False
        if min(float(np.hypot(float(w[0]) - float(h[0]), float(w[1]) - float(h[1]))) for h in heads) > near * torso:
            return False
    return True


def phone_use_pose(k):
    """Both hands together in front of the belly/chest, face toward the camera,
    at least one elbow bent - the posture of looking at something held in the
    hands. NOT evidence of a phone (paper, a glove, clasped hands look the same);
    PO uses it to leave such people alone. Thresholds are in torso lengths."""
    torso = _torso(k)
    if torso is None:
        return False
    if max(float(k[NOSE][2]), float(k[1][2]), float(k[2][2])) < 0.5:     # face visible
        return False
    wl, wr = k[L_WRIST], k[R_WRIST]
    if not (_valid(wl) and _valid(wr) and _valid(k[L_HIP]) and _valid(k[R_HIP])):
        return False
    hip = (k[L_HIP][:2] + k[R_HIP][:2]) / 2.0
    if float(np.hypot(*(wl[:2] - wr[:2]))) >= 0.55 * torso:               # hands together
        return False
    for w in (wl, wr):                                                    # belly to chest
        up = (float(hip[1]) - float(w[1])) / torso
        if not 0.1 <= up <= 0.7:
            return False
    if min(abs(float(w[0]) - float(hip[0])) for w in (wl, wr)) > 0.3 * torso:   # in front
        return False
    elbows = []
    for s, e, w in ((L_SHOULDER, L_ELBOW, L_WRIST), (R_SHOULDER, R_ELBOW, R_WRIST)):
        if _valid(k[s]) and _valid(k[e]) and _valid(k[w]):
            a, b, c = k[s][:2], k[e][:2], k[w][:2]
            ba, bc = a - b, c - b
            cos = float(ba @ bc) / (float(np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6)
            elbows.append(float(np.degrees(np.arccos(max(-1.0, min(1.0, cos))))))
    return bool(elbows) and min(elbows) <= 115.0


def raised_wrists(k):
    """(wrists held up, torso): wrists above the hip line - in front of the
    belly or chest, or at the head - which is where a phone in use is. A hand
    hanging at the side or swinging past the hip is not raised."""
    torso = _torso(k)
    if torso is None:
        return [], None
    hips = [k[i] for i in (L_HIP, R_HIP) if _valid(k[i])]
    if not hips:
        return [], torso
    hip_y = float(np.mean([float(h[1]) for h in hips]))
    return [np.asarray(k[i][:2], float) for i in (L_WRIST, R_WRIST)
            if _valid(k[i]) and hip_y - float(k[i][1]) >= KE_RAISED_MIN * torso], torso


def zoom_box(wrists, torso, w, h):
    """The crop around the raised hand(s): both hands, and the room above them
    where the phone is (held in front of the chest, or against the ear)."""
    xs = [float(p[0]) for p in wrists]
    ys = [float(p[1]) for p in wrists]
    cx = float(np.mean(xs))
    half = max(0.55 * torso, 0.5 * (max(xs) - min(xs)) + 0.35 * torso)
    x1, x2 = int(max(0, cx - half)), int(min(w, cx + half))
    y1, y2 = int(max(0, min(ys) - 0.55 * torso)), int(min(h, max(ys) + 0.30 * torso))
    if x2 - x1 < 12 or y2 - y1 < 12:
        return None
    return x1, y1, x2, y2


def _upright(k):
    heads = [k[i] for i in (NOSE, L_EAR, R_EAR) if _valid(k[i])]
    shs = [k[i] for i in (L_SHOULDER, R_SHOULDER) if _valid(k[i])]
    if not heads or not shs:
        return True                                # missing keypoints -> don't block
    hy = min(float(h[1]) for h in heads)
    sy = float(np.mean([float(s[1]) for s in shs]))
    torso = _torso(k) or 1.0
    return hy < sy - KE_UPRIGHT_MIN * torso


def _feet(k, box):
    pts = [k[i][:2] for i in (15, 16) if _valid(k[i])]
    if pts:
        return np.mean(pts, axis=0)
    x1, _, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, y2])


class KEDetector:
    """Per-track KE state: zoomed phone sightings, and the evidence-held latch."""

    def __init__(self, cfg, ke_cfg):
        c = ke_cfg or {}
        self.walk_ps = float(c.get("walk_speed_ps", 45.0))      # px/s = "moving"
        self.zoom_imgsz = int(c.get("zoom_imgsz", ZOOM_IMGSZ))
        self.zoom_conf = float(c.get("zoom_conf", ZOOM_CONF))
        self.min_hits = int(c.get("zoom_min_hits", ZOOM_MIN_HITS))
        self.window = float(c.get("zoom_window_sec", ZOOM_WINDOW_SEC))
        self.single_conf = float(c.get("zoom_single_conf", 0) or 0)
        # 10 Oct (user): before KE fires, the general object model checks the hand
        # in the person's own crop - a phone there (>= verify_phone_conf) or no KE.
        # On 10 Oct the 5 false KE (paper / card / nothing) had no phone, the 2 real ones had.
        self.verify_conf = c.get("verify_phone_conf")
        self.verify_every = float(c.get("verify_every_sec", 0.5))
        SHADOW["cfg"] = c.get("shadow_checker")    # 13 Oct: the phone checker, shadow only
        self._shadow = {}                          # tid -> (live said yes, checker said yes) as last logged
        self._verified = {}                    # tid -> time a phone was confirmed
        self._vcheck = {}                      # tid -> (time, result) of the last check
        self.every = max(1, int(c.get("zoom_every_n", ZOOM_EVERY)))
        self.max_long = float(c.get("zoom_max_long_torso", ZOOM_MAX_LONG))
        self.det = _shared_model(c.get("phone_model", "yolov8s.pt"))
        self.phone_cls = [i for i, n in self.det.names.items()
                          if n in ("cell phone", "phone", "mobile phone")] or [PHONE_CLASS]
        self._i = 0
        self._hits = defaultdict(deque)        # tid -> [(t, best conf)] frames a phone was found
        self._per_person = defaultdict(list)   # tid -> [(t, box)] phones found (overlay)
        self._phone_seen = defaultdict(float)  # tid -> last time a phone was found
        self._latch = defaultdict(float)       # tid -> last time KE fired
        self._raised_t = defaultdict(float)    # tid -> last time a hand was raised
        self.hands_down_sec = float(c.get("hold_hands_down_sec", KE_HANDS_DOWN_SEC))
        self.max_gap_sec = float(c.get("hold_max_gap_sec", KE_MAX_GAP_SEC))
        self.ear_conf = float(c.get("ear_conf", EAR_CONF))
        self.ear_min_hits = int(c.get("ear_min_hits", EAR_MIN_HITS))
        self.ear_window = float(c.get("ear_window_sec", EAR_WINDOW_SEC))
        self.call_sec = float(c.get("call_sec", CALL_SEC))
        self.call_ratio = float(c.get("call_ratio", CALL_RATIO))
        self.phone_memory = float(c.get("phone_memory_sec", PHONE_MEMORY_SEC))
        self._ear_hits = defaultdict(deque)    # tid -> times a phone was found at the ear
        self._call_ev = defaultdict(deque)     # tid -> [(t, hand at the ear?)] over call_sec
        self._ear_on = {}                      # tid -> KE fired on a call, hand still up
        # both hands up at the head (both_hands_at_head) = no phone evidence
        # from that frame: off unless rules.KE.both_hands_head_veto is set
        self.head_veto = bool(c.get("both_hands_head_veto", False))
        # ear-call phone threshold for a person seen from behind (default: the
        # same as facing - i.e. off)
        self.ear_back_conf = float(c.get("ear_back_conf", self.ear_conf))

    def _zoom(self, frame, p, call=False):
        """[(box, conf, near_head)] phones found in a zoom of p's raised hands,
        in frame coordinates. Only boxes in one of THOSE hands count - not a
        phone lying on a shelf behind the person. With the hand at the ear
        (`call`) weaker boxes come back too (ear_conf), for the ear path."""
        wrists, torso = raised_wrists(p.keypoints)
        if not wrists:
            return []
        h, w = frame.shape[:2]
        box = zoom_box(wrists, torso, w, h)
        if box is None:
            return []
        x1, y1, x2, y2 = box
        crop = frame[y1:y2, x1:x2]
        s = self.zoom_imgsz / float(max(crop.shape[:2]))
        crop = cv2.resize(crop, None, fx=s, fy=s,
                          interpolation=cv2.INTER_CUBIC if s > 1.0 else cv2.INTER_AREA)
        r = self.det.predict(crop, classes=self.phone_cls, imgsz=self.zoom_imgsz,
                             conf=min(self.zoom_conf, self.ear_conf) if call else self.zoom_conf,
                             verbose=False)[0]
        found = []
        if r.boxes is None or len(r.boxes) == 0:
            return found
        k = p.keypoints
        heads = [np.asarray(k[i][:2], float) for i in (NOSE, L_EAR, R_EAR) if _valid(k[i])]
        for (a, b, c, d), conf in zip(r.boxes.xyxy.cpu().numpy().tolist(),
                                      r.boxes.conf.cpu().numpy().tolist()):
            bx = (x1 + a / s, y1 + b / s, x1 + c / s, y1 + d / s)
            # A phone is small: 0.06-0.46 torso long on the real phones of the
            # 26-28 Sep alerts. Paper, a notebook or a trolley handle boxed as a
            # "phone" is bigger: 0.65-0.82 (ID81, ID8217, ID13). Not at the ear:
            # a phone pressed to the ear near the camera, with the hand round
            # it, boxed at 0.86 (ID760, 26 Sep).
            if not call and max(bx[2] - bx[0], bx[3] - bx[1]) > self.max_long * torso:
                continue
            pc = np.array([(bx[0] + bx[2]) / 2.0, (bx[1] + bx[3]) / 2.0])
            if min(float(np.linalg.norm(pc - q)) for q in wrists) <= KE_WRIST_NEAR * torso:
                near_head = bool(heads) and min(float(np.linalg.norm(pc - hd)) for hd in heads) \
                    <= EAR_HEAD_NEAR * torso
                found.append((bx, float(conf), near_head))
        return found

    @staticmethod
    def _owned(found_all, persons):
        """Each phone box stays only with the person whose wrist is nearest to
        it (in their own torso lengths). In a group one person's phone is also
        inside the zoom of the one next to them, and both got KE for it - T7
        for T6's phone in the 28 Sep live test."""
        wr = {}
        for p in persons:
            k = p.keypoints
            t = _torso(k) or max(1.0, (p.bbox[3] - p.bbox[1]) * 0.3)
            wr[p.track_id] = ([np.asarray(k[i][:2], float) for i in (L_WRIST, R_WRIST)
                               if _valid(k[i])], t)

        def dist(tid, pc):
            ws, t = wr.get(tid, ([], 1.0))
            return min((float(np.linalg.norm(pc - w)) / t for w in ws), default=1e9)

        out = {}
        for tid, found in found_all.items():
            keep = []
            for item in found:
                bx = item[0]
                pc = np.array([(bx[0] + bx[2]) / 2.0, (bx[1] + bx[3]) / 2.0])
                mine = dist(tid, pc)
                if all(dist(q, pc) >= mine for q in wr if q != tid):
                    keep.append(item)
            out[tid] = keep
        return out

    def owned_phone_boxes(self):
        """The phones KE found in people's hands just now, in the detector's phone
        format, for the overlay: a KE box on screen then shows the phone it
        was raised for - and a phone that was found but not enough shows too."""
        now = time.time()
        return [{"bbox": np.asarray(b, float), "conf": 1.0}
                for lst in self._per_person.values() for t, b in lst
                if now - t <= KE_BOX_HOLD_SEC]

    def _log_blocked(self, frame, p, tid):
        """12 Oct (user: 'not one KE since the change'): a KE the phone check stopped is
        logged once per person and saved (person crop) to D:/poketenashi-data/ke_blocked/<day>/
        so it can be checked whether real phone users are being missed."""
        done = self.__dict__.setdefault("_blocked_logged", set())
        if tid in done:
            return
        done.add(tid)
        why = VERIFY_WHY.get("why", "")
        try:
            import os
            d = os.path.join("D:/poketenashi-data/ke_blocked", time.strftime("%Y%m%d"))
            os.makedirs(d, exist_ok=True)
            x1, y1, x2, y2 = [int(v) for v in p.bbox]
            H, W = frame.shape[:2]
            pw, ph = int(0.3 * (x2 - x1)), int(0.08 * (y2 - y1))
            crop = frame[max(0, y1 - ph):min(H, y2 + ph), max(0, x1 - pw):min(W, x2 + pw)]
            safe = "".join(ch if ch.isalnum() or ch in ".-" else "_" for ch in why)[:40]
            cv2.imwrite(os.path.join(d, f"{time.strftime('%H%M%S')}_ID{tid}_{safe}.jpg"), crop)
            print(f"[KE] stopped by the phone check: ID{tid} ({why})", flush=True)
        except Exception:
            pass

    def _log_shadow(self, frame, p, tid, live):
        """13 Oct (shadow): one row per person per change of (live, checker) to
        D:/poketenashi-data/ke_shadow/<day>/shadow.csv + the person crop. Never alters KE."""
        if not SHADOW["cfg"]:
            return
        try:
            state = (live, bool(SHADOW["fire"]))
            old = self._shadow.get(tid)
            if old is not None and (old == state or (old[0] >= state[0] and old[1] >= state[1])):
                return                                 # nothing new for this person
            self._shadow[tid] = (max(state[0], old[0]) if old else state[0], max(state[1], old[1]) if old else state[1])
            import os
            d = os.path.join("D:/poketenashi-data/ke_shadow", time.strftime("%Y%m%d"))
            os.makedirs(d, exist_ok=True)
            x1, y1, x2, y2 = [int(v) for v in p.bbox]
            H, W = frame.shape[:2]
            pw, ph = int(0.3 * (x2 - x1)), int(0.08 * (y2 - y1))
            crop = frame[max(0, y1 - ph):min(H, y2 + ph), max(0, x1 - pw):min(W, x2 + pw)]
            ts = time.strftime("%H%M%S")
            img = f"{ts}_ID{tid}_live{int(live)}_chk{int(SHADOW['fire'])}.jpg"
            cv2.imwrite(os.path.join(d, img), crop)
            new = not os.path.exists(os.path.join(d, "shadow.csv"))
            with open(os.path.join(d, "shadow.csv"), "a", encoding="utf-8") as f:
                if new:
                    f.write("time,epoch,track,live_fires,checker_fires,checker_p,phone_boxes,live_why,image\n")
                why = VERIFY_WHY.get("why", "").replace(",", ";")
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{time.time():.0f},{tid},{int(live)},{int(SHADOW['fire'])},"
                        f"{SHADOW['p']:.2f},{SHADOW['n']},{why},{img}\n")
        except Exception:
            pass

    def _prune(self, live):
        for d in (self._hits, self._per_person, self._phone_seen, self._latch, self._raised_t,
                  self._ear_hits, self._call_ev, self._ear_on, self._verified, self._vcheck):
            for k in [k for k in d if k not in live]:
                d.pop(k, None)

    def check(self, persons, phones, frame, engine, zones):
        """Return the set of track_ids currently violating KE."""
        self._i += 1
        now = time.time()
        self._prune({p.track_id for p in persons})
        firing = set()
        capfix = {p.track_id for p in persons if self.head_veto and both_hands_at_head(p.keypoints)}
        calls = {p.track_id: phone_call_pose(p.keypoints) and p.track_id not in capfix for p in persons}
        found_all = {}
        if frame is not None:
            for p in persons:
                if p.track_id in capfix:
                    continue                           # both hands at the head: not a phone
                if (self._i + p.track_id) % self.every == 0:
                    found_all[p.track_id] = self._zoom(frame, p, calls[p.track_id])
            found_all = self._owned(found_all, persons)
        for p in persons:
            tid, k, box = p.track_id, p.keypoints, p.bbox
            hits = self._hits[tid]
            if raised_wrists(k)[0]:
                self._raised_t[tid] = now
            call = calls[tid]
            facing_now = max(float(k[NOSE][2]), float(k[1][2]), float(k[2][2])) >= 0.5
            if tid in found_all:
                found = found_all[tid]
                strong = [(b, c) for b, c, _ in found if c >= self.zoom_conf]
                # seen from behind, a weak "phone" at the head is mostly the
                # background (7 Oct 04:07: a guard's back, the blue kiosk board
                # behind his head at 0.31); a call is then taken only on a
                # clear phone (rules.KE.ear_back_conf). Facing us, the weak
                # threshold stays: a real call hides the phone (0.25-0.42).
                ear_min = self.ear_conf if (facing_now or self._ear_on.get(tid)) else self.ear_back_conf
                ear = [(b, c) for b, c, near_head in found
                       if call and near_head and c >= ear_min]
                if strong:
                    hits.append((now, max(c for _, c in strong)))
                    self._phone_seen[tid] = now
                if ear:
                    self._ear_hits[tid].append(now)
                if strong or ear:
                    self._per_person[tid] = [(now, b) for b, _ in (strong or ear)]
            while hits and now - hits[0][0] > self.window:
                hits.popleft()
            phone_seen = len(hits) >= self.min_hits
            # 10 Oct (user: "fire as soon as the phone is confirmed"): one clear
            # phone (>= zoom_single_conf) is enough - paper, hands and trolleys
            # peaked at 0.49 (zoom_conf note); at a low frame rate the second
            # frame often never came within zoom_window_sec.
            if not phone_seen and self.single_conf and hits and max(c for _, c in hits) >= self.single_conf:
                phone_seen = True
            # The zone gate (rules.py) hands KE only people in the gangway or
            # the side gangways; in_zone says which. Without a gate, the aisle.
            in_gangway = getattr(p, "in_zone", None)
            if in_gangway is None:
                in_gangway = bool(zones is not None and zones.has("gangway")
                                  and zones.contains("gangway", _feet(k, box)))
            walking = engine.speed_ps(tid) >= self.walk_ps
            ke_position = (walking or in_gangway) and _upright(k)
            # A phone in front of the body is hidden from behind: a "phone" at
            # the hand of a back view is something else (4 of 5 overnight KE
            # alerts on 24 Sep were people seen from behind pushing trolleys).
            # A call at the ear shows from behind, so that one still counts.
            facing = max(float(k[NOSE][2]), float(k[1][2]), float(k[2][2])) >= 0.5
            # Ear call: the hand kept at the ear, and a phone there (weakly,
            # EAR_MIN_HITS times) or seen in the hand a moment before.
            ce = self._call_ev[tid]
            ce.append((now, call))
            while ce and now - ce[0][0] > self.call_sec:
                ce.popleft()
            call_held = len(ce) >= 3 and sum(1 for _, v in ce if v) >= self.call_ratio * len(ce)
            eh = self._ear_hits[tid]
            while eh and now - eh[0] > self.ear_window:
                eh.popleft()
            ear_call = call_held and (len(eh) >= self.ear_min_hits
                                      or now - self._phone_seen.get(tid, -1e9) <= self.phone_memory)
            if not call_held:
                self._ear_on.pop(tid, None)
            fire = ke_position and ((phone_seen and (facing or call)) or ear_call)
            if fire and ear_call:
                self._ear_on[tid] = True
            # Hold on the EVIDENCE, never on nothing. Once KE has fired, it
            # stays red while the phone keeps being found (and KE_LATCH_SEC
            # after), and also while the hand stays up at the chest or ear -
            # the phone is not found in every frame (turned edge-on, covered by
            # the thumb), and the box used to blink off in between. Hands down
            # for hands_down_sec, or no sighting for max_gap_sec, ends it. A call
            # holds for as long as the hand stays at the ear: the phone there is
            # hidden most of the time, and the call does not end when it is.
            if (fire and self.verify_conf is not None and frame is not None
                    and not self._latch.get(tid) and tid not in self._verified):
                last = self._vcheck.get(tid)
                if last is None or now - last[0] >= self.verify_every:
                    last = (now, verify_phone_in_hand(self.det, frame, p, float(self.verify_conf)))
                    self._vcheck[tid] = last
                    self._log_shadow(frame, p, tid, bool(last[1]))
                if last[1]:
                    self._verified[tid] = now
                else:
                    fire = False                       # no phone in the hand: not (yet) KE
                    self._log_blocked(frame, p, tid)
            if fire:
                self._latch[tid] = now
            elif ke_position and self._latch.get(tid, 0.0) > 0.0:
                since_phone = now - self._phone_seen.get(tid, 0.0)
                hands_up = now - self._raised_t.get(tid, 0.0) <= self.hands_down_sec
                if (since_phone <= KE_LATCH_SEC or (hands_up and since_phone <= self.max_gap_sec)
                        or self._ear_on.get(tid)):
                    fire = True
            if fire:
                firing.add(tid)
        return firing
