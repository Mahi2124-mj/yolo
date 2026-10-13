"""The five Poketenashi rule checkers.

Each rule looks at the current frame's detections (+ a short motion history)
and decides, per tracked person, whether they are violating. A violation only
"fires" after it persists for `min_frames` consecutive frames, which kills most
single-frame false positives from noisy keypoints.
"""
import os
import time
from collections import defaultdict, deque

import cv2
import numpy as np

from .hands import HandChecker
from .na import NADetector
from .po_crop import wrist_crop
from .ke import KEDetector
from . import po_model, ruleinfo


_HANDS = {}            # min_conf -> the shared MediaPipe HandChecker
_REPOSE = {}           # pose weights -> the shared second pose model (PO evidence_repose)
_PO_REFS = {}           # the user's OK / NG PO masters (po_refs.py), one set per process
_PO_REFS_META = {}      # key -> folder signature / last check (9 Oct: masters reload)


def _angle(a, b, c):
    """Interior angle at vertex b (degrees) for points a-b-c. ~180 = straight limb."""
    ba, bc = a - b, c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


class RuleEngine:
    def __init__(self, cfg, zones):
        self.cfg = cfg
        self.rcfg = cfg["rules"]
        self.enabled = set(self.rcfg.get("enabled", []))
        self.zones = zones

        self.history = defaultdict(lambda: deque(maxlen=30))  # track_id -> recent (time, centre)
        self.streak = defaultdict(int)                        # (track_id, rule) -> consecutive frames
        self.shi_state = defaultdict(lambda: {"pause": 0, "confirmed": False})
        self.po_cand = {}
        self._po_refsim = {}                                  # (tid, side) -> (frame, similarity to OK PO refs, ref name)                                     # tid -> best hand-at-hip look this frame (web.py reads + clears)
        self._warned = set()

        # Head-yaw history (nose vs shoulders) for SHI "looked around / confirmed".
        hw = int(self.rcfg.get("SHI", {}).get("head_window", 25))
        self.head_hist = defaultdict(lambda: deque(maxlen=hw))

        # Wrist movement: body-relative position history per (track_id, side) for the
        # PO "is the hand parked (pocket) or moving (out)?" signal, plus an on-screen trail.
        ww = int(self.rcfg.get("PO", {}).get("wrist_move_window", 15))
        self.wrist_hist = defaultdict(lambda: deque(maxlen=ww))   # (tid, side) -> body-relative [x,y]
        self.wrist_trail = defaultdict(lambda: deque(maxlen=ww))  # (tid, side) -> absolute [x,y] for drawing

        # NA is Naname = DIAGONAL: the company rule is "never cross roads or
        # walkways diagonally", which is a question of heading, not of position
        # or speed. RUN stays its own rule - a slow diagonal is still NA, and a
        # sprint straight down the aisle is RUN, not NA.
        # rules.NA.per_zone {zone: {NA settings}}: one detector per zone, each
        # with its own geometry (the Material Gate: its walkway and its zebra
        # crossing, 7 Oct). Without it, the one gangway as before.
        na_c = self.rcfg.get("NA") or {}
        per = na_c.get("per_zone") or {}
        base_na = {k: v for k, v in na_c.items() if k != "per_zone"}
        self.na_list = ([NADetector({"rules": {"NA": dict(base_na, **(pz or {}), zone=z)}}, zones)
                         for z, pz in per.items()] if per else [NADetector(cfg, zones)])
        self.na = self.na_list[0]

        # KE (phone use while walking / in the gangway): fires only on a phone
        # actually found in a zoomed crop of the person's raised hands (see
        # poketenashi/ke.py). Loads its own phone model and keeps its own per-track
        # state. Only built when enabled so it costs nothing otherwise.
        self.ke = KEDetector(cfg, self.rcfg.get("KE", {})) if "KE" in self.enabled else None

        # Zone gate: the walking rules judge only people whose FEET are inside
        # this zone (the gangway). The camera also sees the racks and the machines
        # either side, and people working there - standing, carrying, gloved -
        # were being judged by rules written for people walking the aisle.
        # TE / SHI / RESTRICTED keep their own zones and are not gated here; NA
        # already looks at the gangway only. `zone` may list several zones (the
        # aisle and the side gangways under the camera): inside any of them counts.
        zg = self.rcfg.get("zone_gate") or {}
        self.gate_zones = self._gate_zones(zg)
        self.gate_rules = set(zg.get("rules", ["PO", "KE", "RUN", "FALL"]))
        self.gate_margin = float(zg.get("margin_px", 0.0))
        if zg.get("zone") and not self.gate_zones:
            print(f"[rules] zone_gate: zone {zg.get('zone')} is not drawn - judging everyone")

        # Debug: when on, rule_PO records per-person measurements for the overlay.
        self.debug = bool(cfg.get("output", {}).get("debug", False))
        self.po_dbg = {}
        self._po_seen = {}          # tid -> last time the pocket test passed
        self._po_ev = defaultdict(deque)   # tid -> [(t, pocket test passed?)] over confirm_sec
        self._po_on = {}            # tid -> PO confirmed and currently shown

        # MediaPipe throttle: run it every Nth frame and reuse cached hands between.
        self._frame_i = 0
        self._hands_cache = {}  # track_id -> last detected hands (frame coords)
        self._crops_dbg = {}    # track_id -> last wrist crop boxes (for the debug overlay)
        # Draw the 21-point finger skeleton for every detected hand (not just pocket
        # candidates) - when on, MediaPipe runs for each person, throttled as usual.
        self.draw_hands = bool(cfg.get("output", {}).get("draw_hand_skeleton", False))

        # The PO stack (MediaPipe + classifier) is light enough to run even in
        # low-memory mode ON CPU (the big RAM hogs were CUDA + the 2nd YOLO model).
        # So it is controlled purely by the PO toggles below, not by low_memory.

        # Optional MediaPipe fingertip check for PO. Only PO fills the hand
        # cache, so an instance that is not running PO would load the model,
        # spend memory on it and never produce a single landmark - while its
        # dashboard still offered a "Hands" toggle that could not do anything.
        self.hand = None
        if ("PO" in self.enabled
                and self.rcfg.get("PO", {}).get("use_fingertips", False)):
            # One MediaPipe model for every camera (it keeps no state between
            # calls, and all cameras are judged from the one detection loop).
            key = float(self.rcfg["PO"].get("fingertip_det_conf", 0.3))
            if key not in _HANDS:
                _HANDS[key] = HandChecker(num_hands=2, min_conf=key)
            self.hand = _HANDS[key]

        # PO pocket model (poketenashi/po_model.py). When present it decides
        # whether a hand at the hip is in a pocket - in place of the shape
        # cues, the stillness test and MediaPipe, none of which can tell a
        # pocket from a hand hanging at the hip on this camera.
        self.po_model = po_model.load(self.rcfg.get("PO", {})) if "PO" in self.enabled else None
        self._po_p = {}             # (tid, side) -> (frame_i, P(pocket))
        # PO hip model (rules.PO.pipeline: evidence): one crop of both hips per
        # person - is ANY hand in a pocket? See _rule_PO_evidence.
        self.po_hip = (po_model.load(self.rcfg.get("PO", {}), key="hip_model_path")
                       if "PO" in self.enabled else None)
        self._po_hp = {}            # tid -> (frame_i, P(a hand in a pocket))
        self._po_arm_ev = {}        # tid -> {side: deque[(t, pocket?)]} (evidence, per arm)
        # rules.PO.evidence_sustain_sec: per arm, the sustained-evidence state
        # (_po_decide_sustain). Kept through tracker drop-outs up to
        # evidence_gap_sec - ByteTrack loses people for single frames at ~5 fps.
        self._po_arm_st = {}        # tid -> {"seen": t, "l": {...}, "r": {...}}
        self._po_ir = (-1, False)   # (frame number, the picture is night IR)
        # The closer re-pose (rules.PO.evidence_repose) is loaded now, not on the
        # first pocket - loading mid-run would stall the detection loop.
        if "PO" in self.enabled and self.rcfg.get("PO", {}).get("evidence_repose", False):
            self._repose_model(self.rcfg["PO"])

        # PO detection method: master | fingertips | geometry.
        po = self.rcfg.get("PO", {})
        self.po_method = po.get("method", "fingertips")
        self.embedder = None
        self.master = None
        if self.po_method == "master":
            try:
                from .embedder import Embedder
                from .master_match import MasterLibrary
                self.master = MasterLibrary(po.get("master_file", "masters/po_masters.npz"))
                if len(self.master) == 0:
                    print("[PO] master library empty -> run 'python -m tools.capture_masters'. "
                          "Falling back to fingertips/geometry for now.")
                    self.master = None
                else:
                    self.embedder = Embedder(device=cfg.get("models", {}).get("device") or None)
                    print(f"[PO] master-match mode: {len(self.master)} reference samples "
                          f"(pocket={self.master.count('pocket')}, out={self.master.count('out')}).")
            except Exception as e:
                print("[PO] master init failed -> fallback:", e)
                self.master = None

    # ---- helpers -------------------------------------------------------

    def refresh_zones(self):
        """The zones changed on the running monitor (zone editor): re-derive what
        was computed from them at start-up - the zone gate and NA's aisle."""
        self.gate_zones = self._gate_zones(self.rcfg.get("zone_gate") or {})
        for na in self.na_list:
            na.refresh()

    def _gate_zones(self, zg):
        """The zone_gate zones that are actually drawn (`zone`: a name or a list)."""
        names = zg.get("zone") or []
        if isinstance(names, str):
            names = [names]
        return [z for z in names if self.zones is not None and self.zones.has(z)]

    def _warn_once(self, msg):
        if msg not in self._warned:
            print(msg)
            self._warned.add(msg)

    def _skip(self, code):
        """Log once that a rule cannot run, in the words /api/state reports.

        One reason string, from poketenashi/ruleinfo.py, so the log line and the
        dashboard can never explain the same silent rule differently.
        """
        miss = ruleinfo.missing_zones(code, self.zones.has)
        hint = f" (python -m tools.zone_drawer --zone {miss[0]})" if miss else ""
        reason = ruleinfo.why(code, self.zones.has) or "its zone is not usable"
        self._warn_once(f"[{code}] skipped: {reason}{hint}")

    def _blocked(self, code):
        """True when `code` has no zone to judge in: logs it once, gives up."""
        if ruleinfo.missing_zones(code, self.zones.has):
            self._skip(code)
            return True
        return False

    def _update_history(self, persons):
        now = time.monotonic()
        for p in persons:
            self.history[p.track_id].append((now, p.center()))

    def speed(self, track_id):
        """Center displacement between the last two frames, in pixels."""
        h = self.history[track_id]
        if len(h) < 2:
            return 0.0
        return float(np.linalg.norm(h[-1][1] - h[-2][1]))

    def speed_ps(self, track_id, window=0.5):
        """Net centre speed in pixels per SECOND, measured over ~`window` seconds.

        speed() is per-FRAME, so its meaning changes with the pipeline frame rate:
        the same walk reads twice as fast when the CPU halves the FPS. This one is
        in real time, so a threshold built on it stays put. The window also smooths
        away the per-frame bbox jitter (arm/leg swing, box wobble) that a two-frame
        difference happily reports as a sprint.
        """
        h = self.history[track_id]
        if len(h) < 2:
            return 0.0
        t_now, p_now = h[-1]
        t_old, p_old = h[-2]
        for t, pt in reversed(h):        # walk back to the oldest sample in the window
            if t_now - t > window:
                break
            t_old, p_old = t, pt
        dt = t_now - t_old
        if dt <= 1e-6:
            return 0.0
        return float(np.linalg.norm(p_now - p_old) / dt)

    def _update_head(self, persons, mc=0.3):
        """Track a head-yaw proxy: nose offset from the shoulder midpoint, scaled by
        shoulder width. Swings left/right when a person looks around."""
        for p in persons:
            nose = p.kp("nose", mc)
            ls, rs = p.kp("l_shoulder", mc), p.kp("r_shoulder", mc)
            if nose is None or ls is None or rs is None:
                continue
            sw = float(np.linalg.norm(ls - rs))
            if sw < 1:
                continue
            self.head_hist[p.track_id].append((nose[0] - (ls[0] + rs[0]) / 2.0) / sw)

    def head_swing(self, track_id):
        """Range of the head-yaw proxy over the window (0 = stared straight ahead)."""
        h = self.head_hist[track_id]
        if len(h) < 5:
            return 0.0
        return float(max(h) - min(h))

    def _update_wrists(self, persons, mc=0.3):
        """Track each wrist's position RELATIVE to the hip (scaled by torso), so the
        person's own walking motion is removed. A hand in a pocket stays put; an
        out/swinging hand wanders. Also keep absolute positions for the trail."""
        for p in persons:
            torso = p.torso_len(mc)
            if not torso or torso <= 1:
                continue
            for side in ("l", "r"):
                W = p.kp(f"{side}_wrist", mc)
                H = p.kp(f"{side}_hip", mc)
                if W is None or H is None:
                    continue
                self.wrist_hist[(p.track_id, side)].append((W - H) / torso)
                self.wrist_trail[(p.track_id, side)].append(W)

    def wrist_movement(self, track_id, side):
        """Body-relative 'wander' of the wrist over the window: mean distance of the
        relative positions from their centroid. ~0 = parked (pocket), higher = moving."""
        h = self.wrist_hist[(track_id, side)]
        if len(h) < 5:
            return None  # not enough history yet
        pts = np.array(h)
        return float(np.mean(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))

    def current_hands(self):
        """All currently cached MediaPipe hands (for drawing the finger skeleton)."""
        out = []
        for hands in self._hands_cache.values():
            out.extend(hands)
        return out

    def wrist_still(self, track_id, side):
        """(is_still, movement_value). A parked wrist (low movement) supports 'pocket'.
        Returns is_still=True while history is too short (don't block early)."""
        c = self.rcfg["PO"]
        if not c.get("use_wrist_motion", True):
            return True, None
        mv = self.wrist_movement(track_id, side)
        if mv is None:
            return True, None
        return mv < c.get("wrist_move_thresh", 0.08), mv

    def _fire(self, results, person, rule, need=None):
        """`need` overrides min_frames for this call.

        A rule that can tell strong evidence from weak should not make both wait
        the same. PO fires on as little as one shape cue, and min_frames is there
        to stop that single cue flickering into an alert - but when all three cues
        agree and no hand is visible at the pocket, there is nothing left to wait
        for and the delay is just a late alert.
        """
        key = (person.track_id, rule)
        self.streak[key] += 1
        if self.streak[key] >= (need if need is not None
                                else self.rcfg.get(rule, {}).get("min_frames", 1)):
            results.append({
                "track_id": person.track_id, "rule": rule,
                "bbox": person.bbox,
            })

    def _clear(self, track_id, rule):
        self.streak[(track_id, rule)] = 0

    # ---- entry point ---------------------------------------------------

    def check(self, persons, phones, frame=None):
        results = []
        # Only people the tracker has confirmed are judged. When it returns no
        # ids for a frame, every detection gets a brand-new negative id, so it
        # has no history (no speed, no wrist stillness, no streak) and no
        # cooldown can apply to it: 233 PO alerts in one day came from those
        # single-frame ids. They are still drawn - just not judged.
        persons = [p for p in persons if p.tracked]
        self._frame_i += 1
        self._update_history(persons)
        self._update_head(persons)
        self._update_wrists(persons)
        # drop cached hands of people no longer tracked (avoid ghost skeletons)
        # Forget everything about tracks that are no longer present, otherwise
        # per-track trails/histories pile up forever and clutter the frame.
        live = {p.track_id for p in persons}
        # 7 Oct 2026: the tracker drops a person for single frames (~5-10 fps). The
        # centre history, head-yaw history and SHI pad state survive a gap of up to
        # track_keep_sec (rules.SHI): wiping them on one missed frame left SHI with
        # no "look" (head swing never reached 5 samples) and lost pad visits.
        keep = float((self.rcfg.get("SHI") or {}).get("track_keep_sec", 2.0))
        tnow = time.monotonic()
        seen = self.__dict__.setdefault("_last_seen", {})
        for tid in live:
            seen[tid] = tnow
        kept = {k for k, t in seen.items() if tnow - t <= keep}
        for k in [k for k in seen if k not in kept]:
            del seen[k]
        for d in (self.history, self.head_hist, self.shi_state):
            for k in [k for k in d if k not in kept]:
                del d[k]
        for d in (self._hands_cache,
                  self._crops_dbg, self._po_seen, self._po_ev, self._po_on,
                  self._po_hp, self._po_arm_ev):
            for k in [k for k in d if k not in live]:
                del d[k]
        for d in (self.wrist_hist, self.wrist_trail, self._po_p, self._po_refsim):   # keyed by (track_id, side)
            for k in [k for k in d if k[0] not in live]:
                del d[k]
        for k in [k for k in self.streak if k[0] not in live]:   # (track_id, rule)
            del self.streak[k]
        # The sustained PO evidence outlives a tracker drop-out (a single missed
        # frame is common at ~5 fps); a person gone longer starts over.
        gap = float(self.rcfg.get("PO", {}).get("evidence_gap_sec", 1.5))
        now = time.monotonic()
        for k in [k for k, st in self._po_arm_st.items() if k not in live and now - st["seen"] > gap]:
            del self._po_arm_st[k]

        # Histories above are kept for everyone, so a person walking INTO the
        # zone arrives with a speed and a wrist history. Only the verdicts are
        # gated - and whoever is outside starts from zero when they come back.
        inside = persons
        if self.gate_zones:
            inside = []
            for p in persons:
                ds = [self.zones.signed_dist(z, p.feet()) for z in self.gate_zones]
                p.in_zone = any(d is not None and d >= -self.gate_margin for d in ds)
                if p.in_zone:
                    inside.append(p)
                else:
                    for rule in self.gate_rules:
                        self._clear(p.track_id, rule)
                    for d in (self._po_seen, self._po_ev, self._po_on, self._po_arm_ev,
                              self._po_arm_st):
                        d.pop(p.track_id, None)

        def judged(rule):
            return inside if rule in self.gate_rules else persons

        if "PO" in self.enabled:
            self.rule_PO(judged("PO"), results, frame)
        if "KE" in self.enabled:
            self.rule_KE(judged("KE"), phones, results, frame)
        if "TE" in self.enabled:
            self.rule_TE(judged("TE"), results)
        if "NA" in self.enabled:
            self.rule_NA(judged("NA"), results)
        if "SHI" in self.enabled:
            self.rule_SHI(judged("SHI"), results)
        if "RUN" in self.enabled:
            self.rule_RUN(judged("RUN"), results)
        if "FALL" in self.enabled:
            self.rule_FALL(judged("FALL"), results, frame)
        if "RESTRICTED" in self.enabled:
            self.rule_RESTRICTED(judged("RESTRICTED"), results)
        return results

    # ---- KE: phone use while walking / in the gangway -----------------

    def rule_KE(self, persons, phones, results, frame):
        """Object-verified phone-use detection (see poketenashi/ke.py). The KE module
        keeps its own streaks + latch, so a firing track is emitted directly (its own
        temporal confirmation already ran); non-firing tracks have their KE streak
        cleared so a later _fire-based consumer stays consistent."""
        if self.ke is None or frame is None:
            return
        firing = self.ke.check(persons, phones, frame, self, self.zones)
        for p in persons:
            if p.track_id in firing:
                self.streak[(p.track_id, "KE")] += 1
                results.append({
                    "track_id": p.track_id, "rule": "KE",
                    "bbox": p.bbox,
                })
            else:
                self._clear(p.track_id, "KE")

    # ---- PO: hands in pockets -----------------------------------------

    def rule_PO(self, persons, results, frame=None):
        """Dispatch to the configured PO method (master / evidence / geometry)."""
        if (self.po_method == "master" and self.master is not None
                and self.embedder is not None and frame is not None):
            self._rule_PO_master(persons, results, frame)
        elif (self.rcfg.get("PO", {}).get("pipeline") == "evidence" and frame is not None
                and (self.po_hip if self.rcfg["PO"].get("evidence_model", "arm") == "hip"
                     else self.po_model) is not None):
            self._rule_PO_evidence(persons, results, frame)
        else:
            self._rule_PO_geom(persons, results, frame)

    def _rule_PO_master(self, persons, results, frame):
        """Master-match PO: crop each candidate wrist, embed it, and match to the
        labelled reference samples. Nearest masters vote pocket vs out."""
        c = self.rcfg["PO"]
        mc, ratio = c["min_keypoint_conf"], c["wrist_hip_ratio"]
        k = c.get("master_k", 5)
        min_conf = c.get("master_min_conf", 0.55)
        scale = c.get("master_scale", 0.6)

        for p in persons:
            torso = p.torso_len(mc)
            violate = False
            dbg_sides = {}
            if torso and torso > 1:
                for side in ("l", "r"):
                    W, H = p.kp(f"{side}_wrist", mc), p.kp(f"{side}_hip", mc)
                    if W is None or H is None:
                        dbg_sides[side] = None
                        continue
                    dist = float(np.linalg.norm(W - H) / torso)
                    near = dist < ratio
                    label, conf = None, 0.0
                    if near:
                        crop, _ = wrist_crop(frame, p, side, scale, mc)
                        label, conf = self.master.predict(self.embedder.embed(crop), k)
                    still, mv = self.wrist_still(p.track_id, side)
                    side_po = near and label == "pocket" and conf >= min_conf and still
                    if side_po:
                        violate = True
                    if self.debug:
                        dbg_sides[side] = {"mode": "master", "dist": dist, "near": near,
                                           "label": label, "conf": round(conf, 2),
                                           "still": still, "mv": (round(mv, 3) if mv is not None else None),
                                           "po": side_po}
            if self.debug:
                self.po_dbg[p.track_id] = {"sides": dbg_sides, "po": violate, "bbox": p.bbox}
            if violate:
                self._po_seen[p.track_id] = time.monotonic()
                self._fire(results, p, "PO")
            elif (time.monotonic() - self._po_seen.get(p.track_id, 0.0)
                    <= self.rcfg.get("PO", {}).get("hold_sec", 1.5)):
                # Hold the box across a flicker. _clear() zeroes the streak on a
                # single missed frame, and min_frames then needs that many
                # CONSECUTIVE frames again - a full second at 5 fps - so the box
                # dropped out every time the fingertip check blinked, even though
                # the hand never left the pocket.
                self._fire(results, p, "PO")
            else:
                self._clear(p.track_id, "PO")

    def _rule_PO_geom(self, persons, results, frame=None):
        """PO via fingertips + geometry - the rule as it ran 10-24 Sep.

        Judged per ARM: one hand in a pocket is PO, whatever the other hand is
        doing (swinging, holding something). A wrist counts as a pocket when it
        is at the hip, parked there relative to the body - so walking and
        standing both count - the arm is shaped like it (>= min_cues of: elbow
        bent, wrist tucked in, not hanging down the thigh) and MediaPipe finds
        no hand with fingers showing at that wrist.

        NOT a pocket (28 Sep: the 10 Sep rule fired 29 times in 24 minutes, and
        none of the 22 alerts checked was a pocket):
          * a glimpse - PO needs the pocket test in most frames of the last
            confirm_sec (_po_decide); one frame of a hand swinging past the hip,
            seen from behind, used to become an alert through the hold
          * the pocket not in the picture - cut off by the bottom of the frame
          * both hands together in front of the body - clasped, or holding
            something; one hand in a pocket leaves the two a hip-width apart
        """

        c = self.rcfg["PO"]
        mc = c["min_keypoint_conf"]

        # With the pocket model the model judges the hand; MediaPipe is not run.
        use_model = self.po_model is not None and frame is not None
        model_thr = float(c.get("model_threshold", 0.6))
        model_every = max(1, int(c.get("model_every_n", 2)))

        use_ft = (
            c.get("use_fingertips", True)
            and self.hand is not None
            and self.hand.ok
            and frame is not None
            and not use_model
        )

        match_r = c.get("hand_match_ratio", 0.65)
        wrist_r = c.get("wrist_region_ratio", 0.75)
        upscale = int(c.get("fingertip_upscale", 512))
        every_n = max(1, int(c.get("fingertip_every_n", 2)))

        bend_min = 70
        bend_max = 140
        tuck_t = c.get("tuck_ratio", 0.10)
        hang_t = c.get("hang_drop_ratio", 0.35)
        min_cues = c.get("min_cues", 2)
        above_max = float(c.get("wrist_above_hip_max", 0.15))

        for p in persons:

            torso = p.torso_len(mc)
            if torso <= 1:
                self._po_decide(p, results, c, None)
                continue

            violate = False
            dbg_sides = {}

            # --------------------------
            # Dynamic pelvis zone
            # --------------------------
            # Pelvis center manually calculate (pelvis_center() method ki zarurat nahi)
            LH = p.kp("l_hip", mc)
            RH = p.kp("r_hip", mc)

            if LH is not None and RH is not None:
                pelvis = (LH + RH) / 2
            elif LH is not None:
                pelvis = LH
            elif RH is not None:
                pelvis = RH
            else:
                self._po_decide(p, results, c, None)
                continue

            # The pocket is not in the picture: cut off by the bottom of the
            # frame above the thighs (walking right under the camera). Hips and
            # wrists are the pose model's guesses there - ID8, 29, 117, 139 on
            # 28 Sep. "Touching" = within 1% of the bottom: ID139's box stopped
            # 5 px short of it on a 720-line clip.
            if (frame is not None and p.bbox[3] >= 0.99 * frame.shape[0]
                    and pelvis[1] + 0.5 * torso > 0.99 * frame.shape[0]):
                self._po_decide(p, results, c, False)
                continue
            # Both hands together in front of the body: clasped, or holding
            # something - not in pockets (ID139 on 28 Sep, 4 alerts).
            if c.get("hands_together_filter", True) and self._hands_together(p, torso):
                self._po_decide(p, results, c, False)
                continue

            body = p.body_scale(mc)
            zone_w = body * 0.28
            zone_h = body * 0.32

            cand = {}

            for side in ("l", "r"):

                W = p.kp(f"{side}_wrist", mc)
                H = p.kp(f"{side}_hip", mc)

                if W is None or H is None:
                    dbg_sides[side] = None
                    continue

                near = (
                    abs(W[0] - pelvis[0]) <= zone_w
                    and abs(W[1] - pelvis[1]) <= zone_h
                )

                cand[side] = (W, H, near)

            # --------------------------
            # MediaPipe wrist crops
            # --------------------------
            hands = []

            if use_ft and any(v[2] for v in cand.values()):

                if self._frame_i % every_n == 0 or p.track_id not in self._hands_cache:

                    boxes = []

                    for _, (W, _, _) in cand.items():

                        r = int(max(wrist_r * torso, 12))
                        yo = int(0.2 * r)

                        box = (
                            int(W[0]) - r,
                            int(W[1]) - r + yo,
                            int(W[0]) + r,
                            int(W[1]) + r + yo,
                        )

                        boxes.append(box)

                        hands.extend(
                            self.hand.detect_in_region(
                                frame,
                                box,
                                upscale_to=upscale,
                            )
                        )

                    self._hands_cache[p.track_id] = hands
                    self._crops_dbg[p.track_id] = boxes

                else:
                    hands = self._hands_cache.get(p.track_id, [])

            # --------------------------
            # Pocket model: P(pocket) for each hand at the hip (both arms in one
            # batch; refreshed every model_every frames, reused in between)
            # --------------------------
            probs = {}
            if use_model:
                need = []
                for side, (W, H, near) in cand.items():
                    m0 = self._po_measure(p, side, mc, torso)
                    if m0 is None or m0["hang"] < -above_max:
                        continue
                    if not (near or float(np.linalg.norm(W - H) / torso) < 0.55):
                        continue
                    cached = self._po_p.get((p.track_id, side))
                    if cached is not None and self._frame_i - cached[0] < model_every:
                        probs[side] = cached[1]
                        continue
                    crop = po_model.arm_crop(frame, W, torso)
                    if crop is not None:
                        need.append((side, crop))
                if need:
                    for (side, _), pr in zip(need, self.po_model.score([cr for _, cr in need])):
                        probs[side] = pr
                        self._po_p[(p.track_id, side)] = (self._frame_i, pr)

            # --------------------------
            # Evaluate each wrist
            # --------------------------

            for side, (W, H, near) in cand.items():

                dist = float(np.linalg.norm(W - H) / torso)

                still, mv = self.wrist_still(p.track_id, side)
                # "Still" has to be SEEN: with too little history wrist_still()
                # says yes, and a brand-new track could fire before anyone knew
                # whether the hand was parked or swinging (~0.3 s at 15 fps).
                if mv is None and c.get("still_needs_history", True) \
                        and c.get("use_wrist_motion", True):
                    still = False

                m = self._po_measure(p, side, mc, torso)

                if m is None:
                    continue

                angle_ok = bend_min <= m["angle"] <= bend_max
                tuck_ok = m["tuck"] > tuck_t
                hang_ok = m["hang"] < hang_t

                cues = int(angle_ok) + int(tuck_ok) + int(hang_ok)

                # A pocket is at the hip: a hand held higher - a phone, papers
                # at the belly or chest - is not in one (PO on phone users,
                # 25 Sep).
                at_hip = m["hang"] >= -above_max

                # visible hand
                visible = False
                best = None
                best_tips = 0

                if use_ft:

                    for hd in hands:

                        d = float(np.linalg.norm(hd["wrist"] - W) / torso)

                        if best is None or d < best:
                            best = d
                            best_tips = hd["n_tips"]

                    visible = (
                        best is not None
                        and best < match_r
                        and best_tips >= 1
                    )

                near_ok = near or (dist < 0.55)

                if use_model:
                    # The model looked at the hand: walking or standing, a hand
                    # it sees in a pocket at the hip is a pocket.
                    side_po = near_ok and at_hip and probs.get(side, 0.0) >= model_thr
                else:
                    side_po = (
                        near_ok
                        and at_hip
                        and still
                        and cues >= min_cues
                        and (not visible if use_ft else True)
                    )

                if self.debug:

                    dbg_sides[side] = {
                        "mode": "ft" if use_ft else "geo",
                        "dist": round(dist, 2),
                        "near": near_ok,
                        "angle": round(m["angle"], 1),
                        "tuck": round(m["tuck"], 2),
                        "hang": round(m["hang"], 2),
                        "cues": cues,
                        "still": still,
                        "mv": round(mv, 3) if mv is not None else None,
                        "visible": visible if use_ft else None,
                        "hand_d": round(best, 2) if best is not None else None,
                        "tips": best_tips,
                        "pocket_p": round(probs[side], 2) if side in probs else None,
                        "po": side_po,
                    }

                    # Per-arm numbers on the console only while tuning: unguarded
                    # this printed ~20 lines a second and buried every real message.
                    print(
                        f"ID={p.track_id} {side} "
                        f"near={near_ok} dist={dist:.2f} "
                        f"still={still} cues={cues} "
                        f"visible={visible} tips={best_tips} hand_d={best}"
                    )

                if side_po:
                    violate = True

            if self.debug:

                self.po_dbg[p.track_id] = {
                    "sides": dbg_sides,
                    "po": violate,
                    "bbox": p.bbox,
                    "crops": self._crops_dbg.get(p.track_id, []),
                }

            self._po_decide(p, results, c, violate if cand else None)

    def _rule_PO_evidence(self, persons, results, frame):
        """PO from evidence (rules.PO.pipeline: evidence) - the rule as the user
        set it: at least ONE hand in a pocket, walking or standing, feet in the
        gangway (the zone gate already ran). The other hand may do anything.

        1. Candidate, per arm: a wrist at its hip (within 0.5 torso up or down,
           0.6 across - a front-pocket wrist sits ABOVE the hip keypoint). An
           arm whose wrist is missing or away from the hip, or a person too
           small to judge, is UNCERTAIN: a hidden or missing hand is never
           evidence either way.
        2. Evidence: the pocket model looks at the hand (evidence_model: arm,
           po_model.arm_crop per arm) or at both hips (hip, po_model.hip_box,
           one verdict for the person). P >= evidence_hi is POCKET, P <=
           evidence_lo is NOT, in between UNCERTAIN. Wrist motion, elbow angle
           and wrist height are NOT used: on 926 labelled arms none of them
           separates a pocket from a free hand (wrist motion AUC 0.50 - a
           walking pocket moves as much as a swinging hand).
           Kept protections (NOT, both arms): the pocket cut off by the bottom
           of the frame, both hands together in front of the body (clasped /
           holding something).
        3. Confirmation per arm (_po_decide_arms): one arm POCKET in most of the
           judged frames of the last confirm_sec; UNCERTAIN frames count neither
           way.
        """
        c = self.rcfg["PO"]
        mc = c["min_keypoint_conf"]
        use_hip = c.get("evidence_model", "arm") == "hip"
        model = self.po_hip if use_hip else self.po_model
        hi = float(c.get("evidence_hi", 0.6))
        lo = float(c.get("evidence_lo", 0.3))
        min_torso = float(c.get("evidence_min_torso", 70))
        every = max(1, int(c.get("model_every_n", 2)))
        fh, fw = frame.shape[:2]
        ir = bool(c.get("evidence_skip_ir", False)) and self._is_ir(frame, c)
        self._po_frame = frame                     # for the object check at fire time (_po_decide_sustain)
        for p in persons:
            tid = p.track_id
            torso = p.torso_len(mc)
            hips = [h for h in (p.kp("l_hip", mc), p.kp("r_hip", mc)) if h is not None]
            ev = {"l": None, "r": None}
            why = {"l": "", "r": ""}
            prob = {"l": None, "r": None}
            crops_dbg = []
            if ir:
                # Night IR (evidence_skip_ir): a grey picture the pocket model
                # was never trained or checked on. On 5 Oct it called a hand
                # behind the back and two hands hanging in plain view pockets.
                # Not judged (UNKNOWN), never "pocket".
                why = {"l": "ir", "r": "ir"}
            # The pocket is not in the picture: cut off by the bottom of the
            # frame (see _rule_PO_geom).
            elif (hips and p.bbox[3] >= 0.99 * fh
                    and float(np.mean(hips, axis=0)[1]) + 0.5 * torso > 0.99 * fh):
                ev = {"l": False, "r": False}
                why = {"l": "cut off", "r": "cut off"}
            # 10 Oct (user, option 2): close to the camera the legs leave the
            # picture first - the frame ends within evidence_cutoff_torso below
            # the hips, so the pocket and the hand
            # in it are (half) out of view (Toolroom ID1426 09:57). Not judged.
            # 10 Oct 13:05 (ID67): a person cut by the LEFT / RIGHT edge of the
            # picture (box touching it) is half out of view - not judged
            elif c.get("evidence_side_edge_px") is not None and (
                    p.bbox[0] <= float(c["evidence_side_edge_px"])
                    or p.bbox[2] >= fw - float(c["evidence_side_edge_px"])):
                why = {"l": "cut off", "r": "cut off"}
            elif (hips and c.get("evidence_cutoff_torso")
                    and float(np.mean(hips, axis=0)[1]) + float(c["evidence_cutoff_torso"]) * torso > 0.98 * fh):
                why = {"l": "cut off", "r": "cut off"}
            elif (c.get("hands_together_filter", True) and torso > 1
                    and self._hands_together(p, torso)):
                ev = {"l": False, "r": False}
                why = {"l": "together", "r": "together"}
            elif torso < min_torso:
                why = {"l": "small", "r": "small"}
            elif c.get("evidence_need_face", False) and not (
                    p.kp("nose", 0.5) is not None
                    and (p.kp("l_eye", 0.5) is not None or p.kp("r_eye", 0.5) is not None)):
                # Seen from behind: a front pocket is out of sight and the hand
                # shows its back - the model guesses. On the 3 Oct raw hour 5 of 8
                # false alerts were people walking away. Not judged (UNCERTAIN),
                # never "not PO".
                why = {"l": "back", "r": "back"}
            elif use_hip:
                box = po_model.hip_box(p, fw, fh)
                if box is None:
                    why = {"l": "no hand", "r": "no hand"}
                else:
                    cached = self._po_hp.get(tid)
                    if cached is not None and self._frame_i - cached[0] < every:
                        pr = cached[1]
                    else:
                        x1, y1, x2, y2 = box
                        pr = model.score([frame[y1:y2, x1:x2]])[0]
                        self._po_hp[tid] = (self._frame_i, pr)
                    ev = {"l": True if pr >= hi else (False if pr <= lo else None), "r": None}
                    prob = {"l": pr, "r": pr}
                    crops_dbg.append(box)
            else:
                need = []
                for side in ("l", "r"):
                    W, H = p.kp(f"{side}_wrist", mc), p.kp(f"{side}_hip", mc)
                    if W is None or H is None:
                        why[side] = "no wrist"
                        continue
                    if abs(W[1] - H[1]) > 0.5 * torso or abs(W[0] - H[0]) > 0.6 * torso:
                        why[side] = "away"
                        continue
                    # A hand in a pocket hangs from the elbow: the forearm runs
                    # DOWN into the pocket (labelled pocket arms: wrist 0.30-0.48
                    # torso below the elbow). A folded forearm is a hand held up -
                    # a pen, a phone, the cap - or a wrist the pose dropped to the
                    # waist while the hand is at the head (ID68 15:25, 3 Oct: cap
                    # + pen, wrist 0.13-0.30 below the elbow). Not judged.
                    drop_min = float(c.get("evidence_min_forearm_drop", 0.0))
                    E = p.kp(f"{side}_elbow", mc)
                    if drop_min > 0 and E is not None and (W[1] - E[1]) < drop_min * torso:
                        why[side] = "folded"
                        continue
                    if self.debug:
                        h = 0.45 * torso
                        crops_dbg.append((W[0] - h, W[1] + 0.1 * torso - h,
                                          W[0] + h, W[1] + 0.1 * torso + h))
                    cached = self._po_p.get((tid, side))
                    if cached is not None and self._frame_i - cached[0] < every:
                        prob[side] = cached[1]
                        continue
                    crop = po_model.arm_crop(frame, W, torso)
                    if crop is None:
                        why[side] = "no crop"
                        continue
                    need.append((side, crop))
                if need:
                    crops_ = [cr for _, cr in need]
                    if c.get("ref_dirs") and hasattr(model, "score_embed"):
                        probs_, embs_ = model.score_embed(crops_)     # one network pass for both
                    else:
                        probs_, embs_ = model.score(crops_), None
                    # 10 Oct: a second pocket model must agree (rules.PO.model2_path):
                    # P = the LOWER of the two - v1 and v8 make different mistakes
                    if c.get("model2_path"):
                        m2 = po_model.load(c, key="model2_path")
                        if m2 is not None:
                            # the second model only where the first one is not already
                            # sure it is out (min() stays below evidence_lo anyway) -
                            # in a crowd most hands score ~0 (10 Oct: 1-2 fps at 12:38)
                            need2 = [k_ for k_, a_ in enumerate(probs_) if a_ > lo]
                            if need2:
                                s2 = m2.score([crops_[k_] for k_ in need2])
                                probs_ = list(probs_)
                                for k_, b_ in zip(need2, s2):
                                    probs_[k_] = min(probs_[k_], b_)
                    for k_, ((side, cr), pr) in enumerate(zip(need, probs_)):
                        prob[side] = pr
                        self._po_p[(tid, side)] = (self._frame_i, pr)
                        # 9 Oct: how close is this hand to the user's OK PO references
                        refs = self._po_refs(c, model)        # starts the master build on the first hand
                        refs = refs if pr >= float(c.get("ref_min_p", hi)) else None
                        if refs is not None:
                            if embs_ is not None:
                                e = embs_[k_]
                            else:
                                from .po_refs import embed
                                e = embed(model, [cr])[0]
                            kk = int(c.get("ref_k", 1))
                            s_ng = refs.ng.topk(e, kk) if getattr(refs, "ng", None) is not None else None
                            self._po_refsim[(tid, side)] = (self._frame_i, refs.topk(e, kk), s_ng)
                # evidence_uncertain_as_not: a hand the model looked at but was
                # unsure about counts against the pocket in the confirm ratio -
                # PO then needs a confident model most of the time, not 3
                # confident frames among many unsure ones (ID37, 15:03 3 Oct:
                # a hand at the specs, the wrist put at the hip by the pose).
                unsure = False if c.get("evidence_uncertain_as_not", False) else None
                for side in ("l", "r"):
                    pr = prob[side]
                    if pr is not None:
                        ev[side] = True if pr >= hi else (False if pr <= lo else unsure)
                # 9 Oct (user): v1 model + the user's OK and NG PO clips as masters.
                # A POCKET hand counts only when it looks more like the OK masters
                # than the NG ones (by ref_margin); masters still loading: no pocket.
                # 9 Oct 14:40 (user): an UNSURE hand (lo < P < hi) is decided in THIS
                # frame by the masters - OK-like by ref_unsure_margin = pocket, else out
                if c.get("ref_ng_dirs") and c.get("ref_unsure_margin") is not None:
                    for side in ("l", "r"):
                        pr = prob[side]
                        if ev[side] is not True and pr is not None and lo < pr < hi:
                            rs = self._po_refsim.get((tid, side))
                            if rs is not None and rs[0] == self._frame_i and rs[2] is not None:
                                ev[side] = (rs[1] - rs[2]) >= float(c["ref_unsure_margin"])
                                why[side] = "unsure->master"
                if c.get("ref_ng_dirs"):
                    for side in ("l", "r"):
                        if ev[side] is True:
                            rs = self._po_refsim.get((tid, side))
                            if (rs is None or rs[0] != self._frame_i or rs[2] is None
                                    or rs[1] - rs[2] < float(c.get("ref_margin", 0.05))):
                                ev[side] = None
                                why[side] = "ng-like"
                # 9 Oct (user): only HANDS in pockets - a hand at the hip holding
                # a bottle, cup, phone, book... is not PO. The KE object model (one
                # shared copy) looks at the hand, only on frames that would count.
                obj_cls = c.get("evidence_object_veto")
                if obj_cls and not c.get("evidence_object_at_fire", True) and any(v is True for v in ev.values()):
                    for side in ("l", "r"):
                        if ev[side] is True and self._po_object_cached(tid, side, frame, p.kp(f"{side}_wrist", mc), torso, c):
                            ev[side] = None
                            why[side] = "object"
                # A closer look before a POCKET counts (evidence_repose). The
                # live pose runs on the whole frame at low resolution - a
                # person is ~80 px tall there - and an arm held up at the face
                # can come out as an arm hanging to the pocket; the pocket
                # model then sees no hand at that "wrist" and says pocket
                # (3 Oct: 15:25 a cap, 18:01 both hands at the face). Posed
                # again on the person's own crop, those wrists land at the
                # chest or the face. No closer pose, or no wrist found in it:
                # the evidence stands.
                if c.get("evidence_repose", False) and any(v is True for v in ev.values()):
                    kp2 = self._po_repose(p, frame, c)
                    if kp2 is not None:
                        for side in ("l", "r"):
                            if ev[side] is True and not self._repose_at_hip(kp2, p.kp(f"{side}_wrist", mc)):
                                ev[side] = None             # the two looks disagree: no evidence
                                why[side] = "re-pose"
            if self.debug:
                # in the overlay's "master" layout: label = why / model, conf = P
                self.po_dbg[tid] = {
                    "sides": {s: {"mode": "master", "dist": 0.0, "near": prob[s] is not None,
                                  "label": why[s] or ("hip" if use_hip else "arm"),
                                  "conf": round(prob[s], 2) if prob[s] is not None else 0.0,
                                  "mv": None, "still": False, "po": ev[s] is True,
                                  "geo": self._po_geo(p, s, mc, torso, fh)}
                              for s in ("l", "r")},
                    "po": any(v is True for v in ev.values()), "bbox": p.bbox, "crops": crops_dbg}
            # 8 Oct (user): every hand at the hip is a PO CANDIDATE for review -
            # web.py saves them (po_candidates in config.yaml), alerts unchanged
            for s_ in ("l", "r"):
                if prob[s_] is not None:
                    cur = self.po_cand.get(tid)
                    if cur is None or prob[s_] > cur["P"]:
                        self.po_cand[tid] = {"P": float(prob[s_]), "side": s_, "bbox": [float(v) for v in p.bbox],
                                             "pocket": ev[s_] is True}
            if float(c.get("evidence_sustain_sec", 0.0)) > 0:
                self._po_decide_sustain(p, results, c, ev, why, prob, hi, lo)
            else:
                self._po_decide_arms(p, results, c, ev)

    def _is_ir(self, frame, c):
        """The camera is in night IR mode: a grey picture. Mean saturation of
        a coarse sample of the frame (day ~90 here, IR ~15). Once per frame."""
        if self._po_ir[0] != self._frame_i:
            import cv2
            s = cv2.cvtColor(np.ascontiguousarray(frame[::16, ::16]), cv2.COLOR_BGR2HSV)[:, :, 1]
            self._po_ir = (self._frame_i, float(s.mean()) < float(c.get("evidence_ir_max_sat", 30)))
        return self._po_ir[1]

    def _po_decide_sustain(self, p, results, c, ev, why, prob, hi, lo):
        """PO from SUSTAINED evidence (rules.PO.evidence_sustain_sec > 0), per arm.

        Each frame, each arm is one of
          POCKET       the pocket model says pocket (>= evidence_hi) and the
                       closer re-pose agrees
          VISIBLE_OUT  a hand is seen: the model says not a pocket (<=
                       evidence_lo), or the wrist is away from the hip, the
                       forearm folded up, both hands together in front
                       (evidence_out_why)
          UNSURE       the model looked and could not tell
          UNKNOWN      not judged: seen from behind, too small, wrist missing,
                       the two poses disagreed, night IR, cut off
        An arm is a CANDIDATE while POCKET time accumulates, and CONFIRMED once
        it holds evidence_sustain_sec of it. VISIBLE_OUT starts it over - a hand
        seen outside is the strongest evidence there is. UNSURE takes time off
        (evidence_unsure: decay). UNKNOWN counts neither way, but only for
        evidence_unknown_max_sec: a hidden hand is never a pocket by itself.
        ONE arm confirmed is PO; the other may do anything.

        Why: the live rule fired after 3 POCKET frames in a 1 s window. On the
        4-6 Oct live alerts, the false ones were mostly 1-5 frame runs of
        POCKET between frames where the hand was seen (hands swinging past the
        hip, holding paper or a phone, hands behind the back), while real
        pockets held 10-26 frames running (dx test, 28 Sep test, 5 Oct live).
        Once confirmed, PO stays hold_sec past the arm's last POCKET frame, and
        ends at once when that hand is seen out.
        """
        tid = p.track_id
        now = time.monotonic()
        need = float(c.get("evidence_sustain_sec", 1.0))
        cap = float(c.get("evidence_dt_cap", 0.4))
        unk_max = float(c.get("evidence_unknown_max_sec", 1.0))
        gap = float(c.get("evidence_gap_sec", 1.5))
        out_why = set(c.get("evidence_out_why", ["away", "folded", "together"]))
        unsure_mode = c.get("evidence_unsure", "decay")
        hold = float(c.get("hold_sec", 1.5))
        # 7 Oct 2026 (user, Dx test: "the red box must stay as long as the alert"):
        # a CONFIRMED arm is latched - it ends only after evidence_release_out_sec
        # of the hand seen OUT (not one frame), or evidence_latch_max_sec with no
        # POCKET frame at all, or the track is gone longer than evidence_gap_sec.
        latch = bool(c.get("evidence_latch", False))
        rel_out = float(c.get("evidence_release_out_sec", 1.0))
        latch_max = float(c.get("evidence_latch_max_sec", 6.0))
        st = self._po_arm_st.setdefault(tid, {"seen": now})
        st["seen"] = now
        shown = False
        states = {}
        for side in ("l", "r"):
            a = st.setdefault(side, {"acc": 0.0, "t": None, "ev_t": None, "pocket_t": None, "on": False})
            was_on = a["on"]                        # 9 Oct: object check only when PO is about to fire
            P = prob[side]
            if ev[side] is True:
                k = "POCKET"
            elif P is not None and P <= lo:
                k = "VISIBLE_OUT"
            elif P is not None and P < hi:
                k = "UNSURE"
            elif P is None and why[side] in out_why:
                k = "VISIBLE_OUT"
            else:
                k = "UNKNOWN"
            dt = 0.0 if a["t"] is None else min(now - a["t"], cap)
            if a["t"] is not None and now - a["t"] > gap:
                a["acc"], a["on"] = 0.0, False
            a["t"] = now
            if k == "POCKET":
                a["acc"] = a["acc"] + dt if a["acc"] > 0 else 1e-6
                a["ev_t"] = a["pocket_t"] = now
                a["out"] = 0.0
            elif k == "VISIBLE_OUT" and latch and a["on"] and P is None:
                pass                     # latched: a pose guess (away / folded) does not release
            elif k == "VISIBLE_OUT" and latch and a["on"]:
                a["out"] = a.get("out", 0.0) + dt
                a["ev_t"] = now
                if a["out"] >= rel_out:
                    a["acc"], a["on"], a["out"] = 0.0, False, 0.0
            elif k == "VISIBLE_OUT":
                a["acc"], a["on"] = 0.0, False
                a["ev_t"] = now
            elif k == "UNSURE":
                if unsure_mode == "reset":
                    a["acc"] = 0.0
                elif unsure_mode == "decay":
                    a["acc"] = max(0.0, a["acc"] - max(dt, 0.15))
                a["ev_t"] = now
            elif a["ev_t"] is not None and now - a["ev_t"] > unk_max:
                a["acc"] = 0.0
            if a["acc"] >= need:
                a["on"] = True
            # 9 Oct (user): a hand that MATCHES the user's OK PO screenshots is PO in
            # ONE frame; any other hand needs pocket_min_frames POCKET frames
            # (within pocket_window_sec). Live counters had shown real pockets with
            # only 1-3 POCKET frames among many 'out' ones - the 0.8 s sustain never filled.
            likely = P is not None and P >= float(c.get("ref_min_p", hi)) and why[side] != "re-pose"
            if k == "POCKET" or likely:
                rs = self._po_refsim.get((tid, side))
                fresh = rs is not None and rs[0] == self._frame_i
                sim = rs[1] if fresh else 0.0
                if fresh and rs[2] is not None:
                    sim = rs[1] - rs[2]            # OK-vs-NG margin (ref_ng_dirs)
                if tid in self.po_cand:
                    cc = self.po_cand[tid].setdefault("cnt", {})
                    if fresh:                           # best OK-vs-NG margin of the track (po_cand teach)
                        cc["refsim"] = max(cc.get("refsim", -1.0), round(sim, 3))
                        cc["refn"] = cc.get("refn", 0) + 1
                # 10 Oct 13:05 (ID67): the one-frame master match only when the
                # pocket models themselves say pocket (P >= hi) - an unsure hand
                # (blurred, half out of the picture) matched a master at 0.15
                matched = (c.get("ref_sim") is not None and sim >= float(c["ref_sim"])
                           and P is not None and P >= hi)
                similar = c.get("ref_sim_similar") is not None and sim >= float(c["ref_sim_similar"])
                if matched:
                    a["on"] = True                     # matches an OK PO master: one frame
                    a["pocket_t"] = now
                mf = c.get("pocket_min_frames")
                if mf is not None and (k == "POCKET" or similar):
                    # a 'likely' hand (P below hi) counts only when it is similar to a master
                    win = float(c.get("pocket_window_sec", 4.0))
                    pk = [(t, sm) for t, sm in a.get("pk", []) if now - t <= win] + [(now, similar)]
                    a["pk"] = pk
                    n_sim = sum(1 for _, sm in pk if sm)
                    if n_sim >= int(c.get("pocket_min_frames_similar", 2)) or len(pk) >= int(mf):
                        a["on"] = True                 # 2 similar frames, or 3 pocket frames
                        a["pocket_t"] = now
            # 9 Oct 16:40: the ~50 ms object model ran on every POCKET frame - with v7's
            # lower threshold that cut the Toolroom to 1-2 fps. Now only on the frame
            # PO would START: a hand holding a bottle / phone / book... does not fire.
            if (a["on"] and not was_on and c.get("evidence_object_veto")
                    and c.get("evidence_object_at_fire", True) and getattr(self, "_po_frame", None) is not None):
                mc_ = c["min_keypoint_conf"]
                if self._po_object_at_hand(self._po_frame, p.kp(f"{side}_wrist", mc_), p.torso_len(mc_), c):
                    a["on"] = False
                    a["pk"] = []
                    a["acc"] = 0.0
            if a["on"] and (a["pocket_t"] is None or now - a["pocket_t"] > (latch_max if latch else hold)):
                a["on"] = False
            shown = shown or a["on"]
            states[side] = ("CONFIRMED" if a["on"] else "CANDIDATE" if a["acc"] > 0 else k, round(a["acc"], 2))
            if tid in self.po_cand and prob[side] is not None:          # 9 Oct: why no alert (po_cand.py)
                cc = self.po_cand[tid].setdefault("cnt", {})
                kk = "REPOSE" if why[side] == "re-pose" else k
                cc[kk] = cc.get(kk, 0) + 1
                cc["acc"] = max(cc.get("acc", 0.0), round(a["acc"], 2))
        if self.debug and tid in self.po_dbg:
            for side, (s, acc) in states.items():
                self.po_dbg[tid]["sides"][side].update(state=s, acc=acc)
        if shown:
            results.append({"track_id": tid, "rule": "PO", "bbox": p.bbox})

    def _po_geo(self, p, side, mc, torso, fh):
        """Arm geometry for one arm, in torso lengths (debug / evidence).
        dx_out: wrist beside its hip, + outward (away from the other hip),
        - toward the middle of the body (behind the back / in front of it);
        dy: wrist below the hip (+); drop: wrist below the elbow (+);
        e_out: elbow out from the shoulder (+ outward, hands on hips)."""
        o ="r" if side == "l" else "l"
        S, E, W, H = (p.kp(f"{side}_{n}", mc) for n in ("shoulder", "elbow", "wrist", "hip"))
        H2 = p.kp(f"{o}_hip", mc)
        if W is None or H is None or not torso or torso <= 1:
            return None
        s = 1.0 if H2 is None else (1.0 if H[0] >= H2[0] else -1.0)
        g = {"dx_out": round(float((W[0] - H[0]) * s / torso), 2),
             "dy": round(float((W[1] - H[1]) / torso), 2),
             "wconf": round(float(p.keypoints[9 if side == "l" else 10][2]), 2),
             "torso": round(float(torso)),
             "hipw": round(float(abs(H[0] - H2[0]) / torso), 2) if H2 is not None else None,
             "face": round(float(max(p.keypoints[0][2], p.keypoints[1][2], p.keypoints[2][2])), 2),
             "legs": bool(self._legs_upright(p)),
             "bottom": bool(p.bbox[3] >= 0.99 * fh),
             "spd": round(self.speed_ps(p.track_id) / torso, 2)}
        if E is not None:
            g["drop"] = round(float((W[1] - E[1]) / torso), 2)
            if S is not None:
                g["angle"] = round(_angle(S, E, W))
                g["e_out"] = round(float((E[0] - S[0]) * s / torso), 2)
        mv = self.wrist_movement(p.track_id, side)
        g["mv"] = round(mv, 3) if mv is not None else None
        return g

    def _po_refs(self, c, model):
        """The user's OK / NG PO masters (rules.PO.ref_dirs / ref_ng_dirs), built on
        a background thread; rebuilt when those folders change (ref_reload_sec,
        9 Oct: the teacher in po_cand.py adds to them) - the old set stays in use
        until the new one is ready."""
        dirs = c.get("ref_dirs")
        if not dirs:
            return None
        key = (tuple(dirs), tuple(c.get("ref_ng_dirs") or ()), id(model))
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        absd = lambda ds: [d if os.path.isabs(d) else os.path.join(root, d) for d in ds]
        meta = _PO_REFS_META.setdefault(key, {"checked": 0.0, "sig": None, "busy": False})

        def sig():
            out = []
            for d in absd(list(dirs) + list(c.get("ref_ng_dirs") or [])):
                for dp, _, fs in os.walk(d):
                    out += [(dp, f, int(os.path.getmtime(os.path.join(dp, f)))) for f in fs]
            return hash(tuple(sorted(out)))

        def build(sg):
            from .po_refs import PORefs
            try:
                kp_ = float(c.get("ref_keep_p", 0.7))   # 9 Oct: the user's label wins - a hand the model doubts is a master too
                r = PORefs(absd(dirs), model, pose, imgsz=int(c.get("evidence_repose_imgsz", 320)), keep_p=kp_)
                print(f"[PO] OK-PO masters: {len(r.E)} pocket hands (of {r.n_arms} hands at the hip) "
                      f"from {r.n_files} files; median master-to-master similarity {r.self_sim}", flush=True)
                r.ng = None
                ng = c.get("ref_ng_dirs")
                if ng:
                    r.ng = PORefs(absd(ng), model, pose, imgsz=int(c.get("evidence_repose_imgsz", 320)), keep_p=kp_)
                    print(f"[PO] NG-PO masters: {len(r.ng.E)} false-pocket hands from {r.ng.n_files} files", flush=True)
                _PO_REFS[key] = r
                meta["sig"] = sg
            except Exception as e:
                print(f"[PO] OK-PO masters not loaded: {e}", flush=True)
            finally:
                meta["busy"] = False

        now = time.time()
        reload_sec = float(c.get("ref_reload_sec", 0) or 0)
        first = key not in _PO_REFS
        if first or (reload_sec > 0 and not meta["busy"] and now - meta["checked"] >= reload_sec):
            meta["checked"] = now
            if first:
                _PO_REFS[key] = None            # until ready: no match
            try:
                sg = sig()
            except OSError:
                sg = None
            if first or sg != meta["sig"]:
                pose = self._repose_model(c)
                meta["busy"] = True
                import threading
                threading.Thread(target=build, args=(sg,), name="po-masters", daemon=True).start()
        return _PO_REFS[key]

    def _po_object_cached(self, tid, side, frame, W, torso, c):
        """_po_object_at_hand at most once per evidence_object_every_sec per hand
        (the object model takes ~50 ms; a held bottle stays in the hand)."""
        cache = self.__dict__.setdefault("_po_obj", {})
        now = time.time()
        hit = cache.get((tid, side))
        if hit is not None and now - hit[0] < float(c.get("evidence_object_every_sec", 1.0)):
            return hit[1]
        res = self._po_object_at_hand(frame, W, torso, c)
        cache[(tid, side)] = (now, res)
        if len(cache) > 500:
            for k in [k for k, v in cache.items() if now - v[0] > 10]:
                del cache[k]
        return res

    def _po_object_at_hand(self, frame, W, torso, c):
        """True when an object of rules.PO.evidence_object_veto (COCO classes) is
        held at this wrist: a box with its centre within evidence_object_near torso."""
        if W is None or torso <= 1:
            return False
        try:
            from .ke import _shared_model
            det = _shared_model((self.rcfg.get("KE") or {}).get("phone_model", "yolo11m.pt"))
            fh, fw = frame.shape[:2]
            h = 0.7 * torso
            x1, y1 = max(0, int(W[0] - h)), max(0, int(W[1] - h))
            x2, y2 = min(fw, int(W[0] + h)), min(fh, int(W[1] + h))
            if x2 - x1 < 8 or y2 - y1 < 8:
                return False
            crop = frame[y1:y2, x1:x2]
            sz = int(c.get("evidence_object_imgsz", 256))
            s = sz / float(max(crop.shape[:2]))
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
            r = det.predict(crop, classes=list(c["evidence_object_veto"]), imgsz=sz,
                            conf=float(c.get("evidence_object_conf", 0.3)), verbose=False)[0]
            if r.boxes is None or len(r.boxes) == 0:
                return False
            near = float(c.get("evidence_object_near", 0.4)) * torso
            for a, b, cc, d in r.boxes.xyxy.cpu().numpy().tolist():
                cx, cy = x1 + (a + cc) / 2.0 / s, y1 + (b + d) / 2.0 / s
                if ((cx - W[0]) ** 2 + (cy - W[1]) ** 2) ** 0.5 <= near:
                    return True
        except Exception as e:
            self._warn_once(f"[PO] object check off: {e}")
        return False

    def _repose_model(self, c):
        """The second pose model (one for every camera). A separate instance:
        the detector's own one carries the tracker state, and predict() on it
        would feed the crop's people into ByteTrack."""
        key = self.cfg.get("models", {}).get("pose", "yolo11n-pose.pt")
        if key not in _REPOSE:
            from ultralytics import YOLO
            _REPOSE[key] = YOLO(key)
            print(f"[PO] closer re-pose ready ({key}, imgsz {int(c.get('evidence_repose_imgsz', 320))})",
                  flush=True)
        return _REPOSE[key]

    def _po_repose(self, p, frame, c):
        """Keypoints (17x3, frame coords) of this person from a second pose
        pass on their own crop, or None (person not found in it)."""
        model = self._repose_model(c)
        x1, y1, x2, y2 = [float(v) for v in p.bbox]
        w, h = x2 - x1, y2 - y1
        fh, fw = frame.shape[:2]
        X1, Y1 = int(max(0, x1 - 0.3 * w)), int(max(0, y1 - 0.1 * h))
        X2, Y2 = int(min(fw, x2 + 0.3 * w)), int(min(fh, y2 + 0.1 * h))
        if X2 - X1 < 16 or Y2 - Y1 < 16:
            return None
        r = model.predict(frame[Y1:Y2, X1:X2], imgsz=int(c.get("evidence_repose_imgsz", 320)),
                                 conf=0.25, verbose=False,
                                 device=self.cfg.get("models", {}).get("device") or "cpu")[0]
        if r.keypoints is None or len(r.boxes) == 0:
            return None
        boxes = r.boxes.xyxy.cpu().numpy() + np.array([X1, Y1, X1, Y1])

        def iou(a, b):
            ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
            return ix * iy / u if u > 0 else 0.0

        k = int(np.argmax([iou(b, (x1, y1, x2, y2)) for b in boxes]))
        if iou(boxes[k], (x1, y1, x2, y2)) < 0.3:
            return None
        kp = r.keypoints.data[k].cpu().numpy().copy()
        kp[:, 0] += X1
        kp[:, 1] += Y1
        return kp

    @staticmethod
    def _repose_at_hip(kp, W, mc=0.3):
        """Does the closer pose also put a hand at the hip near the live wrist W?
        True (also when it cannot tell: no wrist or no torso found); False when
        it found wrists and none of them is at a hip within 0.6 torso of W."""
        if W is None:
            return True
        sh = [kp[i, :2] for i in (5, 6) if kp[i, 2] >= mc]
        hp = [kp[i, :2] for i in (11, 12) if kp[i, 2] >= mc]
        if not sh or not hp:
            return True
        t = float(np.linalg.norm(np.mean(sh, axis=0) - np.mean(hp, axis=0)))
        if t <= 1:
            return True
        found = False
        for wi, hi in ((9, 11), (10, 12)):
            if kp[wi, 2] < mc:
                continue
            found = True
            H = kp[hi, :2] if kp[hi, 2] >= mc else np.mean(hp, axis=0)
            w2 = kp[wi, :2]
            if (abs(w2[1] - H[1]) <= 0.5 * t and abs(w2[0] - H[0]) <= 0.6 * t
                    and float(np.linalg.norm(w2 - W)) <= 0.6 * t):
                return True
        return not found

    def _po_decide_arms(self, p, results, c, ev):
        """_po_decide per ARM: PO once one arm has been POCKET in >= confirm_ratio
        of its judged frames of the last confirm_sec, and in >= confirm_min_frames
        of them (ev: side -> True / False / None = no evidence this frame). A
        pocket seen on the left one frame and on the right the next is two
        glimpses, not a pocket. Once shown, PO holds hold_sec past the last
        POCKET frame of either arm."""
        tid = p.track_id
        now = time.monotonic()
        # evidence_* override the shared confirm_* (which _rule_PO_geom uses),
        # so switching pipelines back and forth leaves the old rule untouched
        win = float(c.get("evidence_confirm_sec", c.get("confirm_sec", 1.0)))
        need = int(c.get("evidence_min_frames", c.get("confirm_min_frames", 3)))
        ratio = float(c.get("evidence_ratio", c.get("confirm_ratio", 0.6)))
        arms = self._po_arm_ev.setdefault(tid, {"l": deque(), "r": deque()})
        confirmed = False
        for side, e in ev.items():
            q = arms[side]
            if e is not None:
                q.append((now, bool(e)))
                if e:
                    self._po_seen[tid] = now
            while q and now - q[0][0] > win:
                q.popleft()
            k = sum(1 for _, v in q if v)
            if len(q) >= need and k >= need and k >= ratio * len(q):
                confirmed = True
        if self._po_on.get(tid):
            if confirmed or now - self._po_seen.get(tid, 0.0) <= float(c.get("hold_sec", 1.5)):
                results.append({"track_id": tid, "rule": "PO",
                                "bbox": p.bbox})
                return
            self._po_on.pop(tid, None)
        if confirmed:
            self._po_on[tid] = True
            results.append({"track_id": tid, "rule": "PO",
                            "bbox": p.bbox})

    def _po_decide(self, p, results, c, pocket):
        """Show PO once the pocket test has passed in most of the last
        confirm_sec: >= confirm_ratio of the frames, and >= confirm_min_frames
        of them. Once shown it holds hold_sec past the last pocket frame, so a
        blink of the test does not blink the alert. `pocket` None = this frame
        could not be judged (no arm measured): no evidence either way.

        The hold used to feed the streak BEFORE any alert: one frame of a hand
        swinging past the hip (a walker seen from behind, 11:23 on 28 Sep) kept
        the streak alive for hold_sec and became an alert. On the 28 Sep alert
        clips this window let 7 of 17 through - all cut off at the frame bottom
        or clasping their hands - and still fired on the 25 Sep pocket tester.
        """
        tid = p.track_id
        now = time.monotonic()
        ev = self._po_ev[tid]
        if pocket is not None:
            ev.append((now, bool(pocket)))
            if pocket:
                self._po_seen[tid] = now
        win = float(c.get("confirm_sec", 1.0))
        while ev and now - ev[0][0] > win:
            ev.popleft()
        if self._po_on.get(tid):
            if now - self._po_seen.get(tid, 0.0) <= float(c.get("hold_sec", 1.5)):
                results.append({"track_id": tid, "rule": "PO",
                                "bbox": p.bbox})
                return
            self._po_on.pop(tid, None)
        need = int(c.get("confirm_min_frames", 3))
        k = sum(1 for _, v in ev if v)
        if len(ev) >= need and k >= need and k >= float(c.get("confirm_ratio", 0.6)) * len(ev):
            self._po_on[tid] = True
            results.append({"track_id": tid, "rule": "PO",
                            "bbox": p.bbox})

    @staticmethod
    def _hands_together(p, torso):
        """Both hands close together in front of the body: clasped, or holding
        something (a phone, papers, a box). Hands in pockets sit a hip-width
        apart, and one hand in a pocket leaves the other free - neither is
        this. Needs a frontal view: side-on, the two wrists overlap whatever
        the hands are doing."""
        Wl, Wr = p.kp("l_wrist", 0.5), p.kp("r_wrist", 0.5)
        if Wl is None or Wr is None:
            return False
        if p.shoulder_width(0.4) < 0.45 * torso:           # side-on
            return False
        return bool(np.linalg.norm(Wl - Wr) < 0.35 * torso)


    @staticmethod
    def _legs_upright(p, mc=0.3):
        """True when the legs go DOWN from the hips, close to vertical: standing
        (or bending over at the waist), not lying on the floor."""
        def mean(names):
            pts = [q for q in (p.kp(n, mc) for n in names) if q is not None]
            return np.mean(pts, axis=0) if pts else None
        hip = mean(("l_hip", "r_hip"))
        foot = mean(("l_ankle", "r_ankle"))
        if foot is None:
            foot = mean(("l_knee", "r_knee"))
        if hip is None or foot is None:
            return False
        v = foot - hip
        if v[1] < 0.5 * p.torso_len(mc):          # feet must be well below the hips
            return False
        return float(np.degrees(np.arctan2(abs(v[0]), v[1]))) < 35.0

    @staticmethod
    def _po_measure(p, side, mc, torso):
        """Raw PO measurements for one arm (None if keypoints missing).

        dist  = wrist-hip distance / torso     (small => wrist at hip/pocket)
        angle = elbow angle (deg)              (~180 straight/hanging, smaller = bent)
        tuck  = how far wrist is pulled toward the hip vs the elbow / torso (+ = tucked in)
        hang  = wrist height below hip / torso (+large = arm hanging down to the thigh)
        """
        S = p.kp(f"{side}_shoulder", mc)
        E = p.kp(f"{side}_elbow", mc)
        W = p.kp(f"{side}_wrist", mc)
        H = p.kp(f"{side}_hip", mc)
        if S is None or E is None or W is None or H is None:
            return None
        return {
            "dist": float(np.linalg.norm(W - H) / torso),
            "angle": _angle(S, E, W),
            "tuck": float((abs(E[0] - H[0]) - abs(W[0] - H[0])) / torso),
            "hang": float((W[1] - H[1]) / torso),
            "W": W,
        }

    # ---- TE: hold the handrail on stairs -------------------------------

    def rule_TE(self, persons, results):
        if self._blocked("TE"):
            return
        for p in persons:
            if not self.zones.contains("stairs", p.feet()):
                self._clear(p.track_id, "TE")
                continue
            holding = any(
                self.zones.contains("handrail", p.kp(w, 0.3))
                for w in ("l_wrist", "r_wrist")
            )
            if holding:
                self._clear(p.track_id, "TE")
            else:
                self._fire(results, p, "TE")

    # ---- NA: use the sides of roads -----------------------------------

    def rule_NA(self, persons, results):
        """Naname = diagonal. See na.py.

        Confirmation is in SECONDS inside the detector, so this does not also go
        through _fire()'s frame streak - stacking the two would make the real hold
        time depend on the frame rate.
        """
        fz = (self.rcfg.get("NA") or {}).get("forbidden_zone")
        if fz:
            return self._na_forbidden(persons, results, fz)
        ready = [na for na in self.na_list if na.ready()]
        if not ready:
            self._skip("NA")
            return
        fired = set()
        for na in ready:                       # one per zone (rules.NA.per_zone)
            for tid, v in na.update(persons).items():
                if v and v[0]:
                    fired.add(tid)
        ex = (self.rcfg.get("NA") or {}).get("exempt_zone")
        if ex and fired and self.zones.has(ex):
            fired -= self._na_exempt(persons, fired, ex)
        for p in persons:
            if p.track_id in fired:
                results.append({
                    "track_id": p.track_id, "rule": "NA",
                    "bbox": p.bbox,
                })

    def _na_forbidden(self, persons, results, zone):
        """rules.NA.forbidden_zone (8 Oct 2026, Material Gate, the user): the
        STOP area below the zebra crossing is for trucks only - a person whose
        feet stay in it for forbidden_sec is NA, shown while they are there
        (+ show_sec). Walking inside the gangway / crossing / working zones,
        and from the crossing into the working zone, is not judged."""
        if not self.zones.has(zone):
            self._skip("NA")
            return
        c = self.rcfg.get("NA") or {}
        need = float(c.get("forbidden_sec", 1.0))
        gap = float(c.get("forbidden_gap_sec", 1.0))
        show = float(c.get("show_sec", 2.0))
        now = time.monotonic()
        st = self.__dict__.setdefault("_na_fz", {})
        for p in persons:
            f = p.feet()
            if f is None:
                continue
            s = st.setdefault(p.track_id, {"in": 0.0, "t": None, "last_in": -1e9, "until": 0.0})
            dt = 0.0 if s["t"] is None else min(now - s["t"], 0.5)
            s["t"] = now
            if self.zones.contains(zone, f):
                s["in"] += dt
                s["last_in"] = now
                if s["in"] >= need:
                    s["until"] = now + show
            elif now - s["last_in"] > gap:
                s["in"] = 0.0
            if s["until"] > now:
                results.append({"track_id": p.track_id, "rule": "NA", "bbox": p.bbox})
        if len(st) > 300:
            live = {p.track_id for p in persons}
            for k in [k for k in st if k not in live][:150]:
                del st[k]

    def _na_exempt(self, persons, fired, zone):
        """rules.NA.exempt_zone (7 Oct 2026, Material Gate 'working' = the gate and
        the booth): a person who crosses INTO that zone is going to work there, so
        the crossing is not NA. Exempt = the feet are in the zone, were in it
        within exempt_hold_sec, or the walk of the last second, carried on for
        up to exempt_lookahead_sec, lands in it."""
        c = self.rcfg.get("NA") or {}
        hold = float(c.get("exempt_hold_sec", 30.0))
        ahead = float(c.get("exempt_lookahead_sec", 4.0))
        now = time.monotonic()
        hist = self.__dict__.setdefault("_na_ex_hist", {})
        seen = self.__dict__.setdefault("_na_ex_seen", {})
        out = set()
        for p in persons:
            tid = p.track_id
            f = p.feet()
            if f is None:
                continue
            f = np.asarray(f, dtype=float)
            h = hist.setdefault(tid, deque(maxlen=40))
            h.append((now, f))
            if self.zones.contains(zone, f):
                seen[tid] = now
            if tid not in fired:
                continue
            if now - seen.get(tid, -1e9) <= hold:
                out.add(tid)
                continue
            old = next(((t, xy) for t, xy in h if now - t <= 1.2), None)
            if old is None or now - old[0] < 0.3:
                continue
            v = (f - old[1]) / (now - old[0])
            if any(self.zones.contains(zone, f + v * k) for k in np.arange(0.5, ahead + 0.01, 0.5)):
                out.add(tid)
        if len(hist) > 200:
            live = {p.track_id for p in persons}
            for k in [k for k in hist if k not in live][:100]:
                hist.pop(k, None)
                seen.pop(k, None)
        return out

    # ---- SHI: confirm the situation (experimental) --------------------

    def rule_SHI(self, persons, results):
        if self._blocked("SHI"):
            return
        c = self.rcfg["SHI"]
        if c.get("mode") == "exit":
            return self._rule_SHI_exit(persons, results, c)
        pspd, pframes = c["pause_speed_px"], c["pause_frames"]
        head_t = c.get("head_move_thresh", 0.25)
        for p in persons:
            st = self.shi_state[p.track_id]
            if not self.zones.contains("checkpoint", p.feet()):
                # left the checkpoint -> reset this visit
                st["pause"], st["confirmed"] = 0, False
                self._clear(p.track_id, "SHI")
                continue
            # "Confirmed" = the person either paused OR looked around (head movement).
            if self.speed(p.track_id) < pspd:
                st["pause"] += 1
            looked_around = self.head_swing(p.track_id) >= head_t
            if st["pause"] >= pframes or looked_around:
                st["confirmed"] = True
            if st["confirmed"]:
                self._clear(p.track_id, "SHI")
            else:
                self._fire(results, p, "SHI")

    def _rule_SHI_exit(self, persons, results, c):
        """SHI, mode exit (the Material Gate): STOP - LOOK - GO at the painted pads.

        7 Oct 2026, from the user's four labelled clips (material gate clips/:
        133009, 133645, 132948 OK; 133357 NG - "had to stop for some seconds"):
        STOP is required - the person must stay nearly in place (the box's bottom
        within stop_move_torso torso lengths) for stop_sec (user: 0.5 s) on the
        pad or within pad_margin of it. A head turn alone does not pass
        (need_stop; the head-yaw proxy is noise at ~200 px tall).
        The verdict is given once they GO: go_min_torso away from where they
        left the pad, in the direction its painted feet point (go_dir, within
        60 deg). Leaving towards exempt_zone (the gate / booth 'working' zone),
        backwards or sideways along the kerb is not judged. Shown show_sec.
        """
        now = time.monotonic()
        go_zone = c.get("go_zone", "")
        go_dir = c.get("go_dir") or {}
        need = float(c.get("stop_sec", c.get("pause_sec", 0.5)))
        move = float(c.get("stop_move_torso", 0.8))
        need_stop = bool(c.get("need_stop", True))
        head_t = float(c.get("head_move_thresh", 0.25))
        mx, mup, mdn = (c.get("pad_margin") or [40, 90, 30])
        go_min = float(c.get("go_min_torso", 1.0))
        ex = c.get("exempt_zone")
        show = float(c.get("show_sec", 3.0))
        pads = {k: np.asarray(v, float).reshape(-1, 2) for k, v in self.zones.polys.items()
                if (k == "checkpoint" or k.startswith("checkpoint_")) and len(v) >= 3}

        def pad_at(f):
            hit = self.zones.part_of("checkpoint", f)
            if hit is not None or f is None:
                return hit
            for n, poly in pads.items():
                if poly[:, 0].min() - mx <= f[0] <= poly[:, 0].max() + mx and \
                        poly[:, 1].min() - mup <= f[1] <= poly[:, 1].max() + mdn:
                    return n
            return None

        for p in persons:
            st = self.shi_state[p.track_id]
            x1, y1, x2, y2 = [float(v) for v in p.bbox]
            foot = np.array([(x1 + x2) / 2.0, y2])
            torso = p.torso_len() or 0.0
            if torso <= 1:
                torso = max(1.0, (y2 - y1) / 3.5)
            pad = pad_at(foot)
            st["trail"] = [q for q in st.get("trail", []) if now - q[0] <= 1.5] + [(now, foot)]
            if pad is not None:
                if st.get("pad") != pad:
                    if st.get("left_pad") != pad:          # a new visit (not a re-entry)
                        st["ok"] = False
                    st.update(pad=pad, pts=[], left_pad=None, left_at=None, slow=False)
                st["pts"].append((now, foot, torso))
                st["pts"] = [q for q in st["pts"] if now - q[0] <= need + 1.0]
                win = [q for q in st["pts"] if now - q[0] <= need]
                min_pts = int(c.get("stop_min_pts", 3))       # 2 on a ~5 fps camera (8 Oct, Help-desk)
                if win and now - win[0][0] >= 0.8 * need and len(win) >= min_pts:
                    far = max(float(np.linalg.norm(q[1] - win[0][1])) for q in win)
                    if far <= move * float(np.median([q[2] for q in win])):
                        st["ok"] = True
                if not need_stop and self.head_swing(p.track_id) >= head_t:
                    st["ok"] = True
                # look_with_slow (8 Oct, Help-desk ID1130): slowing down on the poster
                # AND turning the head counts as STOP-LOOK, even when the stop is short
                lw = c.get("look_with_slow")
                if lw and len(st["pts"]) >= 2:
                    q0, q1 = st["pts"][-2], st["pts"][-1]
                    v = float(np.linalg.norm(q1[1] - q0[1])) / max(1e-3, q1[0] - q0[0]) / max(1.0, q1[2])
                    st["slow"] = st.get("slow", False) or v <= float(lw.get("max_speed_torso_ps", 1.0))
                    if st["slow"] and self.head_swing(p.track_id) >= float(lw.get("head_move", head_t)):
                        st["ok"] = True
                st["at"] = foot
            elif st.get("pad"):
                st.update(left_pad=st["pad"], left_at=st.get("at", foot), left_t=now, pad=None)
            if st.get("left_pad") and pad is None:
                d = foot - st["left_at"]
                dist = float(np.linalg.norm(d))
                if now - st.get("left_t", now) > 4.0:
                    st["left_pad"] = None
                elif dist >= go_min * torso:
                    gd = go_dir.get(st["left_pad"])
                    if gd is not None:
                        g = np.asarray(gd, float)
                        going = float(d @ g) >= float(c.get("go_cos", 0.5)) * dist * float(np.linalg.norm(g))
                    elif go_zone and self.zones.has(go_zone):
                        going = self.zones.contains(go_zone, foot)
                    else:
                        # 8 Oct: a poster whose feet direction was not set (Zone
                        # editor) - leaving it any way without a stop counts
                        going = True
                    if ex and self.zones.has(ex):
                        # heading for the gate / booth: in it now, or the walk of the
                        # last second carried on lands in it within 3 s
                        hist = [q for q in st.get("trail", []) if now - q[0] <= 1.2]
                        v = (foot - hist[0][1]) / max(1e-3, now - hist[0][0]) if hist and now - hist[0][0] >= 0.3 else np.zeros(2)
                        if any(self.zones.contains(ex, foot + v * k) for k in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)):
                            going = False
                    if going and not st.get("ok"):
                        st["bad_until"] = now + show
                    st["left_pad"] = None
            if st.get("bad_until", 0.0) > now:
                results.append({"track_id": p.track_id, "rule": "SHI", "bbox": p.bbox})

    # ---- RUN: running / moving too fast -------------------------------

    def rule_RUN(self, persons, results):
        """Speed in TORSO-LENGTHS PER SECOND, so the threshold means the same thing
        whatever frame rate the pipeline happens to be running at."""
        c = self.rcfg.get("RUN", {})
        thr = c.get("speed_torso_per_sec", 5.0)
        win = c.get("speed_window_sec", 0.5)
        for p in persons:
            torso = p.torso_len()
            if not torso or torso <= 1:
                self._clear(p.track_id, "RUN")
                continue
            if self.speed_ps(p.track_id, win) / torso >= thr:
                self._fire(results, p, "RUN")
            else:
                self._clear(p.track_id, "RUN")

    # ---- FALL: person fallen / lying down -----------------------------

    def rule_FALL(self, persons, results, frame=None):
        """Torso angle is the PRIMARY signal (a sitting or half-occluded person keeps
        a vertical torso, so they don't trigger). The bbox aspect ratio is used only
        as a FALLBACK when the torso keypoints aren't visible."""
        c = self.rcfg.get("FALL", {})
        aspect_t = c.get("aspect_thresh", 1.10)
        angle_t = c.get("torso_angle_deg", 60)
        edge_px = int(c.get("edge_margin_px", 4))
        for p in persons:
            ls, rs = p.kp("l_shoulder"), p.kp("r_shoulder")
            lh, rh = p.kp("l_hip"), p.kp("r_hip")
            if all(x is not None for x in (ls, rs, lh, rh)):
                vec = (lh + rh) / 2.0 - (ls + rs) / 2.0          # shoulders -> hips
                angle = np.degrees(np.arctan2(abs(vec[0]), abs(vec[1]) + 1e-6))  # 0=vertical
                fallen = angle >= angle_t
            else:
                # Torso not visible -> fall back to "wider than tall" box, but
                # ONLY for a box that is whole. A person standing at the edge of
                # the view is cut off by it, so their box is short and therefore
                # wide, and the ratio says "fallen" about somebody perfectly
                # upright. Measured over 70 s: the angle test fired 0 times and
                # the fallback fired 126, every one of them on a clipped box with
                # a ratio of 1.53-1.57 against a 1.30 threshold.
                x1, y1, x2, y2 = p.bbox
                clipped = (frame is not None
                           and (x1 <= edge_px or y1 <= edge_px
                                or x2 >= frame.shape[1] - edge_px
                                or y2 >= frame.shape[0] - edge_px))
                # ...and only for a detection the pose model is sure is a
                # person (FALL.fallback_min_conf). 7 Oct 02:00-02:23: a stack of
                # empty blue crates at the side gangway was a "person" (conf
                # 0.38-0.44, no torso, box ratio 1.31) - 36 FALL alerts in 23
                # minutes, a new track id each time. Real people there: 0.76.
                fallen = (not clipped
                          and float(getattr(p, "conf", 1.0)) >= float(c.get("fallback_min_conf", 0.0))
                          and max(x2 - x1, 1.0) / max(y2 - y1, 1.0) >= aspect_t)

            # A torso tipped past 60 degrees is also everybody who bends over to
            # lift a crate or a bucket - all 19 FALL alerts sampled from the live
            # log were exactly that. What separates them from a person on the
            # floor is the legs: bending, they still stand under the hips.
            if fallen and c.get("legs_veto", True) and self._legs_upright(p):
                fallen = False

            if fallen:
                self._fire(results, p, "FALL")
            else:
                self._clear(p.track_id, "FALL")

    # ---- RESTRICTED: person inside a restricted area ------------------

    def rule_RESTRICTED(self, persons, results):
        if self._blocked("RESTRICTED"):
            return
        for p in persons:
            if self.zones.contains("restricted", p.feet()):
                self._fire(results, p, "RESTRICTED")
            else:
                self._clear(p.track_id, "RESTRICTED")
