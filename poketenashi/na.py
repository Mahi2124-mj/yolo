"""NA (Naname = diagonal) - do not cross a walkway diagonally.

From the company rule sheet: "Never cross roads or walkways diagonally. Use the
designated crossing path."

So NA is about DIRECTION, not speed and not position. Three ways to move through
a gangway, and only one of them is wrong:

    along the aisle      ->  OK   (normal walking)
    straight across      ->  OK   (a proper crossing, shortest exposure)
    at an angle          ->  NA   (cutting the corner - longer time in the lane,
                                   and approaching traffic from an odd bearing)

The measurement is the angle between the person's direction of travel and the
gangway's own long axis, folded into 0-90 degrees so "up the aisle" and "down the
aisle" are the same thing. Near 0 is along, near 90 is across, and the band in
between is the violation.

Guards against firing on nothing:
  * only people whose FEET are inside the gangway are considered
  * a minimum travel distance (in torso lengths) - someone shuffling on the spot
    has a direction, but it is noise
  * the heading is taken over a time window, not frame to frame, so a single
    jittery keypoint cannot invent a diagonal
  * the verdict must hold for a configured number of SECONDS

RUN is deliberately NOT part of this. Running is a speed rule and keeps its own
identity; NA fires on a slow diagonal stroll just the same.

Two modes (rules.NA.mode):
  heading  the rule above - any diagonal heading anywhere inside the gangway
  entry    (3 Oct 2026, the user's rule) the same heading test, but only for
           someone cutting INTO the gangway from its side - the racks, the
           machines, a side gangway once the junction's grace is over. People
           already walking in the gangway who drift, change lane, step round
           material or angle off towards a side gangway are not judged; nor is
           a merge (in from the side, then along). See _update_entry.
"""
import time
from collections import defaultdict, deque

import cv2
import numpy as np

DEFAULTS = {
    "zone": "gangway",          # falls back to walkway_center if that is what exists
    "window_sec": 1.5,          # heading measured over this trailing window
    "min_travel": 0.8,          # ...and only once they have moved this many torsos
    "along_max_deg": 25.0,      # <= this off the axis = walking along, fine
    "across_min_deg": 65.0,     # >= this = a straight crossing, fine
    "confirm_sec": 0.8,         # the diagonal must hold this long
    "min_frames": 4,
    # Visible aisle length / its width, for the top-down view. Measured on the
    # aisle recording from walking speeds (people walk as fast along the aisle
    # as across it): ~8. Only changes how steep a diagonal counts as; along and
    # straight-across stay put.
    "aisle_length_ratio": 8.0,
    # A diagonal only counts once it has carried the person this far ACROSS
    # the aisle (fraction of its width) - a crossing, not a merge or a drift.
    "min_cross": 0.3,
    # Zone where OTHER gangways join this one (a T-junction). Walking left or
    # right there is walking along a gangway, and so is turning from it into
    # this aisle - neither is a diagonal crossing. Nobody inside it, or out of
    # it for less than junction_grace_sec, is judged.
    "junction_zone": "crossing",
    "junction_grace_sec": 2.0,
    # "entry" (3 Oct 2026, the user's rule): NA is a DIAGONAL ENTRY - someone
    # cutting INTO the gangway on a slant from its SIDE (the racks, the
    # machines, the junction once its grace is over). Someone already walking
    # in the gangway who drifts, changes lane, steps round material or angles
    # off towards a side gangway is not judged (see _update_entry).
    # "heading": the earlier rule - any diagonal heading anywhere inside it.
    "mode": "heading",
    # The side band: within this (width fraction) of a long edge counts as AT
    # the side - on the aisle camera the painted walkway lines sit at ~0.09 and
    # ~0.93, so 0.15 is the edge strip plus a step. Standing or walking along
    # there keeps a person "at the side", ready to enter.
    "edge_margin": 0.15,
    "entry_window_sec": 5.0,    # off the side band this long without crossing = no longer an entry
    "settle_sec": 1.5,          # walking ALONG this long off the side band = merged, not crossing
    # At the side this long (working at a rack or a machine) = min_cross is
    # counted from where they leave the side band, not from the edge: a
    # worker's step out with a trolley is not a crossing.
    "side_dwell_sec": 5.0,
    "max_gap_sec": 1.5,         # a track unseen this long starts over (a shorter gap keeps its state)
    "hold_cross": 0.0,          # how far across (width fraction) the person must go WHILE the diagonal is held
    "max_jump_torso": 2.5,      # a foot step longer than this (and faster than 6 torso/s) = tracking glitch
}


def _c(cfg, key):
    return cfg.get(key, DEFAULTS[key])


class NADetector:
    """Diagonal-crossing detector, keyed by track_id."""

    def __init__(self, cfg, zones=None):
        self.c = ((cfg.get("rules") or {}).get("NA") or {})
        self.zones = zones
        self.mc = 0.30
        self._hist = defaultdict(lambda: deque(maxlen=96))   # (t, foot_xy, torso)
        self._since = {}                                     # track -> t of first diagonal
        self._from = {}                                      # track -> ground point it began at
        self._junction_t = {}                                # track -> last time in the junction
        self._st = {}                                        # track -> entry state (mode: entry)
        self._seen = {}                                      # track -> last time seen (mode: entry)
        self._jx = {}                                        # track -> x when last in the junction (entry)
        self.debug = {}

        self.refresh()

    def refresh(self):
        """(Re)derive everything that comes from the gangway polygon - also after
        the zone editor has redrawn it on the running monitor."""
        zones = self.zones
        zone_name, axis, H = None, None, None   # H: image -> top-down aisle plane
        if zones is not None:
            for name in (_c(self.c, "zone"), "walkway_center"):
                if name and zones.has(name):
                    zone_name = name
                    break
        if zone_name is not None:
            poly = zones.polys[zone_name]
            axis = self._axis_of(poly)
            H = self._rectifier(poly, float(_c(self.c, "aisle_length_ratio")))
        self.zone_name, self.axis, self.H = zone_name, axis, H

    @staticmethod
    def _rectifier(poly, length_ratio):
        """Homography from the image to a TOP-DOWN view of the aisle.

        The camera looks down the aisle, so its two long edges converge on a
        vanishing point. Someone walking straight along the side of the aisle
        moves toward that point, which in the picture is a slant of 20-30
        degrees - and NA measured headings in the picture. On the live log 8-9
        of every 12 NA alerts were people walking along the aisle, whole
        groups of them at shift change.

        A four-corner zone is the aisle floor seen in perspective; mapping its
        corners onto a rectangle undoes the perspective. There, walking along
        is vertical and crossing straight over is horizontal wherever it
        happens in the picture. `length_ratio` (visible length / width of the
        aisle) only scales how steep a diagonal looks, not along or across.
        Returns None for zones that are not four-cornered (then the old
        picture-plane axis is used).
        """
        if len(poly) != 4:
            return None
        pts = poly.astype(np.float32)
        edges = [np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
        # The long, converging pair of sides runs ALONG the aisle.
        i0 = 0 if edges[0] + edges[2] >= edges[1] + edges[3] else 1
        # corners: a->b is one long side, d->c the other, a-d and b-c the short ends
        a, b, c, d = (pts[(i0 + k) % 4] for k in range(4))
        dst = np.float32([[0, 0], [0, length_ratio], [1, length_ratio], [1, 0]])
        return cv2.getPerspectiveTransform(np.float32([a, b, c, d]), dst)

    def _ground(self, xy):
        if self.H is None:
            return np.asarray(xy, dtype=float)
        p = cv2.perspectiveTransform(np.float32([[xy]]), self.H)
        return p[0, 0].astype(float)

    @staticmethod
    def _axis_of(poly):
        """Unit vector along the polygon's long side.

        minAreaRect gives the tightest rectangle around the lane; its longer edge
        is the direction the aisle runs. Deriving it from the polygon means the
        operator only ever draws the lane, never an axis.
        """
        (_cx, _cy), (w, h), angle = cv2.minAreaRect(poly.astype(np.float32))
        if w < h:                      # make `angle` describe the LONG side
            angle += 90.0
        rad = np.radians(angle)
        return np.array([np.cos(rad), np.sin(rad)], dtype=float)

    def ready(self):
        return self.zone_name is not None and self.axis is not None

    def update(self, persons, now=None):
        """Returns {track_id: (violating, angle_deg)} for everyone judged."""
        now = time.time() if now is None else now
        # entry mode needs the top-down plane (x = 0..1 across the lane); a zone
        # that is not four-cornered has none, so it keeps the heading rule
        if self.ready() and self.H is not None and _c(self.c, "mode") == "entry":
            return self._update_entry(persons, now)
        return self._update_heading(persons, now)

    def _heading(self, h, now, torso):
        """Heading over the trailing window: (deg to the aisle, ground point the
        window starts at), or None while there is too little track or travel."""
        old = next(((t, xy, ts) for t, xy, ts in h if now - t <= _c(self.c, "window_sec")), None)
        if old is None or now - old[0] < 0.3:
            return None
        travel = h[-1][1] - old[1]
        dist = float(np.linalg.norm(travel))
        if dist < _c(self.c, "min_travel") * torso:
            return None
        if self.H is not None:
            g = self._ground(h[-1][1]) - self._ground(old[1])
            gn = float(np.linalg.norm(g))
            if gn <= 1e-9:
                return None
            cos = abs(float(g[1]) / gn)
        else:
            cos = abs(float(travel @ self.axis) / dist)
        return float(np.degrees(np.arccos(max(0.0, min(1.0, cos))))), self._ground(old[1])

    def _update_entry(self, persons, now):
        """mode: entry - NA is a DIAGONAL ENTRY: cutting into the gangway on a
        slant FROM ITS SIDE.

        Per track, where the person stands relative to the gangway:
          side      outside it beyond a long edge (racks / machines), or inside
                    within edge_margin of one: AT the side. Standing or walking
                    along there keeps them there.
          fresh     came off a side, or out of the junction's grace, and has
                    not settled yet: the entry is being made - JUDGED
          settled   walking in the gangway (walked along settle_sec off the side
                    band, came in through an end of the lane, first seen
                    mid-lane, or off the side entry_window_sec without
                    crossing): NOT judged
          junction  in the junction or within junction_grace_sec of it: not
                    judged (exactly as in the heading mode)
        A fresh person is NA when the old heading test fires - the heading over
        window_sec on the top-down plane in the diagonal band, held for
        confirm_sec - AND they have cut min_cross of the width INTO the gangway,
        away from the side they came from. Counted from the edge they came in
        over; from where they left the side band if they had stood at the side
        side_dwell_sec (working at a rack); from where they left the junction.
        Straight across is fine, and so is a merge (in from the side, a few
        steps across, then along). A crossing that reaches the far side is
        still judged there.
        Track quality: state survives a gap up to max_gap_sec (the tracker drops
        people for single frames); a longer gap or an impossible foot jump
        starts the track over. Speed plays no part.
        """
        out = {}
        along, across = _c(self.c, "along_max_deg"), _c(self.c, "across_min_deg")
        jz = _c(self.c, "junction_zone")
        grace = float(_c(self.c, "junction_grace_sec"))
        e = float(_c(self.c, "edge_margin"))
        ew, settle = float(_c(self.c, "entry_window_sec")), float(_c(self.c, "settle_sec"))
        dwell = float(_c(self.c, "side_dwell_sec"))
        gap, jump = float(_c(self.c, "max_gap_sec")), float(_c(self.c, "max_jump_torso"))
        min_cross, confirm = float(_c(self.c, "min_cross")), float(_c(self.c, "confirm_sec"))

        def restart(tid):
            self._since.pop(tid, None)
            self._from.pop(tid, None)

        for p in persons:
            tid = p.track_id
            foot = p.feet()
            torso = p.torso_len(self.mc)
            if foot is None or not torso or torso <= 1:
                continue
            foot = np.asarray(foot, dtype=float)
            h = self._hist[tid]
            st = self._st.get(tid)
            if h:                                   # track quality: long gap or a jump -> start over
                t_last, xy_last, _ = h[-1]
                dt = now - t_last
                if dt > gap or float(np.linalg.norm(foot - xy_last)) > torso * max(jump, 6.0 * dt):
                    h.clear()
                    st = None
                    restart(tid)
                    self._junction_t.pop(tid, None)
                    self._jx.pop(tid, None)
            h.append((now, foot, float(torso)))
            self._seen[tid] = now
            x = float(self._ground(foot)[0])

            if not self.zones.contains(self.zone_name, foot):
                if x < 0 or x > 1:                  # beyond a long edge: at that side
                    side = -1 if x < 0 else 1
                    stay = st is not None and st["s"] == "side" and st["side"] == side
                    st = {"s": "side", "side": side, "xb": 0.0 if side < 0 else 1.0,
                          "since": st["since"] if stay else now}
                elif st is None or st["s"] != "side":
                    st = {"s": "end"}               # beyond an end of the lane: along it
                self._st[tid] = st
                restart(tid)
                continue
            # The junction as in the heading mode: only time spent in it INSIDE
            # the gangway starts the grace.
            if jz and self.zones.has(jz) and self.zones.contains(jz, foot):
                self._junction_t[tid] = now
                self._jx[tid] = x
            if now - self._junction_t.get(tid, -1e9) < grace:
                self._st[tid] = {"s": "junction"}
                restart(tid)
                continue

            band = -1 if x <= e else (1 if x >= 1 - e else 0)
            if band and st is not None and st["s"] == "fresh" and band != st["side"]:
                pass                                # finishing a crossing at the far side: still judged
            elif band:                              # at a side, in the lane's edge strip
                stay = st is not None and st["s"] == "side" and st["side"] == band
                st = {"s": "side", "side": band, "xb": x, "since": st["since"] if stay else now}
            elif st is None:                        # first seen mid-lane: already walking in it
                st = {"s": "settled"}
            elif st["s"] == "side":                 # just came off the side: the entry starts
                edge = 0.0 if st["side"] < 0 else 1.0
                st = {"s": "fresh", "side": st["side"], "t": now,
                      "x0": min(1.0, max(0.0, st["xb"])) if now - st["since"] >= dwell else edge}
            elif st["s"] == "junction":             # out of the junction's grace
                st = {"s": "fresh", "side": 0, "t": now, "x0": self._jx.get(tid, x)}
            elif st["s"] == "end":
                st = {"s": "settled"}
            self._st[tid] = st
            if st["s"] == "fresh" and now - st["t"] > ew:
                st = self._st[tid] = {"s": "settled"}
            if st["s"] not in ("fresh", "side"):
                restart(tid)
                continue

            hd = self._heading(h, now, torso)
            if hd is None:                          # not really going anywhere
                restart(tid)
                continue
            deg, _g_old = hd
            if deg <= along or deg >= across:       # along the lane, or straight across: fine
                if st["s"] == "fresh" and deg <= along:
                    st.setdefault("along_t", now)
                    if now - st["along_t"] >= settle:
                        self._st[tid] = {"s": "settled"}   # merged into the lane
                restart(tid)
                out[tid] = (False, deg)
                continue
            st.pop("along_t", None)
            self._since.setdefault(tid, now)
            self._from.setdefault(tid, x)           # where across the lane the diagonal began
            held = now - self._since[tid]
            if st["s"] == "side":                   # still at the side: nothing crossed yet
                out[tid] = (False, deg)
                continue
            dx = x - st["x0"]
            into = st["side"] == 0 or dx * st["side"] < 0
            # A turn is not a diagonal: someone who stepped in straight and
            # turns to walk along sweeps the heading window through the
            # diagonal band while no longer going across (18:44, 3 Oct). The
            # diagonal must carry the person on across while it is held.
            # Off by default (0): on the replay set a curve-in - in from the side
            # nearly straight, turning up the lane about a third of the way
            # across - looks just like this, and those were judged real NA.
            hc = float(_c(self.c, "hold_cross"))
            prog = x - self._from[tid]
            prog = abs(prog) if st["side"] == 0 else -prog * st["side"]
            out[tid] = (held >= confirm and into and abs(dx) >= min_cross
                        and (hc <= 0 or prog >= hc), deg)

        for d in (self._hist, self._since, self._from, self._junction_t, self._st, self._seen, self._jx):
            for k in [k for k in d if now - self._seen.get(k, -1e9) > gap]:
                del d[k]
        self.debug = out
        return out

    def _update_heading(self, persons, now):
        """mode: heading - the rule as it ran until 3 Oct: any diagonal heading
        inside the gangway (see the module docstring)."""
        out = {}
        live = set()
        if not self.ready():
            return out

        along = _c(self.c, "along_max_deg")
        across = _c(self.c, "across_min_deg")
        window = _c(self.c, "window_sec")
        min_travel = _c(self.c, "min_travel")

        for p in persons:
            tid = p.track_id
            live.add(tid)
            foot = p.feet()
            torso = p.torso_len(self.mc)
            if foot is None or not torso or torso <= 1:
                continue

            self._hist[tid].append((now, np.asarray(foot, dtype=float), float(torso)))
            if not self.zones.contains(self.zone_name, foot):
                self._since.pop(tid, None)          # left the lane - start over
                continue
            jz = _c(self.c, "junction_zone")
            if jz and self.zones.has(jz) and self.zones.contains(jz, foot):
                self._junction_t[tid] = now
            if now - self._junction_t.get(tid, -1e9) < float(_c(self.c, "junction_grace_sec")):
                self._since.pop(tid, None)          # on / just off the cross gangway
                self._from.pop(tid, None)
                continue

            h = self._hist[tid]
            old = next(((t, xy, ts) for t, xy, ts in h if now - t <= window), None)
            if old is None or now - old[0] < 0.3:
                continue

            travel = h[-1][1] - old[1]
            dist = float(np.linalg.norm(travel))
            if dist < min_travel * torso:
                self._since.pop(tid, None)          # not really going anywhere
                continue

            # Angle to the lane axis, folded to 0-90: "up the aisle" and "down the
            # aisle" are the same heading for this purpose. Measured on the
            # top-down aisle plane when there is one (see _rectifier).
            if self.H is not None:
                g = self._ground(h[-1][1]) - self._ground(old[1])
                gn = float(np.linalg.norm(g))
                if gn <= 1e-9:
                    continue
                cos = abs(float(g[1]) / gn)             # along = the rectified y axis
            else:
                cos = abs(float(travel @ self.axis) / dist)
            deg = float(np.degrees(np.arccos(max(0.0, min(1.0, cos)))))

            diagonal = along < deg < across
            if diagonal:
                # Remember where the diagonal began: the START of this heading
                # window, since the window is what showed the slant.
                self._since.setdefault(tid, now)
                self._from.setdefault(tid, self._ground(old[1]))
                held = now - self._since[tid]
                # A diagonal CROSSING cuts across the aisle. Someone stepping out
                # of the racks and turning to walk along it is also diagonal for
                # a second, but only while turning - they never get far across.
                # On the live log those merges were a large share of NA alerts.
                crossed = (self.H is None or
                           abs(float(self._ground(foot)[0] - self._from[tid][0]))
                           >= float(_c(self.c, "min_cross")))
                out[tid] = (held >= _c(self.c, "confirm_sec") and crossed, deg)
            else:
                self._since.pop(tid, None)
                self._from.pop(tid, None)
                out[tid] = (False, deg)

        for d in (self._hist, self._since, self._from, self._junction_t):
            for k in [k for k in d if k not in live]:
                del d[k]
        self.debug = out
        return out
