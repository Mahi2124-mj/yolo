"""Draw detections, pose skeletons, zones and violation labels onto a frame."""
import numpy as np
import cv2

# COCO skeleton edges (keypoint index pairs) for drawing the pose.
SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),       # arms
    (5, 6), (5, 11), (6, 12), (11, 12),    # torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
    (0, 5), (0, 6),                         # head -> shoulders
]

GREEN = (0, 200, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
GREY = (150, 150, 150)


def draw(frame, persons, phones, violations, zones=None, cfg=None, fps=None, po_dbg=None,
         wrist_trail=None, hand_skeletons=None, names=None):
    """`names`: track id -> employee code from face ID (poketenashi/faceid.py),
    shown in front of the box label as [code]."""
    out = frame
    out_cfg = (cfg or {}).get("output", {})

    if wrist_trail is not None:
        _draw_trails(out, wrist_trail)

    if hand_skeletons:
        _draw_hand_skeletons(out, hand_skeletons)

    if zones is not None and out_cfg.get("draw_zones", True):
        for name, poly in zones.polys.items():
            cv2.polylines(out, [poly], True, CYAN, 2)
            cv2.putText(out, name, tuple(poly[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1)

    # map track_id -> list of violated rule codes
    vmap = {}
    for v in violations:
        vmap.setdefault(v["track_id"], []).append(v["rule"])

    for p in persons:
        x1, y1, x2, y2 = p.bbox.astype(int)
        rules_here = vmap.get(p.track_id, [])
        # Grey = seen, but outside the zone the rules judge (or not yet tracked).
        judged = getattr(p, "in_zone", True) and getattr(p, "tracked", True)
        color = RED if rules_here else (GREEN if judged else GREY)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2 if judged or rules_here else 1)

        label = f"ID{p.track_id}" if getattr(p, "tracked", True) else "new"
        if names and p.track_id in names:
            # a recognised person IS their employee code (user, 7 Oct: "481, not ID991")
            label = str(names[p.track_id])
        if rules_here:
            label += "  ! " + ",".join(rules_here)
        cv2.putText(out, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if out_cfg.get("draw_keypoints", True):
            _draw_pose(out, p.keypoints)

        if po_dbg and p.track_id in po_dbg:
            _draw_po_debug(out, p, po_dbg[p.track_id])

    for ph in phones:
        x1, y1, x2, y2 = ph["bbox"].astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), BLUE, 2)
        cv2.putText(out, "phone", (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 1)

    if fps is not None:
        cv2.putText(out, f"FPS {fps:4.1f}", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(out, "Poketenashi: PO KE TE NA SHI", (10, out.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    return out


# MediaPipe 21-point hand skeleton edges (wrist=0, fingertips=4/8/12/16/20).
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm edge
]
FINGERTIP_IDS = (4, 8, 12, 16, 20)
MAGENTA = (255, 0, 255)


def _draw_hand_skeletons(frame, hands):
    """Draw the full 21-point finger skeleton for each detected hand."""
    for hd in hands:
        lms = hd.get("landmarks")
        if not lms or len(lms) < 21:
            continue
        pts = [tuple(int(v) for v in p) for p in lms]
        for a, b in HAND_EDGES:
            cv2.line(frame, pts[a], pts[b], MAGENTA, 1)
        for p in pts:
            cv2.circle(frame, p, 2, YELLOW, -1)
        for i in FINGERTIP_IDS:                      # fingertips highlighted
            cv2.circle(frame, pts[i], 3, RED, -1)


def _draw_trails(frame, wrist_trail):
    """Draw the recent wrist path; break the line on big jumps (track reuse / bad kp)."""
    jump = 0.18 * frame.shape[1]   # don't connect points farther apart than this
    for (_tid, _side), pts in wrist_trail.items():
        if len(pts) < 2:
            continue
        arr = np.array(pts, dtype=np.int32)
        for i in range(1, len(arr)):
            if np.linalg.norm(arr[i] - arr[i - 1]) <= jump:
                cv2.line(frame, tuple(arr[i - 1]), tuple(arr[i]), (0, 165, 255), 2)
        cv2.circle(frame, tuple(arr[-1]), 4, (0, 90, 255), -1)      # current wrist


def _draw_po_debug(frame, p, dbg):
    """Overlay raw PO numbers per arm so thresholds can be tuned from real footage.

    Fingertip mode:  d=wrist-hip dist  near=wrist at pocket  hand:Y/N(dist)=visible hand
    Geometry mode :  d=dist  a=elbow angle  t=tuck  h=hang  c=cues
    """
    # where MediaPipe is looking for the hand (the wrist zoom-crops)
    for bx in dbg.get("crops", []):
        cv2.rectangle(frame, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), BLUE, 1)

    x1, y1, x2, y2 = p.bbox.astype(int)
    y = min(y2 + 14, frame.shape[0] - 4)
    for side in ("l", "r"):
        s = dbg["sides"].get(side)
        if s is None:
            txt, col = f"{side.upper()}: n/a", (160, 160, 160)
        elif s.get("mode") == "master":
            mv = "-" if s.get("mv") is None else f"{s['mv']:.3f}"
            st = "S" if s.get("still") else "M"   # Still / Moving
            if s.get("state"):
                # sustained evidence (rules.PO.evidence_sustain_sec): the arm's
                # state, the POCKET seconds held, why / P(pocket), the geometry
                g = s.get("geo") or {}
                geo = (f" dy{g['dy']:+.2f} a{g.get('angle', 0):.0f}" if "dy" in g else "")
                txt = (f"{side.upper()} {s['state']} {s.get('acc', 0):.1f}s "
                       f"{s['label']}:{s['conf']:.2f}{geo}")
                col = {"CONFIRMED": (0, 0, 255), "CANDIDATE": (0, 165, 255),
                       "VISIBLE_OUT": (0, 255, 0)}.get(s["state"], (200, 200, 200))
            else:
                txt = (f"{side.upper()} d{s['dist']:.2f} n{int(s['near'])} "
                       f"{s['label']}:{s['conf']:.2f} mv{mv}{st} {'PO' if s['po'] else '-'}")
                col = (0, 0, 255) if s["po"] else (0, 255, 255)
        elif s.get("mode") == "ft":
            hd = "-" if s["hand_d"] is None else f"{s['hand_d']:.2f}"
            mv = "-" if s.get("mv") is None else f"{s['mv']:.3f}"
            st = "S" if s.get("still") else "M"
            txt = (f"{side.upper()} d{s['dist']:.2f} n{int(s['near'])} c{s.get('cues', '?')} "
                   f"hand:{'Y' if s['visible'] else 'N'}({hd}) mv{mv}{st} "
                   f"{'PO' if s['po'] else '-'}")
            col = (0, 0, 255) if s["po"] else (0, 255, 255)
        else:  # geometry mode
            txt = (f"{side.upper()} d{s['dist']:.2f} a{s['angle']:.0f} "
                   f"t{s['tuck']:+.2f} h{s['hang']:+.2f} c{s['cues']} "
                   f"{'PO' if s['po'] else '-'}")
            col = (0, 0, 255) if s["po"] else (0, 255, 255)
        cv2.putText(frame, txt, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3)
        cv2.putText(frame, txt, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
        y += 15


def _draw_pose(frame, kpts, min_conf=0.3):
    for a, b in SKELETON:
        if kpts[a, 2] > min_conf and kpts[b, 2] > min_conf:
            cv2.line(frame, tuple(kpts[a, :2].astype(int)),
                     tuple(kpts[b, :2].astype(int)), YELLOW, 2)
    for i in range(len(kpts)):
        if kpts[i, 2] > min_conf:
            cv2.circle(frame, tuple(kpts[i, :2].astype(int)), 3, GREEN, -1)
