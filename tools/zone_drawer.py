"""Interactively draw a Poketenashi zone on a still frame and save it to config.yaml.

The TE / NA / SHI rules are location-dependent, so each site must mark its own
zones (stairs, handrail, walkway_center, checkpoint).

Usage:
    python -m tools.zone_drawer --zone walkway_center
    python -m tools.zone_drawer --zone stairs --source clip.mp4

Keys:  left-click = add point | u = undo | c = clear | s = save & quit | q = quit
"""
import sys
import time
import argparse
import io

import numpy as np
import cv2
import yaml

ZONE_CHOICES = ["stairs", "handrail", "walkway_center", "gangway",
                "checkpoint", "restricted"]



def _save_zone(path, zone, pts):
    """Rewrite ONE line of the config: the zone's points.

    Emphatically not yaml.safe_dump(cfg) - that round-trips the whole document
    through the parser and drops every comment in it. config.yaml is more comment
    than value (each threshold records why it is what it is), so a full dump
    silently destroys the reason for every number in the file. Editing the single
    line leaves the rest byte-for-byte untouched.
    """
    raw = io.open(path, encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in raw else "\n"
    lines = [l.rstrip("\r") for l in raw.split(nl)]
    flow = "  %s: %s" % (zone, [[int(x), int(y)] for x, y in pts])

    start = None
    for i, l in enumerate(lines):
        if l.strip().startswith(zone + ":") and l.startswith("  "):
            start = i
            break
    if start is None:                       # zone not listed yet - add it under zones:
        z = next(i for i, l in enumerate(lines) if l.startswith("zones:"))
        lines.insert(z + 1, flow)
    else:
        # Absorb any block-list items ("  - - 123") that belonged to the old value.
        end = start + 1
        while end < len(lines) and lines[end].lstrip().startswith("- "):
            end += 1
        comment = ""
        if "#" in lines[start]:
            comment = "   " + lines[start][lines[start].index("#"):]
        lines[start:end] = [flow + comment]
    io.open(path, "w", encoding="utf-8", newline="").write(nl.join(lines))


def grab_frame(source):
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    elif isinstance(source, str) and source.lower().startswith("rtsp"):
        # force RTSP over TCP (reliable frame grab for IP cams)
        import os
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(source)
    frame = None
    for _ in range(120):
        ok, fr = cap.read()
        if ok:
            frame = fr
            break
        time.sleep(0.1)
    cap.release()
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--zone", required=True, choices=ZONE_CHOICES)
    ap.add_argument("--source", default=None, help="override the config camera source")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    src = args.source or cfg["camera"]["source"]
    print(f"[zone_drawer] grabbing a frame from: {src}")
    frame = grab_frame(src)
    if frame is None:
        print("[zone_drawer] could not grab a frame. Check the source / network.")
        sys.exit(1)

    pts = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append([x, y])

    win = f"Draw '{args.zone}'  [click=add  u=undo  c=clear  s=save  q=quit]"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        disp = frame.copy()
        for p in pts:
            cv2.circle(disp, tuple(p), 4, (0, 255, 0), -1)
        if len(pts) >= 2:
            cv2.polylines(disp, [np.array(pts)], len(pts) >= 3, (0, 255, 255), 2)
        cv2.imshow(win, disp)

        k = cv2.waitKey(20) & 0xFF
        if k == ord("u") and pts:
            pts.pop()
        elif k == ord("c"):
            pts.clear()
        elif k == ord("q"):
            print("[zone_drawer] quit without saving.")
            break
        elif k == ord("s"):
            if len(pts) < 3:
                print("[zone_drawer] need at least 3 points to form a polygon.")
                continue
            cfg.setdefault("zones", {})[args.zone] = pts
            _save_zone(args.config, args.zone, pts)
            print(f"[zone_drawer] saved {len(pts)} points to zone '{args.zone}' in {args.config}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
