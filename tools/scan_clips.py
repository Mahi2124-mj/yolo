"""Pull the violation frame out of evidence clips.

Written after five failed attempts at this in one session. Each earlier detector
counted something that was NOT a drawn box:
  * the fire-exit sign and the red/white barrier rope in the room,
  * JPEG compression noise (1500 scattered pixels on a busy frame),
  * and the app's own red badge in the top-left corner, which is in every frame
    and made clips report "42 of 42 frames have a violation".

So this one does not count red pixels. It looks for a red RECTANGLE OUTLINE:
person-shaped, and thin - a drawn box of w x h with a 2-3 px stroke has about
2*(w+h)*t red pixels, while a solid red object fills its whole bounding box.

    python -m tools.scan_clips                 # newest clips from both instances
    python -m tools.scan_clips --since 15:30   # only clips written after a time
"""
import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np

BADGE = (90, 90)          # the app's own red mark, top-left of every frame


def red_boxes(frame):
    """Every red rectangle OUTLINE in the frame: [(x, y, w, h, stroke), ...]."""
    b, g, r = (frame[:, :, 0].astype(int), frame[:, :, 1].astype(int),
               frame[:, :, 2].astype(int))
    m = (((r > 150) & (g < 90) & (b < 90)) * 255).astype(np.uint8)
    m[0:BADGE[1], 0:BADGE[0]] = 0
    out = []
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < 30 or h < 60 or h < w:            # people are taller than wide
            continue
        filled = int(m[y:y + h, x:x + w].sum() / 255)
        stroke = filled / max(1.0, 2.0 * (w + h))
        if 1.0 <= stroke <= 6.0:                 # an outline, not a solid object
            out.append((x, y, w, h, round(stroke, 1)))
    return out


def scan(path):
    cap = cv2.VideoCapture(path)
    frames, hits = [], []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
        hits.append(red_boxes(f))
    cap.release()
    idx = [i for i, h in enumerate(hits) if h]
    return frames, hits, idx


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="*", default=["D:/poketenashi-data/clips"])
    ap.add_argument("--limit", type=int, default=8, help="newest N clips per dir")
    ap.add_argument("--since", default=None, help="only clips written after HH:MM")
    ap.add_argument("--out", default=None, help="write a contact sheet here")
    args = ap.parse_args()

    cut = 0.0
    if args.since:
        h, _, mnt = args.since.partition(":")
        t = time.localtime()
        cut = time.mktime((t.tm_year, t.tm_mon, t.tm_mday, int(h), int(mnt or 0),
                           0, 0, 0, -1))

    tiles = []
    for d in args.dirs:
        files = [p for p in glob.glob(os.path.join(d, "*.mp4"))
                 if os.path.getmtime(p) >= cut]
        files = sorted(files, key=os.path.getmtime)[-args.limit:]
        print(f"\n=== {d}  ({len(files)} clips) ===")
        if not files:
            print("  (none)")
            continue
        print("  clip                                 frames  violation frames")
        for p in files:
            frames, hits, idx = scan(p)
            name = os.path.basename(p)
            note = "" if idx else "   <-- NO VIOLATION IN CLIP"
            print(f"  {name[:36]:<36} {len(frames):>6} {len(idx):>16}{note}")
            if idx:
                k = idx[len(idx) // 2]
                f = frames[k].copy()
                for (x, y, w, h, s) in hits[k]:
                    cv2.rectangle(f, (x - 4, y - 4), (x + w + 4, y + h + 4),
                                  (0, 255, 255), 2)
                cv2.putText(f, f"{name[:30]}  frame {k}/{len(frames)}", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                tiles.append(cv2.resize(f, (640, 360)))

    if args.out and tiles:
        cols = 2
        rows = (len(tiles) + cols - 1) // cols
        while len(tiles) < rows * cols:
            tiles.append(np.zeros((360, 640, 3), np.uint8))
        sheet = np.vstack([np.hstack(tiles[i * cols:(i + 1) * cols])
                           for i in range(rows)])
        cv2.imwrite(args.out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"\ncontact sheet -> {args.out}")


if __name__ == "__main__":
    main()
