"""Capture labelled MASTER samples for the PO (hands-in-pocket) matcher.

Stand in front of the camera and DEMONSTRATE:
  * put a hand in your pocket  -> press  p   (saves the wrist crop as 'pocket')
  * take the hand out / let it hang -> press  o   (saves it as 'out')
Collect ~20-40 of each, in different positions/people/lighting. The more varied,
the better the live matching.

    python -m tools.capture_masters                 # uses config camera
    python -m tools.capture_masters --source 0      # webcam

Keys:  p = label IN-POCKET   o = label HAND-OUT   u = undo last   q = save & quit
"""
import os
import time
import argparse

import cv2
import yaml

from poketenashi.camera import VideoStream
from poketenashi.detector import Detector
from poketenashi.embedder import Embedder
from poketenashi.master_match import MasterLibrary
from poketenashi.po_crop import wrist_crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--source", default=None)
    ap.add_argument("--scale", type=float, default=None,
                    help="wrist crop size (fraction of torso); defaults to config master_scale")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    pcfg = cfg["rules"]["PO"]
    scale = args.scale if args.scale is not None else pcfg.get("master_scale", 0.6)
    src = args.source if args.source is not None else cfg["camera"]["source"]

    det = Detector(cfg)
    emb = Embedder(device=cfg.get("models", {}).get("device") or None)
    lib = MasterLibrary(pcfg.get("master_file", "masters/po_masters.npz"))
    for lbl in ("pocket", "out"):
        os.makedirs(f"masters/po/{lbl}", exist_ok=True)

    stream = VideoStream(src, cfg["camera"].get("rtsp_transport", "tcp")).start()
    win = "Capture PO masters  [p=pocket  o=out  u=undo  q=save&quit]"
    batches = []   # stack of (count, [paths]) for undo
    print(f"[capture] start. pocket={lib.count('pocket')} out={lib.count('out')}  scale={scale}")

    while True:
        ok, frame = stream.read()
        if not ok:
            time.sleep(0.03)
            continue

        persons, _ = det(frame)
        disp = frame.copy()
        crops = []  # (side, crop)
        if persons:
            subj = max(persons, key=lambda z: (z.bbox[2] - z.bbox[0]) * (z.bbox[3] - z.bbox[1]))
            for side in ("l", "r"):
                c, box = wrist_crop(frame, subj, side, scale)
                if box is not None:
                    crops.append((side, c))
                    cv2.rectangle(disp, box[:2], box[2:], (0, 255, 255), 2)
                    cv2.putText(disp, side.upper(), (box[0], box[1] - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.putText(disp, f"pocket:{lib.count('pocket')}  out:{lib.count('out')}",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(disp, "p=pocket  o=out  u=undo  q=save&quit",
                    (10, disp.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.imshow(win, disp)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("p"), ord("o")) and crops:
            label = "pocket" if k == ord("p") else "out"
            paths = []
            for side, c in crops:
                v = emb.embed(c)
                if v is None:
                    continue
                path = f"masters/po/{label}/{int(time.time()*1000)}_{side}.jpg"
                cv2.imwrite(path, c)
                lib.add(v, label)
                paths.append(path)
            if paths:
                batches.append((len(paths), paths))
                lib.save()
                print(f"+{label} x{len(paths)}  ->  pocket={lib.count('pocket')} out={lib.count('out')}")
        elif k == ord("u") and batches:
            n, paths = batches.pop()
            lib.pop(n)
            for pth in paths:
                try:
                    os.remove(pth)
                except OSError:
                    pass
            lib.save()
            print(f"undo {n}  ->  pocket={lib.count('pocket')} out={lib.count('out')}")
        elif k == ord("q"):
            lib.save()
            print(f"[capture] saved {len(lib)} masters (pocket={lib.count('pocket')} out={lib.count('out')})")
            break

    stream.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
