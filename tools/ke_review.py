"""KE review agent (10 Oct 2026, the user: "for KE use object detection - paper, a tablet
or something phone-sized is taken as a phone; detect the object and take a snapshot at
KE time, I will tell OK / NG; do NOT change how the project runs").

A separate, read-only process: it never touches the monitor or its rules. For every
person the monitor logged with a KE alert (persons/<date>/persons.csv, alerts column)
it takes that person's clean clip (cut by tools.person_clips from the raw recording;
falls back to the snapshot when there is no clip), runs the general object model
(yolo11m, all COCO classes) and the pose model on its frames, and keeps the frame
where an object is most clearly in a hand (centre within 0.6 torso of a wrist).
Output, for the user to label:
  D:/poketenashi-data/ke_review/<YYYYMMDD>/<HHMMSS>_<cam>_ID<tid>.jpg  (person crop,
      object boxes + names drawn, wrists marked)
  D:/poketenashi-data/ke_review/<YYYYMMDD>/review.csv  (time, camera, track, objects
      in hand, top object, snapshot, clip, OK_NG to fill in)
COCO has no "paper" / "tablet": paper shows as book (or nothing), a tablet as laptop /
cell phone / book. Below-normal priority (run by tools.person_agent).
"""
import csv
import datetime as dt
import glob
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PERSONS = "D:/poketenashi-data/persons"
OUT = "D:/poketenashi-data/ke_review"
HAND_CLASSES = {24, 26, 39, 41, 43, 63, 64, 65, 66, 67, 73, 74, 76}   # bag, bottle, cup, laptop, phone, book, ...
WAIT_CLIP_MIN = 30


def frames(path, every_sec=0.3, max_n=40):
    if path.lower().endswith(".jpg"):
        im = cv2.imread(path)
        return [im] if im is not None else []
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    step, out, i = max(1, int(round(fps * every_sec))), [], -1
    while len(out) < max_n:
        if not cap.grab():
            break
        i += 1
        if i % step == 0:
            ok, f = cap.retrieve()
            if ok:
                out.append(f)
    cap.release()
    return out


def judge(imgs, det, pose):
    """-> (best score, image with drawings, [(name, conf)] in hand)"""
    best = (-1.0, None, [])
    for f in imgs:
        r = det.predict(f, imgsz=960, conf=0.20, classes=sorted(HAND_CLASSES), verbose=False, device="cpu")[0]
        objs = [(b, int(c), float(s)) for b, c, s in zip(r.boxes.xyxy.numpy(), r.boxes.cls.numpy(), r.boxes.conf.numpy())]
        rp = pose.predict(f, imgsz=640, conf=0.3, verbose=False, device="cpu")[0]
        if rp.keypoints is None or len(rp.boxes) == 0:
            continue
        for pb, kp in zip(rp.boxes.xyxy.numpy(), rp.keypoints.data.numpy()):
            if min(kp[[5, 6, 11, 12], 2]) < 0.3:
                continue
            torso = float(np.linalg.norm(kp[[5, 6], :2].mean(0) - kp[[11, 12], :2].mean(0)))
            wr = [kp[i, :2] for i in (9, 10) if kp[i, 2] >= 0.3]
            held = []
            for b, c, s in objs:
                cxy = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
                if wr and min(np.linalg.norm(cxy - w) for w in wr) <= 0.6 * torso:
                    held.append((b, c, s))
            score = max((s for _, _, s in held), default=0.0) + 0.001 * (pb[3] - pb[1])
            if score > best[0]:
                img = f.copy()
                for b, c, s in held:
                    x1, y1, x2, y2 = map(int, b)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, f"{det.names[c]} {s:.2f}", (x1, max(12, y1 - 4)), 0, 0.6, (0, 0, 255), 2)
                for w in wr:
                    cv2.circle(img, (int(w[0]), int(w[1])), 6, (0, 255, 255), 2)
                H, W = img.shape[:2]
                x1, y1, x2, y2 = pb
                pw, ph = 0.4 * (x2 - x1), 0.1 * (y2 - y1)
                crop = img[max(0, int(y1 - ph)):min(H, int(y2 + ph)), max(0, int(x1 - pw)):min(W, int(x2 + pw))]
                best = (score, crop, [(det.names[c], round(s, 2)) for _, c, s in sorted(held, key=lambda x: -x[2])])
    return best


def main():
    from ultralytics import YOLO
    det, pose = YOLO(os.path.join(ROOT, "yolo11m.pt")), YOLO(os.path.join(ROOT, "yolo11n-pose.pt"))
    done = set()
    while True:
        try:
            for day_dir in sorted(glob.glob(os.path.join(PERSONS, "2*")))[-2:]:
                day = os.path.basename(day_dir)
                odir = os.path.join(OUT, day)
                os.makedirs(odir, exist_ok=True)
                rcsv = os.path.join(odir, "review.csv")
                if os.path.exists(rcsv) and day not in done:
                    with open(rcsv, encoding="utf-8") as f:
                        done |= {r["snapshot"] for r in csv.DictReader(f)}
                done.add(day)
                pc, cc = os.path.join(day_dir, "persons.csv"), os.path.join(day_dir, "clips.csv")
                if not os.path.exists(pc):
                    continue
                clips = {}
                if os.path.exists(cc):
                    with open(cc, encoding="utf-8") as f:
                        clips = {r[0]: r[1] for r in csv.reader(f) if len(r) > 1 and r[1]}
                with open(pc, encoding="utf-8") as f:
                    rows = [r for r in csv.DictReader(f) if "KE" in (r.get("alerts") or "")]
                for r in rows:
                    snap = r["snapshot"]
                    if snap in done:
                        continue
                    clip = clips.get(snap)
                    age_min = (time.time() - time.mktime(time.strptime(r["end"], "%Y-%m-%d %H:%M:%S"))) / 60
                    if not clip and age_min < WAIT_CLIP_MIN:
                        continue                               # the clip is not cut yet
                    src = os.path.join(day_dir, clip if clip else snap)
                    score, img, held = judge(frames(src), det, pose)
                    name = os.path.splitext(os.path.basename(snap))[0] + ".jpg"
                    if img is not None:
                        cv2.imwrite(os.path.join(odir, name), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    new = not os.path.exists(rcsv)
                    with open(rcsv, "a", newline="", encoding="utf-8") as f:
                        w = csv.writer(f)
                        if new:
                            w.writerow(["time", "camera", "track", "objects_in_hand", "top_object", "snapshot", "clip", "review_image", "OK_NG (fill in)"])
                        w.writerow([r["start"], r["camera"], r["track"], "; ".join(f"{n} {s}" for n, s in held),
                                    held[0][0] if held else "none", snap, clip or "", name if img is not None else "", ""])
                    done.add(snap)
        except Exception as e:
            print(f"{dt.datetime.now():%H:%M:%S} {type(e).__name__}: {e}", flush=True)
        if "--once" in sys.argv:
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
