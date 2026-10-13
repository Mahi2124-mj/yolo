"""YOLO-World check of every KE alert that has a clean clip (ke_review rows), next to the
COCO result already in review.csv - read-only, for choosing the KE object model (12 Oct).
Output: D:/poketenashi-data/ke_review/<day>/world.csv + world_<snapshot>.jpg"""
import csv, glob, os, sys
import cv2, numpy as np
ROOT = r"C:\Users\DX-ADMIN\Pictures\yolo"; sys.path.insert(0, ROOT); os.chdir(ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lowprio  # noqa
from ultralytics import YOLO
from tools.ke_review import frames

def held(det, pose, imgs):
    """best frame: objects within 0.6 torso of a wrist of any person -> (list, image)"""
    best = ([], None, -1)
    for f in imgs:
        rp = pose.predict(f, imgsz=640, conf=0.3, verbose=False, device="cpu")[0]
        if rp.keypoints is None or len(rp.boxes) == 0:
            continue
        for pb, kp in zip(rp.boxes.xyxy.numpy(), rp.keypoints.data.numpy()):
            x1, y1, x2, y2 = pb; H, W = f.shape[:2]
            pw, ph = 0.4 * (x2 - x1), 0.1 * (y2 - y1); cx, cy = max(0, int(x1 - pw)), max(0, int(y1 - ph))
            crop = f[cy:min(H, int(y2 + ph)), cx:min(W, int(x2 + pw))]
            if crop.size == 0 or (y2 - y1) < 120:
                continue
            t = float(np.linalg.norm(kp[[5, 6], :2].mean(0) - kp[[11, 12], :2].mean(0))) if min(kp[[5, 6, 11, 12], 2]) >= .3 else .3 * (y2 - y1)
            wr = [kp[i, :2] - [cx, cy] for i in (9, 10) if kp[i, 2] >= .3]
            if not wr:
                continue
            r = det.predict(crop, imgsz=640, conf=0.10, verbose=False, device="cpu")[0]
            objs = []
            for bb, c, s in zip(r.boxes.xyxy.numpy(), r.boxes.cls.numpy(), r.boxes.conf.numpy()):
                pc = np.array([(bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2])
                if min(np.linalg.norm(pc - w) for w in wr) <= 0.6 * t:
                    objs.append((det.names[int(c)], round(float(s), 2), bb))
            if objs and max(o[1] for o in objs) > best[2]:
                img = crop.copy()
                for n, s, bb in objs:
                    cv2.rectangle(img, tuple(bb[:2].astype(int)), tuple(bb[2:].astype(int)), (0, 0, 255), 2)
                    cv2.putText(img, f"{n} {s}", (int(bb[0]), max(12, int(bb[1]) - 4)), 0, 0.6, (0, 0, 255), 2)
                best = (sorted([(n, s) for n, s, _ in objs], key=lambda x: -x[1]), img, max(o[1] for o in objs))
    return best[0], best[1]

world = YOLO(os.path.join(ROOT, "yolov8s-world-ke.pt")); pose = YOLO(os.path.join(ROOT, "yolo11n-pose.pt"))
for rc in sorted(glob.glob("D:/poketenashi-data/ke_review/2*/review.csv")):
    d = os.path.dirname(rc); day = os.path.basename(d); out = []
    for r in csv.DictReader(open(rc, encoding="utf-8")):
        if not r.get("clip"):
            continue
        objs, img = held(world, pose, frames(os.path.join("D:/poketenashi-data/persons", day, r["clip"]), every_sec=0.25, max_n=50))
        name = ""
        if img is not None:
            name = "world_" + os.path.basename(r["snapshot"]).replace("\\\\", "_")
            cv2.imwrite(os.path.join(d, name), img)
        out.append([r["time"], r["camera"], r["track"], r.get("OK_NG (fill in)", ""), r.get("objects_in_hand", ""),
                    "; ".join(f"{n} {s}" for n, s in objs), objs[0][0] if objs else "none", name])
        print(day, r["track"], r.get("OK_NG (fill in)", ""), "| coco:", r.get("objects_in_hand", ""), "| world:", "; ".join(f"{n} {s}" for n, s in objs[:4]), flush=True)
    with open(os.path.join(d, "world.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["time", "camera", "track", "label", "coco_in_hand", "world_in_hand", "world_top", "world_image"]); w.writerows(out)
