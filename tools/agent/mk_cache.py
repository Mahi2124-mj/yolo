"""Detection cache (live detector: pose + ByteTrack, fresh tracker per video),
frames scaled to the live 1920 width.
  mk_cache.py video <video> <out.pkl> <live_fps>      one raw recording, judged at ~live_fps
  mk_cache.py clips <dir> <alerts.jsonl> <out.pkl>    every alert clip (all frames)
Rows: (j, t, [(tid, bbox, conf, kp, tracked)])."""
import lowprio  # noqa
import json, os, pickle, sys, time
import cv2, yaml

ROOT = r"C:\Users\DX-ADMIN\Pictures\yolo"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from poketenashi.detector import Detector  # noqa: E402

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
if os.environ.get("IMGSZ"):
    cfg["models"]["low_mem_imgsz"] = int(os.environ["IMGSZ"])
det = Detector(cfg)


def reset():
    try:
        det.pose.predictor.trackers[0].reset()
    except Exception:
        pass


def run(path, step):
    reset()
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 8.0
    rows, j = [], -1
    while True:
        ok, f = cap.read()
        if not ok:
            break
        j += 1
        if j % step:
            continue
        if f.shape[1] != 1920:
            f = cv2.resize(f, (1920, int(round(f.shape[0] * 1920 / f.shape[1]))), interpolation=cv2.INTER_LINEAR)
        persons, _ = det(f)
        rows.append((j, j / fps, [(p.track_id, p.bbox.tolist(), p.conf, p.keypoints.tolist(), p.tracked)
                                  for p in persons]))
    cap.release()
    return {"video": path, "fps": fps, "step": step, "rows": rows}


t0 = time.time()
if sys.argv[1] == "video":
    path, out, live = sys.argv[2], sys.argv[3], float(sys.argv[4])
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    step = max(1, int(round(fps / live)))
    pickle.dump(run(path, step), open(out, "wb"))
else:
    d, alerts, out = sys.argv[2], sys.argv[3], sys.argv[4]
    res = pickle.load(open(out, "rb")) if os.path.exists(out) else {}
    al = [json.loads(l) for l in open(alerts, encoding="utf-8") if l.strip()]
    for k, a in enumerate(al):
        c = a.get("clip")
        if not c or c in res or not os.path.exists(os.path.join(d, c)):
            continue
        res[c] = dict(run(os.path.join(d, c), 1), alert=a)
        if k % 5 == 0:
            pickle.dump(res, open(out, "wb"))
            print(f"{k + 1}/{len(al)} {time.time() - t0:.0f}s", flush=True)
    pickle.dump(res, open(out, "wb"))
print("done", out, f"{time.time() - t0:.0f}s", flush=True)
