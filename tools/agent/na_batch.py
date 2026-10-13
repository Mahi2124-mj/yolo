"""NA on Material Gate, OLD (live heading rule) vs NEW (truck_lane forbidden zone),
for many detection caches in ONE process. Usage:
  na_batch.py caches <glob of .pkl single-video caches>
  na_batch.py clips <video dir> <alerts.jsonl>   (detects each clip itself, 640, like live cam3)
Prints per clip: old NA tracks / new NA tracks (+ first time)."""
import lowprio  # noqa
import contextlib, copy, glob, io, json, os, pickle, sys
import numpy as np, yaml
ROOT = r"C:\Users\DX-ADMIN\Pictures\yolo"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from poketenashi import rules as R
from poketenashi.detector import Person
from poketenashi.zones import Zones

TRUCK = [[655, 625], [1440, 632], [1880, 930], [1880, 1080], [470, 1080], [560, 800]]
cfg0 = yaml.safe_load(open("config.yaml", encoding="utf-8"))
cam = next(c for c in json.load(open("monitor_cameras.json", encoding="utf-8"))["cameras"] if c["id"] == "cam3")
zones = dict(cam["zones"], truck_lane=TRUCK)


def merge(a, b):
    for k, v in b.items():
        a[k] = merge(a.get(k) or {}, v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return a


def engine(new):
    rc = merge(copy.deepcopy(cfg0["rules"]), copy.deepcopy(cam.get("rules") or {}))
    if new:
        rc["NA"]["forbidden_zone"] = "truck_lane"
        rc["NA"]["forbidden_sec"] = 1.0
    rc["enabled"] = ["NA"]
    cfg = copy.deepcopy(cfg0)
    cfg["rules"], cfg["zones"] = rc, zones
    with contextlib.redirect_stdout(io.StringIO()):
        return R.RuleEngine(cfg, Zones(zones))


clock = [0.0]
R.time.monotonic = lambda: clock[0]
R.time.time = lambda: 1.79e9 + clock[0]


def run(rows):
    out = {}
    for new in (False, True):
        clock[0] = 1000.0
        eng = engine(new)
        hits = {}
        for j, t, pl in rows:
            clock[0] = 1000.0 + t
            persons = [Person(tid, bb, cf, np.array(kp), tracked=tr) for tid, bb, cf, kp, tr in pl if tr]
            with contextlib.redirect_stdout(io.StringIO()):
                res = eng.check(persons, [], None)
            for v in res:
                if v["rule"] == "NA":
                    hits.setdefault(v["track_id"], round(t, 1))
        out[new] = hits
    return out


if sys.argv[1] == "caches":
    for p in sorted(glob.glob(sys.argv[2])):
        c = pickle.load(open(p, "rb"))
        o = run(c["rows"])
        print(f"{os.path.basename(p)[5:-4]} old:{len(o[False])} new:{len(o[True])} {o[True] if o[True] else ''}", flush=True)
else:
    import cv2
    from poketenashi.detector import Detector
    cfg = copy.deepcopy(cfg0)
    cfg["models"]["low_mem_imgsz"] = 640
    det = Detector(cfg)
    for l in open(sys.argv[3], encoding="utf-8"):
        a = json.loads(l)
        p = os.path.join(sys.argv[2], a["clip"])
        if not os.path.exists(p):
            continue
        try:
            det.pose.predictor.trackers[0].reset()
        except Exception:
            pass
        cap = cv2.VideoCapture(p)
        fps = cap.get(cv2.CAP_PROP_FPS) or 5
        rows, j = [], -1
        while True:
            ok, f = cap.read()
            if not ok:
                break
            j += 1
            f = cv2.resize(f, (1920, 1080))
            ps, _ = det(f)
            rows.append((j, j / fps, [(q.track_id, q.bbox.tolist(), q.conf, q.keypoints.tolist(), q.tracked) for q in ps]))
        cap.release()
        o = run(rows)
        print(f"{a['time'][11:]} {a['clip']} old:{len(o[False])} new:{len(o[True])} {o[True] if o[True] else ''}", flush=True)
