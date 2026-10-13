"""Re-check every evidence alert with TODAY's rules (8 Oct): per camera its merged
rules / zones / detector size, the Material Gate dress code and truck exemption
(no face ID: nobody counts as identified). An alert is KEPT if the current rule
fires on a track overlapping the clip's red alert box; else it is FALSE now.
Rules the camera no longer checks are skipped (reported, not judged).
Writes recheck.jsonl (one row per alert) - resumable."""
import lowprio  # noqa
import contextlib, copy, io, json, os, sys, time as _time
import cv2, numpy as np, yaml
ROOT = r"C:\Users\DX-ADMIN\Pictures\yolo"
SP = os.environ.get("AGENT_WORK", "D:/poketenashi-data/agent/work")
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from poketenashi import rules as R
from poketenashi.detector import Detector, Person
from poketenashi.zones import Zones
from poketenashi.dress import DressCode
from poketenashi.vehicles import TruckNear

OUT = os.path.join(SP, "recheck.jsonl")
done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["key"] for l in open(OUT)}
cfg0 = yaml.safe_load(open("config.yaml", encoding="utf-8"))
CAMS = {c["id"]: c for c in json.load(open("monitor_cameras.json", encoding="utf-8"))["cameras"]}


def merge(a, b):
    for k, v in b.items():
        a[k] = merge(a.get(k) or {}, v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return a


clock = [1000.0]
_real_time = _time.time
_time.monotonic = lambda: clock[0]
_time.time = lambda: 1.79e9 + clock[0]
dets = {}


def detector(cam):
    m = merge(copy.deepcopy(cfg0["models"]), copy.deepcopy(cam.get("models") or {}))
    k = m.get("low_mem_imgsz")
    if k not in dets:
        c = copy.deepcopy(cfg0)
        c["models"] = m
        dets[k] = Detector(c)
    return dets[k]


def red_box(f):
    h = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(h, (0, 150, 120), (6, 255, 255)) | cv2.inRange(h, (174, 150, 120), (180, 255, 255))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        x, y, w, hh = cv2.boundingRect(c)
        if w > 25 and hh > 50:
            out.append((x, y, x + w, y + hh))
    return out


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


rows = [json.loads(l) for l in open("violations.jsonl", encoding="utf-8") if l.strip()]
t0 = _real_time()
n = 0
with open(OUT, "a") as fo:
    for r in rows:
        key = f"{r['time']}|{r.get('camera', 'cam1')}|{r['track_id']}|{r['rule']}"
        if key in done:
            continue
        cid = r.get("camera") or "cam1"
        cam = CAMS.get(cid) or {}
        rc = merge(copy.deepcopy(cfg0["rules"]), copy.deepcopy(cam.get("rules") or {}))
        on = rc.get("enabled") or []
        rec = {"key": key, "time": r["time"], "camera": cid, "rule": r["rule"], "clip": r.get("clip")}
        p = os.path.join("D:/poketenashi-data/clips", r.get("clip") or "x")
        if r["rule"] not in on:
            rec["verdict"] = "not_checked_now"
        elif not os.path.exists(p):
            rec["verdict"] = "no_clip"
        else:
          try:
              rc["enabled"] = [r["rule"]]
              cfg = copy.deepcopy(cfg0)
              cfg["rules"], cfg["zones"] = rc, cam.get("zones") or {}
              cfg["samples"] = {"enabled": False}
              clock[0] += 1000
              with contextlib.redirect_stdout(io.StringIO()):
                  eng = R.RuleEngine(cfg, Zones(cfg["zones"]))
              dress = DressCode(cam["dress_code"]) if cam.get("dress_code") else None
              trucks = TruckNear(dict(cam["truck_exempt"], every_sec=0), ROOT) if cam.get("truck_exempt") else None
              det = detector(cam)
              try:
                  det.pose.predictor.trackers[0].reset()
              except Exception:
                  pass
              cap = cv2.VideoCapture(p)
              fps = cap.get(cv2.CAP_PROP_FPS) or 5
              S = 1920.0 / (cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
              j, fired, reds = -1, False, []
              base = clock[0]
              while True:
                  ok, f = cap.read()
                  if not ok:
                      break
                  j += 1
                  clock[0] = base + j / fps
                  reds += [[v * S for v in b] for b in red_box(f)]
                  f2 = cv2.resize(f, (1920, int(round(f.shape[0] * S))))
                  persons, phones = det(f2)
                  judged = dress.judged(f2, persons) if dress else persons
                  if trucks:
                      near = trucks.update(f2, persons)
                      judged = [q for q in judged if q.track_id not in near]
                  with contextlib.redirect_stdout(io.StringIO()):
                      res = eng.check(judged, phones, f2)
                  for v in res:
                      if v["rule"] == r["rule"] and any(iou(v["bbox"], b) >= 0.3 for b in reds[-60:]):
                          fired = True
              cap.release()
              rec["verdict"] = "kept" if fired else "false_now"
          except Exception as e:
            rec["verdict"] = "error"
            rec["err"] = str(e)[:120]
        fo.write(json.dumps(rec) + "\n")
        fo.flush()
        n += 1
        if n % 20 == 0:
            print(n, f"{_real_time() - t0:.0f}s", flush=True)
print("done", n)
