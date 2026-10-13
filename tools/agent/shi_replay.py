"""Replay a Material Gate clip through the live cam3 rules (SHI + NA) on the video
clock; print per track the pad visits (still time, head swing, ok, direction)
and SHI / NA results. Usage: shi_replay.py <cache.pkl> [rules overrides json]"""
import contextlib, copy, io, json, os, pickle, sys
from collections import defaultdict
import numpy as np, yaml

ROOT = r"C:\Users\DX-ADMIN\Pictures\yolo"
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from poketenashi import rules as R                      # noqa: E402
from poketenashi.detector import Person                 # noqa: E402
from poketenashi.zones import Zones                     # noqa: E402

cache = pickle.load(open(sys.argv[1], "rb"))
over = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
cam = next(c for c in json.load(open("monitor_cameras.json", encoding="utf-8"))["cameras"] if c["id"] == "cam3")


def merge(a, b):
    for k, v in b.items():
        a[k] = merge(a.get(k) or {}, v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return a


rc = merge(copy.deepcopy(cfg["rules"]), copy.deepcopy(cam.get("rules") or {}))
rc = merge(rc, over)
rc["enabled"] = ["SHI", "NA"]
cfg["rules"] = rc
cfg["zones"] = cam["zones"]
clock = [0.0]
R.time.monotonic = lambda: clock[0]
R.time.time = lambda: 1.79e9 + clock[0]
with contextlib.redirect_stdout(io.StringIO()):
    eng = R.RuleEngine(cfg, Zones(cam["zones"]))
z = Zones(cam["zones"])
log = defaultdict(list)
fired = defaultdict(set)
for j, t, plist in cache["rows"]:
    clock[0] = 100.0 + t
    persons = [Person(tid, bb, cf, kp, tracked=tr) for tid, bb, cf, kp, tr in plist if tr]
    with contextlib.redirect_stdout(io.StringIO()):
        res = eng.check(persons, [], None)
    for v in res:
        fired[(v["track_id"], v["rule"])].add(round(t, 1))
    for p in persons:
        st = eng.shi_state[p.track_id]
        f = p.feet()
        torso = p.torso_len() or 0
        spd = eng.speed_ps(p.track_id) / torso if torso > 1 else None
        log[p.track_id].append((round(t, 1), z.part_of("checkpoint", f), None if f is None else [int(f[0]), int(f[1])],
                                None if spd is None else round(spd, 2), round(eng.head_swing(p.track_id), 2),
                                round(st.get("still", 0), 1), st.get("ok")))
for tid, rows in sorted(log.items()):
    pads = [r for r in rows if r[1]]
    if not pads and not any(k[0] == tid for k in fired):
        continue
    print(f"ID{tid}: seen {rows[0][0]}-{rows[-1][0]}s  start {rows[0][2]} end {rows[-1][2]}")
    cur = None
    for r in rows:
        if r[1] != cur:
            print(f"   {r[0]:5.1f}s {'on ' + r[1] if r[1] else 'off pad'} at {r[2]}")
            cur = r[1]
        if r[1]:
            print(f"        {r[0]:5.1f}s spd {r[3]} head {r[4]} still {r[5]} ok {r[6]}")
    for k, ts in fired.items():
        if k[0] == tid:
            print(f"   >>> {k[1]} at {min(ts)}-{max(ts)}s")
