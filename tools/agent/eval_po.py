"""Replay a PO variant on every labelled set, on the video clock, through the
real RuleEngine (PO only) and the real AlertManager, and score it.

Usage: eval_po.py <variant.json> <out.json> [sets, default all]
variant.json: {"name", "rules": optional path to a rules.py copy, "po": {rules.PO overrides}}

Sets (detection caches, so YOLO does not run again):
  t25    25 Sep pocket tester (standing)            GT: labelled arm crops (po_arms)
  lt     28 Sep live test (walking + standing)      GT: labelled arm crops (po_arms)
         t25/lt use the arm model HELD OUT from that video (it was trained on them)
  dx     5 Oct staged test on Dx (cam2)             GT: README episodes (by eye, user-checked)
  live6  41 live PO alert clips 4-6 Oct             GT: live6_labels.json (by eye)
  raw    3 Oct 12:29-13:29 raw hour (no pockets found by eye) - every alert is a false one
  raw2   3 Oct 15:26-15:41 raw (pocket testers at 15:23-15:25 are before it) - alerts unverified
"""
import lowprio  # noqa
import contextlib, copy, importlib.util, io, json, os, pickle, sys, time
from collections import defaultdict

import cv2, numpy as np, yaml

ROOT = r"C:\Users\DX-ADMIN\Pictures\yolo"
SP = os.environ.get("AGENT_GUARD", "D:/poketenashi-data/agent/guard")
OLD = os.environ.get("AGENT_OLD", "D:/poketenashi-data/agent/old")
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from tools.replay import VideoClock, install_clock          # noqa: E402
from poketenashi import alerts as alertmod, po_model         # noqa: E402
from poketenashi import rules as rules_mod                   # noqa: E402
from poketenashi.alerts import AlertManager                  # noqa: E402
from poketenashi.detector import Person                      # noqa: E402
from poketenashi.zones import Zones                          # noqa: E402

variant = json.load(open(sys.argv[1], encoding="utf-8"))
out_path = sys.argv[2]
ALL = ["t25", "lt", "dx", "dx6", "live6", "live7", "raw", "raw2"]
which = sys.argv[3].split(",") if len(sys.argv) > 3 else ALL

RuleEngine = rules_mod.RuleEngine
if variant.get("rules"):
    _sp = importlib.util.spec_from_file_location("poketenashi.rules_alt", variant["rules"])
    _m = importlib.util.module_from_spec(_sp)
    _sp.loader.exec_module(_m)

    class _ClockFwd:                       # install_clock() patches rules.time; follow it
        def __getattr__(self, k):
            return getattr(rules_mod.time, k)
    _m.time = _ClockFwd()
    RuleEngine = _m.RuleEngine


class _TimeShim:
    def __init__(self, clock):
        self.clock = clock

    def time(self):
        return self.clock.time()

    def monotonic(self):
        return self.clock.monotonic()

    def __getattr__(self, k):
        return getattr(time, k)


BASE = yaml.safe_load(open("config.yaml", encoding="utf-8"))
BASE["rules"]["enabled"] = ["PO"]
BASE["samples"] = {"enabled": False}
BASE["clips"]["enabled"] = False
BASE["alerts"]["log_file"] = None
BASE["alerts"]["save_snapshots"] = False
BASE["output"]["debug"] = True
CAMS = {c["id"]: c for c in json.load(open("monitor_cameras.json", encoding="utf-8"))["cameras"]}
HELD = {"t25": os.path.join(OLD, "arm_no_t25.pt"), "lt": os.path.join(OLD, "arm_no_lt.pt")}


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def replay(cache, video, zones, model=None):
    """-> alerts [(t, tid, bbox)], shown {tid: [t...]}, tracks {tid: [(t, bbox)]}, dbg rows."""
    cfg = copy.deepcopy(BASE)
    cfg["zones"] = zones
    cfg["rules"]["PO"].update(variant.get("po", {}))
    if model:
        cfg["rules"]["PO"]["model_path"] = model
    po_model._LOADED.clear()
    clock = VideoClock()
    install_clock(clock)
    alertmod.time = _TimeShim(clock)
    with contextlib.redirect_stdout(io.StringIO()):
        eng = RuleEngine(cfg, Zones(zones))
        am = AlertManager(cfg)
    cap = cv2.VideoCapture(video)
    jc, f, t0 = -1, None, clock.t
    alerts, shown, tracks, dbg = [], defaultdict(list), defaultdict(list), []
    for j, t, plist in cache["rows"]:
        while jc < j:
            ok, f = cap.read()
            jc += 1
        if f is not None and f.shape[1] != 1920:
            f = cv2.resize(f, (1920, int(round(f.shape[0] * 1920 / f.shape[1]))), interpolation=cv2.INTER_LINEAR)
        clock.t = t0 + t
        persons = [Person(tid, bb, cf, kp, tracked=tr) for tid, bb, cf, kp, tr in plist]
        with contextlib.redirect_stdout(io.StringIO()):
            res = eng.check(persons, [], f)
        po = [v for v in res if v["rule"] == "PO"]
        for p in persons:
            if p.tracked:
                tracks[p.track_id].append((t, [float(x) for x in p.bbox]))
        for v in po:
            shown[v["track_id"]].append(t)
        for tid, d in eng.po_dbg.items():
            if any(p.track_id == tid for p in persons):
                bb = next(p.bbox for p in persons if p.track_id == tid)
                dbg.append({"t": round(t, 2), "tid": tid, "po": tid in shown and shown[tid][-1] == t,
                            "bbox": [round(float(x)) for x in bb],
                            "sides": {s: {k: (d["sides"].get(s) or {}).get(k)            # geometry: None = arm not seen
                                          for k in ("label", "conf", "po", "state", "geo", "angle", "cues", "still",
                                                    "visible", "elbow_out")}
                                      for s in ("l", "r")}})
        eng.po_dbg.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            fresh = am.handle(po, None, present={p.track_id for p in persons if p.tracked})
        for v in fresh:
            alerts.append((round(t, 2), v["track_id"], [float(x) for x in v["bbox"]]))
    cap.release()
    return alerts, shown, tracks, dbg


# ---- ground truth: labelled arms (t25 / lt) -------------------------------------------
IDX = [json.loads(l) for l in open(r"D:\poketenashi-data\po_arms\index.jsonl", encoding="utf-8")]
LAB = json.load(open(r"D:\poketenashi-data\po_arms\labels.json", encoding="utf-8"))


def arm_episodes(src, tracks):
    by = defaultdict(list)
    for r in IDX:
        if r["src"] == src and LAB.get(r["id"]) == "pocket":
            by[r["tid"]].append(r)
    eps = []
    for tid, rs in by.items():
        rs.sort(key=lambda r: r["t"])
        cur = [rs[0]]
        for r in rs[1:]:
            if r["t"] - cur[-1]["t"] > 8.0:
                eps.append(cur)
                cur = []
            cur.append(r)
        eps.append(cur)
    out = []
    for e in eps:
        tid, ta, tb = e[0]["tid"], e[0]["t"], e[-1]["t"]
        frames = defaultdict(set)
        for r in e:
            frames[r["frame"]].add(r["side"])
        both = any(len(s) == 2 for s in frames.values())
        # walking: the person's box centre moves >= 0.6 torso a second over the episode
        tr = [(t, b) for t, b in tracks.get(tid, []) if ta - 0.5 <= t <= tb + 0.5]
        torso = float(np.median([r["torso"] for r in e]))
        walk = None
        if len(tr) >= 2 and tr[-1][0] > tr[0][0]:
            c = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for _, b in tr]
            path = sum(np.hypot(c[i + 1][0] - c[i][0], c[i + 1][1] - c[i][1]) for i in range(len(c) - 1))
            walk = path / (tr[-1][0] - tr[0][0]) / torso >= 0.6
        # the labels' track ids come from another run of the tracker: map by box
        tids = {k for k, trk in tracks.items() if any(abs(t2 - r["t"]) <= 0.05 and iou(b2, r["bbox"]) >= 0.4
                                                     for r in e for t2, b2 in trk if ta - 0.1 <= t2 <= tb + 0.1)}
        out.append({"name": f"{src}-{tid}@{ta:.0f}", "tid": tid, "tids": sorted(tids), "t0": ta, "t1": tb, "n": len(e),
                    "both": both, "walking": walk, "sustained": len(e) >= 3,
                    "boxes": [(r["t"], r["bbox"]) for r in e]})
    out.sort(key=lambda e: e["t0"])
    return out


def score_arms(src, alerts, tracks):
    eps = arm_episodes(src, tracks)
    for e in eps:
        e["hit"] = None
    fps = []
    for t, tid, bb in alerts:
        hit = None
        for e in eps:
            # the person of the episode: the track(s) mapped onto its labelled
            # boxes, or a box overlapping them (a later id of the same person)
            if e["t0"] - 2.0 <= t <= e["t1"] + 3.0 and (tid in e["tids"] or any(
                    abs(bt - t) <= 1.5 and iou(bb, b) >= 0.3 for bt, b in e["boxes"])):
                hit = e
                break
        if hit is not None:
            hit["hit"] = hit["hit"] if hit["hit"] is not None else t
        else:
            near = {LAB.get(r["id"]) for r in IDX if r["src"] == src and abs(r["t"] - t) <= 1.5
                    and iou(bb, r["bbox"]) >= 0.3}
            fps.append({"t": t, "tid": tid, "gt": "FP(out)" if "out" in near and "pocket" not in near else "unverified"})
    real = [{k: e[k] for k in ("name", "tids", "t0", "t1", "n", "both", "walking", "sustained", "hit")}
            for e in eps if e["sustained"]]
    return {"real": real, "false": fps}


# ---- Dx (README) --------------------------------------------------------------------
def score_dx(alerts, shown, tracks):
    rd = json.load(open(r"D:\poketenashi-data\po_tests\2026-10-05_dx\README.json", encoding="utf-8"))
    # README track ids are those of the 5 Oct replay (dx.jsonl); this cache's
    # tracker numbered people differently - map by box overlap at the same time
    old = defaultdict(list)
    for l in open(r"D:\poketenashi-data\po_tests\2026-10-05_dx\dx.jsonl", encoding="utf-8"):
        o = json.loads(l)
        old[o["tid"]].append((o["t"], o["bbox"]))
    real, false, used = [], [], set()
    for e in rd["episodes_by_eye"]:
        if e["label"] not in ("pocket", "out"):
            continue
        otid, (ta, tb) = e["replay_track"], e["t"]
        ob = [(t, b) for t, b in old[otid] if ta <= t <= tb]
        tids = {tid for tid, tr in tracks.items()
                if any(abs(t - t2) <= 0.05 and iou(b, b2) >= 0.4 for t, b in tr if ta <= t <= tb for t2, b2 in ob)}
        hit = sorted(t for tid in tids for t in shown.get(tid, []) if ta - 1.0 <= t <= tb + 3.0)
        al = [a for a in alerts if a[1] in tids and ta - 1.0 <= a[0] <= tb + 3.0]
        for a in al:
            used.add(id(a))
        rec = {"name": f"dx-{otid}@{ta:.0f}", "tids": sorted(tids), "t0": ta, "t1": tb, "note": e["note"][:80],
               "hit": round(hit[0], 1) if hit else None, "alerts": len(al),
               "shown_s": round(len(hit) / 6.25, 1)}
        if e["label"] == "pocket":
            rec.update(walking=True, both=False)
            real.append(rec)
        else:
            n = e["note"].lower()
            rec["cat"] = "behind_back" if "behind" in n else "straight"
            if hit:
                false.append(rec)
    for a in alerts:
        if id(a) not in used:
            false.append({"t": a[0], "tid": a[1], "cat": "unlabelled"})
    return {"real": real, "false": false}


# ---- 6 Oct Dx test (by eye, full-res zooms dx6/zoom_*.jpg) ------------------------
DX6_REAL = [(29, 67.0, 73.1), (51, 160.0, 165.6), (51, 179.5, 184.0)]
DX6_FALSE = [(51, 105.6, 108.0, "visible_at_pocket"), (51, 143.5, 147.0, "visible_at_pocket"),
             (76, 200.6, 203.5, "visible_at_pocket"), (51, 120.0, 120.6, "object"), (51, 127.5, 130.1, "hands_on_hips"),
             (40, 86.0, 102.0, "straight"), (51, 108.6, 111.0, "straight"), (51, 113.0, 119.5, "behind_back"),
             (51, 168.0, 175.0, "straight"), (51, 186.5, 192.5, "behind_back")]


def score_dx6(alerts, shown):
    real, false = [], []
    for tid, ta, tb in DX6_REAL:
        hit = [t for t in shown.get(tid, []) if ta - 1.0 <= t <= tb + 1.0]
        real.append({"name": f"dx6-{tid}@{ta:.0f}", "tids": [tid], "t0": ta, "t1": tb, "walking": True, "both": False,
                     "hit": round(hit[0], 1) if hit else None})
    for tid, ta, tb, cat in DX6_FALSE:
        hit = [t for t in shown.get(tid, []) if ta <= t <= tb]
        if hit:
            false.append({"name": f"dx6-{tid}@{ta:.0f}", "tids": [tid], "t0": ta, "t1": tb - 3.0, "cat": cat,
                          "shown_s": round(len(hit) / 6.25, 1)})
    return {"real": real, "false": false}


def MODEL(s):
    return (variant.get("model_by_set") or {}).get(s, HELD.get(s) if not variant.get("model_by_set") else None)


# ---- live clips -------------------------------------------------------------------------
def score_live(results):
    lab = json.load(open(os.path.join(SP, "live6_labels.json"), encoding="utf-8"))["labels"]
    idx = json.load(open(os.path.join(SP, "live6big", "po_live_index.json"), encoding="utf-8"))
    real, false = [], []
    for r in idx:
        L = lab.get(str(r["n"]))
        if not L or L[0] == "DX" or r["clip"] not in results:
            continue
        alerts, shown, tracks = results[r["clip"]]
        x, y, w, h = r["box"] or (0, 0, 0, 0)
        pbox = [1.5 * x, 1.5 * y, 1.5 * (x + w), 1.5 * (y + h)]       # 1280x720 poster -> 1920
        who = {tid for tid, tr in tracks.items() if any(abs(t - 4.0) <= 1.5 and iou(b, pbox) >= 0.3 for t, b in tr)}
        fired = [a for a in alerts if a[1] in who]
        rec = {"name": f"live#{r['n']} {r['time'][5:]} ID{r['track_id']}", "cat": L[1], "note": L[2], "clip": r["clip"], "tids": sorted(who),
               "fired": bool(fired), "matched_track": bool(who)}
        if L[0] == "TP":
            rec.update(hit=fired[0][0] if fired else None, walking=False, both=None)
            real.append(rec)
        elif L[0] == "FP" and fired:
            false.append(rec)
    return {"real": real, "false": false}


def score_live7(results):
    """7 Oct: the 32 live PO alerts 6 Oct 14:58 - 7 Oct 11:33 (PO_review, labelled by eye: OK real / NG false).
    The alerted person = the track under the clip's red alert box."""
    rb = json.load(open(os.path.join(SP, "live7_redboxes.json")))
    real, false = [], []
    for l in open(os.path.join(SP, "live7_alerts.jsonl")):
        a = json.loads(l)
        if a["clip"] not in results:
            continue
        alerts, shown, tracks, js = results[a["clip"]]
        boxes = rb.get(a["clip"], {})
        who = set()
        for tid, tr in tracks.items():
            for (t, b), j in zip(tr, js[tid]):
                rbx = boxes.get(str(j))
                if rbx and iou(b, rbx) >= 0.3:
                    who.add(tid)
                    break
        fired = [x for x in alerts if x[1] in who]
        rec = {"name": f"live7#{a['n']} {a['clip'][5:22]}", "cat": a["why"], "clip": a["clip"], "tids": sorted(who), "fired": bool(fired)}
        if a["verdict"] == "OK":
            rec.update(hit=fired[0][0] if fired else None, walking=None, both=None)
            real.append(rec)
        elif fired:
            false.append(rec)
    return {"real": real, "false": false}


# ---- run -------------------------------------------------------------------------------
summary = {"variant": variant, "sets": {}}
for s in which:
    a0 = time.time()
    if s in ("t25", "lt", "raw", "raw2"):
        fn, video = {"t25": ("cache_t25_s2.pkl", "footage/po_test_20260925.mp4"),
                     "lt": ("cache_lt_s2.pkl", "footage/live_test_20260928.mp4"),
                     "raw": ("cache_raw_s4.pkl", r"D:/poketenashi-data/po_raw/cam1_20261003_1229.mp4"),
                     "raw2": ("cache_raw2_s4.pkl", r"D:/poketenashi-data/po_raw/cam1_20261003_1526.mp4")}[s]
        cache = pickle.load(open(os.path.join(OLD, fn), "rb"))
        alerts, shown, tracks, dbg = replay(cache, video, BASE["zones"], MODEL(s))
        if s in ("t25", "lt"):
            r = score_arms(s, alerts, tracks)
        else:
            r = {"real": [], "false": [{"t": a[0], "tid": a[1], "cat": "raw"} for a in alerts]}
    elif s == "dx":
        cache = pickle.load(open(os.path.join(SP, "cache_dx.pkl"), "rb"))
        alerts, shown, tracks, dbg = replay(cache, cache["video"], CAMS["cam2"]["zones"], MODEL(s))
        r = score_dx(alerts, shown, tracks)
    elif s == "dx6":
        cache = pickle.load(open(os.path.join(SP, "cache_dx6.pkl"), "rb"))
        alerts, shown, tracks, dbg = replay(cache, cache["video"], CAMS["cam2"]["zones"], MODEL(s))
        r = score_dx6(alerts, shown)
    elif s == "live7":
        cache = pickle.load(open(os.path.join(SP, "cache_live7.pkl"), "rb"))
        res, dbg = {}, []
        for clip, c in sorted(cache.items()):
            z = CAMS.get(c["alert"]["camera"], {}).get("zones") or BASE["zones"]
            mdl = (variant.get("live7_models") or {}).get(clip[:10]) or MODEL(s)
            al, sh, tr, db = replay(c, c["video"], z, mdl)
            jmap = defaultdict(list)
            for j, t, plist in c["rows"]:
                for p in plist:
                    if p[4]:
                        jmap[p[0]].append(j)
            res[clip] = (al, sh, tr, jmap)
            dbg += [dict(d, clip=clip) for d in db]
        r = score_live7(res)
    elif s == "live6":
        cache = pickle.load(open(os.path.join(SP, "cache_live6.pkl"), "rb"))
        res, dbg = {}, []
        for clip, c in sorted(cache.items()):
            if "cam2" in clip:
                continue
            al, sh, tr, db = replay(c, c["video"], BASE["zones"], MODEL(s))
            res[clip] = (al, sh, tr)
            dbg += [dict(d, clip=clip) for d in db]
        r = score_live(res)
    r["sec"] = round(time.time() - a0)
    r["alerts"] = alerts if s not in ("live6", "live7") else {k: v[0] for k, v in res.items()}
    r["dbg"] = dbg
    summary["sets"][s] = r
    nr = len(r["real"])
    hr = sum(1 for e in r["real"] if e.get("hit") is not None)
    print(f"[{variant.get('name')}] {s}: real {hr}/{nr}  false alerts {len(r['false'])}  ({r['sec']}s)", flush=True)
    for e in r["real"]:
        print(f"    REAL {e['name']:28s} {'walk' if e.get('walking') else 'stand' if e.get('walking') is False else '?':5s} "
              f"{'both' if e.get('both') else 'one ':4s} -> {'HIT ' + str(e['hit']) if e.get('hit') is not None else 'missed'}")
    for e in r["false"]:
        print(f"    FALSE {json.dumps({k: v for k, v in e.items() if k in ('name', 't', 'tid', 'cat', 'gt', 'note')})[:150]}")
json.dump(summary, open(out_path, "w"), default=str)
