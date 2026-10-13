"""Which model blocks the false PO / KE alerts and which loses the real ones (10 Oct 2026, user:
"check every PO and KE alert on every model - which model stopped more NG alerts and which missed
OK ones, with ID, snapshot and clip - include the night alerts, the lighting is different").

Every PO / KE alert in violations.jsonl is replayed through the real pipeline (tools/replay.py:
detector, rule engine, video clock) once per model of its rule, on the footage around the alert:
the raw recording (persons_raw, clean) when it covers the moment, else the alert's own clip (boxes
drawn on it - marked in the report). A model "fires" when its rule fires on the alerted person
(the red box of the alert's poster) within FIRE_WINDOW of the alert. With the OK / NG label of the
alert (labels.csv: the user's review, else Claude's, else open) that gives per model:
NG blocked, NG still fired, OK kept, OK missed - day and night apart.

Only alerts that fired live are judged: a model's NEW false alerts elsewhere are not counted.

    python -m tools.model_compare collect               # alert list -> D:/poketenashi-data/model_compare
    python -m tools.model_compare run [--rule PO] [--limit 5] [--threads 6]   # resumable
    python -m tools.model_compare sheets                # close-ups for labelling (sheets/<key>.jpg)
    python -m tools.model_compare loop --stop 08:00     # overnight: collect + run + sheets every 30 min
    python -m tools.model_compare report                # report.html + summary.csv
"""
import argparse
import copy
import csv
import datetime as dt
import glob
import html
import json
import os
import sys
import time

import cv2
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

WORK = "D:/poketenashi-data/model_compare"
CLIPS = "D:/poketenashi-data/clips"
RAW = "D:/poketenashi-data/persons_raw"
PRE_SEC, POST_SEC = 10.0, 4.0      # footage replayed around each alert (raw clock within ~1 s of the alert)
REPLAY_FPS = 5.0                   # about the live detection rate (0.4-9 fps)
FIRE_WINDOW = 5.0                  # a fire this close to the alert time counts
CLIP_PRE_SEC = 4.0                 # clips.pre_sec: a clip starts this long before the alert that opened it
NIGHT = (19, 7)                    # 19:00-07:00 counts as night

# The models of each rule, each with the thresholds it ran live with (config backups).
VARIANTS = {
    "PO": {
        "po_arm_mnv3 (live)": {"rules.PO.model_path": "models/po_arm_mnv3.pt", "rules.PO.evidence_hi": 0.80, "rules.PO.evidence_lo": 0.30},
        "po_arm_v4": {"rules.PO.model_path": "models/po_arm_v4.pt", "rules.PO.evidence_hi": 0.80, "rules.PO.evidence_lo": 0.30},
        "po_arm_v6": {"rules.PO.model_path": "models/po_arm_v6.pt", "rules.PO.evidence_hi": 0.80, "rules.PO.evidence_lo": 0.30},
        "po_arm_v7": {"rules.PO.model_path": "models/po_arm_v7.pt", "rules.PO.evidence_hi": 0.50, "rules.PO.evidence_lo": 0.15},
        "po_arm_v8": {"rules.PO.model_path": "models/po_arm_v8.pt", "rules.PO.evidence_hi": 0.35, "rules.PO.evidence_lo": 0.10},
    },
    "KE": {
        "yolo11m (live)": {"rules.KE.phone_model": "yolo11m.pt"},
        "yolov8s": {"rules.KE.phone_model": "yolov8s.pt"},
        "yolo11n": {"rules.KE.phone_model": "yolo11n.pt"},
    },
}


def epoch(s):
    return time.mktime(time.strptime(s, "%Y-%m-%d %H:%M:%S"))


def is_night(t):
    h = dt.datetime.fromtimestamp(t).hour
    return h >= NIGHT[0] or h < NIGHT[1]


# ------------------------------------------------------------------ collect
def raw_segment(cam, t):
    """(path, start epoch) of the raw recording holding [t - PRE_SEC, t + POST_SEC], or None."""
    # a recorder keeps writing into the folder of the day it started (11 Oct files in 20261010/raw)
    segs = []
    for p in glob.glob(os.path.join(RAW, "*", "raw", f"{cam}_*.mp4")):
        try:
            s = time.mktime(time.strptime(os.path.basename(p)[len(cam) + 1:-4], "%Y%m%d_%H%M%S"))
        except ValueError:
            continue
        segs.append((s, p))
    segs.sort()
    for i, (s, p) in enumerate(segs):
        end = segs[i + 1][0] if i + 1 < len(segs) else time.time()
        if s <= t - 2.0 and t + POST_SEC <= end:
            # a segment still being written (or cut off by a crash) has no index yet: not readable
            cap = cv2.VideoCapture(p)
            ok = cap.isOpened() and (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) > 0
            cap.release()
            return (p, s) if ok else None
    return None


def collect(a):
    os.makedirs(WORK, exist_ok=True)
    until = epoch(a.until) if a.until else time.time()
    rows = []
    for line in open(os.path.join(ROOT, "violations.jsonl"), encoding="utf-8"):
        try:
            v = json.loads(line)
        except ValueError:
            continue
        if v.get("rule") not in VARIANTS or not v.get("time"):
            continue
        t = epoch(v["time"])
        if t > until:
            continue
        key = os.path.splitext(v.get("poster") or v.get("snapshot") or v.get("clip") or "")[0] or f"{int(t)}_ID{v['track_id']}_{v['rule']}"
        cam = v.get("camera") or "cam1"
        src = raw_segment(cam, t)
        row = {"key": key, "rule": v["rule"], "time": v["time"], "t": t, "camera": cam, "camera_name": v.get("camera_name", cam),
               "track_id": v["track_id"], "clip": v.get("clip"), "poster": v.get("poster"), "snapshot": v.get("snapshot"),
               "night": is_night(t)}
        if src:
            row.update(source="raw", footage=src[0], offset=t - src[1])
        elif v.get("clip") and os.path.exists(os.path.join(CLIPS, v["clip"])):
            opened = float(v["clip"].split("_")[0])
            row.update(source="clip", footage=os.path.join(CLIPS, v["clip"]), offset=t - opened + CLIP_PRE_SEC)
        else:
            row.update(source=None, footage=None, offset=None)
        rows.append(row)
    with open(os.path.join(WORK, "alerts.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)
    by = {}
    for r in rows:
        by.setdefault((r["rule"], r["source"]), 0)
        by[(r["rule"], r["source"])] += 1
    print(f"[collect] {len(rows)} alerts -> {WORK}/alerts.json", {f"{k[0]}/{k[1]}": n for k, n in sorted(by.items(), key=str)})
    labels_init(rows)


def labels_init(rows):
    """labels.csv keeps every label already given; new alerts get a row with an empty label.
    The KE review of 10 Oct (ke_review/clips_check/review.csv) is taken over."""
    p = os.path.join(WORK, "labels.csv")
    have = {}
    if os.path.exists(p):
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            have[r["key"]] = r
    ke_review = "D:/poketenashi-data/ke_review/clips_check/review.csv"
    if os.path.exists(ke_review):
        for r in csv.DictReader(open(ke_review, encoding="utf-8-sig")):
            key = os.path.splitext(r["clip"])[0] if r.get("clip") else None
            for row in rows:
                if row["rule"] == "KE" and row["time"] == r["time"] and str(row["track_id"]) == r["track"]:
                    key = row["key"]
            if key and r.get("your_OK_NG") in ("OK", "NG") and (key not in have or not have[key].get("label")):
                have[key] = {"key": key, "label": r["your_OK_NG"], "by": "user", "note": r.get("reason", "")}
    with open(p, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, ["key", "rule", "time", "camera", "track_id", "label", "by", "note"])
        w.writeheader()
        for row in rows:
            h = have.get(row["key"], {})
            w.writerow({"key": row["key"], "rule": row["rule"], "time": row["time"], "camera": row["camera_name"],
                        "track_id": row["track_id"], "label": h.get("label", ""), "by": h.get("by", ""), "note": h.get("note", "")})
    print(f"[collect] labels.csv: {sum(1 for r in rows if have.get(r['key'], {}).get('label'))} of {len(rows)} labelled")


# ------------------------------------------------------------------ run
def alert_boxes(row, width):
    """Red boxes of the alert's poster (the alerted person), scaled to the replay width."""
    from tools.scan_clips import red_boxes
    p = os.path.join(CLIPS, row["poster"]) if row.get("poster") else None
    img = cv2.imread(p) if p and os.path.exists(p) else None
    if img is None:
        return []
    s = width / float(img.shape[1])
    return [(x * s, y * s, (x + w) * s, (y + h) * s) for x, y, w, h, _ in red_boxes(img)]


def on_person(bbox, boxes):
    if not boxes:
        return True                                      # no red box on the poster: any fire counts
    x0, y0, x1, y1 = bbox
    for a0, b0, a1, b1 in boxes:
        iw, ih = max(0.0, min(x1, a1) - max(x0, a0)), max(0.0, min(y1, b1) - max(y0, b0))
        inter = iw * ih
        union = (x1 - x0) * (y1 - y0) + (a1 - a0) * (b1 - b0) - inter
        cx, cy, ax, ay = (x0 + x1) / 2, (y0 + y1) / 2, (a0 + a1) / 2, (b0 + b1) / 2
        if (union > 0 and inter / union >= 0.2) or abs(cx - ax) <= 0.6 * (a1 - a0) and abs(cy - ay) <= 0.4 * (b1 - b0):
            return True
    return False


def frames_for(row, width):
    cap = cv2.VideoCapture(row["footage"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
    t0 = max(0.0, row["offset"] - PRE_SEC)
    cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000.0)
    out, nxt = [], t0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if ts > row["offset"] + POST_SEC:
            break
        if ts + 1e-6 < nxt:
            continue
        nxt = ts + 1.0 / REPLAY_FPS
        if f.shape[1] != width:
            f = cv2.resize(f, (width, int(f.shape[0] * width / f.shape[1])), interpolation=cv2.INTER_AREA)
        out.append((ts - row["offset"], f))              # seconds relative to the alert
    cap.release()
    return out, fps


def _merged(base, over):
    """web.py _merged (importing web.py would start its logging): base with over laid on top."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merged(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def camera_cfg(base, cam_id):
    """config.yaml with the camera's own zones / rules / models laid over (as web.py CameraRuntime)."""
    cams = json.load(open(os.path.join(ROOT, "monitor_cameras.json"), encoding="utf-8"))["cameras"]
    cam = next((c for c in cams if c["id"] == cam_id), {})
    c = copy.deepcopy(base)
    c["zones"] = copy.deepcopy(cam.get("zones") or {})
    c["clock_box"] = cam.get("clock_box")
    if cam.get("rules"):
        c["rules"] = _merged(base.get("rules") or {}, cam["rules"])
    if cam.get("models"):
        c["models"] = _merged(base.get("models") or {}, cam["models"])
    return c


def run(a):
    from poketenashi import po_refs
    from poketenashi.detector import Detector
    from poketenashi.rules import RuleEngine
    from poketenashi.zones import Zones
    from tools.replay import VideoClock, apply_overrides, install_clock
    import torch
    torch.set_num_threads(a.threads)
    cv2.setNumThreads(2)
    # master caches of the models that are not live go to the work folder, not models/
    po_refs._cache_path = lambda model: os.path.join(WORK, "po_refs_cache_" + os.path.splitext(os.path.basename(model.path))[0] + ".pkl")

    rows = json.load(open(os.path.join(WORK, "alerts.json"), encoding="utf-8"))
    res_p = os.path.join(WORK, "results.jsonl")
    done = set()
    if os.path.exists(res_p):
        for line in open(res_p, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["key"], r["variant"]))
    base = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))
    width = int((base.get("camera") or {}).get("max_width") or 1920)
    # Only the clean raw recording is replayed: the alert clips are 1280 px JPEG with the live
    # boxes drawn on them, and there even the live phone model finds no phone (10 Oct, alert ID57).
    todo = [r for r in rows if r["footage"] and (a.clips or r["source"] == "raw") and (not a.rule or r["rule"] == a.rule)
            and any((r["key"], v) not in done for v in VARIANTS[r["rule"]])]
    if a.limit:
        todo = todo[:a.limit]
    print(f"[run] {len(todo)} alerts to replay ({len(done)} results already)", flush=True)
    for n, row in enumerate(todo, 1):
        t_start = time.time()
        frames, src_fps = frames_for(row, width)
        if len(frames) < 5:
            with open(res_p, "a", encoding="utf-8") as fh:
                for v in VARIANTS[row["rule"]]:
                    fh.write(json.dumps({"key": row["key"], "variant": v, "error": "no footage frames"}) + "\n")
            continue
        boxes = alert_boxes(row, width)
        cfg_cam = camera_cfg(base, row["camera"])
        det = Detector(cfg_cam)                          # a fresh tracker per alert
        dets = [det(f) for _, f in frames]               # shared by every variant
        for vname, sets in VARIANTS[row["rule"]].items():
            if (row["key"], vname) in done:
                continue
            cfg = copy.deepcopy(cfg_cam)
            apply_overrides(cfg, [f"{k}={v}" for k, v in sets.items()])
            clock = VideoClock()
            install_clock(clock)
            engine = RuleEngine(cfg, Zones(cfg.get("zones")))
            fires, any_fire = [], []
            for (ts, f), (persons, phones) in zip(frames, dets):
                clock.advance(1.0 / REPLAY_FPS)
                for v in engine.check(copy.deepcopy(persons), copy.deepcopy(phones), f):
                    if v["rule"] != row["rule"]:
                        continue
                    hit = {"sec": round(ts, 2), "bbox": [round(float(c)) for c in v["bbox"]]}
                    any_fire.append(hit)
                    if abs(ts) <= FIRE_WINDOW and on_person(v["bbox"], boxes):
                        fires.append(hit)
            out = {"key": row["key"], "variant": vname, "fired": bool(fires), "first": fires[0]["sec"] if fires else None,
                   "fires_on_person": len(fires), "fires_any": len(any_fire), "frames": len(frames),
                   "located": bool(boxes), "source": row["source"]}
            with open(res_p, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(out) + "\n")
        print(f"[run] {n}/{len(todo)} {row['key']} ({row['rule']}, {row['source']}, {len(frames)} frames) "
              f"{time.time() - t_start:.0f}s", flush=True)


# ------------------------------------------------------------------ sheets / loop
def sheets(a):
    """sheets/<key>.jpg for every alert with raw footage and no label yet: the poster (where the
    alert was) and six close-ups of that place from the clean recording, 3 s before to 2 s after -
    what the OK / NG label is decided on."""
    os.makedirs(os.path.join(WORK, "sheets"), exist_ok=True)
    rows = json.load(open(os.path.join(WORK, "alerts.json"), encoding="utf-8"))
    labels = {r["key"]: r for r in csv.DictReader(open(os.path.join(WORK, "labels.csv"), encoding="utf-8-sig"))}
    n = 0
    for row in rows:
        out = os.path.join(WORK, "sheets", row["key"] + ".jpg")
        if row["source"] != "raw" or os.path.exists(out) or (labels.get(row["key"]) or {}).get("label"):
            continue
        poster = cv2.imread(os.path.join(CLIPS, row["poster"])) if row.get("poster") else None
        cap = cv2.VideoCapture(row["footage"])
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        boxes = alert_boxes(row, W)
        tiles = []
        for dt_ in (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0):
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, row["offset"] + dt_) * 1000.0)
            ok, f = cap.read()
            if not ok:
                continue
            H = f.shape[0]
            if boxes:
                x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes); x1 = max(b[2] for b in boxes); y1 = max(b[3] for b in boxes)
                bw, bh = x1 - x0, y1 - y0
                c = f[int(max(0, y0 - 0.1 * bh)):int(min(H, y1 + 0.08 * bh)), int(max(0, x0 - 0.6 * bw)):int(min(W, x1 + 0.6 * bw))]
            else:
                c = f
            c = cv2.resize(c, (int(c.shape[1] * 520 / c.shape[0]), 520), interpolation=cv2.INTER_CUBIC)
            cv2.putText(c, f"{dt_:+.0f}s", (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            tiles.append(c)
        cap.release()
        if not tiles:
            continue
        # two rows of three close-ups, the poster (where the alert was) on the right
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        tw = max(t.shape[1] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, 0, 0, tw - t.shape[1], cv2.BORDER_CONSTANT) for t in tiles]
        body = np.vstack([np.hstack(tiles[k:k + 3]) for k in range(0, len(tiles), 3)])
        if poster is not None:
            p = cv2.resize(poster, (int(poster.shape[1] * body.shape[0] / 2 / poster.shape[0]), body.shape[0] // 2))
            p = cv2.copyMakeBorder(p, 0, body.shape[0] - p.shape[0], 0, 0, cv2.BORDER_CONSTANT)
            body = np.hstack([body, p])
        if body.shape[1] > 2600:
            body = cv2.resize(body, (2600, int(body.shape[0] * 2600 / body.shape[1])))
        head = np.zeros((44, body.shape[1], 3), np.uint8)
        cv2.putText(head, f"{row['rule']}  {row['time']}  {row['camera_name']}  ID{row['track_id']}  {row['key']}", (8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(out, np.vstack([head, body]), [cv2.IMWRITE_JPEG_QUALITY, 85])
        n += 1
    print(f"[sheets] {n} new sheets in {WORK}/sheets")


def loop(a):
    """Overnight: every --every minutes until --stop, pick up the new alerts, replay them and make
    their sheets. Run at below-normal priority so the live monitor keeps the CPU."""
    stop = time.mktime(time.strptime(time.strftime("%Y-%m-%d ") + a.stop + ":00", "%Y-%m-%d %H:%M:%S"))
    if stop < time.time():
        stop += 86400
    while True:
        for step in (collect, run, sheets):
            try:
                step(a)
            except Exception as exc:
                print(f"[loop] {step.__name__} failed: {type(exc).__name__}: {exc}", flush=True)
        if time.time() >= stop:
            break
        time.sleep(min(a.every * 60.0, max(0.0, stop - time.time()) + 5))
    print("[loop] done", flush=True)


# ------------------------------------------------------------------ report
def report(a):
    rows = {r["key"]: r for r in json.load(open(os.path.join(WORK, "alerts.json"), encoding="utf-8"))}
    labels = {r["key"]: r for r in csv.DictReader(open(os.path.join(WORK, "labels.csv"), encoding="utf-8-sig"))}
    res = {}
    for line in open(os.path.join(WORK, "results.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        if "error" not in r:
            res[(r["key"], r["variant"])] = r
    summary = []
    for rule, variants in VARIANTS.items():
        for vname in variants:
            for part in ("all", "day", "night"):
                c = {"rule": rule, "model": vname, "part": part, "NG blocked": 0, "NG fired": 0, "OK kept": 0, "OK missed": 0,
                     "open fired": 0, "open blocked": 0}
                for key, row in rows.items():
                    if row["rule"] != rule or (key, vname) not in res:
                        continue
                    if part != "all" and row["night"] != (part == "night"):
                        continue
                    lab = (labels.get(key) or {}).get("label") or "open"
                    fired = res[(key, vname)]["fired"]
                    k = {("NG", False): "NG blocked", ("NG", True): "NG fired", ("OK", True): "OK kept",
                         ("OK", False): "OK missed"}.get((lab, fired), "open fired" if fired else "open blocked")
                    c[k] += 1
                summary.append(c)
    with open(os.path.join(WORK, "summary.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    esc = html.escape
    def link(name, label):
        return f'<a href="file:///{CLIPS}/{esc(name)}">{label}</a>' if name else ""
    parts = []
    for rule, variants in VARIANTS.items():
        head = "".join(f"<th>{esc(v)}</th>" for v in variants)
        srows = "".join(
            f"<tr><td>{esc(s['model'])}</td>" + "".join(f"<td class=n>{s[k]}</td>" for k in ("NG blocked", "NG fired", "OK kept", "OK missed", "open fired", "open blocked"))
            + "".join(f"<td class=n>{x['NG blocked']} / {x['OK missed']}</td>" for x in summary if x["rule"] == rule and x["model"] == s["model"] and x["part"] != "all")
            + "</tr>" for s in summary if s["rule"] == rule and s["part"] == "all")
        arows = []
        for key, row in sorted(rows.items(), key=lambda kv: kv[1]["t"]):
            if row["rule"] != rule:
                continue
            lab = labels.get(key) or {}
            cells = ""
            for v in variants:
                r = res.get((key, v))
                cells += "<td class=c>–</td>" if r is None else (f"<td class='c fire'>fires</td>" if r["fired"] else "<td class='c quiet'>silent</td>")
            arows.append(f"<tr><td>{esc(row['time'])}{' 🌙' if row['night'] else ''}</td><td>{esc(row['camera_name'])}</td><td>ID{row['track_id']}</td>"
                         f"<td class='lab {esc(lab.get('label') or 'open')}'>{esc(lab.get('label') or '?')}</td>"
                         f"<td>{link(row.get('poster'), 'snapshot')} {link(row.get('clip'), 'clip')}</td><td>{esc(row['source'] or 'no footage')}</td>{cells}"
                         f"<td>{esc(lab.get('note', ''))}</td></tr>")
        parts.append(f"<h2>{rule}</h2><table><tr><th>model</th><th>NG blocked</th><th>NG still fired</th><th>OK kept</th><th>OK missed</th>"
                     f"<th>unlabelled fired</th><th>unlabelled silent</th><th>day: NG blocked / OK missed</th><th>night: NG blocked / OK missed</th></tr>{srows}</table>"
                     f"<h3>Every {rule} alert</h3><table><tr><th>time</th><th>camera</th><th>ID</th><th>label</th><th>evidence</th><th>footage</th>{head}<th>note</th></tr>"
                     + "".join(arows) + "</table>")
    page = f"""<!doctype html><meta charset=utf-8><title>PO / KE model comparison</title>
<style>body{{font:13px system-ui,sans-serif;margin:16px}} table{{border-collapse:collapse;margin:8px 0 18px}} td,th{{border:1px solid #ccc;padding:3px 6px}}
.n{{text-align:right}} .c{{text-align:center}} .fire{{background:#fde2e2}} .quiet{{background:#e3f6e3}} .lab.NG{{color:#b00;font-weight:600}} .lab.OK{{color:#070;font-weight:600}}</style>
<h1>PO / KE model comparison</h1>
<p>Each alert replayed through the real rules once per model (footage: raw recording when available, else the alert clip).
"fires" = that model would also have alerted this person; "silent" = it would not. NG blocked and OK missed are what to compare.
Only alerts that fired live are judged: new false alerts a model would make elsewhere are not counted. Night = 19:00-07:00 (🌙).
Made {dt.datetime.now():%Y-%m-%d %H:%M}.</p>{''.join(parts)}"""
    open(os.path.join(WORK, "report.html"), "w", encoding="utf-8").write(page)
    for s in summary:
        if s["part"] == "all":
            print(f"{s['rule']} {s['model']:<20} NG blocked {s['NG blocked']:>3}  NG fired {s['NG fired']:>3}  OK kept {s['OK kept']:>3}  "
                  f"OK missed {s['OK missed']:>3}  unlabelled {s['open fired'] + s['open blocked']}")
    print("[report]", os.path.join(WORK, "report.html"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "run", "sheets", "loop", "report"])
    ap.add_argument("--stop", default="08:00", help="loop: last round at this time (HH:MM)")
    ap.add_argument("--every", type=float, default=30.0, help="loop: minutes between rounds")
    ap.add_argument("--until", default="", help="collect: alerts up to 'YYYY-MM-DD HH:MM:SS' (default now)")
    ap.add_argument("--rule", default="", help="run: PO or KE only")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--clips", action="store_true", help="run: also replay alerts that only have their (drawn, 1280 px) clip")
    a = ap.parse_args()
    {"collect": collect, "run": run, "sheets": sheets, "loop": loop, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
