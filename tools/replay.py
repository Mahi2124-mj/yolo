"""Replay recorded footage through the real pipeline, deterministically.

Nothing here is simulated. The footage is the camera's own, recorded at the same
width the live loop runs on (tools/record.py). The pose model, the phone model
and the rule engine are the ones that run in production, loaded from the same
config.

One thing IS substituted, and it has to be: the clock. Every time-based
threshold in this project - the KE ear streak, the phone memory, the latch, the
parked-phone test - is measured in seconds. On this machine one frame of CPU
inference takes over a second, while that frame represents about 0.08 s of
video. Left on the wall clock, a "2 second" window covers two frames and the
test it guards silently never fires. So the clock advances by 1/fps per frame,
which is real video time, not invented time.

What that buys: the same footage can be run through different settings and the
results compared directly. A live 15 minute window cannot be compared with
another - different people, different traffic - so "13 alerts" versus "51" never
told us how much was the change and how much was the day.

Usage:
    python -m tools.replay footage/20260905_171500.mp4
    python -m tools.replay f.mp4 --set rules.KE.phone_conf=0.5
    python -m tools.replay f.mp4 --json before.json
    python -m tools.replay f.mp4 --set ke.KE_LATCH_SEC=1.0 --json after.json
    python -m tools.replay --diff before.json after.json
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import cv2
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from poketenashi import ke as ke_mod                     # noqa: E402
from poketenashi import na as na_mod                     # noqa: E402
from poketenashi import rules as rules_mod               # noqa: E402
from poketenashi.detector import Detector                # noqa: E402
from poketenashi.rules import RuleEngine                 # noqa: E402
from poketenashi.zones import Zones                      # noqa: E402


class VideoClock:
    """Wall-clock replacement that advances with the video, not the CPU.

    Exposes both names the pipeline uses: ke.py calls time.time(), rules.py
    calls time.monotonic().
    """

    def __init__(self, start=1_000_000.0):
        self.t = float(start)

    def advance(self, dt):
        self.t += dt

    def time(self):
        return self.t

    def monotonic(self):
        return self.t


def install_clock(clock):
    """Point the modules that measure seconds at the video clock.

    Both modules do `import time` and call it through the module object, so
    replacing the attribute is enough and no production code has to grow a test
    seam it would otherwise never use.
    """
    patched = []
    # na.py times its heading window and confirm_sec with time.time() too; left
    # on the wall clock, NA in a replay judged ~1 s of CPU time, not of video.
    for mod in (ke_mod, na_mod, rules_mod):
        assert hasattr(mod, "time"), f"{mod.__name__} no longer imports time"
        mod.time = clock
        patched.append(mod.__name__)
    return patched


def apply_overrides(cfg, pairs):
    """--set a.b.c=value, so one command is one variant.

    `ke.NAME=value` reaches the module constants in ke.py that are not in the
    config at all (KE_LATCH_SEC, KE_EAR_CROP_CONF, ...).
    """
    notes = []
    for pair in pairs or []:
        key, _, raw = pair.partition("=")
        if not raw:
            raise SystemExit(f"--set needs key=value, got {pair!r}")
        try:
            val = yaml.safe_load(raw)
        except Exception:
            val = raw
        parts = key.split(".")
        if parts[0] == "ke":
            name = parts[1]
            if not hasattr(ke_mod, name):
                raise SystemExit(f"ke.py has no constant {name!r}")
            old = getattr(ke_mod, name)
            setattr(ke_mod, name, val)
            notes.append(f"ke.{name}: {old} -> {val}")
            continue
        node = cfg
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                raise SystemExit(f"config has no section {'.'.join(parts[:-1])!r}")
            node = node[p]
        old = node.get(parts[-1])
        node[parts[-1]] = val
        notes.append(f"{key}: {old} -> {val}")
    return notes


def run(path, cfg, args):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[replay] {path}")
    print(f"[replay] {w}x{h}, {fps:.1f} fps, {total} frames "
          f"({total / fps:.0f}s of video)")

    live_w = cfg["camera"].get("max_width", 0)
    if live_w and w != live_w:
        print(f"[replay] WARNING: this footage is {w} px wide but the live pipeline "
              f"runs at {live_w}. Detections will NOT match production - "
              f"re-record with tools/record.py.")

    clock = VideoClock()
    print("[replay] video clock installed in:", ", ".join(install_clock(clock)))

    detector = Detector(cfg)
    zones = Zones(cfg.get("zones"))
    engine = RuleEngine(cfg, zones)

    fires = []                       # (frame, video_sec, rule, track_id)
    why = []
    per_rule = Counter()
    per_rule_tracks = defaultdict(set)
    frames_seen = 0
    step = 1.0 / fps
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.limit and frames_seen >= args.limit:
            print(f"[replay] stopping at --limit {args.limit}")
            break
        clock.advance(step)
        frames_seen += 1
        persons, phones = detector(frame)
        results = engine.check(persons, phones, frame)
        if args.why and getattr(engine, "ke", None) is not None:
            ke = engine.ke
            now = ke_mod.time.time()
            for v in results:
                if v["rule"] != "KE":
                    continue
                tid = v["track_id"]
                hits = ke._hits.get(tid) or []
                why.append({
                    "sec": round(frames_seen * step, 2),
                    "track_id": tid,
                    "zoom_hits_in_window": len(hits),
                    "best_conf_in_window": round(max((c for _, c in hits), default=0.0), 2),
                    "since_phone_seen": round(now - ke._phone_seen.get(tid, 0.0), 2),
                })
        for v in results:
            rule = v["rule"]
            # keep the box: without it a fire is a number, and telling a real
            # detection from a false one means looking at who was flagged.
            fires.append({"frame": frames_seen, "sec": round(frames_seen * step, 2),
                          "rule": rule, "track_id": v["track_id"],
                          "bbox": [round(float(c)) for c in v["bbox"]]})
            per_rule[rule] += 1
            per_rule_tracks[rule].add(v["track_id"])
        if args.progress and frames_seen % args.progress == 0:
            print(f"[replay]   {frames_seen}/{total} frames "
                  f"({frames_seen * step:.0f}s video)", flush=True)
    cap.release()

    secs = frames_seen * step
    print(f"\n[replay] {frames_seen} frames = {secs:.0f}s of video")
    print(f"{'rule':<8} {'frames firing':>13} {'people':>8} {'per video-minute':>17}")
    for rule, n in per_rule.most_common():
        print(f"{rule:<8} {n:>13} {len(per_rule_tracks[rule]):>8} "
              f"{n / (secs / 60.0):>17.1f}")
    if not per_rule:
        print("(nothing fired)")
    if args.why:
        from collections import Counter as _C
        print("")
        print("=== why did KE fire? (%d firing frames) ===" % len(why))
        if why:
            ages = [w["since_phone_seen"] for w in why if w["since_phone_seen"] < 1e6]
            if ages:
                import statistics
                print("  seconds since the zoom last found a phone : median %.2f, max %.2f"
                      % (statistics.median(ages), max(ages)))
            print("  zoom hits in the window : %s" % dict(_C(w["zoom_hits_in_window"] for w in why)))
            print("  best phone conf in window: %s"
                  % sorted({w["best_conf_in_window"] for w in why})[-5:])
    return {"source": path, "frames": frames_seen, "video_sec": round(secs, 1),
            "why": why,
            "per_rule": dict(per_rule),
            "per_rule_people": {k: len(v) for k, v in per_rule_tracks.items()},
            "fires": fires}


def diff(a_path, b_path):
    a = json.load(open(a_path, encoding="utf-8"))
    b = json.load(open(b_path, encoding="utf-8"))
    if a["source"] != b["source"]:
        print(f"WARNING: different footage ({a['source']} vs {b['source']}) - "
              f"these numbers are not comparable.")
    print(f"same footage: {a['source']}  ({a['video_sec']}s)\n")
    print(f"{'rule':<8} {'A':>8} {'B':>8} {'change':>10}")
    for rule in sorted(set(a["per_rule"]) | set(b["per_rule"])):
        x, y = a["per_rule"].get(rule, 0), b["per_rule"].get(rule, 0)
        print(f"{rule:<8} {x:>8} {y:>8} {y - x:>+10}")
    aset = {(f["rule"], f["track_id"]) for f in a["fires"]}
    bset = {(f["rule"], f["track_id"]) for f in b["fires"]}
    gone, new = sorted(aset - bset), sorted(bset - aset)
    print(f"\nrule/person pairs that STOPPED firing ({len(gone)}): {gone[:12]}")
    print(f"rule/person pairs that STARTED firing ({len(new)}): {new[:12]}")


def main():
    ap = argparse.ArgumentParser(description="Replay footage through the real rules.")
    ap.add_argument("video", nargs="?", help="recorded mp4 (see tools/record.py)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--set", action="append", dest="sets", metavar="KEY=VALUE",
                    help="override config (rules.KE.phone_conf=0.5) or a ke.py "
                         "constant (ke.KE_LATCH_SEC=1.0). Repeatable.")
    ap.add_argument("--json", default=None, help="write the result for --diff")
    ap.add_argument("--progress", type=int, default=200, help="0 to silence")
    ap.add_argument("--why", action="store_true",
                    help="for every KE fire, record the evidence behind it - how "
                         "many zoomed-hand phone sightings were in the window, "
                         "their best confidence, and how long ago the phone was "
                         "last found. Answers 'why did this fire' without "
                         "guessing at thresholds.")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N frames - for checking a change quickly "
                         "before committing to the whole recording")
    ap.add_argument("--diff", nargs=2, metavar=("A.json", "B.json"))
    args = ap.parse_args()

    if args.diff:
        diff(*args.diff)
        return
    if not args.video:
        ap.error("give a video, or --diff A.json B.json")

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    # Replay must not touch the evidence the live app is writing.
    cfg.setdefault("clips", {})["enabled"] = False
    cfg.setdefault("alerts", {})["save_snapshots"] = False
    cfg["alerts"]["log_file"] = None
    cfg.setdefault("output", {})["show_window"] = False

    for note in apply_overrides(cfg, args.sets):
        print("[replay] override:", note)

    res = run(args.video, cfg, args)
    if args.json:
        json.dump(res, open(args.json, "w", encoding="utf-8"), indent=1)
        print(f"\n[replay] wrote {args.json}")


if __name__ == "__main__":
    main()
