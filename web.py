"""Poketenashi Safety Monitor — browser edition.

Same pipeline as main.py (YOLO11 pose + tracking -> rule engine -> alerts), but
instead of an OpenCV desktop window the annotated frame is published to a browser
dashboard: live view, violation gallery, and the per-rule charts.

    python web.py                    # camera from config.yaml, opens on :8080
    python web.py --port 9000
    python web.py --source clip.mp4  # replay a recording
    python web.py --no-browser       # don't launch a browser (servers, the Pi)

The inference loop stays on the main thread and the HTTP server runs on daemon
threads, so Ctrl+C stops everything the way it always did.
"""
import argparse
import json
import os
import socket
import sys
import threading
from collections import deque
import time
import webbrowser

# Before cv2 / ultralytics / mediapipe load: from here on everything the process
# prints - Python or native - goes through one filter into logs/monitor.log.
from poketenashi import logsetup
LOG_PATH = logsetup.install(os.environ.get("POKETENASHI_LOG_DIR")
                            or os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))

# How often a second supervised copy re-checks the port while it stands by.
STANDBY_POLL_SEC = 30.0


def _stopped_on_purpose():
    """True for 10 minutes after a copy was stopped with watchdog.stop (the mark
    is STOPPED_MARK below - defined later, this runs before it exists)."""
    mark = os.path.join(os.path.dirname(os.path.abspath(__file__)), "watchdog.stopped")
    try:
        return time.time() - os.path.getmtime(mark) < 600
    except OSError:
        return False


def _standby_while_port_taken():
    """A second SUPERVISED copy waits here, cheaply, instead of crash-looping.

    Two supervisors end up running when run.bat is started while the monitor is
    already up: watchdog.bat frees the port at start by killing whatever holds
    it, so the new copy takes over and the old supervisor's copy is left without
    a port. That copy used to exit at once and be restarted every ~10 s - 94
    restarts in 20 minutes on 2026-09-26, each one importing the whole stack.
    Now it re-checks the port every STANDBY_POLL_SEC, before cv2 and the models
    load (a few MB instead of ~1 GB), and takes over if the running copy stops -
    unless that copy was stopped on purpose with watchdog.stop.
    Unsupervised runs are untouched: main() still fails fast on a taken port.
    """
    if os.environ.get("POKETENASHI_SUPERVISED") != "1":
        return
    import yaml as _yaml
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    a, _ = p.parse_known_args()
    try:
        with open(a.config, "r", encoding="utf-8") as f:
            web_cfg = (_yaml.safe_load(f) or {}).get("web") or {}
        host = a.host or web_cfg.get("host", "0.0.0.0")
        port = a.port or int(web_cfg.get("port", 8080))
    except Exception:
        return                              # main() reports a bad config properly
    waited = False
    while True:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind((host if host != "0.0.0.0" else "", port))
            free = True
        except OSError:
            free = False
        finally:
            probe.close()
        if free:
            if waited and _stopped_on_purpose():
                print("[standby] the running copy was stopped on purpose (watchdog.stop) "
                      "- stopping this one too", flush=True)
                sys.exit(0)
            if waited:
                print(f"[standby] port {port} is free again - taking over", flush=True)
            return
        if not waited:
            print(f"[standby] port {port} is held by another running copy of the monitor: "
                  f"two supervisors (watchdog.bat / run.bat windows) are open. This one "
                  f"stands by - it checks every {STANDBY_POLL_SEC:.0f}s and takes over if "
                  f"the other stops. Close one of the two windows to keep just one.",
                  flush=True)
            waited = True
        time.sleep(STANDBY_POLL_SEC)


_standby_while_port_taken()

import copy

import cv2
import yaml

from poketenashi.camera import VideoStream, mask_url
from poketenashi.detector import Detector
from poketenashi.zones import Zones
from poketenashi.rules import RuleEngine
from poketenashi.alerts import AlertManager
from poketenashi import alerts as alertmod
from poketenashi import clips as clipmod, enhance, retention, visualizer, webserver
from poketenashi.sampler import Sampler
from poketenashi.camstore import CameraStore
from poketenashi.auth import Users, Sessions, Displays

# Face ID (poketenashi/faceid.py): one shared instance for every camera, or
# None when face_id.enabled is off. Set in main().
FACEID = None
LATE_NAMES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "alert_names.jsonl")  # late face names


def _box_iou(a, b):
    ix = max(0.0, min(float(a[2]), b[2]) - max(float(a[0]), b[0]))
    iy = max(0.0, min(float(a[3]), b[3]) - max(float(a[1]), b[1]))
    inter = ix * iy
    u = (float(a[2]) - float(a[0])) * (float(a[3]) - float(a[1])) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def add_late_name(v, who, late_sec):
    """One line per alert that got its face-ID name late (key = the alert's poster /
    clip / snapshot file, else time + camera + track + rule)."""
    rec = {"time": v.get("time"), "camera": v.get("camera"), "track_id": v.get("track_id"),
           "rule": v.get("rule"), "poster": v.get("poster"), "snapshot": v.get("snapshot"),
           "person": who, "late_sec": round(float(late_sec), 1)}
    try:
        with open(LATE_NAMES, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[face] alert {v.get('rule')} ID{v.get('track_id')} {v.get('time')}: named {who} "
              f"{late_sec:.1f} s after the alert", flush=True)
    except Exception as e:
        print(f"[face] late name not saved: {e}", flush=True)

JPEG_QUALITY = 72          # fallback; config web.jpeg_quality wins
# No NEW frame from a camera for this long = the camera is down, not just slow.
# The capture thread gives up on a dead stream after 10 s and reconnects.
STALE_SEC = 10.0
# The main loop not coming round for this long = hung, not slow. Configurable
# as web.hang_restart_sec.
HANG_SEC = 120.0


def _hang_guard(beat, limit_sec, supervised):
    """Exit when the main loop stops coming round, so the supervisor restarts it.

    A hung pipeline is worse than a crashed one: the dashboard keeps showing
    the last frame and the last "LIVE" state, and nothing restarts it. Exiting
    with code 3 lets watchdog.bat bring it back.
    """
    reported = False
    while True:
        time.sleep(5.0)
        stall = time.monotonic() - beat[0]
        if stall < limit_sec:
            reported = False
            continue
        if reported:
            continue
        logsetup.dump_stacks()
        if supervised:
            logsetup.log_direct(f"[watchdog] main loop stalled for {stall:.0f}s - stacks in "
                                f"logs/fatal.log; exiting (code 3) so watchdog.bat restarts it")
            time.sleep(1.0)
            os._exit(3)
        logsetup.log_direct(f"[watchdog] main loop stalled for {stall:.0f}s - stacks in "
                            f"logs/fatal.log. Not supervised, so nothing will restart it: "
                            f"start the monitor with watchdog.bat")
        reported = True


# Pictures a second sent to the dashboard (and sampled into clips, up to the
# clip fps). Configurable as web.view_fps.
VIEW_FPS = 15.0

# Drop one of these next to web.py to stop or restart the monitor CLEANLY.
# watchdog.bat restarts on exit code 4 and stops on 0.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
STOP_FILE = os.path.join(ROOT_DIR, "watchdog.stop")
RESTART_FILE = os.path.join(ROOT_DIR, "restart.request")
# Left behind by a watchdog.stop, so a standby copy does not take the freed port.
STOPPED_MARK = os.path.join(ROOT_DIR, "watchdog.stopped")
EXIT_RESTART = 4

# 10 Oct (user: "the person moves but the KE box moves late - it lands on empty
# space, not on the person"). The newest boxes belong to the frame detection
# last finished, up to a few seconds old when detection is slow, so the live
# view moves each box on along its person's walk (speed between the last two
# detections), at most LEAD_MAX_SEC ahead and one box width away. Clips and
# posters are exact instead (poketenashi/clips.py). output.box_lead: false = off.
LEAD_MAX_SEC = 1.5
LEAD_MAX_GAP = 3.0        # detections further apart than this give no speed


class LiveView(threading.Thread):
    """Paint a camera's NEWEST frame with its newest detections, ~15 times a second.

    The live picture comes straight from the camera, so it stays smooth however
    slowly detection runs; the boxes are moved on along each person's walk until
    the next detection (see LEAD_MAX_SEC). A camera without detection shows its
    plain picture.
    """

    def __init__(self, stream, cam_state, cfg, zones, recorder, fps=VIEW_FPS,
                 stream_w=1280, jpeg_q=JPEG_QUALITY):
        super().__init__(daemon=True, name=f"live-view-{cam_state.id}")
        self.stream, self.state, self.cfg = stream, cam_state, cfg
        self.zones, self.recorder = zones, recorder
        cc = (cfg.get("output") or {}).get("clock_cover", {"enabled": True})
        self.clock = None
        if cc and cc.get("enabled", True):
            from poketenashi.clockfix import ClockCover
            self.clock = ClockCover(dict(cc, box=cfg.get("clock_box")))
        self.period = 1.0 / max(1.0, float(fps))
        self.stream_w, self.jpeg_q = int(stream_w), int(jpeg_q)
        self.overlay = None
        self.stopped = False
        self._err_t = 0.0
        self._err_n = 0
        recorder.draw = self._draw_clip      # clip boxes are drawn when a clip is encoded

    def set_overlay(self, ov):
        """`ov` carries the frame it was judged on ("frame", "seq", "t"): its
        posters are drawn on that frame, and each person's speed since the last
        detection moves the live box on."""
        prev, vel = self.overlay, {}
        if prev is not None and ov.get("t") and prev.get("t") and 0 < ov["t"] - prev["t"] <= LEAD_MAX_GAP:
            dt = ov["t"] - prev["t"]
            before = {p.track_id: p.bbox for p in prev["persons"] if getattr(p, "tracked", True)}
            for p in ov["persons"]:
                b = before.get(p.track_id)
                if b is not None and getattr(p, "tracked", True):
                    vel[p.track_id] = ((p.bbox[0] + p.bbox[2] - b[0] - b[2]) / 2 / dt,
                                       (p.bbox[1] + p.bbox[3] - b[1] - b[3]) / 2 / dt)
        ov["vel"] = vel
        self.overlay = ov                    # one reference assignment: atomic

    def _led(self, ov):
        """The persons of `ov` moved on to now (copies; the rules keep the originals)."""
        if not ov.get("vel") or not ov.get("t") or not (self.cfg.get("output") or {}).get("box_lead", True):
            return ov["persons"]
        dt = min(time.time() - ov["t"], LEAD_MAX_SEC)
        if dt <= 0.05:
            return ov["persons"]
        out = []
        for p in ov["persons"]:
            v = ov["vel"].get(p.track_id)
            if v is None:
                out.append(p)
                continue
            dx, dy = v[0] * dt, v[1] * dt
            lim = max(1.0, p.bbox[2] - p.bbox[0])           # at most one box width
            n = (dx * dx + dy * dy) ** 0.5
            if n > lim:
                dx, dy = dx * lim / n, dy * lim / n
            q = copy.copy(p)
            q.bbox = p.bbox + (dx, dy, dx, dy)
            kp = getattr(p, "keypoints", None)
            if kp is not None and len(kp):
                q.keypoints = kp.copy()
                q.keypoints[:, 0] += dx
                q.keypoints[:, 1] += dy
            out.append(q)
        return out

    def _clip_cfg(self):
        clip_cfg = dict(self.cfg)
        clip_cfg["output"] = dict(self.cfg.get("output") or {})
        clip_cfg["output"]["draw_keypoints"] = False
        clip_cfg["output"]["draw_hand_skeleton"] = False
        return clip_cfg

    def _draw_clip(self, img, persons, phones, violations, names, fps):
        """Evidence overlay: boxes and labels only (clip encoder thread and posters)."""
        return visualizer.draw(img, persons, phones, violations, self.zones, self._clip_cfg(), fps, names=names)

    def _prepare(self, frame):
        shown = enhance.apply(frame, self.cfg.get("output") or {})
        # PC time over the camera's own clock (8 Oct; poketenashi/clockfix.py)
        if self.clock is not None:
            shown = self.clock.apply(shown, raw=frame)
        return shown

    def run(self):
        last_seq = 0
        times = deque()
        while not self.stopped:
            t0 = time.time()
            try:
                seq = self._render(last_seq)
                if seq != last_seq:
                    last_seq = seq
                    times.append(t0)
                    while times and t0 - times[0] > 3.0:
                        times.popleft()
                    if len(times) > 1:
                        self.state.view_fps = (len(times) - 1) / (times[-1] - times[0])
            except Exception as exc:         # a drawing bug must not kill the view
                self._err_n += 1
                if t0 - self._err_t > 60.0:
                    print(f"[view] {self.state.name}: render error ({self._err_n} in the last "
                          f"minute): {type(exc).__name__}: {exc}", flush=True)
                    self._err_t, self._err_n = t0, 0
                time.sleep(0.5)
            time.sleep(max(0.005, self.period - (time.time() - t0)))

    def _render(self, last_seq):
        frame, seq = self.stream.read_new(last_seq)
        if frame is None:
            return last_seq
        self.state.size = (int(frame.shape[1]), int(frame.shape[0]))
        # Posters first, then the overlay: an alert asks for its poster after its
        # overlay is set, so the overlay read below is at least that one.
        posters = self.recorder.take_posters()
        ov = self.overlay
        if posters:
            # the frame the alert was judged on, with its own boxes: exact. A copy -
            # that frame is shared with face ID and drawing paints in place.
            src = ov.get("frame") if ov is not None else None
            base = self._prepare(src.copy() if src is not None else frame.copy())
            pic = self._draw_clip(base, ov["persons"], ov["phones"], ov["violations"], ov.get("names"), ov["fps"]) if ov else base
            for name in posters:
                self.recorder.write_poster(name, pic)
        shown = self._prepare(frame)
        # visualizer.draw() paints IN PLACE: take the clip's clean copy first.
        clip_base = shown.copy() if ov is not None and self.recorder.due() else None
        if ov is not None:
            self.recorder.note(ov)
            vis = visualizer.draw(shown, self._led(ov), ov["phones"], ov["violations"],
                                  self.zones, self.cfg, ov["fps"], po_dbg=ov["po_dbg"],
                                  wrist_trail=ov["trail"], hand_skeletons=ov["hands"],
                                  names=ov.get("names"))
        else:
            vis = shown
        if self.stream_w and vis.shape[1] > self.stream_w:
            sc = self.stream_w / float(vis.shape[1])
            vis = cv2.resize(vis, (self.stream_w, int(vis.shape[0] * sc)),
                             interpolation=cv2.INTER_AREA)
        ok_enc, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_q])
        if ok_enc:
            self.state.publish(buf.tobytes(), ov["fps"] if ov else 0.0,
                               len(ov["persons"]) if ov else 0)
        # Evidence gets its OWN render (boxes and labels only), drawn when the
        # clip is encoded so every box can be placed for its own frame.
        if clip_base is not None:
            if ov.get("seq") is not None:
                self.recorder.feed(clip_base, seq, ov)
            else:
                self.recorder.feed(self._draw_clip(clip_base, ov["persons"], ov["phones"],
                                                   ov["violations"], ov.get("names"), ov["fps"]))
        return seq


def _merged(base, over):
    """base with over laid on top, nested dicts merged (copies, base untouched)."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merged(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


class CameraRuntime:
    """Everything one camera needs: its stream and live view, its own zones and
    tracker and rules (people are tracked per camera), its alerts and clips.
    The heavy models behind the rules (phone, pocket, hands) are shared."""

    def __init__(self, cam, cfg, state, web, primary):
        self.id = cam["id"]
        self.name = cam.get("name") or cam["id"]
        self.detect = bool(cam.get("detect", True))
        self.cam_rules = copy.deepcopy(cam.get("rules") or {})    # its own rule settings (rebuild on change)
        c = dict(cfg)                         # shallow: rules / output / models shared,
        c["zones"] = copy.deepcopy(cam.get("zones") or {})   # so overlay toggles reach all
        c["camera"] = dict(cfg.get("camera") or {}, source=cam["source"])
        c["clock_box"] = cam.get("clock_box")                # where this camera prints its clock
        # Per-camera rule settings (monitor_cameras.json "rules"), laid over
        # config.yaml's: the Material Gate needs NA on two zones with their own
        # geometry and SHI judged on leaving its checkpoints (7 Oct). A camera
        # without them shares config.yaml's rules object as before.
        if cam.get("rules"):
            c["rules"] = _merged(cfg.get("rules") or {}, cam["rules"])
        # Per-camera detector settings (monitor_cameras.json "models"): the Material
        # Gate runs the pose at 640 (7 Oct) - at 480 a person standing on a pad
        # against the signboard dropped below conf 0.35 and vanished.
        if cam.get("models"):
            c["models"] = _merged(cfg.get("models") or {}, cam["models"])
        if not primary:
            c["samples"] = {"enabled": False}  # training samples: the first camera only
        self.cfg = c
        self.zones = Zones(c["zones"])
        self.state = state.cam(self.id, self.name, self.detect)
        self.web_state, self._unnamed = state, []      # alerts still waiting for a face name
        cl = cfg.get("clips") or {}
        self.recorder = clipmod.ClipRecorder(
            out_dir=cl.get("dir", "clips"), pre_sec=cl.get("pre_sec", 4.0),
            post_sec=cl.get("post_sec", 4.0), fps=int(cl.get("fps", clipmod.DEFAULT_FPS)),
            enabled=bool(cl.get("enabled", True)) and self.detect,
            max_width=int(cl.get("max_width", clipmod.MAX_WIDTH)),
            max_sec=float(cl.get("max_sec", clipmod.MAX_SEC)))
        print(f"[init] {self.name} ({self.id}): opening {mask_url(cam['source'])}"
              f"{'' if self.detect else ' - live view only, no rules'}", flush=True)
        self.stream = VideoStream(cam["source"], c["camera"].get("rtsp_transport", "tcp"),
                                  c["camera"].get("reconnect_delay", 3.0),
                                  max_width=c["camera"].get("max_width", 0)).start()
        self.state.raw_source = self.stream.read
        self.view = LiveView(self.stream, self.state, c, self.zones, self.recorder,
                             fps=float(web.get("view_fps", VIEW_FPS)),
                             stream_w=int(web.get("stream_max_width", 1280)),
                             jpeg_q=int(web.get("jpeg_quality", JPEG_QUALITY)))
        self.view.start()
        self.detector = self.rules = self.alerts = self.sampler = None
        # Company dress check - only a camera with "dress_code" in
        # monitor_cameras.json (8 Oct: the Material Gate; guards, drivers and
        # visitors there wear their own clothes). See poketenashi/dress.py.
        self.dress = None
        if cam.get("dress_code"):
            from poketenashi.dress import DressCode
            self.dress = DressCode(cam["dress_code"])
        # People at a truck (getting in / out, around it) whom face ID has not
        # identified are not judged (8 Oct, Material Gate): poketenashi/vehicles.py
        self.trucks = None
        if cam.get("truck_exempt"):
            from poketenashi.vehicles import TruckNear
            self.trucks = TruckNear(cam["truck_exempt"], ROOT_DIR)
        # PO candidates for review (config.yaml po_candidates): every hand at the
        # hip, saved as images until the end time - poketenashi/po_cand.py
        self.po_saver = None
        pc = cfg.get("po_candidates") or {}
        if pc.get("enabled") and self.detect:
            from poketenashi.po_cand import POCandidates
            self.po_saver = POCandidates(pc, self.id, self.name)
        # 10 Oct (user): every person seen -> one snapshot + a row (persons.csv);
        # tools/person_clips.py cuts their clip from the raw recording
        self.person_log = None
        pl = cfg.get("person_log") or {}
        if pl.get("enabled") and self.detect:
            from poketenashi.person_log import PersonLog
            self.person_log = PersonLog(pl, self.id, self.name)
        # 13 Oct (user): PO check - anyone whose hand is hidden at the hip, snapshot for
        # review whatever the model says (poketenashi/po_watch.py; shadow, no alerts)
        self.po_watch = None
        self._po_watch_cfg = cfg.get("po_check") or {}
        # PPE in shadow mode (config.yaml ppe_shadow): cap / shoes judged and
        # logged with crops, NO alerts - poketenashi/ppe_shadow.py
        self.ppe = None
        ps = cfg.get("ppe_shadow") or {}
        if ps.get("enabled") and self.detect:
            try:
                from poketenashi.ppe_shadow import PPEShadow
                self.ppe = PPEShadow(ps, ROOT_DIR, self.id, self.name)
            except Exception as e:
                print(f"[ppe] shadow mode off for {self.name}: {e}", flush=True)
        if self.detect:
            self.detector = Detector(c)       # own tracker: ids are per camera
            self.rules = RuleEngine(c, self.zones)
            self.alerts = AlertManager(c, clip_recorder=self.recorder, camera=(self.id, self.name))
            self.sampler = Sampler(c) if primary else None
        self.last_seq = 0
        self.frame_times = deque()
        self.fps = 0.0
        self.down_since = None
        self._err_t = 0.0

    def rename(self, name):
        self.name = name
        self.state.name = name
        if self.alerts is not None:
            self.alerts.cam_name = name

    def apply_zones(self, zones):
        self.cfg["zones"] = zones
        self.zones.update(zones)
        if self.rules is not None:
            self.rules.refresh_zones()
        print(f"[zones] {self.name}: applied to the running monitor: "
              + ", ".join(sorted(self.zones.polys)), flush=True)

    def _check_down(self):
        age = self.stream.frame_age()
        if age > STALE_SEC:
            if self.down_since is None:
                self.down_since = time.time() - age
                if self.stream.status is None:
                    print(f"[camera] {self.name}: no new frame for {age:.0f}s - waiting for the "
                          f"stream", flush=True)
            self.state.camera_lost(self.stream.status or f"no new frame from the camera for {age:.0f}s")
        elif self.down_since is not None and not self.detect:
            print(f"[camera] {self.name}: live again after {time.time() - self.down_since:.0f}s",
                  flush=True)
            self.down_since = None

    def step(self, push_alert):
        """Judge this camera's newest frame, if a new one arrived. True if it did work."""
        self.state.gray_pct = self.stream.gray_pct      # broken frames skipped (dashboard)
        if not self.detect:
            self._check_down()
            return False
        frame, seq = self.stream.read_new(self.last_seq)
        if frame is None:
            self._check_down()
            return False
        t_frame = time.time()
        self.last_seq = seq
        if self.down_since is not None:
            print(f"[camera] {self.name}: live again after {time.time() - self.down_since:.0f}s",
                  flush=True)
            self.down_since = None
        persons, phones = self.detector(frame)
        # 10 Oct (user): fixed things the pose model takes for a person (the PPE
        # cartoon standee at the Help-desk) are not people - no box, no rule
        nb = (self.cfg.get("not_persons") or {}).get(self.id)
        if nb:
            persons = [p for p in persons if not any(_box_iou(p.bbox, b) >= 0.5 for b in nb)]
        judged = self.dress.judged(frame, persons) if self.dress is not None else persons
        if self.rules is not None:
            self.rules.po_cand = {}
        if self.ppe is not None:
            try:
                self.ppe.observe(frame, persons)
            except Exception as e:
                print(f"[ppe] {self.name}: {e}", flush=True)
                self.ppe = None
        if self.trucks is not None:
            near = self.trucks.update(frame, persons)
            known = FACEID.names(self.id) if FACEID is not None else {}
            judged = [p for p in judged if p.track_id not in near or p.track_id in known]
        violations = self.rules.check(judged, phones, frame)
        if self.person_log is not None:
            try:
                al = {}
                for v in violations:
                    al.setdefault(v["track_id"], set()).add(v["rule"])
                self.person_log.observe(frame, persons, al)
            except Exception as e:
                print(f"[persons] {self.name}: {e}", flush=True)
                self.person_log = None
        if self._po_watch_cfg.get("enabled") and self.detect and self.rules is not None:
            try:
                if self.po_watch is None and getattr(self.rules, "hand", None) is not None:
                    from poketenashi.po_watch import POWatch
                    self.po_watch = POWatch(self._po_watch_cfg, self.id, self.name, self.rules.hand)
                if self.po_watch is not None:
                    self.po_watch.observe(frame, judged, {v["track_id"] for v in violations if v["rule"] == "PO"})
            except Exception as e:
                print(f"[po-check] {self.name}: {e}", flush=True)
                self._po_watch_cfg = {}
        if self.po_saver is not None and self.rules is not None and self.rules.po_cand:
            try:
                self.po_saver.observe(frame, self.rules.po_cand, {v["track_id"] for v in violations if v["rule"] == "PO"})
            except Exception as e:
                print(f"[po-cand] {self.name}: {e}", flush=True)
                self.po_saver = None
        # Face ID: hand the faces to its own thread (never waits), read the
        # names it has settled on; an alert row says who broke the rule.
        names = {}
        if FACEID is not None:
            FACEID.observe(self.id, frame, persons)
            names = FACEID.names(self.id)
            # 9 Oct (user): name first, in the alert itself - unnamed rule breakers
            # get a face look on this very frame before the alert is written
            unnamed = {v["track_id"] for v in violations if v["track_id"] not in names}
            if unnamed:
                try:
                    names.update(FACEID.look_now(self.id, frame, persons, unnamed))
                except Exception as e:
                    print(f"[face] priority look failed: {e}", flush=True)
            for v in violations:
                if v["track_id"] in names:
                    v["person"] = names[v["track_id"]]
            # 8 Oct 2026 (user): the guards are allowed in the Material Gate's
            # truck lane - a rule's exempt_codes (face ID codes, e.g. GUARD1, or
            # a prefix ending in '*') drop that rule for a recognised person.
            ex = {r: (self.cfg["rules"].get(r) or {}).get("exempt_codes") for r in ("NA",)}
            if any(ex.values()):
                def _exempt(v):
                    codes = ex.get(v["rule"]) or []
                    who = str(names.get(v["track_id"]) or "")
                    return bool(who) and any(who == c or (c.endswith("*") and who.startswith(c[:-1])) for c in codes)
                violations = [v for v in violations if not _exempt(v)]
        if self.sampler is not None:
            self.sampler.observe(frame, persons, violations, self.rules)
        now = time.time()
        self.frame_times.append(now)
        while self.frame_times and now - self.frame_times[0] > 3.0:
            self.frame_times.popleft()
        if len(self.frame_times) > 1:
            self.fps = (len(self.frame_times) - 1) / (self.frame_times[-1] - self.frame_times[0])
        if now - getattr(self, "_fps_logged", 0.0) >= 300:      # 9 Oct: speed in the log every 5 min
            self._fps_logged = now
            print(f"[fps] {self.name}: {self.fps:.1f} fps", flush=True)
        # The live view gets a snapshot BEFORE the alerts are logged: an alert
        # asks for its poster from the next rendered frame, which must already
        # carry this violation's red box.
        out = self.cfg.get("output") or {}
        if self.rules.ke is not None:
            phones = list(phones) + self.rules.ke.owned_phone_boxes()
        self.view.set_overlay({
            "persons": persons, "phones": phones, "violations": violations, "fps": self.fps,
            "po_dbg": dict(self.rules.po_dbg) if self.rules.debug else None,
            "trail": ({k: list(v) for k, v in self.rules.wrist_trail.items()}
                      if self.cfg["rules"].get("PO", {}).get("draw_wrist_trail", False) else None),
            "hands": self.rules.current_hands() if out.get("draw_hand_skeleton", False) else None,
            "names": names,
            # the frame these boxes belong to (exact posters, clip boxes, live lead)
            "frame": frame, "seq": seq, "t": t_frame,
        })
        self.state.inferred(self.fps, len(persons))
        # Every person breaking a rule gets their own alert, even side by side.
        for v in self.alerts.handle(violations, frame,
                                    present={p.track_id for p in persons if p.tracked}):
            push_alert(v)
            if FACEID is not None and not v.get("person"):
                self._unnamed.append((time.time(), v))
        self._late_names(names)
        self.recorder.keep_alive({(v["track_id"], v["rule"]) for v in violations})
        return True

    def _late_names(self, names):
        """9 Oct (user): an alert written without a name (face not recognised in
        time) gets it afterwards if face ID names that person within
        late_name_sec: logs/alert_names.jsonl, which the dashboard and the
        evidence archive read alongside the alert log."""
        if not self._unnamed:
            return
        keep_sec = float((self.cfg.get("face_id") or {}).get("late_name_sec", 15.0))
        now, left = time.time(), []
        for t, v in self._unnamed:
            who = names.get(v["track_id"])
            if who:
                add_late_name(v, who, now - t)
                try:
                    self.web_state.name_alert(v, who)
                except Exception:
                    pass
            elif now - t <= keep_sec:
                left.append((t, v))
        self._unnamed = left

    def stop(self):
        """Close this camera properly (RTSP teardown, the clip in progress saved)."""
        self.view.stopped = True
        try:
            self.stream.stop()
        finally:
            self.recorder.flush()


def _stop_later(rt):
    """Stop a camera without holding up the others (stopping waits for its
    capture thread, up to ~12 s on a dead stream)."""
    threading.Thread(target=rt.stop, daemon=True, name=f"stop-{rt.id}").start()


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser(description="Poketenashi Safety Monitor (browser)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--cameras", default=None,
                    help="camera list (default: monitor_cameras.json next to the config)")
    ap.add_argument("--users", default=None,
                    help="dashboard users (default: users.json next to the config)")
    ap.add_argument("--source", default=None,
                    help="override: run ONE camera from this rtsp url / video file / webcam index")
    ap.add_argument("--host", default=None, help="bind address (default 0.0.0.0)")
    ap.add_argument("--port", type=int, default=None, help="HTTP port (default 8080)")
    ap.add_argument("--no-browser", action="store_true", help="don't open a browser")
    ap.add_argument("--debug", action="store_true", help="overlay PO tuning numbers")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    if args.debug:
        cfg.setdefault("output", {})["debug"] = True
    web = cfg.get("web") or {}
    host = args.host or web.get("host", "0.0.0.0")
    port = args.port or int(web.get("port", 8080))
    cfg_dir = os.path.dirname(os.path.abspath(args.config))

    # Claim the port BEFORE loading ~1 GB of models.
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((host if host != "0.0.0.0" else "", port))
    except OSError:
        print(f"[error] port {port} is already in use - another copy of the monitor "
              f"is running. Close it, or start this one with --port.")
        sys.exit(1)
    finally:
        probe.close()

    cams = CameraStore(args.cameras or os.path.join(cfg_dir, "monitor_cameras.json"), cfg)
    if args.source is not None:              # one-off replay: not saved to the camera list
        cams.cams = [{"id": "cam1", "name": "Replay", "source": args.source, "enabled": True,
                      "detect": True, "zones": copy.deepcopy(cfg.get("zones") or {})}]
        cams._write = lambda: None
    users = Users(args.users or os.path.join(cfg_dir, "users.json"))
    # logins survive a restart (sessions.json: token hashes only); display
    # links let a TV screen show its cameras without a login (displays.json)
    sessions = Sessions(os.path.join(cfg_dir, "sessions.json"))
    displays = Displays(os.path.join(cfg_dir, "displays.json"))

    snap_dir = cfg["alerts"].get("snapshot_dir", "snapshots")
    clip_dir = (cfg.get("clips") or {}).get("dir", "clips")
    os.makedirs(clip_dir, exist_ok=True)
    ret = cfg.get("retention") or {}
    retention.start(days=float(ret.get("days", 2)),
                    dirs=[clip_dir, snap_dir],
                    log_path=cfg["alerts"].get("log_file", "violations.jsonl"),
                    trim_log=bool(ret.get("trim_log", True)),
                    sweep_min=float(ret.get("sweep_min", retention.SWEEP_MIN_DEFAULT)),
                    log_lock=alertmod.LOG_LOCK)
    # Every alert kept for good, beyond the retention (poketenashi/archive.py).
    arc = cfg.get("evidence_archive") or {}
    if arc.get("enabled") and cfg["alerts"].get("log_file"):
        from poketenashi.archive import Archiver
        Archiver(cfg["alerts"]["log_file"], clip_dir, arc.get("dir", "D:/poketenashi-data/evidence"),
                 every_sec=float(arc.get("every_sec", 60))).start()

    state = webserver.MonitorState()
    try:
        webserver.serve(state, cfg,
                        log_path=cfg["alerts"].get("log_file", "violations.jsonl"),
                        snap_dir=snap_dir, clip_dir=clip_dir, host=host, port=port,
                        config_path=args.config, cams=cams, users=users, sessions=sessions,
                        displays=displays,
                        audit_path=os.path.join(os.path.dirname(LOG_PATH), "audit.jsonl"))
    except OSError:
        print(f"[error] port {port} is already in use - another copy of the "
              f"monitor is running. Close it, or start this one with --port.")
        sys.exit(1)
    try:
        os.remove(STOPPED_MARK)        # this copy owns the port: an old stop is past
    except OSError:
        pass
    url = f"http://localhost:{port}"
    print(f"[web]  dashboard on {url}")
    print(f"[log]  full log: {LOG_PATH}")
    if not users.any():
        print("[web]  no dashboard users yet - open the dashboard on this PC to create the admin",
              flush=True)

    print("[init] loading models (first run downloads weights) ...")
    global FACEID
    if (cfg.get("face_id") or {}).get("enabled", False):
        from poketenashi.faceid import FaceID
        fid = FaceID(cfg["face_id"], cfg_dir)
        FACEID = fid if fid.ok else None
    runtimes = {}
    first = True
    for cam in cams.all():
        if not cam.get("enabled", True):
            state.cam(cam["id"], cam.get("name"), cam.get("detect", True)).camera_lost("switched off")
            continue
        runtimes[cam["id"]] = CameraRuntime(cam, cfg, state, web, primary=first)
        first = False
    primary_id = next(iter(runtimes), None)

    print("[init] waiting for first frame ...")
    deadline = time.time() + 15.0
    while time.time() < deadline and any(rt.stream.frame_age() == float("inf")
                                         for rt in runtimes.values()):
        time.sleep(0.1)
    for rt in runtimes.values():
        if rt.stream.frame_age() == float("inf"):
            why = rt.stream.status or "no frames received yet"
            rt.state.camera_lost(why)
            print(f"[camera] {rt.name}: no picture yet: {why}. Monitoring starts by itself as "
                  f"soon as the camera answers.", flush=True)
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    supervised = os.environ.get("POKETENASHI_SUPERVISED") == "1"
    beat = [time.monotonic()]
    threading.Thread(target=_hang_guard,
                     args=(beat, float(web.get("hang_restart_sec", HANG_SEC)), supervised),
                     daemon=True, name="hang-guard").start()
    print("[run] monitoring started on %d camera(s)%s. Ctrl+C to stop."
          % (len(runtimes), " (supervised by watchdog.bat)" if supervised else ""))

    exit_code = 0
    next_flag_check = 0.0
    err_t = {}
    try:
        while True:
            beat[0] = time.monotonic()
            if time.time() >= next_flag_check:
                next_flag_check = time.time() + 2.0
                if os.path.exists(RESTART_FILE):
                    os.remove(RESTART_FILE)
                    print("[run] restart requested (restart.request) - closing cleanly", flush=True)
                    exit_code = EXIT_RESTART
                    break
                if os.path.exists(STOP_FILE):
                    os.remove(STOP_FILE)
                    try:                  # a standby copy must not take the port over
                        with open(STOPPED_MARK, "w") as f:
                            f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
                    except OSError:
                        pass
                    print("[run] stop requested (watchdog.stop) - closing cleanly", flush=True)
                    break
            # Changes from the dashboard, applied between two frames.
            for cid, zones in state.take_zones():
                rt = runtimes.get(cid)
                if rt is not None:
                    rt.apply_zones(zones)
            for op, payload in state.take_camera_ops():
                cid = payload if op == "remove" else payload["id"]
                old = runtimes.get(cid)
                if op == "update" and old is not None and payload.get("enabled", True) \
                        and payload["source"] == old.cfg["camera"]["source"] \
                        and bool(payload.get("detect", True)) == old.detect \
                        and (payload.get("rules") or {}) == old.cam_rules:
                    old.rename(payload.get("name") or cid)       # only the name changed
                    continue
                if old is not None:
                    runtimes.pop(cid)
                    _stop_later(old)
                    state.drop_cam(cid)
                    print(f"[cameras] {old.name}: stopped", flush=True)
                if op in ("add", "update"):
                    if payload.get("enabled", True):
                        try:
                            runtimes[cid] = CameraRuntime(payload, cfg, state, web,
                                                          primary=cid == (primary_id or cid))
                        except Exception as e:
                            print(f"[cameras] {payload.get('name')}: could not start: {e}", flush=True)
                    else:
                        state.cam(cid, payload.get("name"), payload.get("detect", True)) \
                            .camera_lost("switched off")
            worked = False
            for rt in list(runtimes.values()):
                try:
                    worked = rt.step(state.push_alert) or worked
                except Exception as e:            # one camera's fault must not stop the rest
                    if time.time() - err_t.get(rt.id, 0) > 60:
                        err_t[rt.id] = time.time()
                        print(f"[run] {rt.name}: frame skipped: {type(e).__name__}: {e}", flush=True)
            if not worked:
                time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n[run] interrupted by user.")
    finally:
        for rt in runtimes.values():
            try:
                rt.stop()
            except Exception:
                pass
        clipmod.wait_idle(20.0)   # let the clips in progress finish encoding
        print("[done] stopped.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
