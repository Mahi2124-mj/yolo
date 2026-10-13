"""Browser front end for the monitor - every camera, settings and users.

Serves everything off one port using nothing but the standard library:

    /                     the dashboard (cameras, live view, evidence, zones,
                          settings, users) - web/index.html + app.css/ui.js/app.js
    /tv                   a plain full-screen view for TV browsers (ES5): the
                          live picture with the violations beside it
    /zones                old address of the zone editor (now inside the dashboard)
    /api/stream?cam=ID    a camera's annotated live view, as MJPEG
    /api/frame?cam=ID     its latest annotated picture as one JPEG
    /api/rawframe?cam=ID  its latest camera picture WITHOUT overlays (zone editor)
    /api/state?cam=ID     fps, people, camera health, every camera's status - and
                          which rules are really running, with the reason for
                          each one that is not (rule_status)
    /api/violations       rows from violations.jsonl (newest first; ?rule= ?cam=)
    /api/alerts           alerts fired since a given id, for the live banners
    /api/stats            counts by rule, by camera and by hour, for the charts
    /api/cameras          the camera list (no passwords); POST add / update /
                          delete / test / discover - admin only
    /api/zones?cam=ID     GET the zones / POST new ones (saved and applied live)
    /api/settings         retention + live-picture overlays (GET, POST admin);
                          /api/settings/preview?days=N what a retention would delete
    /api/users            users (admin); /api/login /api/logout /api/me /api/setup
    /api/password         a logged-in user changes their own password
    /api/audit            who changed what, when (admin)
    /api/clientlog        a browser reports a page error (lands in monitor.log)
    /snapshots/<file>     the saved evidence images
    /clips/<file>         the evidence clips (with Range support, so they seek)

Deliberately no FastAPI/uvicorn: the PC is RAM-tight and the surface is small.
Looking is open to the LAN (the TV, phones). Changing anything - cameras, zones,
settings, users - needs an admin login; until the first user exists only the
monitor PC itself (localhost) can do it, and can create that admin. Every change
is written to the audit log.
"""
import base64
import copy
import json
import mimetypes
import re
import socket
from collections import OrderedDict, deque
import os
import posixpath
import threading
import time
from urllib.parse import urlparse, parse_qs, quote, urlsplit, urlunsplit
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .camera import mask_url
from . import retention, rtsp_probe, settings_store, zone_store, ruleinfo

APP_VERSION = "2026.10.03"

# A browser that goes away mid-response - tab closed, page refreshed, a gallery
# image or a <video> range request cancelled - raises one of these inside the
# handler. That is ordinary traffic, not a fault.
CLIENT_GONE = (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, TimeoutError)

# POKETENASHI_WEB_DIR lets a test copy serve a staged front end (web_next/)
# while the live one keeps serving web/.
WEB_DIR = os.environ.get("POKETENASHI_WEB_DIR") or \
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
BOUNDARY = "frame"
STREAM_SLEEP = 0.045          # ~22 fps to the browser, independent of inference
CLIENT_NOTE_SEC = 600         # log each other machine at most this often
COOKIE = "pk_session"
COOKIE_TV = "pk_tv"           # a display (TV screen) link: the screen's cameras, no login
DENY = object()               # _view(): an error was already sent
_CAM_FILE = re.compile(r"^\d+_(cam\d+)_")   # evidence of cameras other than cam1 names its camera
MAX_BODY = 200000

# Explicit types: Windows' registry can map .js to text/plain, and with nosniff
# a browser then refuses to run the dashboard's script.
MIME = {".html": "text/html; charset=utf-8", ".js": "application/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8", ".json": "application/json", ".svg": "image/svg+xml",
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".ico": "image/x-icon",
        ".mp4": "video/mp4", ".txt": "text/plain; charset=utf-8"}
SECURITY_HEADERS = {"X-Content-Type-Options": "nosniff", "X-Frame-Options": "SAMEORIGIN",
                    "Referrer-Policy": "same-origin"}

# Switches for the live picture. Each maps to the config key the visualizer
# already reads (a change shows on the next frame, on every camera - the
# cameras share these config dicts) and to the config.yaml section it is saved in.
OVERLAYS = {
    "keypoints": ("output", "draw_keypoints"),
    "hands":     ("output", "draw_hand_skeleton"),
    "zones":     ("output", "draw_zones"),
    "trail":     ("rules.PO", "draw_wrist_trail"),
    "enhance":   ("output", "enhance"),
    "sharpen":   ("output", "enhance_sharpen"),
}


class CamState:
    """One camera's live picture and health, between its pipeline and the web.

    The pipeline publishes one encoded JPEG per rendered frame; every connected
    browser reads the latest one - encoding once for all viewers."""

    def __init__(self, cam_id, name="", detect=True):
        self.id, self.name, self.detect = cam_id, name or cam_id, detect
        self._lock = threading.Lock()
        self._jpeg = None
        self._seq = 0
        self.fps = 0.0
        self.persons = 0
        self.camera_ok = False
        self.last_error = None
        self.started = time.time()
        self.last_publish = 0.0
        self.last_infer = 0.0
        self.view_fps = 0.0
        self.raw_source = None        # the stream's read(), for the zone editor
        self.size = None              # (w, h) of the camera picture
        self.gray_pct = 0.0           # % of the last minute's frames skipped as broken

    def raw_frame(self):
        src = self.raw_source
        if src is None:
            return None
        try:
            ok, f = src()
            return f if ok else None
        except Exception:
            return None

    def publish(self, jpeg, fps, persons):
        with self._lock:
            self._jpeg = jpeg
            self._seq += 1
            self.fps = float(fps)
            self.persons = int(persons)
            self.camera_ok = True
            self.last_error = None
            self.last_publish = time.time()

    def inferred(self, fps, persons):
        with self._lock:
            self.fps = float(fps)
            self.persons = int(persons)
            self.last_infer = time.time()

    def camera_lost(self, why):
        with self._lock:
            self.camera_ok = False
            self.last_error = why

    def frame(self):
        with self._lock:
            return self._jpeg, self._seq

    def snapshot(self):
        with self._lock:
            now = time.time()
            age = now - self.last_publish if self.last_publish else None
            iage = now - self.last_infer if self.last_infer else None
            return {
                "id": self.id, "name": self.name, "detect": self.detect,
                # A rate is only meaningful while frames are arriving.
                "fps": round(self.fps, 1) if iage is not None and iage < 3.0 else 0.0,
                "view_fps": round(self.view_fps, 1) if age is not None and age < 3.0 else 0.0,
                "persons": self.persons if iage is not None and iage < 3.0 else 0,
                "camera_ok": self.camera_ok and age is not None and age < 10.0,
                "last_error": self.last_error,
                "frame_age_sec": round(age, 1) if age is not None else None,
                "infer_age_sec": round(iage, 1) if iage is not None else None,
                "size": list(self.size) if self.size else None,
                "gray_pct": self.gray_pct,
            }


class MonitorState:
    """Every camera's state, the live alert feed, and the changes the web side
    asks the pipeline for (applied between frames, never half way)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.cams = OrderedDict()            # id -> CamState
        self.started = time.time()
        self._alerts = deque(maxlen=80)
        self._alert_seq = 0
        self._pending_zones = []             # [(cam_id, zones)]
        self._pending_cams = []              # [("add"|"update"|"remove", cam dict or id)]

    # -- cameras -----------------------------------------------------------
    def cam(self, cam_id, name=None, detect=True):
        with self._lock:
            c = self.cams.get(cam_id)
            if c is None:
                c = self.cams[cam_id] = CamState(cam_id, name, detect)
            else:
                if name:
                    c.name = name
                c.detect = detect
            return c

    def drop_cam(self, cam_id):
        with self._lock:
            self.cams.pop(cam_id, None)

    def get_cam(self, cam_id=None):
        with self._lock:
            if cam_id and cam_id in self.cams:
                return self.cams[cam_id]
            return next(iter(self.cams.values()), None)

    def all_cams(self):
        with self._lock:
            return list(self.cams.values())

    # -- alerts ------------------------------------------------------------
    def push_alert(self, v):
        with self._lock:
            self._alert_seq += 1
            self._alerts.append({
                "id": self._alert_seq, "time": v.get("time"), "rule": v.get("rule"),
                "desc": v.get("desc"), "track_id": v.get("track_id"),
                "snapshot": v.get("snapshot"), "clip": v.get("clip"), "poster": v.get("poster"),
                "camera": v.get("camera") or "cam1", "camera_name": v.get("camera_name"),
                "person": v.get("person"),
            })

    def name_alert(self, v, who):
        """A face-ID name found after the alert was pushed (web.py _late_names)."""
        key = (v.get("time"), v.get("track_id"), v.get("rule"), v.get("camera") or "cam1")
        with self._lock:
            for a in reversed(self._alerts):
                if (a.get("time"), a.get("track_id"), a.get("rule"), a.get("camera")) == key:
                    a["person"] = who
                    break

    def alerts_since(self, since):
        with self._lock:
            return [a for a in self._alerts if a["id"] > since], self._alert_seq

    # -- requests to the pipeline -------------------------------------------
    def request_zones(self, cam_id, zones):
        with self._lock:
            self._pending_zones.append((cam_id, zones))

    def take_zones(self):
        with self._lock:
            z, self._pending_zones = self._pending_zones, []
            return z

    def request_camera(self, op, payload):
        with self._lock:
            self._pending_cams.append((op, payload))

    def take_camera_ops(self):
        with self._lock:
            ops, self._pending_cams = self._pending_cams, []
            return ops


class AuditLog:
    """Who changed what, when: one JSON line per change, login and logout."""
    KEEP = 5000

    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self._n = 0

    def write(self, user, ip, action, detail=""):
        if not self.path:
            return
        rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "user": user or "-", "ip": ip,
               "action": action, "detail": " ".join(str(detail).split())[:400]}
        with self.lock:
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                self._n += 1
                if self._n % 200 == 0:
                    self._trim()
            except OSError as e:
                print(f"[audit] could not write: {e}", flush=True)

    def _trim(self):
        with open(self.path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if len(lines) > self.KEEP * 1.2:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.writelines(lines[-self.KEEP:])
            os.replace(tmp, self.path)

    def tail(self, limit=200):
        if not self.path or not os.path.exists(self.path):
            return []
        with self.lock:
            with open(self.path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - limit * 600))
                chunk = f.read().decode("utf-8", "replace")
        out = []
        for line in reversed(chunk.split("\n")):
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
            if len(out) >= limit:
                break
        return out


# ---------------------------------------------------------------- log reading

def snapshot_name(rec, snap_dir):
    """The row's snapshot filename, or None when there isn't one on disk."""
    name = rec.get("snapshot")
    if not name:
        try:
            epoch = int(time.mktime(time.strptime(rec["time"], "%Y-%m-%d %H:%M:%S")))
        except Exception:
            return None
        name = f"{epoch}_ID{rec['track_id']}_{rec['rule']}.jpg"
    return name if os.path.isfile(os.path.join(snap_dir, name)) else None


def rule_status(enabled, cam):
    """The rules this camera is really running, and why the others are not.

    `cam` is a camstore.public() entry: its `zones` are the zone names drawn on
    that camera, `enabled`/`detect` say whether it runs rules at all. The
    prerequisites themselves live in poketenashi/ruleinfo.py - the same module
    the rule engine reads, so a rule that is switched off, has no zone, or sits
    on a camera that is switched off is reported as inactive with the reason
    instead of being listed as if it were working.
    """
    if cam is None:
        blocked = "no camera"
    elif not cam.get("enabled", True):
        blocked = "this camera is switched off"
    elif not cam.get("detect", True):
        blocked = "live view only on this camera"
    else:
        zones = set(cam.get("zones") or ())
        # a zone in parts counts (checkpoint_left + checkpoint_right = checkpoint),
        # exactly as poketenashi/zones.py Zones.has() judges it for the engine
        rows = ruleinfo.status(enabled, lambda n: any(z == n or z.startswith(n + "_") for z in zones))
        own = cam.get("rules_enabled")
        if own is not None:                           # the camera's own choice (Cameras page)
            for r in rows:
                if r["rule"] not in own and (r["active"] or r["rule"] in (enabled or ())):
                    r["active"], r["reason"] = False, "not checked on this camera (Cameras page)"
        return rows
    return ruleinfo.status(enabled, lambda _n: False, blocked=blocked)


def _parse(line, snap_dir):
    try:
        rec = json.loads(line)
    except Exception:
        return None
    if "rule" not in rec or "time" not in rec:
        return None
    # what the reader sees is the CURRENT label for that rule code, not the text
    # frozen into the row when it fired: rows written before a rule was renamed
    # (NA, 6 Oct 2026) would otherwise be described one way on the TV, which
    # shows this field, and another on the dashboard, which shows the live label.
    rec["desc"] = ruleinfo.LABELS.get(rec["rule"], rec.get("desc") or rec["rule"])
    rec["snapshot"] = snapshot_name(rec, snap_dir)
    clip = rec.get("clip")
    rec["poster"] = rec.get("poster") or ((os.path.splitext(clip)[0] + ".jpg") if clip else None)
    rec.setdefault("camera", "cam1")          # rows from before there were several
    return rec


def read_tail(path, snap_dir, limit, rule=None, cam=None, allow=None):
    """The newest `limit` rows, newest first (reads only the tail of the file)."""
    if not os.path.exists(path):
        return []
    size = os.path.getsize(path)
    want = min(size, max(limit * 400, 128 * 1024) * (4 if (rule or cam) else 1))
    with open(path, "rb") as f:
        f.seek(size - want)
        chunk = f.read()
    lines = chunk.decode("utf-8", "replace").split("\n")
    if want < size:
        lines = lines[1:]
    out = []
    from . import late_names
    names = late_names.load(late_names.path_for(path))
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        rec = _parse(line, snap_dir)
        if rec is not None:
            late_names.fill(rec, names)            # a face name found after the alert
        if rec is None or (rule and rec["rule"] != rule) or (cam and rec["camera"] != cam) \
                or (allow is not None and rec["camera"] not in allow):
            continue
        out.append(rec)
        if len(out) >= limit:
            break
    return out


class StatsCache:
    """Whole-file counts, recomputed only when the log changes."""

    def __init__(self, path, snap_dir):
        self.path = path
        self.snap_dir = snap_dir
        self._key = None
        self._rows = []

    def _load(self):
        try:
            st = os.stat(self.path)
            key = (st.st_mtime_ns, st.st_size)
        except OSError:
            self._rows = []
            return
        if key == self._key:
            return
        rows = []
        with open(self.path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                rec = _parse(line.strip(), self.snap_dir)
                if rec is None:
                    continue
                try:
                    t = time.mktime(time.strptime(rec["time"], "%Y-%m-%d %H:%M:%S"))
                except Exception:
                    t = None
                rows.append((t, rec["rule"], rec["camera"], rec.get("track_id")))
        self._rows, self._key = rows, key

    def get(self, cam=None, allow=None):
        """`allow`: the camera ids a viewer may see (None = all)."""
        self._load()
        now = time.time()
        lt = time.localtime(now)
        midnight = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
        by_rule, by_cam, today_rule, tracks = {}, {}, {}, set()
        hours = [0] * 24
        total = today = last24 = yesterday = yesterday_so_far = 0
        for t, rule, camid, tid in self._rows:
            if allow is not None and camid not in allow:
                continue
            by_cam[camid] = by_cam.get(camid, 0) + 1
            if cam and camid != cam:
                continue
            total += 1
            by_rule[rule] = by_rule.get(rule, 0) + 1
            tracks.add((camid, tid))
            if t is None:
                continue
            if t >= midnight:
                today += 1
                today_rule[rule] = today_rule.get(rule, 0) + 1
            elif t >= midnight - 86400:
                yesterday += 1
                if t < now - 86400:           # yesterday, up to this time of day
                    yesterday_so_far += 1
            age = now - t
            if 0 <= age < 24 * 3600:
                last24 += 1
                hours[23 - int(age // 3600)] += 1
        start_hour = time.localtime(now - 23 * 3600).tm_hour

        def ranked(d):
            return sorted(({"rule": r, "count": c} for r, c in d.items()), key=lambda x: -x["count"])
        return {
            "total": total, "today": today, "last24": last24, "tracks": len(tracks),
            "yesterday": yesterday, "yesterday_so_far": yesterday_so_far,
            "by_rule": ranked(by_rule), "today_by_rule": ranked(today_rule),
            "by_camera": by_cam,
            "by_hour": [{"label": "%02d:00" % ((start_hour + i) % 24), "count": n}
                        for i, n in enumerate(hours)],
        }


_LAN = {"t": 0.0, "ips": []}


def _lan_addresses():
    """This PC's IPv4 addresses on the local networks (cached for a minute)."""
    if time.time() - _LAN["t"] < 60:
        return _LAN["ips"]
    ips = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith(("127.", "169.254.", "100.")) and ip not in ips:
                ips.append(ip)
    except OSError:
        pass
    _LAN.update(t=time.time(), ips=ips)
    return ips


def _dir_usage(path):
    n = size = 0
    try:
        for e in os.scandir(path):
            if e.is_file():
                n += 1
                size += e.stat().st_size
    except OSError:
        pass
    return n, size


def _test_source(source, timeout=12.0):
    """Open a camera address once and grab one picture: (jpeg bytes, w, h) or
    an error string. Runs in a worker thread so a dead address cannot hang the
    request longer than `timeout`."""
    import cv2
    out = {}

    def run():
        try:
            if source.isdigit():
                cap = cv2.VideoCapture(int(source))
            elif source.lower().startswith("rtsp"):
                os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG,
                                       [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000,
                                        cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000])
            else:
                cap = cv2.VideoCapture(source)
            ok, f = cap.read() if cap.isOpened() else (False, None)
            cap.release()
            if not ok or f is None:
                out["error"] = "the camera did not send a picture (address, password or network?)"
                return
            h, w = f.shape[:2]
            s = 640.0 / max(1, w)
            small = cv2.resize(f, (640, int(h * s))) if w > 640 else f
            okj, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
            out["jpeg"], out["w"], out["h"] = buf.tobytes(), w, h
        except Exception as e:
            out["error"] = f"could not open it: {type(e).__name__}"

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None, "no answer within %.0f s (address or network?)" % timeout
    if "error" in out:
        return None, out["error"]
    return out, None


HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]{0,252}$")


def compose_source(data, existing=None):
    """The camera address a dashboard form describes: (source, None) or (None, error).

    The form gives EITHER the full address (`source`), OR the parts - `host`,
    `port`, `user`, `password`, `path` - so nobody has to know RTSP URL syntax,
    OR (for a saved camera) just a new `path`: the saved login is kept, the
    browser never sees it. Editing the host with user and password left empty
    keeps the saved login too."""
    src = str(data.get("source") or "").strip()
    if src:
        return src, None
    host = str(data.get("host") or "").strip()
    path = str(data.get("path") or "").strip()
    if path and not path.startswith("/"):
        path = "/" + path
    if host:
        if not HOST_RE.match(host):
            return None, "the IP address is not valid"
        try:
            port = int(data.get("port") or 554)
        except (TypeError, ValueError):
            return None, "the port must be a number"
        if not 1 <= port <= 65535:
            return None, "the port must be 1-65535"
        user = str(data.get("user") or "")
        pw = str(data.get("password") or "")
        if existing and not user and not pw and existing.get("source", "").lower().startswith("rtsp"):
            userinfo = urlsplit(existing["source"]).netloc.rpartition("@")[0]
        elif user:
            userinfo = quote(user, safe="") + ((":" + quote(pw, safe="")) if pw else "")
        else:
            userinfo = ""
        p, _, q = (path or "/").partition("?")
        return urlunsplit(("rtsp", (userinfo + "@" if userinfo else "") + f"{host}:{port}", p, q, "")), None
    if path and existing:
        if not existing.get("source", "").lower().startswith("rtsp"):
            return None, "only an RTSP camera has a stream path"
        return rtsp_probe.with_path(existing["source"], path), None
    return None, "give the camera's IP address (or its full RTSP address)"


# ------------------------------------------------------------------- handler

def make_handler(state, cfg, log_path, snap_dir, clip_dir, config_path, cams, users, sessions,
                 audit, displays=None):
    stats = StatsCache(log_path, snap_dir)
    seen, seen_lock = {}, threading.Lock()

    def cfg_node(section):
        node = cfg
        for k in section.split("."):
            node = node.setdefault(k, {})
        return node

    class Handler(BaseHTTPRequestHandler):
        server_version = "Poketenashi"
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):        # the console belongs to the pipeline
            pass

        def log_error(self, fmt, *args):
            self._note("sent a bad request: " + (fmt % args)[:160], key="bad")

        def _note(self, what, key=None, remote_only=True):
            """Print `what` for this client, at most once per CLIENT_NOTE_SEC per key."""
            ip = self.client_address[0]
            if remote_only and (ip.startswith("127.") or ip == "::1"):
                return
            now = time.time()
            k = (ip, key or what)
            with seen_lock:
                if now - seen.get(k, 0) < CLIENT_NOTE_SEC:
                    return
                if len(seen) > 500:
                    for old in [x for x, t in seen.items() if now - t > CLIENT_NOTE_SEC]:
                        del seen[old]
                seen[k] = now
            print(f"[web]  client {ip} {what}", flush=True)

        def _note_visit(self):
            ua = (self.headers.get("User-Agent") or "no user-agent")[:180]
            self._note(f"{self.command} {self.path[:60]} - {ua}", key="visit")

        # -- helpers ---------------------------------------------------------
        def _send(self, code, body, ctype, extra=None):
            if isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in SECURITY_HEADERS.items():
                self.send_header(k, v)
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200, extra=None):
            h = {"Cache-Control": "no-store"}
            h.update(extra or {})
            self._send(code, json.dumps(obj), "application/json", h)

        def _file(self, root, name, cache="no-store"):
            safe = posixpath.normpath("/" + name).lstrip("/")
            full = os.path.abspath(os.path.join(root, safe))
            if not full.startswith(os.path.abspath(root) + os.sep) or not os.path.isfile(full):
                return self._send(404, "not found", "text/plain")
            ext = os.path.splitext(full)[1].lower()
            ctype = MIME.get(ext) or mimetypes.guess_type(full)[0] or "application/octet-stream"
            size = os.path.getsize(full)
            hdr = {"Cache-Control": cache, "Accept-Ranges": "bytes"}
            # Byte ranges: a <video> asks for them to seek, and some browsers
            # will not let a clip be scrubbed without them.
            m = re.match(r"^bytes=(\d*)-(\d*)$", (self.headers.get("Range") or "").strip())
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    end = min(int(m.group(2)) if m.group(2) else size - 1, size - 1)
                else:
                    start, end = max(0, size - int(m.group(2))), size - 1
                if start >= size or start > end:
                    return self._send(416, b"", "text/plain", {"Content-Range": f"bytes */{size}"})
                with open(full, "rb") as f:
                    f.seek(start)
                    body = f.read(end - start + 1)
                hdr["Content-Range"] = f"bytes {start}-{end}/{size}"
                return self._send(206, body, ctype, hdr)
            with open(full, "rb") as f:
                return self._send(200, f.read(), ctype, hdr)

        def _read_raw(self):
            """Read this POST's whole body, once, before anything answers.

            A handler that answered without reading it (a refused login check,
            /api/logout, /api/toggle) left the bytes in the kept-alive
            connection, and the browser's NEXT request on it arrived as
            "{}GET /api/..." - a 501 for an ordinary click."""
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = -1
            if n < 0 or n > MAX_BODY:
                self.close_connection = True       # unread bytes: never reuse
                return None
            return self.rfile.read(n) if n else b""

        def _body(self):
            """The POSTed JSON object, or None (an error answer was sent)."""
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            raw = self._raw
            if not raw:
                self._json({"error": "bad request"}, 400)
                return None
            # Only JSON changes anything: a form on some other web page can post
            # text/plain or form data here, but never application/json.
            if ctype != "application/json":
                self._json({"error": "send JSON"}, 415)
                return None
            try:
                data = json.loads(raw.decode("utf-8"))
            except Exception:
                self._json({"error": "not JSON"}, 400)
                return None
            if not isinstance(data, dict):
                self._json({"error": "bad request"}, 400)
                return None
            return data

        def _is_local(self):
            ip = self.client_address[0]
            return ip.startswith("127.") or ip == "::1"

        def _toggles(self):
            return {name: bool(cfg_node(section).get(key, False))
                    for name, (section, key) in OVERLAYS.items()}

        def _cam(self, q):
            """The camera a request names; the first camera only when it names
            none. A named camera that is not running (switched off, removed)
            is None - never another camera's picture: the Dx tab showed the
            Toolroom stream under its own name while Dx was switched off (6 Oct)."""
            cid = (q.get("cam") or [None])[0]
            if cid:
                cs = state.get_cam(cid)
                return cs if cs is not None and cs.id == cid else None
            return state.get_cam(None)

        def _audit(self, action, detail=""):
            audit.write(self._who(), self.client_address[0], action, detail)

        # -- who is asking ---------------------------------------------------
        def _cookie(self, name):
            raw = self.headers.get("Cookie") or ""
            for part in raw.split(";"):
                k, _, v = part.strip().partition("=")
                if k == name:
                    return v
            return None

        def _token(self):
            return self._cookie(COOKIE)

        def _display(self):
            return displays.by_token(self._cookie(COOKIE_TV)) if displays is not None else None

        def _allowed(self):
            """The cameras this request may see: None = all of them, a set =
            only those, False = none (not logged in, no display link).
            Looking needs a login since 7 Oct 2026 (the user: every person sees
            the cameras given to them; a TV screen its display link's)."""
            if not users.any():
                return None if self._is_local() else False
            u = self._user()
            if u is not None:
                if u[1] == "admin":
                    return None
                c = users.cams_of(u[0])
                return None if c is None else set(c)
            d = self._display()
            if d is not None:
                return None if "*" in d["cams"] else set(d["cams"])
            return False

        def _view(self, cam_id=None):
            """The allowed set (None = all) if this request may look - at
            `cam_id` when one is given - else an error was sent and DENY."""
            a = self._allowed()
            if a is False:
                self._json({"error": "Please log in to see the cameras.", "login": True}, 401)
                return DENY
            if cam_id and a is not None and cam_id not in a:
                self._json({"error": "This camera is not one of yours."}, 403)
                return DENY
            return a

        def _cam_id(self, q):
            """The camera a request means: its ?cam=, else the first camera."""
            cid = (q.get("cam") or [None])[0]
            if cid:
                return cid
            cs = state.get_cam(None)
            return cs.id if cs else None

        def _user(self):
            return sessions.get(self._token())

        def _admin(self):
            """True if this request may change things; otherwise an error was sent.
            Before the first user exists, only the monitor PC itself may."""
            if not users.any():
                if self._is_local():
                    return True
                self._json({"error": "No users yet: create the admin account on the monitor PC "
                                     "(open the dashboard there)."}, 403)
                return False
            u = self._user()
            if u is None:
                self._json({"error": "Please log in as an admin.", "login": True}, 401)
                return False
            if u[1] != "admin":
                self._json({"error": "Only an admin can change this."}, 403)
                return False
            return True

        def _who(self):
            return (self._user() or ("localhost" if self._is_local() else "?", ""))[0]

        # -- page error reports ---------------------------------------------
        def _clientlog(self):
            body = (self._raw or b"")[:2000]
            text = " ".join(body.decode("utf-8", "replace").split())[:600]
            if text:
                self._note("page error: " + text, key="page:" + text[:80], remote_only=False)
            self._json({"ok": True})

        # -- login -------------------------------------------------------------
        def _me(self):
            u = self._user()
            a = self._allowed()
            d = self._display() if u is None else None
            return {"user": u[0] if u else None, "role": u[1] if u else None,
                    "cams": None if a is None else sorted(a) if a is not False else [],
                    "may_view": a is not False, "display": d["name"] if d else None,
                    "setup_needed": not users.any(), "local": self._is_local(),
                    "can_admin": (u is not None and u[1] == "admin")
                    or (not users.any() and self._is_local()),
                    # where other screens (the TV) can open this dashboard
                    "lan": _lan_addresses(), "port": self.server.server_address[1],
                    "version": APP_VERSION}

        def _login(self):
            data = self._body()
            if data is None:
                return
            ip = self.client_address[0]
            name = " ".join(str(data.get("user", "")).split())[:40]
            if users.locked(ip):
                return self._json({"error": "Too many wrong attempts - wait 2 minutes."}, 429)
            got = users.verify(data.get("user", ""), data.get("password", ""), ip)
            if got is None:
                self._note("login failed", key="loginfail", remote_only=False)
                audit.write(name or "-", ip, "login failed")
                return self._json({"error": "Wrong user name or password."}, 401)
            tok = sessions.new(*got)
            print(f"[web]  {got[0]} logged in from {ip}", flush=True)
            audit.write(got[0], ip, "logged in", got[1])
            return self._json({"ok": True, "user": got[0], "role": got[1]},
                              extra={"Set-Cookie": f"{COOKIE}={tok}; HttpOnly; SameSite=Lax; Path=/"})

        def _logout(self):
            u = self._user()
            sessions.drop(self._token())
            if u:
                audit.write(u[0], self.client_address[0], "logged out")
            return self._json({"ok": True},
                              extra={"Set-Cookie": f"{COOKIE}=; Max-Age=0; HttpOnly; SameSite=Lax; Path=/"})

        def _password(self):
            """Any logged-in user changes their own password (current one first)."""
            u = self._user()
            if u is None:
                return self._json({"error": "Please log in first.", "login": True}, 401)
            data = self._body()
            if data is None:
                return
            ip = self.client_address[0]
            if users.locked(ip):
                return self._json({"error": "Too many wrong attempts - wait 2 minutes."}, 429)
            if users.verify(u[0], str(data.get("current", "")), ip) is None:
                return self._json({"error": "The current password is wrong."}, 400)
            new = str(data.get("password", ""))
            if new == str(data.get("current", "")):
                return self._json({"error": "The new password is the same as the current one."}, 400)
            err = users.update(u[0], password=new)
            if err:
                return self._json({"error": err}, 400)
            sessions.drop_user(u[0], keep=self._token())     # other browsers log in again
            print(f"[web]  {u[0]} changed their password (from {ip})", flush=True)
            audit.write(u[0], ip, "changed own password")
            return self._json({"ok": True})

        def _setup(self):
            """Create the first admin - only from the monitor PC, only once."""
            if users.any():
                return self._json({"error": "Already set up - log in."}, 400)
            if not self._is_local():
                return self._json({"error": "Create the first admin on the monitor PC itself."}, 403)
            data = self._body()
            if data is None:
                return
            err = users.add(str(data.get("user", "")).strip(), str(data.get("password", "")), "admin")
            if err:
                return self._json({"error": err}, 400)
            tok = sessions.new(str(data.get("user")).strip(), "admin")
            print(f"[web]  first admin '{str(data.get('user')).strip()}' created", flush=True)
            audit.write(str(data.get("user")).strip(), self.client_address[0], "created the first admin")
            return self._json({"ok": True},
                              extra={"Set-Cookie": f"{COOKIE}={tok}; HttpOnly; SameSite=Lax; Path=/"})

        # -- users -------------------------------------------------------------
        def _users_post(self, action):
            if not self._admin():
                return
            data = self._body()
            if data is None:
                return
            name = str(data.get("user", "")).strip()
            if action == "add":
                role = data.get("role", "viewer")
                ucams = data.get("cams") if data.get("cams") is not None else ["*"]
                err = users.add(name, str(data.get("password", "")), role, cams=ucams)
                what = f"added user {name} ({role}, cameras: {', '.join(ucams)})"
            elif action == "update":
                pw = data.get("password") or None
                role = data.get("role") or None
                ucams = data.get("cams")
                err = users.update(name, str(pw) if pw else None, role, cams=ucams)
                if not err and (role or pw):
                    # Log them in again with the new role / password - but an
                    # admin resetting their OWN password stays logged in here.
                    me = self._user()
                    keep = self._token() if me and me[0].lower() == name.lower() and not role else None
                    sessions.drop_user(name, keep=keep)
                what = (f"made {name} {role}" if role else "") + \
                       (" and " if role and pw else "") + (f"reset the password of {name}" if pw else "")
                if ucams is not None:
                    what = (what + "; " if what else "") + f"cameras of {name}: {', '.join(ucams)}"
            elif action == "delete":
                err = users.remove(name)
                if not err:
                    sessions.drop_user(name)
                what = f"removed user {name}"
            else:
                err, what = "unknown action", ""
            if err:
                return self._json({"error": err}, 400)
            print(f"[web]  user {name}: {action} by {self._who()}", flush=True)
            self._audit("users", what)
            return self._json({"ok": True, "users": users.list()})

        # -- displays (TV screens) ---------------------------------------------
        def _displays_get(self):
            if not self._admin():
                return
            return self._json({"displays": displays.list() if displays is not None else []})

        def _displays_post(self, action):
            if not self._admin():
                return
            data = self._body()
            if data is None:
                return
            if displays is None:
                return self._json({"error": "screens are not available"}, 500)
            if action == "add":
                d, err = displays.add(data.get("name"), data.get("cams"))
                what = f"added screen {data.get('name')} (cameras: {', '.join(data.get('cams') or [])})"
            elif action == "update":
                d, err = displays.update(str(data.get("id", "")), name=data.get("name"), cams=data.get("cams"),
                                         regenerate=bool(data.get("regenerate")))
                what = f"changed screen {data.get('id')}" + (" - new link" if data.get("regenerate") else "") + \
                       (f" (cameras: {', '.join(data.get('cams'))})" if data.get("cams") else "")
            elif action == "delete":
                err, d = displays.remove(str(data.get("id", ""))), None
                what = f"removed screen {data.get('id')}"
            else:
                err, d, what = "unknown action", None, ""
            if err:
                return self._json({"error": err}, 400)
            self._audit("screens", what)
            return self._json({"ok": True, "display": d, "displays": displays.list()})

        # -- cameras -----------------------------------------------------------
        def _cams_post(self, action):
            if not self._admin():
                return
            data = self._body()
            if data is None:
                return
            existing = cams.get(str(data.get("id") or "")) if data.get("id") else None
            if data.get("id") and existing is None and action != "add":
                return self._json({"error": "no such camera"}, 404)
            gives_address = any(data.get(k) for k in ("source", "host", "path"))

            if action == "discover":
                # Which stream paths does this camera serve? Asked with the login
                # from the form, or - for a saved camera - with the saved one.
                if data.get("host"):
                    src, err = compose_source(dict(data, path=data.get("path") or "/"), existing)
                elif data.get("source"):
                    src, err = compose_source(data)
                elif existing:
                    src, err = existing["source"], None
                else:
                    src, err = None, "give the camera's IP address"
                if err:
                    return self._json({"error": err}, 400)
                if not src.lower().startswith("rtsp"):
                    return self._json({"ok": False, "error": "Only network (RTSP) cameras can be "
                                       "searched. Give a video file or webcam number directly."})
                res = rtsp_probe.find_paths(src)
                if res["error"]:
                    return self._json({"ok": False, "error": res["error"]})
                if res["login"] == "refused":
                    return self._json({"ok": False, "login": "refused",
                                       "error": "The camera refused the user name / password."})
                found = []
                for f in res["found"][:4]:
                    got, terr = _test_source(rtsp_probe.with_path(src, f["path"]))
                    item = {"label": f["label"], "path": f["path"]}
                    if got:
                        item.update(width=got["w"], height=got["h"], image="data:image/jpeg;base64,"
                                    + base64.b64encode(got["jpeg"]).decode())
                    else:
                        item["error"] = terr
                    found.append(item)
                self._audit("cameras", f"searched streams on {mask_url(rtsp_probe.with_path(src, '/'))}"
                            f" - found {len(found)}")
                return self._json({"ok": bool(found), "login": res["login"], "found": found,
                                   "tried": len(res["tried"]) + 1,
                                   "error": None if found else
                                   "The camera answered, but not on any known stream path. "
                                   "Take the RTSP address from the camera's manual."})

            if action == "test":
                if gives_address:
                    src, err = compose_source(data, existing)
                elif existing:
                    src, err = existing["source"], None
                else:
                    src, err = None, "give the camera address"
                if err:
                    return self._json({"error": err}, 400)
                got, err = _test_source(src)
                if err:
                    return self._json({"ok": False, "error": err})
                return self._json({"ok": True, "width": got["w"], "height": got["h"],
                                   "image": "data:image/jpeg;base64," + base64.b64encode(got["jpeg"]).decode()})

            if action == "add":
                src, err = compose_source(data)
                if err:
                    return self._json({"error": err}, 400)
                cam, err = cams.add(data.get("name"), src, detect=bool(data.get("detect", True)))
                if err:
                    return self._json({"error": err}, 400)
                state.request_camera("add", cam)
                what = (f"added camera {cam['name']} ({cam['id']}) {mask_url(cam['source'])}"
                        f" - rules {'on' if cam['detect'] else 'off'}")
            elif action == "update":
                old = existing
                src = None
                if gives_address:
                    src, err = compose_source(data, old)
                    if err:
                        return self._json({"error": err}, 400)
                rules_en = data.get("rules_enabled")
                if rules_en is not None:
                    if not isinstance(rules_en, list) or any(r not in ruleinfo.CODES for r in rules_en):
                        return self._json({"error": "unknown rule"}, 400)
                cam, err = cams.update(old["id"], name=data.get("name"), source=src,
                                       detect=data.get("detect"), enabled=data.get("enabled"),
                                       rules_enabled=rules_en)
                if err:
                    return self._json({"error": err}, 400)
                if cam["id"] == "cam1" and cam["source"] != old["source"]:
                    try:
                        settings_store.set_camera_source(config_path, cam["source"])
                    except Exception as e:
                        print(f"[cameras] config.yaml camera.source not updated: {e}", flush=True)
                state.request_camera("update", cam)
                changes = []
                if cam["name"] != old["name"]:
                    changes.append(f"name {old['name']} -> {cam['name']}")
                if cam["source"] != old["source"]:
                    changes.append(f"address -> {mask_url(cam['source'])}")
                if cam["detect"] != old.get("detect", True):
                    changes.append(f"rules {'on' if cam['detect'] else 'off'}")
                if rules_en is not None and rules_en != (old.get("rules") or {}).get("enabled"):
                    changes.append("checks " + (", ".join(rules_en) or "no rule"))
                if cam["enabled"] != old.get("enabled", True):
                    changes.append("switched " + ("on" if cam["enabled"] else "off"))
                what = f"camera {cam['name']} ({cam['id']}): " + (", ".join(changes) or "no change")
            elif action == "delete":
                if existing is None:
                    return self._json({"error": "no such camera"}, 404)
                err = cams.remove(existing["id"])
                if err:
                    return self._json({"error": err}, 400)
                state.request_camera("remove", existing["id"])
                what = f"removed camera {existing['name']} ({existing['id']})"
            else:
                return self._json({"error": "unknown action"}, 400)
            print(f"[cameras] {action} {data.get('id') or data.get('name') or ''} by {self._who()}",
                  flush=True)
            self._audit("cameras", what)
            return self._json({"ok": True, "cameras": cams.public()})

        # -- zones -------------------------------------------------------------
        def _zones_info(self, q):
            cs = self._cam(q)
            if cs is None:
                return self._json({"error": "no camera"}, 404)
            cam = cams.get(cs.id) or {}
            w, h = cs.size or (1920, 1080)
            me = self._me()
            return self._json({"cam": cs.id, "name": cs.name,
                               "cameras": [{"id": c["id"], "name": c["name"]} for c in cams.public()],
                               "zones": {k: (v or []) for k, v in (cam.get("zones") or {}).items()},
                               "go_dir": ((cam.get("rules") or {}).get("SHI") or {}).get("go_dir") or {},
                               "frame": [w, h], "help": zone_store.HELP,
                               "local": self._is_local(), "can_save": me["can_admin"],
                               "user": me["user"], "setup_needed": me["setup_needed"]})

        def _save_zones(self, q):
            if not self._admin():
                self._note("zone save refused - not an admin", key="zonepin")
                return
            data = self._body()
            if data is None:
                return
            # a named camera must be THAT camera: falling back to the first one
            # would save one camera's zones onto another
            cs = self._cam({"cam": [data.get("cam") or (q.get("cam") or [None])[0]]})
            if cs is None:
                return self._json({"error": "no such camera"}, 404)
            w, h = cs.size or (1920, 1080)
            clean, err = zone_store.validate(data.get("zones"), w, h)
            if err:
                return self._json({"error": err}, 400)
            # 8 Oct: the feet direction of each STOP-LOOK-GO poster (checkpoint zones)
            gdir = data.get("go_dir")
            if gdir is not None:
                ok_v = ([0, -1], [0, 1], [-1, 0], [1, 0])
                if not isinstance(gdir, dict) or any(
                        not (k == "checkpoint" or str(k).startswith("checkpoint_"))
                        or (v is not None and list(v) not in ok_v) for k, v in gdir.items()):
                    return self._json({"error": "bad feet direction"}, 400)
            before_cam = copy.deepcopy(cams.get(cs.id) or {})
            before = before_cam.get("zones") or {}
            try:
                merged = cams.set_zones(cs.id, clean)
                bak = None
                if cs.id == "cam1":            # the replay/record tools read config.yaml
                    bak = zone_store.backup(config_path)
                    zone_store.save_zones(config_path, clean)
                    z = dict(cfg.get("zones") or {})
                    z.update(clean)
                    cfg["zones"] = z
            except Exception as e:
                return self._json({"error": f"could not save: {e}"}, 500)
            state.request_zones(cs.id, merged)
            if gdir:
                old_gd = ((before_cam.get("rules") or {}).get("SHI") or {}).get("go_dir") or {}
                if any(old_gd.get(k) != (list(v) if v else None) for k, v in gdir.items()):
                    cam_now = cams.set_go_dir(cs.id, gdir)
                    if cam_now is not None:
                        state.request_camera("update", cam_now)     # SHI rebuilt with the new directions
                        names = {(0, -1): "up", (0, 1): "down", (-1, 0): "left", (1, 0): "right"}
                        self._audit("zones", f"{cs.name} ({cs.id}): STOP-LOOK-GO feet " + ", ".join(
                            f"{k} {names.get(tuple(v), 'not set') if v else 'not set'}" for k, v in sorted(gdir.items())))
            changed = [f"{k} {len(before.get(k) or [])}->{len(v)} pts" for k, v in sorted(clean.items())
                       if (before.get(k) or []) != v]
            print(f"[zones] {cs.name}: saved by {self._who()} - "
                  + ", ".join(f"{k} {len(v)} pts" for k, v in clean.items()), flush=True)
            self._audit("zones", f"{cs.name} ({cs.id}): " + (", ".join(changed) or "no change"))
            return self._json({"ok": True, "saved": sorted(clean),
                               "backup": os.path.basename(bak) if bak else None})

        def _rawframe(self, q):
            import cv2                    # only here: the pages themselves need no OpenCV
            cs = self._cam(q)
            f = cs.raw_frame() if cs is not None else None
            if f is None:
                return self._send(503, "no camera picture yet", "text/plain")
            ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                return self._send(500, "encode failed", "text/plain")
            return self._send(200, buf.tobytes(), "image/jpeg", {"Cache-Control": "no-store"})

        # -- settings ----------------------------------------------------------
        def _settings_get(self):
            r = retention.settings()
            n, size = _dir_usage(clip_dir)
            n2, size2 = _dir_usage(snap_dir)
            n, size = n + n2, size + size2
            try:
                import shutil
                du = shutil.disk_usage(clip_dir)
                disk = {"free_gb": round(du.free / 1e9, 1), "total_gb": round(du.total / 1e9, 1)}
            except Exception:
                disk = None
            return self._json({"retention": {"days": r["days"], "sweep_min": r["sweep_min"],
                                             "trim_log": r["trim_log"]},
                               "evidence": {"files": n, "mb": round(size / 1e6, 1), "dir": clip_dir},
                               "disk": disk,
                               "overlays": self._toggles(),
                               "system": {"version": APP_VERSION,
                                          "uptime_sec": int(time.time() - state.started),
                                          "started": time.strftime("%Y-%m-%d %H:%M:%S",
                                                                   time.localtime(state.started)),
                                          "lan": _lan_addresses(),
                                          "port": self.server.server_address[1],
                                          "cameras": len(cams.all()),
                                          "users": len(users.list())}})

        def _settings_preview(self, q):
            try:
                days = float((q.get("days") or ["0"])[0])
            except ValueError:
                return self._json({"error": "days must be a number"}, 400)
            p = retention.preview(days)
            return self._json({"days": days, "files": p["files"], "mb": round(p["bytes"] / 1e6, 1),
                               "log_rows": p["log_rows"]})

        def _settings_post(self):
            if not self._admin():
                return
            data = self._body()
            if data is None:
                return
            done = []
            if "retention" in data:
                r = data.get("retention") or {}
                try:
                    days = float(r.get("days"))
                    sweep = float(r.get("sweep_min", retention.settings()["sweep_min"]))
                    trim = bool(r.get("trim_log", True))
                except (TypeError, ValueError):
                    return self._json({"error": "days and sweep minutes must be numbers"}, 400)
                if not 0 <= days <= 365:
                    return self._json({"error": "keep evidence 0-365 days (0 = never delete)"}, 400)
                if not 5 <= sweep <= 1440:
                    return self._json({"error": "clean up every 5-1440 minutes"}, 400)
                vals = {"days": int(days) if days.is_integer() else days,
                        "sweep_min": int(sweep) if sweep.is_integer() else sweep, "trim_log": trim}
                try:
                    bak = settings_store.set_values(config_path, "retention", vals)
                except Exception as e:
                    return self._json({"error": f"could not save: {e}"}, 500)
                cfg.setdefault("retention", {}).update(vals)
                retention.configure(days=days, sweep_min=sweep, trim_log=trim)
                print(f"[settings] retention -> {vals} by {self._who()} (backup {bak})", flush=True)
                done.append(f"retention {vals['days']} days, sweep {vals['sweep_min']} min, "
                            f"trim log {'on' if trim else 'off'}")
            if "overlays" in data:
                ov = data.get("overlays") or {}
                if not isinstance(ov, dict) or any(k not in OVERLAYS for k in ov):
                    return self._json({"error": "unknown overlay"}, 400)
                by_section = {}
                for name, on in ov.items():
                    section, key = OVERLAYS[name]
                    by_section.setdefault(section, {})[key] = bool(on)
                try:
                    for section, vals in by_section.items():
                        settings_store.set_values(config_path, section, vals)
                except Exception as e:
                    return self._json({"error": f"could not save: {e}"}, 500)
                for section, vals in by_section.items():
                    cfg_node(section).update(vals)
                print(f"[settings] overlays -> {ov} by {self._who()}", flush=True)
                done.append("live picture: " + ", ".join(f"{k} {'on' if v else 'off'}"
                                                         for k, v in ov.items()))
            if not done:
                return self._json({"error": "nothing to change"}, 400)
            self._audit("settings", "; ".join(done))
            return self._settings_get()

        # -- state -------------------------------------------------------------
        def _state(self, q, allow=None):
            if allow is not None and not (q.get("cam") or [None])[0]:
                first = next((c["id"] for c in cams.public() if c["id"] in allow), None)
                q = dict(q, cam=[first]) if first else q
            cs = self._cam(q)
            s = cs.snapshot() if cs else {"fps": 0, "view_fps": 0, "persons": 0,
                                         "camera_ok": False, "last_error": "no camera"}
            s["uptime_sec"] = int(time.time() - state.started)
            pub = [c for c in cams.public() if allow is None or c["id"] in allow]
            sel = next((c for c in pub if cs and c["id"] == cs.id), None)
            s["source"] = sel["source"] if sel else ""
            s["toggles"] = self._toggles()
            # What the selected camera is really running: `rules` are the codes
            # judging right now, `rule_status` every rule with its label and, for
            # the ones that cannot fire, the reason (poketenashi/ruleinfo.py).
            status = rule_status(cfg["rules"].get("enabled", []), sel)
            s["rule_status"] = status
            s["rules"] = [r["rule"] for r in status if r["active"]]
            s["cameras"] = []
            for c in pub:
                st = state.cams.get(c["id"])
                snap = st.snapshot() if st else {}
                s["cameras"].append({**c, "rule_status": rule_status(cfg["rules"].get("enabled", []), c),
                                     "camera_ok": bool(snap.get("camera_ok")),
                                     "fps": snap.get("fps", 0), "view_fps": snap.get("view_fps", 0),
                                     "persons": snap.get("persons", 0), "size": snap.get("size"),
                                     "gray_pct": snap.get("gray_pct", 0),
                                     "last_error": snap.get("last_error") if st else
                                     ("switched off" if not c["enabled"] else "starting…")})
            return self._json(s)

        # -- routes ------------------------------------------------------------
        def do_POST(self):
            self._raw = self._read_raw()       # per request: the handler serves a whole connection
            self._note_visit()
            u = urlparse(self.path)
            p, q = u.path, parse_qs(u.query)
            if p == "/api/clientlog":
                return self._clientlog()
            if p == "/api/login":
                return self._login()
            if p == "/api/logout":
                return self._logout()
            if p == "/api/setup":
                return self._setup()
            if p == "/api/password":
                return self._password()
            if p == "/api/zones":
                return self._save_zones(q)
            if p.startswith("/api/displays/"):
                return self._displays_post(p[len("/api/displays/"):].strip("/"))
            if p.startswith("/api/cameras"):
                return self._cams_post(p[len("/api/cameras"):].strip("/") or "add")
            if p.startswith("/api/users"):
                return self._users_post(p[len("/api/users"):].strip("/") or "add")
            if p == "/api/settings":
                return self._settings_post()
            if p == "/api/toggle":
                # Old switch (pages before 3 Oct): changes the picture for every
                # viewer, so admins only; not saved - Settings saves overlays.
                if not self._admin():
                    return
                name = (q.get("name") or [""])[0]
                if name not in OVERLAYS:
                    return self._json({"error": "unknown toggle"}, 400)
                on = (q.get("on") or ["true"])[0].lower() not in ("0", "false", "no")
                section, key = OVERLAYS[name]
                cfg_node(section)[key] = on
                self._audit("settings", f"live picture: {name} {'on' if on else 'off'} (not saved)")
                return self._json({"toggles": self._toggles()})
            return self._send(404, "not found", "text/plain")

        def do_GET(self):
            self._note_visit()
            u = urlparse(self.path)
            path, q = u.path, parse_qs(u.query)

            if path == "/api/health":
                return self._json({"status": "ok"})
            if path == "/api/frame":
                if self._view(self._cam_id(q)) is DENY:
                    return
                cs = self._cam(q)
                jpeg = cs.frame()[0] if cs else None
                if jpeg is None:
                    return self._send(503, "no picture yet", "text/plain")
                return self._send(200, jpeg, "image/jpeg", {"Cache-Control": "no-store"})
            if path == "/api/state":
                a = self._view((q.get("cam") or [None])[0])
                if a is DENY:
                    return
                return self._state(q, a)
            if path == "/api/violations":
                try:
                    limit = max(1, min(int((q.get("limit") or [100])[0]), 1000))
                except ValueError:
                    limit = 100
                rule = (q.get("rule") or [None])[0] or None
                cam = (q.get("cam") or [None])[0] or None
                a = self._view(cam)
                if a is DENY:
                    return
                return self._json(read_tail(log_path, snap_dir, limit, rule, cam, allow=a))
            if path == "/api/stats":
                cam = (q.get("cam") or [None])[0] or None
                a = self._view(cam)
                if a is DENY:
                    return
                return self._json(stats.get(cam, allow=a))
            if path == "/api/stream":
                if self._view(self._cam_id(q)) is DENY:
                    return
                return self._stream(self._cam(q))
            if path == "/api/alerts":
                try:
                    since = int((q.get("since") or [0])[0])
                except ValueError:
                    since = 0
                a = self._view()
                if a is DENY:
                    return
                items, seq = state.alerts_since(since)
                if a is not None:
                    items = [x for x in items if (x.get("camera") or "cam1") in a]
                return self._json({"seq": seq, "alerts": items})
            if path == "/api/cameras":
                a = self._view()
                if a is DENY:
                    return
                return self._json({"cameras": [c for c in cams.public() if a is None or c["id"] in a]})
            if path == "/api/displays":
                return self._displays_get()
            if path == "/api/me":
                return self._json(self._me())
            if path == "/api/settings":
                return self._settings_get()
            if path == "/api/settings/preview":
                return self._settings_preview(q)
            if path == "/api/users":
                if not self._admin():
                    return
                return self._json({"users": users.list()})
            if path == "/api/audit":
                if not self._admin():
                    return
                try:
                    limit = max(1, min(int((q.get("limit") or [200])[0]), 1000))
                except ValueError:
                    limit = 200
                return self._json({"entries": audit.tail(limit)})
            if path.startswith("/snapshots/") or path.startswith("/clips/"):
                name = path.split("/", 2)[2]
                m = _CAM_FILE.match(name)
                if self._view(m.group(1) if m else "cam1") is DENY:
                    return
                root = snap_dir if path.startswith("/snapshots/") else clip_dir
                return self._file(root, name, "max-age=3600")
            if path in ("/", "/index.html"):
                return self._file(WEB_DIR, "index.html")
            if path in ("/tv", "/tv/"):
                tok = (q.get("d") or [None])[0]
                if tok and displays is not None and displays.by_token(tok):
                    # a display link: this screen keeps its cameras (cookie for 10 years)
                    self._note(f"screen link opened: {displays.by_token(tok)['name']}", key="tvlink")
                    return self._send(302, b"", "text/plain", {
                        "Location": "/tv", "Cache-Control": "no-store",
                        "Set-Cookie": f"{COOKIE_TV}={tok}; Max-Age=315360000; HttpOnly; SameSite=Lax; Path=/"})
                return self._file(WEB_DIR, "tv.html")
            if path in ("/zones", "/zones/"):
                return self._file(WEB_DIR, "zones.html")
            if path == "/api/zones":
                if self._view(self._cam_id(q)) is DENY:
                    return
                return self._zones_info(q)
            if path == "/api/rawframe":
                if self._view(self._cam_id(q)) is DENY:
                    return
                return self._rawframe(q)
            return self._file(WEB_DIR, path.lstrip("/"))

        def _stream(self, cs):
            # Frames are dropped, never queued: while a write blocks, the camera
            # keeps replacing its single latest frame, so the next write sends
            # the CURRENT picture. Dropping frames is right for a live view.
            if cs is None:
                return self._send(404, "no such camera", "text/plain")
            try:
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=" + BOUNDARY)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            last = -1
            try:
                while True:
                    if state.get_cam(cs.id) is not cs:
                        break                      # the camera was removed or rebuilt
                    jpeg, seq = cs.frame()
                    if jpeg is None or seq == last:
                        time.sleep(0.01)
                        continue
                    last = seq
                    self.wfile.write(
                        b"--" + BOUNDARY.encode() + b"\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                        + jpeg + b"\r\n")
                    time.sleep(STREAM_SLEEP)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

    return Handler


class _Server(ThreadingHTTPServer):
    """HTTP server that REFUSES to share its port: a second copy of the monitor
    fails at once instead of loading its models and fighting over it."""
    allow_reuse_address = False
    daemon_threads = True

    def handle_error(self, request, client_address):
        import sys
        if isinstance(sys.exc_info()[1], CLIENT_GONE):
            return
        super().handle_error(request, client_address)


def serve(state, cfg, log_path, snap_dir, clip_dir="clips", host="0.0.0.0", port=8080,
          config_path="config.yaml", cams=None, users=None, sessions=None, audit_path=None,
          displays=None):
    """Start the HTTP server on a daemon thread and return it.

    Raises OSError if the port is already held - meaning another copy is running.
    """
    handler = make_handler(state, cfg, log_path, snap_dir, clip_dir, config_path,
                           cams, users, sessions, AuditLog(audit_path), displays)
    httpd = _Server((host, port), handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd
