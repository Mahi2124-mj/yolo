"""Short evidence clips: a few seconds of video around each alert.

A still frame tells you a rule fired; it does not tell you what led up to it. The
run-up is the useful half - it shows the approach that produced the bypass - so
every clip carries a rolling PRE-roll from before the alert plus a POST-roll
after it.

Why frames are piped to ffmpeg rather than written with cv2.VideoWriter: the only
codec that reliably opens here is mp4v (MPEG-4 Part 2), which no browser will
play in a <video> tag - H.264 needs an openh264 DLL that isn't present. The
ffmpeg binary bundled inside the ``imageio-ffmpeg`` wheel produces real H.264.
Frames are buffered as JPEGs so nothing has to be held as raw BGR, and they are
downscaled first: full-resolution frames in the pre-roll ring plus a queue of
pending clips is hundreds of MB, which once OOM-killed this process.

Encoding happens on one shared worker thread, so a burst of alerts can never
spawn a burst of encoders.
"""
import os
import queue
import subprocess
import threading
import time
from collections import deque
from types import SimpleNamespace

import cv2
import numpy as np

# Frames per second sampled into a clip. Well below the pipeline rate on purpose:
# the ring holds encoded JPEGs in RAM, and this is what keeps it at a few MB
# rather than tens.
DEFAULT_FPS = 8
# Clip frames are downscaled before they are buffered. A full 2304x1296 JPEG is
# ~500 KB, so a few seconds of pre-roll plus a queue of pending clips is hundreds
# of MB of RAM - which is what killed this process once on a 16 GB machine that
# already gives ~1 GB to the model. At 1280 wide a frame is ~4x smaller and the
# evidence is still perfectly readable.
MAX_WIDTH = 1280
JPEG_QUALITY = 70
# Hard ceilings on how much can be in flight at once. A burst of alerts must cost
# bounded memory, not memory proportional to the burst.
MAX_PENDING = 3                 # clips collecting their post-roll at the same time
QUEUE_MAX = 6                   # clips waiting to be encoded
# A clip covers the whole violation, not a fixed window: while the rule keeps
# firing the recording keeps going. MAX_SEC stops a person who simply stands
# with their hands in their pockets from producing an unbounded clip - and an
# unbounded buffer with it.
MAX_SEC = 20.0
# 10 Oct (user: "the KE box moves late - it sits on empty space, not on the
# person"). Detection runs slower than the video (0.4-9 fps against 8-15), so
# the newest boxes belong to an older frame and trail a walking person. Clip
# frames are therefore kept CLEAN and the boxes drawn at encode time, moved
# between the detection before and the one after each frame. A finished clip
# waits up to this long for the detection after its last frame.
CLOSE_WAIT_SEC = 3.0
# Share of the gap between two detections that a person seen in only one of
# them stays drawn (see _scene).
ALONE_SHARE = 0.25

_q = queue.Queue(maxsize=QUEUE_MAX)
_thread = None
_lock = threading.Lock()
_ffmpeg = None
_ffmpeg_checked = False


def _ffmpeg_exe():
    global _ffmpeg, _ffmpeg_checked
    if _ffmpeg_checked:
        return _ffmpeg
    _ffmpeg_checked = True
    try:
        import imageio_ffmpeg
        _ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        _ffmpeg = None
        print("[clips] imageio-ffmpeg not installed -> falling back to mp4v, which "
              "most browsers will NOT play. Fix with: pip install imageio-ffmpeg",
              flush=True)
    return _ffmpeg


def _encode_ffmpeg(exe, path, jpegs, fps):
    cmd = [
        exe, "-y", "-loglevel", "error",
        # "-c:v mjpeg" on the INPUT side is not optional. Left to probe the pipe,
        # ffmpeg reports "unknown codec" whenever the frames are low-detail enough
        # to be small (a near-empty scene) and the clip silently never appears.
        # Stating the codec removes the probe entirely.
        "-f", "image2pipe", "-c:v", "mjpeg", "-framerate", str(fps), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)   # no console pop on Windows
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, creationflags=flags)
    try:
        for j in jpegs:
            proc.stdin.write(j)
        proc.stdin.close()
        _, err = proc.communicate(timeout=120)
    except Exception:
        proc.kill()
        return False
    if proc.returncode != 0:
        print("[clips] ffmpeg failed:", err.decode(errors="ignore")[:200], flush=True)
        return False
    return True


def _encode_cv2(path, jpegs, fps):
    """Last resort without ffmpeg: mp4v, readable by desktop players but usually
    not by a browser."""
    import cv2
    import numpy as np
    frames = [cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR) for j in jpegs]
    frames = [f for f in frames if f is not None]
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        return False
    for f in frames:
        writer.write(f)
    writer.release()
    return True


def _write_poster(path, jpegs):
    """Save one frame of the clip as a small JPEG next to it.

    The gallery needs a thumbnail per clip. Without one the page falls back to a
    <video> element per card, and a few hundred of those - each fetching metadata
    from a ~700 KB file - is enough to hang the browser outright. The frames are
    already encoded in memory here, so a poster costs one file write.

    The alert that opened the clip normally wrote its own poster already - the
    frame it fired on, which is a better thumbnail than mid-clip - so an
    existing file is left alone.
    """
    if not jpegs:
        return
    poster = os.path.splitext(path)[0] + ".jpg"
    if os.path.exists(poster):
        return
    try:
        with open(poster, "wb") as f:
            f.write(jpegs[len(jpegs) // 2])          # mid-clip: the event itself
    except OSError as exc:
        print("[clips] poster write failed:", exc, flush=True)


def _loop():
    while True:
        path, jpegs, fps, render = _q.get()
        try:
            if render is not None:                   # draw the boxes (ClipRecorder._render)
                jpegs = render(jpegs)
            exe = _ffmpeg_exe()
            ok = _encode_ffmpeg(exe, path, jpegs, fps) if exe else _encode_cv2(path, jpegs, fps)
            if ok:
                _write_poster(path, jpegs)
            elif os.path.exists(path):
                os.remove(path)
        except Exception as exc:
            print("[clips] encode error for", os.path.basename(path), "-", exc, flush=True)
        finally:
            _q.task_done()


def wait_idle(timeout=20.0):
    """Block until every queued clip is encoded (or `timeout` passes) - so a
    clean shutdown does not throw away the clip it just flushed."""
    deadline = time.time() + timeout
    while _q.unfinished_tasks and time.time() < deadline:
        time.sleep(0.1)


def submit(path, jpegs, fps=DEFAULT_FPS, render=None):
    """Queue a finished clip for encoding. `render(jpegs) -> jpegs`, when given,
    runs on the encoder thread first (it draws the boxes).

    Drops the job if the encoder is backed up: the snapshot is the primary
    record, so a missing clip must never stall the pipeline.
    """
    global _thread
    if not jpegs:
        return
    with _lock:
        if _thread is None or not _thread.is_alive():
            _thread = threading.Thread(target=_loop, daemon=True, name="clip-encoder")
            _thread.start()
    try:
        _q.put_nowait((path, list(jpegs), fps, render))
    except queue.Full:
        print("[clips] encoder backlog full - dropping one clip", flush=True)


class ClipRecorder:
    """Rolling pre-roll ring plus any clips currently collecting their post-roll.

    The pipeline calls feed() with every rendered frame and start() when an alert
    fires. Sampling to `fps` happens here, so the caller does not have to think
    about it.

    Given the frame's number (`seq`) and the newest detection, feed() keeps the
    frame clean and the boxes are drawn when the clip is encoded (`draw`, set by
    the caller), each box placed between the detection before and the one after
    that frame - so it sits on the person, not where they were a second ago.
    """

    def __init__(self, out_dir, pre_sec=4.0, post_sec=4.0, fps=DEFAULT_FPS,
                 enabled=True, max_width=MAX_WIDTH, max_sec=MAX_SEC):
        self.dir = out_dir
        self.pre_sec = float(pre_sec)
        self.post_sec = float(post_sec)
        self.fps = int(fps)
        self.enabled = bool(enabled)
        self.max_width = int(max_width)
        self.max_sec = float(max_sec)
        self.dropped = 0
        # The ring is trimmed by TIME, not by a frame count. Sized as
        # pre_sec * fps it held 32 frames, but it is filled at whatever rate the
        # pipeline manages - 2.7 fps here - so those 32 frames were 12 seconds of
        # pre-roll, not 4. Clips opened on twelve seconds of nothing with the
        # violation squeezed into the tail, and people reasonably concluded the
        # clip had caught nothing.
        self._ring = deque()        # [(timestamp, seq, scale, jpeg)]
        self._pending = []          # [{"path", "keys", "frames", "times", "seqs", "scales", "until", "deadline"}]
        self._closing = []          # finished clips waiting for the detection after their last frame
        self._dets = []             # detections while frames are kept, ascending seq (_summary)
        self._posters = []          # poster filenames to write from the next rendered frame
        self._last_add = 0.0
        # draw(bgr, persons, phones, violations, names, fps) -> bgr: the clip overlay
        self.draw = None
        # Alerts open and extend clips from the detection loop while the live-view
        # thread feeds them frames, so the shared state is only touched under this.
        self._lock = threading.Lock()
        if self.enabled:
            os.makedirs(self.dir, exist_ok=True)

    def _real_fps(self, times):
        """The rate the frames were actually sampled at.

        Writing the file at the configured fps made every clip play back at the
        wrong speed: sampled at 2.7 fps and written as 8, twelve seconds of
        footage ran past in four. The timestamps are recorded anyway, so use
        them and the clip plays at life speed.
        """
        if len(times) < 2:
            return float(self.fps)
        span = times[-1] - times[0]
        if span <= 0:
            return float(self.fps)
        # Clamp: a stall must not produce a 0.2 fps file some players refuse.
        return max(1.0, min(float(self.fps), (len(times) - 1) / span))

    def _encode(self, frame):
        """Downscale and JPEG-encode one frame for the buffer: (jpeg, scale)."""
        h, w = frame.shape[:2]
        s = 1.0
        if w > self.max_width:
            s = self.max_width / float(w)
            frame = cv2.resize(frame, (int(w * s), int(h * s)),
                               interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return (buf.tobytes(), s) if ok else (None, s)

    def due(self, now=None):
        """True when the next frame will actually be used - sampled into a clip,
        or written as the poster of an alert that just fired.

        Exposed so the caller can render the clip's own overlay only on the
        frames that reach it - a few a second - instead of on every frame.
        """
        if not self.enabled:
            return False
        if self._posters:
            return True
        now = time.time() if now is None else now
        return (now - self._last_add) >= 1.0 / self.fps

    def request_poster(self, name):
        """Ask for `name` to be written from the next rendered frame - the one
        this alert fired on, with its red box. Returns the name, or None.

        Several alerts in a few seconds share one clip, and a clip's own poster
        is its middle frame, so without this every card in that burst showed
        the same picture - often of somebody else.
        """
        if not self.enabled:
            return None
        with self._lock:
            self._posters.append(name)
        return name

    def take_posters(self):
        """The poster names waiting, handed to a caller that draws them itself on
        the frame the detection was made on (write_poster)."""
        with self._lock:
            names, self._posters = self._posters, []
        return names

    def write_poster(self, name, frame):
        jpeg, _ = self._encode(frame)
        if jpeg is None:
            return
        try:
            with open(os.path.join(self.dir, name), "wb") as f:
                f.write(jpeg)
        except OSError as exc:
            print("[clips] poster write failed:", exc, flush=True)

    def note(self, det):
        """Remember a detection ({"seq", "persons", "phones", "violations",
        "names", "fps"}) for the clip boxes. Cheap when it is the one already
        noted, so the caller can pass the newest on every frame."""
        if not self.enabled or not det or det.get("seq") is None:
            return
        with self._lock:
            self._note(det)
            ready = self._ready(time.time())
        for clip in ready:
            self._submit(clip)

    def _note(self, det):
        if self._dets and self._dets[-1]["seq"] >= det["seq"]:
            return
        self._dets.append(_summary(det))
        # keep what the oldest frame still held needs: the detection before it onwards
        held = ([self._ring[0][1]] if self._ring else []) + [c["seqs"][0] for c in self._pending + self._closing if c["seqs"]]
        held = [s for s in held if s is not None]
        first = min(held) if held else self._dets[-1]["seq"]
        k = 0
        while k + 1 < len(self._dets) and self._dets[k + 1]["seq"] <= first:
            k += 1
        del self._dets[:k]

    def _ready(self, now):
        """Finished clips whose last frame has a detection after it (or that waited long enough)."""
        last = self._dets[-1]["seq"] if self._dets else None
        ready = [c for c in self._closing
                 if now >= c["close_by"] or (last is not None and c["seqs"] and c["seqs"][-1] is not None
                                             and last >= c["seqs"][-1])]
        for c in ready:
            self._closing.remove(c)
            c["dets"] = list(self._dets)
        return ready

    def _submit(self, clip):
        render = None
        if self.draw is not None and any(s is not None for s in clip["seqs"]):
            seqs, scales, dets, draw = list(clip["seqs"]), list(clip["scales"]), clip.get("dets") or list(self._dets), self.draw
            render = lambda jpegs: _render(jpegs, seqs, scales, dets, draw)  # noqa: E731
        submit(clip["path"], clip["frames"], self._real_fps(clip["times"]), render)

    def feed(self, frame, seq=None, det=None):
        """Offer one frame (BGR). Ignored unless it is time to sample or a poster
        is waiting for it.

        With `seq` (the camera frame number) and `det` (the newest detection) the
        frame is CLEAN and its boxes are drawn when the clip is encoded; without
        them it is taken as already annotated (the old way).

        Encoding happens here rather than in the caller so it only runs on the
        frames that actually reach a clip - `fps` of them a second, not every
        frame the pipeline renders.
        """
        if not self.enabled or frame is None:
            return
        now = time.time()
        with self._lock:
            if det is not None and seq is not None:
                self._note(det)
            else:
                seq = None
            sample = now - self._last_add >= 1.0 / self.fps
            posters = []
            if seq is None:                     # annotated frame: posters come from it as before
                posters, self._posters = self._posters, []
            if sample:
                self._last_add = now
            ready = self._ready(now)
        for clip in ready:
            self._submit(clip)
        if not sample and not posters:
            return

        jpeg, scale = self._encode(frame)   # outside the lock: the slow part
        if jpeg is None:
            return
        for name in posters:
            try:
                with open(os.path.join(self.dir, name), "wb") as f:
                    f.write(jpeg)
            except OSError as exc:
                print("[clips] poster write failed:", exc, flush=True)
        if not sample:
            return
        with self._lock:
            self._ring.append((now, seq, scale, jpeg))
            while self._ring and now - self._ring[0][0] > self.pre_sec:
                self._ring.popleft()
            for clip in list(self._pending):
                clip["frames"].append(jpeg)
                clip["times"].append(now)
                clip["seqs"].append(seq)
                clip["scales"].append(scale)
                if now >= clip["until"]:
                    self._pending.remove(clip)
                    clip["close_by"] = now + CLOSE_WAIT_SEC
                    self._closing.append(clip)
            ready = self._ready(now)
        for clip in ready:
            self._submit(clip)

    def keep_alive(self, keys):
        """Extend every pending clip whose (track_id, rule) is still violating.

        Called each frame with the rules that are ACTIVE right now, so a clip
        runs for as long as the bypass does instead of cutting off after a fixed
        few seconds - subject to max_sec.
        """
        if not self.enabled or not keys:
            return
        now = time.time()
        with self._lock:
            for clip in self._pending:
                if clip["keys"] & keys and now < clip["deadline"]:
                    clip["until"] = max(clip["until"], min(now + self.post_sec, clip["deadline"]))

    def start(self, name, key=None):
        """Begin a clip named `name`, or join one that is already recording.

        `key` identifies the violation ((track_id, rule)) so keep_alive can extend
        it. Returns the filename the alert should point at, or None when clips
        are off - so the caller logs exactly what will exist on disk.

        A clip records the whole scene, not one person, so every alert inside
        the same few seconds is already on the clip that is running. They used
        to open a clip each; with three recording, every further alert got
        "[clips] 3 already recording - no clip" and no evidence at all - 649
        alerts in one day. Sharing also means one encode instead of three
        overlapping ones.
        """
        if not self.enabled:
            return None
        now = time.time()
        with self._lock:
            for c in self._pending:
                if key in c["keys"] or os.path.basename(c["path"]) == name:
                    return os.path.basename(c["path"])   # already recording this one
            # Join the running clip that can still give this alert its full
            # post-roll. Its pre-roll and what it has recorded since already
            # cover the run-up.
            joinable = [c for c in self._pending if c["deadline"] - now >= self.post_sec]
            if not joinable and len(self._pending) >= MAX_PENDING:
                # Every slot is taken and each clip is about to hit max_sec: the
                # one that runs longest still shows part of this alert, which
                # beats none.
                joinable = self._pending
            if joinable:
                c = max(joinable, key=lambda c: c["deadline"])
                c["keys"].add(key)
                c["until"] = max(c["until"], min(now + self.post_sec, c["deadline"]))
                return os.path.basename(c["path"])
            self._pending.append({
                "path": os.path.join(self.dir, name),
                "keys": {key},
                "frames": [r[3] for r in self._ring],
                "times": [r[0] for r in self._ring],
                "seqs": [r[1] for r in self._ring],
                "scales": [r[2] for r in self._ring],
                "until": now + self.post_sec,
                "deadline": now + self.max_sec,      # hard stop, however long it lasts
            })
        return name

    def flush(self):
        """Encode whatever is still collecting - called on shutdown so a clip in
        progress is not lost."""
        with self._lock:
            pending, self._pending = self._pending + self._closing, []
            self._closing = []
        for clip in pending:
            self._submit(clip)


# ------------------------------------------------------------------ clip boxes
def _summary(det):
    """What the clip overlay needs from one detection (no frames, no keypoints)."""
    return {
        "seq": int(det["seq"]),
        "persons": [(int(p.track_id), np.asarray(p.bbox, float).copy(), bool(getattr(p, "tracked", True)),
                     bool(getattr(p, "in_zone", True))) for p in det.get("persons") or []],
        "phones": [np.asarray(ph["bbox"], float).copy() for ph in det.get("phones") or []],
        "violations": [{"track_id": v["track_id"], "rule": v["rule"]} for v in det.get("violations") or []],
        "names": dict(det.get("names") or {}),
        "fps": det.get("fps"),
    }


def _scene(seq, dets):
    """Boxes for frame `seq`: each tracked person moved linearly between the
    detection just before the frame and the one just after it. A person in only
    one of the two is shown close to that one only; labels and phones come from
    the nearer detection."""
    a = b = None
    for d in dets:
        if d["seq"] <= seq:
            a = d
        else:
            b = d
            break
    if a is None:                                       # before the first detection kept: nothing known
        return [], [], [], {}, None
    if b is None:                                       # after the last one: its boxes as they are
        return ([SimpleNamespace(track_id=t, bbox=bb, tracked=tr, in_zone=z, keypoints=None)
                 for t, bb, tr, z in a["persons"]], a["phones"], a["violations"], a["names"], a["fps"])
    f = (seq - a["seq"]) / float(b["seq"] - a["seq"])
    near = a if f < 0.5 else b
    after = {t: bb for t, bb, tr, _ in b["persons"] if tr}
    persons = []
    # A person in only one of the two detections (walking out of / into view) is
    # drawn only close to that detection: held for half the gap, the box stood on
    # the spot they had just left (first live clip after this change).
    for t, bb, tr, z in a["persons"]:
        if tr and t in after:
            persons.append(SimpleNamespace(track_id=t, bbox=bb + f * (after[t] - bb), tracked=tr, in_zone=z, keypoints=None))
        elif f < ALONE_SHARE:
            persons.append(SimpleNamespace(track_id=t, bbox=bb, tracked=tr, in_zone=z, keypoints=None))
    if f > 1.0 - ALONE_SHARE:
        known = {t for t, _, tr, _ in a["persons"] if tr}
        persons += [SimpleNamespace(track_id=t, bbox=bb, tracked=tr, in_zone=z, keypoints=None)
                    for t, bb, tr, z in b["persons"] if not (tr and t in known)]
    return persons, near["phones"], near["violations"], near["names"], near["fps"]


def _render(jpegs, seqs, scales, dets, draw):
    """Encoder thread: decode each clean frame, draw its boxes, encode again."""
    out = []
    for jpeg, seq, s in zip(jpegs, seqs, scales):
        if seq is None:                                  # already annotated
            out.append(jpeg)
            continue
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            out.append(jpeg)
            continue
        persons, phones, violations, names, fps = _scene(seq, dets)
        if s != 1.0:                                     # boxes are in full-size pixels
            persons = [SimpleNamespace(**dict(vars(p), bbox=p.bbox * s)) for p in persons]
            phones = [b * s for b in phones]
        try:
            img = draw(img, persons, [{"bbox": b} for b in phones], violations, names, fps)
        except Exception as exc:
            print("[clips] box drawing failed:", exc, flush=True)
            out.append(jpeg)
            continue
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        out.append(buf.tobytes() if ok else jpeg)
    return out
