"""Threaded video capture with auto-reconnect, tuned for RTSP bullet cameras.

Reading frames on a background thread keeps the latest frame available and
avoids the buffer lag you get when the main loop is slower than the stream.
"""
import os
import re
import socket
import time
import threading
from urllib.parse import urlparse

import cv2
import numpy as np

# How long one attempt may take to open the stream, and how long a read may wait
# for data before it counts as a dropped stream. OpenCV's own defaults are 30 s
# each, and with the TCP connect retries on top a single reconnect attempt took
# ~110 s - so after an outage the view stayed dead for up to two minutes AFTER
# the camera was back.
OPEN_TIMEOUT_MS = 10000
READ_TIMEOUT_MS = 10000
# While the camera is unreachable, repeat the "still down" note this seldom.
DOWN_NOTE_SEC = 300.0
# Decode on ONE thread. Left to itself OpenCV lets FFmpeg pick, and on this
# 28-thread PC it picks 16 - and H.264 frame threading holds one frame back per
# extra thread: 15 frames = 0.75 s behind at the camera's 20 fps, measured on
# the clock the camera burns into the picture (28 Sep). One thread decodes this
# 1080p stream in real time with room to spare.
DECODE_THREADS = 1

# Frames the H.264 decoder filled with flat gray because the frames they build
# on never arrived: the stream fell behind. On 3 Oct a quarter of the desk
# camera's 3 MP frames came out like that while the PC ran at 80-86 % CPU (the
# same stream read on its own: 27 fps, none). Shown, they flash the picture
# gray; judged, every person "vanishes" and the tracker loses them. Skipped.
GRAY_SKIP = True
GRAY_BLOCKS = (6, 8)       # the picture is judged in this grid of blocks
GRAY_FLAT_STD = 1.5        # a block this flat ...
GRAY_MID_DEV = 10.0        # ... and this close to mid-gray (128) is filler
GRAY_FRAC = 0.3            # skip a frame when at least this share of blocks is
# Measured on both live cameras: real pictures had 0 % such blocks, broken
# frames 88-100 %; a frame gray from the middle down scores 0.33.
GRAY_ONLY_SEC = 5.0        # nothing but gray for this long: say so as the status
GRAY_NOTE_SEC = 600.0      # log the skipped frames at most this often (the dashboard shows the live %)


def gray_fraction(frame, stride=16):
    """Share (0..1) of the picture that is flat mid-gray - the decoder's filler
    for a frame whose reference frames were lost.

    It runs on every frame of every camera, on the very thread that falls
    behind when the PC is busy, so it looks at every 16th pixel of the green
    channel for flatness (~0.2 ms for 3 MP) and checks the colour of the flat
    blocks only: mid-gray in all three channels, not e.g. a flat green wall."""
    g = frame[::stride, ::stride]
    gy, gx = GRAY_BLOCKS
    bh, bw = g.shape[0] // gy, g.shape[1] // gx
    if bh < 2 or bw < 2:
        return 0.0
    g = g[:bh * gy, :bw * gx]
    if g.ndim == 2:
        g = g[:, :, None]
    blocks = g.reshape(gy, bh, gx, bw, g.shape[2]).swapaxes(1, 2).reshape(gy * gx, bh * bw, g.shape[2])
    green = blocks[:, :, min(1, g.shape[2] - 1)].astype(np.float32)
    flat = (green.std(axis=1) < GRAY_FLAT_STD) & (np.abs(green.mean(axis=1) - 128.0) < GRAY_MID_DEV)
    if not flat.any():
        return 0.0
    means = blocks[flat].mean(axis=1)                       # every channel, flat blocks only
    gray = (np.abs(means - 128.0) < GRAY_MID_DEV).all(axis=1)
    return float(gray.sum()) / (gy * gx)


def mask_url(url):
    """rtsp://user:secret@host/... -> rtsp://user:***@host/...

    For everything shown or logged. The dashboard is served to the whole LAN
    and the log is read by other people; neither needs the camera password.
    """
    return re.sub(r"(?<=://)([^:/@]+):[^@/]*@", r"\1:***@", str(url))


class VideoStream:
    def __init__(self, source, rtsp_transport="tcp", reconnect_delay=3.0, max_width=0):
        self.source = source
        self.reconnect_delay = reconnect_delay
        # Downscale on the way in, once, when the stream is wider than this.
        # Every consumer copies the stored frame, so a 2304x1296 grab means 8.5 MB
        # per copy with several in flight - which OOMs on a machine with little
        # free RAM. Shrinking here shrinks every copy downstream. 0 = leave alone.
        self.max_width = int(max_width or 0)

        # Force RTSP over TCP (more reliable than UDP for IP cams).
        if isinstance(source, str) and source.lower().startswith("rtsp"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                f"rtsp_transport;{rtsp_transport}|stimeout;10000000"
            )

        self.cap = None
        self.frame = None
        self.ok = False
        self.seq = 0               # +1 per NEW frame; lets a reader skip repeats
        self.last_frame_t = 0.0    # wall time the newest frame arrived
        self.status = None         # why the camera is down, in words; None while fine
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)

        self._rtsp = isinstance(source, str) and source.lower().startswith("rtsp")
        u = urlparse(source) if self._rtsp else None
        self._host = u.hostname if u else None
        self._port = (u.port or 554) if u else None
        self._down_since = None
        self._last_note = 0.0
        # Broken (gray) frames skipped: in total, and the share of the last full
        # minute in percent - the dashboard shows it per camera.
        self.gray_total = 0
        self.gray_pct = 0.0
        self._gray_since = None          # start of the current run of gray frames
        self._gray_note_t = time.time()
        self._gray_note_n = 0
        self._win_t = time.time()
        self._win_frames = 0
        self._win_gray = 0

    GRAY_STATUS = "the camera sends only broken (gray) frames - its stream falls behind (busy PC or network)"

    def _count(self, gray):
        """Book one frame as good or gray; keep the status and the log honest."""
        now = time.time()
        self._win_frames += 1
        if gray:
            self._win_gray += 1
            self.gray_total += 1
            self._gray_note_n += 1
            if self._gray_since is None:
                self._gray_since = now
            elif now - self._gray_since >= GRAY_ONLY_SEC and not self.status:
                self.status = self.GRAY_STATUS
        else:
            self._gray_since = None
            if self.status == self.GRAY_STATUS:
                self.status = None
        if now - self._win_t >= 60.0:
            self.gray_pct = round(100.0 * self._win_gray / max(1, self._win_frames), 1)
            self._win_t, self._win_frames, self._win_gray = now, 0, 0
        if self._gray_note_n and now - self._gray_note_t >= GRAY_NOTE_SEC:
            print(f"[camera] {self._host or mask_url(self.source)}: skipped {self._gray_note_n} broken "
                  f"(gray) frames in the last {(now - self._gray_note_t) / 60:.0f} min - the stream fell "
                  f"behind", flush=True)
            self._gray_note_t, self._gray_note_n = now, 0

    def _reachable(self, timeout=3.0):
        """Can we open a TCP connection to the camera's RTSP port at all?

        A plain connect answers in milliseconds either way, whereas handing an
        unreachable camera to FFmpeg costs a full open timeout per attempt.
        """
        try:
            with socket.create_connection((self._host, self._port), timeout=timeout):
                return True
        except OSError:
            return False

    def _open(self):
        src = self.source
        if isinstance(src, str) and src.isdigit():  # webcam index passed as string
            src = int(src)
        # Release whatever is there first. A failed open leaves a dead
        # VideoCapture behind, and the reconnect loop calls this every few
        # seconds - so overwriting without releasing leaks one FFmpeg context
        # (and its buffers) per attempt for as long as the camera is unreachable.
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        if self._rtsp and self._host and not self._reachable():
            self.status = (f"camera {self._host}:{self._port} not reachable on the network "
                           f"(no answer on the RTSP port - check its power, cable/PoE and switch)")
            return False
        if self._rtsp:
            params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MS,
                      cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MS]
            if hasattr(cv2, "CAP_PROP_N_THREADS"):
                params += [cv2.CAP_PROP_N_THREADS, DECODE_THREADS]
            self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG, params)
        else:
            self.cap = cv2.VideoCapture(src)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not self.cap.isOpened():
            self.status = (f"camera {self._host}:{self._port} answers, but the stream "
                           f"did not open (check the RTSP path and credentials)"
                           if self._rtsp else f"could not open {mask_url(self.source)}")
            return False
        return True

    def start(self):
        self._open()
        self.thread.start()
        return self

    def _went_down(self, why):
        """Say so ONCE when the stream is lost, then only every few minutes.

        Every retry used to print a line, and an outage of an hour filled the
        console with identical "(re)connecting" messages.
        """
        now = time.time()
        if self._down_since is None:
            self._down_since = now
            self._last_note = now
            print(f"[camera] {why} - reconnecting to {mask_url(self.source)} "
                  f"every {self.reconnect_delay:.0f}s", flush=True)
        elif now - self._last_note >= DOWN_NOTE_SEC:
            self._last_note = now
            print(f"[camera] still down after {(now - self._down_since) / 60:.0f} min: "
                  f"{self.status or why}", flush=True)

    def _update(self):
        # 8 Oct 2026: at 13:13 the PC ran out of memory; cap.read() raised in all
        # four reader threads, they died and the cameras stayed dark for 30 min
        # until a restart. An error now drops the stream and reconnects.
        while not self.stopped:
            try:
                self._step()
            except Exception as e:
                self.status = f"read error: {type(e).__name__}"
                self._went_down(f"read error ({type(e).__name__})")
                try:
                    if self.cap is not None:
                        self.cap.release()
                except Exception:
                    pass
                self.cap = None
                time.sleep(self.reconnect_delay)

    def _step(self):
        """One frame (or one reconnect attempt) of the reader loop."""
        if True:
            if self.cap is None or not self.cap.isOpened():
                if not self._open():
                    self._went_down(self.status or "cannot open the stream")
                    time.sleep(self.reconnect_delay)
                    return
                if self._down_since is not None:
                    print(f"[camera] connected again after "
                          f"{(time.time() - self._down_since):.0f}s", flush=True)
                    self._down_since = None
                self.status = None
            ok, frame = self.cap.read()
            if not ok:
                self.status = "the stream stopped sending frames"
                self._went_down("read failed")
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
                time.sleep(self.reconnect_delay)
                return
            self._pace_file()            # every frame, skipped or not, takes its time
            # Before the resize: a broken frame is not worth the work.
            gray = GRAY_SKIP and gray_fraction(frame) >= GRAY_FRAC
            self._count(gray)
            if gray:
                return
            if self.max_width and frame.shape[1] > self.max_width:
                s = self.max_width / float(frame.shape[1])
                frame = cv2.resize(frame, (self.max_width, int(frame.shape[0] * s)),
                                   interpolation=cv2.INTER_AREA)
            with self.lock:
                self.frame = frame
                self.ok = True
                self.seq += 1
                self.last_frame_t = time.time()

    def _pace_file(self):
        """Play a video file at its own frame rate, like a camera would.

        Unpaced, `--source clip.mp4` decoded as fast as the CPU allowed - a four
        minute recording flew past in half a minute and the rules saw a person
        jump between frames, which is nothing like the live camera.
        """
        if self._rtsp or str(self.source).isdigit():
            return
        if not hasattr(self, "_file_period"):
            fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap is not None else 0
            self._file_period = 1.0 / fps if fps and fps > 0 else 0.0
            self._next_t = time.time()
        if self._file_period:
            self._next_t = max(self._next_t + self._file_period, time.time() - 1.0)
            wait = self._next_t - time.time()
            if wait > 0:
                time.sleep(wait)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def read_new(self, last_seq):
        """(frame, seq) if a frame newer than `last_seq` has arrived, else (None, last_seq).

        read() keeps returning the last frame while the camera is gone; a caller
        that must not judge the same picture twice uses this instead.
        """
        with self.lock:
            if self.frame is None or self.seq == last_seq:
                return None, last_seq
            return self.frame.copy(), self.seq

    def frame_age(self):
        """Seconds since the newest frame arrived (inf before the first one)."""
        with self.lock:
            return time.time() - self.last_frame_t if self.last_frame_t else float("inf")

    def stop(self):
        """Stop and close the stream properly.

        release() ends the RTSP session with a TEARDOWN. A process that is simply
        killed never sends one, and this camera then refused every new connection
        for ~80 s - so each restart cost a minute and a half of dead picture.
        Wait for the capture thread first: releasing the capture while another
        thread is inside read() is how FFmpeg crashes.
        """
        self.stopped = True
        if self.thread.is_alive() and threading.current_thread() is not self.thread:
            self.thread.join(timeout=READ_TIMEOUT_MS / 1000.0 + 2.0)
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
