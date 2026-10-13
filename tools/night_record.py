"""Raw overnight recording of one camera, for replay and training (no overlays:
the alert clips have the boxes burned in and must never be trained on).

ffmpeg stream copy (no re-encode, ~no CPU) in 30-minute files named by their
start time, until the end time; if the camera drops, it reconnects and goes on.
The camera address (with its login) comes from monitor_cameras.json and is
never printed. Uses the camera's second RTSP session (the app holds the first).

    python -m tools.night_record --cam cam1 --until 09:00 --out D:/poketenashi-data/night
"""
import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from poketenashi.camera import mask_url  # noqa: E402


def source(cam_id):
    cams = json.load(open(os.path.join(ROOT, "monitor_cameras.json"), encoding="utf-8"))["cameras"]
    return next(c["source"] for c in cams if c["id"] == cam_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", default="cam1")
    ap.add_argument("--until", default="09:00", help="HH:MM, the next time it comes round")
    ap.add_argument("--out", default="D:/poketenashi-data/night")
    ap.add_argument("--segment-min", type=int, default=30)
    a = ap.parse_args()
    now = dt.datetime.now()
    h, m = map(int, a.until.split(":"))
    end = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if end <= now:
        end += dt.timedelta(days=1)
    out = os.path.join(a.out, now.strftime("%Y%m%d"), "raw")
    os.makedirs(out, exist_ok=True)
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    src = source(a.cam)
    log = open(os.path.join(out, "record.log"), "a", encoding="utf-8")
    log.write(f"{now:%Y-%m-%d %H:%M:%S} recording {a.cam} until {end:%Y-%m-%d %H:%M} -> {out}\n")
    log.flush()
    while dt.datetime.now() < end:
        left = int((end - dt.datetime.now()).total_seconds())
        # -timeout: a camera that stops sending (6-7 Oct: 06:04) must not leave
        # ffmpeg hanging for hours - give up after 15 s and reconnect
        cmd = [ff, "-hide_banner", "-loglevel", "error", "-rtsp_transport", "tcp", "-timeout", "15000000",
               "-i", src, "-an",
               "-c:v", "copy", "-t", str(left), "-f", "segment", "-segment_time", str(a.segment_min * 60),
               "-reset_timestamps", "1", "-strftime", "1",
               os.path.join(out, f"{a.cam}_%Y%m%d_%H%M%S.mp4")]
        # ffmpeg prints the input URL (login included) in its errors: mask it
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                           creationflags=0x00004000)                                # BELOW_NORMAL
        for line in (r.stdout + r.stderr).splitlines():
            log.write(re.sub(r"rtsp://\S+", lambda m: mask_url(m.group(0)), line) + "\n")
        log.write(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} ffmpeg exit {r.returncode}\n")
        log.flush()
        if dt.datetime.now() < end:
            time.sleep(15)                      # camera dropped: wait, reconnect
    log.write(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} done\n")


if __name__ == "__main__":
    sys.exit(main())
