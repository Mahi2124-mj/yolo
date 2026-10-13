"""Raw recording of one camera (ffmpeg stream copy, segments) with reconnect, WITHOUT
importing the app (tools.night_record pulls in OpenCV/torch: ~830 MB each, which
filled the PC's memory on 7 Oct). The camera address comes from
monitor_cameras.json and is never printed or logged.

    python -m tools.light_record --cam cam1 --until 09:00 --out D:/poketenashi-data/night --segment-min 30
"""
import argparse
import datetime as dt
import json
import os
import re
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def mask(s):
    return re.sub(r"rtsp://[^\s\"']+", "rtsp://***", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", default="cam1")
    ap.add_argument("--until", default="09:00")
    ap.add_argument("--out", default="D:/poketenashi-data/night")
    ap.add_argument("--segment-min", type=int, default=30)
    a = ap.parse_args()
    src = next(c["source"] for c in json.load(open(os.path.join(ROOT, "monitor_cameras.json"), encoding="utf-8"))["cameras"]
               if c["id"] == a.cam)
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    now = dt.datetime.now()
    h, m = map(int, a.until.split(":"))
    end = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if end <= now:
        end += dt.timedelta(days=1)
    out = os.path.join(a.out, now.strftime("%Y%m%d"), "raw")
    os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, "record.log"), "a", encoding="utf-8")
    log.write(f"{now:%Y-%m-%d %H:%M:%S} recording {a.cam} until {end:%Y-%m-%d %H:%M} -> {out}\n")
    log.flush()
    while dt.datetime.now() < end:
        left = int((end - dt.datetime.now()).total_seconds())
        cmd = [ff, "-hide_banner", "-loglevel", "error", "-rtsp_transport", "tcp", "-timeout", "15000000", "-i", src,
               "-c:v", "copy", "-an", "-t", str(left), "-f", "segment", "-segment_time", str(a.segment_min * 60),
               "-reset_timestamps", "1", "-strftime", "1", os.path.join(out, f"{a.cam}_%Y%m%d_%H%M%S.mp4")]
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        for line in (r.stdout + r.stderr).splitlines()[-5:]:
            log.write(f"{dt.datetime.now():%H:%M:%S} ffmpeg: {mask(line)}\n")
        log.write(f"{dt.datetime.now():%H:%M:%S} ffmpeg ended (code {r.returncode})\n")
        log.flush()
        if dt.datetime.now() < end:
            time.sleep(10)                         # camera dropped: reconnect
    log.write(f"{dt.datetime.now():%H:%M:%S} finished\n")


if __name__ == "__main__":
    main()
