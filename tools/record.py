"""Record RAW camera footage, for replaying rule changes against real events.

Why this exists: every threshold in this project was being tuned against 15
minute live windows, and two windows are never comparable - different people,
different traffic. Worse, the evidence clips are useless as test input: they are
five seconds long, already have boxes painted on them, and are written at the
clip writer's width rather than the pipeline's. Replaying those produced a
before/after where BOTH sides read zero, which says nothing at all.

So this records what the pipeline actually sees:
  * the same downscale (camera.max_width) the live loop runs on, so a replay is
    faithful rather than approximately similar,
  * no overlay of any kind,
  * a fixed frame rate stamped into the file, so the replay clock can advance in
    video time instead of CPU time.

Usage:
    python -m tools.record --minutes 15
    python -m tools.record --minutes 5 --fps 10 --out footage/lunch.mp4
"""
import argparse
import os
import subprocess
import sys
import time

import cv2
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from poketenashi.camera import VideoStream          # noqa: E402


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Record raw footage for replay.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--source", default=None, help="override the camera source")
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--fps", type=float, default=12.0,
                    help="frames per second to sample and stamp into the file")
    ap.add_argument("--out", default=None, help="output mp4 (default footage/<time>.mp4)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    src = args.source or cfg["camera"]["source"]
    max_width = cfg["camera"].get("max_width", 0)

    out = args.out or os.path.join("footage", time.strftime("%Y%m%d_%H%M%S") + ".mp4")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    exe = ffmpeg_exe()
    if not exe:
        print("[record] imageio-ffmpeg is missing. Install it with:"
              "\n    .venv\\Scripts\\python.exe -m pip install imageio-ffmpeg")
        sys.exit(1)

    print(f"[record] source     : {src}")
    print(f"[record] max_width  : {max_width or 'unchanged'}  (same as the live pipeline)")
    print(f"[record] target fps : {args.fps}")
    print(f"[record] duration   : {args.minutes} min")
    print(f"[record] output     : {out}")

    stream = VideoStream(src, cfg["camera"].get("rtsp_transport", "tcp"),
                         cfg["camera"].get("reconnect_delay", 3.0),
                         max_width=max_width).start()
    print("[record] waiting for the first frame ...")
    frame = None
    for _ in range(150):
        ok, frame = stream.read()
        if ok:
            break
        time.sleep(0.1)
    if frame is None:
        print("[record] no frames - check the URL, network and credentials.")
        stream.stop()
        sys.exit(1)
    h, w = frame.shape[:2]
    print(f"[record] frame size : {w}x{h}")

    cmd = [exe, "-y", "-loglevel", "error",
           "-f", "image2pipe", "-c:v", "mjpeg", "-framerate", str(args.fps), "-i", "-",
           "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, creationflags=flags)

    # Sample on a fixed grid rather than "as fast as possible". The stamped fps
    # then means something, and the file plays back at the speed it was shot.
    period = 1.0 / args.fps
    deadline = time.time() + args.minutes * 60.0
    nxt = time.time()
    n = 0
    try:
        while time.time() < deadline:
            now = time.time()
            if now < nxt:
                time.sleep(min(period, nxt - now))
                continue
            nxt += period
            if now - nxt > 1.0:          # fell far behind; resync instead of racing
                nxt = now + period
            ok, frame = stream.read()
            if not ok:
                continue
            ok_enc, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok_enc:
                continue
            proc.stdin.write(buf.tobytes())
            n += 1
            if n % int(max(args.fps, 1) * 30) == 0:
                left = int(deadline - time.time())
                print(f"[record] {n} frames, {left}s left", flush=True)
    except KeyboardInterrupt:
        print("\n[record] stopped early by Ctrl+C")
    finally:
        stream.stop()
        try:
            proc.stdin.close()
            _, err = proc.communicate(timeout=120)
            if proc.returncode != 0:
                print("[record] ffmpeg failed:", err.decode(errors="ignore")[:300])
        except Exception:
            proc.kill()

    size = os.path.getsize(out) / 1e6 if os.path.exists(out) else 0
    print(f"[record] wrote {n} frames ({n / args.fps:.0f}s of video), {size:.1f} MB -> {out}")
    print(f"[record] replay it with:  python -m tools.replay {out}")


if __name__ == "__main__":
    main()
