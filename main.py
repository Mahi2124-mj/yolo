"""Poketenashi Safety Monitor — entry point.

Pipeline per frame:
    grab frame -> YOLO11 detect+pose+track -> rule engine -> alerts -> draw -> show/save

Examples:
    python main.py                          # use camera + config.yaml
    python main.py --source 0               # test on your webcam
    python main.py --source clip.mp4        # test on a video file
    python main.py --no-show                # headless (server) mode
"""
import sys
import time
import argparse

import cv2
import yaml

from poketenashi.camera import VideoStream
from poketenashi.detector import Detector
from poketenashi.zones import Zones
from poketenashi.rules import RuleEngine
from poketenashi.alerts import AlertManager
from poketenashi import visualizer


WINDOW_NAME = "Poketenashi Monitor"


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser(description="Poketenashi Safety Monitor (YOLO11)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--source", default=None,
                    help="override source: rtsp url / video file / webcam index")
    ap.add_argument("--no-show", action="store_true", help="run headless (no window)")
    ap.add_argument("--save-video", action="store_true", help="write annotated mp4")
    ap.add_argument("--debug", action="store_true", help="overlay PO tuning numbers per person")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    if args.source is not None:
        cfg["camera"]["source"] = args.source
    if args.no_show:
        cfg["output"]["show_window"] = False
    if args.save_video:
        cfg["output"]["save_video"] = True
    if args.debug:
        cfg.setdefault("output", {})["debug"] = True

    print("[init] loading models (first run downloads weights) ...")
    detector = Detector(cfg)
    zones = Zones(cfg.get("zones"))
    rules = RuleEngine(cfg, zones)
    alerts = AlertManager(cfg)

    src = cfg["camera"]["source"]
    print(f"[init] opening source: {src}")
    stream = VideoStream(
        src,
        cfg["camera"].get("rtsp_transport", "tcp"),
        cfg["camera"].get("reconnect_delay", 3.0),
        max_width=cfg["camera"].get("max_width", 0),
    ).start()

    print("[init] waiting for first frame ...")
    ok, frame = False, None
    for _ in range(150):  # ~15s
        ok, frame = stream.read()
        if ok:
            break
        time.sleep(0.1)
    if not ok:
        print("[error] no frames received. Check the RTSP URL, network, and credentials.")
        stream.stop()
        sys.exit(1)

    show = cfg["output"].get("show_window", True)
    if show:
        # The main stream (2304x1296) is bigger than a 1920x1080 screen. The default
        # WINDOW_AUTOSIZE sizes the window to the frame, which pushes the right ~17%
        # and bottom ~23% off-screen (that is where the walkway is). WINDOW_NORMAL
        # scales the frame to fit instead, so the whole field of view stays visible.
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1600, 900)
    writer = None
    t_prev, fps = time.time(), 0.0
    print("[run] monitoring started. Press 'q' in the window to quit (Ctrl+C if headless).")

    try:
        while True:
            ok, frame = stream.read()
            if not ok:
                time.sleep(0.05)
                continue

            persons, phones = detector(frame)
            violations = rules.check(persons, phones, frame)
            alerts.handle(violations, frame)

            now = time.time()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            show_trail = cfg["rules"].get("PO", {}).get("draw_wrist_trail", False)
            show_hands = cfg["output"].get("draw_hand_skeleton", False)
            vis = visualizer.draw(frame.copy(), persons, phones, violations, zones, cfg, fps,
                                  po_dbg=rules.po_dbg,
                                  wrist_trail=rules.wrist_trail if show_trail else None,
                                  hand_skeletons=rules.current_hands() if show_hands else None)

            if cfg["output"].get("save_video", False):
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(
                        cfg["output"]["output_video"],
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        cfg["camera"].get("target_fps", 20), (w, h),
                    )
                writer.write(vis)

            if show:
                cv2.imshow(WINDOW_NAME, vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\n[run] interrupted by user.")
    finally:
        stream.stop()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print("[done] stopped.")


if __name__ == "__main__":
    main()
