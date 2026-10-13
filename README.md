# Poketenashi Safety Monitor (YOLO11)

Real-time monitoring of the **5 Poketenashi safe-walking rules** on a Panasonic
RTSP bullet camera, using Ultralytics **YOLO11** for person detection, pose
keypoints, and tracking, plus a custom rule layer.

| Code | Rule (JTEKT poster) | How it's detected |
|------|---------------------|-------------------|
| **PO**  | Do not put hands in pockets | Pose geometry (elbow bend / forearm tuck / wrist height) + MediaPipe fingertip veto |
| **KE**  | No phone while walking (stop first) | Phone detected near a person who is moving |
| **TE**  | Hold the handrail on stairs | In `stairs` zone, wrist must be in `handrail` zone |
| **NA**  | Use the sides of roads | Foot position inside the `walkway_center` zone = violation |
| **SHI** | Confirm the situation | Must pause inside a `checkpoint` zone *(experimental)* |

## Architecture

```
RTSP frame
  ├─ YOLO11-pose  (person bbox + 17 keypoints + ByteTrack IDs)  -> PO, TE, SHI
  ├─ YOLO11       (cell-phone detection)                         -> KE
  ├─ tracker      (motion history per person)                    -> KE, NA, SHI
  └─ rule layer   (your zones + thresholds in config.yaml)       -> violations
            └─ alerts: console + violations.jsonl + snapshots/
```

## Install

Already set up in a local venv (`.venv`, Python 3.12). To recreate:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
# live camera (uses the RTSP url in config.yaml)
.\.venv\Scripts\python.exe main.py

# test on the webcam or a recorded clip first (recommended)
.\.venv\Scripts\python.exe main.py --source 0
.\.venv\Scripts\python.exe main.py --source clip.mp4

# headless / server (no preview window)
.\.venv\Scripts\python.exe main.py --no-show --save-video
```

Press **q** in the window to quit (Ctrl+C when headless).

## Define zones (needed for TE, NA, SHI)

These rules are location-specific, so mark the areas on a frame from the camera:

```powershell
.\.venv\Scripts\python.exe -m tools.zone_drawer --zone walkway_center
.\.venv\Scripts\python.exe -m tools.zone_drawer --zone stairs
.\.venv\Scripts\python.exe -m tools.zone_drawer --zone handrail
.\.venv\Scripts\python.exe -m tools.zone_drawer --zone checkpoint
```

Left-click to add polygon points, `s` to save into `config.yaml`. Until a zone
is drawn, its rule is skipped (PO and KE work with no zones at all).

## How PO (hands in pockets) works

A hand hanging at the hip while walking and a hand inside a pocket both put the
wrist near the hip, so distance alone gives false positives. PO uses two stages:

1. **Geometry pre-filter** (cheap, every person): wrist near hip **and** ≥2 of
   - `bent` — elbow flexed (pocket), not a near-straight hanging arm
   - `tucked` — forearm angles inward, wrist pulled toward the hip
   - `not_hanging` — wrist at hip/pocket level, not dropped to the thigh
2. **Fingertip veto** (MediaPipe HandLandmarker, only on candidates): crop around
   the wrist and look for a hand. **Fingertips visible → hand is OUT → cancel the
   PO.** "No hand seen" never *creates* a PO (a low-res stream can miss real hands),
   it only fails to veto — so geometry still decides those.

Model: `models/hand_landmarker.task` (downloaded once). Disable the whole veto
with `rules.PO.use_fingertips: false` for a faster, geometry-only PO.

### Master-match mode (recommended — "golden sample" style)

Instead of hand-tuned thresholds, compare each live hand to YOUR labelled
reference samples (few-shot, no training). Set `rules.PO.method: master`, then:

```powershell
# 1) collect references: demonstrate in front of the camera
.\.venv\Scripts\python.exe -m tools.capture_masters
#    p = hand IN pocket   o = hand OUT   u = undo   q = save&quit   (aim ~20-40 of each)

# 2) run - PO now matches live hands to your samples
.\run.bat --debug        # overlay shows  pocket:0.82 / out:0.71
```

Each wrist crop is embedded with MobileNetV3 (GPU) and the k nearest reference
samples vote. Add more varied samples (people, lighting, angles) to improve it,
then `python -m tools.build_masters` rebuilds the library from the saved crops.
If the library is empty it falls back to the fingertip/geometry method.

## Tuning

All thresholds live in `config.yaml` under `rules:`. Common adjustments:

- **Too many false PO alerts** → lower `PO.wrist_hip_ratio` (e.g. 0.40).
- **KE not firing** → lower `KE.walking_speed_px`, or the sub-stream may be too
  low-res for the phone to be detected — switch to the `main` stream.
- **Alerts repeat too fast** → raise `alerts.cooldown_sec`.

## Files

```
main.py                 entry point / main loop
config.yaml             camera, models, rule thresholds, zones
cameras.json            Panasonic camera details
poketenashi/
  camera.py             threaded RTSP capture with auto-reconnect
  detector.py           YOLO11 pose + object detection wrapper
  zones.py              polygon zone tests
  rules.py              the 5 Poketenashi rule checkers
  alerts.py             debounce + logging + snapshots
  visualizer.py         on-frame overlay
tools/
  zone_drawer.py        interactive zone drawing
  test_pipeline.py      headless smoke test (no camera needed)
```

## Notes / limitations

- The **sub** stream (`/sub/`) is low-res; for reliable phone (KE) detection use
  the **main** stream — change `camera.source` in `config.yaml`.
- **SHI** is a heuristic (pause = "confirmed"); real point-and-call gesture
  recognition would need a trained temporal-action model.
- For best accuracy on PPE/site-specific gear, fine-tune YOLO11 on your own
  labeled footage and point `models.pose` / `models.detect` at the new weights.
