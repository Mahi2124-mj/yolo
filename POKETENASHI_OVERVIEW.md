# Poketenashi Safety Monitor — Complete Project Overview

> **Purpose of this document:** full context for an AI assistant (e.g. ChatGPT) or a new engineer.
> **State as of:** 3 October 2026, ~11:30 (IST).
> **Code location:** `C:\Users\DX-ADMIN\Pictures\yolo` (the live copy).
> `README.md`, `PROJECT_CONTEXT.md` and `PO_KE_OVERVIEW.md` in that folder are **stale** (June / early Sep). This file supersedes them.
> **No passwords or camera credentials appear in this document** (camera addresses are shown masked).

---

## 1. What the system does

A factory safety monitor that watches CCTV cameras and flags people breaking the Japanese workplace safety rules **"Po-Ke-Te-Na-Shi"** (ポケテナシ), plus three extra rules:

| Code | Japanese origin | Meaning (what is a violation) |
|---|---|---|
| **PO** | ポケット (pocket) | Hand(s) in pockets while walking or standing in the gangway |
| **KE** | 携帯 (keitai, phone) | Using a phone while walking / in the gangway (looking at it, or a call at the ear) |
| **TE** | 手すり (tesuri, handrail) | Not holding the handrail on stairs |
| **NA** | 斜め (naname, diagonal) | Crossing a walkway diagonally instead of along / straight across |
| **SHI** | 指差し確認 (shisa kakunin) | Not stopping/looking around at a checkpoint (pointing-and-calling confirmation) |
| **RUN** | (added) | Running |
| **FALL** | (added) | A person fallen / lying on the floor |
| **RESTRICTED** | (added) | A person inside a restricted area |

How it works: each camera's RTSP stream goes through these stages. Everything runs on CPU in one Python process.
1. YOLO11 **pose** detection finds people and their 17 body keypoints.
2. **ByteTrack** gives each person a track ID.
3. A per-camera **rule engine** judges each person.
4. Each violation that starts produces an **alert**: a log row, an evidence clip with a poster image, a live banner on the dashboard, and an entry in the TV view's list.
5. A browser **dashboard** (port 6002) shows live views, evidence, charts, cameras, zones, settings and users. A separate full-screen **TV page** shows the live picture with violations at the side.

---

## 2. Site, hardware, network

**PC:**
- Windows 11 Pro, Intel **i7-14700** (20 cores / 28 threads), **16 GB RAM**. RAM is tight: the PC also runs other apps.
- CPU-only inference.
- Total CPU sits at ~80–86% with two detecting cameras plus the other apps.

**Python:** `.venv\Scripts\python.exe` (Python **3.12**).
- Never use the system `python` (3.14).
- Install packages with `.venv\Scripts\python.exe -m pip ...` only: the venv was copied from another folder, so `pip.exe` and `activate.bat` point to the wrong place.

**Other software on this PC — never stop or touch it:**
- an "EOL system"
- a "loop pipe" app
- a separate `python app.py` (Python 3.11, ~1.9 GB)

**Cameras** (stored in `monitor_cameras.json`; credentials never shown):

| id | Name | Address (masked) | Notes |
|---|---|---|---|
| cam1 | Camera 1 | `rtsp://<user>:<pw>@192.168.4.15:554/cam/realmonitor?channel=1&subtype=0` | Dahua-type ceiling camera looking down the main **aisle** (gangway). 1920×1080. Rules on. Has `gangway` (4 corners) and `crossing` (6 corners) zones. |
| cam2 | Dx | `rtsp://<user>:<pw>@192.168.30.241:554/h264/ch1/main/av_stream` | Older generic-OEM camera over a **desk / floor area**. The main stream is 2304×1296, downscaled to 1920 (`camera.max_width`); the sub stream `/h264/ch1/sub/av_stream` is 704×576. The Dahua path `/cam/realmonitor…` does **not** work on it: the camera answers 451. Rules on. **No zones drawn yet.** |

**Network:**
- **Ethernet** 192.168.30.81 (static). Routes to the cameras' 192.168.4.0/24 via 192.168.30.2.
- **Wi-Fi** (USB dongle, DHCP) on SSID **TBDI-DX** = 192.168.100.32.
- **Wall TV:** Samsung Tizen 7.0 browser (Chrome 94) at **192.168.100.56**. It can reach the PC **only** at `http://192.168.100.32:6002`; 192.168.30.81 is unreachable from it.
- Windows Firewall is off on all profiles.
- A permanent fix for the TV link (DHCP reservation, or a cable for the TV) is IT's job.

---

## 3. Architecture

```
            ┌──────────────── per camera (CameraRuntime) ────────────────┐
RTSP ──► VideoStream thread ──► newest frame ──► LiveView thread (~15 fps)
         • TCP, auto-reconnect           │        • draws boxes/keypoints/zones
         • 1 FFmpeg decode thread        │        • JPEG → MJPEG for browsers
         • skips broken gray frames      │        • feeds the ClipRecorder ring
         • downscale > 1920 px           ▼
                      main detection loop (round-robin over cameras)
                      Detector (yolo11n-pose + ByteTrack, own tracker per camera)
                      → RuleEngine (own per camera) → AlertManager (episodes)
                      → violations.jsonl + clip + poster + live alert
            └────────────────────────────────────────────────────────────┘
Shared by all cameras: the KE phone model (yolo11m), MediaPipe HandLandmarker, the PO pocket model (when enabled)
HTTP: stdlib ThreadingHTTPServer on :6002 (dashboard, TV page, JSON API, MJPEG)
```

- **Entry point:** `web.py`. It is started by `watchdog.bat` (`run.bat` only calls `watchdog.bat`).
  - The watchdog restarts the app after a crash, on exit code 3 (hang) and on exit code 4 (restart requested). Exit code 0 stops it.
  - A hang guard exits with code 3 if the main loop stalls for 120 s.
- **Detector** (`poketenashi/detector.py`):
  - `yolo11n-pose.pt` in low-memory mode: imgsz 480, conf 0.35, `torch_threads 4`.
  - Tracker: `trackers/bytetrack_lowfps.yaml` (high 0.25, low 0.1, new 0.25, buffer 30, match 0.9).
  - Only tracker-confirmed people are judged.
- **Latency fix (28 Sep):** OpenCV's FFmpeg defaulted to 16 decode threads, and frame threading added ~0.75 s of delay. `camera.py` forces `CAP_PROP_N_THREADS = 1`. The live view now lags only the camera's own ~0.6 s.
- **Broken-frame skip (3 Oct):**
  - Under CPU load the Dx 3 MP stream sometimes decodes as flat **gray** frames, because reference frames are lost. At one point 24% of sampled frames were gray.
  - `camera.gray_fraction()` checks a 6×8 grid of blocks on every 16th pixel (~0.1 ms per frame). A frame is skipped when ≥ 30% of the blocks are flat mid-gray (block std < 1.5, mean 128 ± 10 in every channel). Measured: real frames 0%, broken frames 88–100%.
  - Skipped frames are never shown, judged or recorded.
  - If only gray frames arrive for 5 s, the camera's status says so; after 10 s the camera shows as offline.
  - The dashboard shows "X% broken frames skipped" per camera.
- **Multi-camera** (since 3 Oct):
  - Each camera has its own CameraRuntime: stream, live view, detector/tracker (track IDs are per camera), rules, alert manager and clip recorder. Heavy models are shared.
  - Cameras can be added, changed or removed **live** from the dashboard; this is applied between frames, with no restart.
  - Up to 8 cameras.
  - Detection is round-robin, so **two detecting cameras each get about half the FPS**:
    - cam1 alone: ~16–20 fps with nobody in view, ~6–7 fps with a person.
    - With Dx also detecting: ~8–9 fps with nobody in view, ~4–5 fps with a person.

---

## 4. Rules — detailed logic (all 8 are enabled in `config.yaml → rules.enabled`)

**Zone gate** (`rules.zone_gate`): PO, KE, RUN and FALL judge **only people whose feet are inside `gangway` or `crossing`** (15 px margin). People at the racks or machines outside are drawn but never flagged.
- If a camera has neither zone drawn, the gate is off and **everyone is judged** (Dx right now).
- TE, SHI and RESTRICTED use their own zones. NA looks at the gangway only.

**Zones** (drawn per camera in the dashboard's Zone editor):
- `gangway` — the aisle floor. Must have **exactly 4 corners**; NA uses it for a top-down view.
- `crossing` — the side gangways / T-junction under the camera.
- `stairs` and `handrail` — for TE.
- `checkpoint` — for SHI.
- `restricted` — for RESTRICTED.

cam1 has only `gangway` and `crossing`. **TE, SHI and RESTRICTED are therefore inactive (skipped) on every camera** until their zones are drawn.

**Multi-person:** every person breaking a rule gets their **own red box and their own alert**, even when people stand side by side.

### PO — hands in pockets
**The user's definition (must be respected):**
- PO = a hand in a pocket while **walking OR standing**, in the gangway (the side gangways count too).
- **ONE hand is enough.** The other hand may move or hold something. Never require both hands.
- Sitting in a work area is not PO.
- A fired PO must produce an alert plus an evidence clip.
- The user rejected over-engineered gates that killed real PO ("faltu cheeje"). "Not PO" filters are only acceptable if real PO alerts keep firing.

**Current logic:** `poketenashi/rules.py → _rule_PO_geom` (the 10 Sep logic) plus the 28 Sep "not PO" filters. The method is `fingertips`.
1. Only tracked people inside the gangway/crossing. Keypoints need conf ≥ 0.4. Lengths are measured in torso lengths (`torso`).
2. **Not judged** (the frame counts as "no pocket"):
   - The person is cut off at the bottom of the frame: bbox bottom ≥ 99% of the frame height and pelvis + 0.5 torso below that. The pocket is not in the picture.
   - **Both hands together in front** of the body: wrists < 0.35 torso apart, in a frontal view (shoulder width ≥ 0.45 torso). These are clasped hands or hands holding something.
3. **Per arm, the pocket test** passes when all of these hold:
   - **near:** the wrist is inside a pelvis box (±0.28 body-scale horizontally, ±0.32 vertically), or the wrist–hip distance is < 0.55 torso.
   - **at_hip:** the wrist is no more than 0.15 torso above the hip (`wrist_above_hip_max`). A hand at the belly or chest holds a phone or paper, not a pocket.
   - **still:** the body-relative wrist movement over the last 15 frames is < **0.07** (`wrist_move_thresh`). There must be enough history first (`still_needs_history`).
   - **shape cues ≥ 2 of 3:** elbow angle between 70° and 140°; wrist tucked toward the hip (tuck > 0.10); arm not hanging down the thigh (hang < 0.35).
   - **no visible hand:** MediaPipe HandLandmarker runs on a wrist crop (radius 0.75 torso, upscaled to 448 px, det conf 0.15, every 2nd frame). A hand with ≥ 1 fingertip within 0.65 torso of the wrist means "visible", so not a pocket.
4. The pocket test passes for the person when **any one arm** passes.
5. **Confirmation** (`_po_decide`): PO is shown once the test has passed in **≥ 60% of the frames of the last 1.0 s, and in ≥ 3 frames**. It then **holds 1.5 s** after the last pocket frame, so a blink does not drop it.
6. **Optional pocket model** (`poketenashi/po_model.py`, `models/po_arm_mnv3.pt`):
   - A fine-tuned MobileNetV3-Small scores an arm crop as P(pocket). With it, `P ≥ 0.6` replaces the still / cues / MediaPipe checks.
   - It is **trained and integrated but NOT enabled**: `rules.PO.model_path` is not set. Its replay validation was never finished.
   - Training data: `D:\poketenashi-data\po_arms` (labelled arm crops).

**Known limits and history:**
- **Missed pockets:**
  - Pockets **while walking** are mostly missed. A walking pocketed wrist moves 0.06–0.19, about the same as a free swinging hand (0.06–0.09), so motion cannot separate them. Standing pockets move 0.05–0.07.
  - The planned fix is the appearance (pocket) model above.
- **False positives seen in the past:**
  - gloved trolley pushers (MediaPipe finds no fingers)
  - people holding a phone, papers or a box
  - people seen from behind
  - people cut off at the frame bottom
  - one-frame glimpses
- **24–26 Sep:** an elaborate PO version with many filters was built. The user ordered it **reverted on 28 Sep** ("old PO worked better"), so the current code is the 10 Sep base plus the filters above.
- **Validation clips — always check both directions:**
  - `footage/po_test_20260925.mp4` (~165–260 s, track 4): a tester standing with both hands in pockets. It **must fire**.
  - `footage/live_test_20260928.mp4`: the 28 Sep 13:01–13:05 live test where testers broke rules on purpose.
  - The 28 Sep false-alert clips must **not** fire.

### KE — phone while walking
**Logic** (`poketenashi/ke.py`, redesigned 26 Sep): fires only when a **phone is actually SEEN in the hand**.
1. **Raised wrists:** wrists at least 0.05 torso above the hip line, i.e. belly, chest or ear level.
2. **Zoom:**
   - A crop around the raised hand(s) and the space above them is cut from the full-resolution frame and resized to **256 px**.
   - **yolo11m** (COCO "cell phone" class) runs on it at conf **0.40**.
   - A full-frame phone search at 1280 px took 70% of the CPU and was removed.
3. **Box sanity:**
   - The phone box must be ≤ **0.5 torso** long (paper or a trolley handle boxes at 0.65–0.82; real phones at 0.06–0.46). This cap is not applied at the ear.
   - The box must be within 0.6 torso of a raised wrist.
   - **Ownership:** each phone box belongs only to the person whose wrist is nearest. A neighbour's phone used to flag both people.
4. **Phone seen:** found in **≥ 2 frames within 3 s**.
5. **Position:**
   - The person is walking (≥ 45 px/s) **or** in the gangway, **and** upright (head above the shoulders).
   - The person must **face the camera** (nose/eyes visible, conf ≥ 0.5): a phone in front of the body is invisible from behind. The exception is the ear-call path.
6. **Ear call:**
   - The call pose (wrist at the ear, beside the face, above the shoulder line) is held for **≥ 70% of 1.5 s**.
   - Plus either **≥ 2 phone boxes ≥ 0.25 conf near the head within 4 s**, or a phone seen in the hand within the last 8 s.
   - It holds while the hand stays at the ear.
7. **Latch (hold on evidence):** once fired, KE stays on while either is true:
   - the phone was seen in the last 3 s;
   - the hand stays raised (down for less than 1.5 s) and the phone was seen within the last 10 s.
   An ear call holds for as long as it lasts.
8. **Removed on 26 Sep:** the pose-only path (hands together at the chest ⇒ KE). About 25 of 33 such alerts were paper, trolleys, bags, gloves or clasped hands.

**Known limits:**
- COCO models (and YOLO-World) cannot see phones held low in the hand, edge-on, or covered by fingers from this ceiling angle. KE misses those rather than guess.
- One ear call was missed because the hand was in front of the face, so no call pose was detected.
- False alerts seen: fixing hair, a worker reaching up.
- Real recall needs a phone detector trained on this camera's images.

### NA — diagonal crossing
**Logic** (`poketenashi/na.py`): NA is about **direction** only. Walking along the aisle is OK and crossing straight over is OK; only an **angled** path is a violation.
1. **Top-down view:** the 4-corner `gangway` polygon defines a homography to a top-down view of the aisle (`aisle_length_ratio: 8`). Without it, perspective slants walking along the aisle by 20–30°, and that was the main source of false NA.
2. **Heading:** measured over a **1.5 s** window, only after the person has moved **≥ 0.8 torso**. The angle to the aisle axis is folded into 0–90°.
3. **Thresholds:** ≤ **30°** is along (OK); ≥ **65°** is straight across (OK). In between is a diagonal.
4. **Confirmation:**
   - The diagonal must hold for **0.8 s**.
   - It must carry the person **≥ 30% of the aisle width across** (`min_cross`). Someone stepping in from the racks and turning to walk along is a merge, not a crossing.
5. **Junction:** inside the `crossing` zone (the T-junction side gangways), and for 2 s after leaving it, nobody is judged. Walking left/right there, or turning into the aisle, is legal.
6. Only feet inside the gangway count.

**Results:** on 88 live NA clips, flags dropped from 59 to 26, and every remaining flag checked was a real diagonal. The site genuinely has many diagonal crossings between the racks and the green lane.

### RUN — running
- Speed in **torso lengths per second** over a 0.5 s window. This is independent of the pipeline's frame rate.
- Fires when speed ≥ **5.0 torso/s** for **6 consecutive frames**.
- Gated to the gangway.

### FALL — person on the floor
1. **Primary signal:** the shoulders→hips vector tilted **≥ 60°** from vertical.
2. **Fallback**, used only when the torso keypoints are not visible: bbox width/height ≥ **1.3**, and only if the box is **not clipped by the frame edge**. People cut off at the edge used to look "wide".
3. **Legs veto:** if the legs still go down from the hips close to vertical (< 35°, feet well below the hips), the person is **bending over** (lifting a crate), not fallen. All 19 old FALL alerts were bend-overs.
4. Needs **12 consecutive frames** (~2 s).
5. Gated to the gangway.

### TE — handrail on stairs
- A person whose feet are in `stairs` and whose wrists are not inside `handrail` for **6 consecutive frames** is a violation.
- **Inactive:** no `stairs` / `handrail` zones are drawn.

### SHI — safety confirmation at a checkpoint (experimental)
- A person in `checkpoint` must either **pause** (8 frames slower than 1.5 px/frame) or **look around** (head-yaw swing ≥ 0.25 over 25 frames). Otherwise SHI fires.
- **Inactive:** no `checkpoint` zone is drawn.

### RESTRICTED — restricted area
- Feet inside `restricted` for **3 consecutive frames** is a violation.
- **Inactive:** no `restricted` zone is drawn.

---

## 5. Alerts, evidence, data

**Episodes** (`poketenashi/alerts.py`): **one alert per incident**, not one per frame.
- An incident ends when its violation has not been seen for **8 s** (`cooldown_sec`).
- If it is still going on, it is re-alerted every **60 s** (`repeat_sec`).
- A new track ID overlapping a running incident (IoU ≥ 0.4) counts as the same person **only if the incident's own track has disappeared** (a real ID switch). Otherwise it is a second person with their own alert.

**Log:** `violations.jsonl`, one row per alert, with these fields:
- `time`, `track_id`, `rule`, `desc`
- `camera`, `camera_name`
- `clip`, `poster` (and `snapshot` if enabled)

Files of cameras other than cam1 carry the camera id in their name.

**Clips:**
- Location: `D:/poketenashi-data/clips`. 4 s before plus 4 s after the alert, 8 fps, max 1280 px wide, max 20 s.
- The clip is kept recording while the same violation continues.
- A **poster** JPEG of the alert frame is saved with it.
- Evidence renders show boxes and labels only, without keypoints.

**Snapshots:** `save_snapshots: false`; the poster images are used instead.

**Training samples** (`poketenashi/sampler.py`): crops of people near the camera with a hand at the hip (PO) or raised (KE) are saved **whether or not a rule fired**, for labelling and training.
- Location: `D:/poketenashi-data/samples`.
- Torso ≥ 90 px, every 1.5 s, 4 per person, at most 4000 per day.
- First camera only.

**Retention:** evidence (clips + snapshots) and log rows older than **2 days** are deleted. The sweep runs every 30 min. Configurable in the dashboard.

---

## 6. Dashboard (http://PC:6002/)

The front end is `web/index.html`, `web/app.css`, `web/ui.js` (popups, confirm, toasts, API helper) and `web/app.js` (pages). It is dark and production-styled, with a sidebar. **Viewing is open to the LAN; every change needs an admin login and goes through a confirmation popup that shows before → after.**

- **Dashboard:**
  - KPI cards:
    - violations today, compared with yesterday at the same time
    - last 24 h, with the busiest hour
    - cameras online
    - people in view
  - Camera tiles; violations by rule (today / all time); hourly activity chart (24 h); latest violations; by camera; system health.
- **Live view:**
  - Per-camera MJPEG view or an "All cameras" grid; snapshot download; full screen.
  - **Overlays** (keypoints, hand skeletons, zones, wrist trail, brighten, sharpen) are saved to `config.yaml` with confirmation.
  - "Recent on this camera" list.
- **Evidence:** filter by camera and rule; grouped by day; a viewer with the clip, prev/next and download; "load more".
- **Cameras:**
  - Each camera card shows: status, rules on/off, masked address, resolution / detection fps / picture fps / people, % broken frames skipped, and offline reasons with a hint.
  - **Add a camera by IP + login (no RTSP URL needed):** "Find stream" makes the server probe the camera with RTSP DESCRIBE (`poketenashi/rtsp_probe.py`, Digest/Basic auth).
    - It checks the login once, then tries the stream paths of common brands: Dahua, Hikvision, generic OEM `/h264/ch1/...`, Uniview, Panasonic, Axis, generic.
    - It shows the streams it found with preview pictures and resolutions.
  - **Edit → Find stream** fixes a wrong path using the **saved** login; the browser never sees the password.
  - Switch a camera on/off, remove a camera.
- **Zone editor** (per camera, inside the dashboard):
  - Draw on the clean camera picture: click to add a corner, drag to move, click an edge to insert, right-click or Del to delete, Ctrl+Z / Ctrl+Y.
  - Hide or show individual zones; warnings (e.g. the gangway must have 4 corners).
  - Save shows a **diff popup** (zone: old → new corners).
  - A warning appears before leaving with unsaved changes.
  - Changes apply live; config.yaml is backed up; cam1's zones are mirrored into config.yaml.
- **Settings:**
  - Retention, with a **preview of how many files would be deleted** before saving.
  - Live picture (overlays).
  - Users: add, change role, reset password, remove. The last admin is protected.
  - My account: change your own password, then other sessions are logged out.
  - **Audit log:** who changed what and when, including logins and failed logins.
  - System: version, uptime, disk, evidence size, LAN addresses.
- **TV view page** in the dashboard: TV addresses with copy buttons, and per-camera TV links.
- **Sound** toggle: a chime on new violations, set per computer. Live violation toasts appear at the bottom right.

**Server** (`poketenashi/webserver.py`, stdlib; no FastAPI to save RAM). Main endpoints:
- Live pictures:
  - `/api/state?cam=` — status of all cameras
  - `/api/stream?cam=` — MJPEG
  - `/api/frame?cam=`, `/api/rawframe?cam=` — single JPEG
- Data:
  - `/api/violations?limit&rule&cam`
  - `/api/stats` (by rule, by camera, by hour, today vs yesterday)
  - `/api/alerts?since=`
- Cameras: `/api/cameras` with `add`, `update`, `delete`, `test`, `discover`.
- Zones: `/api/zones?cam=`.
- Settings: `/api/settings` (retention and overlays) and `/api/settings/preview?days=`.
- Users and login:
  - `/api/users[...]`, `/api/me`
  - `/api/login`, `/api/logout`, `/api/setup`, `/api/password`
- `/api/audit` — the audit log.
- Files: `/snapshots/…` and `/clips/…`, with HTTP Range so videos can seek.

Security headers are set (nosniff, SAMEORIGIN, same-origin referrer), and POST bodies must be JSON.

---

## 7. TV view (`/tv`, file `web/tv.html`)

- **Plain ES5** with no modern CSS, so old TV browsers run it.
- **Layout:** the live picture fills the left 75%; the **violations list on the right (8 items)**. Nothing is ever drawn over the picture.
- **New violations:** a new violation appears at the top of the list with its rule-coloured border and a blinking **NEW** tag for 20 s. The old bottom banner was removed at the user's request.
- **Top bar:**
  - **‹ Dashboard** back button
  - **camera dropdown** ("All cameras" or one camera), remembered per screen in localStorage; a `?cam=camN` link wins
  - monitor and camera status, clock
- **Picture:** MJPEG first. If a browser shows nothing for MJPEG (as the Samsung TV does), the page switches to fetching single JPEGs and remembers that. The grid layout is chosen to make the pictures as large as possible (two cameras are stacked vertically).
- **Self-updating:**
  - It reloads itself when the monitor restarts.
  - Every 10 min it compares its `PAGE_REV` with the server's copy and reloads when tv.html has changed. **Bump `PAGE_REV` on every tv.html change.**

---

## 8. Users and security

- `users.json` holds **PBKDF2-SHA256 hashes only** (200k iterations, salted).
- Roles:
  - **admin** — changes cameras, zones, settings and users
  - **viewer** — looks only; logging in is optional for viewing
- The first admin user is `admin`. The user picked the password; it can be changed in Settings → My account.
- Sessions are in memory (12 h). **A monitor restart logs everyone out.**
- Login lockout: 5 failed attempts from one IP block it for 2 minutes.
- Camera passwords live only in `monitor_cameras.json` and `config.yaml` on the PC. The API and logs always mask them (`rtsp://user:***@…`).
- Note: a file named **`cameras.json`** in the folder belongs to another tool (different format). The monitor does **not** use it; leave it alone.

---

## 9. Files and folders

```
Pictures\yolo\
  web.py                    entry point: multi-camera runtime + main loop
  watchdog.bat / run.bat    supervisor (run.bat just calls watchdog.bat)
  config.yaml               global settings: models, rules, cam1 zones, clips, retention, web, alerts, output
  monitor_cameras.json      camera list (id, name, source, enabled, detect, per-camera zones) – app-managed
  users.json                dashboard users (hashes)
  violations.jsonl          alert log
  restart.request / watchdog.stop   drop-in files for a clean restart / stop
  poketenashi\
    camera.py      RTSP capture, reconnect, 1 decode thread, gray-frame skip, mask_url
    detector.py    YOLO pose + ByteTrack → Person objects (keypoints helpers)
    rules.py       RuleEngine: PO, TE, SHI, RUN, FALL, RESTRICTED, zone gate, KE/NA dispatch
    ke.py          KE (zoomed phone check, ear call, latch)
    na.py          NA (top-down homography, heading, junction)
    hands.py       MediaPipe HandLandmarker wrapper (models/hand_landmarker.task)
    po_model.py    PO pocket classifier (MobileNetV3-Small, disabled)
    zones.py / zone_store.py   zone geometry / saving zone lines into config.yaml
    alerts.py      episodes, log rows, evidence names
    clips.py       evidence clip ring buffer + encoder + posters
    sampler.py     training-sample collector
    retention.py   evidence clean-up (+ preview)
    visualizer.py / enhance.py   overlay drawing / picture enhancement
    webserver.py   HTTP server + API
    camstore.py    monitor_cameras.json store (validated, backed up, atomic)
    auth.py        users + sessions
    settings_store.py   comment-preserving edits of config.yaml (incl. nested sections)
    rtsp_probe.py  RTSP DESCRIBE probe / stream-path discovery
    logsetup.py    all output → logs/monitor.log
  web\  index.html, app.css, ui.js, app.js (dashboard), tv.html (TV), zones.html (redirect to /#zones)
  models\  yolo11n-pose.pt, yolo11m.pt (phone), hand_landmarker.task, po_arm_mnv3.pt
  trackers\bytetrack_lowfps.yaml
  footage\  test recordings (po_test_20260925.mp4, live_test_20260928.mp4, ...)
  tools\    replay.py, record.py, zone_drawer.py, health_watch.py, scan_clips.py, ...
  logs\     monitor.log, fatal.log, audit.jsonl ; watchdog.log
  config_backups\  automatic backups of config.yaml / monitor_cameras.json
  _backup\  code snapshots before big changes (e.g. 20261003_before_multicam, 20261003_dashboard_v1)
D:\poketenashi-data\  clips\, snapshots\, samples\, po_arms\ (labelled PO crops)
```

---

## 10. Operations runbook

- **Live instance:** `Pictures\yolo`, port **6002**, supervised by `watchdog.bat`.
- **Clean restart** (to deploy code): create the file `restart.request` next to `web.py`. The app exits with code 4 and the watchdog restarts it within ~5–20 s.
  - The camera may refuse the new RTSP session for 20–90 s.
  - Everyone is logged out.
- **Clean stop:** create `watchdog.stop`.
- **Never start `run.bat` / `watchdog.bat` while the monitor runs.** It kills the live copy and leaves two supervisors.
  - A supervised second copy stands by quietly (`[standby]` in the log) instead of crash-looping.
  - Do not edit `watchdog.bat` while it runs.
- **Logs:**
  - `logs/monitor.log` — everything, including `[ALERT]`, `[camera]`, `[web] client <ip>` lines and TV page errors
  - `logs/fatal.log` — crashes and hang stack dumps
  - `watchdog.log` — restarts
  - `logs/audit.jsonl` — admin actions
- **Health checks:** `GET /api/health` and `GET /api/state`.
- **Front-end files are served from `web/` on every request.** Editing them changes the live dashboard immediately, so stage new versions elsewhere.
  - The env var `POKETENASHI_WEB_DIR` lets a test copy serve a staged folder.
  - The TV page self-updates via `PAGE_REV`.
- **Testing safely:**
  - Run a second copy on port 6012 with its own config copy, camera list (video files from `footage/`), users and log dir (`POKETENASHI_LOG_DIR`), at BelowNormal priority.
  - **Stop it before dropping `restart.request`:** it watches the same file.
- **Memory:** only ~1–1.5 GB of RAM is free. Run heavy analysis jobs one at a time, at low priority, after checking free commit memory. A replay job once caused a MemoryError and a 3.8 fps live view.
- **Tuning rules:** always with evidence.
  - Replay footage with `tools/replay.py`.
  - Audit alert posters and clips.
  - Check **both directions**: real violations must still fire, and false ones must stop.

---

## 11. Timeline of recent work

| Date | Change |
|---|---|
| Jun 2026 | First version (in `Documents\yolo`). |
| ~2 Sep | Copied to `Pictures\yolo`; browser dashboard. |
| 10 Sep | PO fingertip/geometry logic. This is still the base of PO. |
| 24 Sep | Stability: watchdog, logging, health tools. |
| 25–26 Sep | Live audits and fixes: FALL legs veto; NA top-down homography and junction; KE redesigned to zoom-verified phones (removing the 70%-CPU full-frame pass); TV page; zone editor. PO filter work (later reverted). |
| 28 Sep | PO reverted to the 10 Sep logic plus "not PO" filters. Latency fix (1 decode thread). KE size cap and ear-call path. Per-person alerts and phone ownership. Live test 13:01–13:05. PO pocket model trained (not enabled). |
| 3 Oct | Multi-camera support and dashboard v1 (09:25). **Production dashboard v2** (10:24): confirm popups everywhere, add cameras by IP with stream discovery, audit log, overlays saved, retention preview, Range/security fixes, keep-alive bug fix. Dx camera path fixed. TV dropdown and back button. **Gray-frame skip** (11:09). TV violations side-only (11:20). |

---

## 12. Open issues / next steps

1. **PO recall:** pockets while walking are mostly missed. Next step: validate and enable the pocket model (`rules.PO.model_path: models/po_arm_mnv3.pt`, threshold 0.6) against the 25 Sep tester and the 28 Sep false-alert clips, and collect more labelled data from `samples/`.
2. **KE recall:** phones held low or edge-on are invisible to COCO models. A phone detector should be trained on this camera's crops.
3. **CPU budget:** two detecting cameras halve the FPS per camera (cam1 drops to ~4–5 fps with people, which is borderline for PO's 3-frame confirm). Options:
   - Dx sub stream (lower decode cost, fewer gray frames)
   - Dx with rules off (live view only)
   - fewer torch threads / a lighter model
   - a GPU
4. **Dx has no zones:** draw `gangway` (exactly 4 corners) and `crossing`. Until then PO/KE/RUN/FALL judge everyone on Dx and NA is idle there.
5. **TE / SHI / RESTRICTED** are idle: no `stairs`/`handrail`/`checkpoint`/`restricted` zones are drawn.
6. **TV link** depends on the PC's Wi-Fi staying on TBDI-DX at 192.168.100.32 (DHCP). IT should reserve the address, or cable the TV.
7. **Logins** reset on every restart (sessions are in memory).
8. The admin password chosen at setup is weak; change it in Settings → My account.

---

## 13. How the user works / preferences

- The user writes in **Hinglish**. They judge results by **what the dashboard and the evidence clips show**.
- PO: one hand is enough; walking or standing; in the gangway. Do not add unnecessary gates that kill real PO, but do remove false positives ("haath dikh rahe hain" = hands are visible ⇒ not PO). A hand or paper at the chest is neither PO nor KE.
- Every person breaking a rule must get their own red box and their own alert.
- Every change in the UI must be confirmed through a popup. The UI should be production-level.
- TV: only the live picture and violations at the side, a camera dropdown and a back button.
- Never show or log passwords. Do not touch the other apps on the PC. Ask before system/security settings or persistent changes such as autostart.

---

## 14. Glossary

- **Track / track ID:** the tracker's per-person number. It is per camera and can change on occlusion.
- **Torso length:** the shoulder-midpoint → hip-midpoint distance. All rule thresholds use it, so they work at any distance from the camera.
- **Gangway:** the walking aisle (zone). **Crossing:** the side gangways / T-junction zone.
- **Episode:** one incident = one alert, however long it lasts (re-alerted every 60 s if ongoing).
- **Poster:** the JPEG of the alert frame saved with each clip.
- **Main / sub stream:** a camera's high-resolution vs low-resolution RTSP stream.
