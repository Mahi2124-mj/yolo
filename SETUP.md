# Poketenashi - setup on a PC (Windows, CPU)

## 1. Get the code
```
git clone https://github.com/Mahi2124-mj/yolo.git
cd yolo
```

## 2. Python environment (Python 3.12)
```
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install insightface onnxruntime psutil scipy lap
```
`requirements-lock.txt` lists the exact versions of the working PC. Its torch build is `+cu130`;
on a CPU-only PC the plain `pip install torch torchvision` is fine.

## 3. Private settings (NOT in GitHub - make them on the PC)
```
copy config.example.yaml config.yaml
copy monitor_cameras.example.json monitor_cameras.json
```
In both files replace `rtsp://USER:PASSWORD@...` with the real camera user and password.
Camera IPs, zones and rules are already filled in.

Not in GitHub (privacy): `users.json` (the dashboard asks to create the admin on first start),
`displays.json` (TV links - make again from the dashboard), `images/` (face ID photos - copy from
the old PC if face names are wanted), evidence / clips / logs (`D:/poketenashi-data`).

## 4. Run
```
run.bat            (or watchdog.bat - restarts the monitor if it stops)
```
Dashboard: http://localhost:6002

## Models in the repo
yolo11n-pose.pt (pose), yolo11m.pt (KE phone + object check), yolo11n.pt, yolov8s.pt,
models/po_arm_v7.pt (live PO model), models/po_arm_mnv3.pt (v1), models/po_arm_v4.pt,
models/ke_checker_13oct.pt (KE shadow checker), models/ppe_cap_v1.pt, models/hand_landmarker.task (MediaPipe).
Face ID downloads its own model (insightface buffalo_l) on first use.
