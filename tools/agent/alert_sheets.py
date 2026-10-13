"""One zoom sheet per PO / KE alert in the evidence archive (12 Oct): 4 frames of the
red-boxed person from the alert clip, enlarged, + time / camera / ID. Light: OpenCV only.
Output: <out>/<rule>/<YYYYMMDD_HHMMSS>_<cam>_ID<tid>.jpg and <out>/index.csv"""
import csv, glob, json, os, sys
import cv2, numpy as np
EV = "D:/poketenashi-data/evidence"
def red_box(img):
    """The alert's red box: a THIN red rectangle outline (a red machine or sign is a
    filled blob - rejected by the fill ratio)."""
    m = cv2.inRange(img, (0, 0, 171), (69, 69, 255)); n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
    c = [st[i] for i in range(1, n) if st[i][2] > 12 and st[i][3] > 25 and st[i][4] < 0.25 * st[i][2] * st[i][3]]
    if not c: return None
    x, y, w, h, _ = max(c, key=lambda s: s[2] * s[3]); return [x, y, x + w, y + h]
def main(out):
    rows = []
    for day in sorted(os.path.basename(d) for d in glob.glob(EV + "/2*")):
        aj = os.path.join(EV, day, "alerts.jsonl")
        if not os.path.exists(aj): continue
        seen = set()
        for line in open(aj, encoding="utf-8"):
            try: a = json.loads(line)
            except ValueError: continue
            if a.get("rule") not in ("PO", "KE") or not a.get("clip"): continue
            k = (a["time"], a.get("camera"), a["track_id"], a["rule"])
            if k in seen: continue
            seen.add(k)
            clip = os.path.join(EV, day, a["clip"])
            if not os.path.exists(clip): continue
            cap = cv2.VideoCapture(clip); tot = int(cap.get(7) or 0); boxed = []
            for i in np.linspace(0, max(0, tot - 1), 30).astype(int):
                cap.set(1, int(i)); ok, f = cap.read()
                if not ok: continue
                rb = red_box(f)
                if rb: boxed.append((f, rb))
            cap.release()
            if not boxed: continue
            pick = [boxed[int(j)] for j in np.linspace(0, len(boxed) - 1, min(4, len(boxed))).astype(int)]
            tiles = []
            for f, (x1, y1, x2, y2) in pick:
                H, W = f.shape[:2]; pw, ph = int(.35 * (x2 - x1)), int(.08 * (y2 - y1))
                c = f[max(0, y1 - ph):min(H, y2 + ph), max(0, x1 - pw):min(W, x2 + pw)]
                c = cv2.resize(c, (int(c.shape[1] * 420 / c.shape[0]), 420), interpolation=cv2.INTER_CUBIC)
                tiles.append(c)
            w = sum(t.shape[1] for t in tiles); sheet = np.full((460, w, 3), 255, np.uint8); x = 0
            for t in tiles: sheet[40:, x:x + t.shape[1]] = t; x += t.shape[1]
            hh = a["time"][11:19].replace(":", "")
            cv2.putText(sheet, f"{a['rule']} {a['time']} {a.get('camera_name') or a.get('camera')} ID{a['track_id']}", (6, 28), 0, 0.8, (0, 0, 200), 2)
            name = f"{day}_{hh}_{a.get('camera') or 'cam1'}_ID{a['track_id']}.jpg"
            os.makedirs(os.path.join(out, a["rule"]), exist_ok=True)
            cv2.imwrite(os.path.join(out, a["rule"], name), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
            rows.append([a["rule"], a["time"], a.get("camera_name") or a.get("camera"), a["track_id"], os.path.join(a["rule"], name), clip])
    with open(os.path.join(out, "index.csv"), "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo); w.writerow(["rule", "time", "camera", "track", "sheet", "clip"]); w.writerows(rows)
    print(len(rows), "sheets")
main(sys.argv[1])
