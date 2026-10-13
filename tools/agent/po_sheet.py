"""Per PO alert: 6 frames of its clip, cropped around the red alert box -> one strip.
Strips stacked 4 per sheet in PO_review/_sheets for looking at."""
import glob, os, sys
import cv2, numpy as np

D = r"C:\Users\DX-ADMIN\Pictures\yolo\PO_review\_all"
OUT = r"C:\Users\DX-ADMIN\Pictures\yolo\PO_review\_sheets"
os.makedirs(OUT, exist_ok=True)


def red_box(f):
    h = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(h, (0, 150, 120), (6, 255, 255)) | cv2.inRange(h, (174, 150, 120), (180, 255, 255))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cs:
        x, y, w, hh = cv2.boundingRect(c)
        if w > 25 and hh > 50 and (best is None or w * hh > best[2] * best[3]):
            best = (x, y, w, hh)
    return best


def crop(f, b, H=300):
    if b is None:
        c = f
    else:
        x, y, w, h = b
        p = int(max(w, h) * 0.25)
        c = f[max(0, y - p):y + h + p, max(0, x - p):x + w + p]
    s = H / c.shape[0]
    return cv2.resize(c, (max(1, int(c.shape[1] * s)), H))


strips = []
for mp4 in sorted(glob.glob(D + r"\*.mp4")):
    v = cv2.VideoCapture(mp4)
    frames, boxes = [], []
    while True:
        ok, f = v.read()
        if not ok:
            break
        b = red_box(f)
        if b is not None:
            frames.append(f); boxes.append(b)
    tiles = []
    if frames:
        idx = sorted(set(int(round(k * (len(frames) - 1) / 5)) for k in range(6)))
        tiles = [crop(frames[k], boxes[k], 330) for k in idx]
    if not tiles:
        continue
    row = np.hstack([cv2.copyMakeBorder(t, 0, 0, 0, 4, cv2.BORDER_CONSTANT) for t in tiles])
    name = os.path.basename(mp4)[:-4]
    lab = np.zeros((34, row.shape[1], 3), np.uint8)
    cv2.putText(lab, f"{len(strips)+1:02d}  {name}", (6, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    strips.append((name, np.vstack([lab, row])))

W = 1800
for i in range(0, len(strips), 4):
    rows = []
    for _, s in strips[i:i + 4]:
        s = s[:, :W] if s.shape[1] > W else cv2.copyMakeBorder(s, 0, 0, 0, W - s.shape[1], cv2.BORDER_CONSTANT)
        rows.append(s)
    cv2.imwrite(os.path.join(OUT, f"sheet_{i//4+1:02d}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
print(len(strips), "alerts,", (len(strips) + 3) // 4, "sheets")
for k, (n, _) in enumerate(strips):
    print(k + 1, n)
