"""Face ID: show the employee code (e.g. 481) on a person the camera sees, when
their face matches one of the enrolled people.

Enrolment: one folder per person under `face_id.gallery_dir` (default
images/), named "ID <code>", with a few photos of the face (front and side,
with and without cap / glasses). The code shown is the folder's <code>. The
gallery is rebuilt by itself when the photos change.

The ceiling cameras see a face only ~40-55 px wide, blurred and compressed. So:
  * every photo is also enrolled the way the camera sees it - shrunk to a
    40-55 px face, blurred, JPEG-compressed (measured 6 Oct: same person
    0.46-0.50 median similarity at 35-45 px, other person <= 0.17);
  * the head is cut from the frame, upscaled x4 with a light sharpen (CLAHE +
    unsharp - no AI enhancement, it would invent a face) before detection;
  * ONE look never names anyone: a person gets a code only once several looks
    of the same track agree (votes, top scores and the margin to the other
    codes). A single frame of a stranger reached 0.53 on the 6 Oct footage;
    the testers' tracks held 0.27-0.31 median over dozens of looks.
Runs on its own thread with its own (small) thread budget: the detection loop
only hands it head crops and reads names back, it never waits for a model.

Models: InsightFace buffalo_l (SCRFD det_10g + ArcFace w600k_r50), in
~/.insightface/models/buffalo_l.
"""
import glob
import os
import queue
import threading
import time
from collections import defaultdict, deque

import cv2
import numpy as np

MODEL_DIR = os.path.join(os.path.expanduser("~"), ".insightface", "models", "buffalo_l")


def enhance(img):
    """Light, identity-safe clean-up of an upscaled face crop: local contrast
    (CLAHE on L) and an unsharp mask. Nothing is invented."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(img, (0, 0), 2.0)
    return cv2.addWeighted(img, 1.6, blur, -0.6, 0)


class FaceID:
    def __init__(self, cfg, root):
        c = cfg or {}
        self.ok = False
        self.root = root
        self.gallery_dir = os.path.join(root, c.get("gallery_dir", "images"))
        self.upscale = int(c.get("upscale", 4))
        self.look_sec = float(c.get("look_every_sec", 1.0))          # per person, while unknown
        self.recheck_sec = float(c.get("recheck_sec", 5.0))         # per person, once named
        self.min_face_px = float(c.get("min_face_px", 22))          # smaller faces are not judged
        self.frame_min = float(c.get("frame_min", 0.25))            # a look counts as a vote from this
        self.need_votes = int(c.get("need_votes", 3))
        self.need_ratio = float(c.get("need_ratio", 0.4))           # of the looks that found a face
        self.need_top = float(c.get("need_top", 0.35))              # mean of the 3 best looks
        self.margin = float(c.get("margin", 0.08))                  # over the next code
        self.forget_sec = float(c.get("forget_sec", 10.0))
        # 8 Oct 2026 (Toolroom 08:32: usable face from 08:32:36, named only at
        # 08:32:56): (1) the tracker gave her a new id mid-walk and the votes
        # started over - a new track where one vanished within handover_sec
        # takes over its looks and code; (2) once a look found a face of
        # fast_face_px or more, the next look comes after fast_look_sec, not
        # look_every_sec; (3) a face under vote_min_face_px (far, blurred) is
        # not a vote - it only diluted need_ratio. Defaults = the old behaviour.
        self.handover_sec = float(c.get("handover_sec", 0.0))
        self.fast_face_px = float(c.get("fast_face_px", 1e9))
        self.fast_look_sec = float(c.get("fast_look_sec", self.look_sec))
        self.vote_min_face_px = float(c.get("vote_min_face_px", self.min_face_px))
        # 9 Oct 2026 (user): the name belongs IN the alert - a person about to be
        # alerted with no name yet gets a look right away (look_now), at most
        # every priority_look_sec; the face width is per thread (two lookers).
        self.priority_look_sec = float(c.get("priority_look_sec", 0.5))
        self._tl = threading.local()
        threads = int(c.get("threads", 2))
        try:
            import onnxruntime as ort
            from insightface.model_zoo.arcface_onnx import ArcFaceONNX
            from insightface.model_zoo.scrfd import SCRFD
            from insightface.utils import face_align
            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = 1
            prov = ["CPUExecutionProvider"]
            det_f, rec_f = os.path.join(MODEL_DIR, "det_10g.onnx"), os.path.join(MODEL_DIR, "w600k_r50.onnx")
            self.det = SCRFD(det_f, ort.InferenceSession(det_f, so, providers=prov))
            self.det.prepare(-1, input_size=(320, 320), det_thresh=float(c.get("det_thresh", 0.3)))
            self.rec = ArcFaceONNX(rec_f, ort.InferenceSession(rec_f, so, providers=prov))
            self.rec.prepare(-1)
            self._align = face_align.norm_crop
        except Exception as e:
            print(f"[face] disabled: models not loadable ({e}) - expected in {MODEL_DIR}", flush=True)
            return
        # The gallery is (re)built on the worker thread, and watched: photos
        # added, changed or removed under gallery_dir take effect within
        # gallery_check_sec, no restart. 7 Oct: at the 03:36 restart the photo
        # folders were empty for a moment; face ID wrote an empty gallery and
        # switched itself off for the day - so the testers who broke rules at
        # 08:50 got no code. An empty folder now never replaces a good gallery.
        self.codes, self.G, self._sig = [], {}, None
        self.check_sec = float(c.get("gallery_check_sec", 30.0))
        self._q = queue.Queue(maxsize=int(c.get("queue", 6)))
        self._lock = threading.Lock()
        self._st = {}                     # (cam, tid) -> {"looks": deque, "code", "score", "last_look", "seen"}
        self.ok = True
        threading.Thread(target=self._worker, name="faceid", daemon=True).start()
        print(f"[face] face ID started - enrolling the photos in {self.gallery_dir} ...", flush=True)

    # ---- gallery ---------------------------------------------------------

    def _photos(self):
        out = []
        for d in sorted(glob.glob(os.path.join(self.gallery_dir, "ID *"))):
            code = os.path.basename(d)[2:].strip()
            for p in sorted(glob.glob(os.path.join(d, "*"))):
                if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    out.append((code, p))
        return out

    def _refresh(self):
        """Pick up the photos as they are now (worker thread). Unchanged: no
        work. No photos at all: keep the gallery we have (or the last good
        cache) - a folder briefly emptied must not switch face ID off."""
        try:
            photos = self._photos()
            sig = [(c, os.path.basename(p), os.path.getsize(p), int(os.path.getmtime(p))) for c, p in photos]
        except OSError as e:
            print(f"[face] cannot read the photo folders: {e}", flush=True)
            return
        if sig == self._sig:
            return
        self._sig = sig
        if not photos:
            if not self.codes:
                G = self._cached(None)
                if G:
                    with self._lock:
                        self.codes, self.G = sorted(G), G
            print(f"[face] no photos in {self.gallery_dir} (folders 'ID <code>') - "
                  + (f"keeping the enrolled {', '.join(self.codes)}" if self.codes else "nobody enrolled yet"),
                  flush=True)
            return
        G = self._gallery(photos, sig)
        if not G:
            print("[face] no face found in any photo - keeping the previous gallery", flush=True)
            return
        with self._lock:
            self.codes, self.G = sorted(G), G
            for st in self._st.values():                # judged against the old gallery
                st["looks"].clear()
                st["code"] = None
        print(f"[face] face ID ready: {', '.join(self.codes)} "
              f"({sum(len(v) for v in self.G.values())} gallery embeddings)", flush=True)

    def _cached(self, sig):
        """The cached gallery - for this photo signature, or (sig None) the
        last good one whatever its photos. {} if none."""
        try:
            z = np.load(os.path.join(self.root, "models", "face_gallery.npz"), allow_pickle=True)
            G = z["G"].item()
            if G and (sig is None or list(map(tuple, z["sig"].tolist())) == sig):
                return G
        except Exception:
            pass
        return {}

    def _gallery(self, photos, sig):
        cache = os.path.join(self.root, "models", "face_gallery.npz")
        G = self._cached(sig)
        if G:
            return G
        from PIL import Image, ImageOps
        rng = np.random.default_rng(0)
        G = defaultdict(list)
        for code, p in photos:
            try:
                img = cv2.cvtColor(np.array(ImageOps.exif_transpose(Image.open(p)).convert("RGB")), cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"[face] cannot read {p}: {e}", flush=True)
                continue
            s = 1600 / max(img.shape[:2])
            if s < 1:
                img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
            self.det.prepare(-1, input_size=(640, 640))
            bb, kps = self.det.detect(img, max_num=1)
            self.det.prepare(-1, input_size=(320, 320))
            if len(bb) == 0:
                print(f"[face] no face found in {p} - not enrolled", flush=True)
                continue
            k = kps[0]
            G[code].append(self._embed(img, k))
            # the same face as the ceiling cameras deliver it
            fw = float(np.linalg.norm(k[0] - k[1])) * 2.2
            for w in (55, 45, 35):
                sc = w / max(fw, 1.0)
                small = cv2.resize(img, (max(8, int(img.shape[1] * sc)), max(8, int(img.shape[0] * sc))),
                                   interpolation=cv2.INTER_AREA)
                if rng.random() < 0.7:
                    small = cv2.GaussianBlur(small, (3, 3), rng.uniform(0.4, 1.2))
                ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(40, 75))])
                up = cv2.resize(cv2.imdecode(buf, cv2.IMREAD_COLOR), (img.shape[1], img.shape[0]),
                                interpolation=cv2.INTER_CUBIC)
                G[code].append(self._embed(up, k))
        G = {c: np.stack(v) for c, v in G.items()}
        if G:                                   # never cache an empty gallery
            try:
                np.savez(cache, sig=np.array(sig, dtype=object), G=np.array(G, dtype=object))
            except Exception as e:
                print(f"[face] gallery cache not saved: {e}", flush=True)
        return G

    def _embed(self, img, kps):
        a = self._align(img, landmark=kps, image_size=112)
        e = self.rec.get_feat(a).ravel()
        return e / max(1e-9, float(np.linalg.norm(e)))

    # ---- live ------------------------------------------------------------

    @staticmethod
    def _head_box(p, mc=0.4):
        kp = p.keypoints
        pts = [kp[i, :2] for i in range(5) if kp[i, 2] >= mc]
        face = kp[0, 2] >= 0.5 and max(kp[1, 2], kp[2, 2]) >= 0.5      # nose + an eye: face towards us
        if len(pts) < 3 or not face:
            return None
        cx, cy = np.mean(pts, axis=0)
        h = 0.30 * float(p.bbox[3] - p.bbox[1])
        return int(cx - h), int(cy - h), int(cx + h), int(cy + h)

    def observe(self, cam, frame, persons):
        """Queue head crops of the people worth a look now. Never blocks."""
        if not self.ok or frame is None or not self.codes:
            return
        now = time.time()
        H, W = frame.shape[:2]
        with self._lock:
            live = {p.track_id for p in persons if p.tracked}
            for p in persons:
                if not p.tracked:
                    continue
                key = (cam, p.track_id)
                if key not in self._st and self.handover_sec > 0:
                    self._handover(cam, p, live, now)
                st = self._st.setdefault(key, {"looks": deque(maxlen=40), "code": None,
                                               "score": 0.0, "last_look": 0.0, "seen": now})
                st["seen"] = now
                st["box"] = [float(v) for v in p.bbox]
                every = self.recheck_sec if st["code"] else (
                    self.fast_look_sec if st.get("fw", 0.0) >= self.fast_face_px else self.look_sec)
                if now - st["last_look"] < every:
                    continue
                hb = self._head_box(p)
                if hb is None:
                    continue
                x1, y1, x2, y2 = max(0, hb[0]), max(0, hb[1]), min(W, hb[2]), min(H, hb[3])
                if x2 - x1 < 16 or y2 - y1 < 16:
                    continue
                try:
                    self._q.put_nowait((cam, p.track_id, frame[y1:y2, x1:x2].copy()))
                    st["last_look"] = now
                except queue.Full:
                    break
            for k in [k for k, s in self._st.items() if now - s["seen"] > self.forget_sec]:
                del self._st[k]

    def _handover(self, cam, p, live, now):
        """A new track id: if a track of this camera vanished within
        handover_sec just where this person is (feet within 0.6 of the person's
        height, a similar height), it was the same person - take its state."""
        x1, y1, x2, y2 = [float(v) for v in p.bbox]
        h = max(1.0, y2 - y1)
        foot = np.array([(x1 + x2) / 2, y2])
        best = None
        for (c, tid), s in self._st.items():
            if c != cam or tid in live or "box" not in s or not 0 < now - s["seen"] <= self.handover_sec:
                continue
            b = s["box"]
            bh = max(1.0, b[3] - b[1])
            d = float(np.linalg.norm(foot - np.array([(b[0] + b[2]) / 2, b[3]]))) / h
            if d <= 0.6 and 0.6 <= h / bh <= 1.6 and (best is None or d < best[0]):
                best = (d, (c, tid))
        if best:
            old = self._st.pop(best[1])
            old["seen"], old["was"] = now, best[1][1]
            self._st[(cam, p.track_id)] = old

    def look_now(self, cam, frame, persons, tids):
        """A priority look, in the caller's thread, for the people in `tids` who
        have no name yet (an alert is about to be written for them). Same voting
        as the queued looks - one look only adds a vote. Returns {tid: code}."""
        if not self.ok or frame is None or not self.codes or not tids:
            return {}
        H, W = frame.shape[:2]
        now = time.time()
        todo = []
        with self._lock:
            for p in persons:
                st = self._st.get((cam, p.track_id))
                if p.track_id not in tids or st is None or st["code"]:
                    continue
                if now - st.get("prio_t", 0.0) < self.priority_look_sec:
                    continue
                hb = self._head_box(p)
                if hb is None:
                    continue
                x1, y1, x2, y2 = max(0, hb[0]), max(0, hb[1]), min(W, hb[2]), min(H, hb[3])
                if x2 - x1 < 16 or y2 - y1 < 16:
                    continue
                st["prio_t"] = st["last_look"] = now
                todo.append((p.track_id, frame[y1:y2, x1:x2].copy()))
        got = {}
        for tid, crop in todo:
            self._process(cam, tid, crop)
            code = self.name(cam, tid)
            if code:
                got[tid] = code
        return got

    def name(self, cam, tid):
        if not self.ok:
            return None
        st = self._st.get((cam, tid))
        return st["code"] if st else None

    def names(self, cam):
        if not self.ok:
            return {}
        with self._lock:
            return {tid: s["code"] for (c, tid), s in self._st.items() if c == cam and s["code"]}

    def _look(self, crop):
        up = self.upscale
        big = enhance(cv2.resize(crop, (crop.shape[1] * up, crop.shape[0] * up), interpolation=cv2.INTER_CUBIC))
        bb, kps = self.det.detect(big, max_num=1)
        if len(bb) == 0 or (bb[0][2] - bb[0][0]) / up < self.min_face_px:
            return None
        e = self._embed(big, kps[0])
        G = self.G                                      # one gallery for the whole look
        self._tl.fw = float(bb[0][2] - bb[0][0]) / up
        return {c: float(np.max(G[c] @ e)) for c in sorted(G)}

    def _decide(self, looks):
        """A code once enough looks of this track agree, else None."""
        if not looks:
            return None, 0.0
        best = None
        codes = sorted(looks[-1])
        looks = [l for l in looks if sorted(l) == codes]   # all scored against one gallery
        for c in codes:
            sc = sorted((l[c] for l in looks), reverse=True)
            top = float(np.mean(sc[:3]))
            votes = sum(1 for l in looks if max(l, key=l.get) == c and l[c] >= self.frame_min)
            other = max((float(np.mean(sorted((l[o] for l in looks), reverse=True)[:3]))
                         for o in codes if o != c), default=0.0)
            if (votes >= self.need_votes and votes >= self.need_ratio * len(looks)
                    and top >= self.need_top and top - other >= self.margin):
                if best is None or top > best[1]:
                    best = (c, top)
        return best if best else (None, 0.0)

    def _worker(self):
        last_check = 0.0
        while True:
            if time.time() - last_check >= self.check_sec:
                last_check = time.time()
                try:
                    self._refresh()
                except Exception as e:
                    print(f"[face] gallery refresh failed: {e}", flush=True)
            try:
                cam, tid, crop = self._q.get(timeout=self.check_sec)
            except queue.Empty:
                continue
            self._process(cam, tid, crop)

    def _process(self, cam, tid, crop):
        try:
            self._tl.fw = 0.0
            s = self._look(crop)
        except Exception as e:
            print(f"[face] look failed: {e}", flush=True)
            s = None
        fw = self._tl.fw if s is not None else 0.0
        with self._lock:
            st = self._st.get((cam, tid))
            if st is None:
                for (c, t), x in self._st.items():          # handed over to a new id meanwhile
                    if c == cam and x.get("was") == tid:
                        st = x
                        break
            if st is None:
                return
            st["fw"] = fw
            if s is None or fw < self.vote_min_face_px:
                return
            st["looks"].append(s)
            code, score = self._decide(st["looks"])
            if code and code != st["code"]:
                print(f"[face] {cam} ID{tid} = {code} (score {score:.2f}, {len(st['looks'])} looks)", flush=True)
            st["code"], st["score"] = code, score
