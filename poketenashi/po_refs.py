"""PO reference hands (9 Oct 2026, the user): "the PO screenshots I sent are OK PO -
whatever matches them in ONE frame is PO; anything else needs 2-3 frames of the
hand in the pocket".

The reference folder holds person snapshots the user confirmed as PO (raw, no
overlay). At start each is posed (the re-pose model the PO rule already has),
every arm with its wrist at the hip is cut exactly like live (po_model.arm_crop)
and kept if the pocket model calls it pocket (P >= keep_p) - that is the pocket
hand. Its embedding = the pocket model's own pooled features (mirror-averaged,
L2-normalised). Live, a POCKET arm is compared with every reference: the best
cosine similarity decides whether ONE frame is enough. No extra model in memory."""
import glob
import os

import cv2
import numpy as np
import torch

from . import po_model

EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


@torch.inference_mode()
def embed(model, crops):
    if not crops:
        return np.zeros((0, 576), np.float32)
    ims = []
    for c in crops:
        rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
        im = cv2.resize(rgb, (po_model.NET, po_model.NET)).astype(np.float32) / 255.0
        ims += [im, np.ascontiguousarray(im[:, ::-1])]
    a = (np.stack(ims) - po_model.MEAN) / po_model.STD
    x = torch.from_numpy(a.transpose(0, 3, 1, 2).copy())
    f = model.net.avgpool(model.net.features(x)).flatten(1).numpy()
    f = (f[0::2] + f[1::2]) / 2.0
    return f / np.maximum(1e-9, np.linalg.norm(f, axis=1, keepdims=True))


def _read_images(f, video_step_sec=1.0, max_frames=20):
    """The pictures of one master file: an image, or frames of a clip."""
    if f.lower().endswith(EXTS):
        try:
            from PIL import Image
            yield cv2.cvtColor(np.array(Image.open(f).convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception:
            return
        return
    cap = cv2.VideoCapture(f)
    fps = cap.get(cv2.CAP_PROP_FPS) or 5
    step, j, n = max(1, int(round(fps * video_step_sec))), -1, 0
    while n < max_frames:
        ok = cap.grab()
        if not ok:
            break
        j += 1
        if j % step:
            continue
        ok, fr = cap.retrieve()
        if ok:
            n += 1
            yield fr
    cap.release()


_CACHE = {}


def _cache_path(model):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "models", "po_refs_cache_" + os.path.splitext(os.path.basename(model.path))[0] + ".pkl")


def _cache_for(model):
    import pickle
    p = _cache_path(model)
    if p not in _CACHE:
        try:
            with open(p, "rb") as fh:
                _CACHE[p] = pickle.load(fh)
        except Exception:
            _CACHE[p] = {}
    return _CACHE[p]


def _save_cache(model, cache):
    import pickle
    p = _cache_path(model)
    try:
        with open(p + ".tmp", "wb") as fh:
            pickle.dump(cache, fh)
        os.replace(p + ".tmp", p)
    except Exception as e:
        print(f"[PO] master cache not saved: {e}", flush=True)


class PORefs:
    """Master pocket hands from every OK PO image AND clip in the folders. Every
    person in a picture, every arm with the wrist at the hip, kept when the pocket
    model calls it pocket (P >= keep_p): in an OK PO picture that is the pocket hand.
    Only the hand-at-hip crop is used - no face, no clothes outside it."""

    def __init__(self, folders, model, pose_model, imgsz=320, keep_p=0.7):
        self.E = np.zeros((0, 576), np.float32)
        self.names = []
        files = []
        for d in folders:
            files += [f for f in glob.glob(os.path.join(d, "**", "*"), recursive=True)
                      if f.lower().endswith(EXTS + (".mp4", ".avi", ".mkv"))]
        # 10 Oct: per-file cache (pocket score + embedding of every hand at the hip),
        # in memory and on disk per model - a rebuild only poses the NEW files. Posing
        # all ~700 master files took ~70 s of CPU and cut the live fps to 0.7 (11:50).
        cache = _cache_for(model)
        Ps, Es, names, n_new = [], [], [], 0
        for f in sorted(set(files)):
            try:
                st = os.stat(f)
            except OSError:
                continue
            key = (os.path.abspath(f), int(st.st_mtime), st.st_size)
            hit = cache.get(key)
            if hit is None:
                hit = self._file_arms(f, model, pose_model, imgsz)
                cache[key] = hit
                n_new += 1
            for pv, ev in zip(*hit):
                Ps.append(pv); Es.append(ev); names.append(os.path.basename(f))
        if n_new:
            _save_cache(model, cache)
        keep = [i for i, pv in enumerate(Ps) if pv >= keep_p]
        if keep:
            self.E = np.stack([Es[i] for i in keep]).astype(np.float32)
            self.names = [names[i] for i in keep]
        self.n_new = n_new
        crops = Ps                      # (only its length is used below)
        self.n_files = len(set(files))
        self.n_arms = len(crops)
        # how alike the masters are among themselves (nearest other master, other file)
        self.self_sim = None
        if len(self.E) > 2:
            S = self.E @ self.E.T
            np.fill_diagonal(S, -1)
            same = np.array([[a == b for b in self.names] for a in self.names])
            S[same] = -1
            self.self_sim = float(np.median(S.max(1)))

    @staticmethod
    def _file_arms(f, model, pose_model, imgsz):
        """(pocket scores, embeddings) of every hand at the hip in one master file."""
        crops = []
        for img in _read_images(f):
            if max(img.shape[:2]) < 120:
                continue
            sz = imgsz if max(img.shape[:2]) <= 700 else 960
            r = pose_model.predict(img, imgsz=sz, conf=0.3, verbose=False, device="cpu")[0]
            if r.keypoints is None or len(r.boxes) == 0:
                continue
            for kp in r.keypoints.data.cpu().numpy():
                if min(kp[5, 2], kp[6, 2], kp[11, 2], kp[12, 2]) < 0.3:
                    continue
                torso = float(np.linalg.norm((kp[5, :2] + kp[6, :2]) / 2 - (kp[11, :2] + kp[12, :2]) / 2))
                if torso < 25:
                    continue
                for wi, hi in ((9, 11), (10, 12)):
                    if kp[wi, 2] < 0.3 or kp[hi, 2] < 0.3:
                        continue
                    W, H = kp[wi, :2], kp[hi, :2]
                    if abs(W[1] - H[1]) > 0.5 * torso or abs(W[0] - H[0]) > 0.6 * torso:
                        continue
                    c = po_model.arm_crop(img, W, torso)
                    if c is not None:
                        crops.append(c)
        if not crops:
            return [], []
        if hasattr(model, "score_embed"):
            P, E = model.score_embed(crops)
        else:
            P, E = model.score(crops), embed(model, crops)
        return [float(x) for x in P], [e.astype(np.float32) for e in E]

    def topk(self, e, k=1):
        """Mean of the k best cosine similarities of embedding e to the masters."""
        if len(self.E) == 0:
            return 0.0
        s = self.E @ e
        k = max(1, min(int(k), len(s)))
        return float(np.sort(s)[-k:].mean())

    def match(self, model, crop):
        """Best cosine similarity of this arm crop to the masters (0 if none)."""
        if len(self.E) == 0 or crop is None:
            return 0.0, None
        e = embed(model, [crop])[0]
        s = self.E @ e
        k = int(np.argmax(s))
        return float(s[k]), self.names[k]
