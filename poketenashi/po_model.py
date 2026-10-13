"""PO: is this hand in a pocket? A small image model looks at the hand.

Pose keypoints cannot tell a hand in a pocket from a relaxed hand hanging at
the hip on this camera - from its height both put the wrist at the hip, bend
the arm the same way and, while walking, move the same amount (28 Sep live
test: pockets 0.06-0.19, free hands 0.06-0.09). MediaPipe does not find hands
this small. So a MobileNetV3-Small, fine-tuned on crops of this camera, looks
at the hand itself: a visible hand below the wrist vs a forearm that ends in a
pocket.

Trained on 1090 labelled arm crops (208 in a pocket) from the 28 Sep live test
and the 25 Sep pocket test, `models/po_arm_mnv3.pt`. Grouped cross-validation
(people held out): at 0.6, 8 of 10 pocket episodes caught, 6 of 80 free-hand
episodes flagged. The crop MUST match the training crop (arm_crop below).
"""
import os

import cv2
import numpy as np
import torch
import torch.nn as nn

SIZE = 160      # crop as saved for training
NET = 144       # network input
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def arm_crop(frame, W, torso):
    """Square around the hand: the wrist, the hand below it and the pocket."""
    cx, cy = float(W[0]), float(W[1]) + 0.10 * torso
    half = 0.45 * torso
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = int(cx - half), int(cy - half), int(cx + half), int(cy + half)
    pad = [max(0, -y1), max(0, y2 - h), max(0, -x1), max(0, x2 - w)]
    c = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if c.size == 0:
        return None
    if any(pad):
        c = cv2.copyMakeBorder(c, pad[0], pad[1], pad[2], pad[3], cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(c, (SIZE, SIZE), interpolation=cv2.INTER_AREA if c.shape[0] > SIZE else cv2.INTER_CUBIC)


HIP_MC = 0.4      # keypoint confidence of the sampler, which cut the hip crops


def hip_box(p, frame_w, frame_h):
    """The crop the HIP model (rules.PO.hip_model_path) judges: is ANY hand of
    this person in a pocket? Exactly the box poketenashi/sampler.py saves - both
    hips, plus each wrist that is at its hip, widened by 0.45 torso, from 0.45
    torso above the hip line to 0.75 below - because the model was trained on
    those samples. None when a hip is not seen or no wrist is at a hip."""
    t = p.torso_len(HIP_MC)
    hips = [p.kp("l_hip", HIP_MC), p.kp("r_hip", HIP_MC)]
    if hips[0] is None or hips[1] is None:
        return None
    wrists = [p.kp("l_wrist", HIP_MC), p.kp("r_wrist", HIP_MC)]
    near = [w for w, h in zip(wrists, hips) if w is not None
            and abs(w[1] - h[1]) <= 0.5 * t and abs(w[0] - h[0]) <= 0.6 * t]
    if not near:
        return None
    pel = (hips[0] + hips[1]) / 2.0
    xs = [h[0] for h in hips] + [w[0] for w in near]
    x1, y1 = int(min(xs) - 0.45 * t), int(pel[1] - 0.45 * t)
    x2, y2 = int(max(xs) + 0.45 * t), int(pel[1] + 0.75 * t)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame_w, x2), min(frame_h, y2)
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None
    return x1, y1, x2, y2


class PocketModel:
    def __init__(self, path, threads=2):
        import torchvision.models as tvm
        net = tvm.mobilenet_v3_small(weights=None)
        net.classifier[3] = nn.Linear(net.classifier[3].in_features, 1)
        net.load_state_dict(torch.load(path, map_location="cpu"))
        self.net = net.eval()
        self.threads = int(threads)
        self.path = path

    @torch.inference_mode()
    def score(self, crops):
        """P(pocket) for each BGR crop (arm_crop output); mirrored and averaged,
        as in training."""
        if not crops:
            return []
        old = torch.get_num_threads()
        torch.set_num_threads(self.threads)
        try:
            ims = []
            for c in crops:
                rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
                im = cv2.resize(rgb, (NET, NET)).astype(np.float32) / 255.0
                ims.append(im)
                ims.append(np.ascontiguousarray(im[:, ::-1]))
            a = (np.stack(ims) - MEAN) / STD
            x = torch.from_numpy(a.transpose(0, 3, 1, 2).copy())
            p = torch.sigmoid(self.net(x).squeeze(1)).numpy()
            return [float((p[2 * i] + p[2 * i + 1]) / 2.0) for i in range(len(crops))]
        finally:
            torch.set_num_threads(old)

    @torch.inference_mode()
    def score_embed(self, crops):
        """score() AND the master-matching embedding (po_refs.embed) from ONE pass
        of the network (9 Oct: ~11 ms a hand instead of ~24)."""
        if not crops:
            return [], np.zeros((0, 576), np.float32)
        old = torch.get_num_threads()
        torch.set_num_threads(self.threads)
        try:
            ims = []
            for c in crops:
                rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
                im = cv2.resize(rgb, (NET, NET)).astype(np.float32) / 255.0
                ims.append(im)
                ims.append(np.ascontiguousarray(im[:, ::-1]))
            a = (np.stack(ims) - MEAN) / STD
            x = torch.from_numpy(a.transpose(0, 3, 1, 2).copy())
            f = self.net.avgpool(self.net.features(x)).flatten(1)
            p = torch.sigmoid(self.net.classifier(f).squeeze(1)).numpy()
            f = f.numpy()
            e = (f[0::2] + f[1::2]) / 2.0
            e = e / np.maximum(1e-9, np.linalg.norm(e, axis=1, keepdims=True))
            return [float((p[2 * i] + p[2 * i + 1]) / 2.0) for i in range(len(crops))], e
        finally:
            torch.set_num_threads(old)


def load(cfg_po, key="model_path"):
    """The model named in rules.PO.<key> (model_path: the arm model,
    hip_model_path: the hip model), or None (then PO runs without it)."""
    path = (cfg_po or {}).get(key)
    if not path or (key == "model_path" and not cfg_po.get("use_model", True)):
        return None
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)
    if not os.path.exists(path):
        print(f"[PO] model {path} not found - PO runs on pose + MediaPipe", flush=True)
        return None
    if path in _LOADED:                    # one copy for every camera
        return _LOADED[path]
    m = PocketModel(path, threads=int(cfg_po.get("model_threads", 2)))
    if cfg_po.get("pipeline") == "evidence":
        thr = cfg_po.get("evidence_hi", 0.6)
    else:
        thr = cfg_po.get("hip_threshold", 0.7) if key == "hip_model_path" else cfg_po.get("model_threshold", 0.6)
    print(f"[PO] pocket model loaded: {os.path.basename(path)} "
          f"(a {'person' if key == 'hip_model_path' else 'hand'} counts as pocketed from {float(thr):.2f})",
          flush=True)
    _LOADED[path] = m
    return m


_LOADED = {}
