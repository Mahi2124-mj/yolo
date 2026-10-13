"""Image embedder: turns a hand crop into a compact feature vector ("fingerprint")
using a pretrained MobileNetV3-Small. Live crops are matched to labelled master
crops by cosine similarity in this space (few-shot, no training needed).

Runs on GPU if available.
"""
import numpy as np
import cv2
import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class Embedder:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights = MobileNet_V3_Small_Weights.DEFAULT
        net = mobilenet_v3_small(weights=weights)
        net.classifier = torch.nn.Identity()   # output the 576-d pooled features
        self.net = net.eval().to(self.device)
        self.tf = weights.transforms()          # resize/centre-crop/normalise
        self.dim = 576
        print(f"[embedder] MobileNetV3-Small ready on {self.device}.")

    @torch.inference_mode()
    def embed(self, crop_bgr):
        out = self.embed_batch([crop_bgr])
        return None if out is None else out[0]

    @torch.inference_mode()
    def embed_batch(self, crops):
        """Embed a list of BGR crops. Returns a list aligned with the input
        (None for invalid crops), or None if every crop is invalid."""
        valid = [(i, c) for i, c in enumerate(crops) if c is not None and c.size > 0]
        if not valid:
            return None
        tensors = []
        for _, c in valid:
            rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)  # CHW uint8
            tensors.append(self.tf(t))
        batch = torch.stack(tensors).to(self.device)
        v = self.net(batch).float()
        v = v / (v.norm(dim=1, keepdim=True) + 1e-6)   # L2-normalise -> cosine via dot
        arr = v.cpu().numpy()

        out = [None] * len(crops)
        for j, (i, _) in enumerate(valid):
            out[i] = arr[j]
        return out
