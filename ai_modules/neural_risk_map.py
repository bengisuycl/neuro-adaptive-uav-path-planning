import json
import os

import numpy as np
import torch
import torch.nn as nn

from wcci_conference_project.ai_modules.generate_real_dataset import extract_risk_features


def _build_layers(hidden_dims):
    dims = [4] + list(hidden_dims) + [1]
    layers = []
    for i in range(len(dims) - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    layers.append(nn.Softplus())
    return nn.Sequential(*layers)


class RiskApproximator(nn.Module):
    def __init__(self, hidden_dims=(128, 128, 64)):
        super(RiskApproximator, self).__init__()
        self.net = _build_layers(hidden_dims)

    def forward(self, x):
        return self.net(x)


class NeuralRiskMapper:
    _shared_cache = None

    def __init__(self, verbose=False):
        base_dir = os.path.dirname(__file__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if NeuralRiskMapper._shared_cache is None:
            meta = None
            pth_path = os.path.join(base_dir, "neural_risk.pth")
            meta_path = os.path.join(base_dir, "neural_risk_meta.json")

            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="ascii") as f:
                    meta = json.load(f)

            hidden_dims = tuple(meta.get("hidden_dims", [128, 128, 64])) if meta else (128, 128, 64)
            model = RiskApproximator(hidden_dims=hidden_dims)

            if not os.path.exists(pth_path):
                pth_path = os.path.join(base_dir, "..", "..", "neural_risk.pth")

            model.to(self.device)
            try:
                model.load_state_dict(torch.load(pth_path, map_location=self.device))
                model.eval()
                ready = True
                if verbose:
                    print(f"Neural Oracle Loaded from {pth_path}")
            except Exception:
                ready = False
                if verbose:
                    print(f"Neural Model NOT found at {pth_path}. Using Random Weights!")

            NeuralRiskMapper._shared_cache = {
                "meta": meta,
                "model": model,
                "ready": ready,
            }

        self.meta = NeuralRiskMapper._shared_cache["meta"]
        self.model = NeuralRiskMapper._shared_cache["model"]
        self.ready = bool(NeuralRiskMapper._shared_cache["ready"])

    def _normalize(self, features):
        if not self.meta:
            return features

        x_mean = np.asarray(self.meta.get("x_mean", [0.0] * features.shape[1]), dtype=np.float32)
        x_std = np.asarray(self.meta.get("x_std", [1.0] * features.shape[1]), dtype=np.float32)
        x_std = np.where(x_std < 1e-6, 1.0, x_std)
        return (features - x_mean) / x_std

    def _denormalize_preds(self, preds):
        if not self.meta:
            return preds

        transform_name = self.meta.get("target_transform", "identity")
        if transform_name == "log1p_div_scale":
            label_scale = float(self.meta.get("label_scale", 1.0))
            return np.expm1(preds) * max(label_scale, 1e-6)
        return preds

    def get_risk_at_points(self, points, terrain, threats, agl_m=800.0):
        if self.model is None or len(threats) == 0 or len(points) == 0:
            return [0.0] * len(points)

        features = []
        for px, py in np.asarray(points, dtype=np.float32):
            feat = extract_risk_features(terrain, float(px), float(py), float(agl_m), threats)
            features.append([
                feat["dist_norm"],
                feat["rel_alt_norm"],
                feat["slope"],
                feat["th_level"],
            ])

        x = np.asarray(features, dtype=np.float32)
        x = self._normalize(x)
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            preds = self.model(x_tensor).cpu().numpy().flatten()

        preds = self._denormalize_preds(preds)
        preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
        preds = np.clip(preds, 0.0, None)
        return preds.tolist()
