import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


HIDDEN_DIMS = (128, 128, 64)


class RiskApproximator(nn.Module):
    def __init__(self):
        super(RiskApproximator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, HIDDEN_DIMS[0]),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIMS[0], HIDDEN_DIMS[1]),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIMS[1], HIDDEN_DIMS[2]),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIMS[2], 1),
            nn.Softplus(),
        )

    def forward(self, x):
        return self.net(x)


def _normalize(values, mean, std):
    return (values - mean) / np.where(std < 1e-6, 1.0, std)


def _transform_targets(values, label_scale):
    return np.log1p(np.clip(values, 0.0, None) / max(label_scale, 1e-6))


def _inverse_transform_targets(values, label_scale):
    return np.expm1(values) * max(label_scale, 1e-6)


def _weighted_regression_loss(pred, target, weights):
    huber = nn.functional.smooth_l1_loss(pred, target, reduction="none", beta=0.20)
    mse = (pred - target) ** 2
    return torch.mean(weights * (0.65 * huber + 0.35 * mse))


def train(
    dataset_name="risk_dataset_glos.csv",
    epochs=120,
    lr=1e-3,
    batch_size=192,
    early_stopping_patience=24,
    min_delta=1e-4,
):
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, dataset_name)
    if not os.path.exists(csv_path):
        print("Dataset yok! Once generate_real_dataset.py calistir.")
        return

    print("Loading Data...")
    df = pd.read_csv(csv_path)
    feature_cols = ["dist_norm", "rel_alt_norm", "slope", "th_level"]
    label_col = "risk_label"

    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(np.float32).reshape(-1, 1)

    rng = np.random.default_rng(42)
    indices = rng.permutation(len(X))
    split = int(len(indices) * 0.85)
    train_idx, val_idx = indices[:split], indices[split:]

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)

    label_scale = float(np.percentile(y_train, 90))
    label_scale = max(label_scale, 0.25)
    y_train_t = _transform_targets(y_train, label_scale)
    y_val_t = _transform_targets(y_val, label_scale)

    x_train_n = _normalize(X_train, x_mean, x_std)
    x_val_n = _normalize(X_val, x_mean, x_std)

    train_weights = 1.0 + 6.0 * np.clip(y_train_t, 0.0, None) + 3.0 * (y_train >= np.percentile(y_train, 85)).astype(np.float32)
    val_weights = 1.0 + 6.0 * np.clip(y_val_t, 0.0, None) + 3.0 * (y_val >= np.percentile(y_train, 85)).astype(np.float32)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(x_train_n, dtype=torch.float32),
        torch.tensor(y_train_t, dtype=torch.float32),
        torch.tensor(train_weights.astype(np.float32), dtype=torch.float32),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = RiskApproximator()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6)

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    epochs_without_improvement = 0

    print("Training Neural Oracle...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for bx, by, bw in loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = _weighted_regression_loss(pred, by, bw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += float(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(torch.tensor(x_val_n, dtype=torch.float32))
            val_loss = float(
                _weighted_regression_loss(
                    val_pred,
                    torch.tensor(y_val_t, dtype=torch.float32),
                    torch.tensor(val_weights.astype(np.float32), dtype=torch.float32),
                ).item()
            )
        scheduler.step(val_loss)

        if val_loss < (best_val - min_delta):
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 10 == 0 or epoch == epochs - 1:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch:03d}: "
                f"train_loss={total_loss / max(1, len(loader)):.5f}, "
                f"val_loss={val_loss:.5f}, "
                f"lr={current_lr:.6f}"
            )

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch}, best val: {best_val:.5f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_pred_t = model(torch.tensor(x_val_n, dtype=torch.float32)).cpu().numpy()
    val_pred = _inverse_transform_targets(val_pred_t, label_scale)

    save_path = os.path.join(base_dir, "neural_risk.pth")
    checkpoint_path = os.path.join(base_dir, "neural_risk_best.pth")
    meta_path = os.path.join(base_dir, "neural_risk_meta.json")
    torch.save(model.state_dict(), save_path)
    torch.save(model.state_dict(), checkpoint_path)

    metadata = {
        "feature_columns": feature_cols,
        "label_column": label_col,
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "target_transform": "log1p_div_scale",
        "label_scale": label_scale,
        "dataset_path": os.path.basename(csv_path),
        "validation_loss": best_val,
        "best_epoch": best_epoch,
        "prediction_min": float(np.min(val_pred)),
        "prediction_max": float(np.max(val_pred)),
        "prediction_mean": float(np.mean(val_pred)),
        "target_min": float(np.min(y_val)),
        "target_max": float(np.max(y_val)),
        "target_mean": float(np.mean(y_val)),
        "scenario_ids": sorted(df["scenario_id"].dropna().unique().tolist()) if "scenario_id" in df.columns else [],
        "hidden_dims": list(HIDDEN_DIMS),
        "batch_size": batch_size,
        "epochs_requested": epochs,
        "early_stopping_patience": early_stopping_patience,
    }
    with open(meta_path, "w", encoding="ascii") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model Saved: {save_path}")
    print(f"Best Checkpoint Saved: {checkpoint_path}")
    print(f"Metadata Saved: {meta_path}")
    print(
        "Validation prediction stats:",
        f"min={metadata['prediction_min']:.4f}",
        f"max={metadata['prediction_max']:.4f}",
        f"mean={metadata['prediction_mean']:.4f}",
    )


if __name__ == "__main__":
    train()
