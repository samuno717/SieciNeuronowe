import shutil
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from config import LAB4_DIR, LAB4_ARTIFACT_DIR
from lab1.preprocessing import get_raw_data, process_features
from lab3.MLP import MLP as NumpyMLP

#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")


def save_plot(filename_stem: str):
    LAB4_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    local_path = LAB4_DIR / f"{filename_stem}.svg"
    plt.savefig(local_path, bbox_inches="tight", dpi=300)
    art_path = LAB4_ARTIFACT_DIR / f"{filename_stem}.svg"
    shutil.copy2(local_path, art_path)
    plt.close()


def load_data():
    X, y = get_raw_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val = process_features(X_train, X_val)

    X_train = np.asarray(X_train, dtype=np.float64)
    X_val = np.asarray(X_val, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1, 1)
    y_val = np.asarray(y_val, dtype=np.float64).reshape(-1, 1)
    return X_train, X_val, y_train, y_val


def clipped_bce_np(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    eps = 1e-15
    y_c = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(y_c) + (1.0 - y_true) * np.log(1.0 - y_c)))


class TorchMLP(nn.Module):
    def __init__(
        self,
        layer_dims: list[int],
        init_mode: str = "he",
        dtype: torch.dtype = torch.float64,
        init_strategy: str = "sync_numpy",
        transpose_weights: bool = True,
    ):
        super().__init__()
        self.dtype = dtype
        modules = []

        for i in range(len(layer_dims) - 1):
            in_f, out_f = layer_dims[i], layer_dims[i + 1]
            linear = nn.Linear(in_f, out_f, dtype=dtype)

            if init_mode == "he":
                w_scale = np.sqrt(2.0 / in_f)
            elif init_mode == "normal_1":
                w_scale = 1.0
            elif init_mode == "broken":
                w_scale = 5.0
            else:
                w_scale = 0.01

            if init_strategy == "sync_numpy":
                w_np = np.random.normal(loc=0.0, scale=w_scale, size=(in_f, out_f))
                with torch.no_grad():
                    if transpose_weights:
                        linear.weight.copy_(torch.tensor(w_np.T, dtype=dtype))
                    else:
                        linear.weight.copy_(torch.tensor(w_np.reshape(out_f, in_f), dtype=dtype))
                    linear.bias.zero_()
            elif init_strategy == "native_normal":
                nn.init.normal_(linear.weight, mean=0.0, std=float(w_scale))
                nn.init.zeros_(linear.bias)
            elif init_strategy == "default_uniform":
                pass

            modules.append(linear)
            if i < len(layer_dims) - 2:
                modules.append(nn.ReLU())

        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.clamp(self.forward(x), -500.0, 500.0)
        return torch.sigmoid(logits)

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            x_np = np.asarray(x, dtype=np.float64)
            x_t = torch.tensor(x_np, dtype=self.dtype, device=next(self.parameters()).device)
            probs = self.predict_proba(x_t).cpu().numpy()
        return (probs >= threshold).astype(int)


def fit_torch_mlp(
    model: TorchMLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    optimizer: torch.optim.Optimizer,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    epochs: int = 300,
    batch_size: int | None = None,
    shuffle: bool = True,
) -> tuple[list[float], list[float]]:
    n_samples = X_train.shape[0]
    actual_bs = n_samples if (batch_size is None or batch_size >= n_samples) else max(1, batch_size)

    xt_train = torch.tensor(X_train, dtype=model.dtype, device=DEVICE)
    yt_train = torch.tensor(y_train, dtype=model.dtype, device=DEVICE)

    has_val = X_val is not None and y_val is not None
    if has_val:
        xt_val = torch.tensor(X_val, dtype=model.dtype, device=DEVICE)

    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    train_losses, val_losses = [], []

    for _ in range(epochs):
        model.train()
        if shuffle and actual_bs < n_samples:
            indices = np.random.permutation(n_samples)
        else:
            indices = np.arange(n_samples)

        for start in range(0, n_samples, actual_bs):
            batch_idx = indices[start:start + actual_bs]
            logits = model(xt_train[batch_idx])
            loss = criterion(logits, yt_train[batch_idx])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        with torch.no_grad():
            p_train = model.predict_proba(xt_train).cpu().numpy()
            train_losses.append(clipped_bce_np(y_train, p_train))
            if has_val:
                p_val = model.predict_proba(xt_val).cpu().numpy()
                val_losses.append(clipped_bce_np(y_val, p_val))

    return train_losses, val_losses


def collect_metrics(
    exp_name: str,
    variant_name: str,
    np_model: NumpyMLP,
    th_model: TorchMLP,
    np_tr_loss: list[float],
    np_val_loss: list[float],
    th_tr_loss: list[float],
    th_val_loss: list[float],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> list[dict]:
    y_tr = y_train.ravel()
    y_v = y_val.ravel()

    rows = []
    for impl_name, model, tr_l, val_l in [
        ("NumPy", np_model, np_tr_loss[-1], np_val_loss[-1]),
        ("PyTorch", th_model, th_tr_loss[-1], th_val_loss[-1]),
    ]:
        pred_tr = model.predict(X_train).ravel()
        pred_val = model.predict(X_val).ravel()
        rows.append({
            "Eksperyment": exp_name,
            "Wariant": variant_name,
            "Model": impl_name,
            "Train Loss": round(tr_l, 6),
            "Val Loss": round(val_l, 6),
            "Train Acc": round(accuracy_score(y_tr, pred_tr), 4),
            "Val Acc": round(accuracy_score(y_v, pred_val), 4),
            "Train F1": round(f1_score(y_tr, pred_tr, zero_division=0), 4),
            "Val F1": round(f1_score(y_v, pred_val, zero_division=0), 4),
        })
    return rows