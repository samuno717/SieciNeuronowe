import time
import torch
import torch.nn as nn
from torchvision import datasets
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from loguru import logger

from config import PROJECT_ROOT, LAB5_ARTIFACT_DIR

DATA_DIR = PROJECT_ROOT / "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MLP_BASELINE_ACC = 56.36

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def set_deterministic():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_plot(filename_stem: str):
    LAB5_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(LAB5_ARTIFACT_DIR / f"{filename_stem}.svg", bbox_inches="tight", dpi=300)
    plt.savefig(LAB5_ARTIFACT_DIR / f"{filename_stem}.png", bbox_inches="tight", dpi=200)
    plt.close()


class TensorBatches:
    def __init__(self, X: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool):
        self.X, self.y = X, y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = X.shape[0]

    def __len__(self) -> int:
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            idx = torch.randperm(self.n_samples, device=self.X.device)
        else:
            idx = torch.arange(self.n_samples, device=self.X.device)
        for start in range(0, self.n_samples, self.batch_size):
            b = idx[start:start + self.batch_size]
            yield self.X[b], self.y[b]


def _load_split(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    ds = datasets.CIFAR10(root=DATA_DIR, train=train, download=True)
    X = torch.tensor(ds.data, dtype=torch.float32).permute(0, 3, 1, 2).div_(255.0)  # (N, 3, 32, 32)
    mean = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD).view(1, 3, 1, 1)
    X = (X - mean) / std
    y = torch.tensor(ds.targets, dtype=torch.long)
    return X.to(DEVICE), y.to(DEVICE)


def get_cifar10_loaders(batch_size: int = 128) -> tuple[TensorBatches, TensorBatches, TensorBatches]:
    X_tr, y_tr = _load_split(train=True)
    X_val, y_val = _load_split(train=False)
    train_loader = TensorBatches(X_tr, y_tr, batch_size, shuffle=True)
    train_eval_loader = TensorBatches(X_tr, y_tr, 1000, shuffle=False)
    val_loader = TensorBatches(X_val, y_val, 1000, shuffle=False)
    return train_loader, train_eval_loader, val_loader


class FlexibleCNN(nn.Module):
    def __init__(
        self,
        channels_list: list[int],
        kernel_size: int = 3,
        use_bn: bool = False,
        dropout_p: float = 0.0,
        hidden_dim: int = 256,
        num_classes: int = 10,
    ):
        super().__init__()
        layers = []
        in_channels = 3
        padding = kernel_size // 2

        for out_channels in channels_list:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
            if use_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels

        self.conv_blocks = nn.Sequential(*layers)

        classifier_layers = [
            nn.Flatten(),
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
        ]
        if dropout_p > 0.0:
            classifier_layers.append(nn.Dropout(p=dropout_p))
        classifier_layers.append(nn.Linear(hidden_dim, num_classes))

        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)
        return self.classifier(x)


@torch.no_grad()
def evaluate(model: nn.Module, loader: TensorBatches, loss_fn: nn.Module) -> tuple[float, float]:
    model.eval()
    loss_sum, correct = 0.0, 0
    for X, y in loader:
        pred = model(X)
        loss_sum += loss_fn(pred, y).item() * X.shape[0]
        correct += (pred.argmax(1) == y).sum().item()
    return loss_sum / loader.n_samples, correct / loader.n_samples * 100.0


def train_and_evaluate(
    exp_group: str,
    variant_name: str,
    model: nn.Module,
    train_loader: TensorBatches,
    train_eval_loader: TensorBatches,
    val_loader: TensorBatches,
    epochs: int = 15,
    lr: float = 1e-3,
    use_adamw: bool = False,
    weight_decay: float = 0.0,
) -> tuple[dict, dict, list[dict]]:
    model = model.to(DEVICE)

    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, 32, 32, device=DEVICE))

    loss_fn = nn.CrossEntropyLoss()
    if use_adamw:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"epochs": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    epoch_rows = []

    logger.info(f"Start -> [{exp_group}] Wariant: {variant_name}")
    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        for X, y in train_loader:
            pred = model(X)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # metryki po epoce, w trybie eval
        tr_loss, tr_acc = evaluate(model, train_eval_loader, loss_fn)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn)

        history["epochs"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        epoch_rows.append({
            "Eksperyment": exp_group,
            "Wariant": variant_name,
            "Epoka": epoch,
            "Train Loss": round(tr_loss, 4),
            "Val Loss": round(val_loss, 4),
            "Train Acc [%]": round(tr_acc, 2),
            "Val Acc [%]": round(val_acc, 2),
        })

        if epoch % 3 == 0 or epoch == 1 or epoch == epochs:
            logger.info(
                f"  [{variant_name}] Epoka {epoch:2d}/{epochs} -> "
                f"Train Loss: {tr_loss:.4f} ({tr_acc:.1f}%) | Val Loss: {val_loss:.4f} ({val_acc:.1f}%)"
            )

    elapsed = time.perf_counter() - t_start
    best_val_acc = max(history["val_acc"])
    min_val_loss = min(history["val_loss"])
    n_params = sum(p.numel() for p in model.parameters())

    summary_row = {
        "Eksperyment": exp_group,
        "Wariant": variant_name,
        "Parametry": n_params,
        "Train Loss (koniec)": round(history["train_loss"][-1], 4),
        "Val Loss (min)": round(min_val_loss, 4),
        "Epoka min Val Loss": history["epochs"][history["val_loss"].index(min_val_loss)],
        "Val Loss (koniec)": round(history["val_loss"][-1], 4),
        "Train Acc (koniec) [%]": round(history["train_acc"][-1], 2),
        "Val Acc (koniec) [%]": round(history["val_acc"][-1], 2),
        "Val Acc (max) [%]": round(best_val_acc, 2),
        "Luka Overfittingu (Train-Val Acc) [pkt %]": round(history["train_acc"][-1] - history["val_acc"][-1], 2),
        "Czas [s]": round(elapsed, 1),
    }
    logger.info(
        f"Koniec -> [{variant_name}] ({elapsed:.1f}s) | Max Val Acc: {best_val_acc:.2f}% | "
        f"Końcowy Val Loss: {history['val_loss'][-1]:.4f} (min: {min_val_loss:.4f})"
    )
    return history, summary_row, epoch_rows


def plot_experiment(results_dict: dict[str, dict], title_prefix: str, filename_stem: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5), dpi=300)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]
    assert len(results_dict) <= len(colors), "Za mało kolorów dla liczby wariantów"

    max_ep = 15
    for (variant_name, hist), col in zip(results_dict.items(), colors):
        ep = hist["epochs"]
        max_ep = max(ep)
        ax1.plot(ep, hist["val_loss"], color=col, linestyle="-", marker="o", markersize=3, linewidth=1.8, label=f"{variant_name} (Val)")
        ax1.plot(ep, hist["train_loss"], color=col, linestyle="--", alpha=0.55, linewidth=1.3, label=f"{variant_name} (Train)")

        ax2.plot(ep, hist["val_acc"], color=col, linestyle="-", marker="s", markersize=3, linewidth=1.8, label=f"{variant_name} (Val)")
        ax2.plot(ep, hist["train_acc"], color=col, linestyle="--", alpha=0.45, linewidth=1.2, label=f"{variant_name} (Train)")

    ax2.axhline(MLP_BASELINE_ACC, color="gray", linestyle=":", linewidth=1.5, label=f"MLP Baseline ({MLP_BASELINE_ACC:.1f}%)")

    int_ticks = list(range(2, max_ep + 1, 2))
    for ax in (ax1, ax2):
        ax.set_xticks(int_ticks)
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.set_xlabel("Epoka")
        ax.grid(True, linestyle="--", alpha=0.5)

    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title(f"{title_prefix}: Funkcja straty (Train vs Val)")
    ax2.set_ylabel("Accuracy [%]")
    ax2.set_title(f"{title_prefix}: Dokładność klasyfikacji")

    for ax in (ax1, ax2):
        ax.legend(fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)

    plt.tight_layout()
    save_plot(filename_stem)
    logger.info(f"Zapisano wykres: {filename_stem}.svg")
