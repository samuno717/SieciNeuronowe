import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from loguru import logger

from config import LAB5_ARTIFACT_DIR
from lab5.cnn_utils import DATA_DIR, DEVICE, TensorBatches, save_plot

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CNN_BEST_ACC = 78.78  # najlepsza sieć (BN + Dropout + AdamW 0.1)

CLASSES_PL = ["samolot", "samochód", "ptak", "kot", "jeleń", "pies", "żaba", "koń", "statek", "ciężarówka"]


def _load_split_raw(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    # obrazy w [0, 1], bez normalizacji
    ds = datasets.CIFAR10(root=DATA_DIR, train=train, download=True)
    X = torch.tensor(ds.data, dtype=torch.float32).permute(0, 3, 1, 2).div_(255.0)
    y = torch.tensor(ds.targets, dtype=torch.long)
    return X.to(DEVICE), y.to(DEVICE)


def get_raw_loaders(batch_size: int = 128, n_train: int | None = None, n_train_eval: int = 10_000, seed: int = 0):
    X_tr, y_tr = _load_split_raw(train=True)
    X_val, y_val = _load_split_raw(train=False)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(X_tr.shape[0], generator=g).to(DEVICE)

    if n_train is not None:
        X_tr, y_tr = X_tr[perm[:n_train]], y_tr[perm[:n_train]]
        perm = torch.arange(X_tr.shape[0], device=DEVICE)

    ev = perm[:n_train_eval]
    train_loader = TensorBatches(X_tr, y_tr, batch_size, shuffle=True)
    train_eval_loader = TensorBatches(X_tr[ev], y_tr[ev], 500, shuffle=False)
    val_loader = TensorBatches(X_val, y_val, 500, shuffle=False)
    return train_loader, train_eval_loader, val_loader


def preprocess(x: torch.Tensor, size: int) -> torch.Tensor:
    if size != x.shape[-1]:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=False)

    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def unnormalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def make_head(in_features: int, num_classes: int = 10, hidden: int = 256, dropout: float = 0.5) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, num_classes),
    )


def build_model(arch: str, pretrained: bool) -> nn.Module:
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = make_head(model.fc.in_features)
        head_name, gradcam_layer = "fc", "layer4"
    elif arch == "vgg11_bn":
        model = models.vgg11_bn(weights=models.VGG11_BN_Weights.IMAGENET1K_V1 if pretrained else None)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.classifier = make_head(512)
        head_name, gradcam_layer = "classifier", "features.27"  # ostatni ReLU przed ostatnim poolingiem (14x14 przy 224)
    else:
        raise ValueError(arch)
    model.head_name, model.gradcam_layer_name = head_name, gradcam_layer
    return model


def get_head(model: nn.Module) -> nn.Module:
    return model.get_submodule(model.head_name)


def get_gradcam_layer(model: nn.Module) -> nn.Module:
    return model.get_submodule(model.gradcam_layer_name)


def split_params(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    head = list(get_head(model).parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    return backbone, head


def set_backbone_frozen(model: nn.Module, frozen: bool):
    backbone, _ = split_params(model)
    for p in backbone:
        p.requires_grad_(not frozen)


def set_train_mode(model: nn.Module, backbone_frozen: bool):
    model.train()
    if backbone_frozen:
        for name, m in model.named_children():
            if name != model.head_name:
                m.eval()


@dataclass(frozen=True)
class Strategy:
    name: str
    pretrained: bool = True
    freeze_epochs: int = 0
    lr_head: float = 1e-3
    lr_backbone: float = 1e-3
    warmup_epochs: float = 0.0


@torch.no_grad()
def evaluate(model: nn.Module, loader: TensorBatches, size: int) -> tuple[float, float]:
    model.eval()
    loss_sum, correct = 0.0, 0
    for X, y in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(preprocess(X, size).contiguous(memory_format=torch.channels_last))
        out = out.float()
        loss_sum += F.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
    return loss_sum / loader.n_samples, correct / loader.n_samples * 100.0


def train_transfer(
    exp_group: str,
    variant_name: str,
    arch: str,
    strategy: Strategy,
    size: int,
    loaders,
    epochs: int,
    seed: int = 42,
    flip: bool = True,
) -> tuple[nn.Module, dict, dict, list[dict]]:
    train_loader, train_eval_loader, val_loader = loaders
    torch.manual_seed(seed)
    model = build_model(arch, strategy.pretrained).to(DEVICE).to(memory_format=torch.channels_last)

    backbone, head = split_params(model)
    optimizer = torch.optim.Adam([
        {"params": backbone, "lr": strategy.lr_backbone},
        {"params": head, "lr": strategy.lr_head},
    ])
    steps_per_epoch = len(train_loader)
    warmup_steps = int(strategy.warmup_epochs * steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup_steps) if warmup_steps > 0 else 1.0
    )

    history = {"epochs": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    epoch_rows = []
    logger.info(f"Start -> [{exp_group}] {variant_name} ({arch}, {size}x{size})")
    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        frozen = epoch <= strategy.freeze_epochs
        set_backbone_frozen(model, frozen)
        set_train_mode(model, frozen)
        for X, y in train_loader:
            if flip:
                m = torch.rand(X.shape[0], device=X.device) < 0.5
                X = torch.where(m.view(-1, 1, 1, 1), X.flip(-1), X)
            X = preprocess(X, size).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(X), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        tr_loss, tr_acc = evaluate(model, train_eval_loader, size)
        val_loss, val_acc = evaluate(model, val_loader, size)
        for k, v in zip(history, (epoch, tr_loss, val_loss, tr_acc, val_acc)):
            history[k].append(v)
        epoch_rows.append({
            "Eksperyment": exp_group, "Wariant": variant_name, "Epoka": epoch,
            "Ekstraktor zamrożony": frozen,
            "Train Loss": round(tr_loss, 4), "Val Loss": round(val_loss, 4),
            "Train Acc [%]": round(tr_acc, 2), "Val Acc [%]": round(val_acc, 2),
        })
        logger.info(
            f"  [{variant_name}] Epoka {epoch}/{epochs}{' (zamr.)' if frozen else ''} -> "
            f"Train {tr_loss:.4f} ({tr_acc:.1f}%) | Val {val_loss:.4f} ({val_acc:.1f}%) "
            f"| {time.perf_counter() - t_start:.0f}s"
        )

    elapsed = time.perf_counter() - t_start
    min_val_loss = min(history["val_loss"])
    summary = {
        "Eksperyment": exp_group,
        "Wariant": variant_name,
        "Model": arch,
        "Rozdzielczość": size,
        "Pretrenowany": strategy.pretrained,
        "Epoki zamrożenia": min(strategy.freeze_epochs, epochs),
        "LR ekstraktor": strategy.lr_backbone,
        "LR głowa": strategy.lr_head,
        "Warmup [epoki]": strategy.warmup_epochs,
        "Val Acc po 1. epoce [%]": round(history["val_acc"][0], 2),
        "Val Loss (min)": round(min_val_loss, 4),
        "Epoka min Val Loss": history["epochs"][history["val_loss"].index(min_val_loss)],
        "Val Loss (koniec)": round(history["val_loss"][-1], 4),
        "Train Acc (koniec) [%]": round(history["train_acc"][-1], 2),
        "Val Acc (koniec) [%]": round(history["val_acc"][-1], 2),
        "Val Acc (max) [%]": round(max(history["val_acc"]), 2),
        "Luka (Train-Val Acc) [pkt %]": round(history["train_acc"][-1] - history["val_acc"][-1], 2),
        "Czas [s]": round(elapsed, 1),
    }
    logger.info(f"Koniec -> [{variant_name}] ({elapsed:.0f}s) | Val Acc koniec {history['val_acc'][-1]:.2f}% "
                f"(max {max(history['val_acc']):.2f}%)")
    return model, history, summary, epoch_rows


def plot_transfer(results: dict[str, dict], title: str, filename_stem: str, freeze_marks: dict[str, int] | None = None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5), dpi=300)
    colors = ["tab:gray", "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]
    max_ep = 1
    for (name, h), col in zip(results.items(), colors):
        ep = h["epochs"]
        max_ep = max(max_ep, max(ep))
        ax1.plot(ep, h["val_loss"], color=col, marker="o", markersize=3, linewidth=1.8, label=f"{name} (Val)")
        ax1.plot(ep, h["train_loss"], color=col, linestyle="--", alpha=0.55, linewidth=1.3, label=f"{name} (Train)")
        ax2.plot(ep, h["val_acc"], color=col, marker="s", markersize=3, linewidth=1.8, label=f"{name} (Val)")
        ax2.plot(ep, h["train_acc"], color=col, linestyle="--", alpha=0.45, linewidth=1.2, label=f"{name} (Train)")
        if freeze_marks and name in freeze_marks:
            for ax in (ax1, ax2):
                ax.axvline(freeze_marks[name] + 0.5, color=col, linestyle=":", linewidth=1.0, alpha=0.8)
    ax2.axhline(CNN_BEST_ACC, color="black", linestyle=":", linewidth=1.3, label=f"Najlepsza CNN z cz. 1 ({CNN_BEST_ACC:.1f}%)")

    for ax in (ax1, ax2):
        ax.set_xticks(range(1, max_ep + 1))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.set_xlabel("Epoka")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title(f"{title}: Funkcja straty")
    ax2.set_ylabel("Accuracy [%]")
    ax2.set_title(f"{title}: Dokładność klasyfikacji")
    plt.tight_layout()
    save_plot(filename_stem)
    logger.info(f"Zapisano wykres: {filename_stem}.svg")


def plot_resolution(rows: list[dict], filename_stem: str):
    model_color = {"ResNet18": "#2a78d6", "VGG11-BN": "#eb6834"}
    strategy_style = {"fine-tuning": ("-", "o"), "zamrożony": ("--", "s"), "od zera": (":", "^")}

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    series: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        series.setdefault(r["Seria"], []).append((r["Rozdzielczość"], r["Val Acc (koniec) [%]"]))
    order = lambda n: (list(model_color).index(n.split(",")[0]),
                       next(i for i, k in enumerate(strategy_style) if k in n))
    all_pts = [p for pts in series.values() for p in pts]
    for name in sorted(series, key=order):
        pts = sorted(series[name])
        xs, ys = zip(*pts)
        ls, mk = next(v for k, v in strategy_style.items() if k in name)
        ax.plot(xs, ys, color=model_color[name.split(",")[0]], linestyle=ls, marker=mk, markersize=6,
                linewidth=2, label=name)
        for x, y in pts:
            crowded = any(px == x and 0 < py - y < 2.0 for px, py in all_pts) or 0 < CNN_BEST_ACC - y < 1.5
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, -12 if crowded else 6),
                        ha="center", fontsize=7, color="#3d3d3a")
    ax.axhline(CNN_BEST_ACC, color="#8a8a85", linestyle=(0, (1, 2)), linewidth=1.3,
               label=f"Najlepsza CNN z cz. 1 ({CNN_BEST_ACC:.1f}%)")
    ax.set_xscale("log", base=2)
    xs_all = sorted({r["Rozdzielczość"] for r in rows})
    ax.set_xticks(xs_all)
    ax.set_xticklabels([f"{x}×{x}" for x in xs_all])
    ax.minorticks_off()
    ax.set_xlabel("Rozdzielczość wejścia sieci")
    ax.set_ylabel("Val Accuracy (koniec) [%]")
    ax.set_title("Wpływ przeskalowania obrazów CIFAR-10")
    ax.grid(True, linestyle="--", alpha=0.35)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14), handlelength=3)
    plt.tight_layout()
    save_plot(filename_stem)


def save_model_structure(arch: str):
    LAB5_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ref = getattr(models, arch)(weights=None)
    (LAB5_ARTIFACT_DIR / f"struktura_{arch}.txt").write_text(
        f"# print(model) - oryginalny {arch} z torchvision\n{ref}\n\n"
        f"# po wymianie głowy\n{build_model(arch, pretrained=False)}\n",
        encoding="utf-8",
    )
