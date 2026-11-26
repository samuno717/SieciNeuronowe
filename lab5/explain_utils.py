import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from captum.attr import Saliency, NoiseTunnel, LayerGradCam, LayerAttribution, Occlusion
from loguru import logger

from lab5.cnn_utils import save_plot
from lab5.transfer_utils import CLASSES_PL, preprocess, unnormalize, get_gradcam_layer

OCC_WINDOW_FRAC = 0.20
OCC_STRIDE_FRAC = 0.10
SMOOTHGRAD_SAMPLES = 25
SMOOTHGRAD_STD = 0.15  # odchylenie szumu jako ułamek zakresu wartości wejścia
MAX_IMAGES_PER_PASS = 256  # captum mnoży paczkę przez liczbę próbek szumu / pozycji okna


class Explainer:

    def __init__(self, model: nn.Module, size: int):
        self.model = model.eval()
        self.size = size
        for m in model.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False
        self.saliency = Saliency(model)
        self.smoothgrad = NoiseTunnel(Saliency(model))
        self.gradcam = LayerGradCam(model, get_gradcam_layer(model))
        self.occlusion = Occlusion(model)
        self.occ_window = int(round(OCC_WINDOW_FRAC * size))
        self.occ_stride = int(round(OCC_STRIDE_FRAC * size))

    def prep(self, x_raw: torch.Tensor) -> torch.Tensor:
        return preprocess(x_raw, self.size)

    @torch.no_grad()
    def predict(self, x_raw: torch.Tensor, batch: int = 250) -> tuple[torch.Tensor, torch.Tensor]:
        probs = []
        for i in range(0, x_raw.shape[0], batch):
            probs.append(F.softmax(self.model(self.prep(x_raw[i:i + batch])).float(), dim=1))
        p = torch.cat(probs)
        return p.argmax(1), p

    # mapa (N, H, W)
    def saliency_map(self, x, target):
        a = self.saliency.attribute(x.requires_grad_(), target=target, abs=True)
        return a.amax(dim=1).detach()

    def smoothgrad_map(self, x, target):
        std = SMOOTHGRAD_STD * (x.max() - x.min()).item()
        a = self.smoothgrad.attribute(x.requires_grad_(), nt_type="smoothgrad", nt_samples=SMOOTHGRAD_SAMPLES,
                                      nt_samples_batch_size=max(1, MAX_IMAGES_PER_PASS // 2 // x.shape[0]),
                                      stdevs=std, target=target, abs=True)
        return a.amax(dim=1).detach()

    def gradcam_map(self, x, target):
        a = self.gradcam.attribute(x.requires_grad_(), target=target, relu_attributions=True)
        self.gradcam_res = a.shape[-1]
        a = LayerAttribution.interpolate(a, (x.shape[-2], x.shape[-1]), interpolate_mode="bilinear")
        return a[:, 0].detach()

    @torch.no_grad()
    def occlusion_map(self, x, target):
        w, s = self.occ_window, self.occ_stride
        # baseline 0 w przestrzeni znormalizowanej = średni kolor ImageNet
        a = self.occlusion.attribute(x, target=target, sliding_window_shapes=(3, w, w), strides=(3, s, s),
                                     baselines=0.0, perturbations_per_eval=max(1, MAX_IMAGES_PER_PASS // x.shape[0]))
        return a.sum(dim=1).detach()

    def all_maps(self, x_raw: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.prep(x_raw)
        return {
            "Saliency": self.saliency_map(x.clone(), target),
            "SmoothGrad": self.smoothgrad_map(x.clone(), target),
            "Grad-CAM": self.gradcam_map(x.clone(), target),
            "Occlusion": self.occlusion_map(x.clone(), target),
        }


def _norm_abs(a: np.ndarray, pct: float = 99.0) -> np.ndarray:
    # przycięcie na percentylu
    hi = np.percentile(a, pct)
    return np.clip(a / (hi + 1e-12), 0, 1)


def _show_map(ax, img_gray: np.ndarray, a: np.ndarray, method: str):
    if method in ("Saliency", "SmoothGrad"):
        ax.imshow(_norm_abs(a), cmap="hot", vmin=0, vmax=1)
    elif method == "Grad-CAM":
        ax.imshow(img_gray, cmap="gray", vmin=0, vmax=1)
        ax.imshow(a / (a.max() + 1e-12), cmap="jet", alpha=0.5, vmin=0, vmax=1)
    else:
        m = np.abs(a).max() + 1e-12
        ax.imshow(img_gray, cmap="gray", vmin=0, vmax=1)
        ax.imshow(a, cmap="bwr", alpha=0.55, vmin=-m, vmax=m)
    ax.set_xticks([]), ax.set_yticks([])


def _to_np_img(x_net: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    img = unnormalize(x_net.unsqueeze(0))[0].permute(1, 2, 0).cpu().numpy()
    return img, img.mean(axis=2)


def plot_explanations(expl: Explainer, X_raw, y_true, y_pred, probs, idx: list[int], title: str, filename_stem: str):
    methods = ["Saliency", "SmoothGrad", "Grad-CAM", "Occlusion"]
    n = len(idx)
    fig, axes = plt.subplots(n, 5, figsize=(11, 2.25 * n), dpi=200)
    x_raw = X_raw[idx]
    tgt = y_pred[idx]
    maps = expl.all_maps(x_raw, tgt)
    x_net = expl.prep(x_raw)
    for r, i in enumerate(idx):
        img, gray = _to_np_img(x_net[r])
        axes[r, 0].imshow(img)
        axes[r, 0].set_xticks([]), axes[r, 0].set_yticks([])
        ok = y_true[i] == y_pred[i]
        axes[r, 0].set_ylabel(f"prawda: {CLASSES_PL[y_true[i]]}\npred.: {CLASSES_PL[y_pred[i]]} ({probs[i, y_pred[i]]:.0%})",
                              fontsize=8, color="black" if ok else "tab:red")
        for c, m in enumerate(methods, start=1):
            _show_map(axes[r, c], gray, maps[m][r].cpu().numpy(), m)
    col_titles = ["Obraz (wejście sieci)", "Saliency |∂y/∂x|", f"SmoothGrad (n={SMOOTHGRAD_SAMPLES})",
                  f"Grad-CAM ({expl.gradcam_res}×{expl.gradcam_res})", f"Occlusion (okno {expl.occ_window}, krok {expl.occ_stride})"]
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=8.5)
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    save_plot(filename_stem)
    logger.info(f"Zapisano: {filename_stem}")


def plot_errors_pred_vs_true(expl: Explainer, X_raw, y_true, y_pred, probs, idx: list[int], filename_stem: str):
    n = len(idx)
    fig, axes = plt.subplots(n, 5, figsize=(11, 2.25 * n), dpi=200)
    x_raw = X_raw[idx]
    x_net = expl.prep(x_raw)
    t_pred, t_true = y_pred[idx], y_true[idx]
    gc_pred, gc_true = expl.gradcam_map(x_net.clone(), t_pred), expl.gradcam_map(x_net.clone(), t_true)
    oc_pred, oc_true = expl.occlusion_map(x_net.clone(), t_pred), expl.occlusion_map(x_net.clone(), t_true)
    for r, i in enumerate(idx):
        img, gray = _to_np_img(x_net[r])
        axes[r, 0].imshow(img)
        axes[r, 0].set_xticks([]), axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(f"prawda: {CLASSES_PL[y_true[i]]} ({probs[i, y_true[i]]:.0%})\n"
                              f"pred.: {CLASSES_PL[y_pred[i]]} ({probs[i, y_pred[i]]:.0%})", fontsize=8, color="tab:red")
        for c, (a, m) in enumerate([(gc_pred, "Grad-CAM"), (oc_pred, "Occlusion"), (gc_true, "Grad-CAM"), (oc_true, "Occlusion")], 1):
            _show_map(axes[r, c], gray, a[r].cpu().numpy(), m)
        axes[r, 1].set_xlabel(f"→ {CLASSES_PL[y_pred[i]]}", fontsize=7.5)
        axes[r, 3].set_xlabel(f"→ {CLASSES_PL[y_true[i]]}", fontsize=7.5)
    for c, t in enumerate(["Obraz", "Grad-CAM: klasa przewidziana", "Occlusion: klasa przewidziana",
                           "Grad-CAM: klasa prawdziwa", "Occlusion: klasa prawdziwa"]):
        axes[0, c].set_title(t, fontsize=8.5)
    fig.suptitle("Błędne decyzje: wyjaśnienie klasy przewidzianej vs prawdziwej", fontsize=10)
    plt.tight_layout()
    save_plot(filename_stem)
    logger.info(f"Zapisano: {filename_stem}")


@torch.no_grad()
def _prob_of(expl: Explainer, x_net: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = F.softmax(expl.model(x_net).float(), dim=1)
    return p.gather(1, target.view(-1, 1))[:, 0], p.argmax(1)


def deletion_test(expl: Explainer, X_raw, y_pred, idx: list[int], fractions=(0.05, 0.10, 0.20), batch: int = 25,
                  patch: int = 16, seed: int = 0) -> list[dict]:

    g = torch.Generator(device=X_raw.device).manual_seed(seed)
    acc = {}
    for s in range(0, len(idx), batch):
        b = idx[s:s + batch]
        x_raw, tgt = X_raw[b], y_pred[b]
        maps = expl.all_maps(x_raw, tgt)
        x_net = expl.prep(x_raw)
        p0, _ = _prob_of(expl, x_net, tgt)
        side = x_net.shape[-1] // patch
        maps["Losowa kolejność łat"] = torch.rand(len(b), side, side, device=X_raw.device, generator=g)
        for name, a in maps.items():
            if a.shape[-1] != side:
                a = F.avg_pool2d(a.unsqueeze(1).float(), patch)[:, 0]
            flat = a.flatten(1)
            order = flat.argsort(dim=1, descending=True)
            for f in fractions:
                k = max(1, round(f * side * side))
                mask = torch.zeros_like(flat, dtype=torch.bool)
                mask.scatter_(1, order[:, :k], True)
                mask = mask.view(-1, 1, side, side).repeat_interleave(patch, 2).repeat_interleave(patch, 3)
                x_del = torch.where(mask, torch.zeros_like(x_net), x_net)
                p1, pred1 = _prob_of(expl, x_del, tgt)
                rec = acc.setdefault((name, f), {"drop": 0.0, "flip": 0, "n": 0})
                rec["drop"] += (p0 - p1).sum().item()
                rec["flip"] += (pred1 != tgt).sum().item()
                rec["n"] += len(b)
    rows = []
    for (name, f), r in acc.items():
        rows.append({
            "Metoda": name,
            "Zasłonięta powierzchnia [%]": int(f * 100),
            "Średni spadek p(klasa) [pkt %]": round(r["drop"] / r["n"] * 100, 2),
            "Zmiana decyzji [%]": round(r["flip"] / r["n"] * 100, 2),
            "N obrazów": r["n"],
        })
    return rows
