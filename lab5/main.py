import time
import pandas as pd
import torch
from loguru import logger

from config import LAB5_ARTIFACT_DIR
from lab5.cnn_utils import (
    DEVICE,
    FlexibleCNN,
    get_cifar10_loaders,
    set_deterministic,
    train_and_evaluate,
    plot_experiment,
)

EPOCHS = 15
SEED = 42


def main():
    t_total = time.perf_counter()
    set_deterministic()
    gpu_name = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU"
    logger.info(f"Urządzenie: {DEVICE} ({gpu_name})")

    train_loader, train_eval_loader, val_loader = get_cifar10_loaders(batch_size=128)

    all_summaries = []
    all_epochs = []
    cache: dict[tuple, tuple[dict, dict, list[dict]]] = {}

    def run(exp_group: str, name: str, channels: list[int], kernel: int = 3, use_bn: bool = False,
            dropout_p: float = 0.0, use_adamw: bool = False, wd: float = 0.0) -> dict:

        key = (tuple(channels), kernel, use_bn, dropout_p, use_adamw, wd)
        if key not in cache:
            torch.manual_seed(SEED)
            model = FlexibleCNN(channels_list=channels, kernel_size=kernel, use_bn=use_bn, dropout_p=dropout_p)
            cache[key] = train_and_evaluate(
                exp_group, name, model, train_loader, train_eval_loader, val_loader,
                epochs=EPOCHS, use_adamw=use_adamw, weight_decay=wd,
            )
        else:
            logger.info(f"[{exp_group}] {name}: konfiguracja policzona wcześniej - użyto zapisanego wyniku")
        hist, summary, ep_rows = cache[key]
        all_summaries.append({**summary, "Eksperyment": exp_group, "Wariant": name})
        all_epochs.extend({**r, "Eksperyment": exp_group, "Wariant": name} for r in ep_rows)
        return hist

    logger.info("=== EKSPERYMENT 1: Liczba kanałów wyjściowych ===")
    exp1 = {
        "Mała (8 -> 16)": [8, 16],
        "Średnia (32 -> 64)": [32, 64],
        "Duża (64 -> 128)": [64, 128],
    }
    h1 = {name: run("1. Liczba kanałów", name, ch) for name, ch in exp1.items()}
    plot_experiment(h1, "Eksperyment 1 (Kanały)", "exp1_channels")

    logger.info("=== EKSPERYMENT 2: Liczba warstw konwolucyjnych ===")
    exp2 = {
        "1 warstwa (32)": [32],
        "2 warstwy (32->64)": [32, 64],
        "3 warstwy (32->64->128)": [32, 64, 128],
    }
    h2 = {name: run("2. Liczba warstw", name, ch) for name, ch in exp2.items()}
    plot_experiment(h2, "Eksperyment 2 (Liczba warstw)", "exp2_layers")

    logger.info("=== EKSPERYMENT 3: Rozmiar filtra (kernel_size) ===")
    exp3 = {"Filtr 3x3": 3, "Filtr 5x5": 5, "Filtr 7x7": 7}
    h3 = {name: run("3. Rozmiar filtra", name, [32, 64, 128], kernel=k) for name, k in exp3.items()}
    plot_experiment(h3, "Eksperyment 3 (Rozmiar filtra)", "exp3_kernels")

    logger.info("=== EKSPERYMENT 4: Techniki regularyzacji ===")
    base = [32, 64, 128]
    g4 = "4. Regularyzacja"
    h_base = run(g4, "Brak regularyzacji (CNN Base)", base)
    h_drop = run(g4, "Dropout (p=0.5)", base, dropout_p=0.5)
    h_wd = {
        wd: run(g4, f"AdamW (weight_decay={wd:g})", base, use_adamw=True, wd=wd)
        for wd in (0.01, 0.1, 0.5)
    }
    h_bn = run(g4, "BatchNorm2d", base, use_bn=True)
    h_all = run(g4, "Wszystkie (BN + Dropout + AdamW 0.1)", base, use_bn=True, dropout_p=0.5, use_adamw=True, wd=0.1)

    plot_experiment(
        {
            "Brak regularyzacji (CNN Base)": h_base,
            "Dropout (p=0.5)": h_drop,
            "AdamW (weight_decay=0.1)": h_wd[0.1],
            "BatchNorm2d": h_bn,
            "Wszystkie (BN + Dropout + AdamW 0.1)": h_all,
        },
        "Eksperyment 4 (Regularyzacja)",
        "exp4_regularization",
    )
    plot_experiment(
        {"Adam (bez weight decay)": h_base, **{f"AdamW (wd={wd:g})": h for wd, h in h_wd.items()}},
        "Eksperyment 4b (Weight decay)",
        "exp4b_weight_decay",
    )

    df_summary = pd.DataFrame(all_summaries)
    df_epochs = pd.DataFrame(all_epochs)

    logger.info("Podsumowanie wszystkich eksperymentów:\n" + df_summary.to_string(index=False))

    LAB5_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    excel_path = LAB5_ARTIFACT_DIR / "wyniki_lab5.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Podsumowanie", index=False)
        df_epochs.to_excel(writer, sheet_name="Historia Epok", index=False)

    logger.info(f"Zapisano wyniki do: {excel_path} (całkowity czas: {time.perf_counter() - t_total:.1f}s)")


if __name__ == "__main__":
    main()
