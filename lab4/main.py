import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from loguru import logger

from config import LAB4_ARTIFACT_DIR
from lab3.MLP import MLP as NumpyMLP, Dense, ReLU, Adam as NumpyAdam, SGD as NumpySGD
from lab4.torch_mlp import (
    DEVICE,
    TorchMLP,
    fit_torch_mlp,
    clipped_bce_np,
    collect_metrics,
    load_data,
    save_plot,
)


def log_variant_summary(variant_name: str, rows: list[dict], t_np: float, t_th: float):
    np_row, th_row = rows[0], rows[1]
    logger.info(
        f"[{variant_name}] Czas -> NumPy: {t_np:.2f}s, PyTorch: {t_th:.2f}s | "
        f"Train Loss (NP/TH): {np_row['Train Loss']:.4f} / {th_row['Train Loss']:.4f} | "
        f"Val Loss (NP/TH): {np_row['Val Loss']:.4f} / {th_row['Val Loss']:.4f} | "
        f"Val Acc (NP/TH): {np_row['Val Acc']:.4f} / {th_row['Val Acc']:.4f}"
    )


def check_equivalence_levels(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> pd.DataFrame:
    logger.info("NumPy vs PyTorch")
    t0 = time.perf_counter()
    input_dim = X_train.shape[1]

    np.random.seed(42)
    np_layers = [
        Dense(input_dim, 32, init="he"), ReLU(),
        Dense(32, 16, init="he"), ReLU(),
        Dense(16, 1, init="he"),
    ]
    np_model = NumpyMLP(layers=np_layers, optimizer=NumpyAdam(lr=0.005))

    np.random.seed(42)
    th_model = TorchMLP([input_dim, 32, 16, 1], init_mode="he").to(DEVICE)
    th_opt = torch.optim.Adam(th_model.parameters(), lr=0.005, betas=(0.9, 0.999), eps=1e-8)
    criterion = nn.BCEWithLogitsLoss(reduction="mean")

    # Poziom 1: forward i loss poczatkowy
    y_pred_np = np_model.forward(X_train)
    xt = torch.from_numpy(X_train).to(DEVICE)
    yt = torch.from_numpy(y_train).to(DEVICE)

    logits_t = th_model(xt)
    logger.info(f"Poziom 1 -> grad_fn wyjścia: {logits_t.grad_fn}")
    y_pred_th = torch.sigmoid(torch.clamp(logits_t, -500.0, 500.0)).detach().cpu().numpy()

    fwd_diff = float(np.max(np.abs(y_pred_np - y_pred_th)))
    loss_diff = abs(np_model.loss(y_train, y_pred_np) - clipped_bce_np(y_train, y_pred_th))
    logger.info(f"Poziom 1 -> max różnica forward: {fwd_diff:.3e}, różnica loss: {loss_diff:.3e}")

    # Poziom 2: gradienty po backward
    np_model.backward(y_pred_np, y_train)
    criterion(logits_t, yt).backward()

    np_dense = [l for l in np_layers if isinstance(l, Dense)]
    th_linear = [m for m in th_model.net if isinstance(m, nn.Linear)]

    max_dw_diff = max(
        float(np.max(np.abs(l_np.dW - l_th.weight.grad.detach().cpu().numpy().T)))
        for l_np, l_th in zip(np_dense, th_linear)
    )
    max_db_diff = max(
        float(np.max(np.abs(l_np.db - l_th.bias.grad.detach().cpu().numpy().reshape(1, -1))))
        for l_np, l_th in zip(np_dense, th_linear)
    )
    logger.info(f"Poziom 2 -> max różnica dW: {max_dw_diff:.3e}, max różnica db: {max_db_diff:.3e}")

    # Poziom 3: wagi po kroku optymalizatora
    np_model.step()
    th_opt.step()
    th_opt.zero_grad()

    max_w_step_diff = max(
        float(np.max(np.abs(l_np.W - l_th.weight.detach().cpu().numpy().T)))
        for l_np, l_th in zip(np_dense, th_linear)
    )
    logger.info(f"Poziom 3 -> max różnica wag po kroku Adam: {max_w_step_diff:.3e}")

    # Poziom 4: 50 epok treningu z walidacja
    logger.info("Poziom 4 -> testowy trening 50 epok (batch_size=32)...")
    np.random.seed(123)
    np_tr_l, np_val_l = np_model.fit(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)

    np.random.seed(123)
    th_tr_l, th_val_l = fit_torch_mlp(th_model, X_train, y_train, th_opt, X_val, y_val, epochs=50, batch_size=32)

    tr_curve_diff = float(np.max(np.abs(np.array(np_tr_l) - np.array(th_tr_l))))
    val_curve_diff = float(np.max(np.abs(np.array(np_val_l) - np.array(th_val_l))))
    logger.info(
        f"Poziom 4 -> max różnica krzywej Train: {tr_curve_diff:.3e}, Val: {val_curve_diff:.3e} "
        f"(czas weryfikacji: {time.perf_counter() - t0:.2f}s)"
    )

    levels_data = [
        {"Poziom": "Poziom 1 (Forward)", "Sprawdzany element": "Wyjście modelu y_pred", "Max róznica": f"{fwd_diff:.3e}"},
        {"Poziom": "Poziom 1 (Loss)", "Sprawdzany element": "Wartość początkowa BCE", "Max róznica": f"{loss_diff:.3e}"},
        {"Poziom": "Poziom 2 (Backward dW)", "Sprawdzany element": "Gradienty wag dW vs weight.grad", "Max róznica": f"{max_dw_diff:.3e}"},
        {"Poziom": "Poziom 2 (Backward db)", "Sprawdzany element": "Gradienty biasów db vs bias.grad", "Max róznica": f"{max_db_diff:.3e}"},
        {"Poziom": "Poziom 3 (Krok Adam)", "Sprawdzany element": "Wagi po 1 aktualizacji", "Max róznica": f"{max_w_step_diff:.3e}"},
        {"Poziom": "Poziom 4 (50 epok Train)", "Sprawdzany element": "Krzywa straty treningowej", "Max róznica": f"{tr_curve_diff:.3e}"},
        {"Poziom": "Poziom 4 (50 epok Val)", "Sprawdzany element": "Krzywa straty walidacyjnej", "Max róznica": f"{val_curve_diff:.3e}"},
    ]
    return pd.DataFrame(levels_data)


def run_experiments(X_train: np.ndarray, X_val: np.ndarray, y_train: np.ndarray, y_val: np.ndarray) -> pd.DataFrame:
    input_dim = X_train.shape[1]
    metrics_rows = []
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    logger.info("Start -> Trening i walidacja modelu bazowego (250 epok, batch_size=32)...")
    t0 = time.perf_counter()
    np.random.seed(42)
    np_base = NumpyMLP(
        layers=[Dense(input_dim, 32, init="he"), ReLU(), Dense(32, 1, init="he")],
        optimizer=NumpyAdam(lr=0.002),
    )
    np_tr_base, np_val_base = np_base.fit(X_train, y_train, X_val, y_val, epochs=250, batch_size=32)
    t_np = time.perf_counter() - t0

    t0 = time.perf_counter()
    np.random.seed(42)
    th_base = TorchMLP([input_dim, 32, 1], init_mode="he").to(DEVICE)
    th_opt_base = torch.optim.Adam(th_base.parameters(), lr=0.002)
    th_tr_base, th_val_base = fit_torch_mlp(th_base, X_train, y_train, th_opt_base, X_val, y_val, epochs=250, batch_size=32)
    t_th = time.perf_counter() - t0

    rows = collect_metrics("0. Trening vs Walidacja", "Architektura 32-1 (batch 32)", np_base, th_base, np_tr_base, np_val_base, th_tr_base, th_val_base, X_train, y_train, X_val, y_val)
    metrics_rows.extend(rows)
    log_variant_summary("Model bazowy Train/Val", rows, t_np, t_th)

    plt.figure(figsize=(9, 5), dpi=300)
    plt.plot(np_tr_base, color="tab:blue", linewidth=3, alpha=0.4, label="Train Loss (NumPy)")
    plt.plot(th_tr_base, color="tab:blue", linestyle="--", linewidth=1.5, label="Train Loss (PyTorch)")
    plt.plot(np_val_base, color="tab:orange", linewidth=3, alpha=0.4, label="Val Loss (NumPy)")
    plt.plot(th_val_base, color="tab:orange", linestyle="--", linewidth=1.5, label="Val Loss (PyTorch)")
    plt.xlabel("Epoki")
    plt.ylabel("Loss")
    plt.title("Porównanie uczenia na zbiorze treningowym i walidacyjnym")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    save_plot("exp0_train_val_pytorch")

    logger.info("Start -> Eksperyment 1: Tryby paczkowania (300 epok)...")
    batch_modes = {
        "Full Batch (242)": None,
        "Mini-batch (32)": 32,
        "Stochastyczny (1 - SGD)": 1,
    }
    plt.figure(figsize=(10, 5.5), dpi=300)
    for (name, b_size), col in zip(batch_modes.items(), colors):
        logger.info(f"  [Eksperyment 1] Liczenie wariantu: {name} (batch_size={b_size})...")
        t0 = time.perf_counter()
        np.random.seed(42)
        np_layers = [Dense(input_dim, 32, init="he"), ReLU(), Dense(32, 1, init="he")]
        np_model = NumpyMLP(layers=np_layers, optimizer=NumpyAdam(lr=0.005))
        np_tr, np_val = np_model.fit(X_train, y_train, X_val, y_val, epochs=300, batch_size=b_size)
        t_np = time.perf_counter() - t0

        t0 = time.perf_counter()
        np.random.seed(42)
        th_model = TorchMLP([input_dim, 32, 1], init_mode="he").to(DEVICE)
        th_opt = torch.optim.Adam(th_model.parameters(), lr=0.005)
        th_tr, th_val = fit_torch_mlp(th_model, X_train, y_train, th_opt, X_val, y_val, epochs=300, batch_size=b_size)
        t_th = time.perf_counter() - t0

        rows = collect_metrics("1. Paczkowanie", name, np_model, th_model, np_tr, np_val, th_tr, th_val, X_train, y_train, X_val, y_val)
        metrics_rows.extend(rows)
        log_variant_summary(f"Exp 1 - {name}", rows, t_np, t_th)

        plt.plot(np_tr, color=col, linestyle="-", alpha=0.45, linewidth=3, label=f"{name} [NumPy]")
        plt.plot(th_tr, color=col, linestyle="--", linewidth=1.5, label=f"{name} [PyTorch]")

    plt.xlabel("Epoki")
    plt.ylabel("Loss")
    plt.title("Eksperyment 1: Tryby paczkowania")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(ncol=2, fontsize=8.5)
    plt.tight_layout()
    save_plot("exp_batching_pytorch")
    logger.info("Zapisano wykres: exp_batching_pytorch")

    logger.info("Start -> Eksperyment 2: Inicjalizacje wag (300 epok, batch_size=32)...")
    init_types = {
        "Inicjalizacja He": "he",
        "Rozkład normalny N(0, 1)": "normal_1",
        "Zepsuta inicjalizacja N(0, 5)": "broken",
    }
    plt.figure(figsize=(10, 5.5), dpi=300)
    for (name, init_mode), col in zip(init_types.items(), colors):
        logger.info(f"  [Eksperyment 2] Liczenie wariantu: {name} (init={init_mode})...")
        t0 = time.perf_counter()
        np.random.seed(42)
        np_layers = [
            Dense(input_dim, 32, init=init_mode), ReLU(),
            Dense(32, 32, init=init_mode), ReLU(),
            Dense(32, 32, init=init_mode), ReLU(),
            Dense(32, 1, init=init_mode),
        ]
        np_model = NumpyMLP(layers=np_layers, optimizer=NumpyAdam(lr=0.002))
        np_tr, np_val = np_model.fit(X_train, y_train, X_val, y_val, epochs=300, batch_size=32)
        t_np = time.perf_counter() - t0

        t0 = time.perf_counter()
        np.random.seed(42)
        th_model = TorchMLP([input_dim, 32, 32, 32, 1], init_mode=init_mode).to(DEVICE)
        th_opt = torch.optim.Adam(th_model.parameters(), lr=0.002)
        th_tr, th_val = fit_torch_mlp(th_model, X_train, y_train, th_opt, X_val, y_val, epochs=300, batch_size=32)
        t_th = time.perf_counter() - t0

        rows = collect_metrics("2. Inicjalizacja", name, np_model, th_model, np_tr, np_val, th_tr, th_val, X_train, y_train, X_val, y_val)
        metrics_rows.extend(rows)
        log_variant_summary(f"Exp 2 - {name}", rows, t_np, t_th)

        plt.plot(np_tr, color=col, linestyle="-", alpha=0.45, linewidth=3, label=f"{name} [NumPy]")
        plt.plot(th_tr, color=col, linestyle="--", linewidth=1.5, label=f"{name} [PyTorch]")

    plt.xlabel("Epoki")
    plt.ylabel("Loss")
    plt.title("Eksperyment 2: Inicjalizacja wag")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(ncol=2, fontsize=8.5)
    plt.tight_layout()
    save_plot("exp_initialization_pytorch")
    logger.info("Zapisano wykres: exp_initialization_pytorch")

    logger.info("Start -> Eksperyment 3: SGD vs ADAM (350 epok, batch_size=32)...")
    opt_configs = [
        ("ADAM (lr=0.002)", "adam", 0.002),
        ("Gradient (lr=0.002)", "sgd", 0.002),
        ("Gradient (lr=0.01)", "sgd", 0.01),
        ("Gradient (lr=0.05)", "sgd", 0.05),
    ]
    plt.figure(figsize=(10, 5.5), dpi=300)
    for (name, opt_type, lr_val), col in zip(opt_configs, colors):
        logger.info(f"  [Eksperyment 3] Liczenie wariantu: {name} (opt={opt_type}, lr={lr_val})...")
        t0 = time.perf_counter()
        np.random.seed(42)
        np_layers = [
            Dense(input_dim, 32, init="he"), ReLU(),
            Dense(32, 32, init="he"), ReLU(),
            Dense(32, 1, init="he"),
        ]
        np_opt = NumpyAdam(lr=lr_val) if opt_type == "adam" else NumpySGD(lr=lr_val)
        np_model = NumpyMLP(layers=np_layers, optimizer=np_opt)
        np_tr, np_val = np_model.fit(X_train, y_train, X_val, y_val, epochs=350, batch_size=32)
        t_np = time.perf_counter() - t0

        t0 = time.perf_counter()
        np.random.seed(42)
        th_model = TorchMLP([input_dim, 32, 32, 1], init_mode="he").to(DEVICE)
        th_opt = (
            torch.optim.Adam(th_model.parameters(), lr=lr_val)
            if opt_type == "adam"
            else torch.optim.SGD(th_model.parameters(), lr=lr_val)
        )
        th_tr, th_val = fit_torch_mlp(th_model, X_train, y_train, th_opt, X_val, y_val, epochs=350, batch_size=32)
        t_th = time.perf_counter() - t0

        rows = collect_metrics("3. Optymalizatory", name, np_model, th_model, np_tr, np_val, th_tr, th_val, X_train, y_train, X_val, y_val)
        metrics_rows.extend(rows)
        log_variant_summary(f"Exp 3 - {name}", rows, t_np, t_th)

        plt.plot(np_tr, color=col, linestyle="-", alpha=0.45, linewidth=3, label=f"{name} [NumPy]")
        plt.plot(th_tr, color=col, linestyle="--", linewidth=1.5, label=f"{name} [PyTorch]")

    plt.xlabel("Epoki")
    plt.ylabel("Loss")
    plt.title("Eksperyment 3: SGD vs ADAM")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(ncol=2, fontsize=8.5)
    plt.tight_layout()
    save_plot("exp_optimizers_pytorch")
    logger.info("Zapisano wykres: exp_optimizers_pytorch")

    return pd.DataFrame(metrics_rows)


def main():
    total_start = time.perf_counter()
    logger.info(f"Używane urządzenie: {DEVICE}")
    X_train, X_val, y_train, y_val = load_data()
    logger.info(f"Wczytano dane: X_train={X_train.shape}, X_val={X_val.shape}")

    df_levels = check_equivalence_levels(X_train, y_train, X_val, y_val)
    logger.info("Podsumowanie:\n" + df_levels.to_string(index=False))

    df_metrics = run_experiments(X_train, X_val, y_train, y_val)
    logger.info("Tabela metryk (Train / Val):\n" + df_metrics.to_string(index=False))

    LAB4_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    excel_path = LAB4_ARTIFACT_DIR / "wyniki.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_metrics.to_excel(writer, sheet_name="Metryki Train Val", index=False)
        df_levels.to_excel(writer, sheet_name="Poziomy Zgodnosci", index=False)

    logger.info(f"Zapisano wyniki do: {excel_path} (całkowity czas: {time.perf_counter() - total_start:.2f}s)")


if __name__ == "__main__":
    main()