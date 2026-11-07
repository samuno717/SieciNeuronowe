import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from config import LAB4_ARTIFACT_DIR
from lab3.MLP import MLP as NumpyMLP, Dense, ReLU, Adam as NumpyAdam
from lab4.torch_mlp import DEVICE, TorchMLP, clipped_bce_np, load_data, save_plot


def step1_precision_and_transpose(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    print("krok 1...")
    input_dim = X_train.shape[1]
    dims = [input_dim, 32, 16, 1]

    np.random.seed(42)
    np_layers = [Dense(input_dim, 32, init="he"), ReLU(), Dense(32, 16, init="he"), ReLU(), Dense(16, 1, init="he")]
    np_model = NumpyMLP(layers=np_layers, optimizer=NumpyAdam(lr=0.005))
    y_np = np_model.forward(X_train)
    np_model.backward(y_np, y_train)

    np.random.seed(42)
    th_no_T = TorchMLP(dims, dtype=torch.float64, transpose_weights=False).to(DEVICE)
    y_no_T = th_no_T.predict_proba(torch.tensor(X_train, dtype=torch.float64, device=DEVICE)).detach().cpu().numpy()
    err_fwd_no_T = float(np.max(np.abs(y_np - y_no_T)))

    np.random.seed(42)
    th_f32 = TorchMLP(dims, dtype=torch.float32, transpose_weights=True).to(DEVICE)
    x_f32 = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_t_f32 = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    logits_f32 = th_f32(x_f32)
    y_f32 = torch.sigmoid(torch.clamp(logits_f32, -500.0, 500.0)).detach().cpu().numpy()
    nn.BCEWithLogitsLoss()(logits_f32, y_t_f32).backward()
    err_fwd_f32 = float(np.max(np.abs(y_np - y_f32)))
    err_grad_f32 = float(np.max(np.abs(np_layers[0].dW - th_f32.net[0].weight.grad.detach().cpu().numpy().T)))

    np.random.seed(42)
    th_f64 = TorchMLP(dims, dtype=torch.float64, transpose_weights=True).to(DEVICE)
    x_f64 = torch.tensor(X_train, dtype=torch.float64, device=DEVICE)
    y_t_f64 = torch.tensor(y_train, dtype=torch.float64, device=DEVICE)
    logits_f64 = th_f64(x_f64)
    y_f64 = torch.sigmoid(torch.clamp(logits_f64, -500.0, 500.0)).detach().cpu().numpy()
    nn.BCEWithLogitsLoss()(logits_f64, y_t_f64).backward()
    err_fwd_f64 = float(np.max(np.abs(y_np - y_f64)))
    err_grad_f64 = float(np.max(np.abs(np_layers[0].dW - th_f64.net[0].weight.grad.detach().cpu().numpy().T)))

    epochs = 200
    np.random.seed(42)
    np_m = NumpyMLP([Dense(input_dim, 32, init="he"), ReLU(), Dense(32, 1, init="he")], optimizer=NumpyAdam(lr=0.005))
    loss_np, _ = np_m.fit(X_train, y_train, epochs=epochs, batch_size=None, shuffle=False)

    losses_by_dtype = {}
    for dt_name, dt in [("float32", torch.float32), ("float64", torch.float64)]:
        np.random.seed(42)
        m = TorchMLP([input_dim, 32, 1], dtype=dt, transpose_weights=True).to(DEVICE)
        opt = torch.optim.Adam(m.parameters(), lr=0.005)
        crit = nn.BCEWithLogitsLoss()
        xt = torch.tensor(X_train, dtype=dt, device=DEVICE)
        yt = torch.tensor(y_train, dtype=dt, device=DEVICE)
        l_hist = []
        for _ in range(epochs):
            loss = crit(m(xt), yt)
            loss.backward()
            opt.step()
            opt.zero_grad()
            with torch.no_grad():
                p = m.predict_proba(xt).cpu().numpy()
                l_hist.append(clipped_bce_np(y_train, p))
        losses_by_dtype[dt_name] = np.abs(np.array(loss_np) - np.array(l_hist)) + 1e-17

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)
    labels = ["Brak transpozycji W", "Transpozycja + float32", "Transpozycja + float64"]
    fwd_errs = [err_fwd_no_T, err_fwd_f32, err_fwd_f64]
    bars = ax1.bar(labels, fwd_errs, color=["tab:red", "tab:orange", "tab:green"], alpha=0.85, width=0.55)
    ax1.set_yscale("log")
    ax1.set_ylabel("Maksymalna różnica |y_np - y_torch|")
    ax1.set_title("Wpływ transpozycji wag i typu danych na forward")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)
    for bar, val in zip(bars, fwd_errs):
        ax1.text(bar.get_x() + bar.get_width() / 2, val * 1.8, f"{val:.2e}", ha="center", va="bottom", fontsize=9)
    ax1.set_ylim(1e-18, 5.0)

    ax2.plot(losses_by_dtype["float32"], color="tab:orange", label="PyTorch float32 vs NumPy", linewidth=1.8)
    ax2.plot(losses_by_dtype["float64"], color="tab:green", label="PyTorch float64 vs NumPy", linewidth=1.8)
    ax2.set_yscale("log")
    ax2.set_xlabel("Epoka")
    ax2.set_ylabel("Różnica wartości straty |Loss_np - Loss_torch|")
    ax2.set_title("Kumulacja błędu zaokrągleń w trakcie 200 epok")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    save_plot("proof_step1_precision_transpose")

    return {
        "Krok": "Krok 1",
        "Problem": "Transpozycja wag w nn.Linear oraz precyzja float32 vs float64",
        "Przed poprawka": f"Brak transpozycji: błąd forward = {err_fwd_no_T:.2e}; float32: błąd grad = {err_grad_f32:.2e}",
        "Po poprawce": f"W.T + float64: błąd forward = {err_fwd_f64:.2e}, błąd grad = {err_grad_f64:.2e}",
        "Wykres": "proof_step1_precision_transpose.svg",
    }


def step2_initialization_rng(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    print("krok 2...")
    input_dim = X_train.shape[1]
    dims = [input_dim, 32, 1]
    epochs = 200

    np.random.seed(42)
    np_m = NumpyMLP([Dense(input_dim, 32, init="he"), ReLU(), Dense(32, 1, init="he")], optimizer=NumpyAdam(lr=0.005))
    w_init_np = np_m.layers[0].W.copy()
    loss_np, _ = np_m.fit(X_train, y_train, epochs=epochs, batch_size=None, shuffle=False)

    torch.manual_seed(42)
    m_uni = TorchMLP(dims, init_strategy="default_uniform").to(DEVICE)
    w_init_uni = m_uni.net[0].weight.detach().cpu().numpy().T.copy()

    torch.manual_seed(42)
    m_nat = TorchMLP(dims, init_mode="he", init_strategy="native_normal").to(DEVICE)

    np.random.seed(42)
    m_sync = TorchMLP(dims, init_mode="he", init_strategy="sync_numpy").to(DEVICE)
    w_init_sync = m_sync.net[0].weight.detach().cpu().numpy().T.copy()

    err_w_uni = float(np.max(np.abs(w_init_np - w_init_uni)))
    err_w_sync = float(np.max(np.abs(w_init_np - w_init_sync)))

    curves = {}
    for name, model in [("Domyślny nn.Linear (Uniform)", m_uni), ("PyTorch kaiming_normal (własny seed)", m_nat), ("Wagi zsynchronizowane z NumPy", m_sync)]:
        opt = torch.optim.Adam(model.parameters(), lr=0.005)
        crit = nn.BCEWithLogitsLoss()
        xt = torch.tensor(X_train, dtype=torch.float64, device=DEVICE)
        yt = torch.tensor(y_train, dtype=torch.float64, device=DEVICE)
        l_hist = []
        for _ in range(epochs):
            loss = crit(model(xt), yt)
            loss.backward()
            opt.step()
            opt.zero_grad()
            with torch.no_grad():
                l_hist.append(clipped_bce_np(y_train, model.predict_proba(xt).cpu().numpy()))
        curves[name] = l_hist

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)
    ax1.hist(w_init_np.flatten(), bins=30, alpha=0.55, density=True, label=f"NumPy He Normal (std={w_init_np.std():.2f})", color="tab:blue")
    ax1.hist(w_init_uni.flatten(), bins=30, alpha=0.55, density=True, label=f"Domyślny PyTorch Uniform (std={w_init_uni.std():.2f})", color="tab:red")
    ax1.set_xlabel("Wartość wag początkowych")
    ax1.set_ylabel("Gęstość")
    ax1.set_title("Rozkład wag w warstwie wejściowej")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=8.5)

    ax2.plot(loss_np, color="tab:blue", linewidth=3.5, alpha=0.4, label="NumPy MLP")
    ax2.plot(curves["Domyślny nn.Linear (Uniform)"], color="tab:red", linestyle="-.", label="PyTorch: domyślny Uniform")
    ax2.plot(curves["PyTorch kaiming_normal (własny seed)"], color="tab:orange", linestyle=":", linewidth=2, label="PyTorch: He Normal (inny RNG)")
    ax2.plot(curves["Wagi zsynchronizowane z NumPy"], color="tab:blue", linestyle="--", linewidth=1.5, label="PyTorch: wagi skopiowane z NumPy")
    ax2.set_xlabel("Epoka")
    ax2.set_ylabel("Loss")
    ax2.set_title("Krzywa uczenia w zależności od inicjalizacji wag")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=8.5)

    plt.tight_layout()
    save_plot("proof_step2_initialization_rng")

    max_diff_sync = float(np.max(np.abs(np.array(loss_np) - np.array(curves["Wagi zsynchronizowane z NumPy"]))))
    max_diff_nat = float(np.max(np.abs(np.array(loss_np) - np.array(curves["PyTorch kaiming_normal (własny seed)"]))))
    return {
        "Krok": "Krok 2",
        "Problem": "Domyślny rozkład Uniform w nn.Linear oraz inny generator liczb losowych",
        "Przed poprawka": f"Domyślny Uniform: max roznica W0 = {err_w_uni:.2f}; Wlasny RNG: max roznica Loss = {max_diff_nat:.3f}",
        "Po poprawce": f"Synchronizacja wag z np.random.normal: max roznica W0 = {err_w_sync:.1e}, max roznica Loss = {max_diff_sync:.2e}",
        "Wykres": "proof_step2_initialization_rng.svg",
    }


def step3_bce_saturation(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    print("krok 3...")
    input_dim = X_train.shape[1]
    dims = [input_dim, 32, 32, 32, 1]
    epochs = 300

    np.random.seed(42)
    np_layers = [
        Dense(input_dim, 32, init="broken"), ReLU(),
        Dense(32, 32, init="broken"), ReLU(),
        Dense(32, 32, init="broken"), ReLU(),
        Dense(32, 1, init="broken"),
    ]
    np_m = NumpyMLP(np_layers, optimizer=NumpyAdam(lr=0.002))
    y_pred_0 = np_m.forward(X_train)
    np_m.backward(y_pred_0, y_train)
    dw0_np = np_layers[0].dW.copy()

    np.random.seed(42)
    np_layers_run = [
        Dense(input_dim, 32, init="broken"), ReLU(),
        Dense(32, 32, init="broken"), ReLU(),
        Dense(32, 32, init="broken"), ReLU(),
        Dense(32, 1, init="broken"),
    ]
    np_m_run = NumpyMLP(np_layers_run, optimizer=NumpyAdam(lr=0.002))
    loss_np, _ = np_m_run.fit(X_train, y_train, epochs=epochs, batch_size=32)

    xt = torch.tensor(X_train, dtype=torch.float64, device=DEVICE)
    yt = torch.tensor(y_train, dtype=torch.float64, device=DEVICE)

    np.random.seed(42)
    m_naive = TorchMLP(dims, init_mode="broken").to(DEVICE)
    logits_naive = m_naive(xt)
    probs_naive = torch.sigmoid(torch.clamp(logits_naive, -500.0, 500.0))
    probs_naive.retain_grad()
    logits_naive.retain_grad()
    nn.BCELoss()(probs_naive, yt).backward()
    dw0_naive = m_naive.net[0].weight.grad.detach().cpu().numpy().T
    err_grad_naive = float(np.max(np.abs(dw0_np - dw0_naive)))
    zero_grad_samples_naive = int(torch.sum(torch.abs(logits_naive.grad) < 1e-12).item())

    np.random.seed(42)
    m_logits = TorchMLP(dims, init_mode="broken").to(DEVICE)
    logits_good = m_logits(xt)
    logits_good.retain_grad()
    nn.BCEWithLogitsLoss()(logits_good, yt).backward()
    dw0_good = m_logits.net[0].weight.grad.detach().cpu().numpy().T
    err_grad_good = float(np.max(np.abs(dw0_np - dw0_good)))
    zero_grad_samples_good = int(torch.sum(torch.abs(logits_good.grad) < 1e-12).item())

    losses_torch = {}
    for mode_name, use_logits_loss in [("Sigmoid + nn.BCELoss", False), ("nn.BCEWithLogitsLoss", True)]:
        np.random.seed(42)
        m = TorchMLP(dims, init_mode="broken").to(DEVICE)
        opt = torch.optim.Adam(m.parameters(), lr=0.002)
        crit = nn.BCEWithLogitsLoss() if use_logits_loss else nn.BCELoss()
        n_samples = X_train.shape[0]
        l_hist = []
        for _ in range(epochs):
            indices = np.random.permutation(n_samples)
            for start in range(0, n_samples, 32):
                idx = indices[start:start + 32]
                xb, yb = xt[idx], yt[idx]
                out = m(xb)
                loss = crit(out, yb) if use_logits_loss else crit(torch.sigmoid(torch.clamp(out, -500.0, 500.0)), yb)
                loss.backward()
                opt.step()
                opt.zero_grad()
            with torch.no_grad():
                l_hist.append(clipped_bce_np(y_train, m.predict_proba(xt).cpu().numpy()))
        losses_torch[mode_name] = l_hist

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)
    ax1.plot(loss_np, color="tab:green", linewidth=3.5, alpha=0.4, label="NumPy MLP")
    ax1.plot(losses_torch["Sigmoid + nn.BCELoss"], color="tab:red", linestyle="-.", linewidth=1.8, label="PyTorch: Sigmoid + BCELoss")
    ax1.plot(losses_torch["nn.BCEWithLogitsLoss"], color="tab:green", linestyle="--", linewidth=1.5, label="PyTorch: BCEWithLogitsLoss")
    ax1.set_xlabel("Epoka")
    ax1.set_ylabel("Loss")
    ax1.set_title("Przebieg treningu dla zepsutej inicjalizacji N(0, 5)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=8.5)

    n_total = X_train.shape[0]
    bars = ax2.bar(
        ["NumPy MLP", "PyTorch: Sigmoid + BCELoss", "PyTorch: BCEWithLogitsLoss"],
        [0, zero_grad_samples_naive, zero_grad_samples_good],
        color=["tab:blue", "tab:red", "tab:green"],
        width=0.5,
        alpha=0.85,
    )
    ax2.set_ylabel(f"Liczba próbek z zerowym gradientem (na {n_total})")
    ax2.set_title("Liczba próbek z wyzerowanym gradientem")
    ax2.set_ylim(0, n_total * 1.15)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        h = int(bar.get_height())
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 5, f"{h} / {n_total} ({100*h/n_total:.1f}%)", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    save_plot("proof_step3_bce_saturation")

    return {
        "Krok": "Krok 3",
        "Problem": "Zerowanie gradientu przy nasyceniu dla Sigmoid + nn.BCELoss i N(0, 5)",
        "Przed poprawka": f"Sigmoid + BCELoss: {zero_grad_samples_naive}/{n_total} probek ma grad=0, blad dW = {err_grad_naive:.2e}",
        "Po poprawce": f"BCEWithLogitsLoss: {zero_grad_samples_good}/{n_total} zamrozonych probek, blad dW = {err_grad_good:.2e}",
        "Wykres": "proof_step3_bce_saturation.svg",
    }


def step4_zero_grad(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    print("krok 4...")
    input_dim = X_train.shape[1]
    dims = [input_dim, 32, 1]
    epochs = 100

    np.random.seed(42)
    np_m = NumpyMLP([Dense(input_dim, 32, init="he"), ReLU(), Dense(32, 1, init="he")], optimizer=NumpyAdam(lr=0.005))
    loss_np, _ = np_m.fit(X_train, y_train, epochs=epochs, batch_size=32)

    xt = torch.tensor(X_train, dtype=torch.float64, device=DEVICE)
    yt = torch.tensor(y_train, dtype=torch.float64, device=DEVICE)
    n_samples = X_train.shape[0]

    results_loss = {}
    results_grad_norm = {}

    for label, do_zero_grad in [("Brak zero_grad()", False), ("Z zero_grad()", True)]:
        np.random.seed(42)
        m = TorchMLP(dims, init_mode="he").to(DEVICE)
        opt = torch.optim.Adam(m.parameters(), lr=0.005)
        crit = nn.BCEWithLogitsLoss()
        l_hist, g_norms = [], []

        for ep in range(epochs):
            indices = np.random.permutation(n_samples)
            for start in range(0, n_samples, 32):
                idx = indices[start:start + 32]
                loss = crit(m(xt[idx]), yt[idx])
                loss.backward()
                if ep < 5:
                    g_norms.append(float(torch.norm(m.net[0].weight.grad).item()))
                opt.step()
                if do_zero_grad:
                    opt.zero_grad()
            with torch.no_grad():
                l_hist.append(clipped_bce_np(y_train, m.predict_proba(xt).cpu().numpy()))

        results_loss[label] = l_hist
        results_grad_norm[label] = g_norms

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)
    ax1.plot(loss_np, color="tab:blue", linewidth=3.5, alpha=0.4, label="NumPy MLP")
    ax1.plot(results_loss["Brak zero_grad()"], color="tab:red", linestyle="-.", linewidth=1.6, label="PyTorch: bez zero_grad()")
    ax1.plot(results_loss["Z zero_grad()"], color="tab:blue", linestyle="--", linewidth=1.5, label="PyTorch: z zero_grad()")
    ax1.set_xlabel("Epoka")
    ax1.set_ylabel("Loss")
    ax1.set_title("Wpływ braku zero_grad() na funkcję straty")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=8.5)

    ax2.plot(results_grad_norm["Brak zero_grad()"], color="tab:red", marker="o", markersize=3, label="Bez zero_grad()")
    ax2.plot(results_grad_norm["Z zero_grad()"], color="tab:green", marker="s", markersize=3, label="Z zero_grad()")
    ax2.set_xlabel("Krok aktualizacji (pierwsze 5 epok)")
    ax2.set_ylabel("Norma L2 gradientu wagi pierwszej warstwy")
    ax2.set_title("Akumulacja gradientów w tensorze .grad")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=8.5)

    plt.tight_layout()
    save_plot("proof_step4_zero_grad")

    err_no_zg = float(np.max(np.abs(np.array(loss_np) - np.array(results_loss["Brak zero_grad()"]))))
    err_with_zg = float(np.max(np.abs(np.array(loss_np) - np.array(results_loss["Z zero_grad()"]))))
    return {
        "Krok": "Krok 4",
        "Problem": "Akumulacja gradientow w atrybucie .grad przy braku optimizer.zero_grad()",
        "Przed poprawka": f"Brak zero_grad(): norma gradientu rosnie do {max(results_grad_norm['Brak zero_grad()']):.2f}, blad Loss = {err_no_zg:.3f}",
        "Po poprawce": f"Z optimizer.zero_grad(): blad Loss po 100 epokach = {err_with_zg:.2e}",
        "Wykres": "proof_step4_zero_grad.svg",
    }


def step5_batch_shuffling(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    print("krok 5...")
    input_dim = X_train.shape[1]
    dims = [input_dim, 32, 1]
    epochs = 150

    np.random.seed(42)
    np_m = NumpyMLP([Dense(input_dim, 32, init="he"), ReLU(), Dense(32, 1, init="he")], optimizer=NumpyAdam(lr=0.005))

    xt = torch.tensor(X_train, dtype=torch.float64, device=DEVICE)
    yt = torch.tensor(y_train, dtype=torch.float64, device=DEVICE)
    n_samples = X_train.shape[0]

    np.random.seed(42)
    m_dl = TorchMLP(dims, init_mode="he").to(DEVICE)
    opt_dl = torch.optim.Adam(m_dl.parameters(), lr=0.005)
    torch.manual_seed(42)
    loader = DataLoader(TensorDataset(xt, yt), batch_size=32, shuffle=True)

    np.random.seed(42)
    m_sync = TorchMLP(dims, init_mode="he").to(DEVICE)
    opt_sync = torch.optim.Adam(m_sync.parameters(), lr=0.005)

    crit = nn.BCEWithLogitsLoss()
    loss_np_hist, loss_dl_hist, loss_sync_hist = [], [], []
    w_dist_dl, w_dist_sync = [], []

    for ep in range(epochs):
        np.random.seed(1000 + ep)
        perm = np.random.permutation(n_samples)

        for start in range(0, n_samples, 32):
            idx = perm[start:start + 32]
            yb_pred = np_m.forward(X_train[idx])
            np_m.backward(yb_pred, y_train[idx])
            np_m.step()
        loss_np_hist.append(np_m.loss(y_train, np_m.forward(X_train)))

        for start in range(0, n_samples, 32):
            idx = perm[start:start + 32]
            loss = crit(m_sync(xt[idx]), yt[idx])
            loss.backward()
            opt_sync.step()
            opt_sync.zero_grad()
        with torch.no_grad():
            loss_sync_hist.append(clipped_bce_np(y_train, m_sync.predict_proba(xt).cpu().numpy()))

        for xb, yb in loader:
            loss = crit(m_dl(xb), yb)
            loss.backward()
            opt_dl.step()
            opt_dl.zero_grad()
        with torch.no_grad():
            loss_dl_hist.append(clipped_bce_np(y_train, m_dl.predict_proba(xt).cpu().numpy()))

        w_np_curr = np_m.layers[0].W
        w_dl_curr = m_dl.net[0].weight.detach().cpu().numpy().T
        w_sync_curr = m_sync.net[0].weight.detach().cpu().numpy().T
        w_dist_dl.append(float(np.linalg.norm(w_np_curr - w_dl_curr)) + 1e-17)
        w_dist_sync.append(float(np.linalg.norm(w_np_curr - w_sync_curr)) + 1e-17)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)
    ax1.plot(loss_np_hist, color="tab:orange", linewidth=3.5, alpha=0.4, label="NumPy MLP")
    ax1.plot(loss_dl_hist, color="tab:purple", linestyle="-.", linewidth=1.5, label="PyTorch: DataLoader(shuffle=True)")
    ax1.plot(loss_sync_hist, color="tab:orange", linestyle="--", linewidth=1.5, label="PyTorch: wspólna permutacja")
    ax1.set_xlabel("Epoka")
    ax1.set_ylabel("Loss")
    ax1.set_title("Krzywe uczenia przy różnym tasowaniu batchy")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=8.5)

    ax2.plot(w_dist_dl, color="tab:purple", linewidth=1.8, label="DataLoader vs NumPy")
    ax2.plot(w_dist_sync, color="tab:green", linewidth=1.8, label="Wspólna permutacja vs NumPy")
    ax2.set_yscale("log")
    ax2.set_xlabel("Epoka")
    ax2.set_ylabel("Odległość macierzy wag ||W_np - W_torch^T||")
    ax2.set_title("Odległość euklidesowa wag w trakcie uczenia")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=8.5)

    plt.tight_layout()
    save_plot("proof_step5_shuffling")

    return {
        "Krok": "Krok 5",
        "Problem": "Inna kolejnosc mini-batchy z DataLoader(shuffle=True) niz z np.random.permutation",
        "Przed poprawka": f"DataLoader(shuffle=True): odleglosc wag rosnie do {w_dist_dl[-1]:.2f}",
        "Po poprawce": f"Wspolne indeksowanie np.random.permutation: odleglosc wag po 150 epokach = {w_dist_sync[-1]:.2e}",
        "Wykres": "proof_step5_shuffling.svg",
    }


def step6_loss_reporting(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    print("krok 6...")
    input_dim = X_train.shape[1]
    dims = [input_dim, 32, 1]
    epochs = 120

    np.random.seed(42)
    np_m = NumpyMLP([Dense(input_dim, 32, init="he"), ReLU(), Dense(32, 1, init="he")], optimizer=NumpyAdam(lr=0.005))
    loss_np, _ = np_m.fit(X_train, y_train, epochs=epochs, batch_size=1)

    np.random.seed(42)
    m = TorchMLP(dims, init_mode="he").to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=0.005)
    crit = nn.BCEWithLogitsLoss()
    xt = torch.tensor(X_train, dtype=torch.float64, device=DEVICE)
    yt = torch.tensor(y_train, dtype=torch.float64, device=DEVICE)
    n_samples = X_train.shape[0]

    running_loss_hist = []
    post_epoch_loss_hist = []

    for _ in range(epochs):
        indices = np.random.permutation(n_samples)
        batch_losses = []
        for start in range(0, n_samples, 1):
            idx = indices[start:start + 1]
            logits = m(xt[idx])
            loss = crit(logits, yt[idx])
            batch_losses.append(float(loss.item()))
            loss.backward()
            opt.step()
            opt.zero_grad()

        running_loss_hist.append(float(np.mean(batch_losses)))

        with torch.no_grad():
            post_epoch_loss_hist.append(clipped_bce_np(y_train, m.predict_proba(xt).cpu().numpy()))

    diff_running = np.abs(np.array(loss_np) - np.array(running_loss_hist))
    diff_post = np.abs(np.array(loss_np) - np.array(post_epoch_loss_hist))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)
    ax1.plot(loss_np, color="tab:green", linewidth=3.5, alpha=0.4, label="NumPy: ewaluacja całego X")
    ax1.plot(running_loss_hist, color="tab:red", linestyle="-.", linewidth=1.4, label="PyTorch: średnia z batchy w locie")
    ax1.plot(post_epoch_loss_hist, color="tab:green", linestyle="--", linewidth=1.5, label="PyTorch: ewaluacja całego X")
    ax1.set_xlabel("Epoka (batch_size=1)")
    ax1.set_ylabel("Loss")
    ax1.set_title("Średnia strata z batchy a pełna ewaluacja po epoce")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=8.5)

    ax2.plot(diff_running + 1e-17, color="tab:red", label="Średnia z batchy vs NumPy")
    ax2.plot(diff_post + 1e-17, color="tab:green", label="Pełna ewaluacja po epoce vs NumPy")
    ax2.set_yscale("log")
    ax2.set_xlabel("Epoka")
    ax2.set_ylabel("Różnica straty względem NumPy")
    ax2.set_title("Błąd wyliczenia straty (w locie vs po epoce)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=8.5)

    plt.tight_layout()
    save_plot("proof_step6_loss_reporting")

    return {
        "Krok": "Krok 6",
        "Problem": "Liczenie straty jako sredniej z batchy w trakcie epoki zamiast po epoce na pelnym zbiorze",
        "Przed poprawka": f"Srednia z batchy w trakcie epoki: max roznica Loss = {np.max(diff_running):.3f}",
        "Po poprawce": f"Ewaluacja calego zbioru X_train po epoce: max roznica Loss = {np.max(diff_post):.2e}",
        "Wykres": "proof_step6_loss_reporting.svg",
    }


def main():
    X_train, _, y_train, _ = load_data()

    rows = [
        step1_precision_and_transpose(X_train, y_train),
        step2_initialization_rng(X_train, y_train),
        step3_bce_saturation(X_train, y_train),
        step4_zero_grad(X_train, y_train),
        step5_batch_shuffling(X_train, y_train),
        step6_loss_reporting(X_train, y_train),
    ]

    LAB4_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    excel_path = LAB4_ARTIFACT_DIR / "wyniki_porownanie.xlsx"
    df.to_excel(excel_path, index=False)
    print("zapisano do:", excel_path)


if __name__ == "__main__":
    main()