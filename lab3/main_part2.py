import shutil

from config import PROJECT_ROOT

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from lab3.MLP import MLP, Dense, ReLU, Adam, SGD
from lab1.preprocessing import get_raw_data, process_features

ARTIFACT_DIR = PROJECT_ROOT / 'lab3' / 'artifacts'


def save_plot(filename_stem: str):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in [".svg", ".png"]:
        local_path = PROJECT_ROOT / "lab3" / f"{filename_stem}{ext}"
        plt.savefig(local_path, bbox_inches="tight", dpi=300)
        art_path = ARTIFACT_DIR / f"{filename_stem}{ext}"
        shutil.copy2(local_path, art_path)
    plt.close()


def main():
    X, y = get_raw_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_test = process_features(X_train, X_test)

    input_dim = X_train.shape[1]
    summary = []

    # EKSPERYMENT 1: Tryb optymalizacji
    print("Eksperyment 1: Tryby paczkowania...")
    batch_modes = {
        "Full Batch (242)": None,
        "Mini-batch (32)": 32,
        "Stochastyczny (1 - SGD)": 1,
    }
    epochs_batch = 300

    plt.figure(figsize=(9, 5), dpi=300)
    for name, b_size in batch_modes.items():
        np.random.seed(42)
        layers = [
            Dense(input_dim, 32, init="he"),
            ReLU(),
            Dense(32, 1, init="he")
        ]
        model = MLP(layers=layers, optimizer=Adam(lr=0.005))
        train_l, _ = model.fit(X_train, y_train, epochs=epochs_batch, batch_size=b_size)
        plt.plot(train_l, label=name)

    plt.xlabel("Epoki")
    plt.ylabel("Wartość funkcji straty")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    save_plot("exp_batching")

    # EKSPERYMENT 2: Inicjalizacja wag (He vs N(0, 1) vs Intencjonalnie zepsuta)
    print("Eksperyment 2: Inicjalizacje wag")
    init_types = {
        "Inicjalizacja He": "he",
        "Rozkład normalny N(0, 1)": "normal_1",
        "Zepsuta inicjalizacja N(0, 5)": "broken",
    }
    epochs_init = 300

    plt.figure(figsize=(9, 5), dpi=300)
    for name, init_mode in init_types.items():
        np.random.seed(42)
        layers = [
            Dense(input_dim, 32, init=init_mode),
            ReLU(),
            Dense(32, 32, init=init_mode),
            ReLU(),
            Dense(32, 32, init=init_mode),
            ReLU(),
            Dense(32, 1, init=init_mode)
        ]
        model = MLP(layers=layers, optimizer=Adam(lr=0.002))
        train_l, _ = model.fit(X_train, y_train, epochs=epochs_init, batch_size=32)
        plt.plot(train_l, label=name)

    plt.xlabel("Epoki")
    plt.ylabel("Wartość funkcji straty")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    save_plot("exp_initialization")

    # EKSPERYMENT 3: Prosty gradient (SGD) vs ADAM
    print("Eksperyment 3: SGD vs ADAM")
    optimizers_to_test = {
        "ADAM (lr=0.002)": Adam(lr=0.002),
        "Gradient (lr=0.002)": SGD(lr=0.002),
        "Gradient (lr=0.01)": SGD(lr=0.01),
        "Gradient (lr=0.05)": SGD(lr=0.05),
    }
    epochs_opt = 350

    plt.figure(figsize=(9, 5), dpi=300)
    for name, opt in optimizers_to_test.items():
        np.random.seed(42)
        layers = [
            Dense(input_dim, 32, init="he"),
            ReLU(),
            Dense(32, 32, init="he"),
            ReLU(),
            Dense(32, 1, init="he")
        ]
        model = MLP(layers=layers, optimizer=opt)
        train_l, _ = model.fit(X_train, y_train, epochs=epochs_opt, batch_size=32)
        plt.plot(train_l, label=name)

    plt.xlabel("Epoki")
    plt.ylabel("Wartość funkcji straty")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    save_plot("exp_optimizers")


if __name__ == "__main__":
    main()
