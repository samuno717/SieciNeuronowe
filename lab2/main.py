#%%
import sys
from config import PROJECT_ROOT

sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "lab2"))

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig

from lab2.LogisticRegression import LogisticRegression
from preprocessing import get_raw_data, process_features
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

p = PROJECT_ROOT / "lab2"


@hydra.main(version_base=None, config_path="..", config_name="params")
def main(cfg: DictConfig) -> None:
    X, y = get_raw_data()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=cfg.experiment.seed, stratify=y
    )
    X_train, X_test = process_features(X_train_raw, X_test_raw)

    # LEARNING RATE
    # ------------------------------------------------------------------
    print("Wpływ współczynnika learning rate")

    lr_results = []
    plt.figure(figsize=(10, 6), dpi=300)

    for lr in cfg.convergence_study.learning_rates:
        model = LogisticRegression(
            epochs=cfg.convergence_study.epochs,
            learning_rate=lr,
            weights="zeros",
            tol=cfg.convergence_study.tol,
            track_history=True,
            random_state=cfg.experiment.seed
        )
        model.fit(X_train, y_train, eval_set=(X_test, y_test))
        y_pred = model.predict(X_test)

        lr_results.append({
            "Learning Rate": lr,
            "Liczba epok": model.n_iter_,
            "F1-Score": f1_score(y_test, y_pred, zero_division=0),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "Accuracy": accuracy_score(y_test, y_pred),
            "Final Loss": model.history["train_loss"][-1] if model.history["train_loss"] else np.nan
        })

        plt.plot(model.history["train_loss"], label=f"lr = {lr} (iter={model.n_iter_})")

    #plt.title("Zbieżność funkcji kosztu w zależności od współczynnika uczenia")
    plt.xlabel("Epoka")
    plt.ylabel("Wartość funkcji kosztu")
    plt.yscale("log")
    plt.yticks(
        [0.3, 0.5, 0.7, 1.0, 2.0, 5.0, 10.0],
        labels=["0.3", "0.5", "0.7", "1.0", "2.0", "5.0", "10.0"],
    )
    plt.legend()
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
    plt.savefig(p / 'plots' / "lr_plot.svg", bbox_inches="tight")
    plt.close()

    df_lr = pd.DataFrame(lr_results).round(3)
    print(df_lr.to_string(index=False))
    # ------------------------------------------------------------------

    # WEIGHTS
    # ------------------------------------------------------------------
    print("\nWpływ inicjalizacji wag")

    weights_results = []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    styles = {
        "zeros":   {"ls": "-",  "lw": 2.5, "alpha": 0.8},
        "random":  {"ls": "--", "lw": 1.8, "alpha": 0.9},
        "uniform": {"ls": ":",  "lw": 2.2, "alpha": 0.9},
        "ones":    {"ls": "-.", "lw": 1.8, "alpha": 0.9},
        "extreme": {"ls": "-",  "lw": 1.5, "alpha": 0.7},
    }

    for w_type in cfg.convergence_study.weights:
        model = LogisticRegression(
            epochs=cfg.convergence_study.epochs,
            learning_rate=2.0,
            weights=w_type,
            tol=cfg.convergence_study.tol,
            track_history=True,
            random_state=cfg.experiment.seed
        )
        model.fit(X_train, y_train, eval_set=(X_test, y_test))
        y_pred = model.predict(X_test)

        weights_results.append({
            "Inicjalizacja": w_type,
            "Liczba epok": model.n_iter_,
            "F1-Score": f1_score(y_test, y_pred, zero_division=0),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "Accuracy": accuracy_score(y_test, y_pred),
            "Final Loss": model.history["train_loss"][-1] if model.history["train_loss"] else np.nan
        })

        st = styles.get(w_type, {})
        history = model.history["train_loss"]
        label = f"{w_type} (iter={model.n_iter_})"

        ax1.plot(history, label=label, **st)

        ax2.plot(history[:30], label=label, **st)

    ax1.set_xlabel("Epoka")
    ax1.set_ylabel("Wartość funkcji kosztu (skala log)")
    ax1.set_yscale("log")
    ax1.set_title("Zbieżność w pełnym zakresie (skala log)")
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend()

    ax2.set_xlabel("Epoka")
    ax2.set_ylabel("Wartość funkcji kosztu")
    ax2.set_title("Dynamika początkowa (pierwsze 30 epok)")
    ax2.grid(True, ls="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(p / 'plots' / "weights_plot.svg", bbox_inches="tight")
    plt.close()

    df_weights = pd.DataFrame(weights_results).round(3)
    print(df_weights.to_string(index=False))
    # ------------------------------------------------------------------

    # LEARNING CURVE
    # ------------------------------------------------------------------
    print("\nKrzywa uczenia")

    curve_results = []
    best_model = LogisticRegression(
        epochs=cfg.convergence_study.epochs,
        learning_rate=2.0,
        weights="zeros",
        tol=None,
        track_history=True,
        random_state=cfg.experiment.seed
    )
    best_model.fit(X_train, y_train, eval_set=(X_test, y_test))
    y_pred = best_model.predict(X_test)

    curve_results.append({
        "Liczba epok": best_model.n_iter_,
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_test, y_pred),
        "Final Train Loss": best_model.history["train_loss"][-1],
        "Final Val Loss": best_model.history["val_loss"][-1],
    })

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    epochs_range = range(1, len(best_model.history["train_loss"]) + 1)
    color = "tab:blue"
    ax1.set_xlabel("Epoka")
    ax1.set_ylabel("Wartość funkcji kosztu", color=color)
    line1 = ax1.plot(epochs_range, best_model.history["train_loss"], color=color, label="Train Loss")
    line2 = ax1.plot(epochs_range, best_model.history["val_loss"], color="tab:cyan", linestyle="--", label="Val Loss")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("F1-Score (zbiór walidacyjny)", color=color)
    line3 = ax2.plot(epochs_range, best_model.history["val_f1"], color=color, label="Val F1-Score")
    ax2.tick_params(axis="y", labelcolor=color)

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right")
    #plt.title("Krzywa uczenia: Funkcja kosztu i metryka F1-Score w czasie")
    plt.grid(True)
    plt.savefig(p / 'plots' / "learning_curve.svg", bbox_inches="tight")
    plt.close()

    df_curve = pd.DataFrame(curve_results).round(3)
    print(df_curve.to_string(index=False))
    # ------------------------------------------------------------------

if __name__ == "__main__":
    main()
