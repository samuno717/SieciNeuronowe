#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from lab3.MLP import MLP, Dense, ReLU
from lab1.preprocessing import get_raw_data, process_features

def main():
    X, y = get_raw_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_test = process_features(X_train, X_test)

    input_dim = X_train.shape[1]
    epochs = 60000
    lr = 0.01

    summary = []

    # Różna liczba neuronów
    print("Szerokości warstwy...")
    configs_width = [4, 20, 64]
    losses_width = {}

    plt.figure(figsize=(9, 5), dpi=300)

    for neurons in configs_width:
        np.random.seed(42)
        model = MLP(
            layers=[
                Dense(input_dim, neurons),
                ReLU(),
                Dense(neurons, 1)
            ],
            lr=lr
        )
        train_l, _ = model.fit(X_train, y_train, epochs=epochs)
        y_pred = model.predict(X_test)
        losses_width[neurons] = train_l

        summary.append({
            "Eksperyment": "Szerokość warstwy",
            "Wariant": f"{neurons} neurony",
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred, zero_division=0),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
        })

        plt.plot(train_l, label=f"{neurons} neuronów")

    plt.xlabel("Epoki")
    plt.ylabel("Loss")
    plt.title("Wpływ liczby neuronów w warstwie ukrytej")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("exp_neurons.svg", bbox_inches="tight")
    plt.close()

    # Głębokość sieci
    print("Głębokość sieci...")
    architectures = {
        "1 warstwa": [Dense(input_dim, 20), ReLU(), Dense(20, 1)],
        "2 warstwy": [Dense(input_dim, 20), ReLU(), Dense(20, 20), ReLU(), Dense(20, 1)],
        "3 warstwy": [Dense(input_dim, 20), ReLU(), Dense(20, 20), ReLU(), Dense(20, 20), ReLU(), Dense(20, 1)]
    }

    plt.figure(figsize=(9, 5), dpi=300)

    for name, layers in architectures.items():
        np.random.seed(42)
        model = MLP(layers=layers, lr=lr)

        train_l, _ = model.fit(X_train, y_train, epochs=epochs)
        y_pred = model.predict(X_test)

        summary.append({
            "Eksperyment": "Głębokość sieci",
            "Wariant": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred, zero_division=0),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
        })

        plt.plot(train_l, label=name)

    plt.xlabel("Epoki")
    plt.ylabel("Loss")
    plt.title("Wpływ liczby warstw ukrytych")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("exp_layers.svg", bbox_inches="tight")
    plt.close()

    # Train Loss vs Test Loss
    print("Train vs Test Loss...")
    np.random.seed(42)
    overfit_model = MLP(
        layers=[Dense(input_dim, 64), ReLU(), Dense(64, 1)],
        lr=lr
    )
    train_l, test_l = overfit_model.fit(X_train, y_train, X_val=X_test, y_val=y_test, epochs=epochs)

    plt.figure(figsize=(9, 5), dpi=300)
    plt.plot(train_l, label="Train Loss", color="#1f77b4")
    plt.plot(test_l, label="Test Loss", color="#d62728", linestyle="--")
    plt.xlabel("Epoki")
    plt.ylabel("Loss")
    plt.title("Analiza przeuczenia (64 neurony)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("exp_overfitting.svg", bbox_inches="tight")
    plt.close()

    summary_df = pd.DataFrame(summary)
    print("\n" + summary_df.round(4).to_string(index=False))

if __name__ == "__main__":
    main()