from typing import Self
import numpy as np
from sklearn.metrics import f1_score


class LogisticRegression:
    def __init__(
        self,
        learning_rate: float | int = 0.01,
        epochs: int = 100,
        weights: str = "zeros",
        tol: float | None = None,
        random_state: int | None = 42,
        track_history: bool = False,
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = weights
        self.tol = tol
        self.random_state = random_state
        self.track_history = track_history

        self.W = None
        self.b = None
        self.n_iter_ = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_f1": []
        }

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def count_loss(self, y_true: np.ndarray, p: np.ndarray) -> float:
        p = np.clip(p, 1e-15, 1.0 - 1e-15)
        loss = -y_true * np.log(p) - (1.0 - y_true) * np.log(1.0 - p)
        return float(np.mean(loss))

    def initiate_weights(self, n_features: int) -> Self:
        rng = np.random.default_rng(self.random_state)

        if self.weights == "zeros":
            self.W = np.zeros(n_features)
            self.b = 0.0

        elif self.weights == "ones":
            self.W = np.ones(n_features)
            self.b = 1.0

        elif self.weights == "random":
            self.W = rng.normal(loc=0.0, scale=0.01, size=n_features)
            self.b = 0.0

        elif self.weights == "uniform":
            limit = 1.0 / np.sqrt(n_features)
            self.W = rng.uniform(-limit, limit, n_features)
            self.b = 0.0

        elif self.weights == "extreme":
            self.W = rng.choice([-50.0, 50.0], size=n_features)
            self.b = 0.0

        else:
            raise ValueError(f"Unknown weight type: {self.weights}")

        return self

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> Self:

        X_mat = np.asarray(X, dtype=np.float64)
        y_vec = np.asarray(y, dtype=np.float64)

        if eval_set is not None:
            X_val, y_val = eval_set
            X_val_mat = np.asarray(X_val, dtype=np.float64)
            y_val_vec = np.asarray(y_val, dtype=np.float64)
        else:
            X_val_mat, y_val_vec = None, None

        n_samples, n_features = X_mat.shape
        self.initiate_weights(n_features=n_features)
        self.history = {"train_loss": [], "val_loss": [], "val_f1": []}

        prev_loss = None

        for epoch in range(self.epochs):
            self.n_iter_ = epoch + 1

            z = X_mat @ self.W + self.b
            p = self.sigmoid(z)
            train_loss = self.count_loss(y_vec, p)

            if self.track_history:
                self.history["train_loss"].append(train_loss)

                if eval_set is not None:
                    p_val = self.sigmoid(X_val_mat @ self.W + self.b)
                    val_loss = self.count_loss(y_val_vec, p_val)
                    val_pred = (p_val >= 0.5).astype(int)
                    val_f1 = float(f1_score(y_val_vec, val_pred, zero_division=0))

                    self.history["val_loss"].append(val_loss)
                    self.history["val_f1"].append(val_f1)

            if self.tol is not None and prev_loss is not None:
                if abs(prev_loss - train_loss) < self.tol:
                    break

            prev_loss = train_loss

            error = p - y_vec
            dW = (X_mat.T @ error) / n_samples
            db = np.mean(error)

            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db

        return self

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        X_mat = np.asarray(X, dtype=np.float64)
        return self.sigmoid(X_mat @ self.W + self.b)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_probabilities(X) >= threshold).astype(int)
