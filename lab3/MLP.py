import numpy as np


class Dense:
    def __init__(self, in_features: int, out_features: int):
        self.W = np.random.normal(loc=0.0, scale=0.01, size=(in_features, out_features))
        self.b = np.zeros((1, out_features))
        self.cache = None
        self.dW = None
        self.db = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache = x
        return x @ self.W + self.b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        self.dW = self.cache.T @ dout
        self.db = np.sum(dout, axis=0, keepdims=True)
        return dout @ self.W.T


class ReLU:
    def __init__(self):
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache = x
        return np.maximum(0, x)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * (self.cache > 0)


class MLP:
    def __init__(self, layers: list, lr: float = 0.01):
        self.layers = layers
        self.lr = lr

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        eps = 1e-15
        y_clipped = np.clip(y_pred, eps, 1.0 - eps)
        return float(-np.mean(y_true * np.log(y_clipped) + (1.0 - y_true) * np.log(1.0 - y_clipped)))

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return self.sigmoid(out)

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        dout = (y_pred - y_true) / y_true.shape[0]
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def step(self) -> None:
        for layer in self.layers:
            if isinstance(layer, Dense):
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            epochs: int = 1000) -> tuple[list[float], list[float]]:

        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)
        has_val = X_val is not None and y_val is not None

        if has_val:
            X_val = np.asarray(X_val)
            y_val = np.asarray(y_val).reshape(-1, 1)

        train_losses, val_losses = [], []

        for _ in range(epochs):
            y_pred = self.forward(X)
            train_losses.append(self.loss(y, y_pred))

            self.backward(y_pred, y)
            self.step()

            if has_val:
                val_losses.append(self.loss(y_val, self.forward(X_val)))

        return train_losses, val_losses

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.forward(np.asarray(X)) >= threshold).astype(int)