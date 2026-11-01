import numpy as np


class Dense:
    def __init__(self, in_features: int, out_features: int, init: str = "default", scale: float | None = None):
        if scale is not None:
            w_scale = scale
        elif init == "he":
            w_scale = np.sqrt(2.0 / in_features)
        elif init == "normal_1":
            w_scale = 1.0
        elif init == "broken":
            w_scale = 5.0
        else:
            w_scale = 0.01

        self.W = np.random.normal(loc=0.0, scale=w_scale, size=(in_features, out_features))
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


class SGD:
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def step(self, layers: list) -> None:
        for layer in layers:
            if isinstance(layer, Dense):
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db


class Adam:
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self, layers: list) -> None:
        self.t += 1
        for i, layer in enumerate(layers):
            if not isinstance(layer, Dense):
                continue

            if i not in self.m:
                self.m[i] = {"W": np.zeros_like(layer.W), "b": np.zeros_like(layer.b)}
                self.v[i] = {"W": np.zeros_like(layer.W), "b": np.zeros_like(layer.b)}

            # wagi W
            self.m[i]["W"] = self.beta1 * self.m[i]["W"] + (1.0 - self.beta1) * layer.dW
            self.v[i]["W"] = self.beta2 * self.v[i]["W"] + (1.0 - self.beta2) * (layer.dW ** 2)
            m_W_hat = self.m[i]["W"] / (1.0 - self.beta1 ** self.t)
            v_W_hat = self.v[i]["W"] / (1.0 - self.beta2 ** self.t)
            layer.W -= self.lr * m_W_hat / (np.sqrt(v_W_hat) + self.eps)

            # biasy b
            self.m[i]["b"] = self.beta1 * self.m[i]["b"] + (1.0 - self.beta1) * layer.db
            self.v[i]["b"] = self.beta2 * self.v[i]["b"] + (1.0 - self.beta2) * (layer.db ** 2)
            m_b_hat = self.m[i]["b"] / (1.0 - self.beta1 ** self.t)
            v_b_hat = self.v[i]["b"] / (1.0 - self.beta2 ** self.t)
            layer.b -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)


class MLP:
    def __init__(self, layers: list, optimizer=None, lr: float = 0.01):
        self.layers = layers
        self.lr = lr
        if optimizer == "adam":
            self.optimizer = Adam(lr=lr)
        elif optimizer == "sgd":
            self.optimizer = SGD(lr=lr)
        else:
            self.optimizer = optimizer

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
        if self.optimizer is not None:
            self.optimizer.step(self.layers)
        else:
            for layer in self.layers:
                if isinstance(layer, Dense):
                    layer.W -= self.lr * layer.dW
                    layer.b -= self.lr * layer.db

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            epochs: int = 1000,
            batch_size: int | None = None,
            shuffle: bool = True) -> tuple[list[float], list[float]]:

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        n_samples = X.shape[0]

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val = np.asarray(X_val, dtype=np.float64)
            y_val = np.asarray(y_val, dtype=np.float64).reshape(-1, 1)

        actual_batch_size = n_samples if (batch_size is None or batch_size >= n_samples) else max(1, batch_size)
        train_losses, val_losses = [], []

        for _ in range(epochs):
            if shuffle and actual_batch_size < n_samples:
                indices = np.random.permutation(n_samples)
            else:
                indices = np.arange(n_samples)

            for start in range(0, n_samples, actual_batch_size):
                batch_idx = indices[start:start + actual_batch_size]
                xb, yb = X[batch_idx], y[batch_idx]

                y_pred_batch = self.forward(xb)
                self.backward(y_pred_batch, yb)
                self.step()

            y_pred_full = self.forward(X)
            train_losses.append(self.loss(y, y_pred_full))

            if has_val:
                val_losses.append(self.loss(y_val, self.forward(X_val)))

        return train_losses, val_losses

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.forward(np.asarray(X, dtype=np.float64)) >= threshold).astype(int)
