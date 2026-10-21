from typing import Any

import numpy as np
from numpy import dtype, ndarray


class Dense:
    def __init__(self,
                 in_features: int,
                 out_features: int = 1):

        self.in_features = in_features
        self.out_features = out_features
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros((1, out_features))
        self.cache = None
        self.dW = None
        self.db = None

    def forward(self, x: ndarray) -> ndarray:
        z = x @ self.W + self.b
        self.cache = x
        return z

    def backward(self, dout: ndarray) -> ndarray:
        pass


class ReLU:
    def __init__(self):
        self.cache = None

    def forward(self, x: ndarray) -> ndarray:
        a = np.maximum(0, x)
        self.cache = x
        return a

    def backward(self, dout: ndarray) -> ndarray:
        pass


class MLP:
    def __init__(self, input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 lr: float = 0.01):

        self.lr = lr
        self.layers = [
            Dense(input_dim, hidden_dim),
            ReLU(),
            Dense(hidden_dim, output_dim),
        ]

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, X: np.ndarray) -> np.ndarray:
        pass

