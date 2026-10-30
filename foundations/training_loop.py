import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        m, n = X.shape
        w = np.zeros(n)
        b = 0.0

        bias_learner = np.ones_like(X)

        for i in range(epochs):
            y_hat =  X @ w + b
            error = y_hat - y
            loss = np.mean(np.square(error))
            dw = 2 * X.T @ (error) / m
            db = 2 * np.sum(error) / m

            w = w - lr * dw
            b = b - lr * db

        return np.round(w, decimals=5), np.round(b, decimals=5)