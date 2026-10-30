import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        z = np.dot(x, w) + b
        y_pred = 1 / (1 + np.e ** (-z))
        # Loss: L = 0.5 * (y_hat - y_true)^2
        loss = 0.5 * np.square(y_pred - y_true)
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        dL_by_dw = np.dot((y_pred - y_true), np.dot(y_pred, np.dot((1 - y_pred), x)))
        dL_by_db = np.dot((y_pred - y_true), np.dot(y_pred, (1 - y_pred)))
        return np.round(dL_by_dw, 5), np.round(dL_by_db, 5)