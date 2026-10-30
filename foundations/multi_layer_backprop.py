import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        z1 = x @ np.transpose(W1) + b1
        a1 = np.maximum(0, z1)
        y_pred = a1 @ np.transpose(W2) + b2
        loss = np.mean(np.square(y_pred - y_true))

        dL_by_dy_pred = (2 * (y_pred - y_true)) / len(y_true)
        dL_by_dW2 = np.outer(dL_by_dy_pred, a1)
        dL_by_db2 = dL_by_dy_pred
        dL_by_dz1 = np.transpose(W2) @ dL_by_dy_pred * (z1 > 0)
        dL_by_dW1 = np.outer(dL_by_dz1, x)
        dL_by_db1 = dL_by_dz1

        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        return {
            'loss'  : np.round(loss, 4),
            'dW1'   : np.round(dL_by_dW1, 4),
            'db1'   : np.round(dL_by_db1, 4),
            'dW2'   : np.round(dL_by_dW2, 4),
            'db2'   : np.round(dL_by_db2, 4),
        }
