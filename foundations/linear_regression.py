import numpy as np
from numpy.typing import NDArray


# Helpful functions:
# https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
# https://numpy.org/doc/stable/reference/generated/numpy.mean.html
# https://numpy.org/doc/stable/reference/generated/numpy.square.html

class Solution:
    
    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is an Nx3 NumPy array
        # weights is a 3x1 NumPy array
        y = np.matmul(X,weights)
        # HINT: np.matmul() will be useful
        return np.round(y, 5)
        # return np.round(your_answer, 5)


    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        y_pred = model_prediction
        y = ground_truth
        error = 0
        error = np.square(y - y_pred)
        return round(np.mean(error), 5)
        # model_prediction is an Nx1 NumPy array
        # ground_truth is an Nx1 NumPy array
        # HINT: np.mean(), np.square() will be useful
        # return round(your_answer, 5)
