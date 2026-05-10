import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        pass

        n = len(y_true)
        sum = 0

        for i in range(n):
            sum = sum + (y_true[i] * np.log(y_pred[i] + 1e-7) + (1 - y_true[i]) * np.log(1-y_pred[i] + 1e-7))

        L = -(1/n) * sum

        return round(L,4)


    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        pass
        
        sum = 0
        n = len(y_true)

        for i in range(len(y_true)):
            for c in range(len(y_true[i])):
                sum = sum + y_true[i][c] * np.log(y_pred[i][c])

        L = -(1/n) * sum

        return round(L, 4)


