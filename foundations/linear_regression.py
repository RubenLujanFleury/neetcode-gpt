import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        pass
        ans = np.matmul(X, weights)
        return np.round(ans, 5)


    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        pass

        sum = 0
        n = len(model_prediction)
        
        for i in range(n):
            sum = sum + (model_prediction[i] - ground_truth[i])**2

        MSE = (1/n) * sum
        return round(MSE[0], 5)
