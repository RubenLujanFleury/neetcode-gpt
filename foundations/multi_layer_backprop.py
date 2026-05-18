import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        pass

        # Make everything into a numpy array
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)


        # Forward pass
        z1 = x @ W1.T + b1
        a1 = self.ReLu(z1)
        z2 = a1 @ W2.T + b2
        
        # Loss
        MSE = np.mean((z2 - y_true) ** 2)

        # Backward pass (Iterate backwards using chain rule)

        n = len(y_true) if y_true.ndim > 0 else 1
        print(n)

        dL_dZ2 = 2*(z2 - y_true) / n

        dL_dW2 = dL_dZ2.reshape(-1, 1) @ a1.reshape(1, -1)
        dL_dB2 = dL_dZ2
        dL_dA1 = dL_dZ2.reshape(-1,1) @ W2

        dL_dZ1 = dL_dA1.flatten()
        dL_dZ1 = dL_dZ1 * (z1 > 0).astype(float)

        dL_dW1 = dL_dZ1.reshape(-1, 1) @ x.reshape(1, -1)
        dL_dB1 = dL_dZ1


        return {
            'loss': round(float(MSE), 4), 
            'dW1': np.round(dL_dW1, 4).tolist(),
            'db1': np.round(dL_dB1, 4).tolist(),
            'dW2': np.round(dL_dW2, 4).tolist(),
            'db2': np.round(dL_dB2, 4).tolist()
            }
        

    
    def ReLu(self, z):
        return np.maximum(0,z)

        
