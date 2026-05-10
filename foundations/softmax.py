import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        pass
        
        sum = 0

        for i in range(len(z)):
            sum = sum + np.exp(z[i]-np.max(z))
            print(sum)
        

        z = np.exp(z-np.max(z))/sum
        print(z)


        return np.round(z, 4)
        