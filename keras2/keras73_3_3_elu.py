import numpy as np
import matplotlib.pyplot as plt

# def elu(x, alpha=0.1):
#     if x >= 0:
#         return x
#     else:
#         return alpha * (np.exp(x) - 1)

elu = lambda x, alpha=0.1: np.where(x >= 0, x, alpha * (np.exp(x) - 1))

x = np.arange(-5,5,0.1)
y = elu(x)

plt.plot(x,y)
plt.grid()
plt.show()