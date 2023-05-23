import numpy as np
import matplotlib.pyplot as plt

# def reaky_relu(x):
#     return np.maximum(0.1*x , x)

reaky_relu = lambda x: np.maximum(0.1*x ,x)

x = np.arange(-5,5,0.1)
y = reaky_relu(x)

plt.plot(x,y)
plt.grid()
plt.show()