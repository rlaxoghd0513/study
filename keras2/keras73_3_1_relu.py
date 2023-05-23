import numpy as np
import matplotlib.pyplot as plt

# def relu(x):
    # return np.maximum(0, x) # 0과 x를 비교해서 큰 값을 리턴한다

relu = lambda x: np.maximum(0,x)

x = np.arange(-5,5,0.1)
y = relu(x)

plt.plot(x,y)
plt.grid()
plt.show()