#난 정말 시그모이드 ~~ ♪

import numpy as np
import matplotlib.pyplot as plt

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))   #0에서 1사이로 수렴한다 1/1+(e의 x승)

sigmoid = lambda x : 1 / (1 + np.exp(-x))

x = np.arange(-5,25,0.1) #-5부터 5까지 0.1의 거리로 뽑아내기
print(x)

y = sigmoid(x)

plt.plot(x,y)
plt.grid()
plt.show()