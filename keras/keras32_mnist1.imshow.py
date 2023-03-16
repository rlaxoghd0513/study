import numpy as np
from tensorflow.keras.datasets import mnist
# tensorflow pytorch빼곤 다 sklearn 에 있다
(x_train, y_train), (x_test, y_test) = mnist.load_data()  #사이킷런은 xy로 주는데 얘는 이미 분리를 해서 준다

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)   흑백은 1인데 끝에 명시하지 않을수도 있다  (60000, 28, 28, 1)로 reshape해준다
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

print(x_train)
print(y_train)  #[5 0 4 ... 5 6 8]
print(x_train[0])
print(y_train[3333])  #5

import matplotlib.pyplot as plt
plt.imshow(x_train[3333], 'gray')
plt.show()
