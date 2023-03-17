import numpy as np
from tensorflow.keras.datasets import cifar10


(x_train, y_train), (x_test, y_test) =cifar10.load_data() 

print(x_train.shape, y_train.shape) 
print(x_test.shape, y_test.shape)   

print(x_train)
print(y_train)  
print(x_train[0])
print(y_train[3333])  


import matplotlib.pyplot as plt
plt.imshow(x_train[49995],)
plt.show()