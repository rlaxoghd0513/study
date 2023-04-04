from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd

#1 데이터
(x_train, y_train), (x_test,y_test) = reuters.load_data(
    num_words=10, test_split=0.2#num_words 상위 몇개만 나와라
)

print(x_train)
print(y_train)
print(x_train.shape, y_train.shape) #(8982,) (8982,)
print(x_test.shape, y_test.shape) #(2246,) (2246,)

print(len(x_train[0]), len(x_train[1])) #87 56 넘파이 형태로 리스트

print(np.unique(y_train))
