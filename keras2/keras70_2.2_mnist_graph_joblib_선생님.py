######## 실습(훈련시키지말고 70-1에서 가중치든 뭐든 땡겨다가 그래프를 그려라) #########

#keras32 mnist3

from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential,load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np
import pandas as pd
import tensorflow as tf
tf.random.set_seed(337)

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

########### 실습 #############
scaler=MinMaxScaler()
x_train = x_train.reshape(-1,1)
x_train = scaler.fit_transform(x_train)
x_test = x_test.reshape(-1,1)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

y_train=np.array(pd.get_dummies(y_train,prefix='number'))
y_test=np.array(pd.get_dummies(y_test,prefix='number'))

# 2. 모델구성
# model = load_model('./_save/keras70_1_mnist_grape.h5')
#2. 모델 - 피클 불러오기
# history 객체 로드
import joblib
try:
    hist = joblib.load('./_save/keras70_1_history.dat')
except EOFError:
    print('EOFError 발생')

import matplotlib.pyplot as plt
plt.figure(figsize=(9,5))
#1
plt.subplot(2,1,1)
plt.plot(hist['loss'], marker = '.', c='red', label = 'loss')
plt.plot(hist['val_loss'], marker = '.', c = 'blue', label = 'val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc = 'upper right')

#2
plt.subplot(2,1,2)
plt.plot(hist['acc'], marker = '.', c='red', label = 'acc')
plt.plot(hist['val_acc'], marker = '.', c = 'blue', label = 'val_acc')
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epochs')    
plt.legend(['acc','val_acc'])

plt.show()