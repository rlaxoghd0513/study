# 자동제거, x를 x로 훈련시킨다. 준지도 학습.

import numpy as np
from tensorflow.keras.datasets import mnist

#1 데이터
(x_train, _), (x_test, _) = mnist.load_data() #x로 훈련, 결과를 내기 위해

x_train = x_train.reshape(60000, 784).astype('float32')/255.
x_test = x_test.reshape(10000, 784).astype('float32')/255.

#2 모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input_img = Input(shape=(784,))
# encoded = Dense(64, activation='relu')(input_img) #노드를 너무 줄이면 특성이 사라지기 때문에 뿌얘지는 문제가 있다
encoded = Dense(1024, activation='relu')(input_img)
# encoded = Dense(1, activation='relu')(input_img)
# encoded = Dense(32, activation='relu')(input_img)

# decoded = Dense(784, activation='linear')(encoded)
# decoded = Dense(784, activation='relu')(encoded)
decoded = Dense(784, activation='tanh')(encoded)
# decoded = Dense(784, activation='sigmoid')(encoded)
autoencoder = Model(input_img, decoded)

autoencoder.summary()
# Model: "model"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_1 (InputLayer)        [(None, 784)]             0

#  dense (Dense)               (None, 64)                50240

#  dense_1 (Dense)             (None, 784)               50960

# =================================================================
# Total params: 101,200
# Trainable params: 101,200
# Non-trainable params: 0
# _________________________________________________________________

#오토인코더의 고질적인 문제 : 사진이 뿌얘질수도 있음. -> 압축 -> 풀기를 하기때문에.
#학습자체가 문제있는 사진, 문제없는 사진 이렇게 학습.

# autoencoder.compile(optimizer='adam', loss='mse', metrics=['acc'])
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

#3 컴파일 훈련
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(x_train,x_train, epochs =30, batch_size= 128,
                validation_split=0.2)

#4 평가 예측
decoded_images = autoencoder.predict(x_test)

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize = (20,4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_images[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
    
# encoded 64, decoded linear 일때  loss: 0.0095 - val_loss: 0.0095
# encoded 64, decoded relu   일때  loss: 0.0071 - val_loss: 0.0071 잘나옴
# encoded 64, decoded tanh   일때  loss: 0.0124 - val_loss: 0.0124
# encoded 64, decoded sigmoid 일때  loss: 0.0038 - val_loss: 0.0040 이게 제일 잘나옴

# encoded 1024, decoded linear 일때 loss: 2.1220e-04 - val_loss: 3.2136e-04
# encoded 1024, decoded relu 일때  loss: 0.0038 - val_loss: 0.0037
# encoded 1024, decoded tanh 일때  loss: 0.0015 - val_loss: 0.0015



