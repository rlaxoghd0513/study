#[실습] a03_ae2 카피해서 모델 딥하게 구성


import numpy as np
from tensorflow.keras.datasets import mnist

#1 데이터
(x_train, _), (x_test, _) = mnist.load_data() #x로 훈련, 결과를 내기 위해

x_train = x_train.reshape(60000, 784).astype('float32')/255.
x_test = x_test.reshape(10000, 784).astype('float32')/255.

x_train_noised = x_train + np.random.normal(0, 0.3, size = x_train.shape) # 0에서 0.1사이의 값을 랜덤하게 넣어준다
x_test_noised = x_test + np.random.normal(0, 0.3, size = x_test.shape) 

print(x_train_noised.shape, x_test_noised.shape) #(60000, 784) (10000, 784)

print(np.max(x_train_noised), np.min(x_train_noised)) #1.5147243549280165 -0.5203567546484759
print(np.max(x_test_noised), np.min(x_test_noised)) #1.4720130524999153 -0.5252133015854844

#np.random.normal이 0에서 표준편차를 0.1로 갖는 값이기 때문에 음수가 나올 수도 있다

x_train_noised = np.clip(x_train_noised, a_min = 0, a_max=1) #0보다 작은건 0으로 1보다 큰건 1로 고정시켜준다
x_test_noised = np.clip(x_test_noised, a_min = 0, a_max=1) #0보다 작은건 0으로 1보다 큰건 1로 고정시켜준다
print(np.max(x_train_noised), np.min(x_train_noised)) #1.0 0.0
print(np.max(x_test_noised), np.min(x_test_noised)) #1.0 0.0

#####아까 만든거 가지고 확인####
#2 모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# hidden_layer_size = [512,256,128,256,512]
hidden_layer_size = [128,256,512,256,128]

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units = hidden_layer_size[0], input_shape = (784,)))
    model.add(Dense(units = hidden_layer_size[1]))
    model.add(Dense(units = hidden_layer_size[2]))
    model.add(Dense(units = hidden_layer_size[3]))
    model.add(Dense(units = hidden_layer_size[4]))
    model.add(Dense(784, activation = 'sigmoid'))
    return model

model = autoencoder(hidden_layer_size=hidden_layer_size)

#3 컴파일 훈련
model.compile(optimizer = 'adam', loss='mse')
model.fit(x_train_noised, x_train, epochs=30, batch_size=128)

#4 평가 예측
decoded_images =model.predict(x_test_noised)

###################################################################################################

from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6,ax7,ax8, ax9, ax10),
      (ax11, ax12, ax13, ax14, ax15)) = \
      plt.subplots(3,5,figsize = (20,7))

# 이미지 다섯개를 무작위로 고른다
random_images = random.sample(range(decoded_images.shape[0]),5)

#원본 이미지를 맨위에 그린다
for i,ax in enumerate([ax1,ax2,ax3,ax4,ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap = 'gray')
    if i ==0:
        ax.set_ylabel('INPUT', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

#노이즈를 넣은 이미지
for i,ax in enumerate([ax6,ax7,ax8,ax9,ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28,28), cmap='gray')
    if i ==0:
        ax.set_ylabel('NOISE', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

#오토인코더가 출력한 이미지를 아래에 그린다
for i,ax in enumerate([ax11,ax12,ax13,ax14,ax15]):
    ax.imshow(decoded_images[random_images[i]].reshape(28,28), cmap='gray')
    if i ==0:
        ax.set_ylabel('OUTPUT', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()