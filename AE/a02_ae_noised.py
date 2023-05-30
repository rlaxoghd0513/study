#노이즈 강제로 만들어보기 

import numpy as np
from tensorflow.keras.datasets import mnist

#1 데이터
(x_train, _), (x_test, _) = mnist.load_data() #x로 훈련, 결과를 내기 위해

x_train = x_train.reshape(60000, 784).astype('float32')/255.
x_test = x_test.reshape(10000, 784).astype('float32')/255.

x_train_noised = x_train + np.random.normal(0, 0.1, size = x_train.shape) # 0에서 0.1사이의 값을 랜덤하게 넣어준다
x_test_noised = x_test + np.random.normal(0, 0.1, size = x_test.shape) 

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

input_img = Input(shape=(784,))
# encoded = Dense(64, activation='relu')(input_img) #노드를 너무 줄이면 특성이 사라지기 때문에 뿌얘지는 문제가 있다
encoded = Dense(1024, activation='relu')(input_img)
# encoded = Dense(1, activation='relu')(input_img)
# encoded = Dense(32, activation='relu')(input_img)

# decoded = Dense(784, activation='linear')(encoded)
decoded = Dense(784, activation='relu')(encoded)
# decoded = Dense(784, activation='tanh')(encoded)
# decoded = Dense(784, activation='sigmoid')(encoded)
autoencoder = Model(input_img, decoded)

# autoencoder.summary()

# autoencoder.compile(optimizer='adam', loss='mse', metrics=['acc'])
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

#3 컴파일 훈련
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(x_train_noised,x_train, epochs =30, batch_size= 128,
                validation_split=0.2)

#4 평가 예측
decoded_images = autoencoder.predict(x_test_noised)

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize = (20,4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test_noised[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_images[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


