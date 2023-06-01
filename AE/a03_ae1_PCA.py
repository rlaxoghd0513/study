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

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units = hidden_layer_size, input_shape = (784,)))
    model.add(Dense(784, activation = 'sigmoid'))
    return model

# model = autoencoder(hidden_layer_size=1)
# model = autoencoder(hidden_layer_size=154) #PCA 95퍼 성능
# model = autoencoder(hidden_layer_size=331) #pca 99퍼 성능
# model = autoencoder(hidden_layer_size=486) #pca 99.9퍼 성능
model = autoencoder(hidden_layer_size=713) #pca 100퍼 성능

##################################### m33_pca_mnist1.py 참고하기#########################################
# print(np.argmax(pca_cumsum >= 0.95)+1) #154
# print(np.argmax(pca_cumsum >= 0.99)+1) #331
# print(np.argmax(pca_cumsum >= 0.999)+1) #486
# print(np.argmax(pca_cumsum >= 1.0)+1) #713
# 히든 레이어를 154로 하면 0.95
# 331로 하면 0.99

#3 컴파일 훈련
model.compile(optimizer = 'adam', loss='mse')
model.fit(x_train_noised, x_train, epochs=30, batch_size=128)

#4 평가 예측
decoded_images =model.predict(x_test_noised)

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