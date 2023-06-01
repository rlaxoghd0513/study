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

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units = hidden_layer_size, input_shape = (784,)))
    model.add(Dense(784, activation = 'sigmoid'))
    return model

model_01 = autoencoder(hidden_layer_size=1)
model_08 = autoencoder(hidden_layer_size=8)
model_32 = autoencoder(hidden_layer_size=32)
model_64 = autoencoder(hidden_layer_size=64) 
model_154 = autoencoder(hidden_layer_size=154) #PCA 95퍼 성능
model_331 = autoencoder(hidden_layer_size=331) #PCA 99퍼 성능
model_486 = autoencoder(hidden_layer_size=486) #PCA 99.9퍼 성능
model_713 = autoencoder(hidden_layer_size=713) #PCA 100퍼성능

##################################### m33_pca_mnist1.py 참고하기#########################################
# print(np.argmax(pca_cumsum >= 0.95)+1) #154
# print(np.argmax(pca_cumsum >= 0.99)+1) #331
# print(np.argmax(pca_cumsum >= 0.999)+1) #486
# print(np.argmax(pca_cumsum >= 1.0)+1) #713
# 히든 레이어를 154로 하면 0.95
# 331로 하면 0.99

#3 컴파일 훈련
print('=================================== node 1개 시작 ===========================================')
model_01.compile(optimizer = 'adam', loss='mse')
model_01.fit(x_train_noised, x_train, epochs=10, batch_size=32)

print('=================================== node 8개 시작 ===========================================')
model_08.compile(optimizer = 'adam', loss='mse')
model_08.fit(x_train_noised, x_train, epochs=10, batch_size=32)

print('=================================== node 32개 시작 ===========================================')
model_32.compile(optimizer = 'adam', loss='mse')
model_32.fit(x_train_noised, x_train, epochs=10, batch_size=32)

print('=================================== node 64개 시작 ===========================================')
model_64.compile(optimizer = 'adam', loss='mse')
model_64.fit(x_train_noised, x_train, epochs=10, batch_size=32)

print('=================================== node 154개 시작 ===========================================')
model_154.compile(optimizer = 'adam', loss='mse')
model_154.fit(x_train_noised, x_train, epochs=10, batch_size=32)

print('=================================== node 331개 시작 ===========================================')
model_331.compile(optimizer = 'adam', loss='mse')
model_331.fit(x_train_noised, x_train, epochs=10, batch_size=32)

print('=================================== node 486개 시작 ===========================================')
model_486.compile(optimizer = 'adam', loss='mse')
model_486.fit(x_train_noised, x_train, epochs=10, batch_size=32)

print('=================================== node 713개 시작 ===========================================')
model_713.compile(optimizer = 'adam', loss='mse')
model_713.fit(x_train_noised, x_train, epochs=10, batch_size=32)




#4 평가 예측
decoded_images_01 =model_01.predict(x_test_noised)
decoded_images_08 =model_08.predict(x_test_noised)
decoded_images_32 =model_32.predict(x_test_noised)
decoded_images_64 =model_64.predict(x_test_noised)
decoded_images_154 =model_154.predict(x_test_noised)
decoded_images_331 =model_331.predict(x_test_noised)
decoded_images_486 =model_486.predict(x_test_noised)
decoded_images_713 =model_713.predict(x_test_noised)

###################################################################################################

from matplotlib import pyplot as plt
import random

fig, axes = plt.subplots(7,5, figsize = (15,15))

# 이미지 다섯개를 무작위로 고른다
random_images = random.sample(range(decoded_images_01.shape[0]),5)

outputs = [x_test, decoded_images_01, decoded_images_08, decoded_images_32, decoded_images_64, decoded_images_154, decoded_images_331 , decoded_images_486 , decoded_images_713]

for row_num, row in enumerate(axes):
    for col_num, ax in enumerate(row):
        ax.imshow(outputs[row_num][random_images[col_num]].reshape(28,28), cmap='gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
plt.show()