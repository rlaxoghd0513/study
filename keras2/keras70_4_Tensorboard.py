from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.metrics import accuracy_score
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
model = Sequential()
model.add(Conv2D(64, (2,2), padding='same', input_shape=(28,28,1)))
model.add(MaxPooling2D())
model.add(Conv2D(64, (2,2), padding='valid', activation='relu'))
model.add(Conv2D(33, 2))
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(32, activation='relu')) #앞의 필터의 개수만큼 노드로 만들어줌 33개의 평균값을내서 하나로 만듬.
model.add(Dense(10, activation='softmax'))
model.summary()

# 3. 컴파일, 훈련
hist = model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard

es = EarlyStopping(monitor = 'val_loss', patience = 20, mode = 'min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 10, mode = 'auto', verbose=1, factor = 0.8)
tb = TensorBoard(log_dir = 'c:/users/bitcamp/study/_save/_tensorboard/_graph',
                 histogram_freq=0,
                 write_graph=True,
                 write_images=True)

#실행방법 : 경로로 가서 tensorboard --logdir=.    
#웹 켜고 http://localhost:6006/
#       http://127.0.0.1.6006   둘이 같다

import time
start = time.time()
hist = model.fit(x_train, y_train, epochs=5, batch_size=128, verbose=1, validation_split=0.2, callbacks=[es,reduce_lr,tb])
end = time.time()

# 걸린 시간 계산
elapsed_time = end - start

# 분과 초로 변환
minutes = elapsed_time // 60
seconds = elapsed_time % 60


# 4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result : ', result)
print('loss :', result[0])
print('acc', result[1])
y_predict = model.predict(x_test)
acc = accuracy_score(np.argmax(y_test,axis=1), np.argmax(y_predict,axis=1))
print(f'acc : {acc}')
# 출력
print("걸린 시간: {}분 {}초".format(int(minutes), int(seconds)))

model.save('./_save/keras70_1_mnist_graph2.h5')

# history 객체 저장
path = 'c:/users/bitcamp/study/_save/'
import joblib
joblib.dump(hist.history, path + 'keras70_1_history.dat') #hist.history는 딕셔너리가 모여있는 리스트 형태

#####################시각화############################

import matplotlib.pyplot as plt
plt.figure(figsize=(9,5))
#1
plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker = '.', c='red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc = 'upper right')

#2
plt.subplot(2,1,2)
plt.plot(hist.history['acc'], marker = '.', c='red', label = 'acc')
plt.plot(hist.history['val_acc'], marker = '.', c = 'blue', label = 'val_acc')
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epochs')
plt.legend(['acc','val_acc'])

plt.show()