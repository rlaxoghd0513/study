from tensorflow.keras.datasets import cifar100
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout,MaxPooling2D
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

#1 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

x_train = x_train/255.    #(50000, 32, 32, 3) (50000, 1)

y_test = x_test/255.   #(10000, 32, 32, 3) (10000, 1)

print(np.min(x_test), np.max(x_test))   # 0 255
print(np.unique(y_train, return_counts=True))    #0부터 99까지

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#모델구성
model = Sequential()
model.add(Conv2D(8,(3,3), input_shape = (32,32,3), padding='same'))
model.add(Conv2D(8,3, padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(32,(3,3), padding='same'))
model.add(Conv2D(32,2, padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(32,2, padding='same'))
model.add(Conv2D(32,3))
model.add(Flatten())
model.summary()
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64))
model.add(Dropout(0.4))
model.add(Dense(32))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(100, activation='softmax'))

#컴파일 훈련
model.compile(loss = 'categorical_crossentropy', optimizer= 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 30, batch_size = 128) #배치사이즈 128이상 터짐 


#4 평가 예측
results = model.evaluate(x_test, y_test)
print('results:', results)


y_predict = model.predict(x_test)
y_test_acc = np.argmax(y_test, axis=1)
y_predict_acc = np.argmax(y_predict, axis=1)

acc=accuracy_score(y_test_acc, y_predict_acc)
print('acc:',acc)