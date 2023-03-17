from tensorflow.keras.datasets import cifar100
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout,MaxPooling2D
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
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64))
model.add(Dropout(0.4))
model.add(Dense(32))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(100, activation='softmax'))

#컴파일 훈련
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
filepath = './_save/MCP/cifar100/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
import datetime
date = datetime.datetime.now()  #현재시간을 date에 넣어준다
date = date.strftime("%m%d_%H%M") #시간을 문자데이터로 바꾸겠다 그래야 파일명에 넣을 수 있다    %뒤에있는값을 반환해달라
es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience=5, verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor = 'val_loss', mode = 'min', verbose= 1, save_best_only=True, filepath="".join[(filepath,'cifar100_',date,'_',filename)])

 
model.compile(loss = 'categorical_crossentropy', optimizer= 'adam', metrics = ['acc'])

import time
start_time = time.time()

model.fit(x_train, y_train, epochs = 1000, batch_size = 320, callbacks = [es,mcp], validation_split=0.2)
end_time = time.time()

#4 평가 예측
results = model.evaluate(x_test, y_test)
print('results:', results)
print('걸린시간:', round(end_time - start_time,2))

y_predict = model.predict(x_test)
y_test_acc = np.argmax(y_test, axis=1)
y_predict_acc = np.argmax(y_predict, axis=1)

acc=accuracy_score(y_test_acc, y_predict_acc)
print('acc:',acc)