from tensorflow.keras.datasets import cifar10
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout,MaxPooling2D
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

#1 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# print(x_train.shape, y_train.shape)  #(50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape)    #(10000, 32, 32, 3) (10000, 1)

x_train = x_train/255.
x_test = x_test/255.

print(np.max(x_train),np.min(x_train)) # 스케일러가 2차원만 받는다

print(np.unique(y_train, return_counts = True)) #(0,1,2,3,4,5,6,7,8,9)


from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)

y_test = to_categorical(y_test)


#2 모델구성
model = Sequential()
model.add(Conv2D(8, (3,3), padding='same', input_shape=(32,32,3)))
model.add(Conv2D(8,2))
model.add(MaxPooling2D())    #커널처럼 중첩되지 않고 각 구역에서 가장 큰 애만 뽑는다
model.summary()
model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(Conv2D(32,3, padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(128, 3, padding='same')) #대부분 2,2 3,3으로 커널사이즈를 하니까 귀찮아서 2,2를 2만 써도 된다
model.add(Conv2D(128 ,3))
model.add(Flatten())
model.add(Dense(32))
model.add(Dropout(0.3))
model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(16))
model.add(Dense(10, activation = 'softmax'))
model.summary()

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience=5, verbose=1, restore_best_weights=True)

filepath = './_save/MCP/cifar10/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
import datetime
date = datetime.datetime.now()  #현재시간을 date에 넣어준다
print(date)  
date = date.strftime("%m%d_%H%M") #시간을 문자데이터로 바꾸겠다 그래야 파일명에 넣을 수 있다    %뒤에있는값을 반환해달라
print(date)  

mcp = ModelCheckpoint(monitor = 'val_loss', mode = 'min', verbose= 1, save_best_only=True, filepath="".join[filepath,'cifar10_',date,'_',filename])
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
#acc: 0.623
# 0.6329
