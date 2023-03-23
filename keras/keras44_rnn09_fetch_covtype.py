from sklearn.datasets import fetch_covtype
import numpy as np
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, GRU, SimpleRNN, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
datasets = fetch_covtype()

x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(581012, 54) (581012,)
print('y의 라벨값:',np.unique(y)) #[1 2 3 4 5 6 7]
print(y)

#keras 카테고리컬
y = to_categorical(y)
print(y.shape)       #(581012, 8) 
y = np.delete(y,0,axis=1)   #y의 열에서 0번째 행을 뺀다
print(y)   #(581012, 7)



x_train, x_test, y_train, y_test = train_test_split(x,y, random_state =1234, 
                                             shuffle = True, train_size = 0.8)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test))

print(x_train.shape) #(464809, 54)
print(x_test.shape) #(116203, 54)
x_train =x_train.reshape(464809,9,6)
x_test = x_test.reshape(116203,9,6)


model = Sequential()
model.add(LSTM(16, input_shape = (9,6), return_sequences=True))
model.add(GRU(32,return_sequences=True))
model.add(SimpleRNN(16))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(7, activation='softmax'))

#컴파일 훈련
model.compile(loss='categorical_crossentropy' #sparse_categorical_crossentropy 
              , optimizer='adam', metrics=['acc']) 
es=EarlyStopping(monitor='acc', patience=10, mode='max',verbose=1, restore_best_weights=True)

import time
start_time = time.time()

model.fit(x_train, y_train, batch_size=128, epochs=1, validation_split=0.2,verbose=1, callbacks=[es]) #배치가 크면 터진다

end_time = time.time()

#평가 예측
results = model.evaluate(x_test, y_test)
print('results:', results)
print('걸린시간:', round(end_time - start_time,2))    #훈련시키는거에는 np.round 아무거나엔 그냥 round



y_predict = model.predict(x_test)
print(y_predict.shape)
y_test_acc = np.argmax(y_test, axis=1)
print(y_test_acc)
y_predict_acc = np.argmax(y_predict, axis=1)


acc = accuracy_score(y_test_acc, y_predict_acc)
print('acc:', acc)