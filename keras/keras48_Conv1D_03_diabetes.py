from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

#1 데이터
datasets = load_diabetes()
x= datasets.data
y= datasets.target


import numpy as np
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Input,Dropout,Conv2D, MaxPool2D, Flatten, LSTM, Conv1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=131, shuffle=True)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test))
print(x_train.shape) #(353, 10)
print(x_test.shape) #(89, 10)

x_train = x_train.reshape(353,5,2)
x_test = x_test.reshape(89,5,2)


input1 = Input(shape=(5,2))
Conv1 = Conv1D(32,3, padding ='same')(input1)
conv2 = Conv1D(16,2, padding ='same')(Conv1)
conv3 = Conv1D(16,2)(conv2)
flat = Flatten()(conv3)
dense1 = Dense(16)(flat)
dense2 = Dense(32)(dense1)
dense3 = Dense(16)(dense2)
drop1 = Dropout(0.25)(dense3)
dense4 = Dense(8, activation='relu')(drop1)
output1 = Dense(1)(dense4)
model = Model(inputs = input1, outputs = output1)
 
#컴파일 훈련
model.compile(loss='mse', optimizer='adam')
es=EarlyStopping(monitor='val_loss', patience = 20, mode='min'
              ,verbose=1
              ,restore_best_weights=True) 
hist = model.fit(x_train, y_train, epochs=1 , batch_size=4, validation_split=0.2, callbacks=[es])
print(hist.history['val_loss'])

#평가 예측
loss=model.evaluate(x_test, y_test)
print('loss=', loss)

y_predict=model.predict(x_test, verbose =0) #x를 넣어서 결과를 보는건 되지만  x중 70퍼는 이미 훈련시킨 데이터

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) #훈련시키지 않은 데이터로 평가예측

print('r2스코어 :', r2)