from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler

#1.  데이터
datasets = fetch_california_housing()
x= datasets.data
y=datasets.target

# print(x.shape, y.shape) #(20649, 8) (20640, )


import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, MaxPooling2D, Flatten, LSTM, Conv1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

x_train, x_test, y_train,y_test=train_test_split(x,y,random_state=175 ,train_size=0.9, shuffle=True)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test))
print(x_train.shape)  #(18576, 8)
print(x_test.shape)  #(2064, 8)

x_train = x_train.reshape(18576,8,1)
x_test = x_test.reshape(2064,8,1)



input1 = Input(shape=(8,1))
conv1 = Conv1D(16,3,padding = 'same', strides = 2 )(input1)
conv2 = Conv1D(32,2, padding='valid', strides=1)(conv1)
conv3 = Conv1D(16, 2)(conv2)
flat = Flatten()(conv3)
dense1 = Dense(32)(flat)
dense2 = Dense(16)(dense1)
drop1 = Dropout(0.2)(dense2)
dense3 = Dense(8, activation = 'relu')(drop1)
output1 = Dense(1)(dense3)
model = Model(inputs = input1, outputs = output1)

model.compile(loss='mse', optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping
es=EarlyStopping(monitor='val_loss', patience = 20, mode='min'
              ,verbose=1
              ,restore_best_weights=True) 
hist = model.fit(x_train, y_train, epochs=1, batch_size=32, validation_split=0.2, verbose=1, callbacks=[es])
print(hist.history['val_loss'])

loss= model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test, verbose=0)
r2 = r2_score(y_test, y_predict)
print('r2스코어:', r2)