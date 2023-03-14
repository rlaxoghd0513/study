from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler

#1.  데이터
datasets = fetch_california_housing()
x= datasets.data
y=datasets.target

# print(x.shape, y.shape) #(20649, 8) (20640, )


import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

x_train, x_test, y_train,y_test=train_test_split(x,y,random_state=175 ,train_size=0.9, shuffle=True)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test))



input1 = Input(shape=(8,))
dense1 = Dense(10)(input1) #sequential은 순서대로 함수형은 모델을 구성하고 시작이 어딘지 끝이 어딘지 마지막에 정해준다
dense2 = Dense(15)(dense1) #이 레이어가 어디서 왔는지 꽁다리에 
dense3 = Dense(20)(dense2)
drop1 = Dropout(0.3)(dense3)
dense4 = Dense(25)(drop1)
drop2 = Dropout(0.4)(dense4) 
dense5 = Dense(20)(drop2) 
dense6 = Dense(15)(dense5) 
drop3 = Dropout(0.2)(dense6)
dense7 = Dense(10)(drop3) 
dense8 = Dense(5)(dense7) 
output1 = Dense(1)(dense8) 
model = Model(inputs = input1, outputs = output1)

model.compile(loss='mse', optimizer='adam')
from tensorflow.python.keras.callbacks import EarlyStopping
es=EarlyStopping(monitor='val_loss', patience = 20, mode='min'
              ,verbose=1
              ,restore_best_weights=True) 
hist = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1, callbacks=[es])
print(hist.history['val_loss'])

loss= model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test, verbose=0)
r2 = r2_score(y_test, y_predict)
print('r2스코어:', r2)

# r2스코어: 0.6188119066708546
# loss: 0.5097762942314148

# loss: 0.5111734867095947
# r2스코어: 0.6177670345793483


