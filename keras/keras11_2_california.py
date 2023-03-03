from sklearn.datasets import fetch_california_housing

#1.  데이터
datasets = fetch_california_housing()
x= datasets.data
y=datasets.target

# print(x.shape, y.shape) #(20649, 8) (20640, )

#[실습]
#R2 0.55~0.6이상

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

x_train, x_test, y_train,y_test=train_test_split(x,y,random_state=150,train_size=0.7, shuffle=True)

model=Sequential()
model.add(Dense(16, input_dim=8))
model.add(Dense(30))
model.add(Dense(36))
model.add(Dense(41))
model.add(Dense(36))
model.add(Dense(33))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=100, batch_size=20649)

loss=model.evaluate(x_test,y_test)
print('loss=', loss)

y_predict=model.predict(x_test) #x를 넣어서 결과를 보는건 되지만  x중 70퍼는 이미 훈련시킨 데이터

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) #훈련시키지 않은 데이터로 평가예측

print('r2스코어 :', r2)

