from sklearn.datasets import fetch_california_housing

#1.  데이터
datasets = fetch_california_housing()
x= datasets.data
y=datasets.target

# print(x.shape, y.shape) #(20649, 8) (20640, )


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

x_train, x_test, y_train,y_test=train_test_split(x,y,random_state=175 ,train_size=0.9, shuffle=True)

model=Sequential()
model.add(Dense(10, input_dim=8))
model.add(Dense(15))
model.add(Dense(20))
model.add(Dense(25))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=130, batch_size=27,validation_split=0.2)

loss=model.evaluate(x_test,y_test)
print('loss=', loss)

y_predict=model.predict(x_test) #x를 넣어서 결과를 보는건 되지만  x중 70퍼는 이미 훈련시킨 데이터

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) #훈련시키지 않은 데이터로 평가예측

print('r2스코어 :', r2)