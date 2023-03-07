from sklearn.datasets import load_diabetes

#1 데이터
datasets = load_diabetes()
x= datasets.data
y= datasets.target

# print(x.shape, y.shape) #(442, 10) (442, )

#실습
# R2 0.62이상
#train_size 0.7이상 0.9이하

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.9, random_state=131, shuffle=True)

model=Sequential()
model.add(Dense(12, input_dim=10))
model.add(Dense(15))
model.add(Dense(17))
model.add(Dense(20))
model.add(Dense(21))
model.add(Dense(15))
model.add(Dense(13))
model.add(Dense(8))
model.add(Dense(1))

#컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100 , batch_size=4, validation_split=0.2)

#평가 예측
loss=model.evaluate(x_test, y_test)
print('loss=', loss)

y_predict=model.predict(x_test) #x를 넣어서 결과를 보는건 되지만  x중 70퍼는 이미 훈련시킨 데이터

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) #훈련시키지 않은 데이터로 평가예측

print('r2스코어 :', r2)