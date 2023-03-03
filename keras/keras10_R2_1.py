from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1 데이터
x=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y=np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train, x_test, y_train, y_test=train_test_split(x,y, 
                                                  train_size=0.7, shuffle=True, random_state=1234)
#x가 x_train, x_test y가 y_train, y_test로 분리된다

#모델구성
model=Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(5))
model.add(Dense(11))
model.add(Dense(13))
model.add(Dense(10))
model.add(Dense(9))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(1))

#컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=150, batch_size=32)

#평가 예측
loss=model.evaluate(x_test, y_test)
print('loss=', loss)

y_predict=model.predict(x_test) #x를 넣어서 결과를 보는건 되지만  x중 70퍼는 이미 훈련시킨 데이터

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) #훈련시키지 않은 데이터로 평가예측

print('r2스코어 :', r2)
