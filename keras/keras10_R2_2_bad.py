#1. R2를 음수가 아닌 0.5이하로 만들것
#2. 데이터는 건들지 말것
#3. 레이어는 인풋 아웃풋 포함 7개 이상
#4. batch_size=1
#5. 히든 레이어의 노드는 10개 이상 100개 이하
#6. train사이즈 75%
#7. epochs 100번 이상
#8. loss 지표는 mse, mae
# [실 습]


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1 데이터
x=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y=np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train, x_test, y_train, y_test=train_test_split(x,y, 
                                                  train_size=0.75, shuffle=True, random_state=15)
#x가 x_train, x_test y가 y_train, y_test로 분리된다

#모델구성
model=Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(13))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(13))
model.add(Dense(15))
model.add(Dense(1))

#컴파일 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=300, batch_size=1)

#평가 예측
loss=model.evaluate(x_test, y_test)
print('loss=', loss)

y_predict=model.predict(x_test) #x를 넣어서 결과를 보는건 되지만  x중 70퍼는 이미 훈련시킨 데이터

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) #훈련시키지 않은 데이터로 평가예측

print('r2스코어 :', r2)

#train 사이즈 변경 가능
#random