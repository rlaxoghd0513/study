import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1 데이터
x=np.array([range(10)])  #range() 아래 숫자까지 
print(x.shape)  #몇행몇열인지
x=x.T  

y=np.array([[1,2,3,4,5,6,7,8,9,10],
            [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
            [9,8,7,6,5,4,3,2,1,0]]) #(3,10)
y=y.T  #(10,3)

# 모델구성
model=Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))

#컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x,y, epochs=200, batch_size=10)
# 수집한 데이터를 나누어 훈련에 사용하고 그 나머지 데이터로 평가,예측을 한다

#평가, 예측
loss=model.evaluate(x,y)
print('loss=', loss)

result=model.predict([[9]])
print('[[9]]의 결과값=', result)
