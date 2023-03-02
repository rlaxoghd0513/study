import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1데이터         #행무시,열우선   열은 특성, 행은 데이터 개수
x=np.array(
    [[1,2,3,4,5,6,7,8,9,10],
     [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]]
)
y=np.array([11,12,13,14,15,16,17,18,19,20])


#(2,10)을 (10,2)로 바꾸세요

# x=x.transpose()   #전치행렬 가로와 세로를 바꾸는것 둘중에 하나쓰면 됨
x=x.T


print(x.shape)  #(10,2) -> 2개의 특성을 가진 10개의 데이터
print(y.shape)  #(10, )

   
#2 모델구성
model=Sequential()
model.add(Dense(3, input_dim=2))  # x 특성이 몇개냐
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss='mae',optimizer='adam')
model.fit(x,y, epochs=100, batch_size=3)

#4 평가, 예측
loss=model.evaluate(x,y)
print('loss :',loss)

result=model.predict([[10, 1.4]])
print('[10, 1.4] :의 예측값', result)
