import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1 데이터
x=np.array([range(10), range(21,31), range(201,211)])  #range() 아래 숫자까지 
print(x.shape)  #(3,10) #몇행몇열인지
x=x.T  #(10,3)

y=np.array([[1,2,3,4,5,6,7,8,9,10]]) #(1,10)
y=y.T  #(10,1)

#2 모델구성
model=Sequential()
model.add(Dense(4, input_dim=3))  # x 특성이 몇개냐
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3 컴파일, 훈련
model.compile(loss='mae',optimizer='adam')
model.fit(x,y, epochs=200, batch_size=32)

#4 평가, 예측
loss=model.evaluate(x,y)
print('loss :',loss)

result=model.predict([[9, 30, 210]])
print('[9,30,210] :의 예측값', result)

#결과값 [[9.973509]] mse epochs 110 batch 10 3565431
  #    [[10.017415]] mae epochs 200 batch 32  4565421
