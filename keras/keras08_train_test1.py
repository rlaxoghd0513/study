import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 데이터
x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([10,9,8,7,6,5,4,3,2,1])

# print(x)

x_train = np.array([1,2,3,4,5,6,7])
y_train = np.array([1,2,3,4,5,6,7])
x_test = np.array([8,9,10])
y_test = np.array([8,9,10])

#모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(1))


#컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=150, batch_size=2)

#평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss=',loss)

result=model.predict([11])
print('[11]결과값=', result)

#결과 [[10.965477]] mse epochs 100 batch 2
#     [[11.018163]] mae epochs 150 batch 2
#     [[11.000001]] mse epochs 150 batch 2  10 1
