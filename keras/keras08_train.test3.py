import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 데이터
x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([1,2,3,4,5,6,7,8,9,10])

#[검색] train과 test를 섞어서 7:3으로 찾을 수 있는 방법
#힌트 사이킷런
from sklearn.model_selection import train_test_split
                                                                                         #랜덤스테이트 뒤에 써진 숫자에 맞는 랜덤값 표출 랜덤으로 섞지만 한번 나온 데이터가 안바뀌게
x_train, x_test, y_train, y_test = train_test_split( x,y, train_size=0.7, test_size=0.3, random_state=1998, shuffle=True) #shuffle false로 하면 섞지 않고 순서대로나온다

print(x_train)
print(x_test)

#모델구성
model=Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(3))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=10)

#평가 예측
loss=model.evaluate(x_test, y_test)
print('loss=', loss)

result=model.predict([11])
print('[11]의 결과값=', result)

#결과값 [[10.959655]] mae epochs 100 baych 10 1356421
