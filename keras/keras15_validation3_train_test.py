from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
x_train = np.array(range(1,17))
y_train = np.array(range(1,17))


#실습 ::: 잘라봐
#train_test_split으로만 잘라라
#10:3:3
x_train, x_test, y_train, y_test = train_test_split(x_train,y_train, train_size=0.625, random_state=1234)
x_test, x_val, y_test, y_val = train_test_split(x_test,y_test, test_size=0.5, random_state=1234)

print(x_train)
print(x_test)
print(x_val)

#2. 모델
model = Sequential()
model.add(Dense(3, activation='linear', input_dim =1))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1,
          validation_data=(x_val, y_val))

#4. 평가 예측
loss= model.evaluate(x_test, y_test)
print('loss:', loss)

result = model.predict([17])
print('17의 예측값:',result)