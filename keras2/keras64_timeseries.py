import numpy as np
# import autokeras as ak
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# 1. 데이터
data = fetch_california_housing()
x = data.data
y = data.target

# 1-1 train test 분할
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42,shuffle= True)
#2. 모델

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64, input_dim = 8))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam
learning_rate = 0.1
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss = 'mse', optimizer = optimizer)
model.fit(x_train,y_train, epochs =100,batch_size=32)

#4. 평가, 예측
results = model.evaluate(x_test, y_test)

print("lr : ", learning_rate, "loss : ", results)


# lr : 0.01 , loss :  0.6585977673530579