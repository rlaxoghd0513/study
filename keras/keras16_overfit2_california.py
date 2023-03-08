from sklearn.datasets import fetch_california_housing

#1.  데이터
datasets = fetch_california_housing()
x= datasets.data
y=datasets.target

# print(x.shape, y.shape) #(20649, 8) (20640, )


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

x_train, x_test, y_train,y_test=train_test_split(x,y,random_state=175 ,train_size=0.9, shuffle=True)

model=Sequential()
model.add(Dense(10, input_dim=8))
model.add(Dense(15))
model.add(Dense(20))
model.add(Dense(25))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)
print(hist.history)

#그래프그리기
import matplotlib.pyplot as plt
# plt.plot(x, y)  #x는 명시하지 않아도 된다
plt.plot(hist.history['loss'])
plt.show()


