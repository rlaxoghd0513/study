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
from sklearn.metrics import r2_score, mean_squared_error

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
from tensorflow.python.keras.callbacks import EarlyStopping
es=EarlyStopping(monitor='val_loss', patience = 20, mode='min'
              ,verbose=1
              ,restore_best_weights=True) 
hist = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1, callbacks=[es])
print(hist.history['val_loss'])

loss= model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어:', r2)


import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9, 6))
plt.plot(hist.history['loss'], marker = '.', c= 'red', label = '로스') #label 오른쪽 위에 라벨
plt.plot(hist.history['val_loss'],marker = '.', c='blue', label = '발_로스')

# #이름지어주기
plt.title('캘리포니아')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.legend() #라벨값 화면 표시
plt.grid() #알아보기쉽게 격자 넣기
plt.show()

