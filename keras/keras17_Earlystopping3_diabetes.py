from sklearn.datasets import load_diabetes

#1 데이터
datasets = load_diabetes()
x= datasets.data
y= datasets.target


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=131, shuffle=True)

model=Sequential()
model.add(Dense(12, input_dim=10))
model.add(Dense(15))
model.add(Dense(17))
model.add(Dense(20))
model.add(Dense(21))
model.add(Dense(15))
model.add(Dense(13))
model.add(Dense(8))
model.add(Dense(1))

#컴파일 훈련
model.compile(loss='mse', optimizer='adam')
es=EarlyStopping(monitor='val_loss', patience = 20, mode='min'
              ,verbose=1
              ,restore_best_weights=True) 
hist = model.fit(x_train, y_train, epochs=100 , batch_size=4, validation_split=0.2, callbacks=[es])
print(hist.history['val_loss'])

#평가 예측
loss=model.evaluate(x_test, y_test)
print('loss=', loss)

y_predict=model.predict(x_test) #x를 넣어서 결과를 보는건 되지만  x중 70퍼는 이미 훈련시킨 데이터

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) #훈련시키지 않은 데이터로 평가예측

print('r2스코어 :', r2)

import matplotlib.pyplot as plt
# matplotlib.rcParams['font.family'] = 'Malgun Gothic' #가급이면 나눔체
plt.rcParams['font.family'] = 'Malgun Gothic' #한글 오류 뜨는거 해결법


         
plt.figure(figsize=(9, 6))
# plt.plot(x, y)  #x는 명시하지 않아도 된다
plt.plot(hist.history['loss'], marker = '.', c= 'red', label = '로스') #label 오른쪽 위에 라벨
plt.plot(hist.history['val_loss'],marker = '.', c='blue', label = '발_로스')

# #이름지어주기
plt.title('당뇨')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.legend() #라벨값 화면 표시
plt.grid() #알아보기쉽게 격자 넣기
plt.show()