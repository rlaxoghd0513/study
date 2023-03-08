from sklearn.datasets import load_boston
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
#1.데이터

datasets = load_boston()
x = datasets.data
y = datasets['target'] #x,y 구조 동일
print(x.shape, y.shape) #(506, 13) (506,)
x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, random_state=123, test_size=0.3)

#2.모델구성
model=Sequential()
model.add(Dense(10, input_dim=13, activation='sigmoid')) #sigmoid 0과 1사이 
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일 훈련
model.compile(loss='mse',optimizer='adam')

#얼리스타핑
from tensorflow.python.keras.callbacks import EarlyStopping
es=EarlyStopping(monitor='val_loss', patience = 20, mode='min'
              ,verbose=1
              ,restore_best_weights=True) 


hist = model.fit(x_train, y_train, epochs=2000, batch_size=16, validation_split=0.2, verbose=1, callbacks = [es])
# print(hist)#<tensorflow.python.keras.callbacks.History object at 0x000001F32CE1D3A0>
# print(hist.history) #{'loss': [112.98209381103516, 65.07829284667969, 65.4222412109375, 62.993106842041016,
# #                              60.105281829833984, 59.937808990478516, 57.980159759521484, 62.957942962646484,58.52788162231445, 57.778472900390625], 
# #                     'val_loss': [57.738582611083984, 58.2751579284668,  
# #                                59.8399543762207, 58.79970932006836, 57.46071243286133, 60.7735595703125, 59.76877975463867, 
# #                                59.819053649902344, 60.68880844116211, 61.87255859375]}  
# print(hist.history['loss'])
print(hist.history['val_loss'])

#평가예측
loss= model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어:', r2)


# #그래프그리기
import matplotlib.pyplot as plt
# matplotlib.rcParams['font.family'] = 'Malgun Gothic' #가급이면 나눔체
plt.rcParams['font.family'] = 'Malgun Gothic' #한글 오류 뜨는거 해결법


         
plt.figure(figsize=(9, 6))
# plt.plot(x, y)  #x는 명시하지 않아도 된다
plt.plot(hist.history['loss'], marker = '.', c= 'red', label = '로스') #label 오른쪽 위에 라벨
plt.plot(hist.history['val_loss'],marker = '.', c='blue', label = '발_로스')

# #이름지어주기
plt.title('보스톤')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.legend() #라벨값 화면 표시
plt.grid() #알아보기쉽게 격자 넣기
plt.show()
 