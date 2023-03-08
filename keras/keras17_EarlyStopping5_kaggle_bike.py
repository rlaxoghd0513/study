import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#데이터
path = './_data/kaggle_bike/'
path_save = './_save/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)

print(train_csv) #(10886,11)
print(train_csv.shape)

test_csv = pd.read_csv(path + 'test.csv', index_col=0) 
#index_col= 0번째부터 세고 읽는거에서 뺀다

print(test_csv)
print(test_csv.shape)  #(6493, 8)

print(train_csv.columns)
#(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
    #    'humidity', 'windspeed', 'casual', 'registered', 'count'],
    #   dtype='object')
print(test_csv.columns)
#(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
    #    'humidity', 'windspeed'],
    #   dtype='object')

x = train_csv.drop(['casual','registered','count'], axis=1)

y= train_csv['count']



x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, random_state=1234, train_size=0.7)
print(x_train.shape, x_test.shape) #(7620, 8), (3266,8)
print(y_train.shape, y_test.shape) #(7620, ), (3266, )

#모델구성
#한정화함수 다음레이어로 전하는걸 한정시킨다
#relu 양수는 양수로 음수는 0으로 최종레이어에는 잘 쓰지 않는다
#linear 있으나마나 
model=Sequential()
model.add(Dense(10, input_dim=8))
model.add(Dense(12))
model.add(Dense(15))
model.add(Dense(20, activation = 'linear'))#디폴트값
model.add(Dense(18, activation = 'relu'))
model.add(Dense(14, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1))

#컴파일 훈련
model.compile(loss='mse', optimizer= 'adam')
from tensorflow.python.keras.callbacks import EarlyStopping
es= EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1, restore_best_weights=True)
hist =model.fit(x_train, y_train, batch_size=32, epochs=200, verbose=1,validation_split=0.2, callbacks=[es])
print(hist.history['val_loss'])


#평가 예측
loss= model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)

r2= r2_score(y_test, y_predict)
print('r2스코어:', r2)

#rmse만들기
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
rmse = RMSE(y_test, y_predict)
print('RMSE:', rmse)


import matplotlib.pyplot as plt
plt.rcParams['font.family']='Malgun Gothic'

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red', label='로스')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='발_로스')

plt.title('캐글자전거')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.legend()
plt.grid()
plt.show()


#submit

y_submit = model.predict(test_csv)
print(y_submit)

submission = pd.read_csv(path + 'samplesubmission.csv', index_col =0)

print(submission)
submission['count'] = y_submit
print(submission)

submission.to_csv(path_save + 'submit_0308_1755.csv')