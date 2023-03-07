import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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

print(x.shape) #(10886, 8)
print(y.shape) #(10886, )

print(train_csv.info())
#  0   season      10886 non-null  int64
#  1   holiday     10886 non-null  int64
#  2   workingday  10886 non-null  int64
#  3   weather     10886 non-null  int64
#  4   temp        10886 non-null  float64
#  5   atemp       10886 non-null  float64
#  6   humidity    10886 non-null  int64
#  7   windspeed   10886 non-null  float64
#  8   casual      10886 non-null  int64
#  9   registered  10886 non-null  int64
#  10  count       10886 non-null  int64


x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, random_state=3333, train_size=0.7)
print(x_train.shape, x_test.shape) #(7620, 8), (3266,8)
print(y_train.shape, y_test.shape) #(7620, ), (3266, )

#모델구성
#한정화함수 다음레이어로 전하는걸 한정시킨다
#relu 양수는 양수로 음수는 0으로 최종레이어에는 잘 쓰지 않는다
#linear 있으나마나 
model=Sequential()
model.add(Dense(20, input_dim=8))
model.add(Dense(38))
model.add(Dense(52))
model.add(Dense(77, activation = 'linear'))#디폴트값
model.add(Dense(54, activation = 'relu'))
model.add(Dense(22, activation = 'relu'))
model.add(Dense(11, activation = 'relu'))
model.add(Dense(1))

#컴파일 훈련
model.compile(loss='mse', optimizer= 'adam')
model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=1,validation_split=0.2)

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

#submit

y_submit = model.predict(test_csv)
print(y_submit)

submission = pd.read_csv(path + 'samplesubmission.csv', index_col =0)

print(submission)
submission['count'] = y_submit
print(submission)

submission.to_csv(path_save + 'submit_0307_1440.csv')