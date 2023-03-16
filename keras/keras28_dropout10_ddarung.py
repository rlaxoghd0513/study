#데이콘 따릉이 문제풀이
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error #루트씌우기함수
import pandas as pd #가져온 데이터를 파이썬에서 이용할 때 사용
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, MinMaxScaler, StandardScaler


#데이터
path = './_data/ddarung/' #.하나는 현재폴더(작업폴더) 현재는 study 폴더, /은 밑에
path_save= './_save/ddarung/'


#원래는 이렇게 써야됨 문자+문자는 합쳐짐
# train_csv = pd.read_csv('./_data/ddarung/train.csv')
train_csv = pd.read_csv(path + 'train.csv', index_col=0) #id컬럼은 뺀다, 컬럼명(헤더)은 자동으로 빠짐????????????????????????????


print(train_csv)
print(train_csv.shape)   #(1459,10)
#id는 몇갠지만 알려주는거라서 데이터가 아니라서  칼럼에 포함 x

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv)
print(test_csv.shape)   #(715,9)

#==============================================================
print(train_csv.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#       'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#      'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#      dtype='object')
print(train_csv.info()) #데이터 자료형
#  0   hour                    1459 non-null   int64
#  1   hour_bef_temperature    1457 non-null   float64
#  2   hour_bef_precipitation  1457 non-null   float64
#  3   hour_bef_windspeed      1450 non-null   float64
#  4   hour_bef_humidity       1457 non-null   float64
#  5   hour_bef_visibility     1457 non-null   float64
#  6   hour_bef_ozone          1383 non-null   float64
#  7   hour_bef_pm10           1369 non-null   float64
#  8   hour_bef_pm2.5          1342 non-null   float64
#  9   count                   1459 non-null   float64

print(train_csv.describe()) #mean 평균 std 표준편차 min 최솟값 max 최대값 

print(type(train_csv)) # <class 'pandas.core.frame.dataframe' >

#===========결측치 처리=================
#통데이터일때 결측치 처리안하면 데이터 삐꾸됨
# 결측치처리 1. 제거
# print(train_csv.isnull())
print(train_csv.isnull().sum()) #결측치의 합계 

train_csv = train_csv.dropna() #dropna 결측치삭제
print(train_csv.isnull().sum())
print(train_csv.info())
print(train_csv.shape)

#==================================train_csv데이터에서 x와 y를 분리================================
x = train_csv.drop(['count'], axis=1) #????? [,] 두개이상은 리스트
# axis=0은 행을 기준으로 동작하는 것이고 axis=1은 열을 기준으로 동작하는 것
print(x)

y = train_csv['count']
print(y) #pandas 데이터분리형태
#==================================train_csv데이터에서 x와 y를 분리================================

x_train, x_test, y_train, y_test = train_test_split(x,y,shuffle=True, random_state=1357, train_size=0.9)
print(x_train.shape, x_test.shape) #(1021,9) (438,9)   (929, 9) (399, 9)
print(y_train.shape, y_test.shape) #(1021,) (438,)     (929,) (399,)

scaler= RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test))

test_csv = scaler.transform(test_csv)


input1 = Input(shape=(9,))
dense1 = Dense(10)(input1)
dense2 = Dense(12)(dense1)
drop2 = Dropout(0.4)(dense2)
dense3 = Dense(14)(drop2)
drop3 = Dropout(0.4)(dense3)
dense4 = Dense(12)(drop3)
dense5 = Dense(10, activation = 'relu')(dense4)
drop5 = Dropout(0.2)(dense5)
dense6 = Dense(7, activation = 'relu')(drop5)
output1 = Dense(1)(dense6)
model = Model(inputs = input1, outputs = output1)
#컴파일, 훈련

model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()  
date = date.strftime("%m%d_%H%M") 
 

filepath = './_save/MCP/ddarung/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
es=EarlyStopping(monitor = 'val_loss', patience=40, mode='min', verbose=1,restore_best_weights=True)
mcp = ModelCheckpoint(monitor = 'val_loss', mode= 'auto', save_best_only=True, verbose=1, filepath ="".join([filepath,'dda_',date,'_',filename]))
hist = model.fit(x_train, y_train, epochs=10000, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es,mcp])

#평가 예측
loss=model.evaluate(x_test, y_test)
print('loss=', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)


def RMSE(y_test, y_predict):     
    return np.sqrt(mean_squared_error(y_test, y_predict))  
rmse = RMSE(y_test, y_predict)    
print("RMSE :", rmse)

# RMSE : 53.02481313651768

y_submit = model.predict(test_csv) 
# print(y_submit)

submission = pd.read_csv(path+'submission.csv',index_col=0)
                    
# print(submission)
submission['count'] = y_submit
# print(submission)

submission.to_csv(path_save+'ddarung_'+date+'.csv') 
# 0.62  53.11
#  0.6599120814531592
# RMSE : 50.73676437176139
# r2스코어 : 0.7453649781255187
# RMSE : 43.90219486868363