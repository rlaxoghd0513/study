#데이콘 따릉이 문제풀이
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, LSTM, SimpleRNN, GRU, Conv1D
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error #루트씌우기함수
import pandas as pd #가져온 데이터를 파이썬에서 이용할 때 사용
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor

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

n_splits = 5
kfold = KFold(n_splits = n_splits , shuffle=True, random_state=123 )  #5개로 나눈다 그러면 데이터는 20프로씩 나눠진다 데이터 모이는걸 방지하려고 shuffle
# kfold = KFold() #이렇게 해도 된다 디폴트값 있다

#2 모델구성
model = RandomForestRegressor()

#3,4 컴파일 훈련 평가 예측

scores = cross_val_score(model, x, y, cv = kfold, n_jobs = 4) #(모델, 데이터, 데이터, 크로스 발리데이션을 어떻게 할건지, cpu코어사용갯수)
# scores = cross_val_score(model, x, y, cv = 5) #이렇게 kfold를 따로 정의하지 않고도 가능하다
print(scores)

#[0.96666667 1.         0.93333333 0.93333333 0.9       ]
#다섯번 훈련을 시켰으니까 5개의 값이 나온다

print('acc:', scores, '\n cross_val_score 평균:', round(np.mean(scores),4))