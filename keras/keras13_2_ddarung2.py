#데이콘 따릉이 문제풀이
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error #루트씌우기함수
import pandas as pd #가져온 데이터를 파이썬에서 이용할 때 사용


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

x_train, x_test, y_train, y_test = train_test_split(x,y,shuffle=True, random_state=2033, train_size=0.8)
print(x_train.shape, x_test.shape) #(1021,9) (438,9)   (929, 9) (399, 9)
print(y_train.shape, y_test.shape) #(1021,) (438,)     (929,) (399,)

#모델구성
model=Sequential()
model.add(Dense(13, input_dim=9))           #nan과 0은 다르다  
model.add(Dense(28))
model.add(Dense(45))
model.add(Dense(77))
model.add(Dense(101))
model.add(Dense(87))
model.add(Dense(62))
model.add(Dense(45))
model.add(Dense(27))
model.add(Dense(12))
model.add(Dense(1))

#컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=16, verbose=0)

#평가 예측
loss=model.evaluate(x_test, y_test)
print('loss=', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)

#rmse만들기 rmse라는 함수를 만들거야
def RMSE(y_test, y_predict):     #함수정의
    return np.sqrt(mean_squared_error(y_test, y_predict))  #np.sqrt 루트씌우기
rmse = RMSE(y_test, y_predict)    #RMSE 함수 사용
print("RMSE :", rmse)

########submission.csv를 만들어봅시다###### 
# print(test_csv.isnull().sum())  여기도 결측치가 있음

y_submit = model.predict(test_csv) #submit 제출
print(y_submit)

submission = pd.read_csv(path+'submission.csv',index_col=0)
                    
print(submission)
submission['count'] = y_submit
print(submission)

submission.to_csv(path_save+'submit_0306_0559.csv') #csv로 내보내기







