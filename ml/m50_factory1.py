import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import time
from sklearn.metrics import mean_absolute_error, r2_score
from catboost import CatBoostRegressor


path = './_data/ai_spark/'  #/ // \ \\ 같다. 텍스트에서 /n은 줄바꾸기 \a는 ? 이런 예약어 되는거 빼고는 적용된다
# train
# train_aws
# test_input
# test_aws
# meta
# answer_sample

train_files = glob.glob(path + 'TRAIN/*.csv') #glob 이 폴더안에 들어있는 모든 데이터를 가져와서 텍스트화 시켜준다
print(train_files)
train_aws_files = glob.glob(path + 'TRAIN_AWS/*.csv')#경로에서는 대문자 소문자 상관없다
print(train_aws_files)
test_input_files = glob.glob(path + 'TEST_INPUT/*.csv')
print(test_input_files)
test_aws_files = glob.glob(path + 'TEST_AWS/*.csv')
print(test_aws_files)
meta_files = glob.glob(path+'meta/*.csv')
print(meta_files)

##################### train 폴더 ########################################
li = []
for filename in train_files:
    df = pd.read_csv(filename, index_col=None, header=0, 
                     encoding = 'utf-8-sig')#한글깨지는거해결
    #여기까지만 하면 마지막 홍성읍 데이터만 df에 들어가기때문에 li만들어서 append해준다
    li.append(df)
print(li)
print(len(li)) #17 한개의 데이터로 만들어야되는데 이건 지금 17개의 데이터가 쭉 나열된거 뿐이다

train_dataset = pd.concat(li, axis=0, ignore_index = True) #행단위로 합친다, 인덱스가 각각 3만몇개씩 찍혀있는데 합쳐진걸로 새로 생성이 된다
print(train_dataset) #[596088 rows x 4 columns]

##################### test_input 폴더 ##################################
li = []
for filename in test_input_files:
    df = pd.read_csv(filename, index_col=None, header=0, 
                     encoding = 'utf-8-sig')
    li.append(df)
print(li)
print(len(li)) #17 한개의 데이터로 만들어야되는데 이건 지금 17개의 데이터가 쭉 나열된거 뿐이다

test_input_dataset = pd.concat(li, axis=0, ignore_index = True) #행단위로 합친다, 인덱스가 각각 3만몇개씩 찍혀있는데 합쳐진걸로 새로 생성이 된다
print(test_input_dataset) #[131376 rows x 4 columns]

######################## 측정소 라벨 인코딩 ####################################
le = LabelEncoder()
train_dataset['locate'] = le.fit_transform(train_dataset['측정소']) #칼럼 locate를 만들어서 라벨인코딩 한 값을 넣어준다
test_input_dataset['locate'] = le.transform(test_input_dataset['측정소'])

print(train_dataset) #[596088 rows x 5 columns]
print(test_input_dataset) #[131376 rows x 5 columns]

train_dataset = train_dataset.drop(['측정소'], axis=1)
test_input_dataset = test_input_dataset.drop(['측정소'], axis=1)
print(train_dataset)        #[596088 rows x 4 columns]
print(test_input_dataset)   #[131376 rows x 4 columns]

######################## 일시 -> 월, 시간 분리 !! ###################################
# 12-31 21:00 -> 12와 21 추출
print(train_dataset.info()) #  1 일시 596088 non-null  object
train_dataset['month'] = train_dataset['일시'].str[:2] #두번째까지 뽑겠다
print(train_dataset['month'])
train_dataset['time'] = train_dataset['일시'].str[6:8]
print(train_dataset['time'])

print(train_dataset) #[596088 rows x 6 columns]
train_dataset = train_dataset.drop(['일시'],axis=1)
print(train_dataset) #[596088 rows x 5 columns]

print(train_dataset.info()) #month칼럼과 time칼럼 object 문자열

######################## str -> int #################################################
# train_dataset['month'] = train_dataset['month'].astype('int32')
train_dataset['time'] = pd.to_numeric(train_dataset['time']) # 수치형으로 바꿔준다
train_dataset['month'] = pd.to_numeric(train_dataset['month']).astype('int16') #이렇게도 되는데 장점은 예를 들어 int8로 하면 연산하는 메모리가 줄어든다 메모리터짐방지 
print(train_dataset.info())

######################## test_input_dataset 일시 -> 월, 시간 분리 !! ###################################
# 12-31 21:00 -> 12와 21 추출
print(test_input_dataset.info()) #  1 일시 596088 non-null  object
test_input_dataset['month'] = test_input_dataset['일시'].str[:2] #두번째까지 뽑겠다
print(test_input_dataset['month'])
test_input_dataset['time'] = test_input_dataset['일시'].str[6:8]
print(test_input_dataset['time'])

print(test_input_dataset) #[596088 rows x 6 columns]
test_input_dataset = test_input_dataset.drop(['일시'],axis=1)
print(test_input_dataset) #[596088 rows x 5 columns]

print(test_input_dataset.info()) #month칼럼과 time칼럼 object 문자열

######################## test_input_dataset str -> int #################################################
# test_input_dataset['month'] = test_input_dataset['month'].astype('int32')
test_input_dataset['time'] = pd.to_numeric(test_input_dataset['time']) # 수치형으로 바꿔준다
test_input_dataset['month'] = pd.to_numeric(test_input_dataset['month']).astype('int16') #이렇게도 되는데 장점은 예를 들어 int8로 하면 연산하는 메모리가 줄어든다 메모리터짐방지 
print(test_input_dataset.info())

####################결측치제거 pm25 에 15542개 있다 #################################
#전체 596085 -> 580546 으로 줄인다
train_dataset = train_dataset.dropna()
print(train_dataset.info())
print(test_input_dataset.info())

######################## 파생 피쳐 생각해볼것 ########################################


y = train_dataset['PM2.5']
x = train_dataset.drop(['PM2.5'], axis=1)

print(x, '\n', y)

x_train,x_test, y_train, y_test = train_test_split(x,y, random_state = 333, shuffle=True, train_size = 0.8)

parameter ={'n_estimators' : 100000,
            'learning_rate' : 0.07, #일반적으로 가장 성능에 영향을 많이 끼침. 경사하강법에서 얼만큼씩 하강할것이냐를 뜻함. 웨이트를 찾을때 적절한 러닝레이트 필요
            'max_depth' : 2, #트리형 모델의 깊이.
            'gamma' : 0,
            'min_child_weight' : 0, 
            'subsample' : 0.2, # 드랍아웃의 개념. 0.2만큼 덜어낸다는 의미
            'colsample_bytree' : 0.5,
            'colsample_bylevel': 0,
            'reg_lambda': 1,
            'random_state': 333,
            'n_jobs': -1
            }

#2 모델구성
model = XGBRegressor()

#3 컴파일 훈련
model.set_params(**parameter, eval_metric = 'mae', early_stopping_rounds = 300) # model.compile이라고 생각하면 된다

start_time = time.time()

model.fit(x_train, y_train,verbose=1, eval_set = [(x_train,y_train),(x_test,y_test)])

end_time = time.time()
print('걸린시간:', round(end_time - start_time,2), '초')

#4 평가 예측
y_predict = model.predict(x_test)
result = model.score(x_test, y_test)
print('model.score:', result)

r2 = r2_score(y_test, y_predict)
print('r2:', r2)

mae = mean_absolute_error(y_test, y_predict)
print('mae:', mae)

####################answer는 test 결측치만 추출##################
missing_rows = test_input_dataset[test_input_dataset.isnull().any(axis=1)]
print(missing_rows.shape)#(78336, 5)
print(missing_rows.info())

x_test_miss = missing_rows.drop(['PM2.5'], axis=1)

y_submit = model.predict(x_test_miss)
# print(y_submit)

submission = pd.read_csv(path+'answer_sample.csv',index_col=None, header=0, 
                     encoding = 'utf-8-sig')
                    

submission['PM2.5'] = y_submit


path_save = './_save/ai_spark/'

import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

submission.to_csv(path_save +str(round(mae, 5))+ date +'.csv', index = None) #csv로 내보내기



