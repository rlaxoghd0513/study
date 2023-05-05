#회귀로 만들어라
#회귀데이터 올인 포문
#scaler 6개 올인 포문
#정규분포로 만들고, 분위수를 기준으로 0~1사이로 만들기때문에 이상치에 자유롭다
#이상치 많을때 Robust랑 Quantile 생각할수 있다

from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer #quantile  분위수  성능이 좋을수도 있고 나쁠수도 있다 모든값이 0에서 1사이로 바뀐다
from sklearn.preprocessing import PowerTransformer 
import numpy as np
# from sklearn.linear_model import LinearRegression, LogisticRegression #얘네 분류다 선형모델의 대표
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor                     #트리계열대표
from sklearn.metrics import r2_score, accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split

#1 데이터
data_list = [load_diabetes, fetch_california_housing]
data_name = ['디아벳','캘리포니아']
scaler_list = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler(),QuantileTransformer(), PowerTransformer()]
scaler_name = ['스탠다드', '민맥스', '맥스앱스', '로뷰스트','퀀틸','파워']
#PowerTransformer에서 method='yeo-johnson'은 음수값도 처리할수 있지만 'box-cox'는 양수값만 처리가능하다

for i,value in enumerate(data_list):
    
    x,y = value(return_X_y=True)
    x_train,x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=333, shuffle=True)
    for j,value1 in enumerate(scaler_list):
        scaler = value1
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        model = RandomForestRegressor()
        model.fit(x_train, y_train)
        print(data_name[i],scaler_name[j],'결과',round(model.score(x_test,y_test),4))
        
        
# 디아벳 스탠다드 결과 0.3845
# 디아벳 민맥스 결과 0.3922
# 디아벳 맥스앱스 결과 0.3817
# 디아벳 로뷰스트 결과 0.3978
# 디아벳 퀀틸 결과 0.3622
# 디아벳 파워 결과 0.3811

# 캘리포니아 스탠다드 결과 0.8008
# 캘리포니아 민맥스 결과 0.7996
# 캘리포니아 맥스앱스 결과 0.8021
# 캘리포니아 로뷰스트 결과 0.8011
# 캘리포니아 퀀틸 결과 0.8011
# 캘리포니아 파워 결과 0.8028