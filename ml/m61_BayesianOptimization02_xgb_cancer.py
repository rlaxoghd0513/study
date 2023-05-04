from bayes_opt import BayesianOptimization
from xgboost import XGBClassifier
import numpy as np
from sklearn.datasets import load_breast_cancer, fetch_covtype
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')
import time


#1데이터
data_list = [load_breast_cancer, fetch_covtype]
data_list_name = ['캔서','콥타입']
for i,value in enumerate(data_list):
    
    x,y = load_breast_cancer(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=333, shuffle=True, train_size=0.8)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    #2모델
    bayesian_params = {
        'max_depth':(4,10),
        'learning_rate':(0.001,3),
        'subsample':(0.5,1),
        'colsample_bytree':(0.5,1)}
    
#목적함수 준비하기(정의) 파라미터의 범위, 
    def lgb_hamsu(max_depth, learning_rate, subsample, colsample_bytree):
        params = {
            'n_estimators':10000,
            'learning_rate':learning_rate,
            'max_depth':int(round(max_depth)),
            'subsample':max(min(subsample,1),0),              #드랍아웃과 비슷한 개념 1보다 작고 0보다 커야한다
            'colsample_bytree':colsample_bytree
        }
    
        model = XGBClassifier(**params) #한꺼번에 넣는다
        model.fit(x_train, y_train,
                  eval_set = [(x_train,y_train),(x_test,y_test)],
                  eval_metric = 'error',
                  verbose=0,
                  early_stopping_rounds=50)
        
        y_predict = model.predict(x_test)
        result = accuracy_score(y_test,y_predict)

        return result

    lgb_bo = BayesianOptimization(f = lgb_hamsu,
                                  pbounds = bayesian_params, #pbounds에 있는걸 위에 f=lgb_hamsu에 넣어라
                                  random_state = 333)
    start_time = time.time()
    
    n_iter = 100
    lgb_bo.maximize(init_points=5, n_iter=n_iter) #init_points 초기 포인트 찍고 여기서부터 n_iter가 시작된다
    end_time = time.time()
    print(data_list_name[i])
    print(lgb_bo.max)
    print(n_iter,'번 걸린시간:', end_time-start_time)