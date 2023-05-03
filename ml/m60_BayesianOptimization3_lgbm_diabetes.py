from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
import time


#1데이터
x,y = load_diabetes(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=333, shuffle=True, train_size=0.8)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2모델
bayesian_params = {
    'learning_rate':(0.001,0.3),
    'max_depth':(3,16),
    'num_leaves':(24,64),
    'min_child_samples':(10,200),
    'min_child_weight':(1,50),
    'subsample':(0.5,1),    #드랍아웃개념
    'colsample_bytree':(0.5,1),
    'max_bin':(9,500),
    'reg_lambda':(-0.001,10),  #무조건 양수인데 음수를 실수로 넣었다 그래서 아래서 max써준다
    'reg_alpha':(0.01,50)
}
#목적함수 준비하기(정의) 파라미터의 범위, 
def lgb_hamsu(max_depth, num_leaves, learning_rate, min_child_samples, min_child_weight, subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha):
    params = {
        'n_estimators':10000,
        'learning_rate':learning_rate,
        'max_depth':int(round(max_depth)),
        'num_leaves':int(round(num_leaves)),
        'min_child_samples':int(round(min_child_samples)),
        'min_child_weight':int(round(min_child_weight)),
        'subsample':max(min(subsample,1),0),              #드랍아웃과 비슷한 개념 1보다 작고 0보다 커야한다
        'colsample_bytree':colsample_bytree,
        'max_bin':max(int(round(max_bin)),10), #무조건 10이상. int(round(max_bin),10 둘이 비교해서 큰거 빼라
        'reg_lambda':max(reg_lambda,0),     #무조건 양수. reg_lambda랑 0이랑 비교해서 큰거 빼라 
        'reg_alpha':reg_alpha
    }
    model = LGBMRegressor(**params) #한꺼번에 넣는다
    model.fit(x_train, y_train, 
              eval_set = [(x_train,y_train),(x_test,y_test)],
              eval_metric = 'rmse',
              verbose=0,
              early_stopping_rounds=50)
    y_predict = model.predict(x_test)
    result = r2_score(y_test,y_predict)

    return result

lgb_bo = BayesianOptimization(f = lgb_hamsu,
                              pbounds = bayesian_params, #pbounds에 있는걸 위에 f=lgb_hamsu에 넣어라
                              random_state = 333)
start_time = time.time()
n_iter = 500
lgb_bo.maximize(init_points=5, n_iter=n_iter) #init_points 초기 포인트 찍고 여기서부터 n_iter가 시작된다
end_time = time.time()

print(lgb_bo.max)
print(n_iter,'번 걸린시간:', end_time-start_time)





