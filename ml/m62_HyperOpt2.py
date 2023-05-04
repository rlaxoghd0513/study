from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
import time


#1데이터
x,y = load_diabetes(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=333, shuffle=True, train_size=0.8)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from hyperopt import hp, fmin, tpe, Trials
#2모델
search_space = {
    'learning_rate':hp.uniform('learning_rate',0.001,1), #quniform에서 q는 정수기 때문에 uniform 사용  #정규분포형태 
    'max_depth':hp.quniform('max_depth',3,16,1.0),
    'num_leaves':hp.quniform('num_leaves',24,64,1.0),
    # 'min_child_samples':hp.quniform('min_child_samples',10,200,1),
    # 'min_child_weight':hp.quniform('min_child_weight',1,50,1),
    'subsample':hp.uniform('subsample',0.5,1.0),    #드랍아웃개념
    # 'colsample_bytree':hp.uniform('colsample_bytree',0.5,1),
    # 'max_bin':hp.quniform('max_bin',9,500,1),
    # 'reg_lambda':hp.uniform('reg_lambda',0.001,10),  #무조건 양수인데 음수를 실수로 넣었다 그래서 아래서 max써준다
    # 'reg_alpha':hp.uniform('reg_alpha',0.01,50)
}
# hp.quniform(label,low,high,q) : 최소부터 최대까지 q 간격
# hp.uniform(label, low, high) : 최소부터 최대까지 정규분포 간격
# hp.randint(label, upper) : 0부터 최대값 upper까지 random한 정수값
# hp.loguniform(label,low,high) : exp(uniform(low,high))값 반환. 로그씌운걸 다시 원값으로 지수변환한다. #이거 역시 정규분포 

#목적함수 준비하기(정의) 파라미터의 범위, 
def lgb_hamsu(search_space):
    params = {
        'n_estimators':1000,
        'learning_rate':search_space['learning_rate'],
        'max_depth':int(search_space['max_depth']),
        'num_leaves':int(search_space['num_leaves']),
        # 'min_child_samples':int(round(min_child_samples)),
        # 'min_child_weight':int(round(min_child_weight)),
        'subsample':search_space['subsample'],              #드랍아웃과 비슷한 개념 1보다 작고 0보다 커야한다
        # 'colsample_bytree':colsample_bytree,
        # 'max_bin':max(int(round(max_bin)),10), #무조건 10이상. int(round(max_bin),10 둘이 비교해서 큰거 빼라
        # 'reg_lambda':max(reg_lambda,0),     #무조건 양수. reg_lambda랑 0이랑 비교해서 큰거 빼라 
        # 'reg_alpha':reg_alpha
    }
    model = LGBMRegressor(**params) #한꺼번에 넣는다
    model.fit(x_train, y_train, 
              eval_set = [(x_train,y_train),(x_test,y_test)],
              eval_metric = 'rmse',
              verbose=0,
              early_stopping_rounds=50)
    y_predict = model.predict(x_test)
    result = mean_squared_error(y_test,y_predict)

    return result
trial_val = Trials() #hist보려고

# lgb_bo = BayesianOptimization(f = lgb_hamsu,
#                               pbounds = search_space, #pbounds에 있는걸 위에 f=lgb_hamsu에 넣어라
#                               random_state = 333)
best = fmin(
    fn = lgb_hamsu,
    space = search_space,
    algo = tpe.suggest,
    max_evals = 50,
    trials = trial_val,
    rstate = np.random.default_rng(seed=10)
)
print(best)
print(trial_val.vals)
print(trial_val.results)

import pandas as pd

results= [aaa['loss'] for aaa in trial_val.results]
       
df=pd.DataFrame({'learning_rate':trial_val.vals['learning_rate'],
                'max_depth':trial_val.vals['max_depth'],
                'num_leaves':trial_val.vals['num_leaves'],
                'subsample':trial_val.vals['subsample'],
                'results':results})
print(df)

min_idx = df['results'].idxmin()
print(df.iloc[min_idx])

