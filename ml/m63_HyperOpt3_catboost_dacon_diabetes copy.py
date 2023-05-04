from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
import warnings
warnings.filterwarnings('ignore')
import time

#1. 데이터 
path = './_data/dacon_diabetes/'
path_save = './_save/dacon_diabetes/'

train_csv= pd.read_csv(path+'train.csv', index_col=0)
test_csv= pd.read_csv(path+'test.csv', index_col=0)
x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=337, train_size=0.8, stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

#2. 모델    # 정수형 quniform, 실수형 uniform
search_space = {
    'learning_rate' : hp.uniform('learning_rate', 0.001, 1),
    'depth' : hp.quniform('depth', 3, 16, 1),   # 3부터 16까지 1간격으로
    'one_hot_max_size' : hp.quniform('one_hot_max_size', 24, 64, 1),
    'min_data_in_leaf' : hp.quniform('min_data_in_leaf', 10, 200, 1),
    'bagging_temperature' : hp.uniform('bagging_temperature', 0.5, 1),
    'random_strength' : hp.uniform('random_strength', 0.5, 1),
    'l2_leaf_reg' : hp.uniform('l2_leaf_reg', 0.001, 10)
}
# hp.quniform(lable, low, high, q) : 최소부터 최대까지 q 간격
# hp.uniform(label, low, high) : 최소부터 최대까지 정규분포 간격
# np.randint(label, upper): 0부터 최대값 upper까지 random한 정수값
# hp.loguniform(label, low, high) : exp(uniform(low, high))값 반환 / 이거 역시 정규분포


def lgb_hamsu(search_space):
    params = {
        'iterations' : 10,
        'learning_rate' : search_space['learning_rate'],
        'depth' : int(search_space['depth']),           
        'l2_leaf_reg' : search_space['l2_leaf_reg'],          
        'bagging_temperature' :search_space['bagging_temperature'],   
        'random_strength' : search_space['random_strength'],     
        'one_hot_max_size' : int(search_space['one_hot_max_size']),       
        'min_data_in_leaf' : int(search_space['min_data_in_leaf']),
        'task_type' : 'CPU',    
        'logging_level' : 'Silent'                   
    }
    
    model = CatBoostClassifier(**params)

    model.fit(x_train, y_train,
          eval_set=[(x_train, y_train), (x_test, y_test)],
          verbose=0,
          early_stopping_rounds=50
    )
    
    y_predict = model.predict(x_test)
    results = -1*accuracy_score(y_test, y_predict)
    
    return results

trial_val = Trials()    # hist를 보기위해

# lgb_bo = BayesianOptimization(f= lgb_hamsu,
#                               pbounds= bayesian_params,
#                               random_state= 337
#                               )
best = fmin(
    fn = lgb_hamsu, 
    space = search_space,
    algo= tpe.suggest,
    max_evals= 50,
    trials= trial_val,
    rstate= np.random.default_rng(seed=10)
)
print('best : ', best)
# best :  {'learning_rate': 0.2509097581813602, 'max_depth': 7.0, 'numleaves': 41.0, 'subsample': 0.8759587138041729}



##### pandas 데이터프레임에 trial_val, vals를 넣어봐라 #####
## results컬럼 최소값이 있는 행을 출력
import pandas as pd

results = [aaa['loss'] for aaa in trial_val.results]     # trial_val 결과값들을 반복해라 반복하는 걸 aaa라고 라고 그거에 대한 결과를 aaa['loss']에 저장해라

# for aaa in trial_val.results:
#     losses.append(aaa['loss'])        # 위 한 줄과 동일

df = pd.DataFrame({'learning_rate' : trial_val.vals['learning_rate'],
                   'depth' : trial_val.vals['depth'],
                   'l2_leaf_reg' : trial_val.vals['l2_leaf_reg'],
                   'bagging_temperature' : trial_val.vals['bagging_temperature'],
                   'random_strength' : trial_val.vals['random_strength'],
                   'one_hot_max_size' : trial_val.vals['one_hot_max_size'],
                   'min_data_in_leaf' : trial_val.vals['min_data_in_leaf'],
                   'results' : results})
print(df)

min_idx = df['results'].idxmin()
print(df.iloc[min_idx])