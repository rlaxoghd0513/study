#최솟값 찾는거
#Bayesianoptimization은 최댓값
#pip install hyperopt
import numpy as np
import hyperopt
print(hyperopt.__version__) #0.2.7

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

search_space = {
    'x1':hp.quniform('x1',-10,10,1),  #-10부터 10까지 1단위
    'x2':hp.quniform('x2',-15,15,1)   #quniform 
}      #hp.quniform(label, low, high, q) q를 2로 하면 -10 -8 -6 이런식으로 2단위 0.5같은 소수단위도 가능

print(search_space)

def objective_func(search_space):
    x1 = search_space['x1']
    x2 = search_space['x2']
    return_value = x1**2 - 20*x2
    
    return return_value
    #권장 리턴 방식 return {'loss':return_value, 'status':StaTus_OK}
trial_val = Trials()#hist같은거

best = fmin(
    fn = objective_func,
    space = search_space,#얘는 파라미터를 space라고 한다
    algo = tpe.suggest,
    max_evals = 100, #bayesianoptimization에서 n_iter같은거, 몇번 돌릴지,
    trials =trial_val,
    rstate = np.random.default_rng(seed = 10))#랜덤스테이트같은거, 빼면 돌릴때먀다 값이 달라지니까 더 좋을값이 나올수도 있다

print(best)
# {'x1': -0.0, 'x2': 15.0}
print(trial_val.results)
print(trial_val.vals)

#pandas 데이타프레임에 trail_val_vals를 넣으라

import pandas as pd
# for aaa in trial_val.results:
#     losses.append(aaa['loss'])
# 포문을 편하게 한줄로 쓴거 
results= [aaa['loss'] for aaa in trial_val.results]
       
df=pd.DataFrame({'x1':trial_val.vals['x1'],
                'x2':trial_val.vals['x2'],
                'results':results})
print(df)