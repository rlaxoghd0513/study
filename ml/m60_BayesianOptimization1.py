# BayesianOptimization 최댓값 찾는거
#회귀할때 어쩌라고

param_bounds = {'x1':(-1,5),#범위가 -1에서 5사이    #텍스트 형태로 넣어주고 범위는 튜플 형태로 넣어준다
                'x2':(0,4)}

def y_function(x1,x2):
    return -x1 **2 - (x2 -2) **2 +10           # **2 제곱

#이 함수의 최대값을 찾을거다

from bayes_opt import BayesianOptimization

optimizer = BayesianOptimization(
    f = y_function,
    pbounds = param_bounds,#우리가 찾을 범위
    random_state = 337
)

optimizer.maximize(init_points=5,
                   n_iter=16)

print(optimizer.max)
