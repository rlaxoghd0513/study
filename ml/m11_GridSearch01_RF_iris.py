# 분류
# iris
# cancer
# dacon_diabetes
# wine
# fetch_covttype
# digits

# 회귀
# diabets
# california
# dacon_ddarung
# kaggle_bike

#모델 randomforestclassifier
# train test 나누고
# 스케일링

# parameters = [
#     {'n_estimators' : [100,200],'max_depth':[6,10,12], 'min_samples_leaf':[3,10]},
#     {'max_depth' : [6,8,10,12], 'min_samples_leaf':[3,5,7,10]},
#     {'min_smaples_leaf':[3,5,7,10], 'min_samples_split': [2,3,5,10]},
#     {'n_jobs': [-1,2,4],'min_samples_split':[2,3,5,10]}
# ]
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

x,y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, random_state = 333)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits = n_splits, shuffle =True, random_state=333)

parameters = [
    {'n_estimators':[100,200], 'max_depth':[6,10,12]}
]

model = GridSearchCV(RandomForestClassifier(), parameters,
                     cv=5,
                     verbose=1,
                     n_jobs=-1)

import time
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

y_pred_best = model.best_estimator_.predict(x_test)
import pandas as pd
print('최적의 파라미터:', model.best_params_)
print('최적튠:', accuracy_score(y_test, y_pred_best))
print('걸린시간:', round(end_time-start_time,2),'초')
# 최적의 파라미터: {'max_depth': 10, 'n_estimators': 200}
# 최적튠: 0.9473684210526315
# 걸린시간: 4.2 초
