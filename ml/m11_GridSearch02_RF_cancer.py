import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

x,y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, random_state = 333)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits = n_splits, shuffle =True, random_state=333)

parameters = [
    {'n_estimators':[100,200], 'min_samples_leaf':[3,10],'max_depth':[6,10,12]}]

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
# 최적의 파라미터: {'max_depth': 6, 'min_samples_leaf': 3, 'n_estimators': 200}
# 최적튠: 0.9370629370629371
# 걸린시간: 7.76 초