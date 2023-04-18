import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, LSTM, SimpleRNN, GRU, Conv1D
from sklearn.model_selection import train_test_split, cross_val_score, KFold,GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error 
import pandas as pd 
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor

path = './_data/ddarung/'
path_save= './_save/ddarung/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0) 
train_csv = train_csv.dropna() #dropna 결측치삭제

x = train_csv.drop(['count'], axis=1) 
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=333, shuffle=True)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits = n_splits, shuffle =True, random_state=333)

parameters = [
    {'min_samples_leaf':[3,10],'max_depth':[6,10,12]}]

model =RandomizedSearchCV(RandomForestRegressor(), parameters,
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
print('최적튠:', r2_score(y_test, y_pred_best))
print('걸린시간:', round(end_time-start_time,2),'초')
# 최적의 파라미터: {'min_samples_leaf': 3, 'max_depth': 10}
# 최적튠: 0.7760170835123057
# 걸린시간: 5.81 초