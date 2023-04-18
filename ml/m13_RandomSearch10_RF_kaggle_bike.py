import numpy as np
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, LSTM ,GRU, SimpleRNN
import pandas as pd
from sklearn.model_selection import train_test_split,KFold,GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, RobustScaler, MaxAbsScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor


#데이터
path = './_data/kaggle_bike/'
path_save = './_save/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0) 

x = train_csv.drop(['casual','registered','count'], axis=1)
y= train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, random_state = 333)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits = n_splits, shuffle =True, random_state=333)

parameters = [
    {'n_estimators':[100,200], 'max_depth':[6,10,12]}
]

model = RandomizedSearchCV(RandomForestRegressor(), parameters,
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
# 최적의 파라미터: {'n_estimators': 100, 'max_depth': 10}
# 최적튠: 0.3696465065006843
# 걸린시간: 18.0 초