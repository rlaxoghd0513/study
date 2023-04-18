import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold,GridSearchCV,train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

path = './_data/dacon_diabetes/'
path_save = './_save/dacon_diabetes/'

train_csv = pd.read_csv(path+'train.csv', index_col=0)

test_csv = pd.read_csv(path+'test.csv', index_col=0)

x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=333, shuffle=True)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits = n_splits, shuffle =True, random_state=333)

parameters = [
    {'min_samples_leaf':[3,10],'max_depth':[6,10,12]}]

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

# 최적의 파라미터: {'max_depth': 6, 'min_samples_leaf': 10}
# 최적튠: 0.7361963190184049
# 걸린시간: 4.5 초