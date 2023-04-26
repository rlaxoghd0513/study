import numpy as np
import pandas as pd

path = './_data/kaggle_bike/'
path_save = './_save/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0) 

x = train_csv.drop(['casual','registered','count'], axis=1)
y= train_csv['count']
print(x.shape, y.shape)

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor

imputer = IterativeImputer(estimator=XGBRegressor())
print(x.isnull().sum())
print(y.isnull().sum())

x1 = imputer.fit_transform(x)

model = XGBRegressor()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x1, y, train_size=0.7, random_state=123, shuffle=True)
model.fit(x_train, y_train)

result = model.score(x_test, y_test)
print('result : ', result)

from sklearn.metrics import r2_score
y_pred = model.predict(x_test)
print('r2 : ', r2_score(y_test, y_pred))