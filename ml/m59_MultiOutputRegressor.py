import numpy as np
import pandas as pd
from sklearn.datasets import load_linnerud
from sklearn.linear_model import Lasso,Ridge # :L1규제 가중치에서 절대값, L2규제 가중치에서 제곱규제
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

x,y = load_linnerud(return_X_y=True)
print(x)
print(y)
print(x.shape, y.shape) #(20, 3) (20, 3)

# model = Ridge()
# model = XGBRegressor()

# model.fit(x,y)
# print('스코어:',mean_absolute_percentage_error(y, y_pred))

# print(model.predict([[2,110,43]]))

# Ridge모델
# 정상값     [138.  33.  68.]
# predict값  [187.32842123  37.0873515   55.40215097]

#XGBRegressor모델
#predict값   [[138.00215   33.001656  67.99831 ]]

#LGBMRegressor모델 
# 그냥 쓰면 에러
# ValueError: y should be a 1d array, got an array of shape (20, 3) instead.
# 1차원이 들어가야된다 그래서 3번 훈련해서 컨캣한다 너무 번거롭다 그래서 multioutputregressor로 lgbmregressor를 감싸준다

# model = MultiOutputRegressor(LGBMRegressor())
# model.fit(x,y)
# y_pred = model.predict(x)
# print(model.__class__.__name__,'스코어',mean_absolute_percentage_error(y, y_pred))

# print(model.predict([[2,110,43]]))
# [[178.6  35.4  56.1]]

# model = MultiOutputRegressor(CatBoostRegressor())
# model.fit(x,y)
# y_pred = model.predict(x)
# print(model.__class__.__name__,'스코어',mean_absolute_percentage_error(y, y_pred))
# print(model.predict([[2,110,43]]))
# # 캣부스트 그냥 쓰면 에러
# # _catboost.CatBoostError: C:/Program Files (x86)/Go Agent/pipelines/BuildMaster/catboost.git/catboost/private/libs/target/data_providers.cpp:612:
# #     Currently only multi-regression, multilabel and survival objectives work with multidimensional target 
# # MultiOutputRegressor 스코어 0.0021921173091030005
# # [[138.97756017  33.09066774  67.61547996]]

model = CatBoostRegressor(loss_function='MultiRMSE')
model.fit(x,y)
y_pred = model.predict(x)
print(model.__class__.__name__,'스코어',mean_absolute_percentage_error(y, y_pred))
print(model.predict([[2,110,43]]))
