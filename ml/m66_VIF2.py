from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler #통상 스케일러는 스탠다드를 씀.

#1. 데이터
datasets =  fetch_california_housing()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names) # x
df['target'] = datasets.target # y
print(df)

x = df.drop(['target'], axis =1)
y = df['target']

vif = pd.DataFrame()
vif['variables'] = x.columns

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

vif['VIF'] = [variance_inflation_factor(x_scaled, i) for i in range(x_scaled.shape[1])]

print(vif)

#     variables       VIF
# 0      MedInc  2.501295
# 1    HouseAge  1.241254
# 2    AveRooms  8.342786
# 3   AveBedrms  6.994995
# 4  Population  1.138125
# 5    AveOccup  1.008324
# 6    Latitude  9.297624
# 7   Longitude  8.962263

# x= x.drop(['Latitude'], axis = 1)
x= x.drop(['Latitude', 'Longitude'], axis = 1)
x_train,x_test, y_train, y_test = train_test_split(
    x, y, shuffle= True, random_state=337, test_size=0.2
)#스케일러된 x를 넣지 않음.

scaler2 =StandardScaler()
x_train = scaler2.fit_transform(x_train)
x_test =  scaler2.transform(x_test)

#2. 모델
model = RandomForestRegressor()

#3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
results = model.score(x_test,y_test)
print("결과 : ", results)

#결과 :  0.7296584507816609
# 결과 :  0.6747887972204057

# 실습
# 하나를 삭제했을 떄 바뀐 값들을 다시 표시.