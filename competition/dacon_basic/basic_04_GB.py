from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import datetime


path = './_data/dacon_basic/'
save_path = './_save/dacon_basic/'
# 데이터 불러오기
train_df = pd.read_csv(path + 'train.csv')
test_df = pd.read_csv(path + 'test.csv')
sample_submission_df = pd.read_csv(path + 'sample_submission.csv')

# ID 열 제거
train_df = train_df.drop('ID', axis=1)
test_df = test_df.drop('ID', axis=1)

# Weight_Status, Gender 열을 숫자 데이터로 변환
train_df['Weight_Status'] = train_df['Weight_Status'].map({'Normal Weight': 0, 'Overweight': 1, 'Obese': 2})
train_df['Gender'] = train_df['Gender'].map({'M': 0, 'F': 1})
test_df['Weight_Status'] = test_df['Weight_Status'].map({'Normal Weight': 0, 'Overweight': 1, 'Obese': 2})
test_df['Gender'] = test_df['Gender'].map({'M': 0, 'F': 1})

# PolynomialFeatures를 사용하여 데이터 전처리
poly = PolynomialFeatures(degree=1, include_bias=False)
X = poly.fit_transform(train_df.drop('Calories_Burned', axis=1))
y = train_df['Calories_Burned']

# 표준화
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# train, valid 데이터 나누기
X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

n_splits = 5
kfold = KFold(n_splits = n_splits, shuffle=True, random_state=42)
parameters = [{'n_estimators':[100,200,300], 'learning_rate':[0.1,0.01,0.001], 'max_depth':[3,6,9]}]

# 회귀 신경망 모델 학습
mlp =RandomizedSearchCV(GradientBoostingRegressor(), parameters, cv=kfold, n_jobs=-1, verbose=1)
mlp.fit(X_train, y_train)

# valid 데이터 예측 및 평가
y_pred_valid = mlp.predict(X_valid)
rmse_valid = np.sqrt(mean_squared_error(y_valid, y_pred_valid))
print(f"Valid 데이터 RMSE: {rmse_valid:.3f}")

# test 데이터 예측
X_test = test_df.values
X_poly_test = poly.transform(X_test)
X_test_scaled = scaler.transform(X_poly_test)
y_pred_test = mlp.predict(X_test_scaled)

# 결과 저장
sample_submission_df['Calories_Burned'] = y_pred_test
date = datetime.datetime.now().strftime("%m%d_%H%M")
sample_submission_df.to_csv(save_path +date+ 'submission.csv', index=False) 