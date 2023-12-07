import optuna
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import random
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42)  # Seed 고정.

path = 'c:/users/bitcamp/study/_data/dacon_범죄/'
save_path = 'c:/users/bitcamp/study/_save/dacon_범죄/'

train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')

# train['날씨'] = train['강수량(mm)'] + train['강설량(mm)'] + train['적설량(cm)']
# test['날씨'] = test['강수량(mm)'] + test['강설량(mm)'] + test['적설량(cm)']

# train = train.drop(['강수량(mm)' , '적설량(cm)', '강설량(mm)'], axis=1)
# test = test.drop(['강수량(mm)','강설량(mm)','적설량(cm)'], axis=1)

# x_train = train.drop(['ID', 'TARGET'], axis = 1)
x_train = train.drop(['ID', 'TARGET'], axis = 1)
y_train = train['TARGET']
x_test = test.drop('ID', axis = 1)

le = LabelEncoder()

# '요일'과 '범죄발생지' 특성에 대해 Label Encoding 진행
for feature in ['요일', '범죄발생지']:
    # Train 데이터에 대해 fit과 transform을 동시에 수행
    x_train[feature] = le.fit_transform(x_train[feature])
    # Test 데이터에 대해 transform만 수행
    x_test[feature] = le.transform(x_test[feature])

x_train['강수량(mm)'] = x_train['강수량(mm)'].apply(lambda x: 0 if x == 0.000 else 1)
x_test['강수량(mm)'] = x_test['강수량(mm)'].apply(lambda x: 0 if x == 0.000 else 1)

x_train['강설량(mm)'] = x_train['강설량(mm)'].apply(lambda x: 0 if x == 0.0 else 1)
x_test['강설량(mm)'] = x_test['강설량(mm)'].apply(lambda x: 0 if x == 0.0 else 1)

x_train['적설량(cm)'] = x_train['적설량(cm)'].apply(lambda x: 0 if x == 0.0 else 1)
x_test['적설량(cm)'] = x_test['적설량(cm)'].apply(lambda x: 0 if x == 0.0 else 1)

# Create a new feature 'is_weekend'
x_train['is_weekend'] = x_train['요일'].apply(lambda x: 1 if x in ['토', '일'] else 0)
x_test['is_weekend'] = x_test['요일'].apply(lambda x: 1 if x in ['토', '일'] else 0)

# Create a new feature 'is_night'

x_train['morn'] = x_train['시간'].apply(lambda x: 0 if 0 <= x < 3 else (1 if 3 <= x < 6 else (2 if 6 <= x < 9 else (4 if 9 <= x <= 12 else None))))
x_test['morn'] = x_test['시간'].apply(lambda x: 0 if 0 <= x < 3 else (1 if 3 <= x < 6 else (2 if 6 <= x < 9 else (4 if 9 <= x <= 12 else None))))

#계절
x_train['계절'] = x_train['월'].apply(lambda x: 0 if 1 <= x < 3 else (1 if 3 <= x < 6 else (2 if 6 <= x < 9 else (4 if 9 <= x < 12 else 0))))
x_test['계절'] = x_test['월'].apply(lambda x: 0 if 1 <= x < 3 else (1 if 3 <= x < 6 else (2 if 6 <= x < 9 else (4 if 9 <= x < 12 else 0))))

it_column = ['안개','짙은안개','번개','진눈깨비','서리', '연기/연무', '눈날림']
for i,value in enumerate(it_column):
    x_train[value] = x_train[value].astype(int)
    x_test[value] = x_test[value].astype(int)

print(x_train.columns)
# Index(['월', '요일', '시간', '소관경찰서', '소관지역', '사건발생거리', '강수량(mm)', '강설량(mm)',
#        '적설량(cm)', '풍향', '안개', '짙은안개', '번개', '진눈깨비', '서리', '연기/연무', '눈날림',
#        '범죄발생지', '날씨', 'is_weekend', 'morn', '계절'],
print(x_train.head(10))
print(x_train.info())
print(y_train.info())
# 원핫인코딩할 칼럼 선택
# columns_to_encode = ['계절', 'morn', 'is_weekend','강수량(mm)', '적설량(cm)', '강설량(mm)','안개', '짙은안개','번개','진눈깨비', '서리', '연기/연무', '눈날림']

# # 선택된 칼럼들을 원핫인코딩
# x_train_encoded = pd.get_dummies(x_train, columns=columns_to_encode)
# x_test_encoded = pd.get_dummies(x_test, columns = columns_to_encode)

# # 원핫인코딩하지 않을 칼럼들은 그대로 유지
# columns_to_keep = ['월', '요일', '시간','소관경찰서', '소관지역', '사건발생거리', '풍향','범죄발생지']
# x_train_encoded[columns_to_keep] = x_train[columns_to_keep]
# x_test_encoded[columns_to_keep] = x_test[columns_to_keep]

# x_train = x_train_encoded
# x_test = x_test_encoded

# print(x_train.head(10))

# Scaling the data
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

print(x_train)
print(y_train)

x_train = x_train.to_numpy()
x_test = x_test.to_numpy()
y_train = y_train.to_numpy()

# Handle Imbalanced Data
smote = SMOTE(random_state= 42, k_neighbors = 15)
x_train, y_train = smote.fit_resample(x_train, y_train)


from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.9, shuffle=True, stratify=y_train, random_state=42)

print(np.unique(y_train))
print(np.unique(y_val))

print(x_train.shape)
print(x_val.shape)
print(x_test.shape)

# Define your XGBClassifier
xgb = XGBClassifier()

# Define the parameter grid to search through
param_grid = {
    'learning_rate': [0.1, 0.3],
    'n_estimators': [100, 200],
    'max_depth': [4,6],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0, 0.5],
    'reg_lambda': [1, 0.5]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(xgb, param_grid, cv=4, scoring='accuracy')
grid_search.fit(x_train, y_train)

# Print the best parameters found
print("Best parameters: ", grid_search.best_params_)

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
pred = best_model.predict(x_test)
# 제출 파일을 읽어옵니다.
submit = pd.read_csv(path + 'sample_submission.csv')

# 예측한 값을 TARGET 컬럼에 할당합니다.
submit['TARGET'] = pred

#time'
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

# 예측한 결과를 파일로 저장합니다. index 인자의 값을 False로 설정하지 않으면 제출이 정상적으로 진행되지 않습니다.
submit.to_csv(save_path + date +'.csv', index= False)