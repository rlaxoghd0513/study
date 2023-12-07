import pandas as pd
import numpy as np
import random
import os
import optuna
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
from lightgbm import LGBMClassifier

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(777)  # Seed 고정.

path = 'c:/users/bitcamp/study/_data/dacon_범죄/'
save_path = 'c:/users/bitcamp/study/_save/dacon_범죄/'

train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')

print(np.unique(train['시간']))

train['날씨'] = train['강수량(mm)'] + train['강설량(mm)'] + train['적설량(cm)']
test['날씨'] = test['강수량(mm)'] + test['강설량(mm)'] + test['적설량(cm)']

train = train.drop(['강수량(mm)' , '적설량(cm)', '강설량(mm)'], axis=1)
test = test.drop(['강수량(mm)','강설량(mm)','적설량(cm)'], axis=1)

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

ordinal_features = ['요일', '범죄발생지']


# Create a new feature 'is_weekend'
# x_train['is_weekend'] = x_train['요일'].apply(lambda x: 1 if x in ['토', '일'] else 0)
# x_test['is_weekend'] = x_test['요일'].apply(lambda x: 1 if x in ['토', '일'] else 0)

# Create a new feature 'is_night'
x_train['is_night'] = x_train['시간'].apply(lambda x: 1 if 0 <= x < 6 else 0)
x_test['is_night'] = x_test['시간'].apply(lambda x: 1 if 0 <= x < 6 else 0)

# Create a new feature 'is_weekend_night'
# x_train['is_weekend_night'] = x_train['is_weekend'] * x_train['is_night']
# x_test['is_weekend_night'] = x_test['is_weekend'] * x_test['is_night']


# Feature Engineering: one-hot encoding
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
x_train_ohe = ohe.fit_transform(x_train[ordinal_features])
x_test_ohe = ohe.transform(x_test[ordinal_features])

x_train = pd.concat([x_train, pd.DataFrame(x_train_ohe, columns=ohe.get_feature_names_out(ordinal_features))], axis=1)
x_test = pd.concat([x_test, pd.DataFrame(x_test_ohe, columns=ohe.get_feature_names_out(ordinal_features))], axis=1)

# from sklearn.preprocessing import PolynomialFeatures
# poly = PolynomialFeatures(degree=2)
# x_train = poly.fit_transform(x_train)
# x_test = poly.transform(x_test)
# print(x_train.shape, x_test.shape) #(103891, 40) (103891,)

# Scaling the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Handle Imbalanced Data
smote = SMOTE(random_state= 55, k_neighbors = 15)
x_train, y_train = smote.fit_resample(x_train, y_train)


import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x_train,x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.95, random_state=42, shuffle=True)
print(x_train.shape, x_val.shape)

def objective(trial):
    params = {
        'objective': 'multiclass',
        'num_class': num_classes,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 50),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 20)
    }

    # Create a LightGBM model
    model = lgb.LGBMClassifier(**params)

    # Train the model
    model.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(x_val)

    # Calculate accuracy as the objective metric
    accuracy = accuracy_score(y_val, y_pred)

    return accuracy

# Split your data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
num_classes = len(np.unique(y_train))

# Define the study object and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=500)

# Get the best parameters and the best accuracy achieved
best_params = study.best_params
best_accuracy = study.best_value

# Create a new model with the best parameters
best_model = lgb.LGBMClassifier(**best_params)

# Train the best model on the entire dataset
best_model.fit(x_train, y_train)

# Make predictions using the best model
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
submit.to_csv(save_path + date +'_'+ str(round(best_accuracy,4))+'.csv', index= False)