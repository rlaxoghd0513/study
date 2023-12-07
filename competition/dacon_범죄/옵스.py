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
x_train['is_weekend'] = x_train['요일'].apply(lambda x: 1 if x in ['토', '일'] else 0)
x_test['is_weekend'] = x_test['요일'].apply(lambda x: 1 if x in ['토', '일'] else 0)

# Create a new feature 'is_night'
x_train['is_night'] = x_train['시간'].apply(lambda x: 1 if 0 <= x < 6 else 0)
x_test['is_night'] = x_test['시간'].apply(lambda x: 1 if 0 <= x < 6 else 0)

# Create a new feature 'is_weekend_night'
x_train['is_weekend_night'] = x_train['is_weekend'] * x_train['is_night']
x_test['is_weekend_night'] = x_test['is_weekend'] * x_test['is_night']

# Feature Engineering: one-hot encoding
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
x_train_ohe = ohe.fit_transform(x_train[ordinal_features])
x_test_ohe = ohe.transform(x_test[ordinal_features])

x_train = pd.concat([x_train, pd.DataFrame(x_train_ohe, columns=ohe.get_feature_names_out(ordinal_features))], axis=1)
x_test = pd.concat([x_test, pd.DataFrame(x_test_ohe, columns=ohe.get_feature_names_out(ordinal_features))], axis=1)

# Scaling the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Handle Imbalanced Data
smote = SMOTE(random_state= 42, k_neighbors = 15)
x_train, y_train = smote.fit_resample(x_train, y_train)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.95, shuffle=True, stratify=y_train, random_state=42)

print(np.unique(y_train))
print(np.unique(y_val))

print(x_train.shape)
print(x_val.shape)
print(x_test.shape)

base_models = [
    ('catboost', CatBoostClassifier()),
    ('lgbm', LGBMClassifier()),
    ('xgb', XGBClassifier())
]

# Define meta model
meta_model = RandomForestClassifier()

# Define the stacking ensemble model
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)

# Define the objective function for Optuna optimization
def objective(trial):
    # Define the hyperparameters to optimize
    params = {
        'catboost__learning_rate': trial.suggest_loguniform('catboost__learning_rate', 0.01, 0.1),
        'lgbm__learning_rate': trial.suggest_loguniform('lgbm__learning_rate', 0.01, 0.1),
        'xgb__learning_rate': trial.suggest_loguniform('xgb__learning_rate', 0.01, 0.1)
    }
    
    # Set the hyperparameters for the stacking model
    stacking_model.set_params(**params)
    stacking_model.fit(x_train, y_train)
    pred = stacking_model.predict(x_val)
    acc = accuracy_score(y_val, pred)
    return acc

# Optimize the hyperparameters using Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Get the best hyperparameters
best_params = study.best_params
print("Best Hyperparameters:", best_params)

# Train the stacking model with the best hyperparameters
stacking_model.set_params(**best_params)
stacking_model.fit(x_train, y_train)

# Make predictions with the trained stacking model
y_pred = stacking_model.predict(x_val)
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)


predictions = stacking_model.predict(x_test)

# 제출 파일을 읽어옵니다.
submit = pd.read_csv(path + 'sample_submission.csv')

# 예측한 값을 TARGET 컬럼에 할당합니다.
submit['TARGET'] = predictions

#time'
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

# 예측한 결과를 파일로 저장합니다. index 인자의 값을 False로 설정하지 않으면 제출이 정상적으로 진행되지 않습니다.
submit.to_csv(save_path + date +'.csv', index= False)