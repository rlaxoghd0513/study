import pandas as pd
from unidecode import unidecode

path = './_data/dacon_범죄/'
train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
print(train_csv.shape) #(84406, 19)
print(train_csv.columns) #Index(['월', '요일', '시간', '소관경찰서', '소관지역', '사건발생거리', '강수량(mm)', '강설량(mm)','적설량(cm)', 
                                # '풍향', '안개', '짙은안개', '번개', '진눈깨비', '서리', '연기/연무', '눈날림','범죄발생지', 'TARGET'],

new_columns = [unidecode(col) for col in train_csv.columns]
train_csv.columns = new_columns

new_columns1 = [unidecode(col) for col in test_csv.columns]
test_csv.columns = new_columns1

print(train_csv.columns)
#요일 범죄발생지
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_csv['yoil'] = le.fit_transform(train_csv['yoil'])
test_csv['yoil'] = le.transform(test_csv['yoil'])

train_csv['beomjoebalsaengji'] = le.fit_transform(train_csv['beomjoebalsaengji'])
test_csv['beomjoebalsaengji'] = le.transform(test_csv['beomjoebalsaengji'])

x = train_csv.drop('TARGET', axis=1)
y = train_csv['TARGET']

# print(x.shape, y.shape)#(84406, 18) (84406,)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, shuffle=True, random_state=42, stratify=y)
#################################################################################################################

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
scaler = QuantileTransformer()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

###################################################################################################################
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold,KFold,GridSearchCV, cross_val_score, cross_val_predict

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

parameters = 

lgbm = LGBMClassifier()
model = GridSearchCV(lgbm, parameters, cv=kfold, n_jobs=-1)

model.fit(x_train, y_train)
model.score(x_test, y_test)
y_predict = model.predict(x_test)

from sklearn.metrics import accuracy_score, f1_score
acc = accuracy_score(y_test, y_predict)
print('acc:', acc)

f1 = f1_score(y_test, y_predict, average= 'macro')
print('f1:', f1)

path_save = './_save/dacon_범죄/'

y_submit = model.predict(test_csv)

submission = pd.read_csv(path + 'sample_submission.csv', index_col =0)

submission['TARGET'] = y_submit

import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

submission.to_csv(path_save + date + '_' + str(round(f1, 2)) + '.csv')

#xgb f1: 0.5232632950598683
#cat f1: 0.5216000797653145
#lgbm nonscale f1: 0.5257959760487064
    # qunatile f1: 0.5272061978293162
    # power    f1: 0.5258312600411942
    # minmax   f1: 0.526436585231461
    # maxabs   f1: 0.5257959760487064
    # standard f1: 0.5253788149592035
    # robust   f1: 0.526006062755756
#rf f1: 0.496937101245706

