import numpy as np
import pandas as pd

#1 데이터
path = './_data/dacon_도서/'
train_csv = pd.read_csv(path + 'train.csv', encoding='cp949')
test_csv = pd.read_csv(path + 'test.csv', encoding='cp949')

# print(train_csv.columns) #Index(['ID', 'User-ID', 'Book-ID', 'Book-Rating', 'Age', 'Location','Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher'],dtype='object')

train_csv = train_csv.drop('ID', axis=1)
test_csv = test_csv.drop('ID', axis=1)
# print(train_csv.columns) #Index(['User-ID', 'Book-ID', 'Book-Rating', 'Age', 'Location', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher'],dtype='object')

train_csv['User-ID'] = train_csv['User-ID'].str[5:10]
test_csv['User-ID'] = test_csv['User-ID'].str[5:10]

train_csv['Book-ID'] = train_csv['Book-ID'].str[5:11]
test_csv['Book-ID'] = test_csv['Book-ID'].str[5:11]

train_csv['legion1'] = train_csv['Location'].apply(lambda x:x.split(",")[0])
train_csv['legion2'] = train_csv['Location'].apply(lambda x:x.split(",")[1])
train_csv['country'] = train_csv['Location'].apply(lambda x:x.split(",")[2])

test_csv['legion1'] = test_csv['Location'].apply(lambda x:x.split(",")[0])
test_csv['legion2'] = test_csv['Location'].apply(lambda x:x.split(",")[1])
test_csv['country'] = test_csv['Location'].apply(lambda x:x.split(",")[2])

train_csv = train_csv.drop(['Location'], axis=1)
test_csv = test_csv.drop(['Location'], axis=1)

# print(train_csv)
# print(test_csv)
# print(train_csv.info())

#  0   User-ID              871393 non-null  object
#  1   Book-ID              871393 non-null  object
#  2   Book-Rating          871393 non-null  int64
#  3   Age                  871393 non-null  float64
#  4   Book-Title           871393 non-null  object
#  5   Book-Author          871393 non-null  object
#  6   Year-Of-Publication  871393 non-null  float64
#  7   Publisher            871393 non-null  object
#  8   legion1              871393 non-null  object
#  9   legion2              871393 non-null  object
#  10  country              871393 non-null  object

train_csv['User-ID'] = pd.to_numeric(train_csv['User-ID'])
test_csv['User-ID'] = pd.to_numeric(test_csv['User-ID'])

train_csv['Book-ID'] = pd.to_numeric(train_csv['Book-ID'])
test_csv['Book-ID'] = pd.to_numeric(test_csv['Book-ID'])

# print(train_csv.info())
# print(test_csv.info())
# print(train_csv.shape)#(871393, 11)
# print(test_csv.shape)#(159621, 10)

#  0   User-ID              159621 non-null  int64
#  1   Book-ID              159621 non-null  int64
#  2   Age                  159621 non-null  float64
#  3   Book-Title           159621 non-null  object
#  4   Book-Author          159620 non-null  object
#  5   Year-Of-Publication  159621 non-null  float64
#  6   Publisher            159621 non-null  object
#  7   legion1              159621 non-null  object
#  8   legion2              159621 non-null  object
#  9   country              159621 non-null  object

from sklearn.preprocessing import LabelEncoder

############################################### book title #########################################################
unique_books = train_csv['Book-Title'].unique()

# LabelEncoder 객체를 생성하고, train data에서 'Book-Title'을 학습시킵니다.
le = LabelEncoder()
le.fit(unique_books)

# test data에서 새로운 레이블이 있는지 확인합니다.
new_labels = set(test_csv['Book-Title'].unique()) - set(unique_books)

# 만약 새로운 레이블이 있다면, train data에 추가합니다.
if new_labels:
    unique_books = pd.concat([train_csv, test_csv[test_csv['Book-Title'].isin(new_labels)]])

    # LabelEncoder 객체를 다시 학습시킵니다.
    unique_books = unique_books['Book-Title'].unique()
    le = LabelEncoder()
    le.fit(unique_books)

# train data와 test data에서 'Book-Title'을 변환합니다.
train_csv['Book-Title'] = le.transform(train_csv['Book-Title'])
test_csv['Book-Title'] = le.transform(test_csv['Book-Title'])

############################################ book author ################################################################

unique_books1 = train_csv['Book-Author'].unique()

# LabelEncoder 객체를 생성하고, train data에서 'Book-Title'을 학습시킵니다.
le = LabelEncoder()
le.fit(unique_books1)

# test data에서 새로운 레이블이 있는지 확인합니다.
new_labels1 = set(test_csv['Book-Author'].unique()) - set(unique_books1)

# 만약 새로운 레이블이 있다면, train data에 추가합니다.
if new_labels1:
    unique_books1 = pd.concat([train_csv, test_csv[test_csv['Book-Author'].isin(new_labels1)]])

    # LabelEncoder 객체를 다시 학습시킵니다.
    unique_books1 = unique_books1['Book-Author'].unique()
    le = LabelEncoder()
    le.fit(unique_books1)

train_csv['Book-Author'] = le.transform(train_csv['Book-Author'])
test_csv['Book-Author'] = le.transform(test_csv['Book-Author'])

####################################### publisher ####################################################################

unique_books2 = train_csv['Publisher'].unique()

# LabelEncoder 객체를 생성하고, train data에서 'Book-Title'을 학습시킵니다.
le = LabelEncoder()
le.fit(unique_books2)

# test data에서 새로운 레이블이 있는지 확인합니다.
new_labels2 = set(test_csv['Publisher'].unique()) - set(unique_books2)

# 만약 새로운 레이블이 있다면, train data에 추가합니다.
if new_labels2:
    unique_books2 = pd.concat([train_csv, test_csv[test_csv['Publisher'].isin(new_labels2)]])

    # LabelEncoder 객체를 다시 학습시킵니다.
    unique_books2 = unique_books2['Publisher'].unique()
    le = LabelEncoder()
    le.fit(unique_books2)

train_csv['Publisher'] = le.transform(train_csv['Publisher'])
test_csv['Publisher'] = le.transform(test_csv['Publisher'])

######################################## legion1 #############################################################################

unique_books3 = train_csv['legion1'].unique()

le = LabelEncoder()
le.fit(unique_books3)

new_labels3 = set(test_csv['legion1'].unique()) - set(unique_books3)

if new_labels3:
    unique_books3 = pd.concat([train_csv, test_csv[test_csv['legion1'].isin(new_labels3)]])

    unique_books3 = unique_books3['legion1'].unique()
    le = LabelEncoder()
    le.fit(unique_books3)

train_csv['legion1'] = le.transform(train_csv['legion1'])
test_csv['legion1'] = le.transform(test_csv['legion1'])

##################################### legion2 ###################################################################

unique_books4 = train_csv['legion2'].unique()

le = LabelEncoder()
le.fit(unique_books4)

new_labels4 = set(test_csv['legion2'].unique()) - set(unique_books4)

if new_labels4:
    unique_books4 = pd.concat([train_csv, test_csv[test_csv['legion2'].isin(new_labels4)]])

    unique_books4 = unique_books4['legion2'].unique()
    le = LabelEncoder()
    le.fit(unique_books4)

train_csv['legion2'] = le.transform(train_csv['legion2'])
test_csv['legion2'] = le.transform(test_csv['legion2'])

##################################### country ###################################################################

unique_books5 = train_csv['country'].unique()

le = LabelEncoder()
le.fit(unique_books5)

new_labels5 = set(test_csv['country'].unique()) - set(unique_books5)

if new_labels5:
    unique_books5 = pd.concat([train_csv, test_csv[test_csv['country'].isin(new_labels5)]])

    unique_books5 = unique_books5['country'].unique()
    le = LabelEncoder()
    le.fit(unique_books5)

train_csv['country'] = le.transform(train_csv['country'])
test_csv['country'] = le.transform(test_csv['country'])

############################################################################################################

# print(train_csv.info())
# print(test_csv.info())

x = train_csv.drop('Book-Rating',axis=1)
y = train_csv['Book-Rating']
# print(x.shape) #(871393, 10)
# print(y.shape) #(871393,)
# print(x)
# print(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, train_size=0.9, random_state=42)

####################################### scaler #########################################
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer

scaler = QuantileTransformer()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

######################################## fit ###################################################################

from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

model = ExtraTreesRegressor(max_features=0.05, min_samples_split=12, random_state=42)

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
rmse =  np.sqrt(mean_squared_error(y_test, y_predict))
print('rmse:', rmse)

################################## submit ###############################################

path_save = './_save/dacon_도서/'

y_submit = model.predict(test_csv)

submission = pd.read_csv(path + 'sample_submission.csv', index_col =0)

submission['Book-Rating'] = y_submit

import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

submission.to_csv(path_save + date + '_' + str(round(rmse, 2)) + '.csv')

#################################### rmse ###############################################

# 스케일링 안했을때 rmse: 3.5327626373561776
# quantile  rmse: 3.5324078047198526
# power rmse: 3.5327644258372977
# minmax  rmse: 3.5327487730416918
# maxabs rmse: 3.5327371545987156
# robust rmse: 3.532752829322522
# standard rmse: 3.5327667364127984

# XGB  rmse: 3.5317749022060148
# CAT  rmse: 3.5345745146187717
# LGBM rmse: 3.6295209985515973
# RFR  