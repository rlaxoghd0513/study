#스모트# 실습!! 시작!!!
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


#1. 데이터
path = './_data/dacon_wine/'
train_set = pd.read_csv(path + 'train.csv')
test_set = pd.read_csv(path + 'test.csv')

# data_set2 = train_set.to_numpy()
# print(type(data_set2))
# print(data_set2.shape)  # (4898, 12)

le = LabelEncoder()
le.fit(train_set['type'])

aaa = le.transform(train_set['type'])
train_set['type'] = aaa
test_set['type'] = le.transform(test_set['type'])

x = train_set.drop(['quality'], axis= 1)
y = train_set['quality']

print(x.shape, y.shape) # (4898, 11) (4898,)
print(np.unique(y, return_counts=True))  

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=123, shuffle=True, train_size=0.8,
    stratify=y
)
print(pd.Series(y_train).value_counts())
# 6.0    1758
# 5.0    1166
# 7.0     704
# 8.0     140
# 4.0     130
# 3.0      16
# 9.0       4
model = RandomForestClassifier()
model.fit(x_train, y_train)
print("#========================== SMOTE 적용 전 ============================ ")

#4. 평가, 예측
from sklearn.metrics import accuracy_score, f1_score
y_predict = model.predict(x_test) 
score = model.score(x_test, y_test)
# print('model.score : ', score)     
print('SMOTE 이전 acc : ', accuracy_score(y_test, y_predict))
print('SMOTE 이전 f1_score(macro) : ', f1_score(y_test, y_predict, average='macro'))   # f1_score는 이진분류이므로 average를 사용하여 다중분류에 사용
# print('f1_score(micro) : ', f1_score(y_test, y_predict, average='micro'))







print("#========================== SMOTE 적용 후 ============================ ")
smote = SMOTE(random_state=123, k_neighbors=3) 
x_train, y_train = smote.fit_resample(x_train, y_train)

# print(pd.Series(y_train).value_counts())
# 4.0    1758
# 5.0    1758
# 6.0    1758
# 7.0    1758
# 8.0    1758
# 3.0    1758
# 9.0    1758

#2. 모델 #3. 훈련
model = RandomForestClassifier()
model.fit(x_train, y_train)

#4. 평가, 예측
from sklearn.metrics import accuracy_score, f1_score

y_predict = model.predict(x_test) 
score = model.score(x_test, y_test)
# print('model.score : ', score)     
print('SMOTE 적용 후 acc_score : ', accuracy_score(y_test, y_predict))
print('SMOTE 적용 후 f1_score(macro) : ', f1_score(y_test, y_predict, average='macro'))   # f1_score는 이진분류이므로 average를 사용하여 다중분류에 사용
# print('f1_score(micro) : ', f1_score(y_test, y_predict, average='micro'))


# #========================== SMOTE 적용 전 ============================ 
# SMOTE 이전 acc :  0.6827272727272727
# SMOTE 이전 f1_score(macro) :  0.3578523454725242
# #========================== SMOTE 적용 후 ============================
# SMOTE 적용 후 acc_score :  0.6272727272727273
# SMOTE 적용 후 f1_score(macro) :  0.3646122436826789