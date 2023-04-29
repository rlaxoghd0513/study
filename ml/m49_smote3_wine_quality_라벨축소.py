# [실습] y클래스를 3개까지 줄이고 그것을 smote 해서
# 성능 비교
# 3개로 줄인 결과와, 3개를 smote로 증폭한 결과 비교

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
print(pd.Series(y_train).value_counts().sort_index())
# 3      21
# 4     149
# 5    1430
# 6    1932
# 7     739
# 8     122
# 9       4

new_y = []
for i in y :
    if i<=5 :
        new_y += [0]
    elif i<=6:
        new_y += [1]
    elif i <=9:
        new_y += [2]
    else:
        new_y += [3]

y = np.array(new_y)
print(np.unique(y, return_counts=True))


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




# 라벨 축소 이전
# #========================== SMOTE 적용 전 ============================ 
# SMOTE 이전 acc :  0.6827272727272727
# SMOTE 이전 f1_score(macro) :  0.3578523454725242
# #========================== SMOTE 적용 후 ============================
# SMOTE 적용 후 acc_score :  0.6272727272727273
# SMOTE 적용 후 f1_score(macro) :  0.3646122436826789


# 라벨 축소
# #========================== SMOTE 적용 전 ============================ 
# SMOTE 이전 acc :  0.6609090909090909
# SMOTE 이전 f1_score(macro) :  0.3381443912197954
# #========================== SMOTE 적용 후 ============================
# SMOTE 적용 후 acc_score :  0.6345454545454545
# SMOTE 적용 후 f1_score(macro) :  0.3670485730531699