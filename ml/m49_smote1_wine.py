#데이터 오버샘플링에 관하여
#SMOTE 오버샘플링. 시간이 너무 오래걸린다. 정말 너무너무 오래걸림.

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier


#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape)  #(178, 13) (178,)
print(np.unique(y, return_counts=True))
print(pd.Series(y).value_counts().sort_index())
# 1    71
# 0    59
# 2    48
# dtype: int64

#sort_index 후
# 0    59
# 1    71
# 2    48
# dtype: int64
print(y)


# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]

x = x[:-25]
y = y[:-25]

# print(x.shape, y.shape)  #(153, 13) (153,)

# print(y)

print(pd.Series(y).value_counts().sort_index())
# 0    59
# 1    71
# 2    23

x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 train_size=0.75,
                                                 shuffle=True,
                                                 random_state=3377,
                                                 stratify=y,
                                                 )


print(pd.Series(y_train).value_counts().sort_index())

# 0    44
# 1    53
# 2    17

model = RandomForestClassifier()
model.fit(x_train, y_train)



score = model.score(x_test,y_test)
y_pred = model.predict(x_test)
print('model.score :', score)
print('accuracy_score :', accuracy_score(y_test, y_pred))
print('f1_score(macro) :', f1_score(y_test, y_pred, average='macro'))
# print('f1_score(micro) :', f1_score(y_test, y_pred, average='micro'))
# print('f1_score(micro) :', f1_score(y_test, y_pred))
# f1스코어는 이진분류에서 사용한다. 
# average 항목을 macro나 micro를 입력하면 다중분류에서 사용할 수 있게 된다.
# 클래스간의 불균형이 심하다면 f1스코어가 acc보다 정확할 수 있다.

# 0    44
# 1    53
# 2    17
# 이 데이터에 SOMTE를 사용하면 모든 클래스가 53개로 변함.
# 가장 쉬운 증폭은 copy
print(x_train.shape, y_train.shape)

print("====================SMOTE 적용 후==========================")
smote = SMOTE(random_state=321, 
              k_neighbors=8,#최근접 이웃 방식, k개의 데이터의 영향을 받는다.
              )
x_train, y_train = smote.fit_resample(x_train, y_train)
print(x_train.shape, y_train.shape)
print(pd.Series(y_train).value_counts().sort_index())
# (114, 13) (114,) SMOTE 적용 전
# (159, 13) (159,) SMOTE 적용 후
# 0    53
# 1    53
# 2    53


model = RandomForestClassifier()
model.fit(x_train, y_train)



score = model.score(x_test,y_test)
y_pred = model.predict(x_test)
print('model.score :', score)
print('accuracy_score :', accuracy_score(y_test, y_pred))
print('f1_score(macro) :', f1_score(y_test, y_pred, average='macro'))
