#스태킹 최대 문제점
# 프리딕한걸 또 프리딕한다
# 결국 x_test로 또 훈련을 하는거니까 과적합의 문제가 생길 수도 있다
# 스태킹할땐 항상 과적합문제 신경써야한다 -> train을 두개로 나누었다?

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer,load_wine, load_digits, fetch_covtype
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier

#1. 데이터
path = './_data/dacon_diabetes/'
path_save = './_save/dacon_diabetes/'

train_set = pd.read_csv(path + 'train.csv',index_col=0)
test_set = pd.read_csv(path + 'test.csv',index_col=0)

x=train_set.drop(['Outcome'], axis=1)
y=train_set['Outcome']

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle= True, train_size=0.8, random_state=1030,stratify=y
)

scaler = StandardScaler()
x_train =  scaler.fit_transform(x_train)
x_test =  scaler.fit_transform(x_test)

#2. 모델
lr = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=8)
dt = DecisionTreeClassifier()

# model = VotingClassifier(
model = StackingClassifier(
    estimators=[('LR', lr), ('KNN', knn), ('DT', dt)],#voting안먹힘
    # final_estimator=LogisticRegression(), #디폴트는 logisticregression #predict한걸 훈련할 모델 
    # final_estimator=KNeighborsClassifier(),
    final_estimator=RandomForestClassifier(),
    # final_estimator=VotingClassifier('주저리주저리') 넣을순 있는데 성능이 좋을지는 모른다
    ) 

#3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
y_pred = model.predict(x_test)
print('model.score : ', model.score(x_test,y_test))
print("Stacking.acc : ", accuracy_score(y_test,y_pred))

Classifiers = [lr,knn,dt]

for model2 in Classifiers:
    model2.fit(x_train,y_train)
    y_pred = model2.predict(x_test)
    score2 = accuracy_score(y_test,y_pred)
    class_name = model2.__class__.__name__ 
    print("{0}정확도 : {1:4f}".format(class_name, score2))

# model.score :  0.7480916030534351
# Stacking.acc :  0.7480916030534351
# LogisticRegression정확도 : 0.801527
# KNeighborsClassifier정확도 : 0.755725
# DecisionTreeClassifier정확도 : 0.656489