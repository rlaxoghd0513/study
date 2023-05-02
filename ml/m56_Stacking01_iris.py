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
data_list = [load_iris, load_breast_cancer,load_wine, load_digits,fetch_covtype]
data_list_name = ['아이리스','캔서','와인','디깃','콥타입']
for i,value in enumerate(data_list):
    x,y  = value(return_X_y=True)

    x_train, x_test, y_train, y_test = train_test_split(
        x,y, shuffle= True, train_size=0.8, random_state=1030
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
    print(data_list_name[i])
    print('model.score : ', model.score(x_test,y_test))
    print("Stacking.acc : ", accuracy_score(y_test,y_pred))

    Classifiers = [lr,knn,dt]

    for model2 in Classifiers:
        model2.fit(x_train,y_train)
        y_pred = model2.predict(x_test)
        score2 = accuracy_score(y_test,y_pred)
        class_name = model2.__class__.__name__ 
        print("{0}정확도 : {1:4f}".format(class_name, score2))

# 아이리스
# model.score :  0.8666666666666667
# Stacking.acc :  0.8666666666666667
# LogisticRegression정확도 : 0.900000
# KNeighborsClassifier정확도 : 0.900000
# DecisionTreeClassifier정확도 : 0.900000
# 캔서
# model.score :  0.9736842105263158
# Stacking.acc :  0.9736842105263158
# LogisticRegression정확도 : 0.964912
# KNeighborsClassifier정확도 : 0.929825
# DecisionTreeClassifier정확도 : 0.885965
# 와인
# model.score :  1.0
# Stacking.acc :  1.0
# LogisticRegression정확도 : 1.000000
# KNeighborsClassifier정확도 : 0.972222
# DecisionTreeClassifier정확도 : 0.916667
# 디깃
# model.score :  0.9722222222222222
# Stacking.acc :  0.9722222222222222
# LogisticRegression정확도 : 0.950000
# KNeighborsClassifier정확도 : 0.963889
# DecisionTreeClassifier정확도 : 0.808333