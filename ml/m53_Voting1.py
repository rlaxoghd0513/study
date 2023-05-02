#m51_bagging1 카피

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from sklearn.ensemble import VotingClassifier #투표

#1. 데이터
x,y  = load_breast_cancer(return_X_y=True)

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

model = VotingClassifier(
    estimators=[('LR', lr), ('KNN', knn), ('DT', dt)],
                voting='soft', #디폴트는 하드, 성능은 소프트가 더 좋음.
)

#3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
y_pred = model.predict(x_test)
print('model.score : ', model.score(x_test,y_test))
print("Voting.acc : ", accuracy_score(y_test,y_pred))

#Hard Voting
# model.score :  0.9473684210526315
# acc :  0.9473684210526315 

#Soft Voting
# model.score :  0.9649122807017544
# acc :  0.9649122807017544

Classifiers = [lr,knn,dt]

for model2 in Classifiers:
    model2.fit(x_train,y_train)
    y_pred = model2.predict(x_test)
    score2 = accuracy_score(y_test,y_pred)
    class_name = model2.__class__.__name__ 
    print("{0}정확도 : {1:4f}".format(class_name, score2))