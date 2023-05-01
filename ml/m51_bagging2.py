import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#1데이터
x,y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=123, train_size=0.8,shuffle=True, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2모델
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier #Decision모여있는게 RandomForest
from sklearn.linear_model import LogisticRegression

# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
aaa = LogisticRegression()
model = BaggingClassifier(aaa,
                          n_estimators=10,
                          n_jobs=-1,
                          random_state=337,
                          bootstrap=True) #배깅에서 bootstrap 파라미터만 알면 됨, 디폴트:True 
# for i,value in enumerate(model_list):
#     model = BaggingClassifier(value,
#                               n_estimators=10,
#                               bootstrap=True)
    
#샘플의 중복을 허용했을때 통상적으로 성능이 더 좋다

#3훈련
model.fit(x_train, y_train)

#4 평가예측
print('model.score:',model.score(x_test, y_test))
y_predict = model.predict(x_test)
print('acc:',accuracy_score(y_test, y_predict))