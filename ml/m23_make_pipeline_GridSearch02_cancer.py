import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split,GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import SVC

#1 데이터
x,y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8,random_state = 123, shuffle=True)

########################################################################################
#make_pipeline은 내가 지은 이름이 없기 때문에 모델의 풀네임을 파라미터들 앞에 소문자로 쓰고 아래작대기 두개 __ 붙인다

parameters = [
    {'randomforestclassifier__n_estimators':[100,200], 'randomforestclassifier__max_depth':[6,10,12]},
    {'randomforestclassifier__max_depth':[6,8,10,10], 'randomforestclassifier__min_samples_leaf':[3,5,7,10]},
    {'randomforestclassifier__min_samples_leaf':[3,5,7,10]}
]

search_list = [GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV]
search_name = ['그리드','랜덤','할빙']

max_score = 0
max_acc = 0
max_search_name = ''

#2 모델
for index,value in enumerate(search_list):
    pipe = make_pipeline(StandardScaler(),RandomForestClassifier())
    model = value(pipe, parameters, cv=5, verbose=1)
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    y_predict = model.predict(x_test)
    acc = accuracy_score(y_test, y_predict)
    if acc>max_acc:
        max_acc = acc
        max_search_name = search_name[index]
        
print(max_search_name, round(max_acc,4))
