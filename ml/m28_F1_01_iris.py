#실습
#피처임포턴스가 전체 중요도에서 하위 20-25 컬럼들 제거
#재구성 후
#모델을 돌려서 결과 도출
#기존 모델들과 성능 비교

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

model_list = [DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier, XGBClassifier]
model_name_list = ['DecisionTreeClassifier','RandomForestClassifier', 'GraidientBoostingClassifier','XGBClassifier']
for i,value in enumerate(model_list):
    x,y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8, shuffle=True, random_state=333)
    model = value()
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    x1 = np.delete(x,2,axis=1)
    x1_train, x1_test, y_train, y_test = train_test_split(x1,y,train_size=0.8, shuffle=True, random_state=333)
    model = value()
    model.fit(x1_train, y_train)
    result1 = model.score(x1_test, y_test)
    print(model_name_list[i])
    print('기존acc:', result)
    print('칼럼삭제후acc:', result1)
    
    
    

