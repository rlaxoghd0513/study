import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.preprocessing import MinMaxScaler, RobustScaler, MaxAbsScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data_list = [load_iris, load_breast_cancer, load_wine, load_digits]
model_list = [RandomForestClassifier, LogisticRegression, SVC, DecisionTreeClassifier]
scaler_list = [MinMaxScaler, RobustScaler, MaxAbsScaler, StandardScaler]
data_name_list = ['아이리스', '캔서', '와인', '디깃츠']
model_name_list = ['randomforestclassifier', 'logisticregression', 'svc', 'decisiontreeclassifier']
scaler_name_list = ['minmaxscaler', 'robustscaler', 'maxabsscaler', 'standardscaler']

for i, data in enumerate(data_list):
    x, y = data(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=333)
    
    max_score = 0
    max_name = ''
    max_scaler_name = ''
    max_model_name = ''
    
    for scaler in scaler_list:
        for model in model_list:
            model = Pipeline([('sc',scaler()), ('md',model())])
            model.fit(x_train, y_train)
            score = model.score(x_test, y_test)
            if score > max_score:
                max_score = score
                max_name = data_name_list[i]
                max_scaler_name = scaler_name_list[i]
                max_model_name = model_name_list[i]
                
    print(f"{max_name} 최적의 scaler: {max_scaler_name}, 최적의 model: {max_model_name}, 최고 점수: {max_score:.4f}")
    
# 아이리스 최적의 scaler: minmaxscaler, 최적의 model: randomforestclassifier, 최고 점수: 0.9667
# 캔서 최적의 scaler: robustscaler, 최적의 model: logisticregression, 최고 점수: 0.9825
# 와인 최적의 scaler: maxabsscaler, 최적의 model: svc, 최고 점수: 1.0000
# 디깃츠 최적의 scaler: standardscaler, 최적의 model: decisiontreeclassifier, 최고 점수: 0.9833