import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, RobustScaler, MaxAbsScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score

data_list = [load_diabetes, fetch_california_housing]
model_list = [RandomForestRegressor, DecisionTreeRegressor]
scaler_list = [MinMaxScaler, RobustScaler, MaxAbsScaler, StandardScaler]
data_name_list = ['디아벳', '캘리포니아']
model_name_list = ['randomforestRegressor', 'decisiontreeRegressor']
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
            model = make_pipeline(scaler(), model())
            model.fit(x_train, y_train)
            score = model.score(x_test, y_test)
            if score > max_score:
                max_score = score
                max_name = data_name_list[i]
                max_scaler_name = scaler_name_list[i]
                max_model_name = model_name_list[i]
                
    print(f"{max_name} 최적의 scaler: {max_scaler_name}, 최적의 model: {max_model_name}, 최고 점수: {max_score:.4f}")
    
# 디아벳 최적의 scaler: minmaxscaler, 최적의 model: randomforestRegressor, 최고 점수: 0.4149
# 캘리포니아 최적의 scaler: robustscaler, 최적의 model: decisiontreeRegressor, 최고 점수: 0.8023