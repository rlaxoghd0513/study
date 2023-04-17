import numpy as np
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler,RobustScaler, StandardScaler
import warnings 
warnings.filterwarnings('ignore')
from sklearn.utils import all_estimators


datasets = [load_diabetes(return_X_y=True),
            fetch_california_housing(return_X_y = True)]

data_name = ['디아벳','캘리포니아']

n_splits= 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

for index, value in enumerate(datasets):
    x,y = value
    #모델구성
    allAlgorithms = all_estimators(type_filter = 'regressor')
    
    max_score = 0
    max_name = '바보'
    for (name, algorithm) in allAlgorithms:
        try:
            if name in ['MinMaxScaler', 'RobustScaler','MaxabsScaler','StandardScaler']:
               model = algorithm()
               model = model[1](model[0]())
            else:
                model = algorithm()
                
            scores = cross_val_score(model, x,y, cv = kfold)
            results = round(np.mean(scores),4)
            
            if max_score < results:
                max_score = results
                max_name = name
        except:
            continue
        
    print("========", data_name[index],"========")
    print("최고모델:", max_name, max_score)
    print("======================") 