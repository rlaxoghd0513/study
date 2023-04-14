#분류
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import all_estimators
import sklearn as sk
print(sk.__version__)

x,y = load_digits(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(x,y, shuffle=True, random_state =121, test_size = 0.2)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# model = RandomForestRegressor(n_jobs=4) #?
allAlgorithms = all_estimators(type_filter ='classifier')
# allAlgorithms = all_estimators(type_filter ='regressor')
print(allAlgorithms)
print('모델갯수:', len(allAlgorithms))

max_acc = 0
max_name = '최대'
for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        results = model.score(x_test, y_test)
        print(name,'의 정답률', results)
        
        if max_acc < results:
            max_acc = results
            max_name = name
        # y_predict = model.predict(x_test)
        # # print(y_test.dtype)
        # # print(y_predict.dtype)
        # aaa = r2_score(y_test, y_predict)
        # print('r2:',aaa)
    except:
        print(name)
print("===========================")
print('최고모델:', max_name, max_acc)
print("===========================")

# 최고모델: ExtraTreesClassifier 0.9805555555555555