import numpy as np
from sklearn.datasets import load_iris,load_breast_cancer, load_wine,load_digits, fetch_covtype
from sklearn.model_selection import train_test_split, cross_val_score, KFold #결국 cross_val_score를 사용했을때 이것또한 과적합이라고 볼 수 있다
#과적합을 포기하고 전체데이터를 할지 , 과적합은 안된다 버릴건 버리자 하면 train_test 나누고 train만 cross_val할지
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
import warnings 
warnings.filterwarnings('ignore')
from sklearn.utils import all_estimators
from sklearn.model_selection import cross_val_predict


#1 데이터
datasets = [load_iris(return_X_y=True),
            load_breast_cancer(return_X_y=True),
            load_wine(return_X_y=True),
            load_digits(return_X_y=True)]

data_name = ['아이리스','캔서','와인','디지트']

n_splits= 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

for index, value in enumerate(datasets):
    x,y = value
    x_train, x_test,y_train, y_test = train_test_split(x,y, train_size = 0.8, random_state=123, shuffle = True)
    #모델구성
    allAlgorithms = all_estimators(type_filter = 'classifier')
    
    max_score = 0
    max_name = '바보'
    max_acc = 0
    max_predict = '일'
    for (name, algorithm) in allAlgorithms:
        try:
            model = algorithm()
            scores = cross_val_score(model, x_train,y_train, cv = kfold)
            results = round(np.mean(scores),4)
            
            y_predict = cross_val_predict(model, x_test, y_test, cv = kfold)
            acc = accuracy_score(y_test, y_predict)
            if max_score < results:
                max_score = results
                max_name = name
            if max_acc < acc:
                max_acc = acc 
                max_predict = name
                          
        except:
            continue
        
    print("========", data_name[index],"========")
    print("최고score모델:", max_name, max_score)
    print('최고predict모델:', max_predict, max_acc)
    print("======================") 
