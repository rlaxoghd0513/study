#가중치 역시 데이터이다.
#데이터(가중치)를 저장하는 방법에 대하여import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# 경사하강법
# 그래디언트 디센트
 


#1. 데이터



parameter ={'n_estimators' : 1000,
            'learning_rate' : 0.3, #일반적으로 가장 성능에 영향을 많이 끼침. 경사하강법에서 얼만큼씩 하강할것이냐를 뜻함. 웨이트를 찾을때 적절한 러닝레이트 필요
            'max_depth' : 2, #트리형 모델의 깊이.
            'gamma' : 0,
            'min_child_weight' : 0, 
            'subsample' : 0.2, # 드랍아웃의 개념. 0.2만큼 덜어낸다는 의미
            'colsample_bytree' : 0.5,
            'colsample_bylevel': 0,
            'colsample_bynode': 0,
            'reg_alpha': 1, #알파와 람다 l1, l2 규제
            'reg_lambda': 1,
            'random_state': 337,
            }

#값을 리스트 형태로 넣으면 에러. 파라미터는 항상 한개의 값만을 받을 수 있기 때문이다.

#2. 모델
scaler=RobustScaler()
model = XGBRegressor(**parameter
                      )
print("========================================================")

def Runmodel(a, x, y):
    x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                        random_state=337,train_size=0.8)
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    model.fit(x_train, y_train, eval_set=[(x_train,y_train),(x_test,y_test)], early_stopping_rounds=10, verbose=0,eval_metric='rmse')
    results = model.score(x_test,y_test)
    print(a, 'score :', results)
    y_predict = model.predict(x_test)
    r2 = r2_score(y_test, y_predict)
    print(a, 'r2 :', r2)
    
    mse = mean_squared_error(y_test, y_predict)
    print(a, 'rmse :', np.sqrt(mse))
    
x, y = load_diabetes(return_X_y=True)

Runmodel('기존 데이터', x, y)


for i in range(x.shape[1]-1) :
    a = model.feature_importances_
    b = np.argmin(a, axis=0)
    x = pd.DataFrame(pd.DataFrame(x).drop(b,axis=1).values)
    Runmodel(f'{9-i}개의 column 삭제', x, y)