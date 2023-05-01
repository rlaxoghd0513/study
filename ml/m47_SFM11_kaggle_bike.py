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
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')
# 경사하강법
# 그래디언트 디센트
 


#1. 데이터

# 1. 데이터
path='./_data/kaggle_bike/'
path_save='./_save/kaggle_bike/'    

train_csv = pd.read_csv(path + 'train.csv',
                        index_col=0)                 

train_csv = train_csv.dropna()   

x = train_csv.drop(['count'], axis=1)   
y = train_csv['count']

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
scaler = RobustScaler()
model = XGBRegressor(**parameter
                      )
# print("========================================================")
# x, y = load_diabetes(return_X_y=True)


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    random_state=337,train_size=0.8)
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
model.fit(x_train, y_train, eval_set=[(x_train,y_train),(x_test,y_test)], 
          early_stopping_rounds=10,
          verbose=0,
          eval_metric='rmse')

# results = model.score(x_test,y_test)
# print('score :', results)

# y_predict = model.predict(x_test)
# r2 = r2_score(y_test, y_predict)

# print('r2 :', r2)

# mse = mean_squared_error(y_test, y_predict)
# print('rmse :', np.sqrt(mse))
    

# print(model.feature_importances)
thresholds = np.sort(model.feature_importances_)                    

print(thresholds)
# [0.03175544 0.06501069 0.07197672 0.07753727 0.08700569 0.10525618
#  0.12046603 0.13875969 0.14712931 0.15510292]     10개의 컬런중 지정한 값과 나머지것들을 비교하여 지정한 값과 그 값보다 큰 값을 유지함. 따라서 for문에 입력하면 1번 값부터 값을 비교하며 지정한 값보다 낮은 값은 drop됨

for i in thresholds :
    selection = SelectFromModel(model,threshold=i,
                                prefit=True, #prefit 사전 훈련
                                # prefit=False, #사전훈련한 가중치를 사용하지 않는다는 의미. 다시 훈련시켜서 가중치를 사용함
                            )
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print('변형된 x_train :', select_x_train.shape, '변형된 x_test :', select_x_test.shape)
    
    selection_model = XGBRegressor()
    
    selection_model.set_params(early_stopping_rounds=10, **parameter, eval_metric='rmse',)
    
    selection_model.fit(select_x_train,y_train,
                        eval_set=[(select_x_train,y_train), (select_x_test, y_test)],
                        verbose=0,
                        )
    
    select_y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test, select_y_predict)
    print("Tresh=%.3f, 남은 컬런 갯수=%d, R2: %.2f%%"%(i, select_x_train.shape[1], score*100))
    #%.3f는 float 소수 셋째짜리까지의 숫자를 넣으라는 의미, 그 숫자는 뒤 % 뒤로 정의한 내용중 첫번째.
    #%d는 정수형으로 값을 빼라는 의미, 그 숫자는 뒤 % 뒤로 정의한 내용중 두번째.
    #%.2f%% 는 float 소수 둘째자리까지의 숫자를 넣으라는 의미. %%는 글자 %를 입력하고 싶을때 쓰는 문법. 출력되는 숫자는 %뒤로 정의한 내용중 세번째

# 변형된 x_train : (8708, 10) 변형된 x_test : (2178, 10)
# Tresh=0.000, 남은 컬런 갯수=10, R2: 98.11%
# 변형된 x_train : (8708, 9) 변형된 x_test : (2178, 9)
# Tresh=0.002, 남은 컬런 갯수=9, R2: 99.59%
# 변형된 x_train : (8708, 8) 변형된 x_test : (2178, 8)
# Tresh=0.004, 남은 컬런 갯수=8, R2: 99.24%
# 변형된 x_train : (8708, 7) 변형된 x_test : (2178, 7)
# Tresh=0.006, 남은 컬런 갯수=7, R2: 99.73%
# 변형된 x_train : (8708, 6) 변형된 x_test : (2178, 6)
# Tresh=0.007, 남은 컬런 갯수=6, R2: 99.45%
# 변형된 x_train : (8708, 5) 변형된 x_test : (2178, 5)
# Tresh=0.032, 남은 컬런 갯수=5, R2: 99.53%
# 변형된 x_train : (8708, 4) 변형된 x_test : (2178, 4)
# Tresh=0.042, 남은 컬런 갯수=4, R2: 99.83%
# 변형된 x_train : (8708, 3) 변형된 x_test : (2178, 3)
# Tresh=0.065, 남은 컬런 갯수=3, R2: 99.98%
# 변형된 x_train : (8708, 2) 변형된 x_test : (2178, 2)
# Tresh=0.074, 남은 컬런 갯수=2, R2: 99.99%
# 변형된 x_train : (8708, 1) 변형된 x_test : (2178, 1)
# Tresh=0.768, 남은 컬런 갯수=1, R2: 94.82%