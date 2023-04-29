#가중치 역시 데이터이다.
#데이터(가중치)를 저장하는 방법에 대하여import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine,fetch_covtype,load_digits,fetch_california_housing
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')

x, y = load_digits(return_X_y=True)
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

#2. 모델
scaler = RobustScaler()
model = XGBClassifier(**parameter
                      )



x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    random_state=337,train_size=0.8)

model.fit(x_train, y_train, eval_set=[(x_train,y_train),(x_test,y_test)], 
          early_stopping_rounds=10,
          verbose=0,
          eval_metric='auc')



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
    
    selection_model = XGBClassifier()
    
    selection_model.set_params(**parameter, eval_metric='merror',)
    
    selection_model.fit(select_x_train,y_train,
                        # eval_set=[(select_x_train,y_train), (select_x_test, y_test)],
                        verbose=0, )
    
    select_y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test, select_y_predict)
    print("Tresh=%.3f, 남은 컬런 갯수=%d, R2: %.2f%%"%(i, select_x_train.shape[1], score*100))
    #%.3f는 float 소수 셋째짜리까지의 숫자를 넣으라는 의미, 그 숫자는 뒤 % 뒤로 정의한 내용중 첫번째.
    #%d는 정수형으로 값을 빼라는 의미, 그 숫자는 뒤 % 뒤로 정의한 내용중 두번째.
    #%.2f%% 는 float 소수 둘째자리까지의 숫자를 넣으라는 의미. %%는 글자 %를 입력하고 싶을때 쓰는 문법. 출력되는 숫자는 %뒤로 정의한 내용중 세번째

# 변형된 x_train : (1437, 64) 변형된 x_test : (360, 64)
# Tresh=0.000, 남은 컬런 갯수=64, R2: 87.00%
# 변형된 x_train : (1437, 61) 변형된 x_test : (360, 61)
# Tresh=0.000, 남은 컬런 갯수=61, R2: 87.13%
# 변형된 x_train : (1437, 60) 변형된 x_test : (360, 60)
# Tresh=0.000, 남은 컬런 갯수=60, R2: 88.74%
# 변형된 x_train : (1437, 59) 변형된 x_test : (360, 59)
# Tresh=0.000, 남은 컬런 갯수=59, R2: 87.36%
# 변형된 x_train : (1437, 58) 변형된 x_test : (360, 58)
# Tresh=0.000, 남은 컬런 갯수=58, R2: 90.91%
# 변형된 x_train : (1437, 57) 변형된 x_test : (360, 57)
# Tresh=0.000, 남은 컬런 갯수=57, R2: 85.71%
# 변형된 x_train : (1437, 56) 변형된 x_test : (360, 56)
# Tresh=0.000, 남은 컬런 갯수=56, R2: 91.93%
# 변형된 x_train : (1437, 55) 변형된 x_test : (360, 55)
# Tresh=0.001, 남은 컬런 갯수=55, R2: 89.43%
# 변형된 x_train : (1437, 54) 변형된 x_test : (360, 54)
# Tresh=0.001, 남은 컬런 갯수=54, R2: 88.41%
# 변형된 x_train : (1437, 53) 변형된 x_test : (360, 53)
# Tresh=0.002, 남은 컬런 갯수=53, R2: 89.37%
# 변형된 x_train : (1437, 52) 변형된 x_test : (360, 52)
# Tresh=0.003, 남은 컬런 갯수=52, R2: 86.77%
# 변형된 x_train : (1437, 51) 변형된 x_test : (360, 51)
# Tresh=0.004, 남은 컬런 갯수=51, R2: 86.50%
# 변형된 x_train : (1437, 50) 변형된 x_test : (360, 50)
# Tresh=0.005, 남은 컬런 갯수=50, R2: 93.12%
# 변형된 x_train : (1437, 49) 변형된 x_test : (360, 49)
# Tresh=0.005, 남은 컬런 갯수=49, R2: 86.34%
# 변형된 x_train : (1437, 48) 변형된 x_test : (360, 48)
# Tresh=0.006, 남은 컬런 갯수=48, R2: 87.36%
# 변형된 x_train : (1437, 47) 변형된 x_test : (360, 47)
# Tresh=0.007, 남은 컬런 갯수=47, R2: 85.15%
# 변형된 x_train : (1437, 46) 변형된 x_test : (360, 46)
# Tresh=0.007, 남은 컬런 갯수=46, R2: 92.30%
# 변형된 x_train : (1437, 45) 변형된 x_test : (360, 45)
# Tresh=0.007, 남은 컬런 갯수=45, R2: 84.46%
# 변형된 x_train : (1437, 44) 변형된 x_test : (360, 44)
# Tresh=0.009, 남은 컬런 갯수=44, R2: 88.31%
# 변형된 x_train : (1437, 43) 변형된 x_test : (360, 43)
# Tresh=0.009, 남은 컬런 갯수=43, R2: 87.29%
# 변형된 x_train : (1437, 42) 변형된 x_test : (360, 42)
# Tresh=0.009, 남은 컬런 갯수=42, R2: 85.48%
# 변형된 x_train : (1437, 41) 변형된 x_test : (360, 41)
# Tresh=0.010, 남은 컬런 갯수=41, R2: 87.62%
# 변형된 x_train : (1437, 40) 변형된 x_test : (360, 40)
# Tresh=0.013, 남은 컬런 갯수=40, R2: 92.53%
# 변형된 x_train : (1437, 39) 변형된 x_test : (360, 39)
# Tresh=0.013, 남은 컬런 갯수=39, R2: 84.13%
# 변형된 x_train : (1437, 38) 변형된 x_test : (360, 38)
# Tresh=0.013, 남은 컬런 갯수=38, R2: 86.07%
# 변형된 x_train : (1437, 37) 변형된 x_test : (360, 37)
# Tresh=0.014, 남은 컬런 갯수=37, R2: 89.17%
# 변형된 x_train : (1437, 36) 변형된 x_test : (360, 36)
# Tresh=0.014, 남은 컬런 갯수=36, R2: 92.56%
# 변형된 x_train : (1437, 35) 변형된 x_test : (360, 35)
# Tresh=0.014, 남은 컬런 갯수=35, R2: 89.40%
# 변형된 x_train : (1437, 34) 변형된 x_test : (360, 34)
# Tresh=0.015, 남은 컬런 갯수=34, R2: 86.93%
# 변형된 x_train : (1437, 33) 변형된 x_test : (360, 33)
# Tresh=0.016, 남은 컬런 갯수=33, R2: 83.80%
# 변형된 x_train : (1437, 32) 변형된 x_test : (360, 32)
# Tresh=0.016, 남은 컬런 갯수=32, R2: 85.42%
# 변형된 x_train : (1437, 31) 변형된 x_test : (360, 31)
# Tresh=0.016, 남은 컬런 갯수=31, R2: 88.21%
# 변형된 x_train : (1437, 30) 변형된 x_test : (360, 30)
# Tresh=0.016, 남은 컬런 갯수=30, R2: 85.45%
# 변형된 x_train : (1437, 29) 변형된 x_test : (360, 29)
# Tresh=0.016, 남은 컬런 갯수=29, R2: 80.41%
# 변형된 x_train : (1437, 28) 변형된 x_test : (360, 28)
# Tresh=0.017, 남은 컬런 갯수=28, R2: 84.92%
# 변형된 x_train : (1437, 27) 변형된 x_test : (360, 27)
# Tresh=0.018, 남은 컬런 갯수=27, R2: 84.99%
# 변형된 x_train : (1437, 26) 변형된 x_test : (360, 26)