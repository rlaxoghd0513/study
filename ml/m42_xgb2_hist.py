import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from xgboost import XGBClassifier, XGBRegressor


# 경사하강법
# 그래디언트 디센트
 


#1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=337, train_size =0.8, stratify=y
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# 이상치는 결측치라고 볼 수도 있다.
# 발견된 이상치를 Nan으로 바꿔서 모두 결측치 처리해버릴 수도 잇음

# 'n_estimators' : [100, 200, 300, 400, 500, 1000] 디폴트 100 / 1~inf / 정수
# 'learning_rate' : [0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001] 디폴트 0.3 / 0~1 /
# 'max_depth' : [None, 2, 3, 4, 5, 6, 7, 8, 9, 10] 디폴트 6 / 0~inf/ 정수
# 'gamma' : [0, 1, 2, 3, 4, 5, 7, 10, 100] 디폴트 0 / 0~inf
# 'min_child_weight' : [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10, 100] 디폴트1 
# 'subsample' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1 / 0~1
# 'colsample_bytree' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트1 / 0~1
# 'colsample_bylevel': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트1 / 0~1
# 'colsample_bynode': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트1 / 0~1
# 'reg_alpha': [0, 0.1, 0.01, 0.001, 1, 2, 10] 디폴트0 / 0~inf / L1 절대값 가중치 규제
# 'reg_lambda': [0, 0.1, 0.01, 0.001, 1, 2, 10] 디폴트0 / 0~inf / L2 제곱 가중치 규제

# L1 규제 = 절대값으로 규제하여 레이어상에서 양수로 만들겠다는 소리이다. 라쏘
# L2 규제 = 제곱으로 규제하여 레이어상에서 양수로 만듬 릿지


n_splits = 5

kfold = KFold(n_splits=n_splits)

parameter ={'n_estimators' : 1000,
            'learning_rate' : 0.3, #일반적으로 가장 성능에 영향을 많이 끼침. 경사하강법에서 얼만큼씩 하강할것이냐를 뜻함. 웨이트를 찾을때 적절한 러닝레이트 필요
            'max_depth' : 2, #트리형 모델의 깊이.
            'gamma' : 0,
            'min_child_weight' : 0, 
            'subsample' : 0.2, # 드랍아웃의 개념. 0.2만큼 덜어낸다는 의미
            'colsample_bytree' : 0.5,
            'colsample_bylevel': 0,
            'colsample_bynode': 1,
            'reg_alpha': 1, #알파와 람다 l1, l2 규제
            'reg_lambda': 1,
            'random_state': 337,
            }

#값을 리스트 형태로 넣으면 에러. 파라미터는 항상 한개의 값만을 받을 수 있기 때문이다.

#2. 모델

model = XGBClassifier(**parameter
                      )

#3. 훈련
model.set_params(early_stopping_rounds=50)
model.fit(x_train, y_train,
          eval_set =[(x_train, y_train),(x_test, y_test)],    #각 튜플 항목에 대한 로스가 나오기 때문에 train 항목을 넣으면 케라스의 loss와 같다.
          #발리데이션 데이터다.
        #   early_stopping_rounds=10,  #더 이상 지표가 감소하지 않는 최대 반복횟수
        #   verbose=False,   #verbose=true of false
        #   eval_metric = 'error',#이진분류
        #   eval_metric = 'logloss', #이진분류
          eval_metric = 'auc', #이진분류
        #   eval_metric = 'merror',#다중분류
        #   eval_metric = 'mlogloss',#다중분류
        #   eval_metric = 'rmse','mae','rmsle....#회귀
          )

print("========================================================")
hist = model.evals_result()    #평가의 결과. 케라스의 hist와 같다.
print(hist)


#4. 평가, 예측
# print("최상의 매개변수 :", model.best_params_)
print('최상의 점수 :', model.score(x_test,y_test))

results = model.score(x_test, y_test)
print("최종점수 : ", results)


# 최상의 점수 : 0.9714285714285715
# 최종점수 :  0.956140350877193

#*args  인자의 갯수에 대해서 제한이 없다.
#**params  키 밸류 형태. 딕셔너리 형태. 이전 소스에서 정리한 parameter 딕셔너리를 한번에 받을 수 있다.



import matplotlib.pyplot as plt
train_score = hist['validation_0']['auc']
test_score = hist['validation_1']['auc']

plt.plot(train_score, label='Train')
plt.plot(test_score, label='Test')
plt.xlabel('epochs')
plt.ylabel('error')
plt.title('XGBoost')
plt.legend()
plt.show()