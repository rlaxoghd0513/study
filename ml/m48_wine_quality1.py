import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical

#1. 데이터
path = './_data/dacon_wine/'

train_csv = pd.read_csv(path + 'train.csv',
                        index_col=0)                 

test_csv = pd.read_csv(path + 'test.csv',
                        index_col=0)                 
print(train_csv.shape)
# (5497, 13)

print(test_csv.shape)
# (1000, 12)

# print(train_csv['quality'].value_counts())
# 6    2416
# 5    1788
# 7     924
# 4     186
# 8     152
# 3      26
# 9       5

# print(train_csv.columns)
# Index(['quality', 'fixed acidity', 'volatile acidity', 'citric acid',
#        'residual sugar', 'chlorides', 'free sulfur dioxide',
#        'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol',
#        'type'],
#       dtype='object')

# print(test_csv.columns)
# Index(['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
#        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
#        'pH', 'sulphates', 'alcohol', 'type'],
#       dtype='object')


le = LabelEncoder()
le.fit(train_csv['type'])

aaa = le.transform(train_csv['type'])
train_csv['type'] = aaa
test_csv['type'] = le.transform(test_csv['type'])

x = train_csv.drop(['quality'], axis= 1)
y = train_csv['quality']-3

# 



x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    random_state=3377,
                                                    shuffle=True,
                                                    stratify= y
                                                    )

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

print(y_train.shape)
print(y)

# 2. 모델구성
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

params ={'n_estimators' : 30000,
        'learning_rate' : 0.01, #일반적으로 가장 성능에 영향을 많이 끼침. 경사하강법에서 얼만큼씩 하강할것이냐를 뜻함. 웨이트를 찾을때 적절한 러닝레이트 필요
        'max_depth' : 30, #트리형 모델의 깊이.
        'gamma' : 0,
        'min_child_weight' : 0, 
        'subsample' : 0.4, # 드랍아웃의 개념. 0.2만큼 덜어낸다는 의미
        'colsample_bytree' : 0.8,
        'colsample_bylevel': 0.7,
        'colsample_bynode': 1,
        'reg_alpha': 0, #알파와 람다 l1, l2 규제
        'reg_lambda': 1,
        'random_state': 3377,
        # 
        }
model = XGBClassifier(**params)

# 3. 컴파일, 훈련

# model.set_params(early_stopping_rounds=50,
#                  )

# model.fit(x_train,y_train,
#         eval_set =[(x_train, y_train),(x_test, y_test)],    #각 튜플 항목에 대한 로스가 나오기 때문에 train 항목을 넣으면 케라스의 loss와 같다.
# )

# y_pred = model.predict(x_test)

# print('score :', model.score(x_test,y_test))

# print('acc :', accuracy_score(y_test, y_pred))

model.set_params(early_stopping_rounds=1000, n_jobs = -1,
                 eval_metric='merror',
                 )
model.fit(x_train, y_train,
          eval_set =[(x_train, y_train),(x_test, y_test)],    #각 튜플 항목에 대한 로스가 나오기 때문에 train 항목을 넣으면 케라스의 loss와 같다.
          #발리데이션 데이터다.
        #   early_stopping_rounds=10,  #더 이상 지표가 감소하지 않는 최대 반복횟수
        #   verbose=False,   #verbose=true of false
          
          )

#4. 평가, 예측
# print("최상의 매개변수 :", model.best_params_)
# print('최상의 점수 :', model.score(x_test,y_test))
# results = model.score(x_test, y_test)
# print("최종점수 : ", results)

y_pred = np.argmax(model.predict(test_csv), axis=1)

submission = pd.read_csv('./_data/dacon_wine/sample_submission.csv', index_col=0)

submission['quality'] = y_pred+3

submission.to_csv('./_data/dacon_wine/sub.csv')