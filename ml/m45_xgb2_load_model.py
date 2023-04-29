#가중치 역시 데이터이다.
#데이터(가중치)를 저장하는 방법에 대하여import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score
import pickle
import joblib

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

#2. 모델 - 피클 불러오기
path = './_save/pickle_test/'
# model = joblib.load(path + 'm43_pickle1_save.dat') 
model = XGBClassifier()
model.load_model(path + 'm44_xgb1_save_model.dat')

#4. 평가, 예측
# print("최상의 매개변수 :", model.best_params_)
results = model.score(x_test, y_test)
print("최종점수 : ", results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc :', acc)


##########################################
# path = './_save/pickle_test/'
# pickle.dump(model, open(path + 'm43_pickle1_save.dat', 'wb')) # wb = write binary