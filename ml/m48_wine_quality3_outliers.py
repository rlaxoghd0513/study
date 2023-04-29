#아웃라이어 확인

#아웃라이어 처리

#돌려

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

#1. 데이터
path = './_data/dacon_wine/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)  
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

# pandas를 numpy로 바꾸는 법

le = LabelEncoder()
le.fit(train_csv['type'])

aaa = le.transform(train_csv['type'])
train_csv['type'] = aaa
test_csv['type'] = le.transform(test_csv['type'])

data = train_csv
# x = train_csv.drop(['quality'], axis= 1)
# y = train_csv['quality']

# x = x.to_numpy()
# y= y.to_numpy()
data = data.to_numpy()

def outliers(data_out) :
    quartile_1, q2, quartile_3 = np.percentile(data_out,[25, 50, 75])
    print('1사분위 : ',quartile_1)
    print('q2 : ', q2)
    print('3사분위 : ',quartile_3)
    iqr = quartile_3 - quartile_1
    print('iqr : ', iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound)|(data_out<lower_bound))
    
outliers_loc = outliers(data)
print('이상치의 위치 : ', outliers_loc)     

data = np.delete(data, outliers_loc, 0)


data = pd.DataFrame(data)

x = data.drop([0], axis= 1)
y = data[0]

# print(x.shape, y.shape) # (4898, 11) (4898,)

# print(np.unique(y, return_counts=True))
# (array([3., 4., 5., 6., 7., 8., 9.]), array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))
# print(data['quality'].value_counts())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=123, shuffle=True,
                                                    train_size=0.8)

#2. 모델
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

#3. 훈련
model.fit(x_train,  y_train)

#4. 평가,예측
y_predict = model.predict(x_test)

from sklearn.metrics import accuracy_score, f1_score    

score = model.score(x_test, y_test)
print('model.score : ', score)
print('acc_score : ', accuracy_score(y_test,y_predict))
print('f1_score(macro) : ', f1_score(y_test,y_predict, average='macro'))
print('f1_score(micro) : ', f1_score(y_test,y_predict, average='micro'))
