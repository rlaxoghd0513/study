#그림그려보기
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier #트리계열은 결측치에 강하다, 이상치에서도 자유롭다, 스케일링을 하지 않아도 괜찮다
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier       #설치해야됨 xgboost
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline 


#1 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8,random_state = 123, shuffle=True)

# 칼럼에서 필요없는 칼럼을 찾아내보자

#2 모델
model = RandomForestClassifier()


model.fit(x_train, y_train)
result = model.score(x_test, y_test)
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('====================')
print('model.score:', result)
print('acc:', acc)

import matplotlib.pyplot as plt

def plot_feature_importances(model):                #그래프 정의
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features),model.feature_importances_,align='center') #수평 막대 그래프를 그리는 함수입니다.
                                                                              #barh 함수는 x축에는 막대의 길이 값을, y축에는 각 막대의 이름
    plt.yticks(np.arange(n_features),datasets.feature_names) #그래프의 y축 눈금을 설정하는 함수입니다.이 함수는 두 개의 인자를 받습니다. 
                                                            #첫 번째 인자는 y축 눈금의 위치를 나타내는 리스트이고, 두 번째 인자는 해당 눈금 위치에 출력할 레이블들입니다.
    plt.xlabel('Feature Importances')                        
    plt.ylabel('Features')
    plt.ylim(-1, n_features) # y축 범위를 설정하는 함수입니다.
                             #이 함수는 두 개의 인자를 받습니다. 
                             # 첫 번째 인자는 y축의 최소값과 최대값을 나타내는 튜플 (ymin, ymax)이며, 두 번째 인자는 y축 범위를 자동으로 조정하는 옵션 auto입니다
    plt.title(model)

plot_feature_importances(model)
plt.show()
    
