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
import matplotlib.pyplot as plt

#1 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8,random_state = 123, shuffle=True)

# 칼럼에서 필요없는 칼럼을 찾아내보자
model_list = [DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier, XGBClassifier]
#2 모델
for i,value in enumerate(model_list):
    model = value()
    model.fit(x_train,y_train)
    def plot_feature_importances(model):                
        n_features = datasets.data.shape[1]
        plt.barh(np.arange(n_features),model.feature_importances_,align='center') 
                                                                            
        plt.yticks(np.arange(n_features),datasets.feature_names) 
                                                            
        plt.xlabel('Feature Importances')                        
        plt.ylabel('Features')
        plt.ylim(-1, n_features) 
        
        if i !=3:
            plt.title(model)       
        else:
            plt.title('XGBClassifier')
    
    plt.subplot(2,2,i+1)
    plot_feature_importances(model)
plt.show()