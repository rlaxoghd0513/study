#그림그려보기
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier    
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline 
import matplotlib.pyplot as plt
import pandas as pd

path = './_data/dacon_diabetes/'
path_save = './_save/dacon_diabetes/'

train_csv = pd.read_csv(path+'train.csv', index_col=0)

test_csv = pd.read_csv(path+'test.csv', index_col=0)

x = train_csv.drop(['Outcome'], axis=1)

y = train_csv['Outcome']


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8,random_state = 123, shuffle=True)

model_list = [DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier, XGBClassifier]
#2 모델
for i,value in enumerate(model_list):
    model = value()
    model.fit(x_train,y_train)
    def plot_feature_importances(model):                
        n_features = x.shape[1]
        plt.barh(np.arange(n_features),model.feature_importances_,align='center') 
                                                                            
        plt.yticks(np.arange(n_features),x.columns) 
                                                            
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

x1 = train_csv.drop([train_csv.columns[3],train_csv.columns[4]], axis=1)
model_name_list = ['DecisionTreeClassifier','RandomForestClassifier', 'GradientBoostingClassifier','XGBClassifier']

for j,value1 in enumerate(model_list):
    model = value1()
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    x1_train, x1_test, y_train, y_test = train_test_split(x1,y,train_size=0.8, shuffle=True, random_state=333)
    model = value()
    model.fit(x1_train, y_train)
    result1 = model.score(x1_test, y_test)
    print(model_name_list[j])
    print('기존acc:', result)
    print('칼럼삭제후acc:', result1)
    
# DecisionTreeClassifier
# 기존acc: 0.6564885496183206
# 칼럼삭제후acc: 1.0
# RandomForestClassifier
# 기존acc: 0.5954198473282443
# 칼럼삭제후acc: 1.0
# GradientBoostingClassifier
# 기존acc: 0.6106870229007634
# 칼럼삭제후acc: 1.0
# XGBClassifier
# 기존acc: 0.6030534351145038
# 칼럼삭제후acc: 1.0
