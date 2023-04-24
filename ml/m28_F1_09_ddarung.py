#그림그려보기
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier , DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline 
import matplotlib.pyplot as plt
import pandas as pd

path = './_data/ddarung/'
path_save= './_save/ddarung/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0) 
train_csv = train_csv.dropna()

x = train_csv.drop(['count'], axis=1) 
y = train_csv['count']


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8,random_state = 123, shuffle=True)

model_list = [DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor, XGBRegressor]
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
            plt.title('XGBRegressor')
    
    plt.subplot(2,2,i+1)
    plot_feature_importances(model)
plt.show()

x1 = train_csv.drop([train_csv.columns[0],train_csv.columns[5]], axis=1)
model_name_list = ['DecisionTreeRegressor','RandomForestRegressor', 'GradientBoostingRegressor','XGBRegressor']

for j,value1 in enumerate(model_list):
    model = value1()
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    x1_train, x1_test, y_train, y_test = train_test_split(x1,y,train_size=0.8, shuffle=True, random_state=333)
    model = value()
    model.fit(x1_train, y_train)
    result1 = model.score(x1_test, y_test)
for k in model_name_list:
    print(k)
    print('기존r2:', result)
    print('칼럼삭제후r2:', result1)
    
# DecisionTreeRegressor
# 기존r2: -0.2982885064782794
# 칼럼삭제후r2: 0.9996238217704159
# RandomForestRegressor
# 기존r2: -0.2982885064782794
# 칼럼삭제후r2: 0.9996238217704159
# GradientBoostingRegressor
# 기존r2: -0.2982885064782794
# 칼럼삭제후r2: 0.9996238217704159
# XGBRegressor
# 기존r2: -0.2982885064782794
# 칼럼삭제후r2: 0.9996238217704159