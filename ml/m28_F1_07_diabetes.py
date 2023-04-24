import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier , DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline 
import matplotlib.pyplot as plt

#1 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8,random_state = 123, shuffle=True)

model_list = [DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor, XGBRegressor]
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
            plt.title('XGBRegressor')
    
    plt.subplot(2,2,i+1)
    plot_feature_importances(model)
plt.show()

x1 = np.delete(x,[2,8],axis=1)
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
    print('기존acc:', result)
    print('칼럼삭제후acc:', result1)
    
# DecisionTreeRegressor
# 기존acc: -0.26219438808602624
# 칼럼삭제후acc: 0.1181526277170033
# RandomForestRegressor
# 기존acc: -0.26219438808602624
# 칼럼삭제후acc: 0.1181526277170033
# GradientBoostingRegressor
# 기존acc: -0.26219438808602624
# 칼럼삭제후acc: 0.1181526277170033
# XGBRegressor
# 기존acc: -0.26219438808602624
# 칼럼삭제후acc: 0.1181526277170033