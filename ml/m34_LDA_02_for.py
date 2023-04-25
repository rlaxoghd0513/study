import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #LDA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype

#1 데이터
data = [load_iris, load_breast_cancer, load_wine, load_digits, fetch_covtype]
data_name = ['아이리스','캔서','와인','디깃츠','코브타입']
for i,value in enumerate(data):
    x,y = value(return_X_y=True)
    
    x_train,x_test,y_train,y_test = train_test_split(x,y, train_size = 0.8, shuffle=True, random_state=333)
    
    lda = LinearDiscriminantAnalysis()
    x1 = lda.fit_transform(x,y)
    x1_train,x1_test,y1_train,y1_test = train_test_split(x1,y,train_size = 0.8, shuffle=True, random_state=333)
    
    model = RandomForestClassifier(random_state=333)
    
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    
    model = RandomForestClassifier(random_state=333)
    
    model.fit(x1_train, y1_train)
    result1 = model.score(x1_test, y1_test)
    
    print('===================')
    print(data_name[i],x.shape,'->',x1.shape)
    print(data_name[i],'기본acc:', result)
    print(data_name[i],'LDA_acc:', result1)
    
# ===================
# 아이리스 (150, 4) -> (150, 2)
# 아이리스 기본acc: 0.9666666666666667
# 아이리스 LDA_acc: 0.9333333333333333
# ===================
# 캔서 (569, 30) -> (569, 1)
# 캔서 기본acc: 0.9473684210526315
# 캔서 LDA_acc: 0.956140350877193
# ===================
# 와인 (178, 13) -> (178, 2)
# 와인 기본acc: 0.9444444444444444
# 와인 LDA_acc: 1.0
# ===================
# 디깃츠 (1797, 64) -> (1797, 9)
# 디깃츠 기본acc: 0.9777777777777777
# 디깃츠 LDA_acc: 0.9388888888888889
# ===================
# 코브타입 (581012, 54) -> (581012, 6)
# 코브타입 기본acc: 0.9541061762605096
# 코브타입 LDA_acc: 0.8837035188420265
    




