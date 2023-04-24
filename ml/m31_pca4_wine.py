import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
#decomposition 분해


#1. 데이터
datasets = load_wine()

x = datasets['data']
y = datasets.target
print(x.shape, y.shape)  #(178, 13) (178,)
                    
for i in range(13):
    
    pca = PCA(n_components = 13-i)
#n_components 차원(칼럼) 몇개로 줄일건지
    x = pca.fit_transform(x)


    x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, random_state=123, shuffle=True)

#2. 모델구성
    model = RandomForestClassifier(random_state=123) #훈련할때마다 바뀌니까 random_state고정

#3. 훈련
    model.fit(x_train, y_train)

#4. 평가,예측
    results = model.score(x_test, y_test)
    print('차원',i,'개 축소:', results)
    
# 차원 0 개 축소: 0.8888888888888888
# 차원 1 개 축소: 0.9444444444444444
# 차원 2 개 축소: 0.9166666666666666
# 차원 3 개 축소: 0.8888888888888888
# 차원 4 개 축소: 0.9166666666666666
# 차원 5 개 축소: 0.9166666666666666
# 차원 6 개 축소: 0.9166666666666666
# 차원 7 개 축소: 0.9166666666666666
# 차원 8 개 축소: 0.9166666666666666
# 차원 9 개 축소: 0.8888888888888888
# 차원 10 개 축소: 0.7222222222222222
# 차원 11 개 축소: 0.6388888888888888
# 차원 12 개 축소: 0.6111111111111112