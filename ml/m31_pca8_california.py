import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
#decomposition 분해


#1. 데이터
datasets = fetch_california_housing()

x = datasets['data']
y = datasets.target
print(x.shape, y.shape)  #(20640, 8) (20640,)
                    
for i in range(8):
    
    pca = PCA(n_components = 8-i)
#n_components 차원(칼럼) 몇개로 줄일건지
    x = pca.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, random_state=123, shuffle=True)

#2. 모델구성
    model = RandomForestRegressor(random_state=123) #훈련할때마다 바뀌니까 random_state고정

#3. 훈련
    model.fit(x_train, y_train)

#4. 평가,예측
    results = model.score(x_test, y_test)
    print('차원',i,'개 축소:', results)
    
# 차원 0 개 축소: 0.7825857242412009
# 차원 1 개 축소: 0.7786727671384369
# 차원 2 개 축소: 0.7018597110810503
# 차원 3 개 축소: 0.5918722922244304
# 차원 4 개 축소: 0.3241551445937575
# 차원 5 개 축소: 0.0789494633195541
# 차원 6 개 축소: 0.046317003872303086
# 차원 7 개 축소: -0.4412309439201201