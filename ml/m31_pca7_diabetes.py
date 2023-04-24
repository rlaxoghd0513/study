import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
#decomposition 분해


#1. 데이터
datasets = load_diabetes()

x = datasets['data']
y = datasets.target
print(x.shape, y.shape)  #(442, 10) (442,)
                    
for i in range(10):
    
    pca = PCA(n_components = 10-i)
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
    
# 차원 0 개 축소: 0.5175313353682958
# 차원 1 개 축소: 0.5176641914777897
# 차원 2 개 축소: 0.5115188842870858
# 차원 3 개 축소: 0.5135775085581957
# 차원 4 개 축소: 0.47944566216290596
# 차원 5 개 축소: 0.49717844368864106
# 차원 6 개 축소: 0.463420367992899
# 차원 7 개 축소: 0.20456612051196454
# 차원 8 개 축소: 0.08830856693058775
# 차원 9 개 축소: 0.08358980396334348