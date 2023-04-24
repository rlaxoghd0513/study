import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
#decomposition 분해


#1. 데이터
path = './_data/kaggle_bike/'
path_save = './_save/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0) 

x = train_csv.drop(['casual','registered','count'], axis=1)
y= train_csv['count']
print(x.shape, y.shape)  #(10886, 8) (10886,)
                    
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
    
# 차원 0 개 축소: 0.2895771892656308
# 차원 1 개 축소: 0.2938420048759155
# 차원 2 개 축소: 0.2664999926233613
# 차원 3 개 축소: 0.23961437109608652
# 차원 4 개 축소: 0.23390529920394865
# 차원 5 개 축소: 0.12472498676921084
# 차원 6 개 축소: 0.11616342655473855
# 차원 7 개 축소: -0.15106887979914108