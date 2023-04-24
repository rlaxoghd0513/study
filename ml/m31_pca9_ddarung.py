import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
#decomposition 분해


#1. 데이터
path = './_data/ddarung/'
path_save= './_save/ddarung/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0) 
train_csv = train_csv.dropna()

x = train_csv.drop(['count'], axis=1) 
y = train_csv['count']
print(x.shape, y.shape)  #(1328, 9) (1328,)
                    
for i in range(9):
    
    pca = PCA(n_components = 9-i)
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
    
# 차원 0 개 축소: 0.6936276869466653
# 차원 1 개 축소: 0.6784454950446617
# 차원 2 개 축소: 0.6754821123352452
# 차원 3 개 축소: 0.6935666626368776
# 차원 4 개 축소: 0.62252151710199
# 차원 5 개 축소: 0.29494787387754695
# 차원 6 개 축소: 0.2353812467908073
# 차원 7 개 축소: 0.09707754599708784
# 차원 8 개 축소: -0.26917844142607183