#pca 차원축소 칼럼축소 삭제가 아니라 압축하는 개념
#pca 했을때 성능이 좋아질수도 있다
#y는 차원축소할 필요가 없다 target값(통상y) 이기 때문에 차원축소 할수도 없다
#지도학습과 비지도학습 두가지 개념이 공존한다

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
#decomposition 분해


#1. 데이터
datasets = load_breast_cancer()

x = datasets['data']
y = datasets.target
print(x.shape, y.shape)  
                    
for i in range(30):
    
    pca = PCA(n_components = 30-i)
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
    
# 차원 0 개 축소: 0.9736842105263158
# 차원 1 개 축소: 0.9912280701754386
# 차원 2 개 축소: 0.9736842105263158
# 차원 3 개 축소: 0.9736842105263158
# 차원 4 개 축소: 0.9824561403508771
# 차원 5 개 축소: 0.9912280701754386
# 차원 6 개 축소: 0.9824561403508771
# 차원 7 개 축소: 0.9824561403508771
# 차원 8 개 축소: 0.9736842105263158
# 차원 9 개 축소: 0.9824561403508771
# 차원 10 개 축소: 0.9824561403508771
# 차원 11 개 축소: 0.9824561403508771
# 차원 12 개 축소: 0.9824561403508771
# 차원 13 개 축소: 0.9824561403508771
# 차원 14 개 축소: 0.9824561403508771
# 차원 15 개 축소: 0.9912280701754386
# 차원 16 개 축소: 0.9824561403508771
# 차원 17 개 축소: 0.9824561403508771
# 차원 18 개 축소: 0.9824561403508771
# 차원 19 개 축소: 0.9912280701754386
# 차원 20 개 축소: 0.9824561403508771
# 차원 21 개 축소: 0.9912280701754386
# 차원 22 개 축소: 0.9912280701754386
# 차원 23 개 축소: 0.9824561403508771
# 차원 24 개 축소: 0.9912280701754386
# 차원 25 개 축소: 0.9912280701754386
# 차원 26 개 축소: 0.9824561403508771
# 차원 27 개 축소: 0.956140350877193
# 차원 28 개 축소: 0.9473684210526315
# 차원 29 개 축소: 0.868421052631579
    

