#pca 차원축소 칼럼축소 삭제가 아니라 압축하는 개념
#pca 했을때 성능이 좋아질수도 있다
#y는 차원축소할 필요가 없다 target값(통상y) 이기 때문에 차원축소 할수도 없다
#지도학습과 비지도학습 두가지 개념이 공존한다
#선위에다 맵핑한다 또 할때는 직각 또할때는 또 직각

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
#decomposition 분해


#1. 데이터
datasets = load_iris()

x = datasets['data']
y = datasets.target
print(x.shape, y.shape) 

# pca = PCA(n_components = 2)
# #n_components 차원(칼럼) 몇개로 줄일건지
# x = pca.fit_transform(x)
# print(x.shape)  

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, random_state=123, shuffle=True)

#2. 모델구성
model = RandomForestClassifier(random_state=123) #훈련할때마다 바뀌니까 random_state고정

#3. 훈련
model.fit(x_train, y_train)

#4. 평가,예측
results = model.score(x_test, y_test)
print('결과:', results)

# 기본결과: 0.9734776536312849
# pca결과: 0.9333333333333333