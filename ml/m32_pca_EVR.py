#pca 차원축소 칼럼축소 삭제가 아니라 압축하는 개념
#pca 했을때 성능이 좋아질수도 있다
#y는 차원축소할 필요가 없다 target값(통상y) 이기 때문에 차원축소 할수도 없다
#지도학습과 비지도학습 두가지 개념이 공존한다

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
#decomposition 분해

#1. 데이터
datasets = load_breast_cancer()

x = datasets['data']
y = datasets.target
print(x.shape, y.shape)  #(569, 30) (569,)

pca = PCA(n_components = 30)
x = pca.fit_transform(x)
print(x.shape)           #(569, 30)

pca_EVR = pca.explained_variance_ratio_ #설명가능한 변화율
print(pca_EVR)
print(sum(pca_EVR)) #0.9999999999999998  순서대로 pca1개했을때 2개했을때 3개 했을때---

print(np.cumsum(pca_EVR)) #누적합 pca를 몇개하면 데이터에 손실이 없을지 보는거 1부분에 가까울수록 데이터 손실이 없다
pca_cumsum = np.cumsum(pca_EVR)

import matplotlib.pyplot as plt
plt.plot(pca_cumsum)
plt.grid()
plt.show()

