#Linear Discriminant Analysis
#회귀에서 되는지
#결론: 안된다  y에 round씌우면 되긴하지만 결국 데이터 조작이고 소숫점 부분이 중요한 데이터 일수도 있기 때문에 사용하지 않는다

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #LDA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_diabetes, fetch_california_housing
from tensorflow.keras.datasets import cifar100

#1 데이터
x,y = load_diabetes(return_X_y=True)
print(y)
print(np.unique(y))
# 정수값이기 때문에 클래스로 판단했다
# fetch_california는 소수로 나와서 안된다 
# y에 round씌우면 가능하긴 하다
print(len(np.unique(y))) #214

lda = LinearDiscriminantAnalysis()

x_lda = lda.fit_transform(x,y)

print(x_lda.shape) #(442, 10)

#회귀는 원래 안된다 하지만 diabetes는 정수형이라서 LDA에서 y의 크래스로 잘못 인식한거다 그래서 돌아간거다
#성호는 캘리포니아에서 라운드처리했다 그래서 정수형이 되서 클래스로 인식해서 돌아간거다
#회귀데이터는 원칙적으로 에러가 뜬다
#굳이 y에 round씌워서 LDA로 돌릴 필요가 없다
