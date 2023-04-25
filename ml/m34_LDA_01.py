#Linear Discriminant Analysis  #Discriminant 판별
#얘도 차원축소하는애
#PCA는 데이터의 방향성에 따라 선을 긋고 데이터들을 선에 매치시킨다 그리고 평탄화 시키고 직각으로 다시 긋는다
#LDA는 데이터의 클래스별로 매치를 시키는 선을 긋는다 
#LDA는 지도학습 y의 클래스를 알아야지 사용할수 있다
#클래스별로 매치를 시키기 때문에 명확하다
#PCA는 회귀에서도 유용하지만 LDA는 클래스를 구분하는 경계를 찾아내는 데 효과적이다 그래서 분류에서 주로 사용한다
#PCA의 n_components는 칼럼 갯수가 들어가고 LDA의 n_components는 클래스 갯수가 들어간다


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #LDA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype

#1 데이터
x,y = load_iris(return_X_y=True)

# pca = PCA(n_components=4) #디폴트 : 전체 칼럼 갯수 그대로
# x = pca.fit_transform(x)
# print(x.shape) #(150, 4)

lda = LinearDiscriminantAnalysis() #디폴트: 전체 클래스 갯수 -1 #n_components 가 전체 클래스 갯수-1 이하로 넣어야한다
#lda는 클래스별로 매치시킨다했기 때문에 fit_transform할때 y값이 필요하다
#pca가 좋은지 lda가 좋은지는 해봐야 알 수 있다
x = lda.fit_transform(x,y)
print(x.shape) #(150, 2)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size = 0.8, random_state = 333, shuffle=True)

#2모델구성
model = RandomForestClassifier(random_state=123)
#훈련
model.fit(x_train, y_train)
#평가 예측
result = model.score(x_test, y_test)
print('결과:', result)