# PCA 주성분분석 차원(column) 축소 10개짜리 칼럼 갖다가 훈련시킬때 열개에 대해 특성을 잡아낸다
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline 
from sklearn.decomposition import PCA

#1 데이터
x,y = load_digits(return_X_y=True)
print(x.shape) #(1797, 64) 사이킷런의 모든 모델은 2차원만 받는다 8,8을 쭉핀 형태다
print(np.unique(y, return_counts = True))

# pca = PCA(n_components=8) #8개로 압축한다 디폴트는 주성분분석에 의해 값이 변하긴 하지만 크기는 그대로
# x = pca.fit_transform(x)
# print(x.shape) #(1797, 8)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8,random_state = 123, shuffle=True)

#2 모델
# model = RandomForestClassifier()
model = make_pipeline(PCA(n_components=8), StandardScaler(), RandomForestClassifier()) #스케일러 뭐쓸지와 모델 뭐쓸지 
#순서대로 넣어야한다 pca하고 스케일링 할건지 스케일링하고 pca할건지

#3 훈련
model.fit(x_train,y_train)

#4 평가예측
result = model.score(x_test, y_test)
print('model.score:', result)

y_predict = model.predict(x_test)
print('acc', accuracy_score(y_test, y_predict))