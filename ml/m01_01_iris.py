import numpy as np
from sklearn.datasets import load_iris

#1 데이터
# datasets = load_iris()
# x = datasets.data
# y = datasets['target']

x,y = load_iris(return_X_y=True)  #바로 x랑 y가 온다

print(x.shape, y.shape) #(150, 4) (150,)

#2 모델구성
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

# model = Sequential()
# model.add(Dense(10, activation = 'relu', input_shape = (4,)))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(3, activation = 'softmax'))

# model = LinearSVC() #안에 파라미터들이 다 있다   인공지능 선긋는거 괄호안에 c가 작으면 작을수록 직선이다  c가 클수록 데이터를
model = DecisionTreeClassifier()

#3 컴파일 훈련
# model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['acc']) # 위에서 원핫 하지 않았다면 sparse써주면 된다 주의할점은 0부터 시작하는지 확인해야한다 아니면 라벨 틀어진다
# model.fit(x,y, epochs = 100, validation_split = 0.2)

model.fit(x,y)

#4 평가 예측
# results = model.evaluate(x, y)

results = model.score(x,y)

print(results)

# 어떤 모델 쓸건지  
# 모델 안에 알고리즘 어떤 모델을 쓸건지

