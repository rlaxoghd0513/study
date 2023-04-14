import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
#xor는 0과0이 같으면 1 0과1이 다르면 0

#1 데이터
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [0,1,1,0]

#2 모델구성
# model = LinearSVC()
model = SVC()

#컴파일 훈련
model.fit(x_data, y_data)

#4평가 예측
y_predict = model.predict(x_data)

results = model.score(x_data, y_data)
print("model.score : ", results)

acc = accuracy_score(y_data, y_predict)
print('acc:', acc)
#acc 0.5 나온다 선 긋기