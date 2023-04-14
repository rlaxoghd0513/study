import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#xor는 0과0이 같으면 1 0과1이 다르면 0

#1 데이터
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [0,1,1,0]

#2 모델구성
# model = LinearSVC()
#퍼셉트론 케라스로 구현한거?
model = Sequential()
model.add(Dense(1, input_dim = 2, activation = 'sigmoid')) 

#컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_data, y_data, batch_size = 1, epochs = 100)

#4평가 예측
y_predict = model.predict(x_data)


# results = model.score(x_data, y_data)
results= model.evaluate(x_data, y_data)
print("model.score : ", results[1])

acc = accuracy_score(y_data, np.round(y_predict))
print('acc:', acc)
