import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping

#1데이터
datasets = load_iris()
print(datasets.DESCR) #판다스 describe()
# print(datasets.feature_names)  #판다스 columns
#['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(150, 4) (150,)

print(x)
print(y)
print('y의 라벨값:', np.unique(y))   #y의 라벨값 [0,1,2]
#원핫 인코딩 y에 들어있는걸 자리위치, 라벨의 개수만큼 쉐이프가 늘어남

############### 이지점에서 원핫을 해야함 #########이미 나눠진걸 또 나누면귀찮으니까
# 케라스 원핫인코딩 
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y.shape) #(150, 3)

#판다스 겟더미 y=pd.get_dummies(y)
#             print(y.shape)  (150, 3)

#사이킷런 원핫인코더 
# from sklearn.preprocessing import OneHotEncoder
# y=y.reshape(-1,1)
# ohe = OneHotEncoder()
# ohe.fit(y)
# y = ohe.transform(y)
# print(y.shape)  # (150, 3)


#판다스 겟더미, 사이킷런에 원핫인코더  약간의 차이 0부터 시작하냐 안하냐 



#y를 (150, )에서 (150,3)으로 바꾼다

x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, 
                                                    random_state=333,
                                                    train_size=0.8,
                                                    stratify=y # stratify y라벨값들 중 하나로 쏠리지 않게 균등하게 나눠주는거 
                                                    )
#분류모델할땐 y라벨값 확인해야함   
print(y_train)
print(np.unique(y_train, return_counts=True)) #y라벨이랑 각 개수까지 알려주는거

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test))

#2 모델구성
# model = Sequential()
# model.add(Dense(50, activation = 'relu',input_dim=4))
# model.add(Dense(40, activation = 'relu'))
# model.add(Dense(40, activation = 'relu'))
# model.add(Dense(10, activation = 'relu'))
# model.add(Dense(3, activation = 'softmax')) #다중분류에서 무조건 softmax, y라벨의 갯수만큼 아웃풋 노드 ,  값을 다 더하면 1, 3개중에 가장 큰 놈이라고 판단
# 다중분류 데이터를 받았을때 원핫 softmax를 통해 세개가 합이 1인 걸로 0과1사이로 한정됨
# 마지막 레이어 노드 라벨 개수  

input1 = Input(shape=(4, ))
dense1 = Dense(50)(input1)
dense2 = Dense(40)(dense1)
dense3 = Dense(40)(dense2)
dense4 = Dense(10)(dense3)
output1 = Dense(3)(dense4)
model = Model(inputs = input1, outputs = output1)

#3 컴파일 훈련
es = EarlyStopping(monitor='acc', patience = 20, mode='max'
              ,verbose=1
              ,restore_best_weights=True)
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])  #다중분류에서 loss는 
model.fit(x_train, y_train, epochs=100, batch_size = 1, validation_split = 0.2, verbose=1)

#4 평가 예측
# accuracy_score를 사용해서 스코어를 빼세요 밑에랑 비교 
# results = model.evaluate(x_test, y_test)
# print('results:', results)

# y_predict = np.round(model.predict(x_test))
# print(y_predict)
# acc=accuracy_score(y_test, y_predict)
# print('acc:',acc)
###################################################
#4. 평가 예측
results = model.evaluate(x_test, y_test) #단축기 ctrl+space  loss로 안짓고 results로 짓는 이유: 결과가 여러개가 나오기 때문
print(results)    
print('loss:', results[0])
print('acc:',results[1])

y_predict = model.predict(x_test)
# print(y.test.shape) #(30,3)
# print(y_predict.shape) 
# ########소프트 맥스의 결과는 0.1 0.2 0.7이런식으로
# print(y_test[:5])
# print(y_predict[:5])
y_test_acc = np.argmax(y_test, axis=1) #각 행에 있는 열끼리 비교 
y_pred = np.argmax(y_predict, axis = 1) #가독성을 위해 위에서 argmax 씌우지말고 한번 더 쓰는게 좋다
                                           #값들이 y_test, y_predict 둘다 예측값으로 바뀌어 있어야 한다
acc = accuracy_score(y_test_acc, y_pred) 

print('acc:', acc)

#sequential [1.3455125093460083, 0.8666666746139526]
#model      [5.456156253814697, 0.3333333432674408]
#더 안좋아짐