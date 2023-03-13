import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#1데이터
datasets = load_breast_cancer()
#print(datasets)   #**********리스트[]연결가능 '두개이상은 리스트'  딕셔너리{}키벨류 {키: 벨류} 데이터에 대한 값 예시로 사전 땡땡은 뭐뭐야******  튜플 소괄호 연결불가능 과제 파이썬의 자료형
# print(datasets.DESCR)    #사이킷런에서 DESCR pandas에서는  describe()
print(datasets.feature_names)  #pandas에서는 columns()

x=datasets['data']
y=datasets.target

print(x.shape, y.shape)  #(569, 30) (569,)
# print(y)


x_train, x_test, y_train, y_test = train_test_split(x,y, random_state= 123, train_size=0.8, shuffle=True)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test))

#모델구성
# model = Sequential()
# model.add(Dense(10, input_dim=30, activation='relu'))
# model.add(Dense(9, activation='linear'))
# model.add(Dense(8, activation='linear'))
# model.add(Dense(7, activation='linear'))
# model.add(Dense(1, activation='sigmoid')) #0과1 사이로 한정시킨다

input1 = Input(shape = (30,))
dense1 = Dense(10, activation='linear')(input1)
dense2 = Dense(9)(dense1)
dense3 = Dense(8)(dense2)
dense4 = Dense(7)(dense3)
output1 = Dense(1, activation = 'sigmoid')(dense4)
model = Model(inputs = input1, outputs = output1)

#3 컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics= ['accuracy','mse']  # 값을 리스트로 받으면 여러개가 들어갈 수 있다    *****두개이상은 리스트*****  2진 분류에서 mse 무의미, mae알고 싶으면 metrics에 mae박으면 된다
              )  #metrics accuracy를 넣으면 프린트에 np.round 할필요 없다 , 
from tensorflow.python.keras.callbacks import EarlyStopping
es=EarlyStopping(monitor= 'val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True )
hist = model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.2, verbose=1, callbacks=[es])

#4 평가 예측
results = model.evaluate(x_test, y_test)  #evaluate로 돌린 loss와 metrics에 넣은 지표가 출력
print('results:', results)
y_predict= np.round(model.predict(x_test))

# print("======================================") 
# print(y_predict[:5])
# print(np.round(y_predict[:5])) #np.round 반올림
# print("======================================")



# 정확도 accurate
from sklearn.metrics import r2_score, accuracy_score
acc=accuracy_score(y_test, y_predict)
print('acc:', acc)  

#sequential  [0.08199618011713028, 0.9736841917037964, 0.020016854628920555]
#model       [0.06929958611726761, 0.9912280440330505, 0.012348112650215626]