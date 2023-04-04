from sklearn.datasets import fetch_covtype
import numpy as np
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
datasets = fetch_covtype()

x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(581012, 54) (581012,)
print('y의 라벨값:',np.unique(y)) #[1 2 3 4 5 6 7]
print(y)

#keras 카테고리컬
# y = to_categorical(y)
# print(y.shape)       #(581012, 8)  #카테고리컬로 하면 0부터 시작해서 0라벨이 생겨서 8개가 된다
# y = np.delete(y,0,axis=1)   #y의 열에서 0번째 행을 뺀다
# print(y)   #(581012, 7)

#2. sklearn
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder()
# y = y.reshape(-1,1)
# y = ohe.fit_transform(y).toarray()
# print(y.shape) # (581012,7)
# print(type(y)) #<class 'numpy.ndarray'>

#3.pandas get_dummies
# import pandas as pd
# y=pd.get_dummies(y)
# print(y.shape)


x_train, x_test, y_train, y_test = train_test_split(x,y, random_state =1234, 
                                             shuffle = True, train_size = 0.8)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test))

#모델구성
# model = Sequential()
# model.add(Dense(60, input_dim=54))
# model.add(Dense(60))
# model.add(Dense(50))
# model.add(Dense(40))
# model.add(Dense(30))
# model.add(Dense(20))
# model.add(Dense(7, activation = 'softmax'))
# model.summary()

input1 = Input(shape=(54, ))
dense1 = Dense(60)(input1)
dense2 = Dense(60)(dense1)
dense3 = Dense(50)(dense2)
dense4 = Dense(40)(dense3)
dense5 = Dense(30)(dense4)
dense6 = Dense(20)(dense5)
output1 = Dense(7, activation = 'softmax')(dense6)
model = Model(inputs = input1, outputs = output1)


#컴파일 훈련
model.compile(loss='categorical_crossentropy' #sparse_categorical_crossentropy 
              , optimizer='adam', metrics=['acc']) 
es=EarlyStopping(monitor='acc', patience=10, mode='max',verbose=1, restore_best_weights=True)

import time
start_time = time.time()

model.fit(x_train, y_train, batch_size=32, epochs=1, validation_split=0.2,verbose=1, callbacks=[es]) #배치가 크면 터진다

end_time = time.time()

#평가 예측
results = model.evaluate(x_test, y_test)
print('results:', results)
print('걸린시간:', round(end_time - start_time,2))    #훈련시키는거에는 np.round 아무거나엔 그냥 round



y_predict = model.predict(x_test)
print(y_predict.shape)
y_test_acc = np.argmax(y_test, axis=1)
print(y_test_acc)
y_predict_acc = np.argmax(y_predict, axis=1)


acc = accuracy_score(y_test_acc, y_predict_acc)
print('acc:', acc)

#sequential acc 0.7184582153644915
#model      acc 0.7068406151304183