import numpy as np
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D, Flatten
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, RobustScaler, MaxAbsScaler, StandardScaler

#데이터
path = './_data/kaggle_bike/'
path_save = './_save/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)

print(train_csv) #(10886,11)
print(train_csv.shape)

test_csv = pd.read_csv(path + 'test.csv', index_col=0) 
#index_col= 0번째부터 세고 읽는거에서 뺀다

print(test_csv)
print(test_csv.shape)  #(6493, 8)

print(train_csv.columns)
#(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
    #    'humidity', 'windspeed', 'casual', 'registered', 'count'],
    #   dtype='object')
print(test_csv.columns)
#(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
    #    'humidity', 'windspeed'],
    #   dtype='object')

x = train_csv.drop(['casual','registered','count'], axis=1)

y= train_csv['count']



x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, random_state=555, train_size=0.9)
print(x_train.shape, x_test.shape) #(9797, 8) (1089, 8)
print(y_train.shape, y_test.shape) #(9797,) (1089,)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test))

test_csv = scaler.transform(test_csv)

x_train = x_train.reshape(9797,4,2,1)
x_test = x_test.reshape(1089,4,2,1)

#모델구성
#한정화함수 다음레이어로 전하는걸 한정시킨다
#relu 양수는 양수로 음수는 0으로 최종레이어에는 잘 쓰지 않는다
#linear 있으나마나 
# model=Sequential()
# model.add(Dense(8, input_dim=8))
# model.add(Dense(7))
# model.add(Dense(6))
# model.add(Dense(5))#디폴트값
# model.add(Dense(4, activation = 'relu'))
# model.add(Dense(3, activation = 'relu'))
# model.add(Dense(1))
input1 = Input(shape = (4,2,1))
conv1 = Conv2D(8,2,padding ='same')(input1)
conv2 = Conv2D(16,(3,2), padding = 'same')(conv1)
conv3 = Conv2D(8,2,padding='same')(conv2)
flat= Flatten()(conv3)
dense1 = Dense(8)(flat)
dense2 = Dense(16)(dense1)
dense3 = Dense(8)(dense2)
drop1 = Dropout(0.3)(dense3)
dense4 = Dense(8, activation='relu')(drop1)
output1 = Dense(1)(dense4)
model = Model(inputs = input1 , outputs = output1)


#컴파일 훈련
model.compile(loss='mse', optimizer= 'adam')

import datetime
date = datetime.datetime.now()  #현재시간을 date에 넣어준다
print(date)  #2023-03-14 11:10:57.992357
date = date.strftime("%m%d_%H%M") #시간을 문자데이터로 바꾸겠다 그래야 파일명에 넣을 수 있다    %뒤에있는값을 반환해달라
print(date)  #0314_1116

filepath = './_save/MCP/kaggle_bike/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

from tensorflow.python.keras.callbacks import EarlyStopping
es= EarlyStopping(monitor='val_loss', mode='min', patience=40, verbose=1, restore_best_weights=True)
mcp = ModelCheckpoint(monitor= 'val_loss', mode='auto', save_best_only=True,verbose=1, filepath="".join([filepath,'kb_',date,'_',filename]))
hist =model.fit(x_train, y_train, batch_size=16, epochs=10000, verbose=1,validation_split=0.15, callbacks=[es,mcp])
print(hist.history['val_loss'])


#평가 예측
loss= model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)

r2= r2_score(y_test, y_predict)
print('r2스코어:', r2)

# loss: 24760.17578125
# r2스코어: 0.21855691285124823