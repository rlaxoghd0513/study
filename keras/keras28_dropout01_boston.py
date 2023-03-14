#과적합을 해결하는 방법
# 데이터가 많으면 된다
# 신경망을 훈련할때 노드 일부를 랜덤으로 배재하고 훈련을 시킨다
# 무조건 좋아지진 않지만 큰데이터는 효율이 있을 가능성이 높다 해봐야안다

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model  
from tensorflow.python.keras.layers import Dense, Input, Dropout   #노드 일부 뺀다
import numpy as np
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import MaxAbsScaler  
from sklearn.preprocessing import RobustScaler  

#1 데이터
datasets = load_boston()
x= datasets.data    
y= datasets.target

print(type(x))
print(x)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=300)

scaler = MinMaxScaler()  
x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)



# input1 = Input(shape=(13,))
# dense1 = Dense(30)(input1) 
# drop1 = Dropout(0.3)(dense1)    #몇퍼센트를 드랍아웃할지
# dense2 = Dense(20, activation='relu')(drop1) 
# drop2 = Dropout(0.2)(dense2)
# dense3 = Dense(10)(drop2)
# drop3 = Dropout(0.5)(dense3)
# output1 = Dense(1)(drop3) 
# model = Model(inputs = input1, outputs = output1)    

model = Sequential()
model.add(Dense(30, input_shape=(13,)))
model.add(Dropout(0.3))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))


#컴파일 훈련
model.compile(loss='mse', optimizer= 'adam')

import datetime
date = datetime.datetime.now()  #현재시간을 date에 넣어준다
print(date)  #2023-03-14 11:10:57.992357
date = date.strftime("%m%d_%H%M") #시간을 문자데이터로 바꾸겠다 그래야 파일명에 넣을 수 있다    %뒤에있는값을 반환해달라  소문자 m은 month 대문자 M은 minutes
print(date)  #0314_1116

filepath = './_save/MCP/keras27_4/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'


from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience = 10, mode = 'min'
                   ,verbose = 1, 
                   restore_best_weights=True 
                )

mcp = ModelCheckpoint(monitor = 'val_loss', mode = 'auto', 
                        verbose= 1, 
                        save_best_only=True,
                        filepath="".join([filepath,'k27_',date,'_',filename]) #""빈공간에 뭘 합치겠다
                        )

model.fit(x_train, y_train, epochs=10000, batch_size=32, callbacks=[es], validation_split=0.2)



#평가 예측
from sklearn.metrics import r2_score

print("========================== 1. 기본출력 ==============================")
loss = model.evaluate(x_test, y_test, verbose = 0)           # 드랍아웃은 평가예측에서는 적용안된다
print('loss:', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어:',r2)


