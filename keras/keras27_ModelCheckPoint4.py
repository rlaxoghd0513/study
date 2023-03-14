# 저장할때 평가결과값, 훈련시간 등을 파일에 넣어줘

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model  
from tensorflow.python.keras.layers import Dense, Input   
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



input1 = Input(shape=(13,))
dense1 = Dense(30)(input1) 
dense2 = Dense(20)(dense1) 
dense3 = Dense(10)(dense2)
output1 = Dense(1)(dense3) 
model = Model(inputs = input1, outputs = output1)    



#컴파일 훈련
model.compile(loss='mse', optimizer= 'adam')

import datetime
date = datetime.datetime.now()  #현재시간을 date에 넣어준다
print(date)  #2023-03-14 11:10:57.992357
date = date.strftime("%m%d_%H%M") #시간을 문자데이터로 바꾸겠다 그래야 파일명에 넣을 수 있다    %뒤에있는값을 반환해달라
print(date)  #0314_1116

filepath = './_save/MCP/keras27_4/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'


from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience = 10, mode = 'min'
                   ,verbose = 1, 
                #    restore_best_weights=True 
                )

mcp = ModelCheckpoint(monitor = 'val_loss', mode = 'auto', 
                        verbose= 1, 
                        save_best_only=True,
                        filepath="".join([filepath,'k27_',date,'_',filename]) #""빈공간에 뭘 합치겠다
                        )

model.fit(x_train, y_train, epochs=10000, batch_size=32, callbacks=[es,mcp], validation_split=0.2)



#평가 예측
from sklearn.metrics import r2_score

print("========================== 1. 기본출력 ==============================")
loss = model.evaluate(x_test, y_test, verbose = 0)
print('loss:', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어:',r2)


# ========================== 1. 기본출력 ============================== ****restore_best_weights 끌지말지 본인이 판단****
# loss: 26.318572998046875
# r2스코어: 0.6430106298990698
# ===================2. load_model 출력=======================
# loss: 26.318572998046875
# r2스코어: 0.6430106298990698
# ===================3. MCP 출력=======================
# loss: 25.912784576416016
# r2스코어: 0.6485147658619967