#input_dim 행과 열만 있는 2차원 데이터셋만 했음
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model  #함수형모델은 그냥 Model
from tensorflow.python.keras.layers import Dense, Input    #함수형에서는 따로 명시해야됨
import numpy as np
from sklearn.preprocessing import MinMaxScaler #값을 0에서 1사이로 바꾸지만 standardScaler는 평균점을 중심으로 모아준다
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import MaxAbsScaler  
from sklearn.preprocessing import RobustScaler  

#1 데이터
datasets = load_boston()
x= datasets.data    
y= datasets.target

print(type(x)) #<class 'numpy.ndarray'>
print(x)



x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, random_state=300)

scaler = MinMaxScaler()  
x_train = scaler.fit_transform(x_train)


x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test)) 



#2 모델구성
# model=Sequential()
# model.add(Dense(1, input_dim=13))
# model.add(Dense(30, input_shape=(13, )))    #함수형에서는 input_layer랑 첫번재 히든레이어랑 같은 줄에 쓰면 안되고 따로 명시해야한다
# model.add(Dense(20))
# model.add(Dense(10))
# model.add(Dense(1))

input1 = Input(shape=(13,))
dense1 = Dense(30)(input1) 
dense2 = Dense(20)(dense1) 
dense3 = Dense(10)(dense2)
output1 = Dense(1)(dense3) 
model = Model(inputs = input1, outputs = output1)   


#모델저장
# model.save('./_save/keras26_1_save_model.h5') 



#컴파일 훈련
model.compile(loss='mse', optimizer= 'adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience = 10, mode = 'min'
                   ,verbose = 1, restore_best_weights=True )

mcp = ModelCheckpoint(monitor = 'val_loss', mode = 'auto', 
                        verbose= 1, 
                        save_best_only=True,
                        filepath='./_save/MCP/keras27_ModelCheckPoint1.hdf5')

model.fit(x_train, y_train, epochs=10000, batch_size=32, callbacks=[es,mcp], validation_split=0.2)

#평가 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)