import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint

#1 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]
             ,[5,6,7],[6,7,8],[7,8,9],[8,9,10]
             ,[9,10,11],[10,11,12]
             ,[20,30,40],[30,40,50],[40,50,60]])

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x_predict = np.array([50,60,70])

# 만들어
print(x.shape, y.shape)  #(13, 3)  (13,)
print(x_predict.shape) #(3,)

x = x.reshape(13,3,1) 
print(x.shape) #(13, 3, 1)
x_predict = x_predict.reshape(1,3,1)
#모델구성
model = Sequential()
model.add(LSTM(64, input_shape = (3,1)))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(256))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

#컴파일 훈련
model.compile(loss='mse', optimizer = 'adam')
mcp = ModelCheckpoint(monitor = 'loss', mode='min', save_best_only=True, filepath ='./_save/MCP/lstm_scale/lstm_scale_mcp1.hdf5')
model.fit(x,y,epochs = 100, callbacks = [mcp], verbose=1)

#평가 예측
loss = model.evaluate(x,y)
results = model.predict(x_predict)
print('loss:', loss)
print('[50,60,70]결과:', results)

