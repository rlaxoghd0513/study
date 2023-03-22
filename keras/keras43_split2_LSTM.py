import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

dataset = np.array(range(1,101)) #1부터 100까지
timesteps = 5
#데이터갯수 - timestep+1 = 행의 개수


def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps + 1):  #len 개수  5번동안 반복한다  
        subset = dataset[i : (i+timesteps)] #i부터 i+timesteps-1까지
        aaa.append(subset)        #append 넣는다
    return np.array(aaa)

bbb = split_x(dataset, timesteps)
print(bbb)
print(bbb.shape)


x = bbb[:, :-1]    
y = bbb[:, -1]
print(x)
print(y)
print(x.shape)
print(x.shape) #(96, 4)
x = x.reshape(96,4,1)

x_predict = np.array(range(96,106)) 
timesteps_pre = 4

def split_x(x_predict, timesteps_pre):
    aaa = []
    for i in range(len(x_predict) - timesteps_pre + 1):  #len 개수  5번동안 반복한다  
        subset = x_predict[i : (i+timesteps_pre)] #i부터 i+timesteps-1까지
        aaa.append(subset)        #append 넣는다
    return np.array(aaa)

x_predict = split_x(x_predict, timesteps_pre)
print(x_predict)
print(x_predict.shape)  #(7, 4)
x_predict = x_predict.reshape(7,4,1)

#모델구성
model = Sequential()
model.add(LSTM(32,input_shape = (4,1) ,return_sequences=True))
model.add(LSTM(16, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(128))
model.add(Dense(256))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))

model.compile(loss= 'mse', optimizer= 'adam')
model.fit(x,y,epochs=100)

loss = model.evaluate(x,y)


result = model.predict(x_predict)
print('loss:', loss)
print('result:', result)





