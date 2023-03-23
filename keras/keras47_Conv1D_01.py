import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Flatten

#1 데이터
#2 모델구성

model = Sequential()
# model.add(LSTM(10, input_shape=(3,1)))  #541개 총 연산량
model.add(Conv1D(10,2, input_shape=(3,1))) #10은 conv에선 필터 나머지에선 유닛
model.add(Conv1D(10,2)) #conv는 두개이상 쌓아야한다 #141개 총 연산량
model.add(Flatten())
model.add(Dense(5))
model.add(Dense(1))
model.summary()  #LSTM은 상태를 전달하지만 Conv1은 특성을 뽑아낸다
# 2차원은 Dense 3차원은 RNN, conv1d 4차원은 conv2d


