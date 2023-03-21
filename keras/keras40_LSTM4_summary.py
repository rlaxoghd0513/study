import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#1 . 데이터


#2 모델구성
model = Sequential()       # [batch, timesteps, feature].  timesteps 어떤 크기로 자르겠다 여기선 5로 잘랐다
model.add(LSTM(10, input_shape = (5,1)))  #(5,1)이만큼씩 훈련시키겠다 32는 아웃풋 노드갯수  rnn은 3차원으로 입력값을 받아도 출력은 2차원 
#units *(units +features+bias) 파라미터 갯수   lstm simplelnn 히든만 다르고 input output은 완벽히 똑같다
model.add(Dense(7))
model.add(Dense(1))
model.summary()
# lstm 의 파라미터 계산갯수(연산량)는 simpliernn 의 4배다