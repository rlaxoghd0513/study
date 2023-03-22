import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#1 . 데이터


#2 모델구성
model = Sequential()       # [batch, timesteps, feature].  timesteps 어떤 크기로 자르겠다 여기선 5로 잘랐다
#(5,1)이만큼씩 훈련시키겠다 32는 아웃풋 노드갯수  rnn은 3차원으로 입력값을 받아도 출력은 2차원 
#units *(units +features+bias) 파라미터 갯수   lstm simplelnn 히든만 다르고 input output은 완벽히 똑같다
#mode.add(LSTM(10, input_shape=(5,1))) 이렇게 써도 되고
# model.add(LSTM(10, input_length = 5, input_dim=1))   이렇게 써도 되고
model.add(LSTM(10, input_length = 1, input_dim=5)) #이렇게 바꿔도 된다 #[batch, input_length, input_dim]
model.add(Dense(7))
model.add(Dense(1))
model.summary()