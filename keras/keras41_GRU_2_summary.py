import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#1 . 데이터
# GRU는 LSTM의 장기의존성 문제를 해결한거
# GRU에는 Output Gate가 없다.
# GRU는 LSTM과 다르게 Gate가 2개이며, Reset Gate과 Update Gate로 이루어져있다.
# Reset Gate는 이전 상태를 얼마나 반영할지
# Update Gate는 이전 상태와 현재 상태를 얼마만큼의 비율로 반영할지


#2 모델구성
model = Sequential()       # [batch, timesteps, feature].  timesteps 어떤 크기로 자르겠다 여기선 5로 잘랐다
model.add(GRU(10, input_shape = (5,1)))  #(5,1)이만큼씩 훈련시키겠다 32는 아웃풋 노드갯수  rnn은 3차원으로 입력값을 받아도 출력은 2차원 
#units *(units +features+bias) 파라미터 갯수   lstm simplelnn 히든만 다르고 input output은 완벽히 똑같다
model.add(Dense(7))
model.add(Dense(1))
model.summary()