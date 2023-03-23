#양방향 bidirectional
#bidirectional은 rnn에서 사용

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU
from tensorflow.keras.layers import Bidirectional

#LSTM 구조 3개의 게이트와 한개의 스테이트 
# (forgettable gate input gate output gate)   (cell state)
#구조 꼭 기억해야함
#파라미터 연산값 simplernn 4배

#1 . 데이터
datasets = np.array([1,2,3,4,5,6,7,8,10])

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9]])   
y = np.array([4,5,6,7,8,9,10])

print(x.shape, y.shape)


x = x.reshape(7,3,1) #[[[1],[2],[3]],[[2],[3],[4]],......]
print(x.shape)  #(7, 3, 1)

#2 모델구성
model = Sequential()
model.add(Bidirectional(SimpleRNN(10), input_shape = (3,1))) #rnn을 bidirectional로 래핑한다 감싼다
model.add(Dense(1))
model.summary()